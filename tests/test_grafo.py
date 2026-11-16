"""Fase 4: carga no Neo4j com sessão falsa (mock), sem banco real."""
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from t2kg.avaliacao.original import normalizar_tripla
from t2kg.grafo.carregar_neo4j import (
    construir_operacoes,
    executar_ops,
    carregar_credenciais,
    resolver_experimento,
    _CY_REL,
    _CY_APAGAR_RUN,
)


class FakeSession:
    def __init__(self):
        self.chamadas = []

    def run(self, cypher, **params):
        self.chamadas.append((cypher, params))


def _registros():
    return [{
        "id": "s1", "sent": "A dirige B",
        "triples": [["A", "director", "B"], ["A", "genre", "Comedy"]],
    }]


def _gt():
    # só director(A,B) está no gabarito → director=acerto, genre=sobra
    return {"s1": {normalizar_tripla("A", "director", "B")}}


class TestConstruirOperacoes:
    def test_quatro_operacoes_sem_apagar(self):
        ops = construir_operacoes(_registros(), {"condicao": "C1", "modelo": "Q", "run": "r1"}, _gt())
        assert len(ops) == 4

    def test_apagar_run_adiciona_operacao(self):
        ops = construir_operacoes(_registros(), {"condicao": "C1", "modelo": "Q", "run": "r1"},
                                  _gt(), apagar_run=True)
        assert len(ops) == 5
        assert ops[0][0] == _CY_APAGAR_RUN
        assert ops[0][1] == {"run": "r1"}

    def test_status_acerto_e_sobra(self):
        ops = construir_operacoes(_registros(), {"condicao": "C1", "modelo": "Q", "run": "r1"}, _gt())
        rel_rows = next(params["rows"] for cy, params in ops if cy == _CY_REL)
        status = {r["relacao"]: r["status"] for r in rel_rows}
        assert status == {"director": "acerto", "genre": "sobra"}

    def test_entidades_e_frase_na_aresta(self):
        ops = construir_operacoes(_registros(), {"condicao": "C1", "modelo": "Q", "run": "r1"}, _gt())
        rel_rows = next(params["rows"] for cy, params in ops if cy == _CY_REL)
        assert all(r["frase"] == "s1" and r["modelo"] == "Q" and r["run"] == "r1" for r in rel_rows)
        nomes = next(params["nomes"] for cy, params in ops if "Entidade {nome: nome}" in cy)
        assert set(nomes) == {"A", "B", "Comedy"}


class TestExecutarOps:
    def test_executa_todas_as_operacoes(self):
        ops = construir_operacoes(_registros(), {"condicao": "C1", "modelo": "Q", "run": "r1"}, _gt())
        sess = FakeSession()
        executar_ops(ops, sess)
        assert len(sess.chamadas) == len(ops)


class TestCredenciais:
    def test_le_do_ambiente_com_fallback_username(self, monkeypatch, tmp_path):
        monkeypatch.setattr("t2kg.grafo.carregar_neo4j.REPO", str(tmp_path))  # sem .env
        monkeypatch.setenv("NEO4J_URI", "neo4j+s://x")
        monkeypatch.delenv("NEO4J_USER", raising=False)
        monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
        monkeypatch.setenv("NEO4J_PASSWORD", "secret")
        uri, user, pwd = carregar_credenciais()
        assert (uri, user, pwd) == ("neo4j+s://x", "neo4j", "secret")

    def test_erra_sem_credenciais(self, monkeypatch, tmp_path):
        monkeypatch.setattr("t2kg.grafo.carregar_neo4j.REPO", str(tmp_path))
        for k in ("NEO4J_URI", "NEO4J_USER", "NEO4J_USERNAME", "NEO4J_PASSWORD"):
            monkeypatch.delenv(k, raising=False)
        with pytest.raises(RuntimeError):
            carregar_credenciais()


class TestResolverExperimento:
    def test_resolve_pv3(self):
        manifesto = os.path.join(REPO, "configs", "pipeline", "manifesto_piloto_v3.json")
        exp = resolver_experimento("pv3_1movie_c1", manifesto)
        assert exp is not None
        assert exp["dataset"] == "tekgen" and exp["condicao"] == "C1"
