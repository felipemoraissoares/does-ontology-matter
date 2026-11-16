"""Etapa B: valida o gerador e o manifesto do experimento completo.

Critérios: nenhum id repetido, nenhum caminho de saída repetido, 29 ontologias presentes.
"""
import json
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import gerar_manifesto as gm

MANIFESTO_COMPLETO = os.path.join(REPO, "configs", "pipeline", "manifesto_7b_completo.json")
PARES = os.path.join(REPO, "configs", "pipeline", "pares_c5.json")


def _gerar():
    with open(PARES, encoding="utf-8") as f:
        pares = json.load(f)
    exp, avisos = gm.gerar_completo(
        ["Qwen/Qwen2.5-7B-Instruct-AWQ"], gm.TODAS_CONDICOES,
        ["tekgen", "webnlg"], pares, url="http://x/v1", git="deadbeef",
    )
    return exp, avisos


class TestGerarCompleto:
    def test_203_experimentos(self):
        exp, _ = _gerar()
        assert len(exp) == 29 * 7 == 203

    def test_ids_unicos(self):
        exp, _ = _gerar()
        ids = [e["id"] for e in exp]
        assert len(ids) == len(set(ids))

    def test_saidas_unicas(self):
        exp, _ = _gerar()
        saidas = [e["saida"] for e in exp]
        assert len(saidas) == len(set(saidas))

    def test_29_ontologias(self):
        exp, _ = _gerar()
        ontos = {(e["dataset"], e["ontologia_id"]) for e in exp}
        assert len(ontos) == 29

    def test_todas_7_condicoes(self):
        exp, _ = _gerar()
        assert {e["condicao"] for e in exp} == set(gm.TODAS_CONDICOES)

    def test_c2_tem_subdir_por_modelo(self):
        exp, _ = _gerar()
        c2 = [e for e in exp if e["condicao"] == "C2"]
        assert len(c2) == 29
        assert all(e["c2_subdir"] == "v_full/qwen2.5-7b" for e in c2)

    def test_c5_tem_parceira(self):
        exp, _ = _gerar()
        c5 = [e for e in exp if e["condicao"].startswith("C5_")]
        assert len(c5) == 29 * 2
        assert all("ontologia_parceira_id" in e for e in c5)

    def test_exigir_c2_omite_sem_induzida(self):
        # Sem ontologias induzidas em v_full/, modo estrito omite as 29 C2.
        with open(PARES, encoding="utf-8") as f:
            pares = json.load(f)
        exp, avisos = gm.gerar_completo(
            ["Qwen/Qwen2.5-7B-Instruct-AWQ"], gm.TODAS_CONDICOES,
            ["tekgen", "webnlg"], pares, url="http://x/v1", git="d", exigir_c2=True,
        )
        assert not any(e["condicao"] == "C2" for e in exp)
        assert len(avisos) == 29


class TestManifestoCompletoCommitado:
    @pytest.mark.skipif(not os.path.exists(MANIFESTO_COMPLETO),
                        reason="manifesto_7b_completo.json ausente")
    def test_arquivo_valido(self):
        with open(MANIFESTO_COMPLETO, encoding="utf-8") as f:
            m = json.load(f)
        assert len(m) == 203
        assert len({e["id"] for e in m}) == 203
        assert len({e["saida"] for e in m}) == 203
        assert len({(e["dataset"], e["ontologia_id"]) for e in m}) == 29
        for e in m:
            assert "\\" not in e["saida"]
            assert e["saida"].startswith("resultados/full/qwen2.5-7b/")
