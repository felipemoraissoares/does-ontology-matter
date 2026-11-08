"""
Testes de integração do pipeline de extração (Fase 1).

Todos os testes usam o ProvedorMock — sem LLM real necessário.
Requerem os dados do Text2KGBench em ../Text2KGBench (irmão do repo).
"""
import json
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE = os.path.abspath(os.path.join(REPO, ".."))
BENCH = os.path.join(WORKSPACE, "Text2KGBench")
sys.path.insert(0, os.path.join(REPO, "src"))

from t2kg.pipeline.dados import carregar_frases
from t2kg.pipeline.ontologias import carregar_ontologia, formatar_contexto
from t2kg.pipeline.prompts import montar_prompt
from t2kg.pipeline.provedores.mock import ProvedorMock
from t2kg.pipeline.executar import executar

SKIP_SEM_BENCH = pytest.mark.skipif(
    not os.path.exists(BENCH), reason="Text2KGBench não encontrado"
)


# ---------------------------------------------------------------------------
# dados.py
# ---------------------------------------------------------------------------

class TestCarregarFrases:
    @SKIP_SEM_BENCH
    def test_tekgen_retorna_frases(self):
        frases = carregar_frases(BENCH, "tekgen", "test", "1_movie")
        assert len(frases) > 0
        primeira = frases[0]
        assert "id" in primeira
        assert "sent" in primeira
        assert isinstance(primeira["gt_triples"], list)

    @SKIP_SEM_BENCH
    def test_exemplo_extraido_para_c1(self):
        frases = carregar_frases(BENCH, "tekgen", "test", "1_movie")
        # A maioria das frases deve ter exemplo extraído do prompt original
        com_exemplo = [f for f in frases if f["exemplo_sent"] is not None]
        assert len(com_exemplo) > len(frases) * 0.9  # > 90% têm exemplo

    @SKIP_SEM_BENCH
    def test_webnlg_retorna_frases(self):
        frases = carregar_frases(BENCH, "webnlg", "test", "1_university")
        assert len(frases) > 0

    @SKIP_SEM_BENCH
    def test_unseen_retorna_frases(self):
        frases = carregar_frases(BENCH, "tekgen", "unseen", "1_movie")
        assert len(frases) > 0


# ---------------------------------------------------------------------------
# ontologias.py
# ---------------------------------------------------------------------------

class TestOntologias:
    @SKIP_SEM_BENCH
    def test_carregar_tekgen(self):
        ont = carregar_ontologia(BENCH, "tekgen", "1_movie")
        assert "relations" in ont
        assert "concepts" in ont

    @SKIP_SEM_BENCH
    def test_c0_retorna_none(self):
        ont = carregar_ontologia(BENCH, "tekgen", "1_movie")
        assert formatar_contexto(ont, "C0") is None

    @SKIP_SEM_BENCH
    def test_c1_contem_relacoes_e_conceitos(self):
        ont = carregar_ontologia(BENCH, "tekgen", "1_movie")
        ctx = formatar_contexto(ont, "C1")
        assert "Ontology Relations:" in ctx
        assert "Ontology Concepts:" in ctx
        assert "director" in ctx

    @SKIP_SEM_BENCH
    def test_c3_so_relacoes_sem_dominio(self):
        ont = carregar_ontologia(BENCH, "tekgen", "1_movie")
        ctx = formatar_contexto(ont, "C3")
        assert "Relations:" in ctx
        assert "Ontology Concepts:" not in ctx
        # C3 não inclui domain/range
        assert "(film," not in ctx

    @SKIP_SEM_BENCH
    def test_c5_e_json_valido(self):
        ont = carregar_ontologia(BENCH, "tekgen", "1_movie")
        ctx = formatar_contexto(ont, "C5")
        parsed = json.loads(ctx)
        assert "relations" in parsed

    def test_condicao_invalida_levanta_erro(self):
        with pytest.raises(ValueError):
            formatar_contexto({}, "C99")


# ---------------------------------------------------------------------------
# prompts.py
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_c0_sem_contexto(self):
        p = montar_prompt("A cat sat on a mat.", "C0")
        assert "A cat sat on a mat." in p
        assert "Ontology" not in p

    def test_c1_tem_contexto(self):
        ctx = "Ontology Concepts: X\nOntology Relations: rel(X,Y)"
        p = montar_prompt("My sentence.", "C1", contexto_ontologia=ctx)
        assert "Ontology Relations:" in p
        assert "My sentence." in p
        assert "Example Sentence:" not in p

    def test_c1_oneshot_tem_exemplo(self):
        ctx = "Ontology Concepts: X\nOntology Relations: rel(X,Y)"
        p = montar_prompt(
            "My sentence.", "C1_oneshot_original",
            contexto_ontologia=ctx,
            exemplo_sent="Example sentence here.",
            exemplo_saida="rel(A, B)",
        )
        assert "Example Sentence: Example sentence here." in p
        assert "Example Output: rel(A, B)" in p
        assert "Test Sentence: My sentence." in p

    def test_c1_oneshot_sem_exemplo_omite_bloco(self):
        ctx = "Ontology Relations: rel(X,Y)"
        p = montar_prompt("My sentence.", "C1_oneshot_original", contexto_ontologia=ctx)
        assert "Example Sentence:" not in p

    def test_c3_lista_relacoes(self):
        ctx = "Relations: director, screenwriter"
        p = montar_prompt("My sentence.", "C3", contexto_ontologia=ctx)
        assert "Relations: director" in p

    def test_c5_instrucao_json(self):
        ctx = '{"relations": []}'
        p = montar_prompt("My sentence.", "C5", contexto_ontologia=ctx)
        assert "JSON" in p


# ---------------------------------------------------------------------------
# executar.py — integração + checkpoint
# ---------------------------------------------------------------------------

class TestExecutar:
    @SKIP_SEM_BENCH
    def test_executa_n_frases(self, tmp_path):
        saida = str(tmp_path / "saida.jsonl")
        n = executar(
            bench_dir=BENCH,
            dataset="tekgen",
            split="test",
            ontologia_id="1_movie",
            condicao="C1_oneshot_original",
            provedor=ProvedorMock(),
            saida=saida,
            frases_max=10,
        )
        assert n == 10
        linhas = [json.loads(l) for l in open(saida, encoding="utf-8") if l.strip()]
        assert len(linhas) == 10
        for linha in linhas:
            assert "id" in linha
            assert "sent" in linha
            assert "response" in linha
            assert isinstance(linha["triples"], list)

    @SKIP_SEM_BENCH
    def test_checkpoint_nao_repete(self, tmp_path):
        saida = str(tmp_path / "saida.jsonl")
        # Primeira execução: 5 frases
        n1 = executar(
            bench_dir=BENCH, dataset="tekgen", split="test",
            ontologia_id="1_movie", condicao="C0",
            provedor=ProvedorMock(), saida=saida, frases_max=5,
        )
        assert n1 == 5

        # Segunda execução: 10 total — deve processar só 5 novas
        n2 = executar(
            bench_dir=BENCH, dataset="tekgen", split="test",
            ontologia_id="1_movie", condicao="C0",
            provedor=ProvedorMock(), saida=saida, frases_max=10,
        )
        assert n2 == 5  # 5 novas (5 já prontas + 5 novas = 10 total)

        linhas = [json.loads(l) for l in open(saida, encoding="utf-8") if l.strip()]
        assert len(linhas) == 10

        # IDs únicos (sem duplicatas)
        ids = [l["id"] for l in linhas]
        assert len(ids) == len(set(ids))

    @SKIP_SEM_BENCH
    def test_condicoes_todas_funcionam(self, tmp_path):
        for cond in ["C0", "C1", "C1_oneshot_original", "C3", "C5"]:
            saida = str(tmp_path / f"{cond}.jsonl")
            n = executar(
                bench_dir=BENCH, dataset="tekgen", split="test",
                ontologia_id="2_music", condicao=cond,
                provedor=ProvedorMock(), saida=saida, frases_max=3,
            )
            assert n == 3, f"Falhou para condição {cond}"
