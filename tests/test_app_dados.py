"""Camada de dados do app novo (app/dados/). Sem Streamlit."""
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(REPO, "src"))

from dados import descoberta, frases, notas, base, grafo, decisoes  # noqa: E402

WORKSPACE = os.path.abspath(os.path.join(REPO, ".."))
BENCH = os.path.join(WORKSPACE, "Text2KGBench")
SKIP_SEM_BENCH = pytest.mark.skipif(not os.path.exists(BENCH), reason="Text2KGBench ausente")


class TestDescoberta:
    def test_piloto_presente(self):
        ids = [r["id"] for r in descoberta.rodadas()]
        assert "piloto-v3" in ids

    def test_experimentos_piloto(self):
        exps = descoberta.experimentos("piloto-v3")
        assert any(e["onto"] == "1_movie" and e["cond"] == "C1" for e in exps)
        assert any(e["cond"] == "C1_oneshot_original" for e in exps)
        assert all(e["n_linhas"] > 0 for e in exps)

    def test_full_descoberto_se_existir(self):
        r = descoberta.rodada("full/qwen2.5-7b")
        if r is None:
            pytest.skip("rodada full ainda não presente")
        exps = descoberta.experimentos("full/qwen2.5-7b")
        assert any(e["dataset"] == "tekgen" and e["onto"] == "1_movie" for e in exps)


@pytest.fixture(scope="module")
def fr():
    return frases.frase("piloto-v3", "tekgen", "1_movie", "C1", "ont_1_movie_test_37")


@SKIP_SEM_BENCH
class TestClassificacao:
    def test_acerto_erro_fora(self, fr):
        assert fr is not None
        cls = {(t["tripla"][0], t["tripla"][1]): t["classe"] for t in fr["classificadas"]}
        assert cls[("Castle in the Sky", "director")] == "acerto"
        assert cls[("Castle in the Sky", "screenwriter")] == "acerto"
        assert cls[("Castle in the Sky", "publication_date")] == "erro"
        assert cls[("Castle in the Sky", "country_of_origin")] == "fora_do_gabarito"

    def test_genre_faltou(self, fr):
        assert "genre" in {t[1] for t in fr["faltas"]}


@SKIP_SEM_BENCH
class TestContext:
    def test_separa_context(self):
        fr = frases.frase("piloto-v3", "tekgen", "1_movie", "C1", "ont_1_movie_test_37")
        partes = fr["prompt_partes"]
        assert "Ontology Relations" in partes["contexto"]
        assert "Test Sentence:" not in partes["contexto"]
        assert partes["teste"].startswith("Test Sentence:")

    def test_c0_sem_context(self):
        fr = frases.frase("piloto-v3", "tekgen", "1_movie", "C0", "ont_1_movie_test_37")
        assert fr["prompt_partes"]["contexto"] == ""


@SKIP_SEM_BENCH
class TestNotas:
    def test_calculado_bate_com_arquivo(self):
        nt = notas.notas_experimento("piloto-v3", "tekgen", "1_movie", "C1")
        assert "ont_1_movie_test_37" in nt
        do_arquivo = nt["ont_1_movie_test_37"]
        assert do_arquivo["fonte"] == "arquivo"
        fr = frases.frase("piloto-v3", "tekgen", "1_movie", "C1", "ont_1_movie_test_37")
        ont = frases._ont_ref("tekgen", "1_movie")
        calc = notas._calcular_frase(fr["sent"], fr["triples"], fr["gt_triplas"], ont, fr["response"])
        for m in ("precision", "recall", "f1", "onto_conf", "fmt_error"):
            assert abs(calc[m] - do_arquivo[m]) < 0.015, m


@SKIP_SEM_BENCH
class TestGrafoExport:
    def test_exportar_arestas(self):
        ar = grafo.exportar_arestas("piloto-v3", "tekgen", "1_movie", "C1")
        assert ar and all({"frase", "origem", "rel", "destino", "classe"} <= set(a) for a in ar)


class TestDecisoes:
    def test_parse_d01_d37(self):
        t = decisoes.todas()
        assert "D-01" in t and "D-37" in t
        assert t["D-37"]["titulo"]
        assert "0.70" in t["D-37"]["corpo"] or "0,70" in t["D-37"]["corpo"]
