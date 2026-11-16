"""Fase 5: lógica de dados do app Streamlit (classificação de triplas)."""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "app"))

from t2kg.avaliacao.original import normalizar_tripla
import dados_piloto as dp


def _gt_norm(triplas):
    return {normalizar_tripla(*t) for t in triplas}


class TestClassificarSistema:
    def test_acerto_e_sobra(self):
        llm = [["A", "director", "B"], ["A", "genre", "Comedy"]]
        gt_norm = _gt_norm([["A", "director", "B"]])
        res = dict((tuple(tr), s) for tr, s in dp.classificar_sistema(llm, gt_norm))
        assert res[("A", "director", "B")] == "acerto"
        assert res[("A", "genre", "Comedy")] == "sobra"

    def test_erro_de_formato(self):
        ont = {"concepts": [{"label": "film", "qid": "Q1"}], "relations": []}
        llm = [["film", "director", "Noriyuki Abe"]]  # sujeito = nome da classe
        res = dp.classificar_sistema(llm, set(), ontologia=ont)
        assert res[0][1] == "fmt"

    def test_sem_ontologia_nao_marca_fmt(self):
        llm = [["film", "director", "Noriyuki Abe"]]
        res = dp.classificar_sistema(llm, set(), ontologia=None)
        assert res[0][1] == "sobra"


class TestFaltas:
    def test_gt_nao_coberto(self):
        gt = [["A", "director", "B"], ["A", "genre", "Comedy"]]
        sys_norm = {normalizar_tripla("A", "director", "B")}
        falt = dp.faltas(gt, sys_norm)
        assert falt == [["A", "genre", "Comedy"]]


class TestCarregarPiloto:
    def test_lista_ontologias_tekgen(self):
        ontos = dp.listar_ontologias("tekgen")
        assert "1_movie" in ontos and "2_music" in ontos

    def test_carregar_condicao_c1(self):
        recs = dp.carregar_condicao("tekgen", "C1", "1_movie")
        assert len(recs) > 0
        algum = next(iter(recs.values()))
        assert "llm_triples" in algum and "gt_triples" in algum

    def test_condicoes_disponiveis(self):
        conds = dp.condicoes_disponiveis("tekgen", "1_movie")
        assert "C0" in conds and "C1" in conds and "C2" in conds

    def test_tabela_metricas(self):
        linhas = dp.carregar_tabela_metricas()
        assert len(linhas) > 0
        assert "avg_f1" in linhas[0]
