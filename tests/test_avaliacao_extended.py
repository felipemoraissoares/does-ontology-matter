"""
Testes do avaliador estendido (Fase 2, item B).
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from t2kg.avaliacao.extended import (
    prf1_sem_filtro,
    taxa_predicados_unicos,
    redundancia_semantica,
    deduplicar_onto_list,
    sumarizar_estendido,
    avaliar_frase_estendida,
)


def _ont(relacoes):
    return {"concepts": [], "relations": [{"label": r} for r in relacoes]}


class TestPRF1SemFiltro:
    def test_match_exato(self):
        gt = [["A", "director", "B"]]
        sys_ = [["A", "director", "B"]]
        p, r, f = prf1_sem_filtro(gt, sys_)
        assert p == 1.0 and r == 1.0 and f == 1.0

    def test_relacao_inventada_penaliza_precisao(self):
        # gt tem director; sistema extrai invented_relation (não está no gabarito)
        # sem filtro, invented_relation é avaliada e resulta em P=0
        gt = [["A", "director", "B"]]
        sys_ = [["A", "invented_relation", "B"]]
        p, r, f = prf1_sem_filtro(gt, sys_)
        assert p == 0.0 and r == 0.0 and f == 0.0

    def test_sem_sistema(self):
        gt = [["A", "director", "B"]]
        p, r, f = prf1_sem_filtro(gt, [])
        assert p == 0.0 and r == 0.0 and f == 0.0

    def test_relacao_extra_reduz_precisao(self):
        gt = [["A", "director", "B"]]
        sys_ = [["A", "director", "B"], ["A", "invented", "C"]]
        p, r, f = prf1_sem_filtro(gt, sys_)
        assert p == 0.5
        assert r == 1.0


class TestTaxaPredicadosUnicos:
    def test_uma_relacao_unica(self):
        triplas = [["A", "director", "B"], ["C", "director", "D"]]
        assert taxa_predicados_unicos(triplas) == 0.5

    def test_todas_distintas(self):
        triplas = [["A", "director", "B"], ["A", "writer", "C"], ["A", "genre", "D"]]
        assert taxa_predicados_unicos(triplas) == 1.0

    def test_vazio(self):
        assert taxa_predicados_unicos([]) == 0.0

    def test_uma_tripla(self):
        assert taxa_predicados_unicos([["A", "director", "B"]]) == 1.0


class TestDeduplicarOntoList:
    def test_remove_duplicata_1_university(self):
        ontos = [
            {"id": "1_university", "sys": "a"},
            {"id": "1_university", "sys": "b"},
            {"id": "2_musicalwork", "sys": "c"},
        ]
        dedup = deduplicar_onto_list(ontos)
        assert len(dedup) == 2
        assert dedup[0]["id"] == "1_university"
        assert dedup[1]["id"] == "2_musicalwork"

    def test_sem_duplicatas(self):
        ontos = [{"id": "1_movie"}, {"id": "2_music"}]
        assert len(deduplicar_onto_list(ontos)) == 2

    def test_vazio(self):
        assert deduplicar_onto_list([]) == []


class TestSumarizarEstendido:
    def _m(self, pf, rf, ff, pp, rp, fp, fp_frac, tpu):
        return {
            "prf1_sem_filtro": {"precision": pf, "recall": rf, "f1": ff},
            "prf1_projetado": {"precision": pp, "recall": rp, "f1": fp, "fracao_projetada": fp_frac},
            "taxa_predicados_unicos": tpu,
        }

    def test_modo_benchmark(self):
        metricas = [self._m(1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.9, 0.5)]
        s = sumarizar_estendido(metricas, n_total_frases=2, modo_amostra=False)
        assert s["avg_sem_filtro_f1"] == 0.5  # 1.0/2
        assert s["n_frases_total"] == 2

    def test_modo_amostra(self):
        metricas = [self._m(1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.9, 0.5)]
        s = sumarizar_estendido(metricas, n_total_frases=100, modo_amostra=True)
        assert s["avg_sem_filtro_f1"] == 1.0  # 1.0/1


SKIP_EMBS = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("sentence_transformers"),
    reason="sentence-transformers não instalado"
)


class TestRedundanciaSemantica:
    def test_vazio(self):
        r = redundancia_semantica([])
        assert r["valor"] == 0.0
        assert r["n_rotulos"] == 0

    def test_um_rotulo(self):
        r = redundancia_semantica([["A", "director", "B"]])
        assert r["valor"] == 0.0
        assert r["n_rotulos"] == 1
        assert r["n_grupos"] == 1

    @SKIP_EMBS
    def test_sem_sinonimos(self):
        triplas = [["A", "director", "B"], ["X", "birthplace", "Y"]]
        r = redundancia_semantica(triplas, threshold=0.80)
        assert r["n_rotulos"] == 2
        assert r["n_grupos"] == 2
        assert r["valor"] == 0.0

    @SKIP_EMBS
    def test_sinonimos_obvios(self):
        # "director" e "directed_by" devem fundir a threshold baixo
        triplas = [["A", "director", "B"], ["C", "directed_by", "D"]]
        r = redundancia_semantica(triplas, threshold=0.70)
        assert r["n_rotulos"] == 2
        assert "n_grupos" in r
        assert "grupos" in r
        assert r["valor"] is not None


class TestAvaliarFraseEstendida:
    @SKIP_EMBS
    def test_basico(self):
        frase = {"id": "s1", "sent": "Film A was directed by Dir B.", "gt_triples": [["Film A", "director", "Dir B"]]}
        ont = _ont(["director", "screenwriter", "genre"])
        triplas = [["Film A", "director", "Dir B"]]
        r = avaliar_frase_estendida(frase, triplas, ont)
        assert r["id"] == "s1"
        assert r["prf1_sem_filtro"]["f1"] == 1.0
        assert r["taxa_predicados_unicos"] == 1.0

    def test_estrutura_retorno(self):
        frase = {"id": "s1", "sent": "X.", "gt_triples": []}
        ont = _ont([])
        try:
            r = avaliar_frase_estendida(frase, [], ont)
            assert "id" in r
            assert "prf1_sem_filtro" in r
            assert "taxa_predicados_unicos" in r
        except ImportError:
            pytest.skip("sentence-transformers não instalado")
