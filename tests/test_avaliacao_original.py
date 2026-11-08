"""
Testes que travam a reprodução do avaliador Text2KGBench.
Critério de aceitação: métricas globais reproduzidas == métricas publicadas.

Nota sobre encoding: o run_eval.py original usa open() sem encoding,
o que no Linux usa UTF-8 (mesmo comportamento que nosso código).
No Windows, 2 sentenças com caracteres Unicode multi-byte nos dados "unseen"
podem diferir; esses casos estão documentados em DESVIOS_ENCODING.
"""
import json
import os
import sys
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE = os.path.abspath(os.path.join(REPO_ROOT, ".."))
BENCH = os.path.join(WORKSPACE, "Text2KGBench")

sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from t2kg.avaliacao.original import (
    calcular_precisao_recall_f1,
    normalizar_tripla,
    calcular_conformidade_ontologica,
    calcular_alucinacao_sujeito_objeto,
    limpar_entidade,
    avaliar,
)
from nltk.stem import PorterStemmer


# ---------------------------------------------------------------------------
# Testes unitários das funções de métrica
# ---------------------------------------------------------------------------

class TestPrecisaoRecallF1:
    def test_match_exato(self):
        gold = {"abc"}
        pred = {"abc"}
        p, r, f1 = calcular_precisao_recall_f1(gold, pred)
        assert p == 1.0 and r == 1.0 and f1 == 1.0

    def test_sem_predicao(self):
        gold = {"abc"}
        pred = set()
        p, r, f1 = calcular_precisao_recall_f1(gold, pred)
        assert p == 0 and r == 0 and f1 == 0

    def test_sem_intersecao(self):
        gold = {"abc"}
        pred = {"xyz"}
        p, r, f1 = calcular_precisao_recall_f1(gold, pred)
        assert p == 0.0 and r == 0.0 and f1 == 0.0

    def test_precisao_parcial(self):
        gold = {"ab", "cd"}
        pred = {"ab", "ef"}
        p, r, f1 = calcular_precisao_recall_f1(gold, pred)
        assert p == 0.5 and r == 0.5 and f1 == 0.5


class TestNormalizarTripla:
    def test_remove_espacos_e_underscores(self):
        assert normalizar_tripla("Toy Story", "cast_member", "Tom Hanks") == "toystorycastmembertomhanks"

    def test_minusculas(self):
        assert normalizar_tripla("A", "B", "C") == "abc"

    def test_caso_original(self):
        # O ':' NÃO é removido — apenas espaços e underscores são.
        # "Bleach: Hell Verse" → "bleach:hellverse"
        s = normalizar_tripla("Bleach: Hell Verse", "director", "Noriyuki Abe")
        assert s == "bleach:hellversedirectornoriyukiabe"


class TestConformidadeOntologica:
    def _ont(self, labels):
        return {"relations": [{"label": l} for l in labels]}

    def test_tripla_vazia(self):
        ont = self._ont(["parent taxon"])
        conf, rh = calcular_conformidade_ontologica(ont, [])
        assert conf == 1.0 and rh == 0.0

    def test_relacao_conforme(self):
        ont = self._ont(["parent taxon"])
        triplas = [["A", "parent_taxon", "B"]]
        conf, rh = calcular_conformidade_ontologica(ont, triplas)
        assert conf == 1.0 and rh == 0.0

    def test_relacao_nao_conforme(self):
        ont = self._ont(["parent taxon"])
        triplas = [["A", "invented_relation", "B"]]
        conf, rh = calcular_conformidade_ontologica(ont, triplas)
        assert conf == 0.0 and rh == 1.0

    def test_case_sensitive(self):
        ont = self._ont(["Habitat"])
        # "Habitat" → "Habitat" (sem mudança - já sem espaço)
        triplas_ok = [["A", "Habitat", "B"]]
        triplas_nok = [["A", "habitat", "B"]]
        conf_ok, _ = calcular_conformidade_ontologica(ont, triplas_ok)
        conf_nok, _ = calcular_conformidade_ontologica(ont, triplas_nok)
        assert conf_ok == 1.0
        assert conf_nok == 0.0

    def test_espaco_vira_underscore(self):
        # label "mountain range" → rels_ont = ["mountain_range"]
        ont = self._ont(["mountain range"])
        triplas = [["A", "mountain_range", "B"]]
        conf, _ = calcular_conformidade_ontologica(ont, triplas)
        assert conf == 1.0


class TestAlucinacaoSubjetoObjeto:
    def _ont(self):
        return {"concepts": [{"label": "taxon"}, {"label": "human"}]}

    def test_sem_triplas(self):
        ps = PorterStemmer()
        sh, oh = calcular_alucinacao_sujeito_objeto(ps, self._ont(), "sent", [])
        assert sh == 0.0 and oh == 0.0

    def test_sem_alucinacao(self):
        ps = PorterStemmer()
        # "Sacred Kingfisher" está na frase
        frase = "The sacred kingfisher is a bird."
        triplas = [["sacred kingfisher", "habitat", "woodland"]]
        # "woodland" NÃO está na frase nem nos conceitos
        sh, oh = calcular_alucinacao_sujeito_objeto(ps, self._ont(), frase, triplas)
        assert sh == 0.0  # sujeito está na frase
        assert oh == 1.0  # objeto não está

    def test_concatenacao_sem_espaco(self):
        """
        Espelha o comportamento: frase + " ".join(conceitos) SEM espaço entre
        a frase e o primeiro conceito. Então "frase.taxon human".
        """
        ps = PorterStemmer()
        ont = {"concepts": [{"label": "taxon"}]}
        frase = "X is a "  # termina com espaço
        triplas = [["X", "rel", "taxon"]]
        # "taxon" aparece em frase+"taxon" = "X is a taxon"
        sh, oh = calcular_alucinacao_sujeito_objeto(ps, ont, frase, triplas)
        assert oh == 0.0  # taxon está no contexto

    def test_01januari_removido(self):
        ps = PorterStemmer()
        # "01 January 1962" após stemming + normalização vira "01januari1962"
        # então "01januari" é removido e sobra "1962"
        entity = "01 January 1962"
        normalizado = limpar_entidade(ps, entity)
        assert "01januar" not in normalizado


# ---------------------------------------------------------------------------
# Testes de integração: verificam que os números batem com o run_eval.py
# original para os experimentos principais (amostra por dataset).
# ---------------------------------------------------------------------------

def construir_config_tekgen_vicuna():
    onto_list = [
        "1_movie", "2_music", "3_sport", "4_book", "5_military",
        "6_computer", "7_space", "8_politics", "9_nature", "10_culture",
    ]
    items = []
    for onto in onto_list:
        items.append({
            "id": onto,
            "sys": os.path.join(BENCH, "data", "wikidata_tekgen", "baselines", "Vicuna-13B",
                                "llm_responses", f"ont_{onto}_llm_responses.jsonl"),
            "gt": os.path.join(BENCH, "data", "wikidata_tekgen", "ground_truth",
                               f"ont_{onto}_ground_truth.jsonl"),
            "selected_ids": os.path.join(BENCH, "data", "wikidata_tekgen",
                                         "manually_verified_sentences", f"selected_ont_{onto}.txt"),
            "onto": os.path.join(BENCH, "data", "wikidata_tekgen", "ontologies",
                                 f"{onto}_ontology.json"),
            "output": os.path.join(REPO_ROOT, "resultados", "tests", f"tv_{onto}.jsonl"),
        })
    return {
        "onto_list": items,
        "avg_out_file": os.path.join(REPO_ROOT, "resultados", "tests", "tv_avg.jsonl"),
    }


def construir_config_webnlg_vicuna():
    onto_list = [
        "1_university", "1_university", "2_musicalwork", "3_airport", "4_building",
        "5_athlete", "6_politician", "7_company", "8_celestialbody", "9_astronaut",
        "10_comicscharacter", "11_meanoftransportation", "12_monument", "13_food",
        "14_writtenwork", "15_sportsteam", "16_city", "17_artist", "18_scientist", "19_film",
    ]
    items = []
    for onto in onto_list:
        items.append({
            "id": onto,
            "sys": os.path.join(BENCH, "data", "dbpedia_webnlg", "baselines", "Vicuna-13B",
                                "llm_responses", f"{onto}_Vicuna13B_responses.jsonl"),
            "gt": os.path.join(BENCH, "data", "dbpedia_webnlg", "ground_truth",
                               f"ont_{onto}_ground_truth.jsonl"),
            "onto": os.path.join(BENCH, "data", "dbpedia_webnlg", "ontologies",
                                 f"{onto}_ontology.json"),
            "output": os.path.join(REPO_ROOT, "resultados", "tests", f"wv_{onto}.jsonl"),
        })
    return {
        "onto_list": items,
        "avg_out_file": os.path.join(REPO_ROOT, "resultados", "tests", "wv_avg.jsonl"),
    }


def extrair_global(caminho):
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            d = json.loads(linha)
            if d.get("type") == "global":
                return d
    return None


METRICAS = ["avg_precision", "avg_recall", "avg_f1", "avg_onto_conf",
            "avg_sub_halluc", "avg_rel_halluc", "avg_obj_halluc"]

PUBLICADO_TEKGEN_VICUNA = os.path.join(
    BENCH, "data", "wikidata_tekgen", "baselines", "Vicuna-13B",
    "eval_metrics", "ont_llm_avg_stats_v2.jsonl"
)
PUBLICADO_WEBNLG_VICUNA = os.path.join(
    BENCH, "data", "dbpedia_webnlg", "baselines", "Vicuna-13B",
    "eval_metrics", "avg_eval_results.jsonl"
)


@pytest.fixture(scope="module")
def resultado_tekgen_vicuna(tmp_path_factory):
    """Roda avaliação TekGen/Vicuna e retorna o global."""
    tmp = tmp_path_factory.mktemp("tekgen_vicuna")
    config = construir_config_tekgen_vicuna()
    # Redireciona outputs para tmp
    for item in config["onto_list"]:
        item["output"] = str(tmp / f"{item['id']}.jsonl")
    config["avg_out_file"] = str(tmp / "avg.jsonl")

    avaliar(config, diretorio_base="")
    return extrair_global(config["avg_out_file"])


@pytest.fixture(scope="module")
def resultado_webnlg_vicuna(tmp_path_factory):
    """Roda avaliação DBpedia/WebNLG/Vicuna e retorna o global."""
    tmp = tmp_path_factory.mktemp("webnlg_vicuna")
    config = construir_config_webnlg_vicuna()
    for item in config["onto_list"]:
        item["output"] = str(tmp / f"{item['id']}.jsonl")
    config["avg_out_file"] = str(tmp / "avg.jsonl")

    avaliar(config, diretorio_base="")
    return extrair_global(config["avg_out_file"])


@pytest.mark.skipif(not os.path.exists(PUBLICADO_TEKGEN_VICUNA),
                    reason="Benchmark não encontrado")
def test_tekgen_vicuna_global_bate(resultado_tekgen_vicuna):
    publicado = extrair_global(PUBLICADO_TEKGEN_VICUNA)
    assert publicado is not None
    for m in METRICAS:
        assert resultado_tekgen_vicuna[m] == publicado[m], \
            f"TekGen/Vicuna {m}: publicado={publicado[m]}, reproduzido={resultado_tekgen_vicuna[m]}"


@pytest.mark.skipif(not os.path.exists(PUBLICADO_WEBNLG_VICUNA),
                    reason="Benchmark não encontrado")
def test_webnlg_vicuna_global_bate(resultado_webnlg_vicuna):
    publicado = extrair_global(PUBLICADO_WEBNLG_VICUNA)
    assert publicado is not None
    for m in METRICAS:
        assert resultado_webnlg_vicuna[m] == publicado[m], \
            f"DBpedia/Vicuna {m}: publicado={publicado[m]}, reproduzido={resultado_webnlg_vicuna[m]}"
