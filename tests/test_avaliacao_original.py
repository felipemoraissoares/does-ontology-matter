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
    calcular_taxa_erro_formato,
    limpar_entidade,
    avaliar,
    PLACEHOLDER_RELS,
    PLACEHOLDER_ENTITIES,
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


class TestTaxaErroFormato:
    """Testa calcular_taxa_erro_formato com ontologia fictícia."""

    def _ont(self, labels):
        return {"concepts": [{"label": l} for l in labels], "relations": []}

    def test_triplas_vazias_retorna_zero(self):
        ont = self._ont(["film", "human"])
        assert calcular_taxa_erro_formato(ont, []) == 0.0

    def test_sujeito_exatamente_classe_minusculo(self):
        # "film" (todo minúsculo) é o modelo usando o nome da classe — erro
        ont = self._ont(["film", "human"])
        triplas = [["film", "director", "Noriyuki Abe"]]
        assert calcular_taxa_erro_formato(ont, triplas) == 1.0

    def test_sem_falso_positivo_entidade_capitalizada(self):
        # "Composer" é nome legítimo de entidade (ex.: obra musical), não erro de formato
        ont = self._ont(["composer", "human"])
        triplas = [["Composer", "genre", "Classical"]]
        assert calcular_taxa_erro_formato(ont, triplas) == 0.0

    def test_sem_falso_positivo_book_capitalizado(self):
        # "Book" capitalizado não é erro; "book" minúsculo seria erro
        ont = self._ont(["book", "author"])
        triplas_ok = [["Book", "author", "Some Author"]]
        triplas_err = [["book", "author", "Some Author"]]
        assert calcular_taxa_erro_formato(ont, triplas_ok) == 0.0
        assert calcular_taxa_erro_formato(ont, triplas_err) == 1.0

    def test_sujeito_com_prefixo_dois_pontos(self):
        # "film:Bleach: Hell Verse" → começa com "film:" → erro
        ont = self._ont(["film", "human"])
        triplas = [["film:Bleach: Hell Verse", "director", "human:Noriyuki Abe"]]
        # ambos sujeito e objeto são erros; mas a métrica conta por tripla (1 se qualquer)
        assert calcular_taxa_erro_formato(ont, triplas) == 1.0

    def test_sujeito_com_prefixo_espaco_seguido_de_aspas(self):
        # 'film "Keyboard Cat"' → começa com 'film "' → erro (padrão 3)
        ont = self._ont(["film"])
        triplas = [['film "Keyboard Cat"', "genre", "Animation"]]
        assert calcular_taxa_erro_formato(ont, triplas) == 1.0

    def test_sem_falso_positivo_city_of_god(self):
        # "City of God" é um título legítimo que começa com "city " (sem aspas).
        # Com a classe "city" na ontologia, o padrão 3 restrito NÃO deve flagear.
        ont = self._ont(["city", "film"])
        triplas = [["City of God", "director", "Fernando Meirelles"]]
        assert calcular_taxa_erro_formato(ont, triplas) == 0.0

    def test_sem_falso_positivo_sport_competition(self):
        # "sport competition" começa com "sport " mas é entidade real (sem aspas).
        ont = self._ont(["sport"])
        triplas = [["sport competition", "type", "Athletics"]]
        assert calcular_taxa_erro_formato(ont, triplas) == 0.0

    def test_entidade_correta_sem_erro(self):
        ont = self._ont(["film", "human"])
        triplas = [["Bleach: Hell Verse", "director", "Noriyuki Abe"]]
        assert calcular_taxa_erro_formato(ont, triplas) == 0.0

    def test_mix_erros_e_corretos(self):
        ont = self._ont(["film", "human"])
        triplas = [
            ["Bleach: Hell Verse", "director", "Noriyuki Abe"],   # correto
            ["film", "genre", "Animation"],                        # erro (sujeito = classe)
            ["film:Another Film", "director", "Some Director"],    # erro (prefixo com :)
        ]
        # 2 de 3 triplas com erro → 2/3
        assert abs(calcular_taxa_erro_formato(ont, triplas) - 2/3) < 1e-9

    def test_case_insensitive(self):
        # "Film" (capitalizado) deve ser detectado como classe "film"
        ont = self._ont(["film"])
        triplas = [["Film:Bleach", "director", "someone"]]
        assert calcular_taxa_erro_formato(ont, triplas) == 1.0

    def test_dbpedia_classe_capitalizada_exata(self):
        # DBpedia: classe "Film" (capitalizada), entidade exatamente "Film" → fmt_error
        ont = self._ont(["Film"])
        triplas = [["Film", "director", "Someone"]]
        assert calcular_taxa_erro_formato(ont, triplas) == 1.0

    def test_dbpedia_sem_falso_positivo_entidade_capitalizada_composta(self):
        # DBpedia: classe "Film", entidade "Film Critic" é legítima (não exatamente o label)
        ont = self._ont(["Film"])
        triplas = [["Film Critic", "type", "Person"]]
        assert calcular_taxa_erro_formato(ont, triplas) == 0.0

    def test_dbpedia_multiplas_classes_capitalizadas(self):
        # DBpedia típico: "Person", "City", "Film" como classes
        ont = self._ont(["Person", "City", "Film"])
        triplas_erros = [
            ["Person", "birthPlace", "City"],   # sub e obj são classes → 1 erro por tripla
            ["Film", "director", "Someone"],    # sub é classe → erro
        ]
        assert calcular_taxa_erro_formato(ont, triplas_erros) == 1.0  # 2 de 2 triplas com erro
        triplas_ok = [["Nolan", "birthPlace", "London"]]
        assert calcular_taxa_erro_formato(ont, triplas_ok) == 0.0

    # D-32: placeholders da instrução Variante D

    def test_relacao_placeholder_e_fmt_error(self):
        # "relation_name" como relação é placeholder literal — deve ser fmt_error
        ont = self._ont([])
        triplas = [["Bleach: Hell Verse", "relation_name", "Noriyuki Abe"]]
        assert calcular_taxa_erro_formato(ont, triplas) == 1.0

    def test_relacao_placeholder_case_insensitive(self):
        ont = self._ont([])
        triplas = [["A", "Relation_Name", "B"]]
        assert calcular_taxa_erro_formato(ont, triplas) == 1.0

    def test_entidade_placeholder_sujeito_e_fmt_error(self):
        ont = self._ont([])
        sub = "exact name of the first entity as written in the sentence"
        triplas = [[sub, "director", "Noriyuki Abe"]]
        assert calcular_taxa_erro_formato(ont, triplas) == 1.0

    def test_entidade_placeholder_objeto_e_fmt_error(self):
        ont = self._ont([])
        obj = "exact name of the second entity as written in the sentence"
        triplas = [["Bleach: Hell Verse", "director", obj]]
        assert calcular_taxa_erro_formato(ont, triplas) == 1.0

    def test_placeholder_nao_afeta_tripla_correta(self):
        ont = self._ont(["film"])
        triplas_ok = [["Bleach: Hell Verse", "director", "Noriyuki Abe"]]
        assert calcular_taxa_erro_formato(ont, triplas_ok) == 0.0

    def test_placeholder_rels_e_entities_nao_vazios(self):
        # Garante que os conjuntos de placeholders estão definidos e não estão vazios
        assert "relation_name" in PLACEHOLDER_RELS
        assert len(PLACEHOLDER_ENTITIES) == 2


class TestAvaliarPerOntoMode:
    """Fix 1: modo_amostra deve ser por ontologia, não global."""

    def _make_fixture(self, tmp_path, n_sys, n_gt):
        """Cria sys/gt/onto/output fixtures com n_sys e n_gt sentenças."""
        import json as _json

        onto_data = {
            "concepts": [{"label": "entity"}],
            "relations": [{"label": "rel_a"}, {"label": "rel_b"}],
        }
        onto_path = os.path.join(tmp_path, "onto.json")
        with open(onto_path, "w") as f:
            _json.dump(onto_data, f)

        # GT: n_gt sentenças, cada uma com 1 tripla
        gt_records = [
            {"id": f"s{i}", "sent": f"sent {i}", "triples": [{"sub": "A", "rel": "rel_a", "obj": "B"}]}
            for i in range(n_gt)
        ]
        gt_path = os.path.join(tmp_path, "gt.jsonl")
        with open(gt_path, "w") as f:
            for r in gt_records:
                f.write(_json.dumps(r) + "\n")

        # Sistema: n_sys primeiras sentenças, todas com 1 tripla correta
        sys_records = [
            {"id": f"s{i}", "sent": f"sent {i}", "triples": [["A", "rel_a", "B"]], "response": "rel_a(A, B)"}
            for i in range(n_sys)
        ]
        sys_path = os.path.join(tmp_path, "sys.jsonl")
        with open(sys_path, "w") as f:
            for r in sys_records:
                f.write(_json.dumps(r) + "\n")

        output_path = os.path.join(tmp_path, "output.jsonl")
        avg_path = os.path.join(tmp_path, "avg.jsonl")
        return sys_path, gt_path, onto_path, output_path, avg_path

    def test_per_onto_amostra_usa_denom_sys(self):
        import tempfile, json as _json
        with tempfile.TemporaryDirectory() as tmp:
            sys_p, gt_p, onto_p, out_p, avg_p = self._make_fixture(tmp, n_sys=5, n_gt=10)
            config = {
                "onto_list": [{
                    "id": "onto1",
                    "sys": sys_p, "gt": gt_p, "onto": onto_p,
                    "output": out_p,
                    "modo_amostra": True,  # per-onto override
                }],
                "avg_out_file": avg_p,
            }
            result = avaliar(config, diretorio_base="")
            media = result["por_ontologia"]["onto1"]
            # Com modo_amostra=True e 5/5 acertos, F1 deve ser 1.0 (denom=5)
            assert float(media["avg_f1"]) == 1.0, f"expected 1.0, got {media['avg_f1']}"

    def test_per_onto_completo_usa_denom_gt(self):
        import tempfile, json as _json
        with tempfile.TemporaryDirectory() as tmp:
            sys_p, gt_p, onto_p, out_p, avg_p = self._make_fixture(tmp, n_sys=5, n_gt=10)
            config = {
                "onto_list": [{
                    "id": "onto1",
                    "sys": sys_p, "gt": gt_p, "onto": onto_p,
                    "output": out_p,
                    "modo_amostra": False,  # denom = gt = 10
                }],
                "avg_out_file": avg_p,
            }
            result = avaliar(config, diretorio_base="")
            media = result["por_ontologia"]["onto1"]
            # 5 acertos / 10 GT → F1 = 0.5
            assert abs(float(media["avg_f1"]) - 0.5) < 0.01, f"expected ~0.5, got {media['avg_f1']}"


class TestMetricasSemVazias:
    """Fix 3: métricas de conformidade/alucinação sem respostas vazias."""

    def test_pct_vazias_e_conf_nv(self):
        import tempfile, json as _json
        with tempfile.TemporaryDirectory() as tmp:
            onto_data = {
                "concepts": [{"label": "entity"}],
                "relations": [{"label": "rel_a"}],
            }
            onto_path = os.path.join(tmp, "onto.json")
            with open(onto_path, "w") as f:
                _json.dump(onto_data, f)

            gt_records = [
                {"id": "s0", "sent": "s", "triples": [{"sub": "A", "rel": "rel_a", "obj": "B"}]},
                {"id": "s1", "sent": "s", "triples": [{"sub": "A", "rel": "rel_a", "obj": "B"}]},
                {"id": "s2", "sent": "s", "triples": [{"sub": "A", "rel": "rel_a", "obj": "B"}]},
            ]
            gt_path = os.path.join(tmp, "gt.jsonl")
            with open(gt_path, "w") as f:
                for r in gt_records:
                    f.write(_json.dumps(r) + "\n")

            # s0: conformante, 1 tripla; s1: tripla não conformante; s2: vazia
            sys_records = [
                {"id": "s0", "sent": "s", "triples": [["A", "rel_a", "B"]], "response": "rel_a(A,B)"},
                {"id": "s1", "sent": "s", "triples": [["A", "invented", "B"]], "response": "invented(A,B)"},
                {"id": "s2", "sent": "s", "triples": [], "response": ""},
            ]
            sys_path = os.path.join(tmp, "sys.jsonl")
            with open(sys_path, "w") as f:
                for r in sys_records:
                    f.write(_json.dumps(r) + "\n")

            avg_path = os.path.join(tmp, "avg.jsonl")
            config = {
                "onto_list": [{"id": "o", "sys": sys_path, "gt": gt_path,
                                "onto": onto_path, "output": os.path.join(tmp, "out.jsonl")}],
                "avg_out_file": avg_path,
            }
            result = avaliar(config)
            media = result["por_ontologia"]["o"]

            # pct_vazias: 1 vazia de 3 processadas = 1/3 ≈ 0.33
            assert abs(float(media["pct_vazias"]) - 1/3) < 0.01
            # avg_onto_conf_nv: s0 e s1 têm triplas; s0 conf=1.0, s1 conf=0.0 → média=0.5
            assert abs(float(media["avg_onto_conf_nv"]) - 0.5) < 0.01
            # avg_rel_halluc_nv: s0 hal=0.0, s1 hal=1.0 → média=0.5
            assert abs(float(media["avg_rel_halluc_nv"]) - 0.5) < 0.01


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
