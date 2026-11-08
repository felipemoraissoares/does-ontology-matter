#!/usr/bin/env python
# Script para reproduzir as métricas dos baselines e comparar com os publicados.
# Uso: python scripts/reproduzir_baselines.py

import json
import os
import sys

PROJETO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_DIR = os.path.abspath(os.path.join(PROJETO_DIR, ".."))
BENCH_DIR = os.path.join(WORKSPACE_DIR, "Text2KGBench")

sys.path.insert(0, os.path.join(PROJETO_DIR, "src"))

from t2kg.avaliacao.original import carregar_config, avaliar


def config_tekgen_vicuna():
    return {
        "onto_list": ["1_movie", "2_music", "3_sport", "4_book", "5_military",
                      "6_computer", "7_space", "8_politics", "9_nature", "10_culture"],
        "path_patterns": {
            "sys": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "baselines", "Vicuna-13B",
                                "llm_responses", "ont_$$onto$$_llm_responses.jsonl"),
            "gt": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "ground_truth",
                               "ont_$$onto$$_ground_truth.jsonl"),
            "selected_ids": os.path.join(BENCH_DIR, "data", "wikidata_tekgen",
                                         "manually_verified_sentences", "selected_ont_$$onto$$.txt"),
            "onto": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "ontologies",
                                 "$$onto$$_ontology.json"),
            "output": os.path.join(PROJETO_DIR, "resultados", "reproducao", "tekgen_vicuna",
                                   "$$onto$$_metricas.jsonl"),
        },
        "avg_out_file": os.path.join(PROJETO_DIR, "resultados", "reproducao", "tekgen_vicuna",
                                     "avg_metricas.jsonl"),
    }


def config_tekgen_alpaca():
    return {
        "onto_list": ["1_movie", "2_music", "3_sport", "4_book", "5_military",
                      "6_computer", "7_space", "8_politics", "9_nature", "10_culture"],
        "path_patterns": {
            "sys": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "baselines", "Alpaca-LoRA-13B",
                                "llm_responses", "ont_$$onto$$_lora-13b_responses_v2.jsonl"),
            "gt": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "ground_truth",
                               "ont_$$onto$$_ground_truth.jsonl"),
            "selected_ids": os.path.join(BENCH_DIR, "data", "wikidata_tekgen",
                                         "manually_verified_sentences", "selected_ont_$$onto$$.txt"),
            "onto": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "ontologies",
                                 "$$onto$$_ontology.json"),
            "output": os.path.join(PROJETO_DIR, "resultados", "reproducao", "tekgen_alpaca",
                                   "$$onto$$_metricas.jsonl"),
        },
        "avg_out_file": os.path.join(PROJETO_DIR, "resultados", "reproducao", "tekgen_alpaca",
                                     "avg_metricas.jsonl"),
    }


def config_webnlg_vicuna():
    # 1_university repete duas vezes — espelha o original
    ontos = ["1_university", "1_university", "2_musicalwork", "3_airport", "4_building",
             "5_athlete", "6_politician", "7_company", "8_celestialbody", "9_astronaut",
             "10_comicscharacter", "11_meanoftransportation", "12_monument", "13_food",
             "14_writtenwork", "15_sportsteam", "16_city", "17_artist", "18_scientist", "19_film"]
    return {
        "onto_list": ontos,
        "path_patterns": {
            "sys": os.path.join(BENCH_DIR, "data", "dbpedia_webnlg", "baselines", "Vicuna-13B",
                                "llm_responses", "$$onto$$_Vicuna13B_responses.jsonl"),
            "gt": os.path.join(BENCH_DIR, "data", "dbpedia_webnlg", "ground_truth",
                               "ont_$$onto$$_ground_truth.jsonl"),
            "onto": os.path.join(BENCH_DIR, "data", "dbpedia_webnlg", "ontologies",
                                 "$$onto$$_ontology.json"),
            "output": os.path.join(PROJETO_DIR, "resultados", "reproducao", "webnlg_vicuna",
                                   "$$onto$$_metricas.jsonl"),
        },
        "avg_out_file": os.path.join(PROJETO_DIR, "resultados", "reproducao", "webnlg_vicuna",
                                     "avg_metricas.jsonl"),
    }


def config_webnlg_alpaca():
    ontos = ["1_university", "1_university", "2_musicalwork", "3_airport", "4_building",
             "5_athlete", "6_politician", "7_company", "8_celestialbody", "9_astronaut",
             "10_comicscharacter", "11_meanoftransportation", "12_monument", "13_food",
             "14_writtenwork", "15_sportsteam", "16_city", "17_artist", "18_scientist", "19_film"]
    return {
        "onto_list": ontos,
        "path_patterns": {
            "sys": os.path.join(BENCH_DIR, "data", "dbpedia_webnlg", "baselines", "Alpaca-LoRA-13B",
                                "llm_responses", "$$onto$$_AlpacaLora13B_responses.jsonl"),
            "gt": os.path.join(BENCH_DIR, "data", "dbpedia_webnlg", "ground_truth",
                               "ont_$$onto$$_ground_truth.jsonl"),
            "onto": os.path.join(BENCH_DIR, "data", "dbpedia_webnlg", "ontologies",
                                 "$$onto$$_ontology.json"),
            "output": os.path.join(PROJETO_DIR, "resultados", "reproducao", "webnlg_alpaca",
                                   "$$onto$$_metricas.jsonl"),
        },
        "avg_out_file": os.path.join(PROJETO_DIR, "resultados", "reproducao", "webnlg_alpaca",
                                     "avg_metricas.jsonl"),
    }


def config_tekgen_vicuna_unseen():
    return {
        "onto_list": ["1_movie", "2_music", "3_sport", "4_book", "5_military",
                      "6_computer", "7_space", "8_politics", "9_nature", "10_culture"],
        "path_patterns": {
            "sys": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "baselines", "Vicuna-13B",
                                "unseen", "llm_responses", "$$onto$$_unseen_Vicuna13B_responses.jsonl"),
            "gt": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "unseen_sentences",
                               "ground_truth", "ont_$$onto$$_unseen_ground_truth.jsonl"),
            "onto": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "ontologies",
                                 "$$onto$$_ontology.json"),
            "output": os.path.join(PROJETO_DIR, "resultados", "reproducao", "tekgen_vicuna_unseen",
                                   "$$onto$$_metricas.jsonl"),
        },
        "avg_out_file": os.path.join(PROJETO_DIR, "resultados", "reproducao", "tekgen_vicuna_unseen",
                                     "avg_metricas.jsonl"),
    }


def config_tekgen_alpaca_unseen():
    return {
        "onto_list": ["1_movie", "2_music", "3_sport", "4_book", "5_military",
                      "6_computer", "7_space", "8_politics", "9_nature", "10_culture"],
        "path_patterns": {
            "sys": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "baselines", "Alpaca-LoRA-13B",
                                "unseen", "llm_responses", "$$onto$$_unseen_Alpaca13B_responses.jsonl"),
            "gt": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "unseen_sentences",
                               "ground_truth", "ont_$$onto$$_unseen_ground_truth.jsonl"),
            "onto": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "ontologies",
                                 "$$onto$$_ontology.json"),
            "output": os.path.join(PROJETO_DIR, "resultados", "reproducao", "tekgen_alpaca_unseen",
                                   "$$onto$$_metricas.jsonl"),
        },
        "avg_out_file": os.path.join(PROJETO_DIR, "resultados", "reproducao", "tekgen_alpaca_unseen",
                                     "avg_metricas.jsonl"),
    }


EXPERIMENTOS = [
    {
        "nome": "TekGen / Vicuna-13B",
        "config_fn": config_tekgen_vicuna,
        "publicado": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "baselines",
                                  "Vicuna-13B", "eval_metrics", "ont_llm_avg_stats_v2.jsonl"),
    },
    {
        "nome": "TekGen / Alpaca-LoRA-13B",
        "config_fn": config_tekgen_alpaca,
        "publicado": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "baselines",
                                  "Alpaca-LoRA-13B", "eval_metrics", "ont_llm_avg_eval_metrics.jsonl"),
    },
    {
        "nome": "DBpedia/WebNLG / Vicuna-13B",
        "config_fn": config_webnlg_vicuna,
        "publicado": os.path.join(BENCH_DIR, "data", "dbpedia_webnlg", "baselines",
                                  "Vicuna-13B", "eval_metrics", "avg_eval_results.jsonl"),
    },
    {
        "nome": "DBpedia/WebNLG / Alpaca-LoRA-13B",
        "config_fn": config_webnlg_alpaca,
        "publicado": os.path.join(BENCH_DIR, "data", "dbpedia_webnlg", "baselines",
                                  "Alpaca-LoRA-13B", "eval_metrics", "avg_eval_results.jsonl"),
    },
    {
        "nome": "TekGen Unseen / Vicuna-13B",
        "config_fn": config_tekgen_vicuna_unseen,
        "publicado": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "baselines",
                                  "Vicuna-13B", "unseen", "eval_metrics", "ont_llm_avg_eval_metrics.jsonl"),
    },
    {
        "nome": "TekGen Unseen / Alpaca-LoRA-13B",
        "config_fn": config_tekgen_alpaca_unseen,
        "publicado": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "baselines",
                                  "Alpaca-LoRA-13B", "unseen", "eval_metrics", "ont_llm_avg_eval_metrics.jsonl"),
    },
]

METRICAS = ["avg_precision", "avg_recall", "avg_f1", "avg_onto_conf",
            "avg_sub_halluc", "avg_rel_halluc", "avg_obj_halluc"]


def ler_jsonl(caminho):
    with open(caminho, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def extrair_global(caminho):
    for linha in ler_jsonl(caminho):
        if linha.get("type") == "global":
            return linha
    return None


def extrair_por_onto(caminho):
    return {l["onto"]: l for l in ler_jsonl(caminho) if l.get("type") == "all_test_cases"}


def comparar(nome, pub_path, rep_path):
    pub_global = extrair_global(pub_path)
    rep_global = extrair_global(rep_path)

    print(f"\n{'='*72}")
    print(f"  {nome}")
    print(f"{'='*72}")

    if pub_global is None or rep_global is None:
        print("  ERRO: não foi possível ler a linha global.")
        return False

    print(f"  {'Métrica':<22} {'Publicado':>12} {'Reproduzido':>12} {'OK?':>6}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*6}")

    tudo_bate = True
    for m in METRICAS:
        pub_val = pub_global.get(m, "?")
        rep_val = rep_global.get(m, "?")
        bate = pub_val == rep_val
        if not bate:
            tudo_bate = False
        status = "OK" if bate else "DIFF"
        print(f"  {m:<22} {pub_val:>12} {rep_val:>12} {status:>6}")

    pub_onto = extrair_por_onto(pub_path)
    rep_onto = extrair_por_onto(rep_path)
    divergencias = []
    for onto_id in pub_onto:
        for m in METRICAS:
            pv = pub_onto[onto_id].get(m)
            rv = rep_onto.get(onto_id, {}).get(m)
            if pv != rv:
                divergencias.append((onto_id, m, pv, rv))

    if divergencias:
        print(f"\n  Divergências por ontologia (primeiras 10):")
        for onto_id, m, pv, rv in divergencias[:10]:
            print(f"    {onto_id:25s} {m}: publicado={pv}, reproduzido={rv}")
        if len(divergencias) > 10:
            print(f"    ... e mais {len(divergencias)-10} divergências.")
    else:
        print(f"\n  Todas as métricas por ontologia: OK")

    return tudo_bate


def main():
    print("Reproduzindo métricas dos baselines Text2KGBench...\n")
    print(f"  Benchmark em:  {BENCH_DIR}")
    print(f"  Projeto em:    {PROJETO_DIR}")

    todos_ok = True
    for exp in EXPERIMENTOS:
        config = exp["config_fn"]()

        # expande caminhos (os paths já são absolutos, expandir $$onto$$)
        config_expandida = {"onto_list": [], "avg_out_file": config["avg_out_file"]}
        for onto in config["onto_list"]:
            onto_data = {"id": onto}
            for chave, padrao in config["path_patterns"].items():
                onto_data[chave] = padrao.replace("$$onto$$", onto)
            config_expandida["onto_list"].append(onto_data)

        avaliar(config_expandida, diretorio_base="")

        ok = comparar(exp["nome"], exp["publicado"], config["avg_out_file"])
        todos_ok = todos_ok and ok

    print(f"\n{'='*72}")
    if todos_ok:
        print("  RESULTADO FINAL: TODOS OS NÚMEROS BATEM.")
    else:
        print("  RESULTADO FINAL: HÁ DIVERGÊNCIAS (veja detalhes acima).")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
