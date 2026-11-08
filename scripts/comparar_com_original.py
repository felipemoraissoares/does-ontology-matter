#!/usr/bin/env python
"""
Compara a saída de avaliacao/original.py contra o run_eval.py do Text2KGBench.
Ambos são executados sobre os mesmos arquivos de entrada.
Critério de aceitação: todas as métricas globais e por ontologia devem ser iguais.
"""
import json
import os
import sys
import subprocess

PROJETO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_DIR = os.path.abspath(os.path.join(PROJETO_DIR, ".."))
BENCH_DIR = os.path.join(WORKSPACE_DIR, "Text2KGBench")
EVAL_DIR = os.path.join(BENCH_DIR, "src", "evaluation")
TEMP_DIR = os.path.join(PROJETO_DIR, "resultados", "reproducao", "_orig_temp")

sys.path.insert(0, os.path.join(PROJETO_DIR, "src"))

from t2kg.avaliacao.original import avaliar

os.makedirs(TEMP_DIR, exist_ok=True)

METRICAS = ["avg_precision", "avg_recall", "avg_f1", "avg_onto_conf",
            "avg_sub_halluc", "avg_rel_halluc", "avg_obj_halluc"]

EXPERIMENTOS = [
    {
        "nome": "TekGen / Vicuna-13B",
        "onto_list": ["1_movie", "2_music", "3_sport", "4_book", "5_military",
                      "6_computer", "7_space", "8_politics", "9_nature", "10_culture"],
        "sys_tpl": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "baselines", "Vicuna-13B",
                                "llm_responses", "ont_$$onto$$_llm_responses.jsonl"),
        "gt_tpl": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "ground_truth",
                               "ont_$$onto$$_ground_truth.jsonl"),
        "sel_tpl": os.path.join(BENCH_DIR, "data", "wikidata_tekgen",
                                "manually_verified_sentences", "selected_ont_$$onto$$.txt"),
        "onto_tpl": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "ontologies",
                                 "$$onto$$_ontology.json"),
    },
    {
        "nome": "TekGen / Alpaca-LoRA-13B",
        "onto_list": ["1_movie", "2_music", "3_sport", "4_book", "5_military",
                      "6_computer", "7_space", "8_politics", "9_nature", "10_culture"],
        "sys_tpl": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "baselines", "Alpaca-LoRA-13B",
                                "llm_responses", "ont_$$onto$$_lora-13b_responses_v2.jsonl"),
        "gt_tpl": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "ground_truth",
                               "ont_$$onto$$_ground_truth.jsonl"),
        "sel_tpl": os.path.join(BENCH_DIR, "data", "wikidata_tekgen",
                                "manually_verified_sentences", "selected_ont_$$onto$$.txt"),
        "onto_tpl": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "ontologies",
                                 "$$onto$$_ontology.json"),
    },
    {
        "nome": "DBpedia/WebNLG / Vicuna-13B",
        "onto_list": ["1_university", "1_university", "2_musicalwork", "3_airport", "4_building",
                      "5_athlete", "6_politician", "7_company", "8_celestialbody", "9_astronaut",
                      "10_comicscharacter", "11_meanoftransportation", "12_monument", "13_food",
                      "14_writtenwork", "15_sportsteam", "16_city", "17_artist", "18_scientist", "19_film"],
        "sys_tpl": os.path.join(BENCH_DIR, "data", "dbpedia_webnlg", "baselines", "Vicuna-13B",
                                "llm_responses", "$$onto$$_Vicuna13B_responses.jsonl"),
        "gt_tpl": os.path.join(BENCH_DIR, "data", "dbpedia_webnlg", "ground_truth",
                               "ont_$$onto$$_ground_truth.jsonl"),
        "onto_tpl": os.path.join(BENCH_DIR, "data", "dbpedia_webnlg", "ontologies",
                                 "$$onto$$_ontology.json"),
    },
    {
        "nome": "DBpedia/WebNLG / Alpaca-LoRA-13B",
        "onto_list": ["1_university", "1_university", "2_musicalwork", "3_airport", "4_building",
                      "5_athlete", "6_politician", "7_company", "8_celestialbody", "9_astronaut",
                      "10_comicscharacter", "11_meanoftransportation", "12_monument", "13_food",
                      "14_writtenwork", "15_sportsteam", "16_city", "17_artist", "18_scientist", "19_film"],
        "sys_tpl": os.path.join(BENCH_DIR, "data", "dbpedia_webnlg", "baselines", "Alpaca-LoRA-13B",
                                "llm_responses", "$$onto$$_AlpacaLora13B_responses.jsonl"),
        "gt_tpl": os.path.join(BENCH_DIR, "data", "dbpedia_webnlg", "ground_truth",
                               "ont_$$onto$$_ground_truth.jsonl"),
        "onto_tpl": os.path.join(BENCH_DIR, "data", "dbpedia_webnlg", "ontologies",
                                 "$$onto$$_ontology.json"),
    },
    {
        "nome": "TekGen Unseen / Vicuna-13B",
        "onto_list": ["1_movie", "2_music", "3_sport", "4_book", "5_military",
                      "6_computer", "7_space", "8_politics", "9_nature", "10_culture"],
        "sys_tpl": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "baselines", "Vicuna-13B",
                                "unseen", "llm_responses", "$$onto$$_unseen_Vicuna13B_responses.jsonl"),
        "gt_tpl": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "unseen_sentences",
                               "ground_truth", "ont_$$onto$$_unseen_ground_truth.jsonl"),
        "onto_tpl": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "ontologies",
                                 "$$onto$$_ontology.json"),
    },
    {
        "nome": "TekGen Unseen / Alpaca-LoRA-13B",
        "onto_list": ["1_movie", "2_music", "3_sport", "4_book", "5_military",
                      "6_computer", "7_space", "8_politics", "9_nature", "10_culture"],
        "sys_tpl": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "baselines", "Alpaca-LoRA-13B",
                                "unseen", "llm_responses", "$$onto$$_unseen_Alpaca13B_responses.jsonl"),
        "gt_tpl": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "unseen_sentences",
                               "ground_truth", "ont_$$onto$$_unseen_ground_truth.jsonl"),
        "onto_tpl": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "ontologies",
                                 "$$onto$$_ontology.json"),
    },
]


def construir_config(exp, slug):
    """Monta a config dicionário com caminhos absolutos."""
    onto_list_expandida = []
    path_patterns = {
        "sys": exp["sys_tpl"],
        "gt": exp["gt_tpl"],
        "onto": exp["onto_tpl"],
        "output": os.path.join(TEMP_DIR, slug + "_$$onto$$_metricas.jsonl"),
    }
    if "sel_tpl" in exp:
        path_patterns["selected_ids"] = exp["sel_tpl"]

    for onto in exp["onto_list"]:
        onto_data = {"id": onto}
        for chave, padrao in path_patterns.items():
            onto_data[chave] = padrao.replace("$$onto$$", onto)
        onto_list_expandida.append(onto_data)

    return {
        "onto_list": onto_list_expandida,
        "avg_out_file": os.path.join(TEMP_DIR, slug + "_avg.jsonl"),
    }


def construir_config_orig(exp, slug):
    """Constrói um JSON de config no formato do run_eval.py original."""
    # O run_eval.py usa caminhos relativos a partir do diretório src/evaluation
    # Vamos usar caminhos absolutos — o run_eval.py aceita isso
    patterns = {
        "sys": exp["sys_tpl"],
        "gt": exp["gt_tpl"],
        "onto": exp["onto_tpl"],
        "output": os.path.join(TEMP_DIR, slug + "_orig_$$onto$$.jsonl"),
    }
    if "sel_tpl" in exp:
        patterns["selected_ids"] = exp["sel_tpl"]
    return {
        "onto_list": exp["onto_list"],
        "path_patterns": patterns,
        "avg_out_file": os.path.join(TEMP_DIR, slug + "_orig_avg.jsonl"),
    }


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


def comparar_arquivos(nome, path_orig, path_rep):
    orig_global = extrair_global(path_orig)
    rep_global = extrair_global(path_rep)

    print(f"\n{'='*72}")
    print(f"  {nome}")
    print(f"{'='*72}")

    if orig_global is None or rep_global is None:
        print("  ERRO: linha global não encontrada.")
        return False

    tudo_ok = True
    for m in METRICAS:
        ov = orig_global.get(m, "?")
        rv = rep_global.get(m, "?")
        bate = ov == rv
        if not bate:
            tudo_ok = False
        status = "OK" if bate else f"DIFF (orig={ov}, rep={rv})"
        print(f"  {m:<22} {status}")

    orig_onto = extrair_por_onto(path_orig)
    rep_onto = extrair_por_onto(path_rep)
    divs = [(o, m, orig_onto[o].get(m), rep_onto.get(o, {}).get(m))
            for o in orig_onto for m in METRICAS
            if orig_onto[o].get(m) != rep_onto.get(o, {}).get(m)]
    if divs:
        tudo_ok = False
        print(f"\n  Divergências por ontologia ({len(divs)}):")
        for o, m, ov, rv in divs[:5]:
            print(f"    {o:25s} {m}: orig={ov}, rep={rv}")
    else:
        print("  Por ontologia: OK")

    return tudo_ok


def main():
    print("Comparando avaliacao/original.py contra run_eval.py original...")
    print(f"  Temp dir: {TEMP_DIR}\n")

    # Escreve o run_eval.py original path
    orig_script = os.path.join(EVAL_DIR, "run_eval.py")

    todos_ok = True
    for i, exp in enumerate(EXPERIMENTOS):
        slug = f"exp{i}"
        config_rep = construir_config(exp, slug)
        config_orig_dict = construir_config_orig(exp, slug)

        # Roda minha implementação
        avaliar(config_rep, diretorio_base="")

        # Salva config temporária para o original
        config_orig_path = os.path.join(TEMP_DIR, f"{slug}_orig_config.json")
        with open(config_orig_path, "w", encoding="utf-8") as f:
            json.dump(config_orig_dict, f)

        # Roda o original
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            ["python", orig_script, "--eval_config_path", config_orig_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace", env=env
        )
        if result.returncode != 0 and "UnicodeEncodeError" not in result.stderr:
            print(f"  Erro ao rodar original para {exp['nome']}:")
            print("  ", result.stderr[:300])

        ok = comparar_arquivos(exp["nome"], config_orig_dict["avg_out_file"], config_rep["avg_out_file"])
        todos_ok = todos_ok and ok

    print(f"\n{'='*72}")
    if todos_ok:
        print("  RESULTADO: avaliacao/original.py BATE com run_eval.py em todos os cenários.")
    else:
        print("  RESULTADO: HÁ DIVERGÊNCIAS (veja detalhes acima).")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
