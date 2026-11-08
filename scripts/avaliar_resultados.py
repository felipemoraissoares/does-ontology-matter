"""
Avalia os resultados de uma rodada experimental e produz tabela de métricas.

Lê um manifesto JSON (mesmo formato que gerar_manifesto.py produz) e, para cada
experimento, verifica se o arquivo de saída existe e roda o avaliador.

Produz:
  - Um JSONL de médias por (dataset, condicao) em resultados/avaliacao/
  - Uma tabela em stdout por default; --csv salva em CSV

Uso:
  python scripts/avaliar_resultados.py --manifesto configs/pipeline/manifesto_7b_full.json
  python scripts/avaliar_resultados.py --manifesto ... --csv resultados/avaliacao/tabela.csv
"""
import argparse
import csv
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE  = os.path.abspath(os.path.join(REPO_ROOT, ".."))
BENCH_DIR  = os.path.join(WORKSPACE, "Text2KGBench")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from t2kg.avaliacao.original import avaliar

METRICAS = [
    "avg_precision", "avg_recall", "avg_f1",
    "avg_onto_conf", "avg_sub_halluc", "avg_rel_halluc", "avg_obj_halluc",
    "avg_fmt_error",
]

TEKGEN_ONTOS = [
    "1_movie", "2_music", "3_sport", "4_book", "5_military",
    "6_computer", "7_space", "8_politics", "9_nature", "10_culture",
]
WEBNLG_ONTOS = [
    "1_university", "2_musicalwork", "3_airport", "4_building", "5_athlete",
    "6_politician", "7_company", "8_celestialbody", "9_astronaut", "10_comicscharacter",
    "11_meanoftransportation", "12_monument", "13_food", "14_writtenwork", "15_sportsteam",
    "16_city", "17_artist", "18_scientist", "19_film",
]


def _gt_path(bench_dir, dataset, onto_id):
    if dataset == "tekgen":
        return os.path.join(bench_dir, "data", "wikidata_tekgen", "ground_truth",
                            f"ont_{onto_id}_ground_truth.jsonl")
    return os.path.join(bench_dir, "data", "dbpedia_webnlg", "ground_truth",
                        f"ont_{onto_id}_ground_truth.jsonl")


def _onto_path(bench_dir, dataset, onto_id):
    if dataset == "tekgen":
        return os.path.join(bench_dir, "data", "wikidata_tekgen", "ontologies",
                            f"{onto_id}_ontology.json")
    return os.path.join(bench_dir, "data", "dbpedia_webnlg", "ontologies",
                        f"{onto_id}_ontology.json")


def _selected_path(bench_dir, dataset, onto_id):
    if dataset == "tekgen":
        p = os.path.join(bench_dir, "data", "wikidata_tekgen",
                         "manually_verified_sentences", f"selected_ont_{onto_id}.txt")
        return p if os.path.exists(p) else None
    return None


def avaliar_condicao(experimentos_cond, bench_dir, avaliacao_dir, cond, dataset):
    """
    Monta config para o avaliador e roda para todos os experimentos de uma (cond, dataset).
    Retorna o dict global de métricas.
    """
    onto_list = []
    ontos = TEKGEN_ONTOS if dataset == "tekgen" else WEBNLG_ONTOS
    for onto_id in ontos:
        # Localiza o experimento correspondente
        exp = next((e for e in experimentos_cond
                    if e["ontologia_id"] == onto_id), None)
        if exp is None or not os.path.exists(exp["saida"]):
            continue

        item = {
            "id": onto_id,
            "sys": exp["saida"],
            "gt": _gt_path(bench_dir, dataset, onto_id),
            "onto": _onto_path(bench_dir, dataset, onto_id),
            "output": os.path.join(avaliacao_dir, f"{dataset}_{cond}_{onto_id}.jsonl"),
        }
        sel = _selected_path(bench_dir, dataset, onto_id)
        if sel:
            item["selected_ids"] = sel
        onto_list.append(item)

    if not onto_list:
        return None

    os.makedirs(avaliacao_dir, exist_ok=True)
    config = {
        "onto_list": onto_list,
        "avg_out_file": os.path.join(avaliacao_dir, f"{dataset}_{cond}_avg.jsonl"),
    }
    avaliar(config, diretorio_base="")

    # Lê o global
    with open(config["avg_out_file"], encoding="utf-8") as f:
        for linha in f:
            d = json.loads(linha)
            if d.get("type") == "global":
                return {m: d.get(m) for m in METRICAS}
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifesto", required=True)
    p.add_argument("--bench_dir", default=BENCH_DIR)
    p.add_argument("--avaliacao_dir",
                   default=os.path.join(REPO_ROOT, "resultados", "avaliacao"))
    p.add_argument("--csv", default=None, help="Salva tabela em CSV neste caminho.")
    args = p.parse_args()

    with open(args.manifesto, encoding="utf-8") as f:
        experimentos = json.load(f)

    # Agrupa por (dataset, condicao)
    grupos = {}
    for e in experimentos:
        chave = (e["dataset"], e["condicao"])
        grupos.setdefault(chave, []).append(e)

    linhas = []
    for (dataset, cond), exps in sorted(grupos.items()):
        completos = sum(1 for e in exps if os.path.exists(e["saida"]))
        print(f"\n--- {dataset} / {cond}  ({completos}/{len(exps)} arquivos presentes) ---")
        if completos == 0:
            print("  Nenhum resultado ainda.")
            continue

        resultado = avaliar_condicao(exps, args.bench_dir, args.avaliacao_dir, cond, dataset)
        if resultado is None:
            print("  Avaliacao retornou None.")
            continue

        linha = {"dataset": dataset, "condicao": cond, **resultado}
        linhas.append(linha)

        for m, v in resultado.items():
            if v is not None:
                print(f"  {m}: {v:.4f}")

    if args.csv and linhas:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        cabecalho = ["dataset", "condicao"] + METRICAS
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cabecalho)
            w.writeheader()
            w.writerows(linhas)
        print(f"\nTabela salva em: {args.csv}")

    print("\nConcluído.")


if __name__ == "__main__":
    main()
