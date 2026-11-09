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
  python scripts/avaliar_resultados.py --manifesto ... --extended   # inclui C0 projetada

Detecção automática de modo_amostra:
  Sempre que o arquivo de saída do sistema tiver menos registros que o gabarito,
  o modo_amostra=True é ativado automaticamente (denominador = frases avaliadas).
  A flag --auto_modo_amostra_manifesto lê o campo modo_amostra do manifesto em vez disso.
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

from t2kg.avaliacao.original import avaliar, ler_jsonl, ler_json

METRICAS = [
    "avg_precision", "avg_recall", "avg_f1",
    "avg_onto_conf", "avg_sub_halluc", "avg_rel_halluc", "avg_obj_halluc",
    "avg_fmt_error", "avg_parse_error",
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


def _contar_registros(caminho):
    """Conta linhas não-vazias num JSONL."""
    if not os.path.exists(caminho):
        return 0
    return sum(1 for l in open(caminho, encoding="utf-8") if l.strip())


def _detectar_modo_amostra(experimentos, bench_dir, dataset):
    """
    Detecta automaticamente se os arquivos de saída cobrem menos registros que o GT.
    Retorna (modo_amostra: bool, n_sys: int, n_gt: int).
    Verifica o primeiro experimento com arquivo de saída presente.
    """
    for exp in experimentos:
        saida = exp.get("saida", "")
        if not os.path.exists(saida):
            continue
        n_sys = _contar_registros(saida)
        gt = _gt_path(bench_dir, dataset, exp["ontologia_id"])
        n_gt = _contar_registros(gt)
        if n_sys > 0 and n_gt > 0:
            return (n_sys < n_gt), n_sys, n_gt
    return False, 0, 0


def avaliar_condicao(experimentos_cond, bench_dir, avaliacao_dir, cond, dataset,
                     modo_amostra: bool = False):
    """
    Monta config para o avaliador e roda para todos os experimentos de uma (cond, dataset).

    modo_amostra=True  → denominador = frases efetivamente avaliadas (correto para amostras).
    modo_amostra=False → denominador = total do GT (reproduz métrica do benchmark publicado).

    Retorna o dict global de métricas.
    """
    onto_list = []
    ontos = TEKGEN_ONTOS if dataset == "tekgen" else WEBNLG_ONTOS
    for onto_id in ontos:
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
    avaliar(config, diretorio_base="", modo_amostra=modo_amostra)

    with open(config["avg_out_file"], encoding="utf-8") as f:
        for linha in f:
            d = json.loads(linha)
            if d.get("type") == "global":
                return {m: d.get(m) for m in METRICAS}
    return None


def avaliar_condicao_extendida(experimentos_cond, bench_dir, cond, dataset,
                                modo_amostra: bool = False):
    """
    Calcula métricas estendidas (prf1_projetado, prf1_sem_filtro, redundancia) para
    uma condição. Retorna dict com médias por ontologia e global, ou None se indisponível.

    Requer sentence-transformers; retorna None silenciosamente se ausente.
    """
    try:
        from t2kg.avaliacao.extended import avaliar_frase_estendida, sumarizar_estendido
    except ImportError:
        return None

    ontos = TEKGEN_ONTOS if dataset == "tekgen" else WEBNLG_ONTOS
    resultados_onto = {}

    soma_global = {
        "avg_sem_filtro_f1": 0.0,
        "avg_projetado_f1": 0.0,
        "avg_fracao_projetada": 0.0,
        "avg_redundancia_esquema": 0.0,
    }
    n_onto_ok = 0

    for onto_id in ontos:
        exp = next((e for e in experimentos_cond
                    if e["ontologia_id"] == onto_id), None)
        if exp is None or not os.path.exists(exp["saida"]):
            continue

        saida_sistema = {r["id"]: r for r in ler_jsonl(exp["saida"])}
        gabarito_raw  = ler_jsonl(_gt_path(bench_dir, dataset, onto_id))
        gabarito      = {r["id"]: r for r in gabarito_raw}
        ontologia     = ler_json(_onto_path(bench_dir, dataset, onto_id))

        metricas_por_frase = []
        for sent_id, gt_rec in gabarito.items():
            if sent_id not in saida_sistema:
                continue
            gt_triplas = [[t["sub"], t["rel"], t["obj"]] for t in gt_rec["triples"]]
            sys_triplas = saida_sistema[sent_id]["triples"]
            frase_dict = {"id": sent_id, "sent": gt_rec["sent"], "gt_triples": gt_triplas}
            metricas_por_frase.append(
                avaliar_frase_estendida(frase_dict, sys_triplas, ontologia)
            )

        n_gt = len(gabarito)
        resumo = sumarizar_estendido(metricas_por_frase, n_gt, modo_amostra=modo_amostra)
        resultados_onto[onto_id] = resumo

        for k in soma_global:
            soma_global[k] += resumo.get(k, 0.0)
        n_onto_ok += 1

    if n_onto_ok == 0:
        return None

    global_ext = {k: v / n_onto_ok for k, v in soma_global.items()}
    global_ext["n_ontologias"] = n_onto_ok
    return {"global": global_ext, "por_ontologia": resultados_onto}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifesto", required=True)
    p.add_argument("--bench_dir", default=BENCH_DIR)
    p.add_argument("--avaliacao_dir",
                   default=os.path.join(REPO_ROOT, "resultados", "avaliacao"))
    p.add_argument("--csv", default=None, help="Salva tabela em CSV neste caminho.")
    p.add_argument(
        "--auto_modo_amostra", action="store_true",
        help="(Legado) Lê modo_amostra do manifesto. Auto-detecção por contagem de "
             "registros agora é o padrão e não requer flag.",
    )
    p.add_argument(
        "--extended", action="store_true",
        help="Inclui métricas estendidas: P/R/F1 sem filtro, projeção de relações (C0/C2/C3), "
             "redundância de esquema. Requer sentence-transformers.",
    )
    args = p.parse_args()

    with open(args.manifesto, encoding="utf-8") as f:
        experimentos = json.load(f)

    grupos = {}
    for e in experimentos:
        chave = (e["dataset"], e["condicao"])
        grupos.setdefault(chave, []).append(e)

    linhas = []
    for (dataset, cond), exps in sorted(grupos.items()):
        completos = sum(1 for e in exps if os.path.exists(e.get("saida", "")))
        print(f"\n--- {dataset} / {cond}  ({completos}/{len(exps)} experimentos com saída) ---")
        if completos == 0:
            print("  Nenhum resultado ainda.")
            continue

        # Auto-detecção por contagem de registros (padrão)
        detectado, n_sys, n_gt = _detectar_modo_amostra(exps, args.bench_dir, dataset)

        # Legado: lê do manifesto se --auto_modo_amostra passado
        modo_manifesto = bool(exps[0].get("modo_amostra", False)) if args.auto_modo_amostra else False

        modo_amostra = detectado or modo_manifesto
        if detectado:
            print(f"  Modo: AMOSTRA — auto-detectado (sys={n_sys} < gt={n_gt})")
        elif modo_manifesto:
            print(f"  Modo: AMOSTRA — lido do manifesto")
        else:
            n_gt_total = n_gt or _contar_registros(
                _gt_path(args.bench_dir, dataset, exps[0]["ontologia_id"])
            )
            print(f"  Modo: COMPLETO (gt={n_gt_total})")

        resultado = avaliar_condicao(
            exps, args.bench_dir, args.avaliacao_dir, cond, dataset,
            modo_amostra=modo_amostra,
        )
        if resultado is None:
            print("  Avaliação retornou None.")
            continue

        resultado_float = {m: (float(v) if v is not None else None) for m, v in resultado.items()}
        linha = {"dataset": dataset, "condicao": cond, "modo_amostra": modo_amostra,
                 **resultado_float}
        linhas.append(linha)

        for m, v in resultado_float.items():
            if v is not None:
                print(f"  {m}: {v:.4f}")

        # Métricas estendidas (opcional)
        if args.extended:
            ext = avaliar_condicao_extendida(
                exps, args.bench_dir, cond, dataset, modo_amostra=modo_amostra
            )
            if ext:
                g = ext["global"]
                print(f"  [ext] avg_sem_filtro_f1:    {g['avg_sem_filtro_f1']:.4f}")
                print(f"  [ext] avg_projetado_f1:     {g['avg_projetado_f1']:.4f}")
                print(f"  [ext] avg_fracao_projetada: {g['avg_fracao_projetada']:.4f}")
                print(f"  [ext] avg_redundancia:      {g['avg_redundancia_esquema']:.4f}")
                # Acrescenta ao CSV
                linha["ext_sem_filtro_f1"]    = g["avg_sem_filtro_f1"]
                linha["ext_projetado_f1"]     = g["avg_projetado_f1"]
                linha["ext_fracao_projetada"] = g["avg_fracao_projetada"]
            elif cond in ("C0", "C2", "C3"):
                print("  [ext] sentence-transformers não disponível — projeção ignorada.")

    if args.csv and linhas:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        cabecalho = ["dataset", "condicao", "modo_amostra"] + METRICAS
        if args.extended:
            cabecalho += ["ext_sem_filtro_f1", "ext_projetado_f1", "ext_fracao_projetada"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cabecalho, extrasaction="ignore")
            w.writeheader()
            w.writerows(linhas)
        print(f"\nTabela salva em: {args.csv}")

    print("\nConcluído.")


if __name__ == "__main__":
    main()
