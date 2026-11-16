"""Recalibra os limiares da C2 sobre as 29 ontologias (CPU, após a indução no Kaggle).

Aplica as MESMAS regras pré-registradas, agora sobre as 29 ontologias induzidas do modelo
em configs/c2/<c2_base>/<slug>/:

  - Agrupamento (D-21): threshold único = max(breakpoints) + 0.05, onde o breakpoint de
    uma ontologia é a maior similaridade entre duas relações de REFERÊNCIA (o primeiro par
    que se fundiria). Reportado para embedding só-rótulo e rótulo+definição (D-24).
  - Projeção (D-30): menor threshold em [0.40, 0.90] cuja precisão agregada nos pares de
    treino alinhados seja >= 0.80 (prioriza precisão).

O script APENAS calcula e mostra os limiares. Registrar a decisão é tarefa do autor.

Uso (no PC, depois do Kaggle):
  python scripts/recalibrar_c2_29.py --slug qwen2.5-7b
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE = os.path.abspath(os.path.join(REPO, ".."))
BENCH = os.path.join(WORKSPACE, "Text2KGBench")
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from t2kg.avaliacao.extended import _normalizar_rotulo, _get_modelo, normalizar_chave_definicao
from sklearn.metrics.pairwise import cosine_similarity as _cos
# Reaproveita a maquinaria de alinhamento/calibração de projeção do piloto
from calibrar_limiar_projecao import (
    load_gt_train, load_free_train, build_alignment_pairs, calibrate, THRESHOLDS,
)

TEK = ["1_movie", "2_music", "3_sport", "4_book", "5_military", "6_computer",
       "7_space", "8_politics", "9_nature", "10_culture"]
WEB = ["1_university", "2_musicalwork", "3_airport", "4_building", "5_athlete",
       "6_politician", "7_company", "8_celestialbody", "9_astronaut", "10_comicscharacter",
       "11_meanoftransportation", "12_monument", "13_food", "14_writtenwork", "15_sportsteam",
       "16_city", "17_artist", "18_scientist", "19_film"]


def _train_jsonl(ds, onto):
    sub = "wikidata_tekgen" if ds == "tekgen" else "dbpedia_webnlg"
    return os.path.join(BENCH, "data", sub, "train", f"ont_{onto}_train.jsonl")


def _onto_json(ds, onto):
    sub = "wikidata_tekgen" if ds == "tekgen" else "dbpedia_webnlg"
    return os.path.join(BENCH, "data", sub, "ontologies", f"{onto}_ontology.json")


def _max_sim_ref(onto_rels, defs, usar_def):
    """Maior similaridade entre duas relações de referência (breakpoint D-21)."""
    if len(onto_rels) < 2:
        return 0.0
    modelo = _get_modelo()
    if usar_def and defs:
        textos = [f"{_normalizar_rotulo(r)}: {defs[normalizar_chave_definicao(r)]}"
                  if normalizar_chave_definicao(r) in defs else _normalizar_rotulo(r)
                  for r in onto_rels]
    else:
        textos = [_normalizar_rotulo(r) for r in onto_rels]
    embs = modelo.encode(textos, show_progress_bar=False)
    sim = _cos(embs, embs)
    np.fill_diagonal(sim, -1.0)
    return float(sim.max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slug", default="qwen2.5-7b")
    p.add_argument("--c2_base", default="v_full")
    args = p.parse_args()

    c2_dir = os.path.join(REPO, "configs", "c2", args.c2_base, args.slug)
    raw_dir = os.path.join(REPO, "resultados", "c2_inducao", args.c2_base, args.slug)
    if not os.path.isdir(c2_dir):
        print(f"ERRO: ontologias induzidas não encontradas em {c2_dir}. "
              f"Rode a indução no Kaggle (Etapa C) primeiro.", file=sys.stderr)
        sys.exit(1)

    breakpoints_label, breakpoints_def = [], []
    todas_linhas = []
    print(f"{'ontologia':>26} | {'nref':>4} | {'bp(label)':>9} | {'bp(def)':>9} | {'pares':>5}")
    print("-" * 70)

    for ds, ontos in (("tekgen", TEK), ("webnlg", WEB)):
        for onto in ontos:
            induced = os.path.join(c2_dir, f"{ds}_{onto}_induced.json")
            defs_p = os.path.join(c2_dir, f"definicoes_{onto}.json")
            raw = os.path.join(raw_dir, f"raw_{ds}_{onto}_treino.jsonl")
            frases_p = os.path.join(c2_dir, f"{ds}_{onto}_frases_seed42.json")
            if not (os.path.exists(induced) and os.path.exists(defs_p)):
                print(f"{ds+'/'+onto:>26} | AUSENTE (indução incompleta)")
                continue

            with open(_onto_json(ds, onto), encoding="utf-8") as f:
                onto_rels = [r["label"] for r in json.load(f)["relations"]]
            with open(defs_p, encoding="utf-8") as f:
                defs_raw = json.load(f).get("relacoes", {})
            defs = {normalizar_chave_definicao(k): v for k, v in defs_raw.items()}

            bp_l = _max_sim_ref(onto_rels, defs, usar_def=False)
            bp_d = _max_sim_ref(onto_rels, defs, usar_def=True)
            breakpoints_label.append(bp_l)
            breakpoints_def.append(bp_d)

            n_pares = 0
            if os.path.exists(raw) and os.path.exists(frases_p):
                with open(frases_p, encoding="utf-8") as f:
                    ids = set(json.load(f)["ids"])
                fmt = "tekgen" if ds == "tekgen" else "webnlg"
                gt = load_gt_train(_train_jsonl(ds, onto), fmt, ids)
                free = load_free_train(raw, ids)
                pares = build_alignment_pairs(gt, free)
                n_pares = len(pares)
                if n_pares:
                    res = calibrate(pares, onto_rels, defs, "label_def")
                    todas_linhas.extend(res)
            print(f"{ds+'/'+onto:>26} | {len(onto_rels):>4} | {bp_l:>9.3f} | {bp_d:>9.3f} | {n_pares:>5}")

    if not breakpoints_label:
        print("\nNenhuma ontologia induzida encontrada.")
        return

    thr_label = max(breakpoints_label) + 0.05
    thr_def = max(breakpoints_def) + 0.05
    print("\n=== Agrupamento (D-21) ===")
    print(f"  max breakpoint (só-rótulo)   = {max(breakpoints_label):.3f}  ->  threshold = {thr_label:.3f}")
    print(f"  max breakpoint (rótulo+def)  = {max(breakpoints_def):.3f}  ->  threshold = {thr_def:.3f}")

    print("\n=== Projeção (D-30), agregado sobre as 29 ontologias (label+def) ===")
    if todas_linhas:
        print(f"  {'thr':>5} | {'prec':>6} | {'rec':>6} | {'aceitos':>8} | {'corretos':>8}")
        proposto = None
        for t in THRESHOLDS:
            rs = [r for r in todas_linhas if abs(r["threshold"] - t) < 1e-6]
            n_ac = sum(r["n_aceitos"] for r in rs)
            n_co = sum(r["n_corretos"] for r in rs)
            n_to = sum(r["n_total"] for r in rs)
            prec = n_co / n_ac if n_ac else 0.0
            rec = n_ac / n_to if n_to else 0.0
            marca = ""
            if proposto is None and prec >= 0.80:
                proposto = t
                marca = "  <- menor thr com prec>=0.80"
            print(f"  {t:>5.2f} | {prec:>6.3f} | {rec:>6.3f} | {n_ac:>8} | {n_co:>8}{marca}")
        print(f"\n  Threshold de projeção proposto (D-30): {proposto if proposto is not None else 'nenhum >=0.80'}")
    else:
        print("  Sem pares de alinhamento (raws/frases ausentes).")

    print("\nO script apenas calcula. Registre a decisão dos limiares manualmente.")


if __name__ == "__main__":
    main()
