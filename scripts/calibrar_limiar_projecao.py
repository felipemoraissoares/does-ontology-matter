"""
Calibra o limiar de projeção (threshold) para mapeamento livre→ontologia.

Usa pares alinhados (relação livre, relação GT) derivados das 50 frases de treino
de cada ontologia do piloto-v3. Alinhamento por coincidência de par de entidades
(sub, obj) normalizados entre extração livre e GT.

Compara dois formatos de embedding:
  fmt1 — só rótulo: _normalizar_rotulo(label)
  fmt2 — rótulo + definição (D-24): _normalizar_rotulo(label): <def>

Para cada threshold t em [0.40 .. 0.90], calcula:
  precision = P(mapeamento correto | SimE >= t)
  recall    = P(SimE >= t)   [fração dos pares aceitos]
  f1        = harmônica de precision e recall

Propõe threshold com maior F1 ponderando mais precisão (alpha=2 no beta de Fbeta).
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE = os.path.abspath(os.path.join(REPO, ".."))
BENCH_DIR = os.path.join(WORKSPACE, "Text2KGBench")
C2_DIR = os.path.join(REPO, "configs", "c2")
RAW_DIR = os.path.join(REPO, "resultados", "c2_inducao")

sys.path.insert(0, os.path.join(REPO, "src"))

from t2kg.avaliacao.extended import _normalizar_rotulo, _get_modelo
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

# ---------------------------------------------------------------------------
# Config das 4 ontologias do piloto
# ---------------------------------------------------------------------------
ONTOS = [
    {"dataset": "tekgen", "onto": "1_movie",
     "raw_jsonl": os.path.join(RAW_DIR, "raw_tkg_1_movie_treino.jsonl"),
     "frases_json": os.path.join(C2_DIR, "tekgen_1_movie_frases_seed42.json"),
     "train_jsonl": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "train", "ont_1_movie_train.jsonl"),
     "onto_json": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "ontologies", "1_movie_ontology.json"),
     "defs_json": os.path.join(C2_DIR, "definicoes_1_movie.json"),
     "train_format": "tekgen"},
    {"dataset": "tekgen", "onto": "2_music",
     "raw_jsonl": os.path.join(RAW_DIR, "raw_tkg_2_music_treino.jsonl"),
     "frases_json": os.path.join(C2_DIR, "tekgen_2_music_frases_seed42.json"),
     "train_jsonl": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "train", "ont_2_music_train.jsonl"),
     "onto_json": os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "ontologies", "2_music_ontology.json"),
     "defs_json": os.path.join(C2_DIR, "definicoes_2_music.json"),
     "train_format": "tekgen"},
    {"dataset": "webnlg", "onto": "1_university",
     "raw_jsonl": os.path.join(RAW_DIR, "raw_dbp_1_university_treino.jsonl"),
     "frases_json": os.path.join(C2_DIR, "webnlg_1_university_frases_seed42.json"),
     "train_jsonl": os.path.join(BENCH_DIR, "data", "dbpedia_webnlg", "train", "ont_1_university_train.jsonl"),
     "onto_json": os.path.join(BENCH_DIR, "data", "dbpedia_webnlg", "ontologies", "1_university_ontology.json"),
     "defs_json": os.path.join(C2_DIR, "definicoes_1_university.json"),
     "train_format": "webnlg"},
    {"dataset": "webnlg", "onto": "19_film",
     "raw_jsonl": os.path.join(RAW_DIR, "raw_dbp_19_film_treino.jsonl"),
     "frases_json": os.path.join(C2_DIR, "webnlg_19_film_frases_seed42.json"),
     "train_jsonl": os.path.join(BENCH_DIR, "data", "dbpedia_webnlg", "train", "ont_19_film_train.jsonl"),
     "onto_json": os.path.join(BENCH_DIR, "data", "dbpedia_webnlg", "ontologies", "19_film_ontology.json"),
     "defs_json": os.path.join(C2_DIR, "definicoes_19_film.json"),
     "train_format": "webnlg"},
]

THRESHOLDS = [round(t, 2) for t in np.arange(0.40, 0.91, 0.05)]


def _norm_ent(s: str) -> str:
    return re.sub(r"[\s_\-]+", "", s.lower().strip())


def load_gt_train(path: str, fmt: str, ids: set) -> Dict[str, List[Tuple]]:
    """Retorna {sent_id: [(sub_norm, rel, obj_norm), ...]} para IDs no treino."""
    gt = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            sid = d["id"]
            if sid not in ids:
                continue
            if fmt == "tekgen":
                sub = _norm_ent(d.get("sub_label", d.get("sub", "")))
                obj = _norm_ent(d.get("obj_label", d.get("obj", "")))
                rel = d["rel_label"]
                gt[sid].append((sub, rel, obj))
            else:  # webnlg
                for t in d.get("triples", []):
                    sub = _norm_ent(t["sub"])
                    obj = _norm_ent(t["obj"])
                    rel = t["rel"]
                    gt[sid].append((sub, rel, obj))
    return gt


def load_free_train(path: str, ids: set) -> Dict[str, List[Tuple]]:
    """Retorna {sent_id: [(sub_norm, rel_livre, obj_norm), ...]}."""
    free = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            sid = d["id"]
            if sid not in ids:
                continue
            for t in d.get("triples", []):
                sub = _norm_ent(t[0])
                rel = t[1]
                obj = _norm_ent(t[2])
                free[sid].append((sub, rel, obj))
    return free


def build_alignment_pairs(gt: Dict, free: Dict) -> List[Tuple[str, str]]:
    """
    Alinha por par de entidades (sub_norm, obj_norm) iguais.
    Retorna lista de (rel_livre, rel_gt).
    """
    pairs = []
    for sid in gt:
        if sid not in free:
            continue
        gt_triples = gt[sid]
        fr_triples = free[sid]
        for g_sub, g_rel, g_obj in gt_triples:
            for f_sub, f_rel, f_obj in fr_triples:
                # Match se sub E obj coincidem (após normalização)
                if g_sub == f_sub and g_obj == f_obj:
                    pairs.append((f_rel, g_rel))
    return pairs


def calibrate(pairs: List[Tuple[str, str]],
              onto_rels: List[str],
              defs: Optional[Dict],
              label: str) -> List[Dict]:
    """
    Para cada threshold, calcula precision/recall de projeção.
    Retorna lista de dicts {threshold, precision, recall, f_beta, n_aceitos, n_corretos}.
    """
    modelo = _get_modelo()

    # Embeddings das relações de referência (ontologia)
    if defs:
        rels_ref_emb = [
            f"{_normalizar_rotulo(r)}: {defs[r]}" if r in defs else _normalizar_rotulo(r)
            for r in onto_rels
        ]
    else:
        rels_ref_emb = [_normalizar_rotulo(r) for r in onto_rels]

    # Relações livres únicas nos pares
    unique_free = list({p[0] for p in pairs})
    rels_free_emb = [_normalizar_rotulo(r) for r in unique_free]

    embs_ref = modelo.encode(rels_ref_emb, show_progress_bar=False)
    embs_free = modelo.encode(rels_free_emb, show_progress_bar=False)
    sim = _cos_sim(embs_free, embs_ref)  # (n_free_unique, n_onto_rels)

    # Pré-calcular: para cada relação livre, (melhor_rel_onto, melhor_sim)
    free_to_pred = {}
    for i, fr in enumerate(unique_free):
        idx_max = int(np.argmax(sim[i]))
        free_to_pred[fr] = (onto_rels[idx_max], float(sim[i, idx_max]))

    n_total = len(pairs)
    results = []
    for t in THRESHOLDS:
        n_aceitos = 0
        n_corretos = 0
        for f_rel, g_rel in pairs:
            pred_rel, pred_sim = free_to_pred[f_rel]
            if pred_sim >= t:
                n_aceitos += 1
                if pred_rel.replace(" ", "_") == g_rel.replace(" ", "_"):
                    n_corretos += 1
        precision = n_corretos / n_aceitos if n_aceitos > 0 else 0.0
        recall = n_aceitos / n_total if n_total > 0 else 0.0
        # F-beta com beta=0.5 (pesa precision 2× mais que recall)
        beta = 0.5
        fb = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0
        results.append({
            "formato": label,
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f_beta_0.5": fb,
            "n_aceitos": n_aceitos,
            "n_total": n_total,
            "n_corretos": n_corretos,
        })
    return results


def main():
    modelo_ok = True
    try:
        _get_modelo()
    except Exception:
        modelo_ok = False
        print("ERRO: sentence-transformers nao disponivel")
        sys.exit(1)

    all_results = []
    all_pairs_fmt1 = []
    all_pairs_fmt2 = []
    onto_rels_all = []
    defs_all = None

    print("\n" + "="*70)
    print("CALIBRAÇÃO DO LIMIAR DE PROJEÇÃO")
    print("="*70)

    # --- Coleta pares de alinhamento de todas as ontologias ---
    aggregated_pairs = []
    for cfg in ONTOS:
        onto_id = cfg["onto"]
        ds = cfg["dataset"]
        label = f"{ds}/{onto_id}"
        print(f"\n--- {label} ---")

        # IDs de treino
        with open(cfg["frases_json"], encoding="utf-8") as f:
            frases_data = json.load(f)
        train_ids = set(frases_data["ids"])
        print(f"  IDs de treino: {len(train_ids)}")

        # GT de treino
        gt = load_gt_train(cfg["train_jsonl"], cfg["train_format"], train_ids)
        # Extração livre de treino
        free = load_free_train(cfg["raw_jsonl"], train_ids)

        # Pares alinhados
        pairs = build_alignment_pairs(gt, free)
        print(f"  Pares alinhados (sub+obj match): {len(pairs)}")
        if not pairs:
            print("  AVISO: sem pares — verificar alinhamento de entidades")
            continue

        # Ontologia de referência
        with open(cfg["onto_json"], encoding="utf-8") as f:
            onto_data = json.load(f)
        onto_rels = [r["label"] for r in onto_data["relations"]]

        # Definições D-24
        defs = None
        if os.path.exists(cfg["defs_json"]):
            with open(cfg["defs_json"], encoding="utf-8") as f:
                raw = json.load(f)
            defs = raw.get("relacoes", {})
            print(f"  Definicoes D-24: {len(defs)} labels")

        # Resultados por formato para esta ontologia
        res1 = calibrate(pairs, onto_rels, None, "label_only")
        res2 = calibrate(pairs, onto_rels, defs, "label_def")

        for r in res1 + res2:
            r["ontologia"] = onto_id
            r["dataset"] = ds
        all_results.extend(res1 + res2)
        aggregated_pairs.append((pairs, onto_rels, defs))

    # --- Tabela agregada (todas as ontologias juntas) ---
    # Para cada threshold, agrega precision/recall ponderado por n_total
    print("\n" + "="*70)
    print("TABELA AGREGADA (todas as ontologias)")
    print("="*70)

    def aggregate(fmt_label):
        rows = [r for r in all_results if r["formato"] == fmt_label]
        agg = []
        for t in THRESHOLDS:
            rs = [r for r in rows if abs(r["threshold"] - t) < 0.001]
            n_total = sum(r["n_total"] for r in rs)
            n_aceitos = sum(r["n_aceitos"] for r in rs)
            n_corretos = sum(r["n_corretos"] for r in rs)
            precision = n_corretos / n_aceitos if n_aceitos > 0 else 0.0
            recall = n_aceitos / n_total if n_total > 0 else 0.0
            beta = 0.5
            fb = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0
            agg.append({"threshold": t, "precision": precision, "recall": recall,
                        "f_beta_0.5": fb, "n_aceitos": n_aceitos, "n_total": n_total})
        return agg

    agg1 = aggregate("label_only")
    agg2 = aggregate("label_def")

    # Header
    print(f"\n{'t':>6} | {'=== LABEL ONLY ===':^30} | {'=== LABEL + DEF (D-24) ===':^30}")
    print(f"{'':>6} | {'Prec':>7} {'Rec':>7} {'F-b0.5':>7} {'N-ac':>6} | {'Prec':>7} {'Rec':>7} {'F-b0.5':>7} {'N-ac':>6}")
    print("-"*72)

    best_t1 = max(agg1, key=lambda r: r["f_beta_0.5"])
    best_t2 = max(agg2, key=lambda r: r["f_beta_0.5"])

    for a1, a2 in zip(agg1, agg2):
        t = a1["threshold"]
        m1 = " *" if abs(t - best_t1["threshold"]) < 0.001 else "  "
        m2 = " *" if abs(t - best_t2["threshold"]) < 0.001 else "  "
        print(
            f"{t:>6.2f} | {a1['precision']:>7.3f} {a1['recall']:>7.3f} {a1['f_beta_0.5']:>7.3f} {a1['n_aceitos']:>6}{m1}| "
            f"{a2['precision']:>7.3f} {a2['recall']:>7.3f} {a2['f_beta_0.5']:>7.3f} {a2['n_aceitos']:>6}{m2}"
        )

    print(f"\n* = melhor F-beta(0.5) [priorizando precisão]")
    print(f"\nLabel-only  — melhor threshold: {best_t1['threshold']:.2f}  "
          f"(prec={best_t1['precision']:.3f}, rec={best_t1['recall']:.3f}, F-b={best_t1['f_beta_0.5']:.3f})")
    print(f"Label+def   — melhor threshold: {best_t2['threshold']:.2f}  "
          f"(prec={best_t2['precision']:.3f}, rec={best_t2['recall']:.3f}, F-b={best_t2['f_beta_0.5']:.3f})")

    # --- Por ontologia (tabela compacta) ---
    print("\n" + "="*70)
    print("POR ONTOLOGIA — melhor threshold por formato")
    print("="*70)
    print(f"{'Ontologia':>18} | {'label-only':^22} | {'label+def':^22}")
    print(f"{'':>18} | {'t*':>5} {'Prec':>6} {'Rec':>6} {'F-b':>5} | {'t*':>5} {'Prec':>6} {'Rec':>6} {'F-b':>5}")
    print("-"*72)
    for cfg in ONTOS:
        onto = cfg["onto"]
        ds = cfg["dataset"]
        def best_for(fmt):
            rows = [r for r in all_results
                    if r["formato"] == fmt and r["ontologia"] == onto]
            if not rows:
                return None
            return max(rows, key=lambda r: r["f_beta_0.5"])
        b1 = best_for("label_only")
        b2 = best_for("label_def")
        if b1 and b2:
            print(f"{ds}/{onto:>14} | {b1['threshold']:>5.2f} {b1['precision']:>6.3f} {b1['recall']:>6.3f} {b1['f_beta_0.5']:>5.3f} | "
                  f"{b2['threshold']:>5.2f} {b2['precision']:>6.3f} {b2['recall']:>6.3f} {b2['f_beta_0.5']:>5.3f}")

    # --- Salva CSV ---
    csv_out = os.path.join(REPO, "resultados", "avaliacao_piloto_v3", "calibracao_threshold_projecao.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["dataset", "ontologia", "formato", "threshold", "precision",
                      "recall", "f_beta_0.5", "n_aceitos", "n_total", "n_corretos"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    print(f"\nCSV salvo: {csv_out}")

    # --- Proposta final ---
    print("\n" + "="*70)
    print("PROPOSTA DE THRESHOLD")
    print("="*70)
    # Prioriza precisão >= 0.70 com maior recall
    cands = [r for r in agg2 if r["precision"] >= 0.70]
    if cands:
        proposed = max(cands, key=lambda r: r["recall"])
    else:
        proposed = best_t2
    print(f"\nFormato label+def: threshold proposto = {proposed['threshold']:.2f}")
    print(f"  Precision={proposed['precision']:.3f}, Recall={proposed['recall']:.3f}, "
          f"F-b(0.5)={proposed['f_beta_0.5']:.3f}, N-aceitos={proposed['n_aceitos']}/{proposed['n_total']}")
    print(f"\nFormato label-only: threshold proposto = {best_t1['threshold']:.2f}")
    print(f"  Precision={best_t1['precision']:.3f}, Recall={best_t1['recall']:.3f}, "
          f"F-b(0.5)={best_t1['f_beta_0.5']:.3f}")

    return proposed["threshold"], best_t1["threshold"]


if __name__ == "__main__":
    main()
