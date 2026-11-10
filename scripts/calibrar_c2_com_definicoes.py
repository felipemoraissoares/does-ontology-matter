"""
D-24: Calibração C2 comparando "só rótulo" × "rótulo + definição".

Para cada onto × N × freq_min, calcula:
  - n_rels_induzidas
  - cobertura do gabarito de treino (fração de relações GT com mapping ≥ threshold)
  - fração de ruído (relações induzidas sem mapeamento para GT)

Modos:
  --dry_run : mostra só a tabela baseline "label" (sem embeddings)
  --from_cache : usa configs/c2/definicoes_{onto_id}.json se disponível

Uso:
  python scripts/calibrar_c2_com_definicoes.py --onto_id 1_movie --dataset tekgen
  python scripts/calibrar_c2_com_definicoes.py --onto_id 1_movie --dataset tekgen --dry_run
  python scripts/calibrar_c2_com_definicoes.py --onto_id 1_movie --dataset tekgen --from_cache
"""
import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    _EMBS_OK = True
except ImportError:
    _EMBS_OK = False

from t2kg.avaliacao.extended import _normalizar_rotulo  # noqa: E402

BENCH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Text2KGBench"))
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELO_EMBS = "all-MiniLM-L6-v2"
THRESHOLD = 0.70
FREQ_MINS = [2, 3, 5]
N = 50


def _carregar_raw_jsonl(dataset: str, onto_id: str) -> Optional[str]:
    prefixo = "tkg" if dataset == "tekgen" else "dbp"
    path = os.path.join(REPO, "resultados", "c2_inducao", f"raw_{prefixo}_{onto_id}_treino.jsonl")
    return path if os.path.exists(path) else None


def _carregar_triplas_por_frase(raw_jsonl: str) -> Dict[str, List[List[str]]]:
    from t2kg.parser import parsear_resposta
    result = {}
    with open(raw_jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            triplas = r.get("triples") or parsear_resposta(r.get("response", ""))
            result[r["id"]] = triplas
    return result


def _carregar_gt_treino(dataset: str, onto_id: str) -> List[str]:
    """Retorna lista de relações GT do treino (pode ter repetição)."""
    if dataset == "tekgen":
        p = os.path.join(BENCH, "data", "wikidata_tekgen", "train", f"ont_{onto_id}_train.jsonl")
    else:
        p = os.path.join(BENCH, "data", "dbpedia_webnlg", "train", f"ont_{onto_id}_train.jsonl")
    if not os.path.exists(p):
        return []
    rels = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if "rel_label" in d:
                rels.append(d["rel_label"].replace(" ", "_").lower())
            else:
                for t in d.get("triples", []):
                    rels.append(t.get("rel", "").replace(" ", "_").lower())
    return rels


def _carregar_relacoes_ref(dataset: str, onto_id: str) -> List[str]:
    if dataset == "tekgen":
        p = os.path.join(BENCH, "data", "wikidata_tekgen", "ontologies", f"{onto_id}_ontology.json")
    else:
        p = os.path.join(BENCH, "data", "dbpedia_webnlg", "ontologies", f"{onto_id}_ontology.json")
    with open(p, encoding="utf-8") as f:
        onto = json.load(f)
    return [r["label"].replace(" ", "_").lower() for r in onto.get("relations", [])]


def _agrupa_e_filtra(relacoes_por_frase, freq_min, modelo, definicoes=None):
    """Agrupa por freq_min distintas frases e retorna labels induzidos."""
    all_rels = [t[1] for triplas in relacoes_por_frase.values() for t in triplas if len(t) >= 2]
    if not all_rels:
        return []
    contagem = Counter(all_rels)
    unicos = list(contagem.keys())

    if definicoes:
        emb_texts = [f"{_normalizar_rotulo(r)}: {definicoes.get(r, _normalizar_rotulo(r))}" for r in unicos]
    else:
        emb_texts = [_normalizar_rotulo(r) for r in unicos]

    embs = modelo.encode(emb_texts, show_progress_bar=False)
    sim = cosine_similarity(embs, embs)
    n = len(unicos)
    visitado = [False] * n
    grupos = []
    for i in range(n):
        if visitado[i]:
            continue
        grupo = [i]
        visitado[i] = True
        for j in range(i + 1, n):
            if not visitado[j] and sim[i, j] >= THRESHOLD:
                grupo.append(j)
                visitado[j] = True
        membros = [unicos[k] for k in grupo]
        label = max(membros, key=lambda m: contagem[m])
        # Count distinct frases
        membros_set = {m.lower().strip() for m in membros}
        frases_do_grupo = set()
        for fid, triplas in relacoes_por_frase.items():
            for t in triplas:
                if len(t) >= 2 and t[1].lower().strip() in membros_set:
                    frases_do_grupo.add(fid)
                    break
        n_frases = len(frases_do_grupo)
        grupos.append({"label": label, "n_frases": n_frases, "membros": membros})

    return [g["label"] for g in grupos if g["n_frases"] >= freq_min]


def _cobertura_e_ruido(labels_ind, rels_ref, modelo, definicoes=None):
    """Calcula cobertura (fração de GT coberta) e ruído (fração de ind sem GT)."""
    if not labels_ind or not rels_ref:
        return 0.0, 1.0

    def encode(labels, defs=None):
        if defs:
            texts = [f"{_normalizar_rotulo(r)}: {defs.get(r, _normalizar_rotulo(r))}" for r in labels]
        else:
            texts = [_normalizar_rotulo(r) for r in labels]
        return modelo.encode(texts, show_progress_bar=False)

    embs_ind = encode(labels_ind, definicoes)
    embs_ref = encode(rels_ref, definicoes)
    sim = cosine_similarity(embs_ind, embs_ref)

    cobertura_gt = sum(1 for j in range(len(rels_ref)) if sim[:, j].max() >= THRESHOLD) / len(rels_ref)
    ruido = sum(1 for i in range(len(labels_ind)) if sim[i, :].max() < THRESHOLD) / len(labels_ind)
    return cobertura_gt, ruido


def _carregar_definicoes(onto_id):
    p = os.path.join(REPO, "configs", "c2", f"definicoes_{onto_id}.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8")).get("relacoes", {})
    return None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--onto_id", required=True)
    p.add_argument("--dataset", required=True, choices=["tekgen", "webnlg"])
    p.add_argument("--dry_run", action="store_true",
                   help="Mostra tabela baseline label sem embeddings.")
    p.add_argument("--from_cache", action="store_true",
                   help="Usa definicoes_{onto_id}.json se disponível.")
    args = p.parse_args()

    if not _EMBS_OK and not args.dry_run:
        print("ERRO: instale sentence-transformers e scikit-learn.")
        sys.exit(1)

    raw_jsonl = _carregar_raw_jsonl(args.dataset, args.onto_id)
    if not raw_jsonl:
        print(f"ERRO: raw JSONL não encontrado para {args.dataset}/{args.onto_id}")
        print("Execute a re-indução C2 no Kaggle primeiro.")
        sys.exit(1)

    relacoes_por_frase = _carregar_triplas_por_frase(raw_jsonl)
    rels_ref = _carregar_relacoes_ref(args.dataset, args.onto_id)
    print(f"Ontologia: {args.onto_id}  N={len(relacoes_por_frase)}  GT rels: {len(rels_ref)}")

    definicoes = None
    if args.from_cache:
        definicoes = _carregar_definicoes(args.onto_id)
        if definicoes:
            print(f"Definições carregadas: {len(definicoes)} rótulos")
        else:
            print("Aviso: definições não disponíveis (from_cache mas arquivo não encontrado)")

    if args.dry_run:
        print("\nModo --dry_run: tabela label-only (sem embeddings para modo def)")
        print(f"\n{'Onto':<12} {'N':>4} {'f_min':>6} {'modo':<6} {'n_ind':>6} {'cob':>8} {'ruido':>7}")
        print("-" * 55)
        modelo = SentenceTransformer(MODELO_EMBS)
        for fm in FREQ_MINS:
            labels = _agrupa_e_filtra(relacoes_por_frase, fm, modelo)
            cob, ruido = _cobertura_e_ruido(labels, rels_ref, modelo)
            print(f"{args.onto_id:<12} {N:>4} {fm:>6} {'label':<6} {len(labels):>6} {cob:>7.1%} {ruido:>7.1%}")
            if definicoes:
                labels_d = _agrupa_e_filtra(relacoes_por_frase, fm, modelo, definicoes)
                cob_d, ruido_d = _cobertura_e_ruido(labels_d, rels_ref, modelo, definicoes)
                print(f"{args.onto_id:<12} {N:>4} {fm:>6} {'def':<6} {len(labels_d):>6} {cob_d:>7.1%} {ruido_d:>7.1%}")
            else:
                print(f"{'':12} {N:>4} {fm:>6} {'def':<6} {'pendente':>6}")
        sys.exit(0)

    modelo = SentenceTransformer(MODELO_EMBS)
    print(f"\n{'Onto':<12} {'N':>4} {'f_min':>6} {'modo':<6} {'n_ind':>6} {'cob':>8} {'ruido':>7}")
    print("-" * 55)
    for fm in FREQ_MINS:
        labels = _agrupa_e_filtra(relacoes_por_frase, fm, modelo)
        cob, ruido = _cobertura_e_ruido(labels, rels_ref, modelo)
        print(f"{args.onto_id:<12} {N:>4} {fm:>6} {'label':<6} {len(labels):>6} {cob:>7.1%} {ruido:>7.1%}")
        if definicoes:
            labels_d = _agrupa_e_filtra(relacoes_por_frase, fm, modelo, definicoes)
            cob_d, ruido_d = _cobertura_e_ruido(labels_d, rels_ref, modelo, definicoes)
            print(f"{args.onto_id:<12} {N:>4} {fm:>6} {'def':<6} {len(labels_d):>6} {cob_d:>7.1%} {ruido_d:>7.1%}")
        else:
            print(f"{'':12} {N:>4} {fm:>6} {'def':<6} {'pendente (gerar_definicoes_relacoes.py)':>6}")
