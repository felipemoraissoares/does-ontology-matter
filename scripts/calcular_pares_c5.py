"""
Calcula os pares de ontologias para as condições C5_vizinho e C5_incongruente.

Para cada ontologia alvo, encontra:
- Vizinho: parceira com maior similaridade semântica dos labels de relação
- Incongruente: parceira com menor similaridade semântica

Dois critérios computados por par (A, B):
  (i)  overlap = |labels(A) ∩ labels(B)| / |labels(A) ∪ labels(B)|   [Jaccard]
  (ii) sim_emb  = similaridade bidirecional por embeddings dos labels   [all-MiniLM-L6-v2]

Pré-processamento dos labels:
  - snake_case → espaços
  - camelCase → espaços
  - minúsculas

Uso:
  python scripts/calcular_pares_c5.py
"""
import json
import os
import re
import sys
from collections import Counter

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("ERRO: instale sentence-transformers e scikit-learn.")
    print("  pip install sentence-transformers scikit-learn --user")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
BENCH = os.path.join(os.path.dirname(__file__), "..", "..", "Text2KGBench")
MODELO_EMBS = "all-MiniLM-L6-v2"

TEKGEN_ONTOS = [f"{i}_{n}" for i, n in enumerate(
    ["movie", "music", "sport", "book", "military", "computer",
     "space", "politics", "nature", "culture"], 1)]
WEBNLG_ONTOS = [f"{i}_{n}" for i, n in enumerate(
    ["university", "musicalwork", "airport", "building", "athlete", "politician",
     "company", "celestialbody", "astronaut", "comicscharacter", "meanoftransportation",
     "monument", "food", "writtenwork", "sportsteam", "city", "artist", "scientist", "film"], 1)]


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
def normalizar_label(label: str) -> str:
    """snake_case e camelCase para palavras minúsculas."""
    label = label.replace("_", " ").replace(",", " ")
    label = re.sub(r"([a-z])([A-Z])", r"\1 \2", label)
    return label.lower().strip()


def carregar_onto(dataset: str, onto_id: str) -> dict:
    if dataset == "tekgen":
        p = os.path.join(BENCH, "data", "wikidata_tekgen", "ontologies", f"{onto_id}_ontology.json")
    else:
        p = os.path.join(BENCH, "data", "dbpedia_webnlg", "ontologies", f"{onto_id}_ontology.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def num_tokens_json(onto: dict) -> int:
    """Conta palavras no JSON do bloco C5 (proxy para tokens LLM)."""
    return len(json.dumps(onto).split())


def overlap_jaccard(rels_a: list, rels_b: list) -> float:
    """Jaccard sobre conjuntos de labels normalizados."""
    a = {normalizar_label(r) for r in rels_a}
    b = {normalizar_label(r) for r in rels_b}
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def sim_embedding_bidirecional(embs_a: np.ndarray, embs_b: np.ndarray) -> float:
    """
    Similaridade bidirecional de conjuntos de embeddings.
    Para cada embedding em A, encontra o max-cosine com B; repete B→A.
    Retorna a média das duas médias.
    """
    if len(embs_a) == 0 or len(embs_b) == 0:
        return 0.0
    sim_matrix = cosine_similarity(embs_a, embs_b)       # shape (|A|, |B|)
    a_to_b = sim_matrix.max(axis=1).mean()
    b_to_a = sim_matrix.max(axis=0).mean()
    return float(0.5 * (a_to_b + b_to_a))


# ---------------------------------------------------------------------------
# Análise por dataset
# ---------------------------------------------------------------------------
def analisar_dataset(dataset: str, ontos: list, modelo: SentenceTransformer) -> dict:
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset}  ({len(ontos)} ontologias)")
    print(f"{'='*70}")

    # 1. Carrega ontologias e computa embeddings de todos os labels
    dados = {}
    for o in ontos:
        ont = carregar_onto(dataset, o)
        rels = [r["label"] for r in ont["relations"]]
        concs = [c["label"] for c in ont.get("concepts", [])]
        labels_norm = [normalizar_label(r) for r in rels]
        embs = modelo.encode(labels_norm, show_progress_bar=False) if labels_norm else np.zeros((0, 384))
        dados[o] = {
            "ont": ont, "rels": rels, "concs": concs,
            "labels_norm": labels_norm, "embs": embs,
            "n_rels": len(rels), "n_concs": len(concs),
            "n_tokens": num_tokens_json(ont),
        }

    # 2. Computa matriz de similaridade (overlap e embedding) para todos os pares
    n = len(ontos)
    sim = {}  # (a, b) → {"overlap": float, "sim_emb": float}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = ontos[i], ontos[j]
            if (a, b) in sim:
                continue
            ov = overlap_jaccard(dados[a]["rels"], dados[b]["rels"])
            se = sim_embedding_bidirecional(dados[a]["embs"], dados[b]["embs"])
            sim[(a, b)] = sim[(b, a)] = {"overlap": ov, "sim_emb": se}

    # 3. Para cada ontologia, acha vizinho (max sim_emb), incongruente (min) e incongruente_2 (2nd min)
    pares = {}
    contagem_parceiro = Counter()

    for o in ontos:
        candidatos = sorted(
            (c for c in ontos if c != o),
            key=lambda c: sim[(o, c)]["sim_emb"]
        )
        vizinho     = max((c for c in ontos if c != o), key=lambda c: sim[(o, c)]["sim_emb"])
        incongruente   = candidatos[0]
        incongruente_2 = candidatos[1] if len(candidatos) > 1 else None
        pares[o] = {
            "vizinho": vizinho,
            "incongruente": incongruente,
            "incongruente_2": incongruente_2,
        }
        contagem_parceiro[vizinho] += 1
        contagem_parceiro[incongruente] += 1

    # 4. Tabela completa (vizinho e incongruente; inclui dif_tokens)
    print(f"\n{'Alvo':<30} {'#Tok':>5}  "
          f"{'Vizinho':<30} {'OvJ%':>5} {'SimE':>5} {'#Tok':>5} {'dTok':>5}  "
          f"{'Incongruente':<30} {'OvJ%':>5} {'SimE':>5} {'#Tok':>5} {'dTok':>5}  "
          f"{'Incongruente_2':<30} {'SimE':>5}")
    print("-" * 185)
    for o in ontos:
        d = dados[o]
        viz = pares[o]["vizinho"]
        inc = pares[o]["incongruente"]
        inc2 = pares[o]["incongruente_2"]
        dv = dados[viz]; di = dados[inc]
        sv = sim[(o, viz)]; si = sim[(o, inc)]
        dif_v = dv["n_tokens"] - d["n_tokens"]
        dif_i = di["n_tokens"] - d["n_tokens"]
        inc2_str = inc2 if inc2 else "-"
        inc2_sime = f"{sim[(o, inc2)]['sim_emb']:.3f}" if inc2 else "  -  "
        print(
            f"{o:<30} {d['n_tokens']:>5}  "
            f"{viz:<30} {sv['overlap']*100:>5.1f} {sv['sim_emb']:>5.3f} {dv['n_tokens']:>5} {dif_v:>+5}  "
            f"{inc:<30} {si['overlap']*100:>5.1f} {si['sim_emb']:>5.3f} {di['n_tokens']:>5} {dif_i:>+5}  "
            f"{inc2_str:<30} {inc2_sime:>5}"
        )

    # 5. Avisa sobre ontologias muito populares como parceira
    print(f"\n--- Frequencia como parceira (vizinho + incongruente) ---")
    for onto_p, cnt in contagem_parceiro.most_common(5):
        if cnt >= 3:
            print(f"  AVISO: {onto_p} aparece como parceira de {cnt} ontologias")
        else:
            print(f"  {onto_p}: {cnt}x")

    # 6. Resumo de pares escolhidos
    print(f"\n--- Pares vizinho escolhidos ---")
    for o in ontos:
        viz = pares[o]["vizinho"]
        sv = sim[(o, viz)]
        print(f"  {o} -> {viz}  (OvJ={sv['overlap']*100:.1f}%, SimE={sv['sim_emb']:.3f})")

    return pares, dados, sim


def _entrada_par(o: str, parceira: str, dados: dict, sim: dict) -> dict:
    if parceira is None:
        return None
    d = dados[o]; dp = dados[parceira]; s = sim[(o, parceira)]
    return {
        "id": parceira,
        "sim_emb": round(s["sim_emb"], 4),
        "overlap_jaccard": round(s["overlap"], 4),
        "n_tokens_parceira": dp["n_tokens"],
        "dif_tokens": dp["n_tokens"] - d["n_tokens"],
        "n_rels": dp["n_rels"],
        "n_concs": dp["n_concs"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Carregando modelo: {MODELO_EMBS} ...")
    modelo = SentenceTransformer(MODELO_EMBS)

    pares_tkg, dados_tkg, sim_tkg = analisar_dataset("tekgen", TEKGEN_ONTOS, modelo)
    pares_web, dados_web, sim_web = analisar_dataset("webnlg", WEBNLG_ONTOS, modelo)

    def serializar(pares, dados, sim):
        out = {}
        for o, p in pares.items():
            out[o] = {
                "n_tokens_alvo": dados[o]["n_tokens"],
                "n_rels_alvo": dados[o]["n_rels"],
                "n_concs_alvo": dados[o]["n_concs"],
                "vizinho":        _entrada_par(o, p["vizinho"], dados, sim),
                "incongruente":   _entrada_par(o, p["incongruente"], dados, sim),
                "incongruente_2": _entrada_par(o, p["incongruente_2"], dados, sim),
            }
        return out

    resultado = {
        "modelo_embeddings": MODELO_EMBS,
        "tekgen": serializar(pares_tkg, dados_tkg, sim_tkg),
        "webnlg": serializar(pares_web, dados_web, sim_web),
    }
    saida = os.path.join(os.path.dirname(__file__), "..", "configs", "pipeline", "pares_c5.json")
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    print(f"\nResultado salvo em: {saida}")
