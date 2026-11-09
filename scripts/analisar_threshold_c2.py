"""
Análise de calibração do threshold de agrupamento para a condição C2.

Responde três perguntas:
  1. Quais pares de relações da ontologia de REFERÊNCIA se agrupariam em cada threshold?
     (Critério: as relações de referência NÃO devem se fundir — se fundem, o threshold está baixo.)
  2. Qual o tamanho (n_relações, n_tokens) da ontologia induzida vs a de referência?
  3. Qual a distribuição de frequência das relações livres extraídas do piloto C0?

Uso:
  python scripts/analisar_threshold_c2.py
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE = os.path.abspath(os.path.join(REPO, ".."))
BENCH = os.path.join(WORKSPACE, "Text2KGBench")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    _EMBS_OK = True
except ImportError:
    _EMBS_OK = False

ONTOS = ["1_movie", "2_music"]
THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90]


def carregar_relacoes_referencia(onto_id):
    p = os.path.join(BENCH, "data", "wikidata_tekgen", "ontologies",
                     f"{onto_id}_ontology.json")
    d = json.load(open(p, encoding="utf-8"))
    return [r["label"] for r in d["relations"]]


def carregar_ontologia_induzida(onto_id):
    p = os.path.join(REPO, "configs", "c2", f"tekgen_{onto_id}_induced.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def carregar_relacoes_livres_c0(onto_id):
    """Extrai relações das triplas C0 do piloto (extração livre)."""
    fname = os.path.join(REPO, "resultados", "piloto",
                         f"piloto_tkg_{onto_id}_c0_qwen25_7b_instruct_awq.jsonl")
    if not os.path.exists(fname):
        return Counter()
    rels = []
    for l in open(fname, encoding="utf-8"):
        r = json.loads(l)
        for tr in r.get("triples", []):
            rels.append(tr[1])
    return Counter(rels)


def analisar_pares_referencia(rels_ref, modelo, threshold):
    """Retorna pares de relações de referência com sim >= threshold."""
    rels_norm = [r.replace("_", " ").lower() for r in rels_ref]
    embs = modelo.encode(rels_norm, show_progress_bar=False)
    sim = cosine_similarity(embs, embs)
    pares = []
    n = len(rels_ref)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                pares.append((rels_ref[i], rels_ref[j], float(sim[i, j])))
    return pares


def contar_tokens(rels):
    """Proxy de tokens: conta palavras nos labels das relações."""
    return sum(len(r.replace("_", " ").split()) for r in rels)


def main():
    if not _EMBS_OK:
        print("ERRO: sentence-transformers não disponível.")
        sys.exit(1)

    modelo = SentenceTransformer("all-MiniLM-L6-v2")

    for onto_id in ONTOS:
        print(f"\n{'='*70}")
        print(f"Ontologia: {onto_id}")
        print(f"{'='*70}")

        rels_ref = carregar_relacoes_referencia(onto_id)
        print(f"\nRelações de referência ({len(rels_ref)}): {rels_ref}")

        print(f"\n--- Pares que se fundiriam por threshold ---")
        print(f"{'Threshold':>10}  {'#pares':>7}  Pares que colidem")
        any_found = False
        for t in THRESHOLDS:
            pares = analisar_pares_referencia(rels_ref, modelo, t)
            marca = " ← RISCO" if pares else ""
            print(f"  t={t:.2f}       {len(pares):>5}   ", end="")
            if pares:
                any_found = True
                print(f"{pares[0][0]!r} + {pares[0][1]!r} ({pares[0][2]:.3f})", end="")
                if len(pares) > 1:
                    print(f" ... +{len(pares)-1} mais", end="")
                print(marca)
            else:
                print(f"nenhum{marca}")

        if not any_found:
            print("  Nenhum par de referência se funde em nenhum threshold testado.")

        # Detalhe t=0.80 e t=0.75
        for t in [0.75, 0.80]:
            pares = analisar_pares_referencia(rels_ref, modelo, t)
            if pares:
                print(f"\n  Detalhes t={t:.2f}:")
                for a, b, s in sorted(pares, key=lambda x: -x[2]):
                    print(f"    {a!r} + {b!r}  sim={s:.3f}")

        print(f"\n--- Ontologia induzida vs referência ---")
        ind = carregar_ontologia_induzida(onto_id)
        if ind:
            rels_ind = [r["label"] for r in ind.get("relations", [])]
            tok_ref = contar_tokens(rels_ref)
            tok_ind = contar_tokens(rels_ind)
            print(f"  Referência:  {len(rels_ref):3d} relações  {tok_ref:4d} tokens nos labels")
            print(f"  Induzida:    {len(rels_ind):3d} relações  {tok_ind:4d} tokens nos labels")
            meta = ind.get("metadata", {})
            print(f"  Threshold usado: {meta.get('threshold', '?')}  "
                  f"n_frases_treino: {meta.get('n_frases_treino', '?')}")
            print(f"\n  Relações induzidas: {rels_ind[:10]}{'...' if len(rels_ind)>10 else ''}")
        else:
            print("  configs/c2/ não encontrado — execute indução primeiro.")

        print(f"\n--- Frequência das relações livres (C0 piloto) ---")
        freq = carregar_relacoes_livres_c0(onto_id)
        if freq:
            total = sum(freq.values())
            unicas = len(freq)
            print(f"  Total de relações extraídas: {total}  Únicas: {unicas}")
            print(f"  Freq >= 2: {sum(1 for v in freq.values() if v >= 2)} relações")
            print(f"  Freq >= 3: {sum(1 for v in freq.values() if v >= 3)} relações")
            print(f"  Freq >= 5: {sum(1 for v in freq.values() if v >= 5)} relações")
            print(f"\n  Top 15 relações mais frequentes:")
            for rel, cnt in freq.most_common(15):
                print(f"    {cnt:4d}x  {rel!r}")
        else:
            print("  JSONL piloto C0 não encontrado.")

    print(f"\n{'='*70}")
    print("PROPOSTA DE REGRA PRÉ-REGISTRADA (a confirmar na D-19/D-20):")
    print("  - threshold: usar o menor t onde nenhum par de referência se funde")
    print("  - freq_min >= 2: elimina hápax (relações que aparecem só 1x no treino)")
    print("  - cap de tamanho: no máximo 2× o número de relações de referência")
    print("  Critério objetivo: as relações da ontologia de REFERÊNCIA não devem se agrupar.")


if __name__ == "__main__":
    main()
