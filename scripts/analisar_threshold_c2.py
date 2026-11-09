"""
Análise de calibração do threshold de agrupamento para a condição C2.

Responde quatro perguntas:
  1. Quais pares de relações da ontologia de REFERÊNCIA se agrupariam em cada threshold?
     (Critério: as relações de referência NÃO devem se fundir — se fundem, o threshold está baixo.)
     Encontra o ponto de ruptura (menor threshold onde algum par de referência se funde).
     Proposta D-21: threshold = breakpoint + 0.05.
  2. Qual o tamanho (n_relações, n_tokens) da ontologia induzida vs a de referência?
  3. Qual a distribuição de frequência das relações livres extraídas do piloto C0?
  4. Alinhamento free↔reference: para cada relação livre C0, qual a mais similar na referência?

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
# Varredura estendida: de 0.50 até 0.90 para encontrar ponto de ruptura
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


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


def alinhar_free_referencia(rels_free_top, rels_ref, modelo, top_n=20):
    """Para as top-N relações livres, mostra a relação de referência mais similar."""
    if not rels_free_top or not rels_ref:
        return
    rels_free_norm = [r.replace("_", " ").lower() for r in rels_free_top]
    rels_ref_norm = [r.replace("_", " ").lower() for r in rels_ref]
    embs_free = modelo.encode(rels_free_norm, show_progress_bar=False)
    embs_ref = modelo.encode(rels_ref_norm, show_progress_bar=False)
    sim = cosine_similarity(embs_free, embs_ref)
    print(f"  {'Relação livre':<40} {'Mais próxima na ref.':<25} {'sim':>6}")
    print(f"  {'-'*74}")
    for i, rel in enumerate(rels_free_top):
        idx = int(np.argmax(sim[i]))
        s = float(sim[i, idx])
        print(f"  {rel:<40} {rels_ref[idx]:<25} {s:6.3f}")


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

        print(f"\n--- Pares que se fundiriam por threshold (varredura estendida 0.50–0.90) ---")
        print(f"  {'Threshold':>9}  {'#pares':>7}  Pares que colidem")
        breakpoint_t = None
        any_found = False
        for t in THRESHOLDS:
            pares = analisar_pares_referencia(rels_ref, modelo, t)
            if pares and breakpoint_t is None:
                breakpoint_t = t
            marca = " ← RUPTURA" if (pares and breakpoint_t == t) else (" ← RISCO" if pares else "")
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
            print("  Nenhum par de referência se funde em nenhum threshold testado (0.50–0.90).")
            print("  Proposta D-21: usar threshold=0.80 (conservador, sem fusão em toda a faixa).")
        else:
            proposta = breakpoint_t + 0.05 if breakpoint_t is not None else None
            print(f"\n  Ponto de ruptura: t={breakpoint_t:.2f} (primeiro threshold onde algum par de referência funde)")
            print(f"  Proposta D-21:    threshold = breakpoint + 0.05 = {proposta:.2f}")

        # Detalhe nos thresholds problemáticos
        for t in THRESHOLDS:
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
            print("  configs/c2/ não encontrado — execute re-indução primeiro.")

        print(f"\n--- Frequência das relações livres (C0 piloto) ---")
        freq = carregar_relacoes_livres_c0(onto_id)
        if freq:
            total = sum(freq.values())
            unicas = len(freq)
            print(f"  Total de relações extraídas: {total}  Únicas: {unicas}")
            for fmin in [1, 2, 3, 5, 10]:
                n = sum(1 for v in freq.values() if v >= fmin)
                print(f"  freq_min>={fmin:2d}: {n:3d} relações únicas")
            print(f"\n  Top 20 relações mais frequentes:")
            for rel, cnt in freq.most_common(20):
                print(f"    {cnt:4d}x  {rel!r}")

            print(f"\n--- Alinhamento free→referência (top-20 relações C0) ---")
            top20 = [r for r, _ in freq.most_common(20)]
            alinhar_free_referencia(top20, rels_ref, modelo)
        else:
            print("  JSONL piloto C0 não encontrado.")

    print(f"\n{'='*70}")
    print("REGRAS PRÉ-REGISTRADAS (D-21):")
    print("  threshold_agrupamento: breakpoint + 0.05 (onde breakpoint = menor t que funde par ref)")
    print("  Se nenhum breakpoint encontrado em 0.50–0.90: threshold = 0.80 (conservador)")
    print("  freq_min: >= 5 (elimina hápax e relações raras)")
    print("  SEM cap de tamanho (D-19: cap 2× removido para C2)")


if __name__ == "__main__":
    main()
