"""
Análise da ontologia induzida C2: varredura de freq_min e investigação de relações anômalas.

Responde:
  1. Quantas relações únicas restam para cada freq_min em [1, 2, 3, 5, 10]?
  2. A relação "Knocking at the Door" aparece no piloto C0? Em qual frase?
  3. Qual a distribuição de relações por tamanho de grupo (ontologia agrupada)?

Uso:
  python scripts/analisar_ontologia_c2.py
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE = os.path.abspath(os.path.join(REPO, ".."))
BENCH = os.path.join(WORKSPACE, "Text2KGBench")

ONTOS = ["1_movie", "2_music"]
FREQ_MINS = [1, 2, 3, 5, 10]


def carregar_relacoes_livres_c0_completo(onto_id):
    """Retorna lista de (rel, sent_id, tripla) para análise detalhada."""
    fname = os.path.join(REPO, "resultados", "piloto",
                         f"piloto_tkg_{onto_id}_c0_qwen25_7b_instruct_awq.jsonl")
    if not os.path.exists(fname):
        return []
    registros = []
    for l in open(fname, encoding="utf-8"):
        r = json.loads(l)
        for tr in r.get("triples", []):
            registros.append((tr[1], r["id"], tr))
    return registros


def carregar_frases_gt(onto_id):
    """Mapa id → texto da frase (GT de teste)."""
    p = os.path.join(BENCH, "data", "wikidata_tekgen", "ground_truth",
                     f"ont_{onto_id}_ground_truth.jsonl")
    if not os.path.exists(p):
        return {}
    return {json.loads(l)["id"]: json.loads(l)["sent"]
            for l in open(p, encoding="utf-8") if l.strip()}


def carregar_ontologia_induzida(onto_id):
    p = os.path.join(REPO, "configs", "c2", f"tekgen_{onto_id}_induced.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def varredura_freq_min(freq_counter, rels_ref):
    """Mostra quantas relações únicas restam por freq_min."""
    total_extracao = sum(freq_counter.values())
    n_ref = len(rels_ref)
    print(f"  {'freq_min':>8}  {'n_rels_unicas':>14}  {'% de extr.':>10}  {'vs ref':>8}")
    print(f"  {'-'*50}")
    for fmin in FREQ_MINS:
        rels_ok = {r for r, c in freq_counter.items() if c >= fmin}
        n = len(rels_ok)
        frac = sum(freq_counter[r] for r in rels_ok) / total_extracao * 100 if total_extracao else 0
        ratio = f"{n/n_ref:.1f}×" if n_ref else "?"
        print(f"  {fmin:>8}  {n:>14}  {frac:>9.1f}%  {ratio:>8}")


def investigar_anomalias(registros, frases_gt):
    """Busca relações suspeitas (títulos de filmes, frases longas, etc.)."""
    anomalias = []
    for rel, sent_id, tripla in registros:
        # Relações com espaços e maiúsculas = provável título/nome próprio
        palavras = rel.split()
        if len(palavras) >= 3 and any(w[0].isupper() for w in palavras):
            anomalias.append((rel, sent_id, tripla, frases_gt.get(sent_id, "")))
    return anomalias


def main():
    for onto_id in ONTOS:
        print(f"\n{'='*70}")
        print(f"Ontologia: {onto_id}")
        print(f"{'='*70}")

        registros = carregar_relacoes_livres_c0_completo(onto_id)
        if not registros:
            print("  JSONL piloto C0 não encontrado.")
            continue

        freq = Counter(r for r, _, _ in registros)
        frases_gt = carregar_frases_gt(onto_id)

        print(f"\n  Total de triplas C0: {len(registros)}")
        print(f"  Relações únicas (freq_min=1): {len(freq)}")

        # Carrega ontologia de referência
        ref_path = os.path.join(BENCH, "data", "wikidata_tekgen", "ontologies",
                                f"{onto_id}_ontology.json")
        rels_ref = []
        if os.path.exists(ref_path):
            rels_ref = [r["label"] for r in json.load(open(ref_path, encoding="utf-8"))["relations"]]
        print(f"  Relações de referência: {len(rels_ref)}")

        print(f"\n--- Varredura freq_min ---")
        varredura_freq_min(freq, rels_ref)

        print(f"\n--- Ontologia induzida disponível (re-indução com bugs corrigidos) ---")
        ind = carregar_ontologia_induzida(onto_id)
        if ind:
            rels_ind = [r["label"] for r in ind.get("relations", [])]
            meta = ind.get("metadata", {})
            print(f"  Relações induzidas: {len(rels_ind)}")
            print(f"  Metadados: {meta}")
            print(f"  Labels: {rels_ind}")
        else:
            print("  configs/c2/ não encontrado — re-indução ainda não executada no Kaggle.")

        print(f"\n--- Investigação de relações anômalas (títulos/nomes como predicados) ---")
        anomalias = investigar_anomalias(registros, frases_gt)
        if anomalias:
            print(f"  {len(anomalias)} ocorrência(s) de possíveis títulos/nomes como predicados:")
            vistos = set()
            for rel, sent_id, tripla, sent in anomalias[:15]:
                if rel not in vistos:
                    vistos.add(rel)
                    print(f"\n  Relação: {rel!r}")
                    print(f"  Tripla:  {tripla}")
                    print(f"  ID:      {sent_id}")
                    if sent:
                        print(f"  Frase:   {sent[:120]}")
        else:
            print("  Nenhuma relação anômala detectada (critério: >= 3 palavras, alguma maiúscula).")

        print(f"\n--- Busca específica: 'Knocking at the Door' ---")
        knockings = [(rel, sid, tr, frases_gt.get(sid, ""))
                     for rel, sid, tr in registros
                     if "knocking" in rel.lower() or "door" in rel.lower()]
        if knockings:
            for rel, sid, tr, sent in knockings[:5]:
                print(f"  Relação: {rel!r}  ID: {sid}")
                print(f"  Tripla:  {tr}")
                if sent:
                    print(f"  Frase:   {sent[:120]}")
        else:
            print("  'Knocking at the Door' não encontrado no piloto C0 desta ontologia.")

    print(f"\n{'='*70}")
    print("Concluído.")


if __name__ == "__main__":
    main()
