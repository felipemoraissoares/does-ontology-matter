"""
Computa redundância semântica (métrica H1) e taxa de predicados únicos para o piloto-v2.

Lê os 6 JSONL do piloto (C0, C1, C1_oneshot_original × 1_movie, 2_music) e calcula:
  - redundancia_semantica: 1 − (n_grupos / n_rotulos_distintos) com threshold 0.80
  - taxa_predicados_unicos: n_rotulos_distintos / n_total_triplas
  - n_triplas, n_rotulos_distintos, n_grupos por condição/ontologia

Uso:
  python scripts/calcular_redundancia_piloto.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PILOTO = os.path.join(REPO, "resultados", "piloto")

DATASETS = [
    ("piloto_tkg_1_movie_c0_qwen25_7b_instruct_awq.jsonl",                  "C0",               "1_movie"),
    ("piloto_tkg_1_movie_c1_qwen25_7b_instruct_awq.jsonl",                  "C1",               "1_movie"),
    ("piloto_tkg_1_movie_c1_oneshot_original_qwen25_7b_instruct_awq.jsonl", "C1_oneshot",        "1_movie"),
    ("piloto_tkg_2_music_c0_qwen25_7b_instruct_awq.jsonl",                  "C0",               "2_music"),
    ("piloto_tkg_2_music_c1_qwen25_7b_instruct_awq.jsonl",                  "C1",               "2_music"),
    ("piloto_tkg_2_music_c1_oneshot_original_qwen25_7b_instruct_awq.jsonl", "C1_oneshot",        "2_music"),
]

THRESHOLD = 0.80


def carregar_triplas(caminho):
    triplas = []
    for l in open(caminho, encoding="utf-8"):
        r = json.loads(l)
        triplas.extend(r.get("triples", []))
    return triplas


def main():
    try:
        from t2kg.avaliacao.extended import redundancia_semantica, taxa_predicados_unicos
    except ImportError:
        print("ERRO: t2kg não encontrado no path.")
        sys.exit(1)

    try:
        from sentence_transformers import SentenceTransformer
        _ok = True
    except ImportError:
        print("AVISO: sentence-transformers não instalado — redundância semântica indisponível.")
        _ok = False

    print(f"{'Ontologia':<10} {'Condição':<15} {'n_triplas':>9} {'n_rótulos':>9} "
          f"{'n_grupos':>9} {'tpu':>6} {'red_sem':>8}")
    print("-" * 70)

    resultados = []
    for fname, cond, onto_id in DATASETS:
        caminho = os.path.join(PILOTO, fname)
        if not os.path.exists(caminho):
            print(f"  FALTANDO: {fname}")
            continue

        triplas = carregar_triplas(caminho)
        tpu = taxa_predicados_unicos(triplas)

        if _ok:
            res = redundancia_semantica(triplas, threshold=THRESHOLD)
            n_rot = res["n_rotulos"]
            n_grp = res["n_grupos"]
            val = res["valor"]
            val_str = f"{val:.4f}" if val is not None else "N/A"
        else:
            n_rot = len({t[1] for t in triplas})
            n_grp = None
            val_str = "N/A"

        n_triplas = len(triplas)
        print(f"{onto_id:<10} {cond:<15} {n_triplas:>9} {n_rot:>9} "
              f"{str(n_grp) if n_grp is not None else 'N/A':>9} {tpu:>6.3f} {val_str:>8}")
        resultados.append({
            "ontologia": onto_id, "condicao": cond, "n_triplas": n_triplas,
            "n_rotulos": n_rot, "n_grupos": n_grp, "taxa_predicados_unicos": tpu,
            "redundancia_semantica": res["valor"] if _ok else None,
            "threshold": THRESHOLD,
        })

    # Salva resultado
    saida = os.path.join(REPO, "resultados", "piloto", "redundancia_piloto_v2.json")
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em: {saida}")


if __name__ == "__main__":
    main()
