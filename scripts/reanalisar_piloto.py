"""
Diagnóstico: re-parseia os 6 JSONL do piloto com o parser atual (D-18) e compara
com os triples gravados originalmente. Mostra quantas respostas passaram a ter
triplas extraídas após a correção do parser.

Uso:
  python scripts/reanalisar_piloto.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from t2kg.parser import parsear_resposta

PILOTO = os.path.join(os.path.dirname(__file__), "..", "resultados", "piloto")

ARQUIVOS = [
    ("tekgen/1_movie/C0",                "piloto_tkg_1_movie_c0_qwen25_7b_instruct_awq.jsonl"),
    ("tekgen/1_movie/C1",                "piloto_tkg_1_movie_c1_qwen25_7b_instruct_awq.jsonl"),
    ("tekgen/1_movie/C1_oneshot_original","piloto_tkg_1_movie_c1_oneshot_original_qwen25_7b_instruct_awq.jsonl"),
    ("tekgen/2_music/C0",                "piloto_tkg_2_music_c0_qwen25_7b_instruct_awq.jsonl"),
    ("tekgen/2_music/C1",                "piloto_tkg_2_music_c1_qwen25_7b_instruct_awq.jsonl"),
    ("tekgen/2_music/C1_oneshot_original","piloto_tkg_2_music_c1_oneshot_original_qwen25_7b_instruct_awq.jsonl"),
]

print("Re-análise dos 6 JSONL do piloto com o parser D-18")
print("=" * 65)
print(f"{'Condição':<35} {'n':>4} {'orig_vaz':>9} {'novo_vaz':>9} {'ganho':>7}")
print("-" * 65)

total_orig = total_novo = total_n = 0
for label, fname in ARQUIVOS:
    path = os.path.join(PILOTO, fname)
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    orig_vazias = sum(1 for r in recs if not r.get("triples"))
    novo_vazias = sum(1 for r in recs
                     if not parsear_resposta(r.get("response", "")))
    ganho = orig_vazias - novo_vazias

    total_n    += len(recs)
    total_orig += orig_vazias
    total_novo += novo_vazias

    orig_pct = orig_vazias / len(recs) * 100
    novo_pct = novo_vazias / len(recs) * 100
    print(f"{label:<35} {len(recs):>4} {orig_vazias:>4} ({orig_pct:4.1f}%)  "
          f"{novo_vazias:>4} ({novo_pct:4.1f}%)  "
          f"{'+' if ganho >= 0 else ''}{ganho:>3}")

print("-" * 65)
print(f"{'TOTAL':<35} {total_n:>4} {total_orig:>4} ({total_orig/total_n*100:4.1f}%)  "
      f"{total_novo:>4} ({total_novo/total_n*100:4.1f}%)  "
      f"+{total_orig - total_novo:>3}")

print()
print("Notas:")
print("  - 'ganho' = respostas que passaram a ter triplas com o parser D-18")
print("  - Parser D-18: adiciona limpeza de parênteses externos ((rel(sub,obj)))")
print("  - (rel, sub, obj) continua não interpretado — argumento posicional ambíguo")
print("  - A maioria das 'vazias' C1 requer nova execução com o prompt D-18")
