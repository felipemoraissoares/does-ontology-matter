"""Análise de sensibilidade D-37: re-induz a C2 com threshold 0.80 (em vez de 0.70),
reaproveitando as respostas cruas e as definições D-24 já salvas — SEM GPU e SEM LLM.

NÃO altera `configs/c2/v_full/` (threshold 0.70). Escreve em `configs/c2/v_full_s080/<slug>/`.
Ao final, mostra quantas relações cada ontologia teria em 0.70 (v_full) vs 0.80 (v_full_s080).

Uso (no PC, depois do Kaggle):
  python scripts/reinduzir_s080.py --slug qwen2.5-7b
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(REPO, "scripts")

TEK = ["1_movie", "2_music", "3_sport", "4_book", "5_military", "6_computer",
       "7_space", "8_politics", "9_nature", "10_culture"]
WEB = ["1_university", "2_musicalwork", "3_airport", "4_building", "5_athlete",
       "6_politician", "7_company", "8_celestialbody", "9_astronaut", "10_comicscharacter",
       "11_meanoftransportation", "12_monument", "13_food", "14_writtenwork", "15_sportsteam",
       "16_city", "17_artist", "18_scientist", "19_film"]


def _n_rels(caminho):
    if not os.path.exists(caminho):
        return None
    with open(caminho, encoding="utf-8") as f:
        return len(json.load(f).get("relations", []))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slug", default="qwen2.5-7b")
    p.add_argument("--threshold", type=float, default=0.80)
    p.add_argument("--freq_min", type=int, default=3)
    p.add_argument("--n_frases", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src = os.path.join(REPO, "configs", "c2", "v_full", args.slug)
    dst = os.path.join(REPO, "configs", "c2", "v_full_s080", args.slug)
    raw_dir = os.path.join(REPO, "resultados", "c2_inducao", "v_full", args.slug)
    if not os.path.isdir(src):
        print(f"ERRO: {src} não encontrado (rode a indução no Kaggle primeiro).", file=sys.stderr)
        sys.exit(1)
    os.makedirs(dst, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": os.path.join(REPO, "src"), "PYTHONIOENCODING": "utf-8"}

    todos = [("tekgen", o) for o in TEK] + [("webnlg", o) for o in WEB]
    for ds, onto in todos:
        raw = os.path.join(raw_dir, f"raw_{ds}_{onto}_treino.jsonl")
        defs_src = os.path.join(src, f"definicoes_{onto}.json")
        if not (os.path.exists(raw) and os.path.exists(defs_src)):
            print(f"  pulando {ds}/{onto} (raw/def ausente)")
            continue
        # induzir_c2 --use_definicoes lê as definições de --saida_dir; copiamos para lá.
        shutil.copy(defs_src, os.path.join(dst, f"definicoes_{onto}.json"))
        r = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS, "induzir_c2.py"),
             "--dataset", ds, "--onto", onto, "--threshold", str(args.threshold),
             "--freq_min", str(args.freq_min), "--n_frases", str(args.n_frases),
             "--seed", str(args.seed), "--from_raw_jsonl", raw, "--saida_dir", dst,
             "--normalizar", "--use_definicoes", "--run_id", "S080"],
            env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FALHOU {ds}/{onto}:\n{r.stderr[-300:]}")
        else:
            print(f"  OK {ds}/{onto}")

    print("\n=== Relações por ontologia: 0.70 (v_full) x 0.80 (v_full_s080) ===")
    print(f"{'ontologia':>26} | {'0.70':>5} | {'0.80':>5} | {'dif':>4}")
    print("-" * 50)
    tot70 = tot80 = 0
    for ds, onto in todos:
        n70 = _n_rels(os.path.join(src, f"{ds}_{onto}_induced.json"))
        n80 = _n_rels(os.path.join(dst, f"{ds}_{onto}_induced.json"))
        if n70 is None or n80 is None:
            continue
        tot70 += n70
        tot80 += n80
        print(f"{ds+'/'+onto:>26} | {n70:>5} | {n80:>5} | {n80 - n70:>+4}")
    print("-" * 50)
    print(f"{'TOTAL':>26} | {tot70:>5} | {tot80:>5} | {tot80 - tot70:>+4}")


if __name__ == "__main__":
    main()
