"""Induz a C2 das 29 ontologias com o modelo do experimento (Etapa C, D-33/D-34).

Para cada tarefa do manifesto, roda (reaproveitando os scripts existentes):
  (a) extração livre C0 em N frases de TREINO (seed=42, D-16) -> raw jsonl
  (b) definições D-24 dos rótulos livres E das relações de referência -> v_full/<slug>/
  (c) re-indução EDC (--use_definicoes) -> v_full/<slug>/<dataset>_<onto>_induced.json

Roda NO KAGGLE com o Qwen 7B (uma sessão). Use --mock para o smoke local sem GPU.
As saídas vão para configs/c2/v_full/<slug>/ e NÃO sobrescrevem os arquivos do piloto.

Uso (Kaggle):
  python scripts/induzir_c2_lote.py \
      --manifesto configs/pipeline/manifesto_c2_inducao_29.json \
      --modelo_llm Qwen/Qwen2.5-7B-Instruct-AWQ \
      --url http://localhost:8000/v1 \
      --run_id <timestamp>
"""
import argparse
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from t2kg.modelos import slug_modelo  # noqa: E402

SCRIPTS = os.path.join(REPO, "scripts")


def _run(cmd, env):
    print("  $", " ".join(os.path.basename(c) if c.endswith(".py") else c for c in cmd))
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"Falhou (rc={r.returncode}): {' '.join(cmd)}")


def main():
    p = argparse.ArgumentParser(description="Indução C2 em lote (29 ontologias).")
    p.add_argument("--manifesto", required=True)
    p.add_argument("--modelo_llm", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    p.add_argument("--url", default="http://localhost:8000/v1")
    p.add_argument("--slug", default=None, help="Padrão: slug_modelo(--modelo_llm).")
    p.add_argument("--c2_base", default="v_full")
    p.add_argument("--run_id", default="LOTE")
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()

    slug = args.slug or slug_modelo(args.modelo_llm)
    saida_dir = os.path.join(REPO, "configs", "c2", args.c2_base, slug)
    raw_dir = os.path.join(REPO, "resultados", "c2_inducao", args.c2_base, slug)
    os.makedirs(saida_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    with open(args.manifesto, encoding="utf-8") as f:
        tarefas = json.load(f)
    tarefas = [t for t in tarefas if t.get("ativo", True)]

    env = {**os.environ, "PYTHONPATH": os.path.join(REPO, "src"),
           "PYTHONIOENCODING": "utf-8"}
    prov = ["--mock"] if args.mock else ["--url", args.url, "--modelo_llm", args.modelo_llm]

    print(f"Indução C2 em lote: {len(tarefas)} ontologias -> {os.path.relpath(saida_dir, REPO)}")
    for i, t in enumerate(tarefas, 1):
        ds, onto = t["dataset"], t["onto"]
        raw = os.path.join(raw_dir, f"raw_{ds}_{onto}_treino.jsonl")
        thr, fmin = str(t["threshold"]), str(t["freq_min"])
        nfr, seed = str(t["n_frases"]), str(t["seed"])
        print(f"\n[{i}/{len(tarefas)}] {ds}/{onto}")

        # (a) extração livre C0 -> raw
        _run([sys.executable, os.path.join(SCRIPTS, "induzir_c2.py"),
              "--dataset", ds, "--onto", onto, "--threshold", thr, "--freq_min", fmin,
              "--n_frases", nfr, "--seed", seed, "--saida_raw_jsonl", raw,
              "--saida_dir", saida_dir, "--normalizar", "--run_id", args.run_id] + prov, env)

        # (b) definições D-24 (rótulos livres + relações de referência) -> v_full/<slug>/
        _run([sys.executable, os.path.join(SCRIPTS, "gerar_definicoes_relacoes.py"),
              "--onto_id", onto, "--dataset", ds, "--source_jsonl", raw,
              "--saida_dir", saida_dir] + prov, env)

        # (c) re-indução EDC usando as definições -> induced.json definitivo
        _run([sys.executable, os.path.join(SCRIPTS, "induzir_c2.py"),
              "--dataset", ds, "--onto", onto, "--threshold", thr, "--freq_min", fmin,
              "--n_frases", nfr, "--seed", seed, "--from_raw_jsonl", raw,
              "--saida_dir", saida_dir, "--normalizar", "--use_definicoes",
              "--run_id", args.run_id], env)

    print(f"\nIndução completa: {len(tarefas)} ontologias em {os.path.relpath(saida_dir, REPO)}")
    print(f"Definições e raws em {os.path.relpath(raw_dir, REPO)}")


if __name__ == "__main__":
    main()
