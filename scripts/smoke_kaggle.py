#!/usr/bin/env python3
"""
Teste de fumaça que simula o fluxo do lançador Kaggle (sem GPU).

Passos:
  1. Clona does-ontology-matter do GitHub numa pasta temporária
  2. Clona Text2KGBench no commit fixo
  3. Injeta provedor=mock e frases_max=3 no manifesto do piloto
  4. Executa todos os 6 experimentos do piloto
  5. Verifica que os 6 arquivos foram criados em resultados/piloto/
  6. Mostra git status para confirmar que .gitignore os enxerga

Uso:
  python scripts/smoke_kaggle.py [--keep]
  python scripts/smoke_kaggle.py --repo-local   # usa o repo local em vez de clonar
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO_URL     = "https://github.com/felipemoraissoares/does-ontology-matter.git"
BENCH_URL    = "https://github.com/cenguix/Text2KGBench.git"
BENCH_COMMIT = "50a3d255371b8817cdff70fd88459ac82b339cfe"

GIT = shutil.which("git") or "D:\\Programas\\Git\\cmd\\git.exe"


def run(cmd, cwd=None, env=None, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd, env=env)
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.stderr.strip():
        print(r.stderr.strip(), file=sys.stderr)
    if check and r.returncode != 0:
        raise RuntimeError(f"Falhou (rc={r.returncode}): {cmd[:120]}")
    return r


def main():
    p = argparse.ArgumentParser(description="Smoke test — simula Kaggle sem GPU.")
    p.add_argument("--keep", action="store_true",
                   help="Mantém o diretório temporário após o teste.")
    p.add_argument("--repo-local", action="store_true",
                   help="Usa o repo local (sem clonar do GitHub). "
                        "O Text2KGBench ainda é clonado normalmente.")
    args = p.parse_args()

    tmp = tempfile.mkdtemp(prefix="smoke_kaggle_")
    print(f"\n{'='*60}")
    print(f"Smoke test — diretório: {tmp}")
    print(f"{'='*60}\n")

    try:
        repo_dir  = os.path.join(tmp, "does-ontology-matter")
        bench_dir = os.path.join(tmp, "Text2KGBench")

        # ------------------------------------------------------------------
        # 1. Obter repositório
        # ------------------------------------------------------------------
        if args.repo_local:
            local = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            print(f"--- 1. Copiando repo local: {local} ---")
            shutil.copytree(local, repo_dir,
                            ignore=shutil.ignore_patterns(".git", "__pycache__",
                                                          "*.egg-info", ".venv",
                                                          "resultados"))
            os.makedirs(os.path.join(repo_dir, "resultados", "piloto"), exist_ok=True)
            # copia .gitignore e .git para git status funcionar
            shutil.copy(os.path.join(local, ".gitignore"), repo_dir)
            run(f'"{GIT}" -C "{repo_dir}" init')
            run(f'"{GIT}" -C "{repo_dir}" add -A')
        else:
            print("--- 1. Clonando repositório do GitHub ---")
            run(f'"{GIT}" clone --depth=1 {REPO_URL} "{repo_dir}"')

        # ------------------------------------------------------------------
        # 2. Clonar Text2KGBench no commit fixo
        # ------------------------------------------------------------------
        print(f"\n--- 2. Clonando Text2KGBench @ {BENCH_COMMIT[:12]} ---")
        run(f'"{GIT}" init "{bench_dir}"')
        run(f'"{GIT}" -C "{bench_dir}" fetch --depth=1 {BENCH_URL} {BENCH_COMMIT}')
        run(f'"{GIT}" -C "{bench_dir}" checkout FETCH_HEAD')
        actual = run(f'"{GIT}" -C "{bench_dir}" rev-parse HEAD', check=False).stdout.strip()
        if actual != BENCH_COMMIT:
            print(f"AVISO: commit obtido {actual[:12]} ≠ esperado {BENCH_COMMIT[:12]}")
        else:
            print(f"OK — commit {BENCH_COMMIT[:12]}")

        # ------------------------------------------------------------------
        # 3. Preparar manifesto do piloto com Mock + frases_max=3
        # ------------------------------------------------------------------
        print("\n--- 3. Preparando manifesto (Mock, frases_max=3) ---")
        result_dir  = os.path.join(repo_dir, "resultados", "piloto")
        os.makedirs(result_dir, exist_ok=True)

        piloto_src = os.path.join(repo_dir, "configs", "pipeline",
                                  "manifesto_piloto_kaggle.json")
        with open(piloto_src, encoding="utf-8") as f:
            piloto = json.load(f)

        for exp in piloto:
            exp["provedor"]   = "mock"
            exp["frases_max"] = 3
            # garante que saida aponta para result_dir local do tmp
            basename = os.path.basename(exp["saida"])
            exp["saida"] = os.path.join(result_dir, basename).replace("\\", "/")

        smoke_manifest = os.path.join(tmp, "manifesto_smoke.json")
        with open(smoke_manifest, "w", encoding="utf-8") as f:
            json.dump(piloto, f, ensure_ascii=False, indent=2)
        print(f"{len(piloto)} experimentos -> {smoke_manifest}")

        # ------------------------------------------------------------------
        # 4. Executar pipeline com Mock
        # ------------------------------------------------------------------
        print("\n--- 4. Executando pipeline (provedor=mock) ---")
        env = {**os.environ, "PYTHONPATH": os.path.join(repo_dir, "src")}
        run(
            f'python -m t2kg.run '
            f'--manifesto "{smoke_manifest}" '
            f'--bench_dir "{bench_dir}" '
            f'--verbose',
            cwd=repo_dir,
            env=env,
        )

        # ------------------------------------------------------------------
        # 5. Verificar arquivos criados
        # ------------------------------------------------------------------
        print("\n--- 5. Arquivos em resultados/piloto/ ---")
        criados = []
        for fname in sorted(os.listdir(result_dir)):
            fpath = os.path.join(result_dir, fname)
            size  = os.path.getsize(fpath)
            # conta linhas do JSONL
            with open(fpath, encoding="utf-8") as fh:
                n_linhas = sum(1 for _ in fh)
            print(f"  {fname}  ({size} bytes, {n_linhas} linhas)")
            criados.append(fname)

        assert len(criados) == 6, (
            f"Esperado 6 arquivos JSONL, encontrado {len(criados)}."
        )

        # ------------------------------------------------------------------
        # 6. Git status
        # ------------------------------------------------------------------
        print("\n--- 6. Git status (resultados/piloto/ deve aparecer) ---")
        run(f'"{GIT}" -C "{repo_dir}" status --short', check=False)

        print(f"\n{'='*60}")
        print("SMOKE TEST OK — 6/6 experimentos, arquivos em resultados/piloto/")
        print(f"{'='*60}\n")
        return 0

    except Exception as exc:
        print(f"\nERRO: {exc}", file=sys.stderr)
        return 1

    finally:
        if args.keep:
            print(f"Diretório mantido: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"Diretório temporário removido.")


if __name__ == "__main__":
    sys.exit(main())
