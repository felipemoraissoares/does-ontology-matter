#!/usr/bin/env python3
"""
Smoke test que espelha as células 8-13 do kaggle/lancador.ipynb com ProvedorMock.

Executa EXATAMENTE os mesmos comandos do lançador Kaggle, substituindo:
  - vLLM → ProvedorMock (via --mock / provedor=mock)
  - n_frases=50 → n_frases=3  (velocidade)
  - frases_max=100 → frases_max=3

Fases cobertas (correspondência com células do notebook):
  Cel 9  — (a) extração livre --mock  |  (b) definições --mock
           (c) re-indução --use_definicoes  |  (d) calibração --from_cache
  Cel 10 — guard check: run_id correto em todos os induced.json
  Cel 11 — 28 experimentos via t2kg.run (provedor=mock, frases_max=3)
  Cel 13 — avaliar_resultados.py

Coberturas das 4 ontologias do piloto-v3:
  tekgen/1_movie, tekgen/2_music, webnlg/1_university, webnlg/19_film

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
    p = argparse.ArgumentParser(description="Smoke test — espelha lançador Kaggle sem GPU.")
    p.add_argument("--keep", action="store_true",
                   help="Mantém o diretório temporário após o teste.")
    p.add_argument("--repo-local", action="store_true",
                   help="Usa o repo local (sem clonar do GitHub). "
                        "O Text2KGBench ainda é clonado normalmente.")
    args = p.parse_args()

    tmp = tempfile.mkdtemp(prefix="smoke_kaggle_")
    print(f"\n{'='*60}")
    print(f"Smoke test piloto-v3 — dir: {tmp}")
    print(f"{'='*60}\n")

    try:
        repo_dir  = os.path.join(tmp, "does-ontology-matter")
        bench_dir = os.path.join(tmp, "Text2KGBench")

        # Espelha as variáveis do notebook cel 1
        c2_dir     = os.path.join(repo_dir, "configs", "c2")
        c2_raw_dir = os.path.join(repo_dir, "resultados", "c2_inducao")
        pv3_dir    = os.path.join(repo_dir, "resultados", "piloto_v3")
        aval_dir   = os.path.join(repo_dir, "resultados", "avaliacao_piloto_v3")

        # ------------------------------------------------------------------
        # Passo 1 — Obter repositório (espelha cel 1/3)
        # ------------------------------------------------------------------
        if args.repo_local:
            local = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            print(f"--- 1. Copiando repo local: {local} ---")
            shutil.copytree(local, repo_dir,
                            ignore=shutil.ignore_patterns(".git", "__pycache__",
                                                          "*.egg-info", ".venv",
                                                          "resultados"))
            for d in (pv3_dir, c2_raw_dir):
                os.makedirs(d, exist_ok=True)
            shutil.copy(os.path.join(local, ".gitignore"), repo_dir)
            run(f'"{GIT}" -C "{repo_dir}" init')
            run(f'"{GIT}" -C "{repo_dir}" add -A')
        else:
            print("--- 1. Clonando repositório do GitHub ---")
            run(f'"{GIT}" clone --depth=1 {REPO_URL} "{repo_dir}"')
            for d in (pv3_dir, c2_raw_dir):
                os.makedirs(d, exist_ok=True)

        # ------------------------------------------------------------------
        # Passo 2 — Clonar Text2KGBench (espelha cel 3)
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

        env = {**os.environ, "PYTHONPATH": os.path.join(repo_dir, "src")}

        # ------------------------------------------------------------------
        # Passo 3 — Cel 9: C2 induction (a)+(b)+(c)+(d) por tarefa
        # ------------------------------------------------------------------
        print("\n--- 3. Cel 9: C2 induction (a/b/c/d) ---")
        manifesto_c2_path = os.path.join(repo_dir, "configs", "pipeline",
                                         "reinducao_c2_kaggle.json")
        with open(manifesto_c2_path, encoding="utf-8") as f:
            manifesto_c2 = json.load(f)
        tarefas_ativas = [t for t in manifesto_c2 if t.get("ativo", True)]
        print(f"Tarefas C2 ativas: {len(tarefas_ativas)}")

        SMOKE_RUN_ID = "SMOKE_V3_LOCAL"

        for tarefa in tarefas_ativas:
            onto    = tarefa["onto"]
            dataset = tarefa["dataset"]
            seed    = tarefa["seed"]
            raw_basename = os.path.basename(tarefa["saida_raw_jsonl"])
            raw_jsonl = os.path.join(c2_raw_dir, raw_basename)

            # (a) Extração livre treino — --mock substitui vLLM; n_frases=3 para rapidez
            # Célula 9 original usa: --url {PORT}/v1 --modelo_llm {MODELO}
            print(f"\n  (a) extração livre [--mock]: {dataset}/{onto}")
            run(
                f"python \"{os.path.join(repo_dir, 'scripts', 'induzir_c2.py')}\" "
                f"--dataset {dataset} "
                f"--onto {onto} "
                f"--threshold {tarefa['threshold']} "
                f"--freq_min {tarefa['freq_min']} "
                f"--n_frases 3 "
                f"--seed {seed} "
                f"--mock "
                f"--saida_raw_jsonl \"{raw_jsonl}\" "
                f"--saida_dir \"{c2_dir}\" "
                f"--normalizar "
                f"--run_id {SMOKE_RUN_ID}",
                cwd=repo_dir, env=env,
            )

            # (b) Definições D-24 — --mock substitui vLLM
            # Célula 9 original usa: --url {PORT}/v1 --modelo_llm {MODELO}
            print(f"\n  (b) definições D-24 [--mock]: {dataset}/{onto}")
            run(
                f"python \"{os.path.join(repo_dir, 'scripts', 'gerar_definicoes_relacoes.py')}\" "
                f"--onto_id {onto} "
                f"--dataset {dataset} "
                f"--source_jsonl \"{raw_jsonl}\" "
                f"--mock",
                cwd=repo_dir, env=env,
            )

            # (c) Re-indução EDC --use_definicoes (sem LLM: usa --from_raw_jsonl)
            # Célula 9 original: mesma linha sem --url/--modelo_llm, com --use_definicoes
            print(f"\n  (c) re-indução EDC --use_definicoes: {dataset}/{onto}")
            run(
                f"python \"{os.path.join(repo_dir, 'scripts', 'induzir_c2.py')}\" "
                f"--dataset {dataset} "
                f"--onto {onto} "
                f"--threshold {tarefa['threshold']} "
                f"--freq_min {tarefa['freq_min']} "
                f"--n_frases 3 "
                f"--seed {seed} "
                f"--from_raw_jsonl \"{raw_jsonl}\" "
                f"--saida_dir \"{c2_dir}\" "
                f"--normalizar "
                f"--use_definicoes "
                f"--run_id {SMOKE_RUN_ID}",
                cwd=repo_dir, env=env,
            )

            # (d) Tabela calibração --from_cache (check=False igual ao notebook)
            # Célula 9 original: --from_cache (sem LLM)
            print(f"\n  (d) calibração --from_cache: {dataset}/{onto}")
            run(
                f"python \"{os.path.join(repo_dir, 'scripts', 'calibrar_c2_com_definicoes.py')}\" "
                f"--onto_id {onto} "
                f"--dataset {dataset} "
                f"--from_cache",
                cwd=repo_dir, env=env, check=False,
            )

        # ------------------------------------------------------------------
        # Passo 4 — Cel 10: Guard check
        # ------------------------------------------------------------------
        print("\n--- 4. Cel 10: Guard check C2 ---")
        erros = []
        for tarefa in tarefas_ativas:
            onto_path = os.path.join(c2_dir,
                                     f"{tarefa['dataset']}_{tarefa['onto']}_induced.json")
            if not os.path.exists(onto_path):
                erros.append(f"AUSENTE: {onto_path}")
                continue
            with open(onto_path, encoding="utf-8") as f:
                run_id_onto = json.load(f).get("metadata", {}).get("run_id")
            if run_id_onto != SMOKE_RUN_ID:
                erros.append(
                    f"RUN_ID INCORRETO: {os.path.basename(onto_path)}\n"
                    f"  Encontrado: {run_id_onto!r}  Esperado: {SMOKE_RUN_ID!r}"
                )
            else:
                print(f"  OK: {tarefa['dataset']}/{tarefa['onto']} run_id={run_id_onto}")
        if erros:
            raise RuntimeError(
                f"Guard check C2 falhou — {len(erros)} erro(s):\n"
                + "\n".join(f"  * {e}" for e in erros)
            )

        # ------------------------------------------------------------------
        # Passo 5 — Cel 8+11: Preparar manifesto e executar experimentos
        # ------------------------------------------------------------------
        print("\n--- 5. Cel 8+11: Manifesto piloto-v3 + experimentos ---")
        manifesto_src = os.path.join(repo_dir, "configs", "pipeline",
                                     "manifesto_piloto_v3.json")
        with open(manifesto_src, encoding="utf-8") as f:
            exps = json.load(f)
        for exp in exps:
            exp["provedor"]   = "mock"
            exp["frases_max"] = 3
            exp["saida"] = os.path.join(pv3_dir,
                                        os.path.basename(exp["saida"])).replace("\\", "/")
        manifesto_smoke = os.path.join(tmp, "manifesto_smoke_v3.json")
        with open(manifesto_smoke, "w", encoding="utf-8") as f:
            json.dump(exps, f, ensure_ascii=False, indent=2)
        print(f"{len(exps)} experimentos → {manifesto_smoke}")

        run(
            f"python -m t2kg.run "
            f"--manifesto \"{manifesto_smoke}\" "
            f"--bench_dir \"{bench_dir}\" "
            f"--skip_existentes "
            f"--verbose",
            cwd=repo_dir, env=env,
        )

        # ------------------------------------------------------------------
        # Passo 6 — Cel 13: Avaliar resultados
        # ------------------------------------------------------------------
        print("\n--- 6. Cel 13: Avaliando resultados ---")
        os.makedirs(aval_dir, exist_ok=True)
        csv_path = os.path.join(aval_dir, "tabela_smoke_v3.csv")
        run(
            f"python \"{os.path.join(repo_dir, 'scripts', 'avaliar_resultados.py')}\" "
            f"--manifesto \"{manifesto_smoke}\" "
            f"--bench_dir \"{bench_dir}\" "
            f"--avaliacao_dir \"{aval_dir}\" "
            f"--csv \"{csv_path}\" "
            f"--extended",
            cwd=repo_dir, env=env,
        )

        # ------------------------------------------------------------------
        # Passo 7 — Verificar saídas
        # ------------------------------------------------------------------
        print("\n--- 7. Verificando arquivos criados ---")
        criados = []
        for fname in sorted(os.listdir(pv3_dir)):
            fpath = os.path.join(pv3_dir, fname)
            size  = os.path.getsize(fpath)
            with open(fpath, encoding="utf-8") as fh:
                n_linhas = sum(1 for _ in fh)
            print(f"  {fname}  ({size} bytes, {n_linhas} linhas)")
            criados.append(fname)

        assert len(criados) == len(exps), (
            f"Esperado {len(exps)} arquivos JSONL, encontrado {len(criados)}."
        )
        assert os.path.exists(csv_path), f"CSV de avaliação não gerado: {csv_path}"

        print(f"\n{'='*60}")
        print(
            f"SMOKE TEST OK — {len(criados)}/{len(exps)} experimentos, "
            f"C2 induzido para {len(tarefas_ativas)} ontologia(s), avaliação OK"
        )
        print(f"{'='*60}\n")
        return 0

    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"\nERRO: {exc}", file=sys.stderr)
        return 1

    finally:
        if args.keep:
            print(f"Diretório mantido: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
            print("Diretório temporário removido.")


if __name__ == "__main__":
    sys.exit(main())
