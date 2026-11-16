"""Smoke do lançador Kaggle nos DOIS modos, com ProvedorMock e sem GPU.

Exercita o dispatcher scripts/rodar_kaggle.py de ponta a ponta e offline:
  MODO=inducao_c2  -> induzir_c2_lote (mock) grava configs/c2/v_smoke_tmp/mock/
  MODO=experimento -> guard check + t2kg.run (mock) das C2 usando c2_subdir

Usa 2 ontologias e poucas frases. Escreve em diretórios temporários e os remove no fim.

Uso:
  python scripts/smoke_experimento_completo.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE = os.path.abspath(os.path.join(REPO, ".."))
BENCH = os.path.join(WORKSPACE, "Text2KGBench")
SCRIPTS = os.path.join(REPO, "scripts")

SLUG = "mock"
C2_BASE = "v_smoke_tmp"
ONTOS = [("tekgen", "1_movie"), ("webnlg", "1_university")]


def run(cmd, env):
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"Falhou (rc={r.returncode})")
    return r


def main():
    if not os.path.exists(BENCH):
        print(f"ERRO: Text2KGBench não encontrado em {BENCH}", file=sys.stderr)
        return 1

    env = {**os.environ, "PYTHONPATH": os.path.join(REPO, "src"),
           "PYTHONIOENCODING": "utf-8"}
    tmp = tempfile.mkdtemp(prefix="smoke_full_")
    c2_dir = os.path.join(REPO, "configs", "c2", C2_BASE)
    raw_dir = os.path.join(REPO, "resultados", "c2_inducao", C2_BASE)
    out_dir = os.path.join(REPO, "resultados", "full", C2_BASE)
    rodar = os.path.join(SCRIPTS, "rodar_kaggle.py")

    try:
        # ---- MODO inducao_c2 ----
        print("--- MODO inducao_c2 (mock) ---")
        tarefas = [{"dataset": ds, "onto": o, "n_frases": 3, "seed": 42,
                    "threshold": 0.70, "freq_min": 1, "calibrar": False, "ativo": True}
                   for ds, o in ONTOS]
        man_ind = os.path.join(tmp, "inducao.json")
        with open(man_ind, "w", encoding="utf-8") as f:
            json.dump(tarefas, f)
        run([sys.executable, rodar, "--modo", "inducao_c2", "--mock", "--no_push",
             "--manifesto_inducao", man_ind, "--c2_base", C2_BASE, "--slug", SLUG,
             "--repo_dir", REPO, "--run_id", "SMOKE"], env)

        vfull = os.path.join(c2_dir, SLUG)
        for ds, o in ONTOS:
            assert os.path.exists(os.path.join(vfull, f"{ds}_{o}_induced.json")), f"induzida {ds}/{o}"
            assert os.path.exists(os.path.join(vfull, f"definicoes_{o}.json")), f"defs {ds}/{o}"
        print("  OK: induzidas + definições em", os.path.relpath(vfull, REPO))

        # ---- MODO experimento ----
        print("\n--- MODO experimento (mock) ---")
        exps = [{
            "id": f"{ds}_{o}_c2_{SLUG}", "dataset": ds, "split": "test",
            "ontologia_id": o, "condicao": "C2", "provedor": "openai_compat",
            "c2_subdir": f"{C2_BASE}/{SLUG}",
            "saida": f"resultados/full/{C2_BASE}/{ds}_{o}_c2.jsonl",
            "frases_max": 3,
        } for ds, o in ONTOS]
        man_exp = os.path.join(tmp, "completo.json")
        with open(man_exp, "w", encoding="utf-8") as f:
            json.dump(exps, f)
        run([sys.executable, rodar, "--modo", "experimento", "--mock", "--no_push",
             "--manifesto", man_exp, "--repo_dir", REPO, "--bench_dir", BENCH,
             "--work_dir", tmp], env)

        for e in exps:
            saida = os.path.join(REPO, e["saida"])
            assert os.path.exists(saida) and os.path.getsize(saida) > 0, f"saída {e['id']}"
        print("  OK: experimentos C2 geraram saída")

        print("\n" + "=" * 60)
        print("SMOKE OK — lançador nos dois modos (inducao_c2 + experimento)")
        print("=" * 60)
        return 0
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"\nERRO: {exc}", file=sys.stderr)
        return 1
    finally:
        for d in (c2_dir, raw_dir, out_dir, tmp):
            shutil.rmtree(d, ignore_errors=True)
        print("Limpeza concluída.")


if __name__ == "__main__":
    sys.exit(main())
