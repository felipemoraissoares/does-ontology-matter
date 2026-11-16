"""Smoke do fluxo do experimento completo (Etapa C) com ProvedorMock, sem GPU.

Exercita, de ponta a ponta e offline:
  1. indução C2 em lote (induzir_c2_lote.py --mock) -> configs/c2/v_smoke_tmp/mock/
  2. guard check (guard_check_c2.py) de um manifesto com C2 por modelo
  3. execução dos experimentos C2 (t2kg.run, provedor mock) usando c2_subdir

Usa 2 ontologias e 3 frases para rapidez. Escreve em diretórios temporários
(configs/c2/v_smoke_tmp e resultados/c2_inducao/v_smoke_tmp) e os remove no fim.

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
    print("  $", " ".join(os.path.basename(c) if str(c).endswith(".py") else str(c) for c in cmd))
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

    try:
        # 1. Indução C2 em lote (mock)
        print("--- 1. Indução C2 em lote (mock) ---")
        tarefas = [{"dataset": ds, "onto": o, "n_frases": 3, "seed": 42,
                    "threshold": 0.70, "freq_min": 1, "calibrar": False, "ativo": True}
                   for ds, o in ONTOS]
        manif = os.path.join(tmp, "inducao.json")
        with open(manif, "w", encoding="utf-8") as f:
            json.dump(tarefas, f)
        run([sys.executable, os.path.join(SCRIPTS, "induzir_c2_lote.py"),
             "--manifesto", manif, "--mock", "--slug", SLUG,
             "--c2_base", C2_BASE, "--run_id", "SMOKE"], env)

        vfull = os.path.join(c2_dir, SLUG)
        for ds, o in ONTOS:
            assert os.path.exists(os.path.join(vfull, f"{ds}_{o}_induced.json")), \
                f"induzida ausente: {ds}/{o}"
            assert os.path.exists(os.path.join(vfull, f"definicoes_{o}.json")), \
                f"definições ausentes: {ds}/{o}"
        print("  Induzidas e definições OK em", os.path.relpath(vfull, REPO))

        # 2. Guard check de um manifesto completo pequeno
        print("\n--- 2. Guard check ---")
        exps = [{
            "id": f"{ds}_{o}_c2_{SLUG}", "dataset": ds, "split": "test",
            "ontologia_id": o, "condicao": "C2", "provedor": "mock",
            "c2_subdir": f"{C2_BASE}/{SLUG}",
            "saida": os.path.join(tmp, f"{ds}_{o}_c2.jsonl").replace("\\", "/"),
            "frases_max": 3,
        } for ds, o in ONTOS]
        completo = os.path.join(tmp, "completo.json")
        with open(completo, "w", encoding="utf-8") as f:
            json.dump(exps, f)
        run([sys.executable, os.path.join(SCRIPTS, "guard_check_c2.py"),
             "--manifesto", completo], env)

        # 3. Execução dos experimentos C2 (mock) usando c2_subdir
        print("\n--- 3. Execução C2 (mock) ---")
        run([sys.executable, "-m", "t2kg.run", "--manifesto", completo,
             "--bench_dir", BENCH], env)
        for e in exps:
            assert os.path.exists(e["saida"]) and os.path.getsize(e["saida"]) > 0, \
                f"saída vazia: {e['id']}"
        print("  Experimentos C2 geraram saída OK")

        print("\n" + "=" * 60)
        print("SMOKE EXPERIMENTO COMPLETO OK — indução + guard + execução C2")
        print("=" * 60)
        return 0
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"\nERRO: {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(c2_dir, ignore_errors=True)
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)
        print("Limpeza concluída.")


if __name__ == "__main__":
    sys.exit(main())
