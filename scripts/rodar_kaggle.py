"""Dispatcher do lançador Kaggle — um único ponto de entrada para os dois fluxos.

Chamado pela célula de execução do notebook (uma linha). O usuário só troca a variável
MODO no topo do notebook (e MANIFESTO, no modo experimento).

  --modo inducao_c2   : roda induzir_c2_lote.py (manifesto_c2_inducao_29.json) e faz push
                        verificado de configs/c2/v_full/ e resultados/c2_inducao/v_full/.
  --modo experimento  : carrega o MANIFESTO, roda o guard check da C2 e executa com push
                        por experimento (--auto_push).

Flags de teste: --mock (ProvedorMock, sem GPU) e --no_push (não toca no git) — usadas pelo
smoke. Em produção no Kaggle, nenhuma das duas é passada.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(REPO, "scripts")


def sh(cmd, env, cwd=None):
    print("  $", " ".join(os.path.basename(c) if str(c).endswith(".py") else str(c) for c in cmd))
    r = subprocess.run(cmd, env=env, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"Falhou (rc={r.returncode})")


def git(gargs, repo_dir):
    return subprocess.run(["git", "-C", repo_dir] + gargs, capture_output=True, text=True)


def _env(repo_dir):
    return {**os.environ, "PYTHONPATH": os.path.join(repo_dir, "src"),
            "PYTHONIOENCODING": "utf-8"}


def modo_inducao(args):
    env = _env(args.repo_dir)
    run_id = args.run_id or datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cmd = [sys.executable, os.path.join(args.repo_dir, "scripts", "induzir_c2_lote.py"),
           "--manifesto", args.manifesto_inducao, "--run_id", run_id,
           "--c2_base", args.c2_base]
    if args.slug:
        cmd += ["--slug", args.slug]
    if args.mock:
        cmd += ["--mock"]
    else:
        cmd += ["--modelo_llm", args.modelo, "--url", args.url]
    sh(cmd, env, cwd=args.repo_dir)

    if args.no_push:
        print("Indução concluída (--no_push: sem git).")
        return
    alvo = ["configs/c2/v_full/", "resultados/c2_inducao/v_full/"]
    git(["add"] + alvo, args.repo_dir)
    if git(["status", "--porcelain"] + alvo, args.repo_dir).stdout.strip():
        git(["commit", "-m", "auto: inducao C2 v_full (29 ontologias)"], args.repo_dir)
        r = git(["push"], args.repo_dir)
        print("Push indução OK." if r.returncode == 0 else f"AVISO push: {r.stderr[:200]}")
    else:
        print("Indução: nada novo para commitar.")


def modo_experimento(args):
    env = _env(args.repo_dir)
    with open(args.manifesto, encoding="utf-8") as f:
        exps = json.load(f)
    if isinstance(exps, dict):
        exps = exps.get("experimentos", [])
    for e in exps:
        e["saida"] = f"{args.repo_dir}/{e['saida']}"
        if args.mock:
            e["provedor"] = "mock"
    mk = os.path.join(args.work_dir, "manifesto_kaggle.json")
    with open(mk, "w", encoding="utf-8") as f:
        json.dump(exps, f, ensure_ascii=False, indent=2)
    print(f"{len(exps)} experimento(s) -> {mk}")

    # Guard check da C2 (bloqueia se faltar ontologia induzida do modelo)
    sh([sys.executable, os.path.join(args.repo_dir, "scripts", "guard_check_c2.py"),
        "--manifesto", mk, "--repo", args.repo_dir], env, cwd=args.repo_dir)

    cmd = [sys.executable, "-m", "t2kg.run", "--manifesto", mk,
           "--bench_dir", args.bench_dir, "--skip_existentes", "--verbose"]
    if not args.mock and not args.no_push:
        cmd += ["--auto_push", "--push_repo_dir", args.repo_dir]
    sh(cmd, env, cwd=args.repo_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modo", required=True, choices=["inducao_c2", "experimento"])
    p.add_argument("--repo_dir", default=REPO)
    p.add_argument("--bench_dir", default=os.path.abspath(os.path.join(REPO, "..", "Text2KGBench")))
    p.add_argument("--work_dir", default=os.path.abspath(os.path.join(REPO, "..")))
    p.add_argument("--modelo", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    p.add_argument("--url", default="http://localhost:8000/v1")
    p.add_argument("--manifesto", default=None,
                   help="Manifesto do experimento (modo experimento).")
    p.add_argument("--manifesto_inducao",
                   default=os.path.join(REPO, "configs", "pipeline", "manifesto_c2_inducao_29.json"),
                   help="Manifesto de indução (modo inducao_c2).")
    p.add_argument("--c2_base", default="v_full")
    p.add_argument("--slug", default=None)
    p.add_argument("--run_id", default=None)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--no_push", action="store_true")
    args = p.parse_args()

    if args.modo == "inducao_c2":
        modo_inducao(args)
    else:
        if not args.manifesto:
            print("ERRO: modo experimento exige --manifesto.", file=sys.stderr)
            sys.exit(1)
        modo_experimento(args)
    print(f"Concluído: modo {args.modo}")


if __name__ == "__main__":
    main()
