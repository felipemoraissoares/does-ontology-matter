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
import re
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(REPO, "scripts")

# Módulos verificados no teste de import do modo congelar_ambiente.
IMPORTS_TESTE = ["vllm", "torch", "transformers", "sentence_transformers", "nltk"]


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


def _canon(nome: str) -> str:
    return re.sub(r"[-_.]+", "-", nome).strip().lower()


def _pacotes_base(repo_dir: str):
    """Nomes dos pacotes de requirements-kaggle.txt (sem versão)."""
    req = os.path.join(repo_dir, "requirements-kaggle.txt")
    nomes = []
    with open(req, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha or linha.startswith("#"):
                continue
            nome = re.split(r"[<>=!~;\[\s]", linha, 1)[0].strip()
            if nome:
                nomes.append(nome)
    return nomes


def _cuda_torch() -> str:
    try:
        import torch
        return torch.version.cuda or "cpu"
    except Exception:
        return "desconhecido"


def gerar_lock(repo_dir: str) -> str:
    """Monta o conteúdo de requirements-kaggle.lock.txt: pacotes do projeto + suas
    dependências diretas, com as versões EFETIVAS instaladas, e cabeçalho (Python/CUDA/Data).
    """
    import importlib.metadata as md

    tops = _pacotes_base(repo_dir)
    candidatos = {_canon(t) for t in tops}
    for t in tops:
        try:
            for req in (md.requires(t) or []):
                nome = re.split(r"[<>=!~;\[\s(]", req, 1)[0].strip()
                if nome:
                    candidatos.add(_canon(nome))
        except md.PackageNotFoundError:
            pass

    instaladas = {}
    for dist in md.distributions():
        nome = (dist.metadata["Name"] if dist.metadata else None)
        if nome and _canon(nome) in candidatos:
            instaladas[_canon(nome)] = (nome, dist.version)

    linhas = [f"{n}=={v}" for _, (n, v) in sorted(instaladas.items())]
    cab = [
        "# requirements-kaggle.lock.txt — gerado por MODO=congelar_ambiente no Kaggle (D-35).",
        f"# Python: {sys.version.split()[0]}",
        f"# CUDA (torch): {_cuda_torch()}",
        f"# Data: {datetime.date.today().isoformat()}",
        "",
    ]
    return "\n".join(cab + linhas) + "\n"


def modo_congelar(args):
    # 1. Teste rápido de import (pula imports pesados em --mock)
    if args.mock:
        print("[mock] pulando teste de import de vllm/torch/...")
    else:
        faltando = []
        for m in IMPORTS_TESTE:
            try:
                __import__(m)
            except Exception as e:
                faltando.append(f"{m}: {type(e).__name__}: {str(e)[:120]}")
        if faltando:
            raise RuntimeError("Teste de import falhou:\n  " + "\n  ".join(faltando))
        print(f"Teste de import OK: {', '.join(IMPORTS_TESTE)}")

    # 2. Gera e grava o lock
    lock_txt = gerar_lock(args.repo_dir)
    lock_path = args.lock_out or os.path.join(args.repo_dir, "requirements-kaggle.lock.txt")
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(lock_txt)
    n_pkgs = sum(1 for l in lock_txt.splitlines() if l and not l.startswith("#"))
    print(f"Lock gravado: {lock_path} ({n_pkgs} pacotes)")

    # 3. Push verificado (salvo em --no_push)
    if args.no_push:
        print("(--no_push: sem git)")
        return
    rel = os.path.relpath(lock_path, args.repo_dir).replace("\\", "/")
    git(["add", rel], args.repo_dir)
    if git(["status", "--porcelain", rel], args.repo_dir).stdout.strip():
        git(["commit", "-m", "auto: requirements-kaggle.lock.txt (congelar_ambiente)"], args.repo_dir)
        r = git(["push"], args.repo_dir)
        if r.returncode == 0:
            print("Push do lock OK.")
        else:
            raise RuntimeError(f"Push do lock falhou: {r.stderr[:200]}")
    else:
        print("Lock inalterado — nada para commitar.")


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
    p.add_argument("--modo", required=True,
                   choices=["congelar_ambiente", "inducao_c2", "experimento"])
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
    p.add_argument("--lock_out", default=None,
                   help="Caminho do lock a gravar (padrão: <repo>/requirements-kaggle.lock.txt).")
    args = p.parse_args()

    if args.modo == "congelar_ambiente":
        modo_congelar(args)
    elif args.modo == "inducao_c2":
        modo_inducao(args)
    else:
        if not args.manifesto:
            print("ERRO: modo experimento exige --manifesto.", file=sys.stderr)
            sys.exit(1)
        modo_experimento(args)
    print(f"Concluído: modo {args.modo}")


if __name__ == "__main__":
    main()
