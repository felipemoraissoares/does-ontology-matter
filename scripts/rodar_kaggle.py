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
sys.path.insert(0, os.path.join(REPO, "src"))
from t2kg.modelos import slug_modelo  # noqa: E402
from t2kg.git_sync import sincronizar_e_push  # noqa: E402

# Módulos verificados no teste de import do modo congelar_ambiente.
IMPORTS_TESTE = ["vllm", "torch", "transformers", "sentence_transformers", "nltk"]
# Único pacote travado no lock: determina o motor de inferência (D-35).
PACOTE_TRAVADO = "vllm"
# Pacotes cujas versões efetivas são impressas no log após a instalação.
VERSOES_LOG = ["vllm", "torch", "transformers", "sentence-transformers", "numpy"]


def _imprimir_versoes():
    """Imprime as versões efetivas instaladas (logo após a instalação, no run)."""
    import importlib.metadata as md
    print("Versões efetivas instaladas:")
    for pkg in VERSOES_LOG:
        try:
            print(f"  {pkg}=={md.version(pkg)}")
        except Exception:
            print(f"  {pkg}: nao instalado")


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


def _cabecalho(primeira_linha: str):
    return [
        f"# {primeira_linha}",
        f"# Python: {sys.version.split()[0]}",
        f"# CUDA (torch): {_cuda_torch()}",
        f"# Data: {datetime.date.today().isoformat()}",
        "",
    ]


def _pip_freeze():
    r = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                       capture_output=True, text=True)
    return [l.strip() for l in r.stdout.splitlines() if l.strip()]


def gerar_freeze(repo_dir: str) -> str:
    """pip freeze COMPLETO da sessão (auditoria) — NÃO é usado para instalar."""
    cab = _cabecalho("requirements-kaggle.freeze.txt - pip freeze COMPLETO da sessao (auditoria).")
    cab.insert(1, "# NAO e usado para instalar (re-resolver o freeze e irresolvivel). Instale pelo lock.")
    return "\n".join(cab + sorted(_pip_freeze(), key=str.lower)) + "\n"


def gerar_lock(repo_dir: str):
    """Lock = a lista de requirements-kaggle.txt com UMA única trava: vllm==<versão instalada>.
    O resto é resolvido pelo pip sobre a imagem do Kaggle. Travar todas as versões é
    irresolvível porque a imagem pré-instalada traz conflitos "no papel" que o pip só
    aceita numa instalação incremental (D-35). Retorna (texto, faltando).
    """
    import importlib.metadata as md
    linhas, faltando = [], []
    for nome in _pacotes_base(repo_dir):
        if nome == PACOTE_TRAVADO:
            try:
                linhas.append(f"{nome}=={md.version(nome)}")
            except md.PackageNotFoundError:
                faltando.append(nome)
                linhas.append(nome)
        else:
            linhas.append(nome)  # sem travar — pip resolve
    cab = _cabecalho(
        f"requirements-kaggle.lock.txt - apenas {PACOTE_TRAVADO} travado (motor de inferencia) (D-35).")
    cab.insert(1, "# O resto e resolvido pelo pip sobre a imagem do Kaggle. "
                  "Freeze: requirements-kaggle.freeze.txt.")
    return "\n".join(cab + linhas) + "\n", faltando


def _salvar_freeze_run(repo_dir: str, destino_dir: str):
    """Grava o pip freeze real da sessão junto dos resultados (auditoria por run)."""
    try:
        os.makedirs(destino_dir, exist_ok=True)
        with open(os.path.join(destino_dir, "ambiente_pip_freeze.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(_pip_freeze()) + "\n")
    except Exception as e:
        print(f"AVISO: não consegui gravar o pip freeze do run: {e}")


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

    # 2. Grava o freeze COMPLETO (auditoria) e o lock SÓ com os pacotes do projeto.
    freeze_path = args.freeze_out or os.path.join(args.repo_dir, "requirements-kaggle.freeze.txt")
    lock_path = args.lock_out or os.path.join(args.repo_dir, "requirements-kaggle.lock.txt")
    with open(freeze_path, "w", encoding="utf-8") as f:
        f.write(gerar_freeze(args.repo_dir))
    lock_txt, faltando = gerar_lock(args.repo_dir)
    if faltando:
        print(f"AVISO: pacotes do projeto não instalados (fora do lock): {faltando}")
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(lock_txt)
    n_lock = sum(1 for l in lock_txt.splitlines() if l and not l.startswith("#"))
    print(f"Freeze: {freeze_path}  |  Lock ({n_lock} pacotes): {lock_path}")

    # 3. Push verificado (salvo em --no_push)
    if args.no_push:
        print("(--no_push: sem git)")
        return
    rels = [os.path.relpath(p, args.repo_dir).replace("\\", "/") for p in (freeze_path, lock_path)]
    git(["add"] + rels, args.repo_dir)
    if git(["status", "--porcelain"] + rels, args.repo_dir).stdout.strip():
        git(["commit", "-m", "auto: requirements-kaggle lock+freeze (congelar_ambiente)"], args.repo_dir)
        if sincronizar_e_push(args.repo_dir):
            print("Push do lock+freeze OK.")
        else:
            raise RuntimeError("Push do lock+freeze NÃO confirmado após as tentativas.")
    else:
        print("Lock/freeze inalterados — nada para commitar.")


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

    # Registra o pip freeze da sessão junto das saídas da indução (auditoria).
    slug = args.slug or slug_modelo(args.modelo)
    _salvar_freeze_run(
        args.repo_dir,
        os.path.join(args.repo_dir, "resultados", "c2_inducao", args.c2_base, slug))

    if args.no_push:
        print("Indução concluída (--no_push: sem git).")
        return
    alvo = [f"configs/c2/{args.c2_base}/", f"resultados/c2_inducao/{args.c2_base}/"]
    git(["add"] + alvo, args.repo_dir)
    if git(["status", "--porcelain"] + alvo, args.repo_dir).stdout.strip():
        git(["commit", "-m", "auto: inducao C2 v_full (29 ontologias)"], args.repo_dir)
        if sincronizar_e_push(args.repo_dir):
            print("Push indução OK.")
        else:
            print("AVISO: push da indução NÃO confirmado após as tentativas.")
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

    # Registra o pip freeze da sessão junto das saídas do experimento (auditoria).
    if exps:
        _salvar_freeze_run(args.repo_dir, os.path.dirname(exps[0]["saida"]))

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
    p.add_argument("--freeze_out", default=None,
                   help="Caminho do freeze a gravar (padrão: <repo>/requirements-kaggle.freeze.txt).")
    args = p.parse_args()

    # Logo após a instalação (feita antes no lançador): registra as versões efetivas no log.
    if not args.mock:
        _imprimir_versoes()

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
