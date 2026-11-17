"""Push com sincronização (fetch + rebase) — resistente a um remoto que avançou.

O push por experimento falhava quando outra máquina empurrava no meio do lote (o remoto
avança e todo push seguinte é rejeitado). Aqui, antes de empurrar, sincronizamos com
`git fetch` + `git rebase origin/<branch>` e retentamos. Funciona em clone raso (--depth=1):
`fetch` traz os novos commits até a fronteira rasa (o commit base comum), onde o rebase acha
a base de mesclagem.
"""
import subprocess
import sys
from typing import Optional


def _git(repo_dir: str, args, **kw):
    return subprocess.run(["git", "-C", repo_dir] + args,
                          capture_output=True, text=True, **kw)


def branch_atual(repo_dir: str) -> str:
    r = _git(repo_dir, ["rev-parse", "--abbrev-ref", "HEAD"])
    return r.stdout.strip() or "main"


def sincronizar_e_push(repo_dir: str, branch: Optional[str] = None,
                       tentativas: int = 3, remoto: str = "origin") -> bool:
    """fetch + rebase em cima de <remoto>/<branch> e então push; retenta até `tentativas`
    vezes, com rebase entre as tentativas. Retorna True se o push foi aceito.

    Funciona em clone raso. Em conflito de rebase (não esperado para arquivos de resultado,
    que têm nomes distintos por experimento/modelo), aborta o rebase e conta como falha.
    """
    branch = branch or branch_atual(repo_dir)
    for i in range(1, tentativas + 1):
        _git(repo_dir, ["fetch", remoto, branch])
        reb = _git(repo_dir, ["rebase", f"{remoto}/{branch}"])
        if reb.returncode != 0:
            _git(repo_dir, ["rebase", "--abort"])
            print(f"    AVISO: rebase falhou (tentativa {i}/{tentativas}): "
                  f"{(reb.stderr or reb.stdout).strip()[:160]}", file=sys.stderr)
            continue
        push = _git(repo_dir, ["push", remoto, f"HEAD:{branch}"])
        if push.returncode == 0:
            return True
        print(f"    AVISO: push rejeitado (tentativa {i}/{tentativas}): "
              f"{push.stderr.strip()[:200]}", file=sys.stderr)
    return False
