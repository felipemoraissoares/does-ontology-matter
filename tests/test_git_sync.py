"""sincronizar_e_push: push com rebase quando o remoto avançou (outra máquina empurrou)."""
import os
import shutil
import subprocess
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

GITEXE = shutil.which("git") or r"D:\Programas\Git\cmd\git.exe"
if os.path.exists(GITEXE):
    # garante que o 'git' do sincronizar_e_push (via PATH) resolva
    os.environ["PATH"] = os.path.dirname(GITEXE) + os.pathsep + os.environ.get("PATH", "")

from t2kg.git_sync import sincronizar_e_push  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(GITEXE), reason="git ausente")


def g(cwd, *args):
    r = subprocess.run([GITEXE, *args], cwd=str(cwd), capture_output=True, text=True)
    assert r.returncode == 0, f"git {args}: {r.stderr}"
    return r


def _config(d):
    g(d, "config", "user.email", "t@t")
    g(d, "config", "user.name", "teste")
    g(d, "config", "commit.gpgsign", "false")


def test_push_com_rebase_apos_remoto_avancar(tmp_path):
    remote = tmp_path / "remote.git"
    g(tmp_path, "init", "--bare", str(remote))

    # Semeia o remoto com um commit inicial
    seed = tmp_path / "seed"
    g(tmp_path, "clone", str(remote), str(seed))
    _config(seed)
    (seed / "a.txt").write_text("a", encoding="utf-8")
    g(seed, "add", "-A")
    g(seed, "commit", "-m", "seed")
    br = g(seed, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    g(seed, "push", "origin", br)

    # Dois clones rasos
    cloneB = tmp_path / "B"
    cloneC = tmp_path / "C"
    g(tmp_path, "clone", "--depth=1", str(remote), str(cloneB))
    g(tmp_path, "clone", "--depth=1", str(remote), str(cloneC))
    _config(cloneB)
    _config(cloneC)

    # B avança o remoto
    (cloneB / "b.txt").write_text("b", encoding="utf-8")
    g(cloneB, "add", "-A")
    g(cloneB, "commit", "-m", "from B")
    g(cloneB, "push", "origin", br)

    # C faz um commit local (divergiu do remoto)
    (cloneC / "c.txt").write_text("c", encoding="utf-8")
    g(cloneC, "add", "-A")
    g(cloneC, "commit", "-m", "from C")

    # push ingênuo de C seria rejeitado; sincronizar_e_push deve rebase+push
    ok = sincronizar_e_push(str(cloneC), branch=br)
    assert ok is True

    # o remoto tem os dois commits (B e C)
    g(cloneC, "fetch", "origin", br)
    log = g(cloneC, "log", "--oneline", f"origin/{br}").stdout
    assert "from B" in log and "from C" in log


def test_push_simples_quando_remoto_nao_avancou(tmp_path):
    remote = tmp_path / "remote.git"
    g(tmp_path, "init", "--bare", str(remote))
    clone = tmp_path / "A"
    g(tmp_path, "clone", str(remote), str(clone))
    _config(clone)
    (clone / "a.txt").write_text("a", encoding="utf-8")
    g(clone, "add", "-A")
    g(clone, "commit", "-m", "um")
    br = g(clone, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    g(clone, "push", "origin", br)
    (clone / "b.txt").write_text("b", encoding="utf-8")
    g(clone, "add", "-A")
    g(clone, "commit", "-m", "dois")
    assert sincronizar_e_push(str(clone), branch=br) is True
