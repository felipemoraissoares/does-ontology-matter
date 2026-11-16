"""D-35: lock só com pacotes do projeto + freeze completo (auditoria)."""
import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import rodar_kaggle as rk

LOCK = os.path.join(REPO, "requirements-kaggle.lock.txt")
FREEZE = os.path.join(REPO, "requirements-kaggle.freeze.txt")
REQ = os.path.join(REPO, "requirements-kaggle.txt")


def _pins(path):
    """{pacote_canonico: versao} das linhas pkg==ver de um arquivo."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l or l.startswith("#") or "==" not in l:
                continue
            nome, ver = l.split("==", 1)
            out[nome.strip().lower().replace("_", "-")] = ver.strip()
    return out


class TestGerarLock:
    def test_retorna_texto_e_faltando(self):
        texto, faltando = rk.gerar_lock(REPO)
        assert texto.splitlines()[0].startswith("# requirements-kaggle.lock.txt")
        assert any(l.startswith("# Python:") for l in texto.splitlines())
        assert isinstance(faltando, list)

    def test_pacotes_base_inclui_torch_transformers(self):
        nomes = rk._pacotes_base(REPO)
        assert {"vllm", "torch", "transformers", "numpy"} <= set(nomes)
        assert all("==" not in n for n in nomes)


class TestModoCongelar:
    def test_grava_lock_e_freeze_sem_push(self, tmp_path):
        lock_out = str(tmp_path / "req.lock.txt")
        freeze_out = str(tmp_path / "req.freeze.txt")
        args = argparse.Namespace(mock=True, no_push=True, repo_dir=REPO,
                                  lock_out=lock_out, freeze_out=freeze_out)
        rk.modo_congelar(args)
        assert os.path.exists(lock_out) and os.path.exists(freeze_out)
        assert "# Python:" in open(lock_out, encoding="utf-8").read()
        assert "# Python:" in open(freeze_out, encoding="utf-8").read()


class TestLockCoerenteComFreeze:
    """Ponto 3: todo pacote do lock está no freeze com a MESMA versão."""

    def test_lock_subset_do_freeze(self):
        lock, freeze = _pins(LOCK), _pins(FREEZE)
        assert lock, "lock vazio"
        for pkg, ver in lock.items():
            assert pkg in freeze, f"{pkg} do lock ausente no freeze"
            assert freeze[pkg] == ver, f"{pkg}: lock {ver} != freeze {freeze[pkg]}"

    def test_lock_so_tem_pacotes_do_projeto(self):
        projeto = {n.lower().replace("_", "-") for n in rk._pacotes_base(REPO)}
        lock = set(_pins(LOCK))
        assert lock == projeto, f"lock != projeto: {lock ^ projeto}"
