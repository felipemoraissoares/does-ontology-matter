"""D-35: lock trava só o vLLM; freeze completo é só auditoria."""
import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import rodar_kaggle as rk

LOCK = os.path.join(REPO, "requirements-kaggle.lock.txt")
FREEZE = os.path.join(REPO, "requirements-kaggle.freeze.txt")
REQ = os.path.join(REPO, "requirements-kaggle.txt")


def _canon(n):
    return n.strip().lower().replace("_", "-")


def _nomes(path):
    """Nomes de pacotes (com ou sem versão)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l or l.startswith("#"):
                continue
            out.append(_canon(l.split("==")[0]))
    return out


def _pins(path):
    """Só os pacotes travados {nome: versao}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l or l.startswith("#") or "==" not in l:
                continue
            n, v = l.split("==", 1)
            out[_canon(n)] = v.strip()
    return out


class TestGerarLock:
    def test_lista_completa_e_trava_so_vllm(self):
        texto, faltando = rk.gerar_lock(REPO)
        nomes = [_canon(l.split("==")[0]) for l in texto.splitlines()
                 if l and not l.startswith("#")]
        assert {"vllm", "torch", "numpy"} <= set(nomes)
        pins = [l for l in texto.splitlines() if "==" in l and not l.startswith("#")]
        # No máximo o vllm é travado; nenhum outro pacote carrega ==.
        assert all(_canon(p.split("==")[0]) == "vllm" for p in pins)

    def test_pacotes_base_inclui_torch_transformers(self):
        nomes = rk._pacotes_base(REPO)
        assert {"vllm", "torch", "transformers", "numpy"} <= set(nomes)


class TestModoCongelar:
    def test_grava_lock_e_freeze_sem_push(self, tmp_path):
        lock_out = str(tmp_path / "req.lock.txt")
        freeze_out = str(tmp_path / "req.freeze.txt")
        args = argparse.Namespace(mock=True, no_push=True, repo_dir=REPO,
                                  lock_out=lock_out, freeze_out=freeze_out)
        rk.modo_congelar(args)
        assert os.path.exists(lock_out) and os.path.exists(freeze_out)


class TestLockCommitado:
    def test_so_vllm_travado(self):
        pins = _pins(LOCK)
        assert set(pins) == {"vllm"}, f"lock deveria travar só vllm, tem: {set(pins)}"
        assert pins["vllm"], "versão do vllm vazia"

    def test_lista_igual_requirements(self):
        assert set(_nomes(LOCK)) == set(_nomes(REQ))

    def test_vllm_bate_com_freeze(self):
        assert _pins(FREEZE).get("vllm") == _pins(LOCK)["vllm"]
