"""D-35: modo congelar_ambiente — geração do lock (sem GPU, sem push)."""
import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import rodar_kaggle as rk


class TestGerarLock:
    def test_cabecalho_e_pacotes(self):
        lock = rk.gerar_lock(REPO)
        linhas = lock.splitlines()
        assert linhas[0].startswith("# requirements-kaggle.lock.txt")
        assert any(l.startswith("# Python:") for l in linhas)
        assert any(l.startswith("# CUDA (torch):") for l in linhas)
        assert any(l.startswith("# Data:") for l in linhas)
        # Pelo menos um pacote instalado localmente aparece fixado (ex.: numpy/nltk/scipy).
        pkgs = [l for l in linhas if l and not l.startswith("#")]
        assert any("==" in p for p in pkgs)

    def test_pacotes_base_le_lista(self):
        nomes = rk._pacotes_base(REPO)
        assert "vllm" in nomes and "numpy" in nomes
        # Lista sem versões: nenhum token com '=='
        assert all("==" not in n for n in nomes)


class TestModoCongelar:
    def test_grava_lock_sem_push(self, tmp_path):
        lock_out = str(tmp_path / "req.lock.txt")
        args = argparse.Namespace(
            mock=True, no_push=True, repo_dir=REPO, lock_out=lock_out)
        rk.modo_congelar(args)
        assert os.path.exists(lock_out)
        conteudo = open(lock_out, encoding="utf-8").read()
        assert "# Python:" in conteudo and "==" in conteudo
