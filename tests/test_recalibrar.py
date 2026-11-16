"""D-37: exclusão de pares degenerados (embeddings idênticos) no breakpoint."""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import recalibrar_c2_29 as rc


class TestBreakpoint:
    def test_exclui_par_degenerado(self):
        pares = [(1.000, "league", "league"),      # duplicado (degenerado)
                 (0.939, "awards", "award"),        # quase sinônimo (breakpoint real)
                 (0.5, "x", "y")]
        assert rc._breakpoint(pares, excluir_degenerado=True) == (0.939, "awards", "award")

    def test_inclui_degenerado_quando_pedido(self):
        pares = [(1.000, "league", "league"), (0.939, "awards", "award")]
        assert rc._breakpoint(pares, excluir_degenerado=False) == (1.000, "league", "league")

    def test_sem_pares(self):
        assert rc._breakpoint([], excluir_degenerado=True) is None

    def test_todos_degenerados(self):
        pares = [(1.0, "a", "a"), (0.9995, "b", "b")]
        assert rc._breakpoint(pares, excluir_degenerado=True) is None
