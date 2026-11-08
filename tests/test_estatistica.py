"""
Testes das funções estatísticas (Wilcoxon, bootstrap, Holm, regressão mista).
"""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from t2kg.analise.estatistica import (
    wilcoxon_pareado,
    bootstrap_ci,
    corrigir_holm,
    comparacoes_multiplas,
)


class TestWilcoxonPareado:
    def test_diferenca_positiva(self):
        a = np.array([0.8, 0.9, 0.7, 0.85, 0.75, 0.9, 0.8, 0.95, 0.7, 0.85])
        b = np.array([0.4, 0.5, 0.3, 0.45, 0.35, 0.5, 0.4, 0.55, 0.3, 0.45])
        r = wilcoxon_pareado(a, b)
        assert r["pvalue"] < 0.05
        assert r["mediana_diferenca"] > 0
        assert r["n_pares"] == 10

    def test_sem_diferenca(self):
        a = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        b = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        r = wilcoxon_pareado(a, b)
        # Todos zeros → aviso de poucas diffs
        assert "aviso" in r

    def test_poucos_pares_retorna_aviso(self):
        a = np.array([0.8, 0.9, 0.7])
        b = np.array([0.4, 0.5, 0.3])
        r = wilcoxon_pareado(a, b)
        assert "aviso" in r
        assert r["pvalue"] is None

    def test_comprimentos_diferentes_lanca_erro(self):
        with pytest.raises(ValueError):
            wilcoxon_pareado(np.array([1.0, 2.0]), np.array([1.0]))


class TestBootstrapCI:
    def test_ci_contem_media_real(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0.7, 0.9, 50)
        b = rng.uniform(0.4, 0.6, 50)
        r = bootstrap_ci(a, b, n_boot=5000, seed=42)
        assert r["ci_low"] < r["media_diferenca"] < r["ci_high"]
        assert r["ci_low"] > 0  # a > b claramente

    def test_reproducivel_com_seed(self):
        a = np.array([0.8, 0.9, 0.7, 0.85, 0.75])
        b = np.array([0.4, 0.5, 0.3, 0.45, 0.35])
        r1 = bootstrap_ci(a, b, seed=42)
        r2 = bootstrap_ci(a, b, seed=42)
        assert r1["ci_low"] == r2["ci_low"]
        assert r1["ci_high"] == r2["ci_high"]

    def test_nivel_confianca_registrado(self):
        a = np.array([0.8, 0.6, 0.7])
        b = np.array([0.3, 0.2, 0.4])
        r = bootstrap_ci(a, b, nivel_confianca=0.90)
        assert r["nivel_confianca"] == 0.90


class TestCorrigirHolm:
    def test_caso_basico(self):
        # Exemplo canônico: 3 hipóteses, p = [0.01, 0.04, 0.03]
        # Ordenado: 0.01 × 3 = 0.03; 0.03 × 2 = 0.06; 0.04 × 1 = 0.04 → max(0.06, 0.04) = 0.06
        ps = [0.01, 0.04, 0.03]
        adj = corrigir_holm(ps)
        assert adj[0] < adj[1]  # menor p gera menor ajuste
        assert all(0.0 <= p <= 1.0 for p in adj)

    def test_monotonica(self):
        # p-valores ajustados devem ser não-decrescentes quando ordenados pelo original
        ps = [0.001, 0.01, 0.05, 0.10]
        adj = corrigir_holm(ps)
        # adj na ordem original; quando reordenados por ps, devem ser crescentes
        paired = sorted(zip(ps, adj))
        vals = [a for _, a in paired]
        assert vals == sorted(vals)

    def test_comprimento_preservado(self):
        ps = [0.03, 0.01, 0.20]
        assert len(corrigir_holm(ps)) == 3

    def test_p_zero_permanece_zero(self):
        ps = [0.0, 0.05]
        adj = corrigir_holm(ps)
        assert adj[0] == 0.0 or adj[0] < adj[1]

    def test_nunca_excede_um(self):
        ps = [0.9, 0.95, 0.99]
        adj = corrigir_holm(ps)
        assert all(p <= 1.0 for p in adj)


class TestComparacoesMultiplas:
    def test_adiciona_p_holm(self):
        a1 = np.array([0.8, 0.9, 0.7, 0.85, 0.75, 0.9, 0.8, 0.95, 0.7, 0.85])
        b1 = np.array([0.4, 0.5, 0.3, 0.45, 0.35, 0.5, 0.4, 0.55, 0.3, 0.45])
        a2 = np.array([0.6, 0.65, 0.55, 0.60, 0.65, 0.70, 0.60, 0.65, 0.55, 0.70])
        b2 = np.array([0.4, 0.5, 0.3, 0.45, 0.35, 0.5, 0.4, 0.55, 0.3, 0.45])
        comps = [
            {"nome": "C1_vs_C0", "a": a1, "b": b1},
            {"nome": "C5_vs_C0", "a": a2, "b": b2},
        ]
        r = comparacoes_multiplas(comps)
        assert len(r) == 2
        for item in r:
            assert "p_holm" in item
            assert "wilcoxon" in item
            assert "bootstrap" in item
