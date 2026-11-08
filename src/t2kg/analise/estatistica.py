"""
Funções de análise estatística para comparação entre condições experimentais.

Análises pré-registradas (D-09):
  - Wilcoxon signed-rank pareado por frase
  - Bootstrap CI com seed fixa
  - Correção de Holm para comparações múltiplas
  - Regressão linear mista com SimE e dTok como covariáveis (D-08, D-09)

Dependências: scipy, numpy. statsmodels opcional (para regressão mista).
"""
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    import statsmodels.formula.api as smf  # noqa: F401
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False


# ---------------------------------------------------------------------------
# Carregamento de métricas por frase (output do avaliador)
# ---------------------------------------------------------------------------

def carregar_metricas_por_frase(caminho_jsonl: str) -> Dict[str, Dict]:
    """Lê o JSONL de saída do avaliador e retorna dict sent_id → métricas."""
    resultado = {}
    with open(caminho_jsonl, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            d = json.loads(linha)
            if d.get("type") in ("all_test_cases", "selected", "global"):
                continue
            if "id" in d:
                resultado[d["id"]] = d
    return resultado


def pares_por_frase(
    metricas_a: Dict[str, Dict],
    metricas_b: Dict[str, Dict],
    metrica: str = "f1",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna vetores (a, b) com valores da métrica para frases em comum entre A e B.
    Frases ausentes em qualquer um dos dois são ignoradas.
    """
    ids_comuns = sorted(set(metricas_a) & set(metricas_b))
    a = np.array([float(metricas_a[i][metrica]) for i in ids_comuns])
    b = np.array([float(metricas_b[i][metrica]) for i in ids_comuns])
    return a, b


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank pareado
# ---------------------------------------------------------------------------

def wilcoxon_pareado(
    a: np.ndarray,
    b: np.ndarray,
    alternativa: str = "two-sided",
) -> Dict:
    """
    Teste de Wilcoxon signed-rank entre distribuições emparelhadas a e b.

    Retorna dict com statistic, pvalue, n_pares, mediana_diferenca.
    """
    if not SCIPY_OK:
        raise ImportError("scipy necessário: pip install scipy")
    if len(a) != len(b):
        raise ValueError("Vetores devem ter o mesmo comprimento.")
    diffs = a - b
    # Remove zeros (empates)
    diffs_nonzero = diffs[diffs != 0]
    if len(diffs_nonzero) < 6:
        return {
            "statistic": None, "pvalue": None,
            "n_pares": len(a), "n_diffs_nonzero": len(diffs_nonzero),
            "mediana_diferenca": float(np.median(diffs)),
            "aviso": "Poucas diferenças não-zero para o teste (< 6).",
        }
    stat, pval = stats.wilcoxon(diffs_nonzero, alternative=alternativa)
    return {
        "statistic": float(stat),
        "pvalue": float(pval),
        "n_pares": len(a),
        "n_diffs_nonzero": len(diffs_nonzero),
        "mediana_diferenca": float(np.median(diffs)),
    }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
    nivel_confianca: float = 0.95,
) -> Dict:
    """
    Intervalo de confiança bootstrap para a diferença de médias (a - b).

    seed fixa garante reprodutibilidade (pré-registrado em D-09).
    """
    rng = np.random.default_rng(seed)
    diffs = a - b
    n = len(diffs)
    boot_means = np.array([
        rng.choice(diffs, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = 1 - nivel_confianca
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return {
        "media_diferenca": float(diffs.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "nivel_confianca": nivel_confianca,
        "n_boot": n_boot,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Correção de Holm para comparações múltiplas
# ---------------------------------------------------------------------------

def corrigir_holm(pvalues: List[float]) -> List[float]:
    """
    Correção de Holm-Bonferroni para lista de p-valores.
    Retorna lista de p-valores ajustados na mesma ordem de entrada.
    """
    n = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    p_ajustados = [0.0] * n
    p_max_acumulado = 0.0

    for rank, (i_orig, p) in enumerate(indexed):
        p_ajust = p * (n - rank)
        p_ajust = min(p_ajust, 1.0)
        p_ajust = max(p_ajust, p_max_acumulado)
        p_max_acumulado = p_ajust
        p_ajustados[i_orig] = p_ajust

    return p_ajustados


def comparacoes_multiplas(
    comparacoes: List[Dict],
    metrica: str = "f1",
    seed: int = 42,
) -> List[Dict]:
    """
    Recebe lista de dicts com chaves 'nome', 'a' (ndarray), 'b' (ndarray).
    Aplica Wilcoxon + bootstrap em cada par, depois Holm sobre todos os p-valores.

    Retorna lista de dicts com resultados e p_holm.
    """
    resultados = []
    pvalores = []

    for comp in comparacoes:
        a, b = comp["a"], comp["b"]
        w = wilcoxon_pareado(a, b)
        bc = bootstrap_ci(a, b, seed=seed)
        p = w.get("pvalue")
        pvalores.append(p if p is not None else 1.0)
        resultados.append({"nome": comp["nome"], "wilcoxon": w, "bootstrap": bc})

    p_holm = corrigir_holm(pvalores)
    for r, ph in zip(resultados, p_holm):
        r["p_holm"] = ph

    return resultados


# ---------------------------------------------------------------------------
# Regressão linear mista (D-09): SimE + dTok como covariáveis
# ---------------------------------------------------------------------------

def regressao_mista_sim_e(
    dados: List[Dict],
    metrica: str = "f1",
) -> Optional[Dict]:
    """
    Ajusta modelo linear misto:
      metrica ~ sim_e + d_tok + (1 | onto_id)

    dados: lista de dicts com chaves:
      onto_id, frase_id, metrica_valor, sim_e, d_tok

    Retorna dict com coeficientes, p-valores, AIC ou None se statsmodels ausente.
    """
    if not STATSMODELS_OK:
        return {"erro": "statsmodels não instalado: pip install statsmodels"}

    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame(dados)
    df = df.rename(columns={"metrica_valor": metrica})

    formula = f"{metrica} ~ sim_e + d_tok"
    try:
        modelo = smf.mixedlm(formula, df, groups=df["onto_id"]).fit(reml=True, disp=False)
        return {
            "coeficientes": modelo.params.to_dict(),
            "pvalores": modelo.pvalues.to_dict(),
            "aic": float(modelo.aic),
            "bic": float(modelo.bic),
            "n_obs": int(modelo.nobs),
        }
    except Exception as e:
        return {"erro": str(e)}


# ---------------------------------------------------------------------------
# Demonstração com dados existentes
# ---------------------------------------------------------------------------

def demo_com_dados_vicuna(bench_dir: str, resultados_dir: str):
    """
    Demonstra as funções estatísticas usando dados de reprodução Vicuna (Fase 0).
    Requer que test_tekgen_vicuna_global_bate tenha passado (resultados em resultados/tests/).
    """
    print("=== Demo estatística — Fase 0 Vicuna TekGen ===")
    print("(Comparação 1_movie vs 2_music — ilustrativo, não ontologias pareadas)")

    arq1 = os.path.join(resultados_dir, "tv_1_movie.jsonl")
    arq2 = os.path.join(resultados_dir, "tv_2_music.jsonl")

    if not os.path.exists(arq1) or not os.path.exists(arq2):
        print("  Arquivos não encontrados. Execute os testes de reprodução primeiro.")
        return

    m1 = carregar_metricas_por_frase(arq1)
    m2 = carregar_metricas_por_frase(arq2)
    a, b = pares_por_frase(m1, m2, "f1")

    if len(a) == 0:
        print("  Nenhum par encontrado.")
        return

    print(f"  Pares em comum: {len(a)}")
    print(f"  Média F1 1_movie: {a.mean():.3f}  2_music: {b.mean():.3f}")

    w = wilcoxon_pareado(a, b)
    bc = bootstrap_ci(a, b)
    print(f"  Wilcoxon: stat={w.get('statistic')}, p={w.get('pvalue'):.4f}" if w.get("pvalue") else f"  Wilcoxon: {w.get('aviso')}")
    print(f"  Bootstrap CI 95%: [{bc['ci_low']:.3f}, {bc['ci_high']:.3f}]  diff_media={bc['media_diferenca']:.3f}")
