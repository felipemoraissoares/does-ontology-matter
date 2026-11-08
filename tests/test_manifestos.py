"""
Verifica que todos os manifestos JSON em configs/pipeline/ têm caminhos
portáteis: sem barras invertidas e sem caminhos absolutos no campo 'saida'.
"""
import json
import os
import pathlib
import pytest

CONFIGS_DIR = pathlib.Path(__file__).parent.parent / "configs" / "pipeline"
MANIFESTO_GLOB = list(CONFIGS_DIR.glob("manifesto_*.json"))


def _carregar_manifestos():
    items = []
    for path in MANIFESTO_GLOB:
        with open(path, encoding="utf-8") as f:
            exps = json.load(f)
        for exp in exps:
            items.append((path.name, exp))
    return items


TODOS_EXPERIMENTOS = _carregar_manifestos()


@pytest.mark.parametrize("nome_arquivo,exp", TODOS_EXPERIMENTOS, ids=[
    f"{a}::{e['id']}" for a, e in TODOS_EXPERIMENTOS
])
def test_saida_sem_barra_invertida(nome_arquivo, exp):
    saida = exp["saida"]
    assert "\\" not in saida, (
        f"{nome_arquivo} :: {exp['id']}: 'saida' contém barra invertida: {saida!r}\n"
        "Regenere com scripts/gerar_manifesto.py (usa '/' em todos os SO)."
    )


@pytest.mark.parametrize("nome_arquivo,exp", TODOS_EXPERIMENTOS, ids=[
    f"{a}::{e['id']}" for a, e in TODOS_EXPERIMENTOS
])
def test_saida_nao_absoluta(nome_arquivo, exp):
    saida = exp["saida"]
    is_abs = (
        saida.startswith("/")          # Unix absoluto
        or (len(saida) >= 2 and saida[1] == ":")  # Windows C:\ ou D:/
    )
    assert not is_abs, (
        f"{nome_arquivo} :: {exp['id']}: 'saida' é caminho absoluto: {saida!r}\n"
        "Use caminhos relativos à raiz do repositório."
    )


def test_piloto_em_resultados_piloto():
    """Manifesto piloto deve gravar em resultados/piloto/, não em resultados/7b/."""
    piloto = CONFIGS_DIR / "manifesto_piloto_kaggle.json"
    with open(piloto, encoding="utf-8") as f:
        exps = json.load(f)
    for exp in exps:
        assert exp["saida"].startswith("resultados/piloto/"), (
            f"Piloto {exp['id']} aponta para '{exp['saida']}' — "
            "deve estar em resultados/piloto/ (exceção do .gitignore)."
        )


def test_bench_commit_presente():
    """Todos os experimentos dos três manifestos gerados devem ter bench_commit."""
    gerados = ["manifesto_7b_full.json", "manifesto_7b_sensibilidade.json",
               "manifesto_piloto_kaggle.json"]
    for nome in gerados:
        path = CONFIGS_DIR / nome
        with open(path, encoding="utf-8") as f:
            exps = json.load(f)
        for exp in exps:
            assert "bench_commit" in exp, (
                f"{nome} :: {exp['id']}: campo 'bench_commit' ausente."
            )
            assert len(exp["bench_commit"]) == 40, (
                f"{nome} :: {exp['id']}: 'bench_commit' deve ser SHA-1 de 40 chars."
            )
