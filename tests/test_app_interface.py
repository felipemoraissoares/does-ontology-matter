"""Testes de interface (streamlit.testing.v1.AppTest): cada página carrega sem exceção.

`inicio` usa st.page_link (exige o contexto de st.navigation) → testada pela entrada Inicio.py.
As demais páginas não usam navegação e rodam isoladas.
"""
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(REPO, "src"))

from streamlit.testing.v1 import AppTest  # noqa: E402

STANDALONE = ["fases", "explorador", "ontologias", "grafo_agregado", "resultados", "em_breve"]


def _run(caminho, query=None):
    at = AppTest.from_file(caminho, default_timeout=180)
    for k, v in (query or {}).items():
        at.query_params[k] = v
    at.run()
    return at


def test_entrada_inicio():
    at = _run(os.path.join(APP, "Inicio.py"))
    assert not at.exception, at.exception


@pytest.mark.parametrize("pagina", STANDALONE)
def test_pagina_piloto(pagina):
    at = _run(os.path.join(APP, "paginas", f"{pagina}.py"),
              {"rodada": "piloto-v3", "dataset": "tekgen", "onto": "1_movie",
               "frase": "ont_1_movie_test_37"})
    assert not at.exception, f"{pagina}: {at.exception}"


@pytest.mark.parametrize("pagina", ["explorador", "resultados", "grafo_agregado", "ontologias"])
def test_pagina_full_parcial(pagina):
    import dados.descoberta as d
    if d.rodada("full/qwen2.5-7b") is None:
        pytest.skip("rodada full ainda não presente")
    at = _run(os.path.join(APP, "paginas", f"{pagina}.py"),
              {"rodada": "full/qwen2.5-7b", "dataset": "tekgen", "onto": "1_movie"})
    assert not at.exception, f"{pagina} (full): {at.exception}"
