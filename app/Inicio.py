"""Entrypoint do app de exploração.

Rodar:  python -m streamlit run app/Inicio.py
"""
import os
import sys

import streamlit as st

_AQUI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_AQUI, ".."))
for _p in (_AQUI, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

st.set_page_config(page_title="Ontologia & LLM — explorador",
                   page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner=False)
def _preparar_nltk():
    try:
        import nltk
        for pkg in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)
    except Exception:
        pass
    return True


_preparar_nltk()

from componentes.estilo import injetar  # noqa: E402

PAGINAS = {
    "Visão geral": [
        st.Page("paginas/inicio.py", title="Início", icon=":material/home:", default=True),
        st.Page("paginas/fases.py", title="Fases do projeto", icon=":material/timeline:",
                url_path="fases"),
    ],
    "Explorar": [
        st.Page("paginas/explorador.py", title="Explorador de frase",
                icon=":material/search:", url_path="explorador"),
        st.Page("paginas/ontologias.py", title="Ontologias",
                icon=":material/account_tree:", url_path="ontologias"),
        st.Page("paginas/grafo_agregado.py", title="Grafo agregado",
                icon=":material/hub:", url_path="grafo"),
    ],
    "Resultados": [
        st.Page("paginas/resultados.py", title="Resultados agregados",
                icon=":material/bar_chart:", url_path="resultados"),
    ],
    "Em breve": [
        st.Page("paginas/em_breve.py", title="Chat com o grafo",
                icon=":material/chat:", url_path="chat"),
    ],
}

nav = st.navigation(PAGINAS)
injetar()
nav.run()
