"""Injeção única do CSS e constantes visuais compartilhadas."""
import os

import streamlit as st

CORES = {
    "acerto": "#2E7D4F",
    "fora_do_gabarito": "#8a94a0",
    "erro": "#B3261E",
    "erro_formato": "#C99A00",
    "falta": "#8a94a0",
    "primaria": "#2E5A8A",
}
ROTULO_CLASSE = {
    "acerto": "acerto",
    "fora_do_gabarito": "fora do gabarito",
    "erro": "erro",
    "erro_formato": "erro de formato",
    "falta": "falta",
}
_CSS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "estilo.css")


@st.cache_resource
def _ler_css() -> str:
    with open(_CSS, encoding="utf-8") as f:
        return f.read()


def injetar() -> None:
    st.markdown(f"<style>{_ler_css()}</style>", unsafe_allow_html=True)
