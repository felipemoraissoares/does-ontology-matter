"""Linha do tempo das fases, com blocos curtos e chips de decisão D-xx."""
import html
import os

import streamlit as st
import yaml

from dados import decisoes
from componentes import chips

_YAML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "conteudo", "fases.yaml")


@st.cache_data(show_spinner=False)
def _fases():
    with open(_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


st.title("Fases do projeto")
st.caption("Cada fase: o que é, o que foi feito, problemas e decisões. Clique num D-xx para o texto.")

for fase in _fases():
    with st.container():
        chips.mostrar(
            f"<div class='fase'><h3>{fase['id']} · {html.escape(fase['titulo'])}</h3></div>")
        c1, c2 = st.columns(2)
        with c1:
            chips.mostrar(chips.card("O que é", html.escape(fase["o_que_e"])))
            chips.mostrar(chips.card("Problemas encontrados", html.escape(fase["problemas"])))
        with c2:
            chips.mostrar(chips.card("O que foi feito", html.escape(fase["o_que_foi_feito"])))
            fontes = ", ".join(fase.get("fontes", []))
            chips.mostrar(chips.card("Fontes", f"<small class='fonte'>{html.escape(fontes)}</small>"))

        dids = fase.get("decisoes", [])
        if dids:
            cols = st.columns(min(len(dids), 8))
            for i, did in enumerate(dids):
                with cols[i % len(cols)]:
                    with st.popover(did):
                        d = decisoes.decisao(did)
                        st.markdown(f"**{d['id']} · {d['titulo']}**")
                        st.markdown(d["corpo"][:2500] or "_(sem corpo)_")
        st.divider()
