"""Grafo agregado: junta as triplas de todas as frases de uma rodada+ontologia+condição."""
import streamlit as st
import streamlit.components.v1 as components

from dados import descoberta, grafo
from dados.base import COND_ROTULO, CONDICOES
from componentes import chips, grafo_vis
from componentes.filtros import filtros_rodada_ds_onto

st.title("Grafo agregado")

rid, ds, onto = filtros_rodada_ds_onto()
if not (rid and ds and onto):
    st.info("Nenhum experimento disponível.")
    st.stop()

conds = [c for c in CONDICOES if c in descoberta.condicoes_da_ontologia(rid, ds, onto)]
if not conds:
    st.warning("Ontologia sem experimentos nesta rodada.")
    st.stop()

o1, o2, o3, o4 = st.columns([2, 2, 1, 1])
cond_a = o1.selectbox("Condição", conds, format_func=lambda c: COND_ROTULO.get(c, c), key="w_gcA")
comparar = o2.checkbox("Comparar com outra condição")
cond_b = None
if comparar:
    cond_b = o2.selectbox("Condição B", [c for c in conds if c != cond_a],
                          format_func=lambda c: COND_ROTULO.get(c, c), key="w_gcB")
so_acertos = o3.checkbox("Só acertos")
incluir_gab = o4.checkbox("Gabarito por cima")
limite = st.slider("Limite de nós", 20, 400, 150, 10)


@st.cache_data(show_spinner=False)
def _grafo(rid, ds, onto, cond, so_acertos, incluir_gab, limite):
    g = grafo.grafo_agregado(rid, ds, onto, cond, so_acertos=so_acertos,
                             incluir_gabarito=incluir_gab, limite_nos=limite)
    ghtml = grafo_vis.construir_html(g["nos"], g["arestas"], altura=520)
    return g, ghtml


def _painel(g):
    top = "<br>".join(f"{r} · {n}" for r, n in g["top_relacoes"])
    chips.mostrar(chips.card(
        "Resumo",
        f"<b>Relações distintas:</b> {g['n_relacoes_distintas']}<br>"
        f"<b>Redundância:</b> {g['redundancia']}<br>"
        f"<b>Arestas:</b> {len(g['arestas'])}<br><br>"
        f"<b>Top relações</b><br>{top}"))


def _mostrar(cond, largura_grafo=3):
    g, ghtml = _grafo(rid, ds, onto, cond, so_acertos, incluir_gab, limite)
    cg, cp = st.columns([largura_grafo, 1])
    with cg:
        components.html(ghtml, height=540)
    with cp:
        _painel(g)


chips.mostrar(chips.legenda())
if cond_b:
    ca, cb = st.columns(2)
    with ca:
        st.markdown(f"**{COND_ROTULO.get(cond_a, cond_a)}**")
        _mostrar(cond_a, largura_grafo=2)
    with cb:
        st.markdown(f"**{COND_ROTULO.get(cond_b, cond_b)}**")
        _mostrar(cond_b, largura_grafo=2)
else:
    _mostrar(cond_a)
