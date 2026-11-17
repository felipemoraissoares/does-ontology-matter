"""Seletores de filtro reutilizáveis, sincronizados com a URL (query_params)."""
from typing import List, Optional, Tuple

import streamlit as st

from dados import descoberta
from dados.base import DS_ROTULO

from . import estado


def seletor_rodada(col=None) -> Optional[str]:
    rods = descoberta.rodadas()
    if not rods:
        return None
    ids = [r["id"] for r in rods]
    rot = {r["id"]: r["rotulo"] for r in rods}
    atual = estado.obter("rodada", ids[0])
    if atual not in ids:
        atual = ids[0]
    alvo = col or st
    esc = alvo.selectbox("Rodada", ids, index=estado.indice(ids, atual),
                         format_func=lambda i: rot.get(i, i), key="w_rodada")
    estado.definir("rodada", esc)
    return esc


def filtros_rodada_ds_onto(cols=None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Renderiza rodada + dataset + ontologia (em colunas) e devolve (rid, ds, onto)."""
    c = cols or st.columns(3)
    rid = seletor_rodada(c[0])
    if not rid:
        return None, None, None
    pares = descoberta.ontologias_da_rodada(rid)
    if not pares:
        c[1].info("Rodada sem experimentos.")
        return rid, None, None
    datasets = sorted({ds for ds, _ in pares})
    ds_atual = estado.obter("dataset", datasets[0])
    if ds_atual not in datasets:
        ds_atual = datasets[0]
    ds = c[1].selectbox("Base", datasets, index=estado.indice(datasets, ds_atual),
                        format_func=lambda d: DS_ROTULO.get(d, d), key="w_dataset")
    estado.definir("dataset", ds)

    ontos = [o for d, o in pares if d == ds]
    onto_atual = estado.obter("onto", ontos[0] if ontos else None)
    if onto_atual not in ontos:
        onto_atual = ontos[0] if ontos else None
    onto = c[2].selectbox("Ontologia", ontos, index=estado.indice(ontos, onto_atual),
                         key="w_onto")
    estado.definir("onto", onto)
    return rid, ds, onto
