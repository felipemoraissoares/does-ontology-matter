"""Chips HTML de triplas, legenda e cartões de métrica."""
import html

import streamlit as st

from .estilo import CORES, ROTULO_CLASSE


def tripla_html(sub: str, rel: str, obj: str, classe: str) -> str:
    txt = html.escape(f"{rel}({sub}, {obj})")
    titulo = ROTULO_CLASSE.get(classe, classe)
    return f"<span class='chip chip-{classe}' title='{titulo}'>{txt}</span>"


def triplas_html(classificadas, faltas=None) -> str:
    partes = [tripla_html(t["tripla"][0], t["tripla"][1], t["tripla"][2], t["classe"])
              for t in classificadas]
    for s, r, o in (faltas or []):
        partes.append(tripla_html(s, r, o, "falta"))
    return "".join(partes) or "<em>nenhuma tripla</em>"


def legenda() -> str:
    itens = []
    for cls in ("acerto", "fora_do_gabarito", "erro", "erro_formato", "falta"):
        cor = CORES[cls]
        itens.append(f"<span><span class='pilula' style='background:{cor}'></span>"
                     f"{ROTULO_CLASSE[cls]}</span>")
    return "<div class='legenda'>" + "".join(itens) + "</div>"


def cartao_metrica(valor, rotulo: str) -> str:
    return (f"<div class='card metrica'><div class='valor'>{valor}</div>"
            f"<div class='rotulo'>{html.escape(rotulo)}</div></div>")


def card(titulo: str, corpo_html: str) -> str:
    cab = f"<h4>{html.escape(titulo)}</h4>" if titulo else ""
    return f"<div class='card'>{cab}{corpo_html}</div>"


def mostrar(html_str: str) -> None:
    st.markdown(html_str, unsafe_allow_html=True)
