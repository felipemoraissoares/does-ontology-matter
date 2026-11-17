"""Explorador de frase: para uma frase, o que foi ao modelo e o que voltou, por condição."""
import html
import random

import streamlit as st
import streamlit.components.v1 as components

from dados import descoberta, frases, grafo, notas, ontologias
from dados.base import COND_ROTULO, COND_CURTO, CONDICOES
from componentes import chips, estado, grafo_vis
from componentes.filtros import filtros_rodada_ds_onto

st.title("Explorador de frase")

rid, ds, onto = filtros_rodada_ds_onto()
if not (rid and ds and onto):
    st.info("Nenhum experimento disponível para os filtros atuais.")
    st.stop()

conds = [c for c in CONDICOES if c in descoberta.condicoes_da_ontologia(rid, ds, onto)]
if not conds:
    st.warning("Esta ontologia não tem experimentos nesta rodada.")
    st.stop()

frs = frases.listar_frases(rid, ds, onto, conds[0])
ids = [f["id"] for f in frs]
texto = {f["id"]: f["sent"] for f in frs}

st.caption(f"Modelo: **{descoberta.rodada(rid)['modelo']}**  ·  {len(ids)} frases  ·  "
           f"{len(conds)} condições presentes")

# --- Seleção de frase: busca + navegação ---
cbusca, cnav = st.columns([3, 2])
termo = cbusca.text_input("Buscar frase (texto ou id)", key="w_busca")
filtrados = ([i for i in ids if termo.lower() in (i + " " + texto[i]).lower()]
             if termo else ids)
if not filtrados:
    st.warning("Nenhuma frase casa com a busca.")
    st.stop()

atual = estado.obter("frase", filtrados[0])
if atual not in filtrados:
    atual = filtrados[0]

with cnav:
    st.write("")
    b1, b2, b3 = st.columns(3)
    pos = filtrados.index(atual)
    if b1.button("◀ anterior", use_container_width=True):
        atual = filtrados[(pos - 1) % len(filtrados)]
        estado.definir("frase", atual)
        st.rerun()
    if b2.button("próxima ▶", use_container_width=True):
        atual = filtrados[(pos + 1) % len(filtrados)]
        estado.definir("frase", atual)
        st.rerun()
    if b3.button("aleatória ⤮", use_container_width=True):
        atual = random.choice(filtrados)
        estado.definir("frase", atual)
        st.rerun()

sid = st.selectbox("Frase", filtrados, index=filtrados.index(atual),
                   format_func=lambda i: f"{i} · {texto[i][:70]}")
estado.definir("frase", sid)

st.markdown(f"> {html.escape(texto.get(sid, ''))}")


# ---------------------------------------------------------------------------
# Renderização por condição
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _grafo_html(rid, ds, onto, cond, sid, altura=380):
    fr = frases.frase(rid, ds, onto, cond, sid)
    if not fr:
        return ""
    ar = grafo.arestas_frase(fr)
    nos = list({n for a in ar for n in (a["origem"], a["destino"])})
    return grafo_vis.construir_html(nos, ar, altura=altura)


def _bloco_prompt(partes):
    def blk(cls, titulo, conteudo):
        if not conteudo:
            return ""
        return (f"<div style='font-size:.75rem;color:#8a94a0;margin:2px'>{titulo}</div>"
                f"<div class='pp {cls}'>{html.escape(conteudo)}</div>")
    teste = partes["teste"]
    exemplo = ""
    if "Example Sentence:" in teste and "Test Sentence:" in teste:
        i = teste.find("Test Sentence:")
        exemplo, teste = teste[:i].strip(), teste[i:].strip()
    out = blk("pp-instrucao", "instrução", partes["instrucao"])
    out += blk("pp-contexto", "CONTEXT (ontologia)", partes["contexto"])
    out += blk("pp-instrucao", "exemplo (one-shot)", exemplo)
    out += blk("pp-teste", "frase de teste", teste)
    return out


def _resumo_ontologia(env):
    tipo = env.get("tipo")
    linhas = [f"<b>Tipo:</b> {env.get('origem','')}", f"<b>Relações:</b> {env.get('n_relacoes', 0)}"]
    corpo = "<br>".join(linhas)
    rels = env.get("relacoes")
    grupos = env.get("grupos")
    if rels:
        itens = ", ".join(html.escape(
            r["label"] + (f"({r['dominio']},{r['alcance']})" if r.get("dominio") else ""))
            for r in rels[:40])
        corpo += f"<div style='margin-top:6px;font-size:.82rem'>{itens}</div>"
    elif grupos:
        itens = []
        for g in grupos[:40]:
            membros = ", ".join(g["membros"][:6])
            itens.append(f"<b>{html.escape(g['label'])}</b> ← {html.escape(membros)}")
        corpo += "<div style='margin-top:6px;font-size:.82rem'>" + "<br>".join(itens) + "</div>"
    return chips.card("Ontologia enviada", corpo)


def _notas_cards(nt):
    if not nt:
        st.caption("Sem notas para esta frase.")
        return
    campos = [("f1", "F1"), ("precision", "P"), ("recall", "R"),
              ("onto_conf", "conformidade"), ("rel_halluc", "aluc. relação"), ("fmt_error", "erro formato")]
    cols = st.columns(len(campos))
    for col, (k, rot) in zip(cols, campos):
        with col:
            v = nt.get(k)
            chips.mostrar(chips.cartao_metrica(f"{v:.2f}" if isinstance(v, (int, float)) else "—", rot))
    st.caption(f"Fonte das notas: {nt.get('fonte', '—')}"
               + ("  ·  resposta vazia" if nt.get("vazia") else ""))


def _painel_condicao(cond):
    fr = frases.frase(rid, ds, onto, cond, sid)
    if not fr:
        st.info(f"A condição {COND_ROTULO.get(cond, cond)} não tem esta frase nesta rodada.")
        return
    env = ontologias.ontologia_enviada(rid, ds, onto, cond)
    chips.mostrar(_resumo_ontologia(env))

    chips.mostrar(chips.card("Prompt enviado", _bloco_prompt(fr["prompt_partes"])))

    with st.expander("Resposta crua do modelo"):
        st.code(fr["response"] or "(vazia)", language="text")

    chips.mostrar(chips.card(
        "Triplas extraídas", chips.triplas_html(fr["classificadas"], fr["faltas"])))
    chips.mostrar(chips.legenda())

    components.html(_grafo_html(rid, ds, onto, cond, sid), height=400, scrolling=False)
    _notas_cards(notas.notas_frase(rid, ds, onto, cond, sid))


# --- Abas por condição + Comparar ---
rotulos_abas = [COND_CURTO.get(c, c) for c in conds] + ["Comparar"]
abas = st.tabs(rotulos_abas)
for aba, cond in zip(abas[:-1], conds):
    with aba:
        _painel_condicao(cond)

with abas[-1]:
    st.caption("As 7 condições lado a lado para a mesma frase.")
    linhas = []
    for c in conds:
        nt = notas.notas_frase(rid, ds, onto, c, sid) or {}
        linhas.append({"condição": COND_ROTULO.get(c, c), "F1": nt.get("f1"),
                       "P": nt.get("precision"), "R": nt.get("recall"),
                       "conf.": nt.get("onto_conf"), "fmt": nt.get("fmt_error"),
                       "vazia": "sim" if nt.get("vazia") else ""})
    st.dataframe(linhas, use_container_width=True, hide_index=True)
    cols = st.columns(2)
    for i, c in enumerate(conds):
        with cols[i % 2]:
            st.markdown(f"**{COND_ROTULO.get(c, c)}**")
            components.html(_grafo_html(rid, ds, onto, c, sid, altura=260), height=280)
