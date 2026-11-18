"""Ontologias: referência × induzida (C2) × top-3 (C3) × parceiras (C5), lado a lado."""
import html

import streamlit as st
import streamlit.components.v1 as components

from dados import descoberta, ontologias as od
from componentes import chips, grafo_vis
from componentes.filtros import filtros_rodada_ds_onto

st.title("Ontologias")

rid, ds, onto = filtros_rodada_ds_onto()
if not (rid and ds and onto):
    st.info("Nenhum experimento disponível.")
    st.stop()


@st.cache_data(show_spinner=False)
def _mapa_induzida(rid, ds, onto):
    """Mapeia cada relação induzida para a referência: (ref, 'igual'|'parecido'|'sem par')."""
    ind = od.ontologia_induzida(rid, ds, onto)
    if not ind:
        return None
    ref = od.ontologia_referencia(ds, onto)["bruta"]
    defs = od._definicoes(rid, onto)
    try:
        from t2kg.avaliacao.extended import projetar_relacoes, _normalizar_rotulo
        from t2kg.avaliacao.limiares import limiar_projecao_por_rid
    except Exception:
        return {"erro": "sentence-transformers indisponível"}
    limiar = limiar_projecao_por_rid(rid)
    out = []
    for g in ind["grupos"]:
        lab = g["label"]
        try:
            proj, _ = projetar_relacoes([["a", lab, "b"]], ref, threshold=limiar, definicoes_ref=defs or None)
        except Exception as e:
            return {"erro": str(e)[:120]}
        if not proj:
            out.append({"label": lab, "membros": g["membros"], "ref": None, "classe": "sem par"})
        else:
            rlab = proj[0][1]
            classe = "igual" if _normalizar_rotulo(rlab) == _normalizar_rotulo(lab) else "parecido"
            out.append({"label": lab, "membros": g["membros"], "ref": rlab, "classe": classe})
    return {"itens": out}


@st.cache_data(show_spinner=False)
def _diagrama_ref(ds, onto, altura=380):
    rels = od.ontologia_referencia(ds, onto)["relacoes"]
    arestas, nos = [], set()
    for r in rels[:40]:
        dom = r.get("dominio") or "?"
        rng = r.get("alcance") or "?"
        arestas.append({"origem": dom, "rel": r["label"], "destino": rng, "classe": "ref"})
        nos.update([dom, rng])
    return grafo_vis.construir_html(list(nos), arestas, altura=altura)


col1, col2 = st.columns(2)

with col1:
    ref = od.ontologia_referencia(ds, onto)
    itens = ", ".join(html.escape(f"{r['label']}({r['dominio']},{r['alcance']})")
                      for r in ref["relacoes"])
    chips.mostrar(chips.card(
        f"Referência · {len(ref['relacoes'])} relações, {len(ref['conceitos'])} conceitos",
        f"<div style='font-size:.82rem'>{itens}</div>"))
    components.html(_diagrama_ref(ds, onto), height=400)

with col2:
    mapa = _mapa_induzida(rid, ds, onto)
    if mapa is None:
        chips.mostrar(chips.card("Induzida (C2)", "Não há ontologia induzida nesta rodada."))
    elif "erro" in mapa:
        chips.mostrar(chips.card("Induzida (C2)", f"Mapeamento indisponível: {html.escape(mapa['erro'])}"))
    else:
        cor = {"igual": "#2E7D4F", "parecido": "#C99A00", "sem par": "#8a94a0"}
        linhas = []
        for it in mapa["itens"]:
            alvo = f" → <b>{html.escape(it['ref'])}</b>" if it["ref"] else ""
            membros = html.escape(", ".join(it["membros"][:6]))
            linhas.append(
                f"<div style='margin:3px 0'><span class='pilula' style='background:{cor[it['classe']]}'></span>"
                f"<b>{html.escape(it['label'])}</b>{alvo} "
                f"<span style='color:#8a94a0'>({it['classe']})</span><br>"
                f"<span style='font-size:.78rem;color:#53606e'>← {membros}</span></div>")
        chips.mostrar(chips.card(
            f"Induzida (C2) · {len(mapa['itens'])} relações",
            "".join(linhas) + "<div style='margin-top:6px;font-size:.78rem;color:#8a94a0'>"
            "igual · parecido · sem par (mapeamento por embedding label+definição, D-24/D-30)</div>"))

st.divider()
c3, c5 = st.columns(2)
with c3:
    top3 = od.top3_c3(ds, onto)
    itens = "<br>".join(html.escape(f"{r['label']}({r.get('dominio','')},{r.get('alcance','')})")
                        for r in top3) or "—"
    chips.mostrar(chips.card("C3 · top-3 relações do treino", itens))

with c5:
    partes = []
    for papel in ("vizinho", "incongruente"):
        p = od.parceira_c5(ds, onto, papel)
        if p:
            partes.append(f"<b>{papel}:</b> {html.escape(p['parceira_id'])} "
                          f"(SimE={p['sim_emb']}, {p['n_relacoes']} relações)")
    chips.mostrar(chips.card("C5 · parceiras", "<br>".join(partes) or "—"))

# C2: respostas cruas + definições
ind = od.ontologia_induzida(rid, ds, onto)
if ind:
    with st.expander("C2 · definições D-24 das relações induzidas"):
        defs = od._definicoes(rid, onto)
        for g in ind["grupos"][:30]:
            d = defs.get(g["label"]) or defs.get(g["label"].lower()) or ""
            st.markdown(f"**{g['label']}** — {html.escape(d[:160])}")
