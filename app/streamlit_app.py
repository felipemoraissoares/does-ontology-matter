"""App Streamlit — exploração dos resultados do piloto-v3.

Funciona só com os arquivos de resultados/ (Neo4j não é necessário).

Rodar:
  streamlit run app/streamlit_app.py
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
import dados_piloto as dp

REPO = dp.REPO
WORKSPACE = os.path.abspath(os.path.join(REPO, ".."))
BENCH = os.path.join(WORKSPACE, "Text2KGBench")

COR = {"acerto": "#c8e6c9", "sobra": "#fff9c4", "falta": "#ffcdd2", "fmt": "#e0e0e0"}
ROTULO = {"acerto": "acerto", "sobra": "sobra", "falta": "falta", "fmt": "erro de formato"}

st.set_page_config(page_title="does-ontology-matter — piloto v3", layout="wide")


def _fmt_tripla(tr):
    return f"{tr[1]}({tr[0]}, {tr[2]})"


def _chip(texto, status):
    return (f"<span style='background:{COR[status]};padding:2px 6px;border-radius:4px;"
            f"margin:2px;display:inline-block'>{texto}</span>")


def _norm_set(triplas):
    from t2kg.avaliacao.original import normalizar_tripla
    return {normalizar_tripla(t[0], t[1], t[2]) for t in triplas if len(t) >= 3}


@st.cache_resource(show_spinner=False)
def _ontologia_ref(dataset, onto):
    from t2kg.pipeline.ontologias import carregar_ontologia
    return carregar_ontologia(BENCH, dataset, onto)


@st.cache_data(show_spinner=False)
def _definicoes(onto):
    import json
    p = os.path.join(REPO, "configs", "c2", f"definicoes_{onto}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("relacoes", {})


def _projetar(dataset, onto, llm_triples):
    """Projeção best-effort (requer Text2KGBench + sentence-transformers)."""
    from t2kg.avaliacao.extended import projetar_relacoes
    ont = _ontologia_ref(dataset, onto)
    defs = _definicoes(onto)
    proj, _ = projetar_relacoes(llm_triples, ont, definicoes_ref=defs)
    return proj


def pagina_comparar():
    st.header("Comparar condições")
    c1, c2 = st.columns(2)
    dataset = c1.selectbox("Base", ["tekgen", "webnlg"])
    ontos = dp.listar_ontologias(dataset)
    if not ontos:
        st.warning("Sem arquivos de avaliação para esta base.")
        return
    onto = c2.selectbox("Ontologia", ontos)

    conds = dp.condicoes_disponiveis(dataset, onto)
    registros = {c: dp.carregar_condicao(dataset, c, onto) for c in conds}

    # frases disponíveis (id -> sentença), a partir da 1ª condição com dados
    base_cond = next((c for c in conds if registros[c]), None)
    if base_cond is None:
        st.warning("Sem registros.")
        return
    id_para_sent = {sid: r.get("sent", "") for sid, r in registros[base_cond].items()}
    ids = list(id_para_sent)
    sid = st.selectbox("Frase", ids,
                       format_func=lambda i: f"{i}: {id_para_sent[i][:80]}")

    mostrar_proj = st.checkbox(
        "Calcular projeção (C0/C2/C3) — requer Text2KGBench e sentence-transformers",
        value=False)

    st.markdown("**Legenda:** " + " ".join(_chip(ROTULO[k], k) for k in COR))
    st.markdown(f"> {id_para_sent[sid]}")

    # Gabarito
    gt = dp.gt_da_frase(registros[base_cond].get(sid, {}))
    st.subheader("Gabarito")
    st.markdown(" ".join(_chip(_fmt_tripla(t), "acerto") for t in gt) or "_(vazio)_",
                unsafe_allow_html=True)

    gt_norm = _norm_set(gt)
    ont_ref = None
    if os.path.exists(BENCH):
        try:
            ont_ref = _ontologia_ref(dataset, onto)
        except Exception:
            ont_ref = None

    st.subheader("Condições")
    colunas = st.columns(len(conds))
    for col, cond in zip(colunas, conds):
        rec = registros[cond].get(sid)
        with col:
            st.markdown(f"**{dp.COND_ROTULO.get(cond, cond)}**")
            if rec is None:
                st.caption("sem dados")
                continue
            llm = [list(t) for t in rec.get("llm_triples", [])]
            classif = dp.classificar_sistema(llm, gt_norm, ont_ref)
            sys_norm = _norm_set(llm)
            falt = dp.faltas(gt, sys_norm)

            chips = [_chip(_fmt_tripla(tr), status) for tr, status in classif]
            chips += [_chip(_fmt_tripla(t), "falta") for t in falt]
            st.markdown(" ".join(chips) or "_(nenhuma tripla)_", unsafe_allow_html=True)

            if mostrar_proj and cond in dp.CONDICOES_PROJETAVEIS:
                if not os.path.exists(BENCH):
                    st.caption("projeção: Text2KGBench não encontrado")
                else:
                    try:
                        with st.spinner("projetando..."):
                            proj = _projetar(dataset, onto, llm)
                        st.caption("Projetadas: " +
                                   (", ".join(sorted({t[1] for t in proj})) or "nenhuma"))
                    except Exception as e:
                        st.caption(f"projeção indisponível: {str(e)[:80]}")


def pagina_visao_geral():
    st.header("Visão geral — métricas do piloto v3")
    linhas = dp.carregar_tabela_metricas()
    if not linhas:
        st.warning("tabela_piloto_v3.csv não encontrada.")
        return
    try:
        import pandas as pd
        df = pd.DataFrame(linhas)
        cols = [c for c in ["dataset", "condicao", "avg_precision", "avg_recall", "avg_f1",
                            "avg_onto_conf", "avg_fmt_error", "ext_projetado_f1"]
                if c in df.columns]
        st.dataframe(df[cols], use_container_width=True)
    except ImportError:
        st.table(linhas)


def main():
    st.sidebar.title("Piloto v3")
    pagina = st.sidebar.radio("Página", ["Comparar condições", "Visão geral"])
    if pagina == "Comparar condições":
        pagina_comparar()
    else:
        pagina_visao_geral()


main()
