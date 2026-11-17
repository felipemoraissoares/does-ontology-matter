"""Resultados agregados: tabelas e gráficos por condição × base, piloto e rodada full."""
import pandas as pd
import plotly.express as px
import streamlit as st

from dados import agregados, descoberta
from dados.base import (BASELINE_ARTIGO, COND_CURTO, CONDICOES, DS_ROTULO)
from componentes import estado
from componentes.filtros import seletor_rodada

st.title("Resultados agregados")

rid = seletor_rodada()
if not rid:
    st.info("Nenhuma rodada disponível.")
    st.stop()


@st.cache_data(show_spinner=True)
def _df(rid):
    regs = agregados.agregados_rodada(rid)
    return pd.DataFrame(regs)


df = _df(rid)
if df.empty:
    st.warning("Rodada ainda sem experimentos avaliáveis.")
    st.stop()

METRICAS = {
    "avg_f1": "F1", "avg_precision": "Precisão", "avg_recall": "Recall",
    "avg_onto_conf": "Conformidade", "avg_onto_conf_nv": "Conformidade s/ vazias",
    "avg_rel_halluc": "Aluc. relação", "avg_fmt_error": "Erro de formato",
    "pct_vazias": "% vazias",
}
met = st.selectbox("Métrica", list(METRICAS), format_func=lambda m: METRICAS[m],
                   index=estado.indice(list(METRICAS), estado.obter("metrica", "avg_f1")))
estado.definir("metrica", met)

df["cond_curto"] = df["cond"].map(lambda c: COND_CURTO.get(c, c))
ordem = [COND_CURTO[c] for c in CONDICOES]

# --- Barras por condição × base (média ± desvio entre ontologias) ---
resumo = (df.groupby(["dataset", "cond_curto"])[met]
          .agg(["mean", "std", "count"]).reset_index())
resumo["dataset"] = resumo["dataset"].map(lambda d: DS_ROTULO.get(d, d))
fig = px.bar(resumo, x="cond_curto", y="mean", color="dataset", barmode="group",
             error_y="std", category_orders={"cond_curto": ordem},
             labels={"cond_curto": "condição", "mean": METRICAS[met], "dataset": "base"},
             color_discrete_sequence=["#2E5A8A", "#C99A00"])
if met == "avg_f1":
    for ds, val in BASELINE_ARTIGO.items():
        fig.add_hline(y=val, line_dash="dot", line_color="#8a94a0",
                      annotation_text=f"Vicuna-13B {DS_ROTULO[ds]} ({val})",
                      annotation_position="top left")
fig.update_layout(height=430, margin=dict(t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

# --- Mapa de calor ontologia × condição ---
st.subheader("Mapa de calor · ontologia × condição")
for ds in sorted(df["dataset"].unique()):
    sub = df[df["dataset"] == ds]
    piv = sub.pivot_table(index="onto", columns="cond_curto", values=met, aggfunc="mean")
    piv = piv.reindex(columns=[c for c in ordem if c in piv.columns])
    if piv.empty:
        continue
    fig2 = px.imshow(piv, text_auto=".2f", aspect="auto", color_continuous_scale="Blues",
                     labels=dict(color=METRICAS[met]))
    fig2.update_layout(height=max(220, 26 * len(piv) + 90), margin=dict(t=30, b=10),
                       title=DS_ROTULO.get(ds, ds))
    st.plotly_chart(fig2, use_container_width=True)

# --- Tabela ---
st.subheader("Tabela")
tab = (df.groupby(["dataset", "cond_curto"])[list(METRICAS)].mean().reset_index()
       .sort_values(["dataset", "cond_curto"]))
st.dataframe(tab.round(3), use_container_width=True, hide_index=True)
st.caption(f"Rodada {descoberta.rodada(rid)['rotulo']} · {len(df)} experimentos avaliados. "
           "Métricas calculadas sob demanda quando não há arquivo de avaliação (rodada full).")
