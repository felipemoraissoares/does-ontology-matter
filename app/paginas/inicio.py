"""Página inicial: o que é o projeto + números-chave + atalhos."""
import datetime
import os

import streamlit as st

from dados import descoberta
from dados.base import DATASETS, RESULTADOS, caminho_gt, ontologias
from componentes import chips


@st.cache_data(show_spinner=False)
def _numeros():
    total_frases = 0
    n_ontos = 0
    for ds in DATASETS:
        for o in ontologias(ds):
            n_ontos += 1
            p = caminho_gt(ds, o)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    total_frases += sum(1 for l in f if l.strip())
    # experimentos da rodada full (qwen2.5-7b), se houver
    full_feitos = len(descoberta.experimentos("full/qwen2.5-7b")) if descoberta.rodada("full/qwen2.5-7b") else 0
    # última atualização: mtime mais recente em resultados/
    ult = 0.0
    for raiz, _, arqs in os.walk(RESULTADOS):
        for a in arqs:
            if a.endswith(".jsonl"):
                ult = max(ult, os.path.getmtime(os.path.join(raiz, a)))
    data = datetime.datetime.fromtimestamp(ult).strftime("%d/%m %H:%M") if ult else "—"
    return {"frases": total_frases, "ontos": n_ontos, "rodadas": len(descoberta.rodadas()),
            "full_feitos": full_feitos, "data": data}


st.title("Ontologia & LLM — explorador do projeto")
st.markdown(
    "Mede o **efeito da estratégia de ontologia** na extração de triplas por um LLM, sobre o "
    "Text2KGBench. Sete condições variam só o bloco de ontologia no prompt, mantendo modelo, "
    "instrução e frase fixos. Aqui dá para ver, **fase a fase**, o que foi feito e, para "
    "**qualquer frase**, o que foi ao modelo e o que voltou."
)

n = _numeros()
cols = st.columns(6)
cartoes = [
    (n["frases"], "frases de teste"),
    (n["ontos"], "ontologias"),
    (7, "condições"),
    (n["rodadas"], "rodadas disponíveis"),
    (f"{n['full_feitos']}/203", "experimentos (full)"),
    (n["data"], "última atualização"),
]
for col, (v, r) in zip(cols, cartoes):
    with col:
        chips.mostrar(chips.cartao_metrica(v, r))

st.subheader("Por onde começar")
a, b, c = st.columns(3)
with a:
    st.page_link("paginas/explorador.py", label="Explorador de frase", icon=":material/search:")
    st.caption("Para uma frase, o prompt, a resposta e as triplas classificadas.")
with b:
    st.page_link("paginas/fases.py", label="Fases do projeto", icon=":material/timeline:")
    st.caption("A linha do tempo das 9 fases, com decisões D-xx.")
with c:
    st.page_link("paginas/resultados.py", label="Resultados agregados", icon=":material/bar_chart:")
    st.caption("Métricas por condição × base, piloto e rodada completa.")
