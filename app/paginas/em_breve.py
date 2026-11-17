"""Esqueleto: chat com o grafo (GraphRAG). Não implementado — só o plano."""
import streamlit as st

from componentes import chips

st.title("Chat com o grafo · em breve")
st.markdown(
    "Esta página vai permitir **perguntar em linguagem natural** sobre o grafo de uma rodada "
    "— um retriever busca subgrafos relevantes e um LLM responde citando as triplas (GraphRAG)."
)

chips.mostrar(chips.card("O que já está pronto para isso", (
    "A camada de dados exporta as triplas de qualquer rodada/condição como lista de arestas "
    "(<code>dados.grafo.exportar_arestas</code>), reutilizável por um carregamento no Neo4j e "
    "por um retriever. Nada aqui depende de serviço externo ainda."
)))

st.subheader("Prévia do que será indexado")
st.code(
    "from dados.grafo import exportar_arestas\n"
    "arestas = exportar_arestas('piloto-v3', 'tekgen', '1_movie', 'C1')\n"
    "# [{'frase': 'ont_1_movie_test_37', 'origem': 'Castle in the Sky',\n"
    "#   'rel': 'director', 'destino': 'Hayao Miyazaki', 'classe': 'acerto'}, ...]",
    language="python",
)
st.info("Planejado para depois do fechamento do experimento completo.")
