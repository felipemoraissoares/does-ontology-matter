"""Estado de filtros compartilhado entre páginas: session_state + query_params (URL)."""
import streamlit as st


def obter(chave: str, default=None):
    """session_state tem prioridade; senão lê da URL; senão o default."""
    if chave in st.session_state and st.session_state[chave] is not None:
        return st.session_state[chave]
    qp = st.query_params.get(chave)
    if qp not in (None, ""):
        st.session_state[chave] = qp
        return qp
    return default


def definir(chave: str, valor) -> None:
    st.session_state[chave] = valor
    if valor is None:
        st.query_params.pop(chave, None)
    else:
        st.query_params[chave] = str(valor)


def indice(opcoes, valor, padrao: int = 0) -> int:
    try:
        return opcoes.index(valor)
    except (ValueError, AttributeError):
        return padrao
