"""Camada de dados do app (sem Streamlit). Reaproveita src/t2kg/."""
from . import (agregados, base, cache, classificacao, decisoes, descoberta,
               frases, grafo, notas, ontologias)

__all__ = ["agregados", "base", "cache", "classificacao", "decisoes",
           "descoberta", "frases", "grafo", "notas", "ontologias"]
