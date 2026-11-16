"""Parser de docs/DECISOES.md: lê os blocos `## D-xx: Título` e seus corpos.

Mantém o texto das decisões sempre sincronizado com o documento-fonte.
"""
import os
import re
from functools import lru_cache
from typing import Dict, List

from .base import DOCS

_RE_CABECALHO = re.compile(r"^##\s*(D-\d+)\s*:\s*(.+?)\s*$", re.MULTILINE)


@lru_cache(maxsize=1)
def _bruto() -> str:
    p = os.path.join(DOCS, "DECISOES.md")
    if not os.path.exists(p):
        return ""
    with open(p, encoding="utf-8") as f:
        return f.read()


@lru_cache(maxsize=1)
def todas() -> Dict[str, Dict[str, str]]:
    """{ 'D-01': {'id','titulo','corpo'} } na ordem do documento."""
    texto = _bruto()
    achados = list(_RE_CABECALHO.finditer(texto))
    out: Dict[str, Dict[str, str]] = {}
    for i, m in enumerate(achados):
        ini = m.end()
        fim = achados[i + 1].start() if i + 1 < len(achados) else len(texto)
        corpo = texto[ini:fim].strip()
        out[m.group(1)] = {"id": m.group(1), "titulo": m.group(2), "corpo": corpo}
    return out


def decisao(did: str) -> Dict[str, str]:
    return todas().get(did, {"id": did, "titulo": "(não encontrada)", "corpo": ""})


def titulo(did: str) -> str:
    return decisao(did)["titulo"]


def ids() -> List[str]:
    return list(todas().keys())
