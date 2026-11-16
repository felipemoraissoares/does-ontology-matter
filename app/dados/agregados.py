"""Médias por experimento (rodada × dataset × ontologia × condição), com cache em disco."""
import os
from typing import Dict, List

from . import cache
from .descoberta import experimento, experimentos
from .notas import notas_experimento

_CAMPOS = ("precision", "recall", "f1", "onto_conf", "rel_halluc",
           "sub_halluc", "obj_halluc", "fmt_error")


def agregado_experimento(rid: str, dataset: str, onto: str, cond: str) -> Dict:
    exp = experimento(rid, dataset, onto, cond)
    if not exp:
        return {}
    chave = f"agg::{exp['caminho']}"
    mt = os.path.getmtime(exp["caminho"])
    em_cache = cache.ler(chave, mt)
    if em_cache is not None:
        return em_cache

    notas = notas_experimento(rid, dataset, onto, cond)
    n = len(notas)
    if n == 0:
        return {}
    soma = {c: 0.0 for c in _CAMPOS}
    n_vazias = 0
    soma_conf_nv = 0.0
    n_nv = 0
    for nt in notas.values():
        for c in _CAMPOS:
            soma[c] += nt.get(c, 0.0)
        if nt.get("vazia"):
            n_vazias += 1
        else:
            n_nv += 1
            soma_conf_nv += nt.get("onto_conf", 0.0)
    agg = {f"avg_{c}": round(soma[c] / n, 4) for c in _CAMPOS}
    agg["pct_vazias"] = round(n_vazias / n, 4)
    agg["avg_onto_conf_nv"] = round(soma_conf_nv / n_nv, 4) if n_nv else 0.0
    agg["n"] = n
    cache.gravar(chave, mt, agg)
    return agg


def agregados_rodada(rid: str) -> List[Dict]:
    """Um registro por experimento presente: {dataset, onto, cond, avg_*, pct_vazias, n}."""
    out = []
    for e in experimentos(rid):
        agg = agregado_experimento(rid, e["dataset"], e["onto"], e["cond"])
        if agg:
            out.append({"dataset": e["dataset"], "onto": e["onto"], "cond": e["cond"], **agg})
    return out
