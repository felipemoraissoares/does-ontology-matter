"""Descoberta dinâmica de rodadas e experimentos a partir de resultados/.

Uma *rodada* é um conjunto de execuções de um modelo (piloto-v3, full/qwen2.5-7b, …).
Novos arquivos que chegarem por git pull aparecem sem mudar código: a descoberta casa
cada arquivo contra (dataset, ontologia, condição) gerando o stem esperado pelo esquema de
nomes da rodada — não depende de regex frágil.
"""
import os
from functools import lru_cache
from typing import Dict, List, Optional

from .base import (CONDICOES, DATASETS, DS_PREFIXO, RESULTADOS, ontologias)

_MODELO_PILOTO = "Qwen/Qwen2.5-7B-Instruct-AWQ"


def _cond_full(c: str) -> str:
    return c.lower()


def _cond_piloto(c: str) -> str:
    return c.lower().replace("_", "")


def _stem_full(ds: str, onto: str, cond: str, slug: str) -> str:
    return f"{DS_PREFIXO[ds]}_{onto}_{_cond_full(cond)}_{slug}"


def _stem_piloto(ds: str, onto: str, cond: str) -> str:
    return f"pv3_{onto.replace('_', '')}_{_cond_piloto(cond)}"


def _contar_linhas(caminho: str) -> int:
    try:
        with open(caminho, encoding="utf-8") as f:
            return sum(1 for l in f if l.strip())
    except OSError:
        return 0


@lru_cache(maxsize=1)
def rodadas() -> List[Dict]:
    """Lista de rodadas disponíveis (piloto-v3 + full/<slug>)."""
    out = []
    piloto_dir = os.path.join(RESULTADOS, "piloto_v3")
    if os.path.isdir(piloto_dir):
        out.append({
            "id": "piloto-v3", "rotulo": "Piloto v3", "esquema": "pv3",
            "modelo": _MODELO_PILOTO, "slug": "qwen2.5-7b", "dir": piloto_dir,
            "amostra": True,
            "avaliacao_dir": os.path.join(RESULTADOS, "avaliacao_piloto_v3"),
        })
    full_dir = os.path.join(RESULTADOS, "full")
    if os.path.isdir(full_dir):
        for slug in sorted(os.listdir(full_dir)):
            d = os.path.join(full_dir, slug)
            if os.path.isdir(d):
                out.append({
                    "id": f"full/{slug}", "rotulo": f"Completo · {slug}", "esquema": "full",
                    "modelo": slug, "slug": slug, "dir": d, "amostra": False,
                    "avaliacao_dir": os.path.join(RESULTADOS, "avaliacao_full", slug),
                })
    return out


def rodada(rid: str) -> Optional[Dict]:
    return next((r for r in rodadas() if r["id"] == rid), None)


@lru_cache(maxsize=None)
def _mapa_stems(rid: str) -> Dict[str, tuple]:
    """stem -> (dataset, onto, cond) para a rodada."""
    r = rodada(rid)
    mapa = {}
    if not r:
        return mapa
    for ds in DATASETS:
        for onto in ontologias(ds):
            for cond in CONDICOES:
                if r["esquema"] == "pv3":
                    stem = _stem_piloto(ds, onto, cond)
                else:
                    stem = _stem_full(ds, onto, cond, r["slug"])
                mapa[stem] = (ds, onto, cond)
    return mapa


def experimentos(rid: str) -> List[Dict]:
    """Experimentos presentes na rodada: {dataset, onto, cond, caminho, n_linhas}."""
    r = rodada(rid)
    if not r:
        return []
    mapa = _mapa_stems(rid)
    out = []
    for nome in os.listdir(r["dir"]):
        if not nome.endswith(".jsonl"):
            continue
        chave = mapa.get(nome[:-len(".jsonl")])
        if chave is None:
            continue
        ds, onto, cond = chave
        caminho = os.path.join(r["dir"], nome)
        out.append({"dataset": ds, "onto": onto, "cond": cond,
                    "caminho": caminho, "n_linhas": _contar_linhas(caminho)})
    out.sort(key=lambda e: (e["dataset"], e["onto"], CONDICOES.index(e["cond"])))
    return out


def experimento(rid: str, dataset: str, onto: str, cond: str) -> Optional[Dict]:
    for e in experimentos(rid):
        if e["dataset"] == dataset and e["onto"] == onto and e["cond"] == cond:
            return e
    return None


def ontologias_da_rodada(rid: str) -> List[tuple]:
    """(dataset, onto) distintos com ao menos um experimento na rodada."""
    vistos = []
    for e in experimentos(rid):
        chave = (e["dataset"], e["onto"])
        if chave not in vistos:
            vistos.append(chave)
    return vistos


def condicoes_da_ontologia(rid: str, dataset: str, onto: str) -> List[str]:
    return [e["cond"] for e in experimentos(rid)
            if e["dataset"] == dataset and e["onto"] == onto]
