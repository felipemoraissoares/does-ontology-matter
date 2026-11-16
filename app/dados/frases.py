"""Carrega frases completas de um experimento: prompt, resposta, triplas, gabarito e
classificação. Reaproveita o ground_truth do Text2KGBench e a ontologia de referência.
"""
import json
import os
from functools import lru_cache
from typing import Dict, List, Optional

from t2kg.pipeline.ontologias import carregar_ontologia

from .base import BENCH, caminho_gt, separar_prompt
from .classificacao import classificar, contagem_classes, faltas
from .descoberta import experimento


@lru_cache(maxsize=64)
def _ler_por_id(caminho: str, _mtime: float) -> Dict[str, Dict]:
    d = {}
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            try:
                r = json.loads(linha)
            except json.JSONDecodeError:
                continue
            if "id" in r:
                d[r["id"]] = r
    return d


def _por_id(caminho: str) -> Dict[str, Dict]:
    mt = os.path.getmtime(caminho) if os.path.exists(caminho) else 0.0
    return _ler_por_id(caminho, mt)


@lru_cache(maxsize=None)
def gabarito(dataset: str, onto: str) -> Dict[str, Dict]:
    """{id: {'sent', 'triplas':[[s,r,o],...]}} do ground_truth."""
    p = caminho_gt(dataset, onto)
    g = {}
    if not os.path.exists(p):
        return g
    with open(p, encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            r = json.loads(linha)
            g[r["id"]] = {"sent": r.get("sent", ""),
                          "triplas": [[t["sub"], t["rel"], t["obj"]] for t in r.get("triples", [])]}
    return g


@lru_cache(maxsize=64)
def _ont_ref(dataset: str, onto: str) -> Dict:
    return carregar_ontologia(BENCH, dataset, onto)


def listar_frases(rid: str, dataset: str, onto: str, cond: str) -> List[Dict]:
    exp = experimento(rid, dataset, onto, cond)
    if not exp:
        return []
    return [{"id": r["id"], "sent": r.get("sent", "")}
            for r in _por_id(exp["caminho"]).values()]


def frase(rid: str, dataset: str, onto: str, cond: str, sent_id: str) -> Optional[Dict]:
    exp = experimento(rid, dataset, onto, cond)
    if not exp:
        return None
    rec = _por_id(exp["caminho"]).get(sent_id)
    if rec is None:
        return None
    gt = gabarito(dataset, onto).get(sent_id, {"sent": rec.get("sent", ""), "triplas": []})
    ont = _ont_ref(dataset, onto)
    triplas = rec.get("triples", [])
    classificadas = classificar(triplas, gt["triplas"], ont)
    return {
        "id": sent_id,
        "sent": rec.get("sent") or gt["sent"],
        "prompt": rec.get("prompt", ""),
        "prompt_partes": separar_prompt(rec.get("prompt", "")),
        "response": rec.get("response", ""),
        "finish_reason": rec.get("finish_reason"),
        "triples": triplas,
        "gt_triplas": gt["triplas"],
        "classificadas": classificadas,
        "faltas": faltas(gt["triplas"], triplas),
        "contagem": contagem_classes(classificadas),
    }
