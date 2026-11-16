"""Métricas por frase. Usa o arquivo de avaliação quando existe; senão calcula com as
funções de src/t2kg/avaliacao/original (sem reimplementar fórmulas), com cache em disco.
"""
import json
import os
from functools import lru_cache
from typing import Dict, Optional

from t2kg.avaliacao.original import (
    calcular_precisao_recall_f1, normalizar_tripla, calcular_conformidade_ontologica,
    calcular_alucinacao_sujeito_objeto, calcular_taxa_erro_formato,
)

from . import cache
from .descoberta import experimento, rodada
from .frases import _ont_ref, _por_id, gabarito

_METRICAS = ("precision", "recall", "f1", "onto_conf", "rel_halluc",
             "sub_halluc", "obj_halluc", "fmt_error")


@lru_cache(maxsize=1)
def _stemmer():
    from nltk.stem import PorterStemmer
    return PorterStemmer()


def _caminho_eval(rid: str, dataset: str, onto: str, cond: str) -> Optional[str]:
    r = rodada(rid)
    if not r:
        return None
    p = os.path.join(r["avaliacao_dir"], f"{dataset}_{cond}_{onto}.jsonl")
    return p if os.path.exists(p) else None


def _ler_eval(caminho: str) -> Dict[str, Dict]:
    out = {}
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            d = json.loads(linha)
            if d.get("type") in ("all_test_cases", "global", "selected_test_cases"):
                continue
            if "id" in d:
                nota = {m: float(d[m]) for m in _METRICAS if m in d}
                nota["vazia"] = len(d.get("llm_triples", [])) == 0
                nota["fonte"] = "arquivo"
                out[d["id"]] = nota
    return out


def _calcular_frase(sent: str, sys_triplas, gt_triplas, ont, response: str) -> Dict:
    gt_rels = {t[1].replace(" ", "_") for t in gt_triplas}
    fil = [t for t in sys_triplas if len(t) >= 3 and t[1] in gt_rels]
    gt_norm = {normalizar_tripla(t[0], t[1], t[2]) for t in gt_triplas}
    sys_norm = {normalizar_tripla(t[0], t[1], t[2]) for t in fil}
    p, r, f1 = calcular_precisao_recall_f1(gt_norm, sys_norm)
    conf, relh = calcular_conformidade_ontologica(ont, sys_triplas)
    subh, objh = calcular_alucinacao_sujeito_objeto(_stemmer(), ont, sent, sys_triplas)
    fmt = calcular_taxa_erro_formato(ont, sys_triplas)
    return {"precision": round(p, 2), "recall": round(r, 2), "f1": round(f1, 2),
            "onto_conf": round(conf, 2), "rel_halluc": round(relh, 2),
            "sub_halluc": round(subh, 2), "obj_halluc": round(objh, 2),
            "fmt_error": round(fmt, 2),
            "vazia": len(sys_triplas) == 0, "fonte": "calculado"}


def _calcular_experimento(rid, dataset, onto, cond) -> Dict[str, Dict]:
    exp = experimento(rid, dataset, onto, cond)
    if not exp:
        return {}
    chave = f"notas::{exp['caminho']}"
    mt = os.path.getmtime(exp["caminho"])
    em_cache = cache.ler(chave, mt)
    if em_cache is not None:
        return em_cache
    ont = _ont_ref(dataset, onto)
    gt = gabarito(dataset, onto)
    out = {}
    for sid, rec in _por_id(exp["caminho"]).items():
        g = gt.get(sid, {"sent": rec.get("sent", ""), "triplas": []})
        out[sid] = _calcular_frase(rec.get("sent") or g["sent"], rec.get("triples", []),
                                   g["triplas"], ont, rec.get("response", ""))
    cache.gravar(chave, mt, out)
    return out


def notas_experimento(rid: str, dataset: str, onto: str, cond: str) -> Dict[str, Dict]:
    """{id: notas}. Prioriza o arquivo de avaliação; senão calcula (com cache em disco)."""
    cam = _caminho_eval(rid, dataset, onto, cond)
    if cam:
        return _ler_eval(cam)
    return _calcular_experimento(rid, dataset, onto, cond)


def notas_frase(rid: str, dataset: str, onto: str, cond: str, sent_id: str) -> Optional[Dict]:
    return notas_experimento(rid, dataset, onto, cond).get(sent_id)
