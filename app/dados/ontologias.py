"""Qual ontologia foi enviada ao modelo em cada condição (ref/induzida/top-3/parceira C5).

Reaproveita src/t2kg/pipeline (carregar_ontologia, carregar_ontologia_induzida) e
carregar_frequencias_treino. Não monta prompt aqui (o prompt vem do próprio JSONL).
"""
import json
import os
from functools import lru_cache
from typing import Dict, List, Optional

from t2kg.pipeline.ontologias import carregar_ontologia, carregar_ontologia_induzida
from t2kg.pipeline.dados import carregar_frequencias_treino

from .base import BENCH, CONFIGS, caminho_ontologia_ref
from .descoberta import rodada


@lru_cache(maxsize=1)
def _pares_c5() -> Dict:
    p = os.path.join(CONFIGS, "pipeline", "pares_c5.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _ref_relacoes(ont: Dict) -> List[Dict]:
    id2label = {c.get("qid", c["label"]): c["label"] for c in ont.get("concepts", [])}
    rels = []
    for r in ont.get("relations", []):
        rels.append({
            "label": r["label"],
            "dominio": id2label.get(r.get("domain", ""), r.get("domain", "")),
            "alcance": id2label.get(r.get("range", ""), r.get("range", "")),
        })
    return rels


def ontologia_referencia(dataset: str, onto: str) -> Dict:
    ont = carregar_ontologia(BENCH, dataset, onto)
    return {
        "conceitos": [c["label"] for c in ont.get("concepts", [])],
        "relacoes": _ref_relacoes(ont),
        "bruta": ont,
    }


def _c2_subdir(rid: str) -> Optional[str]:
    r = rodada(rid)
    if not r or r["esquema"] == "pv3":
        return None
    return f"v_full/{r['slug']}"


def ontologia_induzida(rid: str, dataset: str, onto: str) -> Optional[Dict]:
    try:
        ind = carregar_ontologia_induzida(dataset, onto, c2_subdir=_c2_subdir(rid))
    except FileNotFoundError:
        return None
    meta = ind.get("metadata", {})
    grupos = [{"label": r["label"], "freq": r.get("frequencia"),
               "membros": r.get("membros", [])} for r in ind.get("relations", [])]
    return {"run_id": meta.get("run_id"), "threshold": meta.get("threshold_agrupamento"),
            "grupos": grupos, "n_relacoes": len(grupos)}


def _definicoes(rid: str, onto: str) -> Dict[str, str]:
    r = rodada(rid)
    base = CONFIGS + "/c2"
    if r and r["esquema"] != "pv3":
        base = os.path.join(CONFIGS, "c2", "v_full", r["slug"])
    p = os.path.join(base, f"definicoes_{onto}.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("relacoes", {})


def top3_c3(dataset: str, onto: str) -> List[Dict]:
    try:
        labels = carregar_frequencias_treino(BENCH, dataset, onto, k=3)
    except (FileNotFoundError, OSError):
        return []
    ref_rels = ontologia_referencia(dataset, onto)["relacoes"]
    ref = {r["label"].replace(" ", "_"): r for r in ref_rels}
    out = []
    for lab in labels:
        r = ref.get(lab) or ref.get(lab.replace(" ", "_"))
        out.append(r or {"label": lab, "dominio": "", "alcance": ""})
    return out


def parceira_c5(dataset: str, onto: str, papel: str) -> Optional[Dict]:
    """papel: 'vizinho' | 'incongruente'."""
    info = _pares_c5().get(dataset, {}).get(onto, {}).get(papel)
    if not info:
        return None
    pid = info["id"]
    rels = ontologia_referencia(dataset, pid)["relacoes"]
    return {"parceira_id": pid, "sim_emb": info.get("sim_emb"),
            "overlap_jaccard": info.get("overlap_jaccard"), "papel": papel,
            "relacoes": rels, "n_relacoes": len(rels)}


def ontologia_enviada(rid: str, dataset: str, onto: str, cond: str) -> Dict:
    """Resumo estruturado da ontologia no prompt daquela condição."""
    if cond == "C0":
        return {"tipo": "nenhuma", "origem": "Nenhuma ontologia no prompt."}
    if cond in ("C1", "C1_oneshot_original"):
        ref = ontologia_referencia(dataset, onto)
        return {"tipo": "referencia", "origem": "Ontologia de referência do alvo.",
                "relacoes": ref["relacoes"], "conceitos": ref["conceitos"],
                "n_relacoes": len(ref["relacoes"])}
    if cond == "C2":
        ind = ontologia_induzida(rid, dataset, onto)
        if not ind:
            return {"tipo": "induzida", "origem": "Ontologia induzida (ausente nesta rodada).",
                    "n_relacoes": 0, "grupos": []}
        return {"tipo": "induzida",
                "origem": f"Induzida pelo modelo (run {ind['run_id']}).",
                "grupos": ind["grupos"], "n_relacoes": ind["n_relacoes"],
                "definicoes": _definicoes(rid, onto)}
    if cond == "C3":
        rels = top3_c3(dataset, onto)
        return {"tipo": "top3", "origem": "3 relações mais frequentes no treino.",
                "relacoes": rels, "n_relacoes": len(rels)}
    if cond in ("C5_vizinho", "C5_incongruente"):
        papel = "vizinho" if cond == "C5_vizinho" else "incongruente"
        p = parceira_c5(dataset, onto, papel)
        if not p:
            return {"tipo": "parceira", "origem": "Parceira C5 (ausente).", "n_relacoes": 0}
        return {"tipo": "parceira",
                "origem": f"Ontologia da parceira {p['parceira_id']} "
                          f"({papel}, SimE={p['sim_emb']}).",
                "relacoes": p["relacoes"], "n_relacoes": p["n_relacoes"],
                "parceira_id": p["parceira_id"], "sim_emb": p["sim_emb"], "papel": papel}
    return {"tipo": "desconhecida", "origem": cond}
