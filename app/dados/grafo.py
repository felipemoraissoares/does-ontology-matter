"""Triplas → arestas de grafo. Base dos grafos da interface e exportação reutilizável
por um futuro carregamento no Neo4j / retriever GraphRAG (página "Em breve").
"""
from typing import Dict, List, Optional

from .classificacao import classificar
from .descoberta import experimento
from .frases import _ont_ref, _por_id, gabarito


def arestas_frase(frase: Dict) -> List[Dict]:
    """Arestas de uma frase já carregada (saída de frases.frase): sistema + faltas."""
    out = []
    for item in frase.get("classificadas", []):
        s, r, o = item["tripla"]
        out.append({"origem": s, "rel": r, "destino": o, "classe": item["classe"]})
    for s, r, o in frase.get("faltas", []):
        out.append({"origem": s, "rel": r, "destino": o, "classe": "falta"})
    return out


def exportar_arestas(rid: str, dataset: str, onto: str, cond: str,
                     incluir_faltas: bool = False) -> List[Dict]:
    """Todas as arestas da rodada+ontologia+condição (reutilizável por Neo4j/GraphRAG).

    Cada aresta: {frase, origem, rel, destino, classe}.
    """
    exp = experimento(rid, dataset, onto, cond)
    if not exp:
        return []
    ont = _ont_ref(dataset, onto)
    gt = gabarito(dataset, onto)
    out = []
    for sid, rec in _por_id(exp["caminho"]).items():
        g = gt.get(sid, {"triplas": []})
        for item in classificar(rec.get("triples", []), g["triplas"], ont):
            s, r, o = item["tripla"]
            out.append({"frase": sid, "origem": s, "rel": r, "destino": o,
                        "classe": item["classe"]})
        if incluir_faltas:
            from .classificacao import faltas
            for s, r, o in faltas(g["triplas"], rec.get("triples", [])):
                out.append({"frase": sid, "origem": s, "rel": r, "destino": o,
                            "classe": "falta"})
    return out


def grafo_agregado(rid: str, dataset: str, onto: str, cond: str,
                   so_acertos: bool = False, incluir_gabarito: bool = False,
                   limite_nos: Optional[int] = None) -> Dict:
    """Nós e arestas agregados de uma rodada+ontologia+condição + painel resumo."""
    arestas = exportar_arestas(rid, dataset, onto, cond, incluir_faltas=incluir_gabarito)
    if so_acertos:
        arestas = [a for a in arestas if a["classe"] == "acerto"]
    rels = [a["rel"] for a in arestas]
    distintas = sorted(set(rels))
    from collections import Counter
    top = Counter(rels).most_common(8)
    redundancia = 1.0 - (len(distintas) / len(rels)) if rels else 0.0

    nos = []
    vistos = set()
    for a in arestas:
        for e in (a["origem"], a["destino"]):
            if e not in vistos:
                vistos.add(e)
                nos.append(e)
    if limite_nos and len(nos) > limite_nos:
        manter = set(nos[:limite_nos])
        arestas = [a for a in arestas if a["origem"] in manter and a["destino"] in manter]
        nos = [n for n in nos if n in manter]
    return {"nos": nos, "arestas": arestas,
            "n_relacoes_distintas": len(distintas), "redundancia": round(redundancia, 3),
            "top_relacoes": top}
