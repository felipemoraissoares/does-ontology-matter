"""Classificação de triplas vs. gabarito, reaproveitando src/t2kg/avaliacao/original.

Classes:
  acerto          — tripla normalizada igual a uma do gabarito (normalizar_tripla)
  erro_formato    — entidade é tipo/placeholder (calcular_taxa_erro_formato)
  erro            — a relação está no gabarito, mas a tripla não bate
  fora_do_gabarito— a relação não aparece no gabarito desta frase (o avaliador descartaria)
E `faltas()` devolve as triplas do gabarito não cobertas por nenhuma tripla do sistema.
"""
from typing import Dict, List, Optional

from t2kg.avaliacao.original import normalizar_tripla, calcular_taxa_erro_formato

CLASSES = ("acerto", "fora_do_gabarito", "erro", "erro_formato")


def _triplas_lista(triplas) -> List[List[str]]:
    """Normaliza para lista [sub, rel, obj] (aceita dicts {sub,rel,obj} ou listas)."""
    out = []
    for t in triplas:
        if isinstance(t, dict):
            out.append([t.get("sub", ""), t.get("rel", ""), t.get("obj", "")])
        elif len(t) >= 3:
            out.append([t[0], t[1], t[2]])
    return out


def rels_gabarito(gt_triplas) -> set:
    return {t[1].replace(" ", "_") for t in _triplas_lista(gt_triplas)}


def _norm_set(triplas) -> set:
    return {normalizar_tripla(t[0], t[1], t[2]) for t in _triplas_lista(triplas)}


def classificar_tripla(tr, gt_norm: set, gt_rels: set,
                       ontologia: Optional[Dict]) -> str:
    if len(tr) < 3:
        return "erro_formato"
    if normalizar_tripla(tr[0], tr[1], tr[2]) in gt_norm:
        return "acerto"
    if ontologia is not None and calcular_taxa_erro_formato(ontologia, [tr]) > 0:
        return "erro_formato"
    if tr[1] in gt_rels or tr[1].replace(" ", "_") in gt_rels:
        return "erro"
    return "fora_do_gabarito"


def classificar(triplas, gt_triplas, ontologia: Optional[Dict] = None) -> List[Dict]:
    """Lista de {tripla:[s,r,o], classe} para cada tripla do sistema."""
    gt = _triplas_lista(gt_triplas)
    gt_norm = _norm_set(gt)
    gt_rels = rels_gabarito(gt)
    return [{"tripla": [t[0], t[1], t[2]], "classe": classificar_tripla(t, gt_norm, gt_rels, ontologia)}
            for t in _triplas_lista(triplas)]


def faltas(gt_triplas, triplas_sistema) -> List[List[str]]:
    sis = _norm_set(triplas_sistema)
    return [t for t in _triplas_lista(gt_triplas)
            if normalizar_tripla(t[0], t[1], t[2]) not in sis]


def contagem_classes(classificadas: List[Dict]) -> Dict[str, int]:
    c = {k: 0 for k in CLASSES}
    for x in classificadas:
        c[x["classe"]] = c.get(x["classe"], 0) + 1
    return c
