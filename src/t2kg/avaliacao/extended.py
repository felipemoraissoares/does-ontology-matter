"""
Avaliador estendido — métricas adicionais além do benchmark original.

Métricas implementadas (pré-registradas em Fase 2 / item B):

1. P/R/F1 sem filtro ontológico
   O avaliador original filtra as triplas do sistema mantendo apenas as cujas relações
   aparecem no gabarito daquela frase (rule #1 do espelho). O modo sem filtro avalia
   TODAS as triplas extraídas, penalizando relações inventadas via precisão.

2. DBpedia sem duplicata
   O benchmark original inclui "1_university" duas vezes na onto_list de Vicuna-13B/WebNLG
   (ver regra #8 do original.py). Nesta função, o ID é deduplicado antes de calcular
   a média global, produzindo médias sobre 19 ontologias distintas.

3. Projeção de relações (C0/C2/C3)
   Mapeia cada relação livre extraída para a relação MAIS similar da ontologia de
   referência via embedding (all-MiniLM-L6-v2). Aceita o mapeamento apenas se
   SimE >= threshold_projecao. As triplas projetadas são então avaliadas com P/R/F1.
   Threshold provisório: 0.80 (a calibrar no piloto, ver D-16).

4. Redundância de esquema
   Fração de relações únicas sobre total de relações extraídas por ontologia.
   Alta redundância = modelo repete poucas relações; baixa = vocabulário variado.

5. Custo de tokens de contexto
   Conta palavras no bloco de ontologia do prompt (mesmo proxy de num_tokens_texto).
"""
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from t2kg.avaliacao.original import (
    normalizar_tripla,
    calcular_precisao_recall_f1,
    limpar_entidade,
)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
    _EMBS_OK = True
except ImportError:
    _EMBS_OK = False


THRESHOLD_PROJECAO_PROVISORIO = 0.80
_MODELO_EMBS = "all-MiniLM-L6-v2"
_modelo_cache: Optional[object] = None


def _get_modelo() -> object:
    global _modelo_cache
    if _modelo_cache is None:
        if not _EMBS_OK:
            raise ImportError("sentence-transformers necessário para projeção.")
        _modelo_cache = SentenceTransformer(_MODELO_EMBS)
    return _modelo_cache


# ---------------------------------------------------------------------------
# 1. P/R/F1 sem filtro
# ---------------------------------------------------------------------------

def prf1_sem_filtro(
    gabarito_triplas: List,
    sistema_triplas: List,
) -> Tuple[float, float, float]:
    """
    P/R/F1 sobre TODAS as triplas do sistema, sem filtrar por relações do gabarito.
    Normalização idêntica ao avaliador original.
    """
    gold = {normalizar_tripla(*t) for t in gabarito_triplas}
    pred = {normalizar_tripla(*t) for t in sistema_triplas}
    return calcular_precisao_recall_f1(gold, pred)


# ---------------------------------------------------------------------------
# 2. Projeção de relações
# ---------------------------------------------------------------------------

def projetar_relacoes(
    triplas: List,
    ontologia_ref: Dict,
    threshold: float = THRESHOLD_PROJECAO_PROVISORIO,
) -> Tuple[List, float]:
    """
    Mapeia relações livres nas triplas para as relações da ontologia de referência.

    Para cada relação extraída, computa SimE com todos os labels da ontologia e aceita
    o mapeamento se SimE >= threshold. Triplas sem mapeamento são descartadas.

    Retorna: (triplas_projetadas, fracao_projetada)
    """
    if not triplas:
        return [], 0.0

    modelo = _get_modelo()
    rels_ref = [r["label"].replace("_", " ").lower() for r in ontologia_ref["relations"]]
    rels_ref_orig = [r["label"] for r in ontologia_ref["relations"]]

    if not rels_ref:
        return [], 0.0

    rels_sistema = list({t[1] for t in triplas})
    rels_sistema_norm = [r.replace("_", " ").lower() for r in rels_sistema]

    embs_ref = modelo.encode(rels_ref, show_progress_bar=False)
    embs_sis = modelo.encode(rels_sistema_norm, show_progress_bar=False)
    sim = _cos_sim(embs_sis, embs_ref)

    mapeamento = {}
    for i, rel_livre in enumerate(rels_sistema):
        idx_max = int(np.argmax(sim[i]))
        if sim[i, idx_max] >= threshold:
            mapeamento[rel_livre] = rels_ref_orig[idx_max]

    projetadas = []
    for sub, rel, obj in triplas:
        if rel in mapeamento:
            projetadas.append([sub, mapeamento[rel], obj])

    fracao = len(projetadas) / len(triplas) if triplas else 0.0
    return projetadas, fracao


def prf1_projetado(
    gabarito_triplas: List,
    sistema_triplas: List,
    ontologia_ref: Dict,
    threshold: float = THRESHOLD_PROJECAO_PROVISORIO,
) -> Dict:
    """
    Projeta relações do sistema na ontologia de referência e calcula P/R/F1.
    Retorna dict com p, r, f1, fracao_projetada, n_projetadas.
    """
    projetadas, fracao = projetar_relacoes(sistema_triplas, ontologia_ref, threshold)
    gold = {normalizar_tripla(*t) for t in gabarito_triplas}
    pred = {normalizar_tripla(*t) for t in projetadas}
    p, r, f1 = calcular_precisao_recall_f1(gold, pred)
    return {
        "precision": p, "recall": r, "f1": f1,
        "fracao_projetada": fracao,
        "n_projetadas": len(projetadas),
        "n_sistema": len(sistema_triplas),
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# 3. Redundância de esquema
# ---------------------------------------------------------------------------

def redundancia_esquema(triplas: List) -> float:
    """
    Fração de relações únicas sobre total de relações extraídas.
    0.0 = todas as relações são iguais; 1.0 = todas são distintas.
    Alta redundância (valor baixo) indica que o modelo usa poucos predicados.
    """
    if not triplas:
        return 0.0
    total = len(triplas)
    unicas = len({t[1] for t in triplas})
    return unicas / total


# ---------------------------------------------------------------------------
# 4. Avaliação estendida por frase
# ---------------------------------------------------------------------------

def avaliar_frase_estendida(
    frase: Dict,
    triplas_sistema: List,
    ontologia_ref: Dict,
    threshold_proj: float = THRESHOLD_PROJECAO_PROVISORIO,
) -> Dict:
    """
    Calcula métricas estendidas para uma única frase.

    frase: dict com chaves 'id', 'sent', 'gt_triples'
    triplas_sistema: triplas extraídas pelo modelo
    """
    gabarito = frase.get("gt_triples", [])

    p_sf, r_sf, f1_sf = prf1_sem_filtro(gabarito, triplas_sistema)
    proj = prf1_projetado(gabarito, triplas_sistema, ontologia_ref, threshold_proj)
    red = redundancia_esquema(triplas_sistema)

    return {
        "id": frase["id"],
        "prf1_sem_filtro": {"precision": p_sf, "recall": r_sf, "f1": f1_sf},
        "prf1_projetado": proj,
        "redundancia_esquema": red,
        "n_triplas_sistema": len(triplas_sistema),
        "n_triplas_gabarito": len(gabarito),
    }


# ---------------------------------------------------------------------------
# 5. Deduplicar onto_list DBpedia (remove segunda ocorrência de 1_university)
# ---------------------------------------------------------------------------

def deduplicar_onto_list(onto_list: List[Dict]) -> List[Dict]:
    """
    Remove entradas duplicadas (mesmo 'id') mantendo a primeira ocorrência.
    Aplica-se ao WebNLG onde 1_university aparece duas vezes.
    """
    vistos = set()
    resultado = []
    for item in onto_list:
        if item["id"] not in vistos:
            vistos.add(item["id"])
            resultado.append(item)
    return resultado


# ---------------------------------------------------------------------------
# 6. Sumarização de métricas estendidas por ontologia
# ---------------------------------------------------------------------------

def sumarizar_estendido(
    metricas_por_frase: List[Dict],
    n_total_frases: int,
    modo_amostra: bool = False,
) -> Dict:
    """
    Agrega métricas estendidas por frase em médias por ontologia.

    n_total_frases: total de frases do gabarito (denominador em modo benchmark)
    modo_amostra: se True, usa len(metricas_por_frase) como denominador
    """
    denom = len(metricas_por_frase) if modo_amostra else n_total_frases
    if denom == 0:
        denom = 1

    soma_prf_sf = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    soma_proj = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "fracao_projetada": 0.0}
    soma_red = 0.0

    for m in metricas_por_frase:
        for k in soma_prf_sf:
            soma_prf_sf[k] += m["prf1_sem_filtro"][k]
        for k in soma_proj:
            soma_proj[k] += m["prf1_projetado"][k]
        soma_red += m["redundancia_esquema"]

    return {
        "avg_sem_filtro_precision": soma_prf_sf["precision"] / denom,
        "avg_sem_filtro_recall": soma_prf_sf["recall"] / denom,
        "avg_sem_filtro_f1": soma_prf_sf["f1"] / denom,
        "avg_projetado_precision": soma_proj["precision"] / denom,
        "avg_projetado_recall": soma_proj["recall"] / denom,
        "avg_projetado_f1": soma_proj["f1"] / denom,
        "avg_fracao_projetada": soma_proj["fracao_projetada"] / denom,
        "avg_redundancia_esquema": soma_red / denom,
        "n_frases_avaliadas": len(metricas_por_frase),
        "n_frases_total": n_total_frases,
    }
