"""Lógica de dados do app Streamlit (pura, sem Streamlit — testável).

Lê os arquivos de avaliação do piloto-v3 em resultados/avaliacao_piloto_v3/ e classifica
as triplas de cada condição vs. o gabarito: acerto / sobra / falta / erro de formato.
"""
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from t2kg.avaliacao.original import normalizar_tripla, calcular_taxa_erro_formato

AVAL_DIR = os.path.join(REPO, "resultados", "avaliacao_piloto_v3")
CONDICOES = ["C0", "C1", "C1_oneshot_original", "C2", "C3", "C5_vizinho", "C5_incongruente"]
COND_ROTULO = {
    "C0": "C0 (sem ontologia)", "C1": "C1 (ontologia)",
    "C1_oneshot_original": "C1+ex (one-shot)", "C2": "C2 (induzida)",
    "C3": "C3 (top-3)", "C5_vizinho": "C5 vizinho", "C5_incongruente": "C5 incongruente",
}
CONDICOES_PROJETAVEIS = {"C0", "C2", "C3"}


def caminho_condicao(dataset: str, cond: str, onto: str, aval_dir: str = AVAL_DIR) -> str:
    return os.path.join(aval_dir, f"{dataset}_{cond}_{onto}.jsonl")


def carregar_condicao(dataset: str, cond: str, onto: str,
                      aval_dir: str = AVAL_DIR) -> Dict[str, Dict]:
    """Retorna {sent_id: registro_de_avaliacao} ou {} se o arquivo não existe."""
    p = caminho_condicao(dataset, cond, onto, aval_dir)
    if not os.path.exists(p):
        return {}
    registros = {}
    with open(p, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            r = json.loads(linha)
            if "id" in r:
                registros[r["id"]] = r
    return registros


def listar_ontologias(dataset: str, aval_dir: str = AVAL_DIR) -> List[str]:
    """Ontologias com arquivos de avaliação para o dataset (ordenadas por índice)."""
    ontos = set()
    for p in glob.glob(os.path.join(aval_dir, f"{dataset}_C1_*.jsonl")):
        nome = os.path.basename(p)[len(f"{dataset}_C1_"):-len(".jsonl")]
        if not nome.startswith("oneshot"):
            ontos.add(nome)
    return sorted(ontos, key=lambda o: int(o.split("_")[0]) if o.split("_")[0].isdigit() else 999)


def condicoes_disponiveis(dataset: str, onto: str, aval_dir: str = AVAL_DIR) -> List[str]:
    return [c for c in CONDICOES
            if os.path.exists(caminho_condicao(dataset, c, onto, aval_dir))]


def classificar_sistema(
    llm_triples: List, gt_norm: set, ontologia: Optional[Dict] = None
) -> List[Tuple[List, str]]:
    """Classifica cada tripla do sistema: 'acerto' | 'fmt' | 'sobra'.

    'fmt' (erro de formato) só é detectado se a ontologia for fornecida.
    """
    out = []
    for tr in llm_triples:
        if len(tr) < 3:
            continue
        if normalizar_tripla(tr[0], tr[1], tr[2]) in gt_norm:
            status = "acerto"
        elif ontologia is not None and calcular_taxa_erro_formato(ontologia, [tr]) > 0:
            status = "fmt"
        else:
            status = "sobra"
        out.append((tr, status))
    return out


def faltas(gt_triples: List, sys_norm: set) -> List[List]:
    """Triplas do gabarito não cobertas por nenhuma tripla do sistema."""
    return [t for t in gt_triples
            if normalizar_tripla(t[0], t[1], t[2]) not in sys_norm]


def gt_da_frase(registro: Dict) -> List[List]:
    """gt_triples como listas [sub, rel, obj]."""
    return [list(t) for t in registro.get("gt_triples", [])]


def carregar_tabela_metricas(aval_dir: str = AVAL_DIR) -> List[Dict]:
    """Lê tabela_piloto_v3.csv como lista de dicts (ou [] se ausente)."""
    p = os.path.join(aval_dir, "tabela_piloto_v3.csv")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))
