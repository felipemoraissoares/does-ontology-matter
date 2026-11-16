"""Constantes e utilidades base da camada de dados (sem Streamlit).

Caminhos do repositório e do Text2KGBench, lista de condições, descoberta dinâmica das
ontologias (a partir do ground_truth) e separação do bloco CONTEXT do prompt.
"""
import os
import re
import sys
from functools import lru_cache
from typing import Dict, List, Optional

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(APP_DIR, ".."))
WORKSPACE = os.path.abspath(os.path.join(REPO, ".."))
BENCH = os.path.join(WORKSPACE, "Text2KGBench")
RESULTADOS = os.path.join(REPO, "resultados")
CONFIGS = os.path.join(REPO, "configs")
DOCS = os.path.join(REPO, "docs")
CACHE_DIR = os.path.join(APP_DIR, ".cache")

# Reaproveita src/t2kg (não reimplementa avaliação/prompt no app)
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))

CONDICOES = ["C0", "C1", "C1_oneshot_original", "C2", "C3", "C5_vizinho", "C5_incongruente"]
COND_ROTULO = {
    "C0": "C0 · sem ontologia",
    "C1": "C1 · ontologia",
    "C1_oneshot_original": "C1 one-shot",
    "C2": "C2 · induzida",
    "C3": "C3 · top-3",
    "C5_vizinho": "C5 vizinho",
    "C5_incongruente": "C5 incongruente",
}
COND_CURTO = {
    "C0": "C0", "C1": "C1", "C1_oneshot_original": "C1+ex", "C2": "C2",
    "C3": "C3", "C5_vizinho": "C5v", "C5_incongruente": "C5i",
}
CONDICOES_PROJETAVEIS = {"C0", "C2", "C3"}

DATASETS = ("tekgen", "webnlg")
DS_PREFIXO = {"tekgen": "tkg", "webnlg": "web"}
DS_SUBDIR = {"tekgen": "wikidata_tekgen", "webnlg": "dbpedia_webnlg"}
DS_ROTULO = {"tekgen": "TekGen (Wikidata)", "webnlg": "DBpedia (WebNLG)"}

# Linha de referência do artigo (Vicuna-13B), para marcar nos gráficos.
BASELINE_ARTIGO = {"tekgen": 0.35, "webnlg": 0.30}


def _gt_dir(dataset: str) -> str:
    return os.path.join(BENCH, "data", DS_SUBDIR[dataset], "ground_truth")


@lru_cache(maxsize=None)
def ontologias(dataset: str) -> List[str]:
    """Ontologias do dataset, descobertas dos arquivos de ground_truth (ordem por índice)."""
    d = _gt_dir(dataset)
    if not os.path.isdir(d):
        return []
    ontos = []
    for nome in os.listdir(d):
        m = re.match(r"ont_(.+)_ground_truth\.jsonl$", nome)
        if m:
            ontos.append(m.group(1))
    return sorted(ontos, key=lambda o: int(o.split("_")[0]) if o.split("_")[0].isdigit() else 999)


@lru_cache(maxsize=None)
def dataset_da_ontologia(onto: str) -> Optional[str]:
    for ds in DATASETS:
        if onto in ontologias(ds):
            return ds
    return None


def caminho_gt(dataset: str, onto: str) -> str:
    return os.path.join(_gt_dir(dataset), f"ont_{onto}_ground_truth.jsonl")


def caminho_ontologia_ref(dataset: str, onto: str) -> str:
    return os.path.join(BENCH, "data", DS_SUBDIR[dataset], "ontologies", f"{onto}_ontology.json")


# ---------------------------------------------------------------------------
# Separação do bloco CONTEXT do prompt (para destacar na interface)
# ---------------------------------------------------------------------------

def separar_prompt(prompt: str) -> Dict[str, str]:
    """Divide o prompt em: instrucao (antes de CONTEXT), contexto (bloco CONTEXT:) e
    teste (a partir de 'Test Sentence:'). Campos ausentes vêm como ''.
    """
    if not prompt:
        return {"instrucao": "", "contexto": "", "teste": ""}
    contexto, instrucao, teste = "", prompt, ""
    m_ctx = re.search(r"\nCONTEXT:\n", prompt)
    m_test = re.search(r"\nTest Sentence:", prompt)
    if m_test:
        teste = prompt[m_test.start() + 1:]
    if m_ctx:
        instrucao = prompt[:m_ctx.start()]
        fim_ctx = m_test.start() if m_test else len(prompt)
        contexto = prompt[m_ctx.end():fim_ctx].strip()
    elif m_test:
        instrucao = prompt[:m_test.start()]
    return {"instrucao": instrucao.strip(), "contexto": contexto, "teste": teste.strip()}
