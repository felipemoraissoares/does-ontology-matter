# Carrega e formata ontologias para cada condição experimental.
#
# Condições:
#   C0  — sem ontologia
#   C1  — conceitos + relações com domínio/range (zero-shot, sem exemplo)
#   C1_oneshot_original — igual C1, mas com exemplo (fornecido por dados.py)
#   C3  — só labels das relações, sem domínio/range e sem conceitos
#   C5  — JSON bruto da ontologia

import json
import os
from typing import Dict, Optional

CONDICOES = {"C0", "C1", "C1_oneshot_original", "C3", "C5"}


def _caminhos_ontologia(bench_dir: str, dataset: str, ontologia_id: str) -> str:
    if dataset == "tekgen":
        return os.path.join(
            bench_dir, "data", "wikidata_tekgen", "ontologies",
            f"{ontologia_id}_ontology.json",
        )
    return os.path.join(
        bench_dir, "data", "dbpedia_webnlg", "ontologies",
        f"{ontologia_id}_ontology.json",
    )


def carregar_ontologia(bench_dir: str, dataset: str, ontologia_id: str) -> Dict:
    caminho = _caminhos_ontologia(bench_dir, dataset, ontologia_id)
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def _formatar_relacoes_c1(ontologia: Dict) -> str:
    """
    Formato do benchmark: rel_label(domain_label,range_label), ...
    Espaços nos labels são preservados (o benchmark usa espaços no nome dos conceitos).
    """
    partes = []
    for r in ontologia["relations"]:
        label = r["label"].replace(" ", "_")
        dom = r.get("domain", "")
        rng = r.get("range", "")
        partes.append(f"{label}({dom},{rng})")
    return ", ".join(partes)


def _formatar_conceitos(ontologia: Dict) -> str:
    return ", ".join(c["label"] for c in ontologia["concepts"]) + ","


def formatar_contexto(ontologia: Dict, condicao: str) -> Optional[str]:
    """
    Retorna o bloco de contexto de ontologia para inserir no prompt.
    Retorna None para C0 (sem ontologia).
    """
    if condicao not in CONDICOES:
        raise ValueError(f"Condição desconhecida: {condicao!r}. Use: {CONDICOES}")

    if condicao == "C0":
        return None

    if condicao in ("C1", "C1_oneshot_original"):
        conceitos = _formatar_conceitos(ontologia)
        relacoes = _formatar_relacoes_c1(ontologia)
        return (
            f"Ontology Concepts: {conceitos}\n"
            f"Ontology Relations: {relacoes}"
        )

    if condicao == "C3":
        labels = ", ".join(
            r["label"].replace(" ", "_") for r in ontologia["relations"]
        )
        return f"Relations: {labels}"

    # C5 — JSON bruto
    return json.dumps(ontologia, ensure_ascii=False)
