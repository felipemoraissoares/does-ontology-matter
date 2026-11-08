# Carrega e formata ontologias para cada condição experimental.
#
# Condições:
#   C0  — sem ontologia
#   C1  — conceitos + relações com domínio/range (zero-shot, sem exemplo)
#   C1_oneshot_original — igual C1, mas com exemplo (fornecido por dados.py)
#   C3  — top-k relações mais frequentes no treino, com domínio/range
#   C5  — JSON bruto da ontologia CORRETA
#   C5_vizinho      — JSON bruto de ontologia de domínio vizinho (perturbação nível 1)
#   C5_incongruente — JSON bruto de ontologia de domínio incompatível (perturbação nível 2)

import json
import os
from typing import Dict, List, Optional

CONDICOES = {"C0", "C1", "C1_oneshot_original", "C3", "C5", "C5_vizinho", "C5_incongruente"}


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
    O benchmark usa os *labels* dos conceitos (ex.: 'film', 'human'), não os QIDs.
    Para TekGen o mapeamento é QID→label; para DBpedia domain/range já são labels.
    """
    id_para_label = {c["qid"]: c["label"] for c in ontologia.get("concepts", [])}

    partes = []
    for r in ontologia["relations"]:
        label = r["label"].replace(" ", "_")
        dom_raw = r.get("domain", "")
        rng_raw = r.get("range", "")
        # Converte QID para label se existir no mapeamento; caso contrário mantém como está
        dom = id_para_label.get(dom_raw, dom_raw)
        rng = id_para_label.get(rng_raw, rng_raw)
        partes.append(f"{label}({dom},{rng})")
    return ", ".join(partes)


def _formatar_conceitos(ontologia: Dict) -> str:
    return ", ".join(c["label"] for c in ontologia["concepts"]) + ","


def formatar_contexto(
    ontologia: Dict,
    condicao: str,
    top_k_rels: Optional[List[str]] = None,
    ontologia_parceira: Optional[Dict] = None,
) -> Optional[str]:
    """
    Retorna o bloco de contexto de ontologia para inserir no prompt.
    Retorna None para C0 (sem ontologia).

    Parâmetros
    ----------
    ontologia : Dict
        Ontologia alvo (sempre a correta para a sentença).
    condicao : str
        C0 | C1 | C1_oneshot_original | C3 | C5 | C5_vizinho | C5_incongruente
    top_k_rels : list[str] | None
        Para C3: labels das k relações mais frequentes no treino (com _ no lugar de espaço).
        Se None, usa todas as relações (comportamento legado).
    ontologia_parceira : Dict | None
        Para C5_vizinho / C5_incongruente: ontologia a fornecer ao modelo no lugar da correta.
        Para C5: ignorado (usa a ontologia correta).
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
        if top_k_rels is not None:
            # C3 k=3: apenas as top_k relações com domínio/range
            id_para_label = {c.get("qid", c["label"]): c["label"] for c in ontologia.get("concepts", [])}
            partes = []
            for r in ontologia["relations"]:
                label_norm = r["label"].replace(" ", "_")
                if label_norm in top_k_rels:
                    dom = id_para_label.get(r.get("domain", ""), r.get("domain", ""))
                    rng = id_para_label.get(r.get("range", ""), r.get("range", ""))
                    partes.append(f"{label_norm}({dom},{rng})")
            return f"Relations: {', '.join(partes)}"
        # Legado: todas as relações sem domínio/range
        labels = ", ".join(r["label"].replace(" ", "_") for r in ontologia["relations"])
        return f"Relations: {labels}"

    # C5* — JSON bruto; C5_vizinho/C5_incongruente usam a ontologia parceira
    ont_json = ontologia_parceira if ontologia_parceira is not None else ontologia
    return json.dumps(ont_json, ensure_ascii=False)
