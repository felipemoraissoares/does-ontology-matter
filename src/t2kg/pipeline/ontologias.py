# Carrega e formata ontologias para cada condição experimental.
#
# Condições:
#   C0  — sem ontologia
#   C1  — conceitos + relações com domínio/range (zero-shot, sem exemplo)
#   C1_oneshot_original — igual C1, mas com exemplo (fornecido por dados.py)
#   C2  — ontologia induzida a partir de extração livre no treino (D-16)
#   C3  — top-k relações mais frequentes no treino, com domínio/range
#   C5  — ontologia correta em formato texto (igual C1)
#   C5_vizinho      — ontologia parceira vizinha em formato texto (perturbação nível 1)
#   C5_incongruente — ontologia parceira incongruente em formato texto (perturbação nível 2)

import json
import os
from typing import Dict, List, Optional

CONDICOES = {"C0", "C1", "C1_oneshot_original", "C2", "C3", "C5", "C5_vizinho", "C5_incongruente"}

# Diretório raiz do repo (dois níveis acima deste arquivo)
_REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


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


def carregar_ontologia_induzida(dataset: str, ontologia_id: str) -> Dict:
    """
    Carrega a ontologia induzida pela condição C2.
    Espera o arquivo em configs/c2/{dataset}_{onto}_induced.json (gerado por induzir_c2.py).
    Lança FileNotFoundError com mensagem clara se o arquivo não existir.
    """
    caminho = os.path.join(_REPO_DIR, "configs", "c2", f"{dataset}_{ontologia_id}_induced.json")
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Ontologia induzida C2 não encontrada: {caminho}\n"
            f"Execute: python scripts/induzir_c2.py --dataset {dataset} --onto {ontologia_id}"
        )
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


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
        C0 | C1 | C1_oneshot_original | C2 | C3 | C5 | C5_vizinho | C5_incongruente
    top_k_rels : list[str] | None
        Para C3: labels das k relações mais frequentes no treino (com _ no lugar de espaço).
        Se None, usa todas as relações (comportamento legado).
    ontologia_parceira : Dict | None
        Para C2: ontologia induzida (pré-carregada). Para C5_vizinho/C5_incongruente:
        ontologia parceira. Para C5: ignorado.
    """
    if condicao not in CONDICOES:
        raise ValueError(f"Condição desconhecida: {condicao!r}. Use: {CONDICOES}")

    if condicao == "C0":
        return None

    if condicao == "C2":
        # Usa ontologia induzida fornecida via ontologia_parceira.
        # Apenas relações são definidas (sem conceitos nem domain/range).
        if ontologia_parceira is None:
            raise ValueError("C2 requer ontologia_parceira com a ontologia induzida (vide D-16).")
        labels = ", ".join(r["label"].replace(" ", "_") for r in ontologia_parceira.get("relations", []))
        return f"Ontology Relations: {labels}"

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

    # C5* — mesmo formato texto de C1, mudando apenas o conteúdo da ontologia.
    # C5_vizinho/C5_incongruente usam a ontologia parceira; C5 usa a ontologia alvo.
    # Manter o mesmo formato isola o efeito do CONTEÚDO vs. o efeito de FORMAT.
    ont_to_use = ontologia_parceira if ontologia_parceira is not None else ontologia
    conceitos = _formatar_conceitos(ont_to_use)
    relacoes = _formatar_relacoes_c1(ont_to_use)
    return (
        f"Ontology Concepts: {conceitos}\n"
        f"Ontology Relations: {relacoes}"
    )
