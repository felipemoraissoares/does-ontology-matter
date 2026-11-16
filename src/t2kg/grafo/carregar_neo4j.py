"""Carrega um JSONL de resultados no Neo4j (Fase 4).

Modelo de grafo:
  (:Entidade {nome})
  (:Frase {id, texto})
  (:Frase)-[:MENCIONA]->(:Entidade)
  (:Entidade)-[:REL {relacao, condicao, modelo, run, status, frase}]->(:Entidade)

status (conforme o avaliador): "acerto" se a tripla normalizada do sistema coincide com
uma tripla do gabarito da frase; "sobra" caso contrário.

Carga POR RUN (o AuraDB Free limita ~200k nós / ~400k relações). Use --apagar_run para
remover um run antes de recarregá-lo.

Credenciais SÓ do .env (NEO4J_URI, NEO4J_USER ou NEO4J_USERNAME, NEO4J_PASSWORD) — nunca no
código nem em commit.

Uso:
  python -m t2kg.grafo.carregar_neo4j --run pv3_1movie_c1
  python -m t2kg.grafo.carregar_neo4j --run pv3_1movie_c1 --apagar_run
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

from t2kg.avaliacao.original import normalizar_tripla
from t2kg.pipeline.dados import carregar_frases

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WORKSPACE = os.path.abspath(os.path.join(REPO, ".."))
BENCH_PADRAO = os.path.join(WORKSPACE, "Text2KGBench")
MANIFESTO_PADRAO = os.path.join(REPO, "configs", "pipeline", "manifesto_piloto_v3.json")

# Cypher (parametrizado via UNWIND — 4 operações por carga, eficiente para o limite do Aura)
_CY_FRASES = "UNWIND $rows AS row MERGE (f:Frase {id: row.id}) SET f.texto = row.texto"
_CY_ENTIDADES = "UNWIND $nomes AS nome MERGE (:Entidade {nome: nome})"
_CY_MENCIONA = (
    "UNWIND $rows AS row MATCH (f:Frase {id: row.fid}) "
    "MATCH (e:Entidade {nome: row.nome}) MERGE (f)-[:MENCIONA]->(e)"
)
_CY_REL = (
    "UNWIND $rows AS row MATCH (s:Entidade {nome: row.sub}) "
    "MATCH (o:Entidade {nome: row.obj}) "
    "CREATE (s)-[:REL {relacao: row.relacao, condicao: row.condicao, modelo: row.modelo, "
    "run: row.run, status: row.status, frase: row.frase}]->(o)"
)
_CY_APAGAR_RUN = "MATCH ()-[r:REL {run: $run}]->() DELETE r"


def carregar_gt_normalizado(bench_dir: str, dataset: str, ontologia: str,
                            split: str = "test") -> Dict[str, set]:
    """Retorna {sent_id: set(triplas_normalizadas do gabarito)}."""
    gt = {}
    for frase in carregar_frases(bench_dir, dataset, split, ontologia):
        gt[frase["id"]] = {
            normalizar_tripla(t["sub"], t["rel"], t["obj"])
            for t in frase.get("gt_triples", [])
        }
    return gt


def construir_operacoes(
    registros: List[Dict],
    meta: Dict,
    gt_norm: Dict[str, set],
    apagar_run: bool = False,
) -> List[Tuple[str, Dict]]:
    """Monta as operações Cypher (cypher, params). Função pura — testável sem Neo4j.

    meta: {"condicao", "modelo", "run"}.
    """
    run = meta["run"]
    condicao = meta["condicao"]
    modelo = meta["modelo"]

    frases_rows, menciona_rows, rel_rows = [], [], []
    nomes = set()
    for r in registros:
        fid = r["id"]
        frases_rows.append({"id": fid, "texto": r.get("sent", "")})
        gt_set = gt_norm.get(fid, set())
        ents_frase = set()
        for tr in r.get("triples", []):
            if len(tr) < 3:
                continue
            sub, rel, obj = tr[0], tr[1], tr[2]
            nomes.add(sub)
            nomes.add(obj)
            ents_frase.add(sub)
            ents_frase.add(obj)
            status = "acerto" if normalizar_tripla(sub, rel, obj) in gt_set else "sobra"
            rel_rows.append({
                "sub": sub, "obj": obj, "relacao": rel, "condicao": condicao,
                "modelo": modelo, "run": run, "status": status, "frase": fid,
            })
        for nome in ents_frase:
            menciona_rows.append({"fid": fid, "nome": nome})

    ops: List[Tuple[str, Dict]] = []
    if apagar_run:
        ops.append((_CY_APAGAR_RUN, {"run": run}))
    ops.append((_CY_FRASES, {"rows": frases_rows}))
    ops.append((_CY_ENTIDADES, {"nomes": sorted(nomes)}))
    ops.append((_CY_MENCIONA, {"rows": menciona_rows}))
    ops.append((_CY_REL, {"rows": rel_rows}))
    return ops


def executar_ops(ops: List[Tuple[str, Dict]], session) -> None:
    """Executa as operações numa sessão Neo4j (ou num fake que exponha .run)."""
    for cypher, params in ops:
        session.run(cypher, **params)


# ---------------------------------------------------------------------------
# Credenciais e resolução de experimento
# ---------------------------------------------------------------------------

def carregar_credenciais() -> Tuple[str, str, str]:
    """Lê NEO4J_* do ambiente e do .env (nunca do código). Erra se faltar algo."""
    vals = {k: os.environ[k] for k in
            ("NEO4J_URI", "NEO4J_USER", "NEO4J_USERNAME", "NEO4J_PASSWORD")
            if os.environ.get(k)}
    env_path = os.path.join(REPO, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for linha in f:
                linha = linha.strip()
                if not linha or linha.startswith("#") or "=" not in linha:
                    continue
                k, v = linha.split("=", 1)
                k, v = k.strip(), v.strip()
                if k.startswith("NEO4J_") and k not in vals:
                    vals[k] = v
    uri = vals.get("NEO4J_URI")
    user = vals.get("NEO4J_USER") or vals.get("NEO4J_USERNAME")
    pwd = vals.get("NEO4J_PASSWORD")
    if not (uri and user and pwd):
        raise RuntimeError(
            "Credenciais Neo4j ausentes. Defina NEO4J_URI, NEO4J_USER (ou NEO4J_USERNAME) "
            "e NEO4J_PASSWORD no arquivo .env (veja .env.example)."
        )
    return uri, user, pwd


def resolver_experimento(run: str, manifesto_path: str) -> Optional[Dict]:
    """Acha o experimento cujo id ou basename de saída corresponde ao run."""
    if not os.path.exists(manifesto_path):
        return None
    with open(manifesto_path, encoding="utf-8") as f:
        data = json.load(f)
    exps = data.get("experimentos", []) if isinstance(data, dict) else data
    for e in exps:
        if e.get("id") == run or os.path.basename(e.get("saida", "")) == f"{run}.jsonl":
            return e
    return None


def _resolver_jsonl(run: str, jsonl: Optional[str], exp: Optional[Dict]) -> Optional[str]:
    if jsonl:
        return jsonl
    if exp and os.path.exists(os.path.join(REPO, exp["saida"])):
        return os.path.join(REPO, exp["saida"])
    # Procura em resultados/
    for raiz, _, arqs in os.walk(os.path.join(REPO, "resultados")):
        if f"{run}.jsonl" in arqs:
            return os.path.join(raiz, f"{run}.jsonl")
    return None


def main():
    p = argparse.ArgumentParser(description="Carrega um run de resultados no Neo4j.")
    p.add_argument("--run", required=True, help="Id/basename do run, ex.: pv3_1movie_c1")
    p.add_argument("--jsonl", default=None, help="Caminho do JSONL (senão, resolvido pelo run).")
    p.add_argument("--manifesto", default=MANIFESTO_PADRAO)
    p.add_argument("--dataset", default=None)
    p.add_argument("--ontologia", default=None)
    p.add_argument("--condicao", default=None)
    p.add_argument("--modelo", default=None)
    p.add_argument("--bench_dir", default=BENCH_PADRAO)
    p.add_argument("--apagar_run", action="store_true",
                   help="Apaga as arestas REL deste run antes de recarregar.")
    args = p.parse_args()

    exp = resolver_experimento(args.run, args.manifesto)
    dataset = args.dataset or (exp or {}).get("dataset")
    ontologia = args.ontologia or (exp or {}).get("ontologia_id")
    condicao = args.condicao or (exp or {}).get("condicao")
    modelo = args.modelo or (exp or {}).get("modelo", "?")
    if not (dataset and ontologia and condicao):
        print("ERRO: não consegui resolver dataset/ontologia/condicao. "
              "Passe --dataset --ontologia --condicao (ou use um run do manifesto).",
              file=sys.stderr)
        sys.exit(1)

    jsonl = _resolver_jsonl(args.run, args.jsonl, exp)
    if not jsonl or not os.path.exists(jsonl):
        print(f"ERRO: JSONL do run '{args.run}' não encontrado.", file=sys.stderr)
        sys.exit(1)

    registros = [json.loads(l) for l in open(jsonl, encoding="utf-8") if l.strip()]
    gt_norm = carregar_gt_normalizado(args.bench_dir, dataset, ontologia)
    meta = {"condicao": condicao, "modelo": modelo, "run": args.run}
    ops = construir_operacoes(registros, meta, gt_norm, apagar_run=args.apagar_run)

    n_rel = len(ops[-1][1]["rows"])
    n_frases = len(registros)
    print(f"Run '{args.run}': {n_frases} frases, {n_rel} arestas REL "
          f"({dataset}/{ontologia}/{condicao}).")

    uri, user, pwd = carregar_credenciais()
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("ERRO: pacote 'neo4j' não instalado. pip install neo4j", file=sys.stderr)
        sys.exit(1)

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    try:
        with driver.session() as session:
            executar_ops(ops, session)
    finally:
        driver.close()
    print("Carga concluída no Neo4j.")


if __name__ == "__main__":
    main()
