"""
D-24: Gera definições semânticas de 1 frase para rótulos de relação.

Para cada rótulo único (das extrações livres de treino + da ontologia de referência),
o LLM gera uma definição curta. O embedding "rótulo: definição" melhora o agrupamento
e a projeção C2 em relação ao embedding só de rótulo (Zhang & Soh 2024, estilo EDC).

Uso:
  # Gerar definições (requer vLLM rodando):
  python scripts/gerar_definicoes_relacoes.py --onto_id 1_movie --dataset tekgen

  # Apenas ler cache existente:
  python scripts/gerar_definicoes_relacoes.py --onto_id 1_movie --dataset tekgen --from_cache

  # A partir de JSONL de extração livre:
  python scripts/gerar_definicoes_relacoes.py --onto_id 1_movie --dataset tekgen \\
      --source_jsonl resultados/c2_inducao/raw_tkg_1_movie_treino.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from t2kg.parser import parsear_resposta  # noqa: E402

BENCH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Text2KGBench"))
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PROMPT_DEFINICAO = (
    "Define the semantic relationship '{label}' in one concise sentence. "
    "Focus on what it means when connecting two entities. "
    "Reply with only the definition, no preamble."
)


def _carregar_relacoes_jsonl(source_jsonl: str):
    """Extrai rótulos únicos de um JSONL de extração livre."""
    rotulos = set()
    with open(source_jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for t in r.get("triples", []):
                if len(t) >= 2:
                    rotulos.add(t[1].strip().lower())
    return rotulos


def _carregar_relacoes_ontologia(dataset: str, onto_id: str):
    """Extrai rótulos da ontologia de referência."""
    if dataset == "tekgen":
        p = os.path.join(BENCH, "data", "wikidata_tekgen", "ontologies", f"{onto_id}_ontology.json")
    else:
        p = os.path.join(BENCH, "data", "dbpedia_webnlg", "ontologies", f"{onto_id}_ontology.json")
    with open(p, encoding="utf-8") as f:
        onto = json.load(f)
    return {r["label"].lower().replace(" ", "_") for r in onto.get("relations", [])}


def _caminho_cache(onto_id: str):
    return os.path.join(REPO, "configs", "c2", f"definicoes_{onto_id}.json")


def _carregar_cache(onto_id: str):
    p = _caminho_cache(onto_id)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {"modelo": None, "relacoes": {}}


def _salvar_cache(onto_id: str, cache: dict):
    p = _caminho_cache(onto_id)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def gerar_definicoes(rotulos, cache, provedor, modelo_nome):
    """Gera definições para rótulos ainda não em cache."""
    pendentes = [r for r in rotulos if r not in cache["relacoes"]]
    if not pendentes:
        print(f"  Todos {len(rotulos)} rótulos já têm definição em cache.")
        return cache

    print(f"  Gerando {len(pendentes)} definições (de {len(rotulos)} total)...")
    cache["modelo"] = modelo_nome
    for i, rotulo in enumerate(pendentes):
        prompt = PROMPT_DEFINICAO.format(label=rotulo)
        resp = provedor.gerar(prompt, max_tokens=128, temperatura=0.0).strip()
        cache["relacoes"][rotulo] = resp
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(pendentes)}: {rotulo!r} → {resp[:60]}...")

    return cache


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--onto_id", required=True)
    p.add_argument("--dataset", required=True, choices=["tekgen", "webnlg"])
    p.add_argument("--source_jsonl", default=None,
                   help="JSONL de extração livre (raw_treino). Se omitido, usa só a ontologia ref.")
    p.add_argument("--url", default="http://localhost:8000/v1")
    p.add_argument("--modelo_llm", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    p.add_argument("--from_cache", action="store_true",
                   help="Apenas lê e exibe o cache existente, sem chamar o LLM.")
    args = p.parse_args()

    cache = _carregar_cache(args.onto_id)

    # Coleta rótulos
    rotulos = _carregar_relacoes_ontologia(args.dataset, args.onto_id)
    print(f"Rótulos da ontologia de referência: {len(rotulos)}")
    if args.source_jsonl and os.path.exists(args.source_jsonl):
        livres = _carregar_relacoes_jsonl(args.source_jsonl)
        print(f"Rótulos de extração livre: {len(livres)}")
        rotulos = rotulos | livres

    print(f"Total a definir: {len(rotulos)} rótulos únicos")
    print(f"Já em cache: {len(cache['relacoes'])}")

    if args.from_cache:
        print("\nModo --from_cache: exibindo definições existentes.")
        for rotulo, defn in sorted(cache["relacoes"].items())[:20]:
            print(f"  {rotulo!r}: {defn[:80]}")
        if len(cache["relacoes"]) > 20:
            print(f"  ... (+{len(cache['relacoes'])-20} mais)")
    else:
        from t2kg.pipeline.provedores.openai_compat import ProvedorOpenAICompat
        provedor = ProvedorOpenAICompat(url=args.url, api_key="token-nao-usado", modelo=args.modelo_llm)
        cache = gerar_definicoes(rotulos, cache, provedor, args.modelo_llm)
        _salvar_cache(args.onto_id, cache)
        print(f"\nDefinições salvas em: {_caminho_cache(args.onto_id)}")
        print(f"Total de definições: {len(cache['relacoes'])}")
