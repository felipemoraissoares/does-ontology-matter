"""
Gera manifesto do piloto-v3 e mostra prompts reais de C3, C5_vizinho, C5_incongruente.

Piloto-v3: instrução Variante D, todas condições, 4 ontologias, 100 frases.
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH = os.path.abspath(os.path.join(REPO, "..", "Text2KGBench"))

from t2kg.pipeline.prompts import montar_prompt
from t2kg.pipeline.ontologias import formatar_contexto, carregar_ontologia
from t2kg.pipeline.dados import carregar_frequencias_treino

def _load_onto(bench, dataset, onto_id):
    return carregar_ontologia(bench, dataset, onto_id)

GIT_HASH = "cfa7e1a"
BENCH_COMMIT = "50a3d255371b8817cdff70fd88459ac82b339cfe"
MODELO = "Qwen/Qwen2.5-7B-Instruct-AWQ"
MODELO_QUANTIZACAO = "AWQ-4bit"
CHAT_TEMPLATE = "qwen2"
PROVEDOR_URL = "http://localhost:8000/v1"
FRASES_MAX = 100
SEED = 42


def onto_path_tekgen(onto_id):
    return os.path.join(BENCH, "data", "wikidata_tekgen", "ontologies", f"{onto_id}_ontology.json")


def onto_path_dbpedia(onto_id):
    return os.path.join(BENCH, "data", "dbpedia_webnlg", "ontologies", f"{onto_id}_ontology.json")


def gt_path_tekgen(onto_id):
    return os.path.join(BENCH, "data", "wikidata_tekgen", "ground_truth", f"ont_{onto_id}_ground_truth.jsonl")


def gt_path_dbpedia(onto_id):
    return os.path.join(BENCH, "data", "dbpedia_webnlg", "ground_truth", f"{onto_id}_ground_truth.jsonl")


def exemplo_oneshot_tekgen(onto_id):
    """Pega o primeiro item do GT como exemplo para C1_oneshot_original."""
    path = gt_path_tekgen(onto_id)
    with open(path, encoding="utf-8") as f:
        item = json.loads(f.readline())
    triplas = " ".join(
        f"{t['rel']}({t['sub']},{t['obj']})" for t in item["triples"]
    ) if item.get("triples") else ""
    return item["sent"], triplas


def item_manifesto(id_, dataset, onto_id, condicao, parceira_id=None, suffix="v3"):
    item = {
        "id": f"p{suffix}_{onto_id.replace('_','')}_{condicao.lower().replace('_','')}",
        "dataset": dataset,
        "split": "test",
        "ontologia_id": onto_id,
        "condicao": condicao,
        "modelo": MODELO,
        "quantizacao": MODELO_QUANTIZACAO,
        "chat_template": CHAT_TEMPLATE,
        "provedor": "openai_compat",
        "provedor_url": PROVEDOR_URL,
        "api_key": "token-nao-usado",
        "temperatura": 0,
        "seed": SEED,
        "max_tokens": 512,
        "frases_max": FRASES_MAX,
        "frases_seed": SEED,
        "git_hash": GIT_HASH,
        "bench_url": "https://github.com/cenguix/Text2KGBench.git",
        "bench_commit": BENCH_COMMIT,
        "saida": f"resultados/piloto_v3/p{suffix}_{onto_id.replace('_','')}_{condicao.lower().replace('_','')}.jsonl",
    }
    if parceira_id:
        item["ontologia_parceira_id"] = parceira_id
    return item


def main():
    # === PROMPTS REAIS ===
    print("=" * 70)
    print("PROMPTS REAIS — 1_movie, primeira frase do GT, Variante D")
    print("=" * 70)

    onto_1movie = _load_onto(BENCH, "tekgen", "1_movie")
    onto_4book = _load_onto(BENCH, "tekgen", "4_book")
    onto_9nature = _load_onto(BENCH, "tekgen", "9_nature")

    # Primeira frase do GT
    with open(gt_path_tekgen("1_movie"), encoding="utf-8") as f:
        primeira = json.loads(f.readline())
    sent = primeira["sent"]
    print(f"\nFrase usada: {sent[:80]}...")

    # C3: top-3 relações de treino
    top3 = carregar_frequencias_treino(BENCH, "tekgen", "1_movie", k=3)
    ctx_c3 = formatar_contexto(onto_1movie, "C3", top_k_rels=top3)
    prompt_c3 = montar_prompt(sent, "C3", contexto_ontologia=ctx_c3)
    print(f"\n--- PROMPT C3 (top_k=3: {top3}) ---")
    print(prompt_c3[:1500])

    # C5_vizinho: 4_book
    # Tentar tekgen primeiro, depois webnlg
    try:
        ctx_v = formatar_contexto(onto_1movie, "C5_vizinho", ontologia_parceira=onto_4book)
    except Exception:
        ctx_v = formatar_contexto(onto_1movie, "C5_vizinho", ontologia_parceira=onto_1movie)
    prompt_c5v = montar_prompt(sent, "C5_vizinho", contexto_ontologia=ctx_v)
    print(f"\n--- PROMPT C5_vizinho (parceira=4_book) ---")
    print(prompt_c5v[:1500])

    # C5_incongruente: 9_nature
    try:
        ctx_i = formatar_contexto(onto_1movie, "C5_incongruente", ontologia_parceira=onto_9nature)
    except Exception:
        ctx_i = formatar_contexto(onto_1movie, "C5_incongruente", ontologia_parceira=onto_1movie)
    prompt_c5i = montar_prompt(sent, "C5_incongruente", contexto_ontologia=ctx_i)
    print(f"\n--- PROMPT C5_incongruente (parceira=9_nature) ---")
    print(prompt_c5i[:1500])

    # === MANIFESTO PILOTO-V3 ===
    print("\n" + "=" * 70)
    print("MANIFESTO PILOTO-V3")
    print("=" * 70)

    # Pares C5 de pares_c5.json
    pares_path = os.path.join(REPO, "configs", "pipeline", "pares_c5.json")
    pares_c5_data = json.load(open(pares_path, encoding="utf-8"))
    pares_tekgen = pares_c5_data.get("tekgen", {})
    pares_dbpedia = pares_c5_data.get("dbpedia", pares_c5_data.get("webnlg", {}))

    # Ontologias induzidas C2 disponíveis
    c2_ontos = {}
    for onto_id in ["1_movie", "2_music"]:
        p = os.path.join(REPO, "configs", "c2", f"tekgen_{onto_id}_induced.json")
        if os.path.exists(p):
            c2_ontos[onto_id] = p

    manifesto = []

    # TekGen: 1_movie, 2_music
    tekgen_ontos = ["1_movie", "2_music"]
    for onto_id in tekgen_ontos:
        condicoes = ["C0", "C1", "C1_oneshot_original", "C3", "C5_vizinho", "C5_incongruente"]
        # C2 apenas se houver ontologia induzida
        if onto_id in c2_ontos:
            condicoes.insert(3, "C2")

        p_v = pares_tekgen.get(onto_id, {}).get("vizinho", {}).get("id")
        p_i = pares_tekgen.get(onto_id, {}).get("incongruente", {}).get("id")

        for cond in condicoes:
            parceira = None
            if cond == "C5_vizinho":
                parceira = p_v
            elif cond == "C5_incongruente":
                parceira = p_i
            manifesto.append(item_manifesto(
                f"pv3_{onto_id}_{cond}", "tekgen", onto_id, cond, parceira
            ))

    # DBpedia: 1_university, 19_film
    dbpedia_ontos = ["1_university", "19_film"]
    for onto_id in dbpedia_ontos:
        condicoes = ["C0", "C1", "C1_oneshot_original", "C3", "C5_vizinho", "C5_incongruente"]
        # Sem C2 (sem raw extractions de DBpedia ainda)

        p_v = pares_dbpedia.get(onto_id, {}).get("vizinho", {}).get("id")
        p_i = pares_dbpedia.get(onto_id, {}).get("incongruente", {}).get("id")

        for cond in condicoes:
            parceira = None
            if cond == "C5_vizinho":
                parceira = p_v
            elif cond == "C5_incongruente":
                parceira = p_i
            manifesto.append(item_manifesto(
                f"pv3_{onto_id}_{cond}", "webnlg", onto_id, cond, parceira
            ))

    n_total = len(manifesto) * FRASES_MAX
    tempo_h = n_total / 5300
    print(f"\nTotal de experimentos: {len(manifesto)}")
    print(f"Total de inferências: {n_total}")
    print(f"Tempo estimado a 5300 fr/h: {tempo_h:.1f}h ({tempo_h*60:.0f} min)")
    print(f"\nCondições por ontologia:")
    for onto_id in tekgen_ontos + dbpedia_ontos:
        conds = [e["condicao"] for e in manifesto if e["ontologia_id"] == onto_id]
        print(f"  {onto_id}: {conds}")

    # Salvar manifesto
    saida = os.path.join(REPO, "configs", "pipeline", "manifesto_piloto_v3.json")
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(manifesto, f, ensure_ascii=False, indent=2)
    print(f"\nManifesto salvo: {saida}")


if __name__ == "__main__":
    main()
