"""
Calibração C2: tabela N × freq_min.

Para N=50 (dados em raw_tkg_{onto}_treino.jsonl), calcular para freq_min ∈ {2,3,5}:
  1. n_relações induzidas (n_grupos com threshold=0.70, D-22)
  2. cobertura treino: fração das triplas de treino que tem relação correspondente (cosine ≥ 0.70)
  3. ruído: fração das relações induzidas sem correspondência no gabarito de treino

Para N=100: marcar como pendente (dados ainda no Kaggle).
"""
import io
import json
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH = os.path.abspath(os.path.join(REPO, "..", "Text2KGBench"))

ONTOS = ["1_movie", "2_music"]
FREQ_MINS = [2, 3, 5]
THRESHOLD = 0.70


def _normalizar_rotulo(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[_\-/]", " ", s)
    s = re.sub(r"^(has|is)\s+", "", s)
    s = re.sub(r"\s+(by|of|in|to|for)$", "", s)
    return s.strip()


def agrupar_relacoes(relacoes_por_frase, modelo, threshold, freq_min=1):
    """Agrupa relações por similaridade semântica (D-22 norm), conta frases distintas."""
    from collections import Counter
    import numpy as np

    # Coleta labels únicos
    todos_labels = []
    contagem_frases = Counter()
    for fid, rels in relacoes_por_frase.items():
        for rel in set(rels):
            contagem_frases[rel] += 1
            if rel not in todos_labels:
                todos_labels.append(rel)

    if not todos_labels:
        return []

    # Normaliza labels para embedding
    labels_norm = [_normalizar_rotulo(l) for l in todos_labels]

    # Embedding
    embs = modelo.encode(labels_norm, batch_size=64, show_progress_bar=False)
    # Normaliza para cosine
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embs = embs / norms

    # Union-Find para clusters
    parent = list(range(len(todos_labels)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    sims = embs @ embs.T
    for i in range(len(todos_labels)):
        for j in range(i + 1, len(todos_labels)):
            if sims[i, j] >= threshold:
                union(i, j)

    # Agrupa
    grupos_map = {}
    for i, label in enumerate(todos_labels):
        root = find(i)
        if root not in grupos_map:
            grupos_map[root] = []
        grupos_map[root].append(label)

    grupos = []
    for root, members in grupos_map.items():
        # Label canônico = mais frequente
        canonico = max(members, key=lambda l: contagem_frases[l])
        n_frases = sum(contagem_frases[m] for m in members)
        # n_frases de frases distintas (aproximação: soma)
        # Para ser exato precisaria rastrear por frase — aqui é a soma dos membros
        grupos.append({
            "label": canonico,
            "members": members,
            "n_frases": n_frases,
        })

    # Filtra por freq_min
    grupos = [g for g in grupos if g["n_frases"] >= freq_min]

    return sorted(grupos, key=lambda g: -g["n_frases"])


def carregar_raw_jsonl(path):
    """Lê raw_tkg_*.jsonl e retorna {frase_id: [rel_labels]}."""
    relacoes = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        fid = r["id"]
        rels = [t[1] for t in r.get("triples", [])]
        if rels:
            relacoes[fid] = rels
    return relacoes


def carregar_rels_treino(onto_id):
    """Retorna lista de relações do gabarito de treino (labels únicos)."""
    path = os.path.join(BENCH, "data", "wikidata_tekgen", "train", f"ont_{onto_id}_train.jsonl")
    rels = set()
    triplas_treino = []
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        rel = r.get("rel_label", "")
        if rel:
            rels.add(rel)
            triplas_treino.append(rel)
    return list(rels), triplas_treino


def calcular_cobertura(rels_treino_unicos, grupos_induzidos, modelo, threshold):
    """Fração de relações únicas do treino que têm ao menos 1 grupo induzido correspondente."""
    if not rels_treino_unicos or not grupos_induzidos:
        return 0.0

    import numpy as np

    labels_treino_norm = [_normalizar_rotulo(r) for r in rels_treino_unicos]
    labels_induzidos_norm = [_normalizar_rotulo(g["label"]) for g in grupos_induzidos]
    # Inclui membros também para melhor cobertura
    labels_induzidos_all = []
    for g in grupos_induzidos:
        for m in g.get("members", [g["label"]]):
            n = _normalizar_rotulo(m)
            if n not in labels_induzidos_all:
                labels_induzidos_all.append(n)

    embs_treino = modelo.encode(labels_treino_norm, batch_size=64, show_progress_bar=False)
    embs_ind = modelo.encode(labels_induzidos_all, batch_size=64, show_progress_bar=False)

    n_t = np.linalg.norm(embs_treino, axis=1, keepdims=True)
    n_t[n_t == 0] = 1
    embs_treino = embs_treino / n_t

    n_i = np.linalg.norm(embs_ind, axis=1, keepdims=True)
    n_i[n_i == 0] = 1
    embs_ind = embs_ind / n_i

    sims = embs_treino @ embs_ind.T  # (n_treino, n_induzido)
    cobertas = (sims.max(axis=1) >= threshold).sum()
    return cobertas / len(rels_treino_unicos)


def calcular_ruido(grupos_induzidos, rels_treino_unicos, modelo, threshold):
    """Fração de grupos induzidos sem nenhuma relação de treino correspondente."""
    if not grupos_induzidos or not rels_treino_unicos:
        return 1.0

    import numpy as np

    labels_induzidos = [_normalizar_rotulo(g["label"]) for g in grupos_induzidos]
    labels_treino = [_normalizar_rotulo(r) for r in rels_treino_unicos]

    embs_ind = modelo.encode(labels_induzidos, batch_size=64, show_progress_bar=False)
    embs_tr = modelo.encode(labels_treino, batch_size=64, show_progress_bar=False)

    n_i = np.linalg.norm(embs_ind, axis=1, keepdims=True)
    n_i[n_i == 0] = 1
    embs_ind = embs_ind / n_i

    n_t = np.linalg.norm(embs_tr, axis=1, keepdims=True)
    n_t[n_t == 0] = 1
    embs_tr = embs_tr / n_t

    sims = embs_ind @ embs_tr.T
    ruido = (sims.max(axis=1) < threshold).sum()
    return ruido / len(grupos_induzidos)


def main():
    from sentence_transformers import SentenceTransformer

    print("Carregando modelo de embeddings...")
    modelo = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"\n{'Onto':<8} {'N':>4} {'freq':>5} {'n_ind':>6} {'cob_tr':>7} {'ruido':>7}  {'labels_induzidos'}")
    print("-" * 90)

    resultados = []

    for onto_id in ONTOS:
        raw_path = os.path.join(REPO, "resultados", "c2_inducao", f"raw_tkg_{onto_id}_treino.jsonl")

        if not os.path.exists(raw_path):
            print(f"{onto_id:<8} {'50':>4}   ---   JSONL não encontrado: {raw_path}")
            continue

        relacoes_por_frase = carregar_raw_jsonl(raw_path)
        n_frases_raw = len(relacoes_por_frase)

        rels_treino_unicos, _ = carregar_rels_treino(onto_id)

        for fm in FREQ_MINS:
            grupos = agrupar_relacoes(relacoes_por_frase, modelo, THRESHOLD, freq_min=fm)
            n_ind = len(grupos)

            if n_ind == 0:
                cob = 0.0
                ruido = 1.0
            else:
                cob = calcular_cobertura(rels_treino_unicos, grupos, modelo, THRESHOLD)
                ruido = calcular_ruido(grupos, rels_treino_unicos, modelo, THRESHOLD)

            labels_str = ", ".join(g["label"] for g in grupos[:8])
            if len(grupos) > 8:
                labels_str += f" (+{len(grupos)-8})"

            print(f"{onto_id:<8} {'50':>4} {fm:>5} {n_ind:>6} {cob:>7.3f} {ruido:>7.3f}  {labels_str}")

            resultados.append({
                "onto": onto_id, "N": 50, "freq_min": fm,
                "n_ind": n_ind, "cobertura_treino": round(cob, 4),
                "ruido": round(ruido, 4),
                "labels": [g["label"] for g in grupos],
            })

        # N=100: dados pendentes
        print(f"{onto_id:<8} {'100':>4} {'--':>5} {'--':>6} {'--':>7} {'--':>7}  [dados pendentes — Kaggle necessário]")

    # Salvar resultados
    saida = os.path.join(REPO, "resultados", "c2_inducao", "calibracao_c2_grid.json")
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em: {saida}")

    # Interpretação
    print("\n=== INTERPRETAÇÃO ===")
    for r in resultados:
        if r["n_ind"] >= 5:
            status = "OK — suficiente para C2"
        elif r["n_ind"] >= 3:
            status = "MARGINAL — testável mas sub-especificado"
        else:
            status = "INSUFICIENTE — considerar EDC"
        print(f"  {r['onto']} N={r['N']} freq_min={r['freq_min']}: {r['n_ind']} rels → {status}")


if __name__ == "__main__":
    main()
