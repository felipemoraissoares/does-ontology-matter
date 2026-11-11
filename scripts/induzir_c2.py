"""
Gera a ontologia induzida para a condição C2 (D-16).

Protocolo:
  1. Sorteia N frases de treino com seed fixa (padrão: N=50, seed=42)
  2. Extrai triplas livres com o modelo em modo C0 (sem ontologia)
  3. Agrupa rótulos de relação por similaridade de embedding (threshold provisório 0.80)
  4. Cada grupo vira uma relação da ontologia induzida (rótulo = mais frequente do grupo)
  5. Se --calibrar: gera curvas de agrupamento e projeção, salva em calibracao_{dataset}_{onto}.json

Outputs em saida_dir/:
  {dataset}_{onto}_induced.json        — ontologia induzida
  {dataset}_{onto}_frases_seed{s}.json — IDs das frases sorteadas
  calibracao_{dataset}_{onto}.json     — curvas de calibração (com --calibrar)

Uso:
  python scripts/induzir_c2.py --dataset tekgen --onto 1_movie \\
      --url http://localhost:8000/v1 --threshold 0.80 --calibrar
"""
import argparse
import json
import os
import re
import random
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    _EMBS_OK = True
except ImportError:
    _EMBS_OK = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from t2kg.pipeline.dados import carregar_frases, carregar_frases_treino  # noqa: E402
from t2kg.pipeline.provedores.openai_compat import ProvedorOpenAICompat  # noqa: E402
from t2kg.pipeline.prompts import montar_prompt  # noqa: E402
from t2kg.parser import parsear_resposta  # noqa: E402
from t2kg.avaliacao.extended import _normalizar_rotulo  # noqa: E402

BENCH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Text2KGBench"))
MODELO_EMBS = "all-MiniLM-L6-v2"
N_FRASES_DEFAULT = 50
SEED_DEFAULT = 42
THRESHOLD_PROVISORIO = 0.80

# D-32: rótulos de relação que são placeholders literais da instrução Variante D — excluídos antes do agrupamento.
PLACEHOLDER_RELS = frozenset({"relation_name"})


# ---------------------------------------------------------------------------
# Extração de triplas livres (C0) no treino
# ---------------------------------------------------------------------------

def sortear_frases_treino(bench: str, dataset: str, onto_id: str, n: int, seed: int) -> List[Dict]:
    """Sorteia n frases de TREINO com seed fixa (sem vazar gt_triples para o modelo)."""
    frases = list(carregar_frases_treino(bench, dataset, onto_id))
    rng = random.Random(seed)
    rng.shuffle(frases)
    return frases[:n]


def extrair_triplas_livres(
    frases: List[Dict],
    provedor,
) -> Dict[str, List[List[str]]]:
    """
    Extrai triplas livres (modo C0) de cada frase de treino.
    Retorna dict {frase_id: [[sub, rel, obj], ...]} para permitir alinhamento com GT.
    """
    triplas_por_frase: Dict[str, List[List[str]]] = {}
    for f in frases:
        # Envia APENAS o texto da frase — gt_triples NUNCA aparecem no prompt (D-16)
        prompt = montar_prompt(sentenca=f["sent"], condicao="C0")
        resp = provedor.gerar(prompt, max_tokens=512, temperatura=0.0)
        triplas = parsear_resposta(resp)
        triplas_por_frase[f["id"]] = triplas
    return triplas_por_frase


def extrair_triplas_com_respostas(
    frases: List[Dict],
    provedor,
) -> Tuple[Dict[str, List[List[str]]], List[Dict]]:
    """
    Como extrair_triplas_livres, mas também retorna lista de registros brutos para salvar
    em JSONL: [{"id": ..., "sent": ..., "response": ..., "triples": [...]}].
    """
    triplas_por_frase: Dict[str, List[List[str]]] = {}
    registros_raw: List[Dict] = []
    for f in frases:
        prompt = montar_prompt(sentenca=f["sent"], condicao="C0")
        resp = provedor.gerar(prompt, max_tokens=512, temperatura=0.0)
        triplas = parsear_resposta(resp)
        triplas_por_frase[f["id"]] = triplas
        registros_raw.append({"id": f["id"], "sent": f["sent"], "response": resp, "triples": triplas})
    return triplas_por_frase, registros_raw


# ---------------------------------------------------------------------------
# Agrupamento por embedding
# ---------------------------------------------------------------------------

def agrupar_relacoes(
    relacoes: List[str],
    modelo: SentenceTransformer,
    threshold: float,
    normalizar: bool = False,
    definicoes: Optional[Dict] = None,
) -> List[Dict]:
    """
    Agrupa rótulos de relação por similaridade de embedding (greedy single-link).
    Retorna lista de dicts: {label: str, frequencia: int, membros: List[str]}.

    normalizar=True aplica _normalizar_rotulo (D-22) antes do embedding.
    definicoes: dict {rotulo: definicao} para D-24 (embedding aumentado com definição).
    """
    if not relacoes:
        return []

    contagem = Counter(relacoes)
    unicos = list(contagem.keys())
    if definicoes:
        norm_fn = _normalizar_rotulo if normalizar else (lambda r: r.replace("_", " ").lower())
        unicos_embs = [f"{norm_fn(r)}: {definicoes[r]}" if r in definicoes else norm_fn(r) for r in unicos]
    elif normalizar:
        unicos_embs = [_normalizar_rotulo(r) for r in unicos]
    else:
        unicos_embs = [r.replace("_", " ").lower() for r in unicos]
    embs = modelo.encode(unicos_embs, show_progress_bar=False)
    sim = cosine_similarity(embs, embs)
    n = len(unicos)
    visitado = [False] * n
    grupos = []

    for i in range(n):
        if visitado[i]:
            continue
        grupo = [i]
        visitado[i] = True
        for j in range(i + 1, n):
            if not visitado[j] and sim[i, j] >= threshold:
                grupo.append(j)
                visitado[j] = True
        membros = [unicos[k] for k in grupo]
        label = max(membros, key=lambda m: contagem[m])
        freq_total = sum(contagem[m] for m in membros)
        grupos.append({"label": label, "frequencia": freq_total, "membros": membros})

    grupos.sort(key=lambda g: -g["frequencia"])
    return grupos


def contar_frases_por_grupo(
    grupos: List[Dict],
    triplas_por_frase: Dict[str, List[List[str]]],
) -> None:
    """Adiciona campo 'n_frases' a cada grupo: número de frases distintas com ≥1 tripla do grupo."""
    for g in grupos:
        membros_set = {m.lower().strip() for m in g["membros"]}
        frases_do_grupo: set = set()
        for fid, triplas in triplas_por_frase.items():
            for t in triplas:
                if len(t) >= 2 and t[1].lower().strip() in membros_set:
                    frases_do_grupo.add(fid)
                    break
        g["n_frases"] = len(frases_do_grupo)


def construir_ontologia_induzida(grupos: List[Dict]) -> Dict:
    return {
        "concepts": [],
        "relations": [
            {"label": g["label"], "frequencia": g["frequencia"], "membros": g["membros"]}
            for g in grupos
        ],
    }


# ---------------------------------------------------------------------------
# Calibração de threshold (D-16) — salva curvas em JSON
# ---------------------------------------------------------------------------

def _normalizar(s: str) -> str:
    return re.sub(r"[\s_]+", "", s.lower())


def _alinhar_pares(
    triplas_por_frase: Dict[str, List[List[str]]],
    frases: List[Dict],
) -> List[Tuple[str, str]]:
    """
    Alinha triplas livres com GT pelo par normalizado (sub, obj).
    Retorna lista de pares (rel_livre, rel_gt) para frases em que houve match.
    """
    pares: List[Tuple[str, str]] = []
    # Índice GT: {frase_id: {(sub_norm, obj_norm): rel_gt}}
    idx_gt: Dict[str, Dict[Tuple[str, str], str]] = {}
    for f in frases:
        idx = {}
        for t in f.get("gt_triples", []):
            if isinstance(t, dict):
                s, r, o = t.get("sub", ""), t.get("rel", ""), t.get("obj", "")
            else:
                s, r, o = t[0], t[1], t[2]
            idx[(_normalizar(s), _normalizar(o))] = r
        idx_gt[f["id"]] = idx

    for fid, triplas in triplas_por_frase.items():
        gt_idx = idx_gt.get(fid, {})
        for t in triplas:
            if len(t) < 3:
                continue
            s_n, r_l, o_n = _normalizar(t[0]), t[1], _normalizar(t[2])
            rel_gt = gt_idx.get((s_n, o_n))
            if rel_gt is not None:
                pares.append((r_l.lower().strip(), rel_gt.lower().strip()))
    return pares


def calibrar_e_salvar(
    triplas_por_frase: Dict[str, List[List[str]]],
    frases: List[Dict],
    onto_ref: Dict,
    modelo: SentenceTransformer,
    saida_dir: str,
    dataset: str,
    onto_id: str,
    thresholds: Optional[List[float]] = None,
) -> str:
    """
    Calcula curvas de calibração de agrupamento e projeção.
    Salva resultado em {saida_dir}/calibracao_{dataset}_{onto_id}.json.
    Retorna o caminho do arquivo salvo.
    """
    if thresholds is None:
        thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]

    # Relações livres únicas
    relacoes_livres: List[str] = []
    for triplas in triplas_por_frase.values():
        relacoes_livres.extend(t[1].lower().strip() for t in triplas if len(t) >= 3)
    unicos = list(set(relacoes_livres))

    # Alinhamento pares (rel_livre, rel_gt)
    pares_alinhados = _alinhar_pares(triplas_por_frase, frases)
    n_pares = len(pares_alinhados)

    # Embeddings para curvas
    embs_unicos = modelo.encode(unicos, show_progress_bar=False) if unicos else np.zeros((0, 384))

    # Relações da ontologia de referência
    rels_ref = [r["label"].replace("_", " ").lower() for r in onto_ref.get("relations", [])]
    embs_ref = modelo.encode(rels_ref, show_progress_bar=False) if rels_ref else np.zeros((0, 384))

    # --- Curva de agrupamento ---
    curva_agrupamento = {}
    if len(unicos) >= 2:
        sim_livre = cosine_similarity(embs_unicos, embs_unicos)
        n_u = len(unicos)
        total_pares = n_u * (n_u - 1) // 2

        # Se temos pares alinhados, calcula precision/recall/F1 do agrupamento
        if n_pares >= 5:
            # Pares "mesma relação" (rel_livre_A e rel_livre_B mapeiam para mesmo rel_gt)
            rel_para_gt: Dict[str, str] = {r_l: r_g for r_l, r_g in pares_alinhados}
            pos_set: set = set()
            neg_set: set = set()
            for i, ri in enumerate(unicos):
                for j, rj in enumerate(unicos):
                    if i >= j:
                        continue
                    gi = rel_para_gt.get(ri)
                    gj = rel_para_gt.get(rj)
                    if gi is not None and gj is not None:
                        if _normalizar(gi) == _normalizar(gj):
                            pos_set.add((i, j))
                        else:
                            neg_set.add((i, j))
            n_pos = len(pos_set)
            n_neg = len(neg_set)
        else:
            pos_set = neg_set = set()
            n_pos = n_neg = 0

        for t in thresholds:
            pares_agrupados = sum(
                1 for i in range(n_u) for j in range(i + 1, n_u) if sim_livre[i, j] >= t
            )
            entrada: Dict = {
                "pares_agrupados": pares_agrupados,
                "total_pares": total_pares,
                "fracao_agrupada": pares_agrupados / total_pares if total_pares > 0 else 0.0,
            }
            if n_pos > 0:
                tp = sum(1 for i, j in pos_set if sim_livre[i, j] >= t)
                fp = sum(1 for i, j in neg_set if sim_livre[i, j] >= t)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec  = tp / n_pos
                f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                entrada.update({"tp": tp, "fp": fp, "n_positivos": n_pos,
                                 "precisao": round(prec, 4), "recall": round(rec, 4),
                                 "f1": round(f1, 4)})
            curva_agrupamento[str(t)] = entrada
    else:
        curva_agrupamento = {"aviso": "Poucos rótulos únicos (< 2)."}

    # --- Curva de projeção ---
    curva_projecao = {}
    if len(unicos) >= 1 and len(rels_ref) >= 1:
        sim_proj = cosine_similarity(embs_unicos, embs_ref)  # (n_unicos, n_ref)

        if n_pares >= 5:
            # Precision: fração de projeções corretas (rel_livre → rel_gt_mais_similar)
            # Ground truth: rel_livre → rel_gt via alinhamento
            rel_para_gt = {r_l: _normalizar(r_g) for r_l, r_g in pares_alinhados}
            rels_ref_norm = [_normalizar(r) for r in rels_ref]
        else:
            rel_para_gt = {}
            rels_ref_norm = []

        for t in thresholds:
            projetados = 0
            tp_proj = 0
            for i, rel in enumerate(unicos):
                max_sim = float(sim_proj[i].max())
                if max_sim >= t:
                    projetados += 1
                    if rel_para_gt and rel in rel_para_gt:
                        idx_pred = int(sim_proj[i].argmax())
                        if rels_ref_norm and rels_ref_norm[idx_pred] == rel_para_gt[rel]:
                            tp_proj += 1
            entrada = {
                "relacoes_projetadas": projetados,
                "total_unicas": len(unicos),
                "fracao_projetada": projetados / len(unicos) if unicos else 0.0,
                "threshold": t,
            }
            if rel_para_gt:
                entrada["tp_projecao"] = tp_proj
                entrada["precisao_projecao"] = round(tp_proj / projetados, 4) if projetados > 0 else 0.0
            curva_projecao[str(t)] = entrada
    else:
        curva_projecao = {"aviso": "Sem relações livres ou ontologia de referência vazia."}

    # --- Serialização ---
    resultado = {
        "dataset": dataset,
        "onto_id": onto_id,
        "n_frases": len(frases),
        "n_relacoes_livres": len(relacoes_livres),
        "n_relacoes_unicas": len(unicos),
        "n_pares_alinhados": n_pares,
        "relacoes_unicas_exemplo": unicos[:20],
        "curva_agrupamento": curva_agrupamento,
        "curva_projecao": curva_projecao,
    }
    saida = os.path.join(saida_dir, f"calibracao_{dataset}_{onto_id}.json")
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    print(f"  Curvas de calibracao salvas em: {saida}")
    return saida


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_provedor(url: str, modelo_llm: str, mock: bool = False):
    if mock:
        from t2kg.pipeline.provedores.mock import ProvedorMock
        return ProvedorMock()
    return ProvedorOpenAICompat(url=url, api_key="token-nao-usado", modelo=modelo_llm)


def _carregar_onto_ref(dataset: str, onto_id: str) -> Dict:
    if dataset == "tekgen":
        p = os.path.join(BENCH, "data", "wikidata_tekgen", "ontologies", f"{onto_id}_ontology.json")
    else:
        p = os.path.join(BENCH, "data", "dbpedia_webnlg", "ontologies", f"{onto_id}_ontology.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["tekgen", "webnlg"])
    p.add_argument("--onto", required=True)
    p.add_argument("--url", default="http://localhost:8000/v1")
    p.add_argument("--modelo_llm", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    p.add_argument("--threshold", type=float, default=THRESHOLD_PROVISORIO)
    p.add_argument("--n_frases", type=int, default=N_FRASES_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--calibrar", action="store_true",
                   help="Calcula e salva curvas de calibracao em calibracao_{dataset}_{onto}.json")
    p.add_argument("--freq_min", type=int, default=1,
                   help="Frequência mínima (frases DISTINTAS) para incluir relação. "
                        "freq_min=5 corresponde a >=10%% de N=50 frases de treino.")
    p.add_argument("--saida_raw_jsonl", default=None,
                   help="Caminho para salvar respostas cruas da extração livre (JSONL).")
    p.add_argument("--from_raw_jsonl", default=None,
                   help="Carrega respostas cruas de JSONL existente, sem chamar o LLM (D-22).")
    p.add_argument("--normalizar", action="store_true",
                   help="Aplica normalização D-22 aos rótulos antes do embedding.")
    p.add_argument("--saida_dir", default=os.path.join(
        os.path.dirname(__file__), "..", "configs", "c2"))
    p.add_argument("--run_id", default=None,
                   help="ID único desta execução (armazenado em metadata.run_id para guard check).")
    p.add_argument("--use_definicoes", action="store_true",
                   help="Usa definições D-24 no embedding (rótulo: definição). "
                        "Requer configs/c2/definicoes_{onto}.json gerado por gerar_definicoes_relacoes.py.")
    p.add_argument("--mock", action="store_true",
                   help="Usa ProvedorMock em vez de vLLM (para testes locais sem GPU).")
    args = p.parse_args()

    if not _EMBS_OK:
        print("ERRO: instale sentence-transformers e scikit-learn.")
        sys.exit(1)

    os.makedirs(args.saida_dir, exist_ok=True)

    print(f"Carregando modelo de embeddings: {MODELO_EMBS}")
    modelo_embs = SentenceTransformer(MODELO_EMBS)

    print(f"Sorteando {args.n_frases} frases de treino ({args.dataset}/{args.onto}, seed={args.seed})")
    frases = sortear_frases_treino(BENCH, args.dataset, args.onto, args.n_frases, args.seed)
    print(f"  Frases sorteadas: {len(frases)}")

    ids_sorteados = [f["id"] for f in frases]
    seed_path = os.path.join(args.saida_dir, f"{args.dataset}_{args.onto}_frases_seed{args.seed}.json")
    with open(seed_path, "w", encoding="utf-8") as fout:
        json.dump({"dataset": args.dataset, "onto": args.onto, "seed": args.seed,
                   "n_frases": len(frases), "ids": ids_sorteados}, fout, ensure_ascii=False, indent=2)
    print(f"  IDs registrados em: {seed_path}")

    if args.from_raw_jsonl:
        # D-22: reprocessamento sem GPU a partir de respostas cruas
        print(f"Carregando respostas cruas de: {args.from_raw_jsonl}")
        with open(args.from_raw_jsonl, encoding="utf-8") as fin:
            raw_data = [json.loads(l) for l in fin if l.strip()]
        triplas_por_frase = {
            r["id"]: (r.get("triples") or parsear_resposta(r.get("response", "")))
            for r in raw_data
        }
        print(f"  {len(triplas_por_frase)} frases carregadas (sem chamada ao LLM).")
        ids_raw = set(triplas_por_frase)
        ids_frases = set(ids_sorteados)
        if ids_raw != ids_frases:
            extras = ids_raw - ids_frases
            faltando = ids_frases - ids_raw
            if extras:
                print(f"  AVISO: {len(extras)} IDs no raw JSONL nao estao no sorteio (seed={args.seed}).")
            if faltando:
                print(f"  AVISO: {len(faltando)} IDs do sorteio ausentes no raw JSONL.")
    else:
        print(f"Extraindo triplas livres (C0) com LLM: {args.modelo_llm}"
              f"{' [MOCK]' if args.mock else ''}")
        provedor = _build_provedor(args.url, args.modelo_llm, mock=args.mock)
        triplas_por_frase, registros_raw = extrair_triplas_com_respostas(frases, provedor)

        if args.saida_raw_jsonl:
            os.makedirs(os.path.dirname(os.path.abspath(args.saida_raw_jsonl)), exist_ok=True)
            with open(args.saida_raw_jsonl, "w", encoding="utf-8") as fout:
                for rec in registros_raw:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  Respostas cruas salvas em: {args.saida_raw_jsonl}")

    all_relacoes_raw = [t[1] for triplas in triplas_por_frase.values() for t in triplas if len(t) >= 3]
    n_placeholder = sum(1 for r in all_relacoes_raw if r.lower().strip() in PLACEHOLDER_RELS)
    all_relacoes = [r for r in all_relacoes_raw if r.lower().strip() not in PLACEHOLDER_RELS]  # D-32
    if n_placeholder:
        print(f"  D-32: {n_placeholder} triplas com placeholder filtradas ({', '.join(PLACEHOLDER_RELS)})")
    print(f"  Relacoes extraidas: {len(all_relacoes)} total, {len(set(all_relacoes))} unicas")

    if args.calibrar:
        print("\n--- Calibracao de threshold ---")
        onto_ref = _carregar_onto_ref(args.dataset, args.onto)
        calibrar_e_salvar(
            triplas_por_frase, frases, onto_ref, modelo_embs,
            args.saida_dir, args.dataset, args.onto,
        )

    # D-24: carrega definições se --use_definicoes
    definicoes = None
    if args.use_definicoes:
        def_path = os.path.join(args.saida_dir, f"definicoes_{args.onto}.json")
        if os.path.exists(def_path):
            with open(def_path, encoding="utf-8") as _f:
                definicoes = json.load(_f).get("relacoes", {})
            print(f"  Definições D-24 carregadas: {len(definicoes)} rótulos de {os.path.basename(def_path)}")
        else:
            print(f"  AVISO: --use_definicoes mas arquivo não encontrado: {def_path}")

    print(f"\nAgrupando relacoes com threshold={args.threshold}"
          f"{' + normalizacao D-22' if args.normalizar else ''}"
          f"{' + definicoes D-24' if definicoes else ''}")
    grupos = agrupar_relacoes(all_relacoes, modelo_embs, args.threshold,
                              normalizar=args.normalizar, definicoes=definicoes)
    print(f"  Grupos formados (antes de freq_min): {len(grupos)}")

    # Conta frases DISTINTAS por grupo (D-21 fix: freq_min conta frases, nao triplas)
    contar_frases_por_grupo(grupos, triplas_por_frase)

    if args.freq_min > 1:
        grupos = [g for g in grupos if g["n_frases"] >= args.freq_min]
        pct = args.freq_min / args.n_frases * 100
        print(f"  Grupos apos freq_min>={args.freq_min} frases distintas (>={pct:.0f}% de {args.n_frases}): {len(grupos)}")

    for g in grupos[:10]:
        print(f"    {g['label']:30s} n_frases={g['n_frases']:3d}  freq={g['frequencia']:3d}  membros={g['membros'][:3]}")

    ontologia_induzida = construir_ontologia_induzida(grupos)
    saida = os.path.join(args.saida_dir, f"{args.dataset}_{args.onto}_induced.json")
    metadados = {
        "dataset": args.dataset, "onto_id": args.onto,
        "n_frases_treino": len(frases), "seed": args.seed,
        "threshold_agrupamento": args.threshold,
        "freq_min_frases": args.freq_min,
        "normalizar_rotulo": args.normalizar,
        "use_definicoes": args.use_definicoes,
        "from_raw_jsonl": args.from_raw_jsonl,
        "modelo_llm": args.modelo_llm if not args.from_raw_jsonl else "(reprocessado sem LLM)",
        "modelo_embs": MODELO_EMBS,
        "ids_frases": ids_sorteados,
        "run_id": args.run_id,
    }
    output = {"metadata": metadados, **ontologia_induzida}
    with open(saida, "w", encoding="utf-8") as fout:
        json.dump(output, fout, ensure_ascii=False, indent=2)
    print(f"\nOntologia induzida salva em: {saida}")
    print(f"  Relacoes: {len(ontologia_induzida['relations'])}")
