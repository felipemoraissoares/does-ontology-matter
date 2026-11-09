# Carrega frases de teste e exemplos do benchmark.
#
# Estrutura de diretórios esperada (relativa ao BENCH_DIR):
#   TekGen test:    data/wikidata_tekgen/ground_truth/ont_{onto}_ground_truth.jsonl
#   TekGen unseen:  data/wikidata_tekgen/unseen_sentences/ground_truth/ont_{onto}_unseen_ground_truth.jsonl
#   DBpedia test:   data/dbpedia_webnlg/ground_truth/ont_{onto}_ground_truth.jsonl
#   Prompts TekGen: data/wikidata_tekgen/baselines/prompts/[unseen/]ont_{onto}[_unseen]_prompts.jsonl
#   Prompts DBpedia:data/dbpedia_webnlg/baselines/prompts/ont_{onto}_prompts.jsonl
#
# Dados de treino (para cálculo de frequências sem vazamento):
#   TekGen train:   data/wikidata_tekgen/train/ont_{onto}_train.jsonl
#                   formato: um dict por linha, campo "rel_label"
#   DBpedia train:  data/dbpedia_webnlg/train/ont_{onto}_train.jsonl
#                   formato: um dict por linha, campo "triples" com lista de {"sub","rel","obj"}

import json
import os
import re
from collections import Counter
from typing import List, Dict, Optional, Tuple


def _ler_jsonl(caminho: str) -> List[Dict]:
    with open(caminho, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _extrair_exemplo(prompt_texto: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extrai (exemplo_sentença, exemplo_saída) do texto do prompt original.
    Suporta tanto o formato TekGen (saída inline) quanto DBpedia (saída na linha seguinte).
    """
    m = re.search(
        r"Example Sentence:\s*(.+?)\nExample Output:\s*\n?(.*?)\n\nTest Sentence:",
        prompt_texto,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def _caminhos_gt(bench_dir: str, dataset: str, split: str, ontologia_id: str) -> str:
    if dataset == "tekgen":
        if split == "unseen":
            return os.path.join(
                bench_dir, "data", "wikidata_tekgen", "unseen_sentences",
                "ground_truth", f"ont_{ontologia_id}_unseen_ground_truth.jsonl",
            )
        return os.path.join(
            bench_dir, "data", "wikidata_tekgen",
            "ground_truth", f"ont_{ontologia_id}_ground_truth.jsonl",
        )
    # webnlg
    return os.path.join(
        bench_dir, "data", "dbpedia_webnlg",
        "ground_truth", f"ont_{ontologia_id}_ground_truth.jsonl",
    )


def _caminhos_prompts(bench_dir: str, dataset: str, split: str, ontologia_id: str) -> str:
    if dataset == "tekgen":
        if split == "unseen":
            return os.path.join(
                bench_dir, "data", "wikidata_tekgen", "baselines", "prompts",
                "unseen", f"ont_{ontologia_id}_unseen_prompts.jsonl",
            )
        return os.path.join(
            bench_dir, "data", "wikidata_tekgen", "baselines", "prompts",
            f"ont_{ontologia_id}_prompts.jsonl",
        )
    return os.path.join(
        bench_dir, "data", "dbpedia_webnlg", "baselines", "prompts",
        f"ont_{ontologia_id}_prompts.jsonl",
    )


def _caminhos_treino(bench_dir: str, dataset: str, ontologia_id: str) -> str:
    if dataset == "tekgen":
        return os.path.join(
            bench_dir, "data", "wikidata_tekgen",
            "train", f"ont_{ontologia_id}_train.jsonl",
        )
    return os.path.join(
        bench_dir, "data", "dbpedia_webnlg",
        "train", f"ont_{ontologia_id}_train.jsonl",
    )


def carregar_frases_treino(
    bench_dir: str, dataset: str, ontologia_id: str
) -> List[Dict]:
    """
    Carrega frases de TREINO para indução C2.

    Retorna dicts mínimos: {"id", "sent", "gt_triples"}.
    gt_triples é usado apenas para calibração — NÃO deve aparecer no prompt enviado ao modelo.

    TekGen:  uma tripla por linha; agrupa em frases únicas por "sent".
    DBpedia: uma frase por linha com campo "triples".
    """
    caminho = _caminhos_treino(bench_dir, dataset, ontologia_id)
    por_sent: Dict[str, Dict] = {}  # sent_text → dict acumulado
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            d = json.loads(linha)
            sent = d.get("sent", "")
            if not sent:
                continue
            if dataset == "tekgen":
                if sent not in por_sent:
                    por_sent[sent] = {"id": d.get("id", sent), "sent": sent, "gt_triples": []}
                # Uma linha = uma tripla
                por_sent[sent]["gt_triples"].append({
                    "sub": d.get("sub_label", ""),
                    "rel": d.get("rel_label", "").replace(" ", "_"),
                    "obj": d.get("obj_label", ""),
                })
            else:
                # DBpedia: uma linha = uma frase
                if sent not in por_sent:
                    por_sent[sent] = {
                        "id": d.get("id", sent),
                        "sent": sent,
                        "gt_triples": d.get("triples", []),
                    }
    return list(por_sent.values())


def carregar_frequencias_treino(
    bench_dir: str, dataset: str, ontologia_id: str, k: int = 3
) -> List[str]:
    """
    Retorna os k labels de relação mais frequentes nos dados de TREINO (nunca no GT de teste).

    Usado exclusivamente para definir o contexto de C3 sem vazamento de informação do conjunto
    de teste.

    Formatos de treino:
      TekGen: um dict por linha, campo "rel_label" (uma tripla por linha)
      DBpedia: um dict por linha, campo "triples" com lista de {"sub","rel","obj"}
    """
    caminho = _caminhos_treino(bench_dir, dataset, ontologia_id)
    cont: Counter = Counter()
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            d = json.loads(linha)
            if "rel_label" in d:
                # TekGen: uma tripla por linha
                cont[d["rel_label"].replace(" ", "_")] += 1
            else:
                # DBpedia: múltiplas triplas por linha
                for t in d.get("triples", []):
                    cont[t["rel"].replace(" ", "_")] += 1
    return [r for r, _ in cont.most_common(k)]


def carregar_frases(
    bench_dir: str,
    dataset: str,
    split: str,
    ontologia_id: str,
) -> List[Dict]:
    """
    Retorna lista de dicts, um por sentença:
      {
        "id": str,
        "sent": str,
        "gt_triples": [ {"sub": ..., "rel": ..., "obj": ...} ],
        "exemplo_sent": str | None,   # da condição C1_oneshot_original
        "exemplo_saida": str | None,
        "prompt_original": str,       # texto completo do prompt do benchmark
      }
    """
    gt_path = _caminhos_gt(bench_dir, dataset, split, ontologia_id)
    prompt_path = _caminhos_prompts(bench_dir, dataset, split, ontologia_id)

    gt_por_id = {r["id"]: r for r in _ler_jsonl(gt_path)}

    prompts_por_id: Dict[str, str] = {}
    if os.path.exists(prompt_path):
        for r in _ler_jsonl(prompt_path):
            prompts_por_id[r["id"]] = r["prompt"]

    frases = []
    for sent_id, gt in gt_por_id.items():
        prompt_txt = prompts_por_id.get(sent_id, "")
        ex_sent, ex_saida = _extrair_exemplo(prompt_txt) if prompt_txt else (None, None)
        frases.append(
            {
                "id": sent_id,
                "sent": gt["sent"],
                "gt_triples": gt.get("triples", []),
                "exemplo_sent": ex_sent,
                "exemplo_saida": ex_saida,
                "prompt_original": prompt_txt,
            }
        )
    return frases
