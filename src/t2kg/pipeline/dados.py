# Carrega frases de teste e exemplos do benchmark.
#
# Estrutura de diretórios esperada (relativa ao BENCH_DIR):
#   TekGen test:    data/wikidata_tekgen/ground_truth/ont_{onto}_ground_truth.jsonl
#   TekGen unseen:  data/wikidata_tekgen/unseen_sentences/ground_truth/ont_{onto}_unseen_ground_truth.jsonl
#   DBpedia test:   data/dbpedia_webnlg/ground_truth/ont_{onto}_ground_truth.jsonl
#   Prompts TekGen: data/wikidata_tekgen/baselines/prompts/[unseen/]ont_{onto}[_unseen]_prompts.jsonl
#   Prompts DBpedia:data/dbpedia_webnlg/baselines/prompts/ont_{onto}_prompts.jsonl

import json
import os
import re
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
