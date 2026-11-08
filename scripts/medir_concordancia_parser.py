#!/usr/bin/env python
"""
Mede a concordância entre o parser aplicado ao campo 'response'
e o campo 'triples' publicado nos arquivos de baseline.
"""
import json
import os
import sys

PROJETO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_DIR = os.path.abspath(os.path.join(PROJETO_DIR, ".."))
BENCH_DIR = os.path.join(WORKSPACE_DIR, "Text2KGBench")

sys.path.insert(0, os.path.join(PROJETO_DIR, "src"))
from t2kg.parser import parsear_resposta


def normalizar_tripla(t):
    """Normalização para comparação: minúsculas, sem espaços/underscores."""
    import re
    return tuple(re.sub(r"[_\s]+", "", x).lower() for x in t)


def ler_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def avaliar_concordancia(nome, path):
    dados = ler_jsonl(path)
    total_frases = 0
    frases_concordam = 0
    total_triplas_pub = 0
    total_triplas_parser = 0
    total_match = 0
    divergencias = []

    for d in dados:
        resposta = d.get("response", "")
        publicado = d.get("triples", [])  # [[sub, rel, obj], ...]
        parseado = parsear_resposta(resposta)

        pub_norm = {normalizar_tripla(t) for t in publicado}
        par_norm = {normalizar_tripla(t) for t in parseado}

        total_frases += 1
        total_triplas_pub += len(publicado)
        total_triplas_parser += len(parseado)
        total_match += len(pub_norm & par_norm)

        if pub_norm == par_norm:
            frases_concordam += 1
        else:
            divergencias.append({
                "id": d.get("id", "?"),
                "response": resposta[:200],
                "publicado": publicado,
                "parseado": parseado,
                "em_pub_nao_parser": [t for t in publicado if normalizar_tripla(t) not in par_norm],
                "em_parser_nao_pub": [t for t in parseado if normalizar_tripla(t) not in pub_norm],
            })

    pct_frases = 100 * frases_concordam / total_frases if total_frases else 0
    pct_triplas = 100 * total_match / total_triplas_pub if total_triplas_pub else 0

    print(f"\n{nome}")
    print(f"  Frases totais:          {total_frases}")
    print(f"  Frases com concordância exata: {frases_concordam} ({pct_frases:.1f}%)")
    print(f"  Triplas publicadas:     {total_triplas_pub}")
    print(f"  Triplas do parser:      {total_triplas_parser}")
    print(f"  Match de triplas:       {total_match} ({pct_triplas:.1f}%)")
    if divergencias:
        print(f"  Divergências (primeiras 3):")
        for div in divergencias[:3]:
            print(f"    ID: {div['id']}")
            print(f"    response: {div['response'][:100]}")
            if div["em_pub_nao_parser"]:
                print(f"    em pub, não parser: {div['em_pub_nao_parser'][:2]}")
            if div["em_parser_nao_pub"]:
                print(f"    em parser, não pub: {div['em_parser_nao_pub'][:2]}")

    return pct_frases, pct_triplas, len(divergencias)


def main():
    ARQUIVOS = [
        # TekGen Vicuna
        ("TekGen/Vicuna-13B/1_movie",
         os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "baselines", "Vicuna-13B",
                      "llm_responses", "ont_1_movie_llm_responses.jsonl")),
        ("TekGen/Vicuna-13B/7_space",
         os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "baselines", "Vicuna-13B",
                      "llm_responses", "ont_7_space_llm_responses.jsonl")),
        # TekGen Alpaca v2
        ("TekGen/Alpaca-LoRA-13B/1_movie",
         os.path.join(BENCH_DIR, "data", "wikidata_tekgen", "baselines", "Alpaca-LoRA-13B",
                      "llm_responses", "ont_1_movie_lora-13b_responses_v2.jsonl")),
        # DBpedia Vicuna
        ("DBpedia/Vicuna-13B/1_university",
         os.path.join(BENCH_DIR, "data", "dbpedia_webnlg", "baselines", "Vicuna-13B",
                      "llm_responses", "1_university_Vicuna13B_responses.jsonl")),
        # DBpedia Alpaca
        ("DBpedia/Alpaca-LoRA-13B/1_university",
         os.path.join(BENCH_DIR, "data", "dbpedia_webnlg", "baselines", "Alpaca-LoRA-13B",
                      "llm_responses", "1_university_AlpacaLora13B_responses.jsonl")),
    ]

    print("Concordância do parser vs. triples publicados")
    print("=" * 60)

    total_concordancia = []
    for nome, path in ARQUIVOS:
        if os.path.exists(path):
            pf, pt, ndiv = avaliar_concordancia(nome, path)
            total_concordancia.append((pf, pt))
        else:
            print(f"\n{nome}: arquivo não encontrado")

    if total_concordancia:
        avg_f = sum(x[0] for x in total_concordancia) / len(total_concordancia)
        avg_t = sum(x[1] for x in total_concordancia) / len(total_concordancia)
        print(f"\n{'='*60}")
        print(f"Média de concordância por frase: {avg_f:.1f}%")
        print(f"Média de concordância por tripla: {avg_t:.1f}%")


if __name__ == "__main__":
    main()
