"""
Verificação crítica permanente: os 6 JSONL do piloto não contêm contaminação de gabarito.

Reconstrói os prompts a partir do Text2KGBench (requer bench local) e verifica:
  - Nenhum prompt contém 'gt_triples', 'exemplo_saida', 'prompt_original' ou padrão de dict Python.
  - C0 não contém bloco CONTEXT nem Example Sentence.
  - C1 contém CONTEXT mas não Example Sentence.
  - C1_oneshot_original contém CONTEXT e Example Sentence.
  - Exibe prompts reais de C0, C1, C1_oneshot para a mesma frase (ont_1_movie_test_10).

Pula automaticamente se Text2KGBench ou os JSONL do piloto não estiverem presentes.
"""
import json
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

WORKSPACE = os.path.abspath(os.path.join(REPO, ".."))
BENCH = os.path.join(WORKSPACE, "Text2KGBench")
PILOTO = os.path.join(REPO, "resultados", "piloto")

DATASETS = {
    "piloto_tkg_1_movie_c0_qwen25_7b_instruct_awq.jsonl":                 ("tekgen", "1_movie", "C0"),
    "piloto_tkg_1_movie_c1_qwen25_7b_instruct_awq.jsonl":                 ("tekgen", "1_movie", "C1"),
    "piloto_tkg_1_movie_c1_oneshot_original_qwen25_7b_instruct_awq.jsonl":("tekgen", "1_movie", "C1_oneshot_original"),
    "piloto_tkg_2_music_c0_qwen25_7b_instruct_awq.jsonl":                 ("tekgen", "2_music", "C0"),
    "piloto_tkg_2_music_c1_qwen25_7b_instruct_awq.jsonl":                 ("tekgen", "2_music", "C1"),
    "piloto_tkg_2_music_c1_oneshot_original_qwen25_7b_instruct_awq.jsonl":("tekgen", "2_music", "C1_oneshot_original"),
}

POISON_TOKENS = ["gt_triples", "exemplo_saida", "prompt_original", "'sub':", "'rel':", "'obj':"]

JSONL_DISPONIVEIS = all(
    os.path.exists(os.path.join(PILOTO, f)) for f in DATASETS
)
BENCH_DISPONIVEL = os.path.exists(BENCH)

SKIP = pytest.mark.skipif(
    not (JSONL_DISPONIVEIS and BENCH_DISPONIVEL),
    reason="Requer resultados/piloto/ e Text2KGBench local.",
)


def _carregar_frases_map(bench, dataset, onto):
    from t2kg.pipeline.dados import carregar_frases
    return {f["id"]: f for f in carregar_frases(bench, dataset, "test", onto)}


def _montar_prompt_para(frase, cond, bench, dataset, onto):
    from t2kg.pipeline.ontologias import carregar_ontologia, formatar_contexto
    from t2kg.pipeline.prompts import montar_prompt
    onto_obj = carregar_ontologia(bench, dataset, onto)
    ctx = formatar_contexto(onto_obj, cond)
    return montar_prompt(
        sentenca=frase["sent"],
        condicao=cond,
        contexto_ontologia=ctx,
        exemplo_sent=frase.get("exemplo_sent"),
        exemplo_saida=frase.get("exemplo_saida"),
    )


# ---------------------------------------------------------------------------
# Teste principal: verificação de todos os 1200 registros
# ---------------------------------------------------------------------------

@SKIP
def test_prompts_sem_contaminacao():
    """
    Reconstrói os prompts dos 1200 registros e verifica ausência de tokens de gabarito.

    Falha com mensagem detalhada listando cada violação encontrada.
    """
    erros = []
    stats = {}

    for fname, (ds, onto, cond) in DATASETS.items():
        path = os.path.join(PILOTO, fname)
        records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        frases_map = _carregar_frases_map(BENCH, ds, onto)

        vazias = dict_resp = 0
        for rec in records:
            fid = rec["id"]
            frase = frases_map[fid]
            prompt = _montar_prompt_para(frase, cond, BENCH, ds, onto)

            for tok in POISON_TOKENS:
                if tok in prompt:
                    erros.append(f"CONTAMINADO | {fname} | {fid} | token={tok!r}")

            if cond == "C0" and "CONTEXT" in prompt:
                erros.append(f"C0 COM CONTEXT | {fname} | {fid}")
            if cond == "C0" and "Example Sentence" in prompt:
                erros.append(f"C0 COM EXEMPLO | {fname} | {fid}")
            if cond == "C1" and "CONTEXT" not in prompt:
                erros.append(f"C1 SEM CONTEXT | {fname} | {fid}")
            if cond == "C1" and "Example Sentence" in prompt:
                erros.append(f"C1 COM EXEMPLO (inesperado) | {fname} | {fid}")
            if cond == "C1_oneshot_original" and "Example Sentence:" not in prompt:
                erros.append(f"C1_oneshot SEM EXEMPLO | {fname} | {fid}")

            resp = rec.get("response", "")
            if "gt_triples" in resp or ("'sent'" in resp and "'id'" in resp):
                dict_resp += 1
                erros.append(f"RESPOSTA COM DICT | {fname} | {fid} | {resp[:60]!r}")
            if len(rec.get("triples", [])) == 0:
                vazias += 1

        stats[fname] = (len(records), vazias, dict_resp)

    # Relatório de estatísticas (sempre impresso)
    for fname, (n, v, d) in stats.items():
        cond = DATASETS[fname][2]
        onto = DATASETS[fname][1]
        print(f"  {onto}/{cond:25s}: {n} frases, {v:3d} vazias ({v/n*100:5.1f}%), dict_resp={d}")

    assert not erros, (
        f"\n{len(erros)} violação(ões) encontradas:\n"
        + "\n".join(f"  {e}" for e in erros[:20])
        + (f"\n  ... e mais {len(erros)-20}" if len(erros) > 20 else "")
    )


# ---------------------------------------------------------------------------
# Exibição de prompts reais para C0, C1, C1_oneshot (mesma frase)
# ---------------------------------------------------------------------------

@SKIP
def test_exibir_prompts_mesma_frase():
    """
    Mostra prompts de C0, C1 e C1_oneshot para ont_1_movie_test_10.

    Este teste sempre passa — serve como evidência documental no log de pytest.
    """
    ID_ALVO = "ont_1_movie_test_10"
    condicoes = [
        ("piloto_tkg_1_movie_c0_qwen25_7b_instruct_awq.jsonl", "C0"),
        ("piloto_tkg_1_movie_c1_qwen25_7b_instruct_awq.jsonl", "C1"),
        ("piloto_tkg_1_movie_c1_oneshot_original_qwen25_7b_instruct_awq.jsonl", "C1_oneshot_original"),
    ]

    frases_map = _carregar_frases_map(BENCH, "tekgen", "1_movie")
    frase = frases_map.get(ID_ALVO)
    assert frase is not None, f"{ID_ALVO} não encontrado no bench"

    for fname, cond in condicoes:
        # Verifica que o ID existe no JSONL piloto
        path = os.path.join(PILOTO, fname)
        ids_jsonl = {json.loads(l)["id"] for l in open(path, encoding="utf-8") if l.strip()}
        assert ID_ALVO in ids_jsonl, f"{ID_ALVO} não está em {fname}"

        prompt = _montar_prompt_para(frase, cond, BENCH, "tekgen", "1_movie")
        trecho = prompt[:600] + ("..." if len(prompt) > 600 else "")
        # Usa buffer UTF-8 direto para evitar erros de encoding no Windows
        sys.stdout.buffer.write(
            f"\n{'='*60}\nCONDIÇÃO: {cond}\n{'='*60}\n".encode("utf-8")
        )
        sys.stdout.buffer.write(trecho.encode("utf-8", errors="replace") + b"\n")
        sys.stdout.buffer.flush()
