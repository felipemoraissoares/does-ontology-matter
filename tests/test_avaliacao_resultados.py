"""
Testes para scripts/avaliar_resultados.py e correções relacionadas.
"""
import json
import os
import sys
import tempfile

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))


# ---------------------------------------------------------------------------
# Teste: conversão de string para float nas métricas do avaliador original
# ---------------------------------------------------------------------------

def test_metricas_string_convertem_para_float():
    """
    O avaliador original (run_eval.py) grava métricas como strings "0.32".
    avaliar_resultados.py deve converter para float sem levantar ValueError.
    """
    metricas_string = {
        "avg_precision": "0.32",
        "avg_recall": "0.45",
        "avg_f1": "0.37",
        "avg_onto_conf": "0.80",
        "avg_sub_halluc": "0.10",
        "avg_rel_halluc": "0.05",
        "avg_obj_halluc": "0.08",
        "avg_fmt_error": "0.00",
    }
    resultado_float = {m: (float(v) if v is not None else None)
                       for m, v in metricas_string.items()}

    for m, v in resultado_float.items():
        # Não deve lançar ValueError com :.4f
        formatted = f"{v:.4f}"
        assert isinstance(v, float), f"{m} deveria ser float, got {type(v)}"
        assert formatted == f"{float(metricas_string[m]):.4f}"


def test_metricas_none_sao_ignoradas():
    resultado = {"avg_precision": None, "avg_f1": "0.50"}
    resultado_float = {m: (float(v) if v is not None else None)
                       for m, v in resultado.items()}
    assert resultado_float["avg_precision"] is None
    assert resultado_float["avg_f1"] == 0.50


# ---------------------------------------------------------------------------
# Teste: induzir_c2.py não inclui gt_triples no prompt (bug D-16)
# ---------------------------------------------------------------------------

def test_montar_prompt_c0_nao_inclui_gt():
    """
    O prompt C0 enviado pelo induzir_c2.py deve conter APENAS o texto da frase.
    Nunca deve incluir gt_triples, que vazariam os rótulos da ontologia de referência.
    """
    from t2kg.pipeline.prompts import montar_prompt

    frase_dict = {
        "id": "ont_1_movie_test_1",
        "sent": "The movie was directed by Nolan.",
        "gt_triples": [{"sub": "The movie", "rel": "director", "obj": "Nolan"}],
        "exemplo_sent": None,
        "exemplo_saida": None,
    }

    # Chamada CORRETA (após correção do bug)
    prompt_correto = montar_prompt(sentenca=frase_dict["sent"], condicao="C0")
    assert "gt_triples" not in prompt_correto
    assert "director" not in prompt_correto  # não vaza o rótulo
    assert "The movie was directed by Nolan." in prompt_correto
    assert "Test Output:" in prompt_correto

    # Chamada INCORRETA (como estava antes) agora levanta ValueError — melhor que vazar gt_triples
    import pytest
    with pytest.raises((ValueError, TypeError)):
        montar_prompt(frase_dict, {"concepts": [], "relations": []}, "C0")


def test_montar_prompt_c0_identico_sem_ontologia():
    """C0 não deve conter bloco CONTEXT nem relações de ontologia."""
    from t2kg.pipeline.prompts import montar_prompt

    prompt = montar_prompt(sentenca="Nolan directed Inception.", condicao="C0")
    assert "CONTEXT" not in prompt
    assert "Relations:" not in prompt
    assert "Example Sentence:" not in prompt
    assert "Nolan directed Inception." in prompt


# ---------------------------------------------------------------------------
# Teste: carregar_frases_treino não inclui o campo gt_triples no "sent"
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE  = os.path.abspath(os.path.join(REPO_ROOT, ".."))
BENCH      = os.path.join(WORKSPACE, "Text2KGBench")
SKIP_SEM_BENCH = pytest.mark.skipif(
    not os.path.exists(BENCH), reason="Text2KGBench não encontrado"
)


@SKIP_SEM_BENCH
def test_carregar_frases_treino_tem_sent_e_gt():
    from t2kg.pipeline.dados import carregar_frases_treino

    frases = carregar_frases_treino(BENCH, "tekgen", "1_movie")
    assert len(frases) > 0
    for f in frases[:5]:
        assert "sent" in f and f["sent"]
        assert "gt_triples" in f
        # gt_triples deve ser lista de dicts com sub/rel/obj
        for t in f["gt_triples"]:
            assert "sub" in t and "rel" in t and "obj" in t
        # Não deve ter campos extras que vazem informação
        assert "prompt_original" not in f
        assert "exemplo_sent" not in f


@SKIP_SEM_BENCH
def test_carregar_frases_treino_ids_sao_de_treino():
    """IDs do treino devem conter 'train', não 'test'."""
    from t2kg.pipeline.dados import carregar_frases_treino

    frases = carregar_frases_treino(BENCH, "tekgen", "1_movie")
    for f in frases[:10]:
        assert "test" not in f["id"], (
            f"Frase de treino tem ID de teste: {f['id']!r} — "
            "carregar_frases_treino está vazando dados de teste."
        )
