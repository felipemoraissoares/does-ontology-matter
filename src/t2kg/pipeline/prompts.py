# Monta o prompt completo para cada condição experimental.
#
# Os prompts são entregues como mensagem de usuário à API de chat.
# O texto segue o mesmo estilo do benchmark para C1_oneshot_original
# e variantes consistentes para as demais condições.

from typing import Optional


# Instrução de entidade: evita que o modelo substitua entidades reais por classes da ontologia.
# Aplicada a C0, C1, C2, C3, C5*. NÃO aplicada a C1_oneshot_original (fidelidade ao benchmark).
_INSTRUCAO_ENTIDADE = (
    "Use as subject and object the exact entity names as written in the sentence, "
    "never the ontology class names."
)

# Instrução de formato — Variante D (D-20, aprovada 2026-11-08).
# Substituiu D-18 (Variante A: "relation(subject, object). Do not write any other text.").
# Vencedora do mini-teste D-19 com score=0.1505 (menor fmt+parse error).
# NÃO aplicada a C1_oneshot_original (fidelidade ao benchmark).
_INSTRUCAO_FORMATO = (
    "Output the triples, one per line, in this exact format: "
    "relation_name(<exact name of the first entity as written in the sentence>, "
    "<exact name of the second entity as written in the sentence>). No other text."
)

_INSTRUCAO_COM_ONT = (
    "Given the following ontology and sentence, please extract the triples "
    "from the sentence according to the relations in the ontology. "
    + _INSTRUCAO_FORMATO + " "
    + _INSTRUCAO_ENTIDADE
)
# O benchmark usa "sentences" (plural) — mantido para C1_oneshot_original; sem instrução de entidade.
_INSTRUCAO_COM_ONT_BENCH = (
    "Given the following ontology and sentences, please extract the triples "
    "from the sentence according to the relations in the ontology. "
    "In the output, only include the triples in the given output format."
)
_INSTRUCAO_C3 = (
    "Given the following sentence and list of key relations, please extract the triples. "
    "You may use relation names not in the list when the text explicitly indicates them. "
    + _INSTRUCAO_FORMATO + " "
    + _INSTRUCAO_ENTIDADE
)
_INSTRUCAO_C5 = (
    "Given the following ontology and sentence, please extract the triples "
    "according to the relations defined in the ontology. "
    + _INSTRUCAO_FORMATO + " "
    + _INSTRUCAO_ENTIDADE
)
_INSTRUCAO_C0 = (
    "Given the following sentence, please extract all relevant triples. "
    + _INSTRUCAO_FORMATO + " "
    + _INSTRUCAO_ENTIDADE
)


def montar_prompt(
    sentenca: str,
    condicao: str,
    contexto_ontologia: Optional[str] = None,
    exemplo_sent: Optional[str] = None,
    exemplo_saida: Optional[str] = None,
    instrucao_formato: Optional[str] = None,
) -> str:
    """
    Monta o prompt completo como string (mensagem de usuário).

    Parâmetros
    ----------
    sentenca : str
        Sentença de teste.
    condicao : str
        C0 | C1 | C1_oneshot_original | C3 | C5 | C5_vizinho | C5_incongruente
    contexto_ontologia : str | None
        Bloco de contexto retornado por `formatar_contexto()`.
        None para C0.
    exemplo_sent : str | None
        Sentença de exemplo (apenas para C1_oneshot_original).
    exemplo_saida : str | None
        Saída esperada do exemplo (apenas para C1_oneshot_original).
    instrucao_formato : str | None
        Sobrescreve _INSTRUCAO_FORMATO (D-19: mini-teste de redações alternativas).
        Ignorado para C1_oneshot_original. Se None, usa o padrão D-18.
    """
    fmt = instrucao_formato if instrucao_formato is not None else _INSTRUCAO_FORMATO

    inst_c0 = (
        "Given the following sentence, please extract all relevant triples. "
        + fmt + " " + _INSTRUCAO_ENTIDADE
    )
    inst_c1 = (
        "Given the following ontology and sentence, please extract the triples "
        "from the sentence according to the relations in the ontology. "
        + fmt + " " + _INSTRUCAO_ENTIDADE
    )
    inst_c3 = (
        "Given the following sentence and list of key relations, please extract the triples. "
        "You may use relation names not in the list when the text explicitly indicates them. "
        + fmt + " " + _INSTRUCAO_ENTIDADE
    )
    inst_c5 = (
        "Given the following ontology and sentence, please extract the triples "
        "according to the relations defined in the ontology. "
        + fmt + " " + _INSTRUCAO_ENTIDADE
    )

    if condicao == "C0":
        return (
            f"{inst_c0}\n\n"
            f"Test Sentence: {sentenca}\n"
            f"Test Output: "
        )

    if condicao == "C3":
        return (
            f"{inst_c3}\n\n"
            f"CONTEXT:\n"
            f"{contexto_ontologia}\n\n"
            f"Test Sentence: {sentenca}\n"
            f"Test Output: "
        )

    if condicao in ("C5", "C5_vizinho", "C5_incongruente"):
        return (
            f"{inst_c5}\n\n"
            f"CONTEXT:\n"
            f"{contexto_ontologia}\n\n"
            f"Test Sentence: {sentenca}\n"
            f"Test Output: "
        )

    # C1 e C2 — zero-shot com ontologia (C2 usa ontologia induzida, mesmo layout que C1)
    if condicao in ("C1", "C2"):
        return (
            f"{inst_c1}\n\n"
            f"CONTEXT:\n"
            f"{contexto_ontologia}\n\n"
            f"Test Sentence: {sentenca}\n"
            f"Test Output: "
        )

    # C1_oneshot_original — replica o formato exato do benchmark
    # Inclui "Test Output: " no final para ser idêntico ao prompt publicado.
    # Para modelos de chat, "Test Output: " sinaliza ao modelo que ele deve
    # continuar a partir daí, sem necessidade de prefixo adicional.
    ex_bloco = ""
    if exemplo_sent and exemplo_saida:
        ex_bloco = (
            f"\nExample Sentence: {exemplo_sent}\n"
            f"Example Output: {exemplo_saida}\n"
        )

    return (
        f"{_INSTRUCAO_COM_ONT_BENCH}\n"
        f"CONTEXT:\n"
        f"{contexto_ontologia}\n"
        f"{ex_bloco}"
        f"\nTest Sentence: {sentenca}\n"
        f"Test Output: "
    )
