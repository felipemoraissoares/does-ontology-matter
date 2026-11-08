# Monta o prompt completo para cada condição experimental.
#
# Os prompts são entregues como mensagem de usuário à API de chat.
# O texto segue o mesmo estilo do benchmark para C1_oneshot_original
# e variantes consistentes para as demais condições.

from typing import Optional


_INSTRUCAO_COM_ONT = (
    "Given the following ontology and sentence, please extract the triples "
    "from the sentence according to the relations in the ontology. "
    "In the output, only include the triples in the given output format."
)
# O benchmark usa "sentences" (plural) — mantido para C1_oneshot_original
_INSTRUCAO_COM_ONT_BENCH = (
    "Given the following ontology and sentences, please extract the triples "
    "from the sentence according to the relations in the ontology. "
    "In the output, only include the triples in the given output format."
)
_INSTRUCAO_C3 = (
    "Given the following sentence and list of relations, please extract the triples. "
    "In the output, only include the triples in the format rel(subject, object), "
    "one triple per line."
)
_INSTRUCAO_C5 = (
    "Given the following ontology (in JSON format) and sentence, please extract "
    "the triples according to the relations defined in the ontology. "
    "In the output, only include the triples in the format rel(subject, object), "
    "one triple per line."
)
_INSTRUCAO_C0 = (
    "Given the following sentence, please extract all relevant triples. "
    "In the output, only include the triples in the format rel(subject, object), "
    "one triple per line."
)


def montar_prompt(
    sentenca: str,
    condicao: str,
    contexto_ontologia: Optional[str] = None,
    exemplo_sent: Optional[str] = None,
    exemplo_saida: Optional[str] = None,
) -> str:
    """
    Monta o prompt completo como string (mensagem de usuário).

    Parâmetros
    ----------
    sentenca : str
        Sentença de teste.
    condicao : str
        C0 | C1 | C1_oneshot_original | C3 | C5
    contexto_ontologia : str | None
        Bloco de contexto retornado por `formatar_contexto()`.
        None para C0.
    exemplo_sent : str | None
        Sentença de exemplo (apenas para C1_oneshot_original).
    exemplo_saida : str | None
        Saída esperada do exemplo (apenas para C1_oneshot_original).
    """
    if condicao == "C0":
        return (
            f"{_INSTRUCAO_C0}\n\n"
            f"Test Sentence: {sentenca}"
        )

    if condicao == "C3":
        return (
            f"{_INSTRUCAO_C3}\n\n"
            f"CONTEXT:\n"
            f"{contexto_ontologia}\n\n"
            f"Test Sentence: {sentenca}"
        )

    if condicao == "C5":
        return (
            f"{_INSTRUCAO_C5}\n\n"
            f"CONTEXT:\n"
            f"{contexto_ontologia}\n\n"
            f"Test Sentence: {sentenca}"
        )

    # C1 — zero-shot com ontologia completa
    if condicao == "C1":
        return (
            f"{_INSTRUCAO_COM_ONT}\n\n"
            f"CONTEXT:\n"
            f"{contexto_ontologia}\n\n"
            f"Test Sentence: {sentenca}"
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
