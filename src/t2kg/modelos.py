"""Utilidades de identificação de modelo (D-33).

O experimento completo tem 8 modelos; cada um induz a própria ontologia C2 e gera as
próprias definições D-24. O `slug` dá um nome curto e estável para o subdiretório por
modelo em `configs/c2/v_full/<slug>/`.
"""

# Sufixos de variante removidos do slug (ordem não importa; removidos repetidamente).
_SUFIXOS = (
    "-instruct", "-chat", "-it", "-awq", "-gptq", "-gguf", "-hf", "-bnb-4bit",
)


def slug_modelo(modelo: str) -> str:
    """Slug curto e estável de um id de modelo HuggingFace.

    Regra (D-33): último componente após '/', minúsculo, removendo os sufixos de
    variante (`-instruct`, `-chat`, `-it`, `-awq`, `-gptq`, `-gguf`, `-hf`, `-bnb-4bit`).

    Exemplos:
        Qwen/Qwen2.5-7B-Instruct-AWQ  -> qwen2.5-7b
        google/gemma-2-9b-it          -> gemma-2-9b
        Qwen/Qwen2.5-32B-Instruct-AWQ -> qwen2.5-32b
    """
    s = modelo.split("/")[-1].lower()
    mudou = True
    while mudou:
        mudou = False
        for suf in _SUFIXOS:
            if s.endswith(suf):
                s = s[: -len(suf)]
                mudou = True
    return s
