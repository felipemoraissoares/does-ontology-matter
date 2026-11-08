# Provedor mock para testes locais sem modelo real.
# Retorna triplas fictícias derivadas do prompt para permitir testes do pipeline.

import re

from .base import ProvedorBase


class ProvedorMock(ProvedorBase):
    """
    Retorna uma resposta fixa ou derivada do prompt.
    Usado em testes de integração do pipeline (checkpoint, formato de saída, etc.)
    sem necessitar de um servidor LLM.
    """

    def __init__(self, resposta_fixa: str = "mock_rel(Mock Subject, Mock Object)"):
        self._resposta = resposta_fixa

    def gerar(self, prompt: str, max_tokens: int = 512, temperatura: float = 0.0) -> str:
        # Extrai a sentença de teste para personalizar minimamente a resposta
        m = re.search(r"Test Sentence:\s*(.+?)$", prompt, re.MULTILINE)
        if m:
            sentenca = m.group(1).strip()[:60]
            return f"mock_rel({sentenca}, mock_object)"
        return self._resposta
