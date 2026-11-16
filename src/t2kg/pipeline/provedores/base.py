from abc import ABC, abstractmethod
from typing import Optional, Tuple


class ProvedorBase(ABC):
    @abstractmethod
    def gerar(self, prompt: str, max_tokens: int = 512, temperatura: float = 0.0) -> str:
        """Envia o prompt e retorna o texto gerado pelo modelo."""
        ...

    def gerar_detalhado(
        self, prompt: str, max_tokens: int = 512, temperatura: float = 0.0
    ) -> Tuple[str, Optional[str]]:
        """Retorna (texto, finish_reason). Default: sem finish_reason (None).

        Thread-safe: cada chamada é independente (não usa estado compartilhado),
        ao contrário de um atributo de instância — importante para workers > 1.
        """
        return self.gerar(prompt, max_tokens=max_tokens, temperatura=temperatura), None
