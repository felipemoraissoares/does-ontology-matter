from abc import ABC, abstractmethod


class ProvedorBase(ABC):
    @abstractmethod
    def gerar(self, prompt: str, max_tokens: int = 512, temperatura: float = 0.0) -> str:
        """Envia o prompt e retorna o texto gerado pelo modelo."""
        ...
