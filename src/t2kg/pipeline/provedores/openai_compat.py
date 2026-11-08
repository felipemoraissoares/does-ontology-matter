# Provedor compatível com a API OpenAI (vLLM, Ollama, LM Studio, etc.)
# Requer: pip install openai

import time
from typing import Optional

from .base import ProvedorBase


class ProvedorOpenAICompat(ProvedorBase):
    """
    Acessa qualquer servidor com API compatível com OpenAI
    (vLLM com --api-key, Ollama, LM Studio, etc.).
    """

    def __init__(
        self,
        url: str,
        modelo: str,
        api_key: str = "token-nao-usado",
        sistema: Optional[str] = None,
        max_tentativas: int = 3,
        espera_tentativa: float = 5.0,
    ):
        """
        Parâmetros
        ----------
        url : str
            URL base da API, ex.: "http://localhost:8000/v1"
        modelo : str
            Nome do modelo como definido no servidor.
        api_key : str
            Chave de API (vLLM aceita qualquer string).
        sistema : str | None
            Mensagem de sistema opcional.
        max_tentativas : int
            Número de tentativas em caso de falha de rede.
        espera_tentativa : float
            Segundos entre tentativas.
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "O pacote 'openai' é necessário. Instale com: pip install openai"
            ) from e

        from openai import OpenAI  # noqa: F811

        self._cliente = OpenAI(base_url=url, api_key=api_key)
        self._modelo = modelo
        self._sistema = sistema
        self._max_tentativas = max_tentativas
        self._espera = espera_tentativa

    def gerar(self, prompt: str, max_tokens: int = 512, temperatura: float = 0.0) -> str:
        mensagens = []
        if self._sistema:
            mensagens.append({"role": "system", "content": self._sistema})
        mensagens.append({"role": "user", "content": prompt})

        ultimo_erro = None
        for tentativa in range(self._max_tentativas):
            try:
                resposta = self._cliente.chat.completions.create(
                    model=self._modelo,
                    messages=mensagens,
                    max_tokens=max_tokens,
                    temperature=temperatura,
                )
                return resposta.choices[0].message.content or ""
            except Exception as e:
                ultimo_erro = e
                if tentativa < self._max_tentativas - 1:
                    time.sleep(self._espera)

        raise RuntimeError(
            f"Falha após {self._max_tentativas} tentativas: {ultimo_erro}"
        ) from ultimo_erro
