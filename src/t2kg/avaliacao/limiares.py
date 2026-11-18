"""Fonte única dos limiares de avaliação (D-36).

Os valores ficam em `configs/avaliacao.json`. O limiar de projeção de relações é
explícito por rodada:

    piloto_v3 -> 0.63   (modo label+def do piloto, D-24/D-30)
    full      -> 0.45   (recalibrado nas 29 ontologias, D-36)

Tanto `scripts/avaliar_full.py` quanto o app leem daqui — nunca redeclaram os números.
"""
import json
import os
from functools import lru_cache

_CFG = os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "avaliacao.json")

# Fallbacks usados só se o JSON estiver ausente (ambiente mínimo / testes isolados).
_FALLBACK = {"piloto_v3": 0.63, "full": 0.45}
_FALLBACK_DEFAULT = 0.45
_FALLBACK_REDUND = 0.80


@lru_cache(maxsize=1)
def _cfg() -> dict:
    try:
        with open(_CFG, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def limiar_projecao(rodada: str = "full") -> float:
    """Limiar de projeção de relações para a rodada ('piloto_v3' ou 'full')."""
    cfg = _cfg()
    mapa = cfg.get("limiar_projecao", _FALLBACK)
    default = cfg.get("limiar_projecao_default", _FALLBACK_DEFAULT)
    return float(mapa.get(rodada, default))


def limiar_projecao_por_rid(rid: str) -> float:
    """Resolve o limiar a partir do id de rodada do app (ex.: 'piloto-v3', 'full/qwen2.5-7b')."""
    chave = "piloto_v3" if rid and "piloto" in rid else "full"
    return limiar_projecao(chave)


def limiar_redundancia() -> float:
    """Limiar de agrupamento da redundância semântica (H1)."""
    return float(_cfg().get("limiar_redundancia", _FALLBACK_REDUND))
