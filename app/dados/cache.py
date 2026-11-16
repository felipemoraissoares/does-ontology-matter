"""Cache simples em disco (app/.cache/, no .gitignore) para métricas calculadas."""
import hashlib
import json
import os

from .base import CACHE_DIR


def _caminho(chave: str) -> str:
    h = hashlib.sha1(chave.encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{h}.json")


def ler(chave: str, mtime: float):
    """Retorna o valor em cache se existir e for >= mtime da fonte; senão None."""
    p = _caminho(chave)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("mtime", 0) >= mtime:
            return d.get("valor")
    except (json.JSONDecodeError, OSError):
        return None
    return None


def gravar(chave: str, mtime: float, valor) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(_caminho(chave), "w", encoding="utf-8") as f:
            json.dump({"mtime": mtime, "valor": valor}, f, ensure_ascii=False)
    except OSError:
        pass
