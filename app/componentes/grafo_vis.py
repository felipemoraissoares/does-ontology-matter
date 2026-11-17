"""Grafo interativo com pyvis (vis.js): nós arrastáveis, rótulos truncados com tooltip."""
from typing import Dict, List

from pyvis.network import Network

from .estilo import CORES


def _trunc(s, n: int = 22) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + "…"


def construir_html(nos: List[str], arestas: List[Dict], altura: int = 460) -> str:
    # cdn_resources="remote": vis.js vem do CDN (navegador faz cache) — mantém o HTML de
    # cada grafo pequeno, essencial porque a página mostra vários grafos de uma vez.
    net = Network(height=f"{altura}px", width="100%", directed=True,
                  cdn_resources="remote", bgcolor="#ffffff", font_color="#1B2430")
    net.toggle_physics(True)
    net.barnes_hut(gravity=-6000, spring_length=120, spring_strength=0.02)

    endpoints = set(nos) | {a["origem"] for a in arestas} | {a["destino"] for a in arestas}
    for n in endpoints:
        net.add_node(n, label=_trunc(n), title=str(n), color="#cfe0f2",
                     shape="dot", size=12)
    for a in arestas:
        cor = CORES.get(a["classe"], "#999999")
        net.add_edge(a["origem"], a["destino"], label=_trunc(a["rel"], 18),
                     title=f"{a['rel']} — {a['classe']}", color=cor,
                     dashes=(a["classe"] == "falta"), arrows="to")
    try:
        return net.generate_html(notebook=False)
    except TypeError:
        return net.generate_html()
