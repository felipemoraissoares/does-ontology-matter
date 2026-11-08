# Executor do pipeline de extração com suporte a checkpoint/retomada.
#
# Formato do JSONL de saída (compatível com o avaliador original):
#   {"id": "...", "sent": "...", "response": "...", "triples": [[sub, rel, obj], ...]}
#
# Checkpoint: ao retomar, lê os IDs já presentes no arquivo de saída e pula essas frases.

import json
import os
from typing import Dict, List, Optional

from ..parser import parsear_resposta
from .dados import carregar_frases, carregar_frequencias_treino
from .ontologias import carregar_ontologia, formatar_contexto
from .prompts import montar_prompt
from .provedores.base import ProvedorBase

_K_RELS = 3  # número de relações para C3


def _ids_ja_processados(caminho_saida: str) -> set:
    """Lê IDs já escritos no arquivo de saída (para checkpoint)."""
    if not os.path.exists(caminho_saida):
        return set()
    ids = set()
    with open(caminho_saida, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                try:
                    ids.add(json.loads(linha)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


def _registrar(caminho_saida: str, entrada: Dict) -> None:
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True) if os.path.dirname(caminho_saida) else None
    with open(caminho_saida, "a", encoding="utf-8") as f:
        f.write(json.dumps(entrada, ensure_ascii=False) + "\n")


def executar(
    bench_dir: str,
    dataset: str,
    split: str,
    ontologia_id: str,
    condicao: str,
    provedor: ProvedorBase,
    saida: str,
    max_tokens: int = 512,
    temperatura: float = 0.0,
    frases_max: Optional[int] = None,
    verbose: bool = False,
    ontologia_parceira_id: Optional[str] = None,
) -> int:
    """
    Executa a extração de triplas para uma (ontologia, condição) e salva em JSONL.

    Retorna o número de frases processadas nesta chamada (excluindo checkpoint).

    Parâmetros
    ----------
    bench_dir : str
        Caminho para o diretório raiz do Text2KGBench.
    dataset : str
        "tekgen" | "webnlg"
    split : str
        "test" | "unseen"
    ontologia_id : str
        ex.: "1_movie"
    condicao : str
        C0 | C1 | C1_oneshot_original | C3 | C5 | C5_vizinho | C5_incongruente
    provedor : ProvedorBase
        Instância do provedor de inferência.
    saida : str
        Caminho do arquivo JSONL de saída.
    frases_max : int | None
        Limita o total de frases processadas (inclui checkpoint).
    verbose : bool
        Imprime progresso.
    ontologia_parceira_id : str | None
        Para C5_vizinho/C5_incongruente: ID da ontologia cujo JSON será fornecido ao modelo.
        Se None em C5_vizinho/C5_incongruente, usa a própria ontologia alvo.
    """
    frases = carregar_frases(bench_dir, dataset, split, ontologia_id)
    ontologia = carregar_ontologia(bench_dir, dataset, ontologia_id)

    # Para C5_vizinho/C5_incongruente carrega a ontologia parceira
    ontologia_parceira = None
    if condicao in ("C5_vizinho", "C5_incongruente") and ontologia_parceira_id:
        ontologia_parceira = carregar_ontologia(bench_dir, dataset, ontologia_parceira_id)

    # Para C3: top-k relações extraídas dos dados de TREINO (nunca do GT de teste)
    top_k_rels = None
    if condicao == "C3":
        top_k_rels = carregar_frequencias_treino(bench_dir, dataset, ontologia_id, k=_K_RELS)

    contexto = formatar_contexto(
        ontologia, condicao,
        top_k_rels=top_k_rels,
        ontologia_parceira=ontologia_parceira,
    )

    ja_prontos = _ids_ja_processados(saida)
    processados = 0

    for frase in frases:
        sent_id = frase["id"]

        # Aplica limite total (checkpoint + novos)
        if frases_max is not None and (len(ja_prontos) + processados) >= frases_max:
            break

        if sent_id in ja_prontos:
            continue

        prompt = montar_prompt(
            sentenca=frase["sent"],
            condicao=condicao,
            contexto_ontologia=contexto,
            exemplo_sent=frase.get("exemplo_sent"),
            exemplo_saida=frase.get("exemplo_saida"),
        )

        resposta = provedor.gerar(prompt, max_tokens=max_tokens, temperatura=temperatura)
        triplas = parsear_resposta(resposta)

        _registrar(
            saida,
            {
                "id": sent_id,
                "sent": frase["sent"],
                "response": resposta,
                "triples": triplas,
            },
        )
        processados += 1

        if verbose:
            print(f"  [{processados}] {sent_id}: {len(triplas)} tripla(s)")

    return processados
