# Executor do pipeline de extração com suporte a checkpoint/retomada.
#
# Formato do JSONL de saída (compatível com o avaliador original):
#   {"id": "...", "sent": "...", "response": "...", "triples": [[sub, rel, obj], ...]}
#
# Checkpoint: ao retomar, lê os IDs já presentes no arquivo de saída e pula essas frases.

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    frases_seed: Optional[int] = None,
    verbose: bool = False,
    ontologia_parceira_id: Optional[str] = None,
    workers: int = 1,
    instrucao_formato: Optional[str] = None,
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
    frases_seed : int | None
        Semente para embaralhar frases antes de aplicar frases_max.
        Garante subsample reprodutível (estratificado por posição aleatória).
        Se None, mantém a ordem original dos dados.
    verbose : bool
        Imprime progresso.
    ontologia_parceira_id : str | None
        Para C5_vizinho/C5_incongruente: ID da ontologia cujo conteúdo será
        fornecido ao modelo. Se None, usa a própria ontologia alvo.
    workers : int
        Número de requisições paralelas ao provedor (padrão 1 = sequencial).
        Com temperatura=0 e greedy decoding, o resultado por frase é determinístico
        e independente da ordem de execução — parallelismo não altera as respostas.
        A saída é sempre escrita na ordem original das frases.
    instrucao_formato : str | None
        Sobrescreve a instrução de formato padrão (D-19: mini-teste).
        None = usa o padrão definido em prompts.py.
    """
    frases = carregar_frases(bench_dir, dataset, split, ontologia_id)

    # Embaralhamento reprodutível para subsample estratificado (item E)
    if frases_seed is not None:
        rng = random.Random(frases_seed)
        frases = list(frases)
        rng.shuffle(frases)
    ontologia = carregar_ontologia(bench_dir, dataset, ontologia_id)

    # Para C2: carrega ontologia induzida (D-16); para C5_vizinho/C5_incongruente: parceira
    ontologia_parceira = None
    if condicao == "C2":
        from .ontologias import carregar_ontologia_induzida
        ontologia_parceira = carregar_ontologia_induzida(dataset, ontologia_id)
    elif condicao in ("C5_vizinho", "C5_incongruente") and ontologia_parceira_id:
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

    # Filtra frases que ainda precisam ser processadas
    pendentes: List[Dict] = []
    for frase in frases:
        if frases_max is not None and (len(ja_prontos) + len(pendentes)) >= frases_max:
            break
        if frase["id"] not in ja_prontos:
            pendentes.append(frase)

    if workers <= 1:
        # Modo sequencial — comportamento original
        for frase in pendentes:
            prompt = montar_prompt(
                sentenca=frase["sent"],
                condicao=condicao,
                contexto_ontologia=contexto,
                exemplo_sent=frase.get("exemplo_sent"),
                exemplo_saida=frase.get("exemplo_saida"),
                instrucao_formato=instrucao_formato,
            )
            resposta = provedor.gerar(prompt, max_tokens=max_tokens, temperatura=temperatura)
            triplas = parsear_resposta(resposta)
            _registrar(saida, {
                "id": frase["id"], "sent": frase["sent"],
                "prompt": prompt, "response": resposta, "triples": triplas,
            })
            processados += 1
            if verbose:
                print(f"  [{processados}] {frase['id']}: {len(triplas)} tripla(s)")
    else:
        # Modo paralelo — mantém a ordem de saída
        # Com temperatura=0 e greedy decoding, resultado por frase é determinístico.
        def _processar_frase(frase: Dict):
            prompt = montar_prompt(
                sentenca=frase["sent"],
                condicao=condicao,
                contexto_ontologia=contexto,
                exemplo_sent=frase.get("exemplo_sent"),
                exemplo_saida=frase.get("exemplo_saida"),
                instrucao_formato=instrucao_formato,
            )
            resposta = provedor.gerar(prompt, max_tokens=max_tokens, temperatura=temperatura)
            return frase, prompt, resposta, parsear_resposta(resposta)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_processar_frase, f) for f in pendentes]
            for future in futures:
                frase, prompt, resposta, triplas = future.result()
                _registrar(saida, {
                    "id": frase["id"], "sent": frase["sent"],
                    "prompt": prompt, "response": resposta, "triples": triplas,
                })
                processados += 1
                if verbose:
                    print(f"  [{processados}] {frase['id']}: {len(triplas)} tripla(s)")

    return processados
