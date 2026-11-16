# Auditoria técnica pré-experimento completo

Data: 2026-11-15. Escopo: código no estado da tag `piloto-v3-fechado` (HEAD `581bb3e`).
Objetivo: identificar riscos antes do experimento completo (8 modelos × 29 ontologias ×
7 condições, ~100 h de GPU no Kaggle). **Nenhum código foi alterado nesta auditoria**;
itens que tocam decisões D-01–D-32 estão marcados como PROPOSTA.

## Verificações executadas

| Verificação | Resultado |
|---|---|
| `pytest tests` | **Falha na coleta** — `test_manifestos.py` quebra ao ler `manifesto_d32_c2_19film.json` (ver C-7). Ignorando esse arquivo: **184 passed** em 74 s. |
| Reprodução dos baselines publicados (`reproduzir_baselines.py`) | Script **aborta** no 2º experimento por caminho errado (ver I-3). Rodando com o caminho corrigido em memória: **5/6 comparações globais batem exatamente** (TekGen/Vicuna, WebNLG/Vicuna, WebNLG/Alpaca, TekGen-unseen/Vicuna, TekGen-unseen/Alpaca). O 6º (TekGen/Alpaca) não tem linha `global` no arquivo publicado do benchmark; comparação por ontologia: **10/10 ontologias × 7 métricas idênticas (0 divergências)**. O modo original do avaliador é espelho fiel do `run_eval.py`. |
| Smoke test (`smoke_kaggle.py --repo-local`, mock, 28 experimentos) | Ver resultado ao final do relatório. |

## Verificações sem achados (OK)

- **Vazamento de teste:** C3 usa `carregar_frequencias_treino` (só `train/`); indução C2 usa
  `carregar_frases_treino` e envia apenas `f["sent"]` ao modelo; calibração D-30 e definições
  D-24 usam apenas pares/rótulos de treino + rótulos da ontologia de referência (esquema
  público). Testes cobrem os caminhos: `test_c3_usa_dados_de_treino_nao_gt`,
  `test_inducao_nao_abre_arquivos_de_teste`, `test_prompt_inducao_bugado_vaza_gt`,
  `test_induzir_c2_raw_jsonl_apenas_treino`, `TestC2UsaOntologiaDaSessao`, guard check de
  `run_id` no lançador.
- **Métricas estendidas:** `avg_fmt_error` conta por tripla com `break` (sem dupla contagem,
  D-07); D-27 (classes capitalizadas), D-28 (métricas `_nv` e `pct_vazias`), D-32
  (placeholders) implementadas conforme registrado; D-30 com constantes 0.50/0.63 corretas.
- **Prompts:** um ramo explícito por condição em `montar_prompt`; condição desconhecida gera
  `ValueError` (`prompts.py:163`, `ontologias.py:104`); a instrução Variante D é a mesma
  string `_INSTRUCAO_FORMATO` em todas as condições exceto `C1_oneshot_original`, que ignora
  `instrucao_formato` (testes `test_todas_condicoes_tem_ramo_explicito`,
  `test_prompts_sem_contaminacao`).
- **Parser:** categorias A–J, placeholder `relation_name` (D-32), depth-aware (D-19),
  recusas em prosa (sem parênteses → ignoradas → contadas em `avg_parse_error`) — cobertos
  por `tests/test_parser.py` (classes TestParser*).
- **Reprodutibilidade (parcial):** `bench_commit` fixado nos manifestos e verificado no
  clone (célula 3 do lançador); `seed=42` e `temperatura=0` nos manifestos; `workers: 1`
  explícito no `manifesto_piloto_v3.json` (D-23); estatística com seeds fixas
  (`bootstrap_ci(seed=42)`).

---

## CRÍTICO — corrigir antes de rodar

### C-1. `.gitignore` não excetua `resultados/7b/` — recorrência do bug do push falso (D-17)

- **Arquivo:** `.gitignore:6-20`; `configs/pipeline/manifesto_7b_full.json` (`saida:
  "resultados/7b/..."`).
- **Problema:** a regra `resultados/*` ignora tudo e as negações listam apenas
  `piloto/`, `tests/`, `d19/`, `c2_inducao/`, `Logs/`, `piloto_v3/`, `avaliacao_piloto_v3/`.
  Verificado: `git check-ignore resultados/7b/teste.jsonl` → ignorado pela linha 6.
- **Impacto:** todo o experimento completo grava em `resultados/7b/`; `--auto_push` faria
  commits **sem os JSONLs** (o aviso de `run.py:100-105` imprime, mas não falha). É
  exatamente o bug 3 da D-17, agora no diretório da fase completa.
- **Correção:** adicionar `!resultados/7b/` e `!resultados/7b/**` (e o diretório de cada
  modelo, se houver) ao `.gitignore`; opcionalmente fazer `_push_resultado` **falhar alto**
  quando o JSONL não estiver staged.

### C-2. `--skip_existentes` trata arquivo parcial como experimento concluído

- **Arquivo:** `src/t2kg/run.py:180-182`; `kaggle/lancador.ipynb` célula 11 (usa
  `--skip_existentes`).
- **Problema:** o pulo é por `os.path.exists(exp["saida"])` — um JSONL com 37/400 frases
  (sessão derrubada) é pulado como "concluído". O checkpoint interno de `executar()`
  (que retoma por IDs) nunca roda.
- **Impacto:** combinado com a auto-detecção de `modo_amostra` em
  `avaliar_resultados.py:123` (`n_sys < n_gt` → amostra), um experimento interrompido vira
  silenciosamente um resultado "válido" com n menor — métricas plausíveis e erradas no
  experimento completo.
- **Correção:** remover `--skip_existentes` do lançador (o checkpoint por IDs já retoma e
  pula frases prontas com custo zero) ou mudar o critério para "pular só se
  `_contar_registros(saida) >= nº de frases esperado`".

### C-3. Linha JSONL truncada por queda de sessão quebra a avaliação inteira

- **Arquivo:** `src/t2kg/pipeline/executar.py:23-36` (`_ids_ja_processados` tolera linha
  inválida) vs. `src/t2kg/avaliacao/original.py:168-176` (`ler_jsonl` faz `json.loads` em
  toda linha não-vazia, sem tratamento).
- **Problema:** se a sessão Kaggle morrer no meio de um `write`, a última linha fica
  truncada. Na retomada, o checkpoint ignora a linha (correto) e reprocessa a frase, mas a
  linha corrompida **permanece no arquivo**. Na avaliação, `ler_jsonl` levanta
  `JSONDecodeError` e derruba a avaliação da condição inteira; `_contar_registros` também
  conta a linha lixo.
- **Impacto:** cenário esperado em ~100 h de Kaggle (sessões de 12 h); risco de descobrir o
  problema só na fase de análise.
- **Correção:** ao retomar em `executar()`, truncar a última linha se inválida (reescrever o
  arquivo sem ela); e/ou `ler_jsonl` pular linha inválida com aviso explícito em stderr.

### C-4. Lançador não faz push por experimento — perda de horas de GPU em queda de sessão

- **Arquivo:** `kaggle/lancador.ipynb` célula 11 (chama `t2kg.run` sem `--auto_push`);
  push só nas células 12–13, após todos os experimentos.
- **Problema:** o mecanismo de push por experimento com verificação do JSONL já existe e
  está correto (`run.py:56-105`, pós-D-17), mas o lançador atual não o usa.
- **Impacto:** sessão derrubada na hora 10 perde todos os resultados desde o início da
  sessão (a célula 4 só restaura de datasets do Kaggle, e apenas com padrões de nome do
  piloto — ver I-6).
- **Correção:** adicionar `--auto_push` ao comando da célula 11 no lançador da fase
  completa (depende de C-1 para funcionar).

### C-5. Condição C2 não existe no plano executável da fase completa — e colide entre modelos (PROPOSTA)

- **Arquivo:** `scripts/gerar_manifesto.py:26` (`CONDICOES_PRINCIPAIS` tem 6 condições, sem
  C2 → `manifesto_7b_full.json` = 29×6 = 174 entradas); `configs/c2/` (ontologias induzidas
  para apenas 4/29 ontologias); `src/t2kg/pipeline/ontologias.py:71` (nome do arquivo
  induzido é `{dataset}_{onto}_induced.json`, sem modelo);
  `configs/pipeline/reinducao_c2_kaggle.json` (4 tarefas).
- **Problema:** (a) o gerador de manifesto não produz C2, logo o experimento "7 condições"
  não é executável como está; (b) a D-16 define que a indução usa "o mesmo modelo" da
  extração — com 8 modelos, cada modelo precisa da própria ontologia induzida, mas o nome
  do arquivo e `carregar_ontologia_induzida()` não têm dimensão de modelo: induções de
  modelos diferentes se sobrescreveriam, ou um modelo usaria a C2 induzida por outro
  (contaminação silenciosa entre modelos).
- **Impacto:** quebra ou invalida a condição C2 na fase completa.
- **Correção (PROPOSTA, requer decisão):** decidir se C2 é por modelo (fiel à D-16 — exige
  29×8 induções + slug do modelo no nome do arquivo, no loader e no guard check) ou se C2
  usa um modelo indutor fixo (desvio da D-16 a registrar). Em ambos os casos, adicionar C2
  ao `gerar_manifesto.py` e gerar as tarefas de indução para as 29 ontologias.

### C-6. Versões de dependências não fixadas no Kaggle

- **Arquivo:** `kaggle/lancador.ipynb` célula 5 (`pip install -q vllm openai
  sentence-transformers ...`, tudo sem versão); não há `requirements.txt` no repo.
- **Problema:** o experimento completo rodará em muitas sessões ao longo de semanas; um
  upgrade do vLLM (kernels/scheduler novos) entre sessões muda a geração em fp16/AWQ.
- **Impacto:** resultados de modelos/condições rodados em sessões diferentes deixam de ser
  comparáveis; irreproduzível a posteriori (viola o requisito de versões fixadas).
- **Correção:** congelar versões (`requirements-kaggle.txt` com `vllm==X.Y.Z`,
  `sentence-transformers==...`, `nltk==...`, etc.) e registrar as versões efetivas
  (`pip freeze`) junto aos resultados de cada sessão.

### C-7. `pytest` não coleta — `test_manifestos.py` quebra com o formato novo de manifesto

- **Arquivo:** `tests/test_manifestos.py:14-28`; `configs/pipeline/manifesto_d32_c2_19film.json`.
- **Problema:** `_carregar_manifestos()` itera `json.load(f)` assumindo lista; o manifesto
  D-32 é dict (`{"reinduzir_c2": false, "experimentos": [...]}`) → iterar produz strings →
  `e['id']` → `TypeError` na coleta, abortando a suíte inteira (0 testes rodam).
- **Impacto:** o gate de verificação pré-experimento está quebrado; regressões nos itens
  (a)/(b)/(c) passariam despercebidas.
- **Correção:** em `_carregar_manifestos`, normalizar o formato
  (`exps = data["experimentos"] if isinstance(data, dict) else data`).

### C-8. Definições D-24 ausentes para 25/29 ontologias → limiar de projeção muda silenciosamente por ontologia

- **Arquivo:** `scripts/avaliar_resultados.py:194-198` (carrega
  `configs/c2/definicoes_{onto_id}.json` se existir; senão `definicoes_ref=None`);
  `src/t2kg/avaliacao/extended.py:146-147` (threshold 0.63 com definições, 0.50 sem).
- **Problema:** existem definições só para as 4 ontologias do piloto. Na fase completa, a
  projeção D-29 usaria label+def@0.63 em 4 ontologias e label-only@0.50 nas outras 25 —
  na mesma condição e na mesma média global, sem qualquer aviso.
- **Impacto:** `F1-proj` e `%Frac-proj` não comparáveis entre ontologias; médias globais de
  C0/C2/C3 projetadas sem significado — muda conclusões (critério (a)).
- **Correção:** incluir a geração de definições das 29 ontologias no fluxo da fase completa
  (célula 9 do lançador já tem o passo (b), basta estendê-la às 29) e fazer
  `avaliar_condicao_extendida` **falhar alto** (ou ao menos avisar por ontologia) quando a
  definição não existir. A recalibração do limiar com 29 ontologias já está prevista na
  D-30.

---

## IMPORTANTE — corrigir se for barato

### I-1. Lookup das definições D-24 falha para labels multi-palavra (representação mista)

- **Arquivo:** `scripts/gerar_definicoes_relacoes.py:58` (chaves do cache =
  `label.lower().replace(" ", "_")`, ex.: `based_on`); `src/t2kg/avaliacao/extended.py:155-159`
  e `scripts/calibrar_limiar_projecao.py:151-153` (lookup `defs[r]` com o label **cru** da
  ontologia, ex.: `based on`).
- **Problema:** verificado em `definicoes_1_movie.json`: `based on`, `cast member`,
  `publication date`, `country of origin`, `production company` etc. **nunca** encontram a
  definição; só labels de uma palavra usam label+def. A calibração D-30 usou o mesmo lookup
  → o limiar 0.63 é autoconsistente com essa representação mista (por isso não invalida o
  piloto fechado), mas a representação "label+def" na prática é parcial.
- **Correção:** normalizar a chave no lookup (mesma transformação da gravação) e
  **recalibrar** o limiar na recalibração já prevista pela D-30 para as 29 ontologias.
  Não reprocessar nada do piloto-v3 (fechado).

### I-2. C5 sem `ontologia_parceira_id` cai silenciosamente na ontologia alvo

- **Arquivo:** `src/t2kg/pipeline/executar.py:117-118`.
- **Problema:** `elif condicao in ("C5_vizinho","C5_incongruente") and ontologia_parceira_id:`
  — se o campo faltar no manifesto, `ontologia_parceira=None` e `formatar_contexto` usa a
  ontologia **alvo**: o experimento vira um C1 rotulado como C5, sem erro.
- **Correção:** levantar `ValueError` quando a condição for C5_vizinho/C5_incongruente e
  `ontologia_parceira_id` for None. (Os manifestos gerados hoje preenchem o campo; a
  validação protege contra manifesto editado à mão.)

### I-3. `reproduzir_baselines.py` aborta: caminho publicado do Alpaca TekGen inexistente

- **Arquivo:** `scripts/reproduzir_baselines.py:152-155` — aponta
  `.../Alpaca-LoRA-13B/eval_metrics/ont_llm_avg_eval_metrics.jsonl`; o arquivo real é
  `ont_llm_avg_stats.jsonl`. Além disso esse arquivo publicado não tem linha `type=global`
  (limitação do benchmark) → `comparar()` precisa cair para comparação por ontologia em vez
  de imprimir "ERRO" e retornar False.
- **Impacto:** a verificação âncora de reprodução não roda até o fim (os 4 experimentos
  após o crash só foram verificados nesta auditoria com patch em memória — todos OK).
- **Correção:** corrigir o caminho e tratar a ausência da linha global comparando apenas
  por ontologia (resultado verificado: 0 divergências em 10 ontologias × 7 métricas).

### I-4. `open()` sem `encoding="utf-8"` em caminho quente da avaliação

- **Arquivo:** `scripts/avaliar_resultados.py:81` (`_contar_registros`).
- **Problema:** no Windows abre com cp1252; bytes como `0x81/0x9D` (presentes em UTF-8 de
  títulos japoneses/letras com macron, comuns no TekGen 1_movie) levantam
  `UnicodeDecodeError`.
- **Correção:** `open(caminho, encoding="utf-8")`. Grep rápido nos demais `open()` dos
  scripts para o mesmo padrão.

### I-5. Saídas e manifestos sem dimensão de modelo — sobrescrita entre os 8 modelos

- **Arquivo:** `scripts/gerar_manifesto.py:274-276` (sempre grava
  `manifesto_7b_full.json` etc., qualquer que seja `--modelo`);
  `scripts/avaliar_resultados.py:130,144` (avaliação grava
  `{dataset}_{cond}_{onto}.jsonl` e `{dataset}_{cond}_avg.jsonl`, sem slug de modelo);
  `avaliar_resultados.py:114-117` (se um manifesto tiver 2+ modelos para a mesma ontologia,
  pega o primeiro silenciosamente).
- **Impacto:** os JSONLs de resultado em si são únicos por modelo (slug no nome — OK), mas
  (a) gerar manifesto do modelo 2 sobrescreve o do modelo 1; (b) avaliar o modelo 2 com o
  mesmo `--avaliacao_dir` sobrescreve as evidências do modelo 1.
- **Correção:** incluir o slug do modelo no nome do manifesto e no `avaliacao_dir` default
  (ou nos nomes dos arquivos de avaliação); adicionar assert de modelo único por chamada em
  `avaliar_condicao`.

### I-6. Restauração de sessão (célula 4) não cobre os arquivos da fase completa

- **Arquivo:** `kaggle/lancador.ipynb` célula 4 — padrões `piloto_*.jsonl`, `d19_*.jsonl`,
  `pv3_*.jsonl`, `raw_*.jsonl`, etc.
- **Problema:** os resultados da fase completa chamam-se `tkg_*`/`web_*` — nenhuma regra os
  restaura de uma sessão anterior.
- **Correção:** adicionar os padrões da fase completa (ou restaurar `*.jsonl` do diretório
  de resultados correspondente). Com C-4 (push por experimento) o git vira o canal
  primário de retomada e a célula 4 vira redundância — ainda assim vale corrigir.

### I-7. `FORCE_RERUN=True` como default na célula 11

- **Arquivo:** `kaggle/lancador.ipynb` célula 11.
- **Problema:** re-executar a célula (ou "Run All" numa retomada) apaga todos os JSONLs do
  manifesto atual antes de rodar.
- **Correção:** default `False` no lançador da fase completa; `True` só por edição
  deliberada.

### I-8. Loop de experimentos aborta no primeiro erro de provedor

- **Arquivo:** `src/t2kg/run.py:171-219` (sem try/except por experimento);
  `src/t2kg/pipeline/provedores/openai_compat.py:58-84` (3 tentativas × 5 s, depois
  `RuntimeError`).
- **Problema:** um soluço do vLLM > 15 s derruba o manifesto inteiro; os experimentos
  seguintes da sessão não rodam.
- **Correção:** capturar exceção por experimento, registrar e continuar (o checkpoint já
  permite re-rodar o que falhou); considerar backoff maior (ex.: 5 tentativas, espera
  exponencial) para o run longo.

### I-9. `finish_reason` não registrado — truncamentos invisíveis em modelos maiores

- **Arquivo:** `src/t2kg/pipeline/provedores/openai_compat.py:75-76` (descarta
  `finish_reason`); `executar.py:155-158` (JSONL sem o campo).
- **Problema:** `max_tokens=512` foi validado para Qwen-7B (0 truncamentos, D-19/D-23).
  Modelos de 14B/32B ou mais verbosos podem truncar; hoje isso apareceria apenas como
  parse_error/respostas estranhas, sem diagnóstico.
- **Correção:** retornar e gravar `finish_reason` (campo extra no JSONL não afeta o
  avaliador). Barato e de alto valor diagnóstico para os 8 modelos.

### I-10. `params_c2.json` desatualizado contradiz a D-21

- **Arquivo:** `configs/c2/params_c2.json` (`"freq_min": 5`); D-21 definitivo = **3**;
  `reinducao_c2_kaggle.json` usa 3 (correto). Nenhum código lê `params_c2.json` — é só
  documentação, mas a DECISOES.md o cita como "parâmetros definitivos".
- **Impacto:** risco de erro humano ao montar as tarefas de indução das 29 ontologias.
- **Correção:** atualizar para `freq_min: 3` com nota do histórico.

### I-11. Riscos de infraestrutura dos modelos novos (verificação, não bug)

- **Gemma 2 em fp16:** o modelo foi treinado em bf16; em fp16 (obrigatório na T4) há risco
  conhecido de overflow/ativações degeneradas em algumas versões de vLLM. **Verificação
  proposta:** smoke de ~5 frases por modelo no início de cada sessão, com inspeção manual
  das respostas, antes de comprometer horas de GPU (vale para os 8 modelos).
- **Memória:** 14B AWQ e 32B AWQ com `tensor-parallel-size 2` em 2×T4 (16 GB) são
  apertados mas plausíveis; variantes fp16 ≥ 27B **não cabem**. Conferir a lista final de
  modelos contra a VRAM antes de agendar; registrar `--max-model-len`/`gpu-memory-utilization`
  por modelo no manifesto.
- **Tempo:** a 0,7 s/frase (medido com 7B), 29 ontologias × 7 condições × ~100–600 frases
  ultrapassa o limite de 12 h por sessão Kaggle para vários modelos → a retomada
  (C-2/C-3/C-4/I-6) não é opcional, é o caminho normal de execução.

---

## OPCIONAL — melhorias que não mudam resultados

### O-1. Código morto e imports não usados
- `src/t2kg/pipeline/prompts.py:27-55` — constantes de módulo `_INSTRUCAO_COM_ONT`,
  `_INSTRUCAO_C3`, `_INSTRUCAO_C5`, `_INSTRUCAO_C0` nunca usadas (a função recomputa
  localmente); só `_INSTRUCAO_COM_ONT_BENCH` é usada.
- `scripts/induzir_c2.py:67-82` — `extrair_triplas_livres` não é chamada (substituída por
  `extrair_triplas_com_respostas`); import de `carregar_frases` (linha 39) não usado.
- `src/t2kg/avaliacao/extended.py:63` — `THRESHOLD_PROJECAO_PROVISORIO` (compat) sem uso
  externo.

### O-2. Documentação desatualizada em docstrings/comentários
- `extended.py:135-137` — docstring de `projetar_relacoes` diz "THRESHOLD_PROJ_DEF (0.55)";
  o valor é 0.63 (D-30 revisada).
- `parser.py` / D-15 citam `_RE_TRIPLA`, que não existe mais (regex inline em
  `_extrair_tripla`).

### O-3. `prf1_projetado` registra `"threshold": None`
- `extended.py:206-212` — quando o limiar é auto-selecionado, o dict de saída grava `None`
  em vez do valor efetivo; dificulta auditoria posterior dos JSONLs de avaliação.

### O-4. Inconsistência `None` vs `{}` em `projetar_relacoes`
- `extended.py:147` usa `definicoes_ref is not None` para escolher o limiar, mas
  `extended.py:155` usa truthiness — com `definicoes_ref={}` (arquivo de definições vazio)
  aplica-se limiar 0.63 a embeddings label-only. Alinhar os dois testes.

### O-5. Configs de avaliação escrevem dentro da árvore do benchmark
- `configs/evaluation/*.json` apontam `output` para dentro de `Text2KGBench/...`
  (vide `*_repro.jsonl` presentes no clone local). Preferir `resultados/` para manter o
  benchmark imaculado.

### O-6. Metadados fixos no manifesto independem do modelo
- `scripts/gerar_manifesto.py:85-87` — `"quantizacao": "AWQ-4bit"` e
  `"chat_template": "qwen2"` são hardcoded; ficariam errados para Gemma 2 fp16. São só
  metadados (nenhum código os lê), mas corrompem a proveniência registrada.

### O-7. Validação de exemplo ausente em C1_oneshot_original
- `prompts.py:147-153` — sem `exemplo_sent/exemplo_saida` o bloco de exemplo é omitido
  silenciosamente e a condição vira zero-shot. Para as 29 ontologias os prompts existem no
  benchmark; uma validação barata (erro se exemplo ausente) protege contra regressão de
  dados.

### O-8. Mock de definições enfraquece o smoke do agrupamento
- `gerar_definicoes_relacoes.py:134` — no smoke, todas as definições são a mesma string
  fixa, tornando os embeddings label+def quase idênticos e o agrupamento D-24 trivialmente
  permissivo. Um mock que ecoe o rótulo na definição exercitaria melhor o caminho.

### O-9. Limpeza
- `scripts/temp_tekgen_alpaca_orig_config.json` (ignorado, mas presente no working tree);
  `.pytest_cache/` versionado na raiz do repo (`llm-kg-ontology/.pytest_cache`).

---

## Resultado do smoke test

`python scripts/smoke_kaggle.py --repo-local` (ProvedorMock, 4 induções C2 + guard check +
28 experimentos com 3 frases + avaliação `--extended`):

**SMOKE TEST OK — 28/28 experimentos, C2 induzido para 4 ontologias, avaliação concluída e
CSV gerado.** (Executado em 2026-11-15 nesta auditoria, em Windows; o mesmo fluxo espelha
as células 8–13 do lançador Kaggle.)

## Observação final

Nenhum achado invalida o piloto-v3 (fechado): I-1 afeta a representação das projeções, mas
a calibração D-30 usou o mesmo código, de modo que os números publicados do piloto são
autoconsistentes. Os itens críticos são todos pré-condições do experimento completo.
