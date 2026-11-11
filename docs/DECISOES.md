# Decisões metodológicas — Fases 1 e 2

Decisões aprovadas que orientam a implementação e o design experimental.
Cada decisão substitui as propostas anteriores onde houver conflito.

---

## Índice

| # | Título | Status | Substituída por |
|---|---|---|---|
| D-01 | Formato do contexto C3 — k=3 com domínio/range | APROVADA | — |
| D-02 | Condição C5 — dois níveis de perturbação de ontologia | APROVADA | — |
| D-03 | Sufixo "Test Output: " em todos os prompts | APROVADA | — |
| D-04 | Modo de avaliação — benchmark vs. amostra | APROVADA | — |
| D-05 | Instrução de entidade — desvio controlado do prompt original | APROVADA | — |
| D-06 | Fonte de dados para C3 k=3 — dados de treino, nunca GT | APROVADA | — |
| D-07 | Métrica auxiliar — taxa de erro de formato (avg_fmt_error) | APROVADA | — |
| D-08 | Diferença de tokens como covariável na análise de C5 | APROVADA | — |
| D-09 | SimE como variável contínua na análise de C5 | APROVADA | — |
| D-10 | Análise de sensibilidade — C5_incongruente com segunda parceira | APROVADA | — |
| D-11 | Terminologia — "similaridade de esquema" | APROVADA | — |
| D-12 | Distinção entre zero-shot e C1_oneshot_original | APROVADA | — |
| D-13 | C0 bruta e C0 projetada | APROVADA | — |
| D-14 | Avaliação sempre contra a ontologia de referência do alvo | APROVADA | — |
| D-15 | Correção do parser — Categoria J | APROVADA | — |
| D-16 | Condição C2 — ontologia induzida a partir de extração livre | APROVADA | — |
| D-17 | Bugs encontrados no piloto Kaggle — C2 inválida, push falso | REGISTRADO | — |
| D-18 | Instrução de formato unificada e métrica de erro de parse | APROVADA | D-20 (instrução) |
| D-19 | Parser depth-aware, workers paralelos, mini-teste de instrução | IMPLEMENTADO | D-20 (instrução), D-23 (workers) |
| D-20 | Regra pré-registrada de escolha da instrução de formato | RESULTADO APURADO | — |
| D-21 | Threshold de agrupamento e freq_min para C2 | RESULTADO APURADO | — |
| D-22 | Normalização de rótulos antes do embedding | APROVADA | — |
| D-23 | Paralelismo — workers=1 fixado em todos os manifestos | APROVADA | — |
| D-24 | Definição semântica para agrupamento e projeção C2 (EDC) | EXECUTADA | — |
| D-25 | Formato do bloco de ontologia em C2 | IMPLEMENTADA | — |
| D-26 | Detecção de modo amostra por ontologia | APROVADA | — |
| D-27 | Correção de FMT_ERR para classes capitalizadas (DBpedia) | APROVADA | — |
| D-28 | Métricas de conformidade/alucinação sem respostas vazias | APROVADA | — |
| D-29 | Embeddings D-24 na projeção de relações (C0/C2/C3) | APROVADA | — |
| D-30 | Recalibração do limiar de projeção para formato D-24 | APROVADA | — |
| D-31 | Piloto-v3 — validação antes da fase completa | EXECUTADA | — |
| D-32 | Placeholders da instrução Variante D tratados como fmt_error | IMPLEMENTADA | — |

---

## D-01: Formato do contexto para a condição C3 — k=3 com domínio/range

**Status:** APROVADA

### Decisão

C3 fornece ao modelo as **3 relações mais frequentes no conjunto de treino** (vide D-06),
acompanhadas de seus conceitos de domínio e alcance. Formato idêntico ao C1 mas sem
listar conceitos nem usar exemplo.

```
Relations: director(film,human), screenwriter(film,human), genre(film,genre)
```

(Exemplo com as top-3 do **treino** para 1_movie TekGen: director, screenwriter, genre.
Vide D-06 para a comparação com as top-3 do GT de teste, que são distintas.)

**Justificativa:** k=3 isola o efeito de nomear as relações mais salientes sem revelar
a estrutura completa da ontologia. A inclusão de domain/range fornece contexto mínimo
sem revelar a lista completa de conceitos. k=0 seria indistinguível de C0 se o modelo
não souber quais relações extrair.

**Implementação:** `executar.py` computa as top-3 relações a partir dos **dados de treino**
(`carregar_frequencias_treino`) antes de iniciar a inferência e passa para `formatar_contexto()` via
`top_k_rels`. Constante `_K_RELS = 3` em `executar.py`.

---

## D-02: Condição C5 — dois níveis de perturbação de ontologia

**Status:** APROVADA

### Decisão

C5 avalia o efeito de fornecer uma **ontologia errada** em formato JSON. Dois níveis:

| Condição | Ontologia fornecida | Objetivo |
|---|---|---|
| **C5_vizinho** (nível 1) | Parceira com **alta similaridade de esquema** (vizinho no espaço de relações) | Mede se o modelo segue a ontologia quando o vocabulário é semanticamente próximo |
| **C5_incongruente** (nível 2) | Parceira com **baixa similaridade de esquema** (incongruente no espaço de relações) | Mede se o modelo ignora ou se guia pela ontologia quando o vocabulário é radicalmente diferente |

**Justificativa:** comparar C5_vizinho vs C5_incongruente permite separar dois efeitos:
(a) o modelo lê e aplica as relações do JSON fornecido; ou (b) o modelo ignora a ontologia
e extrai relações pelo senso comum. Se as métricas de conformidade forem semelhantes nos
dois níveis, o modelo ignora o JSON. Se C5_vizinho tiver conformidade maior, o vocabulário
com alta similaridade de esquema "atrai" o modelo.

**Implementação:** campo opcional `ontologia_parceira_id` no manifesto especifica qual
ontologia usar no JSON do prompt. A ontologia alvo (campo `ontologia_id`) segue sendo
usada para o gabarito.

### Critério objetivo de seleção de pares (pré-registrado)

Os pares C5 são determinados por dois critérios computados sobre os labels de relação de cada
par de ontologias dentro do mesmo dataset:

1. **Overlap Jaccard (OvJ):** `|labels(A) ∩ labels(B)| / |labels(A) ∪ labels(B)|` sobre labels normalizados (camelCase → espaços, underscores → espaços, minúsculas).
2. **Similaridade de embeddings bidirecional (SimE):** para cada label em A, max-cosseno com todos os labels de B; repete B→A; retorna a média das duas médias. Modelo: `all-MiniLM-L6-v2` (mesmo usado na projeção de embeddings do experimento).

**Decisão:** vizinho = parceira com maior SimE; incongruente = parceira com menor SimE.

Resultado salvo em `configs/pipeline/pares_c5.json`. Script reprodutível: `scripts/calcular_pares_c5.py`.

### Tabela de pares C5 — TekGen (Wikidata)

Colunas: `#Tok` = tokens no bloco de texto formato C1/C5 (whitespace-split, proxy de tamanho do contexto; vide D-11); `dTok` = tokens da parceira menos tokens do alvo (covariável na análise estatística); `Inc2` = 2ª parceira com menor SimE (usada na análise de sensibilidade D-10).

| Alvo | #Tok | Vizinho | OvJ% | SimE | #Tok | dTok | Incongruente | OvJ% | SimE | #Tok | dTok | Inc2 | SimE2 |
|---|--:|---|--:|--:|--:|--:|---|--:|--:|--:|--:|---|--:|
| 1_movie      | 193 | 4_book        | 17.4 | 0.591 | 202 |  +9 | 9_nature     |  0.0 | 0.254 | 192 |  -1 | 7_space     | 0.265 |
| 2_music      | 180 | 4_book        | 13.6 | 0.542 | 202 | +22 | 7_space      |  0.0 | 0.243 | 146 | -34 | 5_military  | 0.252 |
| 3_sport      | 215 | 1_movie       |  4.2 | 0.406 | 193 | -22 | 9_nature     |  0.0 | 0.251 | 192 | -23 | 7_space     | 0.272 |
| 4_book       | 202 | 1_movie       | 17.4 | 0.591 | 193 |  -9 | 9_nature     |  0.0 | 0.251 | 192 | -10 | 7_space     | 0.272 |
| 5_military   | 154 | 8_politics    |  0.0 | 0.340 | 153 |  -1 | 7_space      |  0.0 | 0.223 | 146 |  -8 | 9_nature    | 0.233 |
| 6_computer   | 108 | 4_book        |  0.0 | 0.384 | 202 | +94 | 7_space      |  0.0 | 0.187 | 146 | +38 | 9_nature    | 0.226 |
| 7_space      | 146 | 3_sport       |  0.0 | 0.272 | 215 | +69 | 6_computer   |  0.0 | 0.187 | 108 | -38 | 5_military  | 0.223 |
| 8_politics   | 153 | 1_movie       |  0.0 | 0.377 | 193 | +40 | 7_space      |  0.0 | 0.226 | 146 |  -7 | 9_nature    | 0.234 |
| 9_nature     | 192 | 1_movie       |  0.0 | 0.254 | 193 |  +1 | 6_computer   |  0.0 | 0.226 | 108 | -84 | 5_military  | 0.233 |
| 10_culture   | 143 | 4_book        |  0.0 | 0.376 | 202 | +59 | 9_nature     |  0.0 | 0.240 | 192 | +49 | 7_space     | 0.244 |

**Aviso TekGen:** 1_movie, 4_book, 7_space e 9_nature aparecem como parceira de 4 ontologias cada.
Esperado em dataset pequeno (10 ontologias); cada par é determinado unicamente pelo máximo/mínimo de SimE para cada alvo.

### Tabela de pares C5 — DBpedia (WebNLG)

| Alvo | #Tok | Vizinho | OvJ% | SimE | #Tok | dTok | Incongruente | OvJ% | SimE | #Tok | dTok | Inc2 | SimE2 |
|---|--:|---|--:|--:|--:|--:|---|--:|--:|--:|--:|---|--:|
| 1_university          | 435 | 12_monument          | 18.0 | 0.649 | 271 | -164 | 8_celestialbody    |  0.0 | 0.356 | 255 | -180 | 2_musicalwork     | 0.413 |
| 2_musicalwork         | 347 | 17_artist            | 21.3 | 0.663 | 399 |  +52 | 8_celestialbody    |  1.6 | 0.374 | 255 |  -92 | 1_university      | 0.413 |
| 3_airport             | 375 | 16_city              | 24.0 | 0.663 | 235 | -140 | 8_celestialbody    |  0.0 | 0.336 | 255 | -120 | 10_comicscharacter| 0.432 |
| 4_building            | 367 | 15_sportsteam        | 21.6 | 0.650 | 255 | -112 | 8_celestialbody    |  3.2 | 0.404 | 255 | -112 | 10_comicscharacter| 0.444 |
| 5_athlete             | 371 | 15_sportsteam        | 22.0 | 0.680 | 255 | -116 | 8_celestialbody    |  4.9 | 0.449 | 255 | -116 | 10_comicscharacter| 0.461 |
| 6_politician          | 403 | 18_scientist         | 22.5 | 0.673 | 443 |  +40 | 10_comicscharacter |  5.5 | 0.470 | 191 | -212 | 2_musicalwork     | 0.475 |
| 7_company             | 271 | 16_city              | 24.4 | 0.673 | 235 |  -36 | 8_celestialbody    |  0.0 | 0.356 | 255 |  -16 | 10_comicscharacter| 0.466 |
| 8_celestialbody       | 255 | 18_scientist         |  8.8 | 0.497 | 443 | +188 | 3_airport          |  0.0 | 0.336 | 375 | +120 | 7_company         | 0.356 |
| 9_astronaut           | 375 | 6_politician         | 14.7 | 0.607 | 403 |  +28 | 12_monument        |  1.6 | 0.437 | 271 | -104 | 3_airport         | 0.440 |
| 10_comicscharacter    | 191 | 19_film              | 12.7 | 0.603 | 431 | +240 | 16_city            |  0.0 | 0.382 | 235 |  +44 | 8_celestialbody   | 0.397 |
| 11_meanoftransport.   | 631 | 7_company            | 17.1 | 0.632 | 271 | -360 | 8_celestialbody    |  0.0 | 0.396 | 255 | -376 | 9_astronaut       | 0.487 |
| 12_monument           | 271 | 1_university         | 18.0 | 0.649 | 435 | +164 | 8_celestialbody    |  0.0 | 0.366 | 255 |  -16 | 9_astronaut       | 0.437 |
| 13_food               | 247 | 16_city              | 20.5 | 0.580 | 235 |  -12 | 8_celestialbody    |  0.0 | 0.371 | 255 |   +8 | 19_film           | 0.417 |
| 14_writtenwork        | 399 | 6_politician         | 15.1 | 0.596 | 403 |   +4 | 8_celestialbody    |  6.0 | 0.434 | 255 | -144 | 10_comicscharacter| 0.473 |
| 15_sportsteam         | 255 | 5_athlete            | 22.0 | 0.680 | 371 | +116 | 8_celestialbody    |  2.0 | 0.367 | 255 |   +0 | 2_musicalwork     | 0.484 |
| 16_city               | 235 | 18_scientist         | 25.0 | 0.717 | 443 | +208 | 8_celestialbody    |  0.0 | 0.370 | 255 |  +20 | 10_comicscharacter| 0.382 |
| 17_artist             | 399 | 18_scientist         | 26.5 | 0.688 | 443 |  +44 | 10_comicscharacter |  5.6 | 0.475 | 191 | -208 | 8_celestialbody   | 0.489 |
| 18_scientist          | 443 | 16_city              | 25.0 | 0.717 | 235 | -208 | 10_comicscharacter |  4.8 | 0.469 | 191 | -252 | 8_celestialbody   | 0.497 |
| 19_film               | 431 | 2_musicalwork        | 14.5 | 0.607 | 347 |  -84 | 13_food            |  1.5 | 0.417 | 247 | -184 | 8_celestialbody   | 0.440 |

**Aviso WebNLG:** 8_celestialbody é o incongruente de 12/19 ontologias — resultado esperado dado seu
vocabulário astronômico (apoapsis, perihelion, orbitalPeriod...) ser maximalmente dissimilar de
domínios humanos/sociais. Os padrões são coerentes com a estrutura semântica do dataset e não
configuram viés metodológico. Para 6_politician, 17_artist e 18_scientist o incongruente é
10_comicscharacter (SimE ≈ 0.47), não 8_celestialbody, o que reforça a robustez do critério.

---

## D-03: Sufixo "Test Output: " em todos os prompts

**Status:** APROVADA

### Decisão

Todos os prompts de todas as condições (C0, C1, C1_oneshot_original, C3, C5, C5_vizinho,
C5_incongruente) terminam com `\nTest Output: `.

**Justificativa:** o sufixo é um artefato do benchmark original (modelos de completamento
Vicuna/Alpaca). Para modelos de chat, serve como sinal de que o modelo deve começar a
listar triplas sem prefácio. Manter em todas as condições garante consistência e facilita
a comparação direta — o único fator de variação entre condições é o contexto de ontologia,
não o formato do prompt.

**Implementação:** `prompts.py` — cada bloco de condição termina com
`f"Test Sentence: {sentenca}\nTest Output: "`.

---

## D-04: Modo de avaliação — benchmark vs. amostra

**Status:** APROVADA

### Decisão

A função `avaliar()` em `avaliacao/original.py` aceita o parâmetro `modo_amostra: bool`.

| Modo | Denominador por ontologia | Quando usar |
|---|---|---|
| `modo_amostra=False` (padrão) | `len(gabarito)` — total de frases do GT | Runs completos; reprodução dos valores publicados |
| `modo_amostra=True` | `len(metricas_por_frase)` — frases efetivamente processadas | Runs parciais com `frases_max`; smoke tests; runs de desenvolvimento |

**Justificativa:** o artigo original divide pelo total de frases do GT (~400 por ontologia).
Em runs de desenvolvimento com `frases_max=5`, isso produziria conformidade ~0.01 para
5 frases com conformidade real 1.0 — valor sem significado. O modo amostra corrige isso
dividindo pelas frases de fato avaliadas.

O campo `"modo_amostra"` é registrado no JSONL de médias para garantir rastreabilidade.

---

## D-05: Instrução de entidade — desvio controlado do prompt original

**Status:** APROVADA

### Decisão

As condições C0, C1, C3, C5, C5_vizinho e C5_incongruente incluem a instrução adicional:

> *"Use as subject and object the exact entity names as written in the sentence,
> never the ontology class names."*

A condição **C1_oneshot_original** **não inclui** esta instrução (máxima fidelidade ao prompt
publicado no benchmark).

### Justificativa

Sem a instrução, o qwen2.5:3b (e possivelmente modelos maiores) substitui as entidades reais
pelos nomes das classes da ontologia fornecida no contexto. Por exemplo, dado o contexto
`director(film,human)`, o modelo gerava `director(film, Noriyuki Abe)` usando `"film"` como
sujeito em vez do título real do filme. Isso causa P/R/F1=0 mesmo quando a relação está
correta — um artefato de prompt que mascara a capacidade real do modelo.

A instrução adicional é um **desvio controlado e documentado** do benchmark original. Ela
não altera a estrutura do prompt (contexto, formato de saída, "Test Output:"), apenas esclarece
o uso de nomes de entidades. O desvio é registrado aqui para transparência metodológica.

**Efeito observado (qwen2.5:3b, 5 frases, C1 antes→depois):**
- Antes: conformidade=0.80, P/R/F1=0.00, sujeitos = classes puras ("film")
- Depois: conformidade=1.00, P/R/F1=0.00, sujeitos = classe+entidade ("film:Bleach: Hell Verse")
- Problema residual: modelo ainda antepõe o tipo (`film:`, `human:`) — achado metodológico
  registrado para análise na dissertação.

---

## D-06: Fonte de dados para C3 k=3 — dados de treino, nunca GT de teste

**Status:** APROVADA

### Decisão

As frequências de relação usadas para selecionar as k=3 relações da condição C3
são calculadas exclusivamente sobre os **dados de treino** do benchmark
(`data/{dataset}/train/ont_{onto}_train.jsonl`), nunca sobre o ground truth de teste.

### Justificativa

Usar o GT de teste para calcular quais relações incluir no contexto seria **vazamento de dados**:
o pipeline teria acesso direto às respostas corretas antes de executar a inferência. O conjunto
de treino é parte pública do benchmark, disponível antes do teste, e é a fonte correta para
qualquer estatística derivada do corpus.

**Diferença prática (1_movie TekGen):**
| Fonte | Top-3 relações |
|---|---|
| GT de teste (errado) | publication_date, cast_member, director |
| Dados de treino (correto) | director, screenwriter, genre |

As distribuições diferem significativamente — `screenwriter` e `genre` lideram no treino
mas não estão no top-3 do teste. Usar o treino é também mais realista: em produção, a
distribuição de relações do conjunto de teste é desconhecida.

**Implementação:** `dados.py` — função `carregar_frequencias_treino(bench_dir, dataset, ontologia_id, k)`.
Teste: `test_c3_usa_dados_de_treino_nao_gt` verifica que o caminho construído contém
`"train"` e não contém `"ground_truth"` nem `"/test/"`.

---

## D-07: Métrica auxiliar — taxa de erro de formato (avg_fmt_error)

**Status:** APROVADA

### Decisão

Além das métricas do benchmark original, o avaliador computa a **taxa de erro de formato**:
fração de triplas extraídas cujo sujeito **ou** objeto contém o nome de uma classe da
ontologia como prefixo ou valor literal.

**Padrões detectados (após `entidade.lower().strip()` → `e`; classe como `cls`):**
1. `e == cls` **e** `entidade` não contém nenhuma letra maiúscula — entidade é exatamente
   o label do conceito em minúsculas, ex.: `"film"`. A restrição de maiúsculas evita
   flagear entidades legítimas capitalizadas como `"Composer"` ou `"Book"` que existem
   no GT do benchmark.
2. `e.startswith(cls + ":")` — prefixo com dois-pontos, ex.: `"film:Bleach: Hell Verse"`
3. `e.startswith(cls + ' "')` ou `e.startswith(cls + " '")` — prefixo com espaço e aspas,
   ex.: `'film "Keyboard Cat"'`. Restrição a aspas evita `"City of God"` com classe `"city"`.

A métrica conta por **tripla** (1 se ao menos uma entidade falha, 0 caso contrário) —
não duplo-conta triplas onde ambas as entidades contêm erro.

**Implementação:** `calcular_taxa_erro_formato(ontologia, triplas)` em `avaliacao/original.py`.
Integrada ao loop de `avaliar()` via acumuladores `tef`/`stef`/`gef`; saída como campo
`"avg_fmt_error"` nos JSONLs de resultado por ontologia e global.

### Justificativa

O piloto com qwen2.5:3b em C1 mostrou que o modelo antepõe o tipo de classe às entidades
(ex.: `"film:Bleach: Hell Verse"`), causando P/R/F1=0 mesmo quando a relação está correta.
A métrica quantifica essa falha de formato separadamente das métricas de recuperação, permitindo
distinguir dois tipos de erro: (a) relação errada (capturado por F1) e (b) formato errado
mas relação correta (capturado por avg_fmt_error). O piloto com 7B determinará se o problema
persiste em modelos maiores e se justifica intervenção adicional no prompt.

---

## D-08: Diferença de tokens como covariável na análise de C5

**Status:** APROVADA (Fase 2)

### Decisão

A diferença de tokens entre o bloco de ontologia da parceira e o bloco do alvo (`dTok = #Tok_parceira - #Tok_alvo`)
é registrada em `configs/pipeline/pares_c5.json` para cada par C5_vizinho e C5_incongruente e será
incluída como **covariável contínua** na análise estatística dos efeitos de C5.

### Justificativa

Mudar de ontologia alvo para ontologia parceira altera o tamanho do contexto de ontologia injetado no
prompt. Se a parceira tiver significativamente mais ou menos tokens que o alvo (ex.: 11_meanoftransportation
tem 631 tokens vs. 271 da vizinha 7_company, dTok=-360), parte do efeito observado em F1 e conformidade
pode ser atribuído ao tamanho do contexto, não à similaridade de esquema. Incluir dTok como covariável
permite estimar o efeito de similaridade líquido desse confundidor.

**Implementação:** campo `dif_tokens` disponível em `configs/pipeline/pares_c5.json` para cada par.

---

## D-09: SimE como variável contínua na análise de C5

**Status:** APROVADA (Fase 2)

### Decisão

Além da comparação discreta entre níveis (C5_vizinho vs. C5_incongruente), a análise reportará
o **efeito de SimE como variável contínua** sobre F1, conformidade ontológica e taxa de alucinação,
usando regressão linear com ontologia como efeito aleatório (modelo misto).

### Justificativa

A dicotomia vizinho/incongruente colapsa uma variação contínua de similaridade de esquema
(0.187 a 0.717 no TekGen; 0.336 a 0.717 no WebNLG) em dois grupos. A análise contínua
captura o gradiente e permite verificar linearidade — se o efeito for monotônico, fortalece
a interpretação causal; se for não-linear (ex.: limiar), gera hipóteses sobre o mecanismo
de atenção do modelo. Os dois níveis de C5 funcionam como casos extremos no eixo contínuo
de SimE, tornando as duas análises complementares.

**Dados:** os valores de SimE para cada par são exportados em `configs/pipeline/pares_c5.json`
(campo `sim_emb` em cada sub-objeto de vizinho/incongruente).

---

## D-10: Análise de sensibilidade — C5_incongruente com segunda parceira

**Status:** APROVADA (Fase 2, amostra pequena)

### Decisão

Em uma **amostra de 5 ontologias** (3 WebNLG onde incongruente=8_celestialbody + 2 TekGen onde
incongruente=9_nature ou 7_space), executar C5_incongruente também com a **segunda parceira
com menor SimE** (`incongruente_2`, registrado em `configs/pipeline/pares_c5.json`).

As ontologias alvo sugeridas para a amostra de sensibilidade são:
- WebNLG: 3_airport (Inc2=10_comicscharacter, SimE=0.432), 5_athlete (Inc2=10_comicscharacter, SimE=0.461), 16_city (Inc2=10_comicscharacter, SimE=0.382)
- TekGen: 1_movie (Inc2=7_space, SimE=0.265), 5_military (Inc2=9_nature, SimE=0.233)

### Justificativa

A análise primária usa 8_celestialbody como incongruente para 12/19 ontologias WebNLG.
Se os resultados de conformidade ou F1 forem consistentes com os obtidos usando Inc2,
o efeito não depende do outlier astronômico e é generalizável. A análise de sensibilidade
é pré-registrada aqui antes de qualquer execução.

**Critério de robustez:** diferença absoluta de conformidade entre Inc1 e Inc2 < 0.05 para
ao menos 4 das 5 ontologias testadas.

---

## D-11: Terminologia — "similaridade de esquema"

**Status:** APROVADA (Fase 2)

### Decisão

O critério de seleção de parceiras C5 é chamado de **"similaridade de esquema"** em toda a
documentação, dissertação e comunicações. A palavra "domínio" é reservada para o dataset
inteiro (ex.: "domínio de filmes") e não descreve o critério de seleção dos pares.

### Justificativa

"Similaridade de domínio" é ambíguo — domínio pode referir-se à área temática (filmes, música)
ou ao campo `domain` de uma propriedade OWL. "Similaridade de esquema" é mais preciso: o critério
mede a sobreposição semântica entre os **vocabulários de relação** das ontologias (seus esquemas),
não a afinidade temática julgada por um humano. O critério é calculável, reprodutível e agnóstico
a julgamento editorial — nomear como "similaridade de esquema" torna isso explícito.

---

## D-12: Distinção entre zero-shot e C1_oneshot_original

**Status:** APROVADA

### Decisão

As condições C0, C1, C3, C5, C5_vizinho e C5_incongruente são rodadas em modo **zero-shot**
(sem exemplo de entrada/saída no prompt). A condição **C1_oneshot_original** reproduz o prompt
exato do benchmark original, que inclui um par de exemplo (`Input:` / `Output:`) antes da
sentença de teste.

### Justificativa

O benchmark original (Mihindukulasooriya et al., ISWC 2023) usa um único par de exemplo no
prompt de cada ontologia. Para isolar o efeito da estrutura do prompt vs. o conteúdo da
ontologia, as demais condições deste estudo omitem o exemplo e são zero-shot. C1_oneshot_original
serve como âncora de reprodução — permite verificar que nosso pipeline reproduz os valores do
artigo antes de comparar as condições experimentais.

**Implicação:** a comparação direta C1 vs. C1_oneshot_original mede o efeito do exemplo no
prompt, mantendo tudo mais constante.

---

## D-13: C0 bruta e C0 projetada

**Status:** APROVADA

### Decisão

A condição **C0** (sem ontologia) é avaliada em dois modos:

| Modo | Descrição |
|---|---|
| **C0 bruta** | Avaliação direta: triplas do modelo comparadas ao gabarito sem filtragem |
| **C0 projetada** | Triplas do modelo são projetadas para a ontologia alvo antes da avaliação (relações não conformes descartadas) |

### Justificativa

C0 bruta mede o que o modelo gera quando completamente livre; C0 projetada isola a contribuição
do conteúdo semântico das triplas C0 que por acaso estão conformes com a ontologia. A diferença
C0_projetada − C0_bruta quantifica quanto da precisão de C0 é "acidental" (relações corretas
geradas sem guia ontológico). Este delta é relevante para interpretar o ganho de C1 sobre C0.

---

## D-14: Avaliação sempre contra a ontologia de referência do alvo

**Status:** APROVADA

### Decisão

A conformidade ontológica e as métricas de alucinação são calculadas **sempre usando a ontologia
alvo** (a ontologia correta associada ao par sentença/GT), mesmo quando o prompt usa uma
ontologia parceira (condições C5_vizinho e C5_incongruente).

### Justificativa

O objetivo da avaliação é medir quão bem o modelo extrai triplas corretas para o domínio alvo.
Se a conformidade fosse calculada contra a ontologia parceira, C5_vizinho teria conformidade
artificialmente alta (pois o modelo gerou relações da parceira que fornecemos). Usar sempre a
ontologia de referência garante que a conformidade reflita o objetivo real (extração correta)
e não o quão bem o modelo seguiu a ontologia errada que fornecemos.

---

## D-15: Correção do parser — Categoria J

**Status:** APROVADA

### Decisão

O parser aceita triplas no formato `rel(sub, obj),` — com vírgula após o parêntese de
fechamento — sem gerar erro nem descartar a tripla. A vírgula final é simplesmente ignorada.

### Justificativa

Modelos de completamento (Vicuna, Alpaca) geram frequentemente listas de triplas separadas
por vírgula após o parêntese: `rel1(a,b),\nrel2(a,c),`. Sem esta correção, a tripla seria
descartada (a regex de fechamento não encontraria `)` no final da linha), gerando falsos
negativos na avaliação. O padrão é análogo à Categoria D original do benchmark
("trailing comma after triple"), mas afeta o delimitador externo em vez dos argumentos.

**Implementação:** `parser.py` — `_RE_TRIPLA` inclui `,?` após `)` antes do fim da linha.
Teste: `TestParserVirgulaFinal.test_multiplas_com_virgula_final` em `tests/test_parser.py`.

---

## D-16: Condição C2 — ontologia induzida a partir de extração livre no treino

**Status:** APROVADA (Fase 2; execução no piloto Kaggle)

### Decisão

C2 fornece ao modelo uma **ontologia induzida**: o vocabulário de relações extraído pelo
próprio modelo ao processar 50 frases de treino em modo livre (sem ontologia, equivalente
a C0). As relações extraídas são agrupadas por similaridade semântica e cada grupo vira uma
relação da ontologia induzida. O formato do contexto é idêntico ao C1 (texto, não JSON).

### Protocolo de indução (pré-registrado)

1. **Seleção de frases:** 50 frases do conjunto de treino por ontologia, amostradas com
   `seed=42` (mesmo seed do piloto). Registradas para reprodutibilidade.
2. **Extração livre:** o mesmo modelo (Qwen2.5-7B-Instruct-AWQ, temperatura 0) extrai
   triplas das 50 frases SEM ontologia e SEM triplas anotadas — condição equivalente a C0.
   As triplas anotadas do treino **não formam** o vocabulário de C2 (isso seria vazamento);
   são usadas apenas para calibrar os limiares (ver abaixo).
3. **Agrupamento:** os rótulos de relação livres são primeiro normalizados (D-22) e,
   opcionalmente, aumentados com definição semântica gerada pelo LLM (D-24). Em seguida,
   agrupados por embedding bidirecional (all-MiniLM-L6-v2) com limiar definitivo **0.70**
   (D-21). O rótulo mais frequente do grupo vira o nome da relação na ontologia induzida.
4. **Ontologia induzida:** lista de relações únicas com rótulos do modelo, sem domain/range
   (estes são desconhecidos). Salva em `configs/c2/{dataset}_{onto}_induced.json`.

### Calibração dos limiares (corre junto com o piloto Kaggle)

Após a extração livre das 50 frases de treino:
- Alinhar triplas livres com triplas anotadas pelo mesmo par (sujeito, objeto).
- Gerar pares "mesma relação" (rótulo livre ↔ relação anotada que aparece no mesmo par)
  e "relações diferentes".
- Testar limiares de agrupamento: 0.70, 0.75, 0.80, 0.85, 0.90 — escolher o que maximiza
  F1 da separação "mesma vs. diferente".
- Testar limiares de projeção: mesmos valores — escolher o que maximiza **precisão**
  (projetar errado é pior que não projetar).
- Registrar: limiares definitivos, curva F1/precisão, e sensibilidade com o limiar vizinho.
- Reportar % de relações projetadas por condição (C0, C2, C3).

### Limiares provisórios

| Parâmetro | Provisório | Definitivo |
|---|---|---|
| `threshold_agrupamento` | 0.80 | **0.70** (D-21; breakpoint + 0.05, normalização D-22) |
| `threshold_projecao` | 0.80 | **0.63** label+def (D-24, D-30) / **0.50** label-only (D-30) |

**Implementação:** `scripts/induzir_c2.py` — extração + clustering. Resultado em
`configs/c2/`. Condição C2 no pipeline usa o arquivo de ontologia induzida como contexto.

---

## D-17: Bugs encontrados no piloto Kaggle — C2 inválida, push falso

**Status:** REGISTRADO (2026-11-09) — correções aplicadas em 7e74d7f e nesta sessão

### Contexto

O piloto Kaggle rodou em 2026-11-09 (commits cc564e8 → 213a83b). Os 6 experimentos
C0/C1/C1_oneshot para 1_movie e 2_music produziram resultados válidos. A indução C2
(`scripts/induzir_c2.py`) e o `auto_push` apresentaram três problemas independentes.

### Bug 1 — Vazamento de gt_triples no prompt de indução C2

**Código afetado:** `scripts/induzir_c2.py`, chamada a `montar_prompt()`.

**Causa:** chamada com argumentos posicionais:
```python
# ANTES (bugado):
prompt = montar_prompt(f, onto_vazia, "C0")
```
O 1.º parâmetro posicional de `montar_prompt` é `sentenca` (string). Passando o dict
completo da frase (`f`), `sentenca` recebia `str(f)` — incluindo `gt_triples: [{'rel':
'director', ...}]`. O 2.º parâmetro (`condicao`) recebia o dict da ontologia vazia; como
nunca igualava à string `"C0"`, o bloco `C1_oneshot_original` era atingido por fallthrough,
e o campo `Test Sentence:` continha literalmente `{'id': ..., 'gt_triples': [...]}`.

**Efeito:** o modelo leu os rótulos de relação do gabarito diretamente no prompt → copiou
exatamente os labels da ontologia de referência para a ontologia induzida. Os 14 grupos de
`tekgen_1_movie_induced.json` coincidem quase perfeitamente com a ontologia oficial.

**Correção:**
```python
# DEPOIS (correto):
prompt = montar_prompt(sentenca=f["sent"], condicao="C0")
```

### Bug 2 — Dados de teste usados em vez de dados de treino

**Código afetado:** `scripts/induzir_c2.py`, chamada a `carregar_frases()`.

**Causa:** `carregar_frases(bench, dataset, "train", onto_id)` foi usada supondo que
aceitaria `split="train"`. Mas `_caminhos_gt()` não tem caso "train" — cai no `else` e
retorna o caminho do GT de TESTE. Os 50 IDs sorteados tinham o padrão `ont_1_movie_test_*`
(confirmado pelo arquivo `tekgen_1_movie_frases_seed42.json` comprometido).

**Efeito:** a "indução" treinou com frases de teste. Os alignamentos par(sub,obj) para
calibração também usaram triplas do GT de teste → `curva_agrupamento` em
`calibracao_tekgen_1_movie.json` mostra F1=0 em todos os thresholds (sinal de alinhamento
incorreto por ID errado).

**Correção:** nova função `carregar_frases_treino()` em `dados.py` que lê diretamente de
`data/{dataset}/train/ont_{onto_id}_train.jsonl`, sem usar `_caminhos_gt()`.

### Bug 3 — Push falso (`auto_push` enviou commit sem JSONL)

**Causa:** `.gitignore` continha `resultados/` (trailing slash), que faz o git ignorar o
DIRETÓRIO inteiro sem inspecionar subentradas. As negações `!resultados/piloto/**` nunca
eram avaliadas. Resultado: `git add -A` não staged os 6 JSONL; `git commit` commit apenas
os arquivos `configs/pipeline/manifesto_*.json` (que mudaram porque `gerar_manifesto.py`
foi re-executado no Kaggle). A mensagem `"auto: piloto completo"` apareceu no log mas o
JSONL não estava no commit.

**Efeito:** o piloto executou com sucesso, mas os resultados não foram enviados ao GitHub
automaticamente. Os arquivos foram recuperados manualmente do Output do Kaggle.

**Correção:** `.gitignore` alterado para `resultados/*` (sem trailing slash), que permite
o git entrar no diretório e aplicar as negações.

### Dados inválidos — `configs/c2/` do piloto

Os seguintes arquivos gerados pelo piloto são **INVÁLIDOS** e foram removidos do
repositório (commit desta sessão):

| Arquivo | Motivo |
|---|---|
| `tekgen_1_movie_induced.json` | Bug 1: relações copiadas do GT via prompt contaminado |
| `tekgen_2_music_induced.json` | Bug 1: idem |
| `tekgen_1_movie_frases_seed42.json` | Bug 2: IDs são `test_*`, não `train_*` |
| `tekgen_2_music_frases_seed42.json` | Bug 2: idem |
| `calibracao_tekgen_1_movie.json` | Bugs 1+2: alinhamentos incorretos, F1=0 artificialmente |
| `calibracao_tekgen_2_music.json` | Bugs 1+2: precisao_projecao ~0.77 inflada pelo leak |

### Por que os testes anteriores não detectaram esses bugs

Os testes existentes cobrem `executar.py` via `ProvedorMock` — que usa chamadas corretas com
keyword arguments. `induzir_c2.py` não era importado por nenhum teste; era puro `__main__`.
A função `carregar_frases()` com `split="train"` retornava silenciosamente o GT de teste
sem erro, então nada explodiu. Os testes para `carregar_frases_treino()` e para o
interceptor de `open()` foram adicionados retroativamente em `tests/test_c2_inducao.py`.

### Ação de remediação

Re-indução C2 no Kaggle com os bugs corrigidos, usando o manifesto
`configs/pipeline/manifesto_c2_reinducao.json`. Os 6 experimentos C0/C1/C1_oneshot NÃO
precisam ser refeitos — estão válidos.

---

## D-18: Instrução de formato unificada e métrica de erro de parse

**Status:** APROVADA (2026-11-09) — implementada em prompts.py, parser.py, original.py

> **Nota:** instrução de formato parcialmente substituída por D-20 (Variante D aprovada).

### Motivação — dados do piloto (Qwen2.5-7B, TekGen 1_movie + 2_music, n=200)

| Condição | % respostas vazias | % erro de parse |
|---|---|---|
| C0 | 10.0% / 8.0% | 10.0% / 8.0% |
| **C1** | **39.5% / 23.0%** | **39.5% / 23.0%** |
| C1_oneshot_original | 0.5% / 1.5% | 0.5% / 1.5% |

Todas as respostas "vazias" têm `response != ""` — são falhas de parse, não recusas.
Categorização das 79 falhas de C1 (1_movie):
- 31 usam `(rel, sub, obj)` — ordem lógica predicativa, não interpretada (argumento posicional ambíguo)
- 29 adicionam texto de prefácio antes das triplas ("Based on...", "Here are...")
- 4 usam bullet `- (sub, rel, obj)` com ordem errada
- 5 genuinamente vazias (modelo diz que nada há para extrair)

**Causa raiz:** `_INSTRUCAO_COM_ONT` para C1 dizia "in the given output format" (vago, sem
referenciar `rel(sub, obj)` explicitamente). Sem exemplo e sem formato nomeado, o modelo
escolhia `(rel, sub, obj)` por analogia com lógica de predicados. C1_oneshot_original tem
um exemplo e obtém 0.5% de vazias — o exemplo âncora o formato.

### Decisão

#### Instrução de formato unificada

Substituir todas as instruções de formato implícitas por uma única string:

```
Output only the triples, one per line, in the format relation(subject, object).
Do not write any other text.
```

Aplicada a **C0, C1, C2, C3, C5, C5_vizinho, C5_incongruente**.
**NÃO** aplicada a C1_oneshot_original (máxima fidelidade ao benchmark publicado).

Implementação: constante `_INSTRUCAO_FORMATO` em `prompts.py`, compartilhada entre todas
as condições afetadas.

#### Melhoria do parser (Categoria I)

Limpeza de parênteses externos: linha do tipo `(rel(sub, obj))` → strip outer `()` →
`rel(sub, obj)` → parseia normalmente.

Recupera +7 respostas nos 6 JSONL do piloto (3 em C1 1_movie, 4 em C1 2_music).
As 158 falhas restantes requerem nova execução com o prompt corrigido.

**NÃO** interpretado: `(rel, sub, obj)` — argumento posicional ambíguo sem argumento nomeado
que identifique ordem.

#### Métrica nova: `avg_parse_error` (D-18)

Fração de frases avaliadas onde o modelo produziu resposta não-vazia mas o parser não
extraiu nenhuma tripla: `response != ""` e `triples == []`.

Complementa `avg_fmt_error` (que mede uso de nomes de classe como entidades). As duas
métricas juntas cobrem:
- `avg_parse_error` → modelo produziu algo mas parser falhou (problema de formato de saída)
- `avg_fmt_error` → parser extraiu triplas mas entidades têm o nome de uma classe ontológica

### Justificativa para não interpretar `(rel, sub, obj)`

Reordenar argumentos `(rel, sub, obj)` → `rel(sub, obj)` é uma adivinhação perigosa:
o mesmo formato `(a, b, c)` poderia ser `(sub, rel, obj)` — sem forma de distinguir sem
semântica. Uma tripla incorretamente ordenada produz métricas infladas artificialmente
e contaminaria a análise comparativa. A instrução de formato D-18 deve eliminar esse padrão.

---

## D-19: Parser depth-aware, workers paralelos e mini-teste de instrução de formato

**Status:** IMPLEMENTADO (2026-11-09) — parser.py, executar.py, run.py, prompts.py

> **Nota:** parcialmente substituída por D-20 (instrução Variante D aprovada) e D-23 (workers=1 fixo).

### Motivação

Três problemas independentes identificados na análise do piloto D-18:

1. **Parser bug em `_dividir_multiplas`:** o split por regex não rastreava profundidade de
   parênteses → `is_a_song_by(Pavement, Summer Babe (Winter Version))` era dividida
   incorretamente em `is_a_song_by(Pavement, Summer Babe (Winter Version` + `))` ao invés
   de ser tratada como um objeto com parênteses aninhados.

2. **FmtErr C1 duplicou (0,19 → 0,39) após D-18:** causa raiz identificada — a instrução
   D-18 `relation(subject, object)` usa a mesma notação que a assinatura da ontologia
   `director(film, human)` → modelo copia os nomes de classe como entidades (530/668 = 79%).
   Necessário testar redações alternativas que descrevam o formato sem citar domínio.

3. **Velocidade:** vLLM mostra "Running: 1 reqs" — sem paralelismo. Com temperatura=0 e
   greedy decoding, o resultado por frase é determinístico e independente da ordem de
   execução, portanto paralelismo é seguro.

### Decisões

#### Parser depth-aware (correção de bug)

`_dividir_multiplas` substituída por implementação que percorre caractere a caractere e
rastreia `profundidade`. Divisão ocorre apenas em vírgulas com `profundidade == 0`.
A saída é uma lista de múltiplas partes somente se cada parte corresponde ao padrão
`identificador(`. Comportamento anterior preservado em todos os outros casos.

Ganho medido: recupera 4 triplas extras nos 6 JSONL do piloto (casos de objetos com
parênteses aninhados em 2_music).

#### Trailing comma em objeto

`_limpar_string` passa a remover vírgula terminal antes das aspas externas, cobrindo
o padrão `rel(sub, obj,)` que aparecia esporadicamente nas respostas C2.

#### Bullet `\d+\)` (D-19)

Adicionada remoção de bullet numérico com parênteses (`1) rel(a,b)`) além do já existente
`1. rel(a,b)`. Padrão encontrado em respostas de modelos maiores.

#### Workers paralelos

Novo parâmetro `workers: int = 1` em `executar()`. Quando `workers > 1`, usa
`ThreadPoolExecutor` enviando todas as frases pendentes em paralelo, mas coletando
resultados em ordem de submissão (não de conclusão) para garantir a mesma saída JSONL
sequencial. Configurável via campo `workers` no manifesto.

**Justificativa para determinismo:** com `temperatura=0` e greedy decoding, a distribuição
de tokens é calculada independentemente por frase — a resposta não depende de outras frases
em curso. Testado com ProvedorMock.

#### Mini-teste de instrução de formato (C1 FmtErr)

**Critério de escolha:** SOMENTE `avg_fmt_error` + `avg_parse_error` + `% respostas vazias`.
O F1 NÃO é usado como critério — evita selecionar instrução que vaze informação sobre a
estrutura da ontologia.

Quatro redações alternativas em `configs/d19/`:

| Variante | Texto da instrução |
|---|---|
| **A** (padrão D-18) | `Output only the triples, one per line, in the format relation(subject, object). Do not write any other text.` |
| **B** | `List only the extracted triples. Each triple must follow this exact pattern: predicate(entity1, entity2). One triple per line. No explanations, no extra text.` |
| **C** | `Extract triples as: name_of_relation(first_entity, second_entity). Write one triple per line. Nothing else.` |
| **D** | `Output the triples, one per line, in this exact format: relation_name(<exact name of the first entity as written in the sentence>, <exact name of the second entity as written in the sentence>). No other text.` |

Diferenças entre variantes:
- A usa `relation(subject, object)` — espelha a assinatura da ontologia (suspeita D-19)
- B usa `predicate(entity1, entity2)` — termos genéricos de lógica de predicados, sem
  conotação de "tipo de classe"
- C usa `name_of_relation(first_entity, second_entity)` — ênfase em "nome" e "entidade",
  sem parênteses na própria explicação do formato
- D usa angle-brackets `<entity name>` como marcadores explícitos — ênfase em "as written
  in the sentence", espera-se redução de erros de formato. Parser atualizado para remover
  `<>` automaticamente (D-19).

A regra de escolha está pré-registrada em **D-20** (abaixo) — definida ANTES de rodar.

Manifesto: `configs/pipeline/minilancador_d19.json`
17 experimentos: 4 variantes × 2 condições (C0, C1) × 2 ontologias (1_movie, 2_music)
  + 1 teste de paralelismo (C1_oneshot_original, 1_movie, 200 frases, workers=16)
100 frases/experimento (exceto paralelismo: 200), seed=42.

#### Suporte a `instrucao_formato` no manifesto

Novo campo opcional `instrucao_formato` nos experimentos do manifesto. Quando presente,
sobrescreve `_INSTRUCAO_FORMATO` em `prompts.py` para aquele experimento específico.
Implementado em `executar.py` e `run.py`. Ignorado para `C1_oneshot_original`.

### Arquivos modificados

- `src/t2kg/parser.py` — `_dividir_multiplas` depth-aware, trailing comma, bullet `\d+\)`, angle-brackets
- `src/t2kg/pipeline/prompts.py` — parâmetro `instrucao_formato` em `montar_prompt()`
- `src/t2kg/pipeline/executar.py` — `workers` e `instrucao_formato`
- `src/t2kg/run.py` — lê `workers` e `instrucao_formato` do manifesto
- `tests/test_parser.py` — classe `TestParserCasosReaisD19` (9 casos reais)
- `scripts/analisar_threshold_c2.py` — sweep estendido 0.50–0.90, alinhamento free↔ref
- `scripts/analisar_ontologia_c2.py` — varredura freq_min, investigação anomalias C2
- `scripts/calcular_redundancia_piloto.py` — redundância semântica (H1) no piloto-v2
- `src/t2kg/avaliacao/extended.py` — `taxa_predicados_unicos` (renomeado), `redundancia_semantica`
- `configs/d19/instrucao_A.txt`, `instrucao_B.txt`, `instrucao_C.txt`, `instrucao_D.txt`
- `configs/pipeline/minilancador_d19.json`

---

## D-20: Regra pré-registrada de escolha da instrução de formato (mini-teste D-19)

**Status:** PRÉ-REGISTRADA (2026-11-09) — definida ANTES de rodar o mini-teste

### Decisão

A instrução vencedora para uso nas condições C0, C1, C2, C3, C5 é escolhida **exclusivamente**
por critérios de erro de formato/parse, sem considerar F1. Isso evita que a escolha da instrução
introduza viés favorável a alguma condição experimental.

**Critério de seleção:**

1. **Pontuação primária:** `avg_fmt_error + avg_parse_error` (soma sobre condições C0 e C1,
   média entre 1_movie e 2_music) — menor = melhor.
2. **Desempate:** variante mais simples (ordem A < B < C < D).
3. **Critério de exclusão:** variante com `% de respostas prose` (response não-vazia, nenhuma
   tripla extraída) > 20% é eliminada antes do critério 1. **Exceção:** se TODAS as variantes
   ultrapassarem 20% de prosa, o critério de exclusão é ignorado e a seleção aplica apenas os
   critérios 1 e 2. Isso evita que o experimento fique sem vencedor caso o modelo falhe
   generalizadamente com todas as instruções.

**Por que excluir F1 do critério:**
Instruções com terminologia mais próxima da ontologia (ex.: `relation(subject, object)` em A)
podem ter maior conformidade ontológica artificialmente — o modelo "copia" os termos da
ontologia para o formato da instrução. Usar F1 como critério contaminaria a comparação
entre condições experimentais.

**Implementação:** resultado documentado aqui após a execução. Instrução escolhida substitui
`_INSTRUCAO_FORMATO` em `prompts.py` para a fase de execução completa.

### Resultado executado (mini-teste D-19, 2026-11-09)

Executado no Kaggle com Qwen2.5-7B-Instruct-AWQ T4×2, 100 frases/variante.
`d19_A_1movie_c1.jsonl` não foi baixado do Kaggle (ausente localmente); demais 16/17 arquivos presentes.
Scores calculados com `calcular_erros_formato_detalhado()` (C1 usa ontologia de referência; C0 fmt_error=0).

| Variante | score | fmt_err | fmt_a | fmt_b | parse_err | status |
|---|---|---|---|---|---|---|
| A | 0.2962 | 0.1762 | 0.1762 | 0.0000 | 0.1200 | candidata |
| B | 0.2359 | 0.2134 | 0.2134 | 0.0017 | 0.0225 | candidata |
| C | 0.2405 | 0.1680 | 0.1655 | 0.0025 | 0.0725 | candidata |
| **D** | **0.1505** | **0.1380** | **0.1330** | **0.0050** | **0.0125** | **vencedora** |

Nenhuma variante superou 20% de parse_error (máximo: A=12%) — exclusão por prosa não ativada.

**D-20 VENCEDORA: Variante D** — `relation_name(<exact name of the first entity>, <exact name of the second entity>)`

Observações:
- fmt_error dominado por tipo_a (cópia literal do nome da classe como entidade), tipo_b mínimo, tipo_c zero.
- Variante A: alto parse_error em C0 (21–25% de respostas prosa sem triplas).
- Variante B: maior fmt_error — a instrução `predicate(entity1, entity2)` confunde o modelo que usa `entity1`/`entity2` como valores literais.

### Estabilidade com paralelismo (workers=16 vs workers=1)

Validado em `d19_paralelismo_1movie_c1os.jsonl` (workers=16, 200 frases) contra piloto-v2 C1_oneshot_original (workers=1, 200 frases). vLLM T4×2, temperatura=0, greedy decoding.

**Identidade de respostas:** 186/200 idênticas (93.0%) — 14 diferenças.

**Truncamentos (`finish_reason="length"`):** 0 em ambos os arquivos — nenhuma resposta foi cortada por limite de tokens.

**Natureza das 14 diferenças:** variação estocástica residual do scheduling do servidor vLLM (ordem de processamento de requests paralelas afeta seed do CUDA). Exemplo:
- ID `ont_1_movie_test_107`: paralelo gera 1 tripla extra com comentário inline (`# Note: ...`); serial não.
- ID `ont_1_movie_test_228`: paralelo extrai `based_on(novel)`, serial extrai `based_on(The Idiot)` — alucinação diferente, mas ambos erram.
- ID `ont_1_movie_test_265`: mesmo conteúdo semântico, sujeito normalizado diferente ("Test Film" vs "The Film Features").

**Métricas comparadas (n=200, modo_amostra, 1_movie C1_oneshot):**

| Configuração | P | R | F1 | Conf | SH | RH | OH | FmtErr | ParseErr |
|---|---|---|---|---|---|---|---|---|---|
| workers=16 (paralelo) | 0.3875 | 0.2889 | 0.3111 | 0.9526 | 0.1596 | 0.0474 | 0.1353 | 0.0628 | 0.0050 |
| workers=1 (piloto-v2) | 0.3784 | 0.2814 | 0.3025 | 0.9533 | 0.1487 | 0.0467 | 0.1301 | 0.0728 | 0.0050 |

Diferenças absolutas < 0.01 em todas as métricas — paralelismo seguro para a fase completa.

**Tempo de execução:** 540.2s → 1808.3s = 1268s para 1800 frases totais (mini-teste D-19 completo) = **0.70 s/frase** (5143 fr/h). Por-experimento não rastreável (Kaggle registra timestamp de conclusão de célula, não por linha de output).

**Instrução de formato atualizada em `prompts.py`:** `_INSTRUCAO_FORMATO` substituída pela Variante D aprovada (commit pós-D-20).

---

## D-21: Regra pré-registrada — threshold de agrupamento e freq_min para C2

**Status:** RESULTADO APURADO (2026-11-09) — varredura executada, parâmetros definitivos determinados

### Regra (pré-registrada)

**threshold_agrupamento:**
- Varredura `scripts/analisar_threshold_c2.py` sweep 0.50–0.90.
- `threshold_definitivo = max(breakpoints observados) + 0.05` — único valor para todas as ontologias.
- Justificativa para limiar único: comparabilidade entre ontologias exige o mesmo threshold em todas. Usa-se o breakpoint mais restritivo observado. Se nenhuma ontologia apresentar breakpoint, o threshold padrão de calibração é automaticamente seguro para todas.

**freq_min:** `freq_min = 3` (definitivo). Uma relação deve aparecer em pelo menos 3 frases distintas (≥6% das N=50 frases de treino) para entrar na ontologia induzida. Este critério oferece melhor equilíbrio entre cobertura e ruído do que freq_min=5 — a grade de calibração mostra que freq_min=5 produzia apenas 4 relações (1_movie) e 1 relação (2_music), cobertura insuficiente para a condição C2. Derivado do tamanho do conjunto de treino, sem dependência da ontologia de referência. Sem cap de tamanho (2× cap revogado).

> **Histórico:** a especificação original desta D-21 indicava freq_min=5 (≥10% de N=50). Após análise da grade N×freq_min (ver tabela de calibração abaixo), freq_min=3 foi adotado como valor único e definitivo no piloto-v3.

### Resultado da varredura (executada em 2026-11-09)

| Onto | Primeiro par de referência que se funde | Breakpoint |
|---|---|---|
| **1_movie** | `director`+`screenwriter` (sim=0.658) a t=0.65; `award received`+`nominated for` (sim=0.621) a t=0.60 | **t=0.65** |
| **2_music** | Nenhum par se funde até t=0.50 | nenhum (seguro usar qualquer threshold ≤0.90) |

**Threshold D-21 único = 0.70** para todas as ontologias: max(0.65, nenhum) + 0.05 = 0.70.

Como 2_music não apresenta breakpoint até t=0.50, o threshold de 0.70 é seguro para ambas.

**Nota sobre escopo:** o breakpoint t=0.65 foi medido apenas nas ontologias 1_movie e 2_music (piloto-v2/v3). No experimento completo (29 ontologias), a regra `max(breakpoints) + 0.05` será reaplicada ao conjunto completo; o threshold definitivo pode ser revisto se alguma ontologia apresentar breakpoint > 0.65.

Relações de referência 1_movie (15): director, screenwriter, genre, based on, cast member, award received, production company, country of origin, publication date, characters, narrative location, filming location, main subject, nominated for, cost.

Relações de referência 2_music (13): composer, part of, lyrics by, publication date, language of work or name, voice type, instrumentation, tracklist, genre, performer, producer, nominated for, record label.

### Tamanho da ontologia induzida C2 com freq_min=5

| Fonte | Onto | total extraído | freq_min=1 (grupos) | freq_min=5 | Referência |
|---|---|---|---|---|---|
| C2 treino (50 frases) | 1_movie | 181 triplas, 117 únicas | 103 grupos | **7 (s/ D-22) → 4 (c/ D-22)** | 15 |
| C2 treino (50 frases) | 2_music | 158 triplas, 108 únicas | 94 grupos | **4 (s/ D-22) → 1 (c/ D-22)** | 13 |

**Nota:** freq_min conta frases DISTINTAS (D-21 fix) a partir da re-indução com `--from_raw_jsonl` (D-22). Sem a correção D-22, o Kaggle havia contado triplas totais (gerando 7/4 relações). Com D-22, a contagem correta de frases distintas filtra melhor os hápax.

### Alinhamento relações livres → referência (t=0.80, top-15 por frequência no C0 teste)

| Onto | Mapeadas (t≥0.80) | Melhor não-mapeada |
|---|---|---|
| 1_movie | 3/15 (`genre`=1.0, `director`=1.0, `based_on`=1.0) | `directed_by`→`director` (0.582), `produced_by`→`production company` (0.566) |
| 2_music | 0/15 | `produced`→`producer` (0.764), `written_by`→`lyrics by` (0.583) |

Conclusão: relações livres são lexicalmente muito diferentes das referências — mesmo `directed_by` ≠ `director` a t=0.80. Isto confirma a necessidade da projeção C0→referência e justifica o uso do threshold único 0.70 para a métrica H1.

### Redundância semântica H1 — piloto-v2 (threshold=0.70, normalização D-22)

Recalculado com `_normalizar_rotulo` antes do embedding (D-22 aplicada em `extended.py`).

| Onto | Condição | n_triplas | n_rótulos | n_grupos | taxa_pred_uniq | **red_sem (t=0.70)** |
|---|---|---|---|---|---|---|
| 1_movie | C0 | 716 | 331 | 224 | 0.462 | **0.3233** |
| 1_movie | C1 | 781 | 51 | 45 | 0.065 | **0.1176** |
| 1_movie | C1_oneshot | 637 | 39 | 36 | 0.061 | **0.0769** |
| 2_music | C0 | 634 | 326 | 226 | 0.514 | **0.3067** |
| 2_music | C1 | 642 | 40 | 34 | 0.062 | **0.1500** |
| 2_music | C1_oneshot | 535 | 23 | 21 | 0.043 | **0.0870** |

A t=0.70 com normalização D-22, C0 exibe redundância ~0.31–0.32, C1 ~0.12–0.15, C1_oneshot ~0.08–0.09.
C0 tem ~2.6–4.2× mais redundância semântica que as condições com ontologia — sinal consistente com H1.

### Ontologia C2 definitiva (threshold=0.70, freq_min=3 frases, normalização D-22, embeddings D-24)

Reinduzido no Kaggle com `use_definicoes=True` e `from_raw_jsonl`, usando respostas cruas de treino
(50 frases, seed=42). Fonte de verdade: `configs/c2/*_induced.json` (run_id `20261110T213002Z`).
**Nota:** os labels diferem da grade de calibração abaixo, que foi calculada sem D-24 (sem definições semânticas).

**tekgen/1_movie** — 10 relações:
1. `release_date` (freq=13) — membros: release_date, release_location, release_year, released, is_release_year, released_on
2. `genre` (freq=9) — membros: genre, is_genre
3. `year` (freq=8) — membros: year
4. `director` (freq=8) — membros: director, has_director
5. `starring` (freq=8) — membros: starring, starred_in
6. `based_on` (freq=7) — membros: based_on, is_based_on
7. `written_by` (freq=5) — membros: written_by, has_writer, is_written_by
8. `produced` (freq=4) — membros: produced, produced_by
9. `country` (freq=4) — membros: country, country_of
10. `title` (freq=4) — membros: title

**tekgen/2_music** — 11 relações:
1. `recorded` (freq=8) — membros: recorded_by, recorded_between, recorded_in, recorded
2. `produced_by` (freq=8) — membros: produced_by, produced, written_and_produced_by
3. `features` (freq=8) — membros: features
4. `wrote` (freq=7) — membros: written, wrote, writer
5. `genre` (freq=5) — membros: genre
6. `title` (freq=5) — membros: title
7. `performed_by` (freq=4) — membros: performed_by, performed
8. `album` (freq=4) — membros: album
9. `year` (freq=4) — membros: year, was_in_year, in_year
10. `released` (freq=3) — membros: released, release_date, released_on
11. `occupation` (freq=3) — membros: occupation

**webnlg/1_university** — 11 relações:
1. `location` (freq=43) — membros: location, located, located_in, location_of, located in, located_at
2. `has_staff_members` (freq=10) — membros: has_staff, has_staff_number, has_staff_members, has academic staff, has_academic_staff_number, has_academic_staff
3. `educates` (freq=9) — membros: educates
4. `number_of_students` (freq=7) — membros: number_of_students, number_of_doctorate_students, has_student_number, has_student_count
5. `has_students` (freq=7) — membros: has_students, has_students_combined, has students, has_undergraduates
6. `has_doctoral_students` (freq=6) — membros: has_postgraduate_students, has_postgraduates, has_post_graduate_students, has_undergraduate_students, has_doctoral_students
7. `established` (freq=6) — membros: established, established_in
8. `leader_of` (freq=4) — membros: leader, leader_of
9. `employs` (freq=4) — membros: employs
10. `hasStudentCount` (freq=3) — membros: hasStudentCount, hasStudentNumber
11. `capital_of` (freq=3) — membros: capital_of

**webnlg/19_film** — 13 relações (re-induzida após D-32; `relation_name` filtrado antes do agrupamento):
1. `produced` (freq=23) — membros: produced, producer, produced_by
2. `directed` (freq=12) — membros: directed
3. `wrote` (freq=12) — membros: wrote, written_by, wrote_music_for, writer
4. `born` (freq=10) — membros: born, born_in
5. `release_date` (freq=10) — membros: released_on, released, release_date
6. `director` (freq=8) — membros: director
7. `music_composer` (freq=7) — membros: music_composer, music_composed_by, composer, composer_songs_in_film
8. `edited` (freq=6) — membros: edited, editing, edited_by
9. `duration` (freq=6) — membros: duration, film_length, length
10. `editor` (freq=5) — membros: editor_of, editor
11. `cinematography` (freq=5) — membros: cinematographer, in_charge_of_cinematography, cinematography
12. `directed_by` (freq=4) — membros: directed_by
13. `composed` (freq=4) — membros: composing, composed

**Implementação:** parâmetros definitivos em `configs/c2/params_c2.json`; ontologias em `configs/c2/*_induced.json`.

### Calibração C2 — grade N × freq_min (calculado sem GPU, N=50 disponível)

Usando `resultados/c2_inducao/raw_tkg_*_treino.jsonl` (N=50, seed=42) com threshold=0.70 e normalização D-22, **sem** definições semânticas D-24. Os labels desta grade diferem dos definitivos (seção acima) porque D-24 reagrupa as relações com embeddings label+definição. Cobertura = fração das relações únicas do gabarito de treino que têm relação induzida correspondente (cosine ≥ 0.70). Ruído = fração de relações induzidas sem correspondência no gabarito.

| Onto | N | freq_min | n_ind | cob_treino | ruído | labels induzidos (top-7) |
|---|---|---|---|---|---|---|
| 1_movie | 50 | 2 | 21 | 0.286 | 0.810 | released_in, is_a, directed_by, director, genre, produced, follows... |
| 1_movie | 50 | 3 | 7 | 0.143 | 0.714 | released_in, is_a, directed_by, director, genre, produced, follows |
| 1_movie | 50 | 5 | 4 | 0.071 | 0.750 | released_in, is_a, directed_by, director |
| 1_movie | 100 | — | — | — | — | *dados pendentes — Kaggle necessário* |
| 2_music | 50 | 2 | 22 | 0.231 | 0.818 | is_a_song_by, released, recorded_for, produced by, performed, nationality, genre... |
| 2_music | 50 | 3 | 11 | 0.154 | 0.818 | is_a_song_by, released, recorded_for, produced by, performed, nationality, genre, from_album... |
| 2_music | 50 | 5 | 1 | 0.000 | 1.000 | is_a_song_by |
| 2_music | 100 | — | — | — | — | *dados pendentes — Kaggle necessário* |

**Interpretação:** A cobertura é sistematicamente baixa (7–29%) porque o LLM em modo C0 usa vocabulário radicalmente diferente do da ontologia de referência — confirmado pela análise de alinhamento C0→referência (D-21). A abordagem extração livre + clustering consegue alguns conceitos centrais (released_in, directed_by, genre) mas não reconstrói relações periféricas.

---

## D-22: Normalização de rótulos antes do embedding (C2 e métrica H1)

**Status:** APROVADA (2026-11-09)

### Decisão

Antes de computar embeddings de rótulos de relação — tanto na indução C2 quanto na métrica H1 de redundância semântica — aplicar normalização textual `_normalizar_rotulo`:

1. Substituir `_`, `-`, `/` por espaço.
2. Remover prefixos funcionais `has_` / `is_` (tornam-se `has ` / `is ` após passo 1).
3. Remover sufixos funcionais `_by` / `_of` / `_in` / `_to` / `_for` (na forma com espaço).
4. Converter para minúsculas.
5. Remover espaços em branco extras.

**Implementação:** `_normalizar_rotulo()` em `src/t2kg/avaliacao/extended.py`, usada em:
- `redundancia_semantica()` — normaliza rótulos antes do `modelo.encode()`
- `induzir_c2.py` com `--normalizar` — normaliza ao chamar `agrupar_relacoes(normalizar=True)`

### Justificativa

O modelo `all-MiniLM-L6-v2` trata `directed_by`, `has_director` e `director` como semanticamente distintos
porque os tokens de função (`by`, `has`, `of`) afetam o embedding. Sem normalização, `directed_by` e
`director` ficam em grupos separados (sim < 0.70). Com normalização:
- `directed_by` → `directed` (remove ` by`)
- `has_director` → `director` (remove `has `)
- `director_of` → `director` (remove ` of`)

Resultado em 1_movie: `directed_by` (n_frases=5) e `director` (n_frases=5) permanecem separados mesmo
após normalização (sims "directed" vs "director" < 0.70). No entanto, `released_in`, `released` e
`has_release_year` fundem-se em um grupo (sim "released" vs "release year" ≥ 0.70).

### Efeito na H1 (piloto-v2, t=0.70)

| Onto | Condição | red_sem (s/ D-22) | red_sem (c/ D-22) | Δ |
|---|---|---|---|---|
| 1_movie | C0 | 0.3082 | 0.3233 | +0.0151 |
| 1_movie | C1 | 0.0784 | 0.1176 | +0.0392 |
| 1_movie | C1_oneshot | 0.0769 | 0.0769 | 0 |
| 2_music | C0 | 0.3006 | 0.3067 | +0.0061 |
| 2_music | C1 | 0.1500 | 0.1500 | 0 |
| 2_music | C1_oneshot | 0.0870 | 0.0870 | 0 |

A razão C0/C1 se mantém (~2.6–4.2×) com ou sem normalização — o padrão qualitativo de H1 é robusto.

---

## D-23: Paralelismo — workers=1 fixado em todos os manifestos

**Status:** APROVADA (2026-11-10)

### Resultado do teste

Experimento `d19_paralelismo_1movie_c1os.jsonl` (workers=16, 200 frases) comparado ao piloto-v2 C1_oneshot_original (workers=1, 200 frases). vLLM T4×2, temperatura=0, greedy decoding.

| Configuração | Throughput | Tempo/frase |
|---|---|---|
| workers=16 (paralelo) | ~5143 fr/h | ~0.70 s |
| workers=1 (sequencial) | ~5300 fr/h | ~0.68 s |
| Diferença | < 3% | insignificante |

Timing derivado do log do Kaggle: vLLM pronto aos 540s; primeiro resultado de experimento aos 1808s → 1268s para 1800 frases totais (mini-teste D-19 completo). Tempo por-experimento individual não é rastreável (Kaggle registra apenas timestamp de conclusão de célula, não por linha de output).

**Causa da ausência de ganho:** o servidor vLLM local opera em modo single-request (batch=1 por requisição individual quando não há batching explícito). Enviar 16 threads clientes não aumenta o throughput quando o servidor já está saturado — a fila de requisições no servidor permanece essencialmente serial. Paralelismo real requereria batched inference (vLLM continuous batching) ou engine externa.

### Decisão

**Fixar `workers=1` em todos os manifestos** (piloto-v3, fase completa) para garantir 100% de reprodutibilidade. Os 7% de variação estocástica observados em workers=16 são eliminados. Campo `workers: 1` adicionado explicitamente ao `manifesto_piloto_v3.json`.

Paralelismo pode ser reativado em infra com batching real sem qualquer mudança de código (apenas alterar o campo `workers` no manifesto).

---

## D-24: Definição semântica para agrupamento e projeção C2 (estilo EDC)

**Status:** EXECUTADA (2026-11-10)

### Motivação

Cobertura da C2 era 7–29% usando só embedding de rótulo curto. Nomes como `released_in` e `publication_date` não são similares na superfície semântica. Zhang & Soh (2024) propõem "definição e canonicalização" (EDC) — o mesmo LLM que extrai gera definições para canonicalizar vocabulários.

### Implementação

Para cada rótulo de relação livre (das extrações treino C0) e da ontologia de referência, o LLM (mesmo modelo, temperatura=0) gera uma definição de 1 frase via prompt fixo:

```
Define the semantic relationship '{label}' in one concise sentence.
Focus on what it means when connecting two entities.
Reply with only the definition, no preamble.
```

O embedding passa a ser `encode("rótulo: definição")` em vez de `encode("rótulo")`. Agrupamento e projeção usam esse embedding aumentado.

Hipótese: `released_in: "Indicates the year a film was released to the public."` e `publication_date: "Specifies the official date of public release."` terão alta similaridade semântica, melhorando a cobertura do gabarito de treino.

### Arquivos

- `configs/c2/definicoes_{onto_id}.json` — gerado por `scripts/gerar_definicoes_relacoes.py` no Kaggle
- `scripts/gerar_definicoes_relacoes.py` — gera definições para uma ontologia
- `scripts/calibrar_c2_com_definicoes.py` — tabela comparativa label × definição
- `src/t2kg/avaliacao/extended.py` — `gerar_definicao_relacao()`, `redundancia_semantica(..., definicoes=None)`
- `scripts/induzir_c2.py` — `agrupar_relacoes(..., definicoes=None)`

### Status

Scripts implementados e executados no Kaggle. Definições geradas para as ontologias do piloto-v3
e armazenadas em `configs/c2/definicoes_{onto_id}.json`. A tabela comparativa label × definição
foi calculada com `--from_cache` após o run.

---

## D-25: Formato do bloco de ontologia em C2 — rótulo, sem conceitos nem assinaturas

**Status:** IMPLEMENTADA (2026-11-10)

### Decisão

O bloco de contexto de C2 usa o rótulo **`"Ontology Relations:"`** (mesmo rótulo de C1), seguido das relações induzidas separadas por vírgula, sem parênteses de domínio/range:

```
Ontology Relations: released_in, directed_by, genre
```

C2 **não exibe conceitos** (`Ontology Concepts:` ausente) nem **assinaturas de tipo** (`rel(domain,range)`) porque a ontologia induzida por agrupamento de embeddings (D-16/D-24) não produz tipos de entidade — apenas rótulos de relação.

### Impacto na comparação de erros de formato

Ao comparar `avg_fmt_error` entre C1 e C2, deve-se considerar que:

- **C1** exibe conceitos + assinaturas de tipo (ex.: `director(film,human)`) — o modelo pode extrair o tipo como prefixo de entidade.
- **C2** exibe apenas rótulos sem tipo — o modelo não recebe nomes de classe, logo a fonte principal de erro de formato (prefixo de classe) está ausente em C2.

Portanto, espera-se que C2 apresente **menor `avg_fmt_error`** que C1, não por melhor instrução, mas por menor oportunidade de erro de tipo. Comparações de `avg_fmt_error` entre C1 e C2 devem ser interpretadas com essa assimetria estrutural em mente.

### Implementação

`formatar_contexto(ontologia, "C2", ontologia_parceira=induced)` em `ontologias.py` retorna `f"Ontology Relations: {labels}"`.
O bloco é inserido no prompt por `montar_prompt` via `CONTEXT:\n{contexto_ontologia}` — idêntico ao C1.

---

## D-26: Detecção de modo amostra por ontologia (não por dataset)

**Status:** APROVADA

### Decisão

A detecção de `modo_amostra` (denominador = frases avaliadas vs. GT completo) deve ser
feita **por ontologia individualmente**, não por dataset. Implementada em
`avaliar_condicao()` via `_contar_registros(saida) < _contar_registros(gt)` para cada
item, armazenando `item["modo_amostra"]: bool` na config. `avaliar()` em `original.py`
respeita o override por item: `onto.get("modo_amostra", modo_amostra)`.

**Justificativa:** `webnlg/19_film` tem 127 frases de GT mas rodamos 100 (amostra);
`webnlg/1_university` tem 71 frases e rodamos 71 (completo). Com detecção por dataset,
o 1_university (primeiro encontrado, n=71<71→False) determinava modo COMPLETO para
ambos — correto para 1_university mas errado para 19_film.

**Impacto piloto-v3:** 19_film agora avaliado em modo amostra em todas as condições.
Resultados documentados em `docs/piloto_v3_resultados.md`.

---

## D-27: Correção de FMT_ERR para classes capitalizadas (DBpedia)

**Status:** APROVADA

### Decisão

`calcular_taxa_erro_formato()` agora detecta entidades que são exatamente o label
original de uma classe (qualquer capitalização) como fmt_error — adicionando verificação
`entidade.strip() in classes_exact` além da regra existente (entidade toda-minúscula).

**Justificativa:** DBpedia usa classes capitalizadas: "Film", "Person", "City". A regra
antiga (`not any(c.isupper() ...)`) não flagueava "Film" como entidade quando a classe
é "Film". Verificada ausência de falsos positivos: "Film Critic" (composto), "Composer"
(nome legítimo com ontologia `composer`) continuam sem erro.

**Validação:** ~0 falsos positivos confirmados nas 29 ontologias do GT completo
(webnlg baseline). Teste adicionado em `test_avaliacao_original.py`.

---

## D-28: Métricas de conformidade/alucinação sem respostas vazias

**Status:** APROVADA

### Decisão

Adicionar às métricas de avaliação: `avg_onto_conf_nv`, `avg_rel_halluc_nv` (calculados
apenas sobre sentenças com ≥1 tripla extraída) e `pct_vazias` (fração de sentenças com
triples=[] sobre as sentenças processadas).

**Justificativa:** `calcular_conformidade_ontologica([], ...)` retorna 1.0 por default
(ausência de triplas = sem violações). Para C5_incongruente/TekGen, 76–98% das respostas
são vazias (modelo recusa), inflando `avg_onto_conf ≈ 0.87–0.98`. O valor correto
`avg_onto_conf_nv = 0.00` revela que 100% das triplas produzidas são não-conformantes.

**Exemplo crítico:** tekgen/C5_incongruente: `avg_onto_conf=0.87`, `pct_vazias=0.87`,
`avg_onto_conf_nv=0.00`.

---

## D-29: Embeddings D-24 na projeção de relações (C0/C2/C3)

**Status:** APROVADA

### Decisão

`projetar_relacoes()` em `extended.py` agora aceita `definicoes_ref: Optional[Dict]`
e usa o formato D-24 `"label_normalizado: definição"` para os embeddings das relações
de referência quando disponível. Carregado de `configs/c2/definicoes_{onto_id}.json`
em `avaliar_condicao_extendida()`.

**Justificativa:** O agrupamento C2 usa embeddings D-24 (label + definição semântica)
para agrupar relações sinonímicas. Usar o mesmo estilo na projeção de avaliação torna
os dois passos coerentes: a mesma representação que separa clusters na indução também
guia o mapeamento livre→ontologia na avaliação.

**Impacto observado:** Com threshold=0.80, o F1-proj colapsou — o limiar original era
alto demais para o formato label+def (embeddings de strings longas têm similaridades
absolutas menores que embeddings de rótulos curtos). D-30 recalibrou o limiar para
0.63 (label+def) / 0.50 (label-only). Ver `docs/piloto_v3_resultados.md`.

---

## D-30: Recalibração do limiar de projeção para formato D-24

**Status:** APROVADA (revisada — limiar label+def elevado de 0.55 para 0.63)

### Decisão

Dois limiares distintos para `projetar_relacoes()` em `extended.py`, auto-selecionados
pelo formato de embedding:

| Formato | Limiar (D-30) | Anterior (pré-D-30) |
|---|---|---|
| Label-only: `_normalizar_rotulo(label)` | **0.50** | 0.80 |
| Label+def (D-24): `label: definição` | **0.63** | 0.80 |

Implementado como `THRESHOLD_PROJ_LABEL = 0.50`, `THRESHOLD_PROJ_DEF = 0.63`.
`projetar_relacoes(threshold=None)` seleciona automaticamente baseado em `definicoes_ref`.

**Regra de seleção:** menor threshold em [0.40, 0.90] (passo 0.01) tal que a precisão
agregada nos pares de treino alinhados seja ≥ 0.80. Priorizando precisão do mapeamento
sobre cobertura — falso-positivo de projeção contamina o F1-proj com triplas erradas.

### Calibração

**Dados:** pares alinhados (relação livre extraída em C2, relação GT) por coincidência
exata de par de entidades normalizadas (sub_norm, obj_norm) nas 50 frases de treino de
cada uma das 4 ontologias do piloto-v3. **109 pares válidos** no total.

**Aviso de amostra pequena:** `tekgen/2_music` contribui com apenas 12 pares (poucos
alinhamentos por sub+obj), dos quais só 1 tem GT corretamente mapeável com label+def.
As conclusões para essa ontologia são frágeis. No experimento completo (29 ontologias),
a calibração será refeita pela mesma regra (prec ≥ 0.80) com base em todos os pares
de treino disponíveis.

**Tabela de calibração agregada (label+def, 109 pares):**

| Threshold | Prec | Cobertura | N-aceitos | N-corretos |
|---|---|---|---|---|
| 0.45 | 0.630 | 0.670 | 73/109 | 46 |
| 0.50 | 0.646 | 0.596 | 65/109 | 42 |
| 0.55 | 0.750 | 0.440 | 48/109 | 36 |
| 0.60 | 0.773 | 0.404 | 44/109 | 34 |
| 0.63 | **0.800** | 0.367 | **40/109** | 32 |
| 0.65 | 0.789 | 0.349 | 38/109 | 30 |
| 0.80 | 1.000 | 0.055 | 6/109 | 6 |

**Precisão por ontologia em thr=0.63:**

| Ontologia | n | Prec | Cob | Aceitos |
|---|---|---|---|---|
| tekgen/1_movie | 10 | 1.000 | 0.50 | 5/10 |
| tekgen/2_music | 12 | 1.000 | 0.08 | 1/12 |
| webnlg/1_university | 38 | 0.769 | 0.34 | 13/38 |
| webnlg/19_film | 49 | 0.731 | 0.53 | 26/49 |

**Por que 0.80 causou o colapso de F1-proj após D-29?**
O formato label+def produz embeddings com escores de similaridade absolutos menores
(strings longas vs. curtas). Com threshold=0.80, apenas 6/109 pares eram aceitos
(recall=5.5%) — corretos mas volume desprezível.

### Sensibilidade

F1-proj das condições livres (C0, C2, C3) a limiares vizinhos, medido no piloto-v3:

| Dataset | Cond | thr=0.55 | **thr=0.63** | Δ |
|---|---|---|---|---|
| tekgen | C0 | 0.032 | **0.029** | −0.003 |
| tekgen | C2 | 0.063 | **0.065** | +0.002 |
| tekgen | C3 | 0.125 | **0.126** | +0.001 |
| webnlg | C0 | 0.118 | **0.063** | −0.055 |
| webnlg | C2 | 0.035 | **0.016** | −0.019 |
| webnlg | C3 | 0.111 | **0.083** | −0.028 |

A queda mais visível é webnlg/C0 (0.118→0.063): as projeções com sim 0.55–0.62 que eram
aceitas em 0.55 tinham 75% de precisão, algumas corretas, e contribuíam para o F1-proj.
Com 0.63 essas projeções são descartadas; o F1-proj cai mas o mapeamento que resta é
mais limpo (80% precisão). O impacto em tekgen é mínimo (ontologias menores, menos
relações ambíguas). A escolha de 0.63 é conservadora e metodologicamente mais sólida
para a dissertação.

### Otimização de desempenho

Adicionado `_embs_ref_cache: dict` em `extended.py`. Os embeddings das relações de
referência são calculados uma vez por ontologia e reutilizados em todas as sentenças
da mesma ontologia. Reduz as chamadas a `model.encode()` em ~100× para datasets com
100 sentenças por ontologia.

---

## D-31: Piloto-v3 — validação antes da fase completa

**Status:** EXECUTADA (2026-11-10)

### Decisão

Antes de rodar a fase completa (manifesto_7b_full.json, 203 experimentos), executar um piloto curto com:
- **Instrução:** Variante D (D-20 aprovada)
- **Modelo:** Qwen2.5-7B-Instruct-AWQ
- **Frases:** 100/experimento (amostra aleatória, seed=42)
- **Ontologias:** 4 — TekGen: `1_movie`, `2_music`; DBpedia: `1_university`, `19_film`
- **Condições:** C0, C1, C1_oneshot_original, C2 (onde houver ontologia induzida), C3 (top-3 treino), C5_vizinho, C5_incongruente
- **Total:** 28 experimentos × 100 frases = 2800 inferências ≈ 32 min a 5300 fr/h

**Manifesto:** `configs/pipeline/manifesto_piloto_v3.json` (gerado por `scripts/gerar_piloto_v3.py`)

### Condições por ontologia

| Ontologia | Dataset | C0 | C1 | C1_os | C2 | C3 | C5_v | C5_i |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1_movie | TekGen | ✓ | ✓ | ✓ | ✓ (10 rels) | ✓ | ✓ (4_book) | ✓ (9_nature) |
| 2_music | TekGen | ✓ | ✓ | ✓ | ✓ (11 rels) | ✓ | ✓ (4_book) | ✓ (7_space) |
| 1_university | DBpedia | ✓ | ✓ | ✓ | ✓ (11 rels) | ✓ | ✓ | ✓ |
| 19_film | DBpedia | ✓ | ✓ | ✓ | ✓ (13 rels) | ✓ | ✓ | ✓ |

### Prompts reais (Variante D, frase: "Bleach: Hell Verse ... is a 2010 Japanese animated film directed by Noriyuki Abe.")

**C3 (top-3 treino 1_movie: director, screenwriter, genre):**
```
Given the following sentence and list of key relations, please extract the triples. You may use
relation names not in the list when the text explicitly indicates them. Output the triples, one
per line, in this exact format: relation_name(<exact name of the first entity as written in the
sentence>, <exact name of the second entity as written in the sentence>). No other text. Use as
subject and object the exact entity names as written in the sentence, never the ontology class names.

CONTEXT:
Relations: director(film,human), screenwriter(film,human), genre(film,genre)

Test Sentence: Bleach: Hell Verse ... is a 2010 Japanese animated film directed by Noriyuki Abe.
Test Output:
```

**C5_vizinho (parceira=4_book):** fornece ontologia de livros (12 relações) no lugar da de filmes. Espera-se que o modelo extraia relações de livro para uma frase sobre filme — permite medir pressão da ontologia vs. senso comum do LLM.

**C5_incongruente (parceira=9_nature):** fornece ontologia de natureza (13 relações: parent_taxon, mountain_range, tributary...) — completamente incompatível com o domínio do texto.

### Lançador Kaggle — mudanças para piloto-v3

Célula 8 agora usa variável `MANIFESTO` configurável (default = `manifesto_piloto_v3.json`). Célula 9 limpa os JSONLs listados no manifesto atual (não mais `d19_*.jsonl` fixo). Célula 10 (re-indução C2) inclui tarefas N=100 para 1_movie e 2_music — necessárias para completar a tabela de calibração D-21.

### Objetivo do piloto-v3

1. Verificar que a instrução Variante D funciona corretamente em TODAS as condições (incluindo C5 e C2 que ainda não foram testadas com D).
2. Obter primeiras métricas de F1/conformidade/alucinação para 4 ontologias nas 7 condições.
3. Executar re-indução C2 com N=100 para completar a tabela N×freq_min de D-21.
4. Confirmar throughput e estabilidade antes de comprometer as 203 rodadas da fase completa.

---

## D-32: Placeholders da instrução Variante D tratados como fmt_error

**Status:** IMPLEMENTADA (2026-11-11)

### Contexto

A instrução Variante D (D-20) usa este formato de exemplo:

```
relation_name(<exact name of the first entity as written in the sentence>, <exact name of the second entity as written in the sentence>)
```

O modelo, ao operar em modo livre (C0), pode copiar literalmente o placeholder `relation_name` como rótulo de relação, em vez de extrair uma relação semântica. Analogamente, em casos raros os textos `exact name of the first entity...` aparecem como entidades.

Contagem no piloto-v3 (todas as condições e ontologias):

| Condição | 1_movie | 2_music | 1_university | 19_film |
|---|---|---|---|---|
| C0 | 8 | 15 | 0 | 18 |
| C1, C1_oneshot, C2, C3, C5_v, C5_i | 0 | 0 | 0 | 0 |
| Raw induction (treino C2) | 4 | 0 | 0 | 10 |

Placeholders como entidade (todos os JSONLs): 7 como sujeito, 4 como objeto.

### Decisão

Rótulos idênticos aos placeholders literais da instrução são tratados como **erro de formato**, aplicado igualmente a todas as condições:

**`PLACEHOLDER_RELS = {"relation_name"}`** — relações que são literalmente o nome do placeholder.

**`PLACEHOLDER_ENTITIES`** — textos de entidade que são literalmente os descritores do placeholder (`"exact name of the first entity as written in the sentence"`, `"exact name of the second entity as written in the sentence"`).

### Implementação

**(a) Exclusão da indução C2** — `scripts/induzir_c2.py`:
- Constante `PLACEHOLDER_RELS` definida após os demais constants do módulo.
- Antes de `agrupar_relacoes()`, filtra `all_relacoes` removendo qualquer `t[1]` cujo `lower().strip()` esteja em `PLACEHOLDER_RELS`.
- O script imprime quantas triplas foram filtradas quando `n_placeholder > 0`.

**(b) Contagem como fmt_error** — `src/t2kg/avaliacao/original.py`:
- Constantes `PLACEHOLDER_RELS` e `PLACEHOLDER_ENTITIES` no nível do módulo.
- Em `calcular_taxa_erro_formato()`, antes do loop de entidades:
  - Se `tr[1].lower().strip() in PLACEHOLDER_RELS` → incrementa `erros` e continua para a próxima tripla.
- Dentro do loop de entidades:
  - Se `e in PLACEHOLDER_ENTITIES` → `tripla_tem_erro = True; break`.

### Impacto

- **Métricas fmt_error**: mudam apenas para C0/1_movie (+8), C0/2_music (+15), C0/19_film (+18). C2/19_film não é afetado (C2 gerou 0 triplas com `relation_name`).
- **Ontologia C2/19_film**: re-induzida sem `relation_name` → 13 relações (antes 14). Requer re-execução do experimento C2/19_film (100 frases) no Kaggle.
- **Demais experimentos**: as métricas de fmt_error mudam apenas na reavaliação local; os resultados C2 dos outros três ontologias não mudam.

### Resultado do re-run (commits d19f4b4, 9aced71)

**webnlg / 19_film / C2 — antes (commit 6cc389b) vs depois (commit 9aced71):**

| Métrica | Antes | Depois | Δ |
|---|---|---|---|
| Precision | 0.01 | 0.02 | +0.01 |
| Recall | 0.01 | 0.01 | 0 |
| F1 | 0.01 | 0.01 | 0 |
| Conf | 0.30 | 0.24 | −0.06 |
| Conf-nv | 0.12 | 0.10 | −0.02 |
| RelHalluc-nv | 0.88 | 0.90 | +0.02 |
| FmtErr | 0.00 | 0.01 | +0.01 |
| ParseErr | 0.20 | 0.16 | −0.04 |
| %Vaz | 0.20 | 0.16 | −0.04 |

**webnlg / C2 global — projeção:**

| Métrica | Antes | Depois | Δ |
|---|---|---|---|
| F1-proj @0.63 | 0.016 | 0.014 | −0.002 |
| Frac-proj @0.63 | 0.474 | 0.234 | −0.240 |

**FmtErr C0 — efeito colateral do filtro:**

| Onto / Cond | FmtErr antes | FmtErr depois | Δ |
|---|---|---|---|
| tekgen / 1_movie / C0 | 0.01 | 0.05 | +0.04 |
| tekgen / 2_music / C0 | 0.01 | 0.04 | +0.03 |
| webnlg / 19_film / C0 | 0.00 | 0.05 | +0.05 |

**Interpretação:** A conformidade caiu (0.30→0.24) porque `relation_name` era contado como relação conformante antes do filtro. A fração de projeção caiu de 0.474 para 0.234 porque a ontologia induzida sem `relation_name` tem menos falsos vizinhos no espaço de embedding. O F1 do experimento permaneceu 0.01 — sem melhoria de qualidade de extração, esperado pois a ontologia C2 continua sendo induzida do texto do modelo livre. Os aumentos em FmtErr do C0 são o efeito direto do novo filtro de placeholders.
