# Decisões metodológicas — Fase 1

Decisões aprovadas que orientam a implementação e o design experimental.
Cada decisão substitui as propostas anteriores onde houver conflito.

---

## D-01: Formato do contexto para a condição C3 — k=3 com domínio/range

**Status:** APROVADA

### Decisão

C3 fornece ao modelo as **3 relações mais frequentes no conjunto de treino** (vide D-06),
acompanhadas de seus conceitos de domínio e alcance. Formato idêntico ao C1 mas sem
listar conceitos nem usar exemplo.

```
Relations: director(film,human), cast_member(film,human), publication_date(film,time)
```

**Justificativa:** k=3 isola o efeito de nomear as relações mais salientes sem revelar
a estrutura completa da ontologia. A inclusão de domain/range fornece contexto mínimo
sem revelar a lista completa de conceitos. k=0 seria indistinguível de C0 se o modelo
não souber quais relações extrair.

**Implementação:** `executar.py` computa as top-3 relações a partir das triplas do GT
(all_test_cases) antes de iniciar a inferência e passa para `formatar_contexto()` via
`top_k_rels`. Constante `_K_RELS = 3` em `executar.py`.

---

## D-02: Condição C5 — dois níveis de perturbação de ontologia

**Status:** APROVADA

### Decisão

C5 avalia o efeito de fornecer uma **ontologia errada** em formato JSON. Dois níveis:

| Condição | Ontologia fornecida | Objetivo |
|---|---|---|
| **C5_vizinho** (nível 1) | Ontologia de domínio **vizinho/relacionado** | Mede se o modelo segue a ontologia quando o vocabulário é semanticamente próximo |
| **C5_incongruente** (nível 2) | Ontologia de domínio **incompatível** | Mede se o modelo ignora ou se guia pela ontologia quando o vocabulário é radicalmente diferente |

**Justificativa:** comparar C5_vizinho vs C5_incongruente permite separar dois efeitos:
(a) o modelo lê e aplica as relações do JSON fornecido; ou (b) o modelo ignora a ontologia
e extrai relações pelo senso comum. Se as métricas de conformidade forem semelhantes nos
dois níveis, o modelo ignora o JSON. Se C5_vizinho tiver conformidade maior, o vocabulário
vizinho "atrai" o modelo.

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

Colunas: alvo (#R relações, #C conceitos, #Tok tokens no bloco JSON), par vizinho/incongruente com métricas.

| Alvo | #R | #C | #Tok | Vizinho | OvJ% | SimE | #R | #C | #Tok | Incongruente | OvJ% | SimE | #R | #C | #Tok |
|---|--:|--:|--:|---|--:|--:|--:|--:|--:|---|--:|--:|--:|--:|--:|
| 1_movie      | 15 | 12 | 193 | 4_book        | 17.4 | 0.591 | 12 | 20 | 202 | 9_nature     |  0.0 | 0.254 | 13 | 14 | 192 |
| 2_music      | 13 | 13 | 180 | 4_book        | 13.6 | 0.542 | 12 | 20 | 202 | 7_space      |  0.0 | 0.243 |  7 | 15 | 146 |
| 3_sport      | 11 | 20 | 215 | 1_movie       |  4.2 | 0.406 | 15 | 12 | 193 | 9_nature     |  0.0 | 0.251 | 13 | 14 | 192 |
| 4_book       | 12 | 20 | 202 | 1_movie       | 17.4 | 0.591 | 15 | 12 | 193 | 9_nature     |  0.0 | 0.251 | 13 | 14 | 192 |
| 5_military   |  9 | 13 | 154 | 8_politics    |  0.0 | 0.340 |  9 | 13 | 153 | 7_space      |  0.0 | 0.223 |  7 | 15 | 146 |
| 6_computer   |  4 | 15 | 108 | 4_book        |  0.0 | 0.384 | 12 | 20 | 202 | 7_space      |  0.0 | 0.187 |  7 | 15 | 146 |
| 7_space      |  7 | 15 | 146 | 3_sport       |  0.0 | 0.272 | 11 | 20 | 215 | 6_computer   |  0.0 | 0.187 |  4 | 15 | 108 |
| 8_politics   |  9 | 13 | 153 | 1_movie       |  0.0 | 0.377 | 15 | 12 | 193 | 7_space      |  0.0 | 0.226 |  7 | 15 | 146 |
| 9_nature     | 13 | 14 | 192 | 1_movie       |  0.0 | 0.254 | 15 | 12 | 193 | 6_computer   |  0.0 | 0.226 |  4 | 15 | 108 |
| 10_culture   |  8 | 15 | 143 | 4_book        |  0.0 | 0.376 | 12 | 20 | 202 | 9_nature     |  0.0 | 0.240 | 13 | 14 | 192 |

**Aviso TekGen:** 1_movie, 4_book, 7_space e 9_nature aparecem como parceira de 4 ontologias cada.
Isso é esperado em um dataset pequeno (10 ontologias) e não compromete o critério — cada par é
determinado unicamente pelo máximo/mínimo de SimE para cada alvo.

### Tabela de pares C5 — DBpedia (WebNLG)

| Alvo | #R | #C | #Tok | Vizinho | OvJ% | SimE | #R | #C | #Tok | Incongruente | OvJ% | SimE | #R | #C | #Tok |
|---|--:|--:|--:|---|--:|--:|--:|--:|--:|---|--:|--:|--:|--:|--:|
| 1_university          | 46 | 15 | 435 | 12_monument          | 18.0 | 0.649 | 26 | 14 | 271 | 8_celestialbody      |  0.0 | 0.356 | 27 |  8 | 255 |
| 2_musicalwork         | 35 | 15 | 347 | 17_artist            | 21.3 | 0.663 | 39 | 20 | 399 | 8_celestialbody      |  1.6 | 0.374 | 27 |  8 | 255 |
| 3_airport             | 39 | 14 | 375 | 16_city              | 24.0 | 0.663 | 23 | 11 | 235 | 8_celestialbody      |  0.0 | 0.336 | 27 |  8 | 255 |
| 4_building            | 38 | 14 | 367 | 15_sportsteam        | 21.6 | 0.650 | 24 | 14 | 255 | 8_celestialbody      |  3.2 | 0.404 | 27 |  8 | 255 |
| 5_athlete             | 37 | 17 | 371 | 15_sportsteam        | 22.0 | 0.680 | 24 | 14 | 255 | 8_celestialbody      |  4.9 | 0.449 | 27 |  8 | 255 |
| 6_politician          | 40 | 19 | 403 | 18_scientist         | 22.5 | 0.673 | 47 | 15 | 443 | 10_comicscharacter   |  5.5 | 0.470 | 18 | 10 | 191 |
| 7_company             | 28 | 10 | 271 | 16_city              | 24.4 | 0.673 | 23 | 11 | 235 | 8_celestialbody      |  0.0 | 0.356 | 27 |  8 | 255 |
| 8_celestialbody       | 27 |  8 | 255 | 18_scientist         |  8.8 | 0.497 | 47 | 15 | 443 | 3_airport            |  0.0 | 0.336 | 39 | 14 | 375 |
| 9_astronaut           | 38 | 16 | 375 | 6_politician         | 14.7 | 0.607 | 40 | 19 | 403 | 12_monument          |  1.6 | 0.437 | 26 | 14 | 271 |
| 10_comicscharacter    | 18 | 10 | 191 | 19_film              | 12.7 | 0.603 | 44 | 18 | 431 | 16_city              |  0.0 | 0.382 | 23 | 11 | 235 |
| 11_meanoftransportation | 68 | 20 | 631 | 7_company           | 17.1 | 0.632 | 28 | 10 | 271 | 8_celestialbody      |  0.0 | 0.396 | 27 |  8 | 255 |
| 12_monument           | 26 | 14 | 271 | 1_university         | 18.0 | 0.649 | 46 | 15 | 435 | 8_celestialbody      |  0.0 | 0.366 | 27 |  8 | 255 |
| 13_food               | 24 | 12 | 247 | 16_city              | 20.5 | 0.580 | 23 | 11 | 235 | 8_celestialbody      |  0.0 | 0.371 | 27 |  8 | 255 |
| 14_writtenwork        | 44 | 10 | 399 | 6_politician         | 15.1 | 0.596 | 40 | 19 | 403 | 8_celestialbody      |  6.0 | 0.434 | 27 |  8 | 255 |
| 15_sportsteam         | 24 | 14 | 255 | 5_athlete            | 22.0 | 0.680 | 37 | 17 | 371 | 8_celestialbody      |  2.0 | 0.367 | 27 |  8 | 255 |
| 16_city               | 23 | 11 | 235 | 18_scientist         | 25.0 | 0.717 | 47 | 15 | 443 | 8_celestialbody      |  0.0 | 0.370 | 27 |  8 | 255 |
| 17_artist             | 39 | 20 | 399 | 18_scientist         | 26.5 | 0.688 | 47 | 15 | 443 | 10_comicscharacter   |  5.6 | 0.475 | 18 | 10 | 191 |
| 18_scientist          | 47 | 15 | 443 | 16_city              | 25.0 | 0.717 | 23 | 11 | 235 | 10_comicscharacter   |  4.8 | 0.469 | 18 | 10 | 191 |
| 19_film               | 44 | 18 | 431 | 2_musicalwork        | 14.5 | 0.607 | 35 | 15 | 347 | 13_food              |  1.5 | 0.417 | 24 | 12 | 247 |

**Aviso WebNLG:** 8_celestialbody é o incongruente de 12/19 ontologias — resultado esperado, pois seu
vocabulário astronômico (apoapsis, perihelion, orbitalPeriod...) é maximalmente dissimilar do vocabulário
de domínios humanos/sociais. 16_city e 18_scientist são vizinho de 5 e 4 ontologias respectivamente,
pois suas relações (birthPlace, country, leader...) cobrem um espectro semântico amplo.
Esses padrões são coerentes com a estrutura semântica do dataset e não configuram viés metodológico.

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

## D-07: Métrica auxiliar — taxa de erro de formato (avg_fmt_error)

**Status:** APROVADA

### Decisão

Além das métricas do benchmark original, o avaliador computa a **taxa de erro de formato**:
fração de triplas extraídas cujo sujeito **ou** objeto contém o nome de uma classe da
ontologia como prefixo ou valor literal.

**Padrões detectados (case-insensitive, por entidade `e` e classe `cls`):**
1. `e == cls` — entidade é exatamente o label do conceito, ex.: `"film"`
2. `e.startswith(cls + ":")` — prefixo com dois-pontos, ex.: `"film:Bleach: Hell Verse"`
3. `e.startswith(cls + " ")` — prefixo com espaço, ex.: `'film "Keyboard Cat"'`

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
