# Decisões metodológicas — Fases 1 e 2

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
Relations: director(film,human), screenwriter(film,human), genre(film,genre)
```

(Exemplo com as top-3 do **treino** para 1_movie TekGen: director, screenwriter, genre.
Vide D-06 para a comparação com as top-3 do GT de teste, que são distintas.)

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
3. **Agrupamento:** os rótulos de relação livres são agrupados por embedding bidirecional
   (all-MiniLM-L6-v2) com limiar provisório **0.80**. O rótulo mais frequente do grupo
   vira o nome da relação na ontologia induzida.
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
| `threshold_agrupamento` | 0.80 | a calibrar no piloto |
| `threshold_projecao` | 0.80 | a calibrar no piloto |

**Implementação:** `scripts/induzir_c2.py` — extração + clustering. Resultado em
`configs/c2/`. Condição C2 no pipeline usa o arquivo de ontologia induzida como contexto.

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
