# Decisões metodológicas — Fase 1

Decisões aprovadas que orientam a implementação e o design experimental.
Cada decisão substitui as propostas anteriores onde houver conflito.

---

## D-01: Formato do contexto para a condição C3 — k=3 com domínio/range

**Status:** APROVADA

### Decisão

C3 fornece ao modelo as **3 relações mais frequentes no conjunto de teste** (ground truth),
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

### Tabela de pares C5 — TekGen (Wikidata)

| Ontologia alvo | C5_vizinho | Justificativa vizinho | C5_incongruente | Justificativa incongruente |
|---|---|---|---|---|
| 1_movie | 2_music | Ambas são obras de entretenimento com atores/músicos, datas de publicação, prêmios e gêneros | 7_space | Exploração espacial tem estrutura e vocabulário radicalmente distintos de filmes |
| 2_music | 1_movie | Artistas e obras criativas compartilham relações similares (publicação, gênero, prêmios) | 5_military | Militares, conflitos e hierarquias são opostos semânticos da música |
| 3_sport | 8_politics | Ambas têm organizações, membros, cargos e eventos competitivos | 9_nature | Taxonomias biológicas e rios não têm relação com esportes |
| 4_book | 1_movie | Obras criativas (livros→filmes) compartilham autor/diretor, gênero, editora/produtora | 7_space | Missões espaciais e planetas nada têm em comum com literatura |
| 5_military | 8_politics | Governança e forças militares coexistem no mesmo domínio histórico-político | 9_nature | Ecologia e taxonomia são opostos semânticos do militarismo |
| 6_computer | 7_space | Tecnologia de hardware/software compartilha vocabulário com agências e sistemas de missão | 9_nature | Ciências biológicas têm vocabulário ortogonal ao de TI |
| 7_space | 6_computer | Missões espaciais usam sistemas computacionais; agências lembram empresas de tech | 10_culture | Culturas, línguas e tradições são completamente opostos a astronomia |
| 8_politics | 5_military | Política e militarismo são domínios entrelaçados historicamente | 9_nature | Ecologia e taxons não têm correspondência em vocabulário político |
| 9_nature | 7_space | Ciências naturais (taxonomia, biomas) compartilham vocabulário científico com astronomia | 1_movie | Filmes, diretores e atores são completamente alheios à biologia |
| 10_culture | 2_music | Música é uma das expressões centrais de cultura e tradição | 7_space | Exploração espacial é semanticamente distante de cultura e tradições |

### Tabela de pares C5 — DBpedia (WebNLG)

| Ontologia alvo | C5_vizinho | Justificativa vizinho | C5_incongruente | Justificativa incongruente |
|---|---|---|---|---|
| 1_university | 7_company | Ambas são organizações com localização, fundação e afiliações | 8_celestialbody | Corpos celestes nada têm em comum com instituições de ensino |
| 2_musicalwork | 19_film | Obras criativas com gênero, distribuidora/gravadora, duração | 3_airport | Infraestrutura de transporte é oposto semântico de obras musicais |
| 3_airport | 11_meanoftransportation | Transporte e infraestrutura compartilham builder, length, type | 2_musicalwork | Obras musicais têm vocabulário completamente distinto de aeroportos |
| 4_building | 12_monument | Ambas são estruturas físicas com localização, designer, data | 8_celestialbody | Corpos celestes não têm relação com construções |
| 5_athlete | 15_sportsteam | Atletas e times compartilham liga, clube e competições | 8_celestialbody | Astronomia é semanticamente distante de esportes |
| 6_politician | 9_astronaut | Profissionais notáveis com birthPlace, deathPlace, cargo | 11_meanoftransportation | Meios de transporte são opostos a carreiras políticas |
| 7_company | 1_university | Organizações com fundação, localização e afiliações | 8_celestialbody | Corpos celestes nada têm em comum com empresas |
| 8_celestialbody | 9_astronaut | Corpos celestes e astronautas compartilham o domínio espacial | 13_food | Culinária e ingredientes nada têm em comum com astronomia |
| 9_astronaut | 18_scientist | Profissionais científicos com birthPlace, birthDate, knownFor | 13_food | Gastronomia é semanticamente distante de carreiras científicas |
| 10_comicscharacter | 19_film | Personagens ficcionais de mídia (quadrinhos/filmes) | 3_airport | Infraestrutura aeroportuária é oposta a personagens de quadrinhos |
| 11_meanoftransportation | 3_airport | Transporte compartilha builder, type, comprimento | 9_astronaut | Carreiras de astronautas são distantes de meios de transporte |
| 12_monument | 4_building | Estruturas físicas com localização, designer, inauguração | 9_astronaut | Carreiras espaciais não têm relação com monumentos |
| 13_food | 16_city | Ambas têm atributos country e étnicos (ethnicGroup, language) | 8_celestialbody | Astronomia é semanticamente distante de gastronomia |
| 14_writtenwork | 2_musicalwork | Obras criativas com país, língua, gênero | 8_celestialbody | Corpos celestes nada têm em comum com literatura |
| 15_sportsteam | 5_athlete | Times e atletas compartilham liga, local e membros | 8_celestialbody | Astronomia é distante de esportes |
| 16_city | 1_university | Ambas têm isPartOf, country, ethnicGroup — entidades geográfico-culturais | 8_celestialbody | Corpos celestes são opostos a cidades |
| 17_artist | 2_musicalwork | Artistas musicais e obras musicais compartilham gênero e gravadora | 3_airport | Infraestrutura aeroportuária é semanticamente distante de artistas |
| 18_scientist | 9_astronaut | Profissionais científicos com birthPlace, birthDate, knownFor | 13_food | Gastronomia é distante de carreiras científicas |
| 19_film | 2_musicalwork | Obras criativas com atores/músicos, gênero, distribuidora | 3_airport | Infraestrutura aeroportuária é oposta a filmes |

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
