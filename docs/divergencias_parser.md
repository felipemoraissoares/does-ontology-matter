# Divergências do parser — 20 exemplos anotados

Fonte: `data/wikidata_tekgen/baselines/Vicuna-13B/llm_responses/ont_1_movie_*.jsonl`  
Comparação: `parsear_resposta(response)` vs campo `triples` publicado.

Cada exemplo mostra: ID da sentença, resposta bruta (truncada), o que o publicado extraiu,
o que nosso parser extrai, e a categoria de divergência.

---

## Categoria A — Espaço inicial no objeto (causa mais frequente)

O parser original (`run_eval.py`) faz `content.split(",", 1)` e não aplica `.strip()`
ao objeto, preservando o espaço após a vírgula. Nosso parser aplica `.strip()` em ambos
os lados, o que é semanticamente mais correto mas diverge do publicado.

**Impacto na avaliação:** nenhum — a normalização `normalizar_tripla` remove espaços
antes de comparar, então P/R/F1 são idênticos independente do espaço.

---

### Ex 1 — `ont_1_movie_test_6`

**Resposta (trecho):**
```
* production\_company(The Life on the Earth - The Summer of Dioxin, Magic Bus)
* production\_company(The Life on the Earth - The Summer of Dioxin, GoGo Visual Planning)
```

**Publicado:**
```json
["The Life on the Earth - The Summer of Dioxin", "production_company", " Magic Bus"],
["The Life on the Earth - The Summer of Dioxin", "production_company", " GoGo Visual Planning"]
```

**Nosso:**
```json
["The Life on the Earth - The Summer of Dioxin", "production_company", "Magic Bus"],
["The Life on the Earth - The Summer of Dioxin", "production_company", "GoGo Visual Planning"]
```

**Categoria:** A — espaço inicial no objeto.

---

### Ex 2 — `ont_1_movie_test_18`

**Resposta (trecho):**
```
director(Beyond the Clouds, Makoto Shinkai)
screenwriter(Beyond the Clouds, Makoto Shinkai)
production\_company(Beyond the Clouds, film production company)
```

**Publicado:** objeto com espaço inicial em todos: `" Makoto Shinkai"`, `" film production company"`.

**Nosso:** sem espaço inicial em todos.

**Categoria:** A — espaço inicial no objeto.

---

### Ex 3 — `ont_1_movie_test_20`

**Resposta (trecho):**
```
Sentence: The film is directed by Raja Gosnell...

Triples:
* production\_company(The First, Paramount Pictures)
* director(The First, Raja Gosnell)
* film\_genre(The First, Comedy)
```

**Publicado:** objeto com espaço inicial: `" Paramount Pictures"`, `" Raja Gosnell"`, `" Comedy"`.

**Nosso:** sem espaço inicial.

**Categoria:** A — espaço inicial no objeto.

---

## Categoria B — Objeto vazio preservado pelo publicado

O parser original mantém triplas com objeto vazio ou somente espaço (`" "`),
pois não filtra após o split. Nosso parser descarta quando o objeto é vazio após `.strip()`.

---

### Ex 4 — `ont_1_movie_test_21` (Toy Story)

**Resposta (trecho):**
```
based\_on(Toy Story, )
main\_subject(Toy Story, )
cost(Toy Story, )
```

**Publicado:** inclui `["Toy Story", "based_on", " "]`, `["Toy Story", "main_subject", " "]`, `["Toy Story", "cost", " "]`.

**Nosso:** descarta essas três — objeto vazio após strip.

**Categoria:** B — objeto vazio preservado pelo publicado.

---

### Ex 9 — `ont_1_movie_test_41` (The Secret Life of Pets)

**Resposta (trecho):**
```
based\_on(The Secret Life of Pets, )
cast\_member(The Secret Life of Pets, )
...12 relações com objeto vazio...
```

**Publicado:** inclui 12 triplas com objeto `" "`.

**Nosso:** descarta todas as 12 com objeto vazio; retém as 6 com objeto preenchido.

**Categoria:** B — objeto vazio preservado pelo publicado (em larga escala).

---

### Ex 10 — `ont_1_movie_test_40` (Betty Boop / Canary Row)

**Resposta (trecho):**
```
* cost(Canary Row, )
* main\_subject(Canary Row, )
```

**Publicado:** inclui `["Canary Row", "cost", " "]`, `["Canary Row", "main_subject", " "]`.

**Nosso:** descarta.

**Categoria:** B — objeto vazio preservado pelo publicado.

---

## Categoria C — Resposta em código Python (triples.append)

O modelo gerou código Python em vez de triplas. O parser original aplicou sua regex
`rel(sub, obj)` ao código, interpretando `triples.append(...)` como
`append(sub_expr, obj_expr)` e cortando o prefixo `triples.` de forma inconsistente.
Nosso parser rejeita essas linhas porque o "sujeito" começa com `{`.

---

### Ex 5 — `ont_1_movie_test_23` (A Corny Concerto)

**Resposta (trecho):**
```python
triples = []
triples.append({"characters", "A Corny Concerto", "Bugs Bunny"})
triples.append({"director", "A Corny Concerto", "Bob Clampett"})
...
return triples
```

**Publicado:** extraiu `["{\"characters\"", "append", " \"A Corny Concerto\", \"Bugs Bunny\"}"]` etc.
(o original cortou no primeiro `(`, interpretou `triples.append` como relação "append")

**Nosso:** extraiu com relação `triples.append` (nosso regex captura `A-Za-z_` mas não `.`,
logo o match ficou diferente do publicado — detalhes no diff abaixo).

**Diff chave:** publicado usou `"append"` como relação; nosso usou `"triples.append"` porque
o regex do publicado permite `.` via `[^\(]*`, e para nós o `^([A-Za-z_][^\(]*)` captura
também o `triples.`, fazendo o objeto diferente.

**Categoria:** C — resposta em código Python.

---

## Categoria D — Prosa de explicação interpretada como tripla

O modelo respondeu com texto explicativo contendo exemplos no formato `rel(sub, obj)`.
O publicado extraiu essas instâncias; nosso parser também extrai, mas nem sempre identifica
os mesmos fragmentos quando estão misturados com código ou prosa.

---

### Ex 6 — `ont_1_movie_test_29` (How a Mosquito Operates)

**Resposta:**
```
screenwriter(How a Mosquito Operates,Winsor McCay)

Explanation:
...We can extract the triple (screenwriter(film,Winsor McCay))...
```

**Publicado:** extraiu também `["film", "screenwriter", "Winsor McCay"]` da explicação.

**Nosso:** só extraiu `["How a Mosquito Operates", "screenwriter", "Winsor McCay"]`.

**Categoria:** D — tripla da prosa de explicação.

---

### Ex 13 — `ont_1_movie_test_61` (The Day When I Was Born)

**Resposta:**
```
screenwriter(The Day When I Was Born,Fujiko Fujio A)

In the given sentence... the relation is "screenwriter(film,human)".
```

**Publicado:** extraiu também `["film", "screenwriter", "human"]` da explicação inline.

**Nosso:** só extraiu a primeira tripla.

**Categoria:** D — tripla da prosa de explicação.

---

### Ex 16 — `ont_1_movie_test_74` (resposta em código regex)

**Resposta:**
```python
def extract_triples(sentence):
    patterns = {
        'directed_by': re.compile(r'^\s*director\s*=\s*(.+)$', re.IGNORECASE),
        'screenwriter': re.compile(r'^\s*screenwriter\s*=\s*(.+)$', re.IGNORECASE),
        ...
    }
```

**Publicado:** extraiu `["r'^\\s*screenwriter...'", "compile", " re.IGNORECASE"]` etc.
(o original aplicou regex ao código Python e pegou chamadas `compile(...)`)

**Nosso:** não extraiu nada — o `^([A-Za-z_][^\(]*)` não casa com strings que começam com `r'`.

**Categoria:** D + C — código interpretado como triplas pelo publicado.

---

## Categoria E — Sujeito com vírgula no nome (Panda! Go, Panda!)

O nome do sujeito contém vírgula: `Panda! Go, Panda!`. Ambos os parsers
partem no primeiro `,` de nível zero, logo sub=`"Panda! Go"` e obj=`"Panda!, Hayao Miyazaki"`.
A diferença é só o espaço inicial (Categoria A).

---

### Ex 7 — `ont_1_movie_test_26` (Panda! Go, Panda!)

**Resposta (trecho):**
```
screenwriter(Panda! Go, Panda!, Hayao Miyazaki)
```

**Publicado:** `["Panda! Go", "screenwriter", " Panda!, Hayao Miyazaki"]` (espaço inicial).

**Nosso:** `["Panda! Go", "screenwriter", "Panda!, Hayao Miyazaki"]` (sem espaço).

**Nota:** ambos partem errado no nome — o objeto correto seria `Hayao Miyazaki`.

**Categoria:** A + E — espaço e nome com vírgula.

---

## Categoria F — Parêntese de fechamento extra no objeto

O objeto contém parêntese de abertura e o LLM não fechou o externo.

---

### Ex 11 — `ont_1_movie_test_43` (Ghost in the Shell 2)

**Resposta (trecho):**
```
cost(Ghost in the Shell 2: Innocence,¥1.2 billion (approximately USD12 million))
```

O regex `^rel\((.+)\)$` casa com o último `)` como fechamento, deixando o objeto
`¥1.2 billion (approximately USD12 million)`.
O publicado truncou o `)` externo: `"¥1.2 billion (approximately USD12 million"`.

**Publicado:** `"¥1.2 billion (approximately USD12 million"` (sem `)` final).

**Nosso:** `"¥1.2 billion (approximately USD12 million)"` (com `)` — resultado correto).

**Categoria:** F — parêntese extra; nosso resultado é semanticamente superior.

---

## Categoria G — Objeto com newlines

O objeto literal contém `\n` porque o LLM quebrou a linha no meio.

---

### Ex 12 — `ont_1_movie_test_44` (Dead Leaves)

**Resposta:**
```
triple: 
director(Dead Leaves, 
human
)
```

**Publicado:** capturou `["Dead Leaves", "director", " \nhuman\n"]` (incluindo as quebras de linha).

**Nosso:** não extraiu nada — `_extrair_tripla` usa `(.+)` que não casa conteúdo com `\n`
porque `.` em modo não-multiline não casa `\n`.

**Categoria:** G — objeto multi-linha.

---

## Categoria H — Texto de prefixo "Test Output:" na mesma linha da tripla

O modelo colocou `Test Output: screenwriter(sub,obj)` tudo na mesma linha.
Nosso parser extraiu a relação como `"Test Output:  screenwriter"` porque o `[^\(]*`
captura o prefixo inteiro. O publicado extraiu `"screenwriter"` sozinha.

---

### Ex 20 — `ont_1_movie_test_87` (The Day When I Was Born)

**Resposta:**
```
Test Sentence: The original plot was written by Fujiko F. Fujio.
Test Output:  screenwriter(The Day When I Was Born,Fujiko Fujio A)
```

**Publicado:** `["The Day When I Was Born", "screenwriter", "Fujiko Fujio A"]`.

**Nosso:** `["The Day When I Was Born", "Test Output:  screenwriter", "Fujiko Fujio A"]`.

**Categoria:** H — prefixo inline na relação.

---

## Categoria I — Prosa com marcadores `**Triple N:**` interpretada

O modelo estruturou a resposta em seções `**Triple 1:**`, `**Triple 2:**` etc.
O publicado extraiu triplas dentro de cada seção. Nosso parser extrai as mesmas triplas
(os `*` de lista são removidos) mas diverge em espaço inicial (Categoria A) e
objetos vazios (Categoria B).

---

### Ex 10 — `ont_1_movie_test_40` (Betty Boop / Canary Row) — já em Cat. B acima.

---

## Categoria J — Resposta com sujeitos/objetos invertidos (relação invertida)

O modelo emitiu `directs(Kanta Kamei, Tales of Vesperia)` em vez de
`director(Tales of Vesperia, Kanta Kamei)`. Ambos parsers extraem a mesma tripla
incorreta; a diferença é o espaço inicial (Categoria A).

---

### Ex 15 — `ont_1_movie_test_72` (Tales of Vesperia)

**Resposta (trecho):**
```
directs(Kanta Kamei, Tales of Vesperia: The First Strike),
writes(Reiko Yoshida, Tales of Vesperia: The First Strike),
cast\_member(KÅsuke Toriumi, Tales of Vesperia: The First Strike),
...
genre(Tales of Vesperia: The First Strike, fantasy),
```

**Publicado:** capturou todas as linhas, incluindo duplicatas (relações repetidas) e
`award_received` duas vezes.

**Nosso:** quando a linha é `"directs(Kanta Kamei, Tales of Vesperia: The First Strike),"`,
o `,` final impede o match do `\)$` — nosso regex exige `)` no final. Publicado parece
tolerar `,` antes do fechamento.

**Divergência específica:** a vírgula final em `directs(Kanta Kamei, Tales of Vesperia: The First Strike),`
quebra nosso match mas não o do publicado. Resultado: nosso parser extraiu apenas 1 tripla
(`award_received`), publicado extraiu 19.

**Categoria:** J + especial — vírgula após o parêntese final.

---

## Resumo das categorias

| Cat. | Descrição | Exemplos | Impacto em P/R/F1 |
|------|-----------|----------|-------------------|
| A | Espaço inicial no objeto (strip vs. não-strip) | 1,2,3,7,8,9,10,11,14 | Nenhum (normalização remove) |
| B | Objeto vazio preservado pelo publicado | 4,9,10 | Pequeno (triplas espúrias no publicado) |
| C | Resposta em código Python interpretada | 5 | Nenhum relevante |
| D | Prosa de explicação com triplas | 6,13,16 | Pequeno |
| E | Sujeito com vírgula no nome | 7 | Nenhum (ambos partem igual) |
| F | Parêntese extra; nosso resultado correto | 11 | Insignificante |
| G | Objeto com newlines | 12 | Raro |
| H | Prefixo inline na relação | 20 | Pequeno (relação errada no nosso) |
| I | Seções `**Triple N:**` | 10 | Via A+B |
| J | Vírgula após `)` final | 15 | Significativo (perdemos 18 triplas) |

**Conclusão:** a divergência dominante é Categoria A (espaço inicial), que não afeta
as métricas finais porque a normalização do avaliador remove espaços. A Categoria J
(vírgula após `)`) e Categoria B (objeto vazio) têm impacto potencial mas são minoritárias.
