# Decisões de projeto — aguardando OK

Este arquivo documenta decisões metodológicas que requerem aprovação explícita
antes de implementação.

---

## D-01: Formato do JSON para a condição C5

**Status:** AGUARDANDO OK

### Contexto

A condição C5 fornece a ontologia ao modelo como JSON embutido no prompt.
O arquivo de ontologia do benchmark contém campos em dois níveis de detalhe:

**Nível 1 — campos semânticos essenciais** (suficiente para identificar relações):
```json
{
  "title": "Movie Ontology",
  "concepts": [{"label": "film"}, {"label": "human"}, ...],
  "relations": [{"label": "director", "domain": "film", "range": "human"}, ...]
}
```

**Nível 2 — JSON completo do benchmark** (inclui identificadores Wikidata/DBpedia):
```json
{
  "title": "Movie Ontology",
  "id": "ont_1_movie",
  "concepts": [{"qid": "Q11424", "label": "film"}, ...],
  "relations": [{"pid": "P57", "label": "director", "domain": "Q11424", "range": "Q5"}, ...]
}
```

### Proposta

Implementar **duas variantes** de C5:

| Condição | JSON | Raciocínio |
|---|---|---|
| **C5** (nível 1) | Campos `title`, `concepts[label]`, `relations[label, domain, range]` com labels resolvidos | Minimiza ruído: o modelo não precisa interpretar QIDs (`Q11424`, `P57`). Domínio/range são os labels humanos como no C1. |
| **C5_full** (nível 2) | JSON bruto exatamente como no arquivo, incluindo `qid`/`pid` e domínio como QID | Permite testar se o modelo sabe interpretar identificadores Wikidata. Hipótese: pode ajudar modelos treinados em dados Wikidata. |

**Recomendação:** Implementar C5 (nível 1) como variante principal e C5_full como variante secundária para a análise de ablação. Nível 1 é mais limpo e comparável com C1.

---

## D-02: Formato do contexto para a condição C3

**Status:** AGUARDANDO OK

### Contexto

A condição C3 fornece ao modelo apenas os labels das relações (sem domain/range, sem conceitos, sem exemplos). É zero-shot por definição.

Implementação atual:
```
Relations: director, screenwriter, genre, based_on, cast_member, ...
```

### Questão em aberto

Manter zero-shot (k=0) ou incluir um exemplo (k=1)?

| Variante | Descrição | Vantagem | Desvantagem |
|---|---|---|---|
| **C3** (k=0, proposta) | Só a lista de labels, sem exemplo | Mede o ganho mínimo de fornecer nomes de relações | Pode ser muito difícil sem exemplo de formato |
| C3_1shot | Lista de labels + 1 exemplo do benchmark | Separa o efeito do formato vs. o efeito da ontologia | Confunde o efeito com C1_oneshot |

**Recomendação:** k=0. O objetivo de C3 é isolar o efeito de *nomear as relações disponíveis* sem revelar a estrutura (domain/range/conceitos). Adicionar um exemplo introduziria um segundo fator de variação e aproximaria C3 demais de C1_oneshot. Se a comparação C3 vs C1 mostrar diferença grande, podemos incluir C3_1shot como ablação na Fase 2.

---

## D-03: Estratégia para o campo "Test Output: " nos prompts de chat

**Status:** AGUARDANDO OK

### Contexto

O prompt publicado no benchmark termina com `Test Output: ` — artefato de modelos de completamento (Vicuna, Alpaca). Os modelos que rodaremos (Qwen2.5-AWQ, etc.) são modelos de *chat* acessados via API OpenAI-compat, onde o papel de "usuário" é separado do papel de "assistente".

### Variantes consideradas

| Opção | Descrição |
|---|---|
| **A — incluir "Test Output: "** (implementado em C1_oneshot) | Mantém fidelidade máxima ao benchmark; o modelo de chat recebe o prefixo como parte da mensagem e geralmente o ignora ou continua a partir dele |
| B — remover "Test Output: " | Mais limpo para modelos de chat; o modelo responde diretamente sem o prefixo |
| C — mover para mensagem de assistente | Passar `{"role":"assistant","content":"Test Output: "}` como prefixo forçado — suportado por alguns backends mas não padrão OpenAI |

**Recomendação:** Opção A para C1_oneshot_original (máxima fidelidade ao benchmark); opção B para C0/C1/C3/C5 (são condições novas, sem prompt publicado de referência). Permite comparação direta dos resultados do C1_oneshot com os do benchmark original.

---

*Decisões aprovadas serão movidas para docs/DECISOES_APROVADAS.md.*
