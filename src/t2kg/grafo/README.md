# Fase 4 — Grafo no Neo4j

Carrega os resultados de um run no Neo4j para exploração visual e consultas.

## Modelo de grafo

```
(:Entidade {nome})
(:Frase {id, texto})
(:Frase)-[:MENCIONA]->(:Entidade)
(:Entidade)-[:REL {relacao, condicao, modelo, run, status, frase}]->(:Entidade)
```

- **REL** é uma tripla extraída pelo modelo. `status` = `acerto` (coincide com o gabarito da
  frase) ou `sobra` (não coincide), conforme o avaliador.
- `frase` na aresta REL guarda o id da frase — permite comparar condições na MESMA frase.

## Credenciais

Apenas do `.env` (nunca no código nem em commit). Copie `.env.example` para `.env` e preencha:

```
NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
```

## Carregar um run

Carga **por run** (o AuraDB Free limita ~200k nós / ~400k relações):

```
python -m t2kg.grafo.carregar_neo4j --run pv3_1movie_c1
```

Para recarregar um run (apaga as arestas REL dele antes):

```
python -m t2kg.grafo.carregar_neo4j --run pv3_1movie_c1 --apagar_run
```

O run é resolvido pelo `manifesto_piloto_v3.json` (dataset/ontologia/condição/modelo) e o
JSONL é localizado em `resultados/`. Para outros manifestos/arquivos use `--manifesto`,
`--jsonl`, `--dataset`, `--ontologia`, `--condicao`.

## Consulta de exemplo (Cypher)

**Arestas da C0 que não existem na C1 para a mesma frase** (triplas que o modelo só produz
sem ontologia):

```cypher
MATCH (s:Entidade)-[c0:REL {condicao: 'C0'}]->(o:Entidade)
WHERE NOT EXISTS {
  MATCH (s)-[c1:REL {condicao: 'C1'}]->(o)
  WHERE c1.frase = c0.frase AND c1.relacao = c0.relacao AND c1.modelo = c0.modelo
}
RETURN c0.frase AS frase, s.nome AS sujeito, c0.relacao AS relacao,
       o.nome AS objeto, c0.status AS status
ORDER BY frase
```

(Para comparar runs específicos, acrescente `AND c0.run = '...'` / `c1.run = '...'`.)
