# Resultados — Experimento completo Qwen 2.5-7B

**Modelo:** Qwen/Qwen2.5-7B-Instruct-AWQ (AWQ-4bit) · temperatura 0 · seed 42 · vLLM (OpenAI-compat).
**Escopo:** 203 experimentos (manifesto `configs/pipeline/manifesto_7b_completo.json`), 42,532 frases
de teste — TekGen (10 ontologias × 7 condições) e WebNLG (19 ontologias × 7 condições).
**Resultados brutos:** `resultados/full/qwen2.5-7b/*.jsonl`.
**Avaliação por frase:** `resultados/avaliacao_full/qwen2.5-7b/` (mesmo formato do piloto-v3, lido pelo app).
**Reprodução:** `python scripts/avaliar_full.py --manifesto configs/pipeline/manifesto_7b_completo.json --slug qwen2.5-7b --rodada full`.

## Metodologia da avaliação

- **Régua original** (espelho do Text2KGBench, `t2kg.avaliacao.original`): F1 só sobre triplas do
  sistema cuja relação consta do gabarito da frase; normalização sub+rel+obj sem espaços/underscores.
- **F1 projetado** (régua estendida): cada relação livre é mapeada para a relação mais similar da
  ontologia de referência (`all-MiniLM-L6-v2`, modo *label+def*), aceitando quando SimE ≥ **0.45**
  — limiar lido da fonte única `configs/avaliacao.json` (full = 0,45, D-36).
- **Conformidade:** fração das triplas do sistema cuja relação pertence à ontologia de referência.
  Saída vazia conta conformidade = 1 (regra original); por isso **Conf. sem vazias** (só frases com
  ≥1 tripla) é reportada à parte.
- **% vazias:** frações de frases com `triples == []`.
- **Redundância:** redundância semântica de esquema (H1), `1 − n_grupos/n_rótulos`, limiar **0.80**.
- **Alucinações:** alucinação de entidade = média de sujeito e objeto fora do contexto (frase + conceitos).
- **Agregação:** macro-média entre ontologias (como no benchmark original).
- **Intervalos [IC 95%]:** *cluster-bootstrap* com **seed 42** (1000 reamostragens): reamostram-se
  as frases dentro de cada ontologia e recompõe-se a macro-média; a redundância e a sobreposição
  (nível de esquema) usam bootstrap sobre as ontologias.

## TekGen (10 ontologias)

| Condição | F1 original | F1 projetado | Conformidade | Conf. sem vazias | % vazias | Redundância | Alucinações |
|---|---|---|---|---|---|---|---|
| C0 (livre) | 0.017 [0.013–0.021] | 0.033 [0.027–0.038] | 0.040 [0.036–0.045] | 0.029 [0.027–0.033] | 1.1% [0.8–1.5] | 0.171 [0.139–0.204] | 0.180 [0.169–0.191] |
| C1 | 0.188 [0.175–0.201] | 0.135 [0.125–0.144] | 0.933 [0.926–0.940] | 0.930 [0.923–0.937] | 5.5% [4.7–6.3] | 0.054 [0.029–0.080] | 0.077 [0.071–0.083] |
| C1 one-shot orig. | 0.409 [0.393–0.424] | 0.350 [0.337–0.365] | 0.934 [0.926–0.941] | 0.933 [0.926–0.940] | 0.8% [0.5–1.2] | 0.069 [0.039–0.102] | 0.089 [0.083–0.096] |
| C2 (induzida) | 0.035 [0.031–0.041] | 0.068 [0.060–0.076] | 0.132 [0.123–0.141] | 0.083 [0.078–0.088] | 5.4% [4.7–6.2] | 0.077 [0.048–0.106] | 0.054 [0.049–0.059] |
| C3 | 0.172 [0.160–0.185] | 0.123 [0.113–0.131] | 0.913 [0.906–0.920] | 0.912 [0.904–0.920] | 1.2% [0.8–1.6] | 0.058 [0.027–0.090] | 0.072 [0.067–0.078] |
| C5 vizinho | 0.007 [0.006–0.009] | 0.009 [0.007–0.011] | 0.454 [0.440–0.469] | 0.099 [0.094–0.104] | 36.6% [35.3–38.2] | 0.076 [0.050–0.102] | 0.172 [0.162–0.181] |
| C5 incongruente | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.787 [0.773–0.800] | 0.000 [0.000–0.000] | 78.7% [77.3–80.2] | 0.113 [0.017–0.241] | 0.107 [0.099–0.115] |

## WebNLG (19 ontologias)

| Condição | F1 original | F1 projetado | Conformidade | Conf. sem vazias | % vazias | Redundância | Alucinações |
|---|---|---|---|---|---|---|---|
| C0 (livre) | 0.034 [0.027–0.041] | 0.132 [0.121–0.143] | 0.135 [0.121–0.149] | 0.124 [0.112–0.137] | 1.3% [0.8–1.8] | 0.276 [0.246–0.307] | 0.105 [0.093–0.118] |
| C1 | 0.220 [0.206–0.233] | 0.204 [0.192–0.216] | 0.962 [0.955–0.969] | 0.961 [0.953–0.967] | 3.3% [2.6–4.1] | 0.019 [0.008–0.032] | 0.103 [0.093–0.113] |
| C1 one-shot orig. | 0.435 [0.419–0.452] | 0.411 [0.397–0.428] | 0.978 [0.974–0.983] | 0.978 [0.974–0.982] | 0.1% [0.0–0.3] | 0.023 [0.006–0.050] | 0.123 [0.115–0.131] |
| C2 (induzida) | 0.013 [0.010–0.017] | 0.082 [0.074–0.090] | 0.157 [0.146–0.167] | 0.119 [0.111–0.127] | 4.7% [3.7–5.6] | 0.079 [0.053–0.103] | 0.039 [0.034–0.045] |
| C3 | 0.172 [0.160–0.185] | 0.156 [0.145–0.166] | 0.920 [0.911–0.928] | 0.918 [0.910–0.927] | 1.0% [0.6–1.3] | 0.041 [0.018–0.068] | 0.099 [0.088–0.111] |
| C5 vizinho | 0.103 [0.092–0.115] | 0.105 [0.094–0.117] | 0.716 [0.700–0.733] | 0.672 [0.654–0.690] | 13.2% [11.6–14.8] | 0.039 [0.023–0.054] | 0.164 [0.150–0.179] |
| C5 incongruente | 0.040 [0.034–0.046] | 0.045 [0.038–0.052] | 0.507 [0.486–0.527] | 0.326 [0.305–0.348] | 26.0% [24.1–27.9] | 0.038 [0.024–0.053] | 0.168 [0.156–0.181] |

## Sobreposição da C5 (interpretação da H3)

A C5 troca a ontologia-alvo pela de uma parceira (vizinha ou incongruente). Parte do desempenho
residual da C5 pode vir de **relações que a parceira compartilha com a referência**. A tabela abaixo
mostra, por dataset e tipo de C5, a sobreposição média (fração das relações da referência que também
existem na parceira) e o F1 da C5 recalculado **apenas sobre triplas cujas relações do gabarito NÃO
estão na parceira** — isolando o sinal que não pode ser explicado por sobreposição de esquema.

| Dataset | Tipo C5 | Sobreposição ref∩parceira / ref | F1 C5 (todas) | F1 C5 (relações fora da parceira) |
|---|---|---|---|---|
| TekGen | vizinho | 8.3% [0.0–16.3] | 0.007 [0.006–0.009] | 0.005 [0.003–0.006] |
| TekGen | incongruente | 0.0% [0.0–0.0] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] |
| WebNLG (DBpedia) | vizinho | 33.7% [29.5–38.5] | 0.103 [0.092–0.115] | 0.021 [0.016–0.028] |
| WebNLG (DBpedia) | incongruente | 2.9% [1.6–4.5] | 0.040 [0.034–0.046] | 0.023 [0.018–0.029] |

Ver nota de interpretação da **H3** em `docs/DECISOES.md`.

## Leitura rápida

- **C1 one-shot original** lidera em F1 nos dois datasets, com alta conformidade e pouquíssimas vazias.
- **C0 e C2** têm conformidade baixa (vocabulário livre / induzido): o F1 **projetado** recupera parte
  do sinal, confirmando triplas corretas com rótulos fora da ontologia-alvo.
- **C5 incongruente** colapsa em TekGen (F1 ≈ 0; maioria das respostas vazias); em WebNLG o colapso é
  parcial, coerente com ontologias-alvo mais próximas e maior sobreposição de esquema (ver H3).
