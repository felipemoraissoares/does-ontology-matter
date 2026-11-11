# Resultados Piloto v3 — Métricas Corrigidas

> Gerado automaticamente. Avaliação com fixes D-26 a D-30.

> am = modo amostra; Conf-nv / Hal-nv = excluindo respostas vazias; F1-proj = projeção D-24.


## Tabela compacta: F1 por ontologia × condição

| Dataset | Onto | C0 | C1 | C1+ex | C2 | C3 | C5v | C5i |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tekgen | 1_movie | 0.04† | 0.25† | 0.30† | 0.12† | 0.15† | 0.02† | 0.00† |
| tekgen | 2_music | 0.02† | 0.22† | 0.37† | 0.03† | 0.19† | 0.01† | 0.00† |
| **tekgen** | **média** | **0.03** | **0.23** | **0.33** | **0.07** | **0.17** | **0.01** | **0.00** |
| webnlg | 1_university | 0.05 | 0.23 | 0.41 | 0.02 | 0.14 | 0.08 | 0.01 |
| webnlg | 19_film | 0.06† | 0.07† | 0.32† | 0.01† | 0.01† | 0.01† | 0.01† |
| **webnlg** | **média** | **0.06** | **0.15** | **0.36** | **0.01** | **0.08** | **0.04** | **0.01** |

† = modo amostra (n_sys < n_gt)


## Tabela detalhada: todas as métricas por ontologia × condição

| Dataset | Onto | Cond | am | P | R | F1 | Conf | Conf-nv | Hal-nv | FmtErr | ParseErr | %Vaz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tekgen | 1_movie | C0 | S | 0.08 | 0.03 | 0.04 | 0.11 | 0.09 | 0.91 | 0.01 | 0.02 | 0.02 |
| tekgen | 2_music | C0 | S | 0.02 | 0.01 | 0.02 | 0.06 | 0.02 | 0.98 | 0.01 | 0.04 | 0.04 |
| tekgen | 1_movie | C1 | S | 0.29 | 0.24 | 0.25 | 0.95 | 0.94 | 0.06 | 0.18 | 0.05 | 0.05 |
| tekgen | 2_music | C1 | S | 0.28 | 0.19 | 0.22 | 0.94 | 0.93 | 0.07 | 0.27 | 0.08 | 0.08 |
| tekgen | 1_movie | C1+ex | S | 0.36 | 0.29 | 0.30 | 0.97 | 0.97 | 0.03 | 0.08 | 0.01 | 0.01 |
| tekgen | 2_music | C1+ex | S | 0.47 | 0.34 | 0.37 | 0.98 | 0.98 | 0.02 | 0.06 | 0.03 | 0.03 |
| tekgen | 1_movie | C2 | S | 0.20 | 0.09 | 0.12 | 0.31 | 0.27 | 0.73 | 0.03 | 0.06 | 0.06 |
| tekgen | 2_music | C2 | S | 0.04 | 0.02 | 0.03 | 0.10 | 0.05 | 0.95 | 0.01 | 0.05 | 0.05 |
| tekgen | 1_movie | C3 | S | 0.22 | 0.12 | 0.15 | 0.84 | 0.84 | 0.16 | 0.07 | 0.00 | 0.00 |
| tekgen | 2_music | C3 | S | 0.28 | 0.16 | 0.19 | 0.97 | 0.97 | 0.03 | 0.06 | 0.02 | 0.02 |
| tekgen | 1_movie | C5v | S | 0.02 | 0.02 | 0.02 | 0.38 | 0.35 | 0.65 | 0.19 | 0.05 | 0.05 |
| tekgen | 2_music | C5v | S | 0.01 | 0.01 | 0.01 | 0.46 | 0.37 | 0.63 | 0.07 | 0.14 | 0.14 |
| tekgen | 1_movie | C5i | S | 0.00 | 0.00 | 0.00 | 0.76 | 0.00 | 1.00 | 0.01 | 0.76 | 0.76 |
| tekgen | 2_music | C5i | S | 0.00 | 0.00 | 0.00 | 0.98 | 0.00 | 1.00 | 0.00 | 0.98 | 0.98 |
| webnlg | 1_university | C0 | N | 0.10 | 0.04 | 0.05 | 0.27 | 0.27 | 0.73 | 0.00 | 0.00 | 0.00 |
| webnlg | 19_film | C0 | S | 0.08 | 0.05 | 0.06 | 0.15 | 0.14 | 0.86 | 0.00 | 0.01 | 0.01 |
| webnlg | 1_university | C1 | N | 0.29 | 0.20 | 0.23 | 0.96 | 0.96 | 0.04 | 0.03 | 0.00 | 0.00 |
| webnlg | 19_film | C1 | S | 0.08 | 0.07 | 0.07 | 0.93 | 0.92 | 0.08 | 0.43 | 0.07 | 0.07 |
| webnlg | 1_university | C1+ex | N | 0.50 | 0.36 | 0.41 | 0.96 | 0.96 | 0.04 | 0.00 | 0.00 | 0.00 |
| webnlg | 19_film | C1+ex | S | 0.33 | 0.31 | 0.32 | 0.96 | 0.96 | 0.04 | 0.04 | 0.00 | 0.00 |
| webnlg | 1_university | C2 | N | 0.04 | 0.02 | 0.02 | 0.69 | 0.69 | 0.31 | 0.00 | 0.01 | 0.01 |
| webnlg | 19_film | C2 | S | 0.01 | 0.01 | 0.01 | 0.30 | 0.12 | 0.88 | 0.00 | 0.20 | 0.20 |
| webnlg | 1_university | C3 | N | 0.23 | 0.12 | 0.14 | 0.78 | 0.78 | 0.22 | 0.01 | 0.00 | 0.00 |
| webnlg | 19_film | C3 | S | 0.01 | 0.01 | 0.01 | 0.58 | 0.57 | 0.43 | 0.10 | 0.02 | 0.02 |
| webnlg | 1_university | C5v | N | 0.12 | 0.06 | 0.08 | 0.70 | 0.70 | 0.30 | 0.02 | 0.01 | 0.01 |
| webnlg | 19_film | C5v | S | 0.01 | 0.01 | 0.01 | 0.52 | 0.45 | 0.55 | 0.06 | 0.12 | 0.12 |
| webnlg | 1_university | C5i | N | 0.02 | 0.01 | 0.01 | 0.10 | 0.09 | 0.91 | 0.31 | 0.01 | 0.01 |
| webnlg | 19_film | C5i | S | 0.01 | 0.01 | 0.01 | 0.49 | 0.18 | 0.82 | 0.26 | 0.38 | 0.38 |


## Métricas de projeção (global por dataset × condição)

> D-30: limiares recalibrados — label-only=0.50; label+def (D-24)=0.55.
> Antes da calibração (thr=0.80): webnlg/C0=0.005 (frac=0.8%); C3=0.000. Após: C0=0.118 (34%), C3=0.111 (64%).

| Dataset | Cond | F1-proj | %Frac-proj |
| --- | --- | --- | --- |
| tekgen | C0 | 0.032 | 0.246 |
| tekgen | C1 | 0.174 | 0.894 |
| tekgen | C1+ex | 0.278 | 0.950 |
| tekgen | C2 | 0.063 | 0.441 |
| tekgen | C3 | 0.125 | 0.948 |
| tekgen | C5v | 0.014 | 0.403 |
| tekgen | C5i | 0.000 | 0.010 |
| webnlg | C0 | 0.118 | 0.342 |
| webnlg | C1 | 0.139 | 0.712 |
| webnlg | C1+ex | 0.317 | 0.765 |
| webnlg | C2 | 0.035 | 0.275 |
| webnlg | C3 | 0.111 | 0.637 |
| webnlg | C5v | 0.039 | 0.353 |
| webnlg | C5i | 0.024 | 0.141 |


## Fix 5: Comportamento C5_incongruente — recusa vs invenção


**TekGen 1_movie e 2_music**: 76–98% de respostas vazias. O modelo reconhece que a
ontologia é incompatível com a sentença e recusa a extração explicitamente:
- *"There are no entities related to the ontology relations in the given test sentence."*
- *"The provided sentence does not contain any entities or relationships that match..."*
- *"NoOutput"*, *"No triples extracted."*

Resultado: `pct_vazias ≈ 0.76–0.98`, `avg_onto_conf_nv = 0.00` (13–24% que extraem
algo, extraem relações não conformantes). O alto `avg_onto_conf ≈ 0.87–0.98` reflete
o viés de conformidade das respostas vazias (triples=[] → conformidade=1.0 por padrão).

**WebNLG 1_university**: apenas 1% de respostas vazias. O modelo **inventa** relações
usando a ontologia incongruente como molde:
- Extrai `affiliatedTo`, `offers`, `founder`, `president` usando ontologia de astronomia
- Usa **nomes de classe como entidades** (`CelestialBody`, `Person`, `School`) → fmt_error
- Resultado: `avg_onto_conf_nv = 0.09`, `avg_rel_halluc_nv = 0.91`, `avg_fmt_error = 0.31`

**Hipótese explicativa**: TekGen usa ontologias de domínio fechado (biologia, esportes)
com relações muito específicas e sem overlap semântico com filmes/músicas. WebNLG 19_film
recebe ontologia de alimentação/países — mais "genérica" e com maior chance de match
superficial. 1_university recebe ontologia de astronomia/taxons.
