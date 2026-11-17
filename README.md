# Does Ontology Matter? — LLM Triple Extraction Benchmark

A reproducible experimental framework to measure the effect of ontology provision strategy
on knowledge graph triple extraction by large language models, built on top of the
[Text2KGBench](https://github.com/nandana/Text2KGBench) benchmark (Mihindukulasooriya et al., ISWC 2023).

## Overview

Knowledge graph construction from natural language requires a model to identify entities and
the semantic relations between them. Contemporary approaches provide the target ontology as
context, but the optimal way to do so remains unclear: does showing the full ontology help
or hurt compared to showing nothing? Does a corrupted ontology mislead the model or is it
ignored? Does an automatically-induced vocabulary outperform a hand-crafted one?

This project answers these questions through a controlled experiment with seven conditions
that vary only the ontology block injected into the prompt, keeping model, instruction, and
sentence fixed. All conditions are executed on the same benchmark (Text2KGBench), using the
same model (Qwen2.5-7B-Instruct-AWQ, temperature 0), and evaluated with an exact mirror of
the original evaluation script. This design allows direct attribution of metric differences
to the ontology strategy rather than to other confounds.

The two primary hypotheses are:
- **H1**: Models operating without an ontology (C0) produce semantically redundant relation
  vocabularies — their free extractions are less constrained than those produced under
  ontology guidance. Measured by semantic redundancy at cosine threshold 0.70.
- **H2**: Providing the correct ontology (C1) increases extraction quality over no ontology
  (C0) on conformity and F1, and that effect degrades progressively as the provided ontology
  diverges from the correct one (C5_vizinho > C5_incongruente).

The expected practical contribution is a set of pre-registered, reproducible experimental
results on two benchmark datasets (TekGen/Wikidata and DBpedia/WebNLG) that characterize
the marginal value of different ontology provision strategies for triple extraction.

## Research Design

### Experimental Conditions

| Condition | Ontology in prompt | Zero-shot | Instruction |
|---|---|:---:|---|
| **C0** | None | Yes | Free extraction — extract all relevant triples |
| **C1** | Full reference ontology (concepts + typed relations) | Yes | Guided by ontology |
| **C1_oneshot_original** | Full reference ontology, exact benchmark format | No (1-shot) | Original benchmark instruction |
| **C2** | Induced ontology — clustered from free C0 extractions on 50 training sentences | Yes | Guided by ontology |
| **C3** | Top-3 most frequent relations from training data, with domain/range | Yes | Guided by key relations |
| **C5_vizinho** | Neighbour ontology (highest schema similarity to target) | Yes | Guided by ontology |
| **C5_incongruente** | Incongruent ontology (lowest schema similarity to target) | Yes | Guided by ontology |

Schema similarity is measured by bidirectional embedding similarity (SimE) over relation
labels using `all-MiniLM-L6-v2`, with pairs pre-registered in `configs/pipeline/pares_c5.json`
before any experimental execution (decision D-02).

All conditions except C1_oneshot_original use the format instruction selected by the D-19
mini-test (decision D-20, Variant D): relation names and entities described with angle-bracket
markers to reduce class-name copying errors.

### Datasets

| Dataset | Ontologies | Test sentences per ontology |
|---|---|---|
| TekGen (Wikidata) | 10 | ~400 |
| DBpedia (WebNLG) | 19 | ~200–400 |

### Pilot Scope (piloto-v3)

28 experiments across 4 ontologies (TekGen: 1_movie, 2_music; DBpedia: 1_university, 19_film)
× 7 conditions, 100 sentences each, run on Kaggle T4×2.

## Repository Structure

```
llm-kg-ontology/
├── src/t2kg/
│   ├── pipeline/
│   │   ├── dados.py           # Loads benchmark sentences and training frequencies
│   │   ├── ontologias.py      # Loads/formats ontology context (C2 per-model: v_full/<slug>)
│   │   ├── prompts.py         # Assembles the full prompt string per condition
│   │   ├── executar.py        # Runs extraction with checkpoint/resume support
│   │   └── provedores/        # Inference backends (openai_compat, mock)
│   ├── avaliacao/
│   │   ├── original.py        # Exact mirror of Text2KGBench run_eval.py
│   │   └── extended.py        # Extended metrics: semantic redundancy, projection (D-34)
│   ├── grafo/
│   │   └── carregar_neo4j.py  # Fase 4: loads a run into Neo4j (acerto/sobra per edge)
│   ├── modelos.py             # slug_modelo() for per-model C2 paths (D-33)
│   ├── parser.py              # Parses LLM responses into (sub, rel, obj) triples
│   └── run.py                 # CLI entry point: reads manifesto, calls executar()
│
├── scripts/
│   ├── gerar_manifesto.py     # Builds the full-experiment manifest (203/model, session split)
│   ├── montar_wheelhouse.py   # Builds the Kaggle wheelhouse (fast offline install)
│   ├── rodar_kaggle.py        # Dispatcher: MODO inducao_c2 | experimento
│   ├── induzir_c2_lote.py     # C2 induction over the 29 ontologies (per model)
│   ├── induzir_c2.py          # Single-ontology C2 induction: free extraction + clustering
│   ├── gerar_definicoes_relacoes.py  # LLM definitions for EDC (D-24), --saida_dir
│   ├── guard_check_c2.py      # Blocks C2 experiments without the model's induced ontology
│   ├── recalibrar_c2_29.py    # Recomputes D-21/D-30 thresholds over 29 ontologies (CPU)
│   ├── reproduzir_baselines.py       # Reproduces the published baseline metrics (6/6)
│   ├── avaliar_resultados.py  # Evaluates results using a manifesto JSON
│   └── smoke_experimento_completo.py # Mock smoke of the full flow (both MODO)
│
├── configs/
│   ├── pipeline/
│   │   ├── manifesto_piloto_v3.json    # 28-experiment pilot manifest
│   │   ├── manifesto_7b_completo.json  # Full experiment (203: 29 onto × 7 cond)
│   │   ├── manifesto_c2_inducao_29.json# C2 induction tasks for the 29 ontologies
│   │   └── pares_c5.json               # Pre-registered C5 ontology pairs
│   └── c2/
│       ├── *_induced.json              # Pilot induced ontologies
│       ├── definicoes_*.json           # Pilot relation definitions (D-24)
│       └── v_full/<slug>/              # Per-model induced ontologies + definitions (D-33)
│
├── app/                       # App Streamlit de exploração (Fases 4–5)
│   ├── Inicio.py              # Entrypoint: st.navigation (python -m streamlit run app/Inicio.py)
│   ├── dados/                 # Camada de dados (sem Streamlit, testável)
│   ├── paginas/              # Início, Fases, Explorador, Ontologias, Grafo, Resultados, Chat
│   └── componentes/          # CSS, grafo pyvis, chips, filtros, estado
│
├── kaggle/
│   └── lancador.ipynb         # Kaggle launcher with a single MODO switch
│
├── resultados/
│   ├── piloto_v3/             # Pilot JSONL results (auto-pushed from Kaggle)
│   ├── avaliacao_piloto_v3/   # Evaluation metrics and CSV summary
│   ├── full/<slug>/           # Full-experiment results per model
│   └── c2_inducao/            # Raw C0 extractions from the C2 training phase
│
├── requirements-kaggle.txt    # Pinned Kaggle stack (D-35)
├── requirements-app.txt       # Local tooling (Neo4j, Streamlit) for Fases 4–5
│
├── tests/                     # pytest suite (1082 tests, all pass locally)
│
└── docs/
    ├── DECISOES.md                    # Authoritative log of all methodological decisions (D-01–D-35)
    ├── auditoria_pre_experimento.md   # Pre-experiment audit + correction status
    ├── experimento_completo_kaggle.md # Step-by-step for the full experiment on Kaggle
    └── kaggle_wheelhouse.md           # How to build/attach the wheelhouse dataset
```

The Text2KGBench benchmark data is **not included** in this repository. It must be cloned
separately at the sibling path `../Text2KGBench` (see Setup below).

## Setup

### Prerequisites

- Python 3.12
- Git
- NLTK data (`punkt`, `punkt_tab`) — downloaded automatically on first test run
- For C2 induction and extended metrics: `sentence-transformers`, `scikit-learn`
- For GPU inference: Kaggle account with T4×2 GPU (vLLM requires CUDA; not needed for local evaluation)

### Installation

```bash
# Clone this repository
git clone https://github.com/felipemoraissoares/does-ontology-matter.git
cd does-ontology-matter

# Clone Text2KGBench at the sibling path expected by the pipeline
git clone https://github.com/cenguix/Text2KGBench.git ../Text2KGBench
cd ../Text2KGBench
git checkout 50a3d255371b8817cdff70fd88459ac82b339cfe
cd ../does-ontology-matter

# Local evaluation/testing and exploration tooling (no GPU inference):
pip install pytest nltk sentence-transformers scikit-learn numpy scipy pandas statsmodels
pip install -r requirements-app.txt   # Neo4j + Streamlit (Fases 4–5)

# On Kaggle, the launcher installs the pinned GPU stack from requirements-kaggle.txt (D-35)
# (or from the attached wheelhouse — see docs/kaggle_wheelhouse.md).
```

### Data

The benchmark data is in `../Text2KGBench/data/`. No additional downloads are needed beyond
the repository clone above. The benchmark is pinned to commit `50a3d255` for reproducibility.

## Running Locally

### Tests

The full test suite runs without GPU or vLLM. All inference calls go through `ProvedorMock`.

```bash
# Run all 1082 tests
python -m pytest

# Run only prompt-assembly tests
python -m pytest tests/test_pipeline.py -v

# Run C2 smoke tests (requires ../Text2KGBench)
python -m pytest tests/test_c2_smoke_v3.py -v

# Run with coverage
python -m pytest --cov=src/t2kg --cov-report=term-missing
```

Expected output: 1082 passed. Tests do not require GPU, network, or vLLM.

### Evaluation

After obtaining result JSOLs (either from Kaggle push or local run), evaluate with:

```bash
# Standard evaluation
python scripts/avaliar_resultados.py \
    --manifesto configs/pipeline/manifesto_piloto_v3.json \
    --avaliacao_dir resultados/avaliacao_piloto_v3 \
    --csv resultados/avaliacao_piloto_v3/tabela_piloto_v3.csv

# With extended metrics (semantic redundancy H1, projection)
python scripts/avaliar_resultados.py \
    --manifesto configs/pipeline/manifesto_piloto_v3.json \
    --avaliacao_dir resultados/avaliacao_piloto_v3 \
    --csv resultados/avaliacao_piloto_v3/tabela_piloto_v3.csv \
    --extended
```

The script auto-detects partial runs (`frases_max < GT total`) and switches to
`modo_amostra=True` (denominator = sentences evaluated, not total GT sentences).

### Local Inference (optional, requires vLLM + GPU)

```bash
# Start vLLM server
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --port 8000 --dtype half --tensor-parallel-size 1

# Run experiments using the manifesto
python -m t2kg.run \
    --manifesto configs/pipeline/manifesto_piloto_v3.json \
    --bench_dir ../Text2KGBench \
    --skip_existentes \
    --verbose
```

## Running on Kaggle

The Kaggle notebook `kaggle/lancador.ipynb` has a single switch at the top — you only edit
the configuration cell:

```python
MODO = "inducao_c2"   # or "experimento"
MANIFESTO = "configs/pipeline/manifesto_7b_completo.json"   # used in "experimento" mode
MODELO = "Qwen/Qwen2.5-7B-Instruct-AWQ"
```

- **`MODO = "inducao_c2"`** (once per model) — the model induces its own C2 ontology for the
  29 ontologies and generates the D-24 definitions, into `configs/c2/v_full/<slug>/`. Pushes
  the induced ontologies and raw extractions (verified).
- **`MODO = "experimento"`** — runs the chosen `MANIFESTO` with a **guard check** of the C2
  (blocks if the model's induced ontology is missing) and **push per experiment**
  (`--auto_push`, so a dropped session loses nothing and resumes).

Both modes are dispatched by `scripts/rodar_kaggle.py`; no other cell needs manual editing.
Run order and exact commands are in [`docs/experimento_completo_kaggle.md`](docs/experimento_completo_kaggle.md).

### Configuration

1. **Settings → Accelerator → GPU T4 x2** (vLLM requires two GPUs for the AWQ model)
2. **Settings → Internet → On** (for GitHub push and model download)
3. **Settings → Secrets** — `GITHUB_TOKEN` (repo write scope) and `HF_TOKEN`, both attached.
4. (Optional) Attach the **wheelhouse** dataset for fast offline install — see
   [`docs/kaggle_wheelhouse.md`](docs/kaggle_wheelhouse.md). The install cell detects it
   automatically; otherwise it falls back to the pinned `requirements-kaggle.txt` (D-35).

### Pilot results (already committed)

The piloto-v3 (28 experiments, 4 ontologies) is **closed** (tag `piloto-v3-fechado`). Its
raw results, per-ontology metrics and `tabela_piloto_v3.csv` are in `resultados/piloto_v3/`
and `resultados/avaliacao_piloto_v3/`. `git pull` to download, or browse on GitHub.

## Full Experiment

The full experiment covers 29 ontologies × 7 conditions. Generate the manifest with:

```bash
python scripts/gerar_manifesto.py \
    --modelos Qwen/Qwen2.5-7B-Instruct-AWQ \
    --saida configs/pipeline/manifesto_7b_completo.json \
    [--dividir_por_sessao --horas_max 10 --fator 1.0]
```

This writes `manifesto_7b_completo.json` (**203 experiments**, ~8.0 h for the 7B at
~5300 sentences/h, fitting one 12 h Kaggle session) with unique ids and output paths per
model (`resultados/full/<slug>/`). C2 entries carry `c2_subdir: v_full/<slug>` so each model
uses its own induced ontology (D-33). `--dividir_por_sessao` splits slower models into
`..._parteN.json` chunks of ≤ `--horas_max`.

After inducing C2 on Kaggle, recompute the thresholds over the 29 ontologies (CPU):

```bash
python scripts/recalibrar_c2_29.py --slug qwen2.5-7b
```

It prints the clustering threshold (D-21: max breakpoint + 0.05) and the projection
threshold (D-30: lowest threshold with aggregate precision ≥ 0.80). It only computes; the
decision is registered by the author.

## Graph Exploration (Neo4j, Fase 4)

Load a run into Neo4j for visual exploration. Credentials come **only** from `.env`
(`NEO4J_URI`, `NEO4J_USER`/`NEO4J_USERNAME`, `NEO4J_PASSWORD` — never committed):

```bash
pip install -r requirements-app.txt
python -m t2kg.grafo.carregar_neo4j --run pv3_1movie_c1            # load a run
python -m t2kg.grafo.carregar_neo4j --run pv3_1movie_c1 --apagar_run  # reload
```

Graph model: `(:Frase)-[:MENCIONA]->(:Entidade)` and
`(:Entidade)-[:REL {relacao, condicao, modelo, run, status, frase}]->(:Entidade)`, where
`status` is `acerto`/`sobra` per the evaluator. Loading is **per run** (AuraDB Free limits).
An example Cypher query ("C0 edges absent in C1 for the same sentence") is in
[`src/t2kg/grafo/README.md`](src/t2kg/grafo/README.md).

## Exploration App (Streamlit, Fases 4–5)

A multipage Streamlit app to understand the whole project — what each phase did and, for any
sentence, what went to the model and what came back. Reads only repository files; Neo4j not
required. New result files (e.g., the ongoing `resultados/full/qwen2.5-7b/`) appear
automatically without code changes.

```bash
python -m pip install --user -r requirements-app.txt
python -m streamlit run app/Inicio.py
```

Pages (`st.navigation`): **Início** (key numbers), **Fases do projeto** (timeline with
clickable D-xx decisions), **Explorador de frase** (prompt, raw response and triples
classified as acerto/fora do gabarito/erro/erro de formato/falta, per condition, with an
interactive graph and a "Comparar" view of the 7 conditions), **Ontologias** (reference vs
induced-C2 vs top-3 vs C5 partners), **Grafo agregado**, **Resultados agregados** (Plotly
charts/heatmap with the Vicuna-13B baseline), and **Chat com o grafo** (skeleton/GraphRAG).

The data layer lives in `app/dados/` (no Streamlit, tested in `tests/test_app_dados.py`) and
reuses `src/t2kg/` for parsing, evaluation, prompts and ontologies. Screenshots of every page
are in [`docs/app_capturas/`](docs/app_capturas/).

## Experiment Configuration

Each experiment is a JSON object in a manifesto file. The key fields are:

```json
{
  "id": "pv3_1movie_c1",
  "dataset": "tekgen",
  "split": "test",
  "ontologia_id": "1_movie",
  "condicao": "C1",
  "modelo": "Qwen/Qwen2.5-7B-Instruct-AWQ",
  "temperatura": 0,
  "seed": 42,
  "frases_max": 100,
  "frases_seed": 42,
  "workers": 1,
  "saida": "resultados/piloto_v3/pv3_1movie_c1.jsonl",
  "git_hash": "cfa7e1a",
  "bench_commit": "50a3d255371b8817cdff70fd88459ac82b339cfe"
}
```

Key design decisions encoded in the manifesto:
- `workers: 1` — fixed for reproducibility (D-23): parallelism at vLLM batch size 1 showed
  no throughput gain (< 3%) and introduced stochastic variation in 7% of responses
- `temperatura: 0` — greedy decoding for deterministic output
- `frases_seed: 42` — reproducible subsampling; same 100 sentences across all conditions
  for the same ontology, enabling paired analysis
- `bench_commit` — pinned benchmark commit prevents silent data drift

The full-experiment manifesto (203 experiments) is in
`configs/pipeline/manifesto_7b_completo.json` (see the Full Experiment section above).

## Results

Results will be stored in `resultados/avaliacao_piloto_v3/tabela_piloto_v3.csv` after the
Kaggle run. The CSV contains the following columns per (dataset, condition) combination:

```
dataset, condicao, modo_amostra,
avg_precision, avg_recall, avg_f1,
avg_onto_conf, avg_sub_halluc, avg_rel_halluc, avg_obj_halluc,
avg_fmt_error, avg_parse_error,
ext_sem_filtro_f1, ext_projetado_f1, ext_fracao_projetada
```

Preliminary results from piloto-v2 (C0/C1/C1_oneshot, 200 sentences, TekGen only):

| Condition | F1 | Conformity | Semantic Redundancy (H1) |
|---|---|---|---|
| C0 | — | — | 0.32 |
| C1 | — | — | 0.12 |
| C1_oneshot_original | ~0.30 | ~0.95 | 0.08 |

Semantic redundancy ratio C0/C1 ≈ 2.6–4.2× across ontologies, consistent with H1.
Full piloto-v3 results pending.

## Decisions Log

All methodological decisions (D-01–D-35) are documented in [`docs/DECISOES.md`](docs/DECISOES.md)
and visualized at [`docs/decisions.html`](docs/decisions.html).

| Decision | Summary |
|---|---|
| D-01 | C3 uses top-3 most frequent training relations with domain/range |
| D-02 | C5 has two perturbation levels: neighbour (highest SimE) and incongruent (lowest SimE) |
| D-03 | All prompts end with `\nTest Output:` for consistency across conditions |
| D-04 | Evaluation mode: `modo_amostra=True` when output sentences < GT total |
| D-05 | Entity instruction added to C0–C5 to prevent class-name substitution |
| D-06 | C3 relation frequencies from training data only, never from test GT |
| D-07 | `avg_fmt_error` metric: fraction of triples with ontology class name as entity prefix |
| D-08 | Token difference (dTok) between partner and target ontology as a C5 covariate |
| D-09 | SimE as continuous variable in C5 regression analysis |
| D-10 | Sensitivity analysis: C5_incongruente with second-least-similar partner |
| D-11 | Terminology: "schema similarity" for SimE-based pair selection |
| D-12 | All experimental conditions are zero-shot except C1_oneshot_original |
| D-13 | C0 evaluated in two modes: raw and projected onto reference ontology |
| D-14 | Conformity always computed against target ontology, even for C5 conditions |
| D-15 | Parser fix: trailing comma after closing parenthesis |
| D-16 | C2 induced ontology protocol: 50 training sentences, seed 42, free extraction + clustering |
| D-17 | Bugs from first Kaggle pilot: GT leakage in C2 prompt, wrong split, false git push |
| D-18 | Unified format instruction; `avg_parse_error` metric added |
| D-19 | Depth-aware parser; parallel workers; D-19 format mini-test across 4 instruction variants |
| D-20 | Pre-registered selection rule for format instruction; Variant D wins (score=0.1505) |
| D-21 | C2 clustering: threshold=0.70, freq_min=3 distinct sentences (D-21 fix from initial freq_min=5) |
| D-22 | Label normalisation before embedding: strip `has_`, `is_`, `_by`, `_of` prefixes/suffixes |
| D-23 | `workers=1` fixed in all manifests for 100% reproducibility |
| D-24 | Embedding-based Definition Canonicalization (EDC): embed `"label: LLM-definition"` |
| D-25 | C2 context uses `Ontology Relations:` label, no concepts or domain/range signatures |
| D-26 | `modo_amostra` detected per ontology, not per dataset |
| D-27 | `fmt_error` detects capitalized class labels (DBpedia: "Film", "Person") |
| D-28 | Conformity/hallucination metrics excluding empty responses (`_nv`, `pct_vazias`) |
| D-29 | D-24 embeddings (label+definition) used in relation projection (C0/C2/C3) |
| D-30 | Projection threshold recalibrated: 0.63 (label+def) / 0.50 (label-only) |
| D-31 | Piloto-v3 — validation before the full phase |
| D-32 | Variant-D placeholders (`relation_name`) treated as format errors |
| D-33 | Per-model induced C2 ontology under `configs/c2/v_full/<slug>/` |
| D-34 | Single normalized definition key; projection fails if a definition is missing |
| D-35 | Pinned Kaggle dependency versions (`requirements-kaggle.txt`) |

## Citation / Acknowledgements

This project extends the Text2KGBench benchmark:

> Mihindukulasooriya, N., Tiwari, S., Enguix, C. F., & Lata, K. (2023).
> Text2KGBench: A Benchmark for Ontology-Driven Knowledge Graph Generation from Text.
> *International Semantic Web Conference (ISWC 2023)*.

The evaluation code in `src/t2kg/avaliacao/original.py` is a faithful adaptation of the
original `run_eval.py` (Apache 2.0 license). Deviations from the original are limited to:
encoding fixes (UTF-8 explicit), addition of `modo_amostra` parameter, and new metrics
(`avg_fmt_error`, `avg_parse_error`). The core P/R/F1 and conformity calculations are
unchanged.

Embedding-based Definition Canonicalization (D-24) is inspired by:

> Zhang, Y., & Soh, H. (2024). EDC: Towards Accurate and Hallucination-Free Information
> Extraction with Error Detection and Correction. *arXiv:2401.xxxxx*.

Developed as part of a Master's dissertation at Instituto Tecnológico de Aeronáutica (ITA),
São José dos Campos, Brazil.
