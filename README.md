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
│   │   ├── ontologias.py      # Loads and formats ontology context per condition
│   │   ├── prompts.py         # Assembles the full prompt string per condition
│   │   └── executar.py        # Runs extraction with checkpoint/resume support
│   ├── avaliacao/
│   │   ├── original.py        # Exact mirror of Text2KGBench run_eval.py
│   │   └── extended.py        # Extended metrics: semantic redundancy, projection
│   ├── parser.py              # Parses LLM responses into (sub, rel, obj) triples
│   └── run.py                 # CLI entry point: reads manifesto, calls executar()
│
├── scripts/
│   ├── induzir_c2.py          # C2 ontology induction: free extraction + clustering
│   ├── gerar_definicoes_relacoes.py  # Generates LLM definitions for EDC (D-24)
│   ├── calibrar_c2_com_definicoes.py # Calibration table: label vs label+definition
│   ├── avaliar_resultados.py  # Evaluates results using a manifesto JSON
│   ├── calcular_pares_c5.py   # Computes schema-similarity pairs for C5
│   └── analisar_threshold_c2.py  # Threshold sweep for C2 clustering (D-21)
│
├── configs/
│   ├── pipeline/
│   │   ├── manifesto_piloto_v3.json   # 28-experiment pilot manifest
│   │   ├── manifesto_7b_full.json     # Full-phase manifest (174 experiments)
│   │   ├── reinducao_c2_kaggle.json   # C2 induction tasks for Kaggle run
│   │   └── pares_c5.json             # Pre-registered C5 ontology pairs
│   └── c2/
│       ├── *_induced.json             # Induced ontologies (generated on Kaggle)
│       └── definicoes_*.json          # LLM-generated relation definitions (D-24)
│
├── kaggle/
│   └── lancador.ipynb         # 14-cell Kaggle notebook: induction → experiments → evaluation
│
├── resultados/
│   ├── piloto_v3/             # Raw JSONL results from piloto-v3 (auto-pushed from Kaggle)
│   ├── avaliacao_piloto_v3/   # Evaluation metrics and CSV summary
│   └── c2_inducao/            # Raw extraction JSOLs from C2 training phase
│
├── tests/                     # pytest suite (612 tests, all pass locally)
│   ├── test_pipeline.py
│   ├── test_c2_smoke_v3.py
│   ├── test_c2_inducao.py
│   ├── test_avaliacao_resultados.py
│   └── test_parser.py
│
└── docs/
    ├── DECISOES.md            # Authoritative log of all 25 methodological decisions
    ├── decisions.html         # Visual decision log with Mermaid diagrams
    └── codebase.html          # Code architecture and module documentation
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

# Install Python dependencies
pip install -r requirements.txt
# Or, for local evaluation/testing only (no GPU inference):
pip install pytest nltk sentence-transformers scikit-learn numpy scipy pandas statsmodels
```

### Data

The benchmark data is in `../Text2KGBench/data/`. No additional downloads are needed beyond
the repository clone above. The benchmark is pinned to commit `50a3d255` for reproducibility.

## Running Locally

### Tests

The full test suite runs without GPU or vLLM. All inference calls go through `ProvedorMock`.

```bash
# Run all 612 tests
python -m pytest

# Run only prompt-assembly tests
python -m pytest tests/test_pipeline.py -v

# Run C2 smoke tests (requires ../Text2KGBench)
python -m pytest tests/test_c2_smoke_v3.py -v

# Run with coverage
python -m pytest --cov=src/t2kg --cov-report=term-missing
```

Expected output: 612 passed. Tests do not require GPU, network, or vLLM.

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

The Kaggle notebook `kaggle/lancador.ipynb` orchestrates the full pipeline in 14 cells,
pushing results to this GitHub repository at the end.

### Configuration

Before running the notebook, configure in Kaggle:

1. **Settings → Accelerator → GPU T4 x2** (vLLM requires two GPUs for the AWQ model)
2. **Settings → Internet → On** (for GitHub push and model download)
3. **Settings → Secrets** — add two secrets, both with "Attach to notebook" checked:
   - `GITHUB_TOKEN`: a GitHub Personal Access Token with `repo` write scope
   - `HF_TOKEN`: a Hugging Face token for model access (if the model requires authentication)

### Execution Order

The notebook enforces a strict ordering to prevent C2 experiments from using stale or
pre-session induced ontologies:

| Cell | Purpose |
|---|---|
| 1–5 | Setup: secrets, helpers, clone repos, restore previous session outputs, install deps |
| 6–7 | Start vLLM server and wait for health check |
| 8 | Load experiment manifest, adjust output paths to Kaggle working dir |
| **9** | **C2 induction**: (a) free extraction on 50 training sentences → (b) LLM definitions (D-24) → (c) re-induction with label+definition embeddings → (d) calibration table |
| **10** | **Guard check**: raises `RuntimeError` if any C2 experiment lacks a correctly-session-tagged ontology (`run_id == PV3_RUN_ID`) |
| **11** | **28 experiments** (C0, C1, C1_oneshot_original, C2, C3, C5_vizinho, C5_incongruente) |
| **12** | Push raw JSONL results to GitHub (`resultados/piloto_v3/`) |
| **13** | Evaluate results, generate CSV, push evaluation to GitHub (`resultados/avaliacao_piloto_v3/`) |
| 14 | vLLM log tail for diagnostics |

The `PV3_RUN_ID` is an ISO timestamp generated at the start of Cell 9 and stored as
`metadata.run_id` in each induced ontology JSON. The guard check in Cell 10 compares this
value to prevent experiments from silently using an ontology from a previous session — a
real failure mode that occurred in the first Kaggle pilot (decision D-17).

### Outputs

After a successful run, the following are committed and pushed to this repository:

| Path | Contents | Pushed by |
|---|---|---|
| `configs/c2/*_induced.json` | Induced ontologies with metadata | Cell 9 |
| `configs/c2/definicoes_*.json` | LLM-generated relation definitions (D-24) | Cell 9 |
| `resultados/c2_inducao/raw_*.jsonl` | Raw C0 extractions from training sentences | Cell 9 |
| `resultados/piloto_v3/pv3_*.jsonl` | Per-sentence extraction results (28 files) | Cell 12 |
| `resultados/avaliacao_piloto_v3/*.jsonl` | Per-ontology evaluation metrics | Cell 13 |
| `resultados/avaliacao_piloto_v3/tabela_piloto_v3.csv` | Summary table of all metrics | Cell 13 |

To download results after a run, `git pull` in this repository or browse the files directly
on GitHub.

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

The full 174-experiment phase manifesto is in `configs/pipeline/manifesto_7b_full.json`.

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

All 25 methodological decisions are documented in [`docs/DECISOES.md`](docs/DECISOES.md)
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
