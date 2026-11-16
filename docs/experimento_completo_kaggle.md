# Experimento completo no Kaggle — passo a passo (Qwen 7B)

O lançador `kaggle/lancador.ipynb` tem **um único switch** no topo (`MODO`). Você só edita a
célula de configuração; todas as outras rodam iguais em qualquer modo.

```python
MODO = "congelar_ambiente"   # "congelar_ambiente" | "inducao_c2" | "experimento"
MANIFESTO = "configs/pipeline/manifesto_7b_completo.json"   # usado no modo experimento
MODELO = "Qwen/Qwen2.5-7B-Instruct-AWQ"
```

Ordem das sessões: **0) congelar_ambiente → 1) inducao_c2 → 2) experimento.**

## Pré-requisitos (uma vez)

- **Reimportar o notebook** no Kaggle (ele mudou).
- `Settings → Secrets`: `GITHUB_TOKEN` (repo write) e `HF_TOKEN`, ambos anexados.
- `Settings → Environment → **Pin to original environment**` (não "Always use latest") — fixa
  a imagem base do Kaggle entre sessões. O lock trava só o `vllm`; torch/transformers/numpy
  vêm da imagem, então mantê-la fixa garante reprodutibilidade sem travar tudo no pip (D-35).

---

## Sessão 0 — Congelar o ambiente (uma vez, SEM GPU)

As versões dos pacotes dependem do Python/CUDA do Kaggle (não dá para inventar — ex.:
`torch==2.4.0` não existe no Python 3.13). Esta sessão gera o *lock* com as versões reais.

1. `Settings → Accelerator → **None**` (não precisa de GPU) e `Internet → On`.
2. `MODO = "congelar_ambiente"`.
3. Run All.

O lançador instala a lista de `requirements-kaggle.txt` **sem travar**, testa os imports
(`vllm, torch, transformers, sentence_transformers, nltk`), grava
`requirements-kaggle.lock.txt` (versões efetivas + Python + CUDA) e faz **push**. A partir
daí, os outros modos instalam desse lock.

> (Opcional) Para instalar ainda mais rápido, gere o *wheelhouse* a partir do lock e anexe-o
> como Dataset — ver `docs/kaggle_wheelhouse.md`.

---

## Sessão 1 — Indução da C2 (29 ontologias, Qwen 7B)

1. `Settings → Accelerator → **GPU T4 x2**`.
2. `MODO = "inducao_c2"`.
3. Run All.

Gera, para as 29 ontologias: `configs/c2/v_full/qwen2.5-7b/<ds>_<onto>_induced.json`,
`definicoes_<onto>.json` e as extrações cruas em `resultados/c2_inducao/v_full/...`. Faz
push. **Tempo estimado: ~30–45 min de GPU** (com o vLLM já no ar).

### Depois da Sessão 1 (no seu PC)

```
python scripts/recalibrar_c2_29.py --slug qwen2.5-7b
```

Imprime o threshold de agrupamento (D-21) e o de projeção (D-30) nas 29 ontologias — só
calcula; a decisão é sua.

---

## Sessão 2+ — Experimento completo (203 condições)

1. `MODO = "experimento"`, `MANIFESTO = "configs/pipeline/manifesto_7b_completo.json"`.
2. Run All.

O lançador roda o guard check da C2 (bloqueia se faltar a ontologia induzida do modelo) e
executa com **push por experimento** (`--auto_push`) e **retomada** (`--skip_existentes`). Se
a sessão cair, basta rodar de novo: nada se perde. **~8 h** de inferência para o 7B (cabe em
uma sessão de 12 h).

### Dividir em sessões (modelos mais lentos)

```
python scripts/gerar_manifesto.py --modelos <modelo> \
    --saida configs/pipeline/manifesto_<slug>_completo.json \
    --dividir_por_sessao --horas_max 10 --fator <fator_de_lentidao>
```

Cada `..._parteN.json` roda numa sessão (aponte `MANIFESTO` para a parte).
