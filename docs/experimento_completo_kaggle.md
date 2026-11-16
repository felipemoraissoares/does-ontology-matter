# Experimento completo no Kaggle — passo a passo (Qwen 7B)

Dois tipos de sessão, nesta ordem:

1. **Sessão de indução C2** (uma vez por modelo) — o modelo induz a própria ontologia C2
   das 29 ontologias e gera as definições D-24.
2. **Sessões do experimento** (uma ou mais) — roda as 203 condições; a C2 só roda se a
   indução da etapa 1 existir (guard check).

## Pré-requisitos (uma vez)

- **Reimportar o notebook** `kaggle/lancador.ipynb` no Kaggle — ele mudou (célula de
  instalação usa o wheelhouse; célula de execução faz push por experimento;
  `FORCE_RERUN=False`).
- (Opcional, recomendado) Anexar o Dataset **wheelhouse** para instalar rápido
  (ver `docs/kaggle_wheelhouse.md`). Sem ele, a instalação usa `requirements-kaggle.txt`.

---

## Sessão 1 — Indução da C2 (29 ontologias, Qwen 7B)

Rode as células 1–7 normalmente (config, clone, dependências, vLLM pronto). Depois, **no
lugar das células 8–13**, rode uma única célula com o comando abaixo (o `induzir_c2_lote.py`
faz extração livre + definições D-24 dos rótulos induzidos **e** das relações de referência,
e a re-indução EDC — tudo das 29 ontologias, saída em `configs/c2/v_full/qwen2.5-7b/`):

```python
import datetime
PV3 = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
PYPATH = {**os.environ, 'PYTHONPATH': f'{REPO_DIR}/src'}

run(
    f'python {REPO_DIR}/scripts/induzir_c2_lote.py '
    f'--manifesto {REPO_DIR}/configs/pipeline/manifesto_c2_inducao_29.json '
    f'--modelo_llm {MODELO} '
    f'--url http://localhost:{PORT}/v1 '
    f'--run_id {PV3}',
    cwd=REPO_DIR, env=PYPATH,
)

# Commit + push das ontologias induzidas e definições (configs/c2/v_full + resultados/c2_inducao/v_full)
git('add configs/c2/v_full/ resultados/c2_inducao/v_full/')
if git('status --porcelain configs/c2/v_full/ resultados/c2_inducao/v_full/').stdout.strip():
    git('commit -m "auto: inducao C2 v_full qwen2.5-7b (29 ontologias)"')
    git('push')
    print('Push indução C2 OK.')
else:
    print('Indução C2: nada novo para commitar.')
```

Isso produz, para cada uma das 29 ontologias:
- `configs/c2/v_full/qwen2.5-7b/<dataset>_<onto>_induced.json` (ontologia induzida)
- `configs/c2/v_full/qwen2.5-7b/definicoes_<onto>.json` (definições D-24, rótulos livres + referência)
- `resultados/c2_inducao/v_full/qwen2.5-7b/raw_<dataset>_<onto>_treino.jsonl` (extrações cruas)

**Tempo estimado:** ~50 frases de treino × 29 ontologias = ~1450 extrações livres, mais as
definições (poucas centenas de rótulos). A ~5300 fr/h, a extração é ~15–20 min; as
definições somam mais alguns minutos. Com o vLLM já no ar, estime **~30–45 min de GPU**.

### Depois da Sessão 1 (no seu PC)

Recalibre os limiares sobre as 29 ontologias (só calcula e mostra; a decisão é sua):

```
python scripts/recalibrar_c2_29.py --slug qwen2.5-7b
```

Ele imprime o threshold de agrupamento (D-21: max breakpoint + 0,05) e o de projeção
(D-30: menor limiar com precisão ≥ 0,80) agregados nas 29 ontologias.

---

## Sessão 2+ — Experimento completo (203 condições)

Rode as células 1–7. Na **célula 8**, aponte o manifesto do experimento e **não** achate os
caminhos de saída (eles já vêm como `resultados/full/qwen2.5-7b/...`):

```python
MANIFESTO = f'{REPO_DIR}/configs/pipeline/manifesto_7b_completo.json'
with open(MANIFESTO, encoding='utf-8') as f:
    exps = json.load(f)
# saída relativa -> absoluta sob o repo (preserva resultados/full/<slug>/)
for exp in exps:
    exp['saida'] = f"{REPO_DIR}/{exp['saida']}"
manifesto_kaggle_path = f'{WORK_DIR}/manifesto_kaggle.json'
with open(manifesto_kaggle_path, 'w', encoding='utf-8') as f:
    json.dump(exps, f, ensure_ascii=False, indent=2)
print(f'{len(exps)} experimentos')
```

Antes de rodar, uma célula de **guard check** (garante que a C2 das 29 ontologias foi
induzida para o modelo — senão, para com erro):

```python
run(
    f'python {REPO_DIR}/scripts/guard_check_c2.py '
    f'--manifesto {manifesto_kaggle_path}',
    cwd=REPO_DIR, env=PYPATH,
)
```

A **célula de execução** (equivalente à 11) já faz push verificado por experimento
(`--auto_push --push_repo_dir`), retomando de onde parou (`--skip_existentes` só pula o que
está completo). Confirme que ela aponta para `manifesto_kaggle_path`:

```python
run(
    f'python -m t2kg.run '
    f'--manifesto {manifesto_kaggle_path} '
    f'--bench_dir {BENCH_DIR} '
    f'--skip_existentes '
    f'--auto_push '
    f'--push_repo_dir {REPO_DIR} '
    f'--verbose',
    cwd=REPO_DIR, env=PYPATH,
)
```

O experimento completo do 7B é ~**8 h** de inferência (42.532 frases @ ~5300 fr/h, workers=1),
cabendo numa única sessão de 12 h. Se a sessão cair, basta rodar de novo: o checkpoint por
frase e o push por experimento garantem que nada se perde e a execução retoma.

### Dividir em sessões (modelos mais lentos)

Para modelos mais lentos que não cabem em 12 h, gere partes menores:

```
python scripts/gerar_manifesto.py --modelos <modelo> \
    --saida configs/pipeline/manifesto_<slug>_completo.json \
    --dividir_por_sessao --horas_max 10 --fator <fator_de_lentidao>
```

Cada `..._parteN.json` roda numa sessão separada (aponte `MANIFESTO` para a parte).
