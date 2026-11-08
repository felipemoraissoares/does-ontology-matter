# Kaggle — Piloto Text2KG (Qwen2.5-7B-Instruct-AWQ)

## Pré-requisitos

### 1. Configurar Accelerator
Settings → Accelerator → **GPU T4 x2**

### 2. Habilitar Internet
Settings → Internet → **On**

### 3. Adicionar Secrets
Settings → Secrets (cadeado lateral):

| Nome | Valor |
|---|---|
| `GITHUB_TOKEN` | Token GitHub com `repo` (read + push) no repositório privado |
| `HF_TOKEN` | Token HuggingFace com acesso a `Qwen/Qwen2.5-7B-Instruct-AWQ` |

> Para criar o HF_TOKEN: huggingface.co → Settings → Access Tokens → New token (read).
> O modelo AWQ é público; qualquer token read funciona.

---

## Passo a passo

### Célula 1 — Variáveis de ambiente
Lê os secrets e define caminhos. Ajuste `REPO_URL` se o nome do repositório for diferente.

### Célula 2 — Clonar repositórios
Clona `llm-kg-ontology` (privado, usa GITHUB_TOKEN) e `Text2KGBench` (público).
Se já existirem, apenas atualiza com `git pull`.

### Célula 3 — Instalar dependências
Instala vllm, openai, sentence-transformers, nltk, scipy, pandas, statsmodels.
Baixa os dados do NLTK (punkt). Leva ~3 min.

### Célula 4 — Subir vLLM em segundo plano
```
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --port 8000 --dtype half \
    --tensor-parallel-size 2 --max-model-len 4096 --gpu-memory-utilization 0.90
```
- `--dtype half`: obrigatório para T4 (fp16; bf16 não suportado)
- `--tensor-parallel-size 2`: distribui entre as 2 GPUs T4
- Saída redirecionada para `/kaggle/working/vllm.log`

### Célula 5 — Aguardar vLLM pronto
Polling em `/health` a cada 10s. Timeout de 5 min. Se travar, verifique o log na Célula 11.

### Célula 6 — Gerar manifesto com caminhos do Kaggle
Regenera o manifesto do piloto com os caminhos corretos para `/kaggle/working/`.
Lê `configs/pipeline/manifesto_piloto_kaggle.json` e substitui os caminhos `saida`.

### Célula 7 — Executar piloto
Roda 6 experimentos: 2 ontologias TekGen (1_movie, 2_music) × 3 condições (C0, C1, C1_oneshot_original).
- 200 frases por experimento (frases_max=200, frases_seed=42)
- `--skip_existentes`: pula experimentos com arquivo de saída já existente (retomada)

**Tempo estimado:** ~2h para os 6 experimentos.

### Célula 8 — Indução C2 (calibração D-16)
Extrai triplas livres em 50 frases de treino para 1_movie e 2_music:
- Modo C0 sobre o split de treino (sem ontologia, sem triplas anotadas)
- Agrupa relações extraídas com threshold=0.80 (provisório)
- Roda `--calibrar` para gerar curva de agrupamento por threshold
- Salva em `configs/c2/tekgen_{onto}_induced.json`

Esses dados alimentam a calibração dos limiares de agrupamento e projeção (D-16).

### Célula 9 — Avaliar resultados
Chama `scripts/avaliar_resultados.py` e exibe a tabela CSV com P/R/F1/conformidade/fmt_error
para cada condição e ontologia.

### Célula 10 — Commit e push (opcional)
Faz commit dos resultados (`resultados/piloto/`, `configs/c2/`) e push para o repositório.
Execute só após verificar que os resultados estão corretos.

---

## Retomada de sessão interrompida

1. Execute as células **1 a 5** (setup + vLLM)
2. Continue da célula onde parou
3. As células 7 e 8 usam `--skip_existentes` — experimentos concluídos são pulados

---

## Diagnóstico

```python
# Ver log do vLLM
run(f"tail -50 {VLLM_LOG}")

# Verificar GPU
run("nvidia-smi")

# Testar endpoint
run(f"curl http://localhost:8000/v1/models")
```

---

## Resultado esperado

Após a execução completa:
- `resultados/piloto/*.jsonl` — saída por experimento (1 linha por frase)
- `resultados/piloto/tabela_piloto.csv` — métricas agregadas
- `configs/c2/tekgen_1_movie_induced.json` — ontologia induzida 1_movie
- `configs/c2/tekgen_2_music_induced.json` — ontologia induzida 2_music
- `configs/c2/tekgen_*_frases_seed42.json` — IDs das frases de treino sorteadas
