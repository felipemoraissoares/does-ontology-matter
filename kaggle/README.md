# Kaggle — Piloto Text2KG (Qwen2.5-7B-Instruct-AWQ)

## Configuração antes de salvar

### 1. Accelerator
Settings (engrenagem lateral) → Accelerator → **GPU T4 x2**

### 2. Internet
Settings → Internet → **On**

### 3. Secrets
Settings → Secrets (cadeado) → Add a new secret:

| Nome | Valor | Ação obrigatória |
|---|---|---|
| `GITHUB_TOKEN` | Token GitHub fine-grained com **Contents: Read and write** restrito ao repositório `does-ontology-matter` | Clique em **Attach to notebook** |
| `HF_TOKEN` | Token HuggingFace (read) | Clique em **Attach to notebook** |

> **GITHUB_TOKEN fine-grained:** github.com → Settings → Developer settings → Personal access tokens → Fine-grained tokens → New token → Repository access: Only selected → `does-ontology-matter` → Permissions → Repository permissions → Contents: **Read and write** → Generate token.

> **HF_TOKEN:** huggingface.co → Settings → Access Tokens → New token (read). O modelo AWQ é público.

### 4. Para anexar resultados anteriores (retomada)
Add Data (ícone `+` no painel esquerdo) → Your Work → nome do notebook → selecione a versão anterior → **Add**.
O input ficará disponível em `/kaggle/input/{slug}/` e a Célula 4 copia automaticamente os resultados para pular experimentos já concluídos.

---

## Como rodar

**Sempre use Save Version → Save & Run All** (não "Run All Cells"):

1. Faça as configurações acima (GPU, Internet, Secrets)
2. No notebook: **Save Version** (canto superior direito) → **Save & Run All** → Save
3. O notebook roda em fila e os resultados ficam disponíveis na aba **Output** da versão salva

> "Run All Cells" roda na sessão interativa atual, que é descartada quando a janela fecha.
> "Save & Run All" cria uma versão persistida e roda de forma confiável em background.

---

## Onde encontrar os resultados

Após a execução completa:
- Abra a versão no painel **Versions** (à direita)
- Clique na versão → aba **Output**
- Os arquivos listados incluem tudo em `/kaggle/working/`
- Arquivos também são commitados automaticamente ao repositório (`resultados/piloto/`, `configs/c2/`)

---

## Passo a passo das células

### Célula 1 — Configuração
Lê os secrets (GITHUB_TOKEN, HF_TOKEN) e define caminhos. `REPO_DIR = /kaggle/working/does-ontology-matter`.

### Célula 2 — Helpers
Define funções `run()` e `git()` usadas pelas demais células.

### Célula 3 — Clonar repositórios
Clona `does-ontology-matter` (privado, usa GITHUB_TOKEN) e `Text2KGBench` (público).
Se já existirem, apenas faz `git pull`.

### Célula 4 — Restaurar resultados anteriores
Varre `/kaggle/input/` em busca de `.jsonl` e `calibracao_*.json` de versões anteriores.
Copia para `RESULT_DIR` e `C2_DIR` para que `--skip_existentes` os pule na Célula 9.

### Célula 5 — Instalar dependências
`pip install vllm openai sentence-transformers scikit-learn nltk scipy numpy pandas statsmodels`
Leva ~3 min.

### Célula 6 — Subir vLLM
```
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --port 8000 --dtype half \
    --tensor-parallel-size 2 --max-model-len 4096 \
    --gpu-memory-utilization 0.90 > vllm.log 2>&1 &
```
- `--dtype half` obrigatório para T4 (fp16; bf16 não suportado)
- `--tensor-parallel-size 2` distribui entre as 2 GPUs T4
- Saída em `/kaggle/working/vllm.log`

### Célula 7 — Aguardar vLLM pronto
Polling automático em `/health` a cada 10s, timeout de 360s. Nenhuma intervenção necessária.
Se falhar: a célula lê as últimas linhas do log e lança `RuntimeError`.

### Célula 8 — Gerar manifesto
Regenera `manifesto_piloto_kaggle.json` com os caminhos do Kaggle. Exibe status de cada experimento (já existe / pendente).

### Célula 9 — Executar piloto
6 experimentos: 2 TekGen (1_movie, 2_music) × 3 condições (C0, C1, C1_oneshot_original).
- 200 frases/experimento, frases_seed=42
- `--skip_existentes`: pula experimentos com saída já presente
- `--auto_push`: commit + push ao GitHub após **cada experimento** (~a cada 20-30 min)

### Célula 10 — Indução C2 + calibração
Para cada ontologia (1_movie, 2_music):
- Extrai triplas livres (C0) de 50 frases de treino com seed=42
- Agrupa relações (threshold=0.80 provisório)
- Calcula curvas de calibração (thresholds 0.70–0.90, agrupamento + projeção)
- Salva `configs/c2/tekgen_{onto}_induced.json` e `configs/c2/calibracao_tekgen_{onto}.json`
- Push ao GitHub no final

### Célula 11 — Avaliar + push final
Avalia todos os experimentos, gera `resultados/piloto/tabela_piloto.csv`, exibe tabela, push final.

### Célula 12 — Diagnóstico
`tail -30 vllm.log` para diagnóstico de erros.

---

## Retomada de sessão interrompida

1. Na versão incompleta: aba **Output** → baixe os `.jsonl` de resultados parciais
2. Crie uma nova versão do notebook
3. **Add Data → Your Work → versão anterior → Add** (antes de salvar)
4. **Save & Run All** — a Célula 4 restaura os resultados e as Células 9-10 pulam o que já foi feito

---

## Outputs esperados

| Arquivo | Conteúdo |
|---|---|
| `resultados/piloto/piloto_tkg_1_movie_c0_*.jsonl` | Saída do experimento C0 / 1_movie |
| `resultados/piloto/piloto_tkg_2_music_c1_*.jsonl` | Saída do experimento C1 / 2_music |
| `resultados/piloto/tabela_piloto.csv` | Métricas agregadas por condição |
| `configs/c2/tekgen_1_movie_induced.json` | Ontologia induzida 1_movie |
| `configs/c2/calibracao_tekgen_1_movie.json` | Curvas de calibração agrupamento + projeção |
| `configs/c2/tekgen_1_movie_frases_seed42.json` | IDs das 50 frases de treino sorteadas |

---

## Diagnóstico rápido

```python
# Verificar GPU disponível
run("nvidia-smi")

# Testar endpoint vLLM
run("curl -s http://localhost:8000/v1/models | python3 -m json.tool")

# Ver log do vLLM
run("tail -50 /kaggle/working/vllm.log")
```
