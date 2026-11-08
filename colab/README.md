# Colab — Piloto Text2KG (modelos até 3B)

## Configuração antes de rodar

### 1. GPU
Runtime → Change runtime type → **T4 GPU**

### 2. Secrets
No painel lateral esquerdo, clique no ícone de **chave** (🔑):

- Clique em **+ Add new secret**
- Nome: `GITHUB_TOKEN`, Value: seu token GitHub fine-grained
  - Ligue o botão **Notebook access**
- Repita para `HF_TOKEN` (token HuggingFace read)
  - Ligue **Notebook access**

> O painel de Secrets fica na barra lateral esquerda, **não** em Settings.
> O interruptor "Notebook access" precisa estar ligado para o notebook conseguir ler o valor.

> **GITHUB_TOKEN fine-grained:** github.com → Settings → Developer settings → Personal access tokens → Fine-grained tokens → Repository access: `does-ontology-matter` → Permissions → Contents: **Read and write**.

### 3. Rodar
Runtime → **Run all**

---

## Passo a passo

| Célula | O que faz |
|---|---|
| 1 | Lê secrets, define caminhos. Ajuste `MODELO` e `N_FRASES` aqui. |
| 2 | Helpers `run()` e `git()` |
| 3 | Clona `does-ontology-matter` e `Text2KGBench` |
| 4 | `pip install vllm openai sentence-transformers ...` (~3 min) |
| 5 | Sobe vLLM em segundo plano |
| 6 | Polling automático `/health` — aguarda vLLM pronto |
| 7 | Gera manifesto com caminhos do Colab e `frases_max=N_FRASES` |
| 8 | Executa piloto com `--skip_existentes --auto_push` |
| 9 | Indução C2 + curvas de calibração para 1_movie (opcional) |
| 10 | Avalia resultados e gera CSV |
| 11 | Push ao GitHub (ou download local via `files.download`) |

---

## Diferenças em relação ao Kaggle

| | Kaggle | Colab |
|---|---|---|
| GPU | T4 × 2 (`--tensor-parallel-size 2`) | T4 × 1 (`--tensor-parallel-size 1`) |
| Modelo | Qwen2.5-7B-Instruct-AWQ | Qwen2.5-3B-Instruct (ou 1.5B) |
| `N_FRASES` | 200 | 100 |
| `--dtype` | `half` (fp16) | `bfloat16` |
| Retomada | Anexar versão anterior como input | Não suportado (sessão efêmera) |
| Resultados | Aba Output da versão; push automático | Download ou push ao fim |

---

## Modelos testados no T4 Colab

| Modelo | VRAM | Tempo/frase | Observação |
|---|---|---|---|
| `Qwen2.5-1.5B-Instruct` | ~4 GB | ~2s | Mais rápido |
| `Qwen2.5-3B-Instruct` | ~7 GB | ~4s | Recomendado |
| `Qwen2.5-3B-Instruct-AWQ` | ~3 GB | ~3s | Requer AWQ disponível no HF |
