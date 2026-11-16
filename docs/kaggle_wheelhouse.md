# Wheelhouse do Kaggle — acelerar a instalação (Etapa A)

Hoje cada sessão do Kaggle gasta ~14 min instalando os pacotes (vLLM, torch, etc.).
O *wheelhouse* é uma pasta com todos esses pacotes **já baixados**. Você sobe essa pasta
uma única vez como "Dataset" do Kaggle e, nas próximas sessões, a instalação passa a ser
quase instantânea (instala a partir dos arquivos locais, sem baixar nada da internet).

As versões ficam **travadas** (D-35): a mesma pilha em todas as sessões, por semanas, sem
risco de um upgrade silencioso mudar os resultados.

Você faz isto **uma vez**. Depois é só anexar o Dataset ao notebook do experimento.

---

## Parte 0 — Gerar o lock (pré-requisito)

O wheelhouse usa o `requirements-kaggle.lock.txt`, que lista as versões **reais** do Kaggle.
Rode o lançador com `MODO = "congelar_ambiente"` (Accelerator: None) **uma vez** para gerar e
commitar esse lock (ver `docs/experimento_completo_kaggle.md`). Sem o lock, pule para a
instalação normal (o lançador instala sem travar e gera o lock sozinho).

## Parte 1 — Gerar o wheelhouse (uma vez, no Kaggle)

> Precisa ser feito **no Kaggle** porque os pacotes são específicos do sistema do Kaggle
> (Linux). Não dá para gerar no seu PC Windows.

1. No Kaggle, crie um **Notebook** novo (`+ Create` → `New Notebook`).
2. (Opcional) Em **Settings**, ligue **GPU T4 x2** — não é obrigatório para baixar os
   pacotes, mas garante o mesmo ambiente.
3. Cole **uma célula** com o conteúdo abaixo e rode (ícone ▶) — usa o **lock**:

   ```python
   !git clone --depth=1 https://github.com/felipemoraissoares/does-ontology-matter.git
   !python does-ontology-matter/scripts/montar_wheelhouse.py \
       --requirements does-ontology-matter/requirements-kaggle.lock.txt \
       --saida /kaggle/working/wheelhouse
   ```

4. Espere terminar. No fim, a célula imprime quantos arquivos foram baixados e o tamanho
   total (alguns GB — normal, o torch e o vLLM são grandes).

## Parte 2 — Salvar como Dataset

1. No canto superior direito, clique **Save Version** → escolha **Save & Run All (Commit)**
   e confirme. Espere a execução terminar (o Kaggle roda o notebook do começo ao fim).
2. Quando terminar, abra a versão salva. Na aba **Output**, você verá a pasta `wheelhouse`.
3. Clique nos três pontinhos (`⋮`) ao lado de **Output** → **Create Dataset** (ou
   **New Dataset**). Dê um nome, por exemplo **`wheelhouse-does-ontology`**, deixe como
   **Private** e crie.

> Alternativa: `+ Create` → `New Dataset` → aba **Upload** não serve aqui (os arquivos
> estão no Kaggle, não no seu PC). Use o caminho pela **Output** do notebook acima.

## Parte 3 — Usar no experimento

1. Abra o notebook do experimento (`kaggle/lancador.ipynb`, importado no Kaggle).
2. Na barra lateral direita, em **Input**, clique **+ Add Input** → aba **Datasets** →
   procure **`wheelhouse-does-ontology`** → **Add**.
3. Rode o notebook normalmente. A célula de instalação (célula 5) **detecta
   automaticamente** o wheelhouse anexado e instala a partir dele
   (`pip install --no-index --find-links ...`). Se o Dataset não estiver anexado, ela cai
   para a instalação travada normal (`pip install -r requirements-kaggle.txt`), mais lenta.

Você saberá que funcionou pela mensagem: `Wheelhouse detectado: /kaggle/input/... - instalando offline.`

---

## Quando refazer o wheelhouse

Só é preciso refazer se o `requirements-kaggle.txt` mudar (nova versão de algum pacote).
Nesse caso, repita a Parte 1 e **Save Version** de novo: o Dataset é atualizado com a nova
versão e o notebook do experimento passa a usar a nova automaticamente.
