"""Gera um *wheelhouse* — os pacotes (wheels) das dependências travadas já baixados —
para subir como Dataset privado do Kaggle e acelerar a instalação (D-35, Etapa A).

IMPORTANTE: rode este script **no Kaggle** (ou em outro Linux com o MESMO Python do
ambiente de execução). As wheels são específicas de plataforma/versão de Python; baixá-las
no Windows produziria arquivos que o Kaggle não consegue instalar.

Uso (numa célula de um notebook Kaggle, após clonar o repo):
    !python does-ontology-matter/scripts/montar_wheelhouse.py \
        --requirements does-ontology-matter/requirements-kaggle.txt \
        --saida /kaggle/working/wheelhouse

Depois, crie um Dataset do Kaggle a partir da pasta de saída (ver docs/kaggle_wheelhouse.md).
"""
import argparse
import os
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description="Baixa as wheels das dependências travadas.")
    p.add_argument("--requirements", default="requirements-kaggle.txt",
                   help="Arquivo de requisitos travado (D-35).")
    p.add_argument("--saida", default="wheelhouse",
                   help="Pasta onde salvar as wheels (vira o Dataset do Kaggle).")
    args = p.parse_args()

    if not os.path.exists(args.requirements):
        print(f"ERRO: requisitos não encontrados: {args.requirements}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.saida, exist_ok=True)

    # pip download resolve e baixa TODA a árvore transitiva de dependências como wheels.
    cmd = [sys.executable, "-m", "pip", "download",
           "-r", args.requirements, "-d", args.saida]
    print("Executando:", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("ERRO: pip download falhou. Verifique se está rodando no Linux/Kaggle "
              "com o Python correto.", file=sys.stderr)
        sys.exit(r.returncode)

    # Copia o próprio requirements para o wheelhouse (rastreabilidade do lock).
    shutil.copy(args.requirements, os.path.join(args.saida, os.path.basename(args.requirements)))

    arquivos = sorted(os.listdir(args.saida))
    tam_total = sum(os.path.getsize(os.path.join(args.saida, f)) for f in arquivos)
    print(f"\n{len(arquivos)} arquivos em {args.saida}/ ({tam_total/1e6:.0f} MB).")
    print("Primeiros itens:")
    for f in arquivos[:10]:
        print(f"  {f}")
    print("\nPróximo passo: crie um Dataset privado do Kaggle a partir desta pasta e "
          "anexe-o ao notebook do experimento (ver docs/kaggle_wheelhouse.md).")


if __name__ == "__main__":
    main()
