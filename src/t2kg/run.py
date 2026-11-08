# CLI do pipeline de extração.
#
# Uso:
#   python -m t2kg.run --manifesto manifesto.json
#   python -m t2kg.run --manifesto manifesto.json --frases_max 10 --verbose
#
# Formato do manifesto (lista de experimentos):
#   [
#     {
#       "id": "exp_01",
#       "dataset": "tekgen",
#       "split": "test",
#       "ontologia_id": "1_movie",
#       "condicao": "C1_oneshot_original",
#       "modelo": "Qwen/Qwen2.5-7B-Instruct-AWQ",
#       "provedor": "openai_compat",
#       "provedor_url": "http://localhost:8000/v1",
#       "api_key": "token-nao-usado",
#       "temperatura": 0,
#       "max_tokens": 512,
#       "saida": "resultados/exp01/1_movie.jsonl"
#     }
#   ]
#
# Para o provedor "mock", os campos "modelo", "provedor_url" e "api_key" são ignorados.
# Para "openai_compat", todos são obrigatórios exceto "api_key" (default: "token-nao-usado").

import argparse
import json
import os
import sys

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BENCH_DIR_PADRAO = os.path.join(WORKSPACE, "Text2KGBench")


def _criar_provedor(exp: dict):
    tipo = exp.get("provedor", "openai_compat")

    if tipo == "mock":
        from t2kg.pipeline.provedores.mock import ProvedorMock
        return ProvedorMock()

    if tipo == "openai_compat":
        from t2kg.pipeline.provedores.openai_compat import ProvedorOpenAICompat
        return ProvedorOpenAICompat(
            url=exp["provedor_url"],
            modelo=exp["modelo"],
            api_key=exp.get("api_key", "token-nao-usado"),
        )

    raise ValueError(f"Provedor desconhecido: {tipo!r}. Use 'openai_compat' ou 'mock'.")


def main():
    parser = argparse.ArgumentParser(
        description="Executa o pipeline de extração de triplas Text2KG."
    )
    parser.add_argument(
        "--manifesto", required=True,
        help="Caminho para o arquivo JSON com a lista de experimentos.",
    )
    parser.add_argument(
        "--bench_dir", default=BENCH_DIR_PADRAO,
        help=f"Diretório raiz do Text2KGBench (padrão: {BENCH_DIR_PADRAO})",
    )
    parser.add_argument(
        "--frases_max", type=int, default=None,
        help="Limite de frases por experimento (útil para testes rápidos).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Imprime progresso linha a linha.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.manifesto):
        print(f"Manifesto não encontrado: {args.manifesto}", file=sys.stderr)
        sys.exit(1)

    with open(args.manifesto, encoding="utf-8") as f:
        experimentos = json.load(f)

    if isinstance(experimentos, dict):
        experimentos = [experimentos]

    from t2kg.pipeline.executar import executar

    for exp in experimentos:
        exp_id = exp.get("id", "?")
        print(f"\n=== Experimento: {exp_id} ===")
        print(f"    dataset={exp['dataset']}  split={exp['split']}  "
              f"onto={exp['ontologia_id']}  cond={exp['condicao']}")

        provedor = _criar_provedor(exp)

        n = executar(
            bench_dir=args.bench_dir,
            dataset=exp["dataset"],
            split=exp["split"],
            ontologia_id=exp["ontologia_id"],
            condicao=exp["condicao"],
            provedor=provedor,
            saida=exp["saida"],
            max_tokens=exp.get("max_tokens", 512),
            temperatura=exp.get("temperatura", 0.0),
            frases_max=args.frases_max,
            verbose=args.verbose,
        )
        print(f"    {n} frase(s) processada(s)  →  {exp['saida']}")

    print("\nConcluído.")


if __name__ == "__main__":
    main()
