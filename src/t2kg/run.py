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
            seed=exp.get("seed"),
        )

    raise ValueError(f"Provedor desconhecido: {tipo!r}. Use 'openai_compat' ou 'mock'.")


def _contar_registros_validos(caminho: str) -> int:
    """Conta linhas JSON válidas com 'id' no JSONL de saída.

    Ignora linhas vazias e uma eventual linha final truncada (sessão derrubada no
    meio de uma escrita) — essas não contam como frase concluída (C-2/C-3).
    """
    if not os.path.exists(caminho):
        return 0
    n = 0
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            try:
                obj = json.loads(linha)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "id" in obj:
                n += 1
    return n


def _push_resultado(repo_dir: str, exp_id: str, saida: str) -> bool:
    """git add + commit + push SINCRONIZADO após um experimento. Retorna True se empurrado.

    Verifica que o JSONL de saída entrou no commit (evita a falsa confirmação do D-17).
    O push sincroniza (fetch + rebase) e retenta — resiste a um remoto que avançou (ex.:
    outra máquina empurrando no meio do lote).
    """
    import subprocess
    from t2kg.git_sync import sincronizar_e_push

    subprocess.run(["git", "-C", repo_dir, "add", "-A"], capture_output=True, text=True)

    saida_rel = os.path.relpath(saida, repo_dir).replace("\\", "/")
    r_status = subprocess.run(
        ["git", "-C", repo_dir, "status", "--porcelain", "--", saida_rel],
        capture_output=True, text=True)
    jsonl_staged = bool(r_status.stdout.strip())

    r_commit = subprocess.run(
        ["git", "-C", repo_dir, "commit", "-m", f"auto: {exp_id}"],
        capture_output=True, text=True)
    commit_saida = r_commit.stdout + r_commit.stderr
    commit_ok = r_commit.returncode == 0 or "nothing to commit" in commit_saida
    if not commit_ok:
        print(f"    AVISO push — git commit falhou: {r_commit.stderr.strip()[:200]}", file=sys.stderr)
        return False
    if not jsonl_staged:
        print(f"    AVISO: commit SEM o JSONL de resultado ({saida_rel}). "
              f"Verifique .gitignore. Experimento: {exp_id}", file=sys.stderr)

    pushed = sincronizar_e_push(repo_dir)
    if pushed:
        print(f"    Push OK: {exp_id}")
    else:
        print(f"    AVISO: push de {exp_id} NÃO confirmado após as tentativas.", file=sys.stderr)
    return pushed


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
        help="Limite de frases por experimento (substitui frases_max do manifesto).",
    )
    parser.add_argument(
        "--saida_base", default=None,
        help="Substitui o diretório-base dos arquivos 'saida' do manifesto. "
             "Útil para redirecionar resultados em ambientes remotos (Kaggle, Colab).",
    )
    parser.add_argument(
        "--skip_existentes", action="store_true",
        help="Pula experimentos cujo arquivo de saída já existe. "
             "Permite retomar uma execução interrompida.",
    )
    parser.add_argument(
        "--auto_push", action="store_true",
        help="Faz git add + commit + push após cada experimento concluído. "
             "Requer GITHUB_TOKEN configurado no remote origin.",
    )
    parser.add_argument(
        "--push_repo_dir", default=None,
        help="Diretório raiz do repositório git para --auto_push. "
             "Padrão: detecta automaticamente a partir do caminho do manifesto.",
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
        # Formato novo: {"reinduzir_c2": bool, "experimentos": [...]}
        if "experimentos" in experimentos:
            experimentos = experimentos["experimentos"]
        else:
            experimentos = [experimentos]

    # Detecta diretório do repo para auto_push
    push_repo_dir = args.push_repo_dir
    if args.auto_push and push_repo_dir is None:
        push_repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    from t2kg.pipeline.executar import executar
    from t2kg.pipeline.dados import carregar_frases

    sem_push = []  # experimentos cujo push não foi confirmado
    for exp in experimentos:
        exp_id = exp.get("id", "?")

        # Redireciona saida se --saida_base fornecido
        saida_original = exp["saida"]
        if args.saida_base:
            exp["saida"] = os.path.join(args.saida_base, os.path.basename(saida_original))

        # frases_max: CLI tem precedência; fallback para o manifesto
        frases_max_efetivo = args.frases_max if args.frases_max is not None else exp.get("frases_max")
        frases_seed = exp.get("frases_seed")

        # --skip_existentes: só pula se o JSONL tem o número ESPERADO de frases válidas.
        # Arquivo parcial (sessão Kaggle derrubada) é retomado pelo checkpoint de executar(),
        # nunca dado como concluído (C-2).
        if args.skip_existentes:
            n_tem = _contar_registros_validos(exp["saida"])
            if n_tem > 0:
                n_total = len(carregar_frases(
                    args.bench_dir, exp["dataset"], exp["split"], exp["ontologia_id"]))
                n_esperado = (min(frases_max_efetivo, n_total)
                              if frases_max_efetivo is not None else n_total)
                if n_tem >= n_esperado:
                    print(f"\n=== Experimento: {exp_id} "
                          f"[PULADO — {n_tem}/{n_esperado} frases completas] ===")
                    continue
                print(f"\n=== Experimento: {exp_id} "
                      f"[RETOMADA — {n_tem}/{n_esperado} frases já feitas] ===")

        print(f"\n=== Experimento: {exp_id} ===")
        print(f"    dataset={exp['dataset']}  split={exp['split']}  "
              f"onto={exp['ontologia_id']}  cond={exp['condicao']}")

        provedor = _criar_provedor(exp)

        if frases_max_efetivo is not None:
            print(
                f"    AVISO: frases_max={frases_max_efetivo} ativo — "
                "use modo_amostra=True na avaliação (avaliar_resultados.py --auto_modo_amostra)."
            )

        workers = int(exp.get("workers", 1))
        instrucao_formato = exp.get("instrucao_formato")
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
            frases_max=frases_max_efetivo,
            frases_seed=frases_seed,
            verbose=args.verbose,
            ontologia_parceira_id=exp.get("ontologia_parceira_id"),
            workers=workers,
            instrucao_formato=instrucao_formato,
            c2_subdir=exp.get("c2_subdir"),
        )
        print(f"    {n} frase(s) processada(s)  ->  {exp['saida']}")

        if args.auto_push:
            if not _push_resultado(push_repo_dir, exp_id, exp["saida"]):
                sem_push.append(exp_id)

    if args.auto_push:
        from t2kg.git_sync import sincronizar_e_push
        # Push final de tudo o que ficou pendente no lote.
        if sincronizar_e_push(push_repo_dir):
            sem_push = []
            print("\nPush final do lote OK — todos os resultados enviados.")
        else:
            print("\n" + "=" * 60, file=sys.stderr)
            print("⚠️  PUSH FINAL DO LOTE FALHOU — resultados podem estar só locais.",
                  file=sys.stderr)
            print("=" * 60, file=sys.stderr)
        if sem_push:
            print(f"⚠️  {len(sem_push)} experimento(s) SEM push confirmado: "
                  f"{', '.join(sem_push)}", file=sys.stderr)

    print("\nConcluído.")


if __name__ == "__main__":
    main()
