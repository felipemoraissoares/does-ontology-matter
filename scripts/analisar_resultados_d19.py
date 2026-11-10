"""
Análise automática pós-mini-teste D-19.

Computa:
  1. Score D-20 por variante: avg_fmt_error + avg_parse_error (C0+C1 mean, 1_movie+2_music mean)
  2. % de prosa por variante (response != "" mas triples == [])
  3. Vencedora (regra D-20 pré-registrada, incluindo exceção de prosa universal)
  4. Paralelismo: comparação response-a-response entre d19_parallel e piloto-v2 C1_oneshot

Uso:
  python scripts/analisar_resultados_d19.py \\
      --d19_dir resultados/d19 \\
      --piloto_dir resultados/piloto \\
      --bench_dir ../Text2KGBench \\
      --manifesto /tmp/minilancador_d19_kaggle.json
"""
import argparse
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

VARIANTES = ["A", "B", "C", "D"]
CONDICOES = ["C0", "C1"]
ONTOS     = ["1_movie", "2_music"]


def ler_jsonl(caminho):
    if not os.path.exists(caminho):
        return []
    with open(caminho, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d19_dir",    required=True)
    p.add_argument("--piloto_dir", required=True)
    p.add_argument("--bench_dir",  required=True)
    p.add_argument("--manifesto",  required=True)
    args = p.parse_args()

    with open(args.manifesto, encoding="utf-8") as f:
        exps = json.load(f)

    print("\n" + "=" * 70)
    print("ANALISE D-20 — MINI-TESTE DE INSTRUCAO DE FORMATO")
    print("=" * 70)

    scores = {}
    rows   = []

    for var in VARIANTES:
        fmt_errs_v  = []
        parse_errs_v = []

        for cond in CONDICOES:
            for onto in ONTOS:
                # Localiza o experimento correspondente no manifesto
                exp = None
                for e in exps:
                    if (e.get("d19_variante") == var
                            and e.get("condicao") == cond
                            and e.get("ontologia_id") == onto):
                        exp = e
                        break
                if exp is None:
                    continue

                caminho = exp.get("saida", "")
                if not os.path.isabs(caminho):
                    caminho = os.path.join(args.d19_dir, os.path.basename(caminho))

                registros = ler_jsonl(caminho)
                if not registros:
                    print(f"  FALTANDO: var={var} cond={cond} onto={onto} ({caminho})")
                    continue

                n      = len(registros)
                fmt_e  = sum(float(r.get("fmt_error", 0)) for r in registros) / n
                parse_e = sum(
                    1 for r in registros
                    if r.get("response", "") and not r.get("triples")
                ) / n

                fmt_errs_v.append(fmt_e)
                parse_errs_v.append(parse_e)
                rows.append({
                    "variante": var, "condicao": cond, "onto": onto,
                    "n": n, "fmt_error": fmt_e, "parse_error": parse_e,
                })

        if fmt_errs_v:
            avg_fmt   = sum(fmt_errs_v) / len(fmt_errs_v)
            avg_parse = sum(parse_errs_v) / len(parse_errs_v)
            scores[var] = {
                "avg_fmt_error":   avg_fmt,
                "avg_parse_error": avg_parse,
                "avg_prosa":       avg_parse,  # prosa ≈ parse_error
                "score":           avg_fmt + avg_parse,
            }

    # Tabela detalhada
    print(f"\n{'Var':>4} {'Cond':>5} {'Onto':>8} {'n':>5} {'fmt_err':>8} {'parse_err':>10}")
    print("-" * 50)
    for row in rows:
        print(f"  {row['variante']:>2}  {row['condicao']:>4}  {row['onto']:>8}  "
              f"{row['n']:>4}  {row['fmt_error']:>8.4f}  {row['parse_error']:>10.4f}")

    # Tabela resumida de scores
    print(f"\n{'Var':>4} {'score':>7} {'fmt_err':>8} {'parse_err':>10} {'prosa':>7}  status")
    print("-" * 60)

    todas_acima_prosa = all(s["avg_prosa"] > 0.20 for s in scores.values()) if scores else False
    candidatas  = []
    eliminadas  = []

    for var in VARIANTES:
        if var not in scores:
            continue
        s = scores[var]
        if not todas_acima_prosa and s["avg_prosa"] > 0.20:
            status = "ELIMINADA (prosa>20%)"
            eliminadas.append(var)
        else:
            status = "candidata"
            candidatas.append(var)
        print(f"  {var:>2}  {s['score']:>7.4f}  {s['avg_fmt_error']:>8.4f}  "
              f"{s['avg_parse_error']:>10.4f}  {s['avg_prosa']:>7.4f}  {status}")

    if todas_acima_prosa:
        print("\n  NOTA: todas as variantes >20% prosa — criterio de exclusao ignorado (D-20 excecao).")

    if candidatas:
        melhor_var = min(candidatas, key=lambda v: (scores[v]["score"], VARIANTES.index(v)))
        print(f"\n  VENCEDORA: Variante {melhor_var}  (score={scores[melhor_var]['score']:.4f})")
    else:
        print("\n  ERRO: nenhuma candidata disponivel.")

    # Comparação de paralelismo
    print("\n" + "=" * 70)
    print("COMPARACAO DE PARALELISMO — workers=16 vs piloto-v2 C1_oneshot_original")
    print("=" * 70)

    parallel_exp = next(
        (e for e in exps if e.get("workers", 1) > 1 and e.get("condicao") == "C1_oneshot_original"),
        None,
    )
    if parallel_exp:
        par_path = parallel_exp.get("saida", "")
        if not os.path.isabs(par_path):
            par_path = os.path.join(args.d19_dir, os.path.basename(par_path))
        par_registros = {r["id"]: r for r in ler_jsonl(par_path)}

        pilot_path = os.path.join(
            args.piloto_dir,
            "piloto_tkg_1_movie_c1_oneshot_original_qwen25_7b_instruct_awq.jsonl",
        )
        pilot_registros = {r["id"]: r for r in ler_jsonl(pilot_path)}

        comuns    = set(par_registros) & set(pilot_registros)
        identicas = sum(
            1 for sid in comuns
            if par_registros[sid].get("response") == pilot_registros[sid].get("response")
        )
        diferentes = len(comuns) - identicas

        print(f"\n  Registros no paralelo:  {len(par_registros)}")
        print(f"  Registros no piloto-v2: {len(pilot_registros)}")
        print(f"  Em comum:               {len(comuns)}")
        print(f"  Respostas IDENTICAS:    {identicas} / {len(comuns)}")
        print(f"  Respostas DIFERENTES:   {diferentes}")
        if comuns:
            taxa = identicas / len(comuns)
            print(f"  Taxa de identidade:     {taxa:.1%}")
            if identicas == len(comuns):
                print("  RESULTADO: workers=16 e DETERMINÍSTICO (100% identico a workers=1)")
            elif taxa > 0.95:
                print("  RESULTADO: quase determinístico (>95%)")
            else:
                print(f"  AVISO: {diferentes} respostas diferem — verificar temperatura/seed")
    else:
        print("  Experimento paralelo nao encontrado no manifesto.")

    print("\nAnalise concluida.")


if __name__ == "__main__":
    main()
