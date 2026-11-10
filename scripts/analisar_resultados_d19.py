"""
Análise automática pós-mini-teste D-19.

Computa:
  1. Score D-20 por variante: avg_fmt_error + avg_parse_error (C0+C1 mean, 1_movie+2_music mean)
  2. fmt_error subcategorias: tipo_a (cópia do nome da classe), tipo_b (classe:valor),
     tipo_c (classe "valor") — apenas para C1 (ontologia disponível)
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

from t2kg.avaliacao.original import calcular_taxa_erro_formato  # noqa: E402

VARIANTES = ["A", "B", "C", "D"]
CONDICOES = ["C0", "C1"]
ONTOS     = ["1_movie", "2_music"]


def ler_jsonl(caminho):
    if not os.path.exists(caminho):
        return []
    with open(caminho, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def carregar_ontologia_ref(bench_dir, dataset, onto_id):
    """Carrega a ontologia de referência para computar fmt_error em C1."""
    if dataset == "tekgen":
        p = os.path.join(bench_dir, "data", "wikidata_tekgen", "ontologies", f"{onto_id}_ontology.json")
    else:
        p = os.path.join(bench_dir, "data", "dbpedia_webnlg", "ontologies", f"{onto_id}_ontology.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def calcular_erros_formato_detalhado(ontologia, triplas):
    """Retorna dict com total, tipo_a, tipo_b, tipo_c (frações sobre n_triplas).

    tipo_a: entidade == nome da classe (cópia literal)
    tipo_b: entidade começa com 'classe:' (prefixo com dois-pontos)
    tipo_c: entidade começa com 'classe "' ou "classe '" (prefixo com aspas)
    """
    if not triplas:
        return {"total": 0.0, "tipo_a": 0.0, "tipo_b": 0.0, "tipo_c": 0.0}
    classes = {c["label"].lower() for c in ontologia.get("concepts", [])}
    n = len(triplas)
    total = tipo_a = tipo_b = tipo_c = 0
    for tr in triplas:
        tripla_tipo = None
        for entidade in (tr[0], tr[2]):
            e = entidade.lower().strip()
            for cls in classes:
                padrao1 = (e == cls and not any(c.isupper() for c in entidade))
                padrao2 = e.startswith(cls + ":")
                padrao3 = e.startswith(cls + ' "') or e.startswith(cls + " '")
                if padrao1:
                    tripla_tipo = "a"
                    break
                elif padrao2:
                    tripla_tipo = "b"
                    break
                elif padrao3:
                    tripla_tipo = "c"
                    break
            if tripla_tipo:
                break
        if tripla_tipo == "a":
            tipo_a += 1; total += 1
        elif tripla_tipo == "b":
            tipo_b += 1; total += 1
        elif tripla_tipo == "c":
            tipo_c += 1; total += 1
    return {"total": total / n, "tipo_a": tipo_a / n, "tipo_b": tipo_b / n, "tipo_c": tipo_c / n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d19_dir",    required=True)
    p.add_argument("--piloto_dir", required=True)
    p.add_argument("--bench_dir",  required=True)
    p.add_argument("--manifesto",  required=True)
    args = p.parse_args()

    with open(args.manifesto, encoding="utf-8") as f:
        exps = json.load(f)

    # Pré-carrega ontologias de referência (para fmt_error em C1)
    ontos_ref = {}
    for onto in ONTOS:
        for e in exps:
            dataset = e.get("dataset", "tekgen")
            if e.get("ontologia_id") == onto and dataset not in ontos_ref.get(onto, {}):
                ont = carregar_ontologia_ref(args.bench_dir, dataset, onto)
                if ont:
                    ontos_ref[onto] = {"dataset": dataset, "ontologia": ont}
                break

    print("\n" + "=" * 70)
    print("ANALISE D-20 — MINI-TESTE DE INSTRUCAO DE FORMATO")
    print("=" * 70)

    scores = {}
    rows   = []

    for var in VARIANTES:
        fmt_errs_v   = []
        parse_errs_v = []

        for cond in CONDICOES:
            for onto in ONTOS:
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

                n = len(registros)
                # fmt_error: computed from triplas vs ontologia (C1 only; C0 = 0 por definição)
                if cond == "C1" and onto in ontos_ref:
                    ontologia = ontos_ref[onto]["ontologia"]
                    erros_det = [calcular_erros_formato_detalhado(ontologia, r.get("triples", []))
                                 for r in registros]
                    fmt_e      = sum(e["total"]  for e in erros_det) / n
                    fmt_a      = sum(e["tipo_a"] for e in erros_det) / n
                    fmt_b      = sum(e["tipo_b"] for e in erros_det) / n
                    fmt_c      = sum(e["tipo_c"] for e in erros_det) / n
                else:
                    fmt_e = fmt_a = fmt_b = fmt_c = 0.0

                parse_e = sum(
                    1 for r in registros
                    if r.get("response", "") and not r.get("triples")
                ) / n

                fmt_errs_v.append(fmt_e)
                parse_errs_v.append(parse_e)
                rows.append({
                    "variante": var, "condicao": cond, "onto": onto,
                    "n": n, "fmt_error": fmt_e, "fmt_a": fmt_a, "fmt_b": fmt_b, "fmt_c": fmt_c,
                    "parse_error": parse_e,
                })

        if fmt_errs_v:
            avg_fmt   = sum(fmt_errs_v) / len(fmt_errs_v)
            avg_parse = sum(parse_errs_v) / len(parse_errs_v)
            scores[var] = {
                "avg_fmt_error":   avg_fmt,
                "avg_parse_error": avg_parse,
                "score":           avg_fmt + avg_parse,
            }

    # Tabela detalhada
    print(f"\n{'Var':>4} {'Cond':>5} {'Onto':>8} {'n':>5} "
          f"{'fmt_err':>8} {'fmt_a':>6} {'fmt_b':>6} {'fmt_c':>6} {'parse_err':>10}")
    print("-" * 65)
    for row in rows:
        print(f"  {row['variante']:>2}  {row['condicao']:>4}  {row['onto']:>8}  "
              f"{row['n']:>4}  {row['fmt_error']:>8.4f}  {row['fmt_a']:>6.4f}  "
              f"{row['fmt_b']:>6.4f}  {row['fmt_c']:>6.4f}  {row['parse_error']:>10.4f}")

    # Tabela resumida de scores
    print(f"\n{'Var':>4} {'score':>7} {'fmt_err':>8} {'parse_err':>10}  status")
    print("-" * 55)

    todas_acima_prosa = all(s["avg_parse_error"] > 0.20 for s in scores.values()) if scores else False
    candidatas  = []
    eliminadas  = []

    for var in VARIANTES:
        if var not in scores:
            continue
        s = scores[var]
        if not todas_acima_prosa and s["avg_parse_error"] > 0.20:
            status = "ELIMINADA (parse>20%)"
            eliminadas.append(var)
        else:
            status = "candidata"
            candidatas.append(var)
        print(f"  {var:>2}  {s['score']:>7.4f}  {s['avg_fmt_error']:>8.4f}  "
              f"{s['avg_parse_error']:>10.4f}  {status}")

    if todas_acima_prosa:
        print("\n  NOTA: todas as variantes >20% parse — criterio de exclusao ignorado (D-20 excecao).")

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
            elif taxa >= 0.90:
                print(f"  RESULTADO: majoritariamente determinístico ({taxa:.1%})")
            else:
                print(f"  AVISO: {diferentes} respostas diferem — verificar temperatura/seed")

        # Mostra 3 exemplos de diferenças
        if diferentes > 0:
            print(f"\n  3 exemplos de respostas DIFERENTES:")
            n_ex = 0
            for sid in sorted(comuns):
                if par_registros[sid].get("response") != pilot_registros[sid].get("response"):
                    print(f"\n  ID: {sid}")
                    r_par = par_registros[sid].get("response", "")[:200]
                    r_pil = pilot_registros[sid].get("response", "")[:200]
                    print(f"  workers=16: {repr(r_par)}")
                    print(f"  piloto-v2:  {repr(r_pil)}")
                    n_ex += 1
                    if n_ex >= 3:
                        break
    else:
        print("  Experimento paralelo nao encontrado no manifesto.")

    print("\nAnalise concluida.")


if __name__ == "__main__":
    main()
