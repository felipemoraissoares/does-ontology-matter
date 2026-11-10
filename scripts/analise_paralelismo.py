"""Análise de paralelismo: workers=16 vs workers=1 para d19 piloto."""
import json
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

BENCH = os.path.join(REPO, "..", "Text2KGBench")

f_par = os.path.join(REPO, "resultados/d19/d19_paralelismo_1movie_c1os.jsonl")
f_pil = os.path.join(REPO, "resultados/piloto/piloto_tkg_1_movie_c1_oneshot_original_qwen25_7b_instruct_awq.jsonl")

par = {}
for l in open(f_par, encoding="utf-8"):
    r = json.loads(l)
    par[r["id"]] = r

pil = {}
for l in open(f_pil, encoding="utf-8"):
    r = json.loads(l)
    pil[r["id"]] = r

ids_comum = sorted(set(par) & set(pil))
n_identicas = sum(1 for id in ids_comum if par[id]["response"] == pil[id]["response"])
diffs = [id for id in ids_comum if par[id]["response"] != pil[id]["response"]]

print("=== PARALELISMO ===")
print(f"n_total:     {len(ids_comum)}")
print(f"n_identicas: {n_identicas}")
print(f"n_diferentes:{len(diffs)}")
print(f"taxa:        {n_identicas/len(ids_comum)*100:.1f}%")

# Verificar finish_reason
for label, dados in [("workers=16 (paralelo)", par), ("workers=1 (piloto)", pil)]:
    truncados = [(id, r) for id, r in dados.items() if r.get("finish_reason") == "length"]
    print(f"\n{label} - truncamentos finish_reason=length: {len(truncados)}")
    for id, r in truncados[:5]:
        resp = r["response"]
        print(f"  ID: {id}")
        print(f"  resp[:300]: {resp[:300]!r}")

# Verificar se diferenças coincidem com truncamentos
ids_truncado_par = {id for id, r in par.items() if r.get("finish_reason") == "length"}
ids_truncado_pil = {id for id, r in pil.items() if r.get("finish_reason") == "length"}
diffs_set = set(diffs)
print(f"\nDiferenças que são truncamentos no paralelo: {len(diffs_set & ids_truncado_par)}")
print(f"Diferenças que são truncamentos no piloto:  {len(diffs_set & ids_truncado_pil)}")

# 3 exemplos das diferenças
print("\n=== 3 EXEMPLOS DE DIFERENÇAS ===")
for id in diffs[:3]:
    sent = par[id].get("sent", pil[id].get("sent", ""))[:80]
    resp_par = par[id]["response"]
    resp_pil = pil[id]["response"]
    print(f"\nID: {id}")
    print(f"Sent: {sent}")
    print(f"Paralelo (w16): {resp_par!r}")
    print(f"Piloto  (w1):   {resp_pil!r}")

# Calcular métricas de avaliação usando funções de baixo nível
print("\n=== MÉTRICAS DE AVALIAÇÃO ===")
try:
    import tempfile
    from nltk.stem import PorterStemmer
    from t2kg.avaliacao.original import (
        calcular_taxa_erro_formato, calcular_precisao_recall_f1,
        calcular_conformidade_ontologica, calcular_alucinacao_sujeito_objeto,
        normalizar_tripla
    )

    gt_path = os.path.join(BENCH, "data/wikidata_tekgen/ground_truth/ont_1_movie_ground_truth.jsonl")
    onto_path = os.path.join(BENCH, "data/wikidata_tekgen/ontologies/1_movie_ontology.json")

    onto = json.load(open(onto_path, encoding="utf-8"))
    gt = {}
    for l in open(gt_path, encoding="utf-8"):
        r = json.loads(l)
        gt[r["id"]] = r

    ps = PorterStemmer()

    for label, dados_dict in [("workers=16 (paralelo)", par), ("workers=1 (piloto)", pil)]:
        ids_avaliacao = [id for id in ids_comum if id in gt]

        tp = tr_ = tf1 = tc = trh = tsh = toh = tef = tpe = 0.0
        n = 0

        for id in ids_avaliacao:
            r = dados_dict[id]
            gt_r = gt[id]
            sys_triplas = r.get("triples", [])
            resp = r.get("response", "")
            frase = gt_r["sent"]
            gt_triplas = [[t["sub"], t["rel"], t["obj"]] for t in gt_r["triples"]]

            # parse error
            if resp.strip() and not sys_triplas:
                tpe += 1.0

            rels_gabarito = {t[1].replace(" ", "_") for t in gt_triplas}
            fil_triplas = [t for t in sys_triplas if t[1] in rels_gabarito]
            sys_norm = {normalizar_tripla(t[0], t[1], t[2]) for t in fil_triplas}
            gt_norm = {normalizar_tripla(t[0], t[1], t[2]) for t in gt_triplas}

            precision, recall, f1 = calcular_precisao_recall_f1(gt_norm, sys_norm)
            conformidade, rel_hal = calcular_conformidade_ontologica(onto, sys_triplas)
            sub_hal, obj_hal = calcular_alucinacao_sujeito_objeto(ps, onto, frase, sys_triplas)
            fmt_err = calcular_taxa_erro_formato(onto, sys_triplas)

            tp += precision; tr_ += recall; tf1 += f1
            tc += conformidade; trh += rel_hal; tsh += sub_hal; toh += obj_hal
            tef += fmt_err
            n += 1

        total = n if n > 0 else 1
        print(f"\n{label} (n={n}):")
        print(f"  P={tp/total:.4f}  R={tr_/total:.4f}  F1={tf1/total:.4f}")
        print(f"  Conf={tc/total:.4f}")
        print(f"  SH={tsh/total:.4f}  RH={trh/total:.4f}  OH={toh/total:.4f}")
        print(f"  FmtErr={tef/total:.4f}  ParseErr={tpe/total:.4f}")

except Exception as e:
    print(f"ERRO ao calcular métricas: {e}")
    import traceback
    traceback.print_exc()
