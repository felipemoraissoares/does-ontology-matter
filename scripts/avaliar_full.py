"""
Avaliação reprodutível de uma rodada completa (full) a partir de um manifesto.

O que faz:
  1. Régua ORIGINAL (t2kg.avaliacao.original, via scripts/avaliar_resultados.py):
     grava a avaliação por frase e as médias por (dataset, condição) em
     resultados/avaliacao_full/<slug>/  —  mesmo formato de resultados/avaliacao_piloto_v3/
     (consumido pelo app).
  2. Régua ESTENDIDA: F1 projetado (projeção de relações pela ontologia de referência),
     com o limiar lido da fonte única configs/avaliacao.json via t2kg.avaliacao.limiares
     (full = 0.45, D-36). Redundância semântica (H1). % de respostas vazias.
  3. Tabela com IC 95% por cluster-bootstrap (seed fixa) para TODAS as células, mais a
     tabela de sobreposição da C5, gravadas em docs/resultados_7b_completo.md.

Uso:
  python scripts/avaliar_full.py \
      --manifesto configs/pipeline/manifesto_7b_completo.json \
      --slug qwen2.5-7b --rodada full \
      --doc docs/resultados_7b_completo.md
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH = os.path.abspath(os.path.join(REPO, "..", "Text2KGBench"))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import avaliar_resultados as ar  # régua original + agrupamento por (dataset, condição)
from t2kg.avaliacao.original import (
    normalizar_tripla, calcular_precisao_recall_f1,
    calcular_conformidade_ontologica, calcular_alucinacao_sujeito_objeto,
    ler_jsonl, ler_json,
)
from t2kg.avaliacao.extended import (
    _normalizar_rotulo, _get_modelo, normalizar_chave_definicao, redundancia_semantica,
)
from t2kg.avaliacao.limiares import limiar_projecao, limiar_redundancia
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

COND_ORDER = ["C0", "C1", "C1_oneshot_original", "C2", "C3", "C5_vizinho", "C5_incongruente"]
COND_LABEL = {
    "C0": "C0 (livre)", "C1": "C1", "C1_oneshot_original": "C1 one-shot orig.",
    "C2": "C2 (induzida)", "C3": "C3", "C5_vizinho": "C5 vizinho",
    "C5_incongruente": "C5 incongruente",
}
C5_LABEL = {"C5_vizinho": "vizinho", "C5_incongruente": "incongruente"}
ps = PorterStemmer()


def _norm_rel(s):
    return s.replace(" ", "_").lower()


# ---------------------------------------------------------------------------
# Projeção de relações — mapeamento determinístico por rótulo (espelho de
# extended.projetar_relacoes, mas construído uma vez por ontologia/célula).
# ---------------------------------------------------------------------------
class Projetor:
    def __init__(self, def_dir, threshold):
        self.def_dir = def_dir
        self.threshold = threshold
        self._ref_cache = {}
        self._defs_cache = {}

    def defs(self, onto):
        if onto not in self._defs_cache:
            p = os.path.join(self.def_dir, f"definicoes_{onto}.json")
            self._defs_cache[onto] = ler_json(p).get("relacoes", {}) if os.path.exists(p) else None
        return self._defs_cache[onto]

    def _ref(self, onto, ontologia):
        if onto in self._ref_cache:
            return self._ref_cache[onto]
        rels_ref_orig = [r["label"] for r in ontologia["relations"]]
        defs = self.defs(onto)
        if defs is not None:
            defs_norm = {normalizar_chave_definicao(k): v for k, v in defs.items()}
            faltando = [r for r in rels_ref_orig if normalizar_chave_definicao(r) not in defs_norm]
            if faltando:
                rels_ref = [_normalizar_rotulo(r) for r in rels_ref_orig]
            else:
                rels_ref = [f"{_normalizar_rotulo(r)}: {defs_norm[normalizar_chave_definicao(r)]}"
                            for r in rels_ref_orig]
        else:
            rels_ref = [_normalizar_rotulo(r) for r in rels_ref_orig]
        embs = _get_modelo().encode(rels_ref, show_progress_bar=False)
        self._ref_cache[onto] = (rels_ref_orig, embs)
        return self._ref_cache[onto]

    def mapeamento(self, labels, onto, ontologia):
        labels = list(labels)
        if not labels:
            return {}
        rels_ref_orig, embs_ref = self._ref(onto, ontologia)
        embs_sys = _get_modelo().encode([_normalizar_rotulo(l) for l in labels], show_progress_bar=False)
        sim = cosine_similarity(embs_sys, embs_ref)
        mapping = {}
        for i, lab in enumerate(labels):
            j = int(np.argmax(sim[i]))
            if sim[i, j] >= self.threshold:
                mapping[lab] = rels_ref_orig[j]
        return mapping


def _f1(gt_norm, triplas):
    if not gt_norm:
        return 0.0
    _, _, f1 = calcular_precisao_recall_f1(gt_norm, {normalizar_tripla(*t) for t in triplas})
    return f1


def avaliar_cell(exps, ds, cond, proj, redund_thr):
    """Passo de precisão plena por frase (para tabela + bootstrap). Retorna (por_onto, redund, overlap)."""
    por_onto, redund_vals, overlap_vals = {}, [], []
    ontos = ar.TEKGEN_ONTOS if ds == "tekgen" else ar.WEBNLG_ONTOS
    eh_c5 = cond.startswith("C5")
    for onto in ontos:
        exp = next((e for e in exps if e["ontologia_id"] == onto), None)
        if exp is None:
            continue
        sys_path = os.path.join(REPO, exp["saida"].replace("/", os.sep))
        if not os.path.exists(sys_path):
            continue
        saida = {r["id"]: r for r in ler_jsonl(sys_path)}
        gab = {r["id"]: r for r in ler_jsonl(ar._gt_path(BENCH, ds, onto))}
        ontologia = ler_json(ar._onto_path(BENCH, ds, onto))

        parceira_rels = set()
        if eh_c5:
            pid = exp.get("ontologia_parceira_id")
            if pid:
                pont = ler_json(ar._onto_path(BENCH, ds, pid))
                parceira_rels = {_norm_rel(r["label"]) for r in pont["relations"]}
                ref_rels = {_norm_rel(r["label"]) for r in ontologia["relations"]}
                if ref_rels:
                    overlap_vals.append(len(ref_rels & parceira_rels) / len(ref_rels))

        labels = {t[1] for rec in saida.values() for t in rec["triples"]}
        mapping = proj.mapeamento(labels, onto, ontologia)

        f1o, f1p, conf, enth, vazia, f1r = [], [], [], [], [], []
        todas_triplas = []
        for sid, gt_rec in gab.items():
            if sid not in saida:
                continue
            gt_tr = [[t["sub"], t["rel"], t["obj"]] for t in gt_rec["triples"]]
            sys_tr = saida[sid]["triples"]
            todas_triplas.extend(sys_tr)

            rels_gab = {t[1].replace(" ", "_") for t in gt_tr}
            gt_norm = {normalizar_tripla(*t) for t in gt_tr}
            fil = [t for t in sys_tr if t[1] in rels_gab]
            proj_tr = [[s, mapping[r], o] for s, r, o in sys_tr if r in mapping]
            f1o.append(_f1(gt_norm, fil))
            f1p.append(_f1(gt_norm, proj_tr))

            c, _ = calcular_conformidade_ontologica(ontologia, sys_tr)
            sh, oh = calcular_alucinacao_sujeito_objeto(ps, ontologia, gt_rec["sent"], sys_tr)
            conf.append(c)
            enth.append((sh + oh) / 2.0)
            vazia.append(1.0 if len(sys_tr) == 0 else 0.0)

            if eh_c5:
                gt_restr = [t for t in gt_tr if _norm_rel(t[1]) not in parceira_rels]
                rels_gab_r = {t[1].replace(" ", "_") for t in gt_restr}
                gt_norm_r = {normalizar_tripla(*t) for t in gt_restr}
                fil_r = [t for t in sys_tr if t[1] in rels_gab_r]
                f1r.append(_f1(gt_norm_r, fil_r))

        if not f1o:
            continue
        r = redundancia_semantica(todas_triplas, threshold=redund_thr, definicoes=proj.defs(onto))
        redund_vals.append(r["valor"] if r["valor"] is not None else 0.0)
        d = {"f1o": np.array(f1o), "f1p": np.array(f1p), "conf": np.array(conf),
             "enth": np.array(enth), "vazia": np.array(vazia)}
        if eh_c5:
            d["f1r"] = np.array(f1r)
        por_onto[onto] = d
    return por_onto, np.array(redund_vals), np.array(overlap_vals)


def macro_media(por_onto, chave, somente_nao_vazias=False):
    vals = []
    for d in por_onto.values():
        if chave not in d:
            continue
        v = d[chave]
        if somente_nao_vazias:
            v = v[d["vazia"] == 0.0]
            if len(v) == 0:
                continue
        vals.append(v.mean())
    return float(np.mean(vals)) if vals else float("nan")


def cluster_bootstrap(por_onto, chave, n_boot, rng, somente_nao_vazias=False):
    arrays = []
    for d in por_onto.values():
        if chave not in d:
            continue
        v = d[chave]
        if somente_nao_vazias:
            v = v[d["vazia"] == 0.0]
        arrays.append(v)
    if not arrays:
        return (float("nan"), float("nan"))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        per = [v[rng.integers(0, len(v), size=len(v))].mean() for v in arrays if len(v) > 0]
        boots[b] = np.mean(per) if per else np.nan
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return (float(lo), float(hi))


def boot_simples(vec, n_boot, rng):
    if len(vec) <= 1:
        return (float("nan"), float("nan"))
    b = np.array([rng.choice(vec, size=len(vec), replace=True).mean() for _ in range(n_boot)])
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _c(v, ci, pct=False):
    if v != v:  # nan
        return "—"
    if pct:
        return f"{v*100:.1f}% [{ci[0]*100:.1f}–{ci[1]*100:.1f}]"
    return f"{v:.3f} [{ci[0]:.3f}–{ci[1]:.3f}]"


def tabela_metricas(linhas, ds):
    out = ["| Condição | F1 original | F1 projetado | Conformidade | Conf. sem vazias | % vazias | Redundância | Alucinações |",
           "|---|---|---|---|---|---|---|---|"]
    for l in [x for x in linhas if x["dataset"] == ds]:
        out.append("| " + " | ".join([
            COND_LABEL.get(l["cond"], l["cond"]),
            _c(l["f1_orig"], l["f1_orig_ci"]),
            _c(l["f1_proj"], l["f1_proj_ci"]),
            _c(l["conf"], l["conf_ci"]),
            _c(l["conf_nv"], l["conf_nv_ci"]),
            _c(l["vazias"], l["vazias_ci"], pct=True),
            _c(l["redund"], l["redund_ci"]),
            _c(l["alluc"], l["alluc_ci"]),
        ]) + " |")
    return "\n".join(out)


def tabela_overlap(overlaps):
    out = ["| Dataset | Tipo C5 | Sobreposição ref∩parceira / ref | F1 C5 (todas) | F1 C5 (relações fora da parceira) |",
           "|---|---|---|---|---|"]
    for o in overlaps:
        out.append("| " + " | ".join([
            "TekGen" if o["dataset"] == "tekgen" else "WebNLG (DBpedia)",
            C5_LABEL[o["cond"]],
            f"{o['overlap']*100:.1f}% [{o['overlap_ci'][0]*100:.1f}–{o['overlap_ci'][1]*100:.1f}]",
            _c(o["f1_todas"], o["f1_todas_ci"]),
            _c(o["f1_restr"], o["f1_restr_ci"]),
        ]) + " |")
    return "\n".join(out)


def gerar_markdown(linhas, overlaps, slug, limiar_proj, redund_thr, n_boot, seed, n_frases):
    return f"""# Resultados — Experimento completo Qwen 2.5-7B

**Modelo:** Qwen/Qwen2.5-7B-Instruct-AWQ (AWQ-4bit) · temperatura 0 · seed 42 · vLLM (OpenAI-compat).
**Escopo:** 203 experimentos (manifesto `configs/pipeline/manifesto_7b_completo.json`), {n_frases:,} frases
de teste — TekGen (10 ontologias × 7 condições) e WebNLG (19 ontologias × 7 condições).
**Resultados brutos:** `resultados/full/{slug}/*.jsonl`.
**Avaliação por frase:** `resultados/avaliacao_full/{slug}/` (mesmo formato do piloto-v3, lido pelo app).
**Reprodução:** `python scripts/avaliar_full.py --manifesto configs/pipeline/manifesto_7b_completo.json --slug {slug} --rodada full`.

## Metodologia da avaliação

- **Régua original** (espelho do Text2KGBench, `t2kg.avaliacao.original`): F1 só sobre triplas do
  sistema cuja relação consta do gabarito da frase; normalização sub+rel+obj sem espaços/underscores.
- **F1 projetado** (régua estendida): cada relação livre é mapeada para a relação mais similar da
  ontologia de referência (`all-MiniLM-L6-v2`, modo *label+def*), aceitando quando SimE ≥ **{limiar_proj:.2f}**
  — limiar lido da fonte única `configs/avaliacao.json` (full = 0,45, D-36).
- **Conformidade:** fração das triplas do sistema cuja relação pertence à ontologia de referência.
  Saída vazia conta conformidade = 1 (regra original); por isso **Conf. sem vazias** (só frases com
  ≥1 tripla) é reportada à parte.
- **% vazias:** frações de frases com `triples == []`.
- **Redundância:** redundância semântica de esquema (H1), `1 − n_grupos/n_rótulos`, limiar **{redund_thr:.2f}**.
- **Alucinações:** alucinação de entidade = média de sujeito e objeto fora do contexto (frase + conceitos).
- **Agregação:** macro-média entre ontologias (como no benchmark original).
- **Intervalos [IC 95%]:** *cluster-bootstrap* com **seed {seed}** ({n_boot} reamostragens): reamostram-se
  as frases dentro de cada ontologia e recompõe-se a macro-média; a redundância e a sobreposição
  (nível de esquema) usam bootstrap sobre as ontologias.

## TekGen (10 ontologias)

{tabela_metricas(linhas, 'tekgen')}

## WebNLG (19 ontologias)

{tabela_metricas(linhas, 'webnlg')}

## Sobreposição da C5 (interpretação da H3)

A C5 troca a ontologia-alvo pela de uma parceira (vizinha ou incongruente). Parte do desempenho
residual da C5 pode vir de **relações que a parceira compartilha com a referência**. A tabela abaixo
mostra, por dataset e tipo de C5, a sobreposição média (fração das relações da referência que também
existem na parceira) e o F1 da C5 recalculado **apenas sobre triplas cujas relações do gabarito NÃO
estão na parceira** — isolando o sinal que não pode ser explicado por sobreposição de esquema.

{tabela_overlap(overlaps)}

Ver nota de interpretação da **H3** em `docs/DECISOES.md`.

## Leitura rápida

- **C1 one-shot original** lidera em F1 nos dois datasets, com alta conformidade e pouquíssimas vazias.
- **C0 e C2** têm conformidade baixa (vocabulário livre / induzido): o F1 **projetado** recupera parte
  do sinal, confirmando triplas corretas com rótulos fora da ontologia-alvo.
- **C5 incongruente** colapsa em TekGen (F1 ≈ 0; maioria das respostas vazias); em WebNLG o colapso é
  parcial, coerente com ontologias-alvo mais próximas e maior sobreposição de esquema (ver H3).
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifesto", default=os.path.join(REPO, "configs", "pipeline", "manifesto_7b_completo.json"))
    p.add_argument("--slug", default="qwen2.5-7b")
    p.add_argument("--rodada", default="full", help="Chave do limiar de projeção em configs/avaliacao.json.")
    p.add_argument("--avaliacao_dir", default=None)
    p.add_argument("--doc", default=os.path.join(REPO, "docs", "resultados_7b_completo.md"))
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.chdir(REPO)  # saidas do manifesto são relativas à raiz do repo
    avaliacao_dir = args.avaliacao_dir or os.path.join(REPO, "resultados", "avaliacao_full", args.slug)
    os.makedirs(avaliacao_dir, exist_ok=True)
    limiar_proj = limiar_projecao(args.rodada)
    redund_thr = limiar_redundancia()
    def_dir = os.path.join(REPO, "configs", "c2", "v_full", args.slug)
    proj = Projetor(def_dir, limiar_proj)

    with open(args.manifesto, encoding="utf-8") as f:
        manifest = json.load(f)
    grupos = {}
    for e in manifest:
        grupos.setdefault((e["dataset"], e["condicao"]), []).append(e)

    n_frases = 0
    linhas, overlaps = [], []
    for ds in ["tekgen", "webnlg"]:
        for cond in COND_ORDER:
            exps = grupos.get((ds, cond))
            if not exps:
                continue
            print(f"[{ds}/{cond}] régua original -> {avaliacao_dir}", file=sys.stderr)
            # 1) Régua original: grava por frase + médias (formato do app)
            ar.avaliar_condicao(exps, BENCH, avaliacao_dir, cond, ds, modo_amostra=False)
            # 2) Passo de precisão plena para a tabela + bootstrap
            por_onto, redund, overlap = avaliar_cell(exps, ds, cond, proj, redund_thr)
            for d in por_onto.values():
                n_frases += len(d["f1o"])
            rng = np.random.default_rng(args.seed)
            row = {"dataset": ds, "cond": cond}
            for chave, nome in [("f1o", "f1_orig"), ("f1p", "f1_proj"), ("conf", "conf"), ("enth", "alluc")]:
                row[nome] = macro_media(por_onto, chave)
                row[nome + "_ci"] = cluster_bootstrap(por_onto, chave, args.n_boot, rng)
            row["conf_nv"] = macro_media(por_onto, "conf", somente_nao_vazias=True)
            row["conf_nv_ci"] = cluster_bootstrap(por_onto, "conf", args.n_boot, rng, somente_nao_vazias=True)
            row["vazias"] = macro_media(por_onto, "vazia")
            row["vazias_ci"] = cluster_bootstrap(por_onto, "vazia", args.n_boot, rng)
            row["redund"] = float(np.mean(redund)) if len(redund) else float("nan")
            row["redund_ci"] = boot_simples(redund, args.n_boot, rng)
            linhas.append(row)
            print(f"   F1o={row['f1_orig']:.3f} F1p={row['f1_proj']:.3f} conf={row['conf']:.3f} "
                  f"conf_nv={row['conf_nv']:.3f} vazias={row['vazias']:.3f} redund={row['redund']:.3f}",
                  file=sys.stderr)
            if cond.startswith("C5"):
                overlaps.append({
                    "dataset": ds, "cond": cond,
                    "overlap": float(np.mean(overlap)) if len(overlap) else float("nan"),
                    "overlap_ci": boot_simples(overlap, args.n_boot, rng),
                    "f1_todas": row["f1_orig"], "f1_todas_ci": row["f1_orig_ci"],
                    "f1_restr": macro_media(por_onto, "f1r"),
                    "f1_restr_ci": cluster_bootstrap(por_onto, "f1r", args.n_boot, rng),
                })

    md = gerar_markdown(linhas, overlaps, args.slug, limiar_proj, redund_thr,
                        args.n_boot, args.seed, n_frases)
    os.makedirs(os.path.dirname(args.doc), exist_ok=True)
    with open(args.doc, "w", encoding="utf-8") as f:
        f.write(md)
    # resumo-máquina ao lado da avaliação (útil p/ conferência e app)
    with open(os.path.join(avaliacao_dir, "tabela_resumo.json"), "w", encoding="utf-8") as f:
        json.dump({"linhas": linhas, "overlaps": overlaps,
                   "limiar_projecao": limiar_proj, "limiar_redundancia": redund_thr,
                   "n_boot": args.n_boot, "seed": args.seed, "n_frases": n_frases},
                  f, ensure_ascii=False, indent=2)
    print(f"\nDocumento: {args.doc}\nAvaliação por frase: {avaliacao_dir}\nFrases avaliadas: {n_frases}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
