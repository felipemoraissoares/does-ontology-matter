"""
Gera os manifestos JSON para os experimentos da Fase 2.

Produz arquivos em configs/pipeline/:
  manifesto_7b_full.json         — experimento principal (29 onto × 6 cond = 174 entradas)
  manifesto_7b_sensibilidade.json — análise de sensibilidade (5 onto × C5_incongruente_2)
  manifesto_piloto_kaggle.json   — piloto Kaggle (2 TekGen × 3 cond, 200 frases, seed=42)

Uso:
  python scripts/gerar_manifesto.py [--modelo MODELO] [--url URL] [--saida_dir DIR]

Defaults:
  --modelo  Qwen/Qwen2.5-7B-Instruct-AWQ
  --url     http://localhost:8000/v1
  --saida_dir  configs/pipeline
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from t2kg.modelos import slug_modelo  # noqa: E402

TODAS_CONDICOES = [
    "C0", "C1", "C1_oneshot_original", "C2", "C3", "C5_vizinho", "C5_incongruente",
]

# ---------------------------------------------------------------------------
# Configuração fixa
# ---------------------------------------------------------------------------
CONDICOES_PRINCIPAIS = ["C0", "C1", "C1_oneshot_original", "C3", "C5_vizinho", "C5_incongruente"]
CONDICOES_PILOTO = ["C0", "C1", "C1_oneshot_original"]
PILOTO_ONTOS_TEKGEN = ["1_movie", "2_music"]
PILOTO_FRASES_MAX = 200
PILOTO_FRASES_SEED = 42

# Commit fixo do Text2KGBench para reprodutibilidade
BENCH_URL    = "https://github.com/cenguix/Text2KGBench.git"
BENCH_COMMIT = "50a3d255371b8817cdff70fd88459ac82b339cfe"

TEKGEN_ONTOS = [
    "1_movie", "2_music", "3_sport", "4_book", "5_military",
    "6_computer", "7_space", "8_politics", "9_nature", "10_culture",
]
WEBNLG_ONTOS = [
    "1_university", "2_musicalwork", "3_airport", "4_building", "5_athlete",
    "6_politician", "7_company", "8_celestialbody", "9_astronaut", "10_comicscharacter",
    "11_meanoftransportation", "12_monument", "13_food", "14_writtenwork", "15_sportsteam",
    "16_city", "17_artist", "18_scientist", "19_film",
]

# Ontologias para a análise de sensibilidade (D-10)
SENSIBILIDADE_ONTOS = {
    "webnlg": ["3_airport", "5_athlete", "16_city"],
    "tekgen": ["1_movie", "5_military"],
}


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "DESCONHECIDO"


def entrada(
    *,
    exp_id: str,
    dataset: str,
    onto_id: str,
    condicao: str,
    modelo: str,
    url: str,
    saida_base: str,
    git: str,
    onto_parceira_id: str = None,
    c2_subdir: str = None,
    quantizacao: str = "AWQ-4bit",
    chat_template: str = "qwen2",
) -> dict:
    # Sempre usa "/" para garantir portabilidade Linux/Windows
    saida = f"{saida_base}/{exp_id}.jsonl"
    e = {
        "id": exp_id,
        "dataset": dataset,
        "split": "test",
        "ontologia_id": onto_id,
        "condicao": condicao,
        "modelo": modelo,
        "quantizacao": quantizacao,
        "chat_template": chat_template,
        "provedor": "openai_compat",
        "provedor_url": url,
        "api_key": "token-nao-usado",
        "temperatura": 0,
        "seed": 42,
        "max_tokens": 512,
        "workers": 1,
        "git_hash": git,
        "bench_url": BENCH_URL,
        "bench_commit": BENCH_COMMIT,
        "saida": saida,
    }
    if onto_parceira_id is not None:
        e["ontologia_parceira_id"] = onto_parceira_id
    if c2_subdir is not None:
        e["c2_subdir"] = c2_subdir
    return e


def entrada_piloto(
    *,
    exp_id: str,
    dataset: str,
    onto_id: str,
    condicao: str,
    modelo: str,
    url: str,
    saida_base: str,
    git: str,
    frases_max: int,
    frases_seed: int,
) -> dict:
    e = entrada(
        exp_id=exp_id,
        dataset=dataset,
        onto_id=onto_id,
        condicao=condicao,
        modelo=modelo,
        url=url,
        saida_base=saida_base,
        git=git,
    )
    e["frases_max"] = frases_max
    e["frases_seed"] = frases_seed
    e["modo_amostra"] = True
    return e


def gerar_principal(ontos_tkg, ontos_web, pares, modelo, url, saida_base, git):
    exp = []

    def _slug(modelo):
        return modelo.split("/")[-1].lower().replace("-", "_").replace(".", "")

    slug = _slug(modelo)

    for onto_id in ontos_tkg:
        for cond in CONDICOES_PRINCIPAIS:
            parceira = None
            if cond == "C5_vizinho":
                parceira = pares["tekgen"][onto_id]["vizinho"]["id"]
            elif cond == "C5_incongruente":
                parceira = pares["tekgen"][onto_id]["incongruente"]["id"]
            cond_slug = cond.lower()
            exp_id = f"tkg_{onto_id}_{cond_slug}_{slug}"
            exp.append(entrada(
                exp_id=exp_id,
                dataset="tekgen",
                onto_id=onto_id,
                condicao=cond,
                modelo=modelo,
                url=url,
                saida_base=saida_base,
                git=git,
                onto_parceira_id=parceira,
            ))

    for onto_id in ontos_web:
        for cond in CONDICOES_PRINCIPAIS:
            parceira = None
            if cond == "C5_vizinho":
                parceira = pares["webnlg"][onto_id]["vizinho"]["id"]
            elif cond == "C5_incongruente":
                parceira = pares["webnlg"][onto_id]["incongruente"]["id"]
            cond_slug = cond.lower()
            exp_id = f"web_{onto_id}_{cond_slug}_{slug}"
            exp.append(entrada(
                exp_id=exp_id,
                dataset="webnlg",
                onto_id=onto_id,
                condicao=cond,
                modelo=modelo,
                url=url,
                saida_base=saida_base,
                git=git,
                onto_parceira_id=parceira,
            ))

    return exp


def gerar_piloto(modelo, url, saida_base, git):
    exp = []

    def _slug(modelo):
        return modelo.split("/")[-1].lower().replace("-", "_").replace(".", "")

    slug = _slug(modelo)

    for onto_id in PILOTO_ONTOS_TEKGEN:
        for cond in CONDICOES_PILOTO:
            cond_slug = cond.lower()
            exp_id = f"piloto_tkg_{onto_id}_{cond_slug}_{slug}"
            exp.append(entrada_piloto(
                exp_id=exp_id,
                dataset="tekgen",
                onto_id=onto_id,
                condicao=cond,
                modelo=modelo,
                url=url,
                saida_base=saida_base,
                git=git,
                frases_max=PILOTO_FRASES_MAX,
                frases_seed=PILOTO_FRASES_SEED,
            ))

    return exp


def gerar_sensibilidade(pares, modelo, url, saida_base, git):
    exp = []

    def _slug(modelo):
        return modelo.split("/")[-1].lower().replace("-", "_").replace(".", "")

    slug = _slug(modelo)

    for dataset, ontos in SENSIBILIDADE_ONTOS.items():
        ds_prefix = "tkg" if dataset == "tekgen" else "web"
        for onto_id in ontos:
            inc2 = pares[dataset][onto_id].get("incongruente_2")
            if inc2 is None:
                continue
            parceira_id = inc2["id"]
            exp_id = f"{ds_prefix}_{onto_id}_c5_incongruente_2_{slug}"
            exp.append(entrada(
                exp_id=exp_id,
                dataset=dataset,
                onto_id=onto_id,
                condicao="C5_incongruente",
                modelo=modelo,
                url=url,
                saida_base=saida_base,
                git=git,
                onto_parceira_id=parceira_id,
            ))

    return exp


# ---------------------------------------------------------------------------
# Experimento completo (Etapa B) — sem amostragem, 29 ontologias, 7 condições
# ---------------------------------------------------------------------------

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH_DIR_PADRAO = os.path.abspath(os.path.join(REPO, "..", "Text2KGBench"))
C2_DIR = os.path.join(REPO, "configs", "c2")


def _gt_path(bench_dir, dataset, onto):
    if dataset == "tekgen":
        return os.path.join(bench_dir, "data", "wikidata_tekgen", "ground_truth",
                            f"ont_{onto}_ground_truth.jsonl")
    return os.path.join(bench_dir, "data", "dbpedia_webnlg", "ground_truth",
                        f"ont_{onto}_ground_truth.jsonl")


def contar_frases_teste(bench_dir, dataset, onto):
    p = _gt_path(bench_dir, dataset, onto)
    if not os.path.exists(p):
        return 0
    with open(p, encoding="utf-8") as f:
        return sum(1 for l in f if l.strip())


def _parceira(pares, dataset, onto, cond):
    if cond == "C5_vizinho":
        return pares[dataset][onto]["vizinho"]["id"]
    if cond == "C5_incongruente":
        return pares[dataset][onto]["incongruente"]["id"]
    return None


def gerar_completo(modelos, condicoes, datasets, pares, url, git,
                   c2_base="v_full", exigir_c2=False):
    """Monta o manifesto do experimento completo. Retorna (experimentos, avisos).

    C-5/D-33: a condição C2 usa a ontologia induzida do PRÓPRIO modelo em
    configs/c2/<c2_base>/<slug>/. Se o arquivo não existir, registra aviso; com
    exigir_c2=True a entrada C2 é OMITIDA (modo estrito), senão é incluída no plano
    (o guard check do lançador bloqueia a execução até a indução existir).
    """
    exp, avisos = [], []
    ontos_por_ds = {"tekgen": TEKGEN_ONTOS, "webnlg": WEBNLG_ONTOS}
    for modelo in modelos:
        slug = slug_modelo(modelo)
        saida_base = f"resultados/full/{slug}"
        for dataset in datasets:
            ds_prefix = "tkg" if dataset == "tekgen" else "web"
            for onto in ontos_por_ds[dataset]:
                for cond in condicoes:
                    exp_id = f"{ds_prefix}_{onto}_{cond.lower()}_{slug}"
                    c2_subdir = None
                    if cond == "C2":
                        c2_subdir = f"{c2_base}/{slug}"
                        induced = os.path.join(
                            C2_DIR, c2_base, slug, f"{dataset}_{onto}_induced.json")
                        if not os.path.exists(induced):
                            avisos.append(f"{dataset}/{onto} ({slug})")
                            if exigir_c2:
                                continue
                    exp.append(entrada(
                        exp_id=exp_id, dataset=dataset, onto_id=onto, condicao=cond,
                        modelo=modelo, url=url, saida_base=saida_base, git=git,
                        onto_parceira_id=_parceira(pares, dataset, onto, cond),
                        c2_subdir=c2_subdir,
                    ))
    return exp, avisos


def dividir_sessoes(experimentos, bench_dir, horas_max, fr_por_hora, fator):
    """Divide a lista em partes de <= horas_max, estimando por frases de teste."""
    cache = {}

    def horas(e):
        chave = (e["dataset"], e["ontologia_id"])
        if chave not in cache:
            cache[chave] = contar_frases_teste(bench_dir, *chave)
        return cache[chave] * fator / fr_por_hora

    partes, atual, h_atual = [], [], 0.0
    for e in experimentos:
        h = horas(e)
        if atual and h_atual + h > horas_max:
            partes.append((atual, h_atual))
            atual, h_atual = [], 0.0
        atual.append(e)
        h_atual += h
    if atual:
        partes.append((atual, h_atual))
    return partes


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Gera o manifesto do experimento completo.")
    p.add_argument("--modelos", default="Qwen/Qwen2.5-7B-Instruct-AWQ",
                   help="Ids de modelo separados por vírgula.")
    p.add_argument("--condicoes", default=",".join(TODAS_CONDICOES),
                   help="Condições separadas por vírgula.")
    p.add_argument("--datasets", default="tekgen,webnlg")
    p.add_argument("--url", default="http://localhost:8000/v1")
    p.add_argument("--saida", default=os.path.join(
        REPO, "configs", "pipeline", "manifesto_completo.json"))
    p.add_argument("--bench_dir", default=BENCH_DIR_PADRAO)
    p.add_argument("--exigir_c2_induzido", action="store_true",
                   help="C-5 estrito: omite C2 cuja ontologia induzida do modelo não existe.")
    p.add_argument("--dividir_por_sessao", action="store_true",
                   help="Também grava partes de <= --horas_max h (sessões Kaggle de 12 h).")
    p.add_argument("--horas_max", type=float, default=10.0)
    p.add_argument("--fr_por_hora", type=float, default=5300.0,
                   help="Throughput de referência (7B ~5300 fr/h).")
    p.add_argument("--fator", type=float, default=1.0,
                   help="Fator de lentidão do modelo (>1 = mais lento). 7B=1.0.")
    args = p.parse_args()

    modelos = [m.strip() for m in args.modelos.split(",") if m.strip()]
    condicoes = [c.strip() for c in args.condicoes.split(",") if c.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    desconhecidas = [c for c in condicoes if c not in TODAS_CONDICOES]
    if desconhecidas:
        print(f"ERRO: condições desconhecidas: {desconhecidas}", file=sys.stderr)
        sys.exit(1)

    with open(os.path.join(REPO, "configs", "pipeline", "pares_c5.json"), encoding="utf-8") as f:
        pares = json.load(f)

    git = git_hash()
    print(f"git_hash:  {git}")
    print(f"modelos:   {modelos}")
    print(f"condicoes: {condicoes}")
    print(f"datasets:  {datasets}")

    exp, avisos = gerar_completo(
        modelos, condicoes, datasets, pares, args.url, git,
        exigir_c2=args.exigir_c2_induzido,
    )

    os.makedirs(os.path.dirname(args.saida), exist_ok=True)
    with open(args.saida, "w", encoding="utf-8") as f:
        json.dump(exp, f, ensure_ascii=False, indent=2)
    print(f"\nManifesto: {args.saida}  ({len(exp)} experimentos)")

    total_frases = sum(
        contar_frases_teste(args.bench_dir, e["dataset"], e["ontologia_id"]) for e in exp)
    total_h = total_frases * args.fator / args.fr_por_hora
    print(f"Frases totais: {total_frases}  |  Horas estimadas: {total_h:.1f} h "
          f"(@ {args.fr_por_hora:.0f} fr/h, fator {args.fator})")

    if avisos:
        modo = "OMITIDAS" if args.exigir_c2_induzido else "incluídas no plano"
        print(f"\n{len(avisos)} condição(ões) C2 sem ontologia induzida ({modo}):")
        for a in avisos[:8]:
            print(f"  - {a}")
        if len(avisos) > 8:
            print(f"  ... e mais {len(avisos) - 8}.")
        print("  Rode a indução C2 no Kaggle (Etapa C) antes de executar a C2; o guard")
        print("  check do lançador bloqueia a C2 sem a ontologia induzida do modelo.")

    if args.dividir_por_sessao:
        partes = dividir_sessoes(exp, args.bench_dir, args.horas_max,
                                 args.fr_por_hora, args.fator)
        stem = os.path.splitext(args.saida)[0]
        print(f"\nDividido em {len(partes)} parte(s) de <= {args.horas_max} h:")
        for i, (parte, h) in enumerate(partes, 1):
            pp = f"{stem}_parte{i}.json"
            with open(pp, "w", encoding="utf-8") as f:
                json.dump(parte, f, ensure_ascii=False, indent=2)
            print(f"  parte {i}: {len(parte):3d} exp  ~{h:4.1f} h  -> {os.path.basename(pp)}")
