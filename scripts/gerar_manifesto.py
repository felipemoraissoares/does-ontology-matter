"""
Gera o manifesto JSON completo para os experimentos da Fase 2.

Produz dois arquivos em configs/pipeline/:
  manifesto_7b_full.json        — experimento principal (29 onto × 6 cond = 174 entradas)
  manifesto_7b_sensibilidade.json — análise de sensibilidade (5 onto × C5_incongruente_2)

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

# ---------------------------------------------------------------------------
# Configuração fixa
# ---------------------------------------------------------------------------
CONDICOES_PRINCIPAIS = ["C0", "C1", "C1_oneshot_original", "C3", "C5_vizinho", "C5_incongruente"]

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
            ["D:\\Programas\\Git\\cmd\\git.exe", "rev-parse", "HEAD"],
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
) -> dict:
    e = {
        "id": exp_id,
        "dataset": dataset,
        "split": "test",
        "ontologia_id": onto_id,
        "condicao": condicao,
        "modelo": modelo,
        "quantizacao": "AWQ-4bit",
        "chat_template": "qwen2",
        "provedor": "openai_compat",
        "provedor_url": url,
        "api_key": "token-nao-usado",
        "temperatura": 0,
        "seed": 42,
        "max_tokens": 512,
        "git_hash": git,
        "saida": os.path.join(saida_base, f"{exp_id}.jsonl"),
    }
    if onto_parceira_id is not None:
        e["ontologia_parceira_id"] = onto_parceira_id
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
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    p.add_argument("--url", default="http://localhost:8000/v1")
    p.add_argument("--saida_dir", default=os.path.join(
        os.path.dirname(__file__), "..", "configs", "pipeline"))
    args = p.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    saida_resultados = os.path.join(root, "resultados", "7b")

    pares_path = os.path.join(root, "configs", "pipeline", "pares_c5.json")
    with open(pares_path, encoding="utf-8") as f:
        pares = json.load(f)

    git = git_hash()
    print(f"git_hash: {git}")
    print(f"modelo:   {args.modelo}")
    print(f"url:      {args.url}")

    exp_full = gerar_principal(
        TEKGEN_ONTOS, WEBNLG_ONTOS, pares, args.modelo, args.url, saida_resultados, git
    )
    exp_sens = gerar_sensibilidade(pares, args.modelo, args.url, saida_resultados, git)

    full_path = os.path.join(args.saida_dir, "manifesto_7b_full.json")
    sens_path = os.path.join(args.saida_dir, "manifesto_7b_sensibilidade.json")

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(exp_full, f, ensure_ascii=False, indent=2)
    with open(sens_path, "w", encoding="utf-8") as f:
        json.dump(exp_sens, f, ensure_ascii=False, indent=2)

    print(f"\nManifesto principal:    {full_path}  ({len(exp_full)} experimentos)")
    print(f"Manifesto sensibilidade: {sens_path}  ({len(exp_sens)} experimentos)")
    print(f"Total de experimentos:   {len(exp_full) + len(exp_sens)}")
