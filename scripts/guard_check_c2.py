"""Guard check da C2 do experimento completo (Etapa C, D-33/D-34).

Verifica, para cada experimento C2 do manifesto, que existem no repositório:
  - a ontologia induzida do modelo:     configs/c2/<c2_subdir>/<dataset>_<onto>_induced.json
  - as definições D-24 do modelo:       configs/c2/<c2_subdir>/definicoes_<onto>.json

Falha (exit 1) listando o que falta. Chamado pelo lançador antes de executar a C2 — evita
rodar a C2 sem a ontologia induzida daquele modelo (contaminação silenciosa).

Uso:
  python scripts/guard_check_c2.py --manifesto configs/pipeline/manifesto_7b_completo.json
"""
import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _experimentos(data):
    if isinstance(data, dict):
        return data.get("experimentos", [])
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifesto", required=True)
    p.add_argument("--repo", default=REPO, help="Raiz do repositório (padrão: deduzido).")
    args = p.parse_args()

    with open(args.manifesto, encoding="utf-8") as f:
        exps = _experimentos(json.load(f))

    c2 = [e for e in exps if e.get("condicao") == "C2"]
    print(f"Guard check C2: {len(c2)} experimento(s) C2 no manifesto.")

    erros = []
    for e in c2:
        subdir = e.get("c2_subdir")
        if not subdir:
            erros.append(f"{e['id']}: experimento C2 sem campo 'c2_subdir'.")
            continue
        base = os.path.join(args.repo, "configs", "c2", *subdir.split("/"))
        induced = os.path.join(base, f"{e['dataset']}_{e['ontologia_id']}_induced.json")
        defs = os.path.join(base, f"definicoes_{e['ontologia_id']}.json")
        if not os.path.exists(induced):
            erros.append(f"{e['id']}: ontologia induzida ausente -> "
                         f"{os.path.relpath(induced, args.repo)}")
        if not os.path.exists(defs):
            erros.append(f"{e['id']}: definições D-24 ausentes -> "
                         f"{os.path.relpath(defs, args.repo)}")

    if erros:
        print(f"\nGuard check FALHOU — {len(erros)} pendência(s):", file=sys.stderr)
        for msg in erros[:20]:
            print(f"  * {msg}", file=sys.stderr)
        if len(erros) > 20:
            print(f"  ... e mais {len(erros) - 20}.", file=sys.stderr)
        print("\nRode a indução C2 no Kaggle (scripts/induzir_c2_lote.py) antes de "
              "executar a condição C2.", file=sys.stderr)
        sys.exit(1)

    print(f"Guard check OK: ontologia induzida e definições presentes para os {len(c2)} C2.")


if __name__ == "__main__":
    main()
