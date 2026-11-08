# Espelho fiel do avaliador Text2KGBench (run_eval.py)
# Código base: Copyright (c) 2023 Nandana Mihindukulasooriya et al.
# Licença original: Apache 2.0 (https://github.com/nandana/Text2KGBench)
# Adaptado por Felipe Morais Soares, ITA, 2026.
#
# REGRAS DE ESPELHAMENTO (não alterar sem motivo documentado):
#   1. P/R/F1 calculados só sobre triplas do sistema cuja relação
#      aparece nas relações do gabarito daquela frase.
#   2. Normalização: remove espaços e _, minúsculas, concatena sub+rel+obj.
#   3. Conformidade: compara tr[1] com labels das relações da ontologia
#      (espaços → _), case-sensitive, sobre TODAS as triplas do sistema.
#   4. Alucinação de sujeito/objeto: verifica se entidade (stemmed+normalizada)
#      é substring de (frase + labels dos conceitos). Concatenação SEM espaço
#      entre a frase e o primeiro conceito.
#   5. Saída vazia → P=R=F1=0, conformidade=1, alucinações=0.
#   6. Médias por ontologia dividem pelo total de frases do gabarito.
#   7. Média global = média simples das médias por ontologia.
#   8. DBpedia: onto_list pode repetir "1_university" → soma 20 entradas.

import json
import os
import re
from typing import List, Dict, Set, Tuple

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# ---------------------------------------------------------------------------
# Funções de cálculo de métricas (espelho direto do run_eval.py original)
# ---------------------------------------------------------------------------

def calcular_precisao_recall_f1(gabarito: Set, predicao: Set) -> Tuple[float, float, float]:
    """P = corretas/preditas, R = corretas/gabarito, F1 = média harmônica."""
    if len(predicao) == 0:
        return 0.0, 0.0, 0.0
    p = len(gabarito.intersection(predicao)) / len(predicao)
    r = len(gabarito.intersection(predicao)) / len(gabarito)
    f1 = 2 * ((p * r) / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def normalizar_tripla(sub: str, rel: str, obj: str) -> str:
    """Remove espaços e _ e converte para minúsculas; concatena sub+rel+obj."""
    sub = re.sub(r"(_|\s+)", '', sub).lower()
    rel = re.sub(r"(_|\s+)", '', rel).lower()
    obj = re.sub(r"(_|\s+)", '', obj).lower()
    return f"{sub}{rel}{obj}"


def limpar_entidade(ps: PorterStemmer, entidade: str) -> str:
    """Stemmiza e normaliza uma string de sujeito/objeto."""
    stemmed = "".join([ps.stem(w) for w in word_tokenize(entidade)])
    normalizado = re.sub(r"(_|\s+)", '', stemmed).lower()
    # tratamento especial: remove "01janeiro" (datas no formato Wikidata)
    return normalizado.replace("01januari", "")


def calcular_alucinacao_sujeito_objeto(
    ps: PorterStemmer,
    ontologia: Dict,
    frase: str,
    triplas: List,
) -> Tuple[float, float]:
    """
    Verifica se sujeito/objeto de cada tripla aparecem como substring de
    (frase + conceitos da ontologia). A concatenação é feita SEM espaço
    entre a frase e o primeiro conceito — espelho exato do original.
    """
    if len(triplas) == 0:
        return 0.0, 0.0

    # concatenação direta: frase + " ".join(labels dos conceitos)
    # o original usa: test_sentence += " ".join(...)
    contexto = frase + " ".join([c["label"] for c in ontologia["concepts"]])
    stemmed_ctx = "".join([ps.stem(w) for w in word_tokenize(contexto)])
    ctx_norm = re.sub(r"(_|\s+)", '', stemmed_ctx).lower()

    num_sub, num_obj = 0, 0
    for tr in triplas:
        sub_norm = limpar_entidade(ps, tr[0])
        obj_norm = limpar_entidade(ps, tr[2])
        if ctx_norm.find(sub_norm) == -1:
            num_sub += 1
        if ctx_norm.find(obj_norm) == -1:
            num_obj += 1

    return num_sub / len(triplas), num_obj / len(triplas)


def calcular_taxa_erro_formato(ontologia: Dict, triplas: List) -> float:
    """
    Taxa de erro de formato: fração de triplas cujo sujeito OU objeto contém
    o nome de uma classe da ontologia como prefixo ou valor literal.

    Padrões detectados (case-insensitive sobre o label):
      1. Entidade é exatamente o label do conceito, ex.: "film"
      2. Entidade começa com "{label}:", ex.: "film:Bleach: Hell Verse"
      3. Entidade começa com "{label} ", ex.: 'film "Keyboard Cat"'

    Triplas vazias retornam 0.0.
    """
    if not triplas:
        return 0.0
    classes = {c["label"].lower() for c in ontologia.get("concepts", [])}
    erros = 0
    for tr in triplas:
        tripla_tem_erro = False
        for entidade in (tr[0], tr[2]):
            e = entidade.lower().strip()
            for cls in classes:
                if e == cls or e.startswith(cls + ":") or e.startswith(cls + " "):
                    tripla_tem_erro = True
                    break
            if tripla_tem_erro:
                break
        if tripla_tem_erro:
            erros += 1
    return erros / len(triplas)


def calcular_conformidade_ontologica(ontologia: Dict, triplas: List) -> Tuple[float, float]:
    """
    Conformidade = fração das triplas do sistema cuja relação (tr[1]) está
    nos labels de relação da ontologia (espaços → _), case-sensitive.
    Alucinação de relação = 1 − conformidade.
    """
    if len(triplas) == 0:
        return 1.0, 0.0
    rels_ont = [r["label"].replace(" ", "_") for r in ontologia["relations"]]
    conformes = len([tr for tr in triplas if tr[1] in rels_ont])
    conformidade = conformes / len(triplas)
    return conformidade, 1.0 - conformidade


# ---------------------------------------------------------------------------
# Utilitários de I/O
# ---------------------------------------------------------------------------

def ler_jsonl(caminho: str, como_json: bool = True) -> List:
    dados = []
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            dados.append(json.loads(linha) if como_json else linha)
    return dados


def ler_json(caminho: str) -> Dict:
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def salvar_jsonl(dados: List, caminho: str) -> None:
    os.makedirs(os.path.dirname(caminho), exist_ok=True) if os.path.dirname(caminho) else None
    with open(caminho, "w", encoding="utf-8") as f:
        for item in dados:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def acrescentar_jsonl(dado: Dict, caminho: str) -> None:
    os.makedirs(os.path.dirname(caminho), exist_ok=True) if os.path.dirname(caminho) else None
    with open(caminho, "a+", encoding="utf-8") as f:
        f.write(json.dumps(dado, ensure_ascii=False) + "\n")


def para_dicionario(dados: List[Dict], chave: str = "id") -> Dict:
    return {item[chave]: item for item in dados}


def carregar_config(caminho_config: str) -> Dict:
    """
    Lê o JSON de configuração e expande os path_patterns substituindo
    $$onto$$ pelo id de cada ontologia.
    """
    config_raw = ler_json(caminho_config)
    onto_list = config_raw["onto_list"]
    padroes = config_raw["path_patterns"]
    nova_config = {"onto_list": [], "avg_out_file": config_raw["avg_out_file"]}
    for onto in onto_list:
        onto_data = {"id": onto}
        for chave in padroes:
            onto_data[chave] = padroes[chave].replace("$$onto$$", onto)
        nova_config["onto_list"].append(onto_data)
    return nova_config


# ---------------------------------------------------------------------------
# Função principal de avaliação
# ---------------------------------------------------------------------------

def avaliar(config: Dict, diretorio_base: str = "", modo_amostra: bool = False) -> Dict:
    """
    Executa a avaliação completa para todas as ontologias definidas na config.

    Parâmetros
    ----------
    config : Dict
        Configuração já expandida (saída de carregar_config) ou dicionário
        equivalente com chaves 'onto_list' e 'avg_out_file'.
    diretorio_base : str
        Prefixo adicionado a todos os caminhos relativos.
    modo_amostra : bool
        False (padrão): denominador = len(gabarito) — idêntico ao benchmark publicado.
        True: denominador = frases efetivamente na saída do sistema — correto para amostras
              parciais (frases_max < total). Usar sempre que a saída não cubrir o GT completo.

    Retorna
    -------
    Dict com métricas globais e por ontologia.
    """
    ps = PorterStemmer()

    # Remove arquivo de média global anterior para evitar acúmulo
    avg_path = os.path.join(diretorio_base, config["avg_out_file"])
    if os.path.exists(avg_path):
        os.remove(avg_path)

    gp = gr = gf1 = gc = grh = gsh = goh = gef = 0.0
    resultados_por_onto = {}

    for onto in config["onto_list"]:
        onto_id = onto["id"]

        sys_path = os.path.join(diretorio_base, onto["sys"])
        gt_path = os.path.join(diretorio_base, onto["gt"])
        onto_path = os.path.join(diretorio_base, onto["onto"])
        output_path = os.path.join(diretorio_base, onto["output"])

        saida_sistema = para_dicionario(ler_jsonl(sys_path))
        gabarito = para_dicionario(ler_jsonl(gt_path))
        ontologia = ler_json(onto_path)

        ids_selecionados = []
        if "selected_ids" in onto:
            sid_path = os.path.join(diretorio_base, onto["selected_ids"])
            if os.path.exists(sid_path):
                ids_selecionados = ler_jsonl(sid_path, como_json=False)

        tp = tr_ = tf1 = tc = trh = tsh = toh = tef = 0.0
        stp = str_ = stf1 = stc = strh = stsh = stoh = stef = 0.0
        metricas_por_frase = []

        for sent_id in gabarito:
            gt_triplas = [[t["sub"], t["rel"], t["obj"]] for t in gabarito[sent_id]["triples"]]
            frase = gabarito[sent_id]["sent"]

            # Espelha o original: só acumula métricas se a sentença está na saída do sistema.
            # Sentenças ausentes contribuem 0 para todos os acumuladores (by omission),
            # mas o divisor é sempre total_test_cases (len(gabarito)).
            # EXCEÇÃO: se a sentença está presente com triples=[], os helpers
            # calcular_* retornam (0,0,0) para P/R/F1, (1,0) para conformidade, (0,0) para alucinações.
            if sent_id not in saida_sistema:
                # não acumula nada — espelha o comportamento do run_eval.py original
                continue

            sys_triplas = saida_sistema[sent_id]["triples"]

            # relações do gabarito (espaços → _)
            rels_gabarito = {t[1].replace(" ", "_") for t in gt_triplas}

            # filtra triplas do sistema para cálculo de P/R/F1
            fil_triplas = [t for t in sys_triplas if t[1] in rels_gabarito]

            sys_norm = {normalizar_tripla(t[0], t[1], t[2]) for t in fil_triplas}
            gt_norm = {normalizar_tripla(t[0], t[1], t[2]) for t in gt_triplas}

            precision, recall, f1 = calcular_precisao_recall_f1(gt_norm, sys_norm)
            conformidade, rel_hal = calcular_conformidade_ontologica(ontologia, sys_triplas)
            sub_hal, obj_hal = calcular_alucinacao_sujeito_objeto(ps, ontologia, frase, sys_triplas)
            erro_formato = calcular_taxa_erro_formato(ontologia, sys_triplas)

            metricas_por_frase.append({
                "id": sent_id,
                "precision": f"{precision:.2f}",
                "recall": f"{recall:.2f}",
                "f1": f"{f1:.2f}",
                "onto_conf": f"{conformidade:.2f}",
                "rel_halluc": f"{rel_hal:.2f}",
                "sub_halluc": f"{sub_hal:.2f}",
                "obj_halluc": f"{obj_hal:.2f}",
                "fmt_error": f"{erro_formato:.2f}",
                "llm_triples": sys_triplas,
                "filtered_llm_triples": fil_triplas,
                "gt_triples": gt_triplas,
                "sent": frase,
            })

            tp += precision; tr_ += recall; tf1 += f1
            tc += conformidade; trh += rel_hal; tsh += sub_hal; toh += obj_hal
            tef += erro_formato

            if sent_id in ids_selecionados:
                stp += precision; str_ += recall; stf1 += f1
                stc += conformidade; strh += rel_hal; stsh += sub_hal; stoh += obj_hal
                stef += erro_formato

        salvar_jsonl(metricas_por_frase, output_path)

        # Denominador: frases efetivamente avaliadas (modo_amostra) ou GT completo (benchmark)
        total = len(metricas_por_frase) if modo_amostra else len(gabarito)
        if total == 0:
            total = 1  # evita divisão por zero em ontologias sem saída

        media = {
            "onto": onto_id, "type": "all_test_cases",
            "modo_amostra": modo_amostra,
            "avg_precision": f"{tp/total:.2f}",
            "avg_recall": f"{tr_/total:.2f}",
            "avg_f1": f"{tf1/total:.2f}",
            "avg_onto_conf": f"{tc/total:.2f}",
            "avg_sub_halluc": f"{tsh/total:.2f}",
            "avg_rel_halluc": f"{trh/total:.2f}",
            "avg_obj_halluc": f"{toh/total:.2f}",
            "avg_fmt_error": f"{tef/total:.2f}",
        }
        acrescentar_jsonl(media, avg_path)

        gp += tp/total; gr += tr_/total; gf1 += tf1/total
        gc += tc/total; grh += trh/total; gsh += tsh/total; goh += toh/total
        gef += tef/total

        resultados_por_onto[onto_id] = media

        if ids_selecionados:
            total_sel = len(ids_selecionados)
            media_sel = {
                "onto": onto_id, "type": "selected_test_cases",
                "avg_precision": f"{stp/total_sel:.2f}",
                "avg_recall": f"{str_/total_sel:.2f}",
                "avg_f1": f"{stf1/total_sel:.2f}",
                "avg_onto_conf": f"{stc/total_sel:.2f}",
                "avg_sub_halluc": f"{stsh/total_sel:.2f}",
                "avg_rel_halluc": f"{strh/total_sel:.2f}",
                "avg_obj_halluc": f"{stoh/total_sel:.2f}",
                "avg_fmt_error": f"{stef/total_sel:.2f}",
            }
            acrescentar_jsonl(media_sel, avg_path)

    n_onto = len(config["onto_list"])
    global_metricas = {
        "id": "global", "type": "global",
        "modo_amostra": modo_amostra,
        "avg_precision": f"{gp/n_onto:.2f}",
        "avg_recall": f"{gr/n_onto:.2f}",
        "avg_f1": f"{gf1/n_onto:.2f}",
        "avg_onto_conf": f"{gc/n_onto:.2f}",
        "avg_sub_halluc": f"{gsh/n_onto:.2f}",
        "avg_rel_halluc": f"{grh/n_onto:.2f}",
        "avg_obj_halluc": f"{goh/n_onto:.2f}",
        "avg_fmt_error": f"{gef/n_onto:.2f}",
        "onto_list": config["onto_list"],
    }
    acrescentar_jsonl(global_metricas, avg_path)

    return {"global": global_metricas, "por_ontologia": resultados_por_onto}


# ---------------------------------------------------------------------------
# CLI (compatível com o run_eval.py original)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_config_path", type=str, required=True)
    parser.add_argument("--base_dir", type=str, default="",
                        help="Diretório base para resolver caminhos relativos")
    args = parser.parse_args()

    if not os.path.exists(args.eval_config_path):
        print(f"Config não encontrada: {args.eval_config_path}")
        sys.exit(1)

    config = carregar_config(args.eval_config_path)
    avaliar(config, diretorio_base=args.base_dir)
