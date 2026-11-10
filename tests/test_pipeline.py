"""
Testes de integração do pipeline de extração (Fase 1).

Todos os testes usam o ProvedorMock — sem LLM real necessário.
Requerem os dados do Text2KGBench em ../Text2KGBench (irmão do repo).
"""
import json
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE = os.path.abspath(os.path.join(REPO, ".."))
BENCH = os.path.join(WORKSPACE, "Text2KGBench")
sys.path.insert(0, os.path.join(REPO, "src"))

from t2kg.pipeline.dados import carregar_frases
from t2kg.pipeline.ontologias import carregar_ontologia, formatar_contexto
from t2kg.pipeline.prompts import montar_prompt
from t2kg.pipeline.provedores.mock import ProvedorMock
from t2kg.pipeline.executar import executar

SKIP_SEM_BENCH = pytest.mark.skipif(
    not os.path.exists(BENCH), reason="Text2KGBench não encontrado"
)


# ---------------------------------------------------------------------------
# dados.py
# ---------------------------------------------------------------------------

class TestCarregarFrases:
    @SKIP_SEM_BENCH
    def test_tekgen_retorna_frases(self):
        frases = carregar_frases(BENCH, "tekgen", "test", "1_movie")
        assert len(frases) > 0
        primeira = frases[0]
        assert "id" in primeira
        assert "sent" in primeira
        assert isinstance(primeira["gt_triples"], list)

    @SKIP_SEM_BENCH
    def test_exemplo_extraido_para_c1(self):
        frases = carregar_frases(BENCH, "tekgen", "test", "1_movie")
        # A maioria das frases deve ter exemplo extraído do prompt original
        com_exemplo = [f for f in frases if f["exemplo_sent"] is not None]
        assert len(com_exemplo) > len(frases) * 0.9  # > 90% têm exemplo

    @SKIP_SEM_BENCH
    def test_webnlg_retorna_frases(self):
        frases = carregar_frases(BENCH, "webnlg", "test", "1_university")
        assert len(frases) > 0

    @SKIP_SEM_BENCH
    def test_unseen_retorna_frases(self):
        frases = carregar_frases(BENCH, "tekgen", "unseen", "1_movie")
        assert len(frases) > 0


# ---------------------------------------------------------------------------
# ontologias.py
# ---------------------------------------------------------------------------

class TestOntologias:
    @SKIP_SEM_BENCH
    def test_carregar_tekgen(self):
        ont = carregar_ontologia(BENCH, "tekgen", "1_movie")
        assert "relations" in ont
        assert "concepts" in ont

    @SKIP_SEM_BENCH
    def test_c0_retorna_none(self):
        ont = carregar_ontologia(BENCH, "tekgen", "1_movie")
        assert formatar_contexto(ont, "C0") is None

    @SKIP_SEM_BENCH
    def test_c1_contem_relacoes_e_conceitos(self):
        ont = carregar_ontologia(BENCH, "tekgen", "1_movie")
        ctx = formatar_contexto(ont, "C1")
        assert "Ontology Relations:" in ctx
        assert "Ontology Concepts:" in ctx
        assert "director" in ctx

    @SKIP_SEM_BENCH
    def test_c3_so_relacoes_sem_dominio(self):
        ont = carregar_ontologia(BENCH, "tekgen", "1_movie")
        ctx = formatar_contexto(ont, "C3")
        assert "Relations:" in ctx
        assert "Ontology Concepts:" not in ctx
        # C3 não inclui domain/range
        assert "(film," not in ctx

    @SKIP_SEM_BENCH
    def test_c5_mesmo_formato_que_c1(self):
        """C5 deve usar o mesmo formato texto que C1 (Ontology Concepts + Ontology Relations)."""
        ont = carregar_ontologia(BENCH, "tekgen", "1_movie")
        ctx_c1 = formatar_contexto(ont, "C1")
        ctx_c5 = formatar_contexto(ont, "C5")
        # Mesmas chaves de formato
        assert "Ontology Concepts:" in ctx_c5
        assert "Ontology Relations:" in ctx_c5
        # Mesmo conteúdo quando ontologia é a mesma
        assert ctx_c1 == ctx_c5

    def test_condicao_invalida_levanta_erro(self):
        with pytest.raises(ValueError):
            formatar_contexto({}, "C99")


# ---------------------------------------------------------------------------
# prompts.py
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_c0_sem_contexto(self):
        p = montar_prompt("A cat sat on a mat.", "C0")
        assert "A cat sat on a mat." in p
        assert "Ontology" not in p

    def test_c1_tem_contexto(self):
        ctx = "Ontology Concepts: X\nOntology Relations: rel(X,Y)"
        p = montar_prompt("My sentence.", "C1", contexto_ontologia=ctx)
        assert "Ontology Relations:" in p
        assert "My sentence." in p
        assert "Example Sentence:" not in p

    def test_c1_oneshot_tem_exemplo(self):
        ctx = "Ontology Concepts: X\nOntology Relations: rel(X,Y)"
        p = montar_prompt(
            "My sentence.", "C1_oneshot_original",
            contexto_ontologia=ctx,
            exemplo_sent="Example sentence here.",
            exemplo_saida="rel(A, B)",
        )
        assert "Example Sentence: Example sentence here." in p
        assert "Example Output: rel(A, B)" in p
        assert "Test Sentence: My sentence." in p

    def test_c1_oneshot_sem_exemplo_omite_bloco(self):
        ctx = "Ontology Relations: rel(X,Y)"
        p = montar_prompt("My sentence.", "C1_oneshot_original", contexto_ontologia=ctx)
        assert "Example Sentence:" not in p

    def test_c3_lista_relacoes(self):
        ctx = "Relations: director, screenwriter"
        p = montar_prompt("My sentence.", "C3", contexto_ontologia=ctx)
        assert "Relations: director" in p

    def test_c5_mesma_instrucao_que_c1(self):
        """C5 e C1 devem usar a mesma instrução base (formato texto, não JSON)."""
        ctx = "Ontology Concepts: film,\nOntology Relations: director(film,human)"
        p_c5 = montar_prompt("My sentence.", "C5", contexto_ontologia=ctx)
        p_c1 = montar_prompt("My sentence.", "C1", contexto_ontologia=ctx)
        # Ambos devem ter a instrução de entidade
        assert "exact entity names" in p_c5
        # C5 não deve referenciar JSON
        assert "JSON" not in p_c5
        # Ambos devem conter "ontology" na instrução
        assert "ontology" in p_c5.lower()
        assert "ontology" in p_c1.lower()

    def test_instrucao_entidade_presente_em_c0_c1_c3_c5(self):
        """C0, C1, C3, C5 devem conter a instrução de não usar classes da ontologia."""
        trecho = "exact entity names as written in the sentence"
        ctx = "Ontology Relations: rel(X,Y)"
        for cond, extra in [
            ("C0", {}),
            ("C1", {"contexto_ontologia": ctx}),
            ("C3", {"contexto_ontologia": "Relations: rel(X,Y)"}),
            ("C5", {"contexto_ontologia": '{"relations":[]}'}),
        ]:
            p = montar_prompt("My sentence.", cond, **extra)
            assert trecho in p, f"Instrução de entidade ausente em {cond}"

    def test_instrucao_entidade_ausente_em_c1_oneshot_original(self):
        """C1_oneshot_original NÃO deve conter a instrução (fidelidade ao benchmark)."""
        ctx = "Ontology Relations: rel(X,Y)"
        p = montar_prompt("My sentence.", "C1_oneshot_original", contexto_ontologia=ctx)
        assert "exact entity names" not in p

    @pytest.mark.parametrize("condicao,extra", [
        ("C0", {}),
        ("C1",           {"contexto_ontologia": "Ontology Concepts: X,\nOntology Relations: rel(X,Y)"}),
        ("C2",           {"contexto_ontologia": "Relations: rel_induzida_1, rel_induzida_2"}),
        ("C3",           {"contexto_ontologia": "Relations: rel_treino_1, rel_treino_2"}),
        ("C5_vizinho",   {"contexto_ontologia": "Ontology Concepts: X,\nOntology Relations: rel(X,Y)"}),
        ("C5_incongruente", {"contexto_ontologia": "Ontology Concepts: X,\nOntology Relations: rel(X,Y)"}),
    ])
    def test_zero_shot_instrucao_d_sem_exemplo(self, condicao, extra):
        """C0, C1, C2, C3, C5_vizinho, C5_incongruente: zero-shot e instrução D (D-12/D-20)."""
        p = montar_prompt("My sentence.", condicao, **extra)
        assert "Example Sentence:" not in p, (
            f"{condicao}: não deve conter 'Example Sentence:' (viola D-12 zero-shot)"
        )
        assert "Example Output:" not in p, (
            f"{condicao}: não deve conter 'Example Output:' (viola D-12 zero-shot)"
        )
        assert "Output the triples, one per line" in p, (
            f"{condicao}: instrução de formato D ausente"
        )

    def test_c2_identico_a_c1_com_mesmo_contexto(self):
        """C2 com mesmo contexto_ontologia que C1 deve produzir prompt byte-a-byte idêntico."""
        ctx = "Ontology Concepts: film,\nOntology Relations: director(film,human)"
        p_c1 = montar_prompt("My sentence.", "C1", contexto_ontologia=ctx)
        p_c2 = montar_prompt("My sentence.", "C2", contexto_ontologia=ctx)
        assert p_c1 == p_c2, (
            "C2 e C1 com mesmo contexto devem produzir prompts byte-a-byte idênticos.\n"
            f"C1 ({len(p_c1)} chars):\n{p_c1!r}\n\n"
            f"C2 ({len(p_c2)} chars):\n{p_c2!r}"
        )


# ---------------------------------------------------------------------------
# executar.py — integração + checkpoint
# ---------------------------------------------------------------------------

class TestExecutar:
    @SKIP_SEM_BENCH
    def test_executa_n_frases(self, tmp_path):
        saida = str(tmp_path / "saida.jsonl")
        n = executar(
            bench_dir=BENCH,
            dataset="tekgen",
            split="test",
            ontologia_id="1_movie",
            condicao="C1_oneshot_original",
            provedor=ProvedorMock(),
            saida=saida,
            frases_max=10,
        )
        assert n == 10
        linhas = [json.loads(l) for l in open(saida, encoding="utf-8") if l.strip()]
        assert len(linhas) == 10
        for linha in linhas:
            assert "id" in linha
            assert "sent" in linha
            assert "response" in linha
            assert isinstance(linha["triples"], list)

    @SKIP_SEM_BENCH
    def test_checkpoint_nao_repete(self, tmp_path):
        saida = str(tmp_path / "saida.jsonl")
        # Primeira execução: 5 frases
        n1 = executar(
            bench_dir=BENCH, dataset="tekgen", split="test",
            ontologia_id="1_movie", condicao="C0",
            provedor=ProvedorMock(), saida=saida, frases_max=5,
        )
        assert n1 == 5

        # Segunda execução: 10 total — deve processar só 5 novas
        n2 = executar(
            bench_dir=BENCH, dataset="tekgen", split="test",
            ontologia_id="1_movie", condicao="C0",
            provedor=ProvedorMock(), saida=saida, frases_max=10,
        )
        assert n2 == 5  # 5 novas (5 já prontas + 5 novas = 10 total)

        linhas = [json.loads(l) for l in open(saida, encoding="utf-8") if l.strip()]
        assert len(linhas) == 10

        # IDs únicos (sem duplicatas)
        ids = [l["id"] for l in linhas]
        assert len(ids) == len(set(ids))

    @SKIP_SEM_BENCH
    def test_condicoes_todas_funcionam(self, tmp_path):
        for cond in ["C0", "C1", "C1_oneshot_original", "C3", "C5"]:
            saida = str(tmp_path / f"{cond}.jsonl")
            n = executar(
                bench_dir=BENCH, dataset="tekgen", split="test",
                ontologia_id="2_music", condicao=cond,
                provedor=ProvedorMock(), saida=saida, frases_max=3,
            )
            assert n == 3, f"Falhou para condição {cond}"

    @SKIP_SEM_BENCH
    def test_c3_usa_dados_de_treino_nao_gt(self, tmp_path):
        """C3: a fonte das top-k relações deve ser o treino, nunca o GT de teste."""
        from t2kg.pipeline.dados import _caminhos_treino, carregar_frequencias_treino

        # Verifica que o caminho de treino não aponta para ground_truth nem /test/
        path_tkg = _caminhos_treino(BENCH, "tekgen", "1_movie")
        path_web = _caminhos_treino(BENCH, "webnlg", "1_university")
        for path in (path_tkg, path_web):
            norm = path.replace("\\", "/")
            assert "train" in norm, f"Caminho não contém 'train': {path}"
            assert "ground_truth" not in norm, f"Caminho contém 'ground_truth': {path}"
            assert "/test/" not in norm, f"Caminho contém '/test/': {path}"

        # Verifica que as relações retornadas são as do treino (diferentes do GT de teste)
        top3_treino = carregar_frequencias_treino(BENCH, "tekgen", "1_movie", k=3)
        assert isinstance(top3_treino, list)
        assert len(top3_treino) == 3
        # treino de 1_movie tem 'director','screenwriter','genre'; GT de teste tem 'publication_date','cast_member'
        # screenwriter aparece no treino mas NÃO é top-3 no GT de teste
        assert "screenwriter" in top3_treino, (
            f"'screenwriter' deveria estar no top-3 do treino: {top3_treino}"
        )

    @SKIP_SEM_BENCH
    def test_c3_top_k_incluido_no_prompt(self, tmp_path):
        """C3 com k=3 do TREINO: prompt deve conter as top-3 relações com domain/range."""
        from t2kg.pipeline.dados import carregar_frequencias_treino

        ont = carregar_ontologia(BENCH, "tekgen", "1_movie")
        top3 = carregar_frequencias_treino(BENCH, "tekgen", "1_movie", k=3)

        ctx = formatar_contexto(ont, "C3", top_k_rels=top3)
        assert "Relations:" in ctx
        # deve incluir domain/range (parênteses)
        assert "(" in ctx
        # deve conter as relações do treino
        for rel in top3:
            assert rel in ctx

    @SKIP_SEM_BENCH
    def test_c3_nao_abre_arquivo_de_teste_ou_gabarito(self, tmp_path):
        """
        Nenhum arquivo de teste, unseen ou ground_truth deve ser aberto
        durante a construção do contexto C3 (leakagem de dados).
        """
        from unittest.mock import patch
        from t2kg.pipeline.dados import carregar_frequencias_treino

        arquivos_abertos = []
        _open_original = open

        def spy_open(path, *args, **kwargs):
            arquivos_abertos.append(str(path).replace("\\", "/"))
            return _open_original(path, *args, **kwargs)

        with patch("builtins.open", side_effect=spy_open):
            top3 = carregar_frequencias_treino(BENCH, "tekgen", "1_movie", k=3)

        proibidos = ["ground_truth", "/test/", "unseen", "llm_responses"]
        for p in arquivos_abertos:
            for proibido in proibidos:
                assert proibido not in p, (
                    f"Arquivo proibido aberto durante C3: {p!r} contém '{proibido}'"
                )

        # Confirma que as relações são do treino, não do GT de teste
        assert "screenwriter" in top3, (
            f"'screenwriter' deveria estar no top-3 do treino: {top3}"
        )
        assert "publication_date" not in top3, (
            f"'publication_date' é do GT de teste e não deveria aparecer no treino: {top3}"
        )

    @SKIP_SEM_BENCH
    def test_c5_vizinho_usa_ontologia_parceira(self, tmp_path):
        """C5_vizinho deve usar o conteúdo da ontologia parceira (formato texto, igual C1)."""
        ont_alvo = carregar_ontologia(BENCH, "tekgen", "1_movie")
        ont_parceira = carregar_ontologia(BENCH, "tekgen", "2_music")
        ctx_vizinho = formatar_contexto(ont_alvo, "C5_vizinho", ontologia_parceira=ont_parceira)
        ctx_parceira_c1 = formatar_contexto(ont_parceira, "C1")
        # C5_vizinho com parceira deve ser igual a C1 da parceira
        assert ctx_vizinho == ctx_parceira_c1
        # C5_vizinho não deve ser igual a C1 do alvo
        ctx_alvo_c1 = formatar_contexto(ont_alvo, "C1")
        assert ctx_vizinho != ctx_alvo_c1


# ---------------------------------------------------------------------------
# avaliacao/original.py — modo_amostra
# ---------------------------------------------------------------------------

class TestModoAmostra:
    def test_modo_amostra_corrige_denominador(self, tmp_path):
        """
        Dados sintéticos: 10 frases no GT, 5 processadas com triples=[].
        Com triples=[],  calcular_conformidade_ontologica retorna 1.0 por frase.
        modo_amostra=False: 5*1.0 / 10 = 0.50 (denominador = total GT)
        modo_amostra=True:  5*1.0 / 5  = 1.00 (denominador = frases processadas)
        """
        from t2kg.avaliacao.original import avaliar

        # Ontologia mínima sem relações (qualquer tripla seria inconformante)
        ont_data = {"title": "Teste", "concepts": [{"qid": "Q1", "label": "X"}], "relations": []}
        ont_path = str(tmp_path / "ont.json")
        with open(ont_path, "w", encoding="utf-8") as f:
            json.dump(ont_data, f)

        # GT: 10 frases com 1 tripla cada
        gt_path = str(tmp_path / "gt.jsonl")
        with open(gt_path, "w", encoding="utf-8") as f:
            for i in range(10):
                f.write(json.dumps({"id": f"s{i}", "sent": f"Frase {i}.",
                                    "triples": [{"sub": "A", "rel": "rel", "obj": "B"}]}) + "\n")

        # Sistema: só as 5 primeiras frases, todas com triples=[] → conf=1.0 cada
        sys_path = str(tmp_path / "sys.jsonl")
        with open(sys_path, "w", encoding="utf-8") as f:
            for i in range(5):
                f.write(json.dumps({"id": f"s{i}", "sent": f"Frase {i}.",
                                    "response": "", "triples": []}) + "\n")

        def _config(avg_name):
            return {
                "onto_list": [{
                    "id": "test_onto",
                    "sys": sys_path,
                    "gt": gt_path,
                    "onto": ont_path,
                    "output": str(tmp_path / "output.jsonl"),
                }],
                "avg_out_file": str(tmp_path / avg_name),
            }

        # Modo benchmark: denominador = 10 (total GT) → 5*1.0/10 = 0.50
        res_full = avaliar(_config("avg_full.jsonl"), modo_amostra=False)
        conf_full = float(res_full["por_ontologia"]["test_onto"]["avg_onto_conf"])

        # Modo amostra: denominador = 5 (processadas) → 5*1.0/5 = 1.00
        res_sample = avaliar(_config("avg_sample.jsonl"), modo_amostra=True)
        conf_sample = float(res_sample["por_ontologia"]["test_onto"]["avg_onto_conf"])

        assert conf_full == 0.50, f"Esperado 0.50 (modo benchmark), obtido {conf_full}"
        assert conf_sample == 1.0, f"Esperado 1.0 (modo amostra), obtido {conf_sample}"
        assert res_full["global"]["modo_amostra"] is False
        assert res_sample["global"]["modo_amostra"] is True
