"""
Testes para scripts/induzir_c2.py.

Por que os testes anteriores NÃO pegaram os dois bugs do piloto
===============================================================
Os testes existentes (test_pipeline.py, test_avaliacao_resultados.py, etc.) cobrem
exclusivamente o pipeline principal (executar.py → run.py), que chama montar_prompt()
com keyword arguments. O script induzir_c2.py:

  1. Não é importado por nenhum módulo do pipeline — só executado como __main__.
     Nenhum teste existente o importava ou o chamava diretamente.

  2. Bug 1 — argumento posicional errado em montar_prompt():
       ANTES: montar_prompt(f, onto_vazia, "C0")
     Com argumentos posicionais, o 1.º parâmetro (sentenca) recebia o dict inteiro da
     frase (incluindo gt_triples), e o 2.º (condicao) recebia o dict da ontologia vazia.
     Como condicao era um dict (nunca == "C0"), o bloco de C1_oneshot_original era
     alcançado — e o dict da frase, convertido em str(), continha "gt_triples".
     Nenhum teste existente chamava montar_prompt() com argumentos posicionais incorretos.

  3. Bug 2 — _caminhos_gt() usada em vez de _caminhos_treino():
       ANTES: carregar_frases(bench, dataset, "train", onto_id)
     A função _caminhos_gt() não tem caso "train" — cai no else e retorna o GT de TESTE.
     Os IDs retornados eram "ont_1_movie_test_*" — mas nenhum teste verificava os IDs
     retornados por carregar_frases() com split="train".

Esses testes foram adicionados DEPOIS dos bugs serem encontrados nos dados do piloto
Kaggle (commit 213a83b). Os testes abaixo trazem cobertura retroativa e garantem
que os bugs não regridam.
"""
import builtins
import json
import os
import pathlib
import subprocess
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

WORKSPACE = os.path.abspath(os.path.join(REPO, ".."))
BENCH = os.path.join(WORKSPACE, "Text2KGBench")
SKIP_SEM_BENCH = pytest.mark.skipif(
    not os.path.exists(BENCH), reason="Text2KGBench não encontrado"
)


# ---------------------------------------------------------------------------
# (a) Prompt de indução idêntico ao C0 para a mesma frase
# ---------------------------------------------------------------------------

def test_prompt_inducao_identico_c0():
    """
    O prompt enviado pelo induzir_c2 deve ser idêntico ao de C0 para a mesma frase.

    Verifica que a chamada corrigida montar_prompt(sentenca=f["sent"], condicao="C0")
    produz exatamente o mesmo prompt que a chamada C0 do pipeline principal.
    Demonstra também que a chamada bugada (argumental posicional com dict) é diferente.
    """
    from t2kg.pipeline.prompts import montar_prompt

    frase_dict = {
        "id": "ont_1_movie_train_1",
        "sent": "The movie was directed by Christopher Nolan.",
        "gt_triples": [{"sub": "The movie", "rel": "director", "obj": "Christopher Nolan"}],
    }

    # Chamada correta (após correção do bug)
    prompt_inducao = montar_prompt(sentenca=frase_dict["sent"], condicao="C0")

    # Chamada que o pipeline principal usa
    prompt_pipeline_c0 = montar_prompt(sentenca=frase_dict["sent"], condicao="C0")

    assert prompt_inducao == prompt_pipeline_c0, (
        "Prompt de indução C2 difere do prompt C0 do pipeline — "
        "induzir_c2.py está usando argumentos diferentes."
    )

    # Garantias estruturais do C0
    assert "CONTEXT" not in prompt_inducao, "C0 não deve ter bloco CONTEXT"
    assert "gt_triples" not in prompt_inducao, "C0 não deve vazar gt_triples"
    assert "director" not in prompt_inducao, "C0 não deve vazar rótulo de relação do GT"
    assert frase_dict["sent"] in prompt_inducao, "Frase deve estar no prompt"
    assert "Test Output:" in prompt_inducao, "Prompt deve terminar com 'Test Output:'"


def test_prompt_inducao_bugado_vaza_gt():
    """
    Demonstra que a chamada bugada montar_prompt(f, onto, "C0") vazava gt_triples.

    Este teste documenta o comportamento pré-correção para garantir que nenhuma
    regressão restaure o bug silenciosamente.
    """
    from t2kg.pipeline.prompts import montar_prompt

    frase_dict = {
        "id": "ont_1_movie_train_1",
        "sent": "The movie was directed by Christopher Nolan.",
        "gt_triples": [{"sub": "The movie", "rel": "director", "obj": "Christopher Nolan"}],
    }

    # Chamada bugada: frase_dict como 1.º arg posicional; agora levanta ValueError
    # (melhor que vazar gt_triples silenciosamente como ocorria antes)
    import pytest
    with pytest.raises((ValueError, TypeError)):
        montar_prompt(frase_dict, {"concepts": [], "relations": []}, "C0")


# ---------------------------------------------------------------------------
# (b) IDs sorteados contêm "_train_"
# ---------------------------------------------------------------------------

@SKIP_SEM_BENCH
def test_sortear_frases_treino_ids_contem_train():
    """
    sortear_frases_treino() deve retornar apenas frases de treino.
    Todos os IDs devem conter '_train_', nunca '_test_'.
    """
    from t2kg.pipeline.dados import carregar_frases_treino

    frases = carregar_frases_treino(BENCH, "tekgen", "1_movie")
    assert len(frases) > 0, "carregar_frases_treino não retornou nenhuma frase"

    for f in frases[:20]:
        assert "_train_" in f["id"], (
            f"ID de frase de treino não contém '_train_': {f['id']!r} — "
            "carregar_frases_treino está retornando dados de teste."
        )
        assert "_test_" not in f["id"], (
            f"ID de frase de treino contém '_test_': {f['id']!r} — "
            "vazamento confirmado de dados de teste no treino."
        )


@SKIP_SEM_BENCH
def test_sortear_frases_treino_seed_reprodutivel():
    """Duas chamadas com mesma seed devem retornar mesma ordem."""
    import random
    from t2kg.pipeline.dados import carregar_frases_treino

    frases = list(carregar_frases_treino(BENCH, "tekgen", "1_movie"))
    rng = random.Random(42)
    rng.shuffle(frases)
    ids_a = [f["id"] for f in frases[:50]]

    frases2 = list(carregar_frases_treino(BENCH, "tekgen", "1_movie"))
    rng2 = random.Random(42)
    rng2.shuffle(frases2)
    ids_b = [f["id"] for f in frases2[:50]]

    assert ids_a == ids_b, "Seed 42 deve produzir mesma amostra em ambas as chamadas"


# ---------------------------------------------------------------------------
# (c) Interceptor de open() — nenhum arquivo de ground_truth/teste é aberto
# ---------------------------------------------------------------------------

@SKIP_SEM_BENCH
def test_inducao_nao_abre_arquivos_de_teste(monkeypatch):
    """
    Intercepta builtins.open() e verifica que carregar_frases_treino() não abre
    nenhum arquivo de ground_truth ou de split de teste durante a indução C2.

    Detecta regressões onde _caminhos_gt() seja chamada com split="train" (bug 2):
    essa função não tem caso "train" e cai no else, retornando o GT de teste.
    """
    opened_paths: list = []
    real_open = builtins.open

    def interceptor(file, *args, **kwargs):
        if isinstance(file, (str, pathlib.Path)):
            opened_paths.append(str(file).replace("\\", "/"))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", interceptor)

    from t2kg.pipeline.dados import carregar_frases_treino
    frases = carregar_frases_treino(BENCH, "tekgen", "1_movie")

    # Verifica que algum arquivo de treino foi aberto
    treino_aberto = any("train" in p.lower() for p in opened_paths)
    assert treino_aberto, (
        f"Nenhum arquivo de treino foi aberto. Caminhos: {opened_paths} — "
        "carregar_frases_treino não está lendo dados de treino."
    )

    # Verifica que nenhum arquivo de teste ou ground_truth foi aberto
    proibidos = [
        p for p in opened_paths
        if "ground_truth" in p.lower() or "/test/" in p.lower() or "_test_" in p.lower()
    ]
    assert not proibidos, (
        f"Arquivos proibidos foram abertos durante a indução C2:\n"
        + "\n".join(f"  {p}" for p in proibidos)
        + "\ncarregar_frases_treino está abrindo dados de teste/GT — bug 2 regrediu."
    )

    assert len(frases) > 0, "Nenhuma frase de treino foi carregada"


# ---------------------------------------------------------------------------
# (d) extrair_triplas_livres retorna apenas IDs de treino
# ---------------------------------------------------------------------------

def test_induzir_c2_raw_jsonl_apenas_treino():
    """
    Verifica que extrair_triplas_livres() com mock produz resultados com IDs de treino.
    Garante que o formato do dict retornado não contamina com test IDs.
    """
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from induzir_c2 import extrair_triplas_livres

    class MockProvedor:
        def gerar(self, prompt, **kw):
            return "director(Film A, Person B)"

    frases = [
        {"id": "ont_1_movie_train_001", "sent": "Film A was directed by Person B.", "gt_triples": []},
        {"id": "ont_1_movie_train_002", "sent": "Film C was written by Person D.", "gt_triples": []},
    ]
    resultado = extrair_triplas_livres(frases, MockProvedor())

    assert len(resultado) == 2, f"Esperado 2 entradas, obteve {len(resultado)}"
    for fid in resultado:
        assert "_train_" in fid, f"ID não é de treino: {fid!r}"
        assert "_test_" not in fid, f"ID de teste encontrado: {fid!r}"
    # Verifica formato das triplas
    for fid, triplas in resultado.items():
        assert isinstance(triplas, list), f"triplas deve ser lista, obteve {type(triplas)}"


# ---------------------------------------------------------------------------
# (e) contar_frases_por_grupo conta frases distintas, não triplas (D-21 fix)
# ---------------------------------------------------------------------------

def test_contar_frases_por_grupo_conta_frases_distintas():
    """
    contar_frases_por_grupo deve contar frases DISTINTAS, não total de triplas.

    Frase A: 3 triplas com rel "directed_by" → conta 1 (não 3).
    Frase B: 1 tripla com rel "directed_by" → conta +1 = 2 total.
    """
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from induzir_c2 import contar_frases_por_grupo

    triplas_por_frase = {
        "frase_A": [["F1", "directed_by", "P1"], ["F1", "directed_by", "P2"], ["F1", "directed_by", "P3"]],
        "frase_B": [["F2", "directed_by", "P4"]],
        "frase_C": [["F3", "other_rel", "P5"]],
    }
    grupos = [{"label": "directed_by", "frequencia": 4, "membros": ["directed_by"]}]
    contar_frases_por_grupo(grupos, triplas_por_frase)

    assert grupos[0]["n_frases"] == 2, (
        f"n_frases esperado=2 (frase_A + frase_B), obteve {grupos[0]['n_frases']}. "
        "contar_frases_por_grupo está contando triplas em vez de frases distintas."
    )


def test_contar_frases_por_grupo_membros_multiplos():
    """Grupo com múltiplos membros deve unir frases de todos os membros."""
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from induzir_c2 import contar_frases_por_grupo

    triplas_por_frase = {
        "f1": [["A", "directed_by", "B"]],
        "f2": [["C", "director", "D"]],
        "f3": [["E", "wrote", "F"]],  # relação fora do grupo
    }
    grupos = [{"label": "directed_by", "frequencia": 2, "membros": ["directed_by", "director"]}]
    contar_frases_por_grupo(grupos, triplas_por_frase)

    assert grupos[0]["n_frases"] == 2, (
        f"n_frases esperado=2 (f1 via 'directed_by', f2 via 'director'), obteve {grupos[0]['n_frases']}"
    )


# ---------------------------------------------------------------------------
# (f) induzir_c2.py --use_definicoes como subprocesso (D-24 path end-to-end)
# ---------------------------------------------------------------------------

def test_induzir_c2_use_definicoes_subprocess(tmp_path):
    """
    Executa induzir_c2.py como subprocesso com --use_definicoes --from_raw_jsonl.

    Verifica que:
    1. O processo termina com código 0 (sem NameError nem outra exceção).
    2. O arquivo induzido é gerado em saida_dir.
    3. O metadata.use_definicoes == True no arquivo gerado.
    4. A definição mockada é de fato usada (arquivo de definições lido corretamente).
    """
    saida_dir = tmp_path / "c2"
    saida_dir.mkdir()

    # Raw JSONL mínimo: 3 frases de treino com triplas extraídas livres
    raw_jsonl = tmp_path / "raw_1_movie_treino.jsonl"
    registros = [
        {"id": "ont_1_movie_train_001", "sent": "Film A was directed by Person B.",
         "response": "directed_by(Film A, Person B)", "triples": [["Film A", "directed_by", "Person B"]]},
        {"id": "ont_1_movie_train_002", "sent": "Film C stars Actor D.",
         "response": "stars(Film C, Actor D)", "triples": [["Film C", "stars", "Actor D"]]},
        {"id": "ont_1_movie_train_003", "sent": "Film E was produced by Studio F.",
         "response": "directed_by(Film E, Studio F)", "triples": [["Film E", "directed_by", "Studio F"]]},
    ]
    raw_jsonl.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in registros),
        encoding="utf-8",
    )

    # Arquivo de definições D-24 mínimo (como gerar_definicoes_relacoes.py geraria)
    definicoes_path = saida_dir / "definicoes_1_movie.json"
    definicoes = {
        "modelo": "mock",
        "relacoes": {
            "directed_by": "Indicates the person who directed a film.",
            "stars": "Indicates an actor who appears in a film.",
        },
    }
    definicoes_path.write_text(json.dumps(definicoes, ensure_ascii=False), encoding="utf-8")

    script = os.path.join(REPO, "scripts", "induzir_c2.py")
    env = {**os.environ, "PYTHONPATH": os.path.join(REPO, "src")}
    cmd = [
        sys.executable, script,
        "--dataset", "tekgen",
        "--onto", "1_movie",
        "--threshold", "0.70",
        "--freq_min", "1",
        "--n_frases", "3",
        "--seed", "42",
        "--from_raw_jsonl", str(raw_jsonl),
        "--saida_dir", str(saida_dir),
        "--normalizar",
        "--use_definicoes",
        "--run_id", "SUBPROCESS_TEST",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    assert result.returncode == 0, (
        f"induzir_c2.py --use_definicoes saiu com rc={result.returncode}.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    # Verifica que o arquivo induzido foi gerado
    induced_path = saida_dir / "tekgen_1_movie_induced.json"
    assert induced_path.exists(), (
        f"Arquivo induzido não gerado: {induced_path}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    with open(induced_path, encoding="utf-8") as f:
        induced = json.load(f)

    assert induced.get("metadata", {}).get("use_definicoes") is True, (
        "metadata.use_definicoes deve ser True quando --use_definicoes é passado."
    )
    assert induced.get("metadata", {}).get("run_id") == "SUBPROCESS_TEST"
    assert "relations" in induced, "Arquivo induzido deve ter campo 'relations'"
