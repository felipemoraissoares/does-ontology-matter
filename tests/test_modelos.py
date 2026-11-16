"""D-33: slug de modelo e carregamento de ontologia induzida por modelo (v_full/<slug>)."""
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from t2kg.modelos import slug_modelo
from t2kg.pipeline.ontologias import carregar_ontologia_induzida


class TestSlugModelo:
    @pytest.mark.parametrize("modelo,esperado", [
        ("Qwen/Qwen2.5-7B-Instruct-AWQ", "qwen2.5-7b"),
        ("Qwen/Qwen2.5-14B-Instruct-AWQ", "qwen2.5-14b"),
        ("Qwen/Qwen2.5-32B-Instruct-AWQ", "qwen2.5-32b"),
        ("google/gemma-2-9b-it", "gemma-2-9b"),
        ("meta-llama/Llama-3.1-8B-Instruct", "llama-3.1-8b"),
        ("qwen2.5-7b", "qwen2.5-7b"),
    ])
    def test_slug(self, modelo, esperado):
        assert slug_modelo(modelo) == esperado


class TestC2Subdir:
    def test_subdir_compoe_caminho_v_full(self):
        with pytest.raises(FileNotFoundError) as e:
            carregar_ontologia_induzida("tekgen", "1_movie", c2_subdir="v_full/qwen2.5-7b")
        msg = str(e.value).replace("\\", "/")
        assert "configs/c2/v_full/qwen2.5-7b/tekgen_1_movie_induced.json" in msg

    def test_legado_sem_subdir_carrega_piloto(self):
        # O arquivo do piloto existe em configs/c2/ — carregamento legado inalterado.
        ont = carregar_ontologia_induzida("tekgen", "1_movie")
        assert "relations" in ont and len(ont["relations"]) > 0
