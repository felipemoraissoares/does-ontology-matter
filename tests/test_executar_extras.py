"""I-2 (C5 sem parceira erra) e I-9 (finish_reason registrado)."""
import json
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE = os.path.abspath(os.path.join(REPO_ROOT, ".."))
BENCH = os.path.join(WORKSPACE, "Text2KGBench")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from t2kg.pipeline.executar import executar
from t2kg.pipeline.provedores.base import ProvedorBase
from t2kg.pipeline.provedores.mock import ProvedorMock

pytestmark = pytest.mark.skipif(not os.path.exists(BENCH), reason="Text2KGBench ausente")


class _ProvedorComFinish(ProvedorBase):
    def gerar(self, prompt, max_tokens=512, temperatura=0.0):
        return "rel(a, b)"

    def gerar_detalhado(self, prompt, max_tokens=512, temperatura=0.0):
        return "rel(a, b)", "length"


def test_c5_vizinho_sem_parceira_levanta_erro(tmp_path):
    saida = str(tmp_path / "c5.jsonl")
    with pytest.raises(ValueError) as e:
        executar(bench_dir=BENCH, dataset="tekgen", split="test", ontologia_id="1_movie",
                 condicao="C5_vizinho", provedor=ProvedorMock(), saida=saida,
                 frases_max=1, ontologia_parceira_id=None)
    assert "I-2" in str(e.value)


def test_finish_reason_registrado(tmp_path):
    saida = str(tmp_path / "c0.jsonl")
    executar(bench_dir=BENCH, dataset="tekgen", split="test", ontologia_id="1_movie",
             condicao="C0", provedor=_ProvedorComFinish(), saida=saida, frases_max=1)
    linhas = [json.loads(l) for l in open(saida, encoding="utf-8") if l.strip()]
    assert len(linhas) == 1
    assert linhas[0]["finish_reason"] == "length"


def test_mock_sem_finish_reason_nao_adiciona_campo(tmp_path):
    saida = str(tmp_path / "c0mock.jsonl")
    executar(bench_dir=BENCH, dataset="tekgen", split="test", ontologia_id="1_movie",
             condicao="C0", provedor=ProvedorMock(), saida=saida, frases_max=1)
    linhas = [json.loads(l) for l in open(saida, encoding="utf-8") if l.strip()]
    assert "finish_reason" not in linhas[0]
