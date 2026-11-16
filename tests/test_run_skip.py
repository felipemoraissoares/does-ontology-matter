"""
C-2: --skip_existentes só considera concluído o experimento cujo JSONL tem o
número esperado de frases válidas. Arquivo parcial é retomado, nunca pulado.
"""
import json
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE = os.path.abspath(os.path.join(REPO_ROOT, ".."))
BENCH = os.path.join(WORKSPACE, "Text2KGBench")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from t2kg.run import _contar_registros_validos, main as run_main


class TestContarRegistrosValidos:
    def test_conta_validos(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_text('{"id":"a"}\n{"id":"b"}\n', encoding="utf-8")
        assert _contar_registros_validos(str(p)) == 2

    def test_ignora_linha_truncada_final(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_text('{"id":"a"}\n{"id":"b"}\n{"id":"c"', encoding="utf-8")
        assert _contar_registros_validos(str(p)) == 2

    def test_arquivo_ausente(self, tmp_path):
        assert _contar_registros_validos(str(tmp_path / "nope.jsonl")) == 0


def _escrever_manifesto(tmp_path, frases_max):
    saida = str(tmp_path / "exp.jsonl").replace("\\", "/")
    m = [{
        "id": "t", "dataset": "tekgen", "split": "test", "ontologia_id": "1_movie",
        "condicao": "C0", "provedor": "mock", "saida": saida, "frases_max": frases_max,
    }]
    mp = tmp_path / f"manifesto_{frases_max}.json"
    mp.write_text(json.dumps(m), encoding="utf-8")
    return str(mp), saida


@pytest.mark.skipif(not os.path.exists(BENCH), reason="Text2KGBench ausente")
class TestSkipExistentes:
    def _run(self, manifesto):
        old = sys.argv
        sys.argv = ["t2kg.run", "--manifesto", manifesto,
                    "--bench_dir", BENCH, "--skip_existentes"]
        try:
            run_main()
        finally:
            sys.argv = old

    def test_completo_e_pulado(self, tmp_path, capsys):
        manifesto, saida = _escrever_manifesto(tmp_path, frases_max=2)
        self._run(manifesto)
        assert _contar_registros_validos(saida) == 2
        self._run(manifesto)
        assert "PULADO" in capsys.readouterr().out
        assert _contar_registros_validos(saida) == 2

    def test_parcial_e_retomado(self, tmp_path, capsys):
        # Gera 1 frase (parcial em relação ao esperado=3)
        manifesto1, saida = _escrever_manifesto(tmp_path, frases_max=1)
        self._run(manifesto1)
        assert _contar_registros_validos(saida) == 1
        # Reaponta a mesma saída com esperado=3 → deve retomar, não pular
        saida2 = str(tmp_path / "exp.jsonl").replace("\\", "/")
        m = [{
            "id": "t", "dataset": "tekgen", "split": "test", "ontologia_id": "1_movie",
            "condicao": "C0", "provedor": "mock", "saida": saida2, "frases_max": 3,
        }]
        mp = tmp_path / "manifesto_3.json"
        mp.write_text(json.dumps(m), encoding="utf-8")
        self._run(str(mp))
        assert "RETOMADA" in capsys.readouterr().out
        assert _contar_registros_validos(saida) == 3
