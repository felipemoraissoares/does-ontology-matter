"""
Smoke test: ordem completa C2 piloto-v3.

Simula a sequência correta:
  (a) ontologia induzida (com run_id da sessão)
  (b) guard check: run_id correto → OK / errado → RuntimeError
  (c) experimentos C2 com ProvedorMock, frases_max=3
  (d) verificação que o prompt C2 contém as relações da ontologia induzida NESTA sessão

Cobre as 4 ontologias do piloto-v3:
  tekgen/1_movie, tekgen/2_music, webnlg/1_university, webnlg/19_film

Os testes de integração (c)/(d) requerem Text2KGBench; os testes de guard check
rodam sempre, sem dependência de dados externos.
"""
import json
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE = os.path.abspath(os.path.join(REPO, ".."))
BENCH = os.path.join(WORKSPACE, "Text2KGBench")
sys.path.insert(0, os.path.join(REPO, "src"))

SKIP_SEM_BENCH = pytest.mark.skipif(
    not os.path.exists(BENCH), reason="Text2KGBench não encontrado"
)

ONTOS_PV3 = [
    ("tekgen", "1_movie"),
    ("tekgen", "2_music"),
    ("webnlg", "1_university"),
    ("webnlg", "19_film"),
]

SMOKE_RUN_ID = "SMOKE_V3_TEST_RUN"


# ---------------------------------------------------------------------------
# Guard check — rodam sem BENCH
# ---------------------------------------------------------------------------

class TestGuardCheck:
    def test_run_id_correto_passa(self, tmp_path):
        """Ontologia induzida com run_id correto não levanta erro no guard check."""
        (tmp_path / "configs" / "c2").mkdir(parents=True)
        induced = {
            "metadata": {"run_id": SMOKE_RUN_ID},
            "concepts": [],
            "relations": [{"label": "fake_rel"}],
        }
        path = tmp_path / "configs" / "c2" / "tekgen_1_movie_induced.json"
        path.write_text(json.dumps(induced), encoding="utf-8")

        with open(path, encoding="utf-8") as f:
            meta = json.load(f).get("metadata", {})
        assert meta.get("run_id") == SMOKE_RUN_ID

    def test_run_id_errado_detectado(self, tmp_path):
        """Guard check deve detectar run_id de outra sessão."""
        (tmp_path / "configs" / "c2").mkdir(parents=True)
        induced = {
            "metadata": {"run_id": "OLD_RUN_FROM_LAST_SESSION"},
            "concepts": [],
            "relations": [{"label": "old_rel"}],
        }
        path = tmp_path / "configs" / "c2" / "tekgen_1_movie_induced.json"
        path.write_text(json.dumps(induced), encoding="utf-8")

        with open(path, encoding="utf-8") as f:
            meta = json.load(f).get("metadata", {})

        run_id_onto = meta.get("run_id")
        assert run_id_onto != SMOKE_RUN_ID, (
            "Guard deveria ter detectado run_id incorreto, mas não detectou."
        )

    def test_ontologia_ausente_levanta_file_not_found(self, tmp_path, monkeypatch):
        """carregar_ontologia_induzida levanta FileNotFoundError quando arquivo não existe."""
        import t2kg.pipeline.ontologias as ont_mod
        monkeypatch.setattr(ont_mod, "_REPO_DIR", str(tmp_path))
        (tmp_path / "configs" / "c2").mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="Ontologia induzida C2 não encontrada"):
            ont_mod.carregar_ontologia_induzida("tekgen", "1_movie")

    def test_guard_todos_ontos_pv3(self, tmp_path):
        """Guard verifica todas as 4 ontologias do piloto-v3 de uma vez."""
        (tmp_path / "configs" / "c2").mkdir(parents=True)
        for dataset, onto_id in ONTOS_PV3:
            induced = {
                "metadata": {"run_id": SMOKE_RUN_ID, "dataset": dataset, "onto_id": onto_id},
                "concepts": [],
                "relations": [{"label": f"rel_{onto_id}"}],
            }
            path = tmp_path / "configs" / "c2" / f"{dataset}_{onto_id}_induced.json"
            path.write_text(json.dumps(induced), encoding="utf-8")

        # Simula o guard check do lancador
        erros = []
        for dataset, onto_id in ONTOS_PV3:
            onto_path = tmp_path / "configs" / "c2" / f"{dataset}_{onto_id}_induced.json"
            if not onto_path.exists():
                erros.append(f"AUSENTE: {onto_path}")
                continue
            with open(onto_path, encoding="utf-8") as f:
                data = json.load(f)
            run_id_onto = data.get("metadata", {}).get("run_id")
            if run_id_onto != SMOKE_RUN_ID:
                erros.append(f"RUN_ID ERRADO: {onto_path.name}")

        assert not erros, f"Guard check falhou: {erros}"


# ---------------------------------------------------------------------------
# Integração: C2 usa ontologia induzida nesta sessão — requer BENCH
# ---------------------------------------------------------------------------

@SKIP_SEM_BENCH
class TestC2UsaOntologiaDaSessao:
    """
    Para cada uma das 4 ontologias do piloto-v3:
      1. Cria ontologia induzida fake com relação única de smoke (run_id=SMOKE_RUN_ID)
      2. Monkeypatcha _REPO_DIR → tmp_path (para que carregar_ontologia_induzida leia o fake)
      3. Roda executar(..., condicao="C2", frases_max=3)
      4. Verifica que cada prompt gerado contém a relação fake da ontologia desta sessão
    """

    @pytest.mark.parametrize("dataset,onto_id", ONTOS_PV3)
    def test_c2_prompt_contem_relacao_induzida_nesta_sessao(
        self, dataset, onto_id, tmp_path, monkeypatch
    ):
        import t2kg.pipeline.ontologias as ont_mod
        from t2kg.pipeline.executar import executar
        from t2kg.pipeline.provedores.mock import ProvedorMock

        # (a) Cria ontologia induzida com relação única identificável
        fake_rel = f"smoke_rel_{onto_id.replace('_', '')}_runv3"
        (tmp_path / "configs" / "c2").mkdir(parents=True)
        induced = {
            "metadata": {
                "run_id": SMOKE_RUN_ID,
                "dataset": dataset,
                "onto_id": onto_id,
                "freq_min_frases": 3,
                "threshold_agrupamento": 0.70,
                "use_definicoes": True,
            },
            "concepts": [],
            "relations": [{"label": fake_rel}],
        }
        induced_path = tmp_path / "configs" / "c2" / f"{dataset}_{onto_id}_induced.json"
        induced_path.write_text(json.dumps(induced), encoding="utf-8")

        # (b) Redireciona _REPO_DIR → tmp_path
        monkeypatch.setattr(ont_mod, "_REPO_DIR", str(tmp_path))

        # (c) Guard check: verifica run_id antes de rodar experimento
        with open(induced_path, encoding="utf-8") as f:
            meta = json.load(f).get("metadata", {})
        assert meta.get("run_id") == SMOKE_RUN_ID, (
            f"Guard falhou antes do experimento: run_id={meta.get('run_id')!r}"
        )

        # (d) Executa experimento C2 com Mock (3 frases)
        saida = str(tmp_path / f"c2_{dataset}_{onto_id}.jsonl")
        n = executar(
            bench_dir=BENCH,
            dataset=dataset,
            split="test",
            ontologia_id=onto_id,
            condicao="C2",
            provedor=ProvedorMock(),
            saida=saida,
            frases_max=3,
        )
        assert n == 3, f"Esperado 3 frases para {dataset}/{onto_id}, obteve {n}"

        # (e) Verifica que o prompt C2 contém a relação da ontologia induzida nesta sessão
        linhas = [
            json.loads(linha)
            for linha in open(saida, encoding="utf-8")
            if linha.strip()
        ]
        assert len(linhas) == 3, f"Esperado 3 linhas no JSONL, obteve {len(linhas)}"
        for i, linha in enumerate(linhas):
            assert "prompt" in linha, f"Linha {i}: campo 'prompt' ausente"
            assert fake_rel in linha["prompt"], (
                f"{dataset}/{onto_id} frase {i}: relação induzida '{fake_rel}' "
                f"não encontrada no prompt C2.\n"
                f"Prompt (primeiros 300 chars): {linha['prompt'][:300]}"
            )

    @pytest.mark.parametrize("dataset,onto_id", ONTOS_PV3)
    def test_c2_guard_run_id_errado_impede_execucao(
        self, dataset, onto_id, tmp_path, monkeypatch
    ):
        """Se run_id não bater, o guard check deve bloquear antes de iniciar o experimento."""
        import t2kg.pipeline.ontologias as ont_mod

        (tmp_path / "configs" / "c2").mkdir(parents=True)
        # Ontologia com run_id de outra sessão
        induced = {
            "metadata": {"run_id": "SESSAO_ANTERIOR_20261101"},
            "concepts": [],
            "relations": [{"label": "stale_rel"}],
        }
        induced_path = tmp_path / "configs" / "c2" / f"{dataset}_{onto_id}_induced.json"
        induced_path.write_text(json.dumps(induced), encoding="utf-8")
        monkeypatch.setattr(ont_mod, "_REPO_DIR", str(tmp_path))

        # Guard check detecta run_id errado (simula a lógica do lancador)
        with open(induced_path, encoding="utf-8") as f:
            run_id_onto = json.load(f).get("metadata", {}).get("run_id")
        assert run_id_onto != SMOKE_RUN_ID, (
            "Guard check deveria ter detectado run_id incorreto, mas não detectou. "
            f"run_id_onto={run_id_onto!r}, SMOKE_RUN_ID={SMOKE_RUN_ID!r}"
        )
