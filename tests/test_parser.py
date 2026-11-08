"""
Testes do parser de respostas LLM.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from t2kg.parser import parsear_resposta


class TestParserCanônico:
    def test_tripla_simples(self):
        resp = "director(Bleach: Hell Verse,Noriyuki Abe)"
        assert parsear_resposta(resp) == [["Bleach: Hell Verse", "director", "Noriyuki Abe"]]

    def test_tripla_com_espaco(self):
        resp = "cast_member(Toy Story, Tom Hanks)"
        assert parsear_resposta(resp) == [["Toy Story", "cast_member", "Tom Hanks"]]

    def test_multiplas_linhas(self):
        resp = "director(A,B)\nscreenwriter(A,C)"
        result = parsear_resposta(resp)
        assert ["A", "director", "B"] in result
        assert ["A", "screenwriter", "C"] in result

    def test_escapes_markdown(self):
        resp = "cast\\_member(Toy Story,Tom Hanks)"
        assert parsear_resposta(resp) == [["Toy Story", "cast_member", "Tom Hanks"]]

    def test_prefixo_lista(self):
        resp = "- director(A,B)\n* screenwriter(A,C)"
        result = parsear_resposta(resp)
        assert ["A", "director", "B"] in result
        assert ["A", "screenwriter", "C"] in result

    def test_objeto_vazio_descartado(self):
        resp = "director(A,)"
        assert parsear_resposta(resp) == []

    def test_sem_virgula_descartado(self):
        resp = "director(A)"
        assert parsear_resposta(resp) == []

    def test_resposta_vazia(self):
        assert parsear_resposta("") == []
        assert parsear_resposta(None) == []


class TestParserMultiplasTriplas:
    def test_multiplas_mesma_linha(self):
        resp = "director(A,B), screenwriter(A,C)"
        result = parsear_resposta(resp)
        assert ["A", "director", "B"] in result
        assert ["A", "screenwriter", "C"] in result

    def test_sem_multiplas_em_linha_normal(self):
        # Linha normal com vírgula dentro dos argumentos não deve ser dividida errado
        resp = "director(Toy Story, John Lasseter)"
        result = parsear_resposta(resp)
        assert len(result) == 1
        assert result[0] == ["Toy Story", "director", "John Lasseter"]


class TestParserAspas:
    def test_aspas_duplas_no_objeto_removidas(self):
        resp = 'latinName(Univ,"Instituti")'
        result = parsear_resposta(resp)
        assert result == [["Univ", "latinName", "Instituti"]]

    def test_aspas_no_sujeito_removidas(self):
        resp = 'director("Film A",Director B)'
        result = parsear_resposta(resp)
        assert result == [["Film A", "director", "Director B"]]


class TestParserProsaIgnorada:
    def test_linha_em_prosa_ignorada(self):
        resp = "Here is the answer.\ndirector(Film A,Director B)"
        result = parsear_resposta(resp)
        assert result == [["Film A", "director", "Director B"]]

    def test_so_prosa(self):
        resp = "I cannot extract any triples from this sentence."
        assert parsear_resposta(resp) == []
