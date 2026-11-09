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


class TestParserVirgulaFinal:
    def test_virgula_apos_parentese_final(self):
        # Categoria J: "rel(sub, obj)," — vírgula após ) é ignorada
        resp = "director(Film A, Director B),"
        assert parsear_resposta(resp) == [["Film A", "director", "Director B"]]

    def test_multiplas_com_virgula_final(self):
        # Padrão real do Vicuna: cada tripla termina com vírgula
        resp = (
            "directs(Kanta Kamei, Tales of Vesperia: The First Strike),\n"
            "writes(Reiko Yoshida, Tales of Vesperia: The First Strike),\n"
            "genre(Tales of Vesperia: The First Strike, fantasy),"
        )
        result = parsear_resposta(resp)
        assert ["Kanta Kamei", "directs", "Tales of Vesperia: The First Strike"] in result
        assert ["Reiko Yoshida", "writes", "Tales of Vesperia: The First Strike"] in result
        assert ["Tales of Vesperia: The First Strike", "genre", "fantasy"] in result

    def test_ponto_e_virgula_final(self):
        resp = "director(Film A, Director B);"
        assert parsear_resposta(resp) == [["Film A", "director", "Director B"]]


class TestParserProsaIgnorada:
    def test_linha_em_prosa_ignorada(self):
        resp = "Here is the answer.\ndirector(Film A,Director B)"
        result = parsear_resposta(resp)
        assert result == [["Film A", "director", "Director B"]]

    def test_so_prosa(self):
        resp = "I cannot extract any triples from this sentence."
        assert parsear_resposta(resp) == []


class TestParserCategoriaH:
    """Categoria H: prefixos de resposta que o modelo adiciona antes das triplas."""

    def test_test_output_prefix(self):
        resp = "Test Output: director(Film A, Director B)"
        assert parsear_resposta(resp) == [["Film A", "director", "Director B"]]

    def test_output_prefix(self):
        resp = "Output: screenwriter(Film A, Writer C)"
        assert parsear_resposta(resp) == [["Film A", "screenwriter", "Writer C"]]

    def test_answer_prefix(self):
        resp = "Answer: genre(Film A, Action)"
        assert parsear_resposta(resp) == [["Film A", "genre", "Action"]]

    def test_prefix_case_insensitive(self):
        resp = "OUTPUT: director(Film A, Director B)"
        assert parsear_resposta(resp) == [["Film A", "director", "Director B"]]

    def test_prefix_seguido_de_multiplas(self):
        resp = "Test Output: director(Film A, Dir B)\nscreenwriter(Film A, Sc C)"
        result = parsear_resposta(resp)
        assert ["Film A", "director", "Dir B"] in result
        assert ["Film A", "screenwriter", "Sc C"] in result


class TestParserCategoriaG:
    """Categoria G: tripla distribuída em múltiplas linhas."""

    def test_tripla_em_duas_linhas(self):
        resp = "director(Film A,\nDirector B)"
        assert parsear_resposta(resp) == [["Film A", "director", "Director B"]]

    def test_tripla_em_tres_linhas(self):
        resp = "director(\n  Film A,\n  Director B\n)"
        result = parsear_resposta(resp)
        assert len(result) == 1
        assert result[0][1] == "director"

    def test_tripla_multilinhas_seguida_de_normal(self):
        resp = "director(\n  Film A, Director B)\ngenre(Film A, Action)"
        result = parsear_resposta(resp)
        assert ["Film A", "genre", "Action"] in result


class TestParserCasosReaisD19:
    """Casos reais encontrados no piloto — bugs corrigidos em D-19."""

    def test_relacao_com_espaco_ja_funciona(self):
        """directed by(X, Y) já funcionava antes de D-19 — confirmação explícita."""
        assert parsear_resposta("directed by(The Matrix, Wachowskis)") == [
            ["The Matrix", "directed by", "Wachowskis"]
        ]

    def test_relacao_com_hifen_ja_funciona(self):
        assert parsear_resposta("based-on(The Dark Knight, Batman)") == [
            ["The Dark Knight", "based-on", "Batman"]
        ]

    def test_objeto_com_parenteses_aninhados(self):
        """D-19: objeto com parênteses aninhados não divide incorretamente."""
        r = parsear_resposta("is_a_song_by(Pavement, Summer Babe (Winter Version))")
        assert r == [["Pavement", "is_a_song_by", "Summer Babe (Winter Version)"]]

    def test_trailing_comma_no_objeto(self):
        """D-19: trailing comma antes de ) é removida."""
        r = parsear_resposta("director(Film A, Person B,)")
        assert r == [["Film A", "director", "Person B"]]

    def test_bullet_numero_parenteses(self):
        """D-19: '1) rel(a, b)' — remoção de bullet com parênteses."""
        r = parsear_resposta("1) director(The Matrix, Wachowskis)")
        assert r == [["The Matrix", "director", "Wachowskis"]]

    def test_multiplas_triplas_sem_dividir_dentro_do_objeto(self):
        """D-19: duas triplas na mesma linha não dividem argumentos internos."""
        r = parsear_resposta("director(Matrix, W), screenwriter(Matrix, W)")
        assert len(r) == 2

    def test_prose_pura_retorna_vazio(self):
        """Prosa pura sem rel(X,Y) retorna [] — comportamento correto."""
        assert parsear_resposta("The film revolves around a detective.") == []

    def test_relacao_longa_com_espacos_e_hifen(self):
        """'eight-minute animated film(X, Y)' — relação longa com espaços e hífens."""
        r = parsear_resposta("eight-minute animated film(Bambi, Forest)")
        assert r == [["Bambi", "eight-minute animated film", "Forest"]]

    def test_objeto_aninhado_mais_segunda_tripla(self):
        """D-19: objeto aninhado seguido de segunda tripla na mesma linha."""
        r = parsear_resposta(
            "is_a_song_by(Pavement, Summer Babe (Winter Version)), "
            "genre(Pavement, Indie Rock)"
        )
        assert ["Pavement", "is_a_song_by", "Summer Babe (Winter Version)"] in r
        assert ["Pavement", "genre", "Indie Rock"] in r
