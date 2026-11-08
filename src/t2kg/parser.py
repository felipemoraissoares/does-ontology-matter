# Parser de respostas LLM → lista de triplas [[sub, rel, obj], ...]
# O formato esperado pelo Text2KGBench é: relacao(sujeito, objeto)
#
# Estratégias aplicadas em ordem:
#   1. Padrão canônico linha a linha: rel(sub, obj) com escapes markdown removidos
#   2. Múltiplas triplas na mesma linha separadas por ", rel(" — divididas antes
#   3. Aspas externas em sub/obj são removidas
#   4. Triplas com objeto vazio são descartadas (rel(sub,) ou rel(sub))
#   5. Linhas em prosa (sem parênteses) são ignoradas
#   6. (Cat H) Prefixos "Test Output:", "Output:", "Answer:", "Triples:" removidos
#   7. (Cat G) Triplas multi-linha (rel( \n sub, obj \n )) são reunidas

import re
from typing import List, Optional, Tuple

# Prefixos que modelos de chat frequentemente antepõem ao conteúdo (Categoria H)
_RE_PREFIXO_RESPOSTA = re.compile(
    r"^(?:Test\s+Output|Output|Triples|Answer|Result)\s*:\s*",
    re.IGNORECASE,
)


def parsear_resposta(resposta: str) -> List[List[str]]:
    """
    Converte a string de resposta bruta de um LLM em lista de triplas
    [[sub, rel, obj], ...].
    """
    if not resposta or not isinstance(resposta, str):
        return []

    # Remove escapes markdown: \_ → _  \( → (  \) → )
    texto = resposta.replace("\\_", "_").replace("\\(", "(").replace("\\)", ")")

    # Categoria G: junta linhas de triplas incompletas (abertura sem fechamento)
    texto = _juntar_triplas_multiplas_linhas(texto)

    triplas = []
    for linha in texto.splitlines():
        linha = linha.strip()
        # Categoria H: remove prefixos "Test Output:", "Output:", etc.
        linha = _RE_PREFIXO_RESPOSTA.sub("", linha).strip()
        # Remove prefixos de lista: "- ", "* ", "• ", "1. "
        linha = re.sub(r"^[-*•]\s+", "", linha)
        linha = re.sub(r"^\d+\.\s+", "", linha)

        # Divide linha com múltiplas triplas: "rel1(a,b), rel2(c,d)"
        for segmento in _dividir_multiplas(linha):
            tripla = _extrair_tripla(segmento)
            if tripla is not None:
                triplas.append(list(tripla))

    return triplas


def _juntar_triplas_multiplas_linhas(texto: str) -> str:
    """
    Categoria G: detecta linhas com '(' mas sem ')' e junta com a(s) próxima(s)
    até o parêntese fechar. Exemplo:
      director(\\n  Film A,\\n  Person B\\n)  →  director(Film A, Person B)
    """
    linhas = texto.splitlines()
    resultado: List[str] = []
    buffer: Optional[str] = None

    for linha in linhas:
        if buffer is not None:
            buffer += " " + linha.strip()
            if ")" in buffer:
                resultado.append(buffer)
                buffer = None
        elif "(" in linha and ")" not in linha:
            buffer = linha
        else:
            resultado.append(linha)

    if buffer is not None:
        resultado.append(buffer)

    return "\n".join(resultado)


def _dividir_multiplas(linha: str) -> List[str]:
    """
    Divide uma linha que contém múltiplas triplas do formato rel(sub,obj), rel2(sub,obj).
    Usa a heurística: divide antes de um identificador_de_relação( que segue ),
    """
    # Padrão: ", identificador(" onde identificador não contém vírgulas ou parênteses
    partes = re.split(r",\s*(?=[A-Za-z_][^,()\n]*\()", linha)
    # Se a divisão produziu apenas 1 parte, retorna a linha original
    return partes if len(partes) > 1 else [linha]


def _extrair_tripla(linha: str) -> Optional[Tuple[str, str, str]]:
    """
    Tenta extrair (sub, rel, obj) de um fragmento no formato rel(sub, obj).
    Retorna None se não conseguir ou se obj estiver vazio.
    """
    if not linha:
        return None

    # Padrão: identificador_de_relação(conteúdo) com vírgula/ponto-e-vírgula final opcionais
    m = re.match(r"^([A-Za-z_][^\(]*)\((.+)\)\s*[,;]?\s*$", linha)
    if not m:
        return None

    rel_raw = m.group(1).strip()
    conteudo = m.group(2)

    sub, obj = _dividir_argumentos(conteudo)
    if sub is None or obj is None:
        return None

    sub = _limpar_string(sub)
    obj = _limpar_string(obj)

    # Descarta triplas com sujeito ou objeto vazios
    if not sub or not obj:
        return None

    return sub, rel_raw, obj


def _dividir_argumentos(conteudo: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Divide 'sub, obj' no primeiro vírgula de nível superior
    (fora de parênteses aninhados).
    """
    profundidade = 0
    for i, c in enumerate(conteudo):
        if c == "(":
            profundidade += 1
        elif c == ")":
            profundidade -= 1
        elif c == "," and profundidade == 0:
            return conteudo[:i], conteudo[i + 1:]
    return None, None


def _limpar_string(s: str) -> str:
    """Remove espaços externos e aspas duplas/simples externas."""
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()
    elif len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        s = s[1:-1].strip()
    return s
