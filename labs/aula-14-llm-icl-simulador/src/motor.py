"""Simulador auditável de predição condicionada por demonstrações.

IMPORTANTE: isto é uma analogia didática do ICL, não um Transformer/LLM.
O algoritmo é determinístico e não treina parâmetros.
"""

from dataclasses import dataclass
from math import exp


@dataclass(frozen=True)
class Exemplo:
    palavra: str
    rotulo: str


# 01. Base pequena, construída para a aula, com rótulos inventados.
BASE = (
    Exemplo("XAR-11", "LUM"),
    Exemplo("TOV-08", "NEX"),
    Exemplo("XAR-30", "LUM"),
    Exemplo("TOV-17", "NEX"),
)


# 02. Retira espaços e padroniza a comparação.
def normalizar(palavra: str) -> str:
    return palavra.strip().lower()


# 03. Define quais características o simulador consegue observar.
def caracteristicas(palavra: str) -> dict[str, str]:
    palavra = normalizar(palavra)
    return {"prefixo": palavra[:3], "sufixo": palavra[-2:]}


# 04. Seleciona K exemplos; inverter muda o rótulo, não a palavra.
def demonstracoes(k: int, invertidos: bool = False, ordem_inversa: bool = False) -> list[Exemplo]:
    if k not in (0, 1, 2, 4):
        raise ValueError("K permitido: 0, 1, 2 ou 4")
    escolhidos = list(BASE[:k])
    if invertidos:
        trocado = {"LUM": "NEX", "NEX": "LUM"}
        escolhidos = [Exemplo(x.palavra, trocado[x.rotulo]) for x in escolhidos]
    if ordem_inversa:
        escolhidos.reverse()
    return escolhidos


# 05. Calcula uma similaridade INTENCIONALMENTE simples.
def similaridade(consulta: str, exemplo: Exemplo, peso_prefixo: float = 2.0,
                 peso_sufixo: float = 0.5) -> tuple[float, bool, bool]:
    q = caracteristicas(consulta)
    e = caracteristicas(exemplo.palavra)
    mesmo_prefixo = q["prefixo"] == e["prefixo"]
    mesmo_sufixo = q["sufixo"] == e["sufixo"]
    pontuacao = peso_prefixo * int(mesmo_prefixo) + peso_sufixo * int(mesmo_sufixo)
    return pontuacao, mesmo_prefixo, mesmo_sufixo


# 06. Softmax converte pontuações em pesos não negativos que somam 1.
def softmax(valores: list[float]) -> list[float]:
    if not valores:
        return []
    maior = max(valores)
    escalados = [exp(v - maior) for v in valores]  # Evita estouro numérico.
    total = sum(escalados)
    return [valor / total for valor in escalados]


# 07. Soma os pesos por rótulo; NÃO estima probabilidades calibradas.
def prever(consulta: str, exemplos: list[Exemplo], peso_prefixo: float = 2.0,
           peso_sufixo: float = 0.5, vies_posicao: float = 0.0) -> dict:
    if not normalizar(consulta):
        raise ValueError("Informe uma palavra para consultar")
    if not exemplos:  # Zero-shot: divisão neutra por convenção didática.
        return {"passos": [], "pesos_por_classe": {"LUM": 0.5, "NEX": 0.5},
                "predicao": "INDETERMINADO", "motivo": "Sem demonstrações"}

    passos = []
    n = len(exemplos)
    for i, exemplo in enumerate(exemplos):
        base, pref, suf = similaridade(consulta, exemplo, peso_prefixo, peso_sufixo)
        efeito_posicao = vies_posicao * i / max(n - 1, 1)
        passos.append({
            "exemplo": exemplo.palavra, "rotulo": exemplo.rotulo,
            "prefixo_igual": pref, "sufixo_igual": suf,
            "similaridade": round(base, 5), "posicao": round(efeito_posicao, 5),
            "pontuacao": round(base + efeito_posicao, 5),
        })

    pesos = softmax([x["pontuacao"] for x in passos])
    pesos_por_classe = {"LUM": 0.0, "NEX": 0.0}
    for passo, peso in zip(passos, pesos):
        passo["atencao"] = peso
        pesos_por_classe[passo["rotulo"]] += peso

    diferenca = pesos_por_classe["LUM"] - pesos_por_classe["NEX"]
    predicao = "INDETERMINADO" if abs(diferenca) < 1e-9 else "LUM" if diferenca > 0 else "NEX"
    return {"passos": passos, "pesos_por_classe": pesos_por_classe, "predicao": predicao,
            "motivo": "Soma dos pesos normalizados por rótulo"}

