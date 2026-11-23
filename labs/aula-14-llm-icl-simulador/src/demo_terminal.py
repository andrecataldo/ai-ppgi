"""Demonstração sem interface e sem dependências: python3 src/demo_terminal.py.

Mostra todas as contas feitas pelo motor para o mesmo prompt, passo a passo.
"""

from motor import demonstracoes, prever


def exibir_cenario(k: int, inverter: bool = False) -> None:
    """Executa um cenário e imprime as contas relevantes em formato tabular."""
    exemplos = demonstracoes(k, invertidos=inverter)
    resultado = prever("TOV-42", exemplos)

    print(f"\n{'=' * 65}")
    print(f"Consulta: TOV-42 | K={k} | rótulos invertidos={inverter}")
    print("Demonstrações:", [(e.palavra, e.rotulo) for e in exemplos])
    print("exemplo   rótulo  prefixo  sufixo  escore  atenção")
    for p in resultado["passos"]:
        print(
            f"{p['exemplo']:<9} {p['rotulo']:<7} "
            f"{str(p['prefixo_igual']):<8} {str(p['sufixo_igual']):<7} "
            f"{p['pontuacao']:<7.2f} {p['atencao']:.3f}"
        )
    for classe, peso in resultado["pesos_por_classe"].items():
        print(f"Peso agregado de {classe}: {peso:.1%} (não é confiança calibrada)")
    print("PREVISÃO:", resultado["predicao"])


if __name__ == "__main__":
    print("SIMULADOR DE PREDIÇÃO CONDICIONADA — NÃO É UM LLM REAL")
    print("Regra explícita: 2 pontos por prefixo igual, 0,5 por sufixo igual.")
    for quantidade in (0, 1, 2, 4):
        exibir_cenario(quantidade)
    exibir_cenario(4, inverter=True)
