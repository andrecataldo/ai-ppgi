"""Estruturas e funções compartilhadas pelos algoritmos de busca."""

from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    algorithm: str
    path: list[str]
    cost: int
    expanded: int
    trace: list[dict[str, Any]]


def path_to_str(path: list[str]) -> str:
    return " -> ".join(path)


def print_result(result: SearchResult) -> None:
    print()
    print("=" * 72)
    print(result.algorithm)
    print("=" * 72)

    for step in result.trace:
        print(f"\nIteração {step['iteration']}")
        print(f"Frontier: {step['frontier']}")
        print(f"Selecionado: {step['selected']}")

        if "details" in step:
            print(step["details"])

    print("\nResultado")
    print("-" * 72)
    print(f"Caminho:   {path_to_str(result.path)}")
    print(f"Custo:     {result.cost}")
    print(f"Expandidos:{result.expanded}")
