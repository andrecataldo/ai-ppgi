"""Executa BFS, DFS, UCS, Greedy e A* sobre o mesmo problema."""

from astar import astar
from bfs import bfs
from dfs import dfs
from graph import OPTIMAL_COST
from greedy import greedy
from search_common import path_to_str, print_result
from ucs import ucs


def main() -> None:
    algorithms = [
        bfs,
        dfs,
        ucs,
        greedy,
        astar,
    ]

    results = []

    for algorithm in algorithms:
        result = algorithm()
        results.append(result)
        print_result(result)

    print()
    print("=" * 88)
    print("COMPARAÇÃO FINAL")
    print("=" * 88)

    print(
        f"{'Algoritmo':<30}"
        f"{'Caminho':<20}"
        f"{'Custo':>8}"
        f"{'Expandidos':>14}"
        f"{'Ótima?':>10}"
    )

    print("-" * 88)

    for result in results:
        optimal = "SIM" if result.cost == OPTIMAL_COST else "NÃO"

        print(
            f"{result.algorithm:<30}"
            f"{path_to_str(result.path):<20}"
            f"{result.cost:>8}"
            f"{result.expanded:>14}"
            f"{optimal:>10}"
        )

    print()
    print(f"Custo ótimo conhecido do problema: {OPTIMAL_COST}")


if __name__ == "__main__":
    main()
