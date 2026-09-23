"""Compare search strategies on the same NUMPUZ I -> S problem."""

from heuristics import (
    manhattan_distance,
    misplaced_tiles,
)
from puzzle import (
    GOAL_STATE,
    SIZE,
    State,
    shuffle_state,
)
from solver import (
    SearchResult,
    astar,
    bfs,
    dfs,
)


def state_to_string(state: State) -> str:
    """Format a puzzle state for terminal display."""
    rows = []

    for row in range(SIZE):
        values = []

        for col in range(SIZE):
            tile = state[row * SIZE + col]

            values.append(
                "_" if tile == 0 else str(tile)
            )

        rows.append(
            " ".join(
                f"{value:>2}"
                for value in values
            )
        )

    return "\n".join(rows)


def print_results(
    results: list[SearchResult],
) -> None:
    """Print a compact comparison table."""
    print()
    print("=" * 100)
    print("COMPARAÇÃO DOS ALGORITMOS")
    print("=" * 100)

    print(
        f"{'Algoritmo':<40}"
        f"{'Custo':>8}"
        f"{'Expandidos':>14}"
        f"{'Gerados':>12}"
        f"{'Tempo (s)':>14}"
    )

    print("-" * 100)

    for result in results:
        print(
            f"{result.algorithm:<40}"
            f"{result.cost:>8}"
            f"{result.expanded:>14}"
            f"{result.generated:>12}"
            f"{result.elapsed_seconds:>14.6f}"
        )


def main() -> None:
    goal = GOAL_STATE

    initial = shuffle_state(
        moves=20,
        seed=42,
        start=goal,
    )

    print("=" * 50)
    print("NUMPUZ — Experimento de Busca I -> S")
    print("=" * 50)

    print("\nEstado inicial — I:\n")
    print(state_to_string(initial))

    print("\nEstado objetivo — S:\n")
    print(state_to_string(goal))

    results = [
        bfs(
            initial,
            goal,
        ),
        dfs(
            initial,
            goal,
        ),
        astar(
            initial,
            goal,
            heuristic=misplaced_tiles,
        ),
        astar(
            initial,
            goal,
            heuristic=manhattan_distance,
        ),
    ]

    print_results(results)

    optimal_cost = results[0].cost

    print()

    print(
        f"✓ Custo ótimo de referência (BFS): "
        f"{optimal_cost}."
    )

    if (
        results[2].cost == optimal_cost
        and results[3].cost == optimal_cost
    ):
        print(
            "✓ As duas configurações de A* "
            "encontraram o mesmo custo ótimo."
        )

    if results[1].cost == optimal_cost:
        print(
            "✓ Neste experimento, o DFS também "
            "encontrou uma solução de custo ótimo."
        )
    else:
        print(
            "ℹ O DFS encontrou uma solução com custo "
            f"{results[1].cost}. DFS não garante "
            "a solução de menor profundidade."
        )

    fewest = min(
        results,
        key=lambda result: result.expanded,
    )

    print(
        "✓ Menor número de estados expandidos: "
        f"{fewest.algorithm} "
        f"({fewest.expanded})."
    )


if __name__ == "__main__":
    main()