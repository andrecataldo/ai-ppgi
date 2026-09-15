"""Compare search strategies on the same NUMPUZ instance."""

from heuristics import (
    manhattan_distance,
    misplaced_tiles,
)
from puzzle import SIZE, State, shuffle_state
from solver import SearchResult, astar, bfs


def state_to_string(state: State) -> str:
    """Format a puzzle state for terminal display."""
    rows = []

    for row in range(SIZE):
        values = []

        for col in range(SIZE):
            tile = state[row * SIZE + col]
            values.append(" " if tile == 0 else str(tile))

        rows.append(" ".join(f"{value:>2}" for value in values))

    return "\n".join(rows)


def print_results(results: list[SearchResult]) -> None:
    """Print a compact comparison table."""
    print()
    print("=" * 96)
    print("COMPARAÇÃO DOS ALGORITMOS")
    print("=" * 96)

    print(
        f"{'Algoritmo':<36}"
        f"{'Custo':>8}"
        f"{'Expandidos':>14}"
        f"{'Gerados':>12}"
        f"{'Tempo (s)':>14}"
    )

    print("-" * 96)

    for result in results:
        print(
            f"{result.algorithm:<36}"
            f"{result.cost:>8}"
            f"{result.expanded:>14}"
            f"{result.generated:>12}"
            f"{result.elapsed_seconds:>14.6f}"
        )


def main() -> None:
    state = shuffle_state(
        moves=20,
        seed=42,
    )

    print("=" * 40)
    print("NUMPUZ — Experimento de Busca")
    print("=" * 40)

    print("\nEstado inicial:\n")
    print(state_to_string(state))

    results = [
        bfs(state),
        astar(
            state,
            heuristic=misplaced_tiles,
        ),
        astar(
            state,
            heuristic=manhattan_distance,
        ),
    ]

    print_results(results)

    costs = {
        result.cost
        for result in results
    }

    print()

    if len(costs) == 1:
        print(
            "✓ Todos os algoritmos encontraram "
            "uma solução com o mesmo custo ótimo."
        )
    else:
        print(
            "⚠ Os algoritmos retornaram soluções "
            "com custos diferentes."
        )

    best = min(
        results,
        key=lambda result: result.expanded,
    )

    print(
        f"✓ Menor número de estados expandidos: "
        f"{best.algorithm} ({best.expanded})."
    )


if __name__ == "__main__":
    main()