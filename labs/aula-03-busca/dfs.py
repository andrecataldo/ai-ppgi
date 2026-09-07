"""Depth-First Search - Busca em Profundidade."""

from graph import GRAPH, GOAL, START
from search_common import SearchResult


def dfs(
    graph=GRAPH,
    start: str = START,
    goal: str = GOAL,
) -> SearchResult:
    stack = [(start, [start], 0)]
    expanded = 0
    iteration = 0
    trace = []

    while stack:
        # Mostra primeiro o elemento que será removido.
        frontier_view = [
            f"{node}(custo={cost})"
            for node, _, cost in reversed(stack)
        ]

        node, path, cost = stack.pop()

        trace.append(
            {
                "iteration": iteration,
                "frontier": frontier_view,
                "selected": node,
                "details": "Critério: aprofundar o caminho atual / ordem LIFO",
            }
        )

        if node == goal:
            return SearchResult(
                algorithm="DFS - Depth-First Search",
                path=path,
                cost=cost,
                expanded=expanded,
                trace=trace,
            )

        expanded += 1

        # Reverso para que A seja visitado antes de B e C.
        for neighbor, edge_cost in reversed(graph[node]):
            stack.append(
                (
                    neighbor,
                    path + [neighbor],
                    cost + edge_cost,
                )
            )

        iteration += 1

    raise ValueError("Objetivo não encontrado.")
