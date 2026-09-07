"""Uniform Cost Search - Busca de Custo Uniforme."""

import heapq
import itertools

from graph import GRAPH, GOAL, START
from search_common import SearchResult


def ucs(
    graph=GRAPH,
    start: str = START,
    goal: str = GOAL,
) -> SearchResult:
    counter = itertools.count()

    frontier = [
        (0, next(counter), start, [start])
    ]

    expanded = 0
    iteration = 0
    trace = []

    while frontier:
        frontier_view = [
            f"{node}(g={cost})"
            for cost, _, node, _ in sorted(frontier)
        ]

        cost, _, node, path = heapq.heappop(frontier)

        trace.append(
            {
                "iteration": iteration,
                "frontier": frontier_view,
                "selected": node,
                "details": f"Critério: menor custo acumulado g(n) = {cost}",
            }
        )

        if node == goal:
            return SearchResult(
                algorithm="UCS - Uniform Cost Search",
                path=path,
                cost=cost,
                expanded=expanded,
                trace=trace,
            )

        expanded += 1

        for neighbor, edge_cost in graph[node]:
            new_cost = cost + edge_cost

            heapq.heappush(
                frontier,
                (
                    new_cost,
                    next(counter),
                    neighbor,
                    path + [neighbor],
                )
            )

        iteration += 1

    raise ValueError("Objetivo não encontrado.")
