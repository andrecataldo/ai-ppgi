"""Greedy Best-First Search - Busca Gulosa."""

import heapq
import itertools

from graph import GRAPH, GOAL, HEURISTIC, START
from search_common import SearchResult


def greedy(
    graph=GRAPH,
    heuristic=HEURISTIC,
    start: str = START,
    goal: str = GOAL,
) -> SearchResult:
    counter = itertools.count()

    frontier = [
        (
            heuristic[start],
            next(counter),
            start,
            [start],
            0,
        )
    ]

    expanded = 0
    iteration = 0
    trace = []

    while frontier:
        frontier_view = [
            f"{node}(h={h}, custo={cost})"
            for h, _, node, _, cost in sorted(frontier)
        ]

        h, _, node, path, cost = heapq.heappop(frontier)

        trace.append(
            {
                "iteration": iteration,
                "frontier": frontier_view,
                "selected": node,
                "details": (
                    f"Critério: menor heurística h(n) = {h}; "
                    f"custo acumulado g(n) = {cost} é ignorado na prioridade"
                ),
            }
        )

        if node == goal:
            return SearchResult(
                algorithm="Greedy Best-First Search",
                path=path,
                cost=cost,
                expanded=expanded,
                trace=trace,
            )

        expanded += 1

        for neighbor, edge_cost in graph[node]:
            heapq.heappush(
                frontier,
                (
                    heuristic[neighbor],
                    next(counter),
                    neighbor,
                    path + [neighbor],
                    cost + edge_cost,
                )
            )

        iteration += 1

    raise ValueError("Objetivo não encontrado.")
