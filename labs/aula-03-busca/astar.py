"""A* Search."""

import heapq
import itertools

from graph import GRAPH, GOAL, HEURISTIC, START
from search_common import SearchResult


def astar(
    graph=GRAPH,
    heuristic=HEURISTIC,
    start: str = START,
    goal: str = GOAL,
) -> SearchResult:
    counter = itertools.count()

    start_g = 0
    start_h = heuristic[start]
    start_f = start_g + start_h

    frontier = [
        (
            start_f,
            start_h,
            next(counter),
            start_g,
            start,
            [start],
        )
    ]

    expanded = 0
    iteration = 0
    trace = []

    while frontier:
        frontier_view = [
            f"{node}(g={g}, h={h}, f={f})"
            for f, h, _, g, node, _ in sorted(frontier)
        ]

        f, h, _, g, node, path = heapq.heappop(frontier)

        trace.append(
            {
                "iteration": iteration,
                "frontier": frontier_view,
                "selected": node,
                "details": (
                    f"Critério: menor f(n)=g(n)+h(n): "
                    f"{g}+{h}={f}"
                ),
            }
        )

        if node == goal:
            return SearchResult(
                algorithm="A* Search",
                path=path,
                cost=g,
                expanded=expanded,
                trace=trace,
            )

        expanded += 1

        for neighbor, edge_cost in graph[node]:
            new_g = g + edge_cost
            new_h = heuristic[neighbor]
            new_f = new_g + new_h

            heapq.heappush(
                frontier,
                (
                    new_f,
                    new_h,
                    next(counter),
                    new_g,
                    neighbor,
                    path + [neighbor],
                )
            )

        iteration += 1

    raise ValueError("Objetivo não encontrado.")
