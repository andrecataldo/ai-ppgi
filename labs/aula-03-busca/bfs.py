"""Breadth-First Search - Busca em Largura."""

from collections import deque

from graph import GRAPH, GOAL, START
from search_common import SearchResult


def bfs(
    graph=GRAPH,
    start: str = START,
    goal: str = GOAL,
) -> SearchResult:
    frontier = deque([(start, [start], 0)])
    expanded = 0
    iteration = 0
    trace = []

    while frontier:
        frontier_view = [
            f"{node}(custo={cost})"
            for node, _, cost in frontier
        ]

        node, path, cost = frontier.popleft()

        trace.append(
            {
                "iteration": iteration,
                "frontier": frontier_view,
                "selected": node,
                "details": "Critério: menor profundidade / ordem FIFO",
            }
        )

        if node == goal:
            return SearchResult(
                algorithm="BFS - Breadth-First Search",
                path=path,
                cost=cost,
                expanded=expanded,
                trace=trace,
            )

        expanded += 1

        for neighbor, edge_cost in graph[node]:
            frontier.append(
                (
                    neighbor,
                    path + [neighbor],
                    cost + edge_cost,
                )
            )

        iteration += 1

    raise ValueError("Objetivo não encontrado.")
