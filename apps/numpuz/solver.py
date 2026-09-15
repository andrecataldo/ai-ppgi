"""Search algorithms for the NUMPUZ sliding puzzle."""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from math import inf
from time import perf_counter

from heuristics import manhattan_distance
from puzzle import (
    GOAL_STATE,
    Move,
    State,
    is_solvable,
    neighbors,
    validate_state,
)


@dataclass(frozen=True)
class SearchResult:
    """Result returned by a NUMPUZ search algorithm."""

    algorithm: str
    path: tuple[State, ...]
    moves: tuple[Move, ...]
    cost: int
    expanded: int
    generated: int
    elapsed_seconds: float


ParentInfo = tuple[State, Move] | None

HeuristicFunction = Callable[
    [State, State],
    int,
]


def reconstruct_path(
    parents: dict[State, ParentInfo],
    goal: State,
) -> tuple[tuple[State, ...], tuple[Move, ...]]:
    """Reconstruct states and movements from start to goal."""
    states: list[State] = []
    moves: list[Move] = []

    current = goal

    while True:
        states.append(current)

        parent_info = parents[current]

        if parent_info is None:
            break

        parent, move = parent_info

        moves.append(move)
        current = parent

    states.reverse()
    moves.reverse()

    return tuple(states), tuple(moves)


def bfs(
    start: State,
    goal: State = GOAL_STATE,
) -> SearchResult:
    """Solve a NUMPUZ instance using Breadth-First Search.

    Since every puzzle movement has cost 1, BFS returns an optimal
    solution in number of movements.
    """
    validate_state(start)
    validate_state(goal)

    if not is_solvable(start, goal):
        raise ValueError(
            "O estado inicial não possui solução para o objetivo informado."
        )

    started_at = perf_counter()

    frontier: deque[State] = deque([start])

    discovered: set[State] = {
        start,
    }

    parents: dict[State, ParentInfo] = {
        start: None,
    }

    expanded = 0
    generated = 1

    while frontier:
        current = frontier.popleft()

        if current == goal:
            path, moves = reconstruct_path(
                parents,
                current,
            )

            elapsed = perf_counter() - started_at

            return SearchResult(
                algorithm="Breadth-First Search (BFS)",
                path=path,
                moves=moves,
                cost=len(moves),
                expanded=expanded,
                generated=generated,
                elapsed_seconds=elapsed,
            )

        expanded += 1

        for move, successor in neighbors(current):
            if successor in discovered:
                continue

            discovered.add(successor)

            parents[successor] = (
                current,
                move,
            )

            frontier.append(successor)

            generated += 1

    raise RuntimeError(
        "A busca terminou sem encontrar o estado objetivo."
    )


def astar(
    start: State,
    goal: State = GOAL_STATE,
    *,
    heuristic: HeuristicFunction = manhattan_distance,
) -> SearchResult:
    """Solve NUMPUZ using A* Search.

    The priority function is:

        f(n) = g(n) + h(n)

    where:
        g(n) = accumulated path cost
        h(n) = estimated remaining cost

    With an admissible heuristic such as Manhattan Distance or
    Misplaced Tiles, A* returns an optimal solution for the 8-puzzle.
    """
    validate_state(start)
    validate_state(goal)

    if not is_solvable(start, goal):
        raise ValueError(
            "O estado inicial não possui solução para o objetivo informado."
        )

    started_at = perf_counter()

    counter = itertools.count()

    start_g = 0
    start_h = heuristic(
        start,
        goal,
    )
    start_f = start_g + start_h

    # Heap entries:
    #
    # (
    #     f(n),
    #     h(n),
    #     tie_breaker,
    #     g(n),
    #     state,
    # )
    #
    # h(n) is used as a secondary criterion when two states
    # have the same f(n).
    frontier: list[
        tuple[int, int, int, int, State]
    ] = [
        (
            start_f,
            start_h,
            next(counter),
            start_g,
            start,
        )
    ]

    parents: dict[State, ParentInfo] = {
        start: None,
    }

    g_score: dict[State, int] = {
        start: 0,
    }

    # Distinct states ever discovered.
    discovered: set[State] = {
        start,
    }

    # States whose successors have already been examined.
    closed: set[State] = set()

    expanded = 0
    generated = 1

    while frontier:
        (
            _f,
            _h,
            _,
            current_g,
            current,
        ) = heapq.heappop(frontier)

        # The same state may appear in the heap more than once.
        # If this entry no longer represents its best known g(n),
        # it is stale and can safely be ignored.
        if current_g != g_score.get(
            current,
            inf,
        ):
            continue

        if current in closed:
            continue

        if current == goal:
            path, moves = reconstruct_path(
                parents,
                current,
            )

            elapsed = perf_counter() - started_at

            return SearchResult(
                algorithm=f"A* Search ({heuristic.__name__})",
                path=path,
                moves=moves,
                cost=current_g,
                expanded=expanded,
                generated=generated,
                elapsed_seconds=elapsed,
            )

        closed.add(current)

        expanded += 1

        for move, successor in neighbors(current):
            tentative_g = current_g + 1

            previous_g = g_score.get(
                successor,
                inf,
            )

            if tentative_g >= previous_g:
                continue

            parents[successor] = (
                current,
                move,
            )

            g_score[successor] = tentative_g

            # A better route to a previously expanded state was found.
            # Reopening makes the implementation robust even when a
            # future heuristic is admissible but not consistent.
            if successor in closed:
                closed.remove(successor)

            successor_h = heuristic(
                successor,
                goal,
            )

            successor_f = (
                tentative_g
                + successor_h
            )

            heapq.heappush(
                frontier,
                (
                    successor_f,
                    successor_h,
                    next(counter),
                    tentative_g,
                    successor,
                ),
            )

            if successor not in discovered:
                discovered.add(successor)
                generated += 1

    raise RuntimeError(
        "A busca terminou sem encontrar o estado objetivo."
    )