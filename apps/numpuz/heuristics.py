"""Heuristics for informed search in the NUMPUZ sliding puzzle."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from puzzle import (
    BLANK,
    GOAL_STATE,
    SIZE,
    State,
    validate_state,
)

Heuristic = Callable[[State], int]


def misplaced_tiles(
    state: State,
    goal: State = GOAL_STATE,
) -> int:
    """Count tiles that are not in their goal position.

    The blank tile is ignored.
    """
    validate_state(state)
    validate_state(goal)

    return sum(
        1
        for index, tile in enumerate(state)
        if tile != BLANK and tile != goal[index]
    )


@lru_cache(maxsize=None)
def _goal_positions(
    goal: State,
) -> dict[int, tuple[int, int]]:
    """Map each tile to its row and column in the goal state."""
    validate_state(goal)

    return {
        tile: divmod(index, SIZE)
        for index, tile in enumerate(goal)
        if tile != BLANK
    }


def manhattan_distance(
    state: State,
    goal: State = GOAL_STATE,
) -> int:
    """Return the total Manhattan distance to the goal.

    For each numbered tile:

        |current_row - goal_row|
        +
        |current_col - goal_col|

    The blank tile is ignored.
    """
    validate_state(state)
    validate_state(goal)

    goal_positions = _goal_positions(goal)

    total = 0

    for index, tile in enumerate(state):
        if tile == BLANK:
            continue

        current_row, current_col = divmod(index, SIZE)
        goal_row, goal_col = goal_positions[tile]

        total += (
            abs(current_row - goal_row)
            + abs(current_col - goal_col)
        )

    return total


HEURISTICS: dict[str, Heuristic] = {
    "Manhattan Distance": manhattan_distance,
    "Misplaced Tiles": misplaced_tiles,
}