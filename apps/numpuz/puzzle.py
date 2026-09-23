"""Core domain model for the NUMPUZ / sliding puzzle."""

from __future__ import annotations

from random import Random
from typing import Literal

SIZE = 3
BLANK = 0

State = tuple[int, ...]
Move = Literal["UP", "DOWN", "LEFT", "RIGHT"]

GOAL_STATE: State = tuple(range(1, SIZE * SIZE)) + (BLANK,)

_MOVE_DELTAS: dict[Move, tuple[int, int]] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

_OPPOSITE: dict[Move, Move] = {
    "UP": "DOWN",
    "DOWN": "UP",
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
}


def validate_state(state: State) -> None:
    """Raise ValueError when a state is not a valid SIZE x SIZE puzzle."""
    expected = set(range(SIZE * SIZE))

    if len(state) != SIZE * SIZE:
        raise ValueError(
            f"Estado inválido: esperado {SIZE * SIZE} posições, "
            f"recebido {len(state)}."
        )

    if set(state) != expected:
        raise ValueError(
            f"Estado inválido: deve conter exatamente os valores "
            f"{sorted(expected)}."
        )

def parse_state(text: str) -> State:
    """Parse a textual 8-puzzle configuration into a State.

    Accepted examples:

        1,2,3,4,5,6,7,_,8
        123/456/7_8
        1 2 3
        4 5 6
        7 _ 8

    The blank position may be represented by "_" or "0".
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "Informe uma configuração para o tabuleiro."
        )

    separators = {
        " ",
        "\t",
        "\n",
        "\r",
        ",",
        "/",
        ";",
        "|",
    }

    symbols = [
        character
        for character in text
        if character not in separators
    ]

    if len(symbols) != SIZE * SIZE:
        raise ValueError(
            f"Estado inválido: esperado {SIZE * SIZE} posições, "
            f"recebido {len(symbols)}."
        )

    allowed = set("12345678_0")

    invalid = [
        symbol
        for symbol in symbols
        if symbol not in allowed
    ]

    if invalid:
        raise ValueError(
            "Estado inválido: utilize somente os números de 1 a 8 "
            "e '_' ou 0 para representar o espaço vazio."
        )

    state: State = tuple(
        BLANK if symbol == "_" else int(symbol)
        for symbol in symbols
    )

    validate_state(state)

    return state

def blank_position(state: State) -> tuple[int, int]:
    """Return the row and column of the blank position."""
    validate_state(state)

    index = state.index(BLANK)

    return divmod(index, SIZE)


def valid_moves(state: State) -> tuple[Move, ...]:
    """Return every legal movement of the blank for the given state."""
    row, col = blank_position(state)

    moves: list[Move] = []

    if row > 0:
        moves.append("UP")

    if row < SIZE - 1:
        moves.append("DOWN")

    if col > 0:
        moves.append("LEFT")

    if col < SIZE - 1:
        moves.append("RIGHT")

    return tuple(moves)


def apply_move(state: State, move: Move) -> State:
    """Return a new state after moving the blank."""
    validate_state(state)

    if move not in _MOVE_DELTAS:
        raise ValueError(f"Movimento desconhecido: {move!r}.")

    legal_moves = valid_moves(state)

    if move not in legal_moves:
        raise ValueError(
            f"Movimento {move} inválido para o estado atual. "
            f"Movimentos possíveis: {legal_moves}."
        )

    blank_row, blank_col = blank_position(state)

    delta_row, delta_col = _MOVE_DELTAS[move]

    target_row = blank_row + delta_row
    target_col = blank_col + delta_col

    blank_index = blank_row * SIZE + blank_col
    target_index = target_row * SIZE + target_col

    new_state = list(state)

    new_state[blank_index], new_state[target_index] = (
        new_state[target_index],
        new_state[blank_index],
    )

    return tuple(new_state)


def neighbors(state: State) -> tuple[tuple[Move, State], ...]:
    """Return legal moves paired with successor states."""
    return tuple(
        (move, apply_move(state, move))
        for move in valid_moves(state)
    )


def is_goal(
    state: State,
    goal: State = GOAL_STATE,
) -> bool:
    """Return True when state matches the requested goal state."""
    validate_state(state)
    validate_state(goal)

    return state == goal


def inversion_count(state: State) -> int:
    """Count inversions while ignoring the blank tile."""
    validate_state(state)

    tiles = [
        tile
        for tile in state
        if tile != BLANK
    ]

    return sum(
        1
        for i in range(len(tiles))
        for j in range(i + 1, len(tiles))
        if tiles[i] > tiles[j]
    )


def _parity_signature(state: State) -> int:
    """Return the reachability parity signature of a puzzle state."""
    inversions = inversion_count(state)

    if SIZE % 2 == 1:
        return inversions % 2

    blank_row, _ = blank_position(state)
    blank_row_from_bottom = SIZE - blank_row

    return (inversions + blank_row_from_bottom) % 2


def is_solvable(
    state: State,
    goal: State = GOAL_STATE,
) -> bool:
    """Return whether state and goal are mutually reachable."""
    validate_state(state)
    validate_state(goal)

    return _parity_signature(state) == _parity_signature(goal)


def shuffle_state(
    moves: int = 30,
    *,
    seed: int | None = None,
    start: State = GOAL_STATE,
) -> State:
    """Create a solvable puzzle using legal random movements.

    Immediate reversal of the previous movement is avoided whenever
    another legal move exists.

    `moves` represents the length of the random walk. It does not
    necessarily represent the optimal distance from the generated
    state to the goal.
    """
    validate_state(start)

    if moves < 0:
        raise ValueError("moves deve ser maior ou igual a zero.")

    rng = Random(seed)

    current = start
    previous_move: Move | None = None

    for _ in range(moves):
        candidates = list(valid_moves(current))

        if previous_move is not None and len(candidates) > 1:
            reverse = _OPPOSITE[previous_move]

            candidates = [
                move
                for move in candidates
                if move != reverse
            ]

        selected = rng.choice(candidates)

        current = apply_move(current, selected)
        previous_move = selected

    # Evita devolver o próprio estado inicial após um ciclo completo.
    if moves > 0 and current == start:
        selected = rng.choice(valid_moves(current))
        current = apply_move(current, selected)

    return current