"""Grafo utilizado no laboratório comparativo da Aula 03."""

GRAPH = {
    "S": [("A", 1), ("B", 4), ("C", 2)],
    "A": [("G", 9)],
    "B": [("G", 2)],
    "C": [("D", 2)],
    "D": [("G", 3)],
    "G": [],
}

HEURISTIC = {
    "S": 2,
    "A": 1,
    "B": 2,
    "C": 4,
    "D": 3,
    "G": 0,
}

START = "S"
GOAL = "G"
OPTIMAL_COST = 6
