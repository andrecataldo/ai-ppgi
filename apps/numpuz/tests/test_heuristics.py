import unittest

from heuristics import (
    HEURISTICS,
    manhattan_distance,
    misplaced_tiles,
)
from puzzle import (
    GOAL_STATE,
    shuffle_state,
)


class HeuristicTests(unittest.TestCase):

    def test_goal_has_zero_heuristic(self):
        self.assertEqual(
            misplaced_tiles(GOAL_STATE),
            0,
        )

        self.assertEqual(
            manhattan_distance(GOAL_STATE),
            0,
        )

    def test_state_one_move_from_goal(self):
        state = (
            1, 2, 3,
            4, 5, 6,
            7, 0, 8,
        )

        self.assertEqual(
            misplaced_tiles(state),
            1,
        )

        self.assertEqual(
            manhattan_distance(state),
            1,
        )

    def test_manhattan_uses_distance_not_only_wrong_position(self):
        state = (
            8, 2, 3,
            4, 5, 6,
            7, 1, 0,
        )

        self.assertEqual(
            misplaced_tiles(state),
            2,
        )

        self.assertEqual(
            manhattan_distance(state),
            6,
        )

    def test_blank_is_not_counted(self):
        state = (
            1, 2, 3,
            4, 5, 6,
            0, 7, 8,
        )

        self.assertEqual(
            misplaced_tiles(state),
            2,
        )

        self.assertEqual(
            manhattan_distance(state),
            2,
        )

    def test_manhattan_dominates_misplaced_tiles(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                state = shuffle_state(
                    moves=40,
                    seed=seed,
                )

                self.assertGreaterEqual(
                    manhattan_distance(state),
                    misplaced_tiles(state),
                )

    def test_heuristic_registry(self):
        self.assertIs(
            HEURISTICS["Manhattan Distance"],
            manhattan_distance,
        )

        self.assertIs(
            HEURISTICS["Misplaced Tiles"],
            misplaced_tiles,
        )

    def test_heuristics_use_custom_goal(self):
        custom_goal = (
            1, 2, 3,
            4, 5, 6,
            7, 0, 8,
        )

        state = GOAL_STATE

        self.assertEqual(
            misplaced_tiles(
                state,
                custom_goal,
            ),
            1,
        )

        self.assertEqual(
            manhattan_distance(
                state,
                custom_goal,
            ),
            1,
        )

        self.assertEqual(
            misplaced_tiles(
                custom_goal,
                custom_goal,
            ),
            0,
        )

        self.assertEqual(
            manhattan_distance(
                custom_goal,
                custom_goal,
            ),
            0,
        )

if __name__ == "__main__":
    unittest.main()