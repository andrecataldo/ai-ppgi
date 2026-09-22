import unittest

from heuristics import (
    manhattan_distance,
    misplaced_tiles,
)
from puzzle import (
    GOAL_STATE,
    apply_move,
    shuffle_state,
)
from solver import (
    astar,
    bfs,
    dfs,
)


class BFSTests(unittest.TestCase):

    def test_goal_state_requires_zero_moves(self):
        result = bfs(GOAL_STATE)

        self.assertEqual(
            result.cost,
            0,
        )

        self.assertEqual(
            result.moves,
            (),
        )

        self.assertEqual(
            result.path,
            (GOAL_STATE,),
        )

        self.assertEqual(
            result.expanded,
            0,
        )

        self.assertEqual(
            result.generated,
            1,
        )

    def test_one_move_from_goal(self):
        start = (
            1, 2, 3,
            4, 5, 6,
            7, 0, 8,
        )

        result = bfs(start)

        self.assertEqual(
            result.cost,
            1,
        )

        self.assertEqual(
            result.moves,
            ("RIGHT",),
        )

        self.assertEqual(
            result.path[0],
            start,
        )

        self.assertEqual(
            result.path[-1],
            GOAL_STATE,
        )

    def test_two_moves_from_goal(self):
        start = (
            1, 2, 3,
            4, 5, 6,
            0, 7, 8,
        )

        result = bfs(start)

        self.assertEqual(
            result.cost,
            2,
        )

        self.assertEqual(
            result.moves,
            (
                "RIGHT",
                "RIGHT",
            ),
        )

    def test_path_and_moves_are_consistent(self):
        start = (
            1, 2, 3,
            4, 0, 6,
            7, 5, 8,
        )

        result = bfs(start)

        current = start

        for index, move in enumerate(
            result.moves,
            start=1,
        ):
            current = apply_move(
                current,
                move,
            )

            self.assertEqual(
                current,
                result.path[index],
            )

        self.assertEqual(
            current,
            GOAL_STATE,
        )

    def test_path_contains_cost_plus_one_states(self):
        start = (
            1, 2, 3,
            4, 5, 6,
            0, 7, 8,
        )

        result = bfs(start)

        self.assertEqual(
            len(result.path),
            result.cost + 1,
        )

        self.assertEqual(
            len(result.moves),
            result.cost,
        )

    def test_known_unsolvable_state_is_rejected(self):
        unsolvable = (
            1, 2, 3,
            4, 5, 6,
            8, 7, 0,
        )

        with self.assertRaises(ValueError):
            bfs(unsolvable)

    def test_generated_is_greater_than_or_equal_to_expanded(self):
        start = (
            1, 2, 3,
            4, 0, 6,
            7, 5, 8,
        )

        result = bfs(start)

        self.assertGreaterEqual(
            result.generated,
            result.expanded,
        )

    def test_elapsed_time_is_non_negative(self):
        result = bfs(GOAL_STATE)

        self.assertGreaterEqual(
            result.elapsed_seconds,
            0.0,
        )

    def test_custom_goal(self):
        custom_goal = (
            1, 2, 3,
            4, 5, 6,
            7, 0, 8,
        )

        result = bfs(
            GOAL_STATE,
            custom_goal,
        )

        self.assertEqual(
            result.cost,
            1,
        )

        self.assertEqual(
            result.moves,
            ("LEFT",),
        )

        self.assertEqual(
            result.path[-1],
            custom_goal,
        )

class DFSTests(unittest.TestCase):

    def test_goal_state_requires_zero_moves(self):
        result = dfs(GOAL_STATE)

        self.assertEqual(
            result.cost,
            0,
        )

        self.assertEqual(
            result.moves,
            (),
        )

        self.assertEqual(
            result.path,
            (GOAL_STATE,),
        )

        self.assertEqual(
            result.expanded,
            0,
        )

        self.assertEqual(
            result.generated,
            1,
        )

    def test_one_move_to_custom_goal(self):
        start = GOAL_STATE

        custom_goal = (
            1, 2, 3,
            4, 5, 0,
            7, 8, 6,
        )

        result = dfs(
            start,
            custom_goal,
        )

        self.assertEqual(
            result.cost,
            1,
        )

        self.assertEqual(
            result.moves,
            ("UP",),
        )

        self.assertEqual(
            result.path[0],
            start,
        )

        self.assertEqual(
            result.path[-1],
            custom_goal,
        )

    def test_path_and_moves_are_consistent(self):
        start = GOAL_STATE

        custom_goal = (
            1, 2, 3,
            4, 5, 0,
            7, 8, 6,
        )

        result = dfs(
            start,
            custom_goal,
        )

        current = start

        for index, move in enumerate(
            result.moves,
            start=1,
        ):
            current = apply_move(
                current,
                move,
            )

            self.assertEqual(
                current,
                result.path[index],
            )

        self.assertEqual(
            current,
            custom_goal,
        )

    def test_unsolvable_goal_is_rejected(self):
        unsolvable_goal = (
            1, 2, 3,
            4, 5, 6,
            8, 7, 0,
        )

        with self.assertRaises(ValueError):
            dfs(
                GOAL_STATE,
                unsolvable_goal,
            )

class AStarTests(unittest.TestCase):

    def test_goal_state_requires_zero_moves(self):
        result = astar(
            GOAL_STATE,
        )

        self.assertEqual(
            result.cost,
            0,
        )

        self.assertEqual(
            result.moves,
            (),
        )

        self.assertEqual(
            result.path,
            (GOAL_STATE,),
        )

        self.assertEqual(
            result.expanded,
            0,
        )

        self.assertEqual(
            result.generated,
            1,
        )

    def test_one_move_from_goal(self):
        start = (
            1, 2, 3,
            4, 5, 6,
            7, 0, 8,
        )

        result = astar(start)

        self.assertEqual(
            result.cost,
            1,
        )

        self.assertEqual(
            result.moves,
            ("RIGHT",),
        )

    def test_path_and_moves_are_consistent(self):
        start = (
            1, 2, 3,
            4, 0, 6,
            7, 5, 8,
        )

        result = astar(start)

        current = start

        for index, move in enumerate(
            result.moves,
            start=1,
        ):
            current = apply_move(
                current,
                move,
            )

            self.assertEqual(
                current,
                result.path[index],
            )

        self.assertEqual(
            current,
            GOAL_STATE,
        )

    def test_astar_matches_bfs_optimal_cost(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                state = shuffle_state(
                    moves=8,
                    seed=seed,
                )

                bfs_result = bfs(state)

                astar_result = astar(
                    state,
                    heuristic=manhattan_distance,
                )

                self.assertEqual(
                    astar_result.cost,
                    bfs_result.cost,
                )

    def test_both_heuristics_return_optimal_solution(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                state = shuffle_state(
                    moves=8,
                    seed=seed,
                )

                optimal = bfs(state).cost

                misplaced_result = astar(
                    state,
                    heuristic=misplaced_tiles,
                )

                manhattan_result = astar(
                    state,
                    heuristic=manhattan_distance,
                )

                self.assertEqual(
                    misplaced_result.cost,
                    optimal,
                )

                self.assertEqual(
                    manhattan_result.cost,
                    optimal,
                )

    def test_custom_goal(self):
        custom_goal = (
            1, 2, 3,
            4, 5, 6,
            7, 0, 8,
        )

        result = astar(
            GOAL_STATE,
            custom_goal,
            heuristic=manhattan_distance,
        )

        self.assertEqual(
            result.cost,
            1,
        )

        self.assertEqual(
            result.moves,
            ("LEFT",),
        )

        self.assertEqual(
            result.path[-1],
            custom_goal,
        )

    def test_heuristics_are_admissible_on_sample_states(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                state = shuffle_state(
                    moves=8,
                    seed=seed,
                )

                optimal = bfs(state).cost

                self.assertLessEqual(
                    misplaced_tiles(state),
                    optimal,
                )

                self.assertLessEqual(
                    manhattan_distance(state),
                    optimal,
                )

    def test_unsolvable_state_is_rejected(self):
        unsolvable = (
            1, 2, 3,
            4, 5, 6,
            8, 7, 0,
        )

        with self.assertRaises(ValueError):
            astar(unsolvable)

    def test_metrics_are_valid(self):
        state = shuffle_state(
            moves=10,
            seed=42,
        )

        result = astar(state)

        self.assertGreaterEqual(
            result.generated,
            result.expanded,
        )

        self.assertGreaterEqual(
            result.elapsed_seconds,
            0.0,
        )

if __name__ == "__main__":
    unittest.main()