import unittest

from puzzle import (
    GOAL_STATE,
    apply_move,
    blank_position,
    inversion_count,
    is_goal,
    is_solvable,
    neighbors,
    parse_state,
    shuffle_state,
    valid_moves,
)


class PuzzleTests(unittest.TestCase):

    def test_goal_state_is_recognized(self):
        self.assertTrue(is_goal(GOAL_STATE))


    def test_custom_goal_is_recognized(self):
        custom_goal = (
            1, 2, 3,
            4, 5, 6,
            7, 0, 8,
        )

        self.assertTrue(
            is_goal(
                custom_goal,
                custom_goal,
            )
        )

        self.assertFalse(
            is_goal(
                GOAL_STATE,
                custom_goal,
            )
        )

    def test_shuffle_from_custom_goal_is_reachable(self):
        custom_goal = (
            1, 2, 3,
            4, 5, 6,
            7, 0, 8,
        )

        initial = shuffle_state(
            moves=20,
            seed=42,
            start=custom_goal,
        )

        self.assertTrue(
            is_solvable(
                initial,
                custom_goal,
            )
        )
    
    def test_blank_position_on_goal(self):
        self.assertEqual(
            blank_position(GOAL_STATE),
            (2, 2),
        )

    def test_valid_moves_from_bottom_right_corner(self):
        self.assertEqual(
            valid_moves(GOAL_STATE),
            ("UP", "LEFT"),
        )

    def test_valid_moves_from_center(self):
        state = (
            1, 2, 3,
            4, 0, 5,
            6, 7, 8,
        )

        self.assertEqual(
            valid_moves(state),
            ("UP", "DOWN", "LEFT", "RIGHT"),
        )

    def test_apply_move_returns_new_state_without_mutating_original(self):
        original = GOAL_STATE

        moved = apply_move(original, "LEFT")

        self.assertEqual(
            original,
            GOAL_STATE,
        )

        self.assertEqual(
            moved,
            (
                1, 2, 3,
                4, 5, 6,
                7, 0, 8,
            ),
        )

    def test_invalid_move_raises_value_error(self):
        with self.assertRaises(ValueError):
            apply_move(
                GOAL_STATE,
                "RIGHT",
            )

    def test_neighbors_match_legal_moves(self):
        result = neighbors(GOAL_STATE)

        self.assertEqual(
            len(result),
            2,
        )

        self.assertEqual(
            {move for move, _ in result},
            {"UP", "LEFT"},
        )

    def test_goal_has_zero_inversions(self):
        self.assertEqual(
            inversion_count(GOAL_STATE),
            0,
        )

    def test_known_unsolvable_state_is_rejected(self):
        unsolvable = (
            1, 2, 3,
            4, 5, 6,
            8, 7, 0,
        )

        self.assertFalse(
            is_solvable(unsolvable)
        )

    def test_shuffled_states_are_solvable(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                state = shuffle_state(
                    40,
                    seed=seed,
                )

                self.assertTrue(
                    is_solvable(state)
                )

    def test_shuffle_is_reproducible_with_seed(self):
        first = shuffle_state(
            30,
            seed=42,
        )

        second = shuffle_state(
            30,
            seed=42,
        )

        self.assertEqual(
            first,
            second,
        )

    def test_zero_shuffle_returns_goal(self):
        self.assertEqual(
            shuffle_state(
                0,
                seed=42,
            ),
            GOAL_STATE,
        )
        
    def test_parse_state_with_commas(self):
        result = parse_state(
            "1,2,3,4,5,6,7,_,8"
        )

        self.assertEqual(
            result,
            (
                1, 2, 3,
                4, 5, 6,
                7, 0, 8,
            ),
        )

    def test_parse_state_with_compact_rows(self):
        result = parse_state(
            "123/456/78_"
        )

        self.assertEqual(
            result,
            GOAL_STATE,
        )

    def test_parse_state_with_multiline_board(self):
        result = parse_state(
            """
            1 2 3
            4 5 6
            7 _ 8
            """
        )

        self.assertEqual(
            result,
            (
                1, 2, 3,
                4, 5, 6,
                7, 0, 8,
            ),
        )

    def test_parse_state_accepts_zero_as_blank(self):
        result = parse_state(
            "1,2,3,4,5,6,7,8,0"
        )

        self.assertEqual(
            result,
            GOAL_STATE,
        )

    def test_parse_state_rejects_missing_position(self):
        with self.assertRaises(ValueError):
            parse_state(
                "1,2,3,4,5,6,7,_"
            )

    def test_parse_state_rejects_duplicate_tiles(self):
        with self.assertRaises(ValueError):
            parse_state(
                "1,2,3,4,5,6,7,7,_"
            )

    def test_parse_state_rejects_invalid_character(self):
        with self.assertRaises(ValueError):
            parse_state(
                "1,2,3,4,5,6,7,X,_"
            )

if __name__ == "__main__":
    unittest.main()