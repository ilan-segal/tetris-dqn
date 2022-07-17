"""
Helper functions and constants
"""
import numpy as np


BOARD_WIDTH = 10
BOARD_HEIGHT = 24
BOARD_VANISH_HEIGHT = BOARD_HEIGHT - 4


def overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Return True iff there exists some position in both a & b where both arrays
    equal True.
    """
    if a is None or b is None:
        # For when either array represents the position of an out-of-bounds piece.
        return True
    return np.logical_and(a, b).any()


def empty_board(empty_value) -> np.ndarray:
    """
    Return a (BOARD_WIDTH x BOARD_HEIGHT) matrix whose values are all False.
    """
    return np.full((BOARD_WIDTH, BOARD_HEIGHT), empty_value)


def is_out_of_bounds(a: np.ndarray) -> bool:
    """
    Given `a` which represents a single tetromino on an otherwise empty board,
    return True iff the tetromino is partially or completely off of the board.
    This is determined by counting the number of minos (individual squares)
    which are accounted for in `a`. If this count is less than 4, we know that
    at least 1 mino is off the board.
    """
    return np.count_nonzero(a) < 4
