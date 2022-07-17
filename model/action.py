"""
All ways to act on the current board state.
"""
from enum import Enum


class Action(Enum):
    """
    State actions include:
    - Rotate CW
    - Rotate CCW
    - Shift left
    - Shift right
    - Drop
    - Wait (or soft drop)
    """
    ROTATE_CW = 0
    ROTATE_CCW = 1
    LEFT = 2
    RIGHT = 3
    SOFT_DROP = 4
    HARD_DROP = 5


N = len(Action)
