"""
Classes in this file represent the 7 tetrominoes: I, O, T, S, Z, J, L
"""

import numpy as np

from utils import (
    BOARD_WIDTH,
    BOARD_VANISH_HEIGHT,
    overlap,
    empty_board,
    is_out_of_bounds
)

FILLED_CELL = 1
EMPTY_CELL = 0


class Tetromino:
    """
    Abstract class
    """

    def __init__(self, box_size=3):
        self.facing = 0
        self.x = (BOARD_WIDTH - box_size) // 2
        self.y = BOARD_VANISH_HEIGHT - box_size + 1
        self.color_idx = 1
        self.rotation_points_by_facing = None

    def get_grid(self, offset: tuple[int, int] = (0, 0)) -> np.ndarray:
        """
        Return a numpy array which represents this tetromino in the same space
        as the whole board (BOARD_WIDTH x BOARD_HEIGHT). Returns None if the tetromino is
        out of bounds.
        """
        x, y = offset
        self.x += x
        self.y += y
        ret = empty_board(EMPTY_CELL)
        self._add_to_board(ret)
        self.x -= x
        self.y -= y
        return ret * self.color_idx

    def get_ghost_grid(self, board: np.ndarray) -> np.ndarray:
        """
        Return a numpy array in the same space as `get_grid()` but represents
        this piece if it were to hard drop.
        """
        old = self.x, self.y
        self.hard_drop(board)
        ret = self.get_grid()
        self.x, self.y = old
        return ret

    def _add_to_board(self, board: np.ndarray) -> None:
        raise NotImplementedError

    def rotate_CW(self, board: np.ndarray) -> None:
        """
        Rotate this piece clockwise.
        """
        self._rotate(board=board, dir=1)

    def rotate_CCW(self, board: np.ndarray) -> None:
        """
        Rotate this piece counter-clockwise.
        """
        self._rotate(board=board, dir=-1)

    def _rotate(self, board: np.ndarray, dir: int) -> None:
        """
        dir is either 1 (clockwise) or -1 (counter-clockwise)
        """
        src_points = self._get_rotation_points()
        self.facing = (self.facing + dir) % 4
        dst_points = self._get_rotation_points()
        for (src_x, src_y), (dst_x, dst_y) in zip(src_points, dst_points):
            diff_x = dst_x - src_x
            diff_y = dst_y - src_y
            self.x += diff_x
            self.y += diff_y
            if not self._has_invalid_pos(board):
                return
            self.x -= diff_x
            self.y -= diff_y
        # Undo rotation in case no point could be rotated around
        self.facing = (self.facing - dir) % 4

    def _get_rotation_points(self) -> list[tuple]:
        """
        Return a list of all 5 rotation points in order of priority.
        Check page 36 of the 2009 Tetris Design Guideline for details of the
        super rotation system.
        """
        if self.rotation_points_by_facing is None:
            raise NotImplementedError
        return self.rotation_points_by_facing[self.facing]

    def _is_overlapping(self, board: np.ndarray) -> bool:
        return overlap(self.get_grid(), board)

    def shift(self, board: np.ndarray, dir: tuple[int, int]) -> bool:
        """
        Move this tetromino one unit across the board. Return True iff this
        tetromino was able to move down without collision and without leaving
        the boundaries of the board.
        """
        x, y = dir
        self.x += x
        self.y += y
        if self._has_invalid_pos(board):
            self.x -= x
            self.y -= y
            return False
        return True

    def _is_out_of_bounds(self) -> bool:
        return is_out_of_bounds(self.get_grid())

    def _has_invalid_pos(self, board: np.ndarray) -> bool:
        return self._is_overlapping(board) or self._is_out_of_bounds()

    def hard_drop(self, board: np.ndarray) -> int:
        """
        Move this tetromino as far downwards as possible without collision or
        leaving bounds.
        """
        lines_dropped = 0
        while self.shift(board, (0, -1)):
            lines_dropped += 1
        return lines_dropped


class I(Tetromino):
    """
    ####
    """

    def __init__(self):
        super().__init__(box_size=4)
        self.color_idx = 1
        self.rotation_points_by_facing = {
            0: [(1, 2), (0, 2), (3, 2), (0, 2), (3, 2)],
            1: [(1, 2), (2, 2), (2, 2), (2, 3), (2, 0)],
            2: [(1, 2), (3, 2), (0, 2), (3, 1), (0, 1)],
            3: [(1, 2), (1, 2), (1, 2), (1, 0), (1, 3)],
        }

    def _add_to_board(self, board: np.ndarray) -> None:
        match self.facing:
            case 0:
                board[self.x : self.x+4, self.y+2 : self.y+3] = FILLED_CELL
            case 1:
                board[self.x+2 : self.x+3, self.y : self.y+4] = FILLED_CELL
            case 2:
                board[self.x : self.x+4, self.y+1 : self.y+2] = FILLED_CELL
            case 3:
                board[self.x+1 : self.x+2, self.y : self.y+4] = FILLED_CELL
            case _:
                raise ValueError(f'Invalid facing {self.facing}')


class O(Tetromino):
    """
    ##
    ##
    """

    def __init__(self):
        super().__init__()
        self.color_idx = 2

    def _add_to_board(self, board: np.ndarray) -> None:
        board[self.x+1 : self.x+3, self.y+1 : self.y+3] = FILLED_CELL

    def _get_rotation_points(self) -> list[tuple]:
        return [(1, 1)]


class T(Tetromino):
    """
     #
    ###
    """

    def __init__(self):
        super().__init__()
        self.color_idx = 3
        self.rotation_points_by_facing = {
            0: [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            1: [(1, 1), (1, 2), (0, 2), (1, 3), (2, 3)],
            2: [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            3: [(1, 1), (0, 1), (0, 0), (1, 3), (0, 3)],
        }

    def _add_to_board(self, board: np.ndarray) -> None:
        match self.facing:
            case 0:
                board[self.x : self.x+3, self.y+1 : self.y+2] = FILLED_CELL
                board[self.x+1 : self.x+2, self.y+2 : self.y+3] = FILLED_CELL
            case 1:
                board[self.x+1 : self.x+2, self.y : self.y+3] = FILLED_CELL
                board[self.x+2 : self.x+3, self.y+1 : self.y+2] = FILLED_CELL
            case 2:
                board[self.x : self.x+3, self.y+1 : self.y+2] = FILLED_CELL
                board[self.x+1 : self.x+2, self.y : self.y+1] = FILLED_CELL
            case 3:
                board[self.x+1 : self.x+2, self.y : self.y+3] = FILLED_CELL
                board[self.x : self.x+1, self.y+1 : self.y+2] = FILLED_CELL
            case _:
                raise ValueError(f'Invalid facing {self.facing}')


class L(Tetromino):
    """
    #
    #
    ##
    """

    def __init__(self):
        super().__init__()
        self.color_idx = 4
        self.rotation_points_by_facing = {
            0: [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            1: [(1, 1), (1, 2), (0, 2), (1, 3), (2, 3)],
            2: [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            3: [(1, 1), (0, 1), (0, 0), (1, 3), (0, 3)],
        }

    def _add_to_board(self, board: np.ndarray) -> None:
        match self.facing:
            case 0:
                board[self.x : self.x+3, self.y+1 : self.y+2] = FILLED_CELL
                board[self.x+2 : self.x+3, self.y+2 : self.y+3] = FILLED_CELL
            case 1:
                board[self.x+1 : self.x+2, self.y : self.y+3] = FILLED_CELL
                board[self.x+2 : self.x+3, self.y : self.y+1] = FILLED_CELL
            case 2:
                board[self.x : self.x+3, self.y+1 : self.y+2] = FILLED_CELL
                board[self.x : self.x+1, self.y : self.y+1] = FILLED_CELL
            case 3:
                board[self.x+1 : self.x+2, self.y : self.y+3] = FILLED_CELL
                board[self.x : self.x+1, self.y+2 : self.y+3] = FILLED_CELL
            case _:
                raise ValueError(f'Invalid facing {self.facing}')


class J(Tetromino):
    """
     #
     #
    ##
    """

    def __init__(self):
        super().__init__()
        self.color_idx = 5
        self.rotation_points_by_facing = {
            0: [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            1: [(1, 1), (1, 2), (0, 2), (1, 3), (2, 3)],
            2: [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            3: [(1, 1), (0, 1), (0, 0), (1, 3), (0, 3)],
        }

    def _add_to_board(self, board: np.ndarray) -> None:
        match self.facing:
            case 0:
                board[self.x : self.x+3, self.y+1 : self.y+2] = FILLED_CELL
                board[self.x : self.x+1, self.y+2 : self.y+3] = FILLED_CELL
            case 1:
                board[self.x+1 : self.x+2, self.y : self.y+3] = FILLED_CELL
                board[self.x+2 : self.x+3, self.y+2 : self.y+3] = FILLED_CELL
            case 2:
                board[self.x : self.x+3, self.y+1 : self.y+2] = FILLED_CELL
                board[self.x+2 : self.x+3, self.y : self.y+1] = FILLED_CELL
            case 3:
                board[self.x+1 : self.x+2, self.y : self.y+3] = FILLED_CELL
                board[self.x : self.x+1, self.y : self.y+1] = FILLED_CELL
            case _:
                raise ValueError(f'Invalid facing {self.facing}')


class S(Tetromino):
    """
     ##
    ##
    """

    def __init__(self):
        super().__init__()
        self.color_idx = 6
        self.rotation_points_by_facing = {
            0: [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            1: [(1, 1), (1, 2), (0, 2), (1, 3), (2, 3)],
            2: [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            3: [(1, 1), (0, 1), (0, 0), (1, 3), (0, 3)],
        }

    def _add_to_board(self, board: np.ndarray) -> None:
        match self.facing:
            case 0:
                board[self.x : self.x+2, self.y+1 : self.y+2] = FILLED_CELL
                board[self.x+1 : self.x+3, self.y+2 : self.y+3] = FILLED_CELL
            case 1:
                board[self.x+1 : self.x+2, self.y+1 : self.y+3] = FILLED_CELL
                board[self.x+2 : self.x+3, self.y : self.y+2] = FILLED_CELL
            case 2:
                board[self.x : self.x+2, self.y : self.y+1] = FILLED_CELL
                board[self.x+1 : self.x+3, self.y+1 : self.y+2] = FILLED_CELL
            case 3:
                board[self.x : self.x+1, self.y+1 : self.y+3] = FILLED_CELL
                board[self.x+1 : self.x+2, self.y : self.y+2] = FILLED_CELL
            case _:
                raise ValueError(f'Invalid facing {self.facing}')


class Z(Tetromino):
    """
    ##
     ##
    """

    def __init__(self):
        super().__init__()
        self.color_idx = 7
        self.rotation_points_by_facing = {
            0: [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            1: [(1, 1), (1, 2), (0, 2), (1, 3), (2, 3)],
            2: [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            3: [(1, 1), (0, 1), (0, 0), (1, 3), (0, 3)],
        }

    def _add_to_board(self, board: np.ndarray) -> None:
        match self.facing:
            case 0:
                board[self.x : self.x+2, self.y+2 : self.y+3] = FILLED_CELL
                board[self.x+1 : self.x+3, self.y+1 : self.y+2] = FILLED_CELL
            case 1:
                board[self.x+2 : self.x+3, self.y+1 : self.y+3] = FILLED_CELL
                board[self.x+1 : self.x+2, self.y : self.y+2] = FILLED_CELL
            case 2:
                board[self.x : self.x+2, self.y+1 : self.y+2] = FILLED_CELL
                board[self.x+1 : self.x+3, self.y : self.y+1] = FILLED_CELL
            case 3:
                board[self.x+1 : self.x+2, self.y+1 : self.y+3] = FILLED_CELL
                board[self.x : self.x+1, self.y : self.y+2] = FILLED_CELL
            case _:
                raise ValueError(f'Invalid facing {self.facing}')


AnyTetromino = I | J | L | O | S | T | Z
ALL_SHAPE_TYPES = [I, J, L, O, S, T, Z]
