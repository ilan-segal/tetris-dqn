"""
Tracking state and handling state actions.
"""
# from __future__ import annotations

import random
import numpy as np

from utils import empty_board, BOARD_WIDTH, BOARD_HEIGHT, BOARD_VANISH_HEIGHT, overlap
from model.tetromino import AnyTetromino, ALL_SHAPE_TYPES, EMPTY_CELL
from model.action import Action


class State:
    """
    Current state on which the AI will act. Encompasses:
    - Current board as represented by a boolean numpy array (True or False for filled or empty cell, respectively)
    - Current piece.
    - Next piece.
    - Random bag of pieces (not visible to the AI)
    """

    LINE_CLEAR_REWARD = 1

    REVERSIBLE_ACTION_PENALTY = -0.1
    ILLEGAL_MOVE_PENALTY = -100
    GAME_OVER_PENALTY = -100

    def __init__(self):
        self.board = empty_board(EMPTY_CELL)
        self.bag = []
        self.cur_piece = self._generate_next_piece()
        self.next_piece = self._generate_next_piece()
        self.score = 0
        self.lines_cleared = 0
        self.num_moves = 0

    def get_state(self) -> np.ndarray:
        """
        Return a 4xWxH array S where:
        - S[0, :, :] contains all locked minos.
        - S[1, :, :] contains the current tetromino.
        - S[3, :, :] contains the current tetromino ghost.
        - S[4, :, :] contains the next tetromino.

        A filled cell is represented as 1.0, an empty cell 0.0
        """
        state = np.stack([
            self.board,
            self.cur_piece.get_grid(),
            self.cur_piece.get_ghost_grid(self.board),
            self.next_piece.get_grid(),
        ])
        empty_cell_mask = state == EMPTY_CELL
        state[empty_cell_mask] = 0
        state[~empty_cell_mask] = 1
        return state.astype(float)

    def step(self, action: Action) -> tuple[np.ndarray, float, bool]:
        """
        Mutate current state with given action.
        Return a tuple containing:
        - Current state (see `get_state()`)
        - Reward
        - Flag indicating game over
        """
        self.num_moves += 1
        old_score = self.score
        old_state = self.get_state()
        locked = False
        match action:
            case Action.LEFT:
                self.cur_piece.shift(self.board, (-1, 0))
            case Action.RIGHT:
                self.cur_piece.shift(self.board, (1, 0))
            case Action.ROTATE_CW:
                self.cur_piece.rotate_CW(self.board)
            case Action.ROTATE_CCW:
                self.cur_piece.rotate_CCW(self.board)
            case Action.SOFT_DROP:
                locked = not self.cur_piece.shift(self.board, (0, -1))
            case Action.HARD_DROP:
                self.cur_piece.hard_drop(self.board)
                locked = True
        if locked:
            self._lock_piece_and_update()
        reward = self.score - old_score
        if np.all(self.get_state() == old_state):
            reward += State.ILLEGAL_MOVE_PENALTY
        done = self._is_game_over()
        if done:
            reward += State.GAME_OVER_PENALTY
        if action in [Action.LEFT, Action.RIGHT, Action.ROTATE_CW, Action.ROTATE_CCW]:
            reward += State.REVERSIBLE_ACTION_PENALTY
        return self.get_state(), reward, done

    def _is_game_over(self):
        return overlap(self.next_piece.get_grid(), self.board)

    def _generate_next_piece(self) -> AnyTetromino:
        if len(self.bag) == 0:
            self.bag = ALL_SHAPE_TYPES[:]
            random.shuffle(self.bag)
        return self.bag.pop()()

    def _lock_piece_and_update(self):
        self.board = self.board + self.cur_piece.get_grid()
        self.cur_piece = self.next_piece
        self.next_piece = self._generate_next_piece()
        full_lines_mask = self.board.all(axis=0)
        cleared_count = np.count_nonzero(full_lines_mask)
        if cleared_count > 0:
            self.score += (cleared_count * State.LINE_CLEAR_REWARD) ** 2
            self.lines_cleared += cleared_count
            remaining_lines = self.board.T[~full_lines_mask].T
            empty_lines = np.zeros((BOARD_WIDTH, cleared_count))
            self.board = np.concatenate((remaining_lines, empty_lines), axis=1)

    def _get_garbage_hole_area(self) -> int:
        total_area = 0
        cols, rows = self.board.shape
        for col in range(cols):
            cur_area = 0
            for row in range(rows):
                if self.board[col, row]:
                    total_area += cur_area
                    cur_area = 0
                else:
                    cur_area += 1
        return total_area

    def _get_max_height(self) -> int:
        if not np.any(self.board):
            return 0
        row_occupied = np.any(self.board, axis=0)
        dist_from_end = np.argmax(np.flip(row_occupied))
        return BOARD_HEIGHT - dist_from_end

    def __str__(self) -> str:
        ret = ''
        state = self.get_state()
        cast_cur_board = state[0]
        cast_cur_piece = state[1] * 0.5
        all_board = cast_cur_board + cast_cur_piece
        ret += '   ' + '__' * BOARD_WIDTH + '\n'
        for i, row in enumerate(all_board.T[:BOARD_VANISH_HEIGHT][::-1]):
            ret += f'{BOARD_VANISH_HEIGHT - i - 1:2d} '
            squares = [u'\u2588\u2588' if cell == 1 else u'\u2592\u2592' if cell == 0.5 else u'\u2591\u2591' for cell in row]
            ret += ''.join(squares) + u'\u23B8' + '\n'
        ret += '   ' + u' \u0305 \u0305' * BOARD_WIDTH + '\n'
        ret += '   ' + u' '.join(map(str, list(range(BOARD_WIDTH)))) + '\n'
        ret += '\n'
        ret += f'Score = {self.score}\n'
        ret += f'Max height = {self._get_max_height()}\n'
        ret += f'Garbage hole area = {self._get_garbage_hole_area()}\n'
        return ret
