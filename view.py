"""
Display a given model using pygame.
"""
import numpy as np
import pygame

from model.state import State
from utils import BOARD_VANISH_HEIGHT


_WINDOW_SIZE = 800, 800
_MINO_SIZE = 38
_BOARD_POS = 25, 25
_NEXT_PIECE_POS = 350, 50
_NEXT_PIECE_LABEL_POS = 500, 120
_NUM_MOVES_LABEL_POS = 420, 620
_SCORE_LABEL_POS = 420, 670
_LINES_CLEARED_LABEL_POS = 420, 720

_BACKGROUND = 28, 23, 36
_BOARD_BACKGROUND = 13, 13, 13

_MINO_COLOURS = {
    1: (129, 213, 219),
    2: (245, 242, 144),
    3: (230, 108, 217),
    4: (255, 193, 112),
    5: (107, 130, 255),
    6: (119, 237, 138),
    7: (255, 130, 130)
}

_MINO_IMG_PATH = 'assets/mino.png'
_orig_mino = pygame.image.load(_MINO_IMG_PATH)
_orig_mino_w, _orig_mino_h = _orig_mino.get_size()
_orig_mino = pygame.transform.scale(
    _orig_mino,
    (
        _MINO_SIZE,
        _MINO_SIZE,
    )
)


class View:
    """
    Display a game to the _screen.
    """

    def __init__(self):
        pygame.init()
        self._screen = pygame.display.set_mode(_WINDOW_SIZE)
        self._mino_images = dict()
        self._mino_images_ghost = dict()
        for colour_idx, colour in _MINO_COLOURS.items():
            colour += 255,
            new_mino = _orig_mino.copy().convert_alpha()
            new_mino.fill(colour, special_flags=pygame.BLEND_RGB_MULT)
            self._mino_images[colour_idx] = new_mino
            new_mino_ghost = new_mino.copy()
            new_mino_ghost.fill((255, 255, 255, 128), special_flags=pygame.BLEND_RGBA_MULT)
            self._mino_images_ghost[colour_idx] = new_mino_ghost
        self._font = pygame.font.SysFont('agencyfb', 50)

    def quit(self) -> None:
        """
        Quit da game
        """
        pygame.quit()

    def draw(self, s: State) -> None:
        """
        Draw da game
        """
        pygame.event.pump()
        self._screen.fill(_BACKGROUND)
        board_width, board_height = s.board.shape[0], BOARD_VANISH_HEIGHT
        screen_size = (
            board_width * _MINO_SIZE,
            board_height * _MINO_SIZE
        )
        pygame.draw.rect(
            self._screen,
            _BOARD_BACKGROUND,
            pygame.Rect(*_BOARD_POS, *screen_size)
        )
        board = s.board
        cur_piece = s.cur_piece.get_grid()
        cur_ghost = s.cur_piece.get_ghost_grid(board)
        self._draw_board(_BOARD_POS, np.maximum(board, cur_piece), cur_ghost)
        self._draw_board(_NEXT_PIECE_POS, s.next_piece.get_grid((0, -1)))
        self._draw_text('next', _NEXT_PIECE_LABEL_POS)
        self._draw_text(f'moves: {s.num_moves}', _NUM_MOVES_LABEL_POS)
        self._draw_text(f'score: {s.score}', _SCORE_LABEL_POS)
        self._draw_text(f'lines: {s.lines_cleared}', _LINES_CLEARED_LABEL_POS)
        pygame.display.flip()

    def _draw_text(self, text: str, pos: tuple[int, int]) -> None:
        img = self._font.render(text, True, (255, 255, 255))
        rect = img.get_rect()
        rect.topleft = pos
        self._screen.blit(img, rect)

    def _draw_board(self,
                    pos: tuple[int, int],
                    solid_minos: np.ndarray,
                    ghost_minos: np.ndarray = None) -> None:
        board_width, board_height = solid_minos.shape[0], BOARD_VANISH_HEIGHT
        if ghost_minos is None:
            ghost_minos = np.zeros_like(solid_minos)
        for board_y in range(board_height):
            screen_y = board_height - board_y - 1
            screen_y *= _MINO_SIZE
            screen_y += pos[1]
            for board_x in range(board_width):
                is_ghost = False
                screen_x = _MINO_SIZE * board_x + pos[0]
                colour_idx = solid_minos[board_x, board_y]
                if colour_idx == 0:
                    colour_idx = ghost_minos[board_x, board_y]
                    is_ghost = True
                if colour_idx == 0:
                    continue
                colour_dict = self._mino_images_ghost if is_ghost else self._mino_images
                mino_img = colour_dict.get(colour_idx, list(colour_dict.values())[0])
                self._screen.blit(mino_img, (screen_x, screen_y))
