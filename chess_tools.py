import typing
from typing import Tuple, List, Iterable, Union
import numpy as np
from numpy import ndarray
import time
from array import array

# The chess board is an 8x8 array of 1-character strings. uppercase = white, lowercase = black
#chess_board = np.empty((8, 8), dtype='U1')

def is_in_board(pos: Tuple[int, int]) -> bool:
    # Check if a position is in the board.
    row, col = pos
    return 0 <= row < 8 and 0 <= col < 8


class CharArray2D:
    """Fast 2D array of characters. Works like a numpy array of chr, but is much faster and memory efficient with pypy"""
    __slots__ = ('width', 'height', 'array')

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.array = bytearray(width * height)

    def __getitem__(self, index):
        y, x = index
        return chr(self.array[y * self.width + x])

    def __setitem__(self, index, value):
        y, x = index
        self.array[y * self.width + x] = ord(value)

    def copy(self):
        new_array = CharArray2D(self.width, self.height)
        new_array.array = self.array.copy()
        return new_array

    @classmethod
    def from_ndarray(cls, array):
        self = cls(array.shape[1], array.shape[0])
        for y in range(self.height):
            for x in range(self.width):
                self[y, x] = array[y, x]
        return self

    def __str__(self):
        return str(self.array)



def init_board():
    # Initialize the board with the starting positions of the pieces.
    chess_board = np.array([['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'],
                            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
                            ['.', '.', '.', '.', '.', '.', '.', '.'],
                            ['.', '.', '.', '.', '.', '.', '.', '.'],
                            ['.', '.', '.', '.', '.', '.', '.', '.'],
                            ['.', '.', '.', '.', '.', '.', '.', '.'],
                            ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
                            ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r']])
    chess_board = CharArray2D.from_ndarray(np.array(chess_board))
    return chess_board

def move_piece(start: Tuple[int, int], end: Tuple[int, int], chess_board: ndarray) -> ndarray:
    # Move a piece from start to end.
    chess_board = chess_board.copy()


    if chess_board[start] not in 'KkPp':
        chess_board[end] = chess_board[start]
        chess_board[start] = '.'
        return chess_board

    # Check for pawn
    if chess_board[start] == "P":
        if end[0] == 7:
            chess_board[end] = "Q"
            chess_board[start] = '.'
            return chess_board
        #Check for en passant
        if start[1] != end[1] and chess_board[end] == '.':
            chess_board[end[0], end[1]] = 'P'
            chess_board[start[0], start[1]] = '.'
            chess_board[end[0] - 1, end[1]] = '.'
            return chess_board

        chess_board[end] = chess_board[start]
        chess_board[start] = '.'
        return chess_board

    elif chess_board[start] == "p":
        if end[0] == 0:
            chess_board[end] = "q"
            chess_board[start] = '.'
            return chess_board
        #Check for en passant
        if start[1] != end[1] and chess_board[end] == '.':
            chess_board[end[0], end[1]] = 'p'
            chess_board[start[0], start[1]] = '.'
            chess_board[end[0] + 1, end[1]] = '.'
            return chess_board
        chess_board[end] = chess_board[start]
        chess_board[start] = '.'
        return chess_board


    # Check for castling.
    if chess_board[start] == "K":
        if start == (0, 4) and end == (0, 0):
            chess_board[0, 2] = "K"
            chess_board[0, 3] = "R"
            chess_board[0, 0] = '.'
            chess_board[0, 1] = '.'
            chess_board[0, 4] = '.'

        elif start == (0, 4) and end == (0, 7):
            chess_board[0, 5] = "R"
            chess_board[0, 6] = "K"
            chess_board[0, 7] = '.'
            chess_board[0, 4] = '.'

        else:
            chess_board[end] = chess_board[start]
            chess_board[start] = '.'
            return chess_board

    elif chess_board[start] == "k":
        if start == (7, 4) and end == (7, 0):
            chess_board[7, 2] = "k"
            chess_board[7, 3] = "r"
            chess_board[7, 0] = '.'
            chess_board[7, 1] = '.'
            chess_board[7, 4] = '.'

        elif start == (7, 4) and end == (7, 7):
            chess_board[7, 4] = '.'
            chess_board[7, 5] = "r"
            chess_board[7, 6] = "k"
            chess_board[7, 7] = '.'


        else:
            chess_board[end] = chess_board[start]
            chess_board[start] = '.'
            return chess_board


    return chess_board



def get_moves_pawn_white(start: Tuple[int, int], chess_board: ndarray) -> List[Tuple[int, int]]:
    # Get all possible moves for a white pawn at start.
    first_move = (start[0] == 1)
    moves = []

    # Move forward if not blocked
    if chess_board[start[0] + 1, start[1]] == '.':
        moves.append((start[0] + 1, start[1]))
        if first_move and chess_board[start[0] + 2, start[1]] == '.':
            moves.append((start[0] + 2, start[1]))

    # Capture diagonally.
    if start[1] > 0 and chess_board[start[0] + 1, start[1] - 1].islower():
        moves.append((start[0] + 1, start[1] - 1))
    if start[1] < 7 and chess_board[start[0] + 1, start[1] + 1].islower():
        moves.append((start[0] + 1, start[1] + 1))

    # Check for en passant
    if start[0] == 4:
        if start[1] > 0 and chess_board[start[0], start[1] - 1] == 'p':
            moves.append((start[0] + 1, start[1] - 1))
        if start[1] < 7 and chess_board[start[0], start[1] + 1] == 'p':
            moves.append((start[0] + 1, start[1] + 1))

    return moves

def get_moves_pawn_black(start: Tuple[int, int], chess_board: ndarray) -> List[Tuple[int, int]]:
    # Get all possible moves for a black pawn at start.
    first_move = (start[0] == 6)
    moves = []

    # Move forward if not blocked.
    if chess_board[start[0] - 1, start[1]] == '.':
        moves.append((start[0] - 1, start[1]))
        if first_move and chess_board[start[0] - 2, start[1]] == '.':
            moves.append((start[0] - 2, start[1]))

    # Capture diagonally.
    if start[1] > 0 and chess_board[start[0] - 1, start[1] - 1].isupper():
        moves.append((start[0] - 1, start[1] - 1))
    if start[1] < 7 and chess_board[start[0] - 1, start[1] + 1].isupper():
        moves.append((start[0] - 1, start[1] + 1))

    # Check for en passant
    if start[0] == 3:
        if start[1] > 0 and chess_board[start[0], start[1] - 1] == 'P':
            moves.append((start[0] - 1, start[1] - 1))
        if start[1] < 7 and chess_board[start[0], start[1] + 1] == 'P':
            moves.append((start[0] - 1, start[1] + 1))

    return moves

def get_moves_rook_white(start: Tuple[int, int], chess_board: ndarray) -> List[Tuple[int, int]]:
    # Get all possible moves for a white rook at start.
    moves = []

    # Move up.
    for i in range(start[0] + 1, 8):
        if chess_board[i, start[1]] == '.':
            moves.append((i, start[1]))
        elif chess_board[i, start[1]].islower():
            moves.append((i, start[1]))
            break
        else:
            break

    # Move down.
    for i in range(start[0] - 1, -1, -1):
        if chess_board[i, start[1]] == '.':
            moves.append((i, start[1]))
        elif chess_board[i, start[1]].islower():
            moves.append((i, start[1]))
            break
        else:
            break

    # Move right.
    for i in range(start[1] + 1, 8):
        if chess_board[start[0], i] == '.':
            moves.append((start[0], i))
        elif chess_board[start[0], i].islower():
            moves.append((start[0], i))
            break
        else:
            break

    # Move left.
    for i in range(start[1] - 1, -1, -1):
        if chess_board[start[0], i] == '.':
            moves.append((start[0], i))
        elif chess_board[start[0], i].islower():
            moves.append((start[0], i))
            break
        else:
            break

    return moves

def get_moves_rook_black(start: Tuple[int, int], chess_board: ndarray) -> List[Tuple[int, int]]:
    # Get all possible moves for a black rook at start.
    moves = []

    # Move up.
    for i in range(start[0] + 1, 8):
        if chess_board[i, start[1]] == '.':
            moves.append((i, start[1]))
        elif chess_board[i, start[1]].isupper():
            moves.append((i, start[1]))
            break
        else:
            break

    # Move down.
    for i in range(start[0] - 1, -1, -1):
        if chess_board[i, start[1]] == '.':
            moves.append((i, start[1]))
        elif chess_board[i, start[1]].isupper():
            moves.append((i, start[1]))
            break
        else:
            break

    # Move right.
    for i in range(start[1] + 1, 8):
        if chess_board[start[0], i] == '.':
            moves.append((start[0], i))
        elif chess_board[start[0], i].isupper():
            moves.append((start[0], i))
            break
        else:
            break

    # Move left.
    for i in range(start[1] - 1, -1, -1):
        if chess_board[start[0], i] == '.':
            moves.append((start[0], i))
        elif chess_board[start[0], i].isupper():
            moves.append((start[0], i))
            break
        else:
            break

    return moves

def get_moves_knight_white(start: Tuple[int, int], chess_board: ndarray) -> List[Tuple[int, int]]:
    # Get all possible moves for a white knight at start.
    moves = []
    verify = lambda move: is_in_board(move) and (chess_board[move].islower() or chess_board[move] == '.')

    # Move up right.
    move = (start[0] + 2, start[1] + 1)
    if verify(move):
        moves.append(move)

    # Move up left.
    move = (start[0] + 2, start[1] - 1)
    if verify(move):
        moves.append(move)

    # Move down right.
    move = (start[0] - 2, start[1] + 1)
    if verify(move):
        moves.append(move)

    # Move down left.
    move = (start[0] - 2, start[1] - 1)
    if verify(move):
        moves.append(move)

    # Move right up.
    move = (start[0] + 1, start[1] + 2)
    if verify(move):
        moves.append(move)

    # Move right down.
    move = (start[0] - 1, start[1] + 2)
    if verify(move):
        moves.append(move)

    # Move left up.
    move = (start[0] + 1, start[1] - 2)
    if verify(move):
        moves.append(move)

    # Move left down.
    move = (start[0] - 1, start[1] - 2)
    if verify(move):
        moves.append(move)

    return moves

def get_moves_knight_black(start: Tuple[int, int], chess_board: ndarray) -> List[Tuple[int, int]]:
    # Get all possible moves for a black knight at start.
    moves = []
    verify = lambda move: is_in_board(move) and (chess_board[move].isupper() or chess_board[move] == '.')

    # Move up right.
    move = (start[0] + 2, start[1] + 1)
    if verify(move):
        moves.append(move)

    # Move up left.
    move = (start[0] + 2, start[1] - 1)
    if verify(move):
        moves.append(move)

    # Move down right.
    move = (start[0] - 2, start[1] + 1)
    if verify(move):
        moves.append(move)

    # Move down left.
    move = (start[0] - 2, start[1] - 1)
    if verify(move):
        moves.append(move)

    # Move right up.
    move = (start[0] + 1, start[1] + 2)
    if verify(move):
        moves.append(move)

    # Move right down.
    move = (start[0] - 1, start[1] + 2)
    if verify(move):
        moves.append(move)

    # Move left up.
    move = (start[0] + 1, start[1] - 2)
    if verify(move):
        moves.append(move)

    # Move left down.
    move = (start[0] - 1, start[1] - 2)
    if verify(move):
        moves.append(move)

    return moves


def get_moves_bishop_white(start: Tuple[int, int], chess_board: ndarray) -> List[Tuple[int, int]]:
    # Get all possible moves for a white bishop at start.
    moves = []

    # Move up and right.
    for i in range(1, 8):
        if start[0] + i > 7 or start[1] + i > 7:
            break
        if chess_board[start[0] + i, start[1] + i] == '.':
            moves.append((start[0] + i, start[1] + i))
        elif chess_board[start[0] + i, start[1] + i].islower():
            moves.append((start[0] + i, start[1] + i))
            break
        else:
            break

    # Move up and left.
    for i in range(1, 8):
        if start[0] + i > 7 or start[1] - i < 0:
            break
        if chess_board[start[0] + i, start[1] - i] == '.':
            moves.append((start[0] + i, start[1] - i))
        elif chess_board[start[0] + i, start[1] - i].islower():
            moves.append((start[0] + i, start[1] - i))
            break
        else:
            break

    # Move down and right.
    for i in range(1, 8):
        if start[0] - i < 0 or start[1] + i > 7:
            break
        if chess_board[start[0] - i, start[1] + i] == '.':
            moves.append((start[0] - i, start[1] + i))
        elif chess_board[start[0] - i, start[1] + i].islower():
            moves.append((start[0] - i, start[1] + i))
            break
        else:
            break

    # Move down and left.
    for i in range(1, 8):
        if start[0] - i < 0 or start[1] - i < 0:
            break
        if chess_board[start[0] - i, start[1] - i] == '.':
            moves.append((start[0] - i, start[1] - i))
        elif chess_board[start[0] - i, start[1] - i].islower():
            moves.append((start[0] - i, start[1] - i))
            break
        else:
            break

    return moves

def get_moves_bishop_black(start: Tuple[int, int], chess_board: ndarray) -> List[Tuple[int, int]]:
    # Get all possible moves for a black bishop at start.
    moves = []

    # Move up and right.
    for i in range(1, 8):
        if start[0] + i > 7 or start[1] + i > 7:
            break
        if chess_board[start[0] + i, start[1] + i] == '.':
            moves.append((start[0] + i, start[1] + i))
        elif chess_board[start[0] + i, start[1] + i].isupper():
            moves.append((start[0] + i, start[1] + i))
            break
        else:
            break

    # Move up and left.
    for i in range(1, 8):
        if start[0] + i > 7 or start[1] - i < 0:
            break
        if chess_board[start[0] + i, start[1] - i] == '.':
            moves.append((start[0] + i, start[1] - i))
        elif chess_board[start[0] + i, start[1] - i].isupper():
            moves.append((start[0] + i, start[1] - i))
            break
        else:
            break

    # Move down and right.
    for i in range(1, 8):
        if start[0] - i < 0 or start[1] + i > 7:
            break
        if chess_board[start[0] - i, start[1] + i] == '.':
            moves.append((start[0] - i, start[1] + i))
        elif chess_board[start[0] - i, start[1] + i].isupper():
            moves.append((start[0] - i, start[1] + i))
            break
        else:
            break

    # Move down and left.
    for i in range(1, 8):
        if start[0] - i < 0 or start[1] - i < 0:
            break
        if chess_board[start[0] - i, start[1] - i] == '.':
            moves.append((start[0] - i, start[1] - i))
        elif chess_board[start[0] - i, start[1] - i].isupper():
            moves.append((start[0] - i, start[1] - i))
            break
        else:
            break

    return moves

def get_moves_queen_white(start: Tuple[int, int], chess_board: ndarray) -> List[Tuple[int, int]]:
    # Get all possible moves for a white queen at start.
    return get_moves_bishop_white(start, chess_board) + get_moves_rook_white(start, chess_board)

def get_moves_queen_black(start: Tuple[int, int], chess_board: ndarray) -> List[Tuple[int, int]]:
    # Get all possible moves for a black queen at start.
    return get_moves_bishop_black(start, chess_board) + get_moves_rook_black(start, chess_board)

def get_moves_king_white(start: Tuple[int, int], chess_board: ndarray) -> List[Tuple[int, int]]:
    # Get all possible moves for a black king at start.
    moves = []
    # Move up if possible.
    if start[0] < 7:
        if chess_board[start[0] + 1, start[1]] == '.' or chess_board[start[0] + 1, start[1]].islower():
            moves.append((start[0] + 1, start[1]))

    # Move down.
    if start[0] > 0:
        if chess_board[start[0] - 1, start[1]] == '.' or chess_board[start[0] - 1, start[1]].islower():
            moves.append((start[0] - 1, start[1]))

    # Move right.
    if start[1] < 7:
        if chess_board[start[0], start[1] + 1] == '.' or chess_board[start[0], start[1] + 1].islower():
            moves.append((start[0], start[1] + 1))

    # Move left.
    if start[1] > 0:
        if chess_board[start[0], start[1] - 1] == '.' or chess_board[start[0], start[1] - 1].islower():
            moves.append((start[0], start[1] - 1))

    # Move up and right.
    if start[0] < 7:
        if start[1] < 7:
            if chess_board[start[0] + 1, start[1] + 1] == '.' or chess_board[start[0] + 1, start[1] + 1].islower():
                moves.append((start[0] + 1, start[1] + 1))

    # Move up and left.
    if start[0] < 7:
        if start[1] > 0:
            if chess_board[start[0] + 1, start[1] - 1] == '.' or chess_board[start[0] + 1, start[1] - 1].islower():
                moves.append((start[0] + 1, start[1] - 1))

    # Move down and right.
    if start[0] > 0:
        if start[1] < 7:
            if chess_board[start[0] - 1, start[1] + 1] == '.' or chess_board[start[0] - 1, start[1] + 1].islower():
                moves.append((start[0] - 1, start[1] + 1))

    # Move down and left.
    if start[0] > 0:
        if start[1] > 0:
            if chess_board[start[0] - 1, start[1] - 1] == '.' or chess_board[start[0] - 1, start[1] - 1].islower():
                moves.append((start[0] - 1, start[1] - 1))

    # Castling.
    if start == (0, 4):
        # White king side.
        if chess_board[0, 5] == '.' and chess_board[0, 6] == '.' and chess_board[0, 7] == 'R':
            moves.append((0, 7))

        # White queen side.
        if chess_board[0, 3] == '.' and chess_board[0, 2] == '.' and chess_board[0, 1] == '.' and chess_board[0, 0] == 'R':
            moves.append((0, 0))


    return moves

def get_moves_king_black(start: Tuple[int, int], chess_board: ndarray) -> List[Tuple[int, int]]:
    # Get all possible moves for a black king at start.
    moves = []
    # Move up if possible.
    if start[0] < 7:
        if chess_board[start[0] + 1, start[1]] == '.' or chess_board[start[0] + 1, start[1]].isupper():
            moves.append((start[0] + 1, start[1]))

    # Move down.
    if start[0] > 0:
        if chess_board[start[0] - 1, start[1]] == '.' or chess_board[start[0] - 1, start[1]].isupper():
            moves.append((start[0] - 1, start[1]))

    # Move right.
    if start[1] < 7:
        if chess_board[start[0], start[1] + 1] == '.' or chess_board[start[0], start[1] + 1].isupper():
            moves.append((start[0], start[1] + 1))

    # Move left.
    if start[1] > 0:
        if chess_board[start[0], start[1] - 1] == '.' or chess_board[start[0], start[1] - 1].isupper():
            moves.append((start[0], start[1] - 1))

    # Move up and right.
    if start[0] < 7:
        if start[1] < 7:
            if chess_board[start[0] + 1, start[1] + 1] == '.' or chess_board[start[0] + 1, start[1] + 1].isupper():
                moves.append((start[0] + 1, start[1] + 1))

    # Move up and left.
    if start[0] < 7:
        if start[1] > 0:
            if chess_board[start[0] + 1, start[1] - 1] == '.' or chess_board[start[0] + 1, start[1] - 1].isupper():
                moves.append((start[0] + 1, start[1] - 1))

    # Move down and right.
    if start[0] > 0:
        if start[1] < 7:
            if chess_board[start[0] - 1, start[1] + 1] == '.' or chess_board[start[0] - 1, start[1] + 1].isupper():
                moves.append((start[0] - 1, start[1] + 1))

    # Move down and left.
    if start[0] > 0:
        if start[1] > 0:
            if chess_board[start[0] - 1, start[1] - 1] == '.' or chess_board[start[0] - 1, start[1] - 1].isupper():
                moves.append((start[0] - 1, start[1] - 1))

    # Castling.
    if start == (7, 4):
        # Castling kingside.
        if chess_board[7, 5] == '.' and chess_board[7, 6] == '.' and chess_board[7, 7] == 'r':
            moves.append((7, 7))
        # Castling queenside.
        if chess_board[7, 3] == '.' and chess_board[7, 2] == '.' and chess_board[7, 1] == '.' and chess_board[7, 0] == 'r':
            moves.append((7, 0))


    return moves

def get_possible_moves(start: Tuple[int, int], chess_board: ndarray) -> List[Tuple[int, int]]:
    # Get all possible moves for a piece at start.
    piece = chess_board[start]

    if piece == 'P':
        return get_moves_pawn_white(start, chess_board)
    elif piece == 'p':
        return get_moves_pawn_black(start, chess_board)
    elif piece == 'R':
        return get_moves_rook_white(start, chess_board)
    elif piece == 'r':
        return get_moves_rook_black(start, chess_board)
    elif piece == 'N':
        return get_moves_knight_white(start, chess_board)
    elif piece == 'n':
        return get_moves_knight_black(start, chess_board)
    elif piece == 'B':
        return get_moves_bishop_white(start, chess_board)
    elif piece == 'b':
        return get_moves_bishop_black(start, chess_board)
    elif piece == 'Q':
        return get_moves_queen_white(start, chess_board)
    elif piece == 'q':
        return get_moves_queen_black(start, chess_board)
    elif piece == 'K':
        return get_moves_king_white(start, chess_board)
    elif piece == 'k':
        return get_moves_king_black(start, chess_board)

    return []



def get_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter() - start} seconds")
        return result

    return wrapper

