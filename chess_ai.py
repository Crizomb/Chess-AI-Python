from chess_tools import get_possible_moves, move_piece, init_board, get_time
import numpy as np
import time
import random




inf = float("inf")

score_table = {"Q" : 9, "R" : 5, "B" : 3.3, "N" : 3.2, "P" : 1, "K" : 10**5 ,"q" : -9, "r" : -5, "b" : -3, "n" : -3, "p" : -1, "k" : -10**5, '':0, '.':0}

count = 0



class Heuristics:

    # The tables denote the points scored for the position of the chess pieces on the board.
    # Source : https://www.chessprogramming.org/Simplified_Evaluation_Function

    PAWN_TABLE = [
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 5, 10, 10,-20,-20, 10, 10,  5],
        [ 5, -5,-10,  0,  0,-10, -5,  5],
        [ 0,  0,  0, 20, 20,  0,  0,  0],
        [ 5,  5, 10, 25, 25, 10,  5,  5],
        [10, 10, 20, 30, 30, 20, 10, 10],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [ 0,  0,  0,  0,  0,  0,  0,  0]
    ]

    KNIGHT_TABLE = [
        [-50, -40, -30, -30, -30, -30, -40, -50],
        [-40, -20,   0,   5,   5,   0, -20, -40],
        [-30,   5,  10,  15,  15,  10,   5, -30],
        [-30,   0,  15,  20,  20,  15,   0, -30],
        [-30,   5,  15,  20,  20,  15,   0, -30],
        [-30,   0,  10,  15,  15,  10,   0, -30],
        [-40, -20,   0,   0,   0,   0, -20, -40],
        [-50, -40, -30, -30, -30, -30, -40, -50]
    ]

    BISHOP_TABLE = [
        [-20, -10, -10, -10, -10, -10, -10, -20],
        [-10,   5,   0,   0,   0,   0,   5, -10],
        [-10,  10,  10,  10,  10,  10,  10, -10],
        [-10,   0,  10,  10,  10,  10,   0, -10],
        [-10,   5,   5,  10,  10,   5,   5, -10],
        [-10,   0,   5,  10,  10,   5,   0, -10],
        [-10,   0,   0,   0,   0,   0,   0, -10],
        [-20, -10, -10, -10, -10, -10, -10, -20]
    ]

    ROOK_TABLE = [
        [ 0,  0,  0,  5,  5,  0,  0,  0],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [ 5, 10, 10, 10, 10, 10, 10,  5],
        [ 0,  0,  0,  0,  0,  0,  0,  0]
    ]

    QUEEN_TABLE = [
        [-20, -10, -10, -5, -5, -10, -10, -20],
        [-10,   0,   0,  0,  0,   0,   0, -10],
        [-10,   0,   0,  0,  0,   0,   0, -10],
        [  0,   0,   0,  0,  0,   0,   0,  -5],
        [ -5,   0,   0,  0,  0,   0,   0,  -5],
        [-10,   0,   0,  0,  0,   0,   0, -10],
        [-10,   0,   0,  0,  0,   0,   0, -10],
        [-20, -10, -10, -5, -5, -10, -10, -20]
    ]

    KING_TABLE_MG = [
        [ 20,  50,  10,   0,   0,  10,  10,  20],
        [ 20,  20,   0,   0,   0,   0,  20,  20],
        [-10, -20, -20, -20, -20, -20, -20, -10],
        [-20, -30, -30, -40, -40, -30, -30, -20],
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-30, -40, -40, -50, -50, -40, -40, -30]
    ]

    KING_TABLE_EG = [
        [-50, -30, -30, -30, -30, -30, -30, -50],
        [-30, -30,   0,   0,   0,   0, -30, -30],
        [-30, -10,  20,  30,  30,  20, -10, -30],
        [-30, -10,  30,  40,  40,  30, -10, -30],
        [-30, -10,  30,  40,  40,  30, -10, -30],
        [-30, -10,  20,  30,  30,  20, -10, -30],
        [-30, -20, -10,   0,   0, -10, -20, -30],
        [-50, -40, -30, -20, -20, -30, -40, -50]
    ]






def get_position_score(start, board):
    """Evaluate the position of a piece on the board."""
    global game_phase
    piece = board[start]
    if piece == 'P':
        return Heuristics.PAWN_TABLE[start[0]][start[1]]
    if piece == 'p':
        return -Heuristics.PAWN_TABLE[7 - start[0]][start[1]]
    if piece == 'N':
        return Heuristics.KNIGHT_TABLE[start[0]][start[1]]
    if piece == 'n':
        return -Heuristics.KNIGHT_TABLE[7 - start[0]][start[1]]
    if piece == 'B':
        return Heuristics.BISHOP_TABLE[start[0]][start[1]]
    if piece == 'b':
        return -Heuristics.BISHOP_TABLE[7 - start[0]][start[1]]
    if piece == 'R':
        return Heuristics.ROOK_TABLE[start[0]][start[1]]
    if piece == 'r':
        return -Heuristics.ROOK_TABLE[7 - start[0]][start[1]]
    if piece == 'Q':
        return Heuristics.QUEEN_TABLE[start[0]][start[1]]
    if piece == 'q':
        return -Heuristics.QUEEN_TABLE[7 - start[0]][start[1]]
    if piece == 'K':
        if game_phase < 45:
            return Heuristics.KING_TABLE_EG[start[0]][start[1]]
        else:
            return Heuristics.KING_TABLE_MG[start[0]][start[1]]
    if piece == 'k':
        if game_phase < 45:
            return -Heuristics.KING_TABLE_EG[7 - start[0]][start[1]]
        else:
            return -Heuristics.KING_TABLE_MG[7 - start[0]][start[1]]

    return 0


def get_king_safety(start, board):
    """King is safe if protected by a wall of pawns"""
    x, y = start
    piece = board[x, y]
    if piece == "K":
        king_safety = (board[max(x-1, 0), y+1] == "P") +  (board[x, y+1] == "P") +  (board[min(x+1, 7), y+1] == "P")
    elif piece == "k":
        king_safety = (board[max(x-1, 0), y-1] == "p") + (board[x, y-1] == "p") +  (board[min(x+1, 7), y-1] == "p")
        king_safety = -king_safety
    else:
        king_safety = 0
    return king_safety

def get_pawn_defend(start, board):
    """A good pawn is a pawn that defend another piece"""
    x, y = start
    piece = board[x, y]
    if piece == "P" and y+1<=7:
        if x-1 >= 0:
            if x+1<=7:
                pawn_defend = board[x-1, y+1].isupper() +  board[x+1, y+1].isupper()
            else:
                pawn_defend = board[x-1, y+1].isupper()
        else:
            pawn_defend =  board[x+1, y+1].isupper()

    elif piece == "p" and y-1>=0:
        if x-1 >= 0:
            if x+1<=7:
                pawn_defend = board[x-1, y-1].islower() +  board[x+1, y-1].islower()
            else:
                pawn_defend = board[x-1, y-1].islower()
        else:
            pawn_defend =  board[x+1, y-1].islower()

    else:
        pawn_defend = 0

    return pawn_defend

def get_rook_score(start, board):
    """A good rook is a rook that can move in a straight line"""
    x, y = start
    piece = board[x, y]
    rook_score = 0
    if piece == "R" or piece == "r":
        for i in range(1, 8):
            if x-i >= 0:
                if board[x-i, y] == ".":
                    rook_score += 1
                else:
                    break
            else:
                break
        for i in range(1, 8):
            if x+i <= 7:
                if board[x+i, y] == ".":
                    rook_score += 1
                else:
                    break
            else:
                break
        for i in range(1, 8):
            if y-i >= 0:
                if board[x, y-i] == ".":
                    rook_score += 2
                else:
                    break
            else:
                break
        for i in range(1, 8):
            if y+i <= 7:
                if board[x, y+i] == ".":
                    rook_score += 2
                else:
                    break
            else:
                break
    return rook_score



def get_other_eval(start, board):
    """Evaluation not based on the conventional chess pieces values and on the position"""
    king_safety = get_king_safety(start, board)
    pawn_defend = get_pawn_defend(start, board)
    rook_score = get_rook_score(start, board)

    return king_safety*10 + pawn_defend*10 + rook_score*2


def leaf_eval(board):
    """Evaluate a certain board position"""
    global count
    count += 1
    #evalute using score_table, position score and other eval
    evaluate = lambda x,y: score_table[board[x, y]] * 100 + get_position_score((x, y), board) + get_other_eval((x, y), board)
    return sum(evaluate(x, y) for y in range(8) for x in range(8)) + random.random()*10

game_phase = 0
def get_game_phase(board):
    global game_phase
    game_phase = sum(abs(score_table[board[x, y]]) for y in range(8) for x in range(8)) - 2 * score_table['K']


def minmax(node, depth, alpha=-inf, beta=inf):
    is_maximizing = node.color == 'white'
    if depth == 0 or node.is_leaf:
        return node.score
    if is_maximizing:
        score = -inf
        for child in node.children:
            score = max(score, minmax(child, depth - 1, alpha, beta))
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        return score
    else:
        score = inf
        for child in node.children:
            score = min(score, minmax(child, depth - 1, alpha, beta))
            beta = min(beta, score)
            if beta <= alpha:
                break
        return score

def evaluate_move(start, end, board):
    """Evaluate a move. Just used in "create_and_evaluate_tree" to sort the moves to optimize alpha-beta pruning"""
    position_score = get_position_score(start, board)
    take_score = score_table[board[end]]*100
    return position_score + take_score


class Node:
    __slots__ = 'board', 'move', 'parent', 'children', 'score', 'depth', 'color', 'board_score'
    def __init__(self, board, move, parent, children, score, depth, color):
        self.board = board
        self.move = move
        self.parent = parent
        self.children = children
        self.score = score
        self.depth = depth
        self.color = color

    @property
    def is_leaf(self):
        return not self.children

    @property
    def is_root(self):
        return self.parent is None

    def evalute_tree(self):
        if self.is_leaf:
            self.score = leaf_eval(self.board)
        else:
            for child in self.children:
                child.evalute_tree()
            self.score = minmax(self, self.depth)


    def create_and_evaluate_tree(self, depth, alpha=-inf, beta=inf):
        if depth == 0:
            self.score = leaf_eval(self.board)
        else:
            movable_pieces = ((x, y) for x in range(8) for y in range(8) if
                              self.board[x, y] != '.' and self.color == 'white' and self.board[x, y].isupper() or self.color == 'black' and self.board[x, y].islower())
            for piece in movable_pieces:
                possible_moves = get_possible_moves(piece, self.board)
                possible_moves.sort(key=lambda x: abs(evaluate_move(piece, x, self.board)), reverse=True)
                #possible_moves.sort(key=lambda x: evaluate_move(piece, x, self.board), reverse= self.color == "white")
                for move in possible_moves:
                    new_board = move_piece(piece, move, self.board)
                    new_move = (piece, move)
                    new_node = Node(new_board, new_move, self, [], 0, self.depth - 1, 'white' if self.color == 'black' else 'black')
                    self.children.append(new_node)
                    if self.color == 'white':
                        alpha = max(alpha, new_node.create_and_evaluate_tree(depth - 1, alpha, beta))
                        if beta <= alpha:
                            break
                        #To stop exploring if one king is taken
                        if abs(self.score) > 10**4:
                            break
                    else:
                        beta = min(beta, new_node.create_and_evaluate_tree(depth - 1, alpha, beta))
                        if beta <= alpha:
                            break
                        if abs(self.score) > 10**4:
                            break
            if not self.children:
                self.score = leaf_eval(self.board)
            else:
                self.score = max(child.score for child in self.children if child.score is not None) if self.color == 'white' else min(child.score for child in self.children if child.score is not None)
        return self.score


    def get_best_move(self):
        self.evalute_tree()
        maximazing = self.color == 'white'
        if maximazing:
            maxi = max(self.children, key=lambda x: x.score)
            print(f"maxi: {maxi.score}")
            return maxi.move
        else:
            mini = min(self.children, key=lambda x: x.score)
            print(f"mini: {mini.score}")
            return mini.move

    def pretty(self):
        string = f"Node: {self.move} {self.score} {self.depth} {self.color} \n"
        for child in self.children:
            string += "| " + child.pretty() + "\n"
        return string

    def __repr__(self):
        return self.pretty()


get_deeper = 0

def get_best_move(board, color, depth):
    global count, get_deeper
    get_game_phase(board)
    print(f"End game: {game_phase}")
    depth += get_deeper
    print(f"curent depth: {depth}")
    a = time.time()
    root = Node(board, None, None, [], 0, 0, color)
    root.create_and_evaluate_tree(depth)
    best_move = root.get_best_move()
    exec_time = time.time() - a
    print(f"Tree evaluated in {time.time() - a} seconds")
    print(f"Number of leafs: {count}")

    if exec_time < 1.5:
        get_deeper += 1
        print("Getting deeper, current depth: ", depth)
    elif exec_time > 20:
        get_deeper += -1
        print("Getting shallower, current depth: ", depth)

    print(f"Mean number of children : {count**(1/depth)}")
    count = 0


    return best_move

if __name__ == '__main__':
    board = init_board()
    b = get_best_move(board, 'white', 2)
    print(b)






















