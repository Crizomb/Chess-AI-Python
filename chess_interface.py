from tkinter import *
from chess_tools import *
from chess_ai import get_best_move
from PIL import Image, ImageTk, ImageDraw
import time

pieces = {"P": "♙", "R": "♖", "N": "♘", "B": "♗", "Q": "♕", "K": "♔", "p": "♟", "r": "♜", "n": "♞", "b": "♝", "q": "♛", "k": "♚", '':'', '.':'', '\x00': ' '}


MODE = "HUMAN VS AI"
COLOR = "BLACK"


def create_circle(x, y, r, canvas, fill="yellow"): #center coordinates, radius
    x0 = x - r
    y0 = y - r
    x1 = x + r
    y1 = y + r
    return canvas.create_oval(x0, y0, x1, y1, fill=fill)


class ChessInterface:

    __slots__ = 'root', 'canvas', 'size', 'square_size', 'selected', 'chess_board', 'count'
    def __init__(self, chess_board):
        self.root = Tk()
        self.root.title("Chess")
        self.canvas = Canvas(self.root, width=600, height=600, bg="white")
        self.size = 600 // 8 * 8
        self.square_size = self.size // 8
        self.selected = None
        self.chess_board = chess_board
        self.count = 0

        def ai_turn(color_ai):
            print(f"AI turn {color_ai}d")
            move = get_best_move(self.chess_board, color_ai, 3)
            print(f"AI move: {move}")
            self.chess_board = move_piece(move[0], move[1], self.chess_board)
            self.canvas.delete("all")
            self.draw_all(self.chess_board)
            self.count += 1
            self.selected = None


        def on_click_human_vs_ai_black(event):
            """AI play black"""
            global color
            x = event.x // self.square_size
            y = event.y // self.square_size
            self.canvas.delete("all")

            if self.selected is None:
                piece = self.chess_board[7-y, x]
                print(piece)
                color = "white" if piece.isupper() else "black"
                self.selected = (7-y, x)
            else:
                color_turn = "white" if self.count % 2 == 0 else "black"
                print(color, color_turn)
                if color == color_turn == "white":
                    print("Player turn")
                    if (7-y, x) in get_possible_moves(self.selected, self.chess_board):
                        self.count += 1
                        move = (self.selected, (7-y, x))
                        print(f"Player move: {move}")
                        self.chess_board = move_piece(move[0], move[1], self.chess_board)
                        self.draw_all(self.chess_board)
                        self.root.update()

                        # AI turn
                        ai_turn("black")

                self.selected = None
            self.draw_all(self.chess_board)

        first_round = True
        def on_click_human_vs_ai_white(event):
            """AI play white"""
            global color
            nonlocal first_round

            if first_round:
                # AI turn
                print("AI turn")
                move = get_best_move(self.chess_board, "white", 3)
                self.canvas.delete("all")
                self.draw_all(self.chess_board)
                print(f"AI move: {move}")
                self.chess_board = move_piece(move[0], move[1], self.chess_board)
                self.count += 1
                self.selected = None
                first_round = False

            x = event.x // self.square_size
            y = event.y // self.square_size
            self.canvas.delete("all")

            if self.selected is None:
                piece = self.chess_board[7-y, x]
                print(piece)
                color = "white" if piece.isupper() else "black"
                self.selected = (7-y, x)
            else:
                color_turn = "white" if self.count % 2 == 0 else "black"
                print(color, color_turn)
                if color == color_turn == "black":
                    print("Player turn")
                    if (7-y, x) in get_possible_moves(self.selected, self.chess_board):
                        self.count += 1
                        move = (self.selected, (7-y, x))
                        print(f"Player move: {move}")
                        self.chess_board = move_piece(move[0], move[1], self.chess_board)
                        self.draw_all(self.chess_board)
                        self.root.update()

                        # AI turn
                        ai_turn("white")

                self.selected = None
            self.draw_all(self.chess_board)

        def on_click_ai_vs_ai(event):
             """start AI vs AI when one click on the board"""
             while True:
                ai_turn("white")
                self.root.update()
                time.sleep(0.5)
                ai_turn("black")
                self.root.update()
                time.sleep(0.5)
                self.canvas.bind("<Button-1>", on_click_ai_vs_ai)

        game_modes = [on_click_human_vs_ai_black, on_click_human_vs_ai_white, on_click_ai_vs_ai]
        self.canvas.bind("<Button-1>", game_modes[1])
        self.canvas.pack()
        self.run()


    def draw_board(self):
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    color = "white"
                else:
                    color = "grey"
                self.canvas.create_rectangle(i * self.square_size, j * self.square_size, (i + 1) * self.square_size, (j + 1) * self.square_size, fill=color)


    def draw_pieces(self, chess_board):
        for i in range(8):
            for j in range(8):
                if chess_board[i, j] != " ":
                    self.canvas.create_text((j + 0.5) * self.square_size, (7-i + 0.5) * self.square_size, text=pieces[chess_board[i, j]], font=("Arial", 60))


    def draw_possible_moves(self, selected, chess_board):
        moves = get_possible_moves(selected, chess_board)
        if moves:
            for move in moves:
                #circle
                create_circle(move[1] * self.square_size + self.square_size // 2, (7-move[0]) * self.square_size + self.square_size // 2, self.square_size // 8, self.canvas)

    def draw_all(self, chess_board):
        self.draw_board()
        self.draw_pieces(chess_board)
        if self.selected is not None:
            self.draw_possible_moves(self.selected, chess_board)
        self.canvas.pack()

    def run(self):
        self.draw_all(self.chess_board)
        self.root.mainloop()

if __name__ == "__main__":
    chess_board = init_board()
    print(chess_board)
    ChessInterface(chess_board)




