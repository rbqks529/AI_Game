import numpy as np
import random
from itertools import product
import time

class P1():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = board  # 0: empty, 1~16: piece index
        self.available_pieces = available_pieces  # Currently available pieces

    def select_piece(self):
        # Use minimax with alpha-beta pruning to choose the piece that is least beneficial to the opponent
        _, piece = self.minimax_select_piece(depth=3, is_maximizing=False, alpha=-1e9, beta=1e9)
        return piece

    def place_piece(self, selected_piece):
        # Use minimax with alpha-beta pruning to place the selected piece optimally
        _, position = self.minimax_place_piece(selected_piece, depth=3, is_maximizing=True, alpha=-1e9, beta=1e9)
        return position

    def minimax_select_piece(self, depth, is_maximizing, alpha, beta):
        if depth == 0 or self.is_terminal():
            return self.evaluate(), None

        if is_maximizing:
            max_eval = -1e9
            best_piece = None
            for piece in self.available_pieces:
                self.available_pieces.remove(piece)
                eval_score, _ = self.minimax_place_piece(piece, depth - 1, False, alpha, beta)
                self.available_pieces.append(piece)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_piece = piece
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_piece

        else:  # Minimizing opponent's best piece
            min_eval = 1e9
            best_piece = None
            for piece in self.available_pieces:
                self.available_pieces.remove(piece)
                eval_score, _ = self.minimax_place_piece(piece, depth - 1, True, alpha, beta)
                self.available_pieces.append(piece)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_piece = piece

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_piece

    def minimax_place_piece(self, piece, depth, is_maximizing, alpha, beta):
        if depth == 0 or self.is_terminal():
            return self.evaluate(), None

        if is_maximizing:
            max_eval = -1e9
            best_position = None
            for row, col in self.get_available_locations():
                self.board[row][col] = self.pieces.index(piece) + 1
                eval_score, _ = self.minimax_select_piece(depth - 1, False, alpha, beta)
                self.board[row][col] = 0

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_position = (row, col)

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_position

        else:
            min_eval = 1e9
            best_position = None
            for row, col in self.get_available_locations():
                self.board[row][col] = self.pieces.index(piece) + 1
                eval_score, _ = self.minimax_select_piece(depth - 1, True, alpha, beta)
                self.board[row][col] = 0

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_position = (row, col)

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_position

    def check_win(self, board):
        # 보드의 승리 조건을 검사하는 함수 (기존에 사용한 check_win 함수와 동일)
        for col in range(4):
            if self.check_line([board[row][col] for row in range(4)]):
                return True
        for row in range(4):
            if self.check_line([board[row][col] for col in range(4)]):
                return True
        if self.check_line([board[i][i] for i in range(4)]) or self.check_line([board[i][3 - i] for i in range(4)]):
            return True
        return False

    def check_line(self, line):
        # 라인이 동일한 특성을 가진 조각들로 구성되었는지 확인
        if 0 in line:
            return False  # 불완전한 라인
        characteristics = np.array([self.pieces[piece_idx - 1] for piece_idx in line])
        for i in range(4):
            if len(set(characteristics[:, i])) == 1:
                return True
        return False

    def is_board_full(self, board):
        # 보드가 가득 찼는지 확인
        return all(board[row][col] != 0 for row in range(4) for col in range(4))
    
    def is_terminal(self):
        # Check if the board is full or if there's a winner
        return is_board_full() or check_win()

    def evaluate(self):
        # Implement a heuristic evaluation function based on board state
        if check_win():
            return 100 if self.is_winner() else -100
        return 0

    def get_available_locations(self):
        return [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col] == 0]

    def is_winner(self):
        # Define this function to check if the AI has won
        return check_win()  # Adjust according to your check_win function

    
    
    