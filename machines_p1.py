import numpy as np
from itertools import product

class P1:
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.board = board
        self.available_pieces = available_pieces
        self.depth = 3
        self.available_positions = self.get_available_locations()

    def set_dynamic_depth(self):
        empty_spaces = len(self.available_positions)
        if empty_spaces > 12:
            self.depth = 5
        elif 8 <= empty_spaces <= 12:
            self.depth = 10
        else:
            self.depth = 20

    def select_piece(self):
        self.set_dynamic_depth()
        _, piece = self.minimax_select_piece(self.depth, is_maximizing=False, alpha=-1e9, beta=1e9)
        # 선택된 피스가 없을 경우 예외 처리
        if piece is None:
            piece = self.available_pieces[0]  # 유효한 기본값 반환
        return piece

    def place_piece(self, selected_piece):
        self.set_dynamic_depth()
        _, position = self.minimax_place_piece(selected_piece, self.depth, is_maximizing=True, alpha=-1e9, beta=1e9)
        # 선택된 위치가 없을 경우 예외 처리
        if position is None:
            position = self.available_positions[0]  # 유효한 기본값 반환
        return position

    def minimax_select_piece(self, depth, is_maximizing, alpha, beta):
        if depth == 0 or self.is_terminal():
            return self.evaluate(), None

        best_piece = None
        if is_maximizing:
            max_eval = -1e9
            for piece in self.available_pieces:
                self.available_pieces.remove(piece)
                eval_score, _ = self.minimax_place_piece(piece, depth - 1, False, alpha, beta)
                self.available_pieces.append(piece)

                if eval_score > max_eval:
                    max_eval, best_piece = eval_score, piece
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_piece
        else:
            min_eval = 1e9
            for piece in self.available_pieces:
                self.available_pieces.remove(piece)
                eval_score, _ = self.minimax_place_piece(piece, depth - 1, True, alpha, beta)
                self.available_pieces.append(piece)

                if eval_score < min_eval:
                    min_eval, best_piece = eval_score, piece
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_piece

    def minimax_place_piece(self, piece, depth, is_maximizing, alpha, beta):
        if depth == 0 or self.is_terminal():
            return self.evaluate(), None

        best_position = None
        if is_maximizing:
            max_eval = -1e9
            for row, col in self.available_positions:
                self.board[row][col] = self.pieces.index(piece) + 1
                eval_score, _ = self.minimax_select_piece(depth - 1, False, alpha, beta)
                self.board[row][col] = 0

                if eval_score > max_eval:
                    max_eval, best_position = eval_score, (row, col)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_position
        else:
            min_eval = 1e9
            for row, col in self.available_positions:
                self.board[row][col] = self.pieces.index(piece) + 1
                eval_score, _ = self.minimax_select_piece(depth - 1, True, alpha, beta)
                self.board[row][col] = 0

                if eval_score < min_eval:
                    min_eval, best_position = eval_score, (row, col)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_position

    def check_win(self):
        for col in range(4):
            if self.check_line([self.board[row][col] for row in range(4)]):
                return True
        for row in range(4):
            if self.check_line([self.board[row][col] for col in range(4)]):
                return True
        if self.check_line([self.board[i][i] for i in range(4)]) or self.check_line([self.board[i][3 - i] for i in range(4)]):
            return True
        return self.check_2x2_subgrid_win()

    def check_2x2_subgrid_win(self):
        for row in range(3):
            for col in range(3):
                subgrid = [self.board[row + i][col + j] for i in range(2) for j in range(2)]
                if 0 not in subgrid and self.check_line(subgrid):
                    return True
        return False

    def check_line(self, line):
        if 0 in line:
            return False
        characteristics = np.array([self.pieces[piece_idx - 1] for piece_idx in line])
        for i in range(4):
            if len(set(characteristics[:, i])) == 1:
                return True
        return False

    def is_board_full(self):
        return all(self.board[row][col] != 0 for row in range(4) for col in range(4))

    def is_terminal(self):
        return self.is_board_full() or self.check_win()

    def evaluate(self):
        if self.check_win():
            return 100 if self.is_winner() else -100
        return 0

    def get_available_locations(self):
        return [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col] == 0]

    def is_winner(self):
        return self.check_win()
