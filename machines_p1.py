import sys
import pygame
import numpy as np
import random
from itertools import product
import time

class P1:
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.board = board
        self.available_pieces = available_pieces
        self.BOARD_SIZE = 4
        self.best_moves = {}
        self.minimax_calculated = False

    def select_piece(self):
        if len(self.available_pieces) > 10:
            return random.choice(self.available_pieces)
        elif not self.minimax_calculated:
            self.calculate_minimax()
        
        current_state = self.get_board_state()
        return max(self.available_pieces, key=lambda p: self.minimax(self.board, [p], False, float('-inf'), float('inf')))

    def place_piece(self, selected_piece):
        if len(self.available_pieces) > 10:
            available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col] == 0]
            return random.choice(available_locs)
        elif not self.minimax_calculated:
            self.calculate_minimax()
        
        current_state = self.get_board_state()
        return self.best_moves[current_state]['move']

    def calculate_minimax(self):
        self.best_moves = {}
        self.minimax(self.board, self.available_pieces, True, float('-inf'), float('inf'))
        self.minimax_calculated = True

    def minimax(self, board, available_pieces, is_maximizing, alpha, beta):
        state = self.get_board_state(board)
        if state in self.best_moves:
            return self.best_moves[state]['score']

        if self.check_win(board):
            return -1 if is_maximizing else 1

        if not available_pieces:
            return 0

        best_score = float('-inf') if is_maximizing else float('inf')
        best_piece = None
        best_move = None

        for piece in available_pieces:
            for row, col in product(range(self.BOARD_SIZE), range(self.BOARD_SIZE)):
                if board[row][col] == 0:
                    new_board = board.copy()
                    new_board[row][col] = self.pieces.index(piece) + 1
                    new_available_pieces = available_pieces[:]
                    new_available_pieces.remove(piece)

                    score = self.minimax(new_board, new_available_pieces, not is_maximizing, alpha, beta)

                    if is_maximizing:
                        if score > best_score:
                            best_score = score
                            best_piece = piece
                            best_move = (row, col)
                        alpha = max(alpha, best_score)
                    else:
                        if score < best_score:
                            best_score = score
                            best_piece = piece
                            best_move = (row, col)
                        beta = min(beta, best_score)

                    if beta <= alpha:
                        break
                
            if beta <= alpha:
                break

        self.best_moves[state] = {'score': best_score, 'piece': best_piece, 'move': best_move}
        return best_score

    def get_board_state(self, board=None):
        if board is None:
            board = self.board
        return tuple(map(tuple, board))

    def check_win(self, board):
        for i in range(self.BOARD_SIZE):
            if self.check_line([board[i][j] for j in range(self.BOARD_SIZE)]) or self.check_line([board[j][i] for j in range(self.BOARD_SIZE)]):
                return True
        if self.check_line([board[i][i] for i in range(self.BOARD_SIZE)]) or self.check_line([board[i][self.BOARD_SIZE-1-i] for i in range(self.BOARD_SIZE)]):
            return True
        return self.check_2x2_subgrid_win(board)

    def check_line(self, line):
        if 0 in line:
            return False
        characteristics = np.array([self.pieces[piece_idx - 1] for piece_idx in line])
        return any(len(set(characteristics[:, i])) == 1 for i in range(4))

    def check_2x2_subgrid_win(self, board):
        for r in range(self.BOARD_SIZE - 1):
            for c in range(self.BOARD_SIZE - 1):
                subgrid = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
                if 0 not in subgrid:
                    characteristics = [self.pieces[idx - 1] for idx in subgrid]
                    if any(len(set(char[i] for char in characteristics)) == 1 for i in range(4)):
                        return True
        return False