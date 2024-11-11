import numpy as np
import random
from itertools import product
import time
import copy

class P2():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.board = board  # 0: empty / 1~16: piece
        self.available_pieces = available_pieces

    def select_piece(self):
        # 최선의 조각을 찾기 위해 Minimax 알고리즘을 사용
        best_piece = None
        best_score = float('inf')  # 상대방의 최적 점수를 최소화

        for piece in self.available_pieces:
            simulated_score = self.minimax(self.board, piece, depth=3, maximizing_player=False)
            if simulated_score < best_score:
                best_score = simulated_score
                best_piece = piece

        return best_piece

    def place_piece(self, selected_piece):
        # 최선의 위치를 찾기 위해 Minimax 알고리즘을 사용
        best_move = None
        best_score = float('-inf')

        available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col] == 0]
        for row, col in available_locs:
            # 보드를 복사하여 가상의 움직임을 만듭니다.
            temp_board = copy.deepcopy(self.board)
            temp_board[row][col] = self.pieces.index(selected_piece) + 1  # 조각을 위치에 배치

            simulated_score = self.minimax(temp_board, selected_piece, depth=3, maximizing_player=True)
            if simulated_score > best_score:
                best_score = simulated_score
                best_move = (row, col)

        return best_move

    def minimax(self, board, piece, depth, maximizing_player):
        # 종료 조건
        if self.check_win(board):
            return 10 if maximizing_player else -10
        elif depth == 0 or self.is_board_full(board):
            return 0  # 무승부

        # 재귀적으로 모든 가능한 선택을 탐색
        if maximizing_player:
            max_eval = float('-inf')
            available_locs = [(row, col) for row, col in product(range(4), range(4)) if board[row][col] == 0]
            for row, col in available_locs:
                temp_board = copy.deepcopy(board)
                temp_board[row][col] = self.pieces.index(piece) + 1
                eval = self.minimax(temp_board, piece, depth - 1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for new_piece in self.available_pieces:
                eval = self.minimax(board, new_piece, depth - 1, True)
                min_eval = min(min_eval, eval)
            return min_eval

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