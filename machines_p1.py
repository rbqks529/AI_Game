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
            self.depth = 15
    
    def select_piece(self):
        self.set_dynamic_depth()
        _, piece = self.minimax_select_piece(self.depth, is_maximizing=False, alpha=-1e9, beta=1e9)
        return piece if piece is not None else self.available_pieces[0]

    def place_piece(self, selected_piece):
        self.set_dynamic_depth()
        _, position = self.minimax_place_piece(selected_piece, self.depth, is_maximizing=True, alpha=-1e9, beta=1e9)
        return position if position is not None else self.available_positions[0]

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
        
    def check_2x2_subgrid_win(self):
        for r in range(3):
            for c in range(3):
                subgrid = [self.board[r][c], self.board[r][c + 1], self.board[r + 1][c], self.board[r + 1][c + 1]]
                if 0 not in subgrid:
                    characteristics = [self.pieces[idx - 1] for idx in subgrid]
                    for i in range(4):
                        if len(set(char[i] for char in characteristics)) == 1:
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

    def is_board_full(self):
        return all(self.board[row][col] != 0 for row in range(4) for col in range(4))

    def is_terminal(self):
        return self.is_board_full() or self.check_win()

    def evaluate(self):
        if self.check_win():
            return 200 if self.is_winner() else -200
        return self.heuristic_score()

    def heuristic_score(self):
        score = 0
        for line in self.get_all_lines():
            score += self.analyze_line(line)
        score += self.analyze_subgrids()
        return score

    def analyze_line(self, line):
        # 실제 놓여진 피스만 필터링
        pieces = [self.pieces[piece_idx - 1] for piece_idx in line if piece_idx != 0]
        
        # 라인이 비어있으면 0 반환
        if not pieces:
            return 0
            
        # 특성별 분석
        score = 0
        for i in range(4):  # 4개의 특성에 대해
            # i번째 특성값들 추출
            characteristic_values = [piece[i] for piece in pieces]
            unique_values = set(characteristic_values)
            
            # 특성값 개수에 따른 점수 부여
            if len(characteristic_values) >= 2:  # 최소 2개 이상의 피스가 있을 때
                if len(unique_values) == 1:  # 모든 값이 동일
                    score += 150
                elif len(unique_values) == 2 and len(characteristic_values) == 3:  # 3개가 동일
                    score += 50
                elif len(unique_values) == 2 and len(characteristic_values) == 2:  # 2개가 동일
                    score += 10
                    
        return score

    def analyze_subgrids(self):
        score = 0
        for r in range(3):
            for c in range(3):
                subgrid = [self.board[r][c], self.board[r][c + 1], 
                        self.board[r + 1][c], self.board[r + 1][c + 1]]
                
                # 실제 놓여진 피스만 필터링
                pieces = [self.pieces[idx - 1] for idx in subgrid if idx != 0]
                
                # 서브그리드가 비어있으면 스킵
                if not pieces:
                    continue
                    
                # 특성별 분석
                for i in range(4):  # 4개의 특성에 대해
                    characteristic_values = [piece[i] for piece in pieces]
                    unique_values = set(characteristic_values)
                    
                    # 특성값 개수에 따른 점수 부여
                    if len(characteristic_values) >= 2:  # 최소 2개 이상의 피스가 있을 때
                        if len(unique_values) == 1:  # 모든 값이 동일
                            score += 150
                        elif len(unique_values) == 2 and len(characteristic_values) == 3:  # 3개가 동일
                            score += 50
                        elif len(unique_values) == 2 and len(characteristic_values) == 2:  # 2개가 동일
                            score += 10
                            
        return score

    def get_all_lines(self):
        # 모든 행
        lines = [[self.board[row][col] for col in range(4)] for row in range(4)]     
        # 모든 열
        lines += [[self.board[row][col] for row in range(4)] for col in range(4)]       
        # 대각선
        lines.append([self.board[i][i] for i in range(4)])
        lines.append([self.board[i][3-i] for i in range(4)])
        return lines

    def get_available_locations(self):
        return [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col] == 0]

    def is_winner(self):
        return self.check_win()
