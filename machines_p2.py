import numpy as np
import random, time
from itertools import product
from functools import lru_cache

WIN_SCORE = 1
DRAW_SCORE = 0
minimax_place: callable
selected_piece: callable

pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
offsets = [(0,0), (0,1), (1,0), (1,1)]

@lru_cache(maxsize=8192)
def evaluate_place(board_tuple, selected_piece):
    board = np.array(board_tuple)
    available_locs = [(row, col) for row, col in product(range(4), range(4)) if board[row][col]==0]
    for row, col in available_locs:
        new_board = np.copy(board)
        new_board[row][col] = pieces.index(selected_piece) + 1
        if(check_win(new_board)):
            return WIN_SCORE * 0.98
    if len(available_locs) == 1:
        return WIN_SCORE * 0.15
    else:
        return WIN_SCORE * 0.45

@lru_cache(maxsize=8192)
def evaluate_select(board_tuple, available_pieces_tuple):
    board = np.array(board_tuple)
    available_pieces = list(available_pieces_tuple)
    available_locs = [(row, col) for row, col in product(range(4), range(4)) if board[row][col]==0]
    
    for piece in available_pieces:
        safe_piece = True
        for row, col in available_locs:
            new_board = np.copy(board)
            new_board[row][col] = pieces.index(piece) + 1
            if(check_win(new_board)):
                safe_piece = False
                break
        if safe_piece:
            if (len(available_pieces) == 1):
                return WIN_SCORE * 0.35
            else:
                return WIN_SCORE * 0.65
    return -WIN_SCORE * 0.95

BOARD_ROWS = 4
BOARD_COLS = 4

def check_line(line):
    if 0 in line:
        return False
    characteristics = np.array([pieces[piece_idx - 1] for piece_idx in line])
    for i in range(4):
        if len(set(characteristics[:, i])) == 1:
            return True
    return False

def check_2x2_subgrid_win(board):
    for r in range(BOARD_ROWS - 1):
        for c in range(BOARD_COLS - 1):
            subgrid = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
            if 0 not in subgrid:
                characteristics = [pieces[idx - 1] for idx in subgrid]
                for i in range(4):
                    if len(set(char[i] for char in characteristics)) == 1:
                        return True
    return False

def check_win(board):
    for col in range(BOARD_COLS):
        if check_line([board[row][col] for row in range(BOARD_ROWS)]):
            return True
    for row in range(BOARD_ROWS):
        if check_line([board[row][col] for col in range(BOARD_COLS)]):
            return True
    if check_line([board[i][i] for i in range(BOARD_ROWS)]) or check_line([board[i][BOARD_ROWS - i - 1] for i in range(BOARD_ROWS)]):
        return True
    if check_2x2_subgrid_win(board):
        return True
    return False

def is_board_full(board):
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row][col] == 0:
                return False
    return True

def piece_to_mbti(piece):
    result = ""
    result += "E" if piece[0] == 1 else "I"
    result += "N" if piece[1] == 0 else "S"
    result += "F" if piece[2] == 1 else "T"
    result += "P" if piece[3] == 0 else "J"
    return result

def get_depth(available_pieces):
    if len(available_pieces) >= 11:
        return 3
    elif len(available_pieces) >= 10:
        return 5
    elif len(available_pieces) >= 9:
        return 6
    elif len(available_pieces) >= 8:
        return 8
    elif len(available_pieces) >= 6:
        return 9
    else:
        return 10

@lru_cache(maxsize=65536)
def minimax_select(is_maximizing, depth, available_pieces_tuple, board_tuple):
    board = np.array(board_tuple)
    available_pieces = list(available_pieces_tuple)
    
    if check_win(board):
        if is_maximizing:
            #print(f"14,000,605가지 중 이기는 유일한 미래!")
            return WIN_SCORE
        if not is_maximizing:
            #print(f"패배하는 미래...ㅠㅠ 먼 미래를 보고 있군")
            return -WIN_SCORE
            
    if is_board_full(board):
        print(f"비기는 평화의 미래")
        return DRAW_SCORE
        
    if is_maximizing:
        if depth == 0:
            return evaluate_select(board_tuple, available_pieces_tuple)
        best_score = -1e9
        for piece in available_pieces:
            new_available_pieces = list(available_pieces)
            new_available_pieces.remove(piece)
            new_available_pieces_tuple = tuple(new_available_pieces)
            score = minimax_place(False, depth - 1, new_available_pieces_tuple, board_tuple, piece)
            best_score = max(score, best_score)
        return best_score
    else:
        if depth == 0:
            return -evaluate_select(board_tuple, available_pieces_tuple)
        best_score = 1e9
        for piece in available_pieces:
            new_available_pieces = list(available_pieces)
            new_available_pieces.remove(piece)
            new_available_pieces_tuple = tuple(new_available_pieces)
            score = minimax_place(True, depth - 1, new_available_pieces_tuple, board_tuple, piece)
            best_score = min(score, best_score)
        return best_score

@lru_cache(maxsize=65536)
def minimax_place(is_maximizing, depth, available_pieces_tuple, board_tuple, selected_piece):
    board = np.array(board_tuple)
    available_pieces = list(available_pieces_tuple)
    
    if is_maximizing:
        if depth == 0:
            return evaluate_place(board_tuple, selected_piece)
        best_score = -1e9
        for row in range(4):
            for col in range(4):
                if board[row][col] == 0:
                    new_board = np.copy(board)
                    new_board[row][col] = pieces.index(selected_piece) + 1
                    board_tuple = tuple(map(tuple, new_board))
                    score = minimax_select(True, depth - 1, available_pieces_tuple, board_tuple)
                    best_score = max(score, best_score)
        return best_score
    else:
        if depth == 0:
            return -evaluate_place(board_tuple, selected_piece)
        best_score = 1e9
        for row in range(4):
            for col in range(4):
                if board[row][col] == 0:
                    new_board = np.copy(board)
                    new_board[row][col] = pieces.index(selected_piece) + 1
                    board_tuple = tuple(map(tuple, new_board))
                    score = minimax_select(False, depth - 1, available_pieces_tuple, board_tuple)
                    best_score = min(score, best_score)
        return best_score

class P2():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.board = board
        self.available_pieces = available_pieces
        
    def select_piece(self):
        start = time.time()
        print(f"=== 남은 말 {len(self.available_pieces)}개, 미니맥스 1 말 선택 중.. {time.strftime('%H:%M:%S')} ===")
        
        if len(self.available_pieces) == 16:
            random.seed(time.time() + len(self.available_pieces))
            result = random.choice(self.available_pieces)
            print(f"=== 미니맥스 1 {piece_to_mbti(result)} 선택, {int(time.time() - start)} 초 경과 ===")
            return result
            
        depth = get_depth(self.available_pieces)
        best_score = -1e9
        
        for piece in self.available_pieces:
            new_available_pieces = list(self.available_pieces)
            new_available_pieces.remove(piece)
            new_available_pieces_tuple = tuple(new_available_pieces)
            board_tuple = tuple(map(tuple, self.board))
            score = minimax_place(False, depth, new_available_pieces_tuple, board_tuple, piece)
            if score > best_score:
                best_score = score
                select = piece
                
        print(f"=== 미니맥스 1 {piece_to_mbti(select)} 선택, {int(time.time() - start)} 초 경과 ===")
        return select
        
    def place_piece(self, selected_piece):
        available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col]==0]
        
        start = time.time()
        print(f"=== 남은 말 {len(self.available_pieces)}개, 미니맥스 1 말 놓는 중.. {time.strftime('%H:%M:%S')} ===")
        
        if len(available_locs) == 16:
            random.seed(time.time_ns())
            result = random.choice(available_locs)
            print(f"=== 미니맥스 1 {result[0]}, {result[1]} 배치, {int(time.time() - start)} 초 경과 ===")
            return result
            
        depth = get_depth(self.available_pieces)
        best_score = -1e9
        
        for row, col in available_locs:
            new_board = np.copy(self.board)
            new_board[row][col] = self.pieces.index(selected_piece) + 1
            available_pieces_tuple = tuple(self.available_pieces)
            board_tuple = tuple(map(tuple, new_board))
            score = minimax_select(True, depth, available_pieces_tuple, board_tuple)
            if score > best_score:
                best_score = score
                place = (row, col)
                
        print(f"=== 미니맥스 1 {place[0]}, {place[1]} 배치, {int(time.time() - start)} 초 경과 ===")
        return place