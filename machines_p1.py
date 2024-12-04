import numpy as np
from itertools import product
import time
import random
from copy import deepcopy
from functools import lru_cache

BOARD_ROWS = 4
BOARD_COLS = 4
WIN_SCORE = 1
DRAW_SCORE = 0
minimax_place: callable
selected_piece: callable
pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
offsets = [(0,0), (0,1), (1,0), (1,1)]


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
            #print(f"비기는 평화의 미래")
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

class QuartoState:
    def __init__(self, board, available_pieces, selected_piece=None):
        self.board = deepcopy(board)
        self.available_pieces = available_pieces.copy()
        self.selected_piece = selected_piece
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) 
                        for k in range(2) for l in range(2)]
        
    def clone(self):
        return QuartoState(self.board, self.available_pieces, self.selected_piece)
    
    def get_available_positions(self):
        return [(r, c) for r, c in product(range(4), range(4)) 
                if self.board[r][c] == 0]
    
    def check_win(self):
        # Check rows and columns
        for i in range(4):
            if self.check_line([self.board[i][j] for j in range(4)]) or \
                self.check_line([self.board[j][i] for j in range(4)]):
                return True
                
        # Check diagonals
        if self.check_line([self.board[i][i] for i in range(4)]) or \
            self.check_line([self.board[i][3-i] for i in range(4)]):
            return True
            
        # Check 2x2 subgrids
        return self.check_2x2_subgrid()
    
    def check_line(self, line):
        if 0 in line:
            return False
        characteristics = np.array([self.pieces[piece_idx - 1] for piece_idx in line])
        return any(len(set(characteristics[:, i])) == 1 for i in range(4))
    
    def check_2x2_subgrid(self):
        for r in range(3):
            for c in range(3):
                subgrid = [self.board[r][c], self.board[r][c + 1],
                            self.board[r + 1][c], self.board[r + 1][c + 1]]
                if 0 not in subgrid:
                    characteristics = [self.pieces[idx - 1] for idx in subgrid]
                    if any(len(set(char[i] for char in characteristics)) == 1 
                            for i in range(4)):
                        return True
        return False

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = self._get_untried_actions()
        
    def _get_untried_actions(self):
        if self.state.selected_piece is None:
            return [(piece, None) for piece in self.state.available_pieces]
        else:
            return [(None, pos) for pos in self.state.get_available_positions()]
            
    def get_ucb1(self, exploration= 1.41):  #exploration 값 설정, 기존: C ≈ sqrt(2) -> 1.41
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration_term = exploration * np.sqrt(np.log(self.parent.visits) / self.visits)
        return exploitation + exploration_term

    def is_terminal(self):
        return self.state.check_win() or not self.state.get_available_positions()

class P1:
    def __init__(self, board, available_pieces):
        self.state = QuartoState(board, available_pieces)
        self.board = board
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) 
                        for k in range(2) for l in range(2)]

    def adjust_simulation_time(self):
        remaining_piece = len(self.state.available_pieces)

        print(f"[DEBUG] remaining_piece: {remaining_piece}")
        # 진행 상황에 따라 가중치를 조정
        if remaining_piece > 12:  # 초반
            #time_weight = 0.5
            return 12
        elif remaining_piece > 10:  # 중반
            #time_weight = 1.0
            return 40
        elif remaining_piece > 8:  # 중반
            #time_weight = 1.0
            return 50
        elif remaining_piece > 6:  # 중반
            #time_weight = 1.0
            return 35
        else:  # 후반
            return 30
            #time_weight = 2.0
        
        # 남은 턴 수에 따라 동적 분배
        #dynamic_time = max((remaining_time / remaining_piece) * time_weight, 1.0)
        #return min(dynamic_time, 10.0)  # 최대 10초 제   
    
    def select_piece(self):
        """상대방에게 최악의 말을 선택하며 즉각적인 패배를 방지"""
        start_time = time.time()
        simulation_time = self.adjust_simulation_time()
        root = MCTSNode(self.state.clone())
        end_time = start_time + simulation_time

        print(f"[DEBUG] simulation_time: {simulation_time}")

        # 즉각적인 패배를 방지: 상대방이 바로 승리하지 않는 말을 우선적으로 선택
        safe_pieces = []
        for piece in self.state.available_pieces:
            if not self._can_opponent_win_next_turn(self.state, piece):
                safe_pieces.append(piece)

        # 안전한 말이 하나인 경우 바로 반환
        if len(safe_pieces) == 1:
            best_piece = safe_pieces[0]
            print(f"[DEBUG] Only one safe piece available: {best_piece}")
            self.state.available_pieces.remove(best_piece)
            self.state.selected_piece = best_piece
            return best_piece
        

        if len(self.state.available_pieces) <= 12:
            select = None
            depth = get_depth(self.state.available_pieces)
            best_score = -1e9

            if safe_pieces:
                print(f"[미니맥스] Safe pieces: {safe_pieces}")
                for piece in safe_pieces:
                    new_available_pieces = list(safe_pieces)
                    new_available_pieces.remove(piece)
                    new_available_pieces_tuple = tuple(new_available_pieces)
                    board_tuple = tuple(map(tuple, self.state.board))
                    score = minimax_place(False, depth, new_available_pieces_tuple, board_tuple, piece)
                    if score > best_score:
                        best_score = score
                        select = piece
            
            else:
                print(f"[미니맥스] Safe pieces 없음 -> 질걸?")
                for piece in self.state.available_pieces:
                    new_available_pieces = list(self.state.available_pieces)
                    new_available_pieces.remove(piece)
                    new_available_pieces_tuple = tuple(new_available_pieces)
                    board_tuple = tuple(map(tuple, self.state.board))
                    score = minimax_place(False, depth, new_available_pieces_tuple, board_tuple, piece)
                    if score > best_score:
                        best_score = score
                        select = piece

            # 선택된 말이 없으면 기본값 설정 (예: 첫 번째 말 선택)
            if select is None:
                select = safe_pieces[0] if safe_pieces else self.state.available_pieces[0]             

            print(f"=== 미니맥스 1 {piece_to_mbti(select)} 선택, {int(time.time() - start_time)} 초 경과 ===")
            return select

        # MCTS 탐색 시작
        if safe_pieces:  # 안전한 말이 있는 경우
            print(f"[MCTs] Safe pieces: {safe_pieces}")
            while time.time() < end_time:
                node = self._select(root)
                if not node.is_terminal() and node.untried_actions:
                    node = self._expand(node)
                simulation_result = self._simulate(node)
                self._backpropagate(node, simulation_result)

            #self.used_time += time.time() - start_time

            # 각 말을 평가하여 상대방에게 가장 불리한 말 선택
            best_piece = None
            worst_score = float('inf')

            for child in root.children:
                if child.action[0]:  # 피스 선택 액션인 경우
                    piece = child.action[0]
                    # MCTS 결과를 사용한 점수 계산
                    piece_score = (child.wins / child.visits if child.visits > 0 else 0) + \
                                0.3 * self._evaluate_piece_selection(piece, self.state)
                    if piece_score < worst_score:
                        worst_score = piece_score
                        best_piece = piece
            
            if best_piece:
                self.state.available_pieces.remove(best_piece)
                self.state.selected_piece = best_piece
                return best_piece
            
            # MCTS 탐색에서 적절한 결과가 없을 경우 안전한 말 중 임의로 선택
            best_piece = safe_pieces[0]
            # 선택된 말 제거 (게임 상태 반영)
            self.state.available_pieces.remove(best_piece)
            self.state.selected_piece = best_piece  # 현재 선택된 말을 설정
            return best_piece
        
        # 안전한 말이 없는 경우 (모든 말이 상대방의 승리를 유발) -> 혹시 모르니까 MCTS 진행
        print("[MCTs] No safe pieces available, performing MCTS with all pieces.")
        while time.time() < end_time:
            node = self._select(root)
            if not node.is_terminal() and node.untried_actions:
                node = self._expand(node)
            simulation_result = self._simulate(node)
            self._backpropagate(node, simulation_result)

        #self.used_time += time.time() - start_time

        # 모든 말을 대상으로 MCTS 결과를 바탕으로 선택
        best_piece = None
        worst_score = float('inf')

        for child in root.children:
            if child.action[0]:  # 모든 말을 고려
                piece = child.action[0]
                piece_score = (child.wins / child.visits if child.visits > 0 else 0) + \
                            0.3 * self._evaluate_piece_selection(piece, self.state)
                if piece_score < worst_score:
                    worst_score = piece_score
                    best_piece = piece

        # 선택된 말 제거 (게임 상태 반영)
        self.state.available_pieces.remove(best_piece)
        self.state.selected_piece = best_piece
        return best_piece


    def place_piece(self, selected_piece):
        """
        상대방이 선택한 말을 보드에 최적의 위치에 배치.
        """
        start = time.time()
        available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col]==0]

        self.state.selected_piece = selected_piece

        # 즉각적인 승리를 먼저 확인
        for position in self.state.get_available_positions():
            temp_state = self.state.clone()
            r, c = position
            temp_state.board[r][c] = self.pieces.index(selected_piece) + 1
            if temp_state.check_win():
                print(f"[DEBUG] Immediate win at position: {position}")
                return position 
            

        if len(self.state.available_pieces) <= 12:
            safe_locations = []   
            select = None
            safe_locations = self._filter_losing_positions(selected_piece) 
            
        
            # 1. 안전한 위치가 여러 개 있을 경우, minimax를 통해 최적의 위치를 선택
            if safe_locations:
                depth = get_depth(self.state.available_pieces)
                best_score = -1e9
                best_location = None
                for row, col in safe_locations:
                    new_board = np.copy(self.board)
                    new_board[row][col] = self.pieces.index(selected_piece) + 1
                    available_pieces_tuple = tuple(self.state.available_pieces)
                    board_tuple = tuple(map(tuple, new_board))
                    score = minimax_select(True, depth, available_pieces_tuple, board_tuple)
                    if score > best_score:
                        best_score = score
                        best_location = (row, col)
                print(f"=== 안전한 위치 중 최적 선택: {best_location} ===")
                return best_location

            # 2. 안전한 위치가 없으면 기존 minimax 로직 사용
            print("=== 모든 위치에서 상대방 승리를 막을 수 없음. 기존 전략 수행 ===")
            depth = get_depth(self.state.available_pieces)
            best_score = -1e9
            best_location = None
            for row, col in available_locs:
                new_board = np.copy(self.board)
                new_board[row][col] = self.pieces.index(selected_piece) + 1
                available_pieces_tuple = tuple(self.state.available_pieces)
                board_tuple = tuple(map(tuple, new_board))
                score = minimax_select(True, depth, available_pieces_tuple, board_tuple)
                if score > best_score:
                    best_score = score
                    best_location = (row, col)       

            print(f"=== 미니맥스 {best_location[0]}, {best_location[1]} 배치, {int(time.time() - start)} 초 경과 ===")
            return best_location


        start_time = time.time()
        simulation_time = self.adjust_simulation_time()
        end_time = start_time + simulation_time
        root = MCTSNode(self.state.clone())

        # MCTS 수행
        while time.time() < end_time:
            node = self._select(root)
            if not node.is_terminal() and node.untried_actions:
                node = self._expand(node)
            simulation_result = self._simulate(node)
            self._backpropagate(node, simulation_result)

        # 가장 많이 방문된 위치 선택
        best_child = max(root.children, key=lambda c: c.visits)
        best_position = best_child.action[1]

        print(f"[MCTs] Best position determined by MCTS: {best_position}")
        return best_position
    

    def _filter_losing_positions(self, selected_piece):
        """
        모든 위치에 말을 놓아보고, 내가 어떤 말을 줘도 상대방이 승리 가능한 경우를 필터링하여 제거.
        """
        available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col] == 0]
        truly_safe_locations = []  # 내가 어떤 말을 줘도 안전한 위치를 저장

        for row, col in available_locs:
            # 주어진 말을 현재 위치에 배치
            new_board = np.copy(self.board)
            new_board[row][col] = self.pieces.index(selected_piece) + 1

            # 상대방에게 줄 수 있는 모든 말을 시뮬레이션
            is_losing_position = True  # 기본값: 이 위치는 내가 지는 위치로 가정
            for opponent_piece in self.state.available_pieces:
                opponent_can_win = False

                # 상대방이 줄 말을 받았을 때 가능한 모든 위치 확인
                for opp_row, opp_col in [(r, c) for r, c in product(range(4), range(4)) if new_board[r][c] == 0]:
                    simulated_board = np.copy(new_board)
                    simulated_board[opp_row][opp_col] = self.pieces.index(opponent_piece) + 1

                    # 상대방이 승리 가능한 경우
                    if check_win(simulated_board):
                        opponent_can_win = True
                        break

                # 상대방이 승리 불가능한 경우가 하나라도 있다면 이 위치는 안전
                if not opponent_can_win:
                    is_losing_position = False
                    break

            # 이 위치가 안전하다면 추가
            if not is_losing_position:
                truly_safe_locations.append((row, col))

        # 안전한 위치 반환
        print(f"[DEBUG] Safe locations after losing positions filtered: {truly_safe_locations}")
        return truly_safe_locations
    
    
    def _select(self, node):
        while not node.is_terminal():
            if node.untried_actions:
                return node
            node = max(node.children, key=lambda c: c.get_ucb1())
        return node
    
    def _expand(self, node):
        action = node.untried_actions.pop()
        new_state = node.state.clone()
        
        if action[0]:  # piece selection
            new_state.available_pieces.remove(action[0])
            new_state.selected_piece = action[0]
        else:  # piece placement
            r, c = action[1]
            new_state.board[r][c] = self.pieces.index(new_state.selected_piece) + 1
            new_state.selected_piece = None
            
        child = MCTSNode(new_state, node, action)
        node.children.append(child)
        return child
    
    def _simulate(self, node):
        state = node.state.clone()
        current_player = 1  # 1: self, 0: opponent
        max_depth = 20  # 최대 시뮬레이션 깊이
        depth = 0

        while not self._is_game_over(state) and depth < max_depth:
            if state.selected_piece is None:
                if len(state.available_pieces) == 0:
                    return 0.5  # Draw
                
                # 상대방에게 최악의 말을 선택
                piece = self._smart_piece_selection(state, current_player) 
                state.available_pieces.remove(piece)
                state.selected_piece = piece
            else:
                # 최적의 위치에 배치
                positions = state.get_available_positions()
                r, c = self._smart_position_selection(state, positions, current_player)        
                state.board[r][c] = self.pieces.index(state.selected_piece) + 1
                state.selected_piece = None
                
                if state.check_win():
                    return 1 if current_player == 1 else 0
                
                current_player = 1 - current_player
            depth += 1

        return 0.5  # Draw if max_depth reached or game over


    def _smart_piece_selection(self, state, current_player):
        best_piece = None
        best_score = float('-inf') if current_player == 1 else float('inf')

        for piece in state.available_pieces:
            score = self._evaluate_piece_selection(piece, state)

            # 상대방이 이기는 말인지 확인
            if self._can_opponent_win_next_turn(state, piece):
                score -= 1000  # 즉각적인 패배를 방지

            if current_player == 1:
                if score > best_score:
                    best_score = score
                    best_piece = piece
            else:
                if score < best_score:
                    best_score = score
                    best_piece = piece

        return best_piece


    def _smart_position_selection(self, state, positions, current_player):
        best_pos = None
        best_score = float('-inf') if current_player == 1 else float('inf')

        for pos in positions:
            r, c = pos
            temp_state = state.clone()
            temp_state.board[r][c] = self.pieces.index(state.selected_piece) + 1

            # 즉시 승리 가능성 평가
            if temp_state.check_win():
                return pos  # 즉각 승리 가능 위치 선택

            # 위치 점수 계산
            score = self._count_potential_winning_lines(temp_state, r, c)

            # 중앙 위치 가중치 추가
            if r in [1, 2] and c in [1, 2]:
                score += 1  # 중앙 위치 선호

            # 상대방의 응수를 평가
            opponent_best_pos = self._simulate_opponent_response(temp_state)
            if opponent_best_pos:
                score -= 1000  # 상대방이 이기는 위치를 방치하지 않도록 패널티 부여

            if current_player == 1:
                if score > best_score:
                    best_score = score
                    best_pos = pos
            else:
                if score < best_score:
                    best_score = score
                    best_pos = pos

        return best_pos
    

    def _simulate_opponent_response(self, state):
        """
        상대방의 최적 응수를 시뮬레이션
        """
        opponent_best_score = float('-inf')
        opponent_best_pos = None

        for pos in state.get_available_positions():
            r, c = pos
            temp_state = state.clone()
            temp_state.board[r][c] = self.pieces.index(state.selected_piece) + 1
            if temp_state.check_win():
                return pos  # 즉각 승리 가능 위치

            score = self._count_potential_winning_lines(temp_state, r, c)
            if score > opponent_best_score:
                opponent_best_score = score
                opponent_best_pos = pos

        return opponent_best_pos


    
    def _backpropagate(self, node, result):
        depth = 0  # 현재 노드의 깊이를 추적
        while node:
            node.visits += 1
            # 깊이에 따른 가중치 감소
            weight = 1 / (1 + depth)
            node.wins += result * weight
            node = node.parent
            result = 1 - result  # 승패 반전
            depth += 1

    
    def _is_game_over(self, state):
        return state.check_win() or not state.get_available_positions()

    def _evaluate_piece_selection(self, piece, state):
        """
        상대방에게 주었을 때 위험도가 높은 말을 평가
        """
        score = 0
        for pos in state.get_available_positions():
            temp_state = state.clone()
            temp_state.board[pos[0]][pos[1]] = self.pieces.index(piece) + 1

            # 상대방이 이 말을 가지고 승리할 가능성을 평가
            if temp_state.check_win():
                score -= 1000  # 즉시 승리 가능성 높은 말은 높은 패널티

            # 해당 위치에서 생성될 승리 가능성 (승리 조건)을 점수화
            score -= self._count_potential_winning_lines(temp_state, pos[0], pos[1])

        # 피스 자체가 다수의 특성을 충족하는지 평가 (위험도 증가)
        for i in range(4):
            if len(set([p[i] for p in [self.pieces[idx - 1] for idx in state.board.flatten() if idx != 0]])) == 1:
                score -= 500  # 특정 특성을 충족시키는 말에 높은 패널티
        
        return -score



    def _can_opponent_win_next_turn(self, state, piece):
        """상대방이 다음 턴에 승리 가능한지 확인"""
        if piece is None:  # piece가 None일 경우 패스
            return False

        for pos in state.get_available_positions():
            temp_state = state.clone()
            temp_state.board[pos[0]][pos[1]] = self.pieces.index(piece) + 1
            if temp_state.check_win():
                return True  # 상대방이 이길 가능성이 있음
        return False



    def _count_potential_winning_lines(self, state, row, col):
        count = 0
        piece_idx = state.board[row][col] - 1
        piece = self.pieces[piece_idx] if piece_idx >= 0 else None  # 해당 위치의 말

        def is_line_winnable(line_positions):
            """라인이 승리 가능 상태인지 확인"""
            characteristics = [state.board[r][c] for r, c in line_positions]
            pieces_in_line = [p for p in characteristics if p != 0]
            
            if len(pieces_in_line) > 0:
                piece_features = [self.pieces[p-1] for p in pieces_in_line]
                # 동일한 특성을 가진 라인은 잠재적 승리 가능
                return all(
                    any(piece[i] == features[i] for features in piece_features)
                    for i in range(4)
                )
            return True  # 비어 있으면 잠재적 승리 가능

        # 가로, 세로, 대각선, 2x2 라인 점검
        lines = [
            [(row, c) for c in range(4)],  # 가로줄
            [(r, col) for r in range(4)],  # 세로줄
            [(i, i) for i in range(4)] if row == col else [],  # 주요 대각선
            [(i, 3 - i) for i in range(4)] if row + col == 3 else []  # 반대 대각선
        ]

        # 각 라인 점검
        for line in lines:
            if line and is_line_winnable(line):
                count += 1

        # 2x2 서브그리드 점검
        start_row, start_col = row // 2 * 2, col // 2 * 2
        subgrid_positions = [(r, c) for r in range(start_row, start_row + 2) for c in range(start_col, start_col + 2)]
        if is_line_winnable(subgrid_positions):
            count += 1

        return count