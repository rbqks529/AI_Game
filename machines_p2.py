import numpy as np
from itertools import product
import time
import random
from copy import deepcopy

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
            
    def get_ucb1(self, exploration=1.41):
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration_term = exploration * np.sqrt(np.log(self.parent.visits) / self.visits)
        return exploitation + exploration_term

    def is_terminal(self):
        return self.state.check_win() or not self.state.get_available_positions()

class P2:
    def __init__(self, board, available_pieces):
        self.state = QuartoState(board, available_pieces)
        self.simulation_time = 10.0  # 시간 제한을 1초로 조정
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) 
                        for k in range(2) for l in range(2)]
    
    def select_piece(self):
        """상대방에게 최악의 말을 선택"""
        root = MCTSNode(self.state.clone())
        end_time = time.time() + self.simulation_time
        
        while time.time() < end_time:
            node = self._select(root)
            if not node.is_terminal() and node.untried_actions:
                node = self._expand(node)
            simulation_result = self._simulate(node)
            self._backpropagate(node, simulation_result)
        
        # 각 피스에 대한 평가 점수를 계산
        best_piece = None
        worst_score = float('inf')
        
        for child in root.children:
            if child.action[0]:  # 피스 선택 액션인 경우
                piece = child.action[0]
                # MCTS 결과와 피스 평가를 결합
                piece_score = (child.wins / child.visits if child.visits > 0 else 0) + \
                             0.3 * self._evaluate_piece_selection(piece, self.state)
                if piece_score < worst_score:
                    worst_score = piece_score
                    best_piece = piece
        
        return best_piece

    def place_piece(self, selected_piece):
        self.state.selected_piece = selected_piece
        root = MCTSNode(self.state.clone())
        end_time = time.time() + self.simulation_time
        
        while time.time() < end_time:
            node = self._select(root)
            if not node.is_terminal() and node.untried_actions:
                node = self._expand(node)
            simulation_result = self._simulate(node)
            self._backpropagate(node, simulation_result)
        
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action[1]
    
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
        depth = 0
        max_depth = 40  # 최대 시뮬레이션 깊이
        
        while not self._is_game_over(state) and depth < max_depth:
            if state.selected_piece is None:
                if len(state.available_pieces) == 0:
                    return 0.5  # Draw
                
                # 스마트한 피스 선택
                if random.random() < 0.8:  # 80% 확률로 휴리스틱 사용
                    piece = self._smart_piece_selection(state, current_player)
                else:
                    piece = random.choice(state.available_pieces)
                    
                state.available_pieces.remove(piece)
                state.selected_piece = piece
            else:
                # 스마트한 위치 선택
                positions = state.get_available_positions()
                if not positions:
                    return 0.5  # Draw
                    
                if random.random() < 0.8:  # 80% 확률로 휴리스틱 사용
                    r, c = self._smart_position_selection(state, positions, current_player)
                else:
                    r, c = random.choice(positions)
                    
                state.board[r][c] = self.pieces.index(state.selected_piece) + 1
                state.selected_piece = None
                
                if state.check_win():
                    return 1 if current_player == 1 else 0
                
                current_player = 1 - current_player
                depth += 1
                
        return 0.5  # Draw

    def _smart_piece_selection(self, state, current_player):
        """Use heuristics for smarter piece selection in simulation"""
        best_piece = None
        best_score = float('-inf') if current_player == 1 else float('inf')
        
        for piece in state.available_pieces:
            score = self._evaluate_piece_selection(piece, state)
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
        """Use heuristics for smarter position selection in simulation"""
        best_pos = None
        best_score = float('-inf') if current_player == 1 else float('inf')
        
        for pos in positions:
            r, c = pos
            temp_state = state.clone()
            temp_state.board[r][c] = self.pieces.index(state.selected_piece) + 1
            
            score = self._count_potential_winning_lines(temp_state, r, c)
            if (r in [1,2] and c in [1,2]):  # 중앙 위치 선호
                score *= 1.2
                
            if current_player == 1:
                if score > best_score:
                    best_score = score
                    best_pos = pos
            else:
                if score < best_score:
                    best_score = score
                    best_pos = pos
                    
        return best_pos
    
    def _backpropagate(self, node, result):
        while node:
            node.visits += 1
            node.wins += result
            node = node.parent
            result = 1 - result  # 승패 반전
    
    def _is_game_over(self, state):
        return state.check_win() or not state.get_available_positions()

    def _evaluate_piece_selection(self, piece, state):
        """말 선택 시 상대방에게 얼마나 불리한지를 더 정교하게 평가"""
        score = 0
        for pos in state.get_available_positions():
            temp_state = state.clone()
            temp_state.board[pos[0]][pos[1]] = self.pieces.index(piece) + 1
            if temp_state.check_win():
                score -= 1000  # 즉시 승리 가능성이 있는 말은 매우 불리
            score += self._count_potential_winning_lines(temp_state, pos[0], pos[1])
        return -score  # 최악의 점수를 찾기 위해 부호 반전

    def _evaluate_board_state(self, state):
        """Evaluate current board state to adjust piece selection strategy"""
        filled_squares = sum(1 for row in state.board for cell in row if cell != 0)
        if filled_squares < 6:  # 게임 초반
            return 1.2  # 공격적 선택
        elif filled_squares < 12:  # 게임 중반
            return 1.0  # 균형잡힌 선택
        else:  # 게임 후반
            return 0.8  # 보수적 선택

    def _can_opponent_win_next_turn(self, state, last_pos):
        """상대방이 다음 턴에 승리 가능한 모든 조합을 시뮬레이션"""
        for pos in state.get_available_positions():
            temp_state = state.clone()
            temp_state.board[pos[0]][pos[1]] = self.pieces.index(state.selected_piece) + 1
            if temp_state.check_win():
                return True  # 상대방이 이길 가능성이 있음
        return False

    def _count_potential_winning_lines(self, state, row, col):
        """해당 위치에서 승리 가능한 라인의 수를 계산"""
        count = 0
        piece_idx = state.board[row][col] - 1
        current_piece = self.pieces[piece_idx]
        
        # 가로줄 검사
        row_pieces = [state.board[row][c] for c in range(4)]
        if self._has_winning_potential(row_pieces, current_piece):
            count += 1
            
        # 세로줄 검사
        col_pieces = [state.board[r][col] for r in range(4)]
        if self._has_winning_potential(col_pieces, current_piece):
            count += 1
            
        # 대각선 검사
        if row == col:
            diag_pieces = [state.board[i][i] for i in range(4)]
            if self._has_winning_potential(diag_pieces, current_piece):
                count += 1
                
        if row + col == 3:
            anti_diag_pieces = [state.board[i][3-i] for i in range(4)]
            if self._has_winning_potential(anti_diag_pieces, current_piece):
                count += 1
                
        # 2x2 서브그리드 검사
        for r in range(max(0, row-1), min(3, row+1)):
            for c in range(max(0, col-1), min(3, col+1)):
                subgrid = [state.board[r][c], state.board[r][c+1],
                            state.board[r+1][c], state.board[r+1][c+1]]
                if self._has_winning_potential(subgrid, current_piece):
                    count += 1
                
        return count

    def _has_winning_potential(self, line_pieces, current_piece):
        """해당 라인이 승리 가능성이 있는지 검사"""
        non_zero_pieces = [p for p in line_pieces if p != 0]
        if len(non_zero_pieces) <= 1:
            return True
            
        pieces_characteristics = [self.pieces[p-1] for p in non_zero_pieces if p != 0]
        for i in range(4):  # 4가지 특성에 대해
            if all(p[i] == current_piece[i] for p in pieces_characteristics):
                return True
        return False