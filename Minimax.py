from connect_n_3d import Agent, ConnectNBoard3D
from typing import Tuple

import random
import numpy as np

class Minimax(Agent):
    """Agente que usa Minimax para buscar la mejor jugada futura posible."""

    def __init__(self, name: str, depth: int = 3):
        super().__init__(name=name)
        self.depth = depth
        self.pos = None

    def select_action(self, board: ConnectNBoard3D) -> Tuple[int, int]:
        if self.pos is None:
            count = np.count_nonzero(board.grid)
            self.pos = (count % board.num_players) + 1

        _, action = self.minimax(board, self.depth, True, self.pos, float('-inf'), float('inf'))
        if action is None:
            return random.choice(board.legal_moves())
        return action

    def minimax(self, board: ConnectNBoard3D, depth: int, maximizing: bool, current_player: int, alpha: float, beta: float):
        winner = board.check_winner()
        if winner == self.pos:
            return 10000 + depth, None
        elif winner and winner != self.pos:
            return -10000 - depth, None
        elif depth == 0 or board.is_full():
            return self.evaluate(board), None

        best_value = float('-inf') if maximizing else float('inf')
        best_action = None

        for move in board.legal_moves():
            temp_board = board.clone()
            try:
                temp_board.place_token(current_player, *move)
            except ValueError:
                continue

            next_player = (current_player % board.num_players) + 1
            val, _ = self.minimax(temp_board, depth - 1, not maximizing, next_player, alpha, beta)

            if maximizing:
                if val > best_value:
                    best_value = val
                    best_action = move
                alpha = max(alpha, val)
            else:
                if val < best_value:
                    best_value = val
                    best_action = move
                beta = min(beta, val)

            if beta <= alpha:
                break  # poda alpha-beta

        return best_value, best_action

    def evaluate(self, board: ConnectNBoard3D) -> int:
        """Simple heuristic: maximize own alignment, penalize others'."""
        score = 0
        for z in range(board.height):
            for x in range(board.width):
                for y in range(board.depth):
                    player = board.grid[z, x, y]
                    if player == 0:
                        continue
                    val = self.count_alignment(board.grid, x, y, z, player, board)
                    if player == self.pos:
                        score += val
                    else:
                        score -= val
        return score

    def count_alignment(self, grid, x0, y0, z0, player, board):
        max_count = 1
        for dx, dy, dz in board.DIRECTIONS:
            count = 1
            x, y, z = x0 + dx, y0 + dy, z0 + dz
            while 0 <= x < board.width and 0 <= y < board.depth and 0 <= z < board.height and grid[z, x, y] == player:
                count += 1
                x, y, z = x + dx, y + dy, z + dz

            x, y, z = x0 - dx, y0 - dy, z0 - dz
            while 0 <= x < board.width and 0 <= y < board.depth and 0 <= z < board.height and grid[z, x, y] == player:
                count += 1
                x, y, z = x - dx, y - dy, z - dz

            max_count = max(max_count, count)
        return max_count

    def learn(self, obs, action, reward, next_obs, done):
        pass

    def get_training_rounds(self) -> int:
        return 0
