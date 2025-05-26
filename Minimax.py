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
        count = np.count_nonzero(board.grid)
        if self.pos is None and count < 4:
            self.pos = count + 1

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
            return self.custom_reward(board, self.pos), None

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

    
    def count_connections(self, board: ConnectNBoard3D, player_id: int, length: int, open_ends: bool = True) -> int:
        count = 0
        visited = set()

        gx, gy, gz = board.width, board.depth, board.height
        grid = board.grid
        n = board.n_to_connect

        for z in range(gz):
            for x in range(gx):
                for y in range(gy):
                    if grid[z, x, y] != player_id:
                        continue
                    for dx, dy, dz in board.DIRECTIONS:
                        key = tuple(sorted([(x + i * dx, y + i * dy, z + i * dz) for i in range(length)]))
                        if key in visited:
                            continue

                        segment = []
                        for i in range(length):
                            xi, yi, zi = x + i * dx, y + i * dy, z + i * dz
                            if 0 <= xi < gx and 0 <= yi < gy and 0 <= zi < gz and grid[zi, xi, yi] == player_id:
                                segment.append((xi, yi, zi))
                            else:
                                break
                        if len(segment) != length:
                            continue

                        visited.add(key)

                        # Check for open ends if requested
                        if open_ends:
                            before = (x - dx, y - dy, z - dz)
                            after = (x + length * dx, y + length * dy, z + length * dz)

                            open_before = (0 <= before[0] < gx and 0 <= before[1] < gy and 0 <= before[2] < gz and grid[before[2], before[0], before[1]] == 0)
                            open_after = (0 <= after[0] < gx and 0 <= after[1] < gy and 0 <= after[2] < gz and grid[after[2], after[0], after[1]] == 0)

                            if not (open_before or open_after):
                                continue

                        count += 1
        return count
    
    def custom_reward(self, board: ConnectNBoard3D, player_id: int) -> float:
        # Hiperparámetros (puedes ajustarlos luego)
        WEIGHTS = {
            "conn_2": 2.0,
            "conn_3": 6.0,
            "conn_2_fully_opened": 2.5,
            "conn_3_fully_opened": 7.0,
            "conn_blocked_penalty": 0.3,
            "opp_conn_2": 1.5,
            "opp_conn_3": 6.5,
            "center_bonus": 0.2,
            "height_bonus": 0.1,
        }

        score = 0.0

        # Bonificaciones por conexiones propias abiertas
        score += WEIGHTS["conn_2"] * self.count_connections(board, player_id, 2, open_ends=1, exact=True)
        score += WEIGHTS["conn_2_fully_opened"] * self.count_connections(board, player_id, 2, open_ends=2, exact=True)
        score += WEIGHTS["conn_3"] * self.count_connections(board, player_id, 3, open_ends=1, exact=True)
        score += WEIGHTS["conn_3_fully_opened"] * self.count_connections(board, player_id, 3, open_ends=2, exact=True)

        # Penalización por conexiones propias bloqueadas
        score -= WEIGHTS["conn_blocked_penalty"] * self.count_connections(board, player_id, 3, open_ends=False, exact=True)

        # Penalizaciones por conexiones peligrosas del rival
        for opp_id in range(1, board.num_players + 1):
            if opp_id == player_id:
                continue
            score -= WEIGHTS["opp_conn_2"] * self.count_connections(board, opp_id, 2, open_ends=True, exact=False)
            score -= WEIGHTS["opp_conn_3"] * self.count_connections(board, opp_id, 3, open_ends=True, exact=False)

        # Control del centro (bonus por controlar el centro del tablero)
        cx, cy = board.width // 2, board.depth // 2
        score += WEIGHTS["center_bonus"] * np.count_nonzero(board.grid[:, cx, cy] == player_id)

        # Bonus por ocupar capas superiores
        for z in range(board.height):
            score += WEIGHTS["height_bonus"] * np.count_nonzero(board.grid[z] == player_id) * z

        return score

    def learn(self, obs, action, reward, next_obs, done):
        pass

    def get_training_rounds(self) -> int:
        return 0
