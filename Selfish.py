from connect_n_3d import Agent, ConnectNBoard3D
from typing import Tuple

import random
import numpy as np

class Selfish(Agent):
    """Agente que busca ganar lo más rápido posible ignorando a los rivales"""

    def __init__(self, name: str):
        super().__init__(name=name)
        self.pos = None

    def select_action(self, board: ConnectNBoard3D) -> Tuple[int, int]:
        """Selecciona la mejor jugada egoísta posible."""
        grid = board.grid

        # 1. Estimar ID del jugador
        if self.pos is None:
            count = np.count_nonzero(grid)
            self.pos = (count % board.num_players) + 1

        best_move = None
        best_score = -1

        # 2. Revisar todas las jugadas legales
        for x, y in board.legal_moves():
            temp_board = board.clone()

            try:
                z = temp_board.place_token(self.pos, x, y)
            except ValueError:
                continue

            # 2.1. Si gano al colocar aquí, juego esto
            if temp_board.check_winner() == self.pos:
                return (x, y)

            # 2.2. Si no gano aún, evalúo cuántas fichas conecto
            line_score = self.max_aligned(temp_board.grid, x, y, z, self.pos, board)
            if line_score > best_score:
                best_score = line_score
                best_move = (x, y)

        # 3. Si no hay buena jugada, juega algo válido al azar
        return best_move if best_move else random.choice(board.legal_moves())

    def max_aligned(self, grid, x0, y0, z0, player, board) -> int:
        """Evalúa el número máximo de fichas alineadas desde un punto (x0,y0,z0)."""
        max_count = 1
        for dx, dy, dz in board.DIRECTIONS:
            count = 1

            # Adelante
            x, y, z = x0 + dx, y0 + dy, z0 + dz
            while 0 <= x < board.width and 0 <= y < board.depth and 0 <= z < board.height and grid[z, x, y] == player:
                count += 1
                x, y, z = x + dx, y + dy, z + dz

            # Atrás
            x, y, z = x0 - dx, y0 - dy, z0 - dz
            while 0 <= x < board.width and 0 <= y < board.depth and 0 <= z < board.height and grid[z, x, y] == player:
                count += 1
                x, y, z = x - dx, y - dy, z - dz

            max_count = max(max_count, count)

        return max_count

    def learn(self, obs, action, reward, next_obs, done):
        pass  # No aprendizaje

    def get_training_rounds(self) -> int:
        return 0
