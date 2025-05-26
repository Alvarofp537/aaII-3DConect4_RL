from connect_n_3d import Agent, ConnectNBoard3D
from typing import Tuple

import random
import numpy as np



class Blocker(Agent):
    """Agente que bloquea"""

    def __init__(self, name: str, num_players=4, gamma: float = 0.95, lr: float = 1e-3):
        super().__init__(name=name)
        self.pos= None

    def select_action(self, board: ConnectNBoard3D) -> Tuple[int, int]:
        """Selects an action to block any imminent win from opponents.

        :param board: The current game board state.
        :return: (x, y) move for the current board state.
        """
        count = np.count_nonzero(board.grid)
        if count < 4:
            self.pos = count + 1

        for x, y in board.legal_moves():
            # Try placing a token for every opponent (not self)
            for player_id in range(1, board.num_players + 1):
                if self.pos != None and player_id == self.pos:
                    continue # Es él
                temp_board = board.clone()
                try:
                    temp_board.place_token(player_id, x, y)
                    if temp_board.check_winner() == player_id:
                        return (x, y)  # Block this winning move
                except ValueError:
                    continue  # Skip if the column is already full

        # No immediate threat found; pick a random legal move
        return random.choice(board.legal_moves())


    def learn(self, obs, action, reward, next_obs, done):
        """
        Actualiza los parámetros del modelo de Actor-Critic usando la información pasada.
        
        :param obs: El estado actual del tablero.
        :param action: La acción tomada por el agente (x, y).
        :param reward: La recompensa obtenida.
        :param next_obs: El siguiente estado del tablero.
        :param done: Indica si el episodio ha terminado.
        """
        pass # No aprende




        
    def get_training_rounds(self) -> int:
        """Returns training rounds,

        :return: x training rounds.
        """
        return 0


class BlockerDeluxe(Agent):
    """Agente que bloquea al jugador con la cadena más larga."""

    def __init__(self, name: str, num_players=4):
        super().__init__(name=name)
        self.pos = None  # Se detecta al inicio

    def select_action(self, board: ConnectNBoard3D) -> Tuple[int, int]:
        """Selecciona una acción que bloquee al jugador con mayor amenaza."""

        # Identificar nuestra posición (por orden de aparición)
        count = np.count_nonzero(board.grid)
        if count < 4:
            self.pos = count + 1

        # 1. Detectar el jugador con la cadena más larga
        max_len = 0
        biggest_threat = None

        for player_id in range(1, board.num_players + 1):
            if player_id == self.pos:
                continue  # Ignoramos a nosotros mismos
            length = self.max_consecutive_for_player(board, player_id)
            if length > max_len:
                max_len = length
                biggest_threat = player_id

        # 2. Si no hay amenaza, jugamos aleatorio
        if biggest_threat is None:
            return random.choice(board.legal_moves())

        # 3. Buscar la mejor jugada para bloquear al jugador más peligroso
        best_block = None
        max_future_threat = -1

        for x, y in board.legal_moves():
            temp_board = board.clone()
            try:
                temp_board.place_token(biggest_threat, x, y)
                new_threat = self.max_consecutive_for_player(temp_board, biggest_threat)
                if new_threat > max_future_threat:
                    max_future_threat = new_threat
                    best_block = (x, y)
            except ValueError:
                continue

        # 4. Si encontramos la mejor casilla para bloquear, la usamos
        if best_block:
            return best_block

        # 5. Si no, jugamos aleatorio
        return random.choice(board.legal_moves())

    def max_consecutive_for_player(self, board: ConnectNBoard3D, player_id: int) -> int:
        """Devuelve la longitud máxima de fichas conectadas para un jugador."""
        max_count = 0
        directions = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, -1, 0), (1, 0, 1), (0, 1, 1),
            (1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1)
        ]
        for x in range(board.width):
            for y in range(board.depth):
                for z in range(board.height):
                    if board.grid[x, y, z] != player_id:
                        continue
                    for dx, dy, dz in directions:
                        count = 1
                        nx, ny, nz = x + dx, y + dy, z + dz
                        while (
                            0 <= nx < board.width and
                            0 <= ny < board.depth and
                            0 <= nz < board.height and
                            board.grid[nx, ny, nz] == player_id
                        ):
                            count += 1
                            nx += dx
                            ny += dy
                            nz += dz
                        max_count = max(max_count, count)
        return max_count

    def learn(self, *args, **kwargs):
        pass  # No aprende

    def get_training_rounds(self) -> int:
        return 0