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
                # Skip own ID (assuming the agent knows its ID if needed)
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
