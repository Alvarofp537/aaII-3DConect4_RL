from connect_n_3d import Agent, ConnectNBoard3D
import random
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import copy


class DuelingDQNNetwork(nn.Module):
    def __init__(self, height, width, depth):
        """
        height, width, depth: dimensiones del tablero 
        num_acctions: numero de acciones permitidas
        """
        super().__init__()
        self.input_size = height * width * depth
        self.height = height
        self.width = width
        self.depth = depth
        self.num_actions = depth * width

        # Backbone común
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Rama para V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Rama para A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions)
        )

    def forward(self, x):
        """ Devuelve los Q-values para cada acción """
        x = self.net(x)

        value = self.value_stream(x)               # shape: (batch, 1)
        advantage = self.advantage_stream(x)       # shape: (batch, num_actions)

        # Dueling Q-values: Q(s, a) = V(s) + (A(s, a) - mean(A(s, ·)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values  # shape: (batch, num_actions)



class AMagno(Agent):
    def __init__(self, name:str = 'AMagno', model=DuelingDQNNetwork(6,7,4), num_players=4, lr: float = 0.01, gamma: float = 0.95, epsilon: float = 0.1):
        super().__init__(name=name)
        self.model = model
        self.num_players = num_players
        self.epsilon = epsilon
        self.gamma=gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # Cargamos lo aprendido
        self.model.load_state_dict(torch.load(f"./{self.name}.alguel"))
        self.model.eval()

        # Guarda siempre al inicializar
        torch.save(self.model.state_dict(), f"{self.name}.alguel")


        self.pos = None

    def QuienSoy(self, grid: np.ndarray, player_id: int) -> np.ndarray:
        new_grid = grid.copy()
        new_grid[grid == player_id] = -1  # Soy yo
        if player_id < 4:
            new_grid[grid == 4] = player_id # Normalizamos siempre [-1,0,1,2,3]
        # No se toca el resto: los demás jugadores se quedan como están (1, 2, 3, 4), excepto el propio
        return new_grid


    def get_valid_mask(self, board: ConnectNBoard3D) -> torch.Tensor:
        """Devuelve un tensor booleano con 1 para acciones válidas, 0 para inválidas."""
        valid = torch.zeros(self.model.num_actions, dtype=torch.float32)
        for x, y in board.legal_moves():
            idx = x * self.model.depth + y
            valid[idx] = 1.0
        return valid


    def obs_to_board(self, obs: np.ndarray) -> ConnectNBoard3D:
        board = ConnectNBoard3D(
            width=self.model.width,
            depth=self.model.depth,
            height=self.model.height,
            n_to_connect=4,
            num_players=self.num_players
        )
        board.grid = obs.copy()
        return board

    def __select_action(self, board: ConnectNBoard3D) -> int:
        """
        Epsilon-Greedy con máscara de acciones válidas
        """
        count = np.count_nonzero(board.grid)
        if count < self.num_players:
            self.pos = count + 1
            self.steps = 0
        if random.random() < self.epsilon:
            # Acción aleatoria válida
            action = random.choice(board.legal_moves())
            return action
        else:
            # Acción greedy entre las válidas
            obs = self.QuienSoy(board.grid, self.pos)
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  #.unsqueeze(0)  otra vez
            with torch.no_grad():
                q_values = self.model(state).squeeze(0)

            valid_mask = self.get_valid_mask(board)  # shape: (num_actions,)
            masked_q_values = q_values.clone()
            masked_q_values[valid_mask == 0] = -float('inf')  # Invalida las acciones ilegales

            action = masked_q_values.argmax().item()
            x = action // self.model.depth
            y = action % self.model.depth

            return x,y 
    
    def __select_learn(self, board):
        jugador_a_seguir = random.choice([i for i in range(4) if i != self.pos])

        # Nos gusta el centro
        return 0,0

    def select_action(self, board):
        count = np.count_nonzero(board.grid)
        if count < self.num_players:
            self.pos = count + 1
            self.learning = False # Nueva partida, no estamos aprendiendo
        if not self.learning:
            return self.__select_action(board)
        else:

            return self.__select_learn(board)

    def learn(self, obs, action, reward, next_obs, done):
        self.learning = True
        
    def count_connections(self, board: ConnectNBoard3D, player_id: int, length: int, open_ends: int = 1, exact: bool = False) -> int:
        count = 0
        visited = set()

        gx, gy, gz = board.width, board.depth, board.height
        grid = board.grid

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

                        # Evaluación de extremos abiertos
                        before = (x - dx, y - dy, z - dz)
                        after = (x + length * dx, y + length * dy, z + length * dz)

                        open_before = (0 <= before[0] < gx and 0 <= before[1] < gy and 0 <= before[2] < gz and grid[before[2], before[0], before[1]] == 0)
                        open_after  = (0 <= after[0] < gx and 0 <= after[1] < gy and 0 <= after[2] < gz and grid[after[2], after[0], after[1]] == 0)

                        total_open_ends = int(open_before) + int(open_after)

                        if exact:
                            if total_open_ends != open_ends:
                                continue
                        else:
                            if total_open_ends < open_ends:
                                continue

                        count += 1
        return count

    
    def custom_reward(self, board: ConnectNBoard3D, player_id: int, done, winner) -> float:
        # Hiperparámetros (puedes ajustarlos luego)
        WEIGHTS = {
            "conn_2": 2.0,
            "conn_3": 6.0,
            "conn_2_fully_opened": 2.5,
            "conn_3_fully_opened": 7.0,
            "conn_blocked_penalty": 0.3,
            "opp_conn_2": 1.5,
            "opp_conn_3": 6.5,
        }

        score = 0.0

        # En caso de ganar 50 puntos
        if done and winner:
            score += 50

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

        # # Control del centro (bonus por controlar el centro del tablero)
        # cx, cy = board.width // 2, board.depth // 2
        # score += WEIGHTS["center_bonus"] * np.count_nonzero(board.grid[:, cx, cy] == player_id)

        # # Bonus por ocupar capas superiores
        # for z in range(board.height):
        #     score += WEIGHTS["height_bonus"] * np.count_nonzero(board.grid[z] == player_id) * z

        return score

    def get_training_rounds(self) -> int:
        """Returns training rounds,

        :return: x training rounds.
        """
        return 0 # -100
