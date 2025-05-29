import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from connect_n_3d import Agent, ConnectNBoard3D
import os
import random

class PPOActorCritic(nn.Module):
    def __init__(self, height, width, depth, num_actions):
        super().__init__()
        self.input_size = height * width * depth
        self.height = height
        self.width = width
        self.depth = depth
        self.num_actions = depth * width

        self.base = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(128, num_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.base(x)
        return self.actor(x), self.critic(x)
    
class AbderramanIII(Agent):
    def __init__(self, name="AbderramanIII", height=6, width=7, depth=4, num_players=4, lr=1e-4, gamma=0.99, eps_clip=0.2):
        super().__init__(name=name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.height = height
        self.width = width
        self.depth = depth
        self.num_players = num_players
        self.num_actions = width * depth

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.model = PPOActorCritic(height, width, depth, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.rollout = []
        self.rewards = []

        # Cargamos lo que sabemos
        self.model.load_state_dict(torch.load(f"./{self.name}.alguel", map_location=self.device))

        self.model.train()
        self.pos = None
        self.learning = False

    def QuienSoy(self, grid: np.ndarray, player_id: int) -> np.ndarray:
        new_grid = grid.copy()
        new_grid[grid == player_id] = -1
        if player_id < 4:
            new_grid[grid == 4] = player_id
        return new_grid
    

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
    
    def get_valid_mask(self, board: ConnectNBoard3D) -> torch.Tensor:
            """Devuelve un tensor booleano con 1 para acciones válidas, 0 para inválidas."""
            valid = torch.zeros(self.model.num_actions, dtype=torch.float32)
            for x, y in board.legal_moves():
                idx = x * self.model.depth + y
                valid[idx] = 1.0
            return valid
    
    def __select_action(self, board: ConnectNBoard3D):

        obs = self.QuienSoy(board.grid, self.pos)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        logits, value = self.model(obs_tensor)

        # Enmascarar acciones inválidas
        logits = logits.squeeze(0)  # Quita dimensión batch: [28]
        valid_mask = self.get_valid_mask(board).to(self.device)  
        logits[valid_mask == 0] = -1e9  # Muy bajo para que tengan probabilidad ~0

        dist = torch.distributions.Categorical(logits=logits)

        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)

        x = action_idx.item() // self.depth
        y = action_idx.item() % self.depth

        self.last_obs = obs  # Para usarlo en learn()
        self.last_action = action_idx.item()
        self.last_log_prob = log_prob.item()
        self.last_value = value.item()

        return x, y

    def __select_blocker(self, board):
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

    def select_action(self, board):
        count = np.count_nonzero(board.grid)
        if count < self.num_players:
            self.pos = count + 1
            self.learning = False # Nueva partida, no estamos aprendiendo
        if not self.learning:
            return self.__select_action(board)
        else:

            return self.__select_blocker(board)

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



    def get_training_rounds(self):
        # return -100 # Se puede?
        return 0
