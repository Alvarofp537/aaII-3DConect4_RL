from connect_n_3d import Agent, ConnectNBoard3D
import random
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DuelingDQNNetwork(nn.Module):
    def __init__(self, height, width, depth):
        """
        height, width, depth: dimensiones del tablero 
        num_acctions: numero de acciones permitidas
        """
        super().__init__()
        self.input_size = height * width * depth
        self.num_actions = height * width

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



class DuelingDQNAgent(Agent):
    def __init__(self, name:str, model=DuelingDQNNetwork(7,4,6), num_players=4, lr: float = 1e-3, gamma: float = 0.95, epsilon: float = 0.1):
        super().__init__(name=name)
        self.model = model
        self.num_players = num_players
        self.epsilon = epsilon
        self.gamma=gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # Verifica si hay un modelo guardado y cárgalo
        if os.path.exists(f"./{self.name}.pth"):
            try:
                print(f"Cargando modelo desde ./{self.name}.pth")
                self.model.load_state_dict(torch.load(f"./{self.name}.pth"))
            except Exception as e:
                print(f"Error al cargar modelo: {e}")
            self.model.eval()

        # Guarda siempre al inicializar
        torch.save(self.model.state_dict(), f"{self.name}.pth")


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

    def select_action(self, board: ConnectNBoard3D, player_id: int) -> int:
        """
        Epsilon-Greedy con máscara de acciones válidas
        """
        if random.random() < self.epsilon:
            # Acción aleatoria válida
            x, y = random.choice(board.legal_moves())
            idx = x * self.model.depth + y
            return idx
        else:
            # Acción greedy entre las válidas
            obs = self.QuienSoy(board.grid, player_id)
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  #.unsqueeze(0)  otra vez
            with torch.no_grad():
                q_values = self.model(state).squeeze(0)

            mask = self.get_valid_mask(board)
            masked_q_values = q_values * mask + (1 - mask) * -1e9  # penaliza las inválidas
            action = masked_q_values.argmax().item()
            return action
        

    def learn(self, obs, action, reward, next_obs, done):
        """
        Actualiza los parámetros de la red DQN usando Bellman
        """
        try:
            board = self.obs_to_board(obs)
            next_board = self.obs_to_board(next_obs)
            obs_grid = self.QuienSoy(obs, self.pos)
            next_obs_grid = self.QuienSoy(next_obs, self.pos)

            obs = torch.tensor(obs_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            next_obs = torch.tensor(next_obs_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            q_values = self.model(obs)
            action_idx = action[0] * self.model.depth + action[1]
            q_value = q_values.view(-1)[action_idx]

            with torch.no_grad():
                next_q_values = self.model(next_obs)
                max_next_q_value = next_q_values.max(dim=1)[0]

            target_q_value = reward + (1 - done) * self.gamma * max_next_q_value
            loss = F.mse_loss(q_value, target_q_value)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        except Exception as e:
            print(f"Error en learn: {e}")

        finally:
            # Guardado automático tras cada learn, como en A2C
            torch.save(self.model.state_dict(), f"{self.name}.pth")


    def get_training_rounds(self) -> int:
        """Returns training rounds,

        :return: x training rounds.
        """
        return 1000
