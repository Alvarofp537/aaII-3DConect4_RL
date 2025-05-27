from connect_n_3d import Agent, ConnectNBoard3D
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQNNetwork(nn.Module):
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

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent(Agent):
    def __init__(self, name: str, model: DQNNetwork=DQNNetwork(7,4,6),  num_players=4, lr: float = 0.01, gamma: float = 0.95, epsilon: float = 0.1):
        super().__init__(name=name)
        self.model = model
        self.num_players = num_players
        self.epsilon = epsilon
        self.gamma=gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)        
    
    def select_action(self, board: ConnectNBoard3D) -> int:
        count = np.count_nonzero(board.grid)
        if count < 4:
            self.pos = count + 1

        legal_moves = board.legal_moves()

        """ Epsilon-Greedy """
        if random.random() < self.epsilon:
            # Acción aleatoria
            return random.choice(legal_moves)
        
        else:
            # Acción con mayor Q-valor
            state = torch.tensor(board.grid, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(state).detach().squeeze()  # shape: (num_actions)

            # Obtener índices de acciones legales en el vector de logits
            action_indices = [x * board.width + y for x, y in legal_moves]

            # Extraer los logits correspondientes
            legal_q_values = q_values[action_indices]

            # Softmax sobre acciones legales
            probs = torch.softmax(legal_q_values, dim=0).numpy()
            action_idx = np.random.choice(len(legal_moves), p=probs)
            
            return legal_moves[action_idx]

    def learn(self, *, obs, action, reward, next_obs, done):
        # Convertir a tensores
        state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)         # (1, D, W, H)
        next_state = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)

        # Calcular Q(s, a)
        q_values = self.model(state)                      # (1, num_actions)
        action_index = action[0] * self.model.width + action[1]  # acción linealizada
        q_value = q_values[0, action_index]

        # Calcular Q(s', a') para next_state
        with torch.no_grad():
            next_q_values = self.model(next_state)
            max_next_q_value = next_q_values.max()

        # Calcular target: r + gamma * max(Q(s', a')) * (1 - done)
        target = reward + self.gamma * max_next_q_value * (1 - done)

        # Calcular loss (error cuadrático medio)
        loss = nn.MSELoss()(q_value, target)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def get_training_rounds(self) -> int:
        """Returns training rounds,

        :return: x training rounds.
        """
        return 1000
