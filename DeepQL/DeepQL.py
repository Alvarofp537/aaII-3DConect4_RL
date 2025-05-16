from connect_n_3d import Agent, ConnectNBoard3D
import random

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
        self.num_actions = height * width

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


# RMSprop como optimizador 

class DQNAgent(Agent):
    def __init__(self, name: str, model: DQNNetwork=DQNNetwork(7,4,6), gamma: float = 0.95, lr: float = 1e-3, epsilon: float = 0.1):
        super().__init__(name=name)
        self.model = model
        self.gamma = gamma
        self.epsilon = epsilon
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)        

    # def select_action(self, board: ConnectNBoard3D) -> int:
    #     """
    #     Selecciona acción óptima (para jugar)
    #     """
    #     state = torch.tensor(board.grid, dtype=torch.float32).unsqueeze(0)
    #     q_values = self.model(state)
    #     action_index=q_values.argmax(dim=-1).item()
    #     x = action_index // 6
    #     y = action_index % 6 
    #     print("optima",x,y)
    #     return (x,y)
    
    def select_action(self, board: ConnectNBoard3D) -> int:
        """ Epsilon-Greedy """
        if random.random() < self.epsilon:
            # Acción aleatoria
            action_index = random.randint(0, self.model.num_actions - 1)
            x = action_index // 6
            y = action_index % 6 
            return (x, y)
        else:
            # Acción con mayor Q-valor
            state = torch.tensor(board.grid, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(state)
            action_index=q_values.argmax(dim=-1).item()
            x = action_index // 6
            y = action_index % 6 
            return (x,y)

    def learn(self, *, obs, action, reward, next_obs, done):
        """ Actualiza los parámetros de la red DQN """
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)

        # Convertimos (x, y) a índice lineal
        width = obs.shape[2]
        depth = obs.shape[3]
        action_index = action[0] * depth + action[1]
        action_index = torch.tensor([[action_index]], dtype=torch.int64)

        # Q-valor actual para la acción tomada
        q_values = self.model(obs)
        q_value = q_values.gather(1, action_index)

        # Q-valor objetivo usando Bellman
        with torch.no_grad():
            next_q_values = self.model(next_obs)
            max_next_q_value = next_q_values.max(dim=1, keepdim=True)[0]
            target_q_value = reward.unsqueeze(1) + (1 - done.unsqueeze(1)) * self.gamma * max_next_q_value

        # Pérdida MSE entre valor actual y objetivo
        loss = F.mse_loss(q_value.squeeze(), target_q_value)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_training_rounds(self) -> int:
        """Returns training rounds,

        :return: x training rounds.
        """
        return 600

