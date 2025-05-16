from connect_n_3d import Agent, ConnectNBoard3D
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class A2CNetwork(nn.Module):
    def __init__(self, height, width, depth):
        """
        height, width, depth: dimensiones del tablero 
        num_players: número de jugadores
        """
        super().__init__()
        self.height = height
        self.width = width
        self.depth = depth
        self.num_actions = height * width  # Cada posición en la capa superior del tablero es una acción válida
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(height * width * depth, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        # Red para políticas (devuelve probabilidades sobre la capa superior del tablero)
        self.actor = nn.Linear(128, self.num_actions)
        # Red para el valor de estado
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.net(x)

        # Aplicamos softmax a la salida del actor para obtener probabilidades sobre todas las acciones posibles
        action_probs = F.softmax(self.actor(x), dim=-1)

        # Seleccionamos una acción basada en la distribución de probabilidades
        action_idx = torch.multinomial(action_probs, 1).item()

        # Convertimos el índice en coordenadas (x, y)
        x_coord = action_idx // self.width
        y_coord = action_idx % self.width

        # Estimación del valor del estado
        value = self.critic(x)
        
        return action_probs, (x_coord, y_coord), value






class A2C(Agent):
    """A2C agente, siguiend política determinista"""

    def __init__(self, name: str, model: A2CNetwork = A2CNetwork(6, 7, 4), gamma: float = 0.95, lr: float = 1e-3):
        super().__init__(name=name)
        self.model = model
        self.gamma = gamma  #valora recompensas futuras
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Verifica si hay un modelo guardado y cárgalo
        if os.path.exists("{self.name}.pth"):
            self.model.load_state_dict(torch.load("{self.name}.pth"))
            self.model.eval()  # Pone el modelo en modo evaluación para evitar cambios no deseados
        # Verifica si hay un modelo guardado y cárgalo
        elif os.path.exists("a2c_model.pth"):
            self.model.load_state_dict(torch.load("a2c_model.pth"))
            self.model.eval()  # Pone el modelo en modo evaluación para evitar cambios no deseados

    def select_action(self, board: ConnectNBoard3D) -> Tuple[int, int]:
        """Selects a action.

        :param board: The current game board state.
        :return: (x, y) move for the current board state.
        """

        state = torch.tensor(board.grid, dtype=torch.float32).unsqueeze(0)  # Aseguramos que sea un batch
        _, action, _ = self.model(state)
        return action # Devuelve las coordenadas


    def learn(self, obs, action, reward, next_obs, done):
        """
        Actualiza los parámetros del modelo de Actor-Critic usando la información pasada.
        
        :param obs: El estado actual del tablero.
        :param action: La acción tomada por el agente (x, y).
        :param reward: La recompensa obtenida.
        :param next_obs: El siguiente estado del tablero.
        :param done: Indica si el episodio ha terminado.
        """

        # Convertimos las observaciones a tensores
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0) 
        next_obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
        
        # Obtención de probabilidades de acción y valores del estado actual
        action_probs, _, value = self.model(obs)  # `action_probs` es el tensor de probabilidades

        # Estado siguiente
        _, _, next_value = self.model(next_obs)  # Valor del siguiente estado

        # Convertimos `(x, y)` en un índice válido dentro del vector de probabilidades
        action_idx = action[0] * self.model.width + action[1]  

        # **Corrección:** `action_probs` es un tensor, y ahora lo tratamos correctamente
        log_action_prob = torch.log(action_probs.view(-1)[action_idx])  

        # Calcular recompensa futura esperada
        target_value = reward + (1 - done) * self.gamma * next_value

        # Ventaja: diferencia entre el valor esperado y el valor actual
        advantage = target_value - value

        # Pérdidas del Actor y Crítico
        actor_loss = -log_action_prob * advantage.detach()  # La política maximiza la ventaja
        critic_loss = F.mse_loss(value, target_value.detach())  # La red crítica minimiza el error cuadrático

        # Pérdida total combinada
        loss = actor_loss + critic_loss

        # Retropropagación y optimización
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        torch.save(self.model.state_dict(), f"{self.name}.pth")



        
    def get_training_rounds(self) -> int:
        """Returns training rounds,

        :return: x training rounds.
        """
        return 6000
