from connect_n_3d import Agent, ConnectNBoard3D
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class A2CNetwork(nn.Module):
    def __init__(self, height, width, depth, num_players):
        """
        height, width, depth: dimensiones del tablero 
        num_acctions: numero de acciones permitidas
        """
        super().__init__()
        self.input_size=height*width*depth
        self.num_players = num_players
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        # red para politicas (devuelve la acción tomada)
        self.actor = nn.Linear(128, 2)
        # Red para el valor de estado
        self.critic = nn.Linear(128, 1)


    def forward(self, x):
        """
        x: El estado de tablero.
        :return: la acción (policy) y el valor del estado (value).
        """
        x = self.net(x)

        #en caso de no ser determinista aplicar un softmax al actor
        action = self.actor(x).argmax(dim=-1)

        value =  self.critic(x)
        
        return action, value



class A2C(Agent):
    """A2C agente, siguiend política determinista"""

    def __init__(self, name: str, model: A2CNetwork, gamma: float = 0.95, lr: float = 1e-3):
        super().__init__(name=name)
        self.model = model
        self.gamma = gamma  #valora recompensas futuras
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, board: ConnectNBoard3D) -> Tuple[int, int]:
        """Selects a action.

        :param board: The current game board state.
        :return: (x, y) move for the current board state.
        """

        state = torch.tensor(board.grid(), dtype=torch.float32).unsqueeze(0)  # Aseguramos que sea un batch
        action, _ = self.model(state)
        return action.item() # Devuelve las coordenadas


    def learn(self, obs, action, reward, next_obs, done):
        """
        Actualiza los parámetros del modelo de Actor-Critic usando la información pasada.
        
        :param obs: El estado actual del tablero.
        :param action: La acción tomada por el agente.
        :param reward: La recompensa obtenida.
        :param next_obs: El siguiente estado del tablero.
        :param done: Indica si el episodio ha terminado.
        """

        # Convertimos las observaciones a tensores
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0) 
        next_obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
        
        # estado actual
        action, value = self.model(obs)  #accion y valor del estado actual
        # estado siguiente
        _, next_value = self.model(next_obs)  # Valor del siguiente estado

        # recompensa futura (recompensa + valor estimado del siguiente estado)
        target_value = reward + (1 - done) * self.gamma * next_value

        # ventaja: diferencia entre la recompensa obtenida y el valor actual
        advantage = target_value - value
 
        # pérdidas de la política y del valor
        actor_loss = -action * advantage.detach()  # La política maximiza la ventaja
        critic_loss = F.mse_loss(value, target_value.detach())  # La red crítica minimiza el error cuadrático

        # pérdida total
        loss = actor_loss + critic_loss

        # Realizamos el paso de retropropagación
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
    def get_training_rounds(self) -> int:
        """Returns training rounds,

        :return: x training rounds.
        """
        return 6000
