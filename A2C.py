from connect_n_3d import Agent, ConnectNBoard3D
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class NoDeberíaPasar(Exception):
    pass

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
        self.num_actions = depth * width  # Cada posición en la capa superior del tablero es una acción válida
        
        self.net = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * height * width * depth, 128),
            nn.ReLU(),
        )
        
        # Red para políticas (devuelve probabilidades sobre la capa superior del tablero)
        self.actor = nn.Linear(128, self.num_actions)
        # Red para el valor de estado
        self.critic = nn.Linear(128, 1)

    def forward(self, x, valid_mask=None, deterministic=False):
        x = self.net(x)
        logits = self.actor(x)
        logits = torch.clamp(logits, -20, 20)

        if valid_mask is not None:
            # Poner -inf en acciones inválidas antes del softmax
            logits = logits.masked_fill(valid_mask == 0, float('-inf'))
        
        # Aplicamos softmax a la salida del actor para obtener probabilidades sobre todas las acciones posibles
        action_probs = F.softmax(logits, dim=-1)

        if deterministic:
            action_idx = torch.argmax(action_probs, dim=-1).item()
        else:
            action_idx = torch.multinomial(action_probs, 1).item()

        # Convertimos el índice en coordenadas (x, y)
        x_coord = action_idx // self.depth
        y_coord = action_idx % self.depth

        # Estimación del valor del estado
        value = self.critic(x)
        
        return action_probs, (x_coord, y_coord), value






class A2C(Agent):
    """A2C agente, siguiend política determinista"""

    def __init__(self, name: str, model: A2CNetwork = A2CNetwork(6, 7, 4), num_players=4, gamma: float = 0.95, lr: float = 1e-3):
        super().__init__(name=name)
        self.model = model
        self.gamma = gamma  #valora recompensas futuras
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Parámetros para saber si está aprendiendo
        self.played = 0
        self.learnt = 0
        self.num_players = num_players
        # QuienSoy?
        self.pos= None

        # Verifica si hay un modelo guardado y cárgalo
        if os.path.exists(f"./{self.name}.pth"):
            try:
                print(f"./{self.name}.pth", os.path.exists(f"./{self.name}.pth"))
                self.model.load_state_dict(torch.load(f"./{self.name}.pth"))
            except: pass
            self.model.eval()  # Pone el modelo en modo evaluación para evitar cambios no deseados
        torch.save(self.model.state_dict(), f"{self.name}.pth")

    def QuienSoy(self, grid: np.ndarray, player_id: int) -> np.ndarray:
        new_grid = grid.copy()
        new_grid[grid == player_id] = -1  # Soy yo
        if player_id < self.num_players:
            new_grid[grid == self.num_players] = player_id # Normalizamos siempre a [-1,0,1,2,3]
        return new_grid

    def get_valid_mask(self, board: ConnectNBoard3D) -> torch.Tensor:
        """Devuelve un tensor booleano con 1 para acciones válidas, 0 para inválidas."""
        valid = torch.zeros(self.model.num_actions, dtype=torch.float32)

        for x, y in board.legal_moves():
            idx = x * self.model.depth + y
            valid[idx] = 1.0

        return valid


    def select_action(self, board: ConnectNBoard3D, returnValues=False) -> Tuple[int, int]:
        """Selects a action.

        :param board: The current game board state.
        :return: (x, y) move for the current board state.
        """
        count = np.count_nonzero(board.grid)
        if count < self.num_players:
            self.pos = count + 1

        if self.played == self.learnt or count < self.num_players:
            deterministic = False
        else:
            deterministic = False # Mejor acción
        self.played += 1

        grid = self.QuienSoy(board.grid, self.pos)
        state = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        valid_mask = self.get_valid_mask(board).unsqueeze(0)

        _, action, value = self.model.forward(state, valid_mask=valid_mask, deterministic=deterministic)
        if returnValues:
            return action, value
        return action # Devuelve las coordenadas

    
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

    def count_connections(self, board: ConnectNBoard3D, player_id: int, length: int, open_ends: bool = True) -> int:
        count = 0
        visited = set()

        gx, gy, gz = board.width, board.depth, board.height
        grid = board.grid
        n = board.n_to_connect

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

                        # Check for open ends if requested
                        if open_ends:
                            before = (x - dx, y - dy, z - dz)
                            after = (x + length * dx, y + length * dy, z + length * dz)

                            open_before = (0 <= before[0] < gx and 0 <= before[1] < gy and 0 <= before[2] < gz and grid[before[2], before[0], before[1]] == 0)
                            open_after = (0 <= after[0] < gx and 0 <= after[1] < gy and 0 <= after[2] < gz and grid[after[2], after[0], after[1]] == 0)

                            if not (open_before or open_after):
                                continue

                        count += 1
        return count
    
    def custom_reward(self, board: ConnectNBoard3D, player_id: int) -> float:
        # Hiperparámetros (puedes ajustarlos luego)
        WEIGHTS = {
            "conn_2": 1.4,
            "conn_3": 5.0,
            "conn_blocked_penalty": 0.2,
            "opp_conn_2": 0.6,
            "opp_conn_3": 2.0,
            "center_bonus": 0.6,
            "height_bonus": 0.1,
        }

        score = 0.0

        # Bonificaciones por conexiones propias abiertas
        score += WEIGHTS["conn_2"] * self.count_connections(board, player_id, 2, open_ends=True)
        score += WEIGHTS["conn_3"] * self.count_connections(board, player_id, 3, open_ends=True)

        # Penalización por conexiones propias bloqueadas
        score -= WEIGHTS["conn_blocked_penalty"] * self.count_connections(board, player_id, 3, open_ends=False)

        # Penalizaciones por conexiones peligrosas del rival
        for opp_id in range(1, board.num_players + 1):
            if opp_id == player_id:
                continue
            score -= WEIGHTS["opp_conn_2"] * self.count_connections(board, opp_id, 2, open_ends=True)
            score -= WEIGHTS["opp_conn_3"] * self.count_connections(board, opp_id, 3, open_ends=True)

        # Control del centro (bonus por controlar el centro del tablero)
        cx, cy = board.width // 2, board.depth // 2
        score += WEIGHTS["center_bonus"] * np.count_nonzero(board.grid[:, cx, cy] == player_id)

        # Bonus por ocupar capas superiores
        for z in range(board.height):
            score += WEIGHTS["height_bonus"] * np.count_nonzero(board.grid[z] == player_id) * z

        return score


    def learn(self, obs, action, reward, next_obs, done):
        """
        Actualiza los parámetros del modelo de Actor-Critic usando la información pasada.
        
        :param obs: El estado actual del tablero.
        :param action: La acción tomada por el agente (x, y).
        :param reward: La recompensa obtenida.
        :param next_obs: El siguiente estado del tablero.
        :param done: Indica si el episodio ha terminado.
        """
        if reward < 0:
            print('Movimiento ilegal')
            raise NoDeberíaPasar()
        reward_extra = 0
        if reward == 1:
            reward_extra = 20
        torch.save(self.model.state_dict(), f"{self.name}.pth")
        self.learnt += 1
        if self.learnt < self.played: # Ronda nueva
            self.played = 0
            self.learnt = 0


        try:
            # Normalizamos el tablero
            obs = self.QuienSoy(obs, self.pos)
            next_obs = self.QuienSoy(next_obs, self.pos)

            # 
            board = self.obs_to_board(obs)
            next_board = self.obs_to_board(next_obs)
            
            # Convertimos las observaciones a tensores
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            next_obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


            valid_mask = self.get_valid_mask(board).unsqueeze(0) 
            next_valid_mask = self.get_valid_mask(next_board).unsqueeze(0)
            
            # Obtención de probabilidades de acción y valores del estado actual
            action_probs, _, value = self.model.forward(obs, valid_mask=valid_mask) # `action_probs` es el tensor de probabilidades

            if done: # Evitamos errores
                next_value = torch.tensor([[0.0]])
            else:
                _, _, next_value = self.model.forward(next_obs, valid_mask=next_valid_mask)


            # Convertimos `(x, y)` en un índice válido dentro del vector de probabilidades
            action_idx = action[0] * self.model.depth + action[1]  

            log_action_prob = torch.log(action_probs.view(-1)[action_idx])  

            # Calcular recompensa futura esperada
            reward = self.custom_reward(next_board, self.pos) - self.custom_reward(board, self.pos)
            if done:
                reward += reward_extra # Si ha ganado damos gran premio
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
        except Exception as e:
            print(str(e))
            # print(valid_mask)
            print(action_probs)
        finally:
            torch.save(self.model.state_dict(), f"{self.name}.pth")

    def compute_GAE(self, rewards, values, gamma=0.95, lam=0.95):
        """
        Calcula la estimación de ventaja generalizada (GAE).
        
        :param rewards: Lista de recompensas en el episodio.
        :param values: Lista de valores estimados por el crítico.
        :param gamma: Factor de descuento para recompensas futuras.
        :param lam: Parámetro de regularización de GAE.
        :return: Lista de ventajas ajustadas.
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            advantages[t] = last_advantage = delta + gamma * lam * last_advantage

        return advantages

    # Implementación dentro de la clase A2C
    def GAE_learn(self, trajectory, gamma=0.95, lam=0.95):
        """
        Aprende usando Generalized Advantage Estimation.
        
        :param trajectory: Lista de estados, acciones, recompensas y valores.
        :param gamma: Factor de descuento.
        :param lam: Parámetro de regularización.
        """
        advantages = []
        self.model.load_state_dict(torch.load(f"./{self.name}.pth"))
        for (obs, action, reward, next_obs, done, value) in trajectory:
            board = self.obs_to_board(obs)
            next_board = self.obs_to_board(next_obs)
            
            # Convertimos las observaciones a tensores
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            next_obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            valid_mask = self.get_valid_mask(board).unsqueeze(0)
            next_valid_mask = self.get_valid_mask(next_board).unsqueeze(0)

            # Obtención de probabilidades de acción y valores
            action_probs, _, value_pred = self.model.forward(obs, valid_mask=valid_mask)
            
            if done:
                next_value = torch.tensor([[0.0]])  # Valor futuro en estado terminal
            else:
                _, _, next_value = self.model.forward(next_obs, valid_mask=next_valid_mask)

            # Convertimos `(x, y)` en índice válido
            action_idx = action[0] * self.model.depth + action[1]

            log_action_prob = torch.log(action_probs.view(-1)[action_idx])

            # Calculamos ventaja con GAE
            delta = reward + gamma * next_value - value
            advantage = delta + gamma * lam * (advantages[-1] if advantages else 0)
            advantages.append(advantage)

            # Pérdidas Actor y Crítico
            actor_loss = -log_action_prob * advantage.detach()
            critic_loss = F.mse_loss(value_pred, (reward + gamma * next_value).detach())

            # Retropropagación y actualización
            loss = actor_loss + critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        torch.save(self.model.state_dict(), f"{self.name}.pth")

        
    def get_training_rounds(self) -> int:
        """Returns training rounds,

        :return: x training rounds.
        """
        return 1000
