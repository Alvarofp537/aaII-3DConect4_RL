import collections
import copy
import itertools
import random
import statistics
from typing import List
from connect_n_3d import ConnectN3DEnv, RandomAgent, Agent, ConnectNBoard3D

import numpy as np
import matplotlib.pyplot as plt
import torch

from A2C import A2C
from DuelingDQN import DuelingDQNAgent as DQN
from Blocker import Blocker, BlockerDeluxe
from Selfish import Selfish, IntelligentSelfish, Greedier
from Minimax import Minimax
from PPO import PPOAgent as PPO

def play_episode(*, env: ConnectN3DEnv, agents, learn: bool = False, imprimir= False) -> Agent:
    """Juega una partida completa entre varios agentes.

    :param env: El entorno ConnectN3D en el que se va a jugar.
    :param agents: Diccionario con los agentes que van a jugar.
    :param learn: Si los agentes deben aprender de la partida o no.
    :return: El agente que ha sido  el ganador de la partida.
    """
    obs, _ = env.reset()
    done, info = False, {}
    i = 0

    while not done:
        if i < 4 and np.count_nonzero(env.board.grid) < i:
            print('Alguien ha fallado ya!!')
            print(env.board.grid)
            print(f"{np.count_nonzero(env.board.grid) = }, {i = }")
            raise SystemError
        # A ver a quién le toca
        current_p = env.current_player

        agent = agents[current_p - 1]
        # El agente selecciona qué acción hacer
        action = agent.select_action(env.board)
        # La hace y ve qué ha pasado con el entorno
        next_obs, reward, done, _, info = env.step(action=action)
        # Si le permitimos aprender de ello, pues que aprenda
        if learn:
            agent.learn(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
            )
        # Vemos el nuevo estado del entorno para el próximo jugador
        obs = next_obs
        i += 1
    if imprimir:
        if not learn:
            print('Ganador:',agents[info["winner"] - 1] if info["winner"] else None)
        else:
            print('Ganador:',agents[info["winner"] - 1] if info["winner"] else None, end='\r')
    return agents[info["winner"] - 1] if info["winner"] else None




def tournament(
        *,
        agents: List[Agent],
        competitors_per_round: int = 4,
        num_rounds: int = 10,
        max_score: int = 10,
        score_to_substract: int = 5,
        max_training_rounds = np.inf,
        min_training_rounds = -np.inf,
        imprimir= False
):
    """Torneo de *N* jugadores.

    :param agents: Diccionario con los agentes a competir.
    :param competitors_per_round: Número de competidores por ronda.
    :param num_rounds: Número de rondas a jugar para cada grupo de competidores.
    :param max_score: Puntuación máxima que puede obtener un jugador.
    :param score_to_substract: Puntos a restar por cada posición en la que
        haya quedado el jugador.
    """
    # Combinaciones de N agentes tomados de 4 en 4 desordenadas
    groups = list(itertools.combinations(agents, 4))
    random.shuffle(groups)

    # Ahora, a competir por la gloria y la fama
    global_ranking = {agent.name: 0 for agent in agents}
    for i, group in enumerate(groups, start=1):
        print(f"Ronda {i}: {' vs. '.join(map(str, group))}")

        # Copiamos todos los agentes para que en cada ronda tengan su
        # el mismo estado inicial
        group = [copy.deepcopy(agent) for agent in group]
        random.shuffle(group)

        # Ronda de calentamiento (para aprender de los competidores). El
        # número de rondas de entrenamiento se extraerá aleatoriamente
        # a partir de los valores que los competidores declaren como
        # necesarios para aprender.
        rounds_for_all = [a.get_training_rounds() for a in group]
        train_rounds = abs(int(random.gauss(
            mu=statistics.mean(rounds_for_all),
            sigma=statistics.stdev(rounds_for_all)
        )))
        env = ConnectN3DEnv(num_players=competitors_per_round)
        if train_rounds > max_training_rounds:
            train_rounds = max_training_rounds
        if train_rounds < min_training_rounds:
            train_rounds = min_training_rounds

        # ENTRENAMIENTO
        train_winners = collections.Counter()
        evolution = {agent.name: [] for agent in group}
        log_every = max(1, train_rounds // 20)  # Cada cuántas rondas guardamos evolución

        for i in range(1, train_rounds + 1):
            random.shuffle(group)
            winner = play_episode(env=env, agents=group, learn=True, imprimir=imprimir)
            if winner:
                train_winners[winner.name] += 1
            # Guarda el progreso cada X rondas
            if i % log_every == 0 or i == train_rounds:
                for agent in group:
                    evolution[agent.name].append(train_winners[agent.name])
            print(f"Entrenando: {i} de {train_rounds} rondas", end="\r")
        
        if train_rounds > 10:
            # Mostrar resumen al final del entrenamiento
            print(f"\nEntrenamiento finalizado con {train_rounds} rondas")
            print("Victorias acumuladas:")
            for agent in group:
                print(f"\t{agent.name}: {train_winners[agent.name]}")
            # GRAFICAMOS LA EVOLUCIÓN
            plt.figure(figsize=(5, 2.5))
            for name, history in evolution.items():
                tramo_victorias = np.diff([0] + history)  # [0] + history para que diff funcione bien
                x_vals = np.linspace(log_every, train_rounds, num=len(tramo_victorias))
                plt.plot(x_vals, tramo_victorias, label=name)
            plt.xlabel("Rondas de entrenamiento")
            plt.ylabel("Victorias acumuladas")
            plt.title("Evolución de victorias durante el entrenamiento")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.show()

        # Ronda de competición. Ahora no van a aprender, sino que van a
        # competir directamente, y el ranking se asignará según el
        # resultado de la partida. Los puntos se irán repartiendo de
        # acuerdo al máximo en el caso del primer jugador, y se irán
        # restando puntos en función de la posición en la que haya
        # quedado el jugador.
        print('Jugamos')
        winners = collections.Counter([
            play_episode(env=env, agents=group, learn=False, imprimir=imprimir)
            for _ in range(num_rounds)
        ])


        # Ahora seguimos jugando hasta que todos los jugadores tengan
        # un número diferente de victorias. Esto es para evitar que
        # haya empates en el número de victorias.
        scores = list(winners.values())
        additional_matches = 0
        while len(scores) != len(set(scores)):
            winner = play_episode(env=env, agents=group, learn=False)
            winners[winner] += 1
            scores = list(winners.values())
            additional_matches += 1

        if additional_matches > 1:
            print(f"\tSe han necesitado {additional_matches} rondas adicionales")
        elif additional_matches == 1:
            print("\tSe ha necesitado una ronda adicional")

        # Actualizamos el ranking global
        for j, competitor in enumerate([w for w, _ in winners.most_common()]):
            if competitor:
                global_ranking[competitor.name] += max_score - score_to_substract * j

        env.close()

    ranking = sorted(global_ranking.items(), key=lambda r: r[1], reverse=True)

    print()
    print("CLASIFICACIÓN FINAL")
    print(f"{'Agente':>6} | {'Puntos':>7}")
    print("-" * 18)
    for agent, pts in ranking:
        print(f"{str(agent):>6} | {pts:>7}")
    print("-" * 18)

import torch

def train_multi_agents(agents: A2C, episodes=100, gamma=0.95, lam=0.95, imprimir=False):
    """
    Entrena múltiples agentes A2C en partidas entre sí usando GAE.

    :param agents: Lista de instancias A2C, una por jugador.
    :param episodes: Número de partidas para entrenar.
    :param gamma: Factor de descuento para recompensas futuras.
    :param lam: Parámetro de suavizado de ventaja para GAE.
    :param imprimir: Si se desea imprimir la evolución del entrenamiento.
    """
    env = ConnectN3DEnv(num_players=len(agents))

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        trajectory_data = {agent: [] for agent in agents}

        while not done:
            current_p = env.current_player
            agent = agents[current_p - 1]
            action, value = agent.select_action(env.board, returnValues= True)
            next_obs, reward, done, _, info = env.step(action=action)

            obs = agent.QuienSoy(obs, agent.pos)
            next_obs = agent.QuienSoy(next_obs, agent.pos)

            # 
            board = agent.obs_to_board(obs)
            next_board = agent.obs_to_board(next_obs)
            reward = agent.custom_reward(next_board, agent.pos) - agent.custom_reward(board, agent.pos)

            trajectory_data[agent].append((obs.copy(), action, reward, next_obs.copy(), done, value))

            obs = next_obs  # Actualizar el estado del juego para el siguiente turno

        # Una vez finalizado el episodio, aplicamos GAE a cada agente
        for agent in agents:
            agent.GAE_learn(trajectory_data[agent], gamma, lam)

        print(f"Episodio {episode + 1}/{episodes} completado", end="\r")

    env.close()
    print("Entrenamiento finalizado para todos los agentes.")

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch

def plot_3d_board(grid, ax, agent_colors):
    ax.clear()
    ax.set_title("3D Connect-N Board")
    ax.set_xlim(0, grid.shape[2])
    ax.set_ylim(0, grid.shape[1])
    ax.set_zlim(0, grid.shape[0])
    ax.set_xlabel("Y (depth)")
    ax.set_ylabel("X (width)")
    ax.set_zlabel("Z (height)")

    for z in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            for y in range(grid.shape[2]):
                player = grid[z, x, y]
                if player != 0:
                    color = agent_colors.get(player, "black")
                    ax.scatter(y + 0.5, x + 0.5, z + 0.5, c=color, s=200)

    ax.view_init(elev=30, azim=45)
    ax.grid(True)

def visualize_match(states, agent_names):
    """
    Visualize the evolution of a match with a legend.
    
    :param states: list of 3D numpy arrays (shape [height, width, depth])
    :param agent_names: list of agent names in the order they play (index 0 = player 1, etc.)
    """
    # Assign consistent colors to each player
    default_colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "cyan"]
    agent_colors = {i + 1: default_colors[i % len(default_colors)] for i in range(len(agent_names))}

    # For legend
    legend_elements = [
        Patch(facecolor=agent_colors[i + 1], edgecolor='black', label=f"{i+1}: {agent_names[i]}")
        for i in range(len(agent_names))
    ]

    def view_step(step):
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        plot_3d_board(states[step], ax, agent_colors)

        # Place legend outside the plot area
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    interact(
        view_step,
        step=IntSlider(min=0, max=len(states) - 1, step=1, value=0, description="Step")
    )



if __name__ == "__main__":
    tournament(
        agents=[RandomAgent(name=f'Agent {i}') for i in range(2)]+[A2C(name=f'A2c_{i}') for i in range(4)],
        competitors_per_round=4,
        num_rounds=10,
        max_score=3,
        score_to_substract=1,
    )
