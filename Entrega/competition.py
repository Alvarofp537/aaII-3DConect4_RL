import collections
import copy
import itertools
import random
import statistics
from typing import List

from connect_n_3d import ConnectN3DEnv, RandomAgent, Agent


def play_episode(*, env: ConnectN3DEnv, agents, learn: bool = False) -> Agent:
    """Juega una partida completa entre varios agentes.

    :param env: El entorno ConnectN3D en el que se va a jugar.
    :param agents: Diccionario con los agentes que van a jugar.
    :param learn: Si los agentes deben aprender de la partida o no.
    :return: El agente que ha sido  el ganador de la partida.
    """
    obs, _ = env.reset()
    done, info = False, {}

    while not done:
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

    return agents[info["winner"] - 1] if info["winner"] else None


def tournament(
        *,
        agents: List[Agent],
        competitors_per_round: int = 4,
        num_rounds: int = 10,
        max_score: int = 10,
        score_to_substract: int = 5,
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
        for _ in range(train_rounds):
            _ = play_episode(env=env, agents=group, learn=True)

        # Ronda de competición. Ahora no van a aprender, sino que van a
        # competir directamente, y el ranking se asignará según el
        # resultado de la partida. Los puntos se irán repartiendo de
        # acuerdo al máximo en el caso del primer jugador, y se irán
        # restando puntos en función de la posición en la que haya
        # quedado el jugador.
        winners = collections.Counter([
            play_episode(env=env, agents=group, learn=False)
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


if __name__ == "__main__":
    from AbderramanIII import AbderramanIII
    from AMagno import AMagno
    from Heuristicos import Almanzor, Bismark
    tournament(
        agents=[AMagno(), Bismark(), AbderramanIII(), Almanzor()],
        competitors_per_round=4,
        num_rounds=10,
        max_score=10,
        score_to_substract=5,
    )
