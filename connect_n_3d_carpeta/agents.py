"""Agent base class plus two example agents."""
import abc
import random
from typing import Tuple

from .board import ConnectNBoard3D, Board


class Agent(metaclass=abc.ABCMeta):
    """Abstract agent interface."""

    def __init__(self, *, name: str):
        """Initializes this new instance.

        :param name: The agents name. This is used to identify the agent
            in the tournament.
        """
        self.name = name

    @abc.abstractmethod
    def get_training_rounds(self) -> int:
        """Estimated number of training rounds needed to learn.

        This should be an estimation of the number of rounds needed to
        learn the game. This is used to determine how many rounds
        to play in the tournament.

        :return: The number of training rounds.
        """

    @abc.abstractmethod
    def select_action(self, *, board: ConnectNBoard3D) -> Tuple[int, int]:
        """Which action to perform.

        :param board: The current game board state.
        :return: The decided (x, y) move for the current board state.
        """

    def learn(
            self, *,
            obs: Board,
            action: Tuple[int, int],
            reward: float,
            next_obs: Board,
            done: bool,
    ) -> None:
        """Update the agent's knowledge.

        This method is not mandatory for all agents. If the agent does
        not learn, this method can be ignored.

        :param obs: The current observation.
        :param action: The action taken.
        :param reward: The reward received.
        :param next_obs: The next observation.
        :param done: Whether the episode has ended.
        """
        pass

    def __str__(self) -> str:
        """Returns the agent name.

        :return: The agent name.
        """
        return self.name


class RandomAgent(Agent):
    """Uniformly random policy."""

    def select_action(self, board: ConnectNBoard3D) -> Tuple[int, int]:
        """Selects a random action.

        :param board: The current game board state.
        :return: A random (x, y) move for the current board state.
        """
        return random.choice(board.legal_moves())

    def get_training_rounds(self) -> int:
        """Returns 0 training rounds, as this agent does not learn.

        :return: 0 training rounds.
        """
        return 0
