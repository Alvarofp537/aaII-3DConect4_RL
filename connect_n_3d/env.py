"""Gymnasium environment for an N-player 3-D Connect-N board."""
from typing import Tuple, Optional, Dict, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .board import ConnectNBoard3D


class ConnectN3DEnv(gym.Env):
    """Turn-based N-player Gymnasium environment."""
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
            self,
            *,
            height: int = 6,
            width: int = 7,
            depth: int = 4,
            n_to_connect: int = 4,
            num_players: int = 2,
    ) -> None:
        """Initializes this instance.

        :param height: The height of the board (z coordinate).
        :param width: The width (x coordinate, aka. number of columns)
            of the board.
        :param depth: The depth of the board (y coordinate) of the
            board.
        :param n_to_connect: How many tokens must be in a straight line
            to win the game.
        :param num_players: How many players are going to play.
        :raise ValueError: If board is invalid (width, depth or height
            are lower than 1), n_to_connect is lower than 2, num_players
            is lower than 2 or higher than 255, or if there is enough
            space for the players to at least win the match.
        """
        super().__init__()

        try:
            self.board = ConnectNBoard3D(
                width=width,
                depth=depth,
                height=height,
                n_to_connect=n_to_connect,
                num_players=num_players,
            )
        except ValueError as e:
            raise ValueError("Invalid board parameters.") from e

        self.num_players = num_players

        # Action space: All available columns (x, y)
        self.action_space = spaces.MultiDiscrete([width, depth])
        # Observation space: Basically, the board (0 = empty, >0 = player id)
        self.observation_space = spaces.Box(
            low=0,
            high=self.board.num_players,
            shape=self.board.grid.shape,
            dtype=np.int8,
        )
        # Player IDs are 1 to num_players
        self.current_player = 1

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to an initial state.

        :param seed: Random seed for reproducibility.
        :param options: Additional options for the environment.
        :return: The initial observation and an empty info dictionary.
        """
        super().reset(seed=seed, options=options)
        self.board.reset()
        self.current_player = 1
        return self.get_obs(), {}

    def step(
            self, *,
            action: Tuple[int, int]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Performs one step in the environment.

        :param action: The action to perform (x, y) coordinates.
        :return: A tuple containing the next observation, reward,
            terminated flag, truncated flag, and additional info.
        """
        x, y = action

        try:
            self.board.place_token(self.current_player, x, y)
        except ValueError:
            # Illegal move; negative reward and change turn
            penalty = -1.0
            self.current_player = (self.current_player % self.num_players) + 1
            return self.get_obs(), penalty, False, False, {
                "illegal_move": True,
                "offender": (self.current_player - 1) or self.num_players,
            }

        # It was a valid move, so let's first check if we're done
        winner = self.board.check_winner()   # Either the player or None
        terminated = winner is not None or self.board.is_full()

        # ¡And now, the reward!
        if terminated:
            if winner is None:
                reward = 0.0  # Draw
            else:
                reward = 1.0  # The player won the match
        else:
            # There's no winner yet, so we give a reward of 0.0
            reward = 0.0
            self.current_player = (self.current_player % self.num_players) + 1

        return self.get_obs(), reward, terminated, False, {
            "winner": winner
        }

    def get_obs(self) -> np.ndarray:
        """Returns the current observation of the environment.

        :return: The current observation of the environment.
        """
        return self.board.grid.copy()

    def render(self, *, mode: str = "human") -> Optional[str]:
        """Renders the environment.

        :param mode: The rendering mode. Can be "human" or "ansi".
        :return: The rendered representation of the environment.
        """
        representation = str(self.board)
        if mode == "human":
            print(representation)
        return representation
