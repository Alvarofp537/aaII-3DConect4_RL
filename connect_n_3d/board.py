"""3-D Connect-N board implementation.

This module contains an implementation of a generalised Connect-N board
in three dimensions. The board is designed to be used with multiple
players and supports a configurable number of tokens to connect in order
to win the game. The class exposes the minimal operations needed by
search agents or learning environments (e.g., Farama Gymnasium):

- place_token(...): Drops a token in a column (x, y).
- legal_moves(...): Enumerates all non-full columns.
- check_winner(...): Detects a completed line of length n_to_connect
    that includes the last placed token.
- is_full(...): Checks whether the board is full.
- clone(...): Deep-copy the board (useful for rollouts).

The implementation tries to avoid any game-flow logic so that the board
is reusable in environments with different penalty / reward schemes.
"""
from typing import List, Optional, Tuple, Annotated

import numpy as np
from numpy.typing import NDArray

Board = Annotated[NDArray[np.int8], "3D array of board state"]


class ConnectNBoard3D:
    """A 3‑Dimensional Connect‑N board.

    Coordinate system (obviously zero‑based):
        x: column index  [0, width‑1]
        y: depth  index  [0, depth‑1]
        z: height index  [0, height‑1] (0 is the bottom layer)
    """

    # The unique direction vectors needed to scan lines in 3‑D
    DIRECTIONS = [
        (1, 0, 0), (0, 1, 0), (0, 0, 1),  # axes
        (1, 1, 0), (1, 0, 1), (0, 1, 1),  # face diagonals
        (1, 1, 1),  # space diagonal
        (1, -1, 0), (1, 0, -1), (0, 1, -1),  # anti‑face diagonals
        (1, -1, 1), (1, 1, -1), (1, -1, -1),  # anti‑space diagonals
    ]

    def __init__(
            self, *,
            width: int = 7,
            depth: int = 4,
            height: int = 6,
            n_to_connect: int = 4,
            num_players: int = 2,
    ) -> None:
        """Initializes this instance.

        :param width: The width (x coordinate, aka. number of columns)
            of the board.
        :param depth: The depth of the board (y coordinate) of the
            board.
        :param height: The height of the board (z coordinate).
        :param n_to_connect: How many tokens must be in a straight line
            to win the game.
        :param num_players: How many players are going to play.
        :raise ValueError: Either width, depth or height are lower than
            1, n_to_connect is lower than 2, num_players is lower than 2
            or higher than 255, or if there is enough space for the
            players to at least win the match.
        """
        if width < 1:
            raise ValueError("Parameter width must be at least 1")
        if depth < 1:
            raise ValueError("Parameter depth must be at least 1")
        if height < 1:
            raise ValueError("Parameter height must be at least 1")
        if width * depth * height < n_to_connect * num_players:
            raise ValueError("The board is very small. Please increase the boards' width, depth or height")
        if n_to_connect < 2:
            raise ValueError("Parameter n_to_connect must be at least 2")
        if not (1 < num_players < 256):
            raise ValueError("Parameter num_players must be at least 2 and no more than 255")

        self.height = height
        self.width = width
        self.depth = depth
        self.n_to_connect = n_to_connect
        self.num_players = num_players

        self.grid: Board = np.zeros((height, width, depth), dtype=np.int8)
        self.last_move: Optional[Tuple[int, int, int, int]] = None  # (player, x, y, z)

    def reset(self) -> None:
        """Clear the board to its initial empty state."""
        self.grid.fill(0)
        self.last_move = None

    def legal_moves(self) -> List[Tuple[int, int]]:
        """Returns every column that has at least one empty slot.

        :return: Each (x, y) valid column to put a token.
        """
        column: List[Tuple[int, int]] = []
        top_layer = self.grid[self.height - 1]
        for x in range(self.width):
            for y in range(self.depth):
                if top_layer[x, y] == 0:
                    column.append((x, y))
        return column

    def place_token(self, player: int, x: int, y: int) -> int:
        """Drop player's token into column (x, y).

        The token occupies the lowest free z slot and the method returns
            that z coordinate.

        :param player: The player that's going to drop the token.
        :param x: The x position (width).
        :param y: The y position (height).
        :raise ValueError: If the column is full or indices are out of
            range.
        """
        if not (1 <= player <= self.num_players):
            raise ValueError("player id out of range")
        if not (0 <= x < self.width and 0 <= y < self.depth):
            raise ValueError("invalid column coordinates")

        column = self.grid[:, x, y]
        empty = np.where(column == 0)[0]
        if empty.size == 0:
            raise ValueError("column is full")
        else:
            z = int(empty[0])
            column[z] = player
            self.last_move = (player, x, y, z)
            return z

    def is_full(self) -> bool:
        """Whether the every slot in the board is occupied or not.

        :return: True if every slot is occupied, False otherwise.
        """
        return bool((self.grid != 0).all())

    def check_winner(self) -> Optional[int]:
        """Checks if there is a winner in the board and who.

        :return: The winner's id if any. None if there is no winner yet.
        """
        winner = None
        if self.last_move is not None:
            player, x0, y0, z0 = self.last_move
            n = self.n_to_connect
            gx, gy, gz = self.width, self.depth, self.height

            for dx, dy, dz in self.DIRECTIONS:
                count = 1  # include the last placed token

                # Forward
                x, y, z = x0 + dx, y0 + dy, z0 + dz
                while 0 <= x < gx and 0 <= y < gy and 0 <= z < gz and self.grid[z, x, y] == player:
                    count += 1
                    x, y, z = x + dx, y + dy, z + dz

                # Backward
                x, y, z = x0 - dx, y0 - dy, z0 - dz
                while 0 <= x < gx and 0 <= y < gy and 0 <= z < gz and self.grid[z, x, y] == player:
                    count += 1
                    x, y, z = x - dx, y - dy, z - dz

                if count >= n:
                    winner = player
        return winner

    def clone(self) -> "ConnectNBoard3D":
        """Returns a deep copy of the board.

        :return: An exact clone of this ConnectNBoard3D.
        """
        new_board = ConnectNBoard3D(
            height=self.height,
            width=self.width,
            depth=self.depth,
            n_to_connect=self.n_to_connect,
            num_players=self.num_players,
        )
        new_board.grid = self.grid.copy()
        new_board.last_move = self.last_move

        return new_board

    def __str__(self) -> str:  # pragma: no cover
        """A beautifull representation of the board.

        :return: An ASCII representation of the board in its current
            state.
        """
        layers = []
        for z in reversed(range(self.height)):
            layers.append(f"Level z={z}")
            for y in reversed(range(self.depth)):
                row = "".join(
                    str(self.grid[z, x, y]) or "."
                    for x in range(self.width)
                )
                layers.append(row)
            layers.append("")
        return "\n".join(layers)

    def __hash__(self) -> int:
        """A hash of the board state.

        This is very useful for memoization in search algorithms)
        """
        return hash(self.grid.tobytes())
