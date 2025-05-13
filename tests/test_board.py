"""Tests for ConnectNBoard3D"""
import textwrap

import numpy as np
import pytest

from connect_n_3d import ConnectNBoard3D


@pytest.fixture
def default_board() -> ConnectNBoard3D:
    """A standard board for the game.

    :return: The board.
    """
    return ConnectNBoard3D()


@pytest.fixture
def cube_board() -> ConnectNBoard3D:
    """A cube board of size 4 for 2 players with a line of 4 tokens to win.

    :return: The board.
    """
    return ConnectNBoard3D(height=4, width=4, depth=4, n_to_connect=4)


def test_invalid_n_to_connect():
    """ValueError is raised when n_to_connect is less than 2."""
    for i in range(-1, 2):
        with pytest.raises(ValueError, match="n_to_connect"):
            ConnectNBoard3D(n_to_connect=i)


def test_invalid_num_players():
    """ValueError is raised when num_players is less than 2."""
    for i in range(-1, 2):
        with pytest.raises(ValueError, match="num_players"):
            ConnectNBoard3D(num_players=i)


def test_place_token_invalid_player(default_board):
    with pytest.raises(ValueError, match="player id"):
        default_board.place_token(0, 0, 0)


@pytest.mark.parametrize("x, y", [(-1, 0), (0, -1), (7, 0), (0, 4)])
def test_place_token_invalid_column(default_board, x, y):
    with pytest.raises(ValueError, match="invalid column"):
        default_board.place_token(1, x, y)


def test_place_token_column_full():
    board = ConnectNBoard3D(height=2, width=2, depth=1, n_to_connect=2, num_players=2)
    board.place_token(1, 0, 0)
    board.place_token(2, 0, 0)
    with pytest.raises(ValueError, match="column is full"):
        board.place_token(1, 0, 0)


def test_legal_moves_updates():
    board = ConnectNBoard3D(height=2, width=2, depth=1, n_to_connect=2, num_players=2)
    assert set(board.legal_moves()) == {(0, 0), (1, 0)}

    board.place_token(1, 0, 0)  # z=0
    board.place_token(2, 0, 0)  # z=1, columna (0,0) llena

    assert set(board.legal_moves()) == {(1, 0)}


def test_is_full():
    board = ConnectNBoard3D(height=1, width=2, depth=2, n_to_connect=2, num_players=2)
    assert board.is_full() is False
    board.place_token(1, 0, 0)
    board.place_token(2, 0, 1)
    board.place_token(1, 1, 0)
    board.place_token(2, 1, 1)
    assert board.is_full() is True


def test_reset_clears_board(default_board):
    default_board.place_token(1, 0, 0)
    default_board.reset()
    assert np.count_nonzero(default_board.grid) == 0
    assert default_board.last_move is None
    assert set(default_board.legal_moves())  # vuelve a haber movimientos


def test_clone_is_deep_copy(default_board):
    default_board.place_token(1, 0, 0)
    clone = default_board.clone()

    # Igual en contenido pero objetos distintos
    assert np.array_equal(default_board.grid, clone.grid)
    assert default_board is not clone

    # Al modificar el clon, el original no cambia
    clone.place_token(2, 1, 0)
    assert not np.array_equal(default_board.grid, clone.grid)


def test_winner_vertical_column(cube_board):
    """(x,y) fijo, z variable."""
    x, y = 1, 2
    for _ in range(3):
        cube_board.place_token(1, x, y)
    assert cube_board.check_winner() is None
    cube_board.place_token(1, x, y)  # cuarto token
    assert cube_board.check_winner() == 1


def test_winner_horizontal_row_x(cube_board):
    """z=0, y=0; x varía."""
    z, y = 0, 0
    for x in range(4):
        cube_board.place_token(1, x, y)
    assert cube_board.check_winner() == 1


def test_winner_depth_row_y(cube_board):
    """z=0, x=0; y varía."""
    z, x = 0, 0
    for y in range(4):
        cube_board.place_token(1, x, y)
    assert cube_board.check_winner() == 1


def test_no_false_positive(default_board):
    default_board.place_token(1, 0, 0)
    default_board.place_token(2, 1, 0)
    default_board.place_token(1, 2, 0)
    default_board.place_token(2, 3, 0)
    assert default_board.check_winner() is None


@pytest.mark.parametrize("dx, dy, dz", ConnectNBoard3D.DIRECTIONS)
def test_winner_in_all_13_directions(dx: int, dy: int, dz: int):
    """
    Construimos 4 fichas del jugador 1 directamente sobre el grid
    para cada uno de los 13 vectores dirección admitidos
    y comprobamos que `check_winner` las detecta.
    """
    size = 4
    board = ConnectNBoard3D(height=size, width=size, depth=size, n_to_connect=4, num_players=2)

    # Punto de inicio que garantiza quedarse dentro de los límites
    x0 = 0 if dx >= 0 else size - 1
    y0 = 0 if dy >= 0 else size - 1
    z0 = 0 if dz >= 0 else size - 1

    for step in range(board.n_to_connect):
        x = x0 + dx * step
        y = y0 + dy * step
        z = z0 + dz * step
        board.grid[z, x, y] = 1  # colocamos manualmente

    board.last_move = (1, x, y, z)  # último movimiento en la línea
    assert board.check_winner() == 1, (
            f"No detecta victoria para la dirección {(dx, dy, dz)}\n"
            + textwrap.dedent(
        f"""
            Línea generada (4 puntos):
                ({x0},{y0},{z0}) … ({x},{y},{z})
            """
    )
    )


def test_shorter_n_to_connect():
    board = ConnectNBoard3D(height=2, width=2, depth=1, n_to_connect=2)
    board.place_token(1, 0, 0)
    assert board.check_winner() is None
    board.place_token(1, 1, 0)
    assert board.check_winner() == 1


def test_hashable():
    board = ConnectNBoard3D()
    try:
        hash(board)
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"El tablero no debería lanzar al ser hashable: {exc}")


def test_str_contains_levels(cube_board):
    cube_board.place_token(1, 0, 0)
    out = str(cube_board)
    # Debe incluir al menos la etiqueta del nivel superior
    assert "Level z=3" in out
