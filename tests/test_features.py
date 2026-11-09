import numpy as np
import pytest

from possession.features import compute_features

TEAMS = np.array([0, 1], dtype=np.int8)


def _ball(n, xy=(50.0, 30.0)):
    b = np.tile(np.array(xy, dtype=float), (n, 1))
    return b, np.ones(n, bool)


def test_nearest_team_picks_closer_side():
    players = np.array([[[50.0, 31.0], [60.0, 30.0]]] * 3, dtype=float)
    ball, obs = _ball(3)
    f = compute_features(players, ball, obs, TEAMS)
    assert (f.nearest_team == 0).all()
    assert f.d_team[0].tolist() == pytest.approx([1.0, 10.0])


def test_zero_players_is_safe():
    players = np.full((2, 2, 2), np.nan)
    ball, obs = _ball(2)
    with np.errstate(all="raise"):
        f = compute_features(players, ball, obs, TEAMS)
    assert (f.nearest_team == -1).all() and np.isnan(f.d_team).all()


def test_ball_outside_pitch_flag():
    players = np.full((2, 2, 2), np.nan)
    ball = np.array([[-1.0, 30.0], [50.0, 30.0]])
    f = compute_features(players, ball, np.ones(2, bool), TEAMS)
    assert not f.ball_in_pitch[0] and f.ball_in_pitch[1]


def test_speed_in_metres_per_second():
    players = np.full((3, 2, 2), np.nan)
    ball = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    f = compute_features(players, ball, np.ones(3, bool), TEAMS)
    assert f.ball_speed[1] == pytest.approx(25.0)
