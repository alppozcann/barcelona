import numpy as np

from possession.clean import clean_ball
from possession.dead import dead_ball_mask
from possession.features import compute_features

TEAMS = np.array([0, 1], dtype=np.int8)


def _dead(ball):
    filled, obs = clean_ball(np.asarray(ball, float))
    players = np.full((len(ball), 2, 2), np.nan)
    f = compute_features(players, filled, obs, TEAMS)
    return dead_ball_mask(f, filled)


def test_outside_pitch_long_is_dead_brief_exit_is_not():
    i = np.arange(120)
    left = np.stack([20 - 0.5 * i, np.full(120, 30.0)], 1)       # 12.5 m/s, leaves at i~41, stays out
    assert _dead(left)[100]
    blip = np.stack([-2 + 0.5 * np.abs(i - 44), np.full(120, 30.0)], 1)   # out for ~5 frames, returns
    assert not _dead(blip).any()


def test_stationary_midpitch_is_dead():
    ball = np.tile([30.0, 20.0], (100, 1))
    assert _dead(ball)[60]


def test_moving_ball_never_dead():
    ball = np.zeros((200, 2)); ball[:, 0] = 20 + np.arange(200) * 0.48   # 12 m/s
    ball[:, 1] = 30
    assert not _dead(ball).any()


def test_long_loss_dead_short_loss_not():
    ball = np.zeros((300, 2)); ball[:, 0] = 20 + np.arange(300) * 0.2; ball[:, 1] = 30
    ball[50:250] = np.nan
    assert _dead(ball)[150]
    ball = np.zeros((300, 2)); ball[:, 0] = 20 + np.arange(300) * 0.2; ball[:, 1] = 30
    ball[50:100] = np.nan
    assert not _dead(ball)[70]


def test_kickoff_on_centre_spot():
    ball = np.tile([52.65, 34.0], (40, 1))
    assert _dead(ball)[30]
