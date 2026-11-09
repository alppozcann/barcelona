import numpy as np
import pytest

from possession.clean import clean_ball, suppress_duplicate_tracks


def test_interpolates_short_gap_only():
    ball = np.full((40, 2), np.nan)
    ball[0] = (0, 0); ball[5] = (5, 0); ball[30] = (6, 0)
    filled, obs = clean_ball(ball)
    assert filled[2, 0] == pytest.approx(2.0) and not obs[2]
    assert np.isnan(filled[15]).all()            # gap of 24 > MAX_INTERP_GAP stays NaN
    assert obs[0] and obs[5] and obs[30]


def test_spike_is_removed():
    ball = np.zeros((10, 2)); ball[5] = (50.0, 0.0)
    _, obs = clean_ball(ball)
    assert not obs[5]


def test_duplicate_same_team_track_suppressed():
    players = np.array([[[10, 10], [10.1, 10.1], [30, 30]]], dtype=float)
    out = suppress_duplicate_tracks(players, np.array([0, 0, 1]))
    assert np.isnan(out[0, 1]).all() and not np.isnan(out[0, 0]).any()
    assert not np.isnan(out[0, 2]).any()


def test_all_nan_ball_is_safe():
    filled, obs = clean_ball(np.full((5, 2), np.nan))
    assert np.isnan(filled).all() and not obs.any()


def test_bad_first_observation_does_not_wipe_real_track():
    ball = np.tile([10.0, 10.0], (300, 1)); ball[:, 0] += np.arange(300) * 0.05
    ball[0] = (100.0, 60.0)                 # on-pitch glitch accepted first
    _, obs = clean_ball(ball)
    assert obs[1:].all()


def test_out_of_range_coordinates_rejected_up_front():
    ball = np.tile([10.0, 10.0], (300, 1)); ball[:, 0] += np.arange(300) * 0.05
    ball[0] = (1000.0, 1000.0)
    _, obs = clean_ball(ball)
    assert not obs[0] and obs[1:].all()


def test_short_glitch_run_still_rejected():
    ball = np.tile([10.0, 10.0], (50, 1)); ball[20:22] = (90.0, 50.0)
    _, obs = clean_ball(ball)
    assert not obs[20] and not obs[21] and obs[19] and obs[22]
