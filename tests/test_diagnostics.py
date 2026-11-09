import numpy as np

from possession.diagnostics import check_plausible, possession_stats


def _labels(s):
    return np.array(list(s))


def test_dead_ball_splits_same_team_possession():
    s = possession_stats(_labels("HHHAADDHH"), np.ones(9, bool), blocks=[(1, 0, 8)])
    assert s["n_possessions"] == 3


def test_X_is_forward_filled_and_does_not_split():
    s = possession_stats(_labels("HHXHH"), np.ones(5, bool), blocks=[(1, 0, 4)])
    assert s["n_possessions"] == 1


def test_overfragmented_flagged():
    s = possession_stats(_labels("HA" * 500), np.ones(1000, bool), blocks=[(1, 0, 999)])
    assert any("n_possessions" in m for m in check_plausible(s))
