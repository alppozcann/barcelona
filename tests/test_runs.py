import numpy as np

from possession.runs import fill_short_gaps, runs_at_least


def test_runs_at_least():
    m = np.array([1, 1, 0, 1, 1, 1, 0], bool)
    assert runs_at_least(m, 3).tolist() == [False, False, False, True, True, True, False]


def test_fill_short_gaps_only_between_trues():
    m = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0], bool)
    out = fill_short_gaps(m, 2)
    assert out.tolist() == [True, True, True, True, False, False, False, False, True, False]
