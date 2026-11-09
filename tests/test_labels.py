import numpy as np

from possession.labels import assemble_labels


def _run(codes, observed=None, graded=None, forced=None):
    n = len(codes)
    codes = np.asarray(codes, np.int8)
    observed = np.ones(n, bool) if observed is None else observed
    graded = np.ones(n, bool) if graded is None else graded
    forced = np.zeros(n, bool) if forced is None else forced
    return assemble_labels(codes, observed, graded, forced)


def test_ungraded_is_X_and_forced_dead_is_D():
    graded = np.ones(10, bool); graded[:3] = False
    forced = np.zeros(10, bool); forced[3:5] = True
    out = _run([0] * 10, graded=graded, forced=forced)
    assert out[0] == "X" and out[3] == "D" and out[6] == "H"


def test_unobserved_gap_inside_possession_is_X():
    obs = np.ones(300, bool); obs[100:150] = False
    out = _run([0] * 300, observed=obs)
    assert (out[100:150] == "X").all() and out[99] == "H" and out[150] == "H"


def test_gap_spanning_change_stays_labelled():
    codes = [0] * 120 + [1] * 180
    obs = np.ones(300, bool); obs[100:150] = False
    out = _run(codes, observed=obs)
    assert "X" not in set(out)


def test_only_valid_chars_and_length():
    n = 199319
    out = _run(np.zeros(n, np.int8))
    assert len(out) == n and set(out) <= {"H", "A", "D", "X"}
