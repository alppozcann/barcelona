import numpy as np

from possession.atd import Tracking
from possession.tagging import TagPriors
from run import build_labels


def test_build_labels_on_synthetic_match_is_valid():
    n = 3000
    rng = np.random.RandomState(0)
    players = rng.uniform(5, 60, size=(n, 4, 2)).astype(np.float32)
    ball = np.stack([30 + 10 * np.sin(np.arange(n) / 200.0), np.full(n, 30.0)], 1).astype(np.float32)
    ball[1000:1100] = np.nan
    t = Tracking(players, ball, np.array([0, 0, 1, 1], np.int8))
    priors = TagPriors(np.zeros(n, bool), np.full(n, -1, np.int8))
    graded = np.ones(n, bool); graded[:100] = False
    forced = np.zeros(n, bool)
    labels = build_labels(t, priors, graded, forced)
    assert len(labels) == n and set(labels) <= {"H", "A", "D", "X"}
    assert (labels[:100] == "X").all()
    again = build_labels(t, priors, graded, forced)
    assert (labels == again).all()               # deterministic
