import numpy as np

from possession.control import ControlEvent, find_control_events
from possession.features import Features


def _feat(d_home, d_away, speed=None):
    n = len(d_home)
    d = np.stack([np.asarray(d_home, float), np.asarray(d_away, float)], 1).astype(np.float32)
    sp = np.full(n, 1.0, np.float32) if speed is None else np.asarray(speed, np.float32)
    ok = np.ones(n, bool)
    return Features(d, np.zeros(n, np.int8), sp, ok, ok, ok)


def test_sustained_control_gives_one_event():
    f = _feat([1.0] * 40 + [9.0] * 20, [8.0] * 40 + [9.0] * 20)
    assert find_control_events(f) == [ControlEvent(0, 39, 0)]


def test_short_touch_no_event():
    f = _feat([1.0] * 10 + [9.0] * 50, [8.0] * 60)
    assert find_control_events(f) == []


def test_margin_not_met_no_event():
    assert find_control_events(_feat([1.0] * 40, [1.5] * 40)) == []


def test_fast_ball_no_event():
    assert find_control_events(_feat([1.0] * 40, [8.0] * 40, speed=[20.0] * 40)) == []


def test_small_gap_merged():
    d_home = [1.0] * 20 + [9.0] * 3 + [1.0] * 20
    f = _feat(d_home, [8.0] * 43)
    assert find_control_events(f) == [ControlEvent(0, 42, 0)]
