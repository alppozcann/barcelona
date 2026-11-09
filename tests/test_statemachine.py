import numpy as np

from possession.control import ControlEvent
from possession.statemachine import run_state_machine
from possession.tagging import TagPriors


def _priors(n, team=None):
    tp = np.full(n, -1, np.int8)
    if team:
        for i, t in team.items():
            tp[i:] = t
    return TagPriors(np.zeros(n, bool), tp)


def _z(n):
    return np.zeros(n, bool)


def test_flip_only_on_opponent_event():
    ev = [ControlEvent(0, 40, 0), ControlEvent(200, 260, 1)]
    c = run_state_machine(400, ev, _z(400), _z(400), _priors(400))
    assert set(c[:200]) == {0} and set(c[200:]) == {1}


def test_loose_ball_keeps_last_possessor():
    ev = [ControlEvent(0, 40, 0), ControlEvent(300, 340, 0)]
    c = run_state_machine(400, ev, _z(400), _z(400), _priors(400))
    assert set(c) == {0}


def test_dead_interval_breaks_possession_and_restart_uses_tag_prior():
    ev = [ControlEvent(0, 40, 0), ControlEvent(400, 440, 1)]
    dead = _z(600); dead[100:150] = True
    c = run_state_machine(600, ev, dead, _z(600), _priors(600, {150: 0}))
    assert set(c[100:150]) == {2}
    assert set(c[150:400]) == {0} and set(c[400:]) == {1}


def test_short_possession_merged():
    ev = [ControlEvent(0, 60, 0), ControlEvent(100, 130, 1), ControlEvent(140, 200, 0)]
    c = run_state_machine(300, ev, _z(300), _z(300), _priors(300))
    assert set(c) == {0}


def test_forced_dead_is_all_D():
    forced = _z(300); forced[:100] = True
    c = run_state_machine(300, [ControlEvent(120, 200, 1)], _z(300), forced, _priors(300))
    assert set(c[:100]) == {2} and set(c[100:]) == {1}


def test_no_events_at_all_is_defined():
    c = run_state_machine(200, [], _z(200), _z(200), _priors(200))
    assert set(c) == {2}
    c = run_state_machine(200, [], _z(200), _z(200), _priors(200, {0: 1}))
    assert set(c) == {1}


def test_many_short_possessions_are_all_merged():
    ev = [ControlEvent(0, 40, 0)]
    for k in range(30):
        s = 200 + k * 300
        ev += [ControlEvent(s, s + 29, 1), ControlEvent(s + 30, s + 60, 0)]
    n = 200 + 30 * 300 + 400
    c = run_state_machine(n, ev, _z(n), _z(n), _priors(n))
    assert set(c) == {0}


def test_short_live_island_between_dead_stretches_is_dead():
    dead = _z(400); dead[:100] = True; dead[120:] = True
    c = run_state_machine(400, [], dead, _z(400), _priors(400, {100: 1}))
    assert set(c) == {2}
