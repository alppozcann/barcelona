"""H/A/D state machine. Possession flips only when the other team shows a control event."""
from typing import List

import numpy as np

from possession import params
from possession.control import ControlEvent
from possession.runs import _runs
from possession.tagging import TagPriors

H, A, D = 0, 1, 2


def _initial_possessor(seg_start, seg_events, priors: TagPriors) -> int:
    if seg_events and seg_events[0].start <= seg_start + params.RESTART_WINDOW_FRAMES:
        return seg_events[0].team
    prior = int(priors.team_prior[seg_start])
    if prior >= 0:
        return prior
    return seg_events[0].team if seg_events else D


def _merge_short(codes: np.ndarray) -> None:
    """One left-to-right pass: a possession shorter than MIN_POSS_FRAMES joins the possession before
    it (or after it, at the start of a live stretch); one flanked by dead ball on both sides is dead."""
    runs = []
    start = 0
    for i in range(1, len(codes) + 1):
        if i == len(codes) or codes[i] != codes[start]:
            runs.append([start, i, int(codes[start])])
            start = i
    done = []
    for k, (a, b, c) in enumerate(runs):
        if c != D and b - a < params.MIN_POSS_FRAMES:
            if done and done[-1][2] != D:
                c = done[-1][2]
            elif k + 1 < len(runs) and runs[k + 1][2] != D:
                c = runs[k + 1][2]
            else:
                c = D
            codes[a:b] = c
        if done and done[-1][2] == c:
            done[-1][1] = b
        else:
            done.append([a, b, c])


def run_state_machine(n_frames: int, events: List[ControlEvent], dead: np.ndarray,
                      forced_dead: np.ndarray, priors: TagPriors) -> np.ndarray:
    codes = np.full(n_frames, D, dtype=np.int8)
    live = ~(np.asarray(dead, bool) | np.asarray(forced_dead, bool))
    events = sorted(events, key=lambda e: e.start)
    for a, b in _runs(live):
        seg_events = [e for e in events if e.end >= a and e.start < b]
        current = _initial_possessor(a, seg_events, priors)
        codes[a:b] = current
        for e in seg_events:
            if e.team != current and e.start >= a:
                current = e.team
                codes[e.start:b] = current
    _merge_short(codes)
    return codes
