"""Per-frame geometry features: distances, nearest team, ball speed, in-pitch flag."""
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from possession import params


@dataclass(frozen=True)
class Features:
    d_team: np.ndarray         # (n, 2) nearest-player distance per team (m), NaN if undefined
    nearest_team: np.ndarray   # (n,) int8, -1 if undefined
    ball_speed: np.ndarray     # (n,) m/s, NaN where undefined
    ball_observed: np.ndarray  # (n,) bool, original observations only
    ball_present: np.ndarray   # (n,) bool, observed or interpolated
    ball_in_pitch: np.ndarray  # (n,) bool


def compute_features(players: np.ndarray, ball_filled: np.ndarray, observed: np.ndarray,
                     team_of_slot: np.ndarray, pitch: Tuple[float, float] = (105.3, 68.0),
                     fps: int = params.FPS) -> Features:
    n = len(ball_filled)
    present = ~np.isnan(ball_filled[:, 0])
    d_team = np.full((n, 2), np.nan, dtype=np.float32)
    for team in (0, 1):
        slots = np.flatnonzero(team_of_slot == team)
        if len(slots) == 0:
            continue
        diff = players[:, slots, :] - ball_filled[:, None, :]
        with np.errstate(invalid="ignore"):
            dist = np.sqrt((diff ** 2).sum(-1))
        dist = np.where(np.isnan(dist), np.inf, dist)
        best = dist.min(axis=1)
        best[np.isinf(best)] = np.nan
        d_team[:, team] = best
    d_team[~present] = np.nan
    home, away = d_team[:, 0], d_team[:, 1]
    nearest = np.full(n, -1, dtype=np.int8)
    h_ok, a_ok = ~np.isnan(home), ~np.isnan(away)
    nearest[h_ok & ~a_ok] = 0
    nearest[a_ok & ~h_ok] = 1
    both = h_ok & a_ok
    nearest[both] = np.where(home[both] <= away[both], 0, 1)
    speed = np.full(n, np.nan, dtype=np.float32)
    if n > 1:
        with np.errstate(invalid="ignore"):
            speed[1:] = np.hypot(*np.diff(ball_filled, axis=0).T) * fps
    length, width = pitch
    with np.errstate(invalid="ignore"):
        in_pitch = present & (ball_filled[:, 0] >= 0) & (ball_filled[:, 0] <= length) \
            & (ball_filled[:, 1] >= 0) & (ball_filled[:, 1] <= width)
    return Features(d_team, nearest, speed, np.asarray(observed, bool), present, in_pitch)
