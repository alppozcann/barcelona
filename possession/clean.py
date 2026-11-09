"""Ball spike removal and gap interpolation; duplicate-track suppression."""
from typing import Tuple

import numpy as np

from possession import params


def clean_ball(ball: np.ndarray, fps: int = params.FPS,
               pitch: Tuple[float, float] = (105.3, 68.0)) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ball_filled, observed). Spikes become NaN; gaps <= MAX_INTERP_GAP are interpolated.

    ``observed`` is True only for original, non-spike observations. A point is a spike when it is
    implausibly far from the last accepted point AND one of the next few observations agrees with
    that last accepted point (so the point, not the anchor, is the outlier). If none does, the
    anchor itself was the glitch and the point becomes the new anchor.
    """
    out = np.array(ball, dtype=np.float64, copy=True)
    m = params.PITCH_MARGIN_M
    with np.errstate(invalid="ignore"):
        garbage = (out[:, 0] < -m) | (out[:, 0] > pitch[0] + m) | (out[:, 1] < -m) | (out[:, 1] > pitch[1] + m)
    out[garbage] = np.nan
    idx = np.flatnonzero(~np.isnan(out[:, 0]))

    def too_fast(i, j):
        return np.hypot(*(out[i] - out[j])) * fps / abs(i - j) > params.SPIKE_SPEED

    last = -1
    for k, i in enumerate(idx):
        if last >= 0 and too_fast(i, last):
            ahead = idx[k + 1:k + 1 + params.SPIKE_LOOKAHEAD]
            if any(not too_fast(j, last) for j in ahead) or len(ahead) == 0:
                out[i] = np.nan
                continue
        last = i
    observed = ~np.isnan(out[:, 0])
    filled = out.copy()
    idx = np.flatnonzero(observed)
    if len(idx) >= 2:
        gaps = np.diff(idx)
        for a, g in zip(idx[:-1][gaps > 1], gaps[gaps > 1]):
            if g - 1 <= params.MAX_INTERP_GAP:
                t = (np.arange(1, g) / g)[:, None]
                filled[a + 1:a + g] = out[a] + t * (out[a + g] - out[a])
    return filled, observed


def suppress_duplicate_tracks(players: np.ndarray, team_of_slot: np.ndarray,
                              chunk: int = 20000) -> np.ndarray:
    """NaN the higher-index slot of any same-team pair closer than DUP_SEP_M in a frame."""
    out = np.array(players, copy=True)
    for team in (0, 1):
        slots = np.flatnonzero(team_of_slot == team)
        if len(slots) < 2:
            continue
        for s in range(0, len(out), chunk):
            block = out[s:s + chunk][:, slots]                     # (c, k, 2)
            diff = block[:, :, None, :] - block[:, None, :, :]
            with np.errstate(invalid="ignore"):
                dist = np.sqrt((diff ** 2).sum(-1))                # (c, k, k)
                close = np.triu(np.ones((len(slots),) * 2, dtype=bool), 1) & (dist < params.DUP_SEP_M)
            drop = close.any(axis=1)                               # j is dropped if some i<j is close
            for j, slot in enumerate(slots):
                out[s:s + chunk][drop[:, j], slot] = np.nan
    return out
