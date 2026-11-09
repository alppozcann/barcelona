"""Label-free diagnostics. Deliberately takes no ground truth."""
from typing import List, Optional

import numpy as np

from possession import params


def _blocks_default():
    from games import GAME, half_video_blocks
    return half_video_blocks(GAME)


def _ffill(half: np.ndarray) -> np.ndarray:
    out = half.copy()
    known = out != "X"
    if not known.any():
        return out
    idx = np.where(known, np.arange(len(out)), -1)
    np.maximum.accumulate(idx, out=idx)
    first = np.argmax(known)
    idx[idx < 0] = first
    return out[idx]


def possession_stats(labels: np.ndarray, graded: np.ndarray, blocks: Optional[list] = None,
                     fps: int = params.FPS) -> dict:
    blocks = blocks or _blocks_default()
    durations, n_d, n_x, n_all = [], 0, 0, 0
    for _, first, last in blocks:
        half = np.asarray(labels[first:last + 1])
        n_all += len(half); n_d += int((half == "D").sum()); n_x += int((half == "X").sum())
        half = _ffill(half)
        start = 0
        for i in range(1, len(half) + 1):
            if i == len(half) or half[i] != half[start]:
                if half[start] in ("H", "A"):
                    durations.append((i - start) / fps)
                start = i
    d = np.array(durations) if durations else np.array([0.0])
    return {"n_possessions": len(durations), "mean_poss_s": float(d.mean()),
            "median_poss_s": float(np.median(d)), "share_D": n_d / max(n_all, 1),
            "share_X": n_x / max(n_all, 1)}


def check_plausible(stats: dict) -> List[str]:
    out = []
    lo, hi = params.PLAUSIBLE_N_POSS
    if not lo <= stats["n_possessions"] <= hi:
        out.append("n_possessions {} outside plausible {}..{}".format(stats["n_possessions"], lo, hi))
    lo, hi = params.PLAUSIBLE_MEAN_POSS_S
    if not lo <= stats["mean_poss_s"] <= hi:
        out.append("mean_poss_s {:.1f} outside plausible {}..{}".format(stats["mean_poss_s"], lo, hi))
    return out
