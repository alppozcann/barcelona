"""Boolean run-length helpers."""
import numpy as np


def _runs(mask: np.ndarray):
    """[(start, end_exclusive)] of maximal True runs."""
    m = np.concatenate(([False], np.asarray(mask, bool), [False]))
    edges = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(edges[::2], edges[1::2]))


def runs_at_least(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = np.zeros(len(mask), dtype=bool)
    for a, b in _runs(mask):
        if b - a >= min_len:
            out[a:b] = True
    return out


def fill_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Set False gaps of length <= max_gap to True when they sit between two True frames."""
    out = np.array(mask, dtype=bool, copy=True)
    runs = _runs(~out)
    for a, b in runs:
        if a > 0 and b < len(out) and b - a <= max_gap:
            out[a:b] = True
    return out
