"""Stage 2: align the broadcast timeline to the tactical one using the two ball tracks only, and
fill tactical ball gaps from the aligned broadcast ball.

For each block of the broadcast timeline we compute, for EVERY possible start on the tactical
timeline, the mean squared distance between the two ball tracks over frames where both see the
ball (masked, via FFT). The best start gives ``lag`` (tactical_frame = broadcast_frame + lag);
the mirrored-pitch hypothesis is tried too. Replays and cuts simply fail the error gate.
Deterministic and needs no video.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from possession import params

PITCH = (105.3, 68.0)


@dataclass(frozen=True)
class Segment:
    b_start: int       # first broadcast frame (inclusive)
    b_end: int         # last broadcast frame (exclusive)
    lag: int           # tactical_frame = broadcast_frame + lag
    flip: bool         # broadcast pitch is mirrored relative to tactical
    confidence: float  # second-best / best match error (>1 means a distinct peak)


def _mirror(ball: np.ndarray, pitch: Tuple[float, float]) -> np.ndarray:
    return np.stack([pitch[0] - ball[:, 0], pitch[1] - ball[:, 1]], 1)


def _msd(a, m1, ref_ffts, n_ref, nfft):
    """Mean squared distance for every start position, inf where overlap is too small."""
    v1, v2, vx, vy = ref_ffts
    a = np.where(m1[:, None], a, 0.0)
    u1 = np.fft.rfft(m1.astype(float), nfft)
    u2 = np.fft.rfft(m1 * (a ** 2).sum(1), nfft)
    ux = np.fft.rfft(m1 * a[:, 0], nfft)
    uy = np.fft.rfft(m1 * a[:, 1], nfft)
    count = np.fft.irfft(v1 * np.conj(u1), nfft)
    total = np.fft.irfft(v1 * np.conj(u2) + v2 * np.conj(u1) - 2 * (vx * np.conj(ux) + vy * np.conj(uy)), nfft)
    positions = n_ref - len(a) + 1
    count, total = count[:positions], total[:positions]
    out = np.full(positions, np.inf)
    ok = count >= params.ALIGN_MIN_OVERLAP - 0.5
    out[ok] = np.maximum(total[ok], 0.0) / count[ok]
    return out


def estimate_segments(tactical_ball: np.ndarray, broadcast_ball: np.ndarray, valid: np.ndarray,
                      seg_len: int = 750, pitch: Tuple[float, float] = PITCH) -> List[Segment]:
    t = np.asarray(tactical_ball, float)
    n_t = len(t)
    m2 = ~np.isnan(t[:, 0])
    tz = np.where(m2[:, None], t, 0.0)
    nfft = 1 << int(np.ceil(np.log2(n_t + seg_len)))
    ref = (np.fft.rfft(m2.astype(float), nfft), np.fft.rfft(m2 * (tz ** 2).sum(1), nfft),
           np.fft.rfft(m2 * tz[:, 0], nfft), np.fft.rfft(m2 * tz[:, 1], nfft))
    b_all = np.asarray(broadcast_ball, float)
    segments = []
    for start in range(0, len(b_all), seg_len):
        block = b_all[start:start + seg_len]
        m1 = np.asarray(valid[start:start + seg_len], bool) & ~np.isnan(block[:, 0])
        if len(block) < params.ALIGN_MIN_OVERLAP or m1.sum() < params.ALIGN_MIN_OVERLAP:
            continue
        best = None
        for flip in (False, True):
            a = _mirror(block, pitch) if flip else block
            err = _msd(np.nan_to_num(a), m1, ref, n_t, nfft)
            p = int(np.argmin(err))
            if np.isfinite(err[p]) and (best is None or err[p] < best[0]):
                rival = err.copy()
                rival[max(0, p - params.ALIGN_EXCLUDE):p + params.ALIGN_EXCLUDE + 1] = np.inf
                second = float(rival.min())
                conf = 1e3 if not np.isfinite(second) else min(1e3, second / (err[p] + 1e-6))
                best = (float(err[p]), p, flip, conf)
        if best is not None and best[0] <= params.ALIGN_MAX_MSD and best[3] >= params.ALIGN_MIN_CONF:
            segments.append(Segment(start, start + len(block), best[1] - start, best[2], best[3]))
    return segments


def fuse_ball(tactical_ball: np.ndarray, broadcast_ball: np.ndarray, segments: List[Segment],
              min_conf: float = params.ALIGN_MIN_CONF,
              pitch: Tuple[float, float] = PITCH) -> Tuple[np.ndarray, np.ndarray]:
    """Fill NaN tactical ball frames from confident aligned segments. Never overwrites."""
    fused = np.array(tactical_ball, float, copy=True)
    filled = np.zeros(len(fused), dtype=bool)
    for s in segments:
        if s.confidence < min_conf:
            continue
        idx_b = np.arange(s.b_start, min(s.b_end, len(broadcast_ball)))
        idx_t = idx_b + s.lag
        keep = (idx_t >= 0) & (idx_t < len(fused))
        idx_b, idx_t = idx_b[keep], idx_t[keep]
        vals = np.asarray(broadcast_ball, float)[idx_b]
        if s.flip:
            vals = _mirror(vals, pitch)
        take = np.isnan(fused[idx_t, 0]) & ~np.isnan(vals[:, 0]) & ~filled[idx_t]
        fused[idx_t[take]] = vals[take]
        filled[idx_t[take]] = True
    return fused, filled


def load_valid(source: str = "broadcast", data_dir: Optional[str] = None) -> np.ndarray:
    """Homography ``valid`` flag per frame of a source (bool array)."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import pandas as pd
    from games import require
    path = require(["homography"], data_dir=data_dir, source=source)[0]
    return pd.read_csv(path, usecols=["valid"])["valid"].to_numpy().astype(bool)
