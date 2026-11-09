import numpy as np

from possession.align_broadcast import Segment, estimate_segments, fuse_ball

PITCH = (105.3, 68.0)


def _walk(n, seed=0):
    rng = np.random.RandomState(seed)
    v = np.cumsum(rng.normal(0, 0.15, size=(n, 2)), axis=0) * 0.2
    pos = np.cumsum(v, axis=0)
    pos = (pos - pos.min(0)) / (np.ptp(pos, axis=0) + 1e-9) * np.array([90.0, 55.0]) + np.array([7.0, 6.0])
    return pos


def _make(lag, flip=False, n_t=6000, n_b=2500, holes=0.3, seed=1):
    t = _walk(n_t)
    b = t[lag:lag + n_b].copy()
    if flip:
        b = np.stack([PITCH[0] - b[:, 0], PITCH[1] - b[:, 1]], 1)
    rng = np.random.RandomState(seed)
    t_obs = t.copy(); t_obs[rng.rand(n_t) < holes] = np.nan
    b[rng.rand(n_b) < holes] = np.nan
    return t_obs, b, t


def test_recovers_known_lag():
    t, b, _ = _make(lag=837)
    segs = estimate_segments(t, b, np.ones(len(b), bool), seg_len=500)
    assert len(segs) >= 3
    assert all(s.lag == 837 and not s.flip for s in segs)


def test_detects_mirrored_pitch():
    t, b, _ = _make(lag=400, flip=True)
    segs = estimate_segments(t, b, np.ones(len(b), bool), seg_len=500)
    assert len(segs) >= 3 and all(s.lag == 400 and s.flip for s in segs)


def test_unrelated_series_gives_no_segments():
    t, _, _ = _make(lag=0)
    b = _walk(2500, seed=99)
    assert estimate_segments(t, b, np.ones(2500, bool), seg_len=500) == []


def test_invalid_homography_frames_are_ignored():
    t, b, _ = _make(lag=300)
    valid = np.ones(len(b), bool); valid[:1000] = False
    segs = estimate_segments(t, b, valid, seg_len=500)
    assert segs and min(s.b_start for s in segs) >= 1000


def test_fuse_only_fills_missing_and_unflips():
    t, b, truth = _make(lag=300, flip=True)
    segs = estimate_segments(t, b, np.ones(len(b), bool), seg_len=500)
    fused, filled = fuse_ball(t, b, segs, min_conf=1.0, pitch=PITCH)
    keep = ~np.isnan(t[:, 0])
    assert np.array_equal(fused[keep], t[keep])            # never overwrites tactical
    assert filled.sum() > 0
    err = np.abs(fused[filled] - truth[filled]).max()
    assert err < 1e-6                                       # filled values are un-flipped truth


def test_low_confidence_segments_ignored():
    t, b, _ = _make(lag=300)
    seg = [Segment(0, 500, 300, False, 1.5)]
    fused, filled = fuse_ball(t, b, seg, min_conf=5.0, pitch=PITCH)
    assert not filled.any()
