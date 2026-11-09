"""Which frames are graded, and which are forced dead ball."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402
from possession import params  # noqa: E402

N_FRAMES = 199319


def graded_mask(n_frames: int = N_FRAMES) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        mask[first:last + 1] = True
    return mask


def forced_dead_mask(n_frames: int = N_FRAMES) -> np.ndarray:
    """Second half is graded from its first frame but kicks off at SECOND_HALF_KICKOFF: all D."""
    mask = np.zeros(n_frames, dtype=bool)
    first = half_video_blocks(GAME)[1][1]
    mask[first:params.SECOND_HALF_KICKOFF] = True
    return mask
