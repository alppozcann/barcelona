"""Dead-ball detection from tracking alone."""
from typing import Tuple

import numpy as np

from possession import params
from possession.features import Features
from possession.runs import runs_at_least


def dead_ball_mask(features: Features, ball_filled: np.ndarray,
                   pitch: Tuple[float, float] = (105.3, 68.0),
                   centre: Tuple[float, float] = (52.65, 34.0)) -> np.ndarray:
    length, width = pitch
    present = features.ball_present
    x, y = ball_filled[:, 0], ball_filled[:, 1]
    m = params.OUT_MARGIN_M
    with np.errstate(invalid="ignore"):
        outside = present & ((x < -m) | (x > length + m) | (y < -m) | (y > width + m))
        slow = present & (features.ball_speed < params.STATIONARY_SPEED)
        at_centre = present & (np.hypot(x - centre[0], y - centre[1]) <= params.CENTRE_RADIUS_M)
    dead = runs_at_least(outside, params.OUT_FRAMES)
    dead |= runs_at_least(slow, params.STATIONARY_FRAMES)
    dead |= runs_at_least(~present, params.LONG_LOSS_FRAMES)
    dead |= runs_at_least(slow & at_centre, params.KICKOFF_FRAMES)
    return dead
