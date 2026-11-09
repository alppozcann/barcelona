"""Control events: a team sustainedly closest to a slow ball. Touches and deflections do not count."""
from dataclasses import dataclass
from typing import List

import numpy as np

from possession import params
from possession.features import Features
from possession.runs import _runs, fill_short_gaps, runs_at_least


@dataclass(frozen=True)
class ControlEvent:
    start: int  # inclusive
    end: int    # inclusive
    team: int   # 0 home, 1 away


def find_control_events(features: Features) -> List[ControlEvent]:
    events = []
    not_flying = ~(features.ball_speed > params.CONTROL_MAX_SPEED)   # NaN speed is allowed
    for team in (0, 1):
        mine, other = features.d_team[:, team], features.d_team[:, 1 - team]
        with np.errstate(invalid="ignore"):
            near = features.ball_present & (mine <= params.CONTROL_RADIUS_M)
            clear = np.isnan(other) | (other - mine >= params.CONTROL_MARGIN_M)
        support = near & clear & not_flying
        support = runs_at_least(fill_short_gaps(support, params.CONTROL_GAP_TOLERANCE),
                                params.MIN_CONTROL_FRAMES)
        events += [ControlEvent(int(a), int(b) - 1, team) for a, b in _runs(support)]
    return sorted(events, key=lambda e: (e.start, e.team))
