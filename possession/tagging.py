"""Smart Tagging -> per-frame restart mask and possession-team prior. Evidence, never answers."""
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Codes whose Team tag names the team that has/attacks with the ball. DEFENDING* and DEFENSIVE
# TRANSITION are excluded: their Team is the defender, not the possessor.
ATTACKING_CODES = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
                   "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX", "LONG BALL"}
RESTART_CODE = "SET PIECES"


@dataclass(frozen=True)
class TagPriors:
    restart_mask: np.ndarray  # (n,) bool
    team_prior: np.ndarray    # (n,) int8: 0 home, 1 away, -1 none


def team_from_tag(value: Optional[str]) -> int:
    """Home is Barcelona, away is Athletic (spelled 'Atletic Club' in the tagging file)."""
    v = (value or "").strip().lower()
    if v == "barcelona":
        return 0
    if v in ("atletic club", "athletic club"):
        return 1
    return -1


def priors_from_intervals(intervals: Iterable[dict], n_frames: int) -> TagPriors:
    restart = np.zeros(n_frames, bool)
    prior = np.full(n_frames, -1, np.int8)
    events = list(intervals)
    for e in sorted((e for e in events if (e["code"] or "").strip().upper() in ATTACKING_CODES),
                    key=lambda e: e["start_frame"]):
        t = team_from_tag(e["tags"].get("Team"))
        if t >= 0:
            prior[max(0, e["start_frame"]):min(n_frames, e["end_frame"])] = t
    for e in events:
        if (e["code"] or "").strip().upper() == RESTART_CODE:
            a, b = max(0, e["start_frame"]), min(n_frames, e["end_frame"])
            restart[a:b] = True
            t = team_from_tag(e["tags"].get("Team"))
            if t >= 0:
                prior[a:b] = t
    return TagPriors(restart, prior)


def load_tag_priors(n_frames: int = 199319, data_dir: Optional[str] = None) -> TagPriors:
    from starter.load_smart_tagging import load
    return priors_from_intervals(load(data_dir=data_dir), n_frames)
