"""Final label array: abstention policy, forced dead ball, ungraded frames, CSV writer."""
import os
import sys

import numpy as np

from possession import params
from possession.runs import _runs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

LETTERS = np.array(["H", "A", "D"])


def assemble_labels(codes: np.ndarray, observed: np.ndarray, graded: np.ndarray,
                    forced_dead: np.ndarray) -> np.ndarray:
    n = len(codes)
    labels = LETTERS[codes].astype("<U1")
    g = params.TRANSITION_GUARD
    for a, b in _runs(~np.asarray(observed, bool) & np.asarray(graded, bool)):
        if b - a < params.ABSTAIN_MIN_GAP or codes[a] > 1:
            continue
        window = codes[max(0, a - g):min(n, b + g)]
        if (window == codes[a]).all():
            labels[a:b] = "X"
    labels[np.asarray(forced_dead, bool)] = "D"
    labels[~np.asarray(graded, bool)] = "X"
    return labels


def write_labels(path: str, labels: np.ndarray) -> None:
    from starter.baseline import write_submission
    write_submission(path, labels)
