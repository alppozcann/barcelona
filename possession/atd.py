"""Parse the raw ATD (FIFA/EPTS text export) into numpy arrays on the video-frame timeline.

Line format: ``<frame_id>:<player>;<player>;...:<ball>`` with each group ``x,y,speed`` in 0-1
coordinates and ``NaN`` for a missing observation. ``frame_id`` counts from 1, so row = frame_id - 1.
"""
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

FIELD_SEP = ":"
GROUP_SEP = ";"


@dataclass(frozen=True)
class Tracking:
    players: np.ndarray       # (n_frames, n_slots, 2) float32, metres, NaN missing
    ball: np.ndarray          # (n_frames, 2) float32, metres, NaN missing
    team_of_slot: np.ndarray  # (n_slots,) int8, 0 home / 1 away


def _xy(group: str) -> Tuple[float, float]:
    if not group or group[0] == "N":
        return np.nan, np.nan
    parts = group.split(",")
    return float(parts[0]), float(parts[1])


def parse_atd_line(line: str, n_slots: int):
    """(frame_id, players (n_slots, 2) normalised 0-1, ball (2,)). NaN where missing."""
    head, players_part, ball_part = line.rstrip("\n").split(FIELD_SEP)
    players = np.full((n_slots, 2), np.nan, dtype=np.float64)
    for slot, group in enumerate(players_part.split(GROUP_SEP)):
        if slot >= n_slots:
            break
        players[slot] = _xy(group)
    return int(head), players, np.array(_xy(ball_part), dtype=np.float64)


def tracking_from_lines(lines: Iterable[str], team_of_slot: np.ndarray,
                        pitch: Tuple[float, float], n_frames: int) -> Tracking:
    n_slots = len(team_of_slot)
    length, width = pitch
    scale = np.array([length, width], dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    for line in lines:
        if not line.strip():
            continue
        frame_id, p, b = parse_atd_line(line, n_slots)
        row = frame_id - 1
        if 0 <= row < n_frames:
            players[row] = p * scale
            ball[row] = b * scale
    return Tracking(players, ball, np.asarray(team_of_slot, dtype=np.int8))


def load_tracking(source: str = "tactical", data_dir: Optional[str] = None,
                  cache_dir: Optional[str] = ".cache", game: str = GAME) -> Tracking:
    """Load a source's ATD. Parsed arrays are cached as npz (derived data: never commit it)."""
    from starter.baseline import team_of_each_track
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    stamp = np.array([os.path.getsize(raw_path)])
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "{}_tracking.npz".format(source))
        if os.path.isfile(cache_path):
            z = np.load(cache_path)
            if "stamp" in z and z["stamp"].tolist() == stamp.tolist():
                return Tracking(z["players"], z["ball"], z["team_of_slot"])
    team = team_of_each_track(meta_path)
    with open(raw_path) as handle:
        tracking = tracking_from_lines(handle, team, GAMES[game]["pitch"],
                                       source_frames(source, game))
    if cache_path:
        np.savez(cache_path, players=tracking.players, ball=tracking.ball,
                 team_of_slot=tracking.team_of_slot, stamp=stamp)
    return tracking
