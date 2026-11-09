import numpy as np
import pytest

from possession.atd import Tracking, parse_atd_line, tracking_from_lines


def test_parse_atd_line_reads_players_and_ball():
    fid, players, ball = parse_atd_line("5:0.5,0.25,1.0;NaN,NaN,NaN:0.1,0.2,0.0", 2)
    assert fid == 5
    assert players[0].tolist() == pytest.approx([0.5, 0.25])
    assert np.isnan(players[1]).all()
    assert ball.tolist() == pytest.approx([0.1, 0.2])


def test_parse_atd_line_missing_ball():
    _, _, ball = parse_atd_line("1:0.5,0.5,0.0:NaN,NaN,NaN", 1)
    assert np.isnan(ball).all()


def test_tracking_indexes_by_frame_id_minus_one_and_scales_to_metres():
    lines = ["1:0.5,0.5,0.0;NaN,NaN,NaN:0.5,0.5,0.0", "2:0.1,0.2,0.0;0.3,0.4,0.0:NaN,NaN,NaN"]
    t = tracking_from_lines(lines, np.array([0, 1], dtype=np.int8), pitch=(100.0, 50.0), n_frames=3)
    assert isinstance(t, Tracking)
    assert t.players.shape == (3, 2, 2) and t.ball.shape == (3, 2)
    assert t.players[0, 0].tolist() == pytest.approx([50.0, 25.0])
    assert t.players[1, 1].tolist() == pytest.approx([30.0, 20.0])
    assert np.isnan(t.ball[1]).all()
    assert np.isnan(t.players[2]).all()          # frame without a record stays NaN
