import numpy as np

from possession.tagging import priors_from_intervals, team_from_tag


def test_team_from_tag():
    assert team_from_tag("Barcelona") == 0
    assert team_from_tag("Atletic Club") == 1 and team_from_tag("Athletic Club") == 1
    assert team_from_tag("N/A") == -1 and team_from_tag(None) == -1 and team_from_tag("") == -1


def test_restart_mask_and_team_prior():
    iv = [
        {"code": "SET PIECES", "start_frame": 250, "end_frame": 300, "tags": {"Team": "Barcelona", "Type": "Corner Kick"}},
        {"code": "BUILD UP", "start_frame": 400, "end_frame": 500, "tags": {"Team": "Atletic Club"}},
        {"code": "DEFENDING IN MIDDLE THIRD", "start_frame": 400, "end_frame": 500, "tags": {"Team": "Barcelona"}},
        {"code": "SET PIECES", "start_frame": 600, "end_frame": 650, "tags": {"Team": "N/A", "Type": "Throw-in"}},
    ]
    p = priors_from_intervals(iv, 1000)
    assert p.restart_mask[250] and p.restart_mask[299] and not p.restart_mask[300]
    assert p.restart_mask[620]
    assert p.team_prior[260] == 0
    assert p.team_prior[450] == 1            # defending phase must not override
    assert p.team_prior[620] == -1           # N/A -> no prior, no crash
    assert p.team_prior[900] == -1
