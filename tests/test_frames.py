from possession.frames import forced_dead_mask, graded_mask


def test_graded_mask_counts_and_edges():
    m = graded_mask()
    assert m.sum() == 158857
    assert m[11185] and not m[11184] and m[90651] and not m[90652]
    assert m[110808] and m[190197] and not m[190198]


def test_forced_dead_mask_is_second_half_pre_kickoff():
    m = forced_dead_mask()
    assert m.sum() == 2934 and m[110808] and m[113741] and not m[113742]
