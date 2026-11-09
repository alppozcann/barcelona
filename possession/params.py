"""Every tunable constant of the pipeline, frozen as a PRIOR.

Tuning rule (see the design document): these values come from physics and football knowledge and
are fixed before running on the real match. They are never chosen by public-window score. The only
data-driven check allowed is the label-free plausibility range at the bottom.
"""

FPS = 25

# -- cleaning ----------------------------------------------------------------------------------
MAX_INTERP_GAP = 12          # frames (~0.5 s): short enough that a ball cannot change hands unseen
SPIKE_SPEED = 45.0           # m/s: faster than any real ball (hardest shots ~35 m/s) -> tracker glitch
DUP_SEP_M = 0.5              # m: two same-team tracks closer than this are the same player

# -- control events ----------------------------------------------------------------------------
CONTROL_RADIUS_M = 2.5       # m: a player this close can play the ball
CONTROL_MARGIN_M = 1.0       # m: the other team must be this much further to call it control
CONTROL_MAX_SPEED = 12.0     # m/s: faster than this the ball is in flight, not controlled
MIN_CONTROL_FRAMES = 25      # 1 s: touch or deflection is not control (CHALLENGE.md)
CONTROL_GAP_TOLERANCE = 5    # frames: a brief flicker does not end a control run

# -- dead ball ---------------------------------------------------------------------------------
STATIONARY_SPEED = 0.3       # m/s
STATIONARY_FRAMES = 75       # 3 s of a motionless ball on the pitch
OUT_MARGIN_M = 0.5           # m beyond the lines
OUT_FRAMES = 25              # 1 s outside the pitch
LONG_LOSS_FRAMES = 125       # 5 s with no ball at all
CENTRE_RADIUS_M = 1.5        # kick-off: ball on the centre spot
RESTART_WINDOW_FRAMES = 125  # 5 s after play restarts to find the first possessor

# -- state machine / labels --------------------------------------------------------------------
MIN_POSS_FRAMES = 50         # 2 s: shorter "possessions" are merged away
ABSTAIN_MIN_GAP = 25         # frames of missing ball before writing X inside a possession
TRANSITION_GUARD = 25        # never abstain this close to a label change

# -- published match facts ---------------------------------------------------------------------
SECOND_HALF_KICKOFF = 113742  # FAQ.md: frames 110808..113741 are graded and all D

# -- label-free plausibility (general football knowledge, NOT the public ground truth) ----------
PLAUSIBLE_N_POSS = (120, 450)
PLAUSIBLE_MEAN_POSS_S = (6.0, 25.0)

KICKOFF_FRAMES = 25          # 1 s motionless on the centre spot = kick-off, ball not yet in play

# -- broadcast alignment (stage 2) -------------------------------------------------------------
ALIGN_MAX_MSD = 4.0          # m^2: segment must match tactical ball to within ~2 m RMS
ALIGN_MIN_OVERLAP = 100      # frames where both sources see the ball, per segment
ALIGN_EXCLUDE = 100          # frames around the best lag ignored when looking for a rival peak
ALIGN_MIN_CONF = 3.0         # second-best / best match error: the peak must be distinct

# -- cleaning, added after review --------------------------------------------------------------
PITCH_MARGIN_M = 40.0        # beyond this the coordinate is tracker garbage; a ball in the stands is still "out of play"
SPIKE_LOOKAHEAD = 3          # a point is a glitch if one of the next 3 observations agrees with the last good one
