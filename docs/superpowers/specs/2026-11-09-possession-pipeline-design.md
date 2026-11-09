# Possession pipeline: design

Date: 2026-11-09. Deadline for submission: 2026-11-17.

## Intent

Produce the `frame,label` CSV (199,319 rows, labels `H`/`A`/`D`/`X`) for the BAR-ATH match, scored on
`spellF1` (possessions recovered as whole intervals), tie-broken by `macroF1`. The published floor is
`baseline_smooth50` at spellF1 0.2433. The goal is to beat it by a clear margin (well above the ~0.04
noise band) with a method that is reproducible from a public repo and explainable in a short technical
document.

The reason for the design below: the baseline fails on `spellF1` because it makes a per-frame decision
and fragments possessions, and because the ball is absent on 41% of graded frames. Both are fixed by
treating possession as a state that persists until the other team demonstrably takes control, and by
recovering more ball positions.

## Constraints (from RULES.md, CHALLENGE.md, DATA.md)

- No hand labels on this match for any purpose. The public 5 minutes (frames 11185..18684) are a
  pipeline sanity check only: not for training, not for choosing between variants.
- No labelled data from other matches is available to the team, so there is **no supervised model**.
  The method is rule-based / statistical, with parameters set from football priors and from
  label-free properties of this match's tracking (e.g. possession counts and durations).
- No human-produced data about this match from any source (event data, corrected tracking).
- Do not upload videos, datasets or extracts to a hosted LLM. Work from code and aggregate statistics;
  raw data content is not pasted into chat.
- The repo must regenerate the exact submitted CSV, be public, and exclude `data/`.
- Row i of the submission is tactical video frame i. ATD `frame_id = video_frame + 1`.
- Graded frames: 11185..90651 and 110808..190197. Second half is graded from 110808 but kicks off at
  113742, so frames 110808..113741 are all `D`.
- Environment: Python 3.13 locally; the scorer needs numpy and pandas only. Reuse `games.py` for paths.

## Goals

- Stage 1 (tactical only): a state-machine possession labeller on the tactical ATD, Smart Tagging and
  homography, beating the 0.2433 floor.
- Stage 2 (time-boxed to about one day): fuse the broadcast ATD to fill tactical ball gaps, then rerun
  stage 1. Keep only if it measurably helps by label-free criteria and does not harm the public-window
  sanity check.
- Validated submission, public repo, technical document, all before the deadline.

## Non-goals

- No tracking, detection or calibration rebuild. No video-based ball detector.
- No dead-ball ownership (`D` is just `D`).
- No tuning against the public 5 minutes.

## Anti-goals (failure even if the number looks fine)

- A file that only scores because parameters were fitted to the public window or leaked labels.
- A file the repository cannot regenerate exactly.
- A pile of tiny possessions that flatters `macroF1` while destroying `spellF1`.

## Method

### Stage 1: tactical only

1. **Load** tactical ATD (stream the raw file, as `starter/baseline.py` does), team per track slot from
   the metadata, ball and player positions in metres.
2. **Clean the tracking.** Interpolate short ball gaps; mark long gaps as missing. Detect and suppress
   duplicate tracks (the ATD can have more than 22 objects). Tolerate identity switches by working with
   team membership and positions only, never with track identity.
3. **Ball-state features** per frame: distance from the ball to the nearest player of each team,
   nearest-player team, ball speed, ball height proxy where inferable, ball inside the pitch, a
   "ball settled near one team" indicator, and a "ball dead" indicator (stationary / out of the
   playing area / tracking lost the ball for long).
4. **Control events.** A control event for a team is a sustained stretch (a few seconds) in which that
   team's player is clearly the nearest and the ball is close and not in free flight. A touch or
   deflection that does not produce sustained control does not count (CHALLENGE.md "what does not
   count"). Exact thresholds are builder decisions, set from physics (ball speeds, control distances)
   and from label-free statistics, never from the public ground truth.
5. **State machine** over {H, A, D}. Possession flips only on a control event by the other team. A loose
   or contested ball keeps the last possessor. `D` is entered on a dead-ball signal (ball out of play,
   stoppage) and left on a restart. Possessions get a minimum duration so they cannot fragment.
6. **Smart Tagging priors.** Use team tags to disambiguate who is in possession near transitions, and
   set-piece intervals (`SET PIECES`, `Type`) as restart/dead-ball evidence. `Team = N/A` (36 instances,
   all throw-ins) must be handled explicitly. Tags are evidence, never answers; phases are not
   possessions.
7. **Abstention policy.** Write `X` where the ball is lost inside a possession (the scorer forward-fills
   it for free on `spellF1`, though `macroF1` counts it as a miss); commit to a label at transitions.
   Ungraded frames are `X`; frames 110808..113741 are `D`.

### Stage 2: broadcast fusion

1. Align the broadcast ATD (164,488 frames, own timeline) to the tactical timeline using only the two
   ATDs and the homography `valid` flags: cross-correlate player and ball tracks in pitch coordinates
   over valid stretches, solve the offset segment by segment between cuts/replays, and check for
   flipped pitch axes between halves.
2. Exclude replays, close-ups and any segment where the alignment confidence is low.
3. Where the tactical ball is missing and an aligned broadcast ball exists, fill it; where both exist
   and disagree, prefer the tactical one unless it is clearly bad. Rerun stage 1.
4. Report the ball coverage before and after as an aggregate number. Drop the whole stage if alignment
   is unreliable.

## Validation and tuning discipline

- `validate_submission.py` on the exact file to be uploaded, always.
- Public 5 minutes: check alignment and label meaning only. Never pick between two variants by it.
- Tune and compare with **label-free diagnostics**: possession count, mean/median possession
  duration, share of `D`, fraction of frames abstained, agreement with Smart Tagging team tags. Sanity
  anchors: a professional match has on the order of a few hundred possessions of roughly 10-20 s mean
  (the public window shows mean 14.2 s over 17 possessions; treat this only as a sanity anchor, not a
  fit target).
- Use `--bootstrap` before believing any gap. Differences under ~0.04 spellF1 are noise.

## Deliverables

1. `<team_name>.csv`, 199,319 rows, validated.
2. Public repo: source, pinned dependencies, one command that regenerates the CSV from `data/`,
   `.gitignore` excluding `data/`.
3. Technical document (PDF or Markdown, a few pages): approach, assets used and dropped, training data
   (none), possession rules implemented, limitations, reproduction.

## Left to the builder

Module layout, exact thresholds and smoothing parameters, interpolation method, duplicate-track
heuristic, the alignment algorithm details in stage 2, and the test strategy.

## Resolved after builder check

- **Scorer semantics (answered by the repo).** A possession is a maximal run of `H` or `A`; a `D` run
  ends it, so the same team regaining the ball after `D` is a new possession. `D` frames are not scored
  as possessions. `X` on a graded frame is forward-filled with the last non-`X` label in the same half
  before `spellF1`. The builder should read `evaluation/possession_metrics.py` (public) as the spec.
- **File layouts and teams (answered by the repo).** See `DATA.md` and `starter/baseline.py`: ATD line
  format `<frame_id>:<slot>;...:<ball>` with `x,y,speed` in 0-1 coordinates (105.3 x 68 m), home is the
  first team in the ATD metadata. Attack direction per half is not given and must be derived from the
  tracking by the builder.
- **Dead-ball evidence (my call).** `D` is inferred from tracking (ball stationary, out of the playing
  area, long loss of the ball, kick-off geometry after a goal or at the start of a half) plus Smart
  Tagging set-piece intervals. Frames 110808..113741 are `D` by rule.
- **Stage 2 (my call).** Optional stretch goal, cut first if time slips. Keep-criteria are label-free
  only: aligned coverage gain of the ball, alignment confidence, and unchanged-or-more-plausible
  possession counts and durations. The public window is not used to decide whether to keep it.
- **Determinism and runtime (my call).** Every step must be deterministic (fixed seeds, no
  nondeterministic search order). No runtime limit is stated; target under about 30 minutes end to end.
- **Python (my call).** Pipeline code targets Python 3.9+; only the scorer needs 3.8 compatibility.
  Pin numpy and pandas versions that run on the interpreter used to produce the submission.
- **Tuning rule (proposed by me, pending the user's approval).** Parameters come from physical and
  football priors fixed before looking at results (control distance, minimum control time, ball-speed
  limits). The only data-driven calibration allowed uses label-free structure of this match's tracking:
  possession count and duration must land in a football-plausible range set from general knowledge, not
  from the public ground truth. Smart Tagging is an input feature, never an optimisation target, and
  tag agreement is not maximised. The public window is run once at the end as a sanity check and never
  to choose between variants.

## Open items / my calls to be checked

- Team name for file naming is not decided (you must supply it).
- I assume the broadcast **video** is not needed: stage 2 uses broadcast ATD and homography only.
- I assume Python 3.13 locally is acceptable; pin dependencies to versions that also run on 3.8+.
