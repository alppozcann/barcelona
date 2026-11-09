# Technical document: frame-level ball possession (More than a Hack 2027, Phase 1)

Team: `<team_name>`. Draft: numbers below are label-free diagnostics only.

## Approach

A rule-based state machine over the provided tactical tracking, with no learned model. No labelled
data of any kind (this match or others) was used, so there is nothing to train; every constant is a
prior fixed before running on the match (`possession/params.py`).

1. **Clean the ball.** Tracker spikes (implied speed above 45 m/s) are removed and gaps up to
   12 frames (0.5 s) are linearly interpolated. Longer gaps stay missing. Same-team tracks closer
   than 0.5 m in a frame are treated as the tracker duplicating one player and the higher slot is
   dropped.
2. **Per-frame geometry.** Distance from the ball to the nearest player of each team, nearest team,
   ball speed, whether the ball is inside the pitch.
3. **Control events.** A team has *control* when its nearest player is within 2.5 m of the ball, the
   other team is at least 1 m further, and the ball moves slower than 12 m/s, sustained for at least
   25 frames (1 s), bridging flickers of up to 5 frames. Touches, deflections and passes in flight do
   not qualify (CHALLENGE.md "what does not count").
4. **Dead ball** (`D`) from tracking: ball outside the pitch for 1 s, ball motionless for 3 s, ball
   lost for 5 s or more, or ball motionless on the centre spot for 1 s (kick-off). Frames
   110808..113741 (second half, before kick-off) are `D` by the published rule.
5. **State machine.** Start in `D`. When play restarts the possessor is the first control event
   within 5 s, otherwise the Smart Tagging team prior, otherwise the next control event. Possession
   flips only when the *other* team shows a control event; a loose or contested ball keeps the last
   possessor. A possession shorter than 2 s is merged into its neighbour.
6. **Smart Tagging** is used as evidence only: attacking phases (Build Up, Progression, Creating
   Chances, Attacking Transition, Ball in Final Third / in the Box, Players in the Box, Long Ball) and
   set pieces give a team prior at restarts. Defending phases are ignored because their `Team` is
   the defender. `Team = N/A` (every throw-in) gives no prior.
7. **Abstention.** `X` is written where the ball is missing for 25+ frames inside a possession whose
   label does not change within 1 s either side (free on `spellF1`, the scorer forward-fills it);
   labels are always committed near transitions. Ungraded frames are `X`.

## Assets used and dropped

| asset | use |
|---|---|
| tactical ATD (tracking) | main input |
| Smart Tagging | restart and possession-team prior |
| tactical homography | not used |
| broadcast ATD + homography | tried (stage 2), **dropped**, see below |
| tactical / broadcast video | not used |

**Broadcast fusion, tried and dropped.** The broadcast timeline was aligned to the tactical one from
the two ball tracks alone (FFT-based masked matching per 750-frame block, mirrored-pitch hypothesis
included, error and peak-distinctiveness gates). Alignment worked (61 segments accepted, 15 distinct
lags, no mirroring) but ball coverage on graded frames rose only from 59.3% to 60.8%
(about 2.3k frames), stable across block lengths 250/375/750, and the possession count moved from
269 to 270. The keep-criterion (a clear coverage gain) was not met, so the submission is
tactical-only. The code stays behind `--fuse-broadcast`.

## Training data

None. No labelled data from any game, no hand labels, no event data, no corrected tracking.

## Findings worth knowing

Of 138 ball-missing gaps of 25+ frames on graded frames, 132 last 125 frames (5 s) or more (median
about 10 s). The 41% missing ball is therefore dominated by long stretches, mostly stoppages, not by
brief occlusions in live play. That is why dead-ball inference from a long ball loss works, and why
abstention rarely fires.

## Label-free diagnostics (tactical-only submission)

269 possessions, mean 12.9 s, median 8.8 s, dead ball 45% of graded frames. These sit inside the
football-plausibility range set beforehand (120-450 possessions, mean 6-25 s). The public 5-minute
window is used once, at the end, as a sanity check only and was not used to choose any parameter.

## Limitations

- Control is geometric. Aerial duels, rebounds and deflections that end in a different team's control
  are only handled through the 1 s minimum.
- Teams are inferred from tracks that can swap identity; only team membership is used, not identity.
- A ball lost for 5+ s is called dead, which is wrong when the ball is genuinely lost during live play.
- Smart Tagging priors are used only at restarts; they are not trusted as answers.
- With more time: learn control/transition thresholds from other matches, and use the video.

## Reproduction

```bash
pip install -r requirements-pipeline.txt          # numpy, pandas, pytest (no kloppy needed)
python run.py --team <team_name> --data-dir <folder with tactical/ and broadcast/>
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```

Deterministic (the output hash is identical across runs, with or without the parse cache). About
10 s on a laptop once tracking is parsed; first parse of the raw ATD adds under a minute.

**Google Colab** (data on Drive, never committed):

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone <your public repo url> repo && cd repo && pip install -q -r requirements-pipeline.txt
!cd repo && python run.py --team <team_name> --data-dir /content/drive/MyDrive/<data folder> \
      --out-dir /content/out --cache-dir /content/cache
```
Python 3.9+ and Colab's preinstalled numpy/pandas are enough. `--data-dir` (or `MTAH_DATA_DIR`) is the
folder containing `tactical/` and `broadcast/`.

## Parameters (all priors, fixed before running on the match)

See `possession/params.py`, each constant carries its rationale.
