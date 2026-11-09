# Possession Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a validated `frame,label` CSV for BAR-ATH that beats the 0.2433 spellF1 floor, plus a public-repo-ready pipeline and technical document.

**Architecture:** A `possession/` package turns the tactical ATD into per-frame features, finds sustained "control events", and runs a small H/A/D state machine (possession flips only on a control event by the other team; `D` from tracking and Smart Tagging evidence). Stage 2 (optional) aligns the broadcast ATD to the tactical timeline and fills tactical ball gaps, then reruns the same pipeline.

**Tech Stack:** Python 3.9+, numpy, pandas, pytest. No kloppy needed (raw ATD is parsed directly, as in `starter/baseline.py`). Reuse `games.py` for paths and half blocks.

**Spec:** `docs/superpowers/specs/2026-11-09-possession-pipeline-design.md`

## Global Constraints

- No hand labels on this match for any purpose; no labelled data from other matches exists, so **no supervised model**.
- The public 5 minutes (frames 11185..18684) are a sanity check only: run once at the end, never to choose between variants.
- **Tuning rule:** all parameters are set as priors in `possession/params.py` *before* running on real data; the only data-driven check is label-free plausibility (possession count/duration ranges). Smart Tagging is an input feature, never an optimisation target. Changing a parameter after seeing public-window scores is forbidden.
- Do not upload videos, datasets or extracts to a hosted LLM; work from code and aggregate statistics, never paste raw tracking rows or frames into chat. Tests use synthetic data only.
- Submission: exactly 199,319 rows, header `frame,label`, labels `H`/`A`/`D`/`X`; row i = tactical video frame i; ATD `frame_id = video_frame + 1`.
- Graded frames: 11185..90651 and 110808..190197. Frames 110808..113741 are `D` by rule. Ungraded frames are `X`.
- Every step deterministic (no unseeded randomness); target under ~30 min end to end.
- Repo must regenerate the exact CSV, be public, and exclude `data/` and derived tracking exports (`.cache/`).
- **Git:** stage files explicitly by path (never `git add -A` / `git add .`). Do **not** commit `docs/superpowers/` (spec and plan stay local, per the user).

## Review Focus

- Half starts/kickoffs where the ball is missing for a long stretch: state must still be defined (no crash, first possessor from first event or tag prior).
- Frames with zero or >22 tracked objects (halftime, duplicates): nearest-team features must be `-1`/NaN-safe, no warnings-as-errors.
- Smart Tagging `Team` = `N/A` (every throw-in) or an unexpected spelling (`Atletic Club` is spelled that way in the data): map to "no prior", never crash.
- Ball glitches: teleports (implausible speed) and coordinates outside 0..1 must not create false dead balls or false control.
- Off-by-one at the boundaries: ATD `frame_id - 1`, inclusive half bounds, final row count 199,319.

---

## File Structure

| File | Responsibility |
|---|---|
| `possession/__init__.py` | package marker |
| `possession/params.py` | all tunable constants, frozen priors, with a one-line rationale each |
| `possession/atd.py` | parse the raw tactical/broadcast ATD into arrays; npz cache |
| `possession/frames.py` | graded mask, forced-dead mask |
| `possession/clean.py` | ball spike removal, gap interpolation, duplicate-track suppression |
| `possession/features.py` | nearest-team distances, ball speed, in-pitch flag |
| `possession/dead.py` | dead-ball mask from tracking |
| `possession/tagging.py` | Smart Tagging to per-frame restart mask and team prior |
| `possession/control.py` | control events |
| `possession/statemachine.py` | H/A/D state machine and minimum-possession merge |
| `possession/labels.py` | abstention policy, final label array, CSV writer |
| `possession/diagnostics.py` | label-free stats and plausibility check |
| `possession/align_broadcast.py` | stage 2 alignment and ball fusion |
| `run.py` | CLI: build the submission end to end |
| `tests/test_*.py` | one file per module, synthetic data only |
| `TECHNICAL.md` | the technical document deliverable |
| `requirements-pipeline.txt` | pinned deps for the pipeline |

---

### Task 1: Scaffold, params, ATD loader

**Files:**
- Create: `possession/__init__.py`, `possession/params.py`, `possession/atd.py`, `tests/test_atd.py`, `requirements-pipeline.txt`
- Modify: `.gitignore` (add `.cache/`)

**Interfaces:**
- Produces:
  - `Tracking` (frozen dataclass): `players: np.ndarray (n_frames, n_slots, 2) float32` metres, NaN missing; `ball: np.ndarray (n_frames, 2) float32` metres, NaN missing; `team_of_slot: np.ndarray (n_slots,) int8` (0 home, 1 away).
  - `parse_atd_line(line: str, n_slots: int) -> Tuple[int, np.ndarray, np.ndarray]` returns `(frame_id, players (n_slots,2) normalised 0-1, ball (2,))`, NaN where `NaN`.
  - `load_tracking(source: str = "tactical", cache_dir: Optional[str] = ".cache") -> Tracking`; row index is `frame_id - 1`; coordinates converted to metres with `GAMES[GAME]["pitch"]`; npz cache keyed by source.
  - `params.py` constants (frozen priors, rationale in comments): `MAX_INTERP_GAP=12`, `SPIKE_SPEED=45.0` m/s, `DUP_SEP_M=0.5`, `CONTROL_RADIUS_M=2.5`, `CONTROL_MARGIN_M=1.0`, `CONTROL_MAX_SPEED=12.0`, `MIN_CONTROL_FRAMES=25`, `CONTROL_GAP_TOLERANCE=5`, `STATIONARY_SPEED=0.3`, `STATIONARY_FRAMES=75`, `OUT_MARGIN_M=0.5`, `OUT_FRAMES=25`, `LONG_LOSS_FRAMES=125`, `MIN_POSS_FRAMES=50`, `ABSTAIN_MIN_GAP=25`, `PLAUSIBLE_N_POSS=(120, 450)`, `PLAUSIBLE_MEAN_POSS_S=(6.0, 25.0)`, `SECOND_HALF_KICKOFF=113742`.

- [ ] **Step 1: Write the failing test** in `tests/test_atd.py`:

```python
def test_parse_atd_line_reads_players_and_ball():
    fid, players, ball = parse_atd_line("5:0.5,0.25,1.0;NaN,NaN,NaN:0.1,0.2,0.0", 2)
    assert fid == 5
    assert players[0].tolist() == [0.5, 0.25] and np.isnan(players[1]).all()
    assert ball.tolist() == [0.1, 0.2]

def test_parse_atd_line_missing_ball():
    _, _, ball = parse_atd_line("1:0.5,0.5,0.0:NaN,NaN,NaN", 1)
    assert np.isnan(ball).all()

def test_load_tracking_from_tiny_file_indexes_by_frame_id_minus_one(tmp_path):
    # tiny synthetic atd.txt + atd.xml via the data_dir/games.py layout; frame_id 1 -> row 0
    ...
```

- [ ] **Step 2: Run** `pytest tests/test_atd.py -v` — Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `parse_atd_line`, `Tracking`, `load_tracking` in `possession/atd.py`; team per slot via the same metadata logic as `starter/baseline.py:team_of_each_track` (import it, do not copy). Cache arrays with `np.savez` under `.cache/<source>.npz`.
- [ ] **Step 4: Run** `pytest tests/test_atd.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `git add possession/__init__.py possession/params.py possession/atd.py tests/test_atd.py requirements-pipeline.txt .gitignore && git commit -m "feat: ATD loader and frozen params"`

### Task 2: Frame masks

**Files:** Create `possession/frames.py`, `tests/test_frames.py`

**Interfaces:**
- Consumes: `games.half_video_blocks("BAR-ATH")`, `params.SECOND_HALF_KICKOFF`.
- Produces: `graded_mask(n_frames: int = 199319) -> np.ndarray` (bool); `forced_dead_mask(n_frames: int = 199319) -> np.ndarray` (bool, frames `110808..113741`).

- [ ] **Step 1: Failing test:**

```python
def test_graded_mask_counts_and_edges():
    m = graded_mask()
    assert m.sum() == 158857
    assert m[11185] and not m[11184] and m[90651] and not m[90652]
    assert m[110808] and m[190197] and not m[190198]

def test_forced_dead_mask_is_second_half_pre_kickoff():
    m = forced_dead_mask()
    assert m.sum() == 2934 and m[110808] and m[113741] and not m[113742]
```
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement** both functions.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** those two files.

### Task 3: Cleaning

**Files:** Create `possession/clean.py`, `tests/test_clean.py`

**Interfaces:**
- Produces:
  - `clean_ball(ball: np.ndarray, fps: int = 25) -> Tuple[np.ndarray, np.ndarray]` returns `(ball_filled (n,2), observed (n,) bool)`. Frames whose implied speed exceeds `SPIKE_SPEED` are set to NaN first; gaps up to `MAX_INTERP_GAP` frames are linearly interpolated; `observed` is true only for original non-NaN, non-spike frames.
  - `suppress_duplicate_tracks(players: np.ndarray, team_of_slot: np.ndarray) -> np.ndarray`: per frame, if two same-team slots are within `DUP_SEP_M`, set the higher-index slot to NaN.

- [ ] **Step 1: Failing tests:**

```python
def test_interpolates_short_gap_only():
    ball = np.full((40, 2), np.nan); ball[0] = (0, 0); ball[5] = (5, 0); ball[30] = (6, 0)
    filled, obs = clean_ball(ball)
    assert filled[2, 0] == pytest.approx(2.0) and not obs[2]
    assert np.isnan(filled[15]).all()           # gap of 24 > MAX_INTERP_GAP stays NaN

def test_spike_is_removed():
    ball = np.zeros((10, 2)); ball[5] = (50.0, 0.0)   # 50 m in one frame
    _, obs = clean_ball(ball)
    assert not obs[5]

def test_duplicate_same_team_track_suppressed():
    players = np.array([[[10, 10], [10.1, 10.1], [30, 30]]], dtype=float)   # (1,3,2)
    out = suppress_duplicate_tracks(players, np.array([0, 0, 1]))
    assert np.isnan(out[0, 1]).all() and not np.isnan(out[0, 0]).any()
```
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement. Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `possession/clean.py tests/test_clean.py`.

### Task 4: Features

**Files:** Create `possession/features.py`, `tests/test_features.py`

**Interfaces:**
- Consumes: `Tracking`, `clean_ball` outputs.
- Produces: `Features` dataclass with `d_team: (n,2) float32` (min distance from ball to nearest player of home/away, NaN if none or no ball), `nearest_team: (n,) int8` (-1 if undefined), `ball_speed: (n,) float32` (m/s, NaN where undefined), `ball_observed: (n,) bool`, `ball_in_pitch: (n,) bool`; `compute_features(players, ball_filled, observed, team_of_slot, pitch=(105.3, 68.0), fps=25) -> Features`.

- [ ] **Step 1: Failing tests:** nearest team picks the closer side; frame with zero players gives `nearest_team == -1` and NaN distances without warnings (`pytest -W error`); ball at `x=-1.0` m gives `ball_in_pitch` false; two-frame ball move of 1 m gives speed 25 m/s.
- [ ] **Step 2–4:** run FAIL, implement (NaN-safe with `np.nanmin` guarded by all-NaN checks), run PASS.
- [ ] **Step 5: Commit** both files.

### Task 5: Dead-ball detection

**Files:** Create `possession/dead.py`, `tests/test_dead.py`

**Interfaces:**
- Consumes: `Features`, `ball_filled`.
- Produces: `dead_ball_mask(features: Features, ball_filled: np.ndarray, centre=(52.65, 34.0)) -> np.ndarray` (bool). `True` where any holds: ball outside pitch (margin `OUT_MARGIN_M`) for `OUT_FRAMES`; observed stationary (`< STATIONARY_SPEED`) for `STATIONARY_FRAMES`; ball unobserved for `LONG_LOSS_FRAMES`; ball within 1.5 m of the centre spot and stationary (kick-off).

- [ ] **Step 1: Failing tests:** 30 frames outside pitch → dead (and 10 frames outside → not dead); 100 stationary frames mid-pitch → dead; a 12 m/s moving ball → never dead; a 200-frame NaN stretch → dead, a 50-frame NaN stretch → not.
- [ ] **Step 2–4:** FAIL, implement with run-length logic, PASS.
- [ ] **Step 5: Commit.**

### Task 6: Smart Tagging priors

**Files:** Create `possession/tagging.py`, `tests/test_tagging.py`

**Interfaces:**
- Consumes: `starter/load_smart_tagging.py` (read its API; do not reimplement the XML parsing), `games.asset_paths()["smart_tagging"]`.
- Produces: `TagPriors` dataclass `restart_mask (n,) bool` (frames inside `SET PIECES` intervals, `video_frame = round(seconds*25)`), `team_prior (n,) int8` (0 home, 1 away, -1 none); `load_tag_priors(n_frames: int = 199319) -> TagPriors`; `team_from_tag(value: str) -> int` mapping `"Barcelona"→0`, `"Atletic Club"→1` (sic) and anything else including `"N/A"` and `""`→`-1`.

- [ ] **Step 1: Failing tests:** `team_from_tag("N/A") == -1`; `team_from_tag("Atletic Club") == 1` and `team_from_tag("Athletic Club") == 1` (both spellings); `team_from_tag(None) == -1`; interval 10.0–12.0 s sets frames 250..300 restart (use a synthetic interval list helper `priors_from_intervals`).
- [ ] **Step 2–4:** FAIL, implement, PASS.
- [ ] **Step 5: Commit.**

### Task 7: Control events

**Files:** Create `possession/control.py`, `tests/test_control.py`

**Interfaces:**
- Consumes: `Features`.
- Produces: `ControlEvent` (frozen dataclass `start: int, end: int, team: int`, inclusive frames); `find_control_events(features: Features) -> List[ControlEvent]`. A frame supports team T when the ball is observed or interpolated, `d_team[T] <= CONTROL_RADIUS_M`, the other team is at least `CONTROL_MARGIN_M` further, and `ball_speed <= CONTROL_MAX_SPEED`. Support runs tolerate gaps up to `CONTROL_GAP_TOLERANCE` and must last at least `MIN_CONTROL_FRAMES`.

- [ ] **Step 1: Failing tests:** 40 frames of home player within 1 m, away 8 m → one event team 0 spanning those frames; a 10-frame touch → no event; away player 1.5 m vs home 1.0 m (margin not met) → no event; a ball flying at 20 m/s past a player → no event; two support runs separated by 3 frames → one merged event.
- [ ] **Step 2–4:** FAIL, implement, PASS.
- [ ] **Step 5: Commit.**

### Task 8: State machine

**Files:** Create `possession/statemachine.py`, `tests/test_statemachine.py`

**Interfaces:**
- Consumes: `ControlEvent` list, `dead: np.ndarray bool`, `forced_dead: np.ndarray bool`, `TagPriors`.
- Produces: `run_state_machine(n_frames: int, events: List[ControlEvent], dead: np.ndarray, forced_dead: np.ndarray, priors: TagPriors) -> np.ndarray` (int8 codes: 0 H, 1 A, 2 D). Rules: start in `D`; `D` where `dead | forced_dead`; on leaving `D` the possessor is the first control event's team within the next 5 s, else `priors.team_prior` at the restart frame, else the next event's team; a live possession flips only when an event of the other team begins; no event means keep the last possessor; afterwards merge any possession shorter than `MIN_POSS_FRAMES` into its predecessor.

- [ ] **Step 1: Failing tests:**

```python
def test_flip_only_on_opponent_event():
    ev = [ControlEvent(0, 40, 0), ControlEvent(200, 260, 1)]
    codes = run_state_machine(400, ev, np.zeros(400, bool), np.zeros(400, bool), empty_priors(400))
    assert set(codes[0:200]) == {0} and set(codes[200:]) == {1}

def test_loose_ball_keeps_last_possessor():   # no events between 40 and 200 -> still home
def test_dead_interval_breaks_possession_and_restart_uses_tag_prior():
def test_short_possession_merged():           # 30-frame away possession inside home run disappears
def test_forced_dead_second_half_pre_kickoff_is_all_D():
def test_no_events_at_all_is_defined():       # half with no ball: codes are 2 or tag prior, no crash
```
- [ ] **Step 2–4:** FAIL, implement, PASS.
- [ ] **Step 5: Commit.**

### Task 9: Labels, abstention, CSV

**Files:** Create `possession/labels.py`, `tests/test_labels.py`

**Interfaces:**
- Consumes: state-machine codes, `ball_observed`, `graded_mask`, `forced_dead_mask`; `starter.baseline.write_submission(path, labels)` (reuse).
- Produces: `assemble_labels(codes: np.ndarray, observed: np.ndarray, graded: np.ndarray, forced_dead: np.ndarray) -> np.ndarray` (`<U1`, length n). Ungraded → `X`; forced dead → `D`; inside a possession, where ball unobserved for at least `ABSTAIN_MIN_GAP` consecutive frames **and** the label is the same on both sides of the gap → `X`; never `X` on a transition frame or within the 25 frames of a transition. `write_labels(path, labels) -> None`.

- [ ] **Step 1: Failing tests:** ungraded frame is `X`; unobserved 50-frame gap inside `H…H` becomes `X`; a gap spanning an `H→A` change stays labelled; output length 199,319 in a full-size call; only `H/A/D/X` characters.
- [ ] **Step 2–4:** FAIL, implement, PASS.
- [ ] **Step 5: Commit.**

### Task 10: Diagnostics and plausibility check

**Files:** Create `possession/diagnostics.py`, `tests/test_diagnostics.py`

**Interfaces:**
- Produces: `possession_stats(labels: np.ndarray, graded: np.ndarray, fps: int = 25) -> dict` with keys `n_possessions`, `mean_poss_s`, `median_poss_s`, `share_D`, `share_X` computed per run of `H`/`A` after forward-filling `X` within each half (same as the scorer); `check_plausible(stats: dict) -> List[str]` returns violation messages against `PLAUSIBLE_N_POSS` and `PLAUSIBLE_MEAN_POSS_S`. Neither function accepts ground truth.

- [ ] **Step 1: Failing tests:** `H H H A A D D H H` gives 3 possessions (D splits the same team); X inside a run is forward-filled and does not split; an over-fragmented label array yields a violation string mentioning `n_possessions`.
- [ ] **Step 2–4:** FAIL, implement, PASS.
- [ ] **Step 5: Commit.**

### Task 11: CLI end to end, public sanity check, determinism

**Files:** Create `run.py`, `tests/test_run_smoke.py`

**Interfaces:**
- Consumes: all of the above.
- Produces: `python run.py --team NAME [--public-check]` writes `evaluation/results/NAME.csv`, prints `possession_stats` and plausibility messages, runs `evaluation/validate_submission.py` logic, and with `--public-check` runs the scorer once against the public GT and prints the result (no looping, no parameter sweep).

- [ ] **Step 1: Failing test:** smoke test on a synthetic 3000-frame tracking (monkeypatching `load_tracking`) produces a file with the right header and only valid labels.
- [ ] **Step 2–4:** FAIL, implement, PASS.
- [ ] **Step 5: Run on the real data:** `python run.py --team <team_name>` — Expected: validation passes, 199,319 rows, plausibility messages empty. If plausibility fails, adjust parameters only by the tuning rule (priors/physics), never by public-window score; record the change in `TECHNICAL.md`.
- [ ] **Step 6: Determinism check:** run twice, `shasum -a 256 evaluation/results/<team_name>.csv` identical both times.
- [ ] **Step 7: Run once** `python run.py --team <team_name> --public-check`; record the number as a sanity check only.
- [ ] **Step 8: Commit** `run.py tests/test_run_smoke.py`.

### Task 12: Technical document and repo hygiene

**Files:** Create `TECHNICAL.md`; Modify `README.md` (reproduction command only if adding a short section; keep organiser README intact).

- [ ] **Step 1:** Write `TECHNICAL.md` covering: approach and why; assets used and dropped (broadcast video dropped; broadcast ATD stage 2 if kept); training data (none); possession rules implemented (control event definition, loose ball, restarts, dead-ball signals, abstention policy, minimum possession); limitations; reproduction (`pip install -r requirements-pipeline.txt`, `python run.py --team <team_name>`); a list of every parameter in `params.py` with its prior rationale.
- [ ] **Step 2:** Check `git status` shows no `data/`, `.cache/`, results CSVs (except baselines) or `docs/superpowers/` staged.
- [ ] **Step 3:** Pin `requirements-pipeline.txt` with `pip freeze` versions of numpy, pandas, pytest.
- [ ] **Step 4: Commit** `TECHNICAL.md requirements-pipeline.txt`.

### Task 13 (optional stretch, cut first): Broadcast fusion

**Files:** Create `possession/align_broadcast.py`, `tests/test_align_broadcast.py`; Modify `run.py` (flag `--fuse-broadcast`).

**Interfaces:**
- Consumes: tactical and broadcast `Tracking` (via `load_tracking("broadcast")`), broadcast homography `valid` column.
- Produces: `Segment` (frozen dataclass `b_start: int, b_end: int, lag: int, confidence: float`); `estimate_segments(tactical: Tracking, broadcast: Tracking, valid: np.ndarray, seg_len: int = 750, max_lag: int = 3000) -> List[Segment]` (cross-correlation of player centroid tracks over valid stretches; checks axis flip per half); `fuse_ball(tactical_ball: np.ndarray, broadcast_ball: np.ndarray, segments: List[Segment], min_conf: float) -> Tuple[np.ndarray, np.ndarray]` returns `(fused_ball, filled_mask)`: fills only frames where the tactical ball is NaN and a confident aligned broadcast ball exists.

- [ ] **Step 1: Failing tests:** broadcast track equal to a tactical track shifted by 37 frames recovers `lag == 37`; a mirrored (x-flipped) broadcast is detected and corrected; `fuse_ball` never overwrites an existing tactical ball value; low-confidence segments are ignored.
- [ ] **Step 2–4:** FAIL, implement, PASS.
- [ ] **Step 5: Real-data decision (label-free only):** report aligned ball coverage before/after and the plausibility stats for both variants. Keep fusion only if coverage rises clearly, alignment confidence is high on the used segments, and the stats stay plausible. Do not use the public window to decide. Document either outcome in `TECHNICAL.md`.
- [ ] **Step 6: Commit** `possession/align_broadcast.py tests/test_align_broadcast.py run.py`.

---

## Self-review notes

- **Spec coverage:** load/clean (T1,T3), features (T4), control events (T7), state machine (T8), Smart Tagging priors incl. `N/A` (T6), abstention policy and forced-dead frames (T2,T9), label-free tuning rule and plausibility (T10, T11 step 5), validation, determinism, public sanity check once (T11), deliverables (T11, T12), stage 2 optional (T13). Attack direction is not implemented: the dead-ball rules do not need it (YAGNI); revisit only if kick-off geometry proves necessary.
- **Review Focus coverage:** long ball loss at half start → T8 `test_no_events_at_all_is_defined`; zero/duplicate tracks → T3, T4; `N/A` tags → T6; ball glitches → T3 spike test, T4 `ball_in_pitch`; off-by-one → T1, T2, T9.
- **Type consistency:** `Tracking`, `Features`, `TagPriors`, `ControlEvent`, `Segment` are defined once in the task that produces them and referenced by those names elsewhere.
