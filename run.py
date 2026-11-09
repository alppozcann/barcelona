"""Build the possession submission end to end.

    python run.py --team MYTEAM                         # data in ./data or $MTAH_DATA_DIR
    python run.py --team MYTEAM --data-dir /content/drive/MyDrive/mtah2027   # e.g. Google Colab
    python run.py --team MYTEAM --public-check          # also score once on the public 5 minutes

Writes <out-dir>/<team>.csv (default evaluation/results/), prints label-free diagnostics and runs the
official validator on the file. Deterministic: the same data gives the same file.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from possession import params  # noqa: E402
from possession.atd import Tracking, load_tracking  # noqa: E402
from possession.clean import clean_ball, suppress_duplicate_tracks  # noqa: E402
from possession.control import find_control_events  # noqa: E402
from possession.dead import dead_ball_mask  # noqa: E402
from possession.diagnostics import check_plausible, possession_stats  # noqa: E402
from possession.features import compute_features  # noqa: E402
from possession.frames import forced_dead_mask, graded_mask  # noqa: E402
from possession.labels import assemble_labels, write_labels  # noqa: E402
from possession.statemachine import run_state_machine  # noqa: E402
from possession.tagging import TagPriors, load_tag_priors  # noqa: E402


def build_labels(tracking: Tracking, priors: TagPriors, graded: np.ndarray,
                 forced_dead: np.ndarray, ball_override: np.ndarray = None) -> np.ndarray:
    """Tracking + Smart Tagging priors -> one label per frame. Pure and deterministic."""
    ball = tracking.ball if ball_override is None else ball_override
    ball_filled, observed = clean_ball(ball)
    players = suppress_duplicate_tracks(tracking.players, tracking.team_of_slot)
    features = compute_features(players, ball_filled, observed, tracking.team_of_slot)
    events = find_control_events(features)
    dead = dead_ball_mask(features, ball_filled)
    codes = run_state_machine(len(ball), events, dead, forced_dead, priors)
    return assemble_labels(codes, observed, graded, forced_dead)


def fuse_and_write(args, tracking, priors, graded, forced, base_stats) -> str:
    """Stage 2. Prints label-free aggregates only: coverage gain, alignment quality, diagnostics."""
    from possession.align_broadcast import estimate_segments, fuse_ball, load_valid
    print("stage 2: loading broadcast tracking and aligning ...")
    broadcast = load_tracking("broadcast", data_dir=args.data_dir, cache_dir=args.cache_dir or None)
    valid = load_valid("broadcast", args.data_dir)
    b_ball = np.where(valid[:, None], broadcast.ball, np.nan)
    segments = estimate_segments(tracking.ball, b_ball, valid)
    fused, filled = fuse_ball(tracking.ball, b_ball, segments)
    before = float((~np.isnan(tracking.ball[:, 0]))[graded].mean())
    after = float((~np.isnan(fused[:, 0]))[graded].mean())
    covered = sum(s.b_end - s.b_start for s in segments)
    print("alignment: {} segments accepted covering {:.1f}% of the broadcast timeline; "
          "{} distinct lags; {} mirrored".format(
              len(segments), 100.0 * covered / len(b_ball), len({s.lag for s in segments}),
              sum(s.flip for s in segments)))
    print("ball coverage on graded frames: {:.1f}% -> {:.1f}% ({} frames filled)".format(
        100 * before, 100 * after, int(filled[graded].sum())))
    labels = build_labels(tracking, priors, graded, forced, ball_override=fused.astype(np.float32))
    path = os.path.join(args.out_dir, "{}_fused.csv".format(args.team))
    write_labels(path, labels)
    stats = possession_stats(labels, graded)
    print("fused diagnostics (label-free): " + ", ".join(
        "{}={:.3f}".format(k, v) if isinstance(v, float) else "{}={}".format(k, v)
        for k, v in stats.items()))
    for message in check_plausible(stats):
        print("PLAUSIBILITY WARNING (fused): " + message)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", required=True, help="team name, used for the output file name")
    parser.add_argument("--data-dir", default=None, help="folder holding tactical/ and broadcast/")
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "evaluation", "results"))
    parser.add_argument("--cache-dir", default=os.path.join(ROOT, ".cache"),
                        help="parsed-tracking cache; pass '' to disable")
    parser.add_argument("--fuse-broadcast", action="store_true",
                        help="stage 2: also write <team>_fused.csv using the aligned broadcast ball")
    parser.add_argument("--public-check", action="store_true",
                        help="score once against the public 5 minutes (sanity check only)")
    args = parser.parse_args(argv)

    print("loading tactical tracking ...")
    tracking = load_tracking("tactical", data_dir=args.data_dir, cache_dir=args.cache_dir or None)
    n = len(tracking.ball)
    priors = load_tag_priors(n, data_dir=args.data_dir)
    graded, forced = graded_mask(n), forced_dead_mask(n)

    labels = build_labels(tracking, priors, graded, forced)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "{}.csv".format(args.team))
    write_labels(path, labels)
    print("wrote {}".format(path))

    stats = possession_stats(labels, graded)
    print("diagnostics (label-free): " + ", ".join(
        "{}={:.3f}".format(k, v) if isinstance(v, float) else "{}={}".format(k, v)
        for k, v in stats.items()))
    for message in check_plausible(stats):
        print("PLAUSIBILITY WARNING: " + message)

    ok = subprocess.call([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), path])
    if ok != 0:
        return ok
    if args.fuse_broadcast:
        fused_path = fuse_and_write(args, tracking, priors, graded, forced, stats)
        subprocess.call([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), fused_path])
    if args.public_check:
        subprocess.call([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
