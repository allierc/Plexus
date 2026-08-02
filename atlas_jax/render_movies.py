"""render_movies -- give every atlas evidence folder its movie, WITHOUT re-simulating.

Most `log/atlas_jax/<name>/` folders were produced by a differ in a hurry, with `--no-movie`. The
movie is the one artefact a human actually looks at, and it is missing from 13 of 19 folders.

THE POINT OF THIS SCRIPT IS WHAT IT DOES NOT DO. Re-running `run_spec.py` would remake the movie,
and would also overwrite `diag.json`, `metrics.json`, `metrics.npz`, `strip.png` and
`spec_run.yaml` -- the acted ledger and the numbers that sixteen differential tests were scored
against, several of them from stochastic runs that will not reproduce frame-for-frame. Evidence is
not something to regenerate for a nicer picture.

So this renders from the trajectory the run already saved (`graphs_data/atlas/<name>/
trajectory.npz`) via `plexus.plot.plot_dataset`, which reads that file and simulates nothing. The
evidence folder gains `movie.mp4` and loses nothing.

A folder whose render fails is REPORTED, never skipped quietly -- a missing movie that looks
deliberate is the exact failure this campaign keeps catching.

    python render_movies.py              # every folder missing a movie
    python render_movies.py --only division harmonic
    python render_movies.py --force      # re-render even where a movie exists
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PLEXUS, "src"))
sys.path.insert(0, HERE)

LOG_DIR = os.path.join(PLEXUS, "log", "atlas_jax")

from run_spec import load_atlas_candidates  # noqa: E402  -- the anti-chamber, so a spec can load


def targets(only, force):
    """(name, spec_path, traj_path) for every evidence folder that wants a movie."""
    out = []
    for name in sorted(os.listdir(LOG_DIR)):
        d = os.path.join(LOG_DIR, name)
        if not os.path.isdir(d) or (only and name not in only):
            continue
        if os.path.isfile(os.path.join(d, "movie.mp4")) and not force:
            continue
        out.append((name, os.path.join(d, "spec_run.yaml"),
                    os.path.join(PLEXUS, "graphs_data", "atlas_jax", name, "trajectory.npz")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    import plexus.operators  # noqa: F401
    load_atlas_candidates()
    from plexus.plot import plot_dataset
    from plexus.schema import load

    todo = targets(set(a.only) if a.only else None, a.force)
    print(f"[render] {len(todo)} folder(s) without a movie\n")

    done, failed = [], []
    for name, spec_path, traj in todo:
        out_dir = os.path.join(LOG_DIR, name)
        if not os.path.isfile(spec_path):
            failed.append((name, "no spec_run.yaml -- this folder is not a completed run"))
            continue
        if not os.path.isfile(traj):
            failed.append((name, "no saved trajectory -- needs a fresh `run_spec.py` run"))
            continue
        try:
            sim = load(spec_path)
            data_dir = plot_dataset(sim, "atlas_jax", movie=True)
            mp4 = next((f for f in sorted(os.listdir(data_dir)) if f.endswith(".mp4")), None)
            if mp4 is None:
                failed.append((name, "plot_dataset wrote no .mp4 (a set with no positions?)"))
                continue
            dst = os.path.join(out_dir, "movie.mp4")
            shutil.copyfile(os.path.join(data_dir, mp4), dst)
            mb = os.path.getsize(dst) / 1e6
            done.append((name, mb))
            print(f"  ok    {name:<30} {mb:.1f} MB")
        except Exception as e:                    # reported, never swallowed
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"  FAIL  {name:<30} {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)

    print(f"\n[render] {len(done)} rendered, {len(failed)} failed")
    for name, why in failed:
        print(f"  MISSING  {name:<30} {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
