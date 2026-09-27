#!/usr/bin/env python
"""Wait for ONE batch of an experiment to land, in a single background call -- instead of polling.

    PYTHONPATH=src python tools/exp_wait.py 9                        every submitted run not yet landed
    PYTHONPATH=src python tools/exp_wait.py 9 --runs exp09_a_gt_b_s1 exp09_a_gt_b_s2
    PYTHONPATH=src python tools/exp_wait.py 9 --arm a_gt_b --every 120 --timeout 21600

WHY. A session that polls (`tools/exp.py poll` every few minutes, a wake-up per check) spends tokens
on every look at a batch that has not moved. This tool is started ONCE per batch with the shell's
background mode (`run_in_background`), sleeps between checks without the model, and exits when the
batch is decided -- every run landed or died, or the timeout passed -- so the session is woken
exactly once, with the summary.

WHAT IT CHECKS, and only this: each run's own folders, through `tools/exp.py`'s `measure` (the same
definition `exp.py poll` uses): LANDED when the run's `trajectory.npz` reads (and, for a generate
arm, its `movie.mp4` exists too, unless `--no-movie` -- the movie is what the record and the judge
look at); DIED when the job's `cluster.out` carries an LSF `TERM_` reason. Nothing is run on the
cluster and no login node is touched: `/groups` is mounted, so the folders are read locally.

EXIT CODES: 0 every run landed | 1 a run died (the others are reported) | 2 timeout.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

import exp  # noqa: E402


def _state(arm, run, n, need_movie):
    try:
        m = exp.measure(arm, run, n)
    except Exception:                                                        # noqa: BLE001  a half-written npz
        return "running", {}
    if m.get("died"):
        return "died", m
    if m.get("landed"):
        if need_movie and arm["kind"] == "generate":
            if not os.path.exists(os.path.join(exp.out_dir(arm, run), "movie.mp4")):
                return "rendering", m
        return "landed", m
    return m.get("state", "queued"), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--runs", nargs="*", help="run names (default: every submitted run not yet landed)")
    ap.add_argument("--arm", action="append", help="only these arms")
    ap.add_argument("--every", type=int, default=120, help="seconds between checks (default 120)")
    ap.add_argument("--timeout", type=int, default=6 * 3600, help="give up after this many seconds (default 6 h)")
    ap.add_argument("--no-movie", action="store_true", help="landed = trajectory only, do not wait for the movie")
    a = ap.parse_args()
    fm, _ = exp.load(exp.exp_path(a.n))
    todo = []
    submitted = set((fm.get("job_ids") or {}).keys())
    for arm, _p, run in exp.runs(fm):
        if a.arm and arm["id"] not in a.arm:
            continue
        if a.runs:
            if run in a.runs:
                todo.append((arm, run))
        elif run in submitted and _state(arm, run, a.n, not a.no_movie)[0] not in ("landed", "died"):
            todo.append((arm, run))
    if not todo:
        print(f"[wait] exp{a.n:02d}: nothing to wait for (no submitted run is still open)")
        return 0
    print(f"[wait] exp{a.n:02d}: {len(todo)} run(s), a check every {a.every} s, timeout {a.timeout} s: "
          + " ".join(r for _, r in todo), flush=True)
    t0 = time.time()
    while True:
        st = {run: _state(arm, run, a.n, not a.no_movie) for arm, run in todo}
        open_ = [r for r, (s, _) in st.items() if s not in ("landed", "died")]
        if not open_ or time.time() - t0 > a.timeout:
            break
        time.sleep(a.every)
    el = int(time.time() - t0)
    for run, (s, m) in st.items():
        extra = ", ".join(f"{k} {m[k]}" for k in ("died", "wall_s", "frames", "ms_per_frame") if k in m)
        print(f"  {s:9s} {run}  {extra}")
    died = [r for r, (s, _) in st.items() if s == "died"]
    left = [r for r, (s, _) in st.items() if s not in ("landed", "died")]
    print(f"[wait] exp{a.n:02d}: after {el} s -- {len(st) - len(died) - len(left)} landed, {len(died)} died, "
          f"{len(left)} still open{' (TIMEOUT)' if left else ''}")
    return 2 if left else (1 if died else 0)


if __name__ == "__main__":
    sys.exit(main())
