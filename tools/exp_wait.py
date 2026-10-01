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
look at); DIED when the job's `cluster.out` carries an LSF `TERM_` reason or "Exited with exit code N" (a crash). Nothing is run on the
cluster and no login node is touched: `/groups` is mounted, so the folders are read locally.

THEN IT LANDS THE BATCH (2026-09-27, unless `--no-land`): `tools/exp_land.py` checks every run's
health (counts, extent, non-finite values in every set, the growth audit, a still movie), scores the
card, runs `--check`, rebuilds the watcher's record, writes the landing report into the experiment's
markdown, and prints it -- so the session wakes to a batch already checked and scored, and the time
its next batch is due (10 minutes later).

RUNS OUTSIDE THE MARKDOWN'S ARMS: `--spec <group>/<name> ...` waits on those output folders directly
(landed = trajectory.npz, and movie.mp4 unless `--no-movie`; died = an LSF failure in
`log/experiments/expNN/<name>/cluster.out` if that exists).

EXIT CODES: 0 every run landed | 1 a run died (the others are reported) | 2 timeout.
"""
from __future__ import annotations

import argparse
import re
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

import exp  # noqa: E402


def _lsf_died(n, run):
    """The LSF verdict of a finished job that did NOT succeed, read from its cluster.out: 'Exited with exit code
    N' or a TERM_ reason. exp.measure only knew TERM_, so a job that crashed with exit code 1 (exp12's CUDA OOM,
    2026-09-27: movie written, no trajectory) was never reported and the waiter sat on it for 2.5 h."""
    co = os.path.join(exp.job_dir(n, run), "cluster.out")
    if not os.path.isfile(co):
        return None
    txt = open(co, errors="ignore").read()
    m = re.search(r"Exited with exit code (\d+)", txt)
    if m:
        why = re.search(r"(CUDA out of memory|Traceback \(most recent call last\)|Killed|MemoryError)", txt)
        return f"exit code {m.group(1)}" + (f" ({why.group(1)})" if why else "")
    t = re.search(r"(TERM_\w+)", txt)
    return t.group(1) if t else None


def _state(arm, run, n, need_movie):
    dead = _lsf_died(n, run)
    if dead:
        return "died", {"died": dead}
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


def _train_state(run):
    """A TRAINER'S run, `training/<model>/<name>` (exp16): LANDED when its analysis figure
    `results/<name>_test.png` exists (analyse is the last phase, after test); DIED when its job's
    `log/runs/<name>/run.out` (where `tools/submit_runs.py` sends LSF's stdout, and a local run.sh its own)
    reports an LSF TERM_ reason or a non-zero exit code."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from exp_measures.common import run_dir
    d = run_dir(run)
    name = os.path.basename(d.rstrip("/"))
    if os.path.exists(os.path.join(d, "results", f"{name}_test.png")):
        return "landed", {}
    ro = os.path.join(ROOT, "log", "runs", name, "run.out")
    if os.path.isfile(ro):
        txt = open(ro, errors="ignore").read()
        m = re.search(r"Exited with exit code (\d+)|^exit code ([1-9]\d*)", txt, re.M)
        t = re.search(r"(TERM_\w+)", txt)
        if m or t:
            return "died", {"died": (f"exit code {m.group(1) or m.group(2)}" if m else t.group(1))}
        return "running", {}
    return "queued", {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--runs", nargs="*", help="run names (default: every submitted run not yet landed)")
    ap.add_argument("--arm", action="append", help="only these arms")
    ap.add_argument("--every", type=int, default=120, help="seconds between checks (default 120)")
    ap.add_argument("--timeout", type=int, default=6 * 3600, help="give up after this many seconds (default 6 h)")
    ap.add_argument("--no-movie", action="store_true", help="landed = trajectory only, do not wait for the movie")
    ap.add_argument("--spec", nargs="*", default=[], help="<group>/<name> runs not in the markdown's arms")
    ap.add_argument("--no-land", action="store_true", help="do not run tools/exp_land.py when the batch is decided")
    ap.add_argument("--train", nargs="*", default=[], help="training/<model>/<name> runs of the trainer (exp16)")
    a = ap.parse_args()
    if a.train:
        print(f"[wait] exp{a.n:02d}: {len(a.train)} training run(s), a check every {a.every} s: "
              + " ".join(a.train), flush=True)
        t0 = time.time()
        while True:
            st = {r: _train_state(r) for r in a.train}
            if all(v[0] in ("landed", "died") for v in st.values()) or time.time() - t0 > a.timeout:
                break
            time.sleep(a.every)
        el = int(time.time() - t0)
        for r, (s_, m) in st.items():
            print(f"  {s_:9s} {r}  {m.get('died', '')}")
        died = [r for r, v in st.items() if v[0] == "died"]
        left = [r for r, v in st.items() if v[0] not in ("landed", "died")]
        landed = [r for r, v in st.items() if v[0] == "landed"]
        summary = (f"waited {el} s: {len(landed)} landed, {len(died)} died" + (f" ({', '.join(died)})" if died else "")
                   + (f", {len(left)} still open" if left else ""))
        print(f"[wait] exp{a.n:02d}: {summary}")
        if landed and not a.no_land:
            import exp_land
            print(exp_land.land(a.n, landed, waited=summary), flush=True)
        return 2 if left else (1 if died else 0)
    fm, _ = exp.load(exp.exp_path(a.n))
    todo = []
    for sp in a.spec:
        group, name = sp.split("/", 1)
        todo.append(({"id": "(spec)", "kind": "generate", "spec": sp}, name))
    submitted = set((fm.get("job_ids") or {}).keys())
    for arm, _p, run in ([] if a.spec and not (a.runs or a.arm) else exp.runs(fm)):
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
    landed = [(arm, r) for arm, r in todo if st[r][0] == "landed" and arm["kind"] == "generate"]
    if a.no_land or not landed:
        print(f"[wait] NEXT: score this batch now -- PYTHONPATH=src:tools python tools/exp_gate_score.py {a.n}"
              f"  (and --check if a gate reads 'no value' or 'INVALID')")
    else:
        import exp_land
        summary = (f"waited {el} s: {len(st) - len(died) - len(left)} landed, {len(died)} died"
                   + (f" ({', '.join(died)})" if died else "") + (f", {len(left)} still open" if left else ""))
        print(exp_land.land(a.n, [f"{arm['spec'].split('/')[0]}/{r}" for arm, r in landed], waited=summary), flush=True)
    return 2 if left else (1 if died else 0)


if __name__ == "__main__":
    sys.exit(main())
