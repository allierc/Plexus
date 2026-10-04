#!/usr/bin/env python
"""A smoke test on a cluster L4 in about two minutes, instead of 20-30 minutes on the container's CPU.

    PYTHONPATH=src:tools python tools/exp_smoke.py tissue/exp11_p2_base                      100 frames, waits, prints health
    PYTHONPATH=src:tools python tools/exp_smoke.py tissue/exp11_p2_base --frames 200 --set general.seed=3 \\
        --set "operators[bm_bond].overdamped_gamma=5"
    PYTHONPATH=src:tools python tools/exp_smoke.py tissue/exp14_p3_base --full               the LONG-RUN CANARY: full length

WHY. Sessions checked a new rig on the devcontainer CPU before a batch: 200 frames took 20-30 minutes
(exp13's largest single cost per batch), while the same frames take 1-2 minutes on an L4 with no
queue wait. And some defects only appear late (exp14: mesh degradation surfaced at 35,000 frames and
cost a whole round) -- `--full` runs one copy at the spec's own length first, as a canary.

WHAT IT DOES. Copies `config/<group>/<name>.yaml` to `config/<group>/smoke_<name>.yaml` (`canary_`
with `--full`) with `general.n_frames` cut to `--frames` and any `--set` dotted overrides (the same
addressing as an arm's `differs_by`, `operators[<op>].<key>`), submits it to the L4 queue with a short
wall (30 min; 240 with `--full`), waits for the trajectory and the movie, and prints the run's
health (`tools/exp_land.py --health`: counts, extent, non-finite values in every set, the growth
audit, a still movie) with the queue wait and the run time. The output lands in
`graphs_data/<group>/smoke_<name>/`; the `smoke_`/`canary_` prefix keeps it out of the experiment's
record and scoring. `--no-wait` submits and returns (put `--full` in the shell's background mode).

`--exp N` ALSO RUNS THE EXPERIMENT'S RULERS on the smoke run -- every measure `gates.yaml` declares --
and checks every key a gate or cap reads against what the rulers returned (the scorer's UNKNOWN
KEY / UNKNOWN MEASURE / RULER ERROR checks, before any batch exists). This is the "dry run clean"
of a new experiment or a new rig: the rig runs, it is healthy, and the card can read it.
"""
from __future__ import annotations

import argparse
import copy
import os
import re
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

import exp  # noqa: E402


def _value(v):
    try:
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


def make_spec(spec: str, frames: int | None, sets: list[str], full: bool) -> tuple[str, str, str]:
    group, name = spec.split("/", 1)
    src = os.path.join(ROOT, "config", group, f"{name}.yaml")
    if not os.path.isfile(src):
        raise SystemExit(f"{src} does not exist")
    s = copy.deepcopy(yaml.safe_load(open(src)))
    run = ("canary_" if full else "smoke_") + name
    s["general"]["name"] = run
    if not full and frames:
        n0 = int(s["general"].get("n_frames", frames))
        s["general"]["n_frames"] = min(int(frames), n0)
    for kv in sets:
        k, _, v = kv.partition("=")
        exp._set_dotted(s, k.strip(), _value(v))
    dst = os.path.join(ROOT, "config", group, f"{run}.yaml")
    yaml.safe_dump(s, open(dst, "w"), sort_keys=False, default_flow_style=None, width=100)
    return group, run, dst


def _lsf(co):
    if not os.path.isfile(co):
        return {}
    txt = open(co, errors="ignore").read()
    out = {}
    for key, pat in (("run_s", r"Run time :\s+(\d+) sec"), ("exit", r"Exited with exit code (\d+)"),
                     ("term", r"(TERM_\w+)")):
        m = re.search(pat, txt)
        if m:
            out[key] = m.group(1)
    if "exit" in out:
        why = re.search(r"(CUDA out of memory|Traceback \(most recent call last\)|Killed|MemoryError|\w+Error: .{0,120})", txt)
        out["why"] = why.group(1) if why else ""
    return out


def check_keys(n: int, run: str) -> int:
    """Run experiment n's declared measures on `run` and lint gates.yaml against their keys; print; return
    the number of problems."""
    import exp_gate_score as S
    import exp_measures
    G, f = S.load_gates(n)
    vals, errs = {}, []
    for m in G.get("measures") or []:
        try:
            v = exp_measures.run_measure(m["measure"], run, **dict(m.get("kw") or {}))
        except Exception as e:                                                       # noqa: BLE001
            errs.append(f"RULER ERROR {m['measure']}: {type(e).__name__}: {e}")
            continue
        vals.update({f"{m['measure']}.{k}": x for k, x in (v or {}).items()})
    M = {(arm, None): vals for arm in (G.get("runs") or {"(smoke)": None})}
    probs, rerr = S.lint(G, M)
    bad = [f"{gid}: {p}" for gid, ps in probs.items() for p in ps if not p.startswith("NOT ON ARM")] + errs + rerr
    print(f"[smoke] exp{n:02d} rulers on {run}: {len(G.get('measures') or [])} measure(s), {len(vals)} key(s) returned; "
          + ("every gate key found" if not bad else f"{len(bad)} problem(s):"))
    for b in bad:
        print("   ", b)
    return len(bad)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="<group>/<name> under config/")
    ap.add_argument("--frames", type=int, default=100, help="frames to run (default 100; ignored with --full)")
    ap.add_argument("--set", action="append", default=[], help="dotted override KEY=VALUE, repeatable")
    ap.add_argument("--full", action="store_true", help="the long-run canary: the spec's own length, wall 240 min")
    ap.add_argument("--no-wait", action="store_true", help="submit and return")
    ap.add_argument("--timeout", type=int, default=None, help="seconds to wait (default 1800; 14400 with --full)")
    ap.add_argument("--exp", type=int, default=None, help="also run experiment N's rulers on the smoke and check the gate keys")
    a = ap.parse_args()
    group, run, dst = make_spec(a.spec, a.frames, a.set, a.full)
    jd = os.path.join(ROOT, "log", "experiments", "smoke", run)
    os.makedirs(jd, exist_ok=True)
    co = os.path.join(jd, "cluster.out")
    for f in (co, co[:-4] + ".err"):
        if os.path.exists(f):
            os.remove(f)
    from plexus.paths import graphs_data_path
    out = graphs_data_path(group, run)
    tj, mv = os.path.join(out, "trajectory.npz"), os.path.join(out, "movie.mp4")
    t_sub = time.time()
    cmd = f"python -u Plexus_Main.py -o generate {group}/{run} --device cuda:0 --force"
    jid = exp._submit("l4", cmd, jd, run, wall=240 if a.full else 30, job_prefix="")
    if not jid:
        return 3
    print(f"[smoke] {group}/{run}: job {jid}, {'full length' if a.full else str(a.frames) + ' frames'}; "
          f"spec {os.path.relpath(dst, ROOT)}", flush=True)
    if a.no_wait:
        print(f"[smoke] not waiting; health later: PYTHONPATH=src:tools python tools/exp_land.py --health {group}/{run}")
        return 0
    timeout = a.timeout or (14400 if a.full else 1800)
    while time.time() - t_sub < timeout:
        L = _lsf(co)
        done = (os.path.exists(tj) and os.path.getmtime(tj) > t_sub and os.path.exists(mv)
                and os.path.getmtime(mv) > t_sub)
        if done or "exit" in L or "term" in L or "run_s" in L:
            break
        time.sleep(10)
    time.sleep(2)
    L = _lsf(co)
    el = int(time.time() - t_sub)
    if "exit" in L or "term" in L:
        print(f"[smoke] DIED after {el} s: exit code {L.get('exit', '?')} {L.get('term', '')} {L.get('why', '')}")
        print(f"        log: {os.path.relpath(co, ROOT)} and {os.path.relpath(co[:-4] + '.err', ROOT)}")
        return 1
    if not (os.path.exists(tj) and os.path.getmtime(tj) > t_sub):
        print(f"[smoke] no trajectory after {el} s (timeout {timeout} s); log: {os.path.relpath(co, ROOT)}")
        return 2
    import exp_land
    run_s = L.get("run_s")
    print(f"[smoke] landed after {el} s wall" + (f" (LSF run time {run_s} s, so ~{max(el - int(run_s), 0)} s queue)" if run_s else ""))
    print("\n".join(exp_land.health_lines(exp_land.health(f"{group}/{run}"))))
    if a.exp is not None:
        return 4 if check_keys(a.exp, f"{group}/{run}") else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
