#!/usr/bin/env python
"""Sweep one knob of the paddle across several short runs, and report what each did.

    python tools/platynereis_sweep.py torque 0.05 0.5 5.0
    python tools/platynereis_sweep.py youngs 3000 30000 300000 --frames 150

WHY A SWEEP AND NOT A GUESS. The paddle has four numbers that decide whether it beats -- the
torque, its stiffness, its length and its width -- and they trade against each other through the
fluid, which is not something to reason about from a formula when a run takes three minutes.
Each variant is a SPEC on disk, run through the pipeline, measured by the same tools as every
other rung, and recorded in `builder/` beside the rest.

THE NUMBER THAT MATTERS IS THE TIP-TO-ROOT RATIO. A paddle that beats has a tip moving several
times further than its own root; a paddle being carried by the body has a ratio near one, and
one being dragged by the water has a ratio below it. The first run of R14 came out at 0.8 -- the
tip moving LESS than the root -- which is not a weak beat, it is no beat at all.

And every variant reports the invariants too, because a variant that beats beautifully while
creating momentum has not found a better paddle, it has found a worse bug.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))
OUT = os.path.join(REPO, "config", "platynereis")
UM = 195.9

# where each knob lives in the spec, so the sweep edits one place and not a scattered set
KNOBS = {
    "torque":  ("op", "cilium_pose_map", "sweep_deg"),
    "omega":   ("op", "cilium_pose_map", "omega"),
    "duty":    ("op", "cilium_pose_map", "duty"),
    "youngs":  ("type", "cilium_point", "youngs"),
    "radius":  ("set", "cilium_point", "radius_um"),
    "n_react": ("op", "cilium_torque", "n_react"),
}


def variant(base: dict, knob: str, value: float, name: str) -> dict:
    d = json.loads(json.dumps(base))
    d["general"]["name"] = name
    kind, target, key = KNOBS[knob]
    if kind == "op":
        for o in d["operators"]:
            if o.get("op") == target:
                o[key] = value
    elif kind == "type":
        for t in d["sets"][target]["types"].values():
            t[key] = value
    else:                                    # a set-level length, given in micrometres
        d["sets"][target]["radius"] = round(value / UM, 6)
        d["plotting"]["dot_radius"]["cilium_shaft"] = round(value / UM, 6)
    return d


def measure(name: str, dt: float):
    from platynereis_momentum import load, masses, measure as mom
    tr, sp = load(name)
    st = mom(tr, masses(sp), dt)
    P = np.asarray(tr["cilium_point__pos"])
    par = np.asarray(tr["cilium_point__parent"])
    n = int(par.max()) + 1
    per = P.shape[1] // n
    tip, root = P[:, per - 1::per, :], P[:, 0::per, :]
    t0 = int(0.25 * P.shape[0])
    a_t = float(np.median(np.linalg.norm(tip[t0:].max(0) - tip[t0:].min(0), axis=1)) * UM)
    a_r = float(np.median(np.linalg.norm(root[t0:].max(0) - root[t0:].min(0), axis=1)) * UM)
    W = np.asarray(tr["water_particle__pos"])
    occ = np.asarray(tr["water_particle__occ"])[0].astype(bool)
    v = np.linalg.norm(np.diff(W[:, occ], axis=0), axis=2) * UM / dt
    return {
        "tip_um": a_t, "root_um": a_r, "ratio": a_t / max(a_r, 1e-9),
        "water_um_s": float(np.median(v.mean(0))),
        "water_p95": float(np.percentile(v.mean(0), 95)),
        "p_peak": float(np.linalg.norm(st["total"], axis=1).max()),
        "body_pct": float(100 * st["radius"][-1] / st["radius"][0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("knob", choices=sorted(KNOBS))
    ap.add_argument("values", nargs="+", type=float)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--base", default="plat_r14_paddle")
    ap.add_argument("--dt", type=float, default=0.05)
    a = ap.parse_args()

    base = yaml.safe_load(open(os.path.join(OUT, a.base + ".yaml")))
    base["general"]["n_frames"] = a.frames
    base["general"]["record_cap"] = a.frames + 1
    base["plotting"]["max_frames"] = a.frames

    rows = []
    for v in a.values:
        nm = f"sw_{a.knob}_{str(v).replace('.', 'p').replace('-', 'm')}"
        p = os.path.join(OUT, nm + ".yaml")
        with open(p, "w") as fh:
            fh.write(f"# a sweep variant: {a.knob} = {v}. Written by tools/platynereis_sweep.py\n")
            yaml.safe_dump(variant(base, a.knob, v, nm), fh, sort_keys=False, width=110)
        print(f"  running {nm} ({a.knob}={v}) ...", flush=True)
        r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "gui_drive.py"),
                            "opencycle", "--spec", p, "--device", "cuda:0", "--no-caption",
                            "--azim", "20", "--elev", "8", "--zoom", "1.3", "--roll", "180",
                            "--why", f"sweep: {a.knob} = {v}, {a.frames} frames. The number that "
                                     f"matters is the tip-to-root ratio -- a paddle that beats "
                                     f"has a tip moving several times further than its own root, "
                                     f"and one being carried has a ratio near 1."],
                           capture_output=True, text=True, timeout=5400, cwd=REPO)
        try:
            rows.append((v, measure(nm, a.dt)))
        except Exception as e:                                       # noqa: BLE001
            print(f"    FAILED: {type(e).__name__}: {str(e)[:160]}")
            print(f"    {(r.stderr or r.stdout)[-300:]}")

    print(f"\n  {a.knob:>10s} {'tip um':>8s} {'root um':>8s} {'tip/root':>9s} "
          f"{'water um/s':>11s} {'p peak':>8s} {'body %':>7s}")
    for v, m in rows:
        print(f"  {v:10g} {m['tip_um']:8.2f} {m['root_um']:8.2f} {m['ratio']:9.2f} "
              f"{m['water_um_s']:11.3f} {m['p_peak']:8.3f} {m['body_pct']:7.0f}")
    if rows:
        best = max(rows, key=lambda kv: kv[1]["ratio"])
        print(f"\n  best tip/root: {a.knob} = {best[0]:g} at {best[1]['ratio']:.2f}x, "
              f"water {best[1]['water_um_s']:.3f} um/s, body {best[1]['body_pct']:.0f}%")


if __name__ == "__main__":
    main()
