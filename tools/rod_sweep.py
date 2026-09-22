#!/usr/bin/env python
"""Vary one number of the rod rig across several runs, and report what each did.

    python tools/rod_sweep.py cil_s1_onerod bend 500 5e3 5e4 5e5
    python tools/rod_sweep.py cil_s1_onerod moment 0.02 0.1 0.5

THE RIG RUNS IN FOUR SECONDS, which is the point of it. The MPM larva takes twenty minutes a run
and every question about the cilium had to be asked through four hundred thousand water particles
and an animal; here one filament on a bench answers the same question before lunch, and the
answers transfer because the filament is the same object.

THE COLUMN THAT MATTERS IS `tip/base`. A rod whose tip angle equals its base angle is a rigid
stick pivoting at the anchor; one whose tip is dead is a floppy string being wiggled at one end,
and the wiggle dies within a segment or two. A cilium is neither: the tip follows most of the way
and arrives LATE, because drag curls the rod as it sweeps. So the pair to read is `tip/base`
together with `lag` -- a large ratio with no lag is a stick, and a lag with no ratio is a string.
"""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))
OUT = os.path.join(REPO, "config", "platynereis")

# where each knob lives: (which list, the op name, the key)
KNOBS = {
    "bend":    ("op", "rod_bend", "k"),
    "bdamp":   ("op", "rod_bend", "zeta"),
    "stretch": ("op", "rod_stretch", "k"),
    "sdamp":   ("op", "rod_stretch", "zeta"),
    "moment":  ("op", "rod_base_moment", "moment"),
    "omega":   ("op", "rod_base_moment", "omega"),
    "duty":    ("op", "rod_base_moment", "duty"),
    "drag":    ("op", "rod_drag", "zeta_par"),
    "ratio":   ("op", "rod_drag", "ratio"),
    "pin":     ("op", "rod_pin", "omega_n"),
    "clamp":   ("op", "rod_pin", "clamp"),
    "length":  ("seed", "rod_seed", "length"),
    "nodes":   ("set", "rod_node", "n"),
}


def variant(base, knob, v, name):
    d = copy.deepcopy(base)
    d["general"]["name"] = name
    where, target, key = KNOBS[knob]
    if where == "op":
        for o in d["operators"]:
            if o.get("op") == target:
                o[key] = v
    elif where == "seed":
        for o in d["seed"]:
            if o.get("op") == target:
                o[key] = v
    else:
        d["sets"][target][key] = int(v)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("knob", choices=sorted(KNOBS))
    ap.add_argument("values", nargs="+", type=float)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()

    from rod_probe import load, angles
    import numpy as np
    import math

    base = yaml.safe_load(open(os.path.join(OUT, a.base + ".yaml")))
    rows = []
    for v in a.values:
        nm = f"rs_{a.knob}_{('%g' % v).replace('.', 'p').replace('-', 'm').replace('+', '')}"
        with open(os.path.join(OUT, nm + ".yaml"), "w") as fh:
            fh.write(f"# rod sweep: {a.knob} = {v:g} on {a.base}. "
                     f"Written by tools/rod_sweep.py\n")
            yaml.safe_dump(variant(base, a.knob, v, nm), fh, sort_keys=False, width=100)
        r = subprocess.run([sys.executable, os.path.join(REPO, "Plexus_Main.py"),
                            "-o", "generate", f"platynereis/{nm}",
                            "--device", a.device, "--no-describe", "--no-viz"],
                           capture_output=True, text=True, cwd=REPO, timeout=3600)
        try:
            tr, sp = load(nm)
            d = yaml.safe_load(open(sp))
            dt = float(d["general"]["dt"])
            rs = next(o for o in d["seed"] if o["op"] == "rod_seed")
            bm = next(o for o in d["operators"] if o["op"] == "rod_base_moment")
            d0 = np.asarray(rs.get("direction", [0, 0, 1.0]), float); d0 /= np.linalg.norm(d0)
            nn = np.asarray(rs.get("beat_axis", [0, 1.0, 0]), float)
            nn = nn - (nn @ d0) * d0; nn /= np.linalg.norm(nn)
            P = np.asarray(tr["rod_node__pos"]); T = P.shape[0]
            A = angles(P, d0, nn)
            s = slice(int(T / 3), None)
            ab, at = float(np.ptp(A[s, 1])), float(np.ptp(A[s, -1]))
            # THE MEAN THE BEAT IS CENTRED ON, which is a different question from its amplitude
            # and the one the base clamp exists to answer. An unclamped rod beats perfectly well
            # about a rest direction that is itself windmilling round: measured, a clean 1 Hz
            # stroke of 29 degrees whose CENTRE had drifted to +32. Amplitude alone calls that a
            # beat.
            mean_t = float(A[s, -1].mean())
            y1, y2 = A[s, 1] - A[s, 1].mean(), A[s, -1] - A[s, -1].mean()
            per = 2 * math.pi / float(bm.get("omega", 1.0))
            lag = float("nan")
            if np.ptp(y1) > 1e-9 and np.ptp(y2) > 1e-9:
                c = np.correlate(y2, y1, "full")
                lag = 360.0 * ((np.argmax(c) - (y1.size - 1)) * dt) / per
            L = np.linalg.norm(P[:, -1, :] - P[:, 0, :], axis=1)
            rows.append((v, ab, at, mean_t, at / max(ab, 1e-12), lag,
                         100 * float(L[-1]) / float(rs["length"]),
                         float(np.linalg.norm(P[:, 0, :] - P[0, 0, :], axis=1).max())))
        except Exception as e:                                       # noqa: BLE001
            print(f"  {a.knob}={v:g} FAILED {type(e).__name__}: {str(e)[:120]}")
            print(f"    {(r.stderr or r.stdout)[-300:]}")

    print(f"\n  {a.knob:>10s} {'base deg':>9s} {'tip deg':>9s} {'tip mean':>9s} "
          f"{'tip/base':>9s} {'lag deg':>8s} {'length %':>9s} {'base drift':>11s}")
    for v, ab, at, mt, r_, lg, ln, bd in rows:
        print(f"  {v:10g} {ab:9.2f} {at:9.2f} {mt:9.2f} {r_:9.3f} {lg:8.1f} {ln:9.1f} {bd:11.2e}")
    print(f"\n  tip/base: 1 is a rigid stick pivoting at the anchor, 0 a floppy string whose "
          f"wiggle dies in a\n  segment or two. A cilium is in between, WITH a lag -- drag curls "
          f"the rod as it sweeps, so the tip\n  arrives late, and that lag is the time asymmetry "
          f"a symmetric command cannot supply (Machin 1958).\n  length % far from 100 means the "
          f"rod is stretching or curling; base drift should stay tiny.")


if __name__ == "__main__":
    main()
