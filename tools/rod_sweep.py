#!/usr/bin/env python
"""Vary one number of the rod rig across several runs, and report what each did.

    python tools/rod_sweep.py cil_s1_onerod bend=500,5e3,5e4,5e5
    python tools/rod_sweep.py cil_s3_onlywater moment=10,50,200,800 bend=5e4,2e5,5e5

ANY NUMBER OF KNOBS, as `name=v1,v2,...` -- the runs are their cartesian product. One knob is the
special case, so there is no separate one-knob tool and no separate two-knob tool. The question
that forced it was a two-parameter one: is there a regime where the beat is big enough to see and
the fluid's inertia does NOT tip the rod over, and does stiffening the rod open it?

THE RIG RUNS IN FOUR SECONDS, which is the point of it. The MPM larva takes twenty minutes a run
and every question about the cilium had to be asked through four hundred thousand water particles
and an animal; here one filament on a bench answers the same question before lunch, and the
answers transfer because the filament is the same object.

THE COLUMN THAT MATTERS DEPENDS ON THE QUESTION, so both are here. `tip/base` says whether the
rod is beating or being carried. `tip mean` says whether it is beating ABOUT THE RIGHT PLACE -- a
rod can sweep 50 degrees perfectly well about a rest direction that has drifted 70 degrees off
vertical, starting upright and finishing lying down, and amplitude alone calls that a beat. When
the scene has water, `flow` is the time-averaged fluid velocity along the sweep direction: it is
zero for a symmetric stroke in a Stokes fluid, and non-zero means inertial streaming, which is
what tips the rod.

THE OLD COLUMN NOTE, still true: `tip/base`. A rod whose tip angle equals its base angle is a rigid
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
    "bend":    ("op", "rod_elastic", "k_bend"),
    "bdamp":   ("op", "rod_elastic", "zeta_bend"),
    "stretch": ("op", "rod_elastic", "k_stretch"),
    "sdamp":   ("op", "rod_elastic", "zeta_stretch"),
    "moment":  ("op", "rod_base", "moment"),
    "omega":   ("op", "rod_base", "omega"),
    "duty":    ("op", "rod_base", "duty"),
    "drag":    ("op", "drag", "k"),
    "ratio":   ("op", "drag", "ratio"),
    "pin":     ("op", "rod_base", "omega_n"),
    "clamp":   ("op", "rod_base", "clamp"),
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
    ap.add_argument("knobs", nargs="+", help="name=v1,v2,... ; several make a cartesian product")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--frames", type=int, default=None)
    a = ap.parse_args()

    import itertools
    axes = []
    for spec_ in a.knobs:
        if "=" not in spec_:
            raise SystemExit(f"knobs are `name=v1,v2,...`; got {spec_!r}. "
                             f"Known names: {', '.join(sorted(KNOBS))}")
        k, vs = spec_.split("=", 1)
        if k not in KNOBS:
            raise SystemExit(f"unknown knob {k!r}; known: {', '.join(sorted(KNOBS))}")
        axes.append((k, [float(v) for v in vs.split(",")]))

    from rod_probe import load, angles
    import numpy as np
    import math

    base = yaml.safe_load(open(os.path.join(OUT, a.base + ".yaml")))
    if a.frames:
        base["general"]["n_frames"] = a.frames
        base["general"]["record_cap"] = min(base["general"].get("record_cap", 10 ** 9),
                                            a.frames // 4 + 1)
        base["plotting"]["max_frames"] = a.frames // 4
    rows = []
    for combo in itertools.product(*[vs for _, vs in axes]):
        tag = "_".join(f"{k}{('%g' % v).replace('.', 'p').replace('-', 'm').replace('+', '')}"
                       for (k, _), v in zip(axes, combo))
        nm = f"rs_{tag}"
        d_ = base
        for (k, _), v in zip(axes, combo):
            d_ = variant(d_, k, v, nm)
        with open(os.path.join(OUT, nm + ".yaml"), "w") as fh:
            fh.write(f"# rod sweep on {a.base}: "
                     + ", ".join(f"{k} = {v:g}" for (k, _), v in zip(axes, combo))
                     + ". Written by tools/rod_sweep.py\n")
            yaml.safe_dump(d_, fh, sort_keys=False, width=100)
        v = combo
        # EVERY RUN GOES THROUGH THE WATCHER. This used Plexus_Main directly, on the argument
        # that a sweep variant produces only a row of numbers. That argument is wrong: a sweep is
        # twelve SCENES, and twelve scenes that never reach the record are twelve things Cedric
        # cannot see while they run. `gui_drive opencycle` writes builder/{spec,png,why,mp4}/NNNN
        # and captions the movie, so a sweep is in the record BY CONSTRUCTION rather than by my
        # remembering to put it there.
        why = (f"rod sweep on {a.base}: "
               + ", ".join(f"{k} = {v:g}" for (k, _), v in zip(axes, combo))
               + ". Is there a regime where the beat is big enough to see AND the fluid's inertia "
                 "does not tip the rod over? `tip mean` is the angle the beat is centred on, 0 "
                 "being upright, and `flow` is the time-averaged fluid velocity along the sweep "
                 "-- zero for a symmetric stroke in a Stokes fluid, non-zero only from inertial "
                 "streaming, which is what tips it.")
        r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "gui_drive.py"),
                            "opencycle", "--spec", os.path.join(OUT, nm + ".yaml"),
                            "--device", a.device, "--azim", "35", "--elev", "12", "--zoom", "1.3",
                            "--why", why],
                           capture_output=True, text=True, cwd=REPO, timeout=7200)
        try:
            tr, sp = load(nm)
            d = yaml.safe_load(open(sp))
            dt = float(d["general"]["dt"])
            rs = next(o for o in d["seed"] if o["op"] == "rod_seed")
            bm = next(o for o in d["operators"] if o["op"] == "rod_base")
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
            # THE TIME-AVERAGED FLOW ALONG THE SWEEP, when there is water. Zero for a symmetric
            # stroke in a Stokes fluid; non-zero IS inertial streaming, and it is what tips the
            # rod, so it belongs beside the drift rather than in a separate tool.
            flow = float("nan")
            if "water_particle__pos" in tr.files:
                W = np.asarray(tr["water_particle__pos"])
                oc = np.asarray(tr["water_particle__occ"])[0].astype(bool)
                stride = max(1, d["general"]["n_frames"] // max(T - 1, 1))
                c0 = P[0].mean(0)
                near = np.linalg.norm(W[0][oc] - c0, axis=1) < float(rs["length"])
                mv = (np.diff(W[:, oc][:, near], axis=0) / (dt * stride)).mean((0, 1))
                flow = float(mv @ np.cross(nn, d0)) * float(
                    (d["general"].get("units") or {}).get("length_um", 1.0))
            rows.append((combo, ab, at, mean_t, at / max(ab, 1e-12), flow,
                         100 * float(L[-1]) / float(rs["length"])))
        except Exception as e:                                       # noqa: BLE001
            print(f"  {tag} FAILED {type(e).__name__}: {str(e)[:120]}")
            print(f"    {(r.stderr or r.stdout)[-300:]}")

    hdr = "  " + " ".join(f"{k:>9s}" for k, _ in axes)
    print(f"\n{hdr} {'base deg':>9s} {'tip deg':>9s} {'tip mean':>9s} "
          f"{'tip/base':>9s} {'flow um/s':>10s} {'length %':>9s}")
    for combo, ab, at, mt, r_, fl, ln in rows:
        print("  " + " ".join(f"{v:9g}" for v in combo)
              + f" {ab:9.2f} {at:9.2f} {mt:9.2f} {r_:9.3f} {fl:10.4f} {ln:9.1f}")
    print(f"\n  tip/base: 1 is a rigid stick pivoting at the anchor, 0 a floppy string whose "
          f"wiggle dies in a\n  segment or two. A cilium is in between, WITH a lag -- drag curls "
          f"the rod as it sweeps, so the tip\n  arrives late, and that lag is the time asymmetry "
          f"a symmetric command cannot supply (Machin 1958).\n  length % far from 100 means the "
          f"rod is stretching or curling; base drift should stay tiny.")


if __name__ == "__main__":
    main()
