#!/usr/bin/env python
"""p1v_impostor -- ADVERSARIAL VERIFICATION of probe C.

The claim under test has two halves. The second half is:

    "in EVERY regime tested a single uniform modulus imitates the whole planted 40-216 field to
     0.73-2.66 steps"

The 0.73 comes from the SPEC (13-point sweep, p1c_instruments.json) and the 2.66 from the wide
stimulus (p1c_zone.json). The regimes the FIRST half of the claim rests on -- drag 3, drive
amplitude 80, pulse always on -- were only ever swept on an 8-point geomspace(20, 800) grid, whose
neighbouring points differ by a factor 1.66. The minimum of the STAT column there is 6.61 / 15.06 /
15.33 steps, not 0.73-2.66. That could be nothing but grid coarseness, because peak_excursion is
MONOTONE in E in two of those regimes, so an E exists that matches the planted amplitude exactly.

This refines the grid until the amplitude crossing is resolved, in each regime, and reports what
the best uniform impostor actually costs and which instrument limits it.

Same reading surface (10x10 probes, margin 20), same floors, same reference convention (that
configuration's own theta_true rollout) as probe C.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import p1c_lib as L                                                        # noqa: E402
import crash_test as CT                                                    # noqa: E402

T0 = time.time()


def log(s=""):
    print(f"[{time.time() - T0:7.1f}s] {s}" if s else "")
    sys.stdout.flush()


INS = L.INSTRUMENTS

CONFIGS = {
    "baseline": ({}, (40.0, 300.0)),
    "drag 3": ({"drag_k": 3.0}, (60.0, 260.0)),
    "amplitude 80": ({"amplitude": 80.0}, (60.0, 260.0)),
    "duration 150 (always on)": ({"duration": 150.0}, (60.0, 260.0)),
}


def scan(name, kw, lo, hi, n, device, seed=None):
    per = kw.get("period", 150.0)
    args = L.default_args(device=device, warmup=int(per) + 30, window=int(per))
    if seed is not None:
        orig = CT.plant_and_warm
        CT.plant_and_warm = lambda a, l, _o=orig, _s=seed: _o(a, l, seed=_s)
        try:
            rig = L.Rig(args, **kw)
        finally:
            CT.plant_and_warm = orig
    else:
        rig = L.Rig(args, **kw)
    ref = rig.roll(rig.theta_true)
    ref_amp = L.amp_reading(ref)
    log(f"\n  {name}  seed={seed or 2026}  planted E [{float(rig.E_true.min()):.1f}, "
        f"{float(rig.E_true.max()):.1f}] median {float(rig.E_true.median()):.1f}; "
        f"theta_true amplitude {ref_amp:.6g}, path {L.path_reading(ref):.6g}")
    log(f"  {'uniform E':>10s}" + "".join(f"{m[:11]:>12s}" for m in INS)
        + f"{'STAT':>9s}{'amp':>12s}{'amp/ref':>9s}")
    rows, best = [], None
    for E in np.geomspace(lo, hi, n):
        sim = rig.roll(rig.theta(E=float(E)))
        r = L.steps_row(sim, ref)
        r["E"] = float(E)
        r["amp_reading"] = L.amp_reading(sim)
        r.pop("_values", None)
        rows.append(r)
        log(f"  {E:10.1f}" + "".join(f"{r[m]:12.2f}" for m in INS)
            + f"{r['STAT']:9.2f}{r['amp_reading']:12.6f}{r['amp_reading'] / ref_amp:9.3f}")
        if best is None or r["STAT"] < best["STAT"]:
            best = r
    lim = max((m for m in INS if isinstance(best[m], float)), key=lambda m: best[m])
    log(f"    BEST uniform impostor: E = {best['E']:.1f} at {best['STAT']:.2f} steps "
        f"(limiting {lim})")
    rig.free()
    return {"spec": kw, "ref_amp": ref_amp, "rows": rows, "best": best,
            "best_limiting": lim, "seed": seed or 2026}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="p1v_impostor.json")
    a = ap.parse_args()

    out = {}
    for name in a.configs:
        kw, (lo, hi) = CONFIGS[name]
        out[name] = scan(name, kw, lo, hi, a.n, a.device, seed=a.seed)
    log(f"\n{'=' * 96}\n  BEST UNIFORM-E IMPOSTOR FOR THE PLANTED FIELD, per regime\n{'=' * 96}")
    log(f"  {'config':<28s}{'best E':>9s}{'STAT':>9s}{'limiting':>20s}{'8-pt grid min':>15s}")
    for n, v in out.items():
        log(f"  {n:<28s}{v['best']['E']:9.1f}{v['best']['STAT']:9.2f}{v['best_limiting']:>20s}")
    json.dump(out, open(os.path.join(HERE, a.out), "w"), indent=1, default=float)
    log(f"  -> {a.out}")
