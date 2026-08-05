#!/usr/bin/env python
"""gate_geom -- how much of the sphere must be suppressed to make an OVOID, at 90 seconds a point.

    python gate_geom.py --device cuda:0

THE DECOMPOSITION, which is the whole point. "Does a dense matrix at the poles make an oblate
spheroid?" is two independent questions welded together:

    A  given a polar pressure pattern, does gating the cell cycle by it produce an oblate tissue,
       and over what range of suppressed solid angle?
    B  does a matrix with dense polar caps actually produce a pressure pattern in that class?

(B) costs a 15-minute MPM run per geometry. (A) costs a 90-second tissue pass, because the gate reads
a pressure MAP off disk and does not care whether a matrix or this script wrote it. So (A) is settled
first, on synthetic maps, and only the winning geometry is spent on (B). Answering them together would
mean 15 minutes per point and no way to tell which link failed when the answer came back round.

THE GEOMETRY. Two polar caps of half-angle theta cover a fraction (1 - cos theta) of 4.pi:

    theta       40     55     60     70     80
    fraction   0.23   0.43   0.50   0.66   0.83

Both ends must fail, and that is the check that the middle result means anything. Too narrow and only a
patch lags -- a dimple, not an ovoid. Too wide and nearly every direction is suppressed, so the tissue
grows slower ISOTROPICALLY and the aspect ratio returns to 1. A sweep that did not bracket the optimum
would not have found one.

THE SYNTHETIC MAP IS A STEP WITH A SOFT EDGE, not the matrix's own pattern: pressure `hi` inside the
cone, `lo` outside, blended over 10 degrees. It is deliberately the CLEANEST version of the pattern --
if the gate cannot make a shape from a clean polar step it will not make one from a noisy measured
field, and that is worth knowing in 90 seconds rather than 15 minutes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, os.path.join(ROOT, "src"), os.path.join(ROOT, "prototype", "Tyssue"),
          os.path.join(ROOT, "discovery_okuda")):
    if p not in sys.path:
        sys.path.insert(0, p)

LOG = os.path.join(ROOT, "log", "okuda_ECM")
SYN = os.path.join(LOG, "_synthetic")
N_THETA, N_PHI = 32, 64


def synth_map(cone_deg, frames=403, hi=1.0, lo=0.05, edge_deg=10.0, ramp=True):
    """A polar step in pressure, ramping in over the run the way a real contact pressure does.

    The RAMP matters. A map that is at full pressure from frame 0 gates the cell cycle before the
    tissue has touched anything, which is not the experiment -- the measured maps are identically zero
    until contact around frame 30 and climb from there. Without it the answer would be "growth was
    suppressed for 400 frames", which is a different question with a more obvious answer.
    """
    th = (np.arange(N_THETA) + 0.5) / N_THETA * np.pi
    poldist = np.minimum(th, np.pi - th)                        # angle to the nearest pole
    w = 0.5 * (1.0 - np.tanh((np.rad2deg(poldist) - cone_deg) / max(edge_deg, 1e-6) * 2.0))
    row = lo + (hi - lo) * w                                    # [N_THETA]
    M = np.repeat(row[:, None], N_PHI, axis=1)[None]            # [1, nth, nph]
    if ramp:
        t = np.arange(frames) / max(frames - 1, 1)
        g = np.clip((t - 0.08) / 0.92, 0.0, 1.0) ** 1.5         # zero until ~frame 32, then climbs
    else:
        g = np.ones(frames)
    return (M * g[:, None, None]).astype(np.float32)


def frac_of_4pi(cone_deg):
    return 1.0 - math.cos(math.radians(cone_deg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cones", default="0,40,55,70,85")
    ap.add_argument("--p-half", type=float, default=0.10)
    ap.add_argument("--floor", type=float, default=0.20)
    ap.add_argument("--frames", type=int, default=401)
    a = ap.parse_args()

    import tissue as TIS
    os.makedirs(SYN, exist_ok=True)
    rows = []
    for c in [float(x) for x in a.cones.split(",")]:
        # cone 0 = the CONTROL: a uniform map at the same mean pressure, so "suppressed everywhere by
        # the same average amount" is on the table as an explanation and can be ruled out by the shape
        # rather than by argument.
        if c <= 0:
            P = synth_map(90.0, frames=a.frames + 2, hi=0.5, lo=0.5)
            tag, fr = "uniform", 1.0
        else:
            P = synth_map(c, frames=a.frames + 2)
            tag, fr = f"cone{c:g}", frac_of_4pi(c)
        path = os.path.join(SYN, f"load_{tag}.npz")
        np.savez_compressed(path, pmap=P)
        npz = TIS.load_or_build(frames=a.frames, device=a.device, buffer_x=4, gate_npz=path,
                               gate_p_half=a.p_half, gate_floor=a.floor,
                               tag_extra=f"_syn{tag}f{a.floor:g}".replace(".", "p"))
        z = np.load(npz)
        x, y, zz = np.asarray(z["r_xyz"])[-1]
        eq = math.sqrt(max(x * x + y * y, 1e-12) / 2.0)          # RMS in-plane semi-axis
        rows.append({"cone_deg": c, "tag": tag, "suppressed_frac_4pi": fr,
                     "r_x": float(x), "r_y": float(y), "r_z": float(zz),
                     "oblateness_eq_over_z": float(eq / max(zz, 1e-9)),
                     "n_cells": int(z["n_cells"][-1]),
                     "r_apical": float(z["r_apical"][-1])})
        print(f"[gate_geom] {tag:9} suppressed {fr:5.2f} of 4pi -> x/y/z {x:5.2f}/{y:5.2f}/{zz:5.2f}"
              f"   eq/z {eq / max(zz, 1e-9):.3f}   {int(z['n_cells'][-1])} cells", flush=True)
        json.dump(rows, open(os.path.join(LOG, "gate_geom.json"), "w"), indent=1)

    print("\n[gate_geom] suppressed fraction -> oblateness (eq/z):")
    for r in rows:
        print(f"  {r['tag']:9} {r['suppressed_frac_4pi']:5.2f}  eq/z {r['oblateness_eq_over_z']:.3f}"
              f"  cells {r['n_cells']}")
    best = max((r for r in rows if r["tag"] != "uniform"),
               key=lambda r: r["oblateness_eq_over_z"], default=None)
    if best:
        print(f"[gate_geom] most oblate: {best['tag']} at {best['suppressed_frac_4pi']:.2f} of 4pi, "
              f"eq/z {best['oblateness_eq_over_z']:.3f}")


if __name__ == "__main__":
    main()
