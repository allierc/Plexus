#!/usr/bin/env python
"""How far out is the bud, and does it have a neck -- one number each, decided before the runs.

    python budding_metric.py 08a_hole_rot 08b_bud_mech ...

WHAT A BUD IS, AS A MEASUREMENT. "A protrusion emerged" is not a claim until it is a number, and the
obvious number -- the maximum radius -- is the wrong one: a spheroid that grew 4% everywhere has a
larger maximum radius than one with a small bud, and the two look nothing alike. Two are used here and
they answer different halves of the question:

    bud_index   how far the tip reaches ALONG THE HOLE'S AXIS, beyond the body:
                    h_max / R_med  -  1
                where h_max is the largest projection of any epithelial vertex onto the cap
                direction and R_med is the median radius of the whole shell. Zero for a sphere of any
                size, because both scale together -- so it measures SHAPE and not growth.

    neck_ratio  whether it is a bud or a bulge:
                    min(r_perp over the waist band)  /  max(r_perp over the lobe beyond it)
                where r_perp is the p90 cross-sectional radius perpendicular to the axis, binned
                along it. A hemispherical bulge has no waist and scores >= 1; a bud pinched at its
                base scores below 1, and the lower the tighter. Reported only where a lobe exists at
                all (bud_index > 0.02), because the waist of a sphere is not a number about budding.

THE AXIS IS RECOMPUTED, NOT STORED. It is a fixed function of the camera basis and the two rotation
angles the run was launched with, so it is derived here from the spec's own `cap` entry rather than
trusted to a field that could have been written by a different version of the rig.

AND THE CONTROL MATTERS MORE THAN THE NUMBER. Every one of these is also computed on the OPPOSITE
pole, along -d, where there is no hole and nothing should be growing. A sweep that raises `bud_index`
at both ends has changed the tissue's overall shape (or its noise floor) and has not made a bud, and
that is the failure mode this exists to catch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LOG = os.path.abspath(os.path.join(HERE, "..", "..", "log", "okuda_ECM"))

N_BIN = 48                         # bins along the axis; ~2 cells wide at the end of a run
LOBE_MIN = 0.02                    # a neck is only a number once there is something to have a neck


def cap_dir(off_theta=45.0, off_phi=45.0):
    from ecm_render import screen_basis
    import test_06_breach as BR
    _d, right, up = screen_basis(18.0, 30.0)
    a, b = np.radians(off_theta), np.radians(off_phi)
    v = (np.cos(a) * np.asarray(BR.CAM_DIR)
         + np.sin(a) * (np.cos(b) * np.asarray(right) + np.sin(b) * np.asarray(up)))
    return v / np.linalg.norm(v)


def profile(x, c, d):
    """(bud_index, neck_ratio, h_max/R_med, the binned cross-section) along +d."""
    p = x - c
    r = np.linalg.norm(p, axis=1)
    R = float(np.median(r))
    h = p @ d
    perp = np.linalg.norm(p - h[:, None] * d[None, :], axis=1)
    h_max = float(h.max())
    bud = h_max / max(R, 1e-30) - 1.0
    # the cross-section, binned along the axis, from the equator outward
    lo, hi = 0.0, h_max
    if hi <= lo:
        return bud, float("nan"), R, h_max, None
    e = np.linspace(lo, hi, N_BIN + 1)
    k = np.clip(np.digitize(h, e) - 1, 0, N_BIN - 1)
    rp = np.full(N_BIN, np.nan)
    for j in range(N_BIN):
        m = (k == j) & (h >= lo)
        if m.sum() >= 8:
            rp[j] = np.percentile(perp[m], 90)
    if bud <= LOBE_MIN or np.all(~np.isfinite(rp)):
        return bud, float("nan"), R, h_max, rp
    # the waist is looked for OUTSIDE the body: beyond the sphere's own shoulder at h = R
    out = np.where(np.isfinite(rp) & (e[:-1] >= 0.75 * R))[0]
    if out.size < 4:
        return bud, float("nan"), R, h_max, rp
    i_min = out[np.nanargmin(rp[out])]
    beyond = out[out > i_min]
    if beyond.size < 2 or not np.isfinite(rp[beyond]).any():
        return bud, float("nan"), R, h_max, rp
    neck = float(rp[i_min] / max(np.nanmax(rp[beyond]), 1e-30))
    return bud, neck, R, h_max, rp


def go(run, off_theta=45.0, off_phi=45.0):
    d_ = os.path.join(LOG, run)
    z = np.load(os.path.join(d_, "bm_frames.npz"))
    n = int(z["n_kept"])
    sp = yaml.safe_load(open(os.path.join(d_, "spec.yaml")))
    cap = str((sp.get("run") or {}).get("cap", ""))
    if "theta" in cap:                      # "theta 45.0 deg, phi 45.0 deg off the camera axis"
        tok = [float(t) for t in cap.replace(",", " ").split() if t.replace(".", "").isdigit()]
        if len(tok) >= 2:
            off_theta, off_phi = tok[0], tok[1]
    d = cap_dir(off_theta, off_phi)
    S = {k: [] for k in ("t", "bud", "neck", "R", "h_max", "bud_far", "neck_far")}
    for i in range(n):
        x = np.asarray(z[f"e{i}"], float)
        c = x.mean(0)
        b, nk, R, hm, _ = profile(x, c, d)
        bf, nf, _, _, _ = profile(x, c, -d)          # the control: the pole with no hole
        S["t"].append(int(z[f"t{i}"])); S["bud"].append(b); S["neck"].append(nk)
        S["R"].append(R); S["h_max"].append(hm)
        S["bud_far"].append(bf); S["neck_far"].append(nf)
    a = {k: np.asarray(v, float) for k, v in S.items()}
    out = dict(run=run, frames=n, axis=dict(off_theta=off_theta, off_phi=off_phi, d=d.tolist()),
               bud_index=dict(final=float(a["bud"][-1]), max=float(np.nanmax(a["bud"])),
                              at_the_far_pole=float(a["bud_far"][-1]),
                              excess_over_control=float(a["bud"][-1] - a["bud_far"][-1])),
               neck_ratio=dict(final=float(a["neck"][-1]),
                               min=float(np.nanmin(a["neck"])) if np.isfinite(a["neck"]).any()
                               else float("nan"),
                               at_the_far_pole=float(a["neck_far"][-1])),
               series={k: [None if not np.isfinite(x) else float(x) for x in v]
                       for k, v in a.items()})
    json.dump(out, open(os.path.join(d_, "budding.json"), "w"), indent=1)
    b, k = out["bud_index"], out["neck_ratio"]
    print(f"[bud] {run:22s} bud_index {b['final']:+.4f} (max {b['max']:+.4f}, control pole "
          f"{b['at_the_far_pole']:+.4f}, EXCESS {b['excess_over_control']:+.4f})  "
          f"neck {k['final']:.3f}" + ("" if np.isfinite(k["final"]) else "  [no lobe]"), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--off-theta", type=float, default=45.0)
    ap.add_argument("--off-phi", type=float, default=45.0)
    a = ap.parse_args()
    for r in a.runs:
        go(r, a.off_theta, a.off_phi)


if __name__ == "__main__":
    main()
