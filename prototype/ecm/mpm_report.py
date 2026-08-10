#!/usr/bin/env python
"""mpm_report -- one row per run: where the sheet sits, what it carries, what the fibres did.

    python mpm_report.py 144_mpm_integrin_v 145_mpm_integrin_v_2dx ...

WHY IT EXISTS. `pass1.json` reports strain and coverage; the two numbers that decide whether a run
worked are neither of those. `standoff` says whether the membrane is where a basement membrane goes,
and `strain / geometric` says whether the sheet's own material knows it was stretched -- a run can be
perfect on one and meaningless on the other, which is exactly what 130 and 121 are. For the MPM-fibre
runs it also prints what the fibre's two ends did, because a fibre whose outer end never moves is the
signature of a constraint that scatters no momentum (142/143).

Everything is recomputed from `traj.npz` against the recorded surface map, so a run measured here and a
run measured in the session log cannot disagree.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
LOG = os.path.join(_ROOT, "log", "okuda_ECM")
REMOTE = "/groups/saalfeld/home/allierc/Graph/Plexus/log"


def row(run):
    d = os.path.join(LOG, run)
    spec = yaml.safe_load(open(os.path.join(d, "spec_run.yaml")))
    ops = {o["op"]: o for o in spec["operators"] if isinstance(o, dict) and "op" in o}
    ref = ops.get("integrin_adhesion") or ops.get("integrin_track") or ops.get("integrin_seed")
    scale = float(ref.get("scale", 1.0))
    smap = np.asarray(np.load(str(ref["surface"]).replace(REMOTE, os.path.join(_ROOT, "log")))
                      ["smap"], np.float32) * scale
    c = np.array([0.5, 0.5, 0.5])
    z = np.load(os.path.join(d, "traj.npz"))
    P = np.asarray(z["mpos"])
    X = P[-1]
    u = (X - c); u /= np.linalg.norm(u, axis=1, keepdims=True).clip(1e-12)
    th = np.arccos(np.clip(u[:, 2], -1, 1)); ph = np.arctan2(u[:, 1], u[:, 0]) % (2 * math.pi)
    nth, nph = smap.shape[1], smap.shape[2]
    R = smap[-1, np.clip((th / math.pi * nth).astype(int), 0, nth - 1),
             np.clip((ph / (2 * math.pi) * nph).astype(int), 0, nph - 1)]
    so = np.linalg.norm(X - c, axis=1) - R
    r0 = np.linalg.norm(P[0] - c, axis=1).mean(); r1 = np.linalg.norm(X - c, axis=1).mean()
    geo = r1 / max(r0, 1e-12) - 1.0
    # A RUN THAT KEPT NO STRAIN HISTORY IS NOT A RUN WITH ZERO STRAIN. `mstrain` is absent when the
    # membrane operator that fills it never ran; reporting 0.0 there would put a null in the table that
    # the run never claimed.
    ms = np.asarray(z["mstrain"], np.float32)[-1] if "mstrain" in z.files else np.array([np.nan])
    # coverage on a FINE grid: 512 bins is 88 particles per bin, so it cannot see a hole smaller than
    # a tenth of the sheet. 64x128 gives ~5 per bin, which is what a tear looks like.
    bi = (np.clip((th / math.pi * 64).astype(int), 0, 63) * 128
          + np.clip((ph / (2 * math.pi) * 128).astype(int), 0, 127))
    cov = len(np.unique(bi)) / (64 * 128)
    out = dict(run=run, standoff=so.mean(), p5=np.percentile(so, 5), p95=np.percentile(so, 95),
               inside=100 * np.mean(so < 0), strain=float(np.mean(ms)), geometric=geo,
               # THE RATIO IS ONLY MEANINGFUL ONCE THERE IS SOMETHING TO SEE. Below 5% geometric
               # stretch the denominator is noise and the percentage reads as 4088% (143).
               fidelity=(float(np.mean(ms)) / geo if geo > 0.05 else float("nan")), cov=cov,
               L=float(ops.get("integrin_seed", {}).get("length", ref.get("offset", 0.0))))
    if "ipos" in z.files:
        I = np.asarray(z["ipos"]); nf = I.shape[1] // int(ops["integrin_seed"].get("layers", 3))
        out["fib_in"] = np.linalg.norm(I[-1][:nf] - c, axis=1).mean()
        out["fib_out"] = np.linalg.norm(I[-1][-nf:] - c, axis=1).mean()
        out["fib_len"] = out["fib_out"] - out["fib_in"]
    return out


def main(runs):
    print(f"  {'run':26s}{'L':>8}{'standoff':>10}{'p5':>9}{'p95':>9}{'in%':>6}"
          f"{'strain':>8}{'geom':>7}{'F sees':>8}{'cov64':>7}{'fibre in':>10}{'out':>9}{'len':>9}")
    for r in runs:
        try:
            o = row(r)
        except Exception as e:
            print(f"  {r:26s} -- {type(e).__name__}: {e}")
            continue
        print(f"  {o['run']:26s}{o['L']:>8.4f}{o['standoff']:>+10.5f}{o['p5']:>+9.5f}{o['p95']:>+9.5f}"
              f"{o['inside']:>6.1f}{o['strain']:>8.3f}{o['geometric']:>7.2f}"
              + (f"{100*o['fidelity']:>7.0f}%" if np.isfinite(o['fidelity']) else f"{'--':>8s}")
              + f"{o['cov']:>7.3f}"
              + (f"{o['fib_in']:>10.4f}{o['fib_out']:>9.4f}{o['fib_len']:>9.4f}"
                 if "fib_in" in o else ""))


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(os.listdir(LOG)))
