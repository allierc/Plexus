#!/usr/bin/env python
"""p1c_percell -- PROBE C, attack 7. The decisive one: read the observable PER CELL.

WHY
================================================================================================
`p1c_mask` showed that restricting `peak_excursion` to the stiff half of a checkerboard turns 1.63
steps into 8.66. That is a fact about the READING SURFACE, not about the physics: the certified
amplitude instruments are `|median_over_nodes(sim) - median_over_nodes(real)|`, and a median over
the whole tissue cancels a pattern that raises half of it and lowers the other half.

But that mask was built from the answer, and an oracle-derived statistic is the exact error this
campaign has already made three times. So the mask used here comes from the SEGMENTATION instead --
which an estimator genuinely has, since the cell boundaries are the input to the whole model. One
mask per cell; `score_one` is called 100 times with `mask = the particles of cell c`; every number
is still a certified instrument read through `cite()`.

Two questions, and the second is the one that decides P1:

  A. SINGLE-CELL SENSITIVITY. Swing ONE cell's modulus across the whole planted range (40 -> 220)
     and read that cell's own particles. If the cell cannot see its own stiffness change at full
     contrast over a full beat, per-cell E is unobservable and no reading surface repairs it.
  B. THE WHOLE FIELD. Score the planted field against its best uniform impostor, per cell. If many
     cells read above a step and the reading correlates with that cell's own modulus error, the
     information is in the recording and the global median was hiding it.

The reading surface is EVERY PARTICLE (10 000 of them, 100 per cell) -- the most an observer of
this sheet could possibly have, so a negative answer here is a statement about the system rather
than about the sampling.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import p1c_lib as L                                                       # noqa: E402
import accept as ACC                                                      # noqa: E402
import metrics as MET                                                     # noqa: E402
import crash_test as CT                                                   # noqa: E402

INS = L.INSTRUMENTS
T0 = time.time()


def log(s=""):
    print(f"[{time.time() - T0:7.1f}s] {s}")
    sys.stdout.flush()


def steps_masked(sim, ref, mask):
    r = ACC.score_one(sim, ref, L.floors(), mask)
    o = {n: (None if r[n]["steps"] is None else float(r[n]["steps"])) for n in INS}
    live = [v for v in o.values() if isinstance(v, float)]
    o["STAT"] = float(max(live)) if live else None
    return o


def summarise(name, per_cell, log_it=True):
    st = np.array([c["STAT"] for c in per_cell], float)
    pe = np.array([c["peak_excursion"] for c in per_cell], float)
    s = {"n_cells": int(st.size),
         "STAT_median": float(np.median(st)), "STAT_p90": float(np.percentile(st, 90)),
         "STAT_max": float(st.max()),
         "cells_over_1_step": int((st > 1).sum()), "cells_over_5_steps": int((st > 5).sum()),
         "peak_excursion_median": float(np.median(pe)),
         "peak_excursion_p90": float(np.percentile(pe, 90)),
         "peak_excursion_cells_over_1": int((pe > 1).sum())}
    if log_it:
        log(f"  {name:<40s} median {s['STAT_median']:7.2f}  p90 {s['STAT_p90']:7.2f}  "
            f"max {s['STAT_max']:8.2f}   cells >1 step {s['cells_over_1_step']:>3d}/100   "
            f">5 steps {s['cells_over_5_steps']:>3d}/100")
    return s


def main(a):
    rig = L.Rig(L.default_args(device=a.device), quiet=False, log=log)
    sy = rig.sy
    C = rig.C
    E_true = rig.E_true.cpu().numpy()

    # the maximal reading surface: every particle
    allp = torch.arange(sy.Np, device=sy.device)
    rig.tracers = {"all": allp, 20: rig.tracers[20]}
    cid = sy.cid.detach().cpu().numpy() - 1                      # 0-based cell per particle
    cell_mask = [cid == c for c in range(C)]
    log(f"  reading surface: all {sy.Np} particles, {int(np.bincount(cid).min())}-"
        f"{int(np.bincount(cid).max())} per cell")

    def roll(E=None, gain=None):
        tr, *_ = CT.rollout(sy, rig.theta(E=E, gain=gain), rig.t0, rig.G, rig.tracers)
        return tr["all"].detach().cpu().numpy(), tr[20].detach().cpu().numpy()

    out = {"null": L.null_row(), "planted_E": [float(v) for v in E_true]}
    ref_all, ref_20 = roll()
    log(f"  reference peak_excursion: all-particles {L.amp_reading(ref_all):.6g}, "
        f"10x10 surface {L.amp_reading(ref_20):.6g}")

    # ------------------------------------------------------------------------------------------ #
    #  A. SINGLE-CELL SENSITIVITY -- one cell's modulus across the whole planted range
    # ------------------------------------------------------------------------------------------ #
    log(f"\n{'=' * 116}\n  A. ONE CELL AT A TIME. Its modulus is moved 40 -> 220 (x5.5, the whole "
        f"planted range) and its OWN particles are read.\n{'=' * 116}")
    cen = rig.centroids()
    d_centre = np.linalg.norm(cen - 0.5, axis=1)
    picks = [int(np.argmin(d_centre)),                                    # under the pacemaker
             int(np.argsort(d_centre)[C // 4]), int(np.argsort(d_centre)[C // 2]),
             int(np.argsort(d_centre)[3 * C // 4]), int(np.argmax(d_centre))]
    log(f"  {'cell':>5s}{'centroid':>18s}{'r from pulse':>13s}{'E_true':>8s}"
        f"{'own-cell STAT':>15s}{'own peak_exc':>14s}{'WHOLE-sheet STAT':>18s}"
        f"{'10x10 STAT':>12s}")
    single = {}
    for c in picks:
        E_lo = E_true.copy(); E_lo[c] = 40.0
        E_hi = E_true.copy(); E_hi[c] = 220.0
        lo_all, lo_20 = roll(E=E_lo)
        hi_all, hi_20 = roll(E=E_hi)
        own = steps_masked(hi_all, lo_all, cell_mask[c])
        whole = steps_masked(hi_all, lo_all, None)
        s20 = steps_masked(hi_20, lo_20, None)
        single[str(c)] = {"centroid": [float(x) for x in cen[c]],
                          "r_from_pulse": float(d_centre[c]), "E_true": float(E_true[c]),
                          "own_cell": own, "whole_sheet": whole, "surface_10x10": s20}
        log(f"  {c:>5d}{str(np.round(cen[c], 3)):>18s}{d_centre[c]:13.3f}{E_true[c]:8.1f}"
            f"{own['STAT']:15.3f}{own['peak_excursion']:14.3f}{whole['STAT']:18.3f}"
            f"{s20['STAT']:12.3f}")
    out["single_cell"] = single
    log(f"  (a step is a difference the instrument can resolve; the null is "
        f"{min(out['null'].values()):.2f} steps)")

    # ------------------------------------------------------------------------------------------ #
    #  B. THE WHOLE FIELD, read per cell
    # ------------------------------------------------------------------------------------------ #
    log(f"\n{'=' * 116}\n  B. THE WHOLE FIELD, scored per cell against the planted rollout"
        f"\n{'=' * 116}")
    chk = rig.checker(45.0, 220.0, block=0.10, by="space")
    cands = {
        "uniform E = 93 (best impostor)": {"E": 93.0},
        "uniform E = 128.4 (planted mean)": {"E": float(E_true.mean())},
        "planted E SHUFFLED": {"E": np.random.default_rng(7).permutation(E_true)},
        "checkerboard 45/220": {"E": chk},
        "uniform gain = mean (E true) [control]": {"gain": float(rig.gain_true.mean())},
        "gain x1.10 (E true) [control]": {"gain": (rig.gain_true * 1.10).cpu().numpy()},
    }
    log(f"  {'candidate':<40s}{'median':>8s}{'p90':>10s}{'max':>10s}"
        f"{'>1 step':>18s}{'>5 steps':>13s}")
    out["fields"] = {}
    for name, kw in cands.items():
        sim_all, sim_20 = roll(**kw)
        per = [steps_masked(sim_all, ref_all, m) for m in cell_mask]
        s = summarise(name, per)
        s["whole_sheet"] = steps_masked(sim_all, ref_all, None)
        s["surface_10x10"] = steps_masked(sim_20, ref_20, None)
        s["per_cell_STAT"] = [float(p["STAT"]) for p in per]
        s["per_cell_peak_excursion"] = [float(p["peak_excursion"]) for p in per]
        # does the per-cell reading know WHICH cell is wrong?
        if "E" in kw:
            Ec = np.full(C, kw["E"], float) if np.isscalar(kw["E"]) else np.asarray(kw["E"], float)
            err = np.abs(np.log(Ec / E_true))
            st = np.array(s["per_cell_STAT"])
            s["log_E_error"] = [float(v) for v in err]
            if err.std() > 1e-12:
                s["spearman_steps_vs_logEerr"] = float(np.corrcoef(
                    np.argsort(np.argsort(err)), np.argsort(np.argsort(st)))[0, 1])
                s["pearson_steps_vs_logEerr"] = float(np.corrcoef(err, st)[0, 1])
        out["fields"][name] = s
    log(f"\n  whole-sheet (unmasked) and the certified 10x10 surface, for the same candidates:")
    log(f"  {'candidate':<40s}{'all particles':>15s}{'10x10 surface':>15s}"
        f"{'rho(steps, |log E err|)':>26s}")
    for name, s in out["fields"].items():
        rho = s.get("spearman_steps_vs_logEerr")
        log(f"  {name:<40s}{s['whole_sheet']['STAT']:15.2f}{s['surface_10x10']['STAT']:15.2f}"
            + (f"{rho:26.3f}" if rho is not None else f"{'--':>26s}"))

    log(f"\n  the null (knowing nothing): "
        + ", ".join(f"{n} {v:.2f}" for n, v in out["null"].items()))
    rig.free()
    p = os.path.join(HERE, "p1c_percell.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    log(f"  -> {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    main(ap.parse_args())
