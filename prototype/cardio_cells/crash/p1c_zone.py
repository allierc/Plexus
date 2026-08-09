#!/usr/bin/env python
"""p1c_zone -- PROBE C, attack 8. Is E unobservable, or is most of the sheet simply not being TESTED?

WHAT ATTACK 7 FOUND
================================================================================================
Swinging ONE cell's modulus across the whole planted range (40 -> 220) and reading that cell's own
particles gives, in certified steps:

    cell 50, 0.009 from the pacemaker    13.16 steps
    cell 59, 0.218                        8.46
    cell 20, 0.335                        0.95
    cell 81, 0.436                        0.14
    cell 10, 0.605                        0.10

So a per-cell modulus is not invisible. It is invisible AT DISTANCE FROM THE STIMULUS. The drive is
one Gaussian bump of radius 0.12 at the centre of a unit sheet, so most cells are barely strained,
and an unstrained cell's stiffness cannot show up in anything -- that is mechanics, not metrology.

WHAT THIS FILE MEASURES
------------------------------------------------------------------------------------------------
  1. the profile itself: per-cell motion and per-cell sensitivity against distance from the pulse
  2. the same sheet driven by a WIDER stimulus (radius 0.36 and 0.60 instead of 0.12), with the
     force scaled to keep the motion comparable. If E becomes identifiable when the whole sheet is
     strained, then "E is unidentifiable" is a statement about THIS STIMULUS and not about
     cardiomyocyte MPM.
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
import crash_test as CT                                                   # noqa: E402

INS = L.INSTRUMENTS
T0 = time.time()


def log(s=""):
    print(f"[{time.time() - T0:7.1f}s] {s}")
    sys.stdout.flush()


def steps_masked(sim, ref, mask=None):
    r = ACC.score_one(sim, ref, L.floors(), mask)
    o = {n: (None if r[n]["steps"] is None else float(r[n]["steps"])) for n in INS}
    live = [v for v in o.values() if isinstance(v, float)]
    o["STAT"] = float(max(live)) if live else None
    return o


def per_node_amp(p):
    q = p - p.mean(axis=0, keepdims=True)
    return np.linalg.norm(q, axis=-1).max(0)


def build(dev, radius=None, amplitude=None):
    rig = L.Rig(L.default_args(device=dev), radius=radius, amplitude=amplitude)
    sy = rig.sy
    rig.tracers = {"all": torch.arange(sy.Np, device=sy.device), 20: rig.tracers[20]}
    cid = sy.cid.detach().cpu().numpy() - 1
    return rig, cid, [cid == c for c in range(rig.C)]


def roll(rig, E=None, gain=None):
    tr, *_ = CT.rollout(rig.sy, rig.theta(E=E, gain=gain), rig.t0, rig.G, rig.tracers)
    return tr["all"].detach().cpu().numpy(), tr[20].detach().cpu().numpy()


def study(name, dev, radius, amplitude, out):
    rig, cid, cell_mask = build(dev, radius, amplitude)
    C = rig.C
    E_true = rig.E_true.cpu().numpy()
    cen = rig.centroids()
    r_c = np.linalg.norm(cen - 0.5, axis=1)
    ref_all, ref_20 = roll(rig)
    amp = per_node_amp(ref_all)
    per_cell_amp = np.array([float(np.median(amp[m])) for m in cell_mask])
    log(f"\n{'=' * 116}\n  {name}   radius={radius} amplitude={amplitude}\n{'=' * 116}")
    log(f"  reference peak_excursion (10x10 surface) {L.amp_reading(ref_20):.6g}"
        f"   all-particle median {np.median(amp):.6g}")

    # --- 1. the motion profile against distance from the pulse ------------------------------- #
    edges = [0.0, 0.10, 0.20, 0.30, 0.40, 0.55, 1.0]
    prof = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (r_c >= a) & (r_c < b)
        if s.sum():
            prof.append({"r_lo": a, "r_hi": b, "n_cells": int(s.sum()),
                         "median_cell_amplitude": float(np.median(per_cell_amp[s]))})
    log(f"  motion profile:  " + "   ".join(
        f"r[{p['r_lo']:.2f},{p['r_hi']:.2f}) n={p['n_cells']:>2d} amp={p['median_cell_amplitude']:.2e}"
        for p in prof))

    # --- 2. single-cell sensitivity at five radii -------------------------------------------- #
    order = np.argsort(r_c)
    picks = [int(order[0]), int(order[C // 4]), int(order[C // 2]), int(order[3 * C // 4]),
             int(order[-1])]
    log(f"\n  ONE CELL'S MODULUS 40 -> 220, read on its own particles (certified steps):")
    log(f"  {'cell':>5s}{'r from pulse':>13s}{'cell amp':>11s}{'own-cell STAT':>15s}"
        f"{'whole-sheet':>13s}{'10x10':>9s}")
    singles = {}
    for c in picks:
        E_lo = E_true.copy(); E_lo[c] = 40.0
        E_hi = E_true.copy(); E_hi[c] = 220.0
        lo_all, lo_20 = roll(rig, E=E_lo)
        hi_all, hi_20 = roll(rig, E=E_hi)
        own = steps_masked(hi_all, lo_all, cell_mask[c])
        whole = steps_masked(hi_all, lo_all)
        s20 = steps_masked(hi_20, lo_20)
        singles[str(c)] = {"r": float(r_c[c]), "cell_amplitude": float(per_cell_amp[c]),
                           "own_cell": own, "whole_sheet": whole, "surface_10x10": s20}
        log(f"  {c:>5d}{r_c[c]:13.3f}{per_cell_amp[c]:11.2e}{own['STAT']:15.3f}"
            f"{whole['STAT']:13.3f}{s20['STAT']:9.3f}")

    # --- 3. the whole field: best uniform impostor, and the per-cell reading ------------------ #
    log(f"\n  THE PLANTED FIELD against uniform impostors (10x10 surface / all particles):")
    log(f"  {'uniform E':>10s}{'10x10 STAT':>12s}{'all-part STAT':>15s}"
        f"{'cells >1 step':>15s}{'cells >5':>10s}{'median cell':>13s}")
    scan = []
    for E in np.geomspace(40.0, 400.0, 7):
        s_all, s_20 = roll(rig, E=float(E))
        per = np.array([steps_masked(s_all, ref_all, m)["STAT"] for m in cell_mask])
        rec = {"E": float(E), "surface_10x10": steps_masked(s_20, ref_20),
               "all_particles": steps_masked(s_all, ref_all),
               "cells_over_1": int((per > 1).sum()), "cells_over_5": int((per > 5).sum()),
               "per_cell_median": float(np.median(per)),
               "per_cell_STAT": [float(v) for v in per],
               "rho_steps_vs_logEerr": float(np.corrcoef(
                   np.argsort(np.argsort(np.abs(np.log(E / E_true)))),
                   np.argsort(np.argsort(per)))[0, 1])}
        scan.append(rec)
        log(f"  {E:10.1f}{rec['surface_10x10']['STAT']:12.2f}{rec['all_particles']['STAT']:15.2f}"
            f"{rec['cells_over_1']:15d}{rec['cells_over_5']:10d}{rec['per_cell_median']:13.2f}")
    best = min(scan, key=lambda r: r["surface_10x10"]["STAT"])
    log(f"    BEST uniform impostor for the PLANTED field: E = {best['E']:.1f} at "
        f"{best['surface_10x10']['STAT']:.2f} steps on the certified 10x10 surface")

    # rank correlation of per-cell reading with per-cell modulus error, in the STRAINED zone only
    near = per_cell_amp > 0.25 * per_cell_amp.max()
    rec = best
    per = np.array(rec["per_cell_STAT"])
    err = np.abs(np.log(rec["E"] / E_true))
    rho_all = float(np.corrcoef(np.argsort(np.argsort(err)), np.argsort(np.argsort(per)))[0, 1])
    rho_near = (float(np.corrcoef(np.argsort(np.argsort(err[near])),
                                  np.argsort(np.argsort(per[near])))[0, 1])
                if near.sum() > 4 else None)
    log(f"    does the per-cell reading know WHICH cell is wrong?  rho = {rho_all:+.3f} over all "
        f"{C} cells; {rho_near if rho_near is None else f'{rho_near:+.3f}'} over the "
        f"{int(near.sum())} cells that actually move (amplitude > 25% of the maximum)")

    out[name] = {"radius": radius, "amplitude": amplitude,
                 "ref_amplitude_10x10": L.amp_reading(ref_20),
                 "profile": prof, "singles": singles, "impostor_scan": scan,
                 "best_impostor": {"E": best["E"], "steps_10x10": best["surface_10x10"]["STAT"],
                                   "steps_all": best["all_particles"]["STAT"]},
                 "rho_all": rho_all, "rho_strained": rho_near,
                 "n_strained_cells": int(near.sum()),
                 "per_cell_amplitude": [float(v) for v in per_cell_amp],
                 "r_from_pulse": [float(v) for v in r_c],
                 "planted_E": [float(v) for v in E_true]}
    rig.free()


def main(a):
    out = {"null": L.null_row()}
    study("stimulus radius 0.12 (the spec)", a.device, None, None, out)
    study("stimulus radius 0.36, force x3", a.device, 0.36, 60.0, out)
    study("stimulus radius 0.60, force x5", a.device, 0.60, 100.0, out)

    log(f"\n{'=' * 116}\n  SUMMARY -- does widening the stimulus make E identifiable?\n{'=' * 116}")
    log(f"  {'configuration':<34s}{'beat amp':>10s}{'nearest-cell':>14s}{'farthest-cell':>15s}"
        f"{'best impostor':>16s}{'cells >5 steps':>16s}{'rho strained':>14s}")
    for name, s in out.items():
        if name == "null":
            continue
        sing = list(s["singles"].values())
        best_scan = min(s["impostor_scan"], key=lambda r: r["surface_10x10"]["STAT"])
        log(f"  {name:<34s}{s['ref_amplitude_10x10']:10.5f}{sing[0]['own_cell']['STAT']:14.2f}"
            f"{sing[-1]['own_cell']['STAT']:15.2f}"
            f"{s['best_impostor']['steps_10x10']:16.2f}{best_scan['cells_over_5']:16d}"
            + (f"{s['rho_strained']:14.3f}" if s["rho_strained"] is not None else f"{'--':>14s}"))
    log(f"\n  the null (knowing nothing): {min(out['null'].values()):.2f} steps")
    p = os.path.join(HERE, "p1c_zone.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    log(f"  -> {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    main(ap.parse_args())
