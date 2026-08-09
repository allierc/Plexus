#!/usr/bin/env python
"""p1c_mask -- PROBE C, attack 6. Is E invisible, or is the READING SURFACE hiding it?

WHY THIS EXISTS
================================================================================================
Two of the four certified instruments are POPULATION SUMMARIES, not per-node comparisons:

    peak_excursion.compute = | median_over_nodes(sim) - median_over_nodes(real) |
    path_length.compute    = | median_over_nodes(sim) - median_over_nodes(real) |

so a spatial pattern that stiffens half the tissue and softens the other half moves the two medians
hardly at all, by construction, no matter how large the local effect is. The other two --
orientation_error and coordination -- ARE per-node paired (a median of per-node differences), so
they do see arrangement, but the sweep showed they span about one step over 40x of E.

That gives the checkerboard result a second possible reading, and the two have opposite
consequences:

    (a) E genuinely does not change the motion -> no estimator can recover it
    (b) E changes the motion locally, and the median cancels it -> a DIFFERENT reading surface
        recovers it, and the campaign's instruments, not the physics, are the blocker

`score_one` already takes a `mask`, so (b) can be tested without inventing a metric: score the
stiff-cell probes and the soft-cell probes SEPARATELY. If the local effect is real and merely
cancelling, the two masked scores are large and of opposite sign in the raw reading. If both are
small, the physics is the blocker and the conclusion stands.

Everything cited here is still `accept.score_one` on the four certified instruments. The per-node
spread numbers at the end are DIAGNOSTICS, uncertified, and are used only to explain the certified
readings -- never to support the verdict.
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


def dense_points(n=20, margin=20, side=MET.GRID_SIDE):
    """The same patch of sheet as the margin-20 selection, sampled n x n instead of 10 x 10."""
    u = np.linspace(margin, side - 1 - margin, n) / (side - 1.0)
    gx, gy = np.meshgrid(u, u, indexing="ij")
    return np.stack([gy.ravel(), gx.ravel()], 1)


def per_node(p):
    """[M] per-node peak excursion and path length -- the quantities the certified metrics take a
    median of. DIAGNOSTIC ONLY."""
    q = p - p.mean(axis=0, keepdims=True)
    pe = np.linalg.norm(q, axis=-1).max(0)
    d = np.diff(np.concatenate([p, p[:1]], 0), axis=0)
    pl = np.linalg.norm(d, axis=-1).sum(0)
    return pe, pl


def row(sim, ref, mask=None):
    r = ACC.score_one(sim, ref, L.floors(), mask)
    o = {n: (None if r[n]["steps"] is None else float(r[n]["steps"])) for n in INS}
    live = [v for v in o.values() if isinstance(v, float)]
    o["STAT"] = float(max(live)) if live else None
    return o


def show(name, r, extra=""):
    log(f"  {name:<38s}" + "".join(
        (f"{r[n]:12.2f}" if isinstance(r.get(n), float) else f"{'undef':>12s}") for n in INS)
        + f"{r['STAT']:9.2f}  {extra}")


def hdr():
    log(f"  {'comparison':<38s}" + "".join(f"{n[:11]:>12s}" for n in INS) + f"{'STAT':>9s}")


def main(a):
    rig = L.Rig(L.default_args(device=a.device), quiet=False, log=log)
    sy = rig.sy
    out = {"null": L.null_row()}

    # a denser reading surface over the SAME patch of sheet
    for n in (20, 30):
        pts = dense_points(n)
        rig.tracers[f"d{n}"] = CT.tracer_indices(sy.x0, pts)
    surfaces = [("10x10 (the certified selection)", 20), ("20x20 (same patch, denser)", "d20"),
                ("30x30 (same patch, denser)", "d30")]

    lo, hi, mid = 45.0, 220.0, 132.5
    chk = rig.checker(lo, hi, block=0.10, by="space")
    anti = rig.checker(hi, lo, block=0.10, by="space")
    # a composition-EXACT control: the same multiset of moduli, rearranged at random. The
    # checkerboard/anti-checkerboard pair is 51/49 against 49/51, so it is not quite matched.
    perm = np.random.default_rng(11).permutation(chk)

    log(f"\n  checkerboard: {int((chk == hi).sum())} stiff / {int((chk == lo).sum())} soft cells, "
        f"block 0.10 world (cells are ~0.1 across)")

    # roll once per candidate, keeping every reading surface from the SAME rollout
    rolls = {}
    for tag, E in (("checker", chk), ("anti", anti), ("perm", perm),
                   ("uniform_mean", mid), ("uniform_124", 124.0)):
        tr, *_ = CT.rollout(sy, rig.theta(E=E), rig.t0, rig.G, rig.tracers)
        rolls[tag] = {k: v.detach().cpu().numpy() for k, v in tr.items()}
        log(f"  rolled {tag}")

    # ---------------------------------------------------------------------------------------- #
    #  1. THE SAME COMPARISONS ON A DENSER READING SURFACE
    # ---------------------------------------------------------------------------------------- #
    log(f"\n{'=' * 116}\n  1. DOES A DENSER READING SURFACE SEE IT? same rollouts, more probes"
        f"\n{'=' * 116}")
    out["surfaces"] = {}
    for sname, key in surfaces:
        log(f"\n  {sname}   M = {rolls['checker'][key].shape[1]} probes")
        hdr()
        s = {}
        for cmp_name, (x, y) in {"checker vs uniform(132.5)": ("checker", "uniform_mean"),
                                 "checker vs ANTI-checker": ("checker", "anti"),
                                 "checker vs PERMUTED (same multiset)": ("checker", "perm"),
                                 "checker vs uniform(124)": ("checker", "uniform_124")}.items():
            r = row(rolls[x][key], rolls[y][key])
            s[cmp_name] = r
            show(cmp_name, r)
        out["surfaces"][sname] = s

    # ---------------------------------------------------------------------------------------- #
    #  2. MASKED SCORING -- stiff-cell probes and soft-cell probes separately
    # ---------------------------------------------------------------------------------------- #
    log(f"\n{'=' * 116}\n  2. MASKED SCORING -- the median cancels a checkerboard by construction; "
        f"scoring each half alone does not\n{'=' * 116}")
    out["masked"] = {}
    for sname, key in surfaces:
        idx = rig.tracers[key]
        cell_of_probe = sy.cid[idx].detach().cpu().numpy() - 1          # 0-based cell index
        stiff = chk[cell_of_probe] == hi
        soft = ~stiff
        log(f"\n  {sname}: {int(stiff.sum())} probes in stiff cells, {int(soft.sum())} in soft")
        hdr()
        s = {"n_stiff": int(stiff.sum()), "n_soft": int(soft.sum())}
        for cmp_name, (x, y) in {"checker vs uniform(132.5)": ("checker", "uniform_mean"),
                                 "checker vs ANTI-checker": ("checker", "anti")}.items():
            for mname, m in (("all probes", None), ("STIFF probes only", stiff),
                             ("SOFT probes only", soft)):
                r = row(rolls[x][key], rolls[y][key], m)
                # the raw amplitude readings, so the SIGN of the difference is visible
                pe_s = MET.REGISTRY["peak_excursion"].reading(rolls[x][key], m)
                pe_r = MET.REGISTRY["peak_excursion"].reading(rolls[y][key], m)
                r["peak_excursion_sim"], r["peak_excursion_real"] = pe_s, pe_r
                r["signed_delta"] = pe_s - pe_r
                s[f"{cmp_name} | {mname}"] = r
                show(f"{cmp_name[:20]} | {mname}", r,
                     f"peak_exc {pe_s:.6f} vs {pe_r:.6f}  delta {pe_s - pe_r:+.2e}")
        out["masked"][sname] = s

    # ---------------------------------------------------------------------------------------- #
    #  3. THE DIAGNOSTIC -- how much the median throws away (NOT evidence, an explanation)
    # ---------------------------------------------------------------------------------------- #
    log(f"\n{'=' * 116}\n  3. DIAGNOSTIC (uncertified): the per-node spread the median discards"
        f"\n{'=' * 116}")
    log(f"  {'comparison':<40s}{'surface':>10s}{'|d median|':>13s}{'median |d|':>13s}"
        f"{'cancellation':>14s}{'in steps':>10s}")
    out["cancellation"] = {}
    for sname, key in surfaces:
        for cmp_name, (x, y) in {"checker vs uniform(132.5)": ("checker", "uniform_mean"),
                                 "checker vs ANTI-checker": ("checker", "anti")}.items():
            a_pe, _ = per_node(rolls[x][key])
            b_pe, _ = per_node(rolls[y][key])
            dmed = abs(np.median(a_pe) - np.median(b_pe))
            meda = float(np.median(np.abs(a_pe - b_pe)))
            unit = 3.0 * L.floors()["peak_excursion"]["unit"]
            out["cancellation"][f"{cmp_name} @ {sname}"] = {
                "abs_diff_of_medians": float(dmed), "median_abs_diff": meda,
                "ratio": float(meda / max(dmed, 1e-30)),
                "diff_of_medians_steps": float(dmed / unit),
                "median_abs_diff_steps": float(meda / unit)}
            log(f"  {cmp_name:<40s}{sname.split()[0]:>10s}{dmed:13.3e}{meda:13.3e}"
                f"{meda / max(dmed, 1e-30):14.2f}x{meda / unit:9.2f}")

    log(f"\n  one certified step of peak_excursion = {3.0 * L.floors()['peak_excursion']['unit']:.3e} "
        f"world units; the reference beat is {L.amp_reading(rolls['uniform_mean'][20]):.5f}")
    log(f"  the null (knowing nothing): "
        + ", ".join(f"{n} {v:.2f}" for n, v in out["null"].items()))

    rig.free()
    p = os.path.join(HERE, "p1c_mask.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    log(f"  -> {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    main(ap.parse_args())
