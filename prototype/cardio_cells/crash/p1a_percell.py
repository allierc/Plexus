#!/usr/bin/env python
"""p1a_percell.py -- PROBE A. Is PER-CELL gain identifiable, or only GLOBAL gain?

THE QUESTION
====================================================================================================
P0 swept the gain UNIFORMLY over all 100 cells (0.25 -> 4, 16x) and the certified amplitude
instrument moved 308.5 steps, against 13.9 steps for a 40x UNIFORM sweep of E. That is a statement
about ONE number (the sheet's overall contractility) and says nothing whatever about whether cell
37's gain can be read off the recording.

CODEMAP.md already asked exactly that question for E and answered it: "+10% on one cell's E moves
the sheet 0.0016 px (1 substep) / 0.036 px (1 frame) max, on a 1024^2 image" -- three orders below
the ~0.1 px that cell tracking delivers. This file repeats that measurement, unchanged in method,
for BOTH parameters on the SAME system, so the two numbers are comparable, and then converts the
perturbation into the campaign's own unit (certified steps on the margin-20 reading surface).

WHAT IS MEASURED
----------------------------------------------------------------------------------------------------
  1. the ladder     +10% on ONE cell's E_c, and separately on ONE cell's gain_c:
                    max and RMS |dx| over particles after 1 substep, 1 frame, and one full 150-frame
                    beat, in world units and in px of a 1024^2 image (CODEMAP's unit).
                    Run for ALL 100 cells at substep/frame cadence (it is cheap) and for 6 named
                    cells -- interior, edge, wall band, corner -- over the full beat.
  2. the unit       accept.score_one(perturbed, unperturbed, working_floors()) on the margin-20
                    tracers of the 150-frame window: what one cell's +10% is WORTH, in steps.
  3. the Gram       column norms and eigenvalues of G_EE, G_gg, G and the block coupling
                    ||G_Eg||_F / sqrt(||G_EE||_F ||G_gg||_F), raw and dimensionless.

TWO TICKS, AND WHY
----------------------------------------------------------------------------------------------------
The pacemaker is sin(pi (t mod 150)/30) for (t mod 150) < 30 -- a 30-frame bump on a 150-frame
period. The gain multiplies the active-force delta, so OFF-PULSE every gain column of A is
IDENTICALLY zero and no measurement of gain is possible at that instant. The campaign's snapshot is
tick 180, and 180 mod 150 = 30: the pulse has just switched off, clock = 0.0000 exactly. So:

    tick 165   (165 mod 150 = 15 -> clock = 1.0000, the PEAK of the pulse)   both parameters live
    tick 180   (clock = 0.0000, the campaign's own snapshot)                 gain is dead on contact

Both are reported. The 150-frame beat window from tick 180 contains the next whole pulse
(ticks 300-329), so the beat-level numbers are not degenerate even though the first substep is.

usage:
  PYTHONPATH=/workspace/Plexus/src python p1a_percell.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ALG = "/workspace/Plexus/prototype/cardio_cells/algebraic"
DISC = "/workspace/Plexus/discovery_cardio_mpm"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/workspace/Plexus/src")
sys.path.insert(0, ALG)
sys.path.insert(0, HERE)
sys.path.insert(0, DISC)

from assemble import SUBSTEP_TOKENS                                   # noqa: E402
from recover import install_E                                         # noqa: E402

import crash_test as CT                                               # noqa: E402
import accept as AC                                                   # noqa: E402
import metrics as MET                                                 # noqa: E402

PX = 1024.0            # CODEMAP's unit: the sheet is a unit square, the image is 1024^2
TRACK_PX = 0.1         # what cell tracking on this data is optimistically good to (CODEMAP 6b)


# --------------------------------------------------------------------------------------------- #
def clock_of(t, period=150.0, duration=30.0):
    return float(np.sin(np.pi * (t % period) / duration)) if (t % period) < duration else 0.0


def theta_vectors(sy, cell=None, param=None, frac=0.0, uniform=None):
    """(E [C+1], gain [C+1]) = the planted pair, optionally with ONE cell scaled by (1+frac).

    `uniform` scales the whole block instead -- the P0 sweep's move, kept here as the control that
    ties a per-cell number to a global one.
    """
    E = sy.E_true.clone()
    g = sy.gain_true.clone()
    if uniform == "E":
        E[1:] = E[1:] * (1.0 + frac)
    elif uniform == "gain":
        g[1:] = g[1:] * (1.0 + frac)
    elif cell is not None:
        if param == "E":
            E[cell] = E[cell] * (1.0 + frac)
        else:
            g[cell] = g[cell] * (1.0 + frac)
    return E, g


def run(sy, E_cell, gain_cell, t0, n_frames, n_sub_extra=0, tracers=None):
    """Free-run from the snapshot: n_frames whole frames, then n_sub_extra extra substeps.

    Identical to crash_test.rollout with anchor=None and jitter=0 -- the same _outer/_tok order, the
    same per-cell gain injection -- except that it can stop part-way through a frame, which is what
    the one-substep rung of the ladder needs.
    """
    sy.restore()
    install_E(sy, E_cell)
    tr = None
    if tracers is not None:
        tr = {m: torch.zeros(n_frames, t.numel(), 2, device=sy.device, dtype=sy.dtype)
              for m, t in tracers.items()}
    for k in range(n_frames):
        sy._outer(t0 + k, gain_cell=gain_cell)
        sy.H.sub_dt = sy.dt_sub
        for _ in range(sy.n_sub_per_frame):
            for tok in SUBSTEP_TOKENS:
                sy._tok(tok)
        sy.H.sub_dt = None
        if tr is not None:
            x = sy.p.get("pos")
            for m, t in tracers.items():
                tr[m][k] = x[t]
    if n_sub_extra:
        sy._outer(t0 + n_frames, gain_cell=gain_cell)
        sy.H.sub_dt = sy.dt_sub
        for _ in range(n_sub_extra):
            for tok in SUBSTEP_TOKENS:
                sy._tok(tok)
        sy.H.sub_dt = None
    return sy.p.get("pos").clone(), tr


def disp_stats(x, x0):
    """max / RMS displacement over particles, in world units and in px of a 1024^2 image."""
    d = (x - x0).norm(dim=-1)
    nz = d > 0
    return {"max_world": float(d.max()),
            "rms_world": float(d.pow(2).mean().sqrt()),
            "rms_moved_world": float(d[nz].pow(2).mean().sqrt()) if bool(nz.any()) else 0.0,
            "max_px": float(d.max()) * PX,
            "rms_px": float(d.pow(2).mean().sqrt()) * PX,
            "rms_moved_px": (float(d[nz].pow(2).mean().sqrt()) * PX) if bool(nz.any()) else 0.0,
            "n_moved": int(nz.sum()),
            "n_moved_gt_1e-9": int((d > 1e-9).sum()),
            "snr_vs_0.1px": float(d.max()) * PX / TRACK_PX}


def gram_blocks(A, C, theta=None):
    """The Gram matrix, block by block. `theta` != None gives the DIMENSIONLESS map A diag(theta),
    whose columns are the response to a FRACTIONAL change -- the only version in which an E column
    (E ~ 130) and a gain column (gain ~ 1) may be compared."""
    Ad = A.double()
    if theta is not None:
        Ad = Ad * theta.double()[None, :]
    AE, Ag = Ad[:, :C].contiguous(), Ad[:, C:].contiguous()
    out = {}
    for nm, M in (("E", AE), ("gain", Ag)):
        cn = M.norm(dim=0)
        out[f"colnorm_{nm}"] = {"mean": float(cn.mean()), "median": float(cn.median()),
                                "min": float(cn.min()), "max": float(cn.max()),
                                "n_zero": int((cn == 0).sum()),
                                "n_below_1e-12_of_max": int((cn < 1e-12 * cn.max()).sum())
                                if float(cn.max()) > 0 else C}
    GEE, Ggg, GEg = AE.T @ AE, Ag.T @ Ag, AE.T @ Ag
    G = Ad.T @ Ad
    for nm, M in (("G_EE", GEE), ("G_gg", Ggg), ("G", G)):
        ev = torch.linalg.eigvalsh(M)
        lo, hi = float(ev.min()), float(ev.max())
        out[nm] = {"eig_min": lo, "eig_max": hi, "fro": float(M.norm()),
                   "cond": (hi / lo) if lo > 0 else float("inf"),
                   "n_nonpositive_eig": int((ev <= 0).sum())}
    den = float((GEE.norm() * Ggg.norm()).sqrt())
    out["coupling_fro"] = float(GEg.norm()) / den if den > 0 else None
    # the sharper coupling: the largest canonical correlation between the two column spaces. 1.0
    # means some combination of E columns is EXACTLY a combination of gain columns. Undefined when
    # a block is identically zero -- QR of a zero matrix returns an arbitrary orthonormal basis and
    # the "correlation" it reports is a property of LAPACK, not of the sheet.
    if float(AE.norm()) == 0.0 or float(Ag.norm()) == 0.0:
        out["max_canonical_correlation"] = None
    else:
        try:
            QE, _ = torch.linalg.qr(AE)
            Qg, _ = torch.linalg.qr(Ag)
            sv = torch.linalg.svdvals(QE.T @ Qg)
            out["max_canonical_correlation"] = float(sv.max())
        except Exception as e:
            out["max_canonical_correlation"] = f"{type(e).__name__}: {e}"
    # where the near-singularity lives: cond of the block-DIAGONAL system (coupling removed)
    Gbd = torch.zeros_like(G)
    Gbd[:C, :C], Gbd[C:, C:] = GEE, Ggg
    ev = torch.linalg.eigvalsh(Gbd)
    out["G_blockdiag"] = {"eig_min": float(ev.min()), "eig_max": float(ev.max()),
                          "cond": (float(ev.max()) / float(ev.min())) if float(ev.min()) > 0
                          else float("inf")}
    return out


# --------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=180, help="the campaign's snapshot tick")
    ap.add_argument("--pulse-tick", type=int, default=165, help="a tick with the pulse ON")
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--mode", default="full")
    ap.add_argument("--e-lo", type=float, default=40.0)
    ap.add_argument("--e-hi", type=float, default=220.0)
    ap.add_argument("--g-lo", type=float, default=0.5)
    ap.add_argument("--g-hi", type=float, default=1.5)
    ap.add_argument("--frac", type=float, default=0.10, help="the perturbation, CODEMAP's +10%%")
    ap.add_argument("--beat-frames", type=int, default=150)
    ap.add_argument("--all-cells", type=int, default=1, help="0 = only the named cells")
    ap.add_argument("--tag", default="p1a")
    args = ap.parse_args()

    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"argv": vars(args), "PX": PX, "track_px": TRACK_PX}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        # ============================================================ the two systems ==========
        sysd = {}
        for tag, W in (("pulse", args.pulse_tick), ("campaign", args.warmup)):
            a2 = argparse.Namespace(**vars(args))
            a2.warmup = W
            log(f"\n[build {tag}] warm-up to tick {W}, clock(t) = {clock_of(W):.4f}")
            sy, _ = CT.plant_and_warm(a2, log)
            sysd[tag] = (sy, W)

        sy0, W0 = sysd["campaign"]
        C, dx = sy0.C, sy0.g.dx
        R.update({"C": C, "Np": sy0.Np, "dx": dx, "dt": sy0.dt, "dt_sub": sy0.dt_sub,
                  "ticks": {t: {"tick": w, "clock": clock_of(w),
                                "norm_act0": float(sysd[t][0].act0.norm()),
                                "norm_pass0": float(sysd[t][0].pass0.norm())}
                            for t, (s, w) in sysd.items()}})
        for t, v in R["ticks"].items():
            log(f"[tick {t:<8s}] {v['tick']} clock {v['clock']:.4f}  "
                f"||active_force delta|| {v['norm_act0']:.6g}  ||passive|| {v['norm_pass0']:.6g}")

        # ============================================================ cell geometry ============
        cid = sy0.cid
        x0 = sy0.x0
        cnt = torch.bincount(cid, minlength=C + 1).clamp(min=1).double()
        cxy = torch.stack([torch.bincount(cid, weights=x0[:, k].double(), minlength=C + 1) / cnt
                           for k in (0, 1)], 1)                      # [C+1, 2]
        d_wall = torch.minimum(torch.minimum(cxy[:, 0], 1 - cxy[:, 0]),
                               torch.minimum(cxy[:, 1], 1 - cxy[:, 1]))
        d_ctr = (cxy - 0.5).norm(dim=1)
        band = 0.06 / MET.SHEET_SPAN
        # the active force each cell actually receives at the pulse peak: the stimulus is a
        # GAUSSIAN of radius 0.12 about (0.5, 0.5), so the gain of a corner cell multiplies almost
        # nothing. This is a structural fact about the system, not about the estimator.
        syP = sysd["pulse"][0]
        actn = syP.act0.norm(dim=1).double()
        act_cell = torch.bincount(syP.cid, weights=actn, minlength=C + 1) / cnt
        R["cell_geometry"] = {
            "band_world": float(band),
            "cells_in_anchor_band": int((d_wall[1:] < band).sum()),
            "act_per_cell_at_pulse": {"min": float(act_cell[1:].min()),
                                      "median": float(act_cell[1:].median()),
                                      "max": float(act_cell[1:].max()),
                                      "ratio_max_over_min": float(act_cell[1:].max()
                                                                  / act_cell[1:].min().clamp(
                                                                      min=1e-300))}}
        log(f"\n[geometry] anchored band {band:.4f}; {R['cell_geometry']['cells_in_anchor_band']}"
            f"/{C} cell centroids inside it")
        log(f"           mean |active-force delta| per cell at the pulse peak: "
            f"min {float(act_cell[1:].min()):.4g}  median {float(act_cell[1:].median()):.4g}  "
            f"max {float(act_cell[1:].max()):.4g}  "
            f"(max/min = {float(act_cell[1:].max()/act_cell[1:].min()):.4g})")

        def pick(v, target):
            return int(torch.argmin((v[1:] - target).abs()).item()) + 1

        named = {}
        named["center"] = int(torch.argmin(d_ctr[1:]).item()) + 1
        named["wall"] = int(torch.argmin(d_wall[1:]).item()) + 1
        named["edge_outside_band"] = pick(d_wall, 0.12)
        named["mid"] = pick(d_wall, 0.25)
        named["stim_1sigma"] = pick(d_ctr, 0.12)
        named["corner"] = int(torch.argmax(d_ctr[1:]).item()) + 1
        # de-duplicate while keeping the labels
        seen, cells = {}, []
        for k, c in named.items():
            if c in seen:
                seen[c] = seen[c] + "/" + k
            else:
                seen[c] = k
                cells.append(c)
        R["named_cells"] = {int(c): {"label": seen[c], "centroid": [float(cxy[c, 0]),
                                                                    float(cxy[c, 1])],
                                     "d_wall": float(d_wall[c]), "d_center": float(d_ctr[c]),
                                     "in_anchor_band": bool(float(d_wall[c]) < band),
                                     "E_true": float(sy0.E_true[c]),
                                     "gain_true": float(sy0.gain_true[c]),
                                     "n_particles": int((cid == c).sum()),
                                     "mean_act_at_pulse": float(act_cell[c])} for c in cells}
        log(f"\n[cells picked] {len(cells)} of {C}")
        for c in cells:
            v = R["named_cells"][int(c)]
            log(f"   cell {c:>3d} {v['label']:<20s} centroid ({v['centroid'][0]:.3f},"
                f"{v['centroid'][1]:.3f})  d_wall {v['d_wall']:.4f}  d_ctr {v['d_center']:.4f}  "
                f"band {str(v['in_anchor_band']):<5s} E {v['E_true']:6.1f} g {v['gain_true']:.3f}  "
                f"act {v['mean_act_at_pulse']:.4g}")

        # ============================================================ 1. the ladder ============
        #  substep and frame cadence, for EVERY cell (it costs a few seconds a cell) at both ticks
        R["ladder"] = {}
        for tag, (sy, W) in sysd.items():
            base_sub, _ = run(sy, sy.E_true, sy.gain_true, W, 0, n_sub_extra=1)
            base_frm, _ = run(sy, sy.E_true, sy.gain_true, W, 1)
            targets = list(range(1, C + 1)) if args.all_cells else cells
            t_l = time.time()
            per = {}
            for c in targets:
                per[c] = {}
                for param in ("E", "gain"):
                    E, g = theta_vectors(sy, cell=c, param=param, frac=args.frac)
                    xs, _ = run(sy, E, g, W, 0, n_sub_extra=1)
                    xf, _ = run(sy, E, g, W, 1)
                    per[c][param] = {"substep": disp_stats(xs, base_sub),
                                     "frame": disp_stats(xf, base_frm)}
            # the UNIFORM control: the same +10%, on every cell at once (P0's move)
            uni = {}
            for param in ("E", "gain"):
                E, g = theta_vectors(sy, frac=args.frac, uniform=param)
                xs, _ = run(sy, E, g, W, 0, n_sub_extra=1)
                xf, _ = run(sy, E, g, W, 1)
                uni[param] = {"substep": disp_stats(xs, base_sub),
                              "frame": disp_stats(xf, base_frm)}
            R["ladder"][tag] = {"tick": W, "clock": clock_of(W), "per_cell": per, "uniform": uni,
                                "seconds": time.time() - t_l, "n_cells_run": len(targets)}
            log(f"\n{'=' * 108}\n  LADDER at tick {W} (clock {clock_of(W):.4f}) -- "
                f"+{100*args.frac:.0f}% on ONE cell, {len(targets)} cells run in "
                f"{time.time()-t_l:.0f} s\n{'=' * 108}")
            log(f"  {'cadence':<8s} {'param':<5s} {'max|dx| px (1024^2)':>34s} "
                f"{'RMS px':>12s} {'particles moved':>17s}   {'SNR vs 0.1px':>12s}")
            for cad in ("substep", "frame"):
                for param in ("E", "gain"):
                    v = np.array([per[c][param][cad]["max_px"] for c in targets])
                    rm = np.array([per[c][param][cad]["rms_moved_px"] for c in targets])
                    nm = np.array([per[c][param][cad]["n_moved"] for c in targets])
                    log(f"  {cad:<8s} {param:<5s} "
                        f"min {v.min():.3e}  med {np.median(v):.3e}  max {v.max():.3e}  "
                        f"{np.median(rm):>10.3e}   {int(np.median(nm)):>8d}/{sy.Np:<8d} "
                        f"{np.median(v)/TRACK_PX:>12.2e}")
                med_E = np.median([per[c]["E"][cad]["max_px"] for c in targets])
                med_g = np.median([per[c]["gain"][cad]["max_px"] for c in targets])
                mx_E = max(per[c]["E"][cad]["max_px"] for c in targets)
                mx_g = max(per[c]["gain"][cad]["max_px"] for c in targets)
                R["ladder"][tag].setdefault("ratio_gain_over_E", {})[cad] = {
                    "median_of_max_px": (med_g / med_E) if med_E > 0 else None,
                    "max_of_max_px": (mx_g / mx_E) if mx_E > 0 else None,
                    "median_E_px": float(med_E), "median_gain_px": float(med_g)}
                r = R["ladder"][tag]["ratio_gain_over_E"][cad]
                log(f"  {'':<8s} {'RATIO gain/E at ' + cad:<40s} median "
                    f"{('%.4g' % r['median_of_max_px']) if r['median_of_max_px'] else 'n/a':>10s}"
                    f"   max {('%.4g' % r['max_of_max_px']) if r['max_of_max_px'] else 'n/a':>10s}")
            for param in ("E", "gain"):
                for cad in ("substep", "frame"):
                    u = uni[param][cad]
                    log(f"  UNIFORM  {param:<5s} {cad:<8s} max {u['max_px']:.4e} px   "
                        f"rms {u['rms_px']:.4e} px   moved {u['n_moved']}/{sy.Np}")

        # ============================================================ 2. the beat + steps ======
        G = args.beat_frames
        floors = AC.working_floors()
        R["floors"] = {n: floors[n] for n in floors}
        R["beat"] = {}
        pe = MET.REGISTRY["peak_excursion"]
        for btag, (sy, W) in sysd.items():
            tracers = {m: CT.tracer_indices(sy.x0, CT.probe_points(m))
                       for m in (MET.MARGIN_SAFE, MET.MARGIN_INHERITED)}
            log(f"\n{'=' * 108}\n  THE BEAT [{btag}]: {G} free frames from tick {W} "
                f"(ticks {W}..{W+G-1}), scored on the margin-{MET.MARGIN_SAFE} tracers in "
                f"certified steps\n{'=' * 108}")
            t_b = time.time()
            x_base, tr_base = run(sy, sy.E_true, sy.gain_true, W, G, tracers=tracers)
            real = tr_base[MET.MARGIN_SAFE].cpu().numpy()
            amp_real = float(np.median(pe.reading(real)))
            pulse_frames = [k for k in range(G) if clock_of(W + k) > 0]
            log(f"  [baseline] {G} frames in {time.time()-t_b:.0f} s;  reference peak_excursion "
                f"{amp_real:.6g};  pulse live on {len(pulse_frames)}/{G} frames of the window "
                f"({'ticks %d..%d' % (W+pulse_frames[0], W+pulse_frames[-1]) if pulse_frames else 'NONE'})")

            def score(sim, real=real):
                one = AC.score_one(sim, real, floors)
                st = {n: (v["steps"] if v["steps"] is not None else None) for n, v in one.items()}
                live = [n for n in st if st[n] is not None]
                worst = max([st[n] for n in live], default=None)
                lim = max(live, key=lambda n: st[n]) if live else None
                third = G // 3
                pairs = [(sim[k * third:(k + 1) * third], real[k * third:(k + 1) * third])
                         for k in range(3)]
                try:
                    acc = AC.accept(pairs, floors)
                    acc.pop("per_tick", None)
                except Exception as e:
                    acc = {"error": f"{type(e).__name__}: {e}"}
                return {"one_window": one, "steps": st, "worst_steps": worst,
                        "limiting_instrument": lim, "accept_3_thirds": acc}

            B = {"tick": W, "clock": clock_of(W), "frames": G,
                 "pulse_frames_in_window": len(pulse_frames),
                 "reference_peak_excursion": amp_real, "rows": {}}
            rows = []
            jobs = [("uniform_E", None, "E", "uniform"), ("uniform_gain", None, "gain", "uniform")]
            jobs += [(f"cell{c}_{p}", c, p, "cell") for c in cells for p in ("E", "gain")]
            for name, c, param, kind in jobs:
                t_r = time.time()
                if kind == "uniform":
                    E, g = theta_vectors(sy, frac=args.frac, uniform=param)
                else:
                    E, g = theta_vectors(sy, cell=c, param=param, frac=args.frac)
                x_p, tr_p = run(sy, E, g, W, G, tracers=tracers)
                sim = tr_p[MET.MARGIN_SAFE].cpu().numpy()
                rec = {"kind": kind, "cell": c, "param": param,
                       "label": R["named_cells"][int(c)]["label"] if c else "ALL CELLS",
                       "final_frame": disp_stats(x_p, x_base),
                       "peak_excursion_sim": float(np.median(pe.reading(sim))),
                       "tracer_rel_l2": float(np.linalg.norm(sim - real)
                                              / max(np.linalg.norm(real - real.mean(0)), 1e-300)),
                       "score": score(sim), "seconds": time.time() - t_r}
                B["rows"][name] = rec
                rows.append(rec)
                s = rec["score"]
                log(f"  {name:<22s} {rec['label']:<20s} "
                    f"max|dx| {rec['final_frame']['max_px']:>10.4f} px"
                    f"  rms {rec['final_frame']['rms_px']:>10.4f} px  moved "
                    f"{rec['final_frame']['n_moved']:>6d}/{sy.Np}  "
                    f"STEPS {('%8.3f' % s['worst_steps']) if s['worst_steps'] is not None else '  n/a'}"
                    f"  ({s['limiting_instrument']})  [{rec['seconds']:.0f} s]")

            for kind in ("cell", "uniform"):
                e = [r for r in rows if r["kind"] == kind and r["param"] == "E"]
                g = [r for r in rows if r["kind"] == kind and r["param"] == "gain"]
                if not e or not g:
                    continue
                me = float(np.median([r["final_frame"]["max_px"] for r in e]))
                mg = float(np.median([r["final_frame"]["max_px"] for r in g]))
                se = [r["score"]["worst_steps"] for r in e if r["score"]["worst_steps"] is not None]
                sg = [r["score"]["worst_steps"] for r in g if r["score"]["worst_steps"] is not None]
                B[f"ratio_gain_over_E_{kind}"] = {
                    "median_max_px_E": me, "median_max_px_gain": mg,
                    "median_max_px": mg / me if me > 0 else None,
                    "median_steps_E": float(np.median(se)) if se else None,
                    "median_steps_gain": float(np.median(sg)) if sg else None,
                    "median_steps_ratio": (float(np.median(sg)) / float(np.median(se)))
                    if se and float(np.median(se)) > 0 else None}
                v = B[f"ratio_gain_over_E_{kind}"]
                r_px = ("%.4g" % v["median_max_px"]) if v["median_max_px"] else "n/a"
                r_st = ("%.4g" % v["median_steps_ratio"]) if v["median_steps_ratio"] else "n/a"
                log(f"\n  RATIO gain/E over a beat [{btag}/{kind}]: displacement {r_px}x   "
                    f"certified steps {r_st}x  (E {v['median_steps_E']}, "
                    f"gain {v['median_steps_gain']})")
            R["beat"][btag] = B

        # ============================================================ 3. the Gram matrix =======
        R["gram"] = {}
        log(f"\n{'=' * 108}\n  THE GRAM MATRIX, BLOCK BY BLOCK\n{'=' * 108}")
        for tag, (sy_g, W_g) in sysd.items():
            t_a = time.time()
            A, a0, t_asm = sy_g.assemble(n_sub=1)
            Af, a0f, _ = sy_g.assemble(n_sub=sy_g.n_sub_per_frame)
            th = sy_g.theta_true
            blk = {"tick": W_g, "clock": clock_of(W_g), "A_shape": list(A.shape),
                   "assembly_s": t_asm,
                   "substep_raw": gram_blocks(A, C),
                   "substep_dimensionless": gram_blocks(A, C, theta=th),
                   "frame_raw": gram_blocks(Af, C),
                   "frame_dimensionless": gram_blocks(Af, C, theta=th)}
            R["gram"][tag] = blk
            del A, Af
            torch.cuda.empty_cache()
            for kind in ("substep_dimensionless", "frame_dimensionless",
                         "substep_raw", "frame_raw"):
                b = blk[kind]
                log(f"\n  [{tag} tick {W_g}, clock {clock_of(W_g):.4f}] {kind}")
                for nm in ("E", "gain"):
                    cn = b[f"colnorm_{nm}"]
                    log(f"     colnorm {nm:<5s} mean {cn['mean']:.4e}  med {cn['median']:.4e}  "
                        f"min {cn['min']:.4e}  max {cn['max']:.4e}  zero columns {cn['n_zero']}/{C}")
                if b["colnorm_E"]["median"] > 0:
                    log(f"     colnorm RATIO gain/E: median "
                        f"{b['colnorm_gain']['median']/b['colnorm_E']['median']:.4g}   max "
                        f"{b['colnorm_gain']['max']/b['colnorm_E']['max']:.4g}")
                for nm in ("G_EE", "G_gg", "G", "G_blockdiag"):
                    v = b[nm]
                    log(f"     {nm:<12s} eig_min {v['eig_min']:.4e}  eig_max {v['eig_max']:.4e}  "
                        f"cond {v['cond']:.4e}")
                log(f"     coupling ||G_Eg||/sqrt(||G_EE|| ||G_gg||) = "
                    f"{b['coupling_fro'] if b['coupling_fro'] is None else '%.4f' % b['coupling_fro']}"
                    f"   max canonical correlation {b['max_canonical_correlation']}")
            log(f"     [{time.time()-t_a:.0f} s]")

    R["wall_seconds"] = time.time() - t_start
    out = os.path.join(HERE, f"{args.tag}_percell.json")
    json.dump(R, open(out, "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{args.tag}_percell.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {out}\n[{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()
