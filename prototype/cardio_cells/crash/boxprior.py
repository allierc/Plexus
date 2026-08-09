#!/usr/bin/env python
"""boxprior -- the box constraint, re-derived from something that is not the answer. P0.

THE DEFECT
================================================================================================
The box was anchored on the fit it was supposed to constrain:

    nv = out["naive"]                                        # round5_solve.py:194-202
    mE = median(nv[:C][nv[:C] > 0])
    lo, hi = 0.2 * mE, 5.0 * mE

`nv` is the naive least-squares estimate, and the naive estimate is ATTENUATED -- noise in the
measured F biases the regression slope toward zero, which is the campaign's oldest and best
established finding. So the box slides down exactly as far as the bias does, and it does it
silently. Read off the 32 stored configurations, against a planted spread of E in [45, 220]:

    condition                          implied median   planted moduli outside the box
    clean                                    128                0 / 100
    realizable noise  sigma_F = 0.0039        40-46             0-11 / 100
    coarse control grid (48, 61 px)           29-34            30-43 / 100
    high noise        sigma_F = 0.0327         2.2            100 / 100

The true median is 132. **The worse the data, the more confidently the prior excludes the truth**,
and a constrained solve then reports a tight, converged, entirely wrong answer -- with the active
bounds as evidence that the prior was doing useful work. 26 of 32 configurations already excluded
some planted modulus. This is the third instrument in the campaign to fail the same way as the
first two: a quantity derived from the thing it was meant to check.

WHAT IT IS ANCHORED ON INSTEAD
------------------------------------------------------------------------------------------------
Attenuation is a property of the REGRESSION -- of dividing noisy measured strain into noisy
measured acceleration. It is not a property of how far the tissue moves. Displacement amplitude is
observed directly, never differentiated, and therefore never attenuated.

So the anchor is a forward calibration on a CERTIFIED instrument:

    1. run the forward model at several uniform moduli E, spanning decades
    2. read `peak_excursion` off each rollout -- the same instrument the acceptance statistic uses
    3. amplitude falls monotonically with stiffness, so invert: which uniform E reproduces the
       amplitude the tissue actually shows?
    4. that E is the anchor, and the box is a declared multiplicative spread about it

Nothing in that chain touches F, a derivative, or a solve, so there is nothing for the noise to
attenuate. The prior and the acceptance statistic also now rest on the same instrument, which is
the consistency the old pair never had.

WHAT THE WIDTH MEANS, AND WHY IT IS DECLARED AND NOT FITTED
------------------------------------------------------------------------------------------------
The anchor fixes the SCALE. The width has to come from a belief about how much cells in one
tissue differ from each other, and that belief is biological, not measurable from this recording.
It is written here as a number with a reason attached, before any fit is seen, and it is the same
number for every noise level -- which is the property the old box lacked.

    python boxprior.py --calibrate            # the sweep, and the anchor
    python boxprior.py --audit                # what the old box did, from the stored runs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/workspace/Plexus/discovery_cardio_mpm")

import metrics as MET                                                    # noqa: E402

# The declared width. A factor of 5 either way is 25x from softest to stiffest cell, which is
# wider than the planted spread (220/45 = 4.9x) and wider than the cell-to-cell range reported for
# cardiomyocytes on a uniform substrate. It is deliberately generous: a prior's job here is to
# exclude the divergent and the negative, not to do the estimating.
WIDTH = 5.0

# The planted truth on the synthetic sheet, for scoring the prior only. `seed_from_segmentation`
# gives a deterministic spread over [45, 220] with no props file (see any run log).
PLANTED_LO, PLANTED_HI = 45.0, 220.0


# =============================================================================================
# THE AUDIT -- what the old box did, read off runs that already exist
# =============================================================================================
def audit(paths=None):
    """Exclusion of the planted moduli by the naive-anchored box, per stored configuration."""
    paths = paths or (sorted(glob.glob(os.path.join(HERE, "round5_solve*.json")))
                      + sorted(glob.glob(os.path.join(HERE, "refute5_solve*.json"))))
    truth = np.linspace(PLANTED_LO, PLANTED_HI, 100)
    rows = []
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        for name, f in d.get("fits", {}).items():
            for T, r in (f.get("T") or {}).items():
                b = (r.get("box_bounds") or {}).get("E")
                if not b:
                    continue
                rows.append({"file": os.path.basename(p), "config": name, "T": T,
                             "lo": b[0], "hi": b[1], "implied_median": b[0] / 0.2,
                             "excluded": int(((truth < b[0]) | (truth > b[1])).sum())})
    return rows


# =============================================================================================
# THE CALIBRATION -- amplitude against stiffness, on the forward model
# =============================================================================================
def amplitude_curve(sy, rollout, tracers, t0, G, e_grid, gain_ref, vary="E", e_ref=None):
    """Sweep `vary` ("E" or "gain") and read the certified amplitude instrument off each rollout."""
    import torch
    out = []
    pe = MET.REGISTRY["peak_excursion"]
    for v in e_grid:
        if vary == "E":
            th = torch.cat([torch.full((sy.C,), float(v), device=sy.device, dtype=sy.dtype),
                            gain_ref.clone()])
        else:
            th = torch.cat([e_ref.clone(),
                            torch.full((sy.C,), float(v), device=sy.device, dtype=sy.dtype)])
        tr, *_ = rollout(sy, th, t0, G, tracers)
        loops = tr[20].detach().cpu().numpy()
        out.append({"E": float(v), "amplitude": float(np.median(pe.reading(loops)))})
    return out


def _unused_amplitude_curve(sy, rollout, tracers, t0, G, e_grid, gain_ref):
    """`peak_excursion` of a uniform-E rollout, for each E in the grid.

    Uniform, not per-cell: the anchor is a question about SCALE, and one number is all that is
    being asked for. Per-cell structure is what the fit is for.
    """
    import torch
    out = []
    pe = MET.REGISTRY["peak_excursion"]
    for E in e_grid:
        th = torch.cat([torch.full((sy.C,), float(E), device=sy.device, dtype=sy.dtype),
                        gain_ref.clone()])
        tr, *_ = rollout(sy, th, t0, G, tracers)
        loops = tr[20].detach().cpu().numpy()                # the margin-20 reading surface
        out.append({"E": float(E), "amplitude": float(np.median(pe.reading(loops)))})
    return out


def anchor_from_amplitude(curve, observed):
    """Invert the amplitude(E) curve at the observed amplitude, in log-log.

    Over a decade of stiffness the small-deformation response is close to amplitude ~ E^-1, so
    log-log is nearly a straight line and the inversion is a one-line interpolation rather than a
    solve. The fitted exponent is returned so that assumption is visible and checkable, not
    assumed: if it is far from -1 the sheet is not in the regime this reasoning describes.
    """
    E = np.array([c["E"] for c in curve], float)
    A = np.array([c["amplitude"] for c in curve], float)
    ok = np.isfinite(A) & (A > 0)
    E, A = E[ok], A[ok]
    if E.size < 3:
        raise RuntimeError("amplitude curve has fewer than three usable points")
    slope, intercept = np.polyfit(np.log(E), np.log(A), 1)
    E0 = float(np.exp((np.log(observed) - intercept) / slope))
    pred = np.exp(intercept + slope * np.log(E))
    resid = float(np.max(np.abs(np.log(pred) - np.log(A))))
    monotone = bool(np.all(np.diff(A) < 0))
    return {"anchor_E": E0, "exponent": float(slope), "loglog_max_resid": resid,
            "monotone_decreasing": monotone,
            "box": [E0 / WIDTH, E0 * WIDTH], "width": WIDTH,
            "observed_amplitude": float(observed),
            "in_sweep_range": bool(E.min() <= E0 <= E.max())}


def score_box(box, truth=None):
    truth = np.linspace(PLANTED_LO, PLANTED_HI, 100) if truth is None else np.asarray(truth, float)
    n = int(((truth < box[0]) | (truth > box[1])).sum())
    return {"excluded": n, "of": int(truth.size), "contains_truth": n == 0,
            "planted": [float(truth.min()), float(truth.max())]}


def calibrate(args):
    """Run the sweep on the crash test's own system, at several F-noise levels.

    The point of repeating it under noise is not to see the anchor improve -- it is to see that it
    DOES NOT MOVE. The old box moved by a factor of 60 between clean and high noise. If this one
    moves at all, it is because the noise perturbed the observed trajectory, not because it
    perturbed a regression, and the difference should be visible as a much smaller number.
    """
    import torch
    import crash_test as CT

    log = print
    sy, recA = CT.plant_and_warm(args, log)
    t0, G = args.warmup, args.window
    x0 = sy.x0
    tracers = {m: CT.tracer_indices(x0, CT.probe_points(m)) for m in (10, 20)}
    theta_true = sy.theta_true
    gain_ref = theta_true[sy.C:].clone()

    # the OBSERVED amplitude -- from the reference trajectory, which is what a recording gives
    tr_ref, *_ = CT.rollout(sy, theta_true, t0, G, tracers)
    ref_loops = tr_ref[20].detach().cpu().numpy()
    pe = MET.REGISTRY["peak_excursion"]
    observed = float(np.median(pe.reading(ref_loops)))

    e_ref = theta_true[:sy.C].clone()
    fl = _floor_unit()
    e_grid = np.geomspace(args.sweep_lo, args.sweep_hi, args.sweep_n)
    log(f"\n  sweeping {args.sweep_n} uniform moduli over [{args.sweep_lo:g}, {args.sweep_hi:g}] "
        f"-- observed amplitude {observed:.6g}")
    curve = amplitude_curve(sy, CT.rollout, tracers, t0, G, e_grid, gain_ref, vary="E")
    for c in curve:
        log(f"    E    = {c['E']:8.2f}   peak_excursion = {c['amplitude']:.6g}")

    # THE CONTRAST that says whether a flat E curve is a property of the sheet or of the sweep.
    # If amplitude is flat in E and steep in gain, the sheet is drag-dominated and force-driven,
    # E is a small correction to the observable, and no prior of any kind repairs that.
    g_grid = np.geomspace(args.gain_lo_s, args.gain_hi_s, args.sweep_n)
    log(f"\n  the contrast: the same sweep on GAIN over "
        f"[{args.gain_lo_s:g}, {args.gain_hi_s:g}]")
    gcurve = amplitude_curve(sy, CT.rollout, tracers, t0, G, g_grid, gain_ref,
                             vary="gain", e_ref=e_ref)
    for c in gcurve:
        log(f"    gain = {c['E']:8.3f}   peak_excursion = {c['amplitude']:.6g}")

    a = anchor_from_amplitude(curve, observed)
    a["identifiability"] = {"E": _ident(curve, fl), "gain": _ident(gcurve, fl)}
    a["gain_curve"] = gcurve
    # the planted moduli, READ OFF THE SYSTEM. The operator's log line advertises its
    # [youngs_min, youngs_max] request; what is actually installed is the deterministic spread it
    # produced, and scoring a prior against the advertised range instead of the installed one is
    # the same class of error this file exists to fix.
    truth = sy.E_true[1:].detach().cpu().numpy()
    a["scored"] = score_box(a["box"], truth)
    a["true_median"] = float(np.median(truth))
    a["planted_range"] = [float(truth.min()), float(truth.max())]
    a["curve"] = curve

    log(f"\n{'=' * 100}\n  IDENTIFIABILITY -- how much the observable moves when the parameter "
        f"does, in certified steps\n{'=' * 100}")
    for k, v in a["identifiability"].items():
        log(f"    {k:<5s} over x{v['fold']:<5.0f}:  amplitude spans {v['span_steps']:6.1f} steps"
            f"   monotone {str(v['monotone']):<5s}   exponent {v['exponent']:+.3f}"
            f"   invertible: {v['invertible']}")
    log(f"\n{'=' * 100}\n  THE ANCHOR\n{'=' * 100}")
    log(f"    amplitude ~ E^{a['exponent']:.3f}   (log-log max residual {a['loglog_max_resid']:.4f}"
        f", monotone: {a['monotone_decreasing']})")
    log(f"    anchor E0            {a['anchor_E']:.2f}      true median {a['true_median']:.1f}"
        f"      ratio {a['anchor_E'] / a['true_median']:.3f}")
    log(f"    box  [{a['box'][0]:.2f}, {a['box'][1]:.2f}]  (x{WIDTH:g} either way, declared)")
    log(f"    planted E in [{a['planted_range'][0]:.1f}, {a['planted_range'][1]:.1f}]")
    log(f"    planted moduli excluded: {a['scored']['excluded']}/{a['scored']['of']}"
        + ("   -- the truth is inside" if a["scored"]["contains_truth"] else
           "   -- STILL EXCLUDES THE TRUTH"))
    out = os.path.join(HERE, f"boxprior_{args.tag}.json")
    json.dump(a, open(out, "w"), indent=1)
    log(f"\n  -> {out}")
    return a


def _floor_unit():
    """3 x the largest measured floor of peak_excursion = one distinguishable step."""
    import noise as NZ
    for r in NZ.resolving_power(verbose=False):
        if r["metric"] == "peak_excursion":
            return 3.0 * float(r["unit"])
    raise RuntimeError("no measured floor for peak_excursion")


def _ident(curve, step):
    """Can this parameter be read off the amplitude at all?

    Two conditions, and BOTH are needed. A span of many steps says the observable moves enough to
    be seen. Monotonicity says a given amplitude names ONE parameter value rather than two. A
    driven sheet has a resonance, and either side of it the same amplitude comes from a stiff
    tissue and a soft one -- so a large span with a turning point in it is not an anchor.
    """
    E = np.array([c["E"] for c in curve], float)
    A = np.array([c["amplitude"] for c in curve], float)
    slope = float(np.polyfit(np.log(E), np.log(A), 1)[0])
    d = np.diff(A)
    mono = bool(np.all(d > 0) or np.all(d < 0))
    span = float((A.max() - A.min()) / step)
    return {"fold": float(E.max() / E.min()), "span_steps": span, "monotone": mono,
            "exponent": slope, "n_turning_points": int((np.diff(np.sign(d)) != 0).sum()),
            "invertible": bool(mono and span >= MET.MIN_LEVELS),
            "amplitude_min": float(A.min()), "amplitude_max": float(A.max())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=180)
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--mode", default="full")
    ap.add_argument("--e-lo", type=float, default=40.0)
    ap.add_argument("--e-hi", type=float, default=220.0)
    ap.add_argument("--g-lo", type=float, default=0.5)
    ap.add_argument("--g-hi", type=float, default=1.5)
    ap.add_argument("--sweep-lo", type=float, default=20.0)
    ap.add_argument("--sweep-hi", type=float, default=800.0)
    ap.add_argument("--sweep-n", type=int, default=7)
    ap.add_argument("--gain-lo-s", type=float, default=0.25)
    ap.add_argument("--gain-hi-s", type=float, default=4.0)
    ap.add_argument("--tag", default="p0")
    a = ap.parse_args()

    if a.audit or not a.calibrate:
        rows = audit()
        truth_med = 132.5
        print(f"\n{'=' * 104}\n  THE OLD BOX -- anchored on the naive fit, which attenuation "
              f"shrinks. Planted E in [{PLANTED_LO:g}, {PLANTED_HI:g}].\n{'=' * 104}")
        print(f"  {'config':<40s}{'T':>3s}  {'box E':>21s} {'implied med':>12s}  excluded")
        for r in rows:
            print(f"  {r['config'][:39]:<40s}{r['T']:>3s}  [{r['lo']:8.2f},{r['hi']:9.2f}] "
                  f"{r['implied_median']:>12.1f}  {r['excluded']:>3d}/100"
                  + ("   <-- the truth is outside" if r["excluded"] else ""))
        bad = [r for r in rows if r["excluded"]]
        print(f"\n  {len(bad)}/{len(rows)} configurations exclude some planted modulus; "
              f"the true median is {truth_med:g}")
        if rows:
            m = [r["implied_median"] for r in rows]
            print(f"  the anchor itself ranges {min(m):.1f} to {max(m):.1f} -- a factor of "
                  f"{max(m) / min(m):.0f} driven by nothing but noise in the fit it came from")
    else:
        calibrate(a)
