#!/usr/bin/env python
"""curve_shape -- describe a metric's TRAJECTORY, not just its endpoint.

WHY
------------------------------------------------------------------------------------------------
Every run already records each metric frame by frame, and every agent in the loop reads only the
endpoints. `metrics.png` -- the plot of those trajectories -- is referenced ZERO times anywhere in
the codebase. It has been drawn for every run since the beginning and read by nobody.

That is not a cosmetic gap, because a scalar cannot distinguish these two runs:

    A:  protrusion climbs to 2.7 and stays there          -> a tube
    B:  protrusion spikes to 2.7 at the moment the mesh tears, then collapses

`protr_peak` is 2.7 for both, and the campaign ranks on `protr_peak`. Worse, the peak is a maximum
over ALL frames with no validity filter, so B scores exactly as well as A -- which means a long
enough search discovers that the cheapest way to score well is to destroy the tissue.

Classifying the SHAPE of a curve is arithmetic, not judgement. So it is computed here,
deterministically, and written into the record BEFORE any agent reads it -- the same discipline as
the Critic running before the Reflection.

THE SHAPES, and what each one is telling you
------------------------------------------------------------------------------------------------
  flat            nothing happened. If this is your headline metric, the run was pointless.
  rising          still climbing at the final frame -- THE RUN WAS TOO SHORT. The final value is
                  not a result, it is wherever we happened to stop.
  converged       settled to a value and stayed. The final value means something.
  peaked          best in the middle, then declined. THE FINAL FRAME IS NOT THE RESULT -- reading
                  the endpoint reports the decay, not the phenomenon.
  exploded        a late super-linear blow-up. Almost always the mesh failing, not biology.
  pinned          the tail is EXACTLY constant -- not converging, but held against a hard limit.
                  This is the signature of a buffer ceiling, and it is the one that cost us a
                  whole overnight study: 32 runs all ended at exactly 1778 cells because that is
                  where the vertex reservoir runs out. A single glance at `cells` classified as
                  `pinned` would have said so immediately.

`pinned` vs `converged` is the distinction that matters most and it is easy: a converging quantity
still wobbles in its last few samples; a quantity held against a hard limit is *bit-identical*
sample after sample.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

SHAPES = ("too_short", "flat", "pinned", "exploded", "peaked", "rising", "converged")


def classify(y, t=None, tail_frac=0.25, flat_eps=1e-3, decline=0.25, rise=0.15,
             explode_mult=3.0):
    """Describe one trajectory. Pure arithmetic; no model, no judgement.

    Returns a dict with the shape plus the numbers the shape was derived from, so the verdict can
    always be audited against its evidence.
    """
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    n = int(ok.sum())
    out = {"n": n, "n_nonfinite": int((~ok).sum())}
    if n < 4:
        return {**out, "shape": "too_short"}
    y = y[ok]
    t = np.arange(len(y), dtype=float) if t is None else np.asarray(t, dtype=float)[ok]

    lo, hi = float(y.min()), float(y.max())
    rng = hi - lo
    scale = max(abs(float(np.median(y))), 1e-12)
    k = max(3, int(round(tail_frac * len(y))))
    tail = y[-k:]
    i_peak = int(np.argmax(y))
    peak_at = float(i_peak / (len(y) - 1))

    out.update(first=float(y[0]), final=float(y[-1]), peak=hi, min=lo,
               peak_at_frac=round(peak_at, 3),
               tail_spread=float(tail.max() - tail.min()))

    # --- flat: nothing moved, relative to its own magnitude
    if rng <= flat_eps * scale:
        return {**out, "shape": "flat"}

    # --- pinned: the tail is EXACTLY constant. A real quantity converging still wobbles; a
    #     quantity held against a hard ceiling repeats bit-for-bit. This is the 1778 signature.
    if float(tail.max() - tail.min()) == 0.0 and len(y) >= 6:
        return {**out, "shape": "pinned", "pinned_at": float(tail[-1])}

    # --- exploded: the last quarter dwarfs everything before it
    head_rng = float(np.ptp(y[:-k])) if len(y) > k else rng
    tail_jump = float(abs(tail[-1] - tail[0]))
    if head_rng > 0 and tail_jump > explode_mult * head_rng:
        return {**out, "shape": "exploded", "tail_jump": tail_jump, "head_range": head_rng}

    # --- peaked: best in the interior, then a real decline
    fell = (hi - float(y[-1])) / max(rng, 1e-12)
    if 0.05 < peak_at < 0.85 and fell > decline:
        return {**out, "shape": "peaked", "declined_by": round(float(fell), 3)}

    # --- rising vs converged: is it still going anywhere?
    span = max(float(t[-1] - t[-k]), 1e-12)
    slope = float(np.polyfit(t[-k:], tail, 1)[0])
    moved = abs(slope) * span / max(rng, 1e-12)
    out["tail_slope"] = slope
    out["tail_moved_frac_of_range"] = round(moved, 3)
    if moved > rise:
        return {**out, "shape": "rising" if slope > 0 else "peaked"}
    return {**out, "shape": "converged"}


# --------------------------------------------------------------------------- run-level
def classify_npz(path, keys=None):
    """Classify every series in an .npz. Returns {name: verdict}."""
    z = np.load(path)
    t = z["t"] if "t" in z.files else (z["frame"] if "frame" in z.files else None)
    out = {}
    for k in (keys or z.files):
        if k in ("t", "frame"):
            continue
        v = z[k]
        if v.ndim != 1 or v.shape[0] < 2:
            continue
        out[k] = classify(v, t if (t is not None and len(t) == len(v)) else None)
    return out


def evidence_horizon(shapes, series, t=None, n_bar=20, first_bar=1):
    """The first frame at which the mesh stops being trustworthy, as an ABSOLUTE CELL COUNT.

    A run is not all-or-nothing: if the mesh fails at frame 380 of 400, the first 379 frames are
    sound physics. So the record carries a HORIZON, every measurement is taken before it, and
    "final" means the last valid frame.

    WHY A COUNT AND NOT A FRACTION (Cedric, 31 July -- the first version used hollow_frac > 0.05).
    A broken cell is a LOCAL topology violation: the physics is wrong there, and having more
    healthy cells elsewhere does not make it less wrong. Worse, a fractional bar gets MORE
    PERMISSIVE AS THE TISSUE GROWS, because the denominator grows with it. Measured on
    r01_03_5e3159_3, the 5% bar meant:

        frame   0   1431 cells -> tolerated  72 broken cells
        frame 461   1966 cells -> tolerated  98
        frame 807   2389 cells -> tolerated 119

    So the runs that grow the most -- the ones we care about -- were granted the most damage. An
    absolute bar of ~20 puts this run's horizon at frame 461 instead of 692.

    WHICH COUNT. Prefer `broken_n`: cells that are genuinely under-connected or whose ring is not
    a valid polygon. Do NOT threshold the legacy blend if the split is available -- the blend also
    counts folded caps and just-divided slivers, and a dividing tissue always has slivers, so a bar
    of 20 on the blend would fire almost immediately and the rule would simply stop being applied.
    An over-strict rule is its own failure mode.

    THE BLEND IS NOT AN ACCEPTABLE FALLBACK, AND ASSUMING IT WAS COST ME A WRONG VERDICT.
    I first reasoned: the blend counts more cells than `broken_n`, so a blend-based horizon fires
    early -- pessimistic, therefore safe. That is wrong. The blend does not over-count damage; it
    counts a DIFFERENT THING. Measured on r01_03_5e3159_3:

        corr(hollow_n, n_tip) = +0.971

    The blended "hollow" count tracks the number of TIP CELLS almost exactly. Over that run the
    tube grows monotonically (length 2.96 -> 19.00) and narrows smoothly (diameter 2.67 -> 1.55)
    with no discontinuity anywhere -- a healthy tube, confirmed by watching the movie -- while the
    blend climbs from 4 to 162. It was counting the tube.

    So the bias does not run toward caution, it runs AGAINST THE TARGET PHENOTYPE: the better the
    tube, the more "damage" is reported. On that basis I condemned 13 of 24 archived runs as
    "peaked after the mesh broke". That verdict is withdrawn.

    Therefore: if only the blend is available, this returns NO HORIZON and says why. A number
    computed from the wrong quantity is worse than no number, because it will be used.

    `broken_n` is the right key precisely because it is TOPOLOGICAL -- under-connected cells and
    rings that are not valid polygons. Curvature and cell size cannot manufacture it, so a tube
    cannot look like damage to it.
    """
    if "broken_n" not in series:
        return {"horizon": None, "counted": None,
                "why": ("no `broken_n` recorded. REFUSING to fall back to the legacy blended "
                        "`hollow_n`: it correlates with the tip-cell count at r=+0.97, i.e. it "
                        "counts the tube, not the damage. Re-run with the three failure modes "
                        "separated. A horizon from the wrong quantity is worse than none.")}
    key = "broken_n"
    y = np.asarray(series[key], dtype=float)
    tt = np.arange(len(y)) if t is None else np.asarray(t)
    out = {"criterion": f"{key} >= {n_bar} cells", "counted": key}

    dmg = np.where(y >= first_bar)[0]
    out["first_damage"] = int(tt[dmg[0]]) if len(dmg) else None

    bad = np.where(y >= n_bar)[0]
    if not len(bad):
        return {**out, "horizon": int(tt[-1]), "horizon_idx": len(y) - 1, "complete": True,
                "why": f"{key} never reached {n_bar} (max {int(np.nanmax(y))})"}
    i = int(bad[0])
    return {**out, "horizon": int(tt[i]), "horizon_idx": i, "complete": False,
            "valid_frac": round(i / max(len(y) - 1, 1), 3),
            "why": f"{key} first reached {n_bar} at frame {int(tt[i])} "
                   f"(first damage at frame {out['first_damage']})"}


def report(run_dir, write=True):
    """Classify a run's curves and write curves.json beside them."""
    out = {"run": os.path.basename(run_dir.rstrip("/"))}
    series, t = {}, None

    mech = os.path.join(run_dir, "mechanics.npz")
    if os.path.exists(mech):
        out["mechanics"] = classify_npz(mech)
        z = np.load(mech)
        for k in z.files:
            series.setdefault(k, z[k])
        t = z["t"] if "t" in z.files else None

    mnpz = os.path.join(run_dir, "metrics.npz")
    mjson = os.path.join(run_dir, "metrics.json")
    if os.path.exists(mnpz):
        out["metrics"] = classify_npz(mnpz)
        z = np.load(mnpz)
        for k in z.files:
            series[k] = z[k]
        t = z["frame"] if "frame" in z.files else t
    elif os.path.exists(mjson):
        d = json.load(open(mjson))
        rows = d.get("series", [])
        if rows:
            cols = {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in rows[0]}
            t = cols.get("frame")
            out["metrics"] = {k: classify(v, t) for k, v in cols.items() if k != "frame"}
            series.update(cols)

    out["horizon"] = evidence_horizon(out.get("metrics", {}), series, t)

    # The two series are sampled at DIFFERENT rates (mechanics 24, metrics 40 on the run this was
    # written against). They carry their own frame column so they are alignable, but nothing
    # currently aligns them -- flag it rather than silently comparing index to index.
    lens = {n: len(v) for n, v in (("mechanics", series.get("force_mean", [])),
                                   ("metrics", series.get("protr", []))) if len(v)}
    if len(set(lens.values())) > 1:
        out["sampling_mismatch"] = {**lens,
                                    "note": "different sample counts; align on the frame column"}
    if write:
        json.dump(out, open(os.path.join(run_dir, "curves.json"), "w"), indent=1, default=float)
    return out


def summarise(rep, keys=("protr", "hollow_frac", "cells", "tube_diam", "force_mean",
                         "tension_mean", "n_cells")):
    """One line per interesting curve -- what goes into an agent's prompt."""
    lines = []
    for grp in ("metrics", "mechanics"):
        for k, v in (rep.get(grp) or {}).items():
            if keys and k not in keys:
                continue
            extra = ""
            if v["shape"] == "pinned":
                extra = f" at {v.get('pinned_at')}  <-- HELD AGAINST A LIMIT, not converged"
            elif v["shape"] == "peaked":
                extra = (f", best at {v['peak_at_frac']:.0%} through the run "
                         f"(peak {v['peak']:.3g} -> final {v['final']:.3g})"
                         f"  <-- the final frame is NOT the result")
            elif v["shape"] == "rising":
                extra = f" (still climbing at the end: {v['final']:.3g})  <-- run may be too short"
            elif v["shape"] == "exploded":
                extra = f" (late blow-up to {v['final']:.3g})  <-- suspect the mesh, not biology"
            lines.append(f"  {k:16} {v['shape']:10}{extra}")
    h = rep.get("horizon") or {}
    if h.get("first_damage") is not None:
        lines.append(f"  first damaged cell at frame {h['first_damage']}")
    if h.get("horizon") is not None and not h.get("complete", False):
        lines.append(f"  EVIDENCE HORIZON  frame {h['horizon']} "
                     f"({h.get('valid_frac', 0):.0%} of the run is trustworthy) -- {h['why']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    OK, fails = "\033[92mok\033[0m", []

    def eq(got, want, what):
        if got != want:
            fails.append(f"{what}: got {got!r} want {want!r}")
        print(f"  [{OK if got == want else 'FAIL'}] {what:52} -> {got}")

    n = 40
    x = np.arange(n)
    eq(classify(np.ones(n))["shape"], "flat", "a constant is flat")
    eq(classify(np.r_[np.linspace(1, 5, 20), np.full(20, 5.0)])["shape"], "pinned",
       "tail EXACTLY constant -> pinned, not converged")
    eq(classify(np.linspace(1, 5, n))["shape"], "rising", "a straight climb is rising")
    eq(classify(5 - 4 * np.exp(-x / 5.0) + 1e-6 * np.sin(x))["shape"], "converged",
       "an asymptote that still wobbles is converged")
    eq(classify(np.r_[np.linspace(1, 3, 12), np.linspace(3, 1.2, 28)])["shape"], "peaked",
       "up then down is peaked")
    eq(classify(np.r_[np.linspace(1, 1.4, 30), np.linspace(1.4, 40, 10)])["shape"], "exploded",
       "a late blow-up is exploded")
    eq(classify(np.array([1.0, 2.0]))["shape"], "too_short", "two points is not a curve")

    print("\nthe case this exists for -- 32 runs that all ended at 1778:")
    cells = np.r_[np.linspace(150, 1778, 25), np.full(15, 1778.0)]
    v = classify(cells)
    eq(v["shape"], "pinned", "cells_end 1778 classifies as pinned")
    print(f"     pinned_at = {v['pinned_at']}  <- the vertex buffer, not the biology")

    print("\nplateau vs terminal spike -- identical peak, opposite meaning:")
    plateau = np.r_[np.linspace(1, 2.7, 20), 2.7 - 0.01 * np.abs(np.sin(np.arange(20)))]
    spike = np.r_[np.linspace(1, 1.3, 36), [1.5, 2.0, 2.4, 2.7]]
    eq(float(np.max(plateau).round(2)), 2.7, "same peak (plateau)")
    eq(float(np.max(spike).round(2)), 2.7, "same peak (spike)")
    eq(classify(plateau)["shape"], "converged", "  plateau -> converged")
    eq(classify(spike)["shape"], "exploded", "  spike   -> exploded")

    print("\nhorizon on real data:")
    rd = "/workspace/Plexus/log/okuda/r01_03_5e3159_3"
    if os.path.isdir(rd):
        rep = report(rd, write=False)
        print(summarise(rep))
    else:
        print("  (run dir absent, skipped)")

    print("\n" + ("curve_shape OK" if not fails else f"{len(fails)} FAILURES:\n  "
                                                     + "\n  ".join(fails)))
    raise SystemExit(1 if fails else 0)
