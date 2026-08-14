"""sweep_lr_span -- how hard must LR pull, and how stiff must it be, to move the eye 20-25 deg?

    python sweep_lr_span.py --model A --device cuda:1

Model A abducts about 5 degrees on a full LR command. The target is 20-25. Two knobs
are swept TOGETHER, because separately neither answers the question:

    amplitude  A   the active stress the muscle generates (`muscle_contract.amplitude`,
                   scaled here for LR alone via its `strength` entry)
    stiffness  E   the muscle's own passive Young's modulus

A muscle shortens until its passive tension balances its active stress, so the STEADY
SHORTENING is set by the ratio A/E, while the FORCE it can deliver against the globe and
the orbital fat goes as A x cross-section. Raising A alone therefore does not simply give
more travel -- past a point the strap collapses on itself and crushes the sclera (the
prototype has done exactly this: archive/t02_viz_fix). Raising E alone stiffens the muscle
and gives less travel. The span lives on a ridge in the (A, E) plane, and that is what
this sweeps.

Each cell is a SETTLED measurement, not a peak: LR is held at full command for `hold_s`
(default 1.8 s, above the plant's 1.28 s settling time) and the gaze is averaged over the
last quarter of the hold. A sweep of transient peaks would just rank overshoot.

Writes `archive/eye_<M>/sweep_lr_span.json` (every cell, with its settled gaze and
whether the globe stayed intact) and renders the best cell as a movie in the standard
template.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np
import yaml

import plexus.operators            # noqa: F401
import eye_ops                     # noqa: F401
import muscle_ops                  # noqa: F401
import probe_ops
import eye_anatomy as EA
import run_eye
import render_eye_vtk
from plexus.schema import load as load_spec
from run_staircase import base_spec, settled

ARCHIVE = os.path.join(HERE, "archive")

# A/E is what sets the shortening, so the grid is laid out to span that ratio by more
# than an order of magnitude while keeping the delivered force sane at both ends.
GAIN = [1.0, 2.0, 4.0, 7.0]              # multiplier on LR's active stress
YOUNGS = [240.0, 480.0, 960.0]           # the muscle's passive modulus


def cell(model, gain, youngs, device, hold_s=1.8, lead_s=0.4, tail_s=0.3, stride=6,
         muscle="LR", keep_cap=False):
    """One (gain, youngs) point: hold LR at full command, report the SETTLED gaze."""
    d = os.path.join(ARCHIVE, f"eye_{model}")
    spec, src = base_spec(model)
    dt = float(spec["general"]["dt"])
    mi = EA.MUSCLE_KEYS.index(muscle)
    drv = next((o for o in spec["operators"]
                if o["op"] in ("oculomotor_drive", "muscle_probe")), {})
    tonic = float(drv.get("tonic", 0.14))
    hold, lead, tail = (int(round(s / dt)) for s in (hold_s, lead_s, tail_s))
    s_spec = probe_ops.staircase_spec(spec, mi, levels=(1.0,), hold=hold, lead=lead,
                                      tail=tail, tonic=tonic)
    for o in s_spec["operators"]:
        if o["op"] == "muscle_contract":
            w = list(o.get("strength", [1.0] * EA.N_MUSCLE))
            w[mi] = float(w[mi]) * float(gain)          # LR only: this is a sweep of ONE muscle
            o["strength"] = w
        elif o["op"] == "muscle_morphogenesis":
            o["youngs"] = float(youngs)
    s_spec["sets"]["muscle"]["types"]["muscle"]["youngs"] = float(youngs)
    tag = f"g{gain:g}_E{youngs:g}"
    path = os.path.join("/tmp", f"sweep_{model}_{muscle}_{tag}.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(s_spec, fh, sort_keys=False)

    sim = load_spec(path)
    prb = probe_ops.MuscleProbeStaircase({"muscle": mi, "levels": [1.0], "hold": hold,
                                          "lead": lead, "tail": tail, "tonic": tonic})
    t0 = time.time()
    _, cap = run_eye.capture_run(sim, device, stride=stride)
    t = np.asarray(cap["frame"]) * dt
    g = np.asarray(cap["gaze"])
    base, _, _ = settled(t, g, 0.0, lead * dt, frac=0.5)
    mean, sd, ptp = settled(t, g, lead * dt, (lead + hold) * dt)
    ln = np.asarray(cap["length"])[:, mi]
    rest = np.asarray(cap["rest_length"])
    rest = float(rest[0][mi] if np.ndim(rest) > 1 else rest[mi])
    # a globe that has been crushed rather than rotated shows up as its shell losing
    # radius: `radius_spread` is the scatter of the shell's radii, and it grows when the
    # sclera is being deformed instead of turned
    rad = np.asarray(cap["radius"]) if "radius" in cap else np.zeros(len(t))
    spread = np.asarray(cap["radius_spread"]) if "radius_spread" in cap else np.zeros(len(t))
    out = dict(gain=gain, youngs=youngs, A_over_E=round(60.0 * gain / youngs, 4),
               abduction_deg=round(float(mean[0] - base[0]), 3),
               gaze_deg=[round(float(v), 3) for v in (mean - base)],
               settle_ptp_deg=[round(float(v), 4) for v in ptp],
               shortening_pct=round(100.0 * (1.0 - float(np.median(ln[-max(len(ln)//5, 1):])) / rest), 2),
               radius_drift_pct=round(100.0 * float(rad[-1] / max(rad[0], 1e-9) - 1.0), 3),
               shell_spread_growth=round(float(spread[-1] / max(spread[0], 1e-9) - 1.0), 3),
               seconds=round(time.time() - t0, 1))
    print("  gain %4.1f  E %6.0f  A/E %5.3f -> abduction %7.2f deg  (settled p-p %.3f, "
          "shortening %5.1f%%, radius drift %+.2f%%)  [%.0f s]"
          % (gain, youngs, out["A_over_E"], out["abduction_deg"], out["settle_ptp_deg"][0],
             out["shortening_pct"], out["radius_drift_pct"], out["seconds"]), flush=True)
    return out, (cap if keep_cap else None), float(sim.dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="A")
    ap.add_argument("--muscle", default="LR")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--gains", type=float, nargs="*", default=GAIN)
    ap.add_argument("--youngs", type=float, nargs="*", default=YOUNGS)
    ap.add_argument("--hold-s", type=float, default=1.8)
    ap.add_argument("--target", type=float, nargs=2, default=[20.0, 25.0])
    ap.add_argument("--no-movie", action="store_true")
    a = ap.parse_args()

    d = os.path.join(ARCHIVE, f"eye_{a.model}")
    rows = []
    print(f"[sweep] {a.model}/{a.muscle}: {len(a.gains)}x{len(a.youngs)} cells, "
          f"hold {a.hold_s}s, target {a.target[0]}-{a.target[1]} deg", flush=True)
    for E in a.youngs:
        for gv in a.gains:
            try:
                r, _, _ = cell(a.model, gv, E, a.device, hold_s=a.hold_s, muscle=a.muscle)
                rows.append(r)
            except Exception as e:
                print(f"  gain {gv} E {E} FAILED: {type(e).__name__}: {e}", flush=True)
                rows.append(dict(gain=gv, youngs=E, failed=f"{type(e).__name__}: {e}"))
            with open(os.path.join(d, f"sweep_{a.muscle}_span.json"), "w") as fh:
                json.dump(dict(model=a.model, muscle=a.muscle, hold_s=a.hold_s,
                               target_deg=a.target, cells=rows), fh, indent=2)

    ok = [r for r in rows if "abduction_deg" in r]
    if not ok:
        print("[sweep] every cell failed", flush=True)
        return
    lo, hi = a.target
    mid = 0.5 * (lo + hi)
    # the best cell is the one nearest the middle of the target band that did not also
    # deform the globe: travel bought by crushing the sclera is not travel
    clean = [r for r in ok if abs(r["radius_drift_pct"]) < 2.0] or ok
    best = min(clean, key=lambda r: abs(r["abduction_deg"] - mid))
    print("\n[sweep] best: gain %.1f, E %.0f -> %.2f deg abduction (target %g-%g)"
          % (best["gain"], best["youngs"], best["abduction_deg"], lo, hi), flush=True)
    print("[sweep] %d of %d cells reached the target band"
          % (sum(lo <= r["abduction_deg"] <= hi for r in ok), len(ok)), flush=True)

    if not a.no_movie:
        print("[sweep] rendering the best cell", flush=True)
        r, cap, dt = cell(a.model, best["gain"], best["youngs"], a.device, hold_s=a.hold_s,
                          muscle=a.muscle, keep_cap=True)
        render_eye_vtk.render(cap, dt,
                              os.path.join(d, f"sweep_{a.muscle}_best.mp4"),
                              os.path.join(d, f"sweep_{a.muscle}_best_strip.png"))
        with open(os.path.join(d, f"sweep_{a.muscle}_best.json"), "w") as fh:
            json.dump(r, fh, indent=2)
    print("[sweep] ->", os.path.join(d, f"sweep_{a.muscle}_span.json"), flush=True)


if __name__ == "__main__":
    main()
