#!/usr/bin/env python
"""gradient -- prove the gradient is the gradient (Phase 3).

WHY A FINITE DIFFERENCE, AND WHY IT NEEDS THE NOISE WORK FIRST
================================================================================================
`--grad_check` proves a gradient ARRIVES at every learnable tensor. It says nothing about whether
the number is right. The brute-force answer is the derivative of the loss measured by displacing
the parameter and re-running:

        d loss / d p  ~=  ( L(p + h) - L(p - h) ) / 2h

and if the analytic gradient disagrees with that, one of them is wrong.

THE NOISE IS MEASURED FIRST, AND IT LOCATED ITSELF. The same-seed floor showed that two identical
fits diverge -- chirality_match by 0.065 -- so the concern was that L(p + h) and L(p - h) would
differ for two reasons, the displacement and the arithmetic, making a small finite difference
meaningless. Probing the same point repeatedly settles it: **the FORWARD is bit-identical**
(365.65628051757812 every time), so the non-determinism lives in the BACKWARD -- `grid_sample` has
no deterministic backward -- and not in the loss. Finite differences here are clean, and the floor
that actually binds is float32 on a loss of ~366, about 1e-2 on a difference.

That is worth having found rather than assumed: it says the divergence between two identical fits
comes from the gradients, not from the simulation, which is a different and more tractable problem.

Every step size is still judged against a floor, and a step whose signal does not clear it is
reported as saying nothing rather than being quietly averaged in.

The probe runs through the trainer's own forward path -- `--perturb` sets the scalar, `--loss_probe`
prints the loss and exits before the backward -- so the finite difference is not being compared
against a second implementation of the model.

    python gradient.py --device cuda:1
    python gradient.py --device cpu       # deterministic, slow, the referee
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
PY = sys.executable
sys.path.insert(0, HERE)

FIT_SPEC = os.path.join(HERE, "config", "material", "material_aniso_cardio_fit.yaml")
BASE = ["--stiff_src", "siren", "--siren_fibre", "1", "--siren_omega", "5",
        "--allow_nondeterministic_ops", "1", "--learn", "fibre,gain,dur,stiff", "--n_iter", "1"]

# the scalars, and a sensible step for each in its own units
SCALARS = {"f_wl": 0.5, "f_ang": 1e-3, "f_amp": 1e-3, "f_ph": 1e-3,
           "raw_g": 1e-3, "raw_dur": 1e-3}


def _run(extra, device, outdir, timeout=3600):
    env = dict(os.environ, PYTHONPATH=SRC + ":" + os.environ.get("PYTHONPATH", ""))
    cmd = [PY, os.path.join(HERE, "train.py"), FIT_SPEC, "--seed", "7", "--device", device,
           "--outdir", outdir, *BASE, *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    return (r.stdout or "") + (r.stderr or "")


def probe(device, work, perturb=None, tag="p"):
    """The loss at one point, through the trainer's own forward."""
    d = os.path.join(work, tag)
    os.makedirs(d, exist_ok=True)
    extra = ["--loss_probe", "1"] + (["--perturb", perturb] if perturb else [])
    out = _run(extra, device, d)
    m = re.search(r"LOSSPROBE loss=([-\d.eE+]+) r2=([-\d.eE+]+) amp=([-\d.eE+]+)", out)
    if not m:
        return None, (out.strip().splitlines() or ["no output"])[-1][:200]
    return {"loss": float(m.group(1)), "r2": float(m.group(2)), "amp": float(m.group(3))}, None


def analytic(device, work):
    """The gradient the model reports, and the value each scalar sits at."""
    d = os.path.join(work, "analytic")
    os.makedirs(d, exist_ok=True)
    out = _run(["--grad_check", "1"], device, d)
    g = {}
    for m in re.finditer(r"GRADSCALAR (\S+) ([-\d.eE+]+) value ([-\d.eE+]+)", out):
        g[m.group(1)] = {"grad": float(m.group(2)), "value": float(m.group(3))}
    return g, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--work", default=os.path.join(HERE, "_gradient"))
    ap.add_argument("--repeats", type=int, default=3,
                    help="probes of the SAME point, to measure the forward's own noise")
    ap.add_argument("--scalars", nargs="*", default=sorted(SCALARS))
    a = ap.parse_args(argv)
    os.makedirs(a.work, exist_ok=True)

    print(f"\n  1. THE FORWARD'S OWN NOISE -- the same point probed {a.repeats} times on "
          f"{a.device}", flush=True)
    reps = []
    for i in range(a.repeats):
        r, err = probe(a.device, a.work, None, f"rep{i}")
        if not r:
            raise SystemExit(f"baseline probe failed: {err}")
        reps.append(r["loss"])
        print(f"     loss = {r['loss']:.17g}", flush=True)
    L0 = float(np.mean(reps))
    noise = float(np.max(reps) - np.min(reps))
    print(f"     mean {L0:.12g}   spread {noise:.3e}"
          + ("   (bit-reproducible)" if noise == 0 else
             "   <-- every finite difference must clear this"))

    g, raw = analytic(a.device, a.work)
    if not g:
        raise SystemExit("no GRADSCALAR lines; is --grad_check wired?")

    # A RATIO NEAR 1 AT ONE STEP SIZE IS NOT PROOF. A central difference carries an O(h^2)
    # truncation error, so the honest test is whether the ratio CONVERGES to 1 as h shrinks: if it
    # does, the gradient is right and the step was merely coarse; if it plateaus somewhere else,
    # the gradient is wrong and no step size will rescue it. The floor is float32 -- the loss is
    # about 365.7 and single precision resolves ~1e-2 of that -- so h cannot shrink forever, and
    # the signal is checked against that floor as well as against the forward's own noise.
    FLOOR = max(noise, abs(L0) * 2.4e-7 * 4)          # float32 eps, with room for the subtraction
    print(f"\n{'=' * 116}\n  2. ANALYTIC AGAINST BRUTE FORCE -- does the finite difference "
          f"CONVERGE to the gradient as the step shrinks?\n{'=' * 116}")
    print(f"  resolution floor on a difference of losses: {FLOOR:.3e}"
          f"  (forward noise {noise:.1e}, float32 on a loss of {L0:.4g})")
    print(f"\n  {'scalar':<10s} {'analytic':>13s} " +
          "".join(f"{'h=' + f'{f:g}':>21s}" for f in (1.0, 0.25, 0.0625)) + "   verdict")
    rows = {}
    for nm in a.scalars:
        if nm not in g:
            continue
        v, an = g[nm]["value"], g[nm]["grad"]
        cells, seq = [], []
        for k, frac in enumerate((1.0, 0.25, 0.0625)):
            h = SCALARS[nm] * frac
            lp, e1 = probe(a.device, a.work, f"{nm}={v + h}", f"{nm}_p{k}")
            lm, e2 = probe(a.device, a.work, f"{nm}={v - h}", f"{nm}_m{k}")
            if not lp or not lm:
                cells.append(f"{'probe failed':>21s}"); seq.append(None); continue
            d = lp["loss"] - lm["loss"]
            fd = d / (2 * h)
            r = fd / an if an != 0 else float("nan")
            below = abs(d) < FLOOR
            seq.append(None if below else r)
            cells.append(f"{fd:>13.5g}{'*' if below else ' '}{r:>7.3f}")
        good = [r for r in seq if r is not None]
        if not good:
            verdict = "every step drowned -- says nothing"
        elif abs(good[-1] - 1.0) <= 0.10:
            verdict = "AGREES (converged to within 10%)"
        elif len(good) > 1 and abs(good[-1] - 1.0) < abs(good[0] - 1.0):
            verdict = f"converging but not there ({good[-1]:.2f}) -- needs a smaller step"
        elif good[-1] < 0:
            verdict = "WRONG SIGN"
        else:
            verdict = f"DOES NOT CONVERGE (plateaus at {good[-1]:.2f})"
        rows[nm] = {"value": v, "analytic": an, "ratios": seq, "verdict": verdict,
                    "steps": [SCALARS[nm] * f for f in (1.0, 0.25, 0.0625)]}
        print(f"  {nm:<10s} {an:>13.5g} " + "".join(cells) + f"   {verdict}")
    print("  * marks a difference of losses below the resolution floor -- that cell says nothing.")

    ok = [k for k, r in rows.items() if r["verdict"].startswith("AGREES")]
    mute = [k for k, r in rows.items() if "drowned" in r["verdict"]]
    bad = [k for k, r in rows.items() if k not in ok and k not in mute]
    print(f"\n  {len(ok)} of {len(rows)} converge to the analytic gradient within 10%; "
          f"{len(mute)} drowned; {len(bad)} do not.")
    if mute:
        print(f"  DROWNED: {', '.join(mute)} -- a larger step or a deterministic device is needed "
              f"before\n  anything can be said about these. It is not a pass.")
    if bad:
        print(f"  DISAGREE: {', '.join(bad)}")
    print("=" * 116)

    os.makedirs(os.path.join(HERE, "_metrology"), exist_ok=True)
    json.dump({"device": a.device, "baseline_loss": L0, "forward_noise": noise, "rows": rows},
              open(os.path.join(HERE, "_metrology", f"gradient_{a.device.replace(':', '')}.json"),
                   "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
