#!/usr/bin/env python
"""cfl_certify -- does the predicted stability boundary match the measured one?

PHASE 4b. The bound was corrected by argument: the dt cancels, so the CFL number is chi*d and
not dt*chi*d. That argument is worth exactly nothing until the boundary it predicts is the
boundary the engine actually has. A limit nobody has watched fail is not a limit -- it is a
line of code that says "fine", which is precisely what the previous version was for a year.

So this sweeps (chi, d_h) around the predicted wall and asks the engine, on short runs:

    does the chemistry stay finite?      -- the thing the bound claims to predict
    does a pattern actually form?        -- because a bound that only admits DEAD chemistry
                                            is not a stability limit, it is a floor

Both matter. The corrected bound is useless if everything inside it is also inert, and that is
a real risk here: the diffusivity and the reaction were scaled together by the clock fix, so
shrinking chi to satisfy stability also slows the pattern.

WHAT A PASS LOOKS LIKE. Every point predicted stable runs finite, every point predicted unstable
goes non-finite, and the transition sits where the formula says. A single point on the wrong side
falsifies the bound and is worth more than the whole grid agreeing.

    python cfl_certify.py                # the sweep
    python cfl_certify.py --frames 120   # longer, if the transition looks marginal
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "agents"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "prototype", "Tyssue"))

BASE = os.path.join(ROOT, "config", "okuda", "coral_fixed_ball.yaml")


def _engine():
    import plexus.operators                                                     # noqa: F401
    import tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d, tyssue_monolayer       # noqa: F401
    import ckpt, tyssue_shape_to_chem                                           # noqa: F401
    import plexus.schema as S
    from plexus.engine import run as engine_run
    return S, engine_run


def probe(chi, d_h, frames, device="cuda:0"):
    """One short run. Returns (finite, act_max, n_active) -- or None if it could not run."""
    S, engine_run = _engine()
    cfg = yaml.safe_load(open(BASE))
    cfg["general"]["n_frames"] = frames
    cfg["general"]["record_cap"] = frames + 2
    cfg["general"]["name"] = f"cfl_{chi:g}_{d_h:g}"
    for o in cfg["operators"]:
        if o["op"] == "cell_diffuse":
            o["chi"], o["d_h"] = float(chi), float(d_h)
    p = f"/tmp/cfl_{chi:g}_{d_h:g}.yaml"
    yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
    try:
        H, out = engine_run(S.load(p), device=device)
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:40]}"
    ch = out["sets"]["cell"]["state"]["chem"][-1][:, 0]
    live = ch[np.isfinite(ch)]
    finite = bool(np.isfinite(ch).all())
    act = float(np.nanmax(live)) if live.size else float("nan")
    # "a pattern" = some cells clearly above the floor, not a uniform field
    active = int((live > 0.5 * act).sum()) if live.size and act > 1e-6 else 0
    return (finite, act, active), None


def main(frames=80, device="cuda:0"):
    from translate import CFL_LIMIT
    cfg = yaml.safe_load(open(BASE))
    d0 = next(o for o in cfg["operators"] if o["op"] == "cell_diffuse")
    dt = float(cfg["general"]["dt"])
    print("=" * 92)
    print(f"CFL CERTIFICATION -- predicted wall at dt*chi*d_h = {CFL_LIMIT}")
    print(f"base: {os.path.basename(BASE)}  dt={dt}  d_a={d0['d_a']}  ({frames} frames each)")
    print("=" * 92)
    print(f"\n  {'chi':>7}{'d_h':>7}{'predicted':>12}{'CFL':>8}  {'measured':>10}"
          f"{'act_max':>10}{'active':>8}  verdict")

    grid = [(1.3, 0.16), (1.3, 0.5), (1.3, 1.0), (1.3, 2.0),
            (4.0, 0.16), (4.0, 0.5), (4.0, 1.0),
            (0.08, 10.0), (0.05, 10.0), (0.5, 10.0)]
    agree = disagree = 0
    for chi, d_h in grid:
        cfl = dt * chi * max(float(d0["d_a"]), d_h)
        pred = "stable" if cfl <= CFL_LIMIT else "UNSTABLE"
        res, err = probe(chi, d_h, frames, device)
        if res is None:
            print(f"  {chi:>7g}{d_h:>7g}{pred:>12}{cfl:>8.2f}  {'--':>10}{'--':>10}{'--':>8}  {err}")
            continue
        finite, act, active = res
        meas = "finite" if finite else "NON-FINITE"
        ok = (finite and pred == "stable") or (not finite and pred == "UNSTABLE")
        agree, disagree = agree + int(ok), disagree + int(not ok)
        note = "agrees" if ok else "*** BOUND IS WRONG HERE ***"
        if finite and active == 0:
            note += "  (finite but INERT -- no pattern)"
        print(f"  {chi:>7g}{d_h:>7g}{pred:>12}{cfl:>8.2f}  {meas:>10}{act:>10.3f}"
              f"{active:>8}  {note}")

    print(f"\n  {agree} points agree with the bound, {disagree} contradict it")
    print("=" * 92)
    return 1 if disagree else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    raise SystemExit(main(a.frames, a.device))
