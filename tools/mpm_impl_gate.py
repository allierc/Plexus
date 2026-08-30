#!/usr/bin/env python
"""CORRECTNESS GATE for an alternative `implementation:` of an MPM operator.

    python tools/mpm_impl_gate.py --op mpm_gather      --impl torch_loop27 --device cuda:0
    python tools/mpm_impl_gate.py --op mpm_grid_update --impl warp   --device cuda:0

WHAT IS COMPARED, AND WHY THIS SHAPE OF TEST. Both implementations are run on the SAME live state --
carried by real frames of a real spec, not a random tensor -- and the quantities they write are
compared element-wise:

    mpm_gather        vel, C, pos      (the particle state it advects)
    mpm_grid_update   grid v           (the nodal velocity it solves for)

A run-to-run comparison would not do: MPM is chaotic, so two runs that differ at the last ulp
diverge visibly by frame 200, and a diverged picture cannot distinguish "different rounding" from
"different physics". Comparing ONE call from a shared state can.

WHAT A PASS MEANS. An alternative implementation is not bit-identical by construction -- it reorders
reductions, and in the warp cases it lowers the same arithmetic through a different compiler -- so
the gate asserts a BOUND, stated here before the run: max relative error below 1e-5, which is ~100
float32 eps. Anything larger is a real disagreement, not rounding, and the tool says which field.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

TOL = 1e-5
# op -> (spec default, what the operator writes)
OPS = {
    "mpm_gather": ("si_material/si_waterfall", ("vel", "C", "pos")),
    "mpm_grid_update": ("si_material/si_waterfall", ("grid_v",)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="mpm_gather", choices=sorted(OPS))
    ap.add_argument("--impl", default="torch_loop27")
    ap.add_argument("--spec", default=None)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import torch

    import plexus.operators                                          # noqa: F401
    from plexus.engine import run
    from plexus.models.registry import get_operator
    from plexus.schema import load

    spec = a.spec or OPS[a.op][0]
    typ, name = spec.split("/", 1)
    sim = load(os.path.join(ROOT, "config", typ, name + ".yaml"))
    sim.n_frames = int(a.frames)
    # THE SPEC'S OWN CHOICE FOR THIS OPERATOR IS FORCED TO THE DEFAULT, whatever it declares: the
    # point is to compare the alternative AGAINST the default, and a third implementation carrying
    # the state would be a third answer.
    ospec = [o for o in sim.operators if o.op == a.op][0]
    ospec.impl = None
    params = dict(ospec.params)
    params.setdefault("_at", ospec.on.set if hasattr(ospec.on, "set") else "mpm_particle")
    if ospec.frm:
        params["from"] = ospec.frm

    grabbed = {}

    def hook(H, tick):
        if tick == sim.n_frames:
            grabbed["H"] = H

    with contextlib.redirect_stdout(io.StringIO()):
        run(sim, out_path=None, device=a.device, progress=False, on_frame=hook)
    H = grabbed["H"]
    p = H.level("mpm_particle")
    g = H.field("mpm_grid")

    base = get_operator(a.op)(params, device=a.device)
    alt = get_operator(a.op, implementation=a.impl)(params, device=a.device)
    if type(alt) is type(base):
        raise SystemExit(f"{a.impl!r} did not resolve to a distinct class for {a.op} -- got "
                         f"{type(alt).__name__} for both")

    H.sub_dt = float([b for b in sim.schedule
                      if isinstance(b, dict) and "steps" in b][0]["substep_dt"])
    # EVERY BUFFER EITHER IMPLEMENTATION MAY WRITE, so the second call sees the first's inputs and
    # not the first's outputs.
    snap = {"state": p.state.clone(), "C": p.C.clone(), "gv": g.v.clone(),
            "gm": g.m.clone(), "gmv": g.mv.clone(), "gc": g.c.clone()}

    def once(op):
        p.state.copy_(snap["state"]); p.C.copy_(snap["C"])
        g.v.copy_(snap["gv"]); g.m.copy_(snap["gm"])
        g.mv.copy_(snap["gmv"]); g.c.copy_(snap["gc"])
        op.forward(H)
        pa, pb = p.state_schema["pos"]; va, vb = p.state_schema["vel"]
        return {"pos": p.state[:, pa:pb].clone(), "vel": p.state[:, va:vb].clone(),
                "C": p.C.clone(), "grid_v": g.v.clone()}

    out_b, out_a = once(base), once(alt)

    print(f"\n  {name}  state from frame {sim.n_frames}   {p.n:,} particles, "
          f"{g.m.numel():,} grid nodes")
    print(f"  {a.op}: {type(base).__name__} vs {type(alt).__name__}\n")
    print(f"  {'field':<10}{'max |abs|':>14}{'max |rel|':>14}{'scale':>14}{'':>8}")
    print("  " + "-" * 60)
    ok = True
    for f in OPS[a.op][1]:
        b, l = out_b[f], out_a[f]
        d = (b - l).abs()
        scale = b.abs().max().clamp(min=1e-30)
        rel = float(d.max() / scale)
        ok &= rel < TOL
        print(f"  {f:<10}{float(d.max()):>14.3e}{rel:>14.3e}{float(scale):>14.3e}"
              f"{('PASS' if rel < TOL else 'FAIL'):>8}")
    ident = all(bool((out_b[f] == out_a[f]).all()) for f in OPS[a.op][1])
    print(f"\n  bit-identical: {'yes' if ident else 'no (expected -- see the module docstring)'}")
    print(f"  {'ALL PASS' if ok else 'FAILURES ABOVE'}  (bound: max |rel| < {TOL:g})\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
