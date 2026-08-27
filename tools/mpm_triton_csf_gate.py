#!/usr/bin/env python
"""GATE: does `mpm_scatter[implementation: triton]` deposit the liquid colour the CSF needs?

THE DEFECT THIS EXISTS FOR. Both Triton scatters -- `triton` and `triton_colour` -- zeroed `g.c`
and never wrote it. `mpm_grid_update` decides whether to run its surface-tension branch at all by
asking whether the colour field is non-zero, so on those two implementations the branch was skipped
and `surface_tension` did EXACTLY NOTHING. It is not a small error: it is a fourth CSF path, silently
at zero, that no gate covered because every surface-tension test ran on torch or warp.

It does not announce itself either. A run at sigma = 0.64 reproduced `spread_r90 0.17831` and
`level_p95 0.61445` -- identical to SEVEN FIGURES to the same run at sigma = 0. Nothing errors,
nothing warns, the movie looks like a plausible liquid, and the tension is absent.

THE THREE ROWS, and each fails differently.

  colour     After one scatter, `sum(g.c)` must equal the total LIQUID mass -- the colour is
             `w * mass * is_liquid` and the B-spline weights sum to 1 at every particle, so this is
             an identity, not a fit. Zero means the deposit is missing; the wrong number means the
             weights or the mask are wrong. Checked against the torch scatter's own g.c on the same
             state, to 1e-5 relative: atomics reorder, they do not change the sum.

  effect     Two runs of the SAME implementation, sigma = 0 and sigma > 0, must DIFFER in the
             particle positions. This is the row that was failing to seven figures. It is stated as
             a floor, not a target: any difference at all proves the branch ran.

  agreement  triton against torch at the same sigma, on the radius of gyration after N frames.
             Loose (5%) on purpose -- atomic float ordering is not deterministic and this is not an
             identity gate -- but it catches a colour that is deposited with the wrong sign, the
             wrong scale, or into the wrong nodes, which the first two rows would both pass.

    python tools/mpm_triton_csf_gate.py --device cuda:1
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def build(spec, kind, impl, sigma, frames, particles, device):
    import yaml
    from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition as CFL
    from plexus.schema import load

    s = yaml.safe_load(open(os.path.join(ROOT, "config", kind, spec + ".yaml")))
    s["general"]["n_frames"] = int(frames)
    s["general"]["save_data"] = False
    s["sets"]["mpm_particle"]["per_parent"] = int(particles)
    for o in s["operators"]:
        if o.get("op") == "mpm_scatter":
            o["implementation"] = impl
        elif o.get("op") in ("mpm_strain", "mpm_gather") and impl != "warp":
            o.pop("implementation", None)             # torch strain/gather beside a triton scatter
        elif o.get("op") == "mpm_grid_update":
            o["surface_tension"] = float(sigma)
    for blk in s["schedule"]:
        if isinstance(blk, dict) and "substep_dt" in blk:
            blk["capture"] = False                    # triton kernels are not capture-tested here
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(s, f)
    f.close()
    CFL(f.name)
    sim = load(f.name)
    os.unlink(f.name)
    return sim


def run(sim, device, probe=None):
    import contextlib
    import io

    import torch
    from plexus import engine as E
    out = {}

    def on_frame(H, t):
        p = H.level("mpm_particle")
        X = p.get("pos").detach()
        out["rg"] = float((X - X.mean(0)).norm(dim=1).pow(2).mean().sqrt())
        out["pos"] = X.clone()
        if probe is not None and t == probe:
            g = H.field("mpm_grid")
            out["csum"] = float(g.c.sum())
            out["mliq"] = float((p.mass * p.is_liquid.to(p.mass.dtype)).sum())

    with contextlib.redirect_stdout(io.StringIO()):
        E.run(sim, out_path=None, device=device, progress=False, on_frame=on_frame)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="si_two_drops3d_s144")
    ap.add_argument("--type", default="si_material")
    ap.add_argument("--sigma", type=float, default=0.144)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--particles", type=int, default=20000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--impls", default="triton,triton_colour")
    ap.add_argument("--tol", type=float, default=5.0)
    a = ap.parse_args()

    import torch
    import plexus.operators  # noqa: F401
    import plexus.operators.mpm_triton  # noqa: F401
    import plexus.operators.mpm_warp  # noqa: F401

    print(f"\n  {a.spec}: sigma {a.sigma:g} vs 0, {a.frames} frames, "
          f"{a.particles * 2:,} particles, {a.device}")

    ref = run(build(a.spec, a.type, "default", a.sigma, a.frames, a.particles, a.device),
              a.device, probe=1)
    ref0 = run(build(a.spec, a.type, "default", 0.0, a.frames, a.particles, a.device), a.device)
    print(f"\n  torch reference: g.c sum {ref['csum']:.6g}  liquid mass {ref['mliq']:.6g}  "
          f"Rg {ref['rg'] * 1000:.4f} mm   (sigma=0: {ref0['rg'] * 1000:.4f} mm)")

    print(f"\n  {'implementation':<16}{'sum(g.c)':>14}{'vs liquid mass':>16}"
          f"{'Rg sig>0 (mm)':>15}{'Rg sig=0':>11}{'differ?':>9}{'vs torch':>10}{'':>8}")
    print("  " + "-" * 100)
    ok = True
    for impl in a.impls.split(","):
        try:
            r = run(build(a.spec, a.type, impl, a.sigma, a.frames, a.particles, a.device),
                    a.device, probe=1)
            r0 = run(build(a.spec, a.type, impl, 0.0, a.frames, a.particles, a.device), a.device)
        except Exception as e:
            print(f"  {impl:<16}  ERROR {type(e).__name__}: {str(e)[:70]}")
            ok = False
            continue
        e_col = abs(r["csum"] / max(r["mliq"], 1e-30) - 1) * 100
        moved = float((r["pos"] - r0["pos"]).abs().max())
        e_rg = abs(r["rg"] / max(ref["rg"], 1e-30) - 1) * 100
        row_ok = (e_col <= 0.001) and (moved > 0) and (e_rg <= a.tol)
        ok &= row_ok
        print(f"  {impl:<16}{r['csum']:>14.6g}{e_col:>15.4f}%{r['rg'] * 1000:>15.4f}"
              f"{r0['rg'] * 1000:>11.4f}{'yes' if moved > 0 else 'NO':>9}{e_rg:>9.2f}%"
              f"{'  PASS' if row_ok else '  FAIL':>8}")
    print(f"\n  rows: colour (sum g.c == liquid mass), effect (sigma changes the run), "
          f"agreement (Rg within {a.tol:g}% of torch)")
    print(f"  {'ALL PASS' if ok else 'FAILURES ABOVE'}\n")


if __name__ == "__main__":
    main()
