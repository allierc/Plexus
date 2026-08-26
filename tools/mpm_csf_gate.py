#!/usr/bin/env python
"""GATE: is `surface_tension` a surface tension? Today it is a tension divided by a cell mass.

THE DEFECT, IN ONE LINE. `mpm_scatter` deposits `weight * mass * is_liquid`, so the CSF "colour"
`gc` is a liquid MASS PER NODE -- on an all-liquid spec it is bitwise the mass field, max relative
difference 2.7e-7 -- while Brackbill's f = sigma * kappa * grad(c) is written for a dimensionless
VOLUME FRACTION, 0 in air and 1 in liquid. Nothing divides by the mass of one full liquid cell,
rho * dx^D. Two consequences, and they are the two symptoms:

  * THE GAIN is short by exactly 1/(rho*dx^D). `surface_tension: 60` at n_grid 192 is a physical
    sigma of 8.5e-6 against rho*g*R^2 = 0.092 -- Bond 10,900. A sweep of it does nothing because
    gravity outweighs it by four orders of magnitude. And because dx^D is in the conversion, the
    same yaml number means a DIFFERENT physical tension at every resolution.
  * THE INTERFACE TEST has no scale either: `gmag > 0.02 * gmag.max()` is a percentile of a running
    maximum, and P2G shot noise gives the bulk a |grad c| of the interface's order, so it selects
    98% of occupied nodes. It is an occupancy mask.

NEITHER ALONE IS THE FIX -- the gain alone implodes the run between Bond 10 and Bond 2, because the
bulk noise force it amplifies is harmless today only by being 1e6 too small. This gate tests them
together, and the last row tests that not setting them changes nothing.

THE ROWS.
  identity   `csf_rho: 0` must reproduce the legacy sigma sweep EXACTLY, or the fix is a rewrite.
  mask       fmask as a fraction of OCCUPIED nodes -- not of all nodes, which flatters it by 15x.
  floor      masked nodes with zero mass, and nodes where `csf_mass_floor` binds. Both must be 0:
             a band with a mass clause makes that floor inert instead of load-bearing.
  sweep      spread r90 against Bond number. Must fall MONOTONICALLY while the level RISES -- a
             puddle beading up. Both falling together is a numerical crush, not cohesion.
  grid       THE STRONGEST ROW. At one fixed PHYSICAL sigma, r90 at n_grid 64 / 96 / 128 must agree.
             A quantity whose meaning depends on the mesh is not a material property.

    python tools/mpm_csf_gate.py --rows identity,mask,sweep --device cuda:0
    python tools/mpm_csf_gate.py --rows grid --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CFG = os.path.join(ROOT, "config", "material")


def run(spec, frames, particles, n_grid, device, surf=None, csf_rho=None, csf_band=None,
        csf_smooth=None):
    """One short run; returns the shape metrics and a live read of the CSF block's own masks."""
    import torch
    import yaml

    import plexus.operators  # noqa: F401
    import plexus.operators.mpm_warp  # noqa: F401
    from plexus import engine as E
    from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition
    from plexus.schema import load

    s = yaml.safe_load(open(os.path.join(CFG, spec + ".yaml")))
    ncell = int(s["sets"]["cell"]["n"])
    s["sets"]["mpm_particle"]["per_parent"] = max(1, particles // ncell)
    for fc in (s.get("fields") or {}).values():
        if isinstance(fc, dict) and "n_grid" in fc:
            fc["n_grid"] = int(n_grid)
    for o in s["operators"]:
        if o["op"] == "mpm_grid_update":
            if surf is not None:
                o["surface_tension"] = float(surf)
            for k, v in (("csf_rho", csf_rho), ("csf_band", csf_band), ("csf_smooth", csf_smooth)):
                if v is not None:
                    o[k] = v
    s["general"]["n_frames"] = int(frames)
    s["general"]["record_cap"] = 2
    s["general"]["seed"] = 0
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(s, f); f.close()
    Courant_Friedrichs_Lewy_condition(f.name)
    sim = load(f.name); os.unlink(f.name)

    H, _ = E.run(sim, out_path=None, device=device, progress=False)

    p = H.level("mpm_particle").get("pos").detach()
    up = int((s.get("plotting") or {}).get("up_axis", 1))
    hz = [i for i in range(3) if i != up]
    lvl = float(p[:, up].quantile(0.95))
    r = (p[:, hz] - p[:, hz].mean(0)).norm(dim=1)
    g = H.field("mpm_grid")
    dx, inv_dx, D = float(g.dx), float(g.inv_dx), 3
    gm, gc = g.m, g.c
    occ = gm > 0
    rho = float(s["sets"]["mpm_particle"].get("density", 1.0))
    mfull = rho * dx ** D                                # mass of one full liquid cell

    out = {"n_grid": n_grid, "surf": surf, "csf_rho": csf_rho, "csf_band": csf_band,
           "level_p95": lvl, "spread_r90": float(r.quantile(0.90)),
           "gm_max_over_full": float(gm.max()) / mfull,
           "occupied": int(occ.sum()), "m_full": mfull}

    # the mask, recomputed exactly as the operator computes it, on the operator's own final grid
    if surf and surf > 0:
        c = (gc / mfull if csf_rho else gc).view(*g.shape)
        for _ in range(int(csf_smooth or 0)):
            for k in range(D):
                c = 0.25 * torch.roll(c, 1, k) + 0.5 * c + 0.25 * torch.roll(c, -1, k)
        grad = [(torch.roll(c, -1, k) - torch.roll(c, 1, k)) * (0.5 * inv_dx) for k in range(D)]
        gmag = torch.sqrt(sum(gk * gk for gk in grad))
        if csf_band:
            fm = ((c > csf_band) & (c < 1.0 - csf_band)
                  & (gm.view(*g.shape) > csf_band * mfull))
        else:
            fm = gmag > 0.02 * gmag.max()
        fmf = fm.view(-1)
        out["fmask_of_occupied"] = float((fmf & occ).sum()) / max(int(occ.sum()), 1)
        out["masked_zero_mass"] = float((fmf & ~occ).sum()) / max(int(fmf.sum()), 1)
        out["floor_binds"] = float((fmf & (gm < 1e-8)).sum()) / max(int(fmf.sum()), 1)
    else:
        out.update(fmask_of_occupied=0.0, masked_zero_mass=0.0, floor_binds=0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="material_3d_water_drop")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--particles", type=int, default=290_000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rho", type=float, default=1.0, help="the liquid density = csf_rho")
    ap.add_argument("--band", type=float, default=0.2)
    ap.add_argument("--rows", default="identity,mask,sweep,grid")
    ap.add_argument("--bond", default="100,10,2,1", help="Bond numbers for the sweep row")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import torch
    import yaml
    torch.cuda.set_device(int(a.device.split(":")[-1]) if ":" in a.device else 0)
    rows = a.rows.split(",")
    res = {}
    # Bond = rho g L^2 / sigma with L the body's own half-width, so sigma_phys = rho g L^2 / Bond
    s = yaml.safe_load(open(os.path.join(CFG, a.spec + ".yaml")))
    ty = list(s["sets"]["cell"]["types"].values())[0]
    b = ty.get("block") or ty.get("ball")
    L = (max(b[3] - b[0], b[4] - b[1], b[5] - b[2]) / 2 if b and len(b) >= 6 else 0.2)
    g = next((float(o.get("g", 0.0)) for o in s["operators"] if o.get("op") == "gravity"), 14.0)
    sig1 = a.rho * g * L * L                                       # sigma at Bond 1
    print(f"\n  {a.spec}: L {L:.4f}, g {g:g}, rho {a.rho:g}  ->  sigma at Bond 1 = {sig1:.4g}\n")

    if "identity" in rows:
        print("  ROW `identity`: csf_rho 0 must reproduce the legacy path bit for bit\n")
        print(f"  {'sigma':>8}{'level p95':>12}{'spread r90':>12}{'fmask/occ':>12}")
        for st in (0.0, 60.0, 150.0):
            o = run(a.spec, a.frames, a.particles, 96, a.device, surf=st, csf_rho=0.0)
            res[f"identity_{st}"] = o
            print(f"  {st:>8.4g}{o['level_p95']:>12.5f}{o['spread_r90']:>12.5f}"
                  f"{o['fmask_of_occupied'] * 100:>11.1f}%", flush=True)
        print()

    if "mask" in rows:
        print("  ROW `mask` + `floor`: the interface selector, as a fraction of OCCUPIED nodes\n")
        print(f"  {'colour':>22}{'fmask/occ':>12}{'zero-mass':>12}{'floor binds':>13}")
        for tag, kw in (("raw mass, 2% of max", dict(csf_rho=0.0)),
                        ("fraction, band 0.2", dict(csf_rho=a.rho, csf_band=a.band))):
            o = run(a.spec, a.frames, a.particles, 96, a.device, surf=sig1, **kw)
            res[f"mask_{tag}"] = o
            print(f"  {tag:>22}{o['fmask_of_occupied'] * 100:>11.1f}%"
                  f"{o['masked_zero_mass'] * 100:>11.1f}%{o['floor_binds'] * 100:>12.1f}%",
                  flush=True)
        print()

    if "sweep" in rows:
        print("  ROW `sweep`: r90 must fall MONOTONICALLY while level RISES -- beading, not crushing\n")
        print(f"  {'Bond':>8}{'sigma':>11}{'level p95':>12}{'spread r90':>12}{'vs sigma=0':>12}"
              f"{'gm/full':>10}")
        base = run(a.spec, a.frames, a.particles, 96, a.device, surf=0.0, csf_rho=a.rho)
        res["sweep_0"] = base
        print(f"  {'inf':>8}{0.0:>11.4g}{base['level_p95']:>12.5f}{base['spread_r90']:>12.5f}"
              f"{'--':>12}{base['gm_max_over_full']:>10.2f}", flush=True)
        for bo in [float(x) for x in a.bond.split(",")]:
            st = sig1 / bo
            o = run(a.spec, a.frames, a.particles, 96, a.device, surf=st, csf_rho=a.rho,
                    csf_band=a.band)
            res[f"sweep_{bo}"] = o
            d = (o["spread_r90"] / base["spread_r90"] - 1) * 100
            print(f"  {bo:>8g}{st:>11.4g}{o['level_p95']:>12.5f}{o['spread_r90']:>12.5f}"
                  f"{d:>11.1f}%{o['gm_max_over_full']:>10.2f}", flush=True)
        print()

    if "grid" in rows:
        print(f"  ROW `grid`: ONE physical sigma ({sig1:.4g}, Bond 1) at three resolutions.\n"
              f"  A material property does not depend on the mesh.\n")
        # PARTICLES SCALE WITH CELLS, or this row measures undersampling instead of the mesh.
        # Held at a fixed particle count, n_grid 64 -> 128 takes the fluid from 1.1 particles per
        # cell to 0.14 -- below the 0.5 floor the CFL pass itself warns at -- and r90 then moved
        # 12.6%, of which none is attributable to the tension. At fixed particles-per-cell the two
        # discretisations are actually comparable, which is what "grid independence" has to mean.
        print(f"  {'n_grid':>8}{'cells':>12}{'particles':>12}{'level p95':>12}{'spread r90':>12}"
              f"{'fmask/occ':>12}")
        vals = []
        for ng in (64, 96, 128):
            npar = int(round(a.particles * (ng / 96.0) ** 3))
            o = run(a.spec, a.frames, npar, ng, a.device, surf=sig1, csf_rho=a.rho,
                    csf_band=a.band)
            o["particles"] = npar
            res[f"grid_{ng}"] = o
            vals.append(o["spread_r90"])
            print(f"  {ng:>8}{ng ** 3:>12,}{npar:>12,}{o['level_p95']:>12.5f}"
                  f"{o['spread_r90']:>12.5f}{o['fmask_of_occupied'] * 100:>11.1f}%", flush=True)
        sp = (max(vals) - min(vals)) / (sum(vals) / len(vals)) * 100
        print(f"\n  spread of r90 across an 8x change in cell count: {sp:.1f}%"
              f"   {'PASS' if sp < 10 else 'FAIL'} (threshold 10%)\n")
        res["grid_spread_pct"] = sp

    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=float)
        print(f"  rows -> {a.json}\n")


if __name__ == "__main__":
    main()
