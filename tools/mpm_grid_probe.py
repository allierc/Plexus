#!/usr/bin/env python
"""Short benchmark + GRID diagnostics: what does each term actually contribute to the grid?

WHY. A sweep of `surface_tension` from 0 to 150 on material_3d_water_dam_20m moved the centre of
mass by 0.0001 of the box -- i.e. by nothing. A parameter that does nothing is either not reaching
the code, or reaching it and being multiplied by something tiny, and the trajectory cannot tell
those apart. The grid can: every force in MLS-MPM becomes a velocity increment on a node, so the
question "does surface tension matter" is exactly "how big is its increment against gravity's".

WHAT IS MEASURED, from the live grid at the end of a short run:
  * the liquid colour field `gc` -- if the scatter is not depositing it, the CSF branch is skipped
    entirely and sigma is inert no matter what it is set to;
  * `_c_csf`, the operator's own cached predicate, which decides that once and never revisits it;
  * the CSF velocity increment `dt * sigma * kappa * grad(c) * dx^D / gm`, computed here exactly as
    the operator computes it, against `dt * g` -- the ratio is the answer;
  * `fmask`, the interface selector: CSF is applied only where |grad c| > 2% of its max, so a
    small mask means the term is real but almost nowhere.

Deliberately small and short: 2 M particles, 60 frames, so a question can be asked and answered in
under a minute instead of overnight.

    python tools/mpm_grid_probe.py --st 0,60,150 --device cuda:1
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


def run(spec_name, frames, n_particles, n_grid, device, st=None, rho=None, out_json=None):
    import torch
    import yaml

    import plexus.operators  # noqa: F401
    import plexus.operators.mpm_warp  # noqa: F401
    from plexus import engine as E
    from plexus.schema import load

    s = yaml.safe_load(open(os.path.join(CFG, spec_name + ".yaml")))
    ncell = int(s["sets"]["cell"]["n"])
    s["sets"]["mpm_particle"]["per_parent"] = max(1, n_particles // ncell)
    if rho is not None:
        s["sets"]["mpm_particle"]["density"] = float(rho)
    for fc in (s.get("fields") or {}).values():
        if isinstance(fc, dict) and "n_grid" in fc:
            fc["n_grid"] = int(n_grid)
    gu = None
    for o in s["operators"]:
        if o["op"] == "mpm_grid_update":
            if st is not None:
                o["surface_tension"] = float(st)
            gu = o
    s["general"]["n_frames"] = int(frames)
    s["general"]["record_cap"] = 2
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(s, f); f.close()
    # the CFL pass rewrites the temp file in place, so the probe runs at a STABLE step whatever the
    # density is -- otherwise a density sweep measures a blow-up rather than a material
    from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition
    Courant_Friedrichs_Lewy_condition(f.name)
    sim = load(f.name)
    substep_dt = next((x["substep_dt"] for x in yaml.safe_load(open(f.name))["schedule"]
                       if isinstance(x, dict) and "substep_dt" in x), None)
    os.unlink(f.name)

    H, _ = E.run(sim, out_path=None, device=device, progress=False)
    # WHAT THE PARAMETERS ARE SUPPOSED TO DO, stated as two numbers. Density changes how much the
    # liquid compresses under its own weight, so it should move the LEVEL; surface tension holds
    # the body together against spreading, so it should change the FOOTPRINT. Measuring both makes
    # "the sweep does nothing" checkable instead of a visual impression.
    _p = H.level("mpm_particle").get("pos").detach()
    _up = int((s.get("plotting") or {}).get("up_axis", 1))
    _h = [i for i in range(3) if i != _up]
    _lvl = float(_p[:, _up].quantile(0.95))                       # free-surface height
    _c = _p[:, _h].mean(0)
    _r = (_p[:, _h] - _c).norm(dim=1)
    _spread = float(_r.quantile(0.90))                            # radius holding 90% of the fluid
    g = H.field("mpm_grid")
    D = 3
    dx, inv_dx = float(g.dx), float(g.inv_dx)
    dt = float(substep_dt)
    grav = 16.0
    for _o in s["operators"]:                       # from the yaml, not the parsed OpSpec objects
        if _o.get("op") == "gravity":
            grav = float(_o.get("g", grav))

    gm, gc = g.m, g.c
    occ = gm > 0
    out = {"spec": spec_name, "st": st, "rho": rho, "n_grid": n_grid,
           "level_p95": _lvl, "spread_r90": _spread,
           "n_particles": int(H.level("mpm_particle").n), "substep_dt": dt,
           "occupied_nodes": int(occ.sum()),
           "gc_nonzero": int((gc > 0).sum()), "gc_max": float(gc.max()),
           "gm_max": float(gm.max()), "gm_median_occ": float(gm[occ].median()) if occ.any() else 0.0}

    # --- the CSF term, recomputed exactly as MPMGridUpdate does it ---------------------------
    surf = float(st if st is not None else (gu or {}).get("surface_tension", 0.0))
    csf_floor = float((gu or {}).get("csf_mass_floor", 1e-8))
    if surf > 0 and bool((gc > 0).any()):
        c = gc.view(*g.shape)
        grad = [(torch.roll(c, -1, k) - torch.roll(c, 1, k)) * (0.5 * inv_dx) for k in range(D)]
        gmag = torch.sqrt(sum(gk * gk for gk in grad))
        nrm = [gk / (gmag + 1e-6) for gk in grad]
        kappa = -sum((torch.roll(nrm[k], -1, k) - torch.roll(nrm[k], 1, k)) * (0.5 * inv_dx)
                     for k in range(D))
        fmask = (gmag > 0.02 * gmag.max()).to(c.dtype)
        inv_m = (dx ** D) / gm.clamp(min=csf_floor)
        dv = torch.stack([(surf * kappa * grad[k] * fmask).view(-1) * inv_m for k in range(D)], 1)
        dv = dt * dv
        mag = dv.norm(dim=1)
        out.update(csf_active=True,
                   fmask_frac=float(fmask.mean()),
                   kappa_absmax=float(kappa.abs().max()),
                   csf_dv_max=float(mag.max()),
                   csf_dv_mean_on_mask=float(mag[mag > 0].mean()) if (mag > 0).any() else 0.0,
                   floor_binds_frac=float((occ & (gm < csf_floor)).float().sum() /
                                          max(int(occ.sum()), 1)))
    else:
        out.update(csf_active=False, fmask_frac=0.0, csf_dv_max=0.0, csf_dv_mean_on_mask=0.0,
                   floor_binds_frac=0.0)
    out["gravity_dv"] = grav * dt
    out["csf_over_gravity"] = (out["csf_dv_mean_on_mask"] / out["gravity_dv"]
                               if out["gravity_dv"] else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="material_3d_water_dam_20m")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--particles", type=int, default=2_000_000)
    ap.add_argument("--n-grid", type=int, default=96)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--st", default="0,60,150")
    ap.add_argument("--rho", default=None, help="comma list; sweeps density instead of sigma")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    import torch
    torch.cuda.set_device(int(a.device.split(":")[-1]) if ":" in a.device else 0)

    rows = []
    if a.rho:
        print(f"\n  DENSITY SWEEP  {a.spec}  {a.particles:,} particles, {a.frames} frames, "
              f"n_grid {a.n_grid}\n")
        print(f"  {'rho':>7}{'substep_dt':>12}{'substeps':>10}{'LEVEL p95':>11}"
              f"{'SPREAD r90':>12}{'occ nodes':>11}{'gm median':>12}")
        for r in [float(x) for x in a.rho.split(",")]:
            o = run(a.spec, a.frames, a.particles, a.n_grid, a.device, rho=r); rows.append(o)
            print(f"  {r:>7}{o['substep_dt']:>12.3e}{round(0.0036/o['substep_dt']):>10}"
                  f"{o['level_p95']:>11.4f}{o['spread_r90']:>12.4f}"
                  f"{o['occupied_nodes']:>11,}{o['gm_median_occ']:>12.3e}", flush=True)
    else:
        print(f"\n  SURFACE-TENSION SWEEP  {a.spec}  {a.particles:,} particles, {a.frames} frames, "
              f"n_grid {a.n_grid}\n")
        print(f"  {'sigma':>7}{'LEVEL p95':>11}{'SPREAD r90':>12}{'fmask':>9}"
              f"{'|kappa|max':>12}{'CSF dv':>11}{'ratio/g':>10}{'floor binds':>12}")
        for st in [float(x) for x in a.st.split(",")]:
            o = run(a.spec, a.frames, a.particles, a.n_grid, a.device, st=st); rows.append(o)
            print(f"  {st:>7}{o['level_p95']:>11.4f}{o['spread_r90']:>12.4f}"
                  f"{o['fmask_frac'] * 100:>8.2f}%{o.get('kappa_absmax', 0):>12.1f}"
                  f"{o['csf_dv_mean_on_mask']:>11.2e}{o['csf_over_gravity']:>10.1f}"
                  f"{o['floor_binds_frac'] * 100:>11.1f}%", flush=True)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
        print(f"\n  rows -> {a.json}")
    print()


if __name__ == "__main__":
    main()
