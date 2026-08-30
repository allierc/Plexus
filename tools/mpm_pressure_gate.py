#!/usr/bin/env python
"""GATE: does the liquid carry the hydrostatic pressure p = rho g d, in pascals?

WHY THIS NEEDS ITS OWN RUN. Pressure is not a recorded quantity. It is not in `trajectory.npz` and
it is not a grid channel -- it lives in the deformation gradient, p = K(1 - J) with J = det(F), and
`F` is per-particle state the recorder does not store. A first attempt at this test tried to
reconstruct J from the shape of the settled column instead, and that is CIRCULAR: inverting the
linear hydrostatic profile to get a pressure returns rho*g*d by construction, whatever the run did.
It measured the arithmetic, not the simulation, and its numbers are discarded.

So the run happens here, with a hook that reads `p.F` live.

THE CLOSED FORM, and it is the most direct statement that a run is in SI at all:

    p(d) = rho g d          d the depth below the free surface, p in PASCALS

Nothing is fitted. rho comes from the spec, g comes from the spec, d is measured, and the pressure
is K(1 - det F) with K the declared bulk modulus. At rho 1000, g 9.81 and d 20 mm the answer is
196.2 Pa and there is nowhere to hide.

WHAT MAKES IT FAIL, so a pass means something:
  * a settled column is required -- the profile is hydrostatic only at rest, and a sloshing one
    carries dynamic pressure the formula knows nothing about. The tool reports the drift and
    refuses to grade an unsettled run.
  * the TOP of the column is not hydrostatic either: the free surface is one cell thick and the
    shallowest bin sits inside it, so bins shallower than 2*dx are reported but not graded.
  * K must be large enough that 1 - J is resolvable in float32. At K = 1e5 and d = 20 mm,
    1 - J = 1.96e-3, which is 16,000 float32 epsilons -- fine. At water's own 2.2 GPa in a 0.1 m
    box it would be 1.5 epsilons, and this test would be measuring quantisation noise.

    python tools/mpm_pressure_gate.py --spec si_gate --frames 1200 --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="si_gate")
    ap.add_argument("--type", default="si_material")
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--particles", type=int, default=250000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--n-grid", type=int, default=0, help="override n_grid (ppc held)")
    ap.add_argument("--drag", type=float, default=-1.0, help="override mpm_scatter drag")
    ap.add_argument("--liquid-volume", default="",
                    help="det | trace -- how a liquid advances J (forces the torch strain)")
    ap.add_argument("--contact-cells", type=float, default=-1.0,
                    help="override wall_contact_cells on gather/grid_update")
    ap.add_argument("--wall-cells", type=float, default=3.84,
                    help="width of the wall-contact band, in cells (mpm_ops default)")
    a = ap.parse_args()

    import numpy as np
    import torch
    import yaml

    import plexus.operators  # noqa: F401
    import plexus.operators.mpm_warp  # noqa: F401
    from plexus import engine as E
    from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition as CFL
    from plexus.schema import load

    s = yaml.safe_load(open(os.path.join(ROOT, "config", a.type, a.spec + ".yaml")))
    s["general"]["n_frames"] = int(a.frames)
    s["general"]["save_data"] = False
    # RESIZING THE PARTICLE COUNT MUST HOLD THE DENSITY, and on a `block:` spec it does not do so by
    # itself. With a block the positions fill the declared box whatever N is, but each particle
    # still carries `p_vol = particle_mass / density`, so N * p_vol stops equalling the box volume
    # the moment N changes: the same water is represented by more or fewer, unchanged, particles.
    # The stress term is proportional to p_vol, so a `--particles` flag meant to make the gate
    # cheaper would have silently rescaled the material's stiffness per unit volume and the
    # measured pressure with it. Scaling particle_mass by the same factor keeps N * p_vol fixed.
    # REFINING THE GRID MUST HOLD PARTICLES-PER-CELL, or the convergence study measures two things
    # at once. n_grid up by k means k^3 more cells, so the particle count goes up by k^3 and the
    # per-particle mass down by the same factor to keep the density and the block volume fixed.
    if a.n_grid:
        _fk = next(fc for fc in s["fields"].values() if isinstance(fc, dict) and "n_grid" in fc)
        _k = float(a.n_grid) / float(_fk["n_grid"])
        _fk["n_grid"] = int(a.n_grid)
        a.particles = int(round(int(s["sets"]["mpm_particle"]["per_parent"]) * _k ** 3))
    if a.contact_cells >= 0:
        for _o in s["operators"]:
            if _o.get("op") in ("mpm_gather", "mpm_grid_update"):
                _o["wall_contact_cells"] = float(a.contact_cells)
    if a.liquid_volume:
        # BOTH ARMS RUN THE SAME KERNEL. `liquid_volume` is implemented in the torch MPMStrain, so
        # comparing it against a warp-strain baseline would be comparing two things at once.
        for _o in s["operators"]:
            if _o.get("op") == "mpm_strain":
                _o.pop("implementation", None)
                _o["liquid_volume"] = a.liquid_volume
    if a.drag >= 0:
        for _o in s["operators"]:
            if _o.get("op") == "mpm_scatter":
                _o["drag"] = float(a.drag)
    if int(a.particles) != int(s["sets"]["mpm_particle"]["per_parent"]):
        _f = float(s["sets"]["mpm_particle"]["per_parent"]) / float(a.particles)
        _pm = s["sets"]["mpm_particle"].get("particle_mass")
        if _pm is not None:
            s["sets"]["mpm_particle"]["particle_mass"] = float(_pm) * _f
    s["sets"]["mpm_particle"]["per_parent"] = int(a.particles)
    rho = float(s["sets"]["mpm_particle"]["density"])
    K = float(list(s["sets"]["cell"]["types"].values())[0]["bulk_modulus"])
    g = next(float(o["g"]) for o in s["operators"] if o.get("op") == "gravity")
    W = float(s["general"]["world"][1])
    dx = W / int(list(s["fields"].values())[0]["n_grid"])
    up = int((s.get("plotting") or {}).get("up_axis", 1))

    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(s, f); f.close()
    CFL(f.name)
    sim = load(f.name); os.unlink(f.name)

    # keep the last N frames of (height, pressure) so the profile can be averaged over the tail
    keep, tail = [], max(8, a.frames // 20)
    # IS IT CONVERGED, OR JUST SLOW? p = K(1 - det F) is not an equilibrium solve -- F is ADVECTED by
    # the grid velocity gradient, so J holds whatever compression the transient happened to deposit
    # and only approaches the hydrostatic value as fast as the column can actually creep there.
    # A single end-of-run number cannot tell an under-converged run from a wrong one, and guessing
    # between them is how the wall-contact and drag hypotheses both got tested and both refuted.
    # So the slope is fitted at checkpoints THROUGH the run and printed as a curve.
    chk, n_chk = [], 12
    _at = {max(1, int(round(a.frames * (i + 1) / n_chk))) for i in range(n_chk)}

    lat = [k for k in range(3) if k != up][:2] if int(s["general"].get("dim", 3)) == 3 else \
        [k for k in range(2) if k != up]

    def _profile(H):
        p = H.level("mpm_particle")
        J = torch.linalg.det(p.F.detach())
        y = p.get("pos")[:, up].detach().cpu().numpy()
        pr = (K * (1.0 - J)).cpu().numpy()
        t0 = float(np.quantile(y, 0.999))
        ds = [5, 10, 15, 20, 25]
        ps = []
        for d_mm in ds:
            d = d_mm / 1000.0
            b = (y > t0 - d - dx) & (y < t0 - d + dx)
            ps.append(float(np.median(pr[b])) if b.sum() > 200 else np.nan)
        if any(np.isnan(ps)):
            return None
        return np.polyfit(ds, ps, 1), t0

    def on_frame(H, t):
        if t in _at:
            r = _profile(H)
            if r is not None:
                chk.append((t, r[0][0] * 1000, r[1]))
        if t < a.frames - tail:
            return
        p = H.level("mpm_particle")
        J = torch.linalg.det(p.F.detach())
        X = p.get("pos").detach()
        y = X[:, up]
        # DISTANCE TO THE NEAREST LATERAL WALL, so the profile can be split into bulk and boundary
        # layer. The wall-contact band is `wall_contact_cells * dx` wide on each side -- 3.84 cells,
        # which is 4% of a unit box at n_grid 96 and 6% of a 0.1 m box at n_grid 64 -- and every
        # particle inside it is in permanent contact. On a column that fills its footprint that is
        # a fifth of the fluid, and a fifth of the fluid being held by the wall is not a small
        # correction to a hydrostatic measurement.
        d_w = torch.stack([torch.minimum(X[:, k], torch.as_tensor(W, device=X.device) - X[:, k])
                           for k in lat], 0).min(0).values
        keep.append((y.cpu().numpy().copy(), (K * (1.0 - J)).cpu().numpy().copy(),
                     d_w.cpu().numpy().copy()))

    E.run(sim, out_path=None, device=a.device, progress=False, on_frame=on_frame)

    Y = np.concatenate([k[0] for k in keep])
    P = np.concatenate([k[1] for k in keep])
    DW = np.concatenate([k[2] for k in keep])
    top = float(np.quantile(Y, 0.999))
    lvl = [float(np.quantile(k[0], 0.999)) for k in keep]
    drift = abs(lvl[-1] - lvl[0])
    band = a.wall_cells * dx                        # the wall-contact band, in metres

    _sl = np.polyfit([5, 10, 15, 20, 25], [float(np.median(P[(Y > top - d / 1000.0 - dx)
                     & (Y < top - d / 1000.0 + dx)])) for d in (5, 10, 15, 20, 25)], 1)
    print(f"\n  {a.spec}: rho {rho:g} kg/m^3, g {g:g} m/s^2, K {K:.3g} Pa, dx {dx * 1000:.3f} mm, "
          f"n_grid {int(list(s['fields'].values())[0]['n_grid'])}, N {int(a.particles):,}")
    if chk:
        print(f"\n  CONVERGENCE -- dp/dd fitted at checkpoints through the run "
              f"(rho*g = {rho * g:.1f} Pa/m)\n")
        print(f"  {'frame':>8}{'sim t (s)':>12}{'dp/dd (Pa/m)':>15}{'vs rho*g':>11}"
              f"{'free surf (mm)':>17}")
        print("  " + "-" * 64)
        for _t, _m, _s in chk:
            print(f"  {_t:>8}{_t * float(s['general']['dt']):>12.4f}{_m:>15.1f}"
                  f"{100 * (_m / (rho * g) - 1):>10.2f}%{_s * 1000:>17.3f}")
    print(f"  FITTED dp/dd = {_sl[0] * 1000:.1f} Pa/m against rho*g = {rho * g:.1f} "
          f"({100 * (_sl[0] * 1000 / (rho * g) - 1):+.2f}%);  surface offset "
          f"{-_sl[1] / _sl[0]:.3f} mm = {-_sl[1] / _sl[0] / 1000.0 / dx:.2f} cells")
    print(f"  free surface {top * 1000:.2f} mm, drift over the graded window {drift * 1000:.3f} mm"
          f"  ({'settled' if drift < 2 * dx else 'NOT SETTLED -- not gradeable'})\n")
    n_in = int((DW > band).sum())
    print(f"  wall-contact band {band * 1000:.2f} mm per side ({a.wall_cells:g} cells): "
          f"{100 * (1 - n_in / max(len(DW), 1)):.1f}% of the fluid is inside it\n")
    # BULK IS WHAT THE CLOSED FORM DESCRIBES. p = rho*g*d assumes the column above a point is
    # carried by that point. A particle in the wall-contact band is partly carried by the WALL, so
    # its pressure is lower by however much the wall is taking -- that is a boundary layer, not an
    # error in the hydrostatic law, and grading it as one would either fail a correct code or force
    # a tolerance loose enough to pass a broken one. Both columns are printed; only bulk is graded.
    print(f"  {'depth (mm)':>12}{'bulk p (Pa)':>14}{'all p (Pa)':>13}{'rho*g*d (Pa)':>15}"
          f"{'bulk err':>11}{'all err':>10}{'':>8}")
    print("  " + "-" * 86)
    ok = drift < 2 * dx
    for d_mm in (5, 10, 15, 20, 25):
        d = d_mm / 1000.0
        at_d = (Y > top - d - dx) & (Y < top - d + dx)      # a one-cell band at this depth
        bulk = at_d & (DW > band)
        if at_d.sum() < 200 or bulk.sum() < 200:
            continue
        pm = float(np.median(P[bulk]))
        pa = float(np.median(P[at_d]))
        cf = rho * g * d
        e = abs(pm / cf - 1) * 100
        ea = abs(pa / cf - 1) * 100
        graded = d >= 2 * dx
        ok &= (e <= a.tol) if graded else True
        mark = ("  PASS" if e <= a.tol else "  FAIL") if graded \
            else "  (inside the free surface, not graded)"
        print(f"  {d_mm:>12}{pm:>14.1f}{pa:>13.1f}{cf:>15.1f}{e:>10.2f}%{ea:>9.2f}%{mark}")
    print(f"\n  {'ALL PASS' if ok else 'FAILURES ABOVE'}  (tol {a.tol:g}%)\n")


if __name__ == "__main__":
    main()
