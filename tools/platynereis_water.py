#!/usr/bin/env python
"""Is the water a fluid? The sampling test that decides whether any stress in it means anything.

    python tools/platynereis_water.py plat_r14_paddle

WHY THIS IS THE FIRST QUESTION ABOUT THE WATER, ahead of the stress itself. MLS-MPM carries the
material on PARTICLES and solves on a GRID, and the transfer between them is an interpolation:
each grid node's velocity is the mass-weighted average of the particles in its neighbourhood, and
each particle's new velocity is read back from the nodes. That average is only a fluid if the
neighbourhood HAS particles in it. The standard sampling is 8 particles per cell -- 2 per axis --
and below roughly 4 the transfer stops being an average and becomes a lottery: nodes flicker
between carrying mass and carrying none, the pressure computed from the density is whatever the
local count happened to be, and a particle's own deformation gradient F integrates a velocity
gradient assembled from nodes that were empty a substep ago.

WHAT IS MEASURED, all of it per grid cell rather than globally, because the global average is the
number that hides the problem:

  * PARTICLES PER OCCUPIED CELL, as a distribution. The mean over the whole box is misleading
    when the water is a dust; what matters is how many cells hold 0, 1, 2 particles.
  * THE FRACTION OF CELLS THE WATER TOUCHES AT ALL. A fluid that occupies a tenth of the nodes
    in the volume it is supposed to fill is not exerting pressure across the other nine tenths.
  * THE VOLUME EACH PARTICLE STANDS FOR, against the cell volume. A particle carrying several
    cells' worth of mass is not wrong about the total -- the density comes out right -- it is
    wrong about WHERE the mass is, which is the only thing a pressure gradient is made of.
  * THE DENSITY THE GRID ACTUALLY SEES, against the 1025 kg/m^3 the spec asked for. This is the
    test that cannot be argued with: if the grid's own density is a fraction of the declared one,
    the fluid is thinner than the spec says and every force it returns is scaled by that error.
  * THE SPEED FIELD's spatial coherence -- neighbouring cells' velocities correlated or not. Real
    flow is smooth over a few cells. Uncorrelated cell-to-cell velocity is noise wearing the name
    of a flow field.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))
UM = 195.9


def cell_counts(P, n_grid):
    """How many particles land in each grid cell, for one frame of positions."""
    ijk = np.clip((P * n_grid).astype(np.int64), 0, n_grid - 1)
    flat = (ijk[:, 0] * n_grid + ijk[:, 1]) * n_grid + ijk[:, 2]
    return np.bincount(flat, minlength=n_grid ** 3)


def main():
    from platynereis_momentum import load, masses
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="plat_r14_paddle")
    ap.add_argument("--dt", type=float, default=0.05)
    a = ap.parse_args()

    tr, sp = load(a.name)
    import yaml
    d = yaml.safe_load(open(sp))
    n_grid = int(((d.get("fields") or {}).get("mpm_grid") or {}).get("n_grid", 96))
    m_of = masses(sp)
    cell_um = UM / n_grid
    cell_vol = (1.0 / n_grid) ** 3

    print(f"{a.name}\n")
    print(f"  grid {n_grid}^3 = {n_grid ** 3:,d} cells, cell {cell_um:.2f} um "
          f"({cell_vol:.3e} world^3)\n")

    # ---------------------------------------------------------------- sampling, per material set
    print(f"  {'set':16s} {'n':>9s} {'per occupied cell':>19s} {'cells used':>12s} "
          f"{'vol/particle':>14s}")
    for k in ("water_particle", "body_point", "mpm_particle", "cilium_point"):
        key = f"{k}__pos"
        if key not in tr.files:
            continue
        P = np.asarray(tr[key])[0]
        occ = tr.get(f"{k}__occ")
        if occ is not None:
            live = np.asarray(occ)[0].astype(bool)
            if live.ndim == 1 and live.shape[0] == P.shape[0]:
                P = P[live]
        c = cell_counts(P, n_grid)
        used = c[c > 0]
        # the volume this set's particles were given, from the spec's own radius and per_parent
        v_spec = m_of.get(k, np.nan) / float(d["sets"][k].get("density", 1.0))
        print(f"  {k:16s} {P.shape[0]:9,d} {used.mean():19.2f} {used.size:12,d} "
              f"{v_spec / cell_vol:13.2f}x")

    W = np.asarray(tr["water_particle__pos"])
    occ = np.asarray(tr["water_particle__occ"])[0].astype(bool)
    c = cell_counts(W[0][occ], n_grid)

    # the water's own bounding region -- the box it is supposed to fill, not the whole world
    lo, hi = W[0][occ].min(0), W[0][occ].max(0)
    span = np.prod(np.clip((hi - lo) * n_grid, 1, None))
    print(f"\n  THE WATER'S SAMPLING, which decides whether its stress means anything:")
    print(f"    {int(occ.sum()):,d} particles over {int(span):,d} cells of its own bounding box")
    print(f"    mean {occ.sum() / span:.3f} particles per cell   "
          f"(MLS-MPM is written for 8; below about 4 the grid transfer is a lottery)")
    for q in (0, 1, 2, 4, 8):
        print(f"    cells holding > {q:2d}: {100 * (c > q).sum() / span:6.2f}% of the box")

    # ---------------------------------------------------------------- the density the grid sees
    m_w = m_of["water_particle"]
    rho_spec = float(d["sets"]["water_particle"]["density"])
    rho_grid = (c[c > 0].mean() * m_w) / cell_vol
    print(f"\n  THE DENSITY THE GRID ACTUALLY SEES, against what the spec asked for:")
    print(f"    spec {rho_spec:8.1f}    occupied cells {rho_grid:8.1f}    "
          f"averaged over the box {(occ.sum() / span) * m_w / cell_vol:8.1f}")
    print(f"    a fluid thinner than its own spec returns forces scaled by that same error")

    # ---------------------------------------------------------------- is the flow field smooth?
    V = np.diff(W[:, occ], axis=0) / a.dt
    f = min(V.shape[0] - 1, int(0.5 * V.shape[0]))
    ijk = np.clip((W[f][occ] * n_grid).astype(np.int64), 0, n_grid - 1)
    flat = (ijk[:, 0] * n_grid + ijk[:, 1]) * n_grid + ijk[:, 2]
    acc = np.zeros((n_grid ** 3, 3))
    np.add.at(acc, flat, V[f])
    cnt = np.bincount(flat, minlength=n_grid ** 3)
    live = cnt > 0
    mean_v = acc[live] / cnt[live, None]
    # the same cells shifted one step in x: how much does a cell's velocity agree with its neighbour
    nb = np.where(live)[0] + n_grid * n_grid
    ok = (nb < n_grid ** 3) & live[np.clip(nb, 0, n_grid ** 3 - 1)]
    idx = np.searchsorted(np.where(live)[0], nb[ok])
    u, w = mean_v[ok], mean_v[np.clip(idx, 0, mean_v.shape[0] - 1)]
    cos = (u * w).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-30)
    print(f"\n  IS THE FLOW SMOOTH, or is it noise wearing the name of a flow field?")
    print(f"    {ok.sum():,d} pairs of adjacent occupied cells at frame {f}")
    print(f"    their velocities agree at cos = {np.median(cos):+.3f} "
          f"(a real flow is near +1 over one cell; 0 is uncorrelated)")

    sp_um = np.linalg.norm(V, axis=2) * UM
    print(f"\n  WATER SPEED  median {np.median(sp_um.mean(0)):.3f} um/s   "
          f"95th {np.percentile(sp_um.mean(0), 95):.3f}   max {sp_um.max():.1f}")

    # ------------------------------------------------------------------ is the box big enough?
    #
    # THE QUESTION IS NOT HOW CLOSE THE WALLS ARE, IT IS WHETHER THE FLOW HAS DECAYED BEFORE IT
    # REACHES THEM. A swimmer in a box is in trouble when the fluid it pushed arrives at a wall
    # still moving, because the wall then pushes back and the animal is partly swimming against
    # its own reflection. The honest test is the speed profile against distance from the nearest
    # wall, compared with the speed in the shell around the animal that the cilia are stirring.
    #
    # Stokes flow from a finite swimmer decays like 1/r or faster, so a box that is comfortable
    # shows the wall shell at a few percent of the near shell. Anything approaching parity means
    # the run is measuring its container.
    # ------------------------------------------------------------------ which regime is this?
    #
    # THE NUMBER THAT SAYS WHETHER THIS IS A MODEL OF A CILIATED LARVA AT ALL. A cilium's whole
    # design -- a fast straight power stroke and a slow bent recovery -- exists because at low
    # Reynolds number a time-SYMMETRIC stroke moves no net fluid however hard it is driven
    # (Purcell's scallop theorem). That constraint only binds when Re << 1. Above it, thrust comes
    # from the fluid's inertia instead, a symmetric stroke swims perfectly well, and the model is
    # paddling rather than beating.
    #
    #     Re = rho U L / eta          eta the DYNAMIC viscosity (mpm_viscosity's `eta`; the
    #                                 operator's docstring used to call it kinematic, which is
    #                                 this same number divided by rho)
    #
    # A real Platynereis nectochaete: rho 1000 kg/m^3, U about 1 mm/s, L 160 um, mu 1e-3 Pa s,
    # so Re is about 0.16.
    eta = next((float(o.get("eta", 0.0)) for o in (d.get("operators") or [])
                if o.get("op") == "mpm_viscosity"), None)
    B = np.asarray(tr["body_point__pos"])
    if eta:
        U = float(np.median(np.linalg.norm(V, axis=2).mean(0)))          # world/s
        L = float(np.linalg.norm(B[0] - B[0].mean(0), axis=1).mean() * 2)
        re = rho_spec * U * L / eta
        print(f"\n  WHICH REGIME IS THIS? Re = rho U L / eta = {re:,.0f}   "
              f"(rho {rho_spec:g}, U {U * UM:.2f} um/s, L {L * UM:.0f} um, eta {eta:g})")
        print(f"    a real Platynereis nectochaete swims at Re about 0.16. Above Re ~ 1 the "
              f"thrust is INERTIAL and\n    a time-symmetric stroke swims fine -- so Purcell's "
              f"scallop theorem, which is the reason a\n    real cilium beats asymmetrically at "
              f"all, does not bind and the model is paddling, not beating.")
        print(f"    eta for Re = 0.16 would be {rho_spec * U * L / 0.16:.1f}, "
              f"{rho_spec * U * L / 0.16 / eta:,.0f}x the value in this spec")

    com = B.mean(1)[0]
    W0 = W[0][occ]
    r_animal = np.linalg.norm(B[0] - com, axis=1).mean()
    d_wall = np.minimum(W0, 1.0 - W0).min(1)                 # distance to the NEAREST wall
    d_body = np.linalg.norm(W0 - com, axis=1)
    speed = sp_um.mean(0)                                     # per particle, averaged over time
    print(f"\n  IS THE BOX BIG ENOUGH? the animal's mean radius is {r_animal * UM:.0f} um and the "
          f"nearest wall is {d_wall.min() * UM:.0f} um from the nearest water")
    print(f"    {'shell':>22s} {'n':>9s} {'speed um/s':>11s}")
    near = speed[d_body < 1.5 * r_animal]
    for lo, hi in ((0.0, 0.10), (0.10, 0.20), (0.20, 0.35)):
        m = (d_wall >= lo) & (d_wall < hi)
        if m.sum():
            print(f"    {lo * UM:6.0f} - {hi * UM:3.0f} um from wall {int(m.sum()):9,d} "
                  f"{np.median(speed[m]):11.3f}")
    if near.size:
        print(f"    {'within 1.5 body radii':>22s} {near.size:9,d} {np.median(near):11.3f}")
        m = d_wall < 0.10
        if m.sum():
            print(f"    the wall shell runs at {100 * np.median(speed[m]) / max(np.median(near), 1e-12):.0f}% "
                  f"of the shell the cilia are stirring -- a comfortable box shows a few percent, "
                  f"and parity means the run is measuring its container")


if __name__ == "__main__":
    main()
