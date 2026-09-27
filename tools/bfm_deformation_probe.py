#!/usr/bin/env python
"""HOW MUCH THE ROTOR'S PARTS DEFORM while they turn -- a record measurement, per part, over time.

    PLEXUS_BUILDER=... python tools/bfm_deformation_probe.py bfm_d02_cj_emerge_drag5 [--note ...]

The trajectory stores positions only, so three things are read off the points of each axis part
(C-ring posts, FliG ring, MS-ring, rod, bushing, hook), against their first row:
  1. VOLUME, as the count of occupied cells at the run's own grid spacing, times dx^3 -- the
     one volume a point cloud has without a surface fitted to it; a part that keeps its cells
     keeps its volume.
  2. EXTENT: inner and outer radius (1st and 99th percentile of distance to the axis) and height
     (1st-99th percentile along it).
  3. NON-RIGID DISPLACEMENT: the best rigid rotation about the axis is removed (the circular-mean
     angle the rotor probe uses) and what remains, |x_t - R(theta_t) x_0| per point, is the
     deformation; its RMS and its maximum per part over time, and a map of it on the last row.
All in nm through the run's own units. Every number is a change against row 0, so a still rotor
reads as zeros.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "src"))

PARTS = ["c_posts_a_pt", "c_flig_pt", "ms_ring_pt", "rod_pt", "lp_ring_pt", "hook_pt"]


def main():
    import zarr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.paths import graphs_data_path
    from builder_figure import record, style
    note = sys.argv[sys.argv.index("--note") + 1] if "--note" in sys.argv else ""
    names = [a for a in sys.argv[1:] if not a.startswith("--") and a != note]
    if len(names) > 1:
        return compare(names, note)
    name = names[0]
    d = next(os.path.join(graphs_data_path(), sub, name) for sub in ("bacterium", "studio")
             if os.path.isdir(os.path.join(graphs_data_path(), sub, name)))
    spec = yaml.safe_load(open(os.path.join(d, "spec.yaml")))
    u = spec["general"]["units"]; nm = u["length_um"] * 1e3
    n_grid = int(spec["fields"]["mpm_grid"]["n_grid"]); dx = 1.0 / n_grid
    g = zarr.open_group(os.path.join(d, "simulation.zarr"), mode="r")
    T = g[PARTS[0]]["pos"].shape[0]
    rows = np.unique(np.linspace(0, T - 1, 41).astype(int))
    t_ms = rows * float(spec["general"]["dt"]) * u["time_s"] * 1e3
    c = np.asarray(g["stator_unit"]["pos"][0]).mean(0)[:2]
    out, last_map = {}, {}
    for p in PARTS:
        if p not in g:
            continue
        P = np.asarray(g[p]["pos"][rows])                     # [R, N, 3]
        rel = P[..., :2] - c[None, None, :]
        r = np.linalg.norm(rel, axis=2); z = P[..., 2]
        phi = np.arctan2(rel[..., 1], rel[..., 0])
        keep = r[0] > 0.02
        m = np.exp(1j * (phi[:, keep] - phi[0:1, keep])).mean(1)
        theta = np.unwrap(np.angle(m))                       # the rigid rotation per row
        # the rigid prediction of row 0, rotated by theta_t
        ct, stt = np.cos(theta)[:, None], np.sin(theta)[:, None]
        x0, y0 = rel[0, :, 0][None, :], rel[0, :, 1][None, :]
        xr = ct * x0 - stt * y0; yr = stt * x0 + ct * y0
        res = np.sqrt((rel[..., 0] - xr) ** 2 + (rel[..., 1] - yr) ** 2 + (z - z[0:1]) ** 2) * nm
        vol = []
        for i in range(P.shape[0]):
            ijk = np.floor(P[i] / dx).astype(np.int64)
            vol.append(len(np.unique(ijk[:, 0] * n_grid * n_grid + ijk[:, 1] * n_grid + ijk[:, 2])) * dx ** 3 * nm ** 3)
        out[p] = dict(t=t_ms, theta=np.degrees(theta), vol=np.asarray(vol), vol0=vol[0],
                      r_in=np.percentile(r, 1, axis=1) * nm, r_out=np.percentile(r, 99, axis=1) * nm,
                      z0=np.percentile(z, 1, axis=1) * nm, z1=np.percentile(z, 99, axis=1) * nm,
                      rms=np.sqrt((res ** 2).mean(1)), mx=res.max(1))
        last_map[p] = (r[-1] * nm, z[-1] * nm, res[-1])
        o = out[p]
        print(f"{p:12s} volume {o['vol0']:.0f} -> {o['vol'][-1]:.0f} nm^3 ({(o['vol'][-1] / o['vol0'] - 1) * 100:+.2f}%); "
              f"r [{o['r_in'][0]:.1f},{o['r_out'][0]:.1f}] -> [{o['r_in'][-1]:.1f},{o['r_out'][-1]:.1f}] nm; "
              f"z [{o['z0'][0]:.1f},{o['z1'][0]:.1f}] -> [{o['z0'][-1]:.1f},{o['z1'][-1]:.1f}] nm; "
              f"non-rigid rms {o['rms'][-1]:.2f} nm, max {o['mx'][-1]:.2f} nm at the end (turned {o['theta'][-1]:.0f} deg)")
    cols = ["#c0392b", "#7b5aa6", "#2f6fb5", "#4a7c59", "#7f7f7f", "#e08214"]
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.8))
    for (p, o), col in zip(out.items(), cols):
        ax[0].plot(o["t"], (o["vol"] / o["vol0"] - 1) * 100, color=col, lw=1.4, label=p.replace("_pt", ""))
        ax[1].plot(o["t"], o["rms"], color=col, lw=1.4); ax[1].plot(o["t"], o["mx"], color=col, lw=0.8, ls=":")
        ax[2].plot(o["t"], o["r_out"] - o["r_out"][0], color=col, lw=1.4)
        ax[2].plot(o["t"], o["z1"] - o["z1"][0], color=col, lw=0.8, ls="--")
    ax[0].set_xlabel("time (ms)"); ax[0].set_ylabel("volume change (%, occupied cells x dx^3)"); ax[0].legend(frameon=False, fontsize=8)
    ax[1].set_xlabel("time (ms)"); ax[1].set_ylabel("non-rigid displacement (nm): rms solid, max dotted")
    ax[2].set_xlabel("time (ms)"); ax[2].set_ylabel("change in outer radius (solid) and top (dashed), nm")
    p_map = "hook_pt" if "hook_pt" in last_map else list(last_map)[-1]
    rr, zz, res = last_map[p_map]
    sc = ax[3].scatter(rr, zz, c=res, s=1.5, cmap="magma", rasterized=True)
    plt.colorbar(sc, ax=ax[3], label="non-rigid displacement (nm)")
    ax[3].set_xlabel("distance from the axis (nm)"); ax[3].set_ylabel("height (nm)"); ax[3].set_title(f"{p_map.replace('_pt', '')}, last row", fontsize=9)
    for i, a in enumerate(ax):
        a.text(0, 1.03, "ABCD"[i], transform=a.transAxes, fontsize=11); style(a)
    fig.tight_layout()
    worst = max(out.items(), key=lambda kv: kv[1]["rms"][-1])
    why = ((note + " ") if note else "") + (
        f"MEASUREMENT, not a run: THE DEFORMATION OF THE ROTOR'S PARTS in {name}, against their first row, at the run's own "
        f"grid ({n_grid}^3, dx {dx * nm:.2f} nm). A: each part's volume as occupied cells x dx^3, in per cent of its start. "
        f"B: what is left of each point's motion after the part's best rigid rotation about the axis is removed, rms (solid) "
        f"and max (dotted), nm. C: the change of each part's outer radius (solid) and top (dashed), nm. D: that non-rigid "
        f"displacement mapped on the {p_map.replace('_pt', '')}'s points at the last row. Numbers at the end: "
        + "; ".join(f"{p.replace('_pt', '')} volume {(o['vol'][-1] / o['vol0'] - 1) * 100:+.2f}%, non-rigid rms {o['rms'][-1]:.2f} nm "
                    f"max {o['mx'][-1]:.2f} nm" for p, o in out.items())
        + f". The part that deforms most is the {worst[0].replace('_pt', '')}.")
    record(fig, why, name=f"{name}: deformation of the rotor's parts")


def _series(name, part="rod_pt"):
    """(t_ms, vol %, rms nm, max nm, theta deg, bushing angle deg, n_grid) of one part in one run."""
    import zarr
    from plexus.paths import graphs_data_path
    d = next(os.path.join(graphs_data_path(), sub, name) for sub in ("bacterium", "studio")
             if os.path.isdir(os.path.join(graphs_data_path(), sub, name)))
    spec = yaml.safe_load(open(os.path.join(d, "spec.yaml")))
    u = spec["general"]["units"]; nm = u["length_um"] * 1e3
    n_grid = int(spec["fields"]["mpm_grid"]["n_grid"]); dx = 1.0 / n_grid
    g = zarr.open_group(os.path.join(d, "simulation.zarr"), mode="r")
    T = g[part]["pos"].shape[0]
    rows = np.unique(np.linspace(0, T - 1, 41).astype(int))
    t_ms = rows * float(spec["general"]["dt"]) * u["time_s"] * 1e3
    c = np.asarray(g["stator_unit"]["pos"][0]).mean(0)[:2]
    out = {}
    for p in (part, "lp_ring_pt"):
        P = np.asarray(g[p]["pos"][rows]); rel = P[..., :2] - c[None, None, :]
        r = np.linalg.norm(rel, axis=2); z = P[..., 2]; phi = np.arctan2(rel[..., 1], rel[..., 0])
        keep = r[0] > 0.02
        m = np.exp(1j * (phi[:, keep] - phi[0:1, keep])).mean(1); theta = np.unwrap(np.angle(m))
        ct, stt = np.cos(theta)[:, None], np.sin(theta)[:, None]
        x0, y0 = rel[0, :, 0][None, :], rel[0, :, 1][None, :]
        res = np.sqrt((rel[..., 0] - (ct * x0 - stt * y0)) ** 2 + (rel[..., 1] - (stt * x0 + ct * y0)) ** 2 + (z - z[0:1]) ** 2) * nm
        vol = [len(np.unique((np.floor(P[i] / dx).astype(np.int64) * np.array([n_grid * n_grid, n_grid, 1])).sum(1))) * dx ** 3 * nm ** 3
               for i in range(P.shape[0])]
        out[p] = dict(vol=np.asarray(vol), rms=np.sqrt((res ** 2).mean(1)), mx=res.max(1), theta=np.degrees(theta))
    o = out[part]
    return dict(t=t_ms, vol=(o["vol"] / o["vol"][0] - 1) * 100, rms=o["rms"], mx=o["mx"], theta=o["theta"],
                bushing=out["lp_ring_pt"]["theta"], n_grid=n_grid, dx_nm=dx * nm)


def compare(names, note):
    """The grid test: the rod's volume, non-rigid displacement and the bushing's angle, one line per run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from builder_figure import record, style
    runs = [_series(n) for n in names]
    cols = ["#c0392b", "#2f6fb5", "#4a7c59", "#e08214"]
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    for n, r, col in zip(names, runs, cols):
        lab = f"{n} (grid {r['n_grid']}, dx {r['dx_nm']:.2f} nm)"
        ax[0].plot(r["t"], r["vol"], color=col, lw=1.4, label=lab)
        ax[1].plot(r["t"], r["rms"], color=col, lw=1.4); ax[1].plot(r["t"], r["mx"], color=col, lw=0.8, ls=":")
        ax[2].plot(r["t"], r["theta"], color=col, lw=1.4)
        ax[3].plot(r["t"], r["bushing"], color=col, lw=1.4)
        print(f"{n}: grid {r['n_grid']}, rod volume {r['vol'][-1]:+.2f}%, non-rigid rms {r['rms'][-1]:.2f} nm max {r['mx'][-1]:.2f} nm, "
              f"turned {r['theta'][-1]:.0f} deg, bushing {r['bushing'][-1]:+.2f} deg")
    ax[0].set_xlabel("time (ms)"); ax[0].set_ylabel("rod volume change (%, occupied cells x dx^3)"); ax[0].legend(frameon=False, fontsize=8)
    ax[1].set_xlabel("time (ms)"); ax[1].set_ylabel("rod non-rigid displacement (nm): rms solid, max dotted")
    ax[2].set_xlabel("time (ms)"); ax[2].set_ylabel("rod angle (deg)")
    ax[3].set_xlabel("time (ms)"); ax[3].set_ylabel("anchored bushing angle (deg): non-zero = it is being dragged")
    for i, a in enumerate(ax):
        a.text(0, 1.03, "ABCD"[i], transform=a.transAxes, fontsize=11); style(a)
    fig.tight_layout()
    why = ((note + " ") if note else "") + (
        "MEASUREMENT, not a run: THE GRID TEST, the same fast motor (drag 5/s) on " + ", ".join(f"{n} (grid {r['n_grid']})" for n, r in zip(names, runs))
        + ", read off the rod (the part that carries the twist) and the anchored bushing. A: the rod's volume change as occupied "
        "cells x dx^3, each at its own grid. B: its non-rigid displacement after the best rigid rotation is removed, rms and max. "
        "C: its angle. D: the bushing's angle -- anchored, it must stay at zero; a bushing that turns is being gripped through "
        "the grid, the contact a too-coarse grid makes when the 4.5 nm rod-bushing gap falls inside the 3-cell stencil. "
        + "; ".join(f"grid {r['n_grid']}: rod volume {r['vol'][-1]:+.2f}%, non-rigid rms {r['rms'][-1]:.2f} nm, bushing {r['bushing'][-1]:+.2f} deg"
                    for r in runs) + ".")
    record(fig, why, name="grid test: rod deformation and bushing grip")


if __name__ == "__main__":
    main()
