#!/usr/bin/env python
"""run_tyssue_rd -- Goal 2: live Turing reaction-diffusion ON the cell set of the 3D vesicle.

Two sets in a genuine hierarchy: `vertex` (the mechanical mesh, relaxed by shape_energy_3d) and
`cell` (chem=[a,h], the morphogen). cell_geometry_3d AGGREGATES vertices -> per-cell centroid;
cell_adjacency builds the cell-cell graph from the half-edge table (NO Voronoi -- cells ARE mesh
faces, neighbours iff they share an edge); cell_diffuse (graph Laplacian) + cell_react run the RD.

`react` is a plexus2 CONTRACT with interchangeable implementations -> two presets on the SAME cell
set: `coral` (Gray-Scott -> a labyrinth/coral) and `spots` (Brusselator, params transposed verbatim
from Turing_vertex fig4_coral -> round Turing spots). Rendered white->red by activator; the mesh is
relaxed but its topology fixed (RD only), so a single mesh renders every frame. strip + movie.

    python run_tyssue_rd.py                 # both presets
    python run_tyssue_rd.py --only coral
"""
from __future__ import annotations
import os, sys, argparse, json, tempfile, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import plexus.operators   # noqa: F401
import tyssue_ops3d        # noqa: F401  seed_mesh_3d + shape_energy_3d
import tyssue_rd_ops       # noqa: F401  cell_geometry_3d + cell_adjacency + cell_rd_seed + cell_diffuse + cell_react
from tyssue_ops3d import build_sphere_mesh
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_specfmt import write_spec
from run_tyssue_vesicle import _draw, _draw_cross

OUT = os.path.join(HERE, "archive")
NCELLS, RADIUS, JITTER, SEED = 1200, 5.0, 0.16, 0
FRAMES = 500


def presets():
    # `react` is a contract; each preset picks an implementation + its (diffusion, reaction, seed, dt).
    gray_scott = dict(react=dict(implementation="gray_scott", F=0.055, kk=0.062, rate=1.0),
                      diffuse=dict(d_a=0.08, d_h=0.16, chi=1.3), seed=dict(mode="scatter", seed_frac=0.06),
                      dt=1.0)                              # Gray-Scott is stable at dt=1 (classic diffusion scale)
    brusselator = dict(react=dict(implementation="brusselator", gamma=2.0, A=1.0, B=3.0),
                       diffuse=dict(d_a=0.05, d_h=0.7, chi=5.0), seed=dict(mode="noise", A=1.0, B=3.0, noise=0.04),
                       dt=0.02)                            # fig4 params need the fig4 timestep dt=0.02
    return [("coral", gray_scott), ("spots", brusselator)]


def make_spec(name, cfgrd, Nv, nF):
    dt = cfgrd["dt"]
    cfg = {
        "general": {"name": f"tyssue_rd_{name}", "seed": SEED, "n_frames": FRAMES, "dt": dt,
                    "boundary": "free", "dim": 3, "world": [6 * RADIUS, 6 * RADIUS, 6 * RADIUS]},
        "sets": {"vertex": {"n": Nv},
                 "cell": {"n": nF, "state": {"chem": {"width": 2, "integration": "first_order"},
                                             "cen": {"width": 3}, "area": {"width": 1}}}},
        "fields": {},
        "operators": [
            {"op": "seed_mesh_3d", "at": "vertex", "n_cells": NCELLS, "radius": RADIUS,
             "jitter": JITTER, "p0": 3.72, "seed": SEED, "before_frame": 1},
            {"op": "shape_energy_3d", "at": "vertex", "p0": 3.72, "K_A": 1.0, "K_P": 1.0,
             "Lambda": 0.1, "K_V": 1.0, "K_R": 0.4, "mu": 1.0, "dt": dt, "relax_iters": 3,
             "eta": 0.08, "cap_frac": 0.12},              # dt matches global dt -> full relaxation each frame
            {"op": "cell_geometry_3d", "at": "cell"},
            {"op": "cell_adjacency", "at": "cell"},
            {"op": "cell_rd_seed", "at": "cell", "seed": SEED, "before_frame": 3, **cfgrd["seed"]},
            {"op": "cell_diffuse", "at": "cell", **cfgrd["diffuse"]},
            {"op": "cell_react", "at": "cell", **cfgrd["react"]},
        ],
        "schedule": ["seed_mesh_3d", "shape_energy_3d", "cell_geometry_3d", "cell_adjacency",
                     "cell_rd_seed", "cell_diffuse", "cell_react"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def _mesh():
    verts, es, et, ef, nF = build_sphere_mesh(NCELLS, RADIUS, JITTER, SEED)
    return dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, verts0=verts, Nv=verts.shape[0]), nF


def run_all(only=None):
    for name, cfgrd in presets():
        if only and name not in only:
            continue
        odir = os.path.join(OUT, name); os.makedirs(odir, exist_ok=True)
        mesh, nF = _mesh(); Nv = mesh["Nv"]
        print(f"[tyssue_rd] {name}: {cfgrd['react']['implementation']}  (Nv={Nv}, cells={nF})", flush=True)
        rec = {"name": name, "react": cfgrd["react"]["implementation"], "Nv": Nv, "cells": nF}
        try:
            sim, cfg = make_spec(name, cfgrd, Nv, nF)
            write_spec(cfg, os.path.join(odir, "spec.yaml"))
            _, out = engine_run(sim, device="cpu")
            pos = out["sets"]["vertex"]["pos"][:, :Nv, :]
            a = out["sets"]["cell"]["state"]["chem"][:, :nF, 0]     # activator trajectory [T, nF]
            T = pos.shape[0]
            lo, hi = float(np.percentile(a[-1], 5)), float(np.percentile(a[-1], 99) + 1e-6)
            rec.update(a_lo=round(lo, 3), a_hi=round(hi, 3), a_std=round(float(a[-1].std()), 4),
                       patterned=bool(a[-1].std() > 0.05))
            L3, L2 = RADIUS * 1.06, RADIUS * 2.23

            def norm(x):
                return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)

            fig = plt.figure(figsize=(17.6, 9.0)); fig.patch.set_facecolor("black")
            picks = [int(round(fr * (T - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]
            for i, t in enumerate(picks):
                ax3 = fig.add_subplot(2, 4, i + 1, projection="3d")
                _draw(ax3, pos[t].astype(np.float64), mesh, 3.72, azim=30, act=norm(a[t]), Lbox=L3)
                ax2 = fig.add_subplot(2, 4, 4 + i + 1)
                _draw_cross(ax2, pos[t].astype(np.float64), mesh, 3.72, act=norm(a[t]), Lbox=L2)
            fig.subplots_adjust(0.006, 0.005, 0.996, 0.996, wspace=0.02, hspace=0.02)
            fig.savefig(os.path.join(odir, "strip.png"), dpi=120, facecolor="black"); plt.close(fig)
            figm = plt.figure(figsize=(5.0, 5.2)); figm.patch.set_facecolor("black")
            axm = figm.add_subplot(111, projection="3d"); figm.subplots_adjust(0, 0, 1, 1)
            keep = np.linspace(0, T - 1, min(T, 60)).astype(int)
            wri = FFMpegWriter(fps=max(1, round(len(keep) / 8.0)), metadata={"title": name})
            with wri.saving(figm, os.path.join(odir, "movie.mp4"), dpi=110):
                for j, t in enumerate(keep):
                    _draw(axm, pos[t].astype(np.float64), mesh, 3.72, azim=(2 * j) % 360, act=norm(a[t]), Lbox=L3)
                    wri.grab_frame()
            plt.close(figm)
            print(f"           -> activator std={rec['a_std']} patterned={rec['patterned']}", flush=True)
        except Exception as e:
            rec["error"] = repr(e); traceback.print_exc()
        json.dump(rec, open(os.path.join(odir, "diag.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    run_all(ap.parse_args().only)


if __name__ == "__main__":
    main()
