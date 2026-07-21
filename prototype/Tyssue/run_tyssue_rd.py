#!/usr/bin/env python
"""run_tyssue_rd -- Goal 2: live Turing reaction-diffusion ON the cell set of the 3D vesicle.

Two sets in a genuine hierarchy: `vertex` (the mechanical mesh, relaxed by shape_energy_3d) and
`cell` (chem=[a,h], the morphogen). cell_geometry_3d AGGREGATES vertices -> per-cell centroid;
cell_adjacency builds the cell-cell graph from the half-edge table (NO Voronoi -- cells ARE mesh
faces, neighbours iff they share an edge); cell_diffuse (graph Laplacian) + cell_react run the RD.

`react` is a plexus2 CONTRACT with interchangeable implementations. Presets:
  coral         -- Gray-Scott -> a labyrinth/coral pattern (static topology)
  spots         -- Brusselator (params transposed verbatim from Turing_vertex fig4_coral) -> round spots
  rd_coral_grow -- coral pattern + UNIFORM growth + DIVISION (no morphogen bulge yet): tests that
                   divide_3d propagates the morphogen to daughters, so the pattern rides the
                   proliferating vesicle. This is the RD<->mesh coupling on the bridge to Fig 5 tubes.

    python run_tyssue_rd.py                 # all presets
    python run_tyssue_rd.py --only rd_coral_grow
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
import tyssue_ops3d        # noqa: F401  seed_mesh_3d + shape_energy_3d + vesicle_growth + divide_3d + topo_snapshot_3d
import tyssue_rd_ops       # noqa: F401  cell_* RD ops
from tyssue_ops3d import build_sphere_mesh
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_specfmt import write_spec
from run_tyssue_vesicle import _draw, _draw_cross

OUT = os.path.join(HERE, "archive")
RADIUS, JITTER, SEED = 5.0, 0.16, 0

GS = dict(react=dict(implementation="gray_scott", F=0.055, kk=0.062, rate=1.0),
          diffuse=dict(d_a=0.08, d_h=0.16, chi=1.3), seed=dict(mode="scatter", seed_frac=0.06), dt=1.0)
BRUSS = dict(react=dict(implementation="brusselator", gamma=2.0, A=1.0, B=3.0),
             diffuse=dict(d_a=0.05, d_h=0.7, chi=5.0), seed=dict(mode="noise", A=1.0, B=3.0, noise=0.04), dt=0.02)


def presets():
    #      name           rd     n_cells frames grow    divide
    return [("coral",        GS,    1200,  500,   0.0,    False),
            ("spots",        BRUSS, 1200,  500,   0.0,    False),
            ("rd_coral_grow", GS,    500,   450,   0.002,  True)]   # coral + SLOW (quasi-static) uniform grow + division


def make_spec(name, rd, n_cells, frames, grow, divide, buf, cbuf):
    dt = rd["dt"]
    ops = [
        {"op": "seed_mesh_3d", "at": "vertex", "n_cells": n_cells, "radius": RADIUS,
         "jitter": JITTER, "p0": 3.72, "seed": SEED, "before_frame": 1},
        {"op": "shape_energy_3d", "at": "vertex", "p0": 3.72, "K_A": 1.0, "K_P": 1.0, "Gamma": 0.1,
         "Lambda": 0.3, "K_V": 1.0, "K_R": 0.4, "mu": 1.0, "dt": dt,
         "relax_iters": 22 if (grow > 0 or divide) else 6,       # growth+division needs full relaxation to stay smooth
         "eta": 0.08, "cap_frac": 0.12},
        {"op": "cell_geometry_3d", "at": "cell"},
        {"op": "cell_adjacency", "at": "cell"},
        {"op": "cell_rd_seed", "at": "cell", "seed": SEED, "before_frame": 3, **rd["seed"]},
        {"op": "cell_diffuse", "at": "cell", **rd["diffuse"]},
        {"op": "cell_react", "at": "cell", **rd["react"]},
    ]
    sched = ["seed_mesh_3d", "shape_energy_3d", "cell_geometry_3d", "cell_adjacency",
             "cell_rd_seed", "cell_diffuse", "cell_react"]
    if grow > 0:
        ops.append({"op": "vesicle_growth", "at": "vertex", "rate": grow, "every": 1})
        sched.append("vesicle_growth")            # UNIFORM growth (no morphogen bulge yet)
    if divide:
        ops.append({"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "p0": 3.72,
                    "every": 2, "max_div": max(10, n_cells // 20), "cell_set": "cell"})  # scale w/ size -> uniform sizes
        sched.append("divide_3d")                                     # daughters inherit the morphogen
        ops.append({"op": "topo_snapshot_3d", "at": "vertex"}); sched.append("topo_snapshot_3d")
    cfg = {
        "general": {"name": f"tyssue_rd_{name}", "seed": SEED, "n_frames": frames, "dt": dt,
                    "boundary": "free", "dim": 3, "world": [8 * RADIUS, 8 * RADIUS, 8 * RADIUS]},
        "sets": {"vertex": {"n": buf},
                 "cell": {"n": cbuf, "state": {"chem": {"width": 2, "integration": "first_order"},
                                               "cen": {"width": 3}, "area": {"width": 1}}}},
        "fields": {},
        "operators": ops,
        "schedule": sched,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def _mesh(n_cells):
    verts, es, et, ef, nF = build_sphere_mesh(n_cells, RADIUS, JITTER, SEED)
    return dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, verts0=verts, Nv=verts.shape[0]), nF


def run_all(only=None):
    for name, rd, n_cells, frames, grow, divide in presets():
        if only and name not in only:
            continue
        odir = os.path.join(OUT, name); os.makedirs(odir, exist_ok=True)
        mesh0, nF = _mesh(n_cells); Nv = mesh0["Nv"]
        buf = int(Nv * (5.0 if divide else 1.0)); cbuf = int(nF * (5.0 if divide else 1.0))
        print(f"[tyssue_rd] {name}: {rd['react']['implementation']} grow={grow} divide={divide}  (Nv={Nv}, cells={nF})", flush=True)
        rec = {"name": name, "react": rd["react"]["implementation"], "grow": grow, "divide": divide, "Nv": Nv, "cells": nF}
        try:
            sim, cfg = make_spec(name, rd, n_cells, frames, grow, divide, buf, cbuf)
            write_spec(cfg, os.path.join(odir, "spec.yaml"))
            Hf, out = engine_run(sim, device="cpu")
            emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist")
            posf = out["sets"]["vertex"]["pos"]                    # [T, buf, 3]
            chemf = out["sets"]["cell"]["state"]["chem"]           # [T, cbuf, 2]
            T = posf.shape[0]

            def frame(t):
                mt = hist[min(t, len(hist) - 1)] if hist else mesh0
                nf = mt["nF"]
                return mt, posf[t][:mt["Nv"]].astype(np.float64), chemf[t][:nf, 0]

            aT = frame(T - 1)[2]
            lo, hi = float(np.percentile(aT, 5)), float(np.percentile(aT, 99) + 1e-6)
            rec.update(a_std=round(float(aT.std()), 4), patterned=bool(aT.std() > 0.05),
                       cells_end=int(frame(T - 1)[0]["nF"]), n_div=int(emesh.get("n_div", 0)))
            Rmax = max(float(np.linalg.norm(frame(t)[1], axis=1).max()) for t in np.linspace(0, T - 1, 20).astype(int))
            L3, L2 = Rmax * 1.06, Rmax * 2.23

            def norm(x):
                return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)

            fig = plt.figure(figsize=(17.6, 9.0)); fig.patch.set_facecolor("black")
            picks = [int(round(fr * (T - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]
            for i, t in enumerate(picks):
                mt, pt, a = frame(t)
                ax3 = fig.add_subplot(2, 4, i + 1, projection="3d")
                _draw(ax3, pt, mt, 3.72, azim=30, act=norm(a), Lbox=L3)
                ax2 = fig.add_subplot(2, 4, 4 + i + 1)
                _draw_cross(ax2, pt, mt, 3.72, act=norm(a), Lbox=L2)
            fig.subplots_adjust(0.006, 0.005, 0.996, 0.996, wspace=0.02, hspace=0.02)
            fig.savefig(os.path.join(odir, "strip.png"), dpi=120, facecolor="black"); plt.close(fig)
            figm = plt.figure(figsize=(5.0, 5.2)); figm.patch.set_facecolor("black")
            axm = figm.add_subplot(111, projection="3d"); figm.subplots_adjust(0, 0, 1, 1)
            keep = np.linspace(0, T - 1, min(T, 60)).astype(int)
            wri = FFMpegWriter(fps=max(1, round(len(keep) / 8.0)), metadata={"title": name})
            with wri.saving(figm, os.path.join(odir, "movie.mp4"), dpi=110):
                for j, t in enumerate(keep):
                    mt, pt, a = frame(t)
                    _draw(axm, pt, mt, 3.72, azim=(2 * j) % 360, act=norm(a), Lbox=L3)
                    wri.grab_frame()
            plt.close(figm)
            print(f"           -> activator std={rec['a_std']} patterned={rec['patterned']} "
                  f"cells {nF}->{rec['cells_end']} (+{rec['n_div']} div)", flush=True)
        except Exception as e:
            rec["error"] = repr(e); traceback.print_exc()
        json.dump(rec, open(os.path.join(odir, "diag.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    run_all(ap.parse_args().only)


if __name__ == "__main__":
    main()
