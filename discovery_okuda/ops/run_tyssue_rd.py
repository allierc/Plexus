#!/usr/bin/env python
"""run_tyssue_rd -- Goal 2: live Turing reaction-diffusion ON the cell set of the 3D vesicle.

Two sets in a genuine hierarchy: `vertex` (the mechanical mesh, relaxed by cell_mechanics) and
`cell` (chem=[a,h], the morphogen). cell_geometry AGGREGATES vertices -> per-cell centroid;
cell_neighbours builds the cell-cell graph from the half-edge table (NO Voronoi -- cells ARE mesh
faces, neighbours iff they share an edge); cell_chem_diffuse (graph Laplacian) + cell_chem_react run the RD.

`react` is a plexus2 CONTRACT with interchangeable implementations. Presets:
  coral         -- Gray-Scott -> a labyrinth/coral pattern (static topology)
  spots         -- Brusselator (params transposed verbatim from Turing_vertex fig4_coral) -> round spots
  rd_coral_grow -- coral pattern + UNIFORM growth + DIVISION (no morphogen bulge yet): tests that
                   cell_divide propagates the morphogen to daughters, so the pattern rides the
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
import mesh_ops        # noqa: F401  mesh_seed + cell_mechanics + cell_grow + cell_divide + topo_record
import t1_ops     # noqa: F401  edge_flip (anneal division defects -> rounded cells over long runs)
import chem_ops       # noqa: F401  cell_* RD ops
from mesh_ops import build_sphere_mesh
import plexus.schema as S
from plexus.engine import run as engine_run
from specfmt import write_spec
from run_tyssue_vesicle import _draw, _draw_cross

OUT = os.path.join(HERE, "archive")
RADIUS, JITTER, SEED = 5.0, 0.16, 0

GS = dict(react=dict(implementation="gray_scott", F=0.055, kk=0.062, rate=1.0),
          diffuse=dict(d_a=0.08, d_h=0.16, chi=1.3), seed=dict(mode="scatter", seed_frac=0.06), dt=1.0)
BRUSS = dict(react=dict(implementation="brusselator", gamma=2.0, A=1.0, B=3.0),
             diffuse=dict(d_a=0.05, d_h=0.7, chi=5.0), seed=dict(mode="noise", A=1.0, B=3.0, noise=0.04), dt=0.02)


def presets():
    #      name              rd     n_cells frames grow    divide  cv    (cv>0 = stochastic cell cycle: desync'd division)
    return [("coral",            GS,    1200,  500,   0.0,    False,  0.0),
            ("spots",            BRUSS, 1200,  500,   0.0,    False,  0.0),
            ("rd_coral_grow",    GS,    150,   220,   0.003,  True,   0.0),   # EXACTLY vesicle_divide (150c/220f) + coral
            ("rd_coral_grow_big", GS,   500,   1000,  0.002,  True,   0.4),   # SCALED: stochastic cell cycle breaks the
            ("rd_coral_grow_long", GS,  150,   500,   0.003,  True,   0.4)]   # LONG rd_coral_grow (cv=0.4 -> uniform)
    #    synchronised division wave -> stays clean at 500c/1000f -> finer coral pattern


def make_spec(name, rd, n_cells, frames, grow, divide, cv, buf, cbuf):
    dt = rd["dt"]
    rec_cap = frames + 2               # record EVERY frame so posf (engine) and hist (topo_snapshot) are the
    sstride = 1                        # SAME length. A mismatched stride (posf longer than hist) makes the
    #   render/diagnostic pair positions with the WRONG frame's topology -> scrambled rings -> phantom
    #   "hollow" cells. Aligning the two recordings is the fix (the hollow-cell scare was entirely this).
    # VERTEX MECHANICS in the SAME order as vesicle_divide (growth -> shape_energy -> T1 -> divide), then
    # the RD ops (which only COLOUR the cells). rd_coral_grow == vesicle_divide + reaction-diffusion.
    ops = [{"op": "mesh_seed", "at": "vertex", "n_cells": n_cells, "radius": RADIUS,
            "jitter": JITTER, "p0": 3.72, "seed": SEED, "before_frame": 1, "vseed_cv": cv}]  # stochastic volume seed
    sched = ["mesh_seed"]
    if grow > 0:
        ops.append({"op": "cell_grow", "at": "vertex", "rate": grow, "every": 1, "rho": 1.0, "a_sw": 0.0, "vth_frac": 15.625, "conserve_amount": False})
        sched.append("cell_grow")            # capped -> PLATEAUS (~1400 cells); BEFORE shape_energy (== vesicle_divide)
    ops.append({"op": "cell_mechanics", "at": "vertex", "p0": 3.72, "K_A": 1.0, "K_P": 1.0,
                "Gamma": 0.1, "Lambda": 0.5, "K_V": 1.0, "K_R": 0.4, "mu": 1.0, "dt": dt,    # == vesicle_divide (clean)
                "relax_iters": 26 if (grow > 0 or divide) else 6, "eta": 0.08, "cap_frac": 0.12})
    sched.append("cell_mechanics")
    if divide:
        ops.append({"op": "edge_flip", "at": "vertex", "l_th_frac": 0.35, "every": 2,
                    "max_flips": max(20, n_cells // 15)})             # T1 anneals division defects -> rounded cells
        sched.append("edge_flip")
        ops.append({"op": "cell_divide", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "cycle_cv": cv,
                    "p0": 3.72, "every": 2, "max_div": max(10, n_cells // 20), "cell_set": "cell"})  # daughters inherit morphogen
        sched.append("cell_divide")
    ops += [{"op": "cell_geometry", "at": "cell"},            # --- RD (colouring only), after the mechanics ---
            {"op": "cell_neighbours", "at": "cell"},
            {"op": "cell_chem_seed", "at": "cell", "seed": SEED, "before_frame": 3, **rd["seed"]},
            {"op": "cell_chem_diffuse", "at": "cell", **rd["diffuse"]},
            {"op": "cell_chem_react", "at": "cell", **rd["react"]}]
    sched += ["cell_geometry", "cell_neighbours", "cell_chem_seed", "cell_chem_diffuse", "cell_chem_react"]
    if divide:
        ops.append({"op": "topo_record", "at": "vertex", "every": sstride}); sched.append("topo_record")
    cfg = {
        "general": {"name": f"tyssue_rd_{name}", "seed": SEED, "n_frames": frames, "dt": dt, "record_cap": rec_cap,
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
    for name, rd, n_cells, frames, grow, divide, cv in presets():
        if only and name not in only:
            continue
        odir = os.path.join(OUT, name); os.makedirs(odir, exist_ok=True)
        mesh0, nF = _mesh(n_cells); Nv = mesh0["Nv"]
        buf = int(Nv * (10.0 if divide else 1.0)); cbuf = int(nF * (10.0 if divide else 1.0))   # headroom for long dividing runs (~1500+ cells)
        print(f"[tyssue_rd] {name}: {rd['react']['implementation']} grow={grow} divide={divide} cv={cv}  (Nv={Nv}, cells={nF})", flush=True)
        rec = {"name": name, "react": rd["react"]["implementation"], "grow": grow, "divide": divide, "cv": cv, "Nv": Nv, "cells": nF}
        try:
            sim, cfg = make_spec(name, rd, n_cells, frames, grow, divide, cv, buf, cbuf)
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
