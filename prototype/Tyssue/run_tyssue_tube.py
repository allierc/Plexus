#!/usr/bin/env python
"""run_tyssue_tube -- bridge step 3 (Turing_vertex Fig 5): morphogen-driven growth+division on a
FLUID vesicle with T1 reconnection -> does a localized bud EXTEND into a tube instead of jamming?

A localized activator patch (top pole, seed_cell_rd mode=patch) drives grow_3d (grow the
target volume where a is high) and, because those cells inflate past their doubling volume, divide_3d
(which propagates the activator to daughters, so the growing region stays activated and proliferates).
The shell is FLUID (p0 > p0* ~ 3.81) and reconnect_t1_3d fires every tick so the tissue can flow /
intercalate -- the precondition (Bi 2015; Okuda dt_r << tau_cycle) for a bud to extend into a coherent
tube rather than a jammed spike (the SPV route's protr_max ~ 1.4 ceiling). Exploratory: render the
deforming shell + cross-section, coloured white->red by activator. strip + movie.

    python run_tyssue_tube.py
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
import tyssue_ops3d        # noqa: F401
import tyssue_rd_ops       # noqa: F401  cell_geometry_3d + seed_cell_rd + grow_3d
import tyssue_t1_ops3d     # noqa: F401  reconnect_t1_3d
from tyssue_ops3d import build_sphere_mesh
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_specfmt import write_spec
from run_tyssue_vesicle import _draw, _draw_cross
from tyssue_diag import hollow_flags     # geometric hollow-cell diagnostic (folded/tiny caps -> grey walls)

OUT = os.path.join(HERE, "archive")
RADIUS, JITTER, SEED = 5.0, 0.16, 0


def presets():
    #      name     n_cells frames p0    grow_rate a_sw patch_z l_th_frac  cv   (Okuda Fig 5: quasi-static + fluid)
    #  growth-rate sweep (tip mitogen rate): faster tip growth -> longer/thinner tube (Fig 5 tubulation).
    return [("tube_1", 400,  500,   3.90, 0.006,    0.5, 0.90,   0.28,     0.4),
            ("tube_2", 400,  500,   3.90, 0.010,    0.5, 0.90,   0.28,     0.4),
            ("tube_3", 400,  500,   3.90, 0.012,    0.5, 0.90,   0.28,     0.4),
            ("tube_4", 400,  500,   3.90, 0.008,    0.5, 0.90,   0.28,     0.4)]


def make_spec(name, n_cells, frames, p0, grow_rate, a_sw, patch_z, l_th_frac, cv, buf, cbuf, g1=False):
    # NB: g1=True (birth-at-target daughter v_eq) was A/B-tested and made the tube WORSE -- fresh daughters
    # born too small -> short-edge T1 storm (2203->5287 flips) -> hollow_max 0.10->0.15, aspect 1.56->1.43.
    # Kept as an off-by-default option; the real hollow fix is elsewhere (see rd_coral_grow_long study).
    # Fig 5 recipe (Okuda): a localized activator patch (mitogen) grows a cell's target volume; on a
    # FLUID shell (p0 > 3.81, T1 fired every tick so d_tr << tau_cycle) the excess area flows into a
    # coherent TUBE instead of jamming/spiking -- the coherent-neck the SPV could not hold. Growth is
    # QUASI-STATIC (slow rate + many relax iters, so force balance is reached between growth steps), the
    # radial term is ~OFF (the bud MUST leave the sphere; per-cell volume elasticity keeps the base
    # smooth), and division is DESYNCHRONISED (stochastic cell cycle) so the proliferating tip stays smooth.
    ops = [
        {"op": "seed_mesh_3d", "at": "vertex", "n_cells": n_cells, "radius": RADIUS,
         "jitter": JITTER, "p0": p0, "seed": SEED, "before_frame": 1, "vseed_cv": cv},
        {"op": "cell_geometry_3d", "at": "cell"},              # cen (needed by the patch seed) each frame
        {"op": "seed_cell_rd", "at": "cell", "mode": "patch", "patch_z": patch_z},  # RE-SEED EVERY FRAME:
        #   activator tracks the current TIP (top z-band) -> growth stays confined to the advancing tip and
        #   cells left behind switch OFF, so the neck constricts into a TUBE (vs the static patch's broad dome).
        {"op": "grow_3d", "at": "vertex", "cell_set": "cell", "rate": grow_rate,
         "a_sw": a_sw, "hill": 4.0, "cap": 4.0},               # grow v_eq where activator high (the mitogen)
        {"op": "shape_energy_3d", "at": "vertex", "p0": p0, "K_A": 1.0, "K_P": 1.0, "Gamma": 0.05,
         "Lambda": 0.2, "K_V": 1.0, "K_R": 0.02, "mu": 1.0, "dt": 1.0, "relax_iters": 40,
         "eta": 0.08, "cap_frac": 0.12},                       # fluid; radial ~OFF so the tube can leave the sphere;
        #   relax_iters high -> QUASI-STATIC (force balance reached each frame -> fresh tip cells relax -> fewer holes)
        {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": l_th_frac, "every": 1,
         "max_flips": max(40, n_cells // 8)},                  # T1 EVERY tick (d_tr << tau_cycle) -> bud flows to tube
        {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "cycle_cv": cv,
         "p0": p0, "every": 2, "max_div": max(10, n_cells // 20), "cell_set": "cell",
         "g1_ramp": g1},  # stochastic cycle + G1 ramp (daughter v_eq = birth volume) -> smooth proliferating tip
        {"op": "topo_snapshot_3d", "at": "vertex", "every": 1},   # record EVERY frame -> posf/hist aligned
    ]
    sched = ["seed_mesh_3d", "cell_geometry_3d", "seed_cell_rd", "grow_3d",
             "shape_energy_3d", "reconnect_t1_3d", "divide_3d", "topo_snapshot_3d"]
    cfg = {
        "general": {"name": f"tyssue_tube_{name}", "seed": SEED, "n_frames": frames, "dt": 1.0, "record_cap": frames + 2,
                    "boundary": "free", "dim": 3, "world": [12 * RADIUS, 12 * RADIUS, 12 * RADIUS]},
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
    for name, n_cells, frames, p0, grow_rate, a_sw, patch_z, l_th_frac, cv in presets():
        if only and name not in only:
            continue
        odir = os.path.join(OUT, name); os.makedirs(odir, exist_ok=True)
        mesh0, nF = _mesh(n_cells); Nv = mesh0["Nv"]
        buf, cbuf = int(Nv * 5.0), int(nF * 5.0)
        print(f"[tyssue_tube] {name}: p0={p0} grow={grow_rate} patch_z={patch_z} cv={cv}  (Nv={Nv}, cells={nF})", flush=True)
        rec = {"name": name, "p0": p0, "grow": grow_rate, "Nv": Nv, "cells": nF}
        try:
            sim, cfg = make_spec(name, n_cells, frames, p0, grow_rate, a_sw, patch_z, l_th_frac, cv, buf, cbuf)
            write_spec(cfg, os.path.join(odir, "spec.yaml"))
            Hf, out = engine_run(sim, device="cpu")
            emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist")
            posf = out["sets"]["vertex"]["pos"]; chemf = out["sets"]["cell"]["state"]["chem"]
            T = posf.shape[0]

            def frame(t):
                mt = hist[min(t, len(hist) - 1)] if hist else mesh0
                return mt, posf[t][:mt["Nv"]].astype(np.float64), chemf[t][:mt["nF"], 0]

            # aspect ratio: max extent / min extent of the final shell -> "protrusion" measure
            mtT, pT, _ = frame(T - 1); ext = pT.max(0) - pT.min(0)
            _, _, hstat = hollow_flags(pT, mtT)                # geometric hollow-cell diagnostic (final frame)
            # WORST-over-time hollow fraction: the tip defect peaks DURING active growth, not at the annealed
            # final frame, so sample the whole rollout -> a sharper soundness number for the G1/quality fixes.
            hollow_series = []
            for tt in np.linspace(0, T - 1, 24).astype(int):
                mtt, ptt, _ = frame(int(tt))
                hollow_series.append(hollow_flags(ptt, mtt)[2]["frac"])
            rec.update(cells_end=int(mtT["nF"]), n_div=int(emesh.get("n_div", 0)),
                       n_t1=int(emesh.get("n_t1", 0)), extent=[round(float(x), 2) for x in ext],
                       aspect=round(float(ext.max() / max(ext.min(), 1e-6)), 3),
                       hollow_frac=round(hstat["frac"], 3), hollow_dev_mean=round(hstat["dev_mean"], 1),
                       hollow_max=round(float(max(hollow_series)), 3),
                       hollow_mean=round(float(np.mean(hollow_series)), 3))
            Rmax = max(float(np.abs(frame(t)[1]).max()) for t in np.linspace(0, T - 1, 20).astype(int))
            L3 = Rmax * 1.08
            fig = plt.figure(figsize=(17.6, 9.0)); fig.patch.set_facecolor("black")
            picks = [int(round(fr * (T - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]
            for i, t in enumerate(picks):
                mt, pt, a = frame(t)
                act = np.clip(a / max(float(a.max()), 1e-6), 0, 1)
                ax3 = fig.add_subplot(2, 4, i + 1, projection="3d")
                _draw(ax3, pt, mt, p0, azim=30, act=act, Lbox=L3)
                ax2 = fig.add_subplot(2, 4, 4 + i + 1)
                _draw_cross(ax2, pt, mt, p0, act=act, Lbox=L3, axis=1)   # VERTICAL longitudinal cut up the tube axis
            fig.subplots_adjust(0.006, 0.005, 0.996, 0.996, wspace=0.02, hspace=0.02)
            fig.savefig(os.path.join(odir, "strip.png"), dpi=120, facecolor="black"); plt.close(fig)
            figm = plt.figure(figsize=(5.0, 5.2)); figm.patch.set_facecolor("black")
            axm = figm.add_subplot(111, projection="3d"); figm.subplots_adjust(0, 0, 1, 1)
            keep = np.linspace(0, T - 1, min(T, 60)).astype(int)
            wri = FFMpegWriter(fps=max(1, round(len(keep) / 8.0)), metadata={"title": name})
            with wri.saving(figm, os.path.join(odir, "movie.mp4"), dpi=110):
                for j, t in enumerate(keep):
                    mt, pt, a = frame(t)
                    _draw(axm, pt, mt, p0, azim=(2 * j) % 360, act=np.clip(a / max(float(a.max()), 1e-6), 0, 1), Lbox=L3)
                    wri.grab_frame()
            plt.close(figm)
            print(f"           -> cells {nF}->{rec['cells_end']} (+{rec['n_div']} div, {rec['n_t1']} T1)  "
                  f"extent {rec['extent']} aspect {rec['aspect']}  hollow={rec['hollow_frac']} "
                  f"(dev {rec['hollow_dev_mean']} deg)", flush=True)
        except Exception as e:
            rec["error"] = repr(e); traceback.print_exc()
        json.dump(rec, open(os.path.join(odir, "diag.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*", default=None)
    run_all(ap.parse_args().only)


if __name__ == "__main__":
    main()
