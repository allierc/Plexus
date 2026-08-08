#!/usr/bin/env python
"""Twin of vh_K4_cv15_d4_rd but with the CORAL (Gray-Scott labyrinth) pattern instead of Brusselator
spots, UNMODULATED growth (a_sw=50), and a movie that carries the cross-section INSET (bottom-right, no
box) so it reads as a true monolayer. Homogenised recipe (K_V=4, cv=0.15, dur, v_eq cap) + GS coral RD."""
from __future__ import annotations
import os, sys, json, tempfile, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg; matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
import plexus.operators  # noqa
import tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d  # noqa
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_ops3d import build_sphere_mesh, face_geometry_3d
from tyssue_specfmt import write_spec
from run_tyssue_vesicle import _draw, _draw_cross, make_movie_axes, draw_movie_frame
from tyssue_diag import hollow_metric, hollow_flags
import torch

R, J, SEED, FR = 5.0, 0.18, 0, 500
NAME = "vh_K4_cv15_d4_rd_coral"
OUT = os.path.join(HERE, "archive", NAME); os.makedirs(OUT, exist_ok=True)


def make():
    verts, es, et, ef, nF = build_sphere_mesh(150, R, J, SEED); Nv = verts.shape[0]
    ops = [{"op": "seed_mesh_3d", "at": "vertex", "n_cells": 150, "radius": R, "jitter": J, "p0": 3.72, "seed": SEED, "before_frame": 1, "vseed_cv": 0.15},
           {"op": "cell_geometry_3d", "at": "cell"}, {"op": "cell_adjacency", "at": "cell"},
           {"op": "seed_cell_rd", "at": "cell", "seed": SEED, "before_frame": 3, "mode": "scatter", "seed_frac": 0.06},
           {"op": "cell_diffuse", "at": "cell", "d_a": 0.08, "d_h": 0.16, "chi": 1.3},                    # Gray-Scott coral
           {"op": "cell_react", "at": "cell", "model": "gray_scott", "F": 0.055, "kk": 0.062, "rate": 1.0},
           {"op": "grow_3d", "at": "vertex", "cell_set": "cell", "rate": 0.03, "a_sw": 50.0, "hill": 4.0, "rho": 1.0, "vth_frac": 1.4},
           {"op": "shape_energy_3d", "at": "vertex", "p0": 3.72, "K_A": 1.0, "K_P": 1.0, "Gamma": 0.1, "Lambda": 0.5, "K_V": 4.0, "K_R": 0.4, "mu": 1.0, "dt": 1.0, "relax_iters": 26, "eta": 0.08, "cap_frac": 0.12},
           {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.35, "every": 2, "max_flips": 30},
           {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "cycle_cv": 0.15, "p0": 3.72, "every": 2, "max_div": 12, "max_div_frac": 0.03, "cell_set": "cell", "min_cycle": 4, "max_cycle": 12},
           {"op": "topo_snapshot_3d", "at": "vertex", "every": 1}]
    sched = ["seed_mesh_3d", "cell_geometry_3d", "cell_adjacency", "seed_cell_rd", "cell_diffuse", "cell_react", "grow_3d", "shape_energy_3d", "reconnect_t1_3d", "divide_3d", "topo_snapshot_3d"]
    cfg = {"general": {"name": f"tyssue_{NAME}", "seed": SEED, "n_frames": FR, "dt": 1.0, "record_cap": FR + 2, "boundary": "free", "dim": 3, "world": [10 * R] * 3},
           "sets": {"vertex": {"n": int(Nv * 12)}, "cell": {"n": int(nF * 12), "state": {"chem": {"width": 2, "integration": "first_order"}, "cen": {"width": 3}, "area": {"width": 1}}}},
           "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path); return sim, cfg, dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=Nv)


def vol_cv(mt, pt):
    _, _, _, vf = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(np.asarray(mt["E_srce"])), torch.as_tensor(np.asarray(mt["E_trgt"])), torch.as_tensor(np.asarray(mt["E_face"])), mt["nF"])
    vf = vf.numpy(); vf = vf[np.abs(vf) > 1e-9]; return float(vf.std() / (np.abs(vf.mean()) + 1e-9))


rec = {"name": NAME}
try:
    sim, cfg, mesh0 = make(); write_spec(cfg, os.path.join(OUT, "spec.yaml"))
    Hf, out = engine_run(sim, device="cpu")
    emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]; chemf = out["sets"]["cell"]["state"]["chem"]; T = posf.shape[0]
    def frame(t):
        mt = hist[min(t, len(hist) - 1)] if hist else mesh0
        return mt, posf[t][:mt["Nv"]].astype(np.float64), chemf[t][:mt["nF"], 0]
    mtT, pT, aT = frame(T - 1); _, arT, _ = hollow_metric(pT, mtT); arT = arT[arT > 0]
    rec.update(cells_end=int(mtT["nF"]), n_div=int(emesh.get("n_div", 0)), area_cv=round(float(arT.std() / arT.mean()), 3),
               vol_cv=round(vol_cv(mtT, pT), 3), hollow_frac=round(float(hollow_flags(pT, mtT)[2]["frac"]), 3),
               activator_range=[round(float(aT.min()), 3), round(float(aT.max()), 3)], patterned=bool(aT.std() > 0.03))
    lo, hi = float(np.percentile(aT, 5)), float(np.percentile(aT, 99) + 1e-6)
    Rmax = max(float(np.linalg.norm(frame(t)[1], axis=1).max()) for t in np.linspace(0, T - 1, 20).astype(int)); L3, L2 = Rmax * 1.06, Rmax * 2.23
    norm = lambda a: np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    fig = plt.figure(figsize=(17.6, 9.0)); fig.patch.set_facecolor("black")
    for i, t in enumerate([int(round(fr * (T - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]):
        mt, pt, a = frame(t)
        ax3 = fig.add_subplot(2, 4, i + 1, projection="3d"); _draw(ax3, pt, mt, 3.72, azim=30, act=norm(a), Lbox=L3)
        ax2 = fig.add_subplot(2, 4, 4 + i + 1); _draw_cross(ax2, pt, mt, 3.72, act=norm(a), Lbox=L2)
    fig.subplots_adjust(0.006, 0.005, 0.996, 0.996, wspace=0.02, hspace=0.02)
    fig.savefig(os.path.join(OUT, "strip.png"), dpi=120, facecolor="black"); plt.close(fig)
    figm = plt.figure(figsize=(5.0, 5.2)); figm.patch.set_facecolor("black"); axm, axin = make_movie_axes(figm)   # 3D + cross-section inset
    keep = np.arange(0, T, max(1, T // 150)); wri = FFMpegWriter(fps=12, metadata={"title": NAME})
    with wri.saving(figm, os.path.join(OUT, "movie.mp4"), dpi=110):
        for j, t in enumerate(keep):
            mt, pt, a = frame(int(t)); draw_movie_frame(axm, axin, pt, mt, 3.72, (2 * j) % 360, norm(a), L3, L2); wri.grab_frame()
    plt.close(figm)
    print(f"[{NAME}] cells 150->{rec['cells_end']} (+{rec['n_div']} div)  area_cv={rec['area_cv']} vol_cv={rec['vol_cv']} "
          f"hollow={rec['hollow_frac']}  activator={rec['activator_range']} patterned={rec['patterned']}", flush=True)
except Exception as e:
    rec["error"] = repr(e); traceback.print_exc()
json.dump(rec, open(os.path.join(OUT, "diag.json"), "w"), indent=1)
