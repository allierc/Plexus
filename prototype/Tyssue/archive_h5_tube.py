#!/usr/bin/env python
"""Archive the H5 breakthrough (the first PROTRUDING tubes): volume-locked body (rho=0, max_cycle=inf) +
N radial cones -> activated growth protrudes into tubes even with uniform-capped cells (protr ~5). Render
coloured by ACTIVATOR (cones -> red tips) + cross-section INSET movie (monolayer). archive/h5_tube."""
from __future__ import annotations
import os, sys, json, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg; matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
import plexus.operators  # noqa
import tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d  # noqa
from tyssue_ops3d import build_sphere_mesh, face_geometry_3d
from tyssue_specfmt import write_spec
from run_tyssue_vesicle import _draw, _draw_cross, make_movie_axes, draw_movie_frame
from tyssue_diag import hollow_metric, hollow_flags
from tyssue_topology_ops3d import rings_from_flat_3d
import run_tyssue_fig5 as F
import torch

NAME = "h5_tube"; CELLS, FR = 800, 300
OUT = os.path.join(HERE, "archive", NAME); os.makedirs(OUT, exist_ok=True)


def vol_cv(mt, pt):
    _, _, _, vf = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(np.asarray(mt["E_srce"])), torch.as_tensor(np.asarray(mt["E_trgt"])), torch.as_tensor(np.asarray(mt["E_face"])), mt["nF"])
    vf = vf.numpy(); vf = vf[np.abs(vf) > 1e-9]; return float(vf.std() / (np.abs(vf.mean()) + 1e-9))


rec = {"name": NAME}
try:
    verts, es, et, ef, nF = build_sphere_mesh(CELLS, F.RADIUS, F.JITTER, F.SEED); Nv = verts.shape[0]
    # fig5 make_spec: name n_cells frames n_spots cone grow cv rho vth K_V min max buf cbuf. rho=0 + max_cycle=inf = locked body
    sim, cfg = F.make_spec(NAME, CELLS, FR, 5, 12.0, 0.05, 0.15, 0.0, 1.5, 4.0, 0, 10 ** 9, int(Nv * 4), int(nF * 4))
    write_spec(cfg, os.path.join(OUT, "spec.yaml"))
    from plexus.engine import run as engine_run
    Hf, out = engine_run(sim, device="cpu")
    emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]; chemf = out["sets"]["cell"]["state"]["chem"]; T = posf.shape[0]
    def frame(t):
        mt = hist[min(t, len(hist) - 1)] if hist else dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=Nv)
        return mt, posf[t][:mt["Nv"]].astype(np.float64), chemf[t][:mt["nF"], 0]
    mtT, pT, aT = frame(T - 1); ext = pT.max(0) - pT.min(0)
    rings = rings_from_flat_3d(np.asarray(mtT["E_srce"]), np.asarray(mtT["E_trgt"]), np.asarray(mtT["E_face"]), mtT["nF"])
    rad = np.array([np.linalg.norm(pT[r].mean(0)) if (r is not None and len(r)) else 0 for r in rings]); rad = rad[rad > 0]
    rec.update(cells_end=int(mtT["nF"]), n_div=int(emesh.get("n_div", 0)),
               protr=round(float(np.percentile(rad, 95) / (np.median(rad) + 1e-9)), 3), vol_cv=round(vol_cv(mtT, pT), 3),
               hollow_frac=round(float(hollow_flags(pT, mtT)[2]["frac"]), 3), spots=int(F.count_spots(aT, mtT, float(np.percentile(aT, 70)))))
    Rmax = max(float(np.abs(frame(t)[1]).max()) for t in np.linspace(0, T - 1, 20).astype(int)); L3, L2 = Rmax * 1.06, Rmax * 2.23
    norm = lambda a: np.clip(a / max(float(a.max()), 1e-6), 0, 1)
    fig = plt.figure(figsize=(17.6, 9.0)); fig.patch.set_facecolor("black")
    for i, t in enumerate([int(round(fr * (T - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]):
        mt, pt, a = frame(t)
        ax3 = fig.add_subplot(2, 4, i + 1, projection="3d"); _draw(ax3, pt, mt, 3.90, azim=30, act=norm(a), Lbox=L3)
        ax2 = fig.add_subplot(2, 4, 4 + i + 1); _draw_cross(ax2, pt, mt, 3.90, act=norm(a), Lbox=L2, axis=1)
    fig.subplots_adjust(0.006, 0.005, 0.996, 0.996, wspace=0.02, hspace=0.02)
    fig.savefig(os.path.join(OUT, "strip.png"), dpi=120, facecolor="black"); plt.close(fig)
    figm = plt.figure(figsize=(5.0, 5.2)); figm.patch.set_facecolor("black"); axm, axin = make_movie_axes(figm)
    keep = np.arange(0, T, max(1, T // 150)); wri = FFMpegWriter(fps=12, metadata={"title": NAME})
    with wri.saving(figm, os.path.join(OUT, "movie.mp4"), dpi=110):
        for j, t in enumerate(keep):
            mt, pt, a = frame(int(t)); draw_movie_frame(axm, axin, pt, mt, 3.90, (2 * j) % 360, norm(a), L3, L2); wri.grab_frame()
    plt.close(figm)
    print(f"[{NAME}] cells {CELLS}->{rec['cells_end']} (+{rec['n_div']} div)  protr={rec['protr']} vol_cv={rec['vol_cv']} "
          f"hollow={rec['hollow_frac']} spots={rec['spots']} extent={[round(float(x),1) for x in ext]}", flush=True)
except Exception as e:
    rec["error"] = repr(e); traceback.print_exc()
json.dump(rec, open(os.path.join(OUT, "diag.json"), "w"), indent=1)
