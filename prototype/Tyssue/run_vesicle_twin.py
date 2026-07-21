#!/usr/bin/env python
"""Sanity twins of vesicle_grow_divide (the mechanics-only dividing vesicle) at 300 and 400 frames --
identical config, just longer. Archives strip (top: faithful white render; bottom: coloured red by the
hollow flag) + rotating movie + diag with hollow_frac (max/mean/final). Tests whether the 'ok' 220-frame
vesicle_grow_divide stays clean when run longer (per the investigation it should buckle)."""
from __future__ import annotations
import os, sys, json, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg; matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
import plexus.operators  # noqa
import tyssue_ops3d, tyssue_t1_ops3d  # noqa
from plexus.engine import run as engine_run
from tyssue_specfmt import write_spec
from tyssue_diag import hollow_flags
import run_tyssue_vesicle as V

OUT = os.path.join(HERE, "archive")


def run(frames):
    name = f"vesicle_grow_divide_{frames}f"; odir = os.path.join(OUT, name); os.makedirs(odir, exist_ok=True)
    mesh0 = V._mesh_from_build(V.NCELLS); Nv = mesh0["Nv"]; buf = int(Nv * 8.0)   # big buffer -> no early cap
    rec = {"name": name, "frames": frames, "grow": 0.003, "divide": True, "n_cells": V.NCELLS}
    try:
        # ALIGNED recording: topo_snapshot every=1 + record_cap>=frames so the mesh-history (hist) and the
        # position array (posf) are the SAME length -> hollow_flags/render pair matching topology+positions.
        # (The default sstride>1 recording misaligns them and fabricates a spurious 'hollow buckling'.)
        sim, cfg = V.make_spec(name, 3.72, buf, 0.003, True, V.NCELLS, frames)
        for op in cfg["operators"]:
            if op["op"] == "topo_snapshot_3d":
                op["every"] = 1
        cfg["general"]["record_cap"] = frames + 2
        import plexus.schema as _S, tempfile as _tf, yaml as _yaml
        with _tf.NamedTemporaryFile("w", suffix=".yaml", delete=False) as _fh:
            _yaml.safe_dump(cfg, _fh); _p = _fh.name
        sim = _S.load(_p); os.unlink(_p)
        write_spec(cfg, os.path.join(odir, "spec.yaml"))
        Hf, out = engine_run(sim, device="cpu")
        emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist")
        posf = out["sets"]["vertex"]["pos"]; T = posf.shape[0]

        def frame(t):
            mt = hist[min(t, len(hist) - 1)] if hist else mesh0
            return mt, posf[t][:mt["Nv"]].astype(np.float64)

        hs = []
        for tt in np.linspace(0, T - 1, 40).astype(int):
            mt, pt = frame(int(tt)); hs.append(hollow_flags(pt, mt)[2]["frac"])
        mtT, pT = frame(T - 1)
        rec.update(cells_end=int(mtT["nF"]), n_div=int(emesh.get("n_div", 0)),
                   hollow_max=round(float(max(hs)), 3), hollow_mean=round(float(np.mean(hs)), 3),
                   hollow_final=round(float(hs[-1]), 3))
        Rmax = max(float(np.linalg.norm(frame(t)[1], axis=1).max()) for t in np.linspace(0, T - 1, 20).astype(int))
        L3, L2 = Rmax * 1.06, Rmax * 2.23
        picks = [int(round(fr * (T - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]
        fig = plt.figure(figsize=(17.6, 9.0)); fig.patch.set_facecolor("black")
        for i, t in enumerate(picks):
            mt, pt = frame(t)
            _, sc, _ = hollow_flags(pt, mt)
            ax3 = fig.add_subplot(2, 4, i + 1, projection="3d"); V._draw(ax3, pt, mt, 3.72, azim=30, Lbox=L3)  # white
            ax2 = fig.add_subplot(2, 4, 4 + i + 1, projection="3d")
            V._draw(ax2, pt, mt, 3.72, azim=30, act=sc[:mt["nF"]], Lbox=L3)                                    # hollow-red
        fig.subplots_adjust(0.006, 0.005, 0.996, 0.996, wspace=0.02, hspace=0.02)
        fig.savefig(os.path.join(odir, "strip.png"), dpi=120, facecolor="black"); plt.close(fig)
        figm = plt.figure(figsize=(5.0, 5.2)); figm.patch.set_facecolor("black")
        axm = figm.add_subplot(111, projection="3d"); figm.subplots_adjust(0, 0, 1, 1)
        # FIXED stride + fps so movie length is PROPORTIONAL to frame count and COMPARABLE across runs:
        # keyframe j = recorded frame 3j at 12 fps -> duration = frames/36 s; the first part of a longer
        # movie is the same footage as a shorter one (recording is now aligned, so frame t is frame t).
        keep = np.arange(0, T, 3)
        wri = FFMpegWriter(fps=12, metadata={"title": name})
        with wri.saving(figm, os.path.join(odir, "movie.mp4"), dpi=110):
            for j, t in enumerate(keep):
                mt, pt = frame(t); V._draw(axm, pt, mt, 3.72, azim=(2 * j) % 360, Lbox=L3); wri.grab_frame()
        plt.close(figm)
        print(f"[{name}] cells {V.NCELLS}->{rec['cells_end']} (+{rec['n_div']} div)  "
              f"hollow max={rec['hollow_max']} mean={rec['hollow_mean']} final={rec['hollow_final']}", flush=True)
    except Exception as e:
        rec["error"] = repr(e); traceback.print_exc()
    json.dump(rec, open(os.path.join(odir, "diag.json"), "w"), indent=1)


import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--frames", nargs="*", type=int, default=[220, 300, 400])
for fr in ap.parse_args().frames:
    run(fr)
