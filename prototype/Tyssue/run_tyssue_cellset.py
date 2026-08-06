#!/usr/bin/env python
"""run_tyssue_cellset -- the plexus2-vs-tyssue demonstration: a genuine two-level hierarchy.

The model declares TWO sets -- `vertex` (pos, the mechanical DOF) and `cell` (a0, ctype, area,
perim, cen, the biological DOF) -- joined by the half-edge map. A `cell_geometry` AGGREGATE reads
the vertices into per-cell area/perimeter; `cell_paint` gives a central CLONE of cells a fate
(ctype=1) and a larger target area (a0 x gain) -- state that is NOT derivable from geometry; and
the cross-set `shape_energy` reads a0 from the cell set and drives the vertices, so the clone
BULGES. In tyssue the clone is a column of a dataframe; here it is state on a first-class set,
composed with the mechanics by typed Aggregate/Broadcast operators. That composability is the
point -- and the same cell set is what the reaction-diffusion (Goal 2) will live on.

    python run_tyssue_cellset.py            # run + archive (before/after strip + movie)
    python run_tyssue_cellset.py --montage
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
from matplotlib.collections import PolyCollection
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import plexus.operators   # noqa: F401
import tyssue_ops          # noqa: F401  seed_mesh + shape_energy (cross-set)
import tyssue_cell_ops     # noqa: F401  seed_cell + cell_geometry + cell_paint
from tyssue_ops import build_honeycomb, face_polygons
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_specfmt import write_spec

OUT = os.path.join(HERE, "archive")
NX, NY, A, BORDER, JITTER, SEED = 16, 18, 1.0, 1, 0.10, 0
FRAMES = 120


def presets():
    #     name             p0   amp  sigma  mode
    return [("morphogen_bulge", 3.72, 1.0, 1.5, "continuous"),   # Hill growth, coloured white->red by activator
            ("clone_bulge",     3.72, 1.0, 1.5, "discrete")]      # threshold -> a typed clone, coloured by fate


def make_spec(name, p0, amp, sigma, mode, Nv, Fbuf):
    a0_base = float((np.sqrt(3) / 2.0) * A * A)
    # the morphogen -> growth coupling differs by mode: a CONTINUOUS Hill response, or a DISCRETE
    # threshold that differentiates a typed clone (French flag). Both retire the hard-coded disc.
    growth = ({"op": "morphogen_growth", "at": "cell", "g": 1.6, "rho": 1.0, "a_sw": 0.4,
               "hill_n": 4.0, "a0_base": a0_base} if mode == "continuous" else
              {"op": "cell_differentiate", "at": "cell", "threshold": 0.4, "gain": 2.2,
               "a0_base": a0_base})
    gname = growth["op"]
    cfg = {
        "general": {"name": f"tyssue_cell_{name}", "seed": SEED, "n_frames": FRAMES, "dt": 1.0,
                    "boundary": "free", "dim": 2, "world": [float(NX * A + 4), float(NY * A + 4)]},
        "sets": {
            "vertex": {"n": Nv},
            "cell": {"n": Fbuf, "state": {"a0": {"width": 1}, "chem": {"width": 2},
                                          "ctype": {"width": 1}, "area": {"width": 1},
                                          "perim": {"width": 1}, "cen": {"width": 2}}},
        },
        "fields": {},
        "operators": [
            {"op": "seed_mesh", "at": "vertex", "nx": NX, "ny": NY, "a": A, "border": BORDER,
             "jitter": JITTER, "p0": p0, "seed": SEED, "before_frame": 1},
            {"op": "seed_cell", "at": "cell", "a": A, "before_frame": 1},
            {"op": "cell_geometry", "at": "cell"},
            {"op": "cell_morphogen", "at": "cell", "amp": amp, "sigma": sigma, "before_frame": 3},
            growth,
            {"op": "shape_energy", "at": "vertex", "p0": p0, "K_A": 1.0, "K_P": 1.0, "mu": 1.0,
             "dt": 1.0, "relax_iters": 8, "eta": 0.08, "cap_frac": 0.15},
        ],
        "schedule": ["seed_mesh", "seed_cell", "cell_geometry", "cell_morphogen",
                     gname, "shape_energy"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def _mesh(name=None):
    verts, es, et, ef, fc, pin, a0 = build_honeycomb(NX, NY, A, BORDER, JITTER, SEED)
    return verts, es, et, ef, pin, a0, int(ef.max()) + 1


def _draw(ax, pos, mesh, facecolors, title, xlim, ylim):
    ax.clear(); ax.set_facecolor("black")                          # black bg: white (low-activator) cells
    es, et, ef, nF = mesh                                          #   stand out from the void
    polys, _, _, _ = face_polygons(pos, dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF))
    pc = PolyCollection(polys, facecolors=facecolors, edgecolors=(0.45, 0.45, 0.45, 0.6), linewidths=0.5)
    ax.add_collection(pc)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")
    if title:                                                      # movies pass "" -> no text overlay
        ax.text(0.02, 0.98, title, transform=ax.transAxes, color="white", fontsize=8,
                va="top", family="monospace")


def run_all(only=None):
    for name, p0, amp, sigma, mode in presets():
        if only and name not in only:
            continue
        odir = os.path.join(OUT, name); os.makedirs(odir, exist_ok=True)
        verts, es, et, ef, pin, a0, nF = _mesh()
        Nv = verts.shape[0]; Fbuf = nF
        mesh = (es, et, ef, nF)
        print(f"[tyssue_cell] {name}: p0={p0} amp={amp} sigma={sigma} mode={mode}  (Nv={Nv}, cells={nF})", flush=True)
        rec = {"name": name, "p0": p0, "amp": amp, "sigma": sigma, "mode": mode, "Nv": Nv, "cells": nF}
        try:
            sim, cfg = make_spec(name, p0, amp, sigma, mode, Nv, Fbuf)
            write_spec(cfg, os.path.join(odir, "spec.yaml"))
            Hf, out = engine_run(sim, device="cpu")
            pos = out["sets"]["vertex"]["pos"][:, :Nv, :]
            clvl = Hf.level("cell"); sch = clvl.state_schema
            h0, _ = sch["chem"]; act = clvl.state[:nF, h0:h0 + 1].detach().cpu().numpy().ravel()
            ti0, _ = sch["ctype"]; ctype = clvl.state[:nF, ti0:ti0 + 1].detach().cpu().numpy().ravel()
            _, area, _, _ = face_polygons(pos[-1].astype(np.float64),
                                          dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF))
            # colour + SOUND diagnostic per mode
            if mode == "continuous":                                  # white(low) -> red(high) activator LUT
                vmax = float(max(act.max(), 1e-6))
                facecolors = plt.cm.Reds(np.clip(act / vmax, 0.0, 1.0))
                rec.update(corr_a_area=round(float(np.corrcoef(act, area)[0, 1]), 3))
                sub = "morphogen -> Hill growth (white->red = activator)"
                msg = f"corr(activator,area)={rec['corr_a_area']}"
            else:                                                     # discrete fate: clone red, wild-type white
                clone = ctype > 0.5
                facecolors = np.where(clone[:, None], np.array([0.80, 0.12, 0.10, 1.0]),
                                      np.array([0.97, 0.97, 0.97, 1.0]))
                rec.update(n_clone=int(clone.sum()),
                           clone_area=round(float(area[clone].mean()), 4),
                           wt_area=round(float(area[~clone].mean()), 4),
                           area_ratio=round(float(area[clone].mean() / max(area[~clone].mean(), 1e-9)), 3))
                sub = "morphogen -> threshold -> typed clone (red)"
                msg = f"clone {rec['n_clone']} cells, area clone/wt = {rec['area_ratio']}x"
            Wx = verts[~pin, 0]; Wy = verts[~pin, 1]
            cx, cy = 0.5 * (Wx.min() + Wx.max()), 0.5 * (Wy.min() + Wy.max())
            half = 0.6 * max(Wx.max() - Wx.min(), Wy.max() - Wy.min())     # dezoom to see all cells
            xlim = (cx - half, cx + half); ylim = (cy - half, cy + half)
            picks = [int(round(fr * (pos.shape[0] - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]
            sfig, sax = plt.subplots(1, 4, figsize=(17.6, 4.6)); sfig.patch.set_facecolor("black")
            for ax, t in zip(sax, picks):
                _draw(ax, pos[t].astype(np.float64), mesh, facecolors, f"{name}\n{sub}\nt={t}", xlim, ylim)
            sfig.subplots_adjust(0.004, 0.004, 0.996, 0.996, wspace=0.03)
            sfig.savefig(os.path.join(odir, "strip.png"), dpi=120, facecolor="black"); plt.close(sfig)
            figm, axm = plt.subplots(figsize=(4.8, 5.2)); figm.patch.set_facecolor("black"); figm.subplots_adjust(0, 0, 1, 1)
            idx = list(range(0, pos.shape[0], max(1, pos.shape[0] // 100)))
            wri = FFMpegWriter(fps=max(1, round(len(idx) / 7.0)), metadata={"title": name})
            with wri.saving(figm, os.path.join(odir, "movie.mp4"), dpi=120):
                for t in idx:
                    _draw(axm, pos[t].astype(np.float64), mesh, facecolors, "", xlim, ylim)   # no text on movie
                    wri.grab_frame()
            plt.close(figm)
            np.savez_compressed(os.path.join(odir, "traj.npz"), pos=pos[::3].astype("float32"),
                                act=act, ctype=ctype, es=es, et=et, ef=ef, nF=nF, p0=p0)
            print(f"           -> {msg}", flush=True)
        except Exception as e:
            rec["error"] = repr(e); traceback.print_exc()
        json.dump(rec, open(os.path.join(odir, "diag.json"), "w"), indent=1)


def montage():
    from PIL import Image
    names = [p[0] for p in presets()]
    ims = [Image.open(os.path.join(OUT, n, "strip.png")).convert("RGB")
           for n in names if os.path.exists(os.path.join(OUT, n, "strip.png"))]
    if ims:
        w = max(i.width for i in ims); h = sum(i.height for i in ims)
        sheet = Image.new("RGB", (w, h), (0, 0, 0)); y = 0
        for im in ims:
            sheet.paste(im, (0, y)); y += im.height
        sheet.save(os.path.join(OUT, "_montage_tyssue_cellset.png"))
    print(f"[tyssue_cell] montage -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--montage", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    if a.montage:
        montage()
    else:
        run_all(a.only); montage()


if __name__ == "__main__":
    main()
