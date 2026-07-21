#!/usr/bin/env python
"""run_tyssue_apoptosis -- Goal 1, recapitulating tyssue-demo B-Apoptosis (Monier et al. 2015).
A force-balanced sheet with one apoptotic cell: its target area shrinks each tick, T1 lets it shed
neighbours, and once small it is EXTRUDED (its vertices merged to a point, face_collapse). The
tissue closes the gap by force balance. before/after strip + movie.

    python run_tyssue_apoptosis.py
    python run_tyssue_apoptosis.py --montage
"""
from __future__ import annotations
import os, sys, argparse, json, tempfile, traceback, math

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
import tyssue_ops          # noqa: F401
import tyssue_topology_ops  # noqa: F401
from tyssue_ops import build_honeycomb, face_polygons
from tyssue_topology_ops import rings_from_flat, ring_valid
from run_tyssue_topology import _draw, _flatten
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm


def _draw_apop(ax, pos, faces, dying, p0, title, xlim, ylim):
    """Colour cells by shape index (blue solid / red fluid), but paint the DYING cell(s) near-black so
    the viewer watches the specific apoptotic cell shrink and vanish while its neighbours close in."""
    ax.clear(); ax.set_facecolor("black")
    m = _flatten(faces); nF = m["nF"]
    polys, area, perim, shape = face_polygons(pos, m)
    norm = TwoSlopeNorm(vcenter=3.81, vmin=3.70, vmax=max(3.95, p0 + 0.1)); cmap = plt.cm.coolwarm
    cols = []
    for f in range(nF):
        if faces[f] is not None and f in dying:
            cols.append((0.32, 0.32, 0.32, 1.0))                # apoptotic cell: DARK GREY (red is the shape LUT)
        else:
            cols.append(cmap(norm(shape[f] if np.isfinite(shape[f]) else 3.81)))
    pc = PolyCollection(polys, facecolors=cols, edgecolors=(1, 1, 1, 0.35), linewidths=0.5)
    ax.add_collection(pc)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")
    if title:                                                      # movies pass "" -> no text overlay
        ax.text(0.02, 0.98, title, transform=ax.transAxes, color="white", fontsize=8, va="top", family="monospace")
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_specfmt import write_spec

OUT = os.path.join(HERE, "archive")
NX, NY, A, BORDER, JITTER, SEED = 9, 10, 1.0, 1, 0.12, 0    # fewer, larger cells -> clearer demo
A0BASE = (np.sqrt(3) / 2.0) * A * A
FRAMES = 150


def presets():
    #     name         p0    n_apoptotic
    return [("apoptose", 3.72, 1)]      # single-cell apoptosis (as in the notebook) -> tissue heals cleanly


def make_spec(name, p0, cells, Nv):
    cfg = {
        "general": {"name": f"tyssue_apop_{name}", "seed": SEED, "n_frames": FRAMES, "dt": 1.0,
                    "boundary": "free", "dim": 2, "world": [float(NX * A + 4), float(NY * A + 4)]},
        "sets": {"vertex": {"n": Nv}},
        "fields": {},
        "operators": [
            {"op": "mesh_seed", "at": "vertex", "nx": NX, "ny": NY, "a": A, "border": BORDER,
             "jitter": JITTER, "p0": p0, "seed": SEED, "before_frame": 1},
            {"op": "shape_energy", "at": "vertex", "p0": p0, "K_A": 1.0, "K_P": 1.0, "Lambda": 0.12,
             "Gamma": 0.05, "mu": 1.0, "dt": 1.0, "relax_iters": 8, "eta": 0.08, "cap_frac": 0.15},
            #  Lambda = surface tension (line tension), Gamma = cell CONTRACTILITY (cortical tension) --
            #  both round the cells and uniformise the healed neighbourhood.
            {"op": "apoptosis", "at": "vertex", "cells": cells, "shrink_rate": 0.03,
             "critical_frac": 0.45, "a0_base": float(A0BASE), "p0": p0},   # gentle shrink; collapse only the
            #                                                               final triangle (degree-3, no rosette)
            # l_th catches the SHRINKING cell's (short) edges so t1_transition sheds its sides one at a
            # time (6->5->4->3) as it contracts; surface tension keeps the intermediates round.
            {"op": "t1_transition", "at": "vertex", "l_th": 0.16, "p0": p0, "every": 2},
            {"op": "topo_snapshot", "at": "vertex"},
        ],
        "schedule": ["mesh_seed", "shape_energy", "apoptosis", "t1_transition", "topo_snapshot"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def run_all(only=None):
    for name, p0, nap in presets():
        if only and name not in only:
            continue
        odir = os.path.join(OUT, name); os.makedirs(odir, exist_ok=True)
        verts, es, et, ef, fc, pin, a0 = build_honeycomb(NX, NY, A, BORDER, JITTER, SEED)
        Nv = verts.shape[0]; F0 = int(ef.max()) + 1
        # apoptotic cells: the nap faces nearest the tissue centre
        c = fc.mean(0); order = np.argsort(np.linalg.norm(fc - c, axis=1))
        cells = [int(x) for x in order[:nap]]
        faces0 = rings_from_flat(es, et, ef, F0)
        print(f"[tyssue_apop] {name}: p0={p0} apoptotic cells={cells}  (Nv={Nv}, F={F0})", flush=True)
        rec = {"name": name, "p0": p0, "cells": cells, "Nv": Nv, "F_start": F0}
        try:
            sim, cfg = make_spec(name, p0, cells, Nv)
            write_spec(cfg, os.path.join(odir, "spec.yaml"))
            Hf, out = engine_run(sim, device="cpu")
            pos = out["sets"]["vertex"]["pos"][:, :Nv, :]
            m = Hf.level("vertex")._mesh
            F_end = int(sum(1 for r in m["faces"] if r is not None and len(r) >= 3))
            pend = pos[-1].astype(np.float64)
            invalid = sum(0 if r is None else (0 if ring_valid(r, pend) else 1) for r in m["faces"])
            rec.update(F_end=F_end, eliminated=F0 - F_end, invalid_faces=int(invalid),
                       n_t1=int(m.get("n_t1", 0)))            # T1s fired (the dying cell shedding its sides)
            Wx = verts[~pin, 0]; Wy = verts[~pin, 1]
            cx, cy = 0.5 * (Wx.min() + Wx.max()), 0.5 * (Wy.min() + Wy.max())
            half = 0.72 * max(Wx.max() - Wx.min(), Wy.max() - Wy.min())    # dezoom to see all cells
            xlim = (cx - half, cx + half); ylim = (cy - half, cy + half)
            hist = [faces0] + m.get("hist_faces", [])
            Tm = min(len(hist), pos.shape[0])
            picks = [int(round(fr * (Tm - 1))) for fr in (0.0, 0.25, 0.5, 1.0)]
            sfig, sax = plt.subplots(1, 4, figsize=(17.6, 4.6)); sfig.patch.set_facecolor("black")
            dying = set(cells)
            for ax, tt in zip(sax, picks):
                nf = sum(1 for r in hist[tt] if r is not None and len(r) >= 3)
                _draw_apop(ax, pos[tt].astype(np.float64), hist[tt], dying, p0,
                           f"{name}\napoptosis (dying cell grey)\nF={nf} t={tt}", xlim, ylim)
            sfig.subplots_adjust(0.004, 0.004, 0.996, 0.996, wspace=0.03)
            sfig.savefig(os.path.join(odir, "strip.png"), dpi=120, facecolor="black"); plt.close(sfig)
            figm, axm = plt.subplots(figsize=(4.8, 5.4)); figm.patch.set_facecolor("black"); figm.subplots_adjust(0, 0, 1, 1)
            idx = list(range(0, Tm, max(1, Tm // 120)))
            wri = FFMpegWriter(fps=max(1, round(len(idx) / 8.0)), metadata={"title": name})
            with wri.saving(figm, os.path.join(odir, "movie.mp4"), dpi=120):
                for tt in idx:
                    nf = sum(1 for r in hist[tt] if r is not None and len(r) >= 3)
                    _draw_apop(axm, pos[tt].astype(np.float64), hist[tt], dying, p0, "", xlim, ylim)
                    wri.grab_frame()
            plt.close(figm)
            np.savez_compressed(os.path.join(odir, "traj.npz"), pos=pos[::4].astype("float32"), p0=p0)
            print(f"           -> F {F0}->{F_end} ({F0-F_end} eliminated), T1s={rec['n_t1']}, invalid={invalid}", flush=True)
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
        sheet.save(os.path.join(OUT, "_montage_tyssue_apoptosis.png"))
    print(f"[tyssue_apop] montage -> {OUT}")


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
