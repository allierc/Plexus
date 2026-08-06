#!/usr/bin/env python
"""run_tyssue_topology -- Stage 2 of the Tyssue AVM prototype: the explicit topology operators
that the Self-Propelled-Voronoi route could not have. A force-balanced honeycomb GROWS its
per-cell target area (face_growth), DIVIDES cells past a threshold (face_divide, a septum
through two vertices), and resolves any sub-threshold junctions by reversible network
reconnection (t1_transition) -- all while shape_energy keeps the sheet at force balance.

Because the mesh topology changes during the run, we render a BEFORE/AFTER pair: the initial
honeycomb and the final proliferated tissue (final topology read back from the Hierarchy). The
point is not a pretty movie but a demonstration that division + T1 keep a VALID, force-balanced
mesh -- the mechanics the tubulation story needs.

    python run_tyssue_topology.py            # run presets, archive each
    python run_tyssue_topology.py --montage
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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import plexus.operators   # noqa: F401
import tyssue_ops          # noqa: F401  seed_mesh + shape_energy
import tyssue_topology_ops  # noqa: F401  face_growth + face_divide + t1_transition
from tyssue_ops import build_honeycomb, face_polygons
from tyssue_topology_ops import rings_from_flat
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_specfmt import write_spec

OUT = os.path.join(HERE, "archive")
NX, NY, A, BORDER, JITTER, SEED = 14, 16, 1.0, 1, 0.15, 0
A0BASE = (np.sqrt(3) / 2.0) * A * A
FRAMES = 150


def presets():
    #     name            p0   (unused)  frac    one-shot clonal division of `frac` of the
    return [("clonal_divide", 3.90, 0.0, 0.25)]         #  hexagons, then relax to force balance


def make_spec(name, p0, rate, ratio, Nv):
    cfg = {
        "general": {"name": f"tyssue_topo_{name}", "seed": SEED, "n_frames": FRAMES, "dt": 1.0,
                    "boundary": "free", "dim": 2, "world": [float(NX * A + 4), float(NY * A + 4)]},
        "sets": {"vertex": {"n": Nv}},
        "fields": {},
        "operators": [
            {"op": "seed_mesh", "at": "vertex", "nx": NX, "ny": NY, "a": A, "border": BORDER,
             "jitter": JITTER, "p0": p0, "seed": SEED, "before_frame": 1},
            {"op": "face_divide", "at": "vertex", "frac": ratio, "a0_base": float(A0BASE),
             "p0": p0, "before_frame": 2},                    # one-shot clonal division (frac of cells)
            {"op": "shape_energy", "at": "vertex", "p0": p0, "K_A": 1.0, "K_P": 1.0,
             "mu": 1.0, "dt": 1.0, "relax_iters": 6, "eta": 0.08},
            {"op": "t1_transition", "at": "vertex", "l_th": 0.08, "p0": p0, "every": 3},
            {"op": "topo_snapshot", "at": "vertex"},          # capture per-tick topology for the movie
        ],
        "schedule": ["seed_mesh", "face_divide", "shape_energy", "t1_transition", "topo_snapshot"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def _draw(ax, pos, faces, p0, a0, title, xlim, ylim):
    ax.clear(); ax.set_facecolor("black")
    m = _flatten(faces)
    polys, area, perim, shape = face_polygons(pos, m)
    norm = TwoSlopeNorm(vcenter=3.81, vmin=3.70, vmax=max(3.95, p0 + 0.1))
    pc = PolyCollection(polys, array=shape, cmap="coolwarm", norm=norm,
                        edgecolors=(1, 1, 1, 0.45), linewidths=0.6)
    ax.add_collection(pc)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")
    if title:                                                      # movies pass "" -> no text overlay
        ax.text(0.02, 0.98, title, transform=ax.transAxes, color="white", fontsize=8,
                va="top", family="monospace")


def _flatten(faces):
    es, et, ef = [], [], []
    for f, r in enumerate(faces):
        if r is None or len(r) < 3:
            continue
        k = len(r)
        for i in range(k):
            es.append(int(r[i])); et.append(int(r[(i + 1) % k])); ef.append(f)
    return dict(E_srce=np.array(es), E_trgt=np.array(et), E_face=np.array(ef), nF=len(faces))


def run_all(only=None):
    recs = []
    for name, p0, rate, ratio in presets():
        if only and name not in only:
            continue
        odir = os.path.join(OUT, name); os.makedirs(odir, exist_ok=True)
        verts, es, et, ef, fc, pin, a0 = build_honeycomb(NX, NY, A, BORDER, JITTER, SEED)
        Nv = verts.shape[0]; F0 = int(ef.max()) + 1
        faces0 = rings_from_flat(es, et, ef, F0)
        print(f"[tyssue_topo] {name}: p0={p0} rate={rate} ratio={ratio}  (Nv={Nv}, F0={F0})", flush=True)
        rec = {"name": name, "p0": p0, "rate": rate, "ratio": ratio, "Nv": Nv, "F_start": F0}
        try:
            sim, cfg = make_spec(name, p0, rate, ratio, Nv)
            write_spec(cfg, os.path.join(odir, "spec.yaml"))
            Hf, out = engine_run(sim, device="cpu")
            pos = out["sets"]["vertex"]["pos"][:, :Nv, :]
            m = Hf.level("vertex")._mesh
            faces_final = m["faces"]
            F_end = int(sum(1 for r in faces_final if r is not None and len(r) >= 3))
            # T1 count: pull the op instance off the Hierarchy if present
            n_t1 = 0
            for ob in getattr(Hf, "operators", []) or []:
                n_t1 = max(n_t1, int(getattr(ob, "n_t1", 0)))
            # validity of the final mesh
            pend = pos[-1].astype(np.float64)
            from tyssue_topology_ops import ring_valid
            invalid = sum(0 if (r is None) else (0 if ring_valid(r, pend) else 1) for r in faces_final)
            rec.update(F_end=F_end, divided=F_end - F0, n_t1=n_t1, invalid_faces=int(invalid))
            # before/after render
            Wx = verts[~pin, 0]; Wy = verts[~pin, 1]
            cx, cy = 0.5 * (Wx.min() + Wx.max()), 0.5 * (Wy.min() + Wy.max())
            half = 0.5 * max(Wx.max() - Wx.min(), Wy.max() - Wy.min())
            xlim = (cx - half, cx + half); ylim = (cy - half, cy + half)
            fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.6)); fig.patch.set_facecolor("black")
            _draw(ax[0], verts.astype(np.float64), faces0, p0, a0,
                  f"{name}  BEFORE\nF={F0}", xlim, ylim)
            _draw(ax[1], pend, faces_final, p0, a0,
                  f"{name}  AFTER\nF={F_end}  T1={n_t1}", xlim, ylim)
            fig.subplots_adjust(0.005, 0.005, 0.995, 0.995, wspace=0.03)
            fig.savefig(os.path.join(odir, "before_after.png"), dpi=130, facecolor="black"); plt.close(fig)
            # movie from the per-tick topology snapshots (topology changes during the run)
            hist = [faces0] + m.get("hist_faces", [])
            Tm = min(len(hist), pos.shape[0])
            idx = list(range(0, Tm, max(1, Tm // 120)))
            figm, axm = plt.subplots(figsize=(4.8, 5.2)); figm.patch.set_facecolor("black")
            figm.subplots_adjust(0, 0, 1, 1)
            wri = FFMpegWriter(fps=max(1, round(len(idx) / 8.0)), metadata={"title": name})
            with wri.saving(figm, os.path.join(odir, "movie.mp4"), dpi=120):
                for t in idx:
                    nf = sum(1 for r in hist[t] if r is not None and len(r) >= 3)
                    _draw(axm, pos[t].astype(np.float64), hist[t], p0, a0,
                          f"{name}\nF={nf}  t={t} ({int(100*t/max(Tm-1,1))}%)", xlim, ylim)
                    wri.grab_frame()
            plt.close(figm)
            np.savez_compressed(os.path.join(odir, "traj.npz"), pos=pos[::4].astype("float32"), p0=p0)
            print(f"           -> F {F0}->{F_end} (+{F_end-F0}), T1={n_t1}, invalid={invalid}", flush=True)
        except Exception as e:
            rec["error"] = repr(e); traceback.print_exc()
        json.dump(rec, open(os.path.join(odir, "diag.json"), "w"), indent=1)
        recs.append(rec)
    return recs


def montage():
    from PIL import Image
    names = [p[0] for p in presets()]
    ims = [Image.open(os.path.join(OUT, n, "before_after.png")).convert("RGB")
           for n in names if os.path.exists(os.path.join(OUT, n, "before_after.png"))]
    if ims:
        w = max(i.width for i in ims); h = sum(i.height for i in ims)
        sheet = Image.new("RGB", (w, h), (0, 0, 0)); y = 0
        for im in ims:
            sheet.paste(im, (0, y)); y += im.height
        sheet.save(os.path.join(OUT, "_montage_tyssue_topology.png"))
    print(f"[tyssue_topo] montage -> {OUT}")


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
