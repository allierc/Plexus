#!/usr/bin/env python
"""run_tyssue_flow -- Stage 3 of the Tyssue AVM prototype: the ACTIVE self-propelled vertex model.
shape_energy gains a v0 polarity drift; t1_transition resolves the short junctions it creates.
Above the rigidity transition (p0 > p0*) the tissue FLOWS -- cells swap neighbours by T1 and the
mesh rearranges; below it, the solid stays caged and T1s are rare. This is the in-engine proof
that the explicit T1 operator fires and fluidises the tissue -- the flow the Self-Propelled-Voronoi
route needed reversible network reconnection for, and could not do with implicit topology.

    python run_tyssue_flow.py            # solid vs fluid, active driving; archive each (movie + strip)
    python run_tyssue_flow.py --montage
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
import ops_2d          # noqa: F401
import topology_ops_2d  # noqa: F401
from ops_2d import build_honeycomb
from topology_ops_2d import rings_from_flat, ring_valid
from run_tyssue_topology import _draw
import plexus.schema as S
from plexus.engine import run as engine_run
from specfmt import write_spec

OUT = os.path.join(HERE, "archive")
NX, NY, A, BORDER, JITTER, SEED = 14, 16, 1.0, 1, 0.20, 0
A0BASE = (np.sqrt(3) / 2.0) * A * A
FRAMES = 180


def presets():
    # grid to find a SOUND T1 dichotomy: for a matched (v0, l_th) the fluid must T1 >> the solid.
    # The dichotomy lives where the drive is comparable to the solid's energy barrier: too-hard
    # drive (or too-eager l_th) fluidises even the solid, too-gentle gives no T1s at all.
    g = []
    for tag, p0 in [("solid", 3.60), ("fluid", 4.10)]:
        for v0 in (0.06, 0.12):
            for l_th in (0.25, 0.35):
                g.append((f"flow_{tag}_v{int(v0*100):02d}_l{int(l_th*100):02d}", p0, v0, l_th, 4))
    return g


def make_spec(name, p0, v0, l_th, relax, Nv):
    cfg = {
        "general": {"name": f"tyssue_flow_{name}", "seed": SEED, "n_frames": FRAMES, "dt": 1.0,
                    "boundary": "free", "dim": 2, "world": [float(NX * A + 4), float(NY * A + 4)]},
        "sets": {"vertex": {"n": Nv}},
        "fields": {},
        "operators": [
            {"op": "seed_mesh", "at": "vertex", "nx": NX, "ny": NY, "a": A, "border": BORDER,
             "jitter": JITTER, "p0": p0, "seed": SEED, "before_frame": 1},
            {"op": "shape_energy", "at": "vertex", "p0": p0, "K_A": 1.0, "K_P": 1.0, "mu": 1.0,
             "dt": 1.0, "relax_iters": relax, "eta": 0.10, "v0": v0, "Dr": 1.0, "cap_frac": 0.15},
            # bounded overdamped Euler (differentiable) keeps it stable; l_th sets how short an edge
            # must get to flip. Fluid: T1s cheap (fire); solid: shape energy resists (few).
            {"op": "t1_transition", "at": "vertex", "l_th": l_th, "p0": p0, "every": 1},
            {"op": "topo_snapshot", "at": "vertex"},
        ],
        "schedule": ["seed_mesh", "shape_energy", "t1_transition", "topo_snapshot"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def run_all(only=None):
    recs = []
    for name, p0, v0, l_th, relax in presets():
        if only and name not in only:
            continue
        odir = os.path.join(OUT, name); os.makedirs(odir, exist_ok=True)
        verts, es, et, ef, fc, pin, a0 = build_honeycomb(NX, NY, A, BORDER, JITTER, SEED)
        Nv = verts.shape[0]; F0 = int(ef.max()) + 1
        faces0 = rings_from_flat(es, et, ef, F0)
        print(f"[tyssue_flow] {name}: p0={p0} v0={v0} l_th={l_th} relax={relax}  (Nv={Nv}, F={F0})", flush=True)
        rec = {"name": name, "p0": p0, "v0": v0, "l_th": l_th, "relax": relax, "Nv": Nv, "nF": F0}
        try:
            sim, cfg = make_spec(name, p0, v0, l_th, relax, Nv)
            write_spec(cfg, os.path.join(odir, "spec.yaml"))
            Hf, out = engine_run(sim, device="cpu")
            pos = out["sets"]["vertex"]["pos"][:, :Nv, :]
            m = Hf.level("vertex")._mesh
            n_t1 = int(m.get("n_t1", 0))
            # cage-breaking: mean squared displacement of interior vertices (t0 -> tT)
            free = ~pin
            msd = float(np.mean(np.sum((pos[-1][free] - pos[0][free]) ** 2, axis=1)))
            pend = pos[-1].astype(np.float64)
            invalid = sum(0 if r is None else (0 if ring_valid(r, pend) else 1) for r in m["faces"])
            rec.update(n_t1=n_t1, msd=round(msd, 4), invalid_faces=int(invalid))
            # render movie + strip from per-tick topology
            hist = [faces0] + m.get("hist_faces", [])
            Wx = verts[free, 0]; Wy = verts[free, 1]
            cx, cy = 0.5 * (Wx.min() + Wx.max()), 0.5 * (Wy.min() + Wy.max())
            half = 0.5 * max(Wx.max() - Wx.min(), Wy.max() - Wy.min())
            xlim = (cx - half, cx + half); ylim = (cy - half, cy + half)
            Tm = min(len(hist), pos.shape[0])
            picks = [int(round(fr * (Tm - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]
            sfig, sax = plt.subplots(1, 4, figsize=(17.6, 4.6)); sfig.patch.set_facecolor("black")
            for ax, t in zip(sax, picks):
                _draw(ax, pos[t].astype(np.float64), hist[t], p0, a0,
                      f"{name}\np0={p0} v0={v0}\nt={t}  T1(cum)", xlim, ylim)
            sfig.subplots_adjust(0.004, 0.004, 0.996, 0.996, wspace=0.03)
            sfig.savefig(os.path.join(odir, "strip.png"), dpi=120, facecolor="black"); plt.close(sfig)
            idx = list(range(0, Tm, max(1, Tm // 120)))
            figm, axm = plt.subplots(figsize=(4.8, 5.2)); figm.patch.set_facecolor("black"); figm.subplots_adjust(0, 0, 1, 1)
            wri = FFMpegWriter(fps=max(1, round(len(idx) / 9.0)), metadata={"title": name})
            with wri.saving(figm, os.path.join(odir, "movie.mp4"), dpi=120):
                for t in idx:
                    _draw(axm, pos[t].astype(np.float64), hist[t], p0, a0, "", xlim, ylim)   # no text on movie
                    wri.grab_frame()
            plt.close(figm)
            np.savez_compressed(os.path.join(odir, "traj.npz"), pos=pos[::4].astype("float32"), p0=p0)
            print(f"           -> T1(cum)={n_t1}  MSD={msd:.3f}  invalid={invalid}", flush=True)
        except Exception as e:
            rec["error"] = repr(e); traceback.print_exc()
        json.dump(rec, open(os.path.join(odir, "diag.json"), "w"), indent=1)
        recs.append(rec)
    return recs


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
        sheet.save(os.path.join(OUT, "_montage_tyssue_flow.png"))
    print(f"[tyssue_flow] montage -> {OUT}")


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
