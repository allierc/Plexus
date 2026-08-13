#!/usr/bin/env python
"""run_tyssue_growth -- Goal 1(ter), the integration test: a GROWING, DIVIDING tissue. face_growth
inflates each cell's target area; when a cell's target exceeds a threshold it divides by a straight
line at a random angle (face_divide_line, cell-cycle mode, edge-midpoint -> well-shaped daughters,
each targeting half); shape_energy holds force balance and t1_transition resolves short junctions.
This exercises growth + division + mechanics + T1 together and, crucially, SUSTAINS many division
rounds without tangling -- the fix for the Stage-2 through-vertex placeholder that degraded after
1-2 rounds. before/after strip + movie.

    python run_tyssue_growth.py
    python run_tyssue_growth.py --montage
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
NX, NY, A, BORDER, JITTER, SEED = 11, 12, 1.0, 1, 0.15, 0
A0BASE = (np.sqrt(3) / 2.0) * A * A
FRAMES = 170


def presets():
    #     name       p0    rate    ratio
    return [("grow_divide", 3.75, 0.006, 1.6)]     # grow target area, divide at 1.6x base (cell cycle)


def make_spec(name, p0, rate, ratio, buf):
    cfg = {
        "general": {"name": f"tyssue_grow_{name}", "seed": SEED, "n_frames": FRAMES, "dt": 1.0,
                    "boundary": "free", "dim": 2, "world": [float(NX * A + 4), float(NY * A + 4)]},
        "sets": {"vertex": {"n": buf}},
        "fields": {},
        "operators": [
            {"op": "seed_mesh", "at": "vertex", "nx": NX, "ny": NY, "a": A, "border": BORDER,
             "jitter": JITTER, "p0": p0, "seed": SEED, "pin_border": False, "before_frame": 1},   # free -> expands
            {"op": "face_growth", "at": "vertex", "rate": rate, "p0": p0, "every": 1},
            {"op": "shape_energy", "at": "vertex", "p0": p0, "K_A": 1.0, "K_P": 1.0, "Lambda": 0.15,
             "Gamma": 0.05, "mu": 1.0, "dt": 1.0, "relax_iters": 8, "eta": 0.08, "cap_frac": 0.15},
            #  Lambda = line tension (smooths the free boundary), Gamma = cortical CONTRACTILITY (P^2) --
            #  penalises perimeter so daughters round out between divisions instead of pulling into points.
            {"op": "face_divide_line", "at": "vertex", "ratio": ratio, "a0_base": float(A0BASE),
             "p0": p0, "every": 3},                            # cell-cycle straight-line division
            {"op": "t1_transition", "at": "vertex", "l_th": 0.10, "p0": p0, "every": 2},
            {"op": "topo_snapshot", "at": "vertex"},
        ],
        "schedule": ["seed_mesh", "face_growth", "shape_energy", "face_divide_line",
                     "t1_transition", "topo_snapshot"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def run_all(only=None):
    for name, p0, rate, ratio in presets():
        if only and name not in only:
            continue
        odir = os.path.join(OUT, name); os.makedirs(odir, exist_ok=True)
        verts, es, et, ef, fc, pin, a0 = build_honeycomb(NX, NY, A, BORDER, JITTER, SEED)
        Nv = verts.shape[0]; F0 = int(ef.max()) + 1
        buf = Nv + 6 * F0 + 40                                 # generous headroom for sustained division
        faces0 = rings_from_flat(es, et, ef, F0)
        print(f"[tyssue_grow] {name}: p0={p0} rate={rate} ratio={ratio}  (Nv={Nv}, F={F0}, buf={buf})", flush=True)
        rec = {"name": name, "p0": p0, "rate": rate, "ratio": ratio, "Nv": Nv, "F_start": F0}
        try:
            sim, cfg = make_spec(name, p0, rate, ratio, buf)
            write_spec(cfg, os.path.join(odir, "spec.yaml"))
            Hf, out = engine_run(sim, device="cpu")
            pos = out["sets"]["vertex"]["pos"]
            m = Hf.level("vertex")._mesh
            F_end = int(sum(1 for r in m["faces"] if r is not None and len(r) >= 3))
            pend = pos[-1].astype(np.float64)
            invalid = sum(0 if r is None else (0 if ring_valid(r, pend) else 1) for r in m["faces"])
            rec.update(F_end=F_end, divided=F_end - F0, invalid_faces=int(invalid))
            # dezoom to fit ALL cells: the free boundary expands, so frame to the FINAL (largest) extent
            fv = np.unique(np.concatenate([r for r in m["faces"] if r is not None and len(r) >= 3]))
            P = pos[-1][fv]
            cx, cy = 0.5 * (P[:, 0].min() + P[:, 0].max()), 0.5 * (P[:, 1].min() + P[:, 1].max())
            half = 0.55 * max(P[:, 0].max() - P[:, 0].min(), P[:, 1].max() - P[:, 1].min())
            xlim = (cx - half, cx + half); ylim = (cy - half, cy + half)
            hist = [faces0] + m.get("hist_faces", [])
            Tm = min(len(hist), pos.shape[0])
            picks = [int(round(fr * (Tm - 1))) for fr in (0.0, 0.4, 0.7, 1.0)]
            sfig, sax = plt.subplots(1, 4, figsize=(17.6, 4.6)); sfig.patch.set_facecolor("black")
            for ax, tt in zip(sax, picks):
                nf = sum(1 for r in hist[tt] if r is not None and len(r) >= 3)
                _draw(ax, pos[tt].astype(np.float64), hist[tt], p0, a0,
                      f"{name}\ngrow + straight-line division\nF={nf} t={tt}", xlim, ylim)
            sfig.subplots_adjust(0.004, 0.004, 0.996, 0.996, wspace=0.03)
            sfig.savefig(os.path.join(odir, "strip.png"), dpi=120, facecolor="black"); plt.close(sfig)
            figm, axm = plt.subplots(figsize=(4.8, 5.2)); figm.patch.set_facecolor("black"); figm.subplots_adjust(0, 0, 1, 1)
            idx = list(range(0, Tm, max(1, Tm // 120)))
            wri = FFMpegWriter(fps=max(1, round(len(idx) / 9.0)), metadata={"title": name})
            with wri.saving(figm, os.path.join(odir, "movie.mp4"), dpi=120):
                for tt in idx:
                    nf = sum(1 for r in hist[tt] if r is not None and len(r) >= 3)
                    _draw(axm, pos[tt].astype(np.float64), hist[tt], p0, a0, "", xlim, ylim)   # no text on movie
                    wri.grab_frame()
            plt.close(figm)
            np.savez_compressed(os.path.join(odir, "traj.npz"), pos=pos[::4].astype("float32"), p0=p0)
            print(f"           -> F {F0}->{F_end} (+{F_end-F0} divisions), invalid={invalid}", flush=True)
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
        sheet.save(os.path.join(OUT, "_montage_tyssue_growth.png"))
    print(f"[tyssue_grow] montage -> {OUT}")


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
