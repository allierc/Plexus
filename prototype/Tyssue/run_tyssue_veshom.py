#!/usr/bin/env python
"""run_tyssue_veshom -- Round 2: HOMOGENISE cells on a growing/dividing vesicle (the clean test, no tubes,
fixing vesicle_grow_divide_500f). Key change vs vesicle_grow_divide: it grew the shell by INFLATING cells
(grow_3d ramps V0f -> big non-uniform cells); here the shell grows by PROLIFERATION (Okuda) --
grow_3d with rho=1 baseline (all cells, no activator gradient) and v_eq CAPPED at vth*v_ref,
so every cell cycles in [~2/3,vth]*v_ref (uniform) and the sphere grows because there are MORE cells.
divide_3d volume-primary + bounded cell-cycle DURATION (min/max_cycle). Metric = area_cv (cell-size
uniformity); strip is coloured by |area-median|/median so size outliers show RED. Cluster sweep.
"""
from __future__ import annotations
import os, sys, argparse, json, tempfile, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg; matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
import plexus.operators  # noqa
import tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d  # noqa
from tyssue_ops3d import build_sphere_mesh
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_specfmt import write_spec
from run_tyssue_vesicle import _draw, _draw_cross
from tyssue_diag import hollow_flags, hollow_metric

OUT = os.path.join(HERE, "archive")
RADIUS, JITTER, SEED = 5.0, 0.18, 0


def presets():
    #  homogenisation sweep: K_V (volume stiffness) x cv (threshold jitter) x duration bounds. 150c/500f.
    #  cols: name frames rate K_V cv vth min max
    g = []
    for K_V in (1.0, 4.0):
        for cv in (0.15, 0.40):
            for dur in ((0, 10 ** 9), (4, 12)):
                mn, mx = dur
                nm = f"vh_K{int(K_V)}_cv{int(cv * 100)}_d{mn}"
                g.append((nm, 500, 0.03, K_V, cv, 1.4, mn, mx))
    return g


def make_spec(name, frames, rate, K_V, cv, vth, min_cyc, max_cyc, buf, cbuf):
    ops = [{"op": "seed_mesh_3d", "at": "vertex", "n_cells": 150, "radius": RADIUS, "jitter": JITTER,
            "p0": 3.72, "seed": SEED, "before_frame": 1, "vseed_cv": cv},
           {"op": "cell_geometry_3d", "at": "cell"},
           # UNIFORM Okuda growth: rho=1 -> all cells grow (activator-independent), v_eq capped at vth*v_ref
           {"op": "grow_3d", "at": "vertex", "cell_set": "cell", "rate": rate,
            "a_sw": 0.5, "hill": 4.0, "rho": 1.0, "vth_frac": vth},
           {"op": "shape_energy_3d", "at": "vertex", "p0": 3.72, "K_A": 1.0, "K_P": 1.0, "Gamma": 0.1,
            "Lambda": 0.5, "K_V": K_V, "K_R": 0.4, "mu": 1.0, "dt": 1.0, "relax_iters": 26, "eta": 0.08, "cap_frac": 0.12},
           {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.35, "every": 2, "max_flips": 30},
           {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "cycle_cv": cv, "p0": 3.72,
            "every": 2, "max_div": 12, "max_div_frac": 0.03, "cell_set": "cell", "min_cycle": min_cyc, "max_cycle": max_cyc},
           {"op": "topo_snapshot_3d", "at": "vertex", "every": 1}]
    sched = ["seed_mesh_3d", "cell_geometry_3d", "grow_3d", "shape_energy_3d",
             "reconnect_t1_3d", "divide_3d", "topo_snapshot_3d"]
    cfg = {"general": {"name": f"tyssue_veshom_{name}", "seed": SEED, "n_frames": frames, "dt": 1.0,
                       "record_cap": frames + 2, "boundary": "free", "dim": 3, "world": [10 * RADIUS] * 3},
           "sets": {"vertex": {"n": buf}, "cell": {"n": cbuf, "state": {"chem": {"width": 2, "integration": "first_order"},
                                                                        "cen": {"width": 3}, "area": {"width": 1}}}},
           "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path); return sim, cfg


def run_all(only=None):
    for name, frames, rate, K_V, cv, vth, min_cyc, max_cyc in presets():
        if only and name not in only:
            continue
        odir = os.path.join(OUT, name); os.makedirs(odir, exist_ok=True)
        verts, es, et, ef, nF = build_sphere_mesh(150, RADIUS, JITTER, SEED); Nv = verts.shape[0]
        buf, cbuf = int(Nv * 12.0), int(nF * 12.0)
        print(f"[veshom] {name}: rate={rate} K_V={K_V} cv={cv} vth={vth} dur=({min_cyc},{max_cyc})  (Nv={Nv})", flush=True)
        rec = {"name": name, "rate": rate, "K_V": K_V, "cv": cv, "vth": vth, "min_cyc": min_cyc, "max_cyc": max_cyc}
        try:
            sim, cfg = make_spec(name, frames, rate, K_V, cv, vth, min_cyc, max_cyc, buf, cbuf)
            write_spec(cfg, os.path.join(odir, "spec.yaml"))
            Hf, out = engine_run(sim, device="cpu")
            emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]; T = posf.shape[0]

            def frame(t):
                mt = hist[min(t, len(hist) - 1)] if hist else dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=Nv)
                return mt, posf[t][:mt["Nv"]].astype(np.float64)

            # area_cv over the rollout (uniformity), + final split
            cvs = []
            for tt in np.linspace(0, T - 1, 24).astype(int):
                mt, pt = frame(int(tt)); _, ar, _ = hollow_metric(pt, mt); ar = ar[ar > 0]
                cvs.append(float(ar.std() / (ar.mean() + 1e-9)) if ar.size else 0)
            mtT, pT = frame(T - 1); _, arT, _ = hollow_metric(pT, mtT); arT = arT[arT > 0]
            _, _, hstat = hollow_flags(pT, mtT)
            rec.update(cells_end=int(mtT["nF"]), n_div=int(emesh.get("n_div", 0)),
                       area_cv=round(float(arT.std() / (arT.mean() + 1e-9)), 3),
                       area_cv_mean=round(float(np.mean(cvs)), 3), area_cv_max=round(float(np.max(cvs)), 3),
                       hollow_frac=round(hstat["frac"], 3))

            def areacol(mt, pt):                       # colour by |area-median|/median -> size outliers RED
                _, ar, _ = hollow_metric(pt, mt); med = np.median(ar[ar > 0]) if (ar > 0).any() else 1.0
                return np.clip(np.abs(ar[:mt["nF"]] - med) / (med + 1e-9), 0, 1)

            Rmax = max(float(np.linalg.norm(frame(t)[1], axis=1).max()) for t in np.linspace(0, T - 1, 20).astype(int))
            L3, L2 = Rmax * 1.06, Rmax * 2.23
            picks = [int(round(fr * (T - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]
            fig = plt.figure(figsize=(17.6, 9.0)); fig.patch.set_facecolor("black")
            for i, t in enumerate(picks):
                mt, pt = frame(t); col = areacol(mt, pt)
                ax3 = fig.add_subplot(2, 4, i + 1, projection="3d"); _draw(ax3, pt, mt, 3.72, azim=30, act=col, Lbox=L3)
                ax2 = fig.add_subplot(2, 4, 4 + i + 1); _draw_cross(ax2, pt, mt, 3.72, act=col, Lbox=L2)
            fig.subplots_adjust(0.006, 0.005, 0.996, 0.996, wspace=0.02, hspace=0.02)
            fig.savefig(os.path.join(odir, "strip.png"), dpi=120, facecolor="black"); plt.close(fig)
            figm = plt.figure(figsize=(5.0, 5.2)); figm.patch.set_facecolor("black")
            axm = figm.add_subplot(111, projection="3d"); figm.subplots_adjust(0, 0, 1, 1)
            keep = np.arange(0, T, max(1, T // 150))
            wri = FFMpegWriter(fps=12, metadata={"title": name})
            with wri.saving(figm, os.path.join(odir, "movie.mp4"), dpi=110):
                for j, t in enumerate(keep):
                    mt, pt = frame(int(t)); _draw(axm, pt, mt, 3.72, azim=(2 * j) % 360, act=areacol(mt, pt), Lbox=L3); wri.grab_frame()
            plt.close(figm)
            print(f"       -> cells 150->{rec['cells_end']} (+{rec['n_div']} div)  area_cv final={rec['area_cv']} "
                  f"mean={rec['area_cv_mean']} max={rec['area_cv_max']}  hollow={rec['hollow_frac']}", flush=True)
        except Exception as e:
            rec["error"] = repr(e); traceback.print_exc()
        json.dump(rec, open(os.path.join(odir, "diag.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*", default=None)
    run_all(ap.parse_args().only)
