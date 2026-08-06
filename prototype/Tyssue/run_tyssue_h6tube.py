#!/usr/bin/env python
"""H6 RD TUBES from ~2000 cells (the actual Fig-5 rationale): a live Brusselator reaction-diffusion
DYNAMICALLY partitions the cells into activated (red) / quiescent (white); activator->growth on a
VOLUME-LOCKED body (rho=0) makes the red spots PROTRUDE into tubes, while cells behind the moving front
deactivate and re-uniformise (Okuda p7: "the activator stayed around the tip, from which tubes
continuously grew"). Contrast with static cones (fig5_2k) -- here the spots self-organise and track tips.
Cluster-ready: `TV_SCRIPT=run_tyssue_h6tube.py python cluster_gen.py h6_2k_g06 ...` ; --only <preset>.
Metrics: protr (95pct/median radius), n_spots, vol_cv (uniformity behind front), hollow_frac."""
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
from tyssue_topology_ops3d import rings_from_flat_3d
import run_tyssue_fig5 as F
import torch

R, J, SEED = 5.0, 0.16, 0

# preset: n_cells, frames, chi(diameter), gamma(spot size: lower->fewer/bigger), rho(body: 0=locked),
# vth(tip growth cap), rate, a_sw(activator->growth switch; Brusselator activator peaks ~4).
PRESETS = {
    "h6_2k_g06": dict(n=2000, frames=220, chi=4.0, gamma=0.6, rho=0.0, vth=1.6, rate=0.06, a_sw=2.5),
    "h6_2k_g03": dict(n=2000, frames=220, chi=4.0, gamma=0.3, rho=0.0, vth=1.6, rate=0.06, a_sw=2.0),
    "h6_2k_g10": dict(n=2000, frames=220, chi=6.0, gamma=1.0, rho=0.0, vth=1.6, rate=0.06, a_sw=3.0),
    # ROUND spot-size (2026-07-22): local ladders showed coupling gentle=flat / strong=balloon-hollow, and
    # faster division makes hollow WORSE. Clean cones differ from balloon-RD in FEW-BIG vs MANY-TINY spots.
    # Isolate spot size at small 300c (rate=0.05 = protrusion edge, rho=0 locked body), fully archived.
    "h6s_g20": dict(n=300, frames=200, chi=4.0, gamma=2.0, rho=0.0, vth=1.4, rate=0.05, a_sw=2.5),   # many tiny
    "h6s_g10": dict(n=300, frames=200, chi=5.0, gamma=1.0, rho=0.0, vth=1.4, rate=0.05, a_sw=2.5),   # medium
    "h6s_g05": dict(n=300, frames=200, chi=6.0, gamma=0.5, rho=0.0, vth=1.4, rate=0.05, a_sw=2.5),   # few big
    # CORRECTED spot-size round: chi=4 FIXED (CFL-safe; the prior round confounded chi with gamma -- the
    # hollowing tracked chi 4/5/6 -> 54/900/explode, i.e. the RD CFL wall, NOT spot size). Only gamma varies.
    "h6c_g20": dict(n=300, frames=200, chi=4.0, gamma=2.0, rho=0.0, vth=1.4, rate=0.05, a_sw=2.5),   # many tiny
    "h6c_g10": dict(n=300, frames=200, chi=4.0, gamma=1.0, rho=0.0, vth=1.4, rate=0.05, a_sw=2.5),   # medium
    "h6c_g05": dict(n=300, frames=200, chi=4.0, gamma=0.5, rho=0.0, vth=1.4, rate=0.05, a_sw=2.5),   # few big
    # BALLOON-HOLLOW FIX round: base = h6c_g05 (protr 2.83 but hollow 255); add ONE stabiliser each. Target:
    # hollow_n_peak DROPS while protr HOLDS. K_lumen (isoperimetric -> patch buds OUT not wrinkles IN) is the
    # untested favourite; K_bend (dihedral, resist cap tilt); antiinv (block inverting faces per substep).
    "h6x_base":    dict(n=200, frames=200, chi=4.0, gamma=0.4, rho=0.0, vth=1.4, rate=0.05, a_sw=2.5),
    "h6x_lumen":   dict(n=200, frames=200, chi=4.0, gamma=0.4, rho=0.0, vth=1.4, rate=0.05, a_sw=2.5, K_lumen=1.0),
    "h6x_bend":    dict(n=200, frames=200, chi=4.0, gamma=0.4, rho=0.0, vth=1.4, rate=0.05, a_sw=2.5, K_bend=80.0),
    "h6x_antiinv": dict(n=200, frames=200, chi=4.0, gamma=0.4, rho=0.0, vth=1.4, rate=0.05, a_sw=2.5, antiinv=0.4),
    # 2000c LARGE-SPOT EVOLUTION (Fig5 target): few big RD spots, long frames, let tubes extrude. Fire the
    # variant whose stabiliser won the 200c fix round (avoid another g03-style blow-up by validating cheap first).
    "h6t_2k_lumen": dict(n=2000, frames=400, chi=4.0, gamma=0.4, rho=0.0, vth=1.4, rate=0.05, a_sw=2.5, K_lumen=1.0),
    "h6t_2k_bend":  dict(n=2000, frames=400, chi=4.0, gamma=0.4, rho=0.0, vth=1.4, rate=0.05, a_sw=2.5, K_bend=80.0),
    # EVOLUTION (user vision): start 200c, LARGE spots, run LONG -> let it grow (~200->2000c) and extrude
    # tubes via RD + morphogenesis. 200c large spots is clean (hollow~1) but lumps into ONE fat lobe; test
    # whether longer evolution ELONGATES it into distinct tubes, at two spot sizes (0.4 fat vs 0.8 distinct).
    "h6ev_g04": dict(n=200, frames=700, chi=4.0, gamma=0.4, rho=0.0, vth=1.4, rate=0.05, a_sw=2.5),
    "h6ev_g08": dict(n=200, frames=700, chi=4.0, gamma=0.8, rho=0.0, vth=1.4, rate=0.05, a_sw=2.5),
    # SMOKE_HOM-INITIALISED RD tubing (user): take smoke_hom's HOMOGENISED config (rho=0.15 partial body
    # growth -> uniform cells AND not the locked-body balloon; vth=1.35 tight cap; cv=0.4; K_V=3 Lambda=0.2
    # Gamma=0.05; 900c) and drive tubing with RD large spots instead of static cones. rho=0.15 is the middle
    # ground between locked-body balloon (rho=0) and flat homogenised sphere. Vary body growth rho.
    "h6sh_r15": dict(n=900, frames=300, chi=4.0, gamma=0.5, rho=0.15, vth=1.35, rate=0.03, a_sw=2.5, K_V=3.0, Lambda=0.2, Gamma=0.05, cyc_cv=0.4, max_cyc=14),
    "h6sh_r08": dict(n=900, frames=300, chi=4.0, gamma=0.5, rho=0.08, vth=1.35, rate=0.03, a_sw=2.5, K_V=3.0, Lambda=0.2, Gamma=0.05, cyc_cv=0.4, max_cyc=14),
    "h6sh_r30": dict(n=900, frames=300, chi=4.0, gamma=0.5, rho=0.30, vth=1.35, rate=0.03, a_sw=2.5, K_V=3.0, Lambda=0.2, Gamma=0.05, cyc_cv=0.4, max_cyc=14),
    # rho=0 at smoke_hom scale/soft-shape (missing point): locked body should PROTRUDE. Fewer spots via lower
    # gamma (0.3) to fight the fine-speckle-at-900c problem. Does the locked body tube, or balloon at scale?
    "h6sh_r00":    dict(n=900, frames=300, chi=4.0, gamma=0.5, rho=0.0, vth=1.4, rate=0.04, a_sw=2.5, K_V=3.0, Lambda=0.2, Gamma=0.05, cyc_cv=0.4, max_cyc=1000000000),
    "h6sh_r00_g3": dict(n=900, frames=300, chi=4.0, gamma=0.3, rho=0.0, vth=1.4, rate=0.04, a_sw=2.5, K_V=3.0, Lambda=0.2, Gamma=0.05, cyc_cv=0.4, max_cyc=1000000000),
    # H-narrow: cones don't balloon because they activate a NARROW cone (few cells -> narrow tube); an RD
    # Turing spot activates a WIDE patch (many cells -> wide balloon). Raise a_sw so only each spot's PEAK
    # drives growth -> narrow coherent tubes. base a_sw=2.5 (h6sh_r00_g3, hollow787); test 3.0 / 3.5.
    "h6nw_a30": dict(n=900, frames=300, chi=4.0, gamma=0.3, rho=0.0, vth=1.4, rate=0.04, a_sw=3.0, K_V=3.0, Lambda=0.2, Gamma=0.05, cyc_cv=0.4, max_cyc=1000000000),
    "h6nw_a35": dict(n=900, frames=300, chi=4.0, gamma=0.3, rho=0.0, vth=1.4, rate=0.04, a_sw=3.5, K_V=3.0, Lambda=0.2, Gamma=0.05, cyc_cv=0.4, max_cyc=1000000000),
    # H-narrow VALIDATED (hollow 787->50 as a_sw 2.5->3.5, tri-lobe forming). Push narrower (3.8) + LONGER
    # (let fat lobes elongate) + test adding body growth rho=0.10 for uniformity now the balloon is controlled.
    "h6nw_a38":  dict(n=900, frames=450, chi=4.0, gamma=0.3, rho=0.0,  vth=1.4, rate=0.04, a_sw=3.8, K_V=3.0, Lambda=0.2, Gamma=0.05, cyc_cv=0.4, max_cyc=1000000000),
    "h6nw_r10":  dict(n=900, frames=450, chi=4.0, gamma=0.3, rho=0.10, vth=1.4, rate=0.04, a_sw=3.5, K_V=3.0, Lambda=0.2, Gamma=0.05, cyc_cv=0.4, max_cyc=1000000000),
    # ===== round_XX naming (user) =====  WINNER regime: rho=0 locked, a_sw=3.5 narrow, gamma=0.3, KV=3 soft.
    # round_01: elongate the fat tri-lobe into TUBES via longer frames + faster tip growth.
    "round_01_long": dict(n=900, frames=700, chi=4.0, gamma=0.3, rho=0.0, vth=1.4, rate=0.04, a_sw=3.5, K_V=3.0, Lambda=0.2, Gamma=0.05, cyc_cv=0.4, max_cyc=1000000000),
    "round_01_fast": dict(n=900, frames=500, chi=4.0, gamma=0.3, rho=0.0, vth=1.4, rate=0.06, a_sw=3.5, K_V=3.0, Lambda=0.2, Gamma=0.05, cyc_cv=0.4, max_cyc=1000000000),
}


def make(p):
    n, frames = p["n"], p["frames"]
    verts, es, et, ef, nF = build_sphere_mesh(n, R, J, SEED); Nv = verts.shape[0]
    ops = [{"op": "seed_mesh_3d", "at": "vertex", "n_cells": n, "radius": R, "jitter": J, "p0": 3.90, "seed": SEED, "before_frame": 1, "vseed_cv": 0.15},
           {"op": "cell_geometry_3d", "at": "cell"}, {"op": "cell_adjacency", "at": "cell"},
           {"op": "seed_cell_rd", "at": "cell", "seed": SEED, "before_frame": 3, "mode": "noise", "A": 1.0, "B": 3.0, "noise": 0.04},
           {"op": "cell_diffuse", "at": "cell", "d_a": 0.05, "d_h": 0.7, "chi": p["chi"]},
           {"op": "cell_react", "at": "cell", "implementation": "brusselator", "gamma": p["gamma"], "A": 1.0, "B": 3.0},
           # activator->growth on a LOCKED body (rho=0): only red cells grow (v_eq up to vth*v_ref) -> protrude
           {"op": "morphogen_growth_3d", "at": "vertex", "cell_set": "cell", "rate": p["rate"], "a_sw": p["a_sw"], "hill": 4.0, "rho": p["rho"], "vth_frac": p["vth"]},
           # p0>3.81 FLUID (tubes need T1 flow) + modest surface tension for rounder cells; radial ~off so tubes leave the sphere
           {"op": "shape_energy_3d", "at": "vertex", "p0": 3.90, "K_A": 1.0, "K_P": 1.0, "Gamma": p.get("Gamma", 0.2), "Lambda": p.get("Lambda", 0.6), "K_V": p.get("K_V", 4.0), "K_R": 0.02, "mu": 1.0, "dt": 0.02, "relax_iters": 26, "eta": 0.08, "cap_frac": 0.12,
            "K_bend": p.get("K_bend", 0.0), "K_lumen": p.get("K_lumen", 0.0), "antiinv": p.get("antiinv", 0.0)},   # balloon-hollow stabilisers (one per variant)
           {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.28, "every": 1, "max_flips": max(60, n // 8)},
           {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "cycle_cv": p.get("cyc_cv", 0.15), "p0": 3.90, "every": 2, "max_div": 60, "max_div_frac": 0.02, "cell_set": "cell", "min_cycle": 4, "max_cycle": p.get("max_cyc", 30)},
           {"op": "topo_snapshot_3d", "at": "vertex", "every": 1}]
    sched = ["seed_mesh_3d", "cell_geometry_3d", "cell_adjacency", "seed_cell_rd", "cell_diffuse", "cell_react",
             "morphogen_growth_3d", "shape_energy_3d", "reconnect_t1_3d", "divide_3d", "topo_snapshot_3d"]
    cfg = {"general": {"name": "tyssue_h6", "seed": SEED, "n_frames": frames, "dt": 0.02, "record_cap": frames + 2, "boundary": "free", "dim": 3, "world": [16 * R] * 3},
           "sets": {"vertex": {"n": int(Nv * 8)}, "cell": {"n": int(nF * 8), "state": {"chem": {"width": 2, "integration": "first_order"}, "cen": {"width": 3}, "area": {"width": 1}}}},  # big buffer: proliferation must not hit the cell-slot ceiling
           "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path); return sim, cfg, dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=Nv)


def vol_cv(mt, pt):
    _, _, _, vf = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(np.asarray(mt["E_srce"])), torch.as_tensor(np.asarray(mt["E_trgt"])), torch.as_tensor(np.asarray(mt["E_face"])), mt["nF"])
    vf = vf.numpy(); vf = vf[np.abs(vf) > 1e-9]; return float(vf.std() / (np.abs(vf.mean()) + 1e-9))


def do(preset):
    p = PRESETS[preset]; OUT = os.path.join(HERE, "archive", preset); os.makedirs(OUT, exist_ok=True)
    sim, cfg, mesh0 = make(p); write_spec(cfg, os.path.join(OUT, "spec.yaml"))
    rec = {"name": preset, **p}
    try:
        Hf, out = engine_run(sim, device="cpu")
        emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]; chemf = out["sets"]["cell"]["state"]["chem"]; T = posf.shape[0]
        def frame(t):
            mt = hist[min(t, len(hist) - 1)] if hist else mesh0
            return mt, posf[t][:mt["Nv"]].astype(np.float64), chemf[t][:mt["nF"], 0]
        mtT, pT, aT = frame(T - 1); ext = pT.max(0) - pT.min(0)
        rings = rings_from_flat_3d(np.asarray(mtT["E_srce"]), np.asarray(mtT["E_trgt"]), np.asarray(mtT["E_face"]), mtT["nF"])
        rad = np.array([np.linalg.norm(pT[r].mean(0)) if (r is not None and len(r)) else 0 for r in rings]); rad = rad[rad > 0]
        rec.update(cells_end=int(mtT["nF"]), protr=round(float(np.percentile(rad, 95) / (np.median(rad) + 1e-9)), 3),
                   vol_cv=round(vol_cv(mtT, pT), 3), hollow_frac=round(float(hollow_flags(pT, mtT)[2]["frac"]), 3),
                   spots=int(F.count_spots(aT, mtT, float(np.percentile(aT, 70)))), aspect=round(float(ext.max() / (np.median(ext) + 1e-9)), 3),
                   activator=[round(float(aT.min()), 2), round(float(aT.max()), 2)])
        lo, hi = float(np.percentile(aT, 5)), float(np.percentile(aT, 99) + 1e-6)
        Rmax = max(float(np.abs(frame(t)[1]).max()) for t in np.linspace(0, T - 1, 20).astype(int)); L3, L2 = Rmax * 1.06, Rmax * 2.23
        col = lambda mt, pt, a: np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
        fig = plt.figure(figsize=(17.6, 9.0)); fig.patch.set_facecolor("black")
        for i, t in enumerate([int(round(fr * (T - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]):
            mt, pt, a = frame(t)
            ax3 = fig.add_subplot(2, 4, i + 1, projection="3d"); _draw(ax3, pt, mt, 3.90, azim=30, act=col(mt, pt, a), Lbox=L3)
            ax2 = fig.add_subplot(2, 4, 4 + i + 1); _draw_cross(ax2, pt, mt, 3.90, act=col(mt, pt, a), Lbox=L2, axis=1)
        fig.subplots_adjust(0.006, 0.005, 0.996, 0.996, wspace=0.02, hspace=0.02); fig.savefig(os.path.join(OUT, "strip.png"), dpi=120, facecolor="black"); plt.close(fig)
        figm = plt.figure(figsize=(5.0, 5.2)); figm.patch.set_facecolor("black"); axm, axin = make_movie_axes(figm)
        keep = np.arange(0, T, max(1, T // 150)); wri = FFMpegWriter(fps=12, metadata={"title": preset})
        with wri.saving(figm, os.path.join(OUT, "movie.mp4"), dpi=110):
            for j, t in enumerate(keep):
                mt, pt, a = frame(int(t)); draw_movie_frame(axm, axin, pt, mt, 3.90, (2 * j) % 360, col(mt, pt, a), L3, L2); wri.grab_frame()
        plt.close(figm)
        from tissue_analysis import analyze                    # per-frame hollow count / size CV / tube diameter
        mf = []
        for t in np.unique(np.linspace(0, T - 1, 40).astype(int)):
            mt, pt, a = frame(int(t)); mf.append((int(t), pt, mt, a))   # a -> red_frac (activator localisation)
        rec.update(analyze(mf, OUT))
        print(f"[{preset}] cells {p['n']}->{rec['cells_end']}  protr={rec['protr']} spots={rec['spots']} "
              f"tube_diam={rec['tube_diam_final']} n_tubes={rec['n_tubes_final']} hollow_peak={rec['hollow_n_peak']} "
              f"area_cv_peak={rec['area_cv_peak']} vol_cv={rec['vol_cv_final']}", flush=True)
    except Exception as e:
        rec["error"] = repr(e); traceback.print_exc()
    json.dump(rec, open(os.path.join(OUT, "diag.json"), "w"), indent=1)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--only":
        do(args[1])
    else:
        for k in (args or list(PRESETS)):
            do(k)
