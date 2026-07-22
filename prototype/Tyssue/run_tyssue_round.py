#!/usr/bin/env python
"""round_XX tubing runs that GENUINELY initialise from smoke_hom's homogenised vesicle (load_mesh_3d reads
archive/smoke_hom/ckpt.npz) and seed a FEW BIG RD spots on it, then extrude tubes (winning regime: rho=0
locked body + a_sw narrow so only each spot's peak grows -> narrow coherent tubes, not wide balloons).
Archives to archive/<preset>/ (name presets round_XX_<desc>). --only <preset>.  Cluster:
    TV_SCRIPT=run_tyssue_round.py python cluster_gen.py round_01_big ..."""
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
import tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d, ckpt  # noqa
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_ops3d import face_geometry_3d
from tyssue_specfmt import write_spec
from run_tyssue_vesicle import _draw, _draw_cross, make_movie_axes, draw_movie_frame
from tyssue_diag import hollow_flags
from tube_analysis import analyze
import torch

CKPT = os.path.join(HERE, "archive", "smoke_hom", "ckpt.npz")
VBUF, CBUF = 26000, 13000                                   # buffer >= ckpt (~2400 cells) + growth headroom

# few BIG RD spots on the homogenised mesh (gamma low = long wavelength = fewer bigger); rho=0 locked +
# a_sw narrow (only peak grows -> narrow tube). frames long to elongate.
PRESETS = {
    "round_01_big": dict(frames=400, chi=4.0, gamma=0.3, rho=0.0, vth=1.4, rate=0.05, a_sw=3.5),
    "round_01_bigger": dict(frames=400, chi=4.0, gamma=0.2, rho=0.0, vth=1.4, rate=0.05, a_sw=3.5),
    # round_02: from ckpt, match the WINNING controlled rate (h6nw_a35: rate 0.04, hollow 50) -- rate 0.05
    # ran away (hollow 3348). Slower rate keeps the activator from runaway-ballooning. gamma 0.3 a_sw 3.5.
    "round_02_r04": dict(frames=400, chi=4.0, gamma=0.3, rho=0.0, vth=1.4, rate=0.04, a_sw=3.5),
    "round_02_r03": dict(frames=450, chi=4.0, gamma=0.3, rho=0.0, vth=1.4, rate=0.03, a_sw=3.5),
    # round_03: CONES on the homogenised ckpt -> N BIG red spots visible at FRAME 0 (like the target image),
    # re-seeded at the tips = tip-tracking = the clean-tube mechanism. dt=1.0 (no RD/CFL). a_sw=0.5 (binary cone).
    "round_03_cone3": dict(frames=350, spots=3, cone_deg=18.0, rho=0.0, vth=1.5, rate=0.04, a_sw=0.5),
    "round_03_cone5": dict(frames=350, spots=5, cone_deg=16.0, rho=0.0, vth=1.5, rate=0.04, a_sw=0.5),
    # round_04: init nails it (white vesicle + 3 big red spots) but tubes BALLOONED (K_V=3 soft -> cells
    # overshoot instead of divide). Fix like the clean fig5 cones: stiff K_V=4 + tight vth (cells divide-and-
    # extend the tube, proliferation not inflation) + more division throughput + more relaxation.
    "round_04_cone3": dict(frames=200, spots=3, cone_deg=18.0, rho=0.0, vth=1.4,  rate=0.04, a_sw=0.5, K_V=4.0, mdf=0.03, relax=30),
    "round_04_slow":  dict(frames=250, spots=3, cone_deg=18.0, rho=0.0, vth=1.35, rate=0.03, a_sw=0.5, K_V=4.0, mdf=0.03, relax=30),
    # round_05: FAST (~15min target) + TIP-SIZE CONTROL. Narrow cones (deg 12 -> fewer activated cells ->
    # far less proliferation -> fast); stiff K_V=5 (no overshoot) + tight vth=1.3 (tip cells stay ~body size,
    # Okuda [2/3,4/3]v_ref). 150 frames. Two tip caps to compare.
    "round_05_v13": dict(frames=150, spots=3, cone_deg=12.0, rho=0.0, vth=1.30, rate=0.04, a_sw=0.5, K_V=5.0, mdf=0.03, relax=28),
    "round_05_v12": dict(frames=150, spots=3, cone_deg=12.0, rho=0.0, vth=1.20, rate=0.04, a_sw=0.5, K_V=5.0, mdf=0.03, relax=28),
    # round_06: HARD SIZE CAP (vcap) -- force-divide any cell >= vcap x v_ref, bypassing the throttle, so NO
    # cell (tip included) exceeds the cap. Directly bounds cell size. cone 12 (fast), K_V=4, frames 150.
    "round_06_cap15": dict(frames=150, spots=3, cone_deg=12.0, rho=0.0, vth=1.5, rate=0.04, a_sw=0.5, K_V=4.0, mdf=0.03, relax=30, vcap=1.5),
    "round_06_cap13": dict(frames=150, spots=3, cone_deg=12.0, rho=0.0, vth=1.5, rate=0.04, a_sw=0.5, K_V=4.0, mdf=0.03, relax=30, vcap=1.3),
}


def make(p):
    cones = "spots" in p; dt = 1.0 if cones else 0.02
    ops = [{"op": "load_mesh_3d", "at": "vertex", "cell_set": "cell", "ckpt": CKPT, "before_frame": 1},
           {"op": "cell_geometry_3d", "at": "cell"}]
    sched = ["load_mesh_3d", "cell_geometry_3d"]
    if cones:                                                   # N BIG spots at frame 0, re-seeded at tips (tip-tracking)
        ops += [{"op": "cell_rd_seed", "at": "cell", "mode": "cones", "n_spots": p["spots"], "cone_deg": p["cone_deg"]}]
        sched += ["cell_rd_seed"]
    else:                                                       # Brusselator RD (develops from noise -> spots emerge late)
        ops += [{"op": "cell_adjacency", "at": "cell"},
                {"op": "cell_rd_seed", "at": "cell", "seed": 0, "before_frame": 3, "mode": "noise", "A": 1.0, "B": 3.0, "noise": 0.04},
                {"op": "cell_diffuse", "at": "cell", "d_a": 0.05, "d_h": 0.7, "chi": p["chi"]},
                {"op": "cell_react", "at": "cell", "implementation": "brusselator", "gamma": p["gamma"], "A": 1.0, "B": 3.0}]
        sched += ["cell_adjacency", "cell_rd_seed", "cell_diffuse", "cell_react"]
    ops += [{"op": "morphogen_growth_3d", "at": "vertex", "cell_set": "cell", "rate": p["rate"], "a_sw": p["a_sw"], "hill": 4.0, "rho": p["rho"], "vth_frac": p["vth"]},
            {"op": "shape_energy_3d", "at": "vertex", "p0": 3.90, "K_A": 1.0, "K_P": 1.0, "Gamma": 0.05, "Lambda": 0.2, "K_V": p.get("K_V", 4.0), "K_R": 0.02, "mu": 1.0, "dt": dt, "relax_iters": p.get("relax", 30), "eta": 0.08, "cap_frac": 0.12},
            {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.28, "every": 1, "max_flips": 300},
            {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "cycle_cv": 0.4, "p0": 3.90, "every": 2, "max_div": 120, "max_div_frac": p.get("mdf", 0.03), "vcap": p.get("vcap", 0.0), "cell_set": "cell", "min_cycle": 4, "max_cycle": 1000000000},
            {"op": "topo_snapshot_3d", "at": "vertex", "every": 1}]
    sched += ["morphogen_growth_3d", "shape_energy_3d", "reconnect_t1_3d", "divide_3d", "topo_snapshot_3d"]
    cfg = {"general": {"name": "tyssue_round", "seed": 0, "n_frames": p["frames"], "dt": dt, "record_cap": p["frames"] + 2, "boundary": "free", "dim": 3, "world": [16 * 5.0] * 3},
           "sets": {"vertex": {"n": VBUF}, "cell": {"n": CBUF, "state": {"chem": {"width": 2, "integration": "first_order"}, "cen": {"width": 3}, "area": {"width": 1}}}},
           "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path); return sim, cfg


def do(preset):
    p = PRESETS[preset]; OUT = os.path.join(HERE, "archive", preset); os.makedirs(OUT, exist_ok=True)
    sim, cfg = make(p); write_spec(cfg, os.path.join(OUT, "spec.yaml"))
    rec = {"name": preset, **p}
    try:
        Hf, out = engine_run(sim, device="cpu")
        emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]; chemf = out["sets"]["cell"]["state"]["chem"]; T = posf.shape[0]
        def frame(t):
            mt = hist[min(t, len(hist) - 1)]
            return mt, posf[t][:mt["Nv"]].astype(np.float64), chemf[t][:mt["nF"], 0]
        mtT, pT, aT = frame(T - 1)
        lo, hi = float(np.percentile(aT, 5)), float(np.percentile(aT, 99) + 1e-6)
        col = lambda a: np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
        lbox = lambda pt: (float(np.abs(pt).max()) * 1.12, float(np.abs(pt).max()) * 2.3)   # PER-FRAME autoscale so the init (and every stage) is always visible, not a dot next to a balloon
        fig = plt.figure(figsize=(17.6, 9.0)); fig.patch.set_facecolor("black")
        for i, t in enumerate([int(round(fr * (T - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]):
            mt, pt, a = frame(t); l3, l2 = lbox(pt)
            ax3 = fig.add_subplot(2, 4, i + 1, projection="3d"); _draw(ax3, pt, mt, 3.90, azim=30, act=col(a), Lbox=l3)
            ax2 = fig.add_subplot(2, 4, 4 + i + 1); _draw_cross(ax2, pt, mt, 3.90, act=col(a), Lbox=l2, axis=1)
        fig.subplots_adjust(0.006, 0.005, 0.996, 0.996, wspace=0.02, hspace=0.02); fig.savefig(os.path.join(OUT, "strip.png"), dpi=120, facecolor="black"); plt.close(fig)
        figm = plt.figure(figsize=(5.0, 5.2)); figm.patch.set_facecolor("black"); axm, axin = make_movie_axes(figm)
        keep = np.arange(0, T, max(1, T // 110)); wri = FFMpegWriter(fps=11, metadata={"title": preset})
        with wri.saving(figm, os.path.join(OUT, "movie.mp4"), dpi=95):
            for j, t in enumerate(keep):
                mt, pt, a = frame(int(t)); l3, l2 = lbox(pt); draw_movie_frame(axm, axin, pt, mt, 3.90, (2 * j) % 360, col(a), l3, l2); wri.grab_frame()
        plt.close(figm)
        mf = []
        for t in np.unique(np.linspace(0, T - 1, 40).astype(int)):
            mt, pt, a = frame(int(t)); mf.append((int(t), pt, mt, a))
        rec.update(cells_end=int(mtT["nF"])); rec.update(analyze(mf, OUT))
        print(f"[{preset}] cells->{rec['cells_end']} protr={rec['protr_final']} tube_diam={rec['tube_diam_final']} "
              f"n_tubes={rec['n_tubes_final']} hollow_pk={rec['hollow_n_peak']} area_cv_pk={rec['area_cv_peak']} red_frac={rec['red_frac_final']}", flush=True)
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
