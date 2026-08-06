#!/usr/bin/env python
"""Re-archive with ROUNDED cells (perfecting: more surface tension Lambda=3/Gamma=0.4, softer K_V=2, p0=3.5
-> shape index 3.94->3.81, slivers halved) + cross-section INSET movie. Two clean mp4s:
  vh_K4_cv15_d4        -- plain homogenised vesicle (no RD), coloured by cell-size deviation
  vh_K4_cv15_d4_rd_coral -- Gray-Scott coral, coloured by activator
    python archive_rounded.py [plain|coral]     (default: both)"""
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
from tyssue_ops3d import build_sphere_mesh, face_geometry_3d
from tyssue_specfmt import write_spec
from run_tyssue_vesicle import _draw, _draw_cross, make_movie_axes, draw_movie_frame
from tyssue_diag import hollow_metric, hollow_flags
import torch

R, J, SEED = 5.0, 0.18, 0
P0, LAM, GAM, K_V = 3.5, 3.0, 0.4, 2.0     # ROUNDED cells (H10 winner)


def make(coral, frames, dt):
    verts, es, et, ef, nF = build_sphere_mesh(150, R, J, SEED); Nv = verts.shape[0]
    ops = [{"op": "seed_mesh_3d", "at": "vertex", "n_cells": 150, "radius": R, "jitter": J, "p0": P0, "seed": SEED, "before_frame": 1, "vseed_cv": 0.15},
           {"op": "cell_geometry_3d", "at": "cell"}]
    sched = ["seed_mesh_3d", "cell_geometry_3d"]
    if coral:
        ops += [{"op": "cell_adjacency", "at": "cell"},
                {"op": "seed_cell_rd", "at": "cell", "seed": SEED, "before_frame": 3, "mode": "scatter", "seed_frac": 0.06},
                {"op": "cell_diffuse", "at": "cell", "d_a": 0.08, "d_h": 0.16, "chi": 1.3},
                {"op": "cell_react", "at": "cell", "implementation": "gray_scott", "F": 0.055, "kk": 0.062, "rate": 1.0}]
        sched += ["cell_adjacency", "seed_cell_rd", "cell_diffuse", "cell_react"]
    ops += [{"op": "morphogen_growth_3d", "at": "vertex", "cell_set": "cell", "rate": 0.03, "a_sw": 50.0, "hill": 4.0, "rho": 1.0, "vth_frac": 1.4},
            {"op": "shape_energy_3d", "at": "vertex", "p0": P0, "K_A": 1.0, "K_P": 1.0, "Gamma": GAM, "Lambda": LAM, "K_V": K_V, "K_R": 0.4, "mu": 1.0, "dt": dt, "relax_iters": 30, "eta": 0.08, "cap_frac": 0.12},
            # D1 CLOCK MIGRATION. These two read `every: 2` when the operator ALSO gated itself,
            # so their true period was 2 (engine) x 2 (operator) = 4. The engine now owns the
            # clock alone, so the period that reproduces the archived run is 4, not 2. Writing 2
            # here would silently double the division rate and produce a different movie.
            # `engine_clock: true` asserts these numbers are written for the engine-owned clock.
            {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.35, "every": 4, "engine_clock": True, "max_flips": 30},
            {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "cycle_cv": 0.15, "p0": P0, "every": 4, "engine_clock": True, "max_div": 12, "max_div_frac": 0.03, "cell_set": "cell", "min_cycle": 4, "max_cycle": 12},
            {"op": "topo_snapshot_3d", "at": "vertex", "every": 1}]
    sched += ["morphogen_growth_3d", "shape_energy_3d", "reconnect_t1_3d", "divide_3d", "topo_snapshot_3d"]
    nm = "vh_K4_cv15_d4_rd_coral" if coral else "vh_K4_cv15_d4"
    cfg = {"general": {"name": f"tyssue_{nm}", "seed": SEED, "n_frames": frames, "dt": dt, "record_cap": frames + 2, "boundary": "free", "dim": 3, "world": [10 * R] * 3},
           "sets": {"vertex": {"n": int(Nv * 12)}, "cell": {"n": int(nF * 12), "state": {"chem": {"width": 2, "integration": "first_order"}, "cen": {"width": 3}, "area": {"width": 1}}}},
           "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    import plexus.schema as S
    sim = S.load(path); os.unlink(path); return nm, sim, cfg, dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=Nv)


def vol_cv(mt, pt):
    _, _, _, vf = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(np.asarray(mt["E_srce"])), torch.as_tensor(np.asarray(mt["E_trgt"])), torch.as_tensor(np.asarray(mt["E_face"])), mt["nF"])
    vf = vf.numpy(); vf = vf[np.abs(vf) > 1e-9]; return float(vf.std() / (np.abs(vf.mean()) + 1e-9))


def do(coral, repro=False, frames=500):
    """Generate the run. `repro=True` writes *.repro.* and leaves the archived record untouched."""
    dt = 1.0
    sfx = ".repro" if repro else ""
    nm, sim, cfg, mesh0 = make(coral, frames, dt); OUT = os.path.join(HERE, "archive", nm); os.makedirs(OUT, exist_ok=True)
    write_spec(cfg, os.path.join(OUT, f"spec{sfx}.yaml"))
    from plexus.engine import run as engine_run
    rec = {"name": nm}
    try:
        Hf, out = engine_run(sim, device="cpu")
        emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]; chemf = out["sets"]["cell"]["state"]["chem"]; T = posf.shape[0]
        def frame(t):
            mt = hist[min(t, len(hist) - 1)] if hist else mesh0
            return mt, posf[t][:mt["Nv"]].astype(np.float64), chemf[t][:mt["nF"], 0]
        mtT, pT, aT = frame(T - 1); _, arT, _ = hollow_metric(pT, mtT); arT = arT[arT > 0]
        rec.update(cells_end=int(mtT["nF"]), vol_cv=round(vol_cv(mtT, pT), 3), hollow_frac=round(float(hollow_flags(pT, mtT)[2]["frac"]), 3), rounded=True)
        Rmax = max(float(np.linalg.norm(frame(t)[1], axis=1).max()) for t in np.linspace(0, T - 1, 20).astype(int)); L3, L2 = Rmax * 1.06, Rmax * 2.23
        if coral:
            lo, hi = float(np.percentile(aT, 5)), float(np.percentile(aT, 99) + 1e-6)
            col = lambda mt, pt, a: np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
        else:
            def col(mt, pt, a):
                _, ar, _ = hollow_metric(pt, mt); med = np.median(ar[ar > 0]) if (ar > 0).any() else 1.0
                return np.clip(np.abs(ar[:mt["nF"]] - med) / (med + 1e-9), 0, 1)
        fig = plt.figure(figsize=(17.6, 9.0)); fig.patch.set_facecolor("black")
        for i, t in enumerate([int(round(fr * (T - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]):
            mt, pt, a = frame(t); c = col(mt, pt, a)
            ax3 = fig.add_subplot(2, 4, i + 1, projection="3d"); _draw(ax3, pt, mt, P0, azim=30, act=c, Lbox=L3)
            ax2 = fig.add_subplot(2, 4, 4 + i + 1); _draw_cross(ax2, pt, mt, P0, act=c, Lbox=L2)
        fig.subplots_adjust(0.006, 0.005, 0.996, 0.996, wspace=0.02, hspace=0.02); fig.savefig(os.path.join(OUT, f"strip{sfx}.png"), dpi=120, facecolor="black"); plt.close(fig)
        figm = plt.figure(figsize=(5.0, 5.2)); figm.patch.set_facecolor("black"); axm, axin = make_movie_axes(figm)
        keep = np.arange(0, T, max(1, T // 150)); wri = FFMpegWriter(fps=12, metadata={"title": nm})
        with wri.saving(figm, os.path.join(OUT, f"movie{sfx}.mp4"), dpi=110):
            for j, t in enumerate(keep):
                mt, pt, a = frame(int(t)); draw_movie_frame(axm, axin, pt, mt, P0, (2 * j) % 360, col(mt, pt, a), L3, L2); wri.grab_frame()
        plt.close(figm)
        print(f"[{nm}] cells 150->{rec['cells_end']}  vol_cv={rec['vol_cv']} hollow={rec['hollow_frac']} (ROUNDED Lam={LAM} Gam={GAM} K_V={K_V} p0={P0})", flush=True)
    except Exception as e:
        # A FAILED RUN MUST NOT OVERWRITE THE ARCHIVED RESULT.
        # `json.dump` used to sit outside this handler, so any exception replaced
        # archive/<nm>/diag.json -- the reference metrics the movie was validated against -- with
        # {"name":..., "error":...}. Merely IMPORTING this module was enough to do it (see the
        # __main__ guard below): one accidental import wiped the ground truth for both front-page
        # videos. archive/ is documented as immutable research record; now it behaves that way.
        rec["error"] = repr(e)
        traceback.print_exc()
        json.dump(rec, open(os.path.join(OUT, f"diag{sfx}.error.json"), "w"), indent=1)
        print(f"[{nm}] FAILED -- wrote diag.error.json; archived diag.json left untouched",
              flush=True)
        raise                      # never swallow an exception around an artefact
    json.dump(rec, open(os.path.join(OUT, f"diag{sfx}.json"), "w"), indent=1)
    return rec


def compare_to_archive(nm, rec):
    """Did we reproduce the archived numbers? Prints a verdict; returns (ok, deltas)."""
    ref_p = os.path.join(HERE, "archive", nm, "diag.json")
    if not os.path.exists(ref_p):
        print(f"[{nm}] no archived diag.json to compare against"); return None, {}
    ref = json.load(open(ref_p))
    # (relative tolerance, ABSOLUTE floor). The floor is not decoration: hollow_frac's reference
    # is 0.004, so a run that improves it to 0.000 is a 100% relative change and was reported as a
    # reproduction MISMATCH -- a failure of the comparison, not of the simulation. A relative
    # tolerance on a near-zero reference measures nothing. Same class of error as the metrics the
    # instrument gate threw out; it belongs in the checker too.
    TOL = {"cells_end": (0.02, 0.5), "vol_cv": (0.10, 0.01), "hollow_frac": (0.05, 0.01)}
    ok, deltas = True, {}
    for k, (rtol, atol) in TOL.items():
        if k not in ref or k not in rec:
            continue
        a, b = float(ref[k]), float(rec[k])
        ad = abs(b - a)
        rd = ad / max(abs(a), 1e-9)
        hit = (ad > atol) and (rd > rtol)          # must fail BOTH to count as a mismatch
        deltas[k] = {"archived": a, "now": b, "abs_delta": round(ad, 6),
                     "rel_delta": round(rd, 4), "ok": not hit}
        if hit:
            ok = False
    print(f"[{nm}] REPRODUCTION {'MATCH' if ok else 'MISMATCH'}")
    for k, v in deltas.items():
        print(f"    {'ok ' if v['ok'] else 'BAD'} {k:12} archived {v['archived']:<10} "
              f"now {v['now']:<10} abs {v['abs_delta']:<9} rel {v['rel_delta']:+.1%}")
    return ok, deltas


# Guarded. Without this, `import archive_rounded` ran two 500-frame simulations as a side effect
# of the import -- which is how the archived diag.json files came to be overwritten.
#
#   python archive_rounded.py [plain|coral|both]            regenerate + overwrite the archive
#   python archive_rounded.py [plain|coral|both] --repro    reproduce and COMPARE, archive intact
#
# --repro is the one to use before a launch: it answers "can we still make this movie?" without
# betting the reference numbers on the answer.
if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    repro = "--repro" in sys.argv
    which = argv[0] if argv else "both"
    corals = [False] if which == "plain" else [True] if which == "coral" else [False, True]
    verdicts = {}
    for coral in corals:
        rec = do(coral, repro=repro)
        verdicts[rec["name"]] = compare_to_archive(rec["name"], rec)[0]
    if repro:
        print("\n=== reproduction summary ===")
        for nm, ok in verdicts.items():
            print(f"  {nm:26} {'MATCH' if ok else 'MISMATCH' if ok is False else 'no reference'}")
        sys.exit(0 if all(v for v in verdicts.values()) else 1)
