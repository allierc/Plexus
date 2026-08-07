#!/usr/bin/env python
"""measure_1frame -- run ONE composition and measure the analysis at every-frame resolution.

WHAT THIS SETTLES. `okuda_route`'s activator, sampled every 23 frames, shows fourteen separate
100%-red episodes swinging 0.010 -> 17,680 -> 0.010, and the gaps between them are 69, 46, 46,
116, 46, ... -- every one an exact multiple of the sampling interval. That is the signature of
ALIASING: what was measured is the beat between an oscillation and our sampling, not the
oscillation. The true period is not recoverable from that record at all.

It matters mechanically, not just tidily. `red_frac` = 1 means the growth operator is acting on
EVERY CELL AT ONCE, which grows a sphere uniformly -- and `okuda_route` grows correctly to 3,975
cells and stays a sphere. Whether the chemistry is a limit cycle (a finding about the mechanism)
or a numerical blow-up-and-reset (a bug in the integrator) is unanswerable at 40 samples, and the
two demand opposite responses.

Measured beforehand on a real 3,975-cell mesh, per frame of analysis:

    whole frame_metrics   1410 ms      hollow_flags 583 ms, face_polygons 301 ms
    cell centroids          30 ms      -> 1.9 ms vectorised, bit-identical
    CHEMISTRY ONLY        0.12 ms      0.008% of a frame

So the chemistry can be measured at EVERY frame for about a tenth of a second per run, while the
mesh metrics -- whose statistics were shown to be stable under decimation, unlike the chemistry's
-- stay at a coarse stride. This script measures that claim end to end rather than projecting it.

    python measure_1frame.py okuda_route --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, os.path.join(ROOT, "src"), os.path.join(ROOT, "prototype", "Tyssue")):
    if p not in sys.path:
        sys.path.insert(0, p)


def chem_metrics(a, a_sw=0.5):
    """The activator metrics, on ONE frame's per-cell array. Pure numpy: 0.12 ms for 3,975 cells.

    BATCHED where it can be: everything here is a reduction over one axis, so the whole run is a
    single (frames x cells) matrix operation rather than a Python loop -- see `chem_metrics_batch`.
    """
    a = np.asarray(a, float)
    mu, sd = float(a.mean()), float(a.std())
    lo, hi = float(a.min()), float(a.max())
    cv = sd / abs(mu) if abs(mu) > 1e-12 else 0.0
    occ = float((a > lo + 0.5 * (hi - lo)).mean()) if hi > lo + 1e-12 else 0.0
    return dict(act_mean=mu, act_sd=sd, act_cv=cv, act_max=hi, act_min=lo,
                act_occupancy=occ, red_frac=float((a > a_sw).mean()),
                act_alive=float(cv > 0.05 and occ > 0.01))


def chem_metrics_batch(A, occ_mask, a_sw=0.5):
    """ALL FRAMES AT ONCE. `A` is (frames x cells) with dead cells masked out by `occ_mask`.

    A Python loop over 900 frames costs 900 interpreter round-trips for arithmetic that numpy
    does in one pass over contiguous memory. The reductions are identical -- this is the same
    arithmetic as `chem_metrics`, vectorised over the frame axis, and the two are checked against
    each other below rather than assumed equal.
    """
    A = np.where(occ_mask, A, np.nan)
    mu = np.nanmean(A, axis=1)
    sd = np.nanstd(A, axis=1)
    lo = np.nanmin(A, axis=1)
    hi = np.nanmax(A, axis=1)
    cv = np.where(np.abs(mu) > 1e-12, sd / np.maximum(np.abs(mu), 1e-30), 0.0)
    thr = lo + 0.5 * (hi - lo)
    occ = np.nanmean((A > thr[:, None]).astype(float), axis=1)
    occ = np.where(hi > lo + 1e-12, occ, 0.0)
    red = np.nanmean((A > a_sw).astype(float), axis=1)
    alive = ((cv > 0.05) & (occ > 0.01)).astype(float)
    return dict(act_mean=mu, act_sd=sd, act_cv=cv, act_max=hi, act_min=lo,
                act_occupancy=occ, red_frac=red, act_alive=alive)


def cell_centroids_fast(pt, mt):
    """Per-cell centroids, radius and live mask -- VECTORISED.

    `tissue_analysis._cell_centroids` builds the rings as a Python list comprehension over every
    face: 29.7 ms for 3,975 cells. This is the same quantity by scatter-add: 1.9 ms, 15x faster,
    and verified bit-identical (max radius difference 0.000e+00, live masks equal) on a real end
    mesh before being used here.
    """
    es, ef, nF = np.asarray(mt["E_srce"]), np.asarray(mt["E_face"]), mt["nF"]
    live_e = ef < nF
    es, ef = es[live_e], ef[live_e]
    cnt = np.bincount(ef, minlength=nF).astype(float)
    cen = np.zeros((nF, 3))
    np.add.at(cen, ef, pt[es])
    live = cnt > 0
    cen[live] /= cnt[live, None]
    origin = cen[live].mean(0) if live.any() else np.zeros(3)
    cen = cen - origin
    rad = np.linalg.norm(cen, axis=1)
    rad[~live] = 0.0
    return cen, rad, live


def centroid_metrics(pt, mt, act, a_sw=0.5):
    """The shape and coupling metrics that need centroids but no mesh geometry -- 1.9 ms."""
    cen, rad, live = cell_centroids_fast(pt, mt)
    if live.sum() < 8:
        return {}
    r = rad[live]
    med = float(np.median(r))
    out = {}
    if med > 1e-9:
        out["protr"] = float(np.percentile(r, 95) / med)
        out["protr_p99"] = float(np.percentile(r, 99) / med)
        out["r_cv"] = float(r.std() / (r.mean() + 1e-12))
        try:
            w = np.linalg.eigvalsh(np.cov(cen[live].T))[::-1]
            tr = float(w.sum())
            if tr > 1e-12:
                out["gyr_prolate"] = float(w[0] / (0.5 * (w[1] + w[2]) + 1e-12))
                out["gyr_oblate"] = float(1.5 * (w[1] - w[2]) / tr)
        except Exception:
            pass
    a = np.asarray(act, float)
    if a.size == rad.size:
        al, rl = a[live], rad[live]
        # THE COUPLING IS REFUSED ON A DEAD FIELD. Pearson is scale-free by construction, so it
        # returns a confident number for noise: measured 0.294 on an activator whose entire
        # spread across 3,975 cells was 8.4e-05. Same act_cv > 0.05 floor as act_alive.
        cv = float(al.std() / abs(al.mean())) if abs(al.mean()) > 1e-12 else 0.0
        if cv > 0.05 and al.std() > 1e-12 and rl.std() > 1e-12:
            out["corr_act_rad"] = float(np.corrcoef(al, rl)[0, 1])
            # GRIP = CORRELATION x AMPLITUDE, because Pearson normalises the amplitude away and
            # that is exactly what made it useless as the campaign's headline. A perfectly
            # correlated 1% wobble scores identically to a perfectly correlated tube. Measured
            # over 273 runs: `r002_10` reported corr 0.922 -- the SECOND HIGHEST coupling of the
            # campaign -- on a sphere (r_cv 0.081, protr 1.163), and `r017_03` reported 0.356 on
            # a shape Cedric watched and saw nothing happen in. Multiplying by the radial spread
            # the correlation is about drops r002_10 from rank 2 to rank 35 and puts genuinely
            # deformed runs on top. The old number stays, as a diagnostic; this is the headline.
            rcv = float(rl.std() / abs(rl.mean())) if abs(rl.mean()) > 1e-12 else 0.0
            out["grip"] = out["corr_act_rad"] * rcv
            tip = rl >= np.percentile(rl, 90)
            if tip.any() and abs(al.mean()) > 1e-12:
                out["act_at_tip"] = float(al[tip].mean() / al.mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="config name in config/okuda, e.g. okuda_route")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--mesh-stride", type=int, default=25)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import yaml
    import run_one as R
    S, engine_run = R._lazy_engine()          # the same lazy import path a real run uses
    cfg_path = os.path.join(ROOT, "config", "okuda", f"{a.name}.yaml")
    if not os.path.exists(cfg_path):
        alt = os.path.join(ROOT, "log", "okuda", a.name, "spec_run.yaml")
        cfg_path = alt if os.path.exists(alt) else cfg_path
    print(f"[measure] {a.name}  spec={cfg_path}  device={a.device}", flush=True)

    sim = S.load(cfg_path)
    if a.frames:
        sim.n_frames = a.frames
    t0 = time.time()
    Hf, out = engine_run(sim, device=a.device)
    t_sim = time.time() - t0
    print(f"[measure] simulation: {t_sim/60:.1f} min", flush=True)

    vlvl = Hf.level("vertex")
    emesh = getattr(vlvl, "_mesh", None) or {}
    hist = emesh.get("hist", [])
    posf = out["sets"]["vertex"]["pos"]
    chemf = out["sets"]["cell"]["state"]["chem"]
    T = posf.shape[0]
    print(f"[measure] {T} recorded frames, {hist[-1]['nF']} cells at the end", flush=True)

    cfg = yaml.safe_load(open(cfg_path)) or {}
    a_sw = 0.5
    try:
        a_sw = next((float(o["a_sw"]) for o in cfg.get("operators", []) if "a_sw" in o), 0.5)
    except Exception:
        pass
    print(f"[measure] growth switch a_sw = {a_sw}", flush=True)

    # ---------------------------------------------------------------- tier 1: chemistry, batched
    nFmax = max(h["nF"] for h in hist)
    A = np.full((T, nFmax), np.nan)
    M = np.zeros((T, nFmax), bool)
    for t in range(T):
        nF = hist[t]["nF"]
        A[t, :nF] = chemf[t][:nF, 0]
        M[t, :nF] = True
    t0 = time.time()
    chem = chem_metrics_batch(A, M, a_sw=a_sw)
    t_chem_batch = time.time() - t0

    t0 = time.time()
    loopres = [chem_metrics(chemf[t][:hist[t]["nF"], 0], a_sw=a_sw) for t in range(T)]
    t_chem_loop = time.time() - t0
    dmax = max(abs(loopres[t]["act_cv"] - chem["act_cv"][t]) for t in range(T))
    print(f"[measure] chemistry, {T} frames:  batched {t_chem_batch:.3f} s   "
          f"loop {t_chem_loop:.3f} s   ({t_chem_loop/max(t_chem_batch,1e-9):.0f}x)   "
          f"max act_cv disagreement {dmax:.2e}", flush=True)

    # ---------------------------------------------------------------- tier 2: centroids, per frame
    t0 = time.time()
    cent = [centroid_metrics(posf[t][:hist[t]["Nv"]].astype(np.float64), hist[t],
                             chemf[t][:hist[t]["nF"], 0], a_sw=a_sw) for t in range(T)]
    t_cent = time.time() - t0
    print(f"[measure] centroid metrics, {T} frames: {t_cent:.1f} s "
          f"({1000*t_cent/T:.2f} ms/frame)", flush=True)

    # ---------------------------------------------------------------- tier 3: mesh, coarse
    from tissue_analysis import frame_metrics
    idx = np.unique(np.append(np.arange(0, T, a.mesh_stride), T - 1))
    t0 = time.time()
    mesh = [frame_metrics(posf[t][:hist[t]["Nv"]].astype(np.float64), hist[t],
                          act=chemf[t][:hist[t]["nF"], 0], a_sw=a_sw) for t in idx]
    t_mesh = time.time() - t0
    print(f"[measure] mesh metrics, {len(idx)} frames (stride {a.mesh_stride}): {t_mesh:.1f} s "
          f"({1000*t_mesh/len(idx):.0f} ms/frame)", flush=True)
    print(f"[measure] TOTAL analysis {t_chem_batch + t_cent + t_mesh:.1f} s "
          f"(chemistry+centroids at EVERY frame, mesh at {a.mesh_stride})", flush=True)

    # ---------------------------------------------------------------- save
    dst = a.out or os.path.join(ROOT, "log", "okuda", a.name, "frames_1.npz")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cols = {f"chem_{k}": v for k, v in chem.items()}
    for k in ("protr", "protr_p99", "r_cv", "gyr_prolate", "gyr_oblate",
              "corr_act_rad", "act_at_tip"):
        cols[k] = np.array([c.get(k, np.nan) for c in cent], float)
    cols["frame"] = np.arange(T)
    cols["n_cells"] = np.array([h["nF"] for h in hist], float)
    cols["mesh_frame"] = idx.astype(float)
    for k in (mesh[0].keys() if mesh else ()):
        vals = [m.get(k, np.nan) for m in mesh]
        if all(isinstance(v, (int, float, np.floating, np.integer)) or v is None for v in vals):
            cols[f"mesh_{k}"] = np.array([np.nan if v is None else v for v in vals], float)
    np.savez(dst, **cols)
    print(f"[measure] wrote {dst}", flush=True)
    json.dump({"sim_min": t_sim / 60, "frames": int(T),
               "chem_batch_s": t_chem_batch, "chem_loop_s": t_chem_loop,
               "centroid_s": t_cent, "mesh_s": t_mesh, "mesh_frames": int(len(idx)),
               "mesh_stride": a.mesh_stride},
              open(os.path.join(os.path.dirname(dst), "frames_1_timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
