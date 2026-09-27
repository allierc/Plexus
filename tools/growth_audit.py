#!/usr/bin/env python
"""Tissue-growth consistency, scored 0-10 from a run's trajectory alone -- no picture, no VLM.

    PYTHONPATH=src python tools/growth_audit.py tissue/ms3_prism_shell cell/exp02_v5 [--json out.jsonl]

THE SCALE (the auditor's rubric, `experiments/exp03_size_hypotheses/AGENT_growth_audit.md`):

     0-2   explosion / chaotic     the mesh is wrecked: non-finite positions, inverted cells, a shell
                                   that has lost its shape, vertices jumping an edge length a frame
     2-4   smooth but no growth    intact and quiet, but the tissue does not grow
     4-6   growth but not smooth   it grows, and cells flicker or the tissue jerks
     6-8   growth, smooth, not uniform
                                   it grows smoothly, but cells or shape drift apart: thickness,
                                   prisms, asphericity or the size spread leave their bands
     8-10  good growth, smooth and uniform

THE AXES ARE READ IN THAT ORDER AND THE FIRST FAILING ONE SETS THE BAND; the position inside the
band is the quality of the axis that failed (or of the weakest margin, in 8-10):

    explosion   score = 2 x (fraction of the run before the first wrecked frame); the jump and
                finiteness tests run on EVERY frame, the other wreck tests on every `every`-th
    no growth   score = 2 + 2 x smoothness quality
    not smooth  score = 4 + 2 x smoothness quality
    not uniform score = 6 + 2 x uniformity quality
    good        score = 8 + 2 x min(growth, smoothness, uniformity) margin

EVERY THRESHOLD IS A CONSTANT BELOW, with the run it was calibrated on. They are the ruler's; the
auditor may argue with a band in its report but never moves it for one run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

# ----------------------------------------------------------------------------------- thresholds
# EXPLOSION -- any one of these at any sampled frame wrecks the run from that frame on.
X_INV = 0.05          # fraction of cells inverted (wedge volume <= 0); an extrusion in progress is 1/200
X_ASPH = 0.20         # shell asphericity (std/mean of vertex radius); crumpled archives read 0.12-0.44
X_HMIN = 0.10         # thinnest cell thickness over the median (shells); crumpled read 0.008-0.09
X_JUMP = 1.0          # max vertex displacement in one frame, in median edge lengths
X_VMED = (0.2, 5.0)   # median cell volume against its value after the settle window
# GROWTH -- the larger of the cell-count and tissue-volume ratios, end over start.
G_MIN = 1.5           # below this the tissue did not grow
G_FULL = 4.0          # two doublings = full growth quality
# SMOOTHNESS -- each cell's size about its own 5-frame moving average, RMS over a window, as a
# fraction of its size; the 90th percentile over the cells followed through the window.
S_MAX = 0.02          # above this, cells visibly flicker (exp02 v1: 0.056; its fix v3: 0.0008)
S_GOOD = 0.002        # at or below this, full smoothness quality
# UNIFORMITY -- held at the last frame (shells: `spheroid_gauge` bands, the working point's).
U_ASPH = 0.06
U_HCV = 0.15
U_HMIN = 0.5
U_TRAP = 0.05
U_CV = 0.40           # cell-size CV at the end
U_CV_RISE = 0.10      # rise of the cell-size CV from MID-RUN to the end: sizes still diverging late.
                      # Not from the start: a seed of near-identical cells (CV 0.02 at the settle frame)
                      # spreads to 0.2-0.3 as soon as it cycles, whatever the rule -- ms3, the accepted
                      # working point, read 0.217 from the start and failed on it (calibration, 2026-09-25)


def _load(spec):
    from plexus.paths import graphs_data_path
    group, name = spec.split("/", 1)
    d = graphs_data_path(group, name)
    tj = os.path.join(d, "trajectory.npz")
    if not os.path.isfile(tj):
        raise SystemExit(f"{tj} missing")
    settle = 0
    try:
        import yaml
        s = yaml.safe_load(open(os.path.join(d, "spec.yaml")))
        for o in s.get("seed") or []:
            if o.get("op") == "seed_mesh":
                settle = int(o.get("ref_frame", 0) or 0)
    except Exception:                                            # noqa: BLE001
        pass
    from size_report import LazyNpz
    return d, LazyNpz(tj), settle


def _kind(z):
    if "vertex__sep" in z.files:
        return "shell"
    P = np.asarray(z["vertex__pos"][0], float)
    occ = np.asarray(z["vertex__occ"][0]) > 0 if "vertex__occ" in z.files else slice(None)
    P = P[occ]
    ext = P.max(0) - P.min(0)
    return "sheet" if ext[-1] < 0.05 * max(ext[0], ext[1], 1e-9) else "surface"


def _sizes(z, t, kind):
    """Per-slot cell size at frame t (area on a sheet, the rules' volume otherwise), and the live mask."""
    from size_report import _volumes
    nF = int(z["vertex__mesh_nF"][t])
    occ = (np.asarray(z["cell__occ"][t])[:nF] > 0) if "cell__occ" in z.files else np.ones(nF, bool)
    if kind == "sheet" and "cell__area" in z.files:
        v = np.asarray(z["cell__area"][t], float)[:nF, 0]
    else:
        v = _volumes(z, t, "polyhedron" if kind == "shell" else "wedge")[:nF]
    return v, occ & np.isfinite(v)


def _rough(P):
    """ROUGHNESS ABOUT THE BEST-FIT ELLIPSOID: std/mean of the vertex radius after each principal
    axis is scaled by the tissue's extent along it. A clean sphere AND a clean elongated body both
    read ~0; a crumpled shell still does not. The plain asphericity (std/mean of the radius about the
    centroid) cannot tell an elongated body from a crumpled one -- it read 0.20-0.30, "wrecked", on
    the intact corset arms of exp 5 round 1 -- so the shell checks use this instead (2026-09-25)."""
    X = P - P.mean(0)
    w, V = np.linalg.eigh(X.T @ X / max(len(X), 1))
    Y = X @ V
    ext = np.ptp(Y, axis=0)
    Y = Y / np.maximum(ext, 1e-12)
    r = np.linalg.norm(Y, axis=1)
    return float(r.std() / max(r.mean(), 1e-12))


def _jump(z, t):
    """Largest vertex displacement from t-1 to t, in median edge lengths (common vertex prefix)."""
    if t < 1:
        return 0.0
    nv = min(int(z["vertex__mesh_Nv"][t - 1]), int(z["vertex__mesh_Nv"][t]))
    a = np.asarray(z["vertex__pos"][t - 1], float)[:nv]
    b = np.asarray(z["vertex__pos"][t], float)[:nv]
    d = np.linalg.norm(b - a, axis=1)
    off = z["vertex__mesh_offsets"]
    s = np.asarray(z["vertex__mesh_E_srce"][off[t]:off[t + 1]])
    e = np.asarray(z["vertex__mesh_E_trgt"][off[t]:off[t + 1]])
    L = np.linalg.norm(b[np.clip(e, 0, nv - 1)] - b[np.clip(s, 0, nv - 1)], axis=1) if len(s) else [1.0]
    return float(np.nanmax(d) / max(np.nanmedian(L), 1e-9)) if d.size else 0.0


def _every_frame(z, t0, T):
    """THE JUMP AND FINITENESS TESTS ON EVERY FRAME, not every `every`-th (added 2026-09-27). Sampled
    at every 20th frame, a vertex that jumped 4 edge lengths between two samples was never seen, so a
    clean audit did not mean a clean run: exp07's arms read 1.7 against 4.0 by sampling luck (its
    finding 54). The arrays are decompressed once (LazyNpz) and one frame costs ~1 ms at 20,000
    vertices (a 1,501-frame, 1.9 GB run: 1.7 s). Returns (first bad frame or None, why, the largest
    jump in median edge lengths, its frame, the number of frames whose jump exceeds X_JUMP)."""
    jmax, jat, bad, why, n_over = 0.0, None, None, "", 0
    for t in range(max(t0, 1), T):
        P = np.asarray(z["vertex__pos"][t], float)[:int(z["vertex__mesh_Nv"][t])]
        if not np.isfinite(P).all():
            return (t if bad is None else bad), (why or "non-finite positions"), jmax, jat, n_over
        j = _jump(z, t)
        if j > jmax:
            jmax, jat = j, t
        if j > X_JUMP:
            n_over += 1
            if bad is None:
                bad, why = t, f"a vertex jumped {j:.2f} edge lengths in one frame"
    if n_over > 1:
        why += f" ({n_over} frames over {X_JUMP} in the run, the largest {jmax:.2f} at frame {jat})"
    return bad, why, jmax, jat, n_over


def _jitter(z, t0, t1, kind):
    """p90 over cells of RMS(size - 5-frame moving average)/mean, for cells present through [t0, t1)."""
    ids = "cell__cell_id" in z.files
    traces = {}
    for t in range(t0, t1):
        v, live = _sizes(z, t, kind)
        key = (np.asarray(z["cell__cell_id"][t], float)[:len(v), 0].astype(np.int64) if ids
               else np.arange(len(v)))
        for k, x, ok in zip(key, v, live):
            if ok:
                traces.setdefault(int(k), []).append(x)
    n = t1 - t0
    J = []
    for tr in traces.values():
        if len(tr) != n or np.mean(tr) <= 0:
            continue
        tr = np.asarray(tr)
        sm = np.convolve(tr, np.ones(5) / 5, mode="valid")
        J.append(float(np.sqrt(np.mean((tr[2:-2] - sm) ** 2)) / np.mean(tr)))
    return (float(np.percentile(J, 90)), float(np.median(J)), len(J)) if J else (np.nan, np.nan, 0)


def audit(spec, every=20, window=60, jump_every=1):
    from spheroid_gauge import frame_metrics
    d, z, settle = _load(spec)
    T = int(len(z["vertex__mesh_nF"]))
    kind = _kind(z)
    t0 = min(max(settle, 1), T - 1)
    ts = sorted(set(list(range(t0, T, every)) + [T - 1]))
    series, wrecked, why = [], None, ""
    for t in ts:
        v, live = _sizes(z, t, kind)
        P = np.asarray(z["vertex__pos"][t], float)[:int(z["vertex__mesh_Nv"][t])]
        row = dict(t=t, cells=int(live.sum()), total=float(v[live].sum()),
                   median=float(np.median(v[live])) if live.any() else np.nan,
                   cv=float(v[live].std() / v[live].mean()) if live.any() else np.nan,
                   jump=_jump(z, t), finite=bool(np.isfinite(P).all()))
        if kind != "sheet":
            row.update({k: float(val) for k, val in frame_metrics(z, t).items() if k != "cells"})
            row["rough"] = _rough(P[np.isfinite(P).all(1)])
        series.append(row)
        if wrecked is None:
            med0 = series[0]["median"]
            bad = [(not row["finite"], "non-finite positions"),
                   (row.get("inv_wedge", 0) > X_INV, f"{row.get('inv_wedge', 0):.3f} of cells inverted"),
                   (kind == "shell" and row.get("rough", 0) > X_ASPH, f"roughness {row.get('rough', 0):.3f} about the fitted ellipsoid"),
                   (kind == "shell" and row.get("h_min_rel", 1) < X_HMIN, f"thinnest cell {row.get('h_min_rel', 1):.3f} of the median"),
                   (row["jump"] > X_JUMP, f"a vertex jumped {row['jump']:.2f} edge lengths in one frame"),
                   (np.isfinite(med0) and not (X_VMED[0] < row["median"] / max(med0, 1e-12) < X_VMED[1]),
                    f"median cell size {row['median'] / max(med0, 1e-12):.2f}x its settled value")]
            hit = [w for b, w in bad if b]
            if hit:
                wrecked, why = t, "; ".join(hit)
    jump_max, jump_at = max(((r["jump"], r["t"]) for r in series), default=(0.0, None))
    n_over = sum(r["jump"] > X_JUMP for r in series)
    if jump_every == 1:
        ev, ev_why, jump_max, jump_at, n_over = _every_frame(z, t0, T)
        if ev is not None and (wrecked is None or ev < wrecked):
            wrecked, why = ev, ev_why
    first, last = series[0], series[-1]
    growth = max(last["cells"] / max(first["cells"], 1), last["total"] / max(first["total"], 1e-12))
    end = T - 1 if wrecked is None else max(t0 + window + 4, wrecked - 1)
    mid = t0 + (end - t0) // 2
    jit_mid = _jitter(z, max(t0, mid - window // 2), min(end, mid + window // 2), kind)
    jit_end = _jitter(z, max(t0, end - window), end, kind)
    jit = np.nanmax([jit_mid[0], jit_end[0]])
    mid_row = min(series, key=lambda r: abs(r["t"] - (t0 + (end - t0) // 2)))
    u = dict(cv_end=last["cv"], cv_rise=last["cv"] - mid_row["cv"])
    if kind == "shell":
        u.update(asph=last.get("rough"), h_cv=last.get("h_cv"), h_min_rel=last.get("h_min_rel"),
                 trapezoid=last.get("trapezoid"))
    # MARGINS in [0, 1]: 1 is comfortably inside, 0 is on the line.
    g_q = float(np.clip(np.log(max(growth, 1e-9) / 1.0) / np.log(G_FULL), 0, 1))
    s_q = float(np.clip((S_MAX - jit) / (S_MAX - S_GOOD), 0, 1)) if np.isfinite(jit) else 0.0
    u_margins = [1 - u["cv_end"] / U_CV, 1 - max(u["cv_rise"], 0) / U_CV_RISE]
    if kind == "shell":
        u_margins += [1 - u["asph"] / U_ASPH, 1 - u["h_cv"] / U_HCV,
                      (u["h_min_rel"] - U_HMIN) / (1 - U_HMIN), 1 - u["trapezoid"] / U_TRAP]
    u_q = float(np.clip(min(u_margins), 0, 1))
    uniform = min(u_margins) >= 0
    if wrecked is not None:
        band, score = "explosion / chaotic", 2.0 * (wrecked - t0) / max(T - 1 - t0, 1)
        reason = f"wrecked at frame {wrecked}: {why}"
    elif growth < G_MIN:
        band, score = "smooth but no growth", 2.0 + 2.0 * s_q
        reason = f"grew only x{growth:.2f} (cells and tissue volume) -- below x{G_MIN}"
    elif not (jit <= S_MAX):
        band, score = "growth but not smooth", 4.0 + 2.0 * s_q
        reason = f"cells flicker: p90 size wobble {jit:.4f} of their size per frame (limit {S_MAX})"
    elif not uniform:
        band, score = "growth, smooth, not uniform", 6.0 + 2.0 * u_q
        worst = min(zip(u_margins, ["cv_end", "cv_rise", "asph", "h_cv", "h_min_rel", "trapezoid"]))[1]
        reason = f"uniformity fails first on {worst} = {u[worst]:.3f}"
    else:
        band, score = "good growth, smooth and uniform", 8.0 + 2.0 * min(g_q, s_q, u_q)
        reason = f"x{growth:.2f} growth, p90 wobble {jit:.4f}, every uniformity band held"
    return dict(spec=spec, kind=kind, frames=T, settle=t0, score=round(float(score), 2), band=band,
                reason=reason, wrecked_at=wrecked, growth=round(float(growth), 3),
                cells=[first["cells"], last["cells"]],
                jump_max=round(float(jump_max), 3), jump_at=jump_at, jumps_over=int(n_over), jump_every=jump_every, jitter_p90=round(float(jit), 5),
                jitter_mid=[round(x, 5) if isinstance(x, float) else x for x in jit_mid],
                jitter_end=[round(x, 5) if isinstance(x, float) else x for x in jit_end],
                uniformity={k: (round(float(v), 4) if v is not None else None) for k, v in u.items()},
                quality=dict(growth=round(g_q, 3), smooth=round(s_q, 3), uniform=round(u_q, 3)),
                series=series)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+", help="<group>/<run>")
    ap.add_argument("--every", type=int, default=20, help="frame stride for the whole-run series")
    ap.add_argument("--window", type=int, default=60, help="frames per smoothness window")
    ap.add_argument("--sampled-jump", action="store_true",
                    help="the pre-2026-09-27 audit: jump and finiteness only on the --every frames")
    ap.add_argument("--json", default=None, help="append one JSON line per run here")
    a = ap.parse_args()
    for sp in a.specs:
        r = audit(sp, a.every, a.window, jump_every=a.every if a.sampled_jump else 1)
        print(f"{r['spec']:<34} {r['score']:5.2f}  {r['band']:<32} {r['reason']}")
        if a.json:
            with open(a.json, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
