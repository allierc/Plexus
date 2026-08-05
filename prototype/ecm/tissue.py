"""tissue -- PASS 1: run cellfix_B_new itself, and keep everything a matrix or a movie can need.

    from tissue import load_or_build
    T = load_or_build(frames=401, device="cuda:0")      # cached; the 2nd caller pays nothing

WHY THIS IS A MODULE AND NOT A FUNCTION INSIDE THE ECM RUN. Five ECM runs that differ only in the
MATRIX must load the SAME tissue, or the sweep compares two things at once and attributes the
difference to the wrong one. The tissue is therefore built once into
`log/okuda_ECM/_tissue/<name>.npz` and every run reads it. It also means a change of mind about a
colour never costs the vertex model again.

WHAT "COMPLY WITH cellfix_B_new" MEANS HERE, CONCRETELY. The spec is not rewritten, reparameterised
or rescaled: `log/okuda/cellfix_B_new/spec_run.yaml` is loaded verbatim and run by the stock engine,
so the operator stack IS the reference one --

    seed_mesh_3d -> cell_geometry_3d -> morphogen_growth_3d -> shape_energy_3d
                 -> reconnect_t1_3d -> divide_3d -> topo_snapshot_3d

200 cells at radius 5 in a 50-unit box, growing and dividing to ~3,200 under the 3D AVM shape
energy. Nothing about the epithelium is this experiment's variable; the matrix is.

THE SURFACE THE MATRIX FEELS IS BUILT FROM THE APICAL VERTICES, not from cell centroids. The mesh
vertices ARE the apical surface -- `_draw` makes the basal ring by scaling them inward by 0.82 --
so binning them is binning the boundary itself. MEASURED, because the honest version of this claim
is small: the median apical-vertex radius runs 4.66 -> 15.91 against the cell-centroid radius's
4.60 -> 15.90, i.e. ~1%, since a face centroid is a chord's midpoint on the same surface. What
actually changes is the SAMPLING -- 6,400 vertices instead of 3,200 face centroids, so more of the
angular map is covered by real geometry and less of it by a filled-in row mean, and the max-per-bin
boundary is the outermost vertex rather than the outermost centroid.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, os.path.join(ROOT, "src"), os.path.join(ROOT, "prototype", "Tyssue"),
          os.path.join(ROOT, "discovery_okuda")):
    if p not in sys.path:
        sys.path.insert(0, p)

LOG = os.path.join(ROOT, "log", "okuda_ECM")
CACHE = os.path.join(LOG, "_tissue")
CELL_SPEC = os.path.join(ROOT, "log", "okuda", "cellfix_B_new", "spec_run.yaml")

# 32 x 64 rather than 48 x 96. The map has to be resolved by the VERTICES PRESENT, and the opening
# frames have ~1,200 of them: 4,608 bins left two thirds of the sphere empty and filled from a row
# mean, which is a smoother sphere than the tissue and hides the early shape entirely. 2,048 bins
# are covered from frame 0 and still resolve a cell.
N_THETA, N_PHI = 32, 64
RENDER_FRAMES = 90          # frames whose full mesh is kept -- the movie draws exactly these
MESH_PAD = 1.12             # camera headroom, the `run_one.run_box` convention


def apical_map(vp, n_theta=N_THETA, n_phi=N_PHI):
    """Vertex cloud -> R(theta, phi), the FURTHEST apical vertex in each direction.

    Centroid-referenced: `vp` must already have the tissue centroid subtracted. Nothing pins the
    vesicle to the origin, and an origin-referenced radius reads the vesicle's DRIFT as growth --
    the same defect `tube_analysis._cell_centroids` documents for `protr`.
    """
    r = np.linalg.norm(vp, axis=1)
    ok = r > 1e-9
    if ok.sum() < 8:
        return np.zeros((n_theta, n_phi), np.float32)
    u = vp[ok] / r[ok, None]
    th = np.arccos(np.clip(u[:, 2], -1, 1))
    ph = np.arctan2(u[:, 1], u[:, 0]) % (2 * np.pi)
    it = np.clip((th / np.pi * n_theta).astype(int), 0, n_theta - 1)
    ip = np.clip((ph / (2 * np.pi) * n_phi).astype(int), 0, n_phi - 1)
    M = np.zeros((n_theta, n_phi), np.float32)
    np.maximum.at(M, (it, ip), r[ok].astype(np.float32))

    # AN EMPTY BIN IS A GAP IN THE SAMPLING, NOT A HOLE IN THE TISSUE. Left at zero it reads as
    # "the surface is at the centre here", and the matrix flows into a wedge that does not exist.
    for i in range(n_theta):
        row = M[i]
        if (row > 0).any():
            row[row == 0] = row[row > 0].mean()
        else:
            M[i] = M[M > 0].mean() if (M > 0).any() else 0.0
    # ONE SMOOTHING PASS, WRAPPED IN PHI. A max-per-bin map is jagged by construction -- each bin
    # takes its single furthest vertex -- and a jagged boundary makes the contact force a field of
    # spikes rather than a surface pressing. Wrapped in phi because phi is periodic and clamped in
    # theta because the poles are not.
    P = np.pad(M, ((1, 1), (0, 0)), mode="edge")
    P = np.concatenate([P[:, -1:], P, P[:, :1]], axis=1)
    M = sum(P[i:i + n_theta, j:j + n_phi] for i in range(3) for j in range(3)) / 9.0
    return M.astype(np.float32)


def _mesh_of(hist_t, pos_t, centroid):
    """One frame's mesh, centroid-referenced, with the per-cell fields the render colours by."""
    nv = int(hist_t["Nv"])
    d = {"pos": (pos_t[:nv] - centroid).astype(np.float32),
         "nF": np.int32(hist_t["nF"]), "Nv": np.int32(nv)}
    for k in ("E_srce", "E_trgt", "E_face"):
        d[k] = np.asarray(hist_t[k], np.int32)
    # `age` and `ndiv` ARE THE GREEN. divide_3d resets age to 0 on division and ndiv counts the
    # divisions a lineage has had; run_one paints a cell green when `age <= 4 and ndiv > 0`. Both
    # have to travel with the mesh or the ECM movie shows a tissue that never divided -- which is
    # exactly what the reference strip's first two rows are FOR.
    for k in ("age", "ndiv"):
        v = hist_t.get(k)
        d[k] = (np.asarray(v, np.float32) if v is not None
                else np.full(int(hist_t["nF"]), np.nan, np.float32))
    return d


def build(frames, device, out_npz, n_render=RENDER_FRAMES, buffer_x=1):
    """Run cellfix_B_new verbatim and write the cache.

    `buffer_x` MULTIPLIES THE VERTEX AND CELL RESERVOIRS AND NOTHING ELSE. At the reference buffers
    the run reports, loudly and by design, `RESERVOIR FULL: 1723 division(s) refused for want of
    vertex buffer (6396/6396)` -- division stops around frame 310 of 402 and the tissue coasts to
    the end at a fixed 3,170 cells, so the last ~quarter of every movie shows an epithelium that has
    stopped proliferating because of an ARRAY, not because of its biology. Every mechanical
    parameter is untouched: the reservoir is a memory allocation, so growing it changes what the run
    is ALLOWED to do, not what it is trying to do.
    """
    import run_one as R
    S, engine_run = R._lazy_engine()
    from tube_analysis import _cell_centroids

    spec = yaml.safe_load(open(CELL_SPEC))
    spec["general"]["n_frames"] = int(frames)
    spec["general"]["name"] = "cellfix_B_new_for_ecm"
    if buffer_x != 1:
        for st, key in (("cell", "n"), ("vertex", "n")):
            spec["sets"][st][key] = int(spec["sets"][st][key] * buffer_x)
        print(f"[tissue] reservoirs x{buffer_x}: cell {spec['sets']['cell']['n']}, "
              f"vertex {spec['sets']['vertex']['n']}", flush=True)
    # UNIQUE PER CACHE, because two builds can run at once (one per GPU) and a shared temp path
    # would have each write the other's spec out from under it.
    p = os.path.join("/tmp", os.path.basename(out_npz).replace(".npz", "") + ".yaml")
    open(p, "w").write(yaml.safe_dump(spec, sort_keys=False))
    t0 = time.time()
    H, out = engine_run(S.load(p), device=device)
    hist = (H.level("vertex")._mesh or {}).get("hist", [])
    if not hist:
        raise RuntimeError("no topo_snapshot history -- the tissue cannot be drawn or coupled")
    posf = out["sets"]["vertex"]["pos"]
    T = min(posf.shape[0], len(hist))

    maps, r_ap, r_med, ncell, cent = [], [], [], [], []
    for t in range(T):
        mt = hist[t]
        nv = int(mt["Nv"])
        vp = posf[t][:nv].astype(np.float64)
        cen, rad, live = _cell_centroids(vp, mt)
        c = vp.mean(0)                                    # the vertex centroid: what `_draw` centres on
        v = vp - c
        maps.append(apical_map(v))
        r_ap.append(float(np.median(np.linalg.norm(v, axis=1))))
        r_med.append(float(np.median(rad[live])) if live.any() else 0.0)
        ncell.append(int(mt["nF"]))
        cent.append(c)

    keep = np.unique(np.linspace(0, T - 1, min(n_render, T)).astype(int))
    mesh = {"mesh_frames": keep.astype(np.int32)}
    extent = 0.0
    for j, t in enumerate(keep):
        d = _mesh_of(hist[int(t)], posf[int(t)].astype(np.float64), cent[int(t)])
        extent = max(extent, float(np.abs(d["pos"]).max()))
        for k, v in d.items():
            mesh[f"m{j}_{k}"] = v

    M = np.stack(maps)
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz, smap=M, r_apical=np.asarray(r_ap, np.float32),
        r_med=np.asarray(r_med, np.float32), n_cells=np.asarray(ncell, np.int32),
        centroid=np.asarray(cent, np.float32),
        # ONE camera half-width for the WHOLE run, measured over every kept frame. Per-frame
        # autofit is what hid growth in every archived movie until run_one.run_box was written:
        # a vesicle that doubles in radius renders at constant apparent size.
        Lbox=np.float32(extent * MESH_PAD), **mesh)
    print(f"[tissue] cellfix_B_new: {T} frames, {ncell[0]} -> {ncell[-1]} cells, "
          f"apical radius {r_ap[0]:.2f} -> {r_ap[-1]:.2f} (cell-centroid radius "
          f"{r_med[0]:.2f} -> {r_med[-1]:.2f}), {len(keep)} meshes kept, "
          f"{time.time()-t0:.0f}s -> {os.path.relpath(out_npz, ROOT)}", flush=True)
    return out_npz


def load_or_build(frames=401, device="cuda:0", name="cellfix_B_new", rebuild=False,
                  buffer_x=1):
    """The cache path, built if missing. Frames are part of the filename: a 401-frame tissue and a
    120-frame one are different tissues, and silently reusing one for the other would be a run
    whose movie stops before the thing it was testing happened."""
    tag = f"{name}_f{int(frames)}" + (f"_x{int(buffer_x)}" if buffer_x != 1 else "")
    out = os.path.join(CACHE, f"{tag}.npz")
    if rebuild or not os.path.exists(out):
        build(frames, device, out, buffer_x=buffer_x)
    else:
        z = np.load(out)
        print(f"[tissue] reusing {os.path.relpath(out, ROOT)}  "
              f"({z['smap'].shape[0]} frames, {int(z['n_cells'][-1])} cells, "
              f"apical r {float(z['r_apical'][0]):.2f} -> {float(z['r_apical'][-1]):.2f})",
              flush=True)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=401)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--buffer-x", type=int, default=1)
    a = ap.parse_args()
    load_or_build(a.frames, a.device, rebuild=a.rebuild, buffer_x=a.buffer_x)
