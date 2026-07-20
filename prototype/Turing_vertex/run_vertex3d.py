#!/usr/bin/env python
"""run_vertex3d -- Stage 3 smoke test: the 3D vertex (Self-Propelled Voronoi) MECHANICS,
in the paper's two initial tissues.

  * aggregate_3d -- a compacted 3D aggregate (solid ball, NO lumen)
  * vesicle_3d   -- a monolayer vesicle (cells on a shell around a hollow lumen)

Cells are points; the tissue is their finite (ghost-bounded) 3D Voronoi; mechanics from the
shape energy E = sum K_V(V-V0)^2 + K_S(S-S0)^2 (plexus2 operators in vertex3d_ops.py). V0 is
measured from the seeded tissue so the cells start near target (stable relaxation). Renders
cell centres coloured by 3D shape index s=S/V^(2/3), with an octant cutaway -- the cutaway of
the vesicle reveals the hollow lumen.

    python run_vertex3d.py            # run + archive
    python run_vertex3d.py --rerender # re-render from cache (no re-sim)
"""
from __future__ import annotations
import os, sys, argparse, glob, json, tempfile, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import plexus.operators   # noqa: F401
import vertex3d_ops       # noqa: F401
from vertex3d_ops import cell_shape_3d, cell_faces_3d
import plexus.schema as S
from plexus.engine import run as engine_run

OUT = os.path.join(HERE, "archive")
PANEL = 4.4
S0_STAR = 5.41                                        # 3D rigidity transition


def presets():
    #   name           lumen   R     N     s0    frames  dt    mu
    return [
        ("aggregate_3d", False, 6.0, 500, 5.30, 140, 0.02, 0.05),
        ("vesicle_3d",   True,  6.0, 360, 5.30, 140, 0.02, 0.05),
    ]


def measure_V0(R, N, lumen):
    """Seed-only pass -> median seeded cell volume (so V0 matches; the tissue starts near target)."""
    W = 2.6 * R
    cfg = {"general": {"name": "seed", "seed": 0, "n_frames": 1, "dt": 0.02, "boundary": "free",
                       "dim": 3, "world": [W, W, W]},
           "sets": {"cell": {"n": N}}, "fields": {},
           "operators": [{"op": "tissue_seed_3d", "at": "cell", "radius": R, "lumen": lumen,
                          "v0": 1.0, "before_frame": 1}],
           "schedule": ["tissue_seed_3d"]}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    _, out = engine_run(sim, device="cpu")
    p0 = out["sets"]["cell"]["pos"][-1].astype(np.float64)
    vol, _, _, ok = cell_shape_3d(p0, lumen, 0.15 * R)
    return float(np.median(vol[ok > 0]))


def make_spec(name, lumen, R, N, s0, frames, dt, mu, V0):
    W = 2.6 * R
    cfg = {
        "general": {"name": f"vertex3d_{name}", "seed": 0, "n_frames": frames, "dt": dt,
                    "boundary": "free", "dim": 3, "world": [W, W, W]},
        "sets": {"cell": {"n": N}},
        "fields": {},
        "operators": [
            {"op": "tissue_seed_3d", "at": "cell", "radius": R, "lumen": lumen, "v0": V0, "before_frame": 1},
            {"op": "voronoi_tension_3d", "at": "cell", "s0": s0, "radius": R, "V0": V0,
             "K_V": 1.0, "K_S": 0.5, "mu": mu, "lumen": lumen},
        ],
        "schedule": ["tissue_seed_3d", "voronoi_tension_3d"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def _shape_traj(pos_dec, lumen, R):
    """Shape index per cell per (decimated) frame -> [T,N] (nan where invalid)."""
    out = []
    for p in pos_dec:
        _, _, s, ok = cell_shape_3d(p.astype(np.float64), lumen, 0.15 * R)
        out.append(np.where(ok > 0, s, np.nan))
    return np.array(out)


def _slice_polygon(tris, axis, c0):
    """Cross-section polygon of a convex polyhedron (hull triangles [F,3,3]) by the plane
    coord[axis]=c0: intersect each triangle edge, collect the other-two coords, order angularly."""
    other = [a for a in (0, 1, 2) if a != axis]
    pts = []
    for tri in tris:
        w = tri[:, axis]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (w[a] - c0) * (w[b] - c0) < 0:
                t = (c0 - w[a]) / (w[b] - w[a])
                pts.append((tri[a] + t * (tri[b] - tri[a]))[other])
    if len(pts) < 3:
        return None
    pts = np.array(pts); cen = pts.mean(0)
    return pts[np.argsort(np.arctan2(pts[:, 1] - cen[1], pts[:, 0] - cen[0]))]


def render(pos_dec, shp, outdir, name, lumen, R, seconds=10.0, movie_frames=70):
    """Draw the actual 3D VORONOI POLYHEDRA (Poly3DCollection, translucent) + an octant cutaway
    + a cross-section slice (Voronoi polygons; the vesicle's slice is a ring around the lumen)."""
    os.makedirs(outdir, exist_ok=True)
    T = pos_dec.shape[0]; pad = 0.15 * R
    box = float(np.nanmax([np.ptp(pos_dec[..., k]) for k in range(3)])) + 1.0
    fin = shp[np.isfinite(shp)]
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(*np.percentile(fin, [3, 97]))    # sequential over the shape-index spread
    cmap = plt.get_cmap("plasma")

    def poly3d(ax, faces, svals, cens, cen, cutaway):
        ax.clear(); ax.set_facecolor("black")
        tris, cols = [], []
        for ct, s, c in zip(faces, svals, cens):
            if cutaway and (c[0] > cen[0] and c[1] > cen[1] and c[2] > cen[2]):
                continue
            rc = cmap(norm(s)); rgba = (rc[0], rc[1], rc[2], 0.45)
            for tri in ct:
                tris.append(tri); cols.append(rgba)
        ax.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolors=(1, 1, 1, 0.12), linewidths=0.1))
        ax.set_xlim(cen[0] - box / 2, cen[0] + box / 2); ax.set_ylim(cen[1] - box / 2, cen[1] + box / 2)
        ax.set_zlim(cen[2] - box / 2, cen[2] + box / 2); ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()

    def slice2d(ax, faces, svals, axis, c0, cen):
        ax.clear(); ax.set_facecolor("black")
        o = [a for a in (0, 1, 2) if a != axis]
        for ct, s in zip(faces, svals):
            poly = _slice_polygon(ct, axis, c0)
            if poly is not None:
                ax.fill(poly[:, 0], poly[:, 1], facecolor=cmap(norm(s)), alpha=0.92, edgecolor="white", lw=0.5)
        hb = box * 0.82                                   # widen so the 2D slice matches the 3D panels'
        ax.set_xlim(cen[o[0]] - hb, cen[o[0]] + hb)       # apparent scale (mplot3d draws the cube with margin)
        ax.set_ylim(cen[o[1]] - hb, cen[o[1]] + hb); ax.set_aspect("equal"); ax.axis("off")

    def label(ax, txt, is3d=True):
        (ax.text2D if is3d else ax.text)(0.02, 0.98, txt, transform=ax.transAxes, color="white",
                                         fontsize=7, va="top", family="monospace")

    # strip (no labels): external @0/50%, octant cutaway @100% (PhysiCell-style), cross-section @100%
    def faces_at(t):
        return cell_faces_3d(pos_dec[t].astype(np.float64), lumen, pad), pos_dec[t].mean(0)
    (f0, s0, c0), cen0 = faces_at(0)
    (fM, sM, cM), cenM = faces_at(T // 2)
    (fF, sF, cF), cen = faces_at(T - 1)
    sfig = plt.figure(figsize=(4 * PANEL, PANEL)); sfig.patch.set_facecolor("black")
    a1 = sfig.add_subplot(1, 4, 1, projection="3d"); a1.set_facecolor("black")
    poly3d(a1, f0, s0, c0, cen0, False); a1.view_init(18, 35)
    a2 = sfig.add_subplot(1, 4, 2, projection="3d"); a2.set_facecolor("black")
    poly3d(a2, fM, sM, cM, cenM, False); a2.view_init(18, 35)
    a3 = sfig.add_subplot(1, 4, 3, projection="3d"); a3.set_facecolor("black")
    poly3d(a3, fF, sF, cF, cen, True); a3.view_init(24, 45)          # cutaway (reveals lumen)
    a4 = sfig.add_subplot(1, 4, 4); a4.set_facecolor("black")
    slice2d(a4, fF, sF, 2, float(cen[2]), cen)                       # cross-section ONLY in the last panel
    sfig.subplots_adjust(0.005, 0.005, 0.995, 0.995, wspace=0.02)
    sfig.savefig(os.path.join(outdir, "strip.png"), dpi=120, facecolor="black"); plt.close(sfig)

    # movie (no labels): rotating octant cutaway -- PhysiCell-style, interior visible
    midx = np.linspace(0, T - 1, min(T, movie_frames)).astype(int)
    fig = plt.figure(figsize=(PANEL, PANEL)); fig.patch.set_facecolor("black")
    ax = fig.add_subplot(projection="3d"); ax.set_facecolor("black"); fig.subplots_adjust(0, 0, 1, 1)
    w = FFMpegWriter(fps=max(1, round(len(midx) / seconds)), metadata={"title": name})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=110):
        for k, t in enumerate(midx):
            f, sv, c = cell_faces_3d(pos_dec[t].astype(np.float64), lumen, pad)
            cn = pos_dec[t].mean(0)
            poly3d(ax, f, sv, c, cn, True); ax.view_init(18, 35 + 360.0 * k / len(midx))
            w.grab_frame()
    plt.close(fig)


def run_all():
    for name, lumen, R, N, s0, frames, dt, mu in presets():
        odir = os.path.join(OUT, name); os.makedirs(odir, exist_ok=True)
        print(f"[vertex3d] {name}: lumen={lumen} N={N} s0={s0}", flush=True)
        rec = {"lumen": lumen, "N": N, "s0": s0}
        try:
            V0 = measure_V0(R, N, lumen); rec["V0"] = round(V0, 3)
            sim, cfg = make_spec(name, lumen, R, N, s0, frames, dt, mu, V0)
            yaml.safe_dump(cfg, open(os.path.join(odir, "spec.yaml"), "w"), sort_keys=False)
            _, out = engine_run(sim, device="cpu")
            pos = out["sets"]["cell"]["pos"]
            keep = np.linspace(0, pos.shape[0] - 1, min(pos.shape[0], 120)).astype(int)
            pos_dec = pos[keep].astype("float32")
            shp = _shape_traj(pos_dec, lumen, R)
            np.savez_compressed(os.path.join(odir, "traj.npz"), pos=pos_dec, shp=shp.astype("float32"),
                                lumen=lumen, name=name, R=R)
            rec["shape_idx_final"] = round(float(np.nanmean(shp[-1])), 3)
            render(pos_dec, shp, odir, name, lumen, R)
            print(f"           -> V0={V0:.2f} shape_idx_final={rec['shape_idx_final']}", flush=True)
        except Exception as e:
            rec["error"] = repr(e); traceback.print_exc()
        json.dump(rec, open(os.path.join(odir, "diag.json"), "w"), indent=1)


def rerender(only=None):
    for name, lumen, R, N, s0, frames, dt, mu in presets():
        if only and name not in only:
            continue
        odir = os.path.join(OUT, name); tf = os.path.join(odir, "traj.npz")
        if not os.path.exists(tf):
            print(f"[rerender] {name}: no traj.npz"); continue
        d = np.load(tf)
        render(d["pos"], d["shp"], odir, str(d["name"]), bool(d["lumen"]), float(d["R"]))
        print(f"[rerender] {name}", flush=True)


def montage():
    from PIL import Image
    names = [p[0] for p in presets()]
    strips = [(n, Image.open(os.path.join(OUT, n, "strip.png")).convert("RGB"))
              for n in names if os.path.exists(os.path.join(OUT, n, "strip.png"))]
    if strips:
        w = max(i.width for _, i in strips); h = sum(i.height for _, i in strips)
        sheet = Image.new("RGB", (w, h), (0, 0, 0)); y = 0
        for _, im in strips:
            sheet.paste(im, (0, y)); y += im.height
        sheet.save(os.path.join(OUT, "_montage_vertex3d.png"))
    print(f"[vertex3d] montage -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--montage", action="store_true"); ap.add_argument("--rerender", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    if a.montage:
        montage()
    elif a.rerender:
        rerender(a.only); montage()
    else:
        run_all(); montage()


if __name__ == "__main__":
    main()
