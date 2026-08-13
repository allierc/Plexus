#!/usr/bin/env python
"""rerender_05 -- redraw a 05x run's movie/strip/3d from its trajectory, without re-running physics.

    python rerender_05.py log/okuda_ECM/05e_conserve [--zoom 4]

WHY THIS EXISTS. A render choice should not cost a re-run. The 05 rigs write `traj.npz` with the sheet
positions, its per-face stretch, its connectivity AT EACH KEPT FRAME (the mesh changes size, so the
trajectory cannot be one stacked array), the epithelium, and which sheet nodes carry a plaque -- which
is everything the section needs. Changing the window, the colour scale or the projection is then a
minute rather than ten.

ONE THING IS NOT STORED: the plaque ATTACHMENT POINTS, the barycentric points on the epithelial faces.
They are RECONSTRUCTED here rather than omitted -- each anchored node's own direction is cast against
the epithelial mesh, which is exactly how `plaque_seed` chose the face in the first place. That is
faithful up to the drift of the node's direction since seeding, and the panel says "reconstructed" so
the segment is never mistaken for a stored quantity.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.animation import FFMpegWriter                           # noqa: E402
from matplotlib.colors import ListedColormap                           # noqa: E402
from matplotlib.collections import LineCollection                       # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection   # noqa: E402

import ecm_spec as ES                                                   # noqa: E402

try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

CMAP = ListedColormap(ES.STRESS_COLORS)
EPI_C, PLQ_C, SHEET_C = "#e8dcc0", "#e0452b", "#9ad2ff"
C = np.array([0.5, 0.5, 0.5])
# THE SCALE, from the run's own `units:` declaration (plexus/units.py): 1 box unit = 1171 um, measured
# from the cache (a cell is 8.54e-3 box across; an epithelial cell is ~10 um). Reporting a standoff as
# "2.7e-4" asks the reader to hold a conversion in their head; reporting it as 0.32 um does not.
UM = 1171.0


def unroll(P, c=C):
    """Every point at its TRUE 3D radius, replotted in the section plane. A node off the plane by dy
    projects to sqrt(r^2 - dy^2), short by up to a quarter of the standoff for a slab this thick, and
    the zigzag that produces reads as roughness of the sheet. This removes it."""
    r = np.linalg.norm(P - c, axis=1)
    th = np.arctan2(P[:, 2] - c[2], P[:, 0] - c[0])
    return c[0] + r * np.cos(th), c[2] + r * np.sin(th)


def attachment_points(XE, F_epi, X, nod, c=C):
    """Where each anchored sheet node attaches to the epithelium, cast along the node's own direction.

    This is `plaque_seed`'s own rule -- a plaque binds the face its node points into -- re-applied to
    the stored geometry. Barycentric coordinates of the ray in the basis of the face's three UNIT
    vertices; all three non-negative is the containment test, with no tolerance to tune.
    """
    if len(nod) == 0 or XE is None or F_epi is None:
        return np.zeros((0, 3)), np.zeros(0, bool)
    U = XE - c
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    tri = U[F_epi]                                            # (nf,3,3)
    Minv = np.linalg.inv(np.transpose(tri, (0, 2, 1)))
    u = X[nod] - c
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    bc = np.einsum("fij,nj->nfi", Minv, u)                    # (n,nf,3)
    ok = (bc >= -1e-9).all(-1)
    hit = ok.any(1)
    face = np.argmax(ok, axis=1)
    w = np.take_along_axis(bc, face[:, None, None], axis=1)[:, 0, :]
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-30)
    p = (XE[F_epi[face]] * w[:, :, None]).sum(1)
    return p, hit


def write_traj(kept, F_epi, d, **extra):
    """The trajectory a section can be redrawn from: per-frame positions, stretch, CONNECTIVITY (the
    mesh changes size, so it cannot be one stacked array), the epithelium and the plaque nodes."""
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(
        os.path.join(d, "traj.npz"), frames=np.asarray([k[0] for k in kept]),
        faces_epi=np.asarray(F_epi, dtype=np.int32),
        **{f"pos_{k[0]}": k[1].astype(np.float32) for k in kept},
        **{f"lam_{k[0]}": k[2].astype(np.float32) for k in kept},
        **{f"faces_{k[0]}": k[3].astype(np.int32) for k in kept},
        **{f"epi_{k[0]}": k[4].astype(np.float32) for k in kept},
        **{f"plaque_{k[0]}": k[5].astype(np.int32) for k in kept},
        **extra)


def render_from_traj(d, zoom=2.0, l0=6.0e-4, fps=20, title=None):
    z = np.load(os.path.join(d, "traj.npz"))
    frames = [int(t) for t in z["frames"]]
    has_epi = f"epi_{frames[0]}" in z.files
    F_epi = z["faces_epi"] if "faces_epi" in z.files else None
    name = title or os.path.basename(d.rstrip("/"))
    allL = np.concatenate([z[f"lam_{t}"][::7] for t in frames[::4]])
    s_hi = float(np.percentile(allL[np.isfinite(allL)], 99))
    print(f"[rerender] {name}: {len(frames)} frames, colour to lambda = {s_hi:.3f}, "
          f"zoom x{zoom:g}, epithelium {'stored' if has_epi else 'ABSENT -- section shows the sheet '
          'only'}", flush=True)

    fig = plt.figure(figsize=(11.6, 5.8), facecolor="black")
    wri = FFMpegWriter(fps=fps, metadata={"title": name})
    strip, strip_at = [], set(np.round(np.linspace(0, len(frames) - 1, 8)).astype(int).tolist())
    with wri.saving(fig, os.path.join(d, "movie.mp4"), dpi=100):
        for i, t in enumerate(frames):
            X, L, F = z[f"pos_{t}"], z[f"lam_{t}"], z[f"faces_{t}"]
            XE = z[f"epi_{t}"] if has_epi else None
            nod = z[f"plaque_{t}"] if f"plaque_{t}" in z.files else np.zeros(0, np.int32)
            fig.clf()
            lim = 0.165
            nl = np.zeros(X.shape[0]); cnt = np.zeros(X.shape[0])
            np.add.at(nl, F.reshape(-1), np.repeat(L, 3)); np.add.at(cnt, F.reshape(-1), 1)
            liv = cnt > 0
            nl = nl / np.maximum(cnt, 1)
            ax = fig.add_subplot(1, 2, 1, projection="3d", facecolor="black", computed_zorder=False)
            ax.set_facecolor("black"); ax.axis("off")
            kf = X[F][:, :, 1].mean(1) > C[1]
            tri = Poly3DCollection(X[F][kf], linewidths=0.1, edgecolors=(1, 1, 1, 0.16))
            tri.set_facecolor(CMAP(np.clip((L[kf] - 1.0) / max(s_hi - 1.0, 1e-9), 0, 1)))
            ax.add_collection3d(tri)
            if XE is not None and F_epi is not None:
                ke = XE[F_epi][:, :, 1].mean(1) > C[1]
                ax.add_collection3d(Line3DCollection(XE[F_epi][ke][:, [0, 1, 2, 0]], colors=EPI_C,
                                                     linewidths=0.3, alpha=0.5))
            if XE is not None and F_epi is not None and len(nod):
                PP3, hit3 = attachment_points(XE, F_epi, X, nod)
                vis = hit3 & (X[nod][:, 1] > C[1])          # the half the camera can see
                if vis.any():
                    ax.add_collection3d(Line3DCollection(
                        np.stack([PP3[vis], X[nod][vis]], 1), colors=PLQ_C, linewidths=0.8,
                        alpha=0.9))
                    ax.scatter(X[nod][vis][:, 0], X[nod][vis][:, 1], X[nod][vis][:, 2], s=4,
                               c=PLQ_C, marker="o", linewidths=0, depthshade=False)
            ax.set_xlim(C[0] - lim, C[0] + lim); ax.set_ylim(C[1] - lim, C[1] + lim)
            ax.set_zlim(C[2] - lim, C[2] + lim)
            try:
                ax.set_box_aspect((1, 1, 1), zoom=1.55)
            except TypeError:
                ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=16, azim=-58)
            ax.text2D(0.02, 0.97, f"{name}   frame {t}\n"
                                  f"{F.shape[0]} triangles, {int(liv.sum())} nodes, "
                                  f"{len(nod)} plaques\n"
                                  f"$\\lambda_{{geo}}$ to {s_hi:.2f}",
                      transform=ax.transAxes, color="white", fontsize=10.5, va="top")

            a2 = fig.add_subplot(1, 2, 2, facecolor="black")
            Rc = float(np.linalg.norm((XE if XE is not None else X) - C, axis=1).mean())
            half = zoom * max(22.0 * l0, 0.13 * Rc)
            # THE BAND IS SET BY THE MESH, and this is the second render defect the section had. It
            # does not follow the zoom (scaling it with the window merges the two surfaces into one
            # ribbon), and it must hold ONE ring of nodes: at 3.8 node spacings it was squashing four
            # concentric rings onto a single line and joining them in projected-angle order, which
            # produced a zigzag that reads as roughness of the sheet. The sheet's real radial
            # roughness at the last frame is 0.098 um against a node spacing of 2.99 um -- 3% -- so
            # almost none of that zigzag was the sheet.
            edge = float(np.linalg.norm(X[F[:, 1]] - X[F[:, 0]], axis=1).mean())
            band = 0.60 * edge
            cx, cz = C[0] + Rc, C[2]
            if XE is not None:
                se = (np.abs(XE[:, 1] - C[1]) < band) & (XE[:, 0] > C[0])
                if se.sum() > 2:
                    ex, ez = unroll(XE[se])
                    o = np.argsort(np.arctan2(ez - C[2], ex - C[0]))
                    a2.plot(ex[o], ez[o], "-", color=EPI_C, lw=1.8, zorder=2)
                    a2.scatter(ex, ez, s=20, c=EPI_C, marker="o", linewidths=0, zorder=3)
            sl = liv & (np.abs(X[:, 1] - C[1]) < band) & (X[:, 0] > C[0])
            gap = float(np.linalg.norm(X[liv] - C, axis=1).mean() - Rc)
            if sl.sum() > 2:
                sx, sz = unroll(X[sl])
                o2 = np.argsort(np.arctan2(sz - C[2], sx - C[0]))
                a2.plot(sx[o2], sz[o2], "-", color=SHEET_C, lw=1.2, alpha=0.85, zorder=5)
                a2.scatter(sx, sz, s=10, c=np.clip(nl[sl], 1.0, s_hi), cmap=CMAP, vmin=1.0,
                           vmax=s_hi, marker="o", linewidths=0, zorder=6)
            # the plaques: an anchored node, its RECONSTRUCTED attachment point on the epithelium,
            # and the segment between them -- which is the plaque, drawn as the relation it is
            anc = np.zeros(X.shape[0], bool)
            anc[nod] = True
            am = sl & anc
            n_att = 0
            if am.any():
                nod_here = np.nonzero(am)[0]
                PP, hitm = attachment_points(XE, F_epi, X, nod_here)
                ax_, az_ = unroll(X[nod_here])
                if hitm.any():
                    px, pz = unroll(PP[hitm])
                    a2.add_collection(LineCollection(
                        np.stack([np.stack([px, pz], 1),
                                  np.stack([ax_[hitm], az_[hitm]], 1)], 1),
                        colors=PLQ_C, linewidths=1.2, zorder=7))
                    a2.scatter(px, pz, s=16, c="#ffb0a0", marker="o", linewidths=0, zorder=8)
                    n_att = int(hitm.sum())
                a2.scatter(ax_, az_, s=34, c=PLQ_C, marker="o", linewidths=0, zorder=9)
            a2.plot([cx - 0.88 * half] * 2, [cz - 0.85 * half, cz - 0.85 * half + l0], "-",
                    color="white", lw=2.5, zorder=8)
            a2.text(cx - 0.84 * half, cz - 0.85 * half,
                    f"$\\ell_0$ = {l0*UM:.2f} um", color="white", fontsize=8.5,
                    va="bottom", zorder=8)
            a2.set_xlim(cx - half, cx + half); a2.set_ylim(cz - half, cz + half)
            a2.set_aspect("equal"); a2.axis("off")
            a2.text(0.02, 0.98,
                    f"section, {2*half*UM:.1f} um across (points at their true radius)\n"
                    f"{int(sl.sum())} sheet nodes and {n_att} plaques in this slice "
                    f"(one node ring, {band*UM:.1f} um thick)\n"
                    f"cream = epithelium, blue = sheet, red = a plaque: its membrane end,\n"
                    f"its cell end on the epithelium (reconstructed), and the segment between\n"
                    f"sheet is {'outside' if gap > 0 else 'inside'} the epithelium by "
                    f"{abs(gap)*UM:.3f} um = {abs(gap)/l0:.1f} $\\ell_0$",
                    transform=a2.transAxes, color="white" if gap > 0 else "#ff8080", fontsize=9,
                    va="top")
            wri.grab_frame()
            if i in strip_at:
                strip.append((t, X.copy(), nl.copy(), liv.copy(), F.shape[0], len(nod)))
    fig.savefig(os.path.join(d, "3d.png"), dpi=115, facecolor="black")
    plt.close(fig)

    figs = plt.figure(figsize=(3.0 * len(strip), 3.5), facecolor="black")
    for i, (t, X, nl, liv, nf, npq) in enumerate(strip):
        a = figs.add_subplot(1, len(strip), i + 1, facecolor="black")
        sl = liv & (np.abs(X[:, 1] - 0.5) < 0.004)
        a.scatter(X[sl][:, 0], X[sl][:, 2], s=6, c=np.clip(nl[sl], 1.0, s_hi), cmap=CMAP, vmin=1.0,
                  vmax=s_hi, marker=".", linewidths=0)
        a.set_xlim(0.335, 0.665); a.set_ylim(0.335, 0.665); a.set_aspect("equal"); a.axis("off")
        a.text(0.03, 0.97, f"frame {t}\n{nf} faces, {npq} plaques", transform=a.transAxes,
               color="white", fontsize=10, va="top")
    figs.subplots_adjust(0.004, 0.004, 0.996, 0.996, wspace=0.02)
    figs.savefig(os.path.join(d, "strip.png"), dpi=115, facecolor="black")
    plt.close(figs)
    print(f"[rerender] wrote movie.mp4, 3d.png, strip.png in {d}", flush=True)


def main():
    d = sys.argv[1]
    render_from_traj(
        d,
        zoom=float(sys.argv[sys.argv.index("--zoom") + 1]) if "--zoom" in sys.argv else 1.0,
        l0=float(sys.argv[sys.argv.index("--l0") + 1]) if "--l0" in sys.argv else 6.0e-4,
        fps=int(sys.argv[sys.argv.index("--fps") + 1]) if "--fps" in sys.argv else 20)


if __name__ == "__main__":
    main()
