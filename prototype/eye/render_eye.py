"""render_eye -- the eight-panel movie of the zebrafish oculomotor plant.

    A  anterior view    the cosmetic eye (white sclera, silver iris, black pupil, gold
                        iridophore flecks) with the six muscles drawn as what they are here:
                        their own material points, brightening as they are activated
    B  lateral view     the ovoid globe in its bony cup, the obliques, the trochlea
    C  strain           Green-Lagrange ||E|| on a cut through globe AND muscles
    D  von Mises        the stress field on the same cut
    E  grid momentum    |v| on the shared MLS-MPM background grid -- the medium through
                        which a contracting muscle actually reaches the sclera
    F  activation       the six innervations against time
    G  shortening       each muscle's length as % of its rest length -- the contraction
    H  gaze             horizontal / vertical / torsion against the command

Panels C-E follow the codebase's `grid.mp4` diagnostic (objects / deformation / stress /
grid momentum), extended to two coupled bodies.

Only front-facing points are drawn in A/B: with the far hemisphere culled the eye reads as
a lit solid instead of a cloud, and the gold flecks -- which are what make TORSION visible
-- stay legible.
"""
from __future__ import annotations

import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import to_rgb, Normalize
try:                                    # ffmpeg is not on PATH in this container
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import eye_anatomy as EA

BG, FG = "black", "white"

TISSUE_RGB = {
    "pupil":    (0.02, 0.02, 0.04),      # the large round black pupil of the reference photo
    "iris":     (0.70, 0.77, 0.75),      # silvery iridophore ring
    "fleck":    (0.95, 0.81, 0.22),      # gold flecks -- these reveal TORSION
    "cornea":   (0.86, 0.89, 0.90),
    "sclera":   (0.94, 0.94, 0.90),      # white sclera
    "choroid":  (0.48, 0.38, 0.38),
    "vitreous": (0.28, 0.40, 0.50),
    "lens":     (0.78, 0.87, 0.96),
}
TISSUE_ORDER = ["vitreous", "choroid", "sclera", "cornea", "iris", "fleck", "pupil", "lens"]
PALETTE = np.array([TISSUE_RGB[k] for k in TISSUE_ORDER], np.float32)
MUS_RGB = np.array([to_rgb(m["color"]) for m in EA.MUSCLES], np.float32)


# --------------------------------------------------------------------------- #
#  cameras
# --------------------------------------------------------------------------- #
def camera(view):
    """(right, up, forward); `forward` points FROM the camera INTO the scene, so a surface
    point is front-facing when its outward normal has n . forward < 0."""
    if view == "anterior":     # the front of the RIGHT eye: temporal (+x) on the viewer's LEFT
        return (np.array([-1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., -1.]))
    if view == "lateral":      # from the temporal side: anterior points right
        return (np.array([0., 0., 1.]), np.array([0., 1., 0.]), np.array([-1., 0., 0.]))
    if view == "oblique":      # 3/4 view, used for the cut panels and the grid
        az, el = math.radians(40.0), math.radians(18.0)
        r = np.array([math.cos(az), 0.0, -math.sin(az)])
        f = -np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])
        u = np.cross(f, r)
        return r, u / np.linalg.norm(u), f
    raise ValueError(view)


def proj(P, view):
    r, u, f = camera(view)
    P = np.atleast_2d(np.asarray(P, float))
    return P @ r, P @ u, P @ f


def _label(ax, s):
    ax.text(0.015, 0.975, s, transform=ax.transAxes, color=FG, fontsize=10.5,
            ha="left", va="top", zorder=10)


def _frame(ax, centre, view, span):
    cx, cy, _ = proj(np.atleast_2d(centre), view)
    ax.set_xlim(float(cx[0]) - span, float(cx[0]) + span)
    ax.set_ylim(float(cy[0]) - span, float(cy[0]) + span)
    ax.set_aspect("equal"); ax.set_facecolor(BG); ax.axis("off")


def _cup(ax, view, centre, **kw):
    r, u, _ = camera(view)
    th = np.linspace(0, 2 * np.pi, 200)
    P = centre[None, :] + EA.CUP_RADIUS * (np.cos(th)[:, None] * r[None, :]
                                           + np.sin(th)[:, None] * u[None, :])
    sx, sy, _ = proj(P, view)
    ax.plot(sx, sy, **kw)


# --------------------------------------------------------------------------- #
#  A / B: the cosmetic eye and its muscles, both made of particles
# --------------------------------------------------------------------------- #
def draw_scene(ax, k, cap, view, label, span=0.245, dot=7.0, mdot=3.4):
    centre = cap["centre"][k]
    _, _, f = camera(view)

    # --- the muscles: their own material points, coloured per muscle, lit by activation
    Y = cap["mus_pos"][k]
    par = cap["mus_parent"]
    act = np.clip(cap["act"][k], 0, 1)
    msx, msy, mdep = proj(Y, view)
    # HIDE the muscle points that the globe occludes. A depth-sorted scatter alone cannot do
    # this: the globe is drawn as points, so a muscle behind it shows through the gaps -- most
    # visibly straight across the black pupil. The globe is an ellipsoid, so its orthographic
    # silhouette is exactly the ellipse  p^T (A A^T)^-1 p <= 1  with A = [r.M ; u.M],
    # M = diag(a, a, c); anything inside that ellipse and farther than the centre is behind it.
    r_c, u_c, _ = camera(view)
    M = np.array([EA.A_EQ, EA.A_EQ, EA.C_AX])
    A2 = np.stack([r_c * M, u_c * M])                      # [2,3]
    Sinv = np.linalg.inv(A2 @ A2.T)
    ccx, ccy, cdep = proj(np.atleast_2d(centre), view)
    d = np.stack([msx - ccx[0], msy - ccy[0]], 1)
    inside = np.einsum("ij,jk,ik->i", d, Sinv, d) <= 1.0
    vis = ~(inside & (mdep > cdep[0]))
    Y, par = Y[vis], par[vis]
    msx, msy, mdep = msx[vis], msy[vis], mdep[vis]
    base = MUS_RGB[par]
    glow = (0.34 + 0.66 * act[par])[:, None]

    # --- the globe shell, far hemisphere culled
    X = cap["shell"][k]
    n = X - centre[None, :]
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-9)
    front = (n @ f) < 0.02
    Xf, tf, nf = X[front], cap["tissue"][front], n[front]
    gsx, gsy, gdep = proj(Xf, view)
    lit = np.clip(0.30 + 0.85 * np.abs(nf @ f), 0, 1)
    grgb = np.clip(PALETTE[tf] * lit[:, None], 0, 1)

    _cup(ax, view, centre, color="0.38", lw=1.1, ls=(0, (5, 4)), zorder=1)

    # one depth-sorted pass over BOTH bodies, so muscles correctly pass in front of and
    # behind the globe (this is what makes the anterior view read like the anatomical plate)
    sx = np.concatenate([gsx, msx]); sy = np.concatenate([gsy, msy])
    dep = np.concatenate([gdep, mdep])
    rgb = np.concatenate([grgb, np.clip(base * glow, 0, 1)])
    sz = np.concatenate([np.full(gsx.size, dot), np.full(msx.size, mdot)])
    o = np.argsort(dep)[::-1]
    ax.scatter(sx[o], sy[o], s=sz[o], c=rgb[o], edgecolors="none", linewidths=0, zorder=3)

    mus_s_vis = cap["mus_s"][vis]
    for i, m in enumerate(EA.MUSCLES):                     # key at the proximal end
        sel = par == i
        if not sel.any():
            continue
        j = np.argmin(mus_s_vis[sel])
        tx, ty = msx[sel][j], msy[sel][j]
        ax.text(tx, ty, f" {m['key']}", color=m["color"], fontsize=8.5, va="center",
                alpha=0.55 + 0.45 * float(act[i]), zorder=6)
    _frame(ax, centre, view, span)
    _label(ax, label)


# --------------------------------------------------------------------------- #
#  C / D: the continuum fields on a cut through both bodies
# --------------------------------------------------------------------------- #
def draw_field(ax, k, cap, key, label, vmax, cmap, span=0.185):
    X = np.concatenate([cap["cut_pos"][k], cap["mus_pos"][k]])
    v = np.concatenate([cap["cut_" + key][k], cap["mus_" + key][k]])
    sx, sy, dep = proj(X, "oblique")
    o = np.argsort(dep)[::-1]
    ax.scatter(sx[o], sy[o], s=2.4, c=v[o], cmap=cmap, vmin=0.0, vmax=vmax,
               edgecolors="none", linewidths=0)
    _frame(ax, cap["centre"][k], "oblique", span)
    _label(ax, label)


def draw_grid(ax, k, cap, label, vmax, span=0.215):
    P, v = cap["gpos"][k], cap["gvel"][k]
    if P.size:
        sx, sy, dep = proj(P, "oblique")
        o = np.argsort(dep)[::-1]
        ax.scatter(sx[o], sy[o], s=2.2, c=v[o], cmap="viridis", vmin=0.0, vmax=vmax,
                   edgecolors="none", linewidths=0)
    _frame(ax, cap["centre"][k], "oblique", span)
    _label(ax, label)


# --------------------------------------------------------------------------- #
#  F / G / H: the traces
# --------------------------------------------------------------------------- #
def _style_trace(ax, t, k, xlabel=True):
    ax.axvline(t[k], color="0.8", lw=1.0, alpha=0.85)
    ax.set_xlim(t[0], t[-1])
    if xlabel:
        ax.set_xlabel("sim time", color="0.8", fontsize=8)
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color("0.35")
    ax.tick_params(colors="0.7", labelsize=7)


def _muscle_legend(ax):
    leg = ax.legend(ncol=6, fontsize=7.5, frameon=False, loc="upper center",
                    handlelength=1.0, columnspacing=1.0, bbox_to_anchor=(0.5, 1.15))
    for txt, m in zip(leg.get_texts(), EA.MUSCLES):
        txt.set_color(m["color"])


def draw_act(ax, k, cap, label, dt):
    t = cap["frame"] * dt
    for i, m in enumerate(EA.MUSCLES):
        ax.plot(t, cap["act"][:, i], color=m["color"], lw=1.4, label=m["key"])
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("activation", color="0.8", fontsize=8)
    _muscle_legend(ax)
    _style_trace(ax, t, k)
    _label(ax, label)


def draw_length(ax, k, cap, label, dt):
    t = cap["frame"] * dt
    pct = 100.0 * cap["length"] / cap["rest_length"][None, :]
    for i, m in enumerate(EA.MUSCLES):
        ax.plot(t, pct[:, i], color=m["color"], lw=1.4, label=m["key"])
    ax.axhline(100.0, color="0.45", lw=0.8, ls=":")
    ax.set_ylabel("length (% of rest)", color="0.8", fontsize=8)
    _muscle_legend(ax)
    _style_trace(ax, t, k)
    _label(ax, label)


def draw_gaze(ax, k, cap, label, dt):
    t = cap["frame"] * dt
    names, cols = ["horizontal", "vertical", "torsion"], ["#4da3ff", "#7ee081", "#c58cff"]
    for i in range(3):
        ax.plot(t, cap["target"][:, i], color=cols[i], lw=1.0, ls="--", alpha=0.5)
        ax.plot(t, cap["gaze"][:, i], color=cols[i], lw=1.7, label=names[i])
    ax.set_ylabel("degrees", color="0.8", fontsize=8)
    leg = ax.legend(ncol=3, fontsize=7.5, frameon=False, loc="upper center",
                    handlelength=1.0, columnspacing=1.0, bbox_to_anchor=(0.5, 1.15))
    for txt, cc in zip(leg.get_texts(), cols):
        txt.set_color(cc)
    _style_trace(ax, t, k)
    _label(ax, label)


# --------------------------------------------------------------------------- #
def _figure(ranges=None):
    """The 2x4 grid. Colour bars live on their OWN axes, created once here: a colorbar made
    inside the frame loop is a new axes every frame, and a few hundred of those stacked on
    top of each other is what wrecks the figure."""
    fig = plt.figure(figsize=(19.2, 8.6), facecolor=BG)
    gs = fig.add_gridspec(2, 4, wspace=0.06, hspace=0.16,
                          left=0.010, right=0.990, top=0.950, bottom=0.075)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]
    if ranges is None:
        return fig, axes
    s_hi, v_hi, g_hi = ranges
    for i, (hi, cmap, lab) in zip((2, 3, 4),
                                  ((s_hi, "magma", "‖E‖"), (v_hi, "inferno", "σ_vM"),
                                   (g_hi, "viridis", "|v| grid"))):
        pos = axes[i].get_position()
        cax = fig.add_axes([pos.x1 - 0.011, pos.y0 + 0.26 * pos.height,
                            0.0045, 0.44 * pos.height])
        sm = plt.cm.ScalarMappable(norm=Normalize(0.0, hi), cmap=cmap)
        cb = fig.colorbar(sm, cax=cax)
        cb.ax.tick_params(labelsize=6, colors=FG, length=2, width=0.4, pad=1.5)
        cb.ax.locator_params(nbins=4)
        cb.outline.set_edgecolor("0.45"); cb.outline.set_linewidth(0.4)
        cb.set_label(lab, color=FG, fontsize=7)
    return fig, axes


def _draw_all(axes, k, cap, dt, s_hi, v_hi, g_hi):
    for a in axes:
        a.clear()
    draw_scene(axes[0], k, cap, "anterior", "a   anterior view — right eye, six extraocular muscles")
    draw_scene(axes[1], k, cap, "lateral", "b   lateral view — ovoid globe in the bony cup")
    draw_field(axes[2], k, cap, "strain", "c   green–lagrange strain ‖E‖", s_hi, "magma")
    draw_field(axes[3], k, cap, "vm", "d   von mises stress", v_hi, "inferno")
    draw_grid(axes[4], k, cap, "e   mls-mpm grid momentum — the coupling", g_hi)
    draw_act(axes[5], k, cap, "f   muscle activation", dt)
    draw_length(axes[6], k, cap, "g   muscle length — the contraction", dt)
    draw_gaze(axes[7], k, cap, "h   gaze (solid) vs command (dashed)", dt)


def render(cap, dt, out_mp4, out_strip, fps=30):
    s_hi = float(np.percentile(np.concatenate([cap["cut_strain"].ravel(),
                                               cap["mus_strain"].ravel()]), 99.5))
    v_hi = float(np.percentile(np.concatenate([cap["cut_vm"].ravel(),
                                               cap["mus_vm"].ravel()]), 99.5))
    gv = np.concatenate([g for g in cap["gvel"] if g.size]) if len(cap["gvel"]) else np.array([1.0])
    g_hi = float(np.percentile(gv, 99.0)) if gv.size else 1.0
    n = len(cap["frame"])

    fig, axes = _figure((s_hi, v_hi, g_hi))
    writer = FFMpegWriter(fps=fps, bitrate=8000,
                          metadata={"title": "zebrafish oculomotor plant (Plexus2)"})
    with writer.saving(fig, out_mp4, dpi=100):
        for k in range(n):
            _draw_all(axes, k, cap, dt, s_hi, v_hi, g_hi)
            writer.grab_frame(facecolor=BG)
            if k % 40 == 0:
                print(f"    [render] {k}/{n}", flush=True)
    plt.close(fig)

    ks = [int(x) for x in np.linspace(0, n - 1, 5)]
    fig2 = plt.figure(figsize=(20, 4.4), facecolor=BG)
    for j, k in enumerate(ks):
        ax = fig2.add_subplot(1, 5, j + 1)
        g = cap["gaze"][k]
        draw_scene(ax, k, cap, "anterior",
                   f"frame {int(cap['frame'][k])}   h {g[0]:+.1f}°  v {g[1]:+.1f}°  t {g[2]:+.1f}°")
    fig2.subplots_adjust(left=0.004, right=0.996, top=0.97, bottom=0.02, wspace=0.02)
    fig2.savefig(out_strip, dpi=110, facecolor=BG)
    plt.close(fig2)
