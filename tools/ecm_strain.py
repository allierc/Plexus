"""Strain in the matrix a growing spheroid pushes into.

The trajectory stores positions and nothing else, so strain is computed here from the displacement
field rather than read off. The loading is spherically symmetric about the spheroid, which is what
makes that tractable: write u_r(r) for the radial displacement of a particle whose UNDEFORMED radius
was r, measured from the spheroid's centre. Then the two strains that matter follow from it,

    eps_rr = d u_r / d r        radial, the stretch along the line to the centre
    eps_tt = u_r / r            hoop, the stretch around it

and they have opposite signs here: material driven outward is pulled apart circumferentially while
being squeezed radially against the matrix further out, which is the compaction a spheroid makes and
a uniform pressure does not.

WHY THE EXPONENT IS THE INTERESTING NUMBER. An incompressible shell conserves volume through every
spherical surface, so 4 pi r^2 u_r is constant and u_r falls as r^-2. That is the closed form to
measure against, and the sign of the departure says which way the matrix differs: STEEPER than -2 is
displacement dying out sooner than a continuum would carry it, SHALLOWER is displacement reaching
further. Panel C fits it rather than asserting it; on the first run measured here the fit came out at
-1.25, shallower, which is what a fibrous network does and a continuum cannot -- a fibre carries
tension along its whole length, so load arrives at material the strain field alone would not reach.

    python tools/ecm_strain.py --run graphs_data/tissue/spheroid_ecm
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def radial_profile(P0, Pf, centre, n_bins=44, r_lo=None, r_hi=None):
    """Mean radial displacement per shell of undeformed radius, and the bin centres.

    Binned by the UNDEFORMED radius, because eps_rr is a derivative with respect to the material
    coordinate; binning by the current radius would differentiate along a moving axis and mix the
    deformation into the coordinate it is measured against.
    """
    r0 = np.linalg.norm(P0 - centre, axis=1)
    u = np.einsum('ij,ij->i', Pf - P0, (P0 - centre) / np.maximum(r0, 1e-12)[:, None])
    r_lo = r0.min() if r_lo is None else r_lo
    r_hi = r0.max() if r_hi is None else r_hi
    edges = np.linspace(r_lo, r_hi, n_bins + 1)
    which = np.digitize(r0, edges) - 1
    ok = (which >= 0) & (which < n_bins)
    ur = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = ok & (which == b)
        cnt[b] = int(m.sum())
        if cnt[b] > 30:
            ur[b] = float(u[m].mean())
    return 0.5 * (edges[:-1] + edges[1:]), ur, cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="graphs_data/tissue/spheroid_ecm")
    ap.add_argument("--centre", type=float, nargs=3, default=[0.5, 0.5, 0.5])
    ap.add_argument("--out", default=None)
    ap.add_argument("--movie", action="store_true",
                    help="also write ecm_strain.mp4: the hoop-strain slab frame by frame, beside "
                         "the radial profile as it builds. THE COLOUR SCALE IS FIXED FOR THE WHOLE "
                         "CLIP, from the last frame's 99th percentile -- a per-frame range would "
                         "renormalise every frame and the strain would appear to arrive instantly "
                         "and then never grow.")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(os.path.join(args.run, "trajectory.npz"))
    P, occ = d["mpm_particle__pos"], d["mpm_particle__occ"]
    V, Vocc = d["vertex__pos"], d["vertex__occ"]
    c = np.array(args.centre, dtype=np.float64)
    nF = P.shape[0]
    keep = occ[0] & occ[-1]
    P0 = P[0][keep].astype(np.float64)
    r0 = np.linalg.norm(P0 - c, axis=1)

    frames = [int(round(f)) for f in np.linspace(0, nF - 1, 5)][1:]
    fig = plt.figure(figsize=(13.5, 10.5), facecolor="black")
    gs = fig.add_gridspec(2, 2, hspace=0.26, wspace=0.24)
    W = dict(color="white", fontsize=11)

    def style(ax):
        ax.set_facecolor("black")
        for sp in ax.spines.values():
            sp.set_color("#888888")
        ax.tick_params(colors="#cccccc", labelsize=9)
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        return ax

    cmap = plt.get_cmap("plasma")
    cols = [cmap(0.15 + 0.72 * i / max(len(frames) - 1, 1)) for i in range(len(frames))]

    # A -- the radial displacement profile, one curve a frame
    axA = style(fig.add_subplot(gs[0, 0]))
    prof = {}
    for col, f in zip(cols, frames):
        rb, ur, cnt = radial_profile(P0, P[f][keep].astype(np.float64), c)
        prof[f] = (rb, ur)
        axA.plot(rb * 1000, ur * 1000, color=col, lw=1.8, label=f"frame {f}")
    axA.set_xlabel("undeformed radius  r  (um)")
    axA.set_ylabel("radial displacement  u_r  (um)")
    axA.axhline(0.0, color="#555555", lw=0.8)
    leg = axA.legend(frameon=False, fontsize=9, labelcolor="white")
    axA.text(0.02, 0.96, "A", transform=axA.transAxes, va="top", fontweight="bold", **W)

    # B -- the two strains at the last frame. Opposite signs is the point.
    axB = style(fig.add_subplot(gs[0, 1]))
    rb, ur = prof[frames[-1]]
    good = np.isfinite(ur)
    e_tt = np.where(good, ur / np.maximum(rb, 1e-12), np.nan)
    e_rr = np.full_like(ur, np.nan)
    e_rr[1:-1] = (ur[2:] - ur[:-2]) / (rb[2:] - rb[:-2])
    axB.plot(rb * 1000, e_rr, color="#e05a5a", lw=1.9, label="radial  eps_rr")
    axB.plot(rb * 1000, e_tt, color="#4aa3e0", lw=1.9, label="hoop  eps_tt")
    axB.axhline(0.0, color="#555555", lw=0.8)
    axB.set_xlabel("undeformed radius  r  (um)")
    axB.set_ylabel("strain  (dimensionless)")
    axB.legend(frameon=False, fontsize=9, labelcolor="white")
    axB.text(0.02, 0.96, "B", transform=axB.transAxes, va="top", fontweight="bold", **W)

    # C -- the falloff exponent against the incompressible shell's own -2
    axC = style(fig.add_subplot(gs[1, 0]))
    m = good & (ur > 0) & (rb > 0)
    slope = np.nan
    if m.sum() > 6:
        slope, icept = np.polyfit(np.log(rb[m]), np.log(ur[m]), 1)
        axC.plot(np.log(rb[m]), np.log(ur[m]), "o", ms=3.4, color="#ffc242",
                 label="measured")
        xs = np.linspace(np.log(rb[m]).min(), np.log(rb[m]).max(), 8)
        axC.plot(xs, slope * xs + icept, color="#ffc242", lw=1.4,
                 label=f"fit  slope {slope:+.2f}")
        axC.plot(xs, -2.0 * xs + (np.log(ur[m])[0] + 2.0 * np.log(rb[m])[0]),
                 color="#8fd18f", lw=1.4, ls="--",
                 label="incompressible shell  slope -2")
    axC.set_xlabel("log  undeformed radius")
    axC.set_ylabel("log  radial displacement")
    axC.legend(frameon=False, fontsize=9, labelcolor="white")
    axC.text(0.02, 0.96, "C", transform=axC.transAxes, va="top", fontweight="bold", **W)

    # D -- where the strain sits, as a slab through the centre
    axD = style(fig.add_subplot(gs[1, 1]))
    Pf = P[frames[-1]][keep].astype(np.float64)
    u_vec = Pf - P0
    ur_p = np.einsum('ij,ij->i', u_vec, (P0 - c) / np.maximum(r0, 1e-12)[:, None])
    e_tt_p = ur_p / np.maximum(r0, 1e-12)
    slab = np.abs(P0[:, 2] - c[2]) < 0.012
    sc = axD.scatter((P0[slab, 0] - c[0]) * 1000, (P0[slab, 1] - c[1]) * 1000,
                     c=e_tt_p[slab], s=1.1, cmap="magma",
                     vmin=0.0, vmax=float(np.nanpercentile(e_tt_p[slab], 99)))
    rv = np.linalg.norm(V[frames[-1]][Vocc[frames[-1]]] - c, axis=1).max()
    th = np.linspace(0, 2 * np.pi, 256)
    axD.plot(rv * 1000 * np.cos(th), rv * 1000 * np.sin(th), color="white", lw=1.1)
    axD.set_aspect("equal")
    axD.set_xlabel("x  (um)"); axD.set_ylabel("y  (um)")
    cb = fig.colorbar(sc, ax=axD, fraction=0.046, pad=0.02)
    cb.set_label("hoop strain  eps_tt", color="white", fontsize=10)
    cb.ax.tick_params(colors="#cccccc", labelsize=8)
    cb.outline.set_edgecolor("#888888")
    axD.text(0.02, 0.96, "D", transform=axD.transAxes, va="top", fontweight="bold", **W)

    out = args.out or os.path.join(args.run, "ecm_strain.png")
    fig.savefig(out, dpi=150, facecolor="black", bbox_inches="tight")
    print(f"  spheroid surface radius at frame {frames[-1]}: {rv*1000:.1f} um")
    print(f"  hoop strain: max {np.nanmax(e_tt):+.4f}, at r = "
          f"{rb[int(np.nanargmax(e_tt))]*1000:.1f} um")
    print(f"  radial strain: min {np.nanmin(e_rr):+.4f}, max {np.nanmax(e_rr):+.4f}")
    print(f"  falloff exponent {slope:+.2f} against -2 for an incompressible shell")
    print(f"  -> {out}")
    if args.movie:
        write_movie(P, occ, V, Vocc, keep, c, args, os.path.join(args.run, "ecm_strain.mp4"))


def write_movie(P, occ, V, Vocc, keep, c, args, out):
    import imageio.v3 as iio
    import matplotlib.pyplot as plt

    P0 = P[0][keep].astype(np.float64)
    r0 = np.linalg.norm(P0 - c, axis=1)
    slab = np.abs(P0[:, 2] - c[2]) < 0.012
    # ONE RANGE FOR THE WHOLE CLIP, taken from the last frame.
    uN = np.einsum('ij,ij->i', P[-1][keep].astype(np.float64) - P0,
                   (P0 - c) / np.maximum(r0, 1e-12)[:, None])
    vmax = float(np.nanpercentile((uN / np.maximum(r0, 1e-12))[slab], 99))
    rb_N, ur_N, _ = radial_profile(P0, P[-1][keep].astype(np.float64), c)
    y_hi = float(np.nanmax(ur_N) * 1000 * 1.08)

    frames = []
    for f in range(P.shape[0]):
        Pf = P[f][keep].astype(np.float64)
        ur = np.einsum('ij,ij->i', Pf - P0, (P0 - c) / np.maximum(r0, 1e-12)[:, None])
        e_tt = ur / np.maximum(r0, 1e-12)
        fig = plt.figure(figsize=(11.4, 5.3), facecolor="black")
        ax0 = fig.add_subplot(1, 2, 1); ax1 = fig.add_subplot(1, 2, 2)
        for ax in (ax0, ax1):
            ax.set_facecolor("black")
            for sp in ax.spines.values():
                sp.set_color("#888888")
            ax.tick_params(colors="#cccccc", labelsize=8)
            ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        sc = ax0.scatter((P0[slab, 0] - c[0]) * 1000, (P0[slab, 1] - c[1]) * 1000,
                         c=e_tt[slab], s=1.1, cmap="magma", vmin=0.0, vmax=vmax)
        vf = V[f][Vocc[f]]
        rv = np.linalg.norm(vf - c, axis=1).max()
        th = np.linspace(0, 2 * np.pi, 256)
        ax0.plot(rv * 1000 * np.cos(th), rv * 1000 * np.sin(th), color="white", lw=1.1)
        ax0.set_aspect("equal"); ax0.set_xlim(-420, 420); ax0.set_ylim(-420, 420)
        ax0.set_xlabel("x  (um)"); ax0.set_ylabel("y  (um)")
        ax0.text(0.02, 0.97, "A", transform=ax0.transAxes, va="top", color="white",
                 fontsize=11, fontweight="bold")
        ax0.text(0.02, 0.03, f"frame {f}/{P.shape[0]-1}   spheroid r = {rv*1000:.0f} um",
                 transform=ax0.transAxes, color="#cccccc", fontsize=9)
        cb = fig.colorbar(sc, ax=ax0, fraction=0.046, pad=0.02)
        cb.set_label("hoop strain  eps_tt", color="white", fontsize=9)
        cb.ax.tick_params(colors="#cccccc", labelsize=8); cb.outline.set_edgecolor("#888888")

        rb, urp, _ = radial_profile(P0, Pf, c)
        ax1.plot(rb * 1000, urp * 1000, color="#ffc242", lw=1.9)
        ax1.axvline(rv * 1000, color="white", lw=1.0, ls="--")
        ax1.set_xlabel("undeformed radius  r  (um)")
        ax1.set_ylabel("radial displacement  u_r  (um)")
        ax1.set_ylim(-0.02 * y_hi, y_hi); ax1.axhline(0.0, color="#555555", lw=0.8)
        ax1.text(0.02, 0.97, "B", transform=ax1.transAxes, va="top", color="white",
                 fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(img)
        plt.close(fig)
    iio.imwrite(out, frames, fps=args.fps, codec="libx264", macro_block_size=None)
    print(f"  -> {out}  ({len(frames)} frames, {args.fps} fps)")


if __name__ == "__main__":
    main()
