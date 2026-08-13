#!/usr/bin/env python
"""bm_section -- the basement membrane, its integrins and the epithelium, in zoomed section.

    python bm_section.py 121_direct_r25 124_punctate20 ...        -> <run>/section.mp4

WHAT THIS DRAWS THAT THE RUN'S OWN MOVIE DOES NOT. `movie.mp4` shows the membrane as a rim of dots at
whole-tissue framing, where the tether that holds it is a sub-pixel quantity. The question these runs
are actually about -- WHERE the sheet sits relative to the cells, and what the adhesion is doing to put
it there -- is only legible with the anchor drawn. So each anchored particle gets its integrin drawn as
a zig-zag between the particle and the point on the epithelial surface it is tethered to, and the panel
is zoomed to a few cell diameters.

THE SPRINGS ARE RECONSTRUCTED, NOT RECORDED, and that is safe for exactly one reason: `integrin_adhesion`
is a pure function of the seeded direction, the recorded surface map and the current position, all three
of which are in the run's artefacts. The reconstruction here is the operator's own arithmetic
(membrane_ops.IntegrinAdhesion.forward) applied to `traj.npz`, including the punctate mask (same
generator, same seed 0), the `tau_adh` creep and the `detach` rupture test. Where it can differ is
within a frame: the operator sees the position at its slot in the schedule and this sees the position
recorded at the end of the frame. That is a sub-frame difference in a quantity whose per-frame change is
~3e-5 box units, and it is not visible at any zoom.

DRAWN THROUGH `ecm_render.draw_cross`, the same routine the run's own section panel uses, so the cells,
the stroma and the membrane are the same picture at the same scale -- the springs are the only thing
this file adds. A separate cell-drawing path would be a second opinion about where the surface is, which
is the one thing this movie must not have.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "discovery_okuda", "ops")):
    sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

import ecm_render as RD
import ecm_spec as ES

LOG = os.path.join(_ROOT, "log", "okuda_ECM")
# THE SPEC RECORDS THE CLUSTER'S PATH. Same file, two mounts; the run is not re-runnable from here
# either way, and this only ever reads the tissue cache.
REMOTE = "/groups/saalfeld/home/allierc/Graph/Plexus/log"
LOCAL = os.path.join(_ROOT, "log")

# CYAN, BECAUSE NOTHING ELSE IN THIS PANEL IS. The stroma owns inferno, the membrane green-to-amber, the
# cells white. A ruptured integrin goes red and is drawn as the straight line it no longer holds.
SPRING = "#4aa3ff"
# ORANGE FOR THE MATERIAL FIBRE, cyan for the spring, and the difference is not decoration: one is a
# force with a target reconstructed from the operator's arithmetic, the other is particles read off the
# trajectory. `flat_test` and `flat_mpm` already draw fibres orange against a blue sheet, so the two
# rigs and the spheroid now say the same thing with the same colours.
FIBRE = "#f0913a"
BROKEN = "#e0452b"


def _local(p):
    return p.replace(REMOTE, LOCAL)


def _anchor_dirs(P, park=None):
    """`u0` as the operator freezes it: at frame 0, or at the frame a reserve particle is secreted.

    A PARKED PARTICLE HAS NO DIRECTION. The reserve sits at `membrane_park` -- (-0.25, -0.25, -0.25),
    which is 1.30 box units from the centre, outside the box -- so freezing its frame-0 direction would
    anchor it to a point on the far side of the tissue. The operator freezes at birth instead (the
    `born` branch of IntegrinAdhesion.forward), and this reproduces that from the first frame at which
    the particle is no longer at the park point.
    """
    c = np.array([0.5, 0.5, 0.5])
    T, N, _ = P.shape
    born = np.zeros(N, np.int32)
    if park is not None:
        d = np.linalg.norm(P - np.asarray(park)[None, None, :], axis=2) > 1e-6
        born = np.where(d.any(0), d.argmax(0), 0).astype(np.int32)
    X = P[born, np.arange(N)] - c
    u0 = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
    return u0, born


def _radius(smap, u, t):
    """R(u, t) off the recorded map, indexed the way the operator indexes it (nearest cell, not bilinear)."""
    nth, nph = smap.shape[1], smap.shape[2]
    th = np.arccos(np.clip(u[:, 2], -1, 1))
    ph = np.arctan2(u[:, 1], u[:, 0]) % (2 * math.pi)
    ti = min(smap.shape[0] - 1, max(0, t))
    return smap[ti,
                np.clip((th / math.pi * nth).astype(int), 0, nth - 1),
                np.clip((ph / (2 * math.pi) * nph).astype(int), 0, nph - 1)]


def _punctate_mask(n, fraction):
    """The operator's own draw: torch, CPU generator, seed 0. Reproduced rather than approximated --
    a different 20% is a different picture of which patches sag."""
    if fraction >= 1.0:
        return np.ones(n, bool)
    import torch
    g = torch.Generator(device="cpu").manual_seed(0)
    return (torch.rand(n, generator=g) < fraction).numpy()


def _zigzag(A, B, teeth=5, amp=0.18):
    """One polyline per spring: `teeth` half-periods of a triangle wave along A->B.

    `amp` is a fraction of the spring's own CURRENT length, so a stretched integrin draws as a stretched
    spring -- the zig-zag flattens as it is pulled, which is the only way the drawing carries the load.
    """
    d = B - A
    L = np.linalg.norm(d, axis=1, keepdims=True).clip(1e-9)
    t = d / L
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    s = np.linspace(0.0, 1.0, 2 * teeth + 3)[None, :, None]
    w = np.zeros(2 * teeth + 3)
    w[1:-1] = np.where(np.arange(2 * teeth + 1) % 2 == 0, 1.0, -1.0)
    w[1] *= 0.5
    w[-2] *= 0.5
    pts = A[:, None, :] + t[:, None, :] * (s * L[:, None, :]) \
        + n[:, None, :] * (w[None, :, None] * amp * L[:, None, :])
    return pts


def render(run, frames=150, zoom_half=2.2, slab_cells=0.35, teeth=5, fps=15, max_springs=70,
           out=None):
    d = os.path.join(LOG, run)
    spec = yaml.safe_load(open(os.path.join(d, "spec_run.yaml")))
    ops = {o["op"]: o for o in spec["operators"] if isinstance(o, dict) and "op" in o}
    ia = ops.get("integrin_adhesion")
    # TWO KINDS OF INTEGRIN, ONE PANEL. The spring version is reconstructed from the operator's own
    # arithmetic; the MPM version is drawn from the particles themselves, which are in `traj.npz`. A run
    # has one or the other, and the label says which so the two are never read as the same picture.
    seedop, trackop = ops.get("integrin_seed"), ops.get("integrin_track")
    mpm_fibre = ia is None and seedop is not None
    if ia is None and not mpm_fibre:
        sys.exit(f"{run}: no integrin_adhesion and no integrin_seed -- there are no fibres to draw")
    ia = ia or dict(trackop or seedop, offset=float(seedop.get("length", 0.004)), k=0.0,
                    fraction=1.0, detach=0.0, tau_adh=0.0)
    surf = _local(str(ia["surface"]))
    scale = float(ia.get("scale", 1.0))
    offset = float(ia.get("offset", 0.004))
    frac = float(ia.get("fraction", 1.0))
    detach = float(ia.get("detach", 0.0))
    tau_adh = float(ia.get("tau_adh", 0.0))
    seed_bm = ops.get("bm_seed", {})
    park = seed_bm.get("park", None)

    z = np.load(surf)
    smap = np.asarray(z["smap"], np.float32)            # TISSUE units, as the renderer wants it
    Tis = RD.load_tissue(surf, scale)
    tr = np.load(os.path.join(d, "traj.npz"))
    IP = np.asarray(tr["ipos"]) if "ipos" in tr.files else None
    ipf = (np.asarray(tr["ipos_frames"], int) if "ipos_frames" in tr.files
           else (np.arange(IP.shape[0]) if IP is not None else None))
    n_layer = int(seedop.get("layers", 3)) if seedop is not None else 0
    pos, band_all = np.asarray(tr["pos"]), np.asarray(tr["stress"])
    P = np.asarray(tr["mpos"])
    # A RUN WHOSE STRAIN HISTORY WAS LOST STILL HAS A GEOMETRY. `mstrain` is missing from 149-156:
    # the `ipos` save was added one indentation level out and swallowed the two saves below it, so the
    # strain was recorded during the run (153's log says "403 strain frames") and then not written.
    # Fixed in run_ecm, but those trajectories cannot be recovered -- F is not stored -- so the sheet is
    # drawn in one colour and the label says so, rather than a zero field being read as no strain.
    no_strain = "mstrain" not in tr.files
    mst = (np.zeros((1, P.shape[1]), np.float32) if no_strain
           else np.asarray(tr["mstrain"], np.float32))
    alive_end = (np.asarray(tr["malive"], bool) if "malive" in tr.files
                 else np.ones(np.asarray(tr["mpos"]).shape[1], bool))
    c = np.array([0.5, 0.5, 0.5])
    # WHICH SIMULATION FRAME EACH RECORDED MEMBRANE FRAME IS. `mpos` is strided when the set is big
    # enough to blow the 400 MB budget -- 128 keeps 202 of 403 -- while `pos`, `stress` and `mstrain`
    # keep every frame. Indexing all of them by the same integer pairs a membrane at frame 402 with an
    # epithelium at 201, which is how this first read 128's standoff as +0.036 outside when it is
    # -0.098 inside. `mpos_frames` is written for exactly this and is the only honest index.
    mf = (np.asarray(tr["mpos_frames"], int) if "mpos_frames" in tr.files
          else np.arange(P.shape[0], dtype=int))
    T = int(mf[-1]) + 1

    u0, born_i = _anchor_dirs(P, park=park)
    born = mf[np.clip(born_i, 0, len(mf) - 1)]          # birth in SIMULATION frames
    anchored = _punctate_mask(P.shape[1], frac)
    cmap = ListedColormap(ES.STRESS_COLORS)
    # FIXED OVER THE RUN, like every other colour scale here: a per-frame normalisation makes a sheet
    # whose strain is climbing look like one at equilibrium.
    msc = float(np.percentile(mst[::10][mst[::10] > 0], 99)) if (mst > 0).any() else 1.0
    L3 = min(0.5 / scale, Tis["Lbox"] * 1.60)
    L2 = L3 * 1.15
    # ONE CELL DEEP, not the section's 3.07 tissue units. The whole-tissue panel wants a thick slab so
    # the matrix reads as a field; a zoom wants a thin one, or springs from seven cell layers overlay
    # into a hedge and no individual integrin is legible.
    r_cell = float(np.sqrt(4 * math.pi * float(Tis["r_apical"][-1]) ** 2
                           / max(float(Tis["n_cells"][-1]), 1.0)))
    slab = slab_cells * r_cell

    # PAIRED BY NEAREST FRAME, not by equal index. The mesh is cached on its own 200-frame grid and the
    # membrane on `mf`; the two grids need not contain the same integers, and requiring a match silently
    # drops every frame where they disagree. Nearest is at worst one frame out of 402 -- below the
    # tissue's own change between frames -- and it never drops a frame.
    meshes = [(t, m) for t, m in Tis["meshes"] if t < T]
    mesh_t = np.asarray([t for t, _ in meshes])
    pick = np.unique(np.round(np.linspace(0, len(mf) - 1, min(frames, len(mf)))).astype(int))
    draw_at = {int(i): meshes[int(np.argmin(np.abs(mesh_t - int(mf[i]))))][1] for i in pick}

    from matplotlib.animation import FFMpegWriter
    try:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    from run_tyssue_round import _screen_basis
    dvec, uvec, vvec = _screen_basis()

    fig = plt.figure(figsize=(11.0, 5.6), facecolor="black")
    axw = fig.add_subplot(1, 2, 1, facecolor="black")
    axz = fig.add_subplot(1, 2, 2, facecolor="black")
    fig.subplots_adjust(0, 0, 1, 1, wspace=0.01)
    out = out or os.path.join(d, "section.mp4")
    wri = FFMpegWriter(fps=fps, metadata={"title": f"{run} section"})

    # THE CREEP IS A RUNNING STATE, so it has to be integrated over every frame, not sampled at the
    # frames the movie draws. `bound` likewise: an integrin that ruptured at frame 200 is gone at 300
    # whether or not 200 was drawn.
    u = u0.copy()
    bound = np.ones(P.shape[1], bool)
    stat = []
    with wri.saving(fig, out, dpi=100):
        for i, t in enumerate(mf):
            t = int(t)
            R = _radius(smap, u, t)
            A_box = c + u * (R * scale + offset)[:, None]
            delta = A_box - P[i]
            if detach > 0:
                bound &= np.linalg.norm(delta, axis=1) < detach
            if tau_adh > 0:
                # PER SIMULATION FRAME, not per recorded one: `tau_adh` is a number of frames in the
                # operator, so replaying it on a 1-in-2 record has to take two steps' worth of creep.
                dt_f = float(mf[i] - mf[i - 1]) if i else 1.0
                cur = P[i] - c
                cur /= np.linalg.norm(cur, axis=1, keepdims=True).clip(1e-12)
                u = u + (cur - u) * min(1.0, dt_f / tau_adh)
                u /= np.linalg.norm(u, axis=1, keepdims=True).clip(1e-12)
            live = alive_end & (born <= t) if park is not None else alive_end
            stand = (np.linalg.norm(P[i] - c, axis=1) - R * scale)[live]
            stat.append((t, float(stand.mean()), int((~bound & anchored & live).sum())))
            if i not in draw_at:
                continue
            mt = draw_at[i]
            vp = mt["pos"]
            q = (pos[t] - c) / scale
            qm = (P[i] - c) / scale
            sm = np.clip(mst[min(t, mst.shape[0] - 1)] / max(msc, 1e-12), 0, 1)
            mem = (qm[live], sm[live])
            for ax, half in ((axw, None), (axz, zoom_half)):
                ax.clear(); ax.set_facecolor("black")
                # THE SAME SLAB AS THE WHOLE PANEL, IN BOTH. The thin slab was introduced so the
                # springs would not overlay into a hedge, and the spring selection has its own copy of
                # it below -- but applied to `draw_cross` it also thinned the STROMA, which is a 3D
                # cloud: at +/-0.29 tissue units almost nothing of it falls in the plane and the zoom
                # showed the sheet floating in black. The matrix is a field and needs the field's slab.
                RD.draw_cross(ax, mt, vp, q, band_all[t], cmap, L2, np.eye(3)[2], 0.055 / scale,
                              dot_scale=5.0 if half else 1.0, mem=mem, zoom_half=half)
            # THE WINDOW HOLDS THE SURFACE AND THE SHEET, whatever the distance between them. Centring
            # on the sheet's mean radius fails on the two runs that most need the panel: 125 sags 6.6
            # tissue units away from the surface, and 128's secreted half sits in a second shell deep in
            # the lumen, so the mean lands in empty space BETWEEN two populations and the panel shows
            # neither. The span is measured every frame and the zoom opens only as far as it must.
            rs = float(smap[min(smap.shape[0] - 1, t)].mean())
            r_live = np.linalg.norm(qm[live], axis=1)
            rlo, rhi = np.percentile(r_live, [2, 98])
            lo, hi = min(rs, float(rlo)), max(rs, float(rhi))
            half = float(np.clip(0.6 * (hi - lo), zoom_half, 8.0))
            ctr = 0.5 * (lo + hi)
            axz.set_xlim(ctr - half, ctr + half)
            axz.set_ylim(-half, half)
            # ---- the springs, in the zoom panel only: at whole-tissue framing they are sub-pixel
            #
            # DRAWN FROM THE SURFACE TO THE PARTICLE, not from the particle to its rest position. The
            # operator's target sits at R + offset, i.e. 0.22 tissue units OUTSIDE the epithelium, and
            # drawing to it puts the integrin's cell-side end floating in the matrix -- which is where
            # nothing attaches. The cell end of an integrin is IN the basal membrane, at R. The rest
            # position is a length along the fibre, not a place to hang it from, so the drawing starts
            # at R and the fibre's own length shows as how far the sheet sits from that end.
            if mpm_fibre:
                # THE FIBRE AS IT IS, not as it was asked to be: inner particle to outer particle, the
                # two ends of the material column. Nothing is reconstructed here.
                j = int(np.argmin(np.abs(ipf - t)))
                Q = (IP[j] - c) / scale
                nf = Q.shape[0] // max(n_layer, 1)
                qm_i, qa_i = Q[:nf], Q[(n_layer - 1) * nf: n_layer * nf]
                selF = np.abs(qm_i @ dvec) < slab
                A2 = np.stack([qa_i[selF] @ uvec, qa_i[selF] @ vvec], 1)
                B2 = np.stack([qm_i[selF] @ uvec, qm_i[selF] @ vvec], 1)
                lo, hi = axz.get_xlim(), axz.get_ylim()
                # lo IS THE X LIMITS AND hi THE Y LIMITS. Comparing an x against hi[1] asks whether
                # a radius of 16 is below 2.2, which is never, so the first version of this drew
                # "0 of 0" fibres on a frame that had four thousand of them.
                w2 = ((np.minimum(A2[:, 0], B2[:, 0]) < lo[1] + 1)
                      & (np.maximum(A2[:, 0], B2[:, 0]) > lo[0] - 1)
                      & (np.minimum(A2[:, 1], B2[:, 1]) < hi[1] + 1)
                      & (np.maximum(A2[:, 1], B2[:, 1]) > hi[0] - 1))
                Aq, Bq = B2[w2], A2[w2]
                n_win = len(Aq)
                if n_win > max_springs:
                    Aq, Bq = Aq[::int(np.ceil(n_win / max_springs))], \
                        Bq[::int(np.ceil(n_win / max_springs))]
                # THE PARTICLES, NOT A CARTOON OF THEM. A zig-zag is the right drawing for the
                # spring version, where the operator's force IS a line between two points and there is
                # nothing else to show. Here the fibre is material: its state is where its particles
                # are, and a spring glyph drawn over them would assert a connection MPM does not have
                # -- which is precisely what 142 disproved (inner ends at 0.2969, outer at 0.0875, no
                # coupling whatever). So the fibre is drawn as what it is.
                selA = np.abs(Q @ dvec) < slab
                fx_, fy_ = Q[selA] @ uvec, Q[selA] @ vvec
                inwin = ((fx_ > lo[0]) & (fx_ < lo[1]) & (fy_ > hi[0]) & (fy_ < hi[1]))
                n_win = int(inwin.sum())
                if n_win:
                    axz.scatter(fx_[inwin], fy_[inwin], s=16.0, c=FIBRE, marker=".",
                                linewidths=0, zorder=6, alpha=0.95)
                Aq = fx_[inwin]
                lab = (f"{run}\nframe {t}   {int(mt['nF'])} cells\n"
                       f"standoff {stat[-1][1]:+.4f} box units (0 = on the surface)\n"
                       f"MPM fibre particles (orange)\n"
                       f"rest length {offset:g} box = {offset/scale:.2f} tissue units\n"
                       f"{n_win} in window, +/-{half:.1f} tissue units")
                bb = dict(facecolor="black", alpha=0.55, edgecolor="none", pad=4.0)
                axz.text(0.02, 0.98, lab, transform=axz.transAxes, color="white", fontsize=11,
                         va="top", ha="left", linespacing=1.5, zorder=8, bbox=bb)
                axw.text(0.02, 0.98, "whole section", transform=axw.transAxes, color="white",
                         fontsize=11, va="top", zorder=8, bbox=bb)
                wri.grab_frame()
                continue
            qa = u * (R[:, None])
            sel = anchored & live & (np.abs(qm @ dvec) < slab)
            x0, y0 = qm[sel] @ uvec, qm[sel] @ vvec
            x1, y1 = qa[sel] @ uvec, qa[sel] @ vvec
            lo, hi = axz.get_xlim(), axz.get_ylim()
            win = (np.minimum(x0, x1) < lo[1] + 1) & (np.maximum(x0, x1) > lo[0] - 1) & \
                  (np.minimum(y0, y1) < hi[1] + 1) & (np.maximum(y0, y1) > hi[0] - 1)
            Aq = np.stack([x0[win], y0[win]], 1)
            Bq = np.stack([x1[win], y1[win]], 1)
            bsel = bound[sel][win]
            # THINNED, AND THE THINNING IS REPORTED. Every anchored particle in the slab has an
            # integrin; drawn all at once they overlap into a hedge in which no single spring can be
            # read. The stride is even, so what is drawn is a fair sample of the window rather than a
            # patch of it -- but it is a SAMPLE, and the label says how many are really there.
            n_win = len(Aq)
            if n_win > max_springs:
                st = int(np.ceil(n_win / max_springs))
                Aq, Bq, bsel = Aq[::st], Bq[::st], bsel[::st]
            if len(Aq):
                pts = _zigzag(Aq[bsel], Bq[bsel], teeth=teeth)
                if len(pts):
                    axz.add_collection(LineCollection(pts, colors=SPRING, linewidths=0.9,
                                                      zorder=6, alpha=0.95))
                if (~bsel).any():
                    seg = np.stack([Aq[~bsel], Bq[~bsel]], axis=1)
                    axz.add_collection(LineCollection(seg, colors=BROKEN, linewidths=0.8,
                                                      linestyles=":", zorder=6, alpha=0.9))
            lab = (f"{run}\nframe {t}   {int(mt['nF'])} cells\n"
                   f"standoff {stat[-1][1]:+.4f} box units (0 = on the surface)\n"
                   + ("sheet drawn in one colour: this run's strain history was lost\n"
                      if no_strain else "")
                   + f"{100 * frac:.0f}% of the sheet anchored\n"
                   f"{len(Aq)} of {n_win} integrins drawn, window +/-{half:.1f} tissue units"
                   + (f", {stat[-1][2]} ruptured" if detach > 0 else ""))
            # A BOX BEHIND IT, because the epithelium fills this panel with white and white-on-white is
            # not a label. Same convention otherwise: top left, no title, fontsize 11.
            bb = dict(facecolor="black", alpha=0.55, edgecolor="none", pad=4.0)
            axz.text(0.02, 0.98, lab, transform=axz.transAxes, color="white", fontsize=11,
                     va="top", ha="left", linespacing=1.5, zorder=8, bbox=bb)
            axw.text(0.02, 0.98, "whole section", transform=axw.transAxes, color="white",
                     fontsize=11, va="top", zorder=8, bbox=bb)
            wri.grab_frame()
    plt.close(fig)
    print(f"[{run}] section.mp4  standoff {stat[0][1]:+.4f} -> {stat[-1][1]:+.4f} box units, "
          f"{stat[-1][2]} of {int(anchored.sum())} integrins ruptured "
          f"(detach={'off' if detach <= 0 else detach}, tau_adh={'off' if tau_adh <= 0 else tau_adh})",
          flush=True)
    return out


if __name__ == "__main__":
    runs = sys.argv[1:] or ["121_direct_r25", "124_punctate20", "125_punctate5",
                            "126_turnover", "127_rupture", "128_secreting"]
    for r in runs:
        render(r)
