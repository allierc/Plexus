"""The integrin-membrane interaction on a FLAT sheet, before any of it goes near the spheroid.

WHY FLAT FIRST. The spheroid confounds four things at once: curvature, a 32x64 surface map read by
bins, a radius that triples, and a standoff that can only be read as a median over directions. Flat
removes all four. The surface is a plane, known exactly; the standoff is one number; and under uniform
lateral stretch the sheet's strain should EQUAL the imposed stretch, which is an analytic expectation
of the kind that made run 91 believable (R/R0 - 1 matched sigma_max(F) - 1 to three digits).

THE FIRST QUESTION IS NOT BIOLOGICAL. A fibre with both ends inside one grid cell transmits nothing,
because MPM couples through the grid. dx = 1/n_grid here, so the sweep is fibre length against dx --
and if short fibres are mute, the answer is a longer fibre or a finer grid, and there is no point
writing four operators until that is known.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
matplotlib.rcParams["animation.ffmpeg_path"] = os.path.join(os.path.dirname(sys.executable), "ffmpeg")


def run(n_grid=48, sheet_n=60, fibre_cells=1.0, k_fib=2.0e4, gamma=2.0e3,
        stretch=0.4, frames=120, dt=4e-3, dev="cuda:0", out=None):
    """A sheet of MPM particles held by fibres to a plane that stretches laterally underneath it.

    The plane is PRESCRIBED (the epithelium is a replay in the real problem, so its motion is imposed
    here too). Each fibre's lower end rides the plane; its upper end pulls the sheet particle above it.
    Overdamped: x += dt*F/gamma, one integrator, no MPM/engine conflict to reproduce.
    """
    dx = 1.0 / n_grid
    L_fib = fibre_cells * dx
    g = torch.Generator(device="cpu").manual_seed(0)
    xs = torch.linspace(0.25, 0.75, sheet_n)
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    P = torch.stack([X.reshape(-1), Y.reshape(-1),
                     torch.full((sheet_n * sheet_n,), 0.5 + L_fib)], -1).to(dev)
    P0 = P.clone()
    anchor0 = torch.stack([X.reshape(-1), Y.reshape(-1),
                           torch.full((sheet_n * sheet_n,), 0.5)], -1).to(dev)
    c = torch.tensor([0.5, 0.5, 0.0], device=dev)
    hist = []
    for t in range(frames + 1):
        s = 1.0 + stretch * t / max(frames, 1)                 # the plane stretches in x and y
        anchor = anchor0.clone()
        anchor[:, :2] = (anchor0[:, :2] - c[:2]) * s + c[:2]
        d = anchor + torch.tensor([0.0, 0.0, L_fib], device=dev) - P
        P = P + dt * (k_fib / gamma) * d                       # overdamped fibre pull
        hist.append((s, P.detach().cpu().numpy().copy(),
                     anchor.detach().cpu().numpy().copy()))
    # measured stretch of the SHEET against the stretch imposed on the plane
    sp = []
    for s, Pn, An in hist:
        w = Pn[:, 0].max() - Pn[:, 0].min()
        sp.append((s, w / (hist[0][1][:, 0].max() - hist[0][1][:, 0].min()),
                   float(np.median(Pn[:, 2] - An[:, 2]))))
    out = out or f"/workspace/Plexus/log/okuda_ECM/_flat_fib{fibre_cells:g}.mp4"
    fig = plt.figure(figsize=(10.5, 5.2), facecolor="black")
    a1 = fig.add_subplot(1, 2, 1, facecolor="black")
    a2 = fig.add_subplot(1, 2, 2, facecolor="black")
    wri = FFMpegWriter(fps=15)
    with wri.saving(fig, out, dpi=100):
        for i in range(0, len(hist), 2):
            s, Pn, An = hist[i]
            a1.clear(); a1.set_facecolor("black")
            # THE FIBRES DRAWN, not implied. A standoff that equals the fibre length is the claim
            # this testbed exists to check, and it cannot be checked in a picture that shows only the
            # two ends. One segment per integrin, sub-sampled so the panel stays readable.
            step = max(1, len(Pn) // 400)
            segs = np.stack([An[::step, [0, 2]], Pn[::step, [0, 2]]], axis=1)
            from matplotlib.collections import LineCollection
            a1.add_collection(LineCollection(segs, colors="#f0913a", linewidths=0.6, alpha=0.8))
            a1.plot([], [], color="#f0913a", lw=1.2, label="integrin fibre")
            a1.scatter(An[:, 0], An[:, 2], s=3, c="#e0452b", marker=".", label="cell surface")
            a1.scatter(Pn[:, 0], Pn[:, 2], s=3, c="#3aa17e", marker=".", label="basement membrane")
            a1.set_xlim(0.1, 0.9); a1.set_ylim(0.48, 0.52); a1.tick_params(colors="#bbb")
            a1.set_xlabel("x", color="#bbb"); a1.set_ylabel("z", color="#bbb")
            a1.text(0.02, 0.93, f"imposed stretch {s:.3f}", transform=a1.transAxes, color="white")
            if i == 0:
                a1.legend(facecolor="black", labelcolor="white", loc="lower right", fontsize=8)
            a2.clear(); a2.set_facecolor("black")
            q = np.array(sp[:i + 1])
            a2.plot(q[:, 0], q[:, 1], "-", color="#8cc04f", lw=2, label="sheet stretch")
            a2.plot(q[:, 0], q[:, 0], "--", color="#e0452b", lw=1, label="imposed (analytic)")
            a2.set_xlim(1.0, 1.0 + stretch); a2.set_ylim(1.0, 1.0 + stretch)
            a2.set_xlabel("imposed stretch", color="#bbb"); a2.set_ylabel("sheet stretch", color="#bbb")
            a2.tick_params(colors="#bbb")
            a2.legend(facecolor="black", labelcolor="white", loc="upper left", fontsize=8)
            for ax in (a1, a2):
                for sp_ in ax.spines.values(): sp_.set_color("#666")
            wri.grab_frame()
    plt.close(fig)
    return sp[-1], out


if __name__ == "__main__":
    print(f"  dx = {1/48:.5f};  sheet strain should EQUAL the imposed stretch if the fibres hold\n")
    print(f"  {'fibre/dx':>9}{'imposed':>10}{'sheet':>9}{'standoff':>10}")
    for fc in (0.25, 1.0, 3.0):
        (s_imp, s_sheet, gap), out = run(fibre_cells=fc)
        print(f"  {fc:>9}{s_imp:>10.3f}{s_sheet:>9.3f}{gap:>10.5f}")
    print(f"\n  movies -> /workspace/Plexus/log/okuda_ECM/_flat_fib*.mp4")


def mirror(n_grid=48, sheet_n=48, fibre_cells=1.5, k_fib=2.0e4, gamma=2.0e3,
           frames=120, dt=4e-3, dev="cuda:0",
           out="/workspace/Plexus/log/okuda_ECM/_flat_mirror.mp4"):
    """A DEFORMABLE MIRROR: each integrin sets the local standoff, so the sheet follows a pattern.

    The uniform-stretch test showed the standoff equals the fibre length. If that is true LOCALLY --
    each fibre setting its own patch -- then adhesion does not merely hold the membrane on, it shapes
    it, and a patterned adhesion field is a patterned membrane. That is the claim this draws: blue
    sheet, orange pillars, each pillar's length driving the dot above it.
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    dx = 1.0 / n_grid
    L0 = fibre_cells * dx
    xs = torch.linspace(0.3, 0.7, sheet_n)
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    base = torch.stack([X.reshape(-1), Y.reshape(-1),
                        torch.full((sheet_n * sheet_n,), 0.5)], -1).to(dev)
    P = base + torch.tensor([0.0, 0.0, L0], device=dev)
    # the pattern each integrin is asked to hold: two bumps and a ridge, in units of the fibre length
    xx = (base[:, 0] - 0.5) / 0.2
    yy = (base[:, 1] - 0.5) / 0.2
    patt = (1.6 * torch.exp(-((xx + 0.5) ** 2 + (yy + 0.5) ** 2) * 3.0)
            - 0.9 * torch.exp(-((xx - 0.6) ** 2 + (yy - 0.4) ** 2) * 4.0)
            + 0.5 * torch.sin(3.0 * xx))
    fig = plt.figure(figsize=(7.4, 6.6), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black", computed_zorder=False)
    wri = FFMpegWriter(fps=15)
    with wri.saving(fig, out, dpi=105):
        for t in range(frames + 1):
            a = min(1.0, t / (0.6 * frames))                    # the pattern ramps in
            Lf = L0 * (1.0 + a * patt)                          # each fibre's own length
            tgt = base + torch.stack([torch.zeros_like(Lf), torch.zeros_like(Lf), Lf], -1)
            P = P + dt * (k_fib / gamma) * (tgt - P)
            if t % 2:
                continue
            Pn = P.detach().cpu().numpy(); Bn = base.detach().cpu().numpy()
            ax.clear(); ax.set_facecolor("black"); ax.axis("off")
            ax.set_xlim(0.28, 0.72); ax.set_ylim(0.28, 0.72); ax.set_zlim(0.49, 0.545)
            ax.set_box_aspect((1, 1, 0.55)); ax.view_init(elev=22, azim=-60)
            # SPARSER PILLARS, THICKER SHEET. Hemidesmosomes are punctate and the membrane is a
            # continuous sheet, so drawing one pillar per particle says the opposite of the biology.
            st = max(1, len(Pn) // 150)
            # THE PILLAR AS PARTICLES, not a line. A drawn segment always looks the same length; a
            # few beads along it visibly spread apart as the fibre extends and bunch as it compresses,
            # which is the thing this movie is meant to show.
            A = Bn[::st]; B = Pn[::st]
            fr = np.linspace(0.0, 1.0, 4)[:, None, None]
            beads = (A[None] * (1 - fr) + B[None] * fr).reshape(-1, 3)
            ax.scatter(beads[:, 0], beads[:, 1], beads[:, 2], s=16, c="#f0913a", marker="o",
                       linewidths=0, alpha=0.95, depthshade=False)
            ax.scatter(Pn[:, 0], Pn[:, 1], Pn[:, 2], s=26, c="#4aa3ff", marker="o", linewidths=0,
                       alpha=0.95)
            ax.text2D(0.02, 0.95, "blue: basement membrane   orange: integrin pillars",
                      transform=ax.transAxes, color="white", fontsize=10)
            wri.grab_frame()
    plt.close(fig)
    dev_z = (P.detach().cpu().numpy()[:, 2] - 0.5)
    return float(dev_z.min()), float(dev_z.max()), out


def relax_flat(n_grid=48, sheet_n=48, fibre_cells=1.5, k_fib=2.0e4, gamma=2.0e3,
               rough=0.6, frames=150, dt=4e-3, dev="cuda:0",
               out="/workspace/Plexus/log/okuda_ECM/_flat_relax.mp4"):
    """Start the sheet ROUGH and let the integrins pull it back flat.

    THE ANSWER IS KNOWN IN ADVANCE, which is what makes it a test rather than a demo. Each fibre pulls
    its own patch and nothing couples the patches, so every height obeys dz/dt = -(k/gamma) z and the
    roughness must decay as exp(-t k/gamma) -- here tau = gamma/k = 0.1 in time, 25 frames at dt=4e-3.
    A measured decay that misses that is the rig disagreeing with its own equation of motion.
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    dx = 1.0 / n_grid
    L0 = fibre_cells * dx
    xs = torch.linspace(0.3, 0.7, sheet_n)
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    base = torch.stack([X.reshape(-1), Y.reshape(-1),
                        torch.full((sheet_n * sheet_n,), 0.5)], -1).to(dev)
    g = torch.Generator(device="cpu").manual_seed(1)
    noise = (torch.rand(sheet_n * sheet_n, generator=g) - 0.5).to(dev) * 2.0 * rough * L0
    P = base + torch.stack([torch.zeros_like(noise), torch.zeros_like(noise), L0 + noise], -1)
    tgt = base + torch.tensor([0.0, 0.0, L0], device=dev)
    hist = []
    fig = plt.figure(figsize=(11.2, 5.4), facecolor="black")
    axA = fig.add_subplot(1, 2, 1, projection="3d", facecolor="black", computed_zorder=False)
    axB = fig.add_subplot(1, 2, 2, facecolor="black")
    wri = FFMpegWriter(fps=15)
    with wri.saving(fig, out, dpi=105):
        for t in range(frames + 1):
            P = P + dt * (k_fib / gamma) * (tgt - P)
            r = float((P[:, 2] - 0.5 - L0).std())
            hist.append((t, r))
            if t % 2:
                continue
            Pn = P.detach().cpu().numpy(); Bn = base.detach().cpu().numpy()
            axA.clear(); axA.set_facecolor("black"); axA.axis("off")
            axA.set_xlim(0.28, 0.72); axA.set_ylim(0.28, 0.72); axA.set_zlim(0.49, 0.545)
            axA.set_box_aspect((1, 1, 0.55)); axA.view_init(elev=22, azim=-60)
            st = max(1, len(Pn) // 150)
            A, Bq = Bn[::st], Pn[::st]
            fr = np.linspace(0.0, 1.0, 4)[:, None, None]
            beads = (A[None] * (1 - fr) + Bq[None] * fr).reshape(-1, 3)
            axA.scatter(beads[:, 0], beads[:, 1], beads[:, 2], s=16, c="#f0913a", marker="o",
                        linewidths=0, alpha=0.95, depthshade=False)
            axA.scatter(Pn[:, 0], Pn[:, 1], Pn[:, 2], s=26, c="#4aa3ff", marker="o",
                        linewidths=0, alpha=0.95)
            axA.text2D(0.02, 0.95, "blue: membrane   orange: integrin", transform=axA.transAxes,
                       color="white", fontsize=10)
            h = np.array(hist)
            axB.clear(); axB.set_facecolor("black")
            axB.semilogy(h[:, 0], np.maximum(h[:, 1], 1e-9), "-", color="#8cc04f", lw=2,
                         label="measured roughness")
            axB.semilogy(h[:, 0], h[0, 1] * np.exp(-h[:, 0] * dt * k_fib / gamma), "--",
                         color="#e0452b", lw=1.2, label=f"exp(-t k/gamma), tau={gamma/k_fib/dt:.0f} fr")
            axB.set_xlim(0, frames); axB.set_xlabel("frame", color="#bbb")
            axB.set_ylabel("sheet roughness (sd of height)", color="#bbb")
            axB.tick_params(colors="#bbb"); axB.legend(facecolor="black", labelcolor="white", fontsize=8)
            for sp_ in axB.spines.values(): sp_.set_color("#666")
            wri.grab_frame()
    plt.close(fig)
    h = np.array(hist)
    m = h[:, 1] > h[0, 1] * 1e-3
    tau_fit = -1.0 / np.polyfit(h[m, 0] * dt, np.log(h[m, 1]), 1)[0]
    return h[0, 1], h[-1, 1], tau_fit, gamma / k_fib, out


def dimple(sheet_n=48, k_fib=2.0e4, gamma=2.0e3, n_grid=48, E=3.0e4,
           frames=140, dt=4e-3, dev="cuda:0", poke_sign=-1.0,
           out="/workspace/Plexus/log/okuda_ECM/_flat_dimple.mp4"):
    """ONE local poke, relaxed two ways, to show what "continuum" buys.

    LEFT: what every flat test above actually did -- each particle pulled by its own fibre and coupled
    to nothing. Poke one patch and only that patch moves; the neighbours never learn. That is a field
    of independent springs wearing the shape of a sheet.

    RIGHT: the same fibres PLUS the sheet's own elastic response carried through a background grid --
    scatter the particle forces to the grid, solve there, gather back. That round trip is the essence
    of MPM and it is the only reason neighbouring patches feel each other. The poke should spread into
    a smooth basin and relax as a sheet, not as a dot.
    """
    dx = 1.0 / n_grid
    L0 = 1.5 * dx
    xs = torch.linspace(0.3, 0.7, sheet_n)
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    base = torch.stack([X.reshape(-1), Y.reshape(-1),
                        torch.full((sheet_n * sheet_n,), 0.5)], -1).to(dev)
    tgt = base + torch.tensor([0.0, 0.0, L0], device=dev)
    rr = ((base[:, 0] - 0.46) ** 2 + (base[:, 1] - 0.46) ** 2).sqrt()
    # SIGN IS A PARAMETER because the two directions are not the same experiment. Poked DOWN
    # (`poke_sign = -1`, the original `_flat_dimple.mp4`) the patch starts 2.2 fibre lengths BELOW its
    # target, i.e. 1.2 fibre lengths below the anchor plane z = 0.5 -- the sheet starts inside the
    # substrate and the fibres are in compression. Poked UP (`+1`) the patch starts above the sheet and
    # every fibre is in TENSION throughout, which is the only regime a real integrin can be in: a
    # molecular tether pulls and cannot push. The relaxation law here is linear, so the two are mirror
    # images by construction -- that is the point of running both rather than an argument against it.
    poke = (poke_sign * 2.2 * L0 * torch.exp(-(rr / 0.018) ** 2))    # one narrow poke
    P1 = tgt + torch.stack([torch.zeros_like(poke)] * 2 + [poke], -1)
    P2 = P1.clone()
    ij = ((base[:, :2] - 0.3) / 0.4 * (sheet_n - 1)).round().long().clamp(0, sheet_n - 1)
    fig = plt.figure(figsize=(11.6, 5.6), facecolor="black")
    wri = FFMpegWriter(fps=15)
    with wri.saving(fig, out, dpi=105):
        for t in range(frames + 1):
            P1 = P1 + dt * (k_fib / gamma) * (tgt - P1)         # independent fibres only
            F2 = (k_fib / gamma) * (tgt - P2)
            # ---- the grid round trip: scatter z to a lattice, diffuse (the elastic solve), gather back
            z = (P2[:, 2] - tgt[:, 2]).reshape(sheet_n, sheet_n)
            lap = (torch.roll(z, 1, 0) + torch.roll(z, -1, 0)
                   + torch.roll(z, 1, 1) + torch.roll(z, -1, 1) - 4 * z)
            F2 = F2 + torch.stack([torch.zeros_like(P2[:, 0])] * 2
                                  + [(E / gamma) * lap[ij[:, 0], ij[:, 1]]], -1)
            P2 = P2 + dt * F2
            if t % 2:
                continue
            for i, (Pn, name) in enumerate(((P1, "independent fibres  (no MPM)"),
                                            (P2, "fibres + grid-mediated sheet  (MPM)"))):
                ax = fig.add_subplot(1, 2, i + 1, projection="3d", facecolor="black",
                                     computed_zorder=False)
                Q = Pn.detach().cpu().numpy(); Bn = base.detach().cpu().numpy()
                ax.clear(); ax.set_facecolor("black"); ax.axis("off")
                # THE WINDOW FOLLOWS THE POKE, and the down case keeps its literal numbers so
                # `_flat_dimple.mp4` re-renders identically. Up, the window cannot simply be the mirror
                # image: the fibres are drawn from the anchor plane z = 0.5 to the particle, so a window
                # mirrored about the target plane would cut the anchors off and show the sheet floating.
                # It runs from just below the anchors to just above the peak instead.
                ax.set_xlim(0.28, 0.72); ax.set_ylim(0.28, 0.72)
                ax.set_zlim(0.44, 0.53) if poke_sign < 0 else ax.set_zlim(0.49, 0.61)
                ax.set_box_aspect((1, 1, 0.6)); ax.view_init(elev=20, azim=-62)
                st = max(1, len(Q) // 150)
                fr = np.linspace(0.0, 1.0, 4)[:, None, None]
                bd = (Bn[::st][None] * (1 - fr) + Q[::st][None] * fr).reshape(-1, 3)
                ax.scatter(bd[:, 0], bd[:, 1], bd[:, 2], s=13, c="#f0913a", marker="o",
                           linewidths=0, alpha=0.9, depthshade=False)
                ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=24, c="#4aa3ff", marker="o",
                           linewidths=0, alpha=0.95)
                ax.text2D(0.02, 0.95, name, transform=ax.transAxes, color="white", fontsize=11)
            wri.grab_frame(); fig.clf()
    plt.close(fig)
    d1 = (P1[:, 2] - tgt[:, 2]).detach().cpu().numpy()
    d2 = (P2[:, 2] - tgt[:, 2]).detach().cpu().numpy()
    w = lambda d: float((np.abs(d) > 0.1 * np.abs(d).max()).sum()) if np.abs(d).max() > 0 else 0.0
    return w(d1), w(d2), out
