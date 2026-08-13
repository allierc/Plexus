#!/usr/bin/env python
"""block_metrics -- what the falling fibre block is made of, measured frame by frame.

    python block_metrics.py log/okuda_ECM/02b_ecm_block_bounce   ->  material.png + material.json

WHAT THIS ADDS TO `stress.png`. The test's own plot answers "did it bounce and did the stress arrive".
It cannot answer the question a bounce is actually for, which is whether the block comes back to the
SAME MATERIAL it was before. Eight panels, in three groups:

  BULK (a,b,c)   where the block is, how much energy it still has, and what fraction of the drop each
                 impact returns. An elastic solid loses a fixed FRACTION per impact, so the apex
                 heights fall geometrically; a yielding one loses most of it on the first impact,
                 because the first impact is what rearranges it. One bounce cannot tell the two apart,
                 which is why this script wants three.
  SHAPE (d,e)    the block's three dimensions, each against its own value at frame 0, and their
                 product. A block that slumps gets shorter and wider PERMANENTLY; a block that rings
                 gets shorter and wider only while the front is in it.
  FIBRE (f,g,h)  the three things a fibre can do that a continuum cannot: stretch along itself, bow
                 sideways, and turn. End-to-end length is the stretch; the RMS distance of a strand's
                 particles from its own end-to-end line is the bow; the second Legendre coefficient of
                 the strand direction against the drop axis is the turn.

WHY END-TO-END AND NOT CONTOUR LENGTH. `ecm_seed` jitters every particle by 0.004 box units across a
strand whose particles are only 0.0015 apart, so the contour is a random walk at the particle scale and
is SIX TIMES the end-to-end distance before anything has moved (0.163 measured on `02_ecm_block` frame
0, and 0.163 again at frame 360). Any per-segment measure therefore reports the seeding noise, not the
mechanics. End-to-end and the bow about it are both taken over the whole strand, where the jitter
averages down.

THE STRAND IS AN INDEX CONVENTION, NOT DATA. `ecm_seed` lays strand k as particles k*per .. (k+1)*per-1
and keeps no record of it, so `per = n_particles // n_fibres` from the spec is the only handle on which
particle belongs to which fibre. It is read from the run's own `spec.yaml` here rather than passed in,
and it stops being true the moment strands are allowed different lengths.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FG, BG = "white", "black"
C_MAIN, C_ALT, C_BAND = "#7fb3ff", "#ff8a5c", "#4a6fa5"


def _spec(d):
    s = yaml.safe_load(open(os.path.join(d, "spec.yaml")))
    ops = {o["op"]: o for o in s["operators"]}
    per = max(1, int(s["sets"]["mpm_particle"]["per_parent"]) // int(ops["ecm_seed"]["n_fibres"]))
    return dict(per=per, n_fibres=int(ops["ecm_seed"]["n_fibres"]),
                fibre_len=float(ops["ecm_seed"]["fibre_len"]),
                g=float(ops.get("gravity", {}).get("g", 0.0)),
                dt=float(s["general"]["dt"]), up=int(s.get("plotting", {}).get("up_axis", 1)))


def _turning_points(y):
    """Frames where the centre of mass reverses: the impacts (minima) and apices (maxima).

    A sign change of the first difference, with a 5-frame guard so numerical chatter at the top of a
    bounce -- where the velocity passes through zero -- is not counted as several bounces.
    """
    d = np.sign(np.diff(y))
    ch = np.where(np.diff(d) != 0)[0] + 1
    keep, last = [], -99
    for i in ch:
        if i - last >= 5:
            keep.append(int(i)); last = i
    mins = [i for i in keep if y[i] <= y[max(i - 3, 0)] and y[i] <= y[min(i + 3, len(y) - 1)]]
    maxs = [i for i in keep if i not in mins]
    return mins, maxs


def measure(d):
    z = np.load(os.path.join(d, "traj.npz"))
    P = np.asarray(z["pos"], np.float32)
    vm = [np.asarray(v, np.float32) for v in z["vm"]] if z["vm"].size else None
    S = _spec(d)
    per, up = S["per"], S["up"]
    T, N, D = P.shape
    ax_other = [a for a in range(D) if a != up]

    # ---- bulk: height, energy, restitution -------------------------------------------------
    com = P.mean(axis=1)
    yc = com[:, up]
    ylo = np.percentile(P[:, :, up], 2, axis=1)
    v = np.diff(P, axis=0, prepend=P[:1]) / S["dt"]
    ke = 0.5 * (v ** 2).sum(2).mean(1)                       # per unit mass, mean over particles
    pe = S["g"] * (yc - yc.min())                            # per unit mass, referenced to the lowest point
    mins, maxs = _turning_points(yc)
    apex = [yc[0]] + [float(yc[i]) for i in maxs]
    rest = [float((apex[i + 1] - yc[mins[i]]) / max(apex[i] - yc[mins[i]], 1e-9))
            for i in range(min(len(mins), len(apex) - 1))]

    # ---- shape: the three dimensions, each against frame 0 ---------------------------------
    dim = np.stack([np.percentile(P[:, :, a], 98, axis=1) - np.percentile(P[:, :, a], 2, axis=1)
                    for a in range(D)], axis=1)
    dim_rel = dim / dim[0]
    vol_rel = dim_rel.prod(1)

    # ---- fibre: stretch, bow, turn ---------------------------------------------------------
    nf = N // per
    Fb = P[:, : nf * per].reshape(T, nf, per, D)
    e2e_v = Fb[:, :, -1] - Fb[:, :, 0]
    e2e = np.linalg.norm(e2e_v, axis=-1)
    lam = e2e / e2e[0]
    u = e2e_v / np.clip(e2e, 1e-9, None)[..., None]
    r = Fb - Fb[:, :, :1]
    perp = r - (r * u[:, :, None, :]).sum(-1)[..., None] * u[:, :, None, :]
    bow = np.sqrt((perp ** 2).sum(-1).mean(-1))              # RMS distance from the strand's own axis
    cos2 = u[:, :, up] ** 2
    order = 0.5 * (3.0 * cos2.mean(1) - 1.0)                 # +1 all along the drop axis, -0.5 all across

    # ---- what did not come back ------------------------------------------------------------
    # Displacement with the block's own translation removed, so a bouncing block reads zero and only
    # rearrangement counts. Median over particles: a few strands at the free surface always move.
    resid = np.median(np.linalg.norm(P - P[:1] - (com - com[0])[:, None, :], axis=2), axis=1)

    m = dict(
        frames=int(T), n_fibres=int(nf), per=int(per), dt=S["dt"], g=S["g"],
        y_com=yc.tolist(), y_under=ylo.tolist(), ke=ke.tolist(), pe=pe.tolist(),
        impacts=[int(i) for i in mins], apices=[int(i) for i in maxs],
        apex_height=[float(a) for a in apex], restitution=rest,
        dim_rel=dim_rel.tolist(), vol_rel=vol_rel.tolist(),
        lam_mean=lam.mean(1).tolist(),
        lam_p5=np.percentile(lam, 5, axis=1).tolist(),
        lam_p95=np.percentile(lam, 95, axis=1).tolist(),
        bow_mean=bow.mean(1).tolist(), bow_p95=np.percentile(bow, 95, axis=1).tolist(),
        order=order.tolist(), resid=resid.tolist(),
        e2e0=float(e2e[0].mean()), bow0=float(bow[0].mean()),
        up_axis=int(up), other_axes=ax_other,
    )
    if vm is not None:
        m["vm_mean"] = [float(np.mean(a)) for a in vm]
        m["vm_p99"] = [float(np.percentile(a, 99)) for a in vm]
    return m


def plot(m, out, name=""):
    T = m["frames"]
    t = np.arange(T)
    fig, ax = plt.subplots(2, 4, figsize=(19.0, 8.6), facecolor=BG)
    for a in ax.ravel():
        a.set_facecolor(BG)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("bottom", "left"):
            a.spines[s].set_color(FG)
        a.tick_params(colors=FG, labelsize=9)
        a.set_xlabel("frame", color=FG, fontsize=9)

    def lab(a, s):
        # ABOVE the axes, not inside them. Half of these traces are flat lines that sit exactly where
        # a top-left label goes -- the bow is constant to 0.2% and the stretch to 2% -- so a label
        # inside the frame covers the one feature the panel exists to show.
        a.text(0.0, 1.02, s, transform=a.transAxes, color=FG, fontsize=10.5, va="bottom", ha="left")

    def marks(a):
        for i in m["impacts"]:
            a.axvline(i, color="#555", lw=0.8, ls=":")

    a = ax[0, 0]
    a.plot(t, m["y_com"], color=C_MAIN, lw=1.6)
    a.plot(t, m["y_under"], color=C_BAND, lw=1.0, ls="--")
    marks(a); lab(a, f"a  height: centre of mass (solid), underside (dashed)\n"
                    f"{len(m['impacts'])} impact(s) at {m['impacts']}")

    a = ax[0, 1]
    a.plot(t, m["ke"], color=C_MAIN, lw=1.4)
    a.set_yscale("log"); marks(a)
    lab(a, "b  kinetic energy per unit mass\n(log; a spike is an impact)")

    a = ax[0, 2]
    r = m["restitution"]
    if r:
        a.bar(np.arange(1, len(r) + 1), r, color=C_ALT, width=0.55)
        a.set_xticks(np.arange(1, len(r) + 1))
        a.set_ylim(0, 1)
        a.set_xlabel("impact number", color=FG, fontsize=9)
    lab(a, "c  fraction of the drop height returned\n" +
        ("  ".join(f"{x:.2f}" for x in r) if r else "no completed bounce"))

    a = ax[0, 3]
    dr = np.asarray(m["dim_rel"])
    up = m["up_axis"]
    a.plot(t, dr[:, up], color=C_MAIN, lw=1.5)
    for k in m["other_axes"]:
        a.plot(t, dr[:, k], color=C_ALT, lw=1.1, alpha=0.85)
    a.axhline(1.0, color="#555", lw=0.7); marks(a)
    lab(a, "d  size / size at frame 0\nblue = along the drop, orange = across")

    a = ax[1, 0]
    a.plot(t, m["vol_rel"], color=C_MAIN, lw=1.5)
    a.axhline(1.0, color="#555", lw=0.7); marks(a)
    lab(a, "e  bounding volume / frame 0")

    a = ax[1, 1]
    a.fill_between(t, m["lam_p5"], m["lam_p95"], color=C_BAND, alpha=0.35, lw=0)
    a.plot(t, m["lam_mean"], color=C_MAIN, lw=1.5)
    a.axhline(1.0, color="#555", lw=0.7); marks(a)
    lab(a, f"f  fibre stretch, end to end / frame 0\nmean and p5-p95; L0 = {m['e2e0']:.4f} box units")

    a = ax[1, 2]
    a.plot(t, m["bow_mean"], color=C_MAIN, lw=1.5)
    a.plot(t, m["bow_p95"], color=C_ALT, lw=1.0, alpha=0.85)
    a.axhline(m["bow0"], color="#555", lw=0.7); marks(a)
    lab(a, f"g  bow: RMS distance from the strand's own axis\nmean, p95; seeded value {m['bow0']:.4f}")

    a = ax[1, 3]
    a.plot(t, m["order"], color=C_MAIN, lw=1.5, label="orientation")
    a.axhline(0.0, color="#555", lw=0.7)
    a2 = a.twinx()
    a2.plot(t, m["resid"], color=C_ALT, lw=1.3)
    a2.tick_params(colors=C_ALT, labelsize=9)
    for s in ("top", "left"):
        a2.spines[s].set_visible(False)
    a2.spines["right"].set_color(C_ALT)
    a2.set_facecolor(BG)
    marks(a)
    lab(a, "h  blue: strand orientation order about the drop axis\n"
           "orange, right: median displacement with translation removed")

    fig.text(0.004, 0.995, name, color=FG, fontsize=12, va="top", ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=3.4, w_pad=2.0)
    fig.savefig(out, dpi=130, facecolor=BG)
    plt.close(fig)


def report(d):
    name = os.path.basename(os.path.normpath(d))
    m = measure(d)
    json.dump(m, open(os.path.join(d, "material.json"), "w"), indent=1)
    plot(m, os.path.join(d, "material.png"), name=name)
    lam = np.asarray(m["lam_mean"]); dr = np.asarray(m["dim_rel"]); up = m["up_axis"]
    print(f"[{name}] {m['frames']} frames, {m['n_fibres']} strands of {m['per']} particles", flush=True)
    print(f"  impacts at {m['impacts']}, apices at {m['apices']}", flush=True)
    print(f"  height returned per impact: "
          f"{'  '.join(f'{x:.3f}' for x in m['restitution']) or 'none completed'}", flush=True)
    print(f"  fibre stretch  min {lam.min():.4f}  max {lam.max():.4f}  end {lam[-1]:.4f}", flush=True)
    print(f"  bow  seeded {m['bow0']:.5f}  peak {max(m['bow_mean']):.5f}  "
          f"end {m['bow_mean'][-1]:.5f}", flush=True)
    print(f"  size along drop  min {dr[:, up].min():.4f}  end {dr[-1, up]:.4f}   "
          f"volume end {m['vol_rel'][-1]:.4f}", flush=True)
    print(f"  median displacement, translation removed: end {m['resid'][-1]:.5f} box units", flush=True)
    return m


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "log", "okuda_ECM", "02_ecm_block")
    report(root)
