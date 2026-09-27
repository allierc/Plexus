#!/usr/bin/env python
"""Did the connectome carry the rhythm from the pacemakers to the ciliary band?

    python tools/platynereis_rhythm.py plat_r7_rhythm --period 8.0

R7 puts an intrinsic rhythm in the 124 motoneurons, because the connectome cannot make one
itself: every weight is a non-negative synapse count, so by Perron-Frobenius its leading mode is
real and rising gain makes the network latch rather than beat (tools/platynereis_spectrum.py).
The pacemakers oscillating is therefore not a result -- they were told to. The result is whether
anything ELSE oscillates at their frequency, and which things.

THE TEST. Take the power spectrum of every cell's membrane state, find how much of each one's
power sits in the pacemaker's line, and compare populations:

  * the MOTONEURONS are the positive control. They were driven; if they do not carry the line,
    the drive is too weak or the leak too fast and nothing downstream means anything.
  * the CILIARY BAND is the question. It is 255 synapses downstream of the motoneurons.
  * cells with NO EDGES AT ALL are the negative control -- 3,108 of them, same membrane law, same
    noise, no input. They cannot have heard the pacemaker, so whatever fraction of power they
    show in that line is what this measurement reads off pure noise, and the band has to beat it.

WHY A LINE AND NOT A CORRELATION. A cross-correlation between the pools would also be raised by
anything they share -- the settling transient, the noise seed, a common drift. Power at one
frequency that the negative control does not have is specific to the rhythm.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))

REGION = "platynereis_larva_4117"


def load(name: str):
    from plexus.paths import graphs_data_path
    G = graphs_data_path()
    for sub in ("studio", "platynereis"):
        t = os.path.join(G, sub, name, "trajectory.npz")
        if os.path.exists(t):
            break
    else:
        raise SystemExit(f"no trajectory for {name}")
    R = os.path.join(G, "neural_regions", REGION)
    return (np.load(t, allow_pickle=True),
            np.load(os.path.join(R, "neurons.npz"), allow_pickle=True),
            np.load(os.path.join(R, "connectome.npz"), allow_pickle=True))


def line_share(X, dt, f0, width=0.25):
    """Fraction of each column's power within +-`width` of `f0`. X is [T, N]. Returns [N]."""
    X = X - X.mean(0, keepdims=True)
    W = np.hanning(X.shape[0])[:, None]
    P = np.abs(np.fft.rfft(X * W, axis=0)) ** 2
    f = np.fft.rfftfreq(X.shape[0], d=dt)
    P[0] = 0.0
    band = (f > f0 * (1 - width)) & (f < f0 * (1 + width))
    tot = P.sum(0)
    return np.where(tot > 1e-30, P[band].sum(0) / np.maximum(tot, 1e-30), 0.0), f, P


def measure(tr, z, c, dt, period, settle=0.3):
    V = np.asarray(tr["neuron__voltage"])[:, :, 0]
    t0 = int(settle * V.shape[0])
    V = V[t0:]
    f0 = 1.0 / period
    share, f, P = line_share(V, dt, f0)

    cn, ci = list(z["cell_class_names"]), np.asarray(z["cell_class_id"])
    ei = np.asarray(c["edge_index"])
    deg = np.zeros(V.shape[1], int)
    np.add.at(deg, ei[0], 1)
    np.add.at(deg, ei[1], 1)

    pools = {
        "Motoneuron (driven)": ci == cn.index("Motoneuron"),
        "ciliary band": ci == cn.index("ciliary band"),
        "Interneuron": ci == cn.index("Interneuron"),
        "Sensory neuron": ci == cn.index("Sensory neuron"),
        "unwired (no edges)": deg == 0,
    }
    out = {"f0": f0, "period": period, "n_frames": int(V.shape[0]),
           "pools": {k: {"n": int(m.sum()), "share": float(np.median(share[m])),
                         "amp": float(np.median(V[:, m].std(0)))}
                     for k, m in pools.items() if m.any()}}
    base = out["pools"]["unwired (no edges)"]["share"]
    for k in out["pools"]:
        out["pools"][k]["over_noise"] = out["pools"][k]["share"] / max(base, 1e-12)
    return out, V, share, pools, f, P


def draw(V, share, pools, f, P, st, dt, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from platynereis_specs import COLOR

    col = {"Motoneuron (driven)": COLOR["Motoneuron"], "ciliary band": COLOR["ciliary band"],
           "Interneuron": COLOR["Interneuron"], "Sensory neuron": COLOR["Sensory neuron"],
           "unwired (no edges)": "#9aa3b0"}
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3), facecolor="white")
    for a in ax:
        a.set_facecolor("white")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=10)

    t = np.arange(V.shape[0]) * dt
    for k in ("Motoneuron (driven)", "ciliary band", "unwired (no edges)"):
        m = pools[k]
        ax[0].plot(t, V[:, m].mean(1), lw=1.5, color=col[k], label=f"{k} ({int(m.sum())})")
    ax[0].set_xlabel("time after settling (scene seconds)", fontsize=11)
    ax[0].set_ylabel("pool mean membrane state", fontsize=11)
    ax[0].legend(fontsize=9, frameon=False)
    ax[0].text(0, 1.04, f"a   the pacemakers run at 1/{st['period']:.0f} s = "
                        f"{st['f0']:.3f} Hz", transform=ax[0].transAxes, fontsize=12)

    for k in ("Motoneuron (driven)", "ciliary band", "Interneuron", "unwired (no edges)"):
        m = pools[k]
        sp = P[:, m].mean(1)
        ax[1].semilogy(f, sp / max(sp.sum(), 1e-30), lw=1.3, color=col[k], label=k)
    ax[1].axvline(st["f0"], color="#444444", lw=1.0, ls=":")
    ax[1].set_xlim(0, min(1.2, f.max()))
    ax[1].set_xlabel("frequency (Hz)", fontsize=11)
    ax[1].set_ylabel("power, normalised", fontsize=11)
    ax[1].legend(fontsize=8, frameon=False)
    ax[1].text(0, 1.04, "b   the pacemaker's line, and who carries it",
               transform=ax[1].transAxes, fontsize=12)

    ks = list(st["pools"])
    ax[2].barh(range(len(ks)), [st["pools"][k]["over_noise"] for k in ks],
               color=[col[k] for k in ks], height=0.6)
    ax[2].axvline(1.0, color="#444444", lw=1.0, ls=":")
    ax[2].set_yticks(range(len(ks)))
    ax[2].set_yticklabels([f"{k}  ({st['pools'][k]['n']})" for k in ks], fontsize=9)
    ax[2].set_xlabel("power in the line, relative to the unwired cells", fontsize=11)
    ax[2].text(0, 1.04, "c   against the negative control",
               transform=ax[2].transAxes, fontsize=12)

    fig.tight_layout()
    fig.savefig(png, dpi=130, facecolor="white")
    plt.close(fig)


def next_index() -> int:
    d = os.path.join(REPO, "builder", "png")
    os.makedirs(d, exist_ok=True)
    ns = [int(f[:4]) for f in os.listdir(d) if f[:4].isdigit()]
    return (max(ns) + 1) if ns else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="plat_r7_rhythm")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--period", type=float, default=8.0)
    ap.add_argument("--why", default="")
    a = ap.parse_args()
    tr, z, c = load(a.name)
    st, V, share, pools, f, P = measure(tr, z, c, a.dt, a.period)

    print(f"{a.name}: pacemakers at 1/{a.period:.1f} s = {st['f0']:.4f} Hz, "
          f"{st['n_frames']} frames after settling\n")
    print(f"  {'pool':24s} {'n':>6s} {'power in the line':>18s} {'x the unwired':>14s} "
          f"{'amplitude':>10s}")
    for k, d in st["pools"].items():
        print(f"  {k:24s} {d['n']:6d} {100 * d['share']:17.1f}% {d['over_noise']:14.2f} "
              f"{d['amp']:10.4f}")

    i = next_index()
    draw(V, share, pools, f, P, st, a.dt, os.path.join(REPO, "builder", "png", f"{i:04d}.png"))
    from plexus.paths import graphs_data_path
    import shutil
    import time as _t
    for sub in ("studio", "platynereis"):
        sp = os.path.join(graphs_data_path(), sub, a.name, "spec.yaml")
        if os.path.exists(sp):
            shutil.copy(sp, os.path.join(REPO, "builder", "spec", f"{i:04d}.yaml"))
            break
    with open(os.path.join(REPO, "builder", "why", f"{i:04d}.txt"), "w") as fh:
        fh.write(f"time   {_t.strftime('%Y-%m-%d %H:%M:%S')}\nspec   {a.name}\n"
                 f"camera (a figure, not a render)\n"
                 f"why    {a.why or 'did the connectome carry the rhythm, measured'}\n")
    print(f"\n  -> builder/*/{i:04d}.*")
