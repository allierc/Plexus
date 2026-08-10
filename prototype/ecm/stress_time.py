#!/usr/bin/env python
"""stress_time -- the von Mises stress of a few named ECM particles against time, and how much of
the flicker in the movie is the material and how much is the palette.

    python stress_time.py 04_spheroid_ecm 04c_spheroid_fibres [04d_...]
        ->  <run>/stress_time.png  and  a `flicker` block in <run>/metrics.json

WHY A FEW PARTICLES AND NOT A FIELD. A mean over 200,000 particles is smooth by construction --
it averages exactly the thing being asked about. Flicker is a property of ONE particle's history,
so this follows named particles, chosen once by their INITIAL radius so the same shells are
compared across runs of different length.

THE QUESTION THE MOVIE RAISES IS NOT ABOUT STRESS, IT IS ABOUT THE MAP FROM STRESS TO COLOUR.
`ecm_stress` bands the von Mises invariant into eight levels against a fixed full scale, and the
renderer colours a whole STRAND by the median band of its twenty particles. So there are three
places a frame can differ from the one before it, and they are separable:

  physics   the von Mises value itself moves. Reported as the relative frame-to-frame change,
            |dv|/v, per particle: a material under a slowly advancing boundary should sit well
            below the 1/8 of full scale that would move it a whole band.
  banding   the value barely moves but sits on a band EDGE, so the colour flips back and forth.
            Reported as band flips per particle per frame, and as the fraction of those flips whose
            underlying |dv| is under a tenth of a band -- those are pure quantisation.
  drawing   a strand is drawn as one polyline or as several depending on whether an internal gap
            exceeds three grid cells, so a strand near the threshold appears and disappears.
            Reported as the number of strands whose piece-count changes between frames.

The third is the one that only exists in runs drawn by the strand renderer, and it is the one that
a movie shows as flicker in the FIBRES rather than in the colour.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
LOG = os.path.join(_ROOT, "log", "okuda_ECM")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

CENTRE = np.array([0.5, 0.5, 0.5], np.float32)
BANDS = 8
GAP_CELLS = 3.0
DX = 1.0 / 64


def pick(P0, per, n_per_shell=3, shells=(0.055, 0.09, 0.15, 0.25)):
    """Named particles, chosen by the radius they STARTED at, one set of shells for every run.

    Deterministic: the first `n_per_shell` particles whose initial radius is closest to each shell,
    so two runs of different length are followed at the same material positions and the traces can
    be laid on top of each other.
    """
    r0 = np.linalg.norm(P0 - CENTRE, axis=1)
    out = []
    for s in shells:
        idx = np.argsort(np.abs(r0 - s))[: 40 * n_per_shell]
        idx = idx[:: max(1, len(idx) // n_per_shell)][:n_per_shell]
        out += [(int(i), float(r0[i]), s) for i in idx]
    return out


def measure(d, scale=None):
    z = np.load(os.path.join(d, "traj.npz"))
    if "vm" not in z.files or not len(z["vm"]):
        print(f"[{os.path.basename(d)}] no `vm` in traj.npz -- nothing to follow")
        return None
    V = np.asarray(z["vm"], np.float32)              # [T, N]
    P = np.asarray(z["pos"], np.float32)
    T, N = V.shape
    m0 = json.load(open(os.path.join(d, "metrics.json")))
    per = int(m0.get("per_strand", 20))
    sc = scale or float(m0.get("stress_full_scale", np.percentile(V[V > 0], 99) if (V > 0).any()
                                                    else 1.0))
    sel = pick(P[0], per)
    idx = np.array([i for i, _, _ in sel])
    tr = V[:, idx]                                   # [T, k]

    # --- physics: how far the value itself moves between frames, relative to its own size
    dv = np.abs(np.diff(V, axis=0))
    ref = np.maximum(V[:-1], 1e-3 * sc)
    rel = dv / ref
    # --- banding: the colour the movie actually uses
    band = np.clip(V / max(sc, 1e-12), 0, 1)
    band = np.round(band * (BANDS - 1)).astype(np.int8)
    flip = band[1:] != band[:-1]
    tiny = flip & (dv < 0.1 * sc / (BANDS - 1))      # a flip a tenth of a band's worth of change made
    # --- drawing: how many strands change their piece count, at the renderer's own gap
    nf = N // per
    pieces = []
    for t in range(T):
        S = P[t, : nf * per].reshape(nf, per, 3)
        g = np.linalg.norm(np.diff(S, axis=1), axis=2) > GAP_CELLS * DX
        pieces.append(g.sum(1) + 1)
    pieces = np.stack(pieces)
    piece_change = (pieces[1:] != pieces[:-1]).sum(1)

    out = dict(
        n_followed=len(idx), full_scale=float(sc), per_strand=per, frames=int(T),
        followed=[dict(i=int(i), r0=float(r), shell=float(s)) for i, r, s in sel],
        rel_change_median=float(np.median(rel)), rel_change_p99=float(np.percentile(rel, 99)),
        band_flips_per_particle_per_frame=float(flip.mean()),
        band_flips_that_are_quantisation=float(tiny.sum() / max(flip.sum(), 1)),
        strand_piece_changes_per_frame=float(piece_change.mean()),
        strand_piece_changes_frac=float(piece_change.mean() / max(nf, 1)),
        trace=[[float(v) for v in tr[:, k]] for k in range(tr.shape[1])])
    m0["flicker"] = {k: v for k, v in out.items() if k != "trace"}
    m0["stress_trace"] = out["trace"]
    m0["stress_trace_r0"] = [r for _, r, _ in sel]
    json.dump(m0, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    return out



def _panel(ax, letter):
    """A bold letter top-left and no title. The numbers a title used to carry go into the note's
    caption, where they can be read against the gate they belong to; a title repeats them in a place
    the figure cannot explain them."""
    ax.text(0.0, 1.03, letter, transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom")


def plot(runs, out):
    fig, ax = plt.subplots(1, len(runs) + 1, figsize=(4.6 * (len(runs) + 1), 3.6),
                           facecolor="white")
    ax = np.atleast_1d(ax)
    for _k, (a, (name, o)) in enumerate(zip(ax, runs)):
        sc = o["full_scale"]
        edges = np.arange(BANDS) * sc / (BANDS - 1)
        for e in edges[1:]:
            a.axhline(e, color="#ddd", lw=0.6, zorder=0)
        t = np.linspace(0, 1, o["frames"])
        for k, (tr, r0) in enumerate(zip(o["trace"], [f["r0"] for f in o["followed"]])):
            a.plot(t, tr, lw=1.0, color=plt.cm.viridis(k / max(len(o["trace"]) - 1, 1)),
                   label=f"r0 = {r0:.3f}" if k % 3 == 0 else None)
        a.set_yscale("symlog", linthresh=1e-3)
        a.set_xlabel("run fraction")
        a.set_ylabel("von Mises stress")
        _panel(a, "abcdef"[_k])
        a.legend(fontsize=6.5, frameon=False)
        a.spines[["top", "right"]].set_visible(False)
    a = ax[-1]
    w = 0.35
    xs = np.arange(len(runs))
    a.bar(xs - w / 2, [o["band_flips_per_particle_per_frame"] for _, o in runs], w,
          color="#e08a2e", label="band flips / particle / frame")
    a.bar(xs + w / 2, [o["strand_piece_changes_frac"] for _, o in runs], w,
          color="#2b6cb0", label="strands changing piece count / frame")
    a.set_xticks(xs); a.set_xticklabels([n for n, _ in runs], fontsize=7, rotation=12)
    a.set_yscale("log"); a.legend(fontsize=7, frameon=False)
    _panel(a, "abcdef"[len(runs)])
    a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("-")]
    runs = []
    for n in names:
        d = os.path.join(LOG, n)
        o = measure(d)
        if o is None:
            continue
        runs.append((n, o))
        print(f"[{n}] followed {o['n_followed']} particles | value moves "
              f"{100 * o['rel_change_median']:.2f}% median / {100 * o['rel_change_p99']:.0f}% p99 "
              f"per frame | band flips {o['band_flips_per_particle_per_frame']:.4f} per particle "
              f"per frame, {100 * o['band_flips_that_are_quantisation']:.0f}% of them from a change "
              f"under a tenth of a band | strands changing piece count "
              f"{o['strand_piece_changes_per_frame']:.0f} per frame "
              f"({100 * o['strand_piece_changes_frac']:.2f}%)", flush=True)
    if runs:
        for n, o in runs:
            plot([(n, o)], os.path.join(LOG, n, "stress_time.png"))
        plot(runs, os.path.join(LOG, names[0], "stress_time_compare.png"))
        print(f"[stress_time] {LOG}/{names[0]}/stress_time_compare.png", flush=True)


if __name__ == "__main__":
    main()
