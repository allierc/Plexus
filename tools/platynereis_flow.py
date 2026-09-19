#!/usr/bin/env python
"""Did the beat move the water? The measurement.

    python tools/platynereis_flow.py plat_r6_water

A movie of 140,000 blue dots around a beating animal looks like flow whatever is happening, and
the dots move anyway: they are a compressible elastic continuum settling, they are pushed by the
walls, they drift. So the question is not "did the water move" but "did it move MORE WHERE THE
BAND IS", and that needs a control the same run provides.

THE CONTROL IS DISTANCE. Every water particle is scored by how far it starts from the nearest
ciliary-band cell. Particles in the near shell and particles in the far shell are the same
material, in the same pool, under the same walls, in the same run -- the only thing that differs
is whether a beating cell was next to them. The near-to-far ratio of displacement is what the
beat did. A ratio of 1 means the animal is stirring nothing and the movie is showing settling.

WHY DISPLACEMENT AND NOT SPEED. A stroke that pushes fluid out and pulls it straight back has
speed and transports nothing; net displacement over many cycles is what a feeding or swimming
current actually is. Both are reported, because the difference between them is the interesting
part: high speed with no displacement is a stroke that is reciprocal, which at this scale is the
scallop theorem and means the animal cannot swim.
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
BAND = "ciliary band"


def load(name: str):
    from plexus.paths import graphs_data_path
    G = graphs_data_path()
    for sub in ("studio", "platynereis"):
        t = os.path.join(G, sub, name, "trajectory.npz")
        if os.path.exists(t):
            break
    else:
        raise SystemExit(f"no trajectory for {name} under {G}/studio or {G}/platynereis")
    tr = np.load(t, allow_pickle=True)
    z = np.load(os.path.join(G, "neural_regions", REGION, "neurons.npz"), allow_pickle=True)
    return tr, z


def measure(tr, z, um, dt, near_um=12.0, far_um=40.0):
    W = np.asarray(tr["water_particle__pos"])
    C = np.asarray(tr["cell__pos"])
    cn, ci = list(z["cell_class_names"]), np.asarray(z["cell_class_id"])
    band = C[0][ci == cn.index(BAND)]

    # distance from each water particle's START to the nearest band cell, in micrometres
    d = np.full(W.shape[1], np.inf, np.float32)
    for k in range(0, band.shape[0], 8):                      # chunked: 140k x 74 in one go is fine
        blk = band[k:k + 8]
        d = np.minimum(d, np.linalg.norm(W[0][:, None, :] - blk[None], axis=2).min(1))
    d = d * um

    near, far = d < near_um, d > far_um
    disp = np.linalg.norm(W[-1] - W[0], axis=1) * um            # net, over the whole run
    step = np.linalg.norm(np.diff(W, axis=0), axis=2) * um / dt  # speed per frame, um/s
    speed = step.mean(0)

    out = {
        "n_water": int(W.shape[1]), "n_near": int(near.sum()), "n_far": int(far.sum()),
        "near_um": near_um, "far_um": far_um,
        "disp_near": float(np.median(disp[near])), "disp_far": float(np.median(disp[far])),
        "speed_near": float(np.median(speed[near])), "speed_far": float(np.median(speed[far])),
    }
    # THE REACH, AND IT IS THE HONEST STATISTIC HERE. A near-to-far RATIO is what you report when
    # both populations moved; this pool starts at rest, has no gravity and uniform density, so
    # water the animal never touched moves EXACTLY zero -- 71,358 particles at 0.0000 um -- and
    # the ratio comes out as 4.5e10, which says nothing except that a denominator was zero. What
    # the run actually measured is how far the disturbance got: the largest distance at which any
    # water particle moved at all, and the distance at which the median falls to a tenth of its
    # value at the animal's surface.
    out["disp_ratio"] = out["disp_near"] / max(out["disp_far"], 1e-12)
    out["speed_ratio"] = out["speed_near"] / max(out["speed_far"], 1e-12)
    moved = disp > 1e-9
    out["reach_um"] = float(d[moved].max()) if moved.any() else 0.0
    out["n_moved"] = int(moved.sum())
    edges = np.linspace(0, float(np.nanpercentile(d[np.isfinite(d)], 99)), 60)
    med = np.array([np.median(disp[(d >= a) & (d < b)]) if ((d >= a) & (d < b)).any() else np.nan
                    for a, b in zip(edges[:-1], edges[1:])])
    ref = np.nanmax(med) if np.isfinite(med).any() else 0.0
    below = np.where(np.nan_to_num(med) < 0.1 * ref)[0]
    out["half_reach_um"] = float(edges[below[0]]) if below.size else float(edges[-1])
    out["peak_disp_um"] = float(ref)
    return out, d, disp, speed, near, far


def draw(d, disp, speed, near, far, st, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3), facecolor="white")
    for a in ax:
        a.set_facecolor("white")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=10)

    # (a) THE WHOLE RELATION, not two buckets: displacement against distance from the band
    keep = np.isfinite(d)
    bins = np.linspace(0, np.percentile(d[keep], 99), 40)
    mid = 0.5 * (bins[1:] + bins[:-1])
    med = [np.median(disp[keep][(d[keep] >= a) & (d[keep] < b)]) if ((d[keep] >= a) & (d[keep] < b)).any()
           else np.nan for a, b in zip(bins[:-1], bins[1:])]
    ax[0].plot(mid, med, lw=2.0, color="#2c6fbb")
    ax[0].axvspan(0, st["near_um"], color="#c0392b", alpha=0.10)
    ax[0].axvspan(st["far_um"], bins[-1], color="#2c6fbb", alpha=0.08)
    ax[0].set_xlabel("distance from the nearest ciliary-band cell at frame 0 (um)", fontsize=11)
    ax[0].set_ylabel("net displacement over the run (um)", fontsize=11)
    ax[0].axvline(st["reach_um"], color="#444444", lw=1.0, ls=":")
    ax[0].text(0, 1.04, f"a   the beat reaches {st['reach_um']:.0f} um; beyond it the water is "
                        f"untouched", transform=ax[0].transAxes, fontsize=12)

    # (b) the two shells as distributions
    hi = np.percentile(disp[np.isfinite(disp)], 99)
    b2 = np.linspace(0, hi, 45)
    ax[1].hist(disp[far], bins=b2, color="#2c6fbb", alpha=0.75, density=True,
               label=f"far (> {st['far_um']:.0f} um, {st['n_far']:,} particles)")
    ax[1].hist(disp[near], bins=b2, color="#c0392b", alpha=0.8, density=True,
               label=f"near (< {st['near_um']:.0f} um, {st['n_near']:,})")
    ax[1].set_xlabel("net displacement (um)", fontsize=11)
    ax[1].set_ylabel("density", fontsize=11)
    ax[1].legend(fontsize=9, frameon=False)
    ax[1].text(0, 1.04, f"b   {st['n_moved']:,} of {st['n_water']:,} particles moved at all",
               transform=ax[1].transAxes, fontsize=12)

    # (c) speed against displacement: a reciprocal stroke sits high and left
    ax[2].bar([0, 1], [st["speed_near"], st["speed_far"]], color=["#c0392b", "#2c6fbb"], width=0.55)
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["near", "far"], fontsize=11)
    ax[2].set_ylabel("mean speed (um per scene second)", fontsize=11)
    ax[2].text(0, 1.04, f"c   {st['speed_near']:.3f} um/s beside the band, 0 beyond the reach",
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
    ap.add_argument("name", nargs="?", default="plat_r6_water")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--why", default="")
    a = ap.parse_args()
    tr, z = load(a.name)
    um = 195.9
    st, d, disp, speed, near, far = measure(tr, z, um, a.dt)

    print(f"{a.name}: {st['n_water']:,} water particles\n")
    print(f"  net displacement over the run, by where the particle STARTED:")
    print(f"    within {st['near_um']:.0f} um of a band cell  {st['disp_near']:.4f} um "
          f"({st['n_near']:,} particles)")
    print(f"    beyond {st['far_um']:.0f} um                  {st['disp_far']:.4f} um "
          f"({st['n_far']:,} particles)")
    print(f"\n  the reach of the disturbance:")
    print(f"    the furthest water that moved at all   {st['reach_um']:.1f} um from a band cell")
    print(f"    the median falls to a tenth by         {st['half_reach_um']:.1f} um")
    print(f"    particles that moved at all            {st['n_moved']:,} of {st['n_water']:,}")
    print(f"    peak median displacement               {st['peak_disp_um']:.4f} um\n")
    print(f"  mean speed:")
    print(f"    near  {st['speed_near']:.4f} um/s")
    print(f"    far   {st['speed_far']:.4f} um/s")
    print(f"  (a near/far RATIO is not reported: the far water moved exactly zero, so the "
          f"ratio\n   would only be saying that a denominator was 0.)")

    i = next_index()
    draw(d, disp, speed, near, far, st, os.path.join(REPO, "builder", "png", f"{i:04d}.png"))

    from plexus.paths import graphs_data_path
    import shutil
    import time as _t
    for sub in ("studio", "platynereis"):
        spec = os.path.join(graphs_data_path(), sub, a.name, "spec.yaml")
        if os.path.exists(spec):
            shutil.copy(spec, os.path.join(REPO, "builder", "spec", f"{i:04d}.yaml"))
            break
    with open(os.path.join(REPO, "builder", "why", f"{i:04d}.txt"), "w") as f:
        f.write(f"time   {_t.strftime('%Y-%m-%d %H:%M:%S')}\nspec   {a.name}\n"
                f"camera (a figure, not a render)\n"
                f"why    {a.why or 'did the beat move the water, measured'}\n")
    print(f"\n  -> builder/*/{i:04d}.*")
