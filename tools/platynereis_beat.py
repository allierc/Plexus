#!/usr/bin/env python
"""Did the ciliary band beat? The measurement, not the impression.

    python tools/platynereis_beat.py plat_r5_beat

"The cilia move" is a claim. "The band cells' radial excursion is 1.4 um peak-to-peak at 0.48 Hz,
against 0.06 um for cells of every other class" is a result, and only the second can be wrong in
a way anyone notices.

WHAT IS MEASURED, and why it is the radial component and not the speed. The stroke is an active
stress along each band cell's own outward radius, so the motion it is supposed to produce is
along that radius too. Total speed would also count the cell being carried by its neighbours, by
the body settling, by anything at all; the projection onto the cell's own stroke axis counts only
what the stroke did. The axis is recomputed from the cell's own start position, exactly as
`radial_polarity` wrote it, rather than read back out of a block, so this measurement does not
depend on the operator under test having stored anything correctly.

THE CONTROL is every cell of every other class, which carries the same membrane law, the same
noise, the same material and the same MPM grid, and no polarity -- so it has no stroke axis and
`polar_active_stress[driven]` gives it nothing. Whatever separates the two populations is the
beat. If they ever converge, the motor pathway is not delivering, however busy the movie looks.
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
    t = os.path.join(G, "studio", name, "trajectory.npz")
    if not os.path.exists(t):
        raise SystemExit(f"no trajectory at {t} -- run the spec first")
    tr = np.load(t, allow_pickle=True)
    z = np.load(os.path.join(G, "neural_regions", REGION, "neurons.npz"), allow_pickle=True)
    return tr, z


def radial_excursion(P, axis=2, centre=(0.5, 0.5)):
    """Each cell's displacement projected on its OWN outward radius, in world units. [T, N]."""
    flat = [i for i in range(3) if i != axis]
    r0 = P[0][:, flat] - np.asarray(centre)[None, :]
    n = r0 / np.maximum(np.linalg.norm(r0, axis=1, keepdims=True), 1e-12)
    d = P[:, :, flat] - P[0][None, :, flat]
    return np.einsum("tnk,nk->tn", d, n)


def dominant_frequency(x, dt):
    """The strongest non-zero frequency in a [T] signal, in hertz, and its share of the power."""
    x = x - x.mean()
    if x.std() < 1e-14:
        return 0.0, 0.0
    F = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), d=dt)
    F[0] = 0.0
    k = int(np.argmax(F))
    return float(f[k]), float(F[k] / max(F.sum(), 1e-30))


def measure(tr, z, dt, um, settle=0.25):
    P = np.asarray(tr["cell__pos"])
    # The membrane state lives on the `neuron` set, one per cell, because voltage is a neuron's
    # coordinate and pos is a body's -- see the spec's own note. Either name is accepted so a
    # trajectory from before the split still reads.
    vk = next((k for k in ("neuron__voltage", "cell__voltage") if k in tr.files), None)
    V = np.asarray(tr[vk])[:, :, 0] if vk else None
    cn, ci = list(z["cell_class_names"]), np.asarray(z["cell_class_id"])
    band = ci == cn.index(BAND)
    e = radial_excursion(P)
    t0 = int(settle * e.shape[0])                       # drop the settling transient
    ptp = (e[t0:].max(0) - e[t0:].min(0)) * um          # peak-to-peak excursion, in micrometres

    out = {"n_frames": int(P.shape[0]), "n_band": int(band.sum()),
           "band_ptp_um": float(np.median(ptp[band])),
           "other_ptp_um": float(np.median(ptp[~band])),
           "band_ptp_um_max": float(ptp[band].max())}
    out["separation"] = out["band_ptp_um"] / max(out["other_ptp_um"], 1e-12)

    fs, ws = [], []
    for i in np.where(band)[0]:
        f, w = dominant_frequency(e[t0:, i], dt)
        if f > 0:
            fs.append(f); ws.append(w)
    out["freq_hz"] = float(np.median(fs)) if fs else 0.0
    out["freq_power_share"] = float(np.median(ws)) if ws else 0.0
    if V is not None:
        out["band_drive"] = float(np.abs(V[t0:][:, band]).mean())
        out["other_drive"] = float(np.abs(V[t0:][:, ~band]).mean())
    return out, e, band, ptp, t0


def draw(e, band, ptp, t0, dt, um, stats, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(e.shape[0]) * dt
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3), facecolor="white")
    for a in ax:
        a.set_facecolor("white")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=10)

    # (a) eight band cells' own strokes, and the same number of controls behind them
    idx = np.where(band)[0]
    for i in idx[:: max(1, len(idx) // 8)][:8]:
        ax[0].plot(t, e[:, i] * um, lw=1.1, color="#e24a6a", alpha=0.85)
    oth = np.where(~band)[0]
    for i in oth[:: max(1, len(oth) // 8)][:8]:
        ax[0].plot(t, e[:, i] * um, lw=1.0, color="#9aa3b0", alpha=0.6)
    ax[0].axvline(t0 * dt, color="#bbbbbb", lw=0.8, ls="--")
    ax[0].set_xlabel("time (scene seconds)", fontsize=11)
    ax[0].set_ylabel("displacement along the cell's own stroke axis (um)", fontsize=11)
    ax[0].text(0, 1.04, "a   eight band cells (red) and eight of every other class (grey)",
               transform=ax[0].transAxes, fontsize=12)

    # (b) THE CONTROL, as a distribution rather than a mean: two populations, one number each
    # `ptp` IS ALREADY IN MICROMETRES -- `measure` multiplied by `um` when it built it. Applying
    # the conversion again here put a 0.05 um excursion on an axis reading 10 um, so the panel
    # disagreed with the number printed beside it by a factor of 195.9.
    bins = np.linspace(0, max(np.percentile(ptp, 99.5), 1e-6), 45)
    ax[1].hist(ptp[~band], bins=bins, color="#2c6fbb", alpha=0.75,
               label=f"every other class ({int((~band).sum()):,} cells)", density=True)
    ax[1].hist(ptp[band], bins=bins, color="#c0392b", alpha=0.8,
               label=f"ciliary band ({int(band.sum())} cells)", density=True)
    ax[1].set_xlabel("peak-to-peak excursion (um)", fontsize=11)
    ax[1].set_ylabel("density", fontsize=11)
    ax[1].legend(fontsize=9, frameon=False)
    ax[1].text(0, 1.04,
               f"b   the band moves {stats['separation']:.1f}x further than anything else",
               transform=ax[1].transAxes, fontsize=12)

    # (c) is it periodic? the band's mean stroke and its spectrum
    m = e[t0:][:, band].mean(1) * um
    ax[2].plot(np.arange(len(m)) * dt, m, lw=1.4, color="#c0392b")
    ax[2].set_xlabel("time after settling (scene seconds)", fontsize=11)
    ax[2].set_ylabel("band mean displacement (um)", fontsize=11)
    ax[2].text(0, 1.04,
               f"c   {stats['freq_hz']:.3f} Hz, {100 * stats['freq_power_share']:.0f}% of the "
               f"power in that line", transform=ax[2].transAxes, fontsize=12)

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
    ap.add_argument("name", nargs="?", default="plat_r5_beat")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--why", default="")
    a = ap.parse_args()
    tr, z = load(a.name)
    um = 195.9
    st, e, band, ptp, t0 = measure(tr, z, a.dt, um)

    print(f"{a.name}: {st['n_frames']} frames, {st['n_band']} ciliary-band cells\n")
    print(f"  peak-to-peak excursion along each cell's own stroke axis, after settling:")
    print(f"    ciliary band       {st['band_ptp_um']:.4f} um  (largest {st['band_ptp_um_max']:.4f})")
    print(f"    every other class  {st['other_ptp_um']:.4f} um  <- no polarity, no stroke axis")
    print(f"    the band moves     {st['separation']:.1f}x further\n")
    print(f"  periodicity of the band's stroke:")
    print(f"    {st['freq_hz']:.4f} Hz, {100 * st['freq_power_share']:.0f}% of the power in "
          f"that one line")
    if "band_drive" in st:
        print(f"\n  the drive that produced it (mean |membrane state|):")
        print(f"    ciliary band       {st['band_drive']:.4f}")
        print(f"    every other class  {st['other_drive']:.4f}")

    i = next_index()
    png = os.path.join(REPO, "builder", "png", f"{i:04d}.png")
    draw(e, band, ptp, t0, a.dt, um, st, png)

    from plexus.paths import graphs_data_path
    import shutil
    import time as _t
    spec = os.path.join(graphs_data_path(), "studio", a.name, "spec.yaml")
    if os.path.exists(spec):
        shutil.copy(spec, os.path.join(REPO, "builder", "spec", f"{i:04d}.yaml"))
    with open(os.path.join(REPO, "builder", "why", f"{i:04d}.txt"), "w") as f:
        f.write(f"time   {_t.strftime('%Y-%m-%d %H:%M:%S')}\nspec   {a.name}\n"
                f"camera (a figure, not a render)\n"
                f"why    {a.why or 'did the ciliary band beat, measured'}\n")
    print(f"\n  -> builder/*/{i:04d}.*")
