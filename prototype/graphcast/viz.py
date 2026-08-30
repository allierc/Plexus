"""Figures. Every gate points to one, because the failures that matter are visible and not scalar.

Three defects were found while building the toy, and each was a number that looked plausible:

  * the circuit sat at a fixed point -- spatial spread 9.29, temporal spread 0.065, and the
    increments were pure noise. A trace panel shows it instantly;
  * the stimulus field was IDENTICALLY ZERO for three generations, because `activation_pulse`
    reads a clock from `H.signals` that no operator was writing. A field panel shows it instantly;
  * spatial type purity read 6.1x chance, which was the grid being finer than the sampling rather
    than any structure. A positions-coloured-by-type panel shows it instantly.

So the figures here are not documentation of a result; they are the instrument that finds the
defect. `gates.Gate.record` refuses to pass a gate with no artifact for that reason.

Style follows the house convention for this project: black background, no titles, white top-left
panel labels.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG = "#000000"
FG = "#ffffff"
LABEL_KW = dict(color=FG, fontsize=11, fontweight="bold", va="top", ha="left")


def _panel(ax, label):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color("#444444")
    ax.tick_params(colors="#888888", labelsize=7)
    ax.text(0.02, 0.98, label, transform=ax.transAxes, **LABEL_KW)


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------- #
#  stage 0
# --------------------------------------------------------------------------------------- #

def option_matrix(combos, ok_flags, path):
    """G1: the 24 option combinations, one cell each, green when the spec parses."""
    fig, ax = plt.subplots(figsize=(7.5, 3.2), facecolor=BG)
    _panel(ax, "a  option combinations")
    labels = [f"{c['encoder_decoder']}/{c['message'][:5]}/p{c['n_passes']}/{c['embedding'][:4]}"
              for c in combos]
    n = len(combos)
    cols = 6
    for i, (lab, ok) in enumerate(zip(labels, ok_flags)):
        r, c = divmod(i, cols)
        ax.add_patch(plt.Rectangle((c, -r), 0.96, 0.9,
                                   color="#2ea043" if ok else "#cf222e"))
        ax.text(c + 0.48, -r + 0.45, lab, ha="center", va="center", color=BG, fontsize=5.5)
    ax.set_xlim(-0.1, cols + 0.1)
    ax.set_ylim(-(n // cols) - 0.1, 1.1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, 0.03, f"{sum(ok_flags)}/{n} parse", transform=ax.transAxes,
            color=FG, fontsize=9)
    return _save(fig, path)


def scan_coverage(counts, offenders, path):
    """G2: how much code was scanned for dataset identity, and what it found."""
    fig, ax = plt.subplots(figsize=(6.5, 3.0), facecolor=BG)
    _panel(ax, "a  files scanned for dataset identity")
    names = list(counts)
    ax.barh(range(len(names)), [counts[k] for k in names], color="#0969da")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, color="#cccccc", fontsize=7)
    ax.set_xlabel("lines scanned", color="#cccccc", fontsize=8)
    ax.text(0.98, 0.05, f"{len(offenders)} offending", transform=ax.transAxes,
            color="#2ea043" if not offenders else "#cf222e", fontsize=10, ha="right")
    return _save(fig, path)


def unit_ladder(units, path):
    """G7: the declared scales and what each derived quantity is therefore denominated in."""
    fig, ax = plt.subplots(figsize=(6.0, 3.2), facecolor=BG)
    _panel(ax, "a  declared scale, and what it makes available")
    rows = [("length_um", f"{units.length_um:g} um per length unit"),
            ("time_s", f"{units.time_s:g} s per time unit"),
            ("force_nN", "ratios only" if units.force_nN is None else f"{units.force_nN:g} nN"),
            ("-> rate", f"{units.rate_per_s:g} per s"),
            ("-> velocity", f"{units.velocity_um_per_s:g} um/s")]
    for i, (k, v) in enumerate(rows):
        ax.text(0.03, 0.82 - 0.16 * i, k, color="#58a6ff", fontsize=9, family="monospace")
        ax.text(0.42, 0.82 - 0.16 * i, v, color=FG, fontsize=9, family="monospace")
    ax.set_xticks([]); ax.set_yticks([])
    return _save(fig, path)


# --------------------------------------------------------------------------------------- #
#  stage 1 -- the toy
# --------------------------------------------------------------------------------------- #

def toy_summary(gt, path, purity_by_res=None):
    """G16 and the toy's own sanity: four panels that would have caught all three defects."""
    pos, nt, v = gt["positions"], gt["node_type"], gt["voltage"]
    dist, w = gt["distance"], gt["weights"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), facecolor=BG)

    ax = axes[0, 0]; _panel(ax, "a  positions coloured by type")
    ax.scatter(pos[:, 0], pos[:, 1], c=nt, cmap="tab10", s=7)
    ax.set_aspect("equal")

    ax = axes[0, 1]; _panel(ax, "b  interaction kernel vs distance")
    ax.scatter(dist, w, s=2, c="#58a6ff", alpha=0.25)
    ax.axhline(0, color="#666666", lw=0.6)
    ax.set_xlabel("distance (length units)", color="#cccccc", fontsize=8)
    ax.set_ylabel("edge weight", color="#cccccc", fontsize=8)

    ax = axes[1, 0]; _panel(ax, "c  voltage traces, 12 neurons")
    for i in np.linspace(0, v.shape[1] - 1, 12).astype(int):
        ax.plot(v[:, i], lw=0.6, alpha=0.85)
    ax.set_xlabel("frame", color="#cccccc", fontsize=8)

    ax = axes[1, 1]; _panel(ax, "d  spatial type purity / permutation null")
    if purity_by_res:
        res = sorted(purity_by_res)
        ax.plot(res, [purity_by_res[r] for r in res], "o-", color="#2ea043")
        ax.axhline(1.0, color=FG, lw=0.8, ls="--")
        ax.axhline(1.2, color="#cf222e", lw=0.8, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("cells per axis", color="#cccccc", fontsize=8)
        ax.set_ylim(0, max(1.4, max(purity_by_res.values()) * 1.1))
    return _save(fig, path)


def state_movie(v, pos, path, stride=8, fps=20):
    """An MP4 of the state over time. A fixed point is unmistakable here and nowhere else."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lim = float(np.abs(v).max()) or 1.0
    frames = []
    for t in range(0, v.shape[0], stride):
        fig, ax = plt.subplots(figsize=(4.2, 4.2), facecolor=BG)
        _panel(ax, f"t = {t}")
        ax.scatter(pos[:, 0], pos[:, 1], c=v[t], cmap="coolwarm", vmin=-lim, vmax=lim, s=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(buf.copy())
        plt.close(fig)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    return path
