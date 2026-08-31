r"""Figures. Every gate points to one, because the failures that matter are visible and not scalar.

Three defects were found while building the toy, and each was a plausible-looking number:

  * the circuit sat at a fixed point -- spatial spread 9.29, temporal spread 0.065, and the
    increments were pure noise. A trace panel shows it instantly;
  * the stimulus field was IDENTICALLY ZERO for three generations, because `activation_pulse`
    reads a clock from `H.signals` that no operator was writing. A field panel shows it instantly;
  * spatial type purity read 6.1x chance, which was the grid being finer than the sampling rather
    than any structure. A positions-coloured-by-type panel shows it instantly.

So these are not documentation of a result; they are the instrument that finds the defect.
`gates.Gate.record` refuses to pass a gate with no artifact for that reason.

STYLE, and why it is enforced here rather than per figure.

  * WHITE background, black text -- these are read printed, in a document, not on a screen.
  * Panel labels sit ABOVE the axes, not inside the data area, where they can never collide with
    a point or a line. Regular weight, not bold: at this size bold reads as emphasis the label has
    not earned, and the position already distinguishes it.
  * ONE FIGURE WIDTH for every figure (`FIGW`). Every figure in the note is included at
    \linewidth, so a figure drawn 13 inches wide is scaled down twice as hard as one drawn 6.5
    inches wide and its 9pt label lands at half the size on the page. Fixing the width is what
    makes a single point size read the same everywhere; setting the same `fontsize=` in each call
    does not, and that is the mistake this constant exists to prevent. Panel HEIGHT may vary.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGW = 7.0          # inches -- IDENTICAL for every figure, so \linewidth scales them all alike
FS = 9              # the one point size; everything else is derived from it

BG = "#ffffff"
FG = "#111111"
GRIDC = "#c8c8c8"

plt.rcParams.update({
    "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": "#555555",
    "xtick.color": FG, "ytick.color": FG,
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS - 1,
    "ytick.labelsize": FS - 1, "legend.fontsize": FS - 1,
    "axes.titlesize": FS, "axes.linewidth": 0.8,
    "lines.linewidth": 1.0, "figure.dpi": 140,
})


def _panel(ax, label):
    """Label ABOVE the axes, left-aligned, regular weight. Never inside the data area."""
    ax.set_title(label, loc="left", fontsize=FS, color=FG, pad=4)
    ax.tick_params(labelsize=FS - 1)


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------- #
#  stage 0
# --------------------------------------------------------------------------------------- #

def option_matrix(combos, ok_flags, path):
    """G1: the 24 option combinations, one cell each, green when the spec parses."""
    fig, ax = plt.subplots(figsize=(FIGW, 3.0), facecolor=BG)
    _panel(ax, "a  option combinations")
    labels = [f"{c['encoder_decoder']}/{c['message'][:5]}/p{c['n_passes']}/{c['embedding'][:4]}"
              for c in combos]
    n = len(combos)
    cols = 6
    for i, (lab, ok) in enumerate(zip(labels, ok_flags)):
        r, c = divmod(i, cols)
        ax.add_patch(plt.Rectangle((c, -r), 0.96, 0.9,
                                   color="#2ea043" if ok else "#cf222e"))
        ax.text(c + 0.48, -r + 0.45, lab, ha="center", va="center", color="#ffffff", fontsize=FS-3.5)
    ax.set_xlim(-0.1, cols + 0.1)
    ax.set_ylim(-(n // cols) - 0.1, 1.1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, 0.03, f"{sum(ok_flags)}/{n} parse", transform=ax.transAxes,
            color=FG, fontsize=FS)
    return _save(fig, path)


def scan_coverage(counts, offenders, path):
    """G2: how much code was scanned for dataset identity, and what it found."""
    fig, ax = plt.subplots(figsize=(FIGW, 3.2), facecolor=BG)
    _panel(ax, "a  files scanned for dataset identity")
    names = list(counts)
    ax.barh(range(len(names)), [counts[k] for k in names], color="#0969da")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, color=FG, fontsize=FS-1)
    ax.set_xlabel("lines scanned", color=FG, fontsize=FS)
    ax.text(0.98, 0.05, f"{len(offenders)} offending", transform=ax.transAxes,
            color="#2ea043" if not offenders else "#cf222e", fontsize=FS, ha="right")
    return _save(fig, path)


def unit_ladder(units, path):
    """G7: the declared scales and what each derived quantity is therefore denominated in."""
    fig, ax = plt.subplots(figsize=(FIGW, 2.6), facecolor=BG)
    _panel(ax, "a  declared scale, and what it makes available")
    rows = [("length_um", f"{units.length_um:g} um per length unit"),
            ("time_s", f"{units.time_s:g} s per time unit"),
            ("force_nN", "ratios only" if units.force_nN is None else f"{units.force_nN:g} nN"),
            ("-> rate", f"{units.rate_per_s:g} per s"),
            ("-> velocity", f"{units.velocity_um_per_s:g} um/s")]
    for i, (k, v) in enumerate(rows):
        ax.text(0.03, 0.82 - 0.16 * i, k, color="#58a6ff", fontsize=FS, family="monospace")
        ax.text(0.42, 0.82 - 0.16 * i, v, color=FG, fontsize=FS, family="monospace")
    ax.set_xticks([]); ax.set_yticks([])
    return _save(fig, path)


# --------------------------------------------------------------------------------------- #
#  stage 1 -- the toy
# --------------------------------------------------------------------------------------- #

def toy_summary(gt, path, purity_by_res=None):
    """G16 and the toy's own sanity: four panels that would have caught all three defects."""
    pos, nt, v = gt["positions"], gt["node_type"], gt["voltage"]
    dist = gt["distance"]
    w = gt["weights"] if "weights" in gt.files else None
    fig, axes = plt.subplots(2, 2, figsize=(FIGW, 5.6), facecolor=BG)

    ax = axes[0, 0]; _panel(ax, "a  positions coloured by type")
    ax.scatter(pos[:, 0], pos[:, 1], c=nt, cmap="tab10", s=7)
    ax.set_aspect("equal")

    ax = axes[0, 1]
    if w is not None:
        _panel(ax, "b  interaction kernel vs distance")
        ax.scatter(dist, w, s=2, c="#1f77b4", alpha=0.25)
        ax.axhline(0, color=GRIDC, lw=0.6)
        ax.set_ylabel("edge weight", color=FG, fontsize=FS)
    else:
        _panel(ax, "b  signed gain in space (the heterogeneity)")
        lim = float(np.abs(gt["gain"]).max()) or 1.0
        ax.scatter(pos[:, 0], pos[:, 1], c=gt["gain"], cmap="coolwarm", vmin=-lim, vmax=lim, s=8)
        ax.set_aspect("equal")
    ax.set_xlabel("distance (length units)" if w is not None else "x", color=FG, fontsize=FS)

    ax = axes[1, 0]; _panel(ax, "c  voltage traces, 12 neurons")
    for i in np.linspace(0, v.shape[1] - 1, 12).astype(int):
        ax.plot(v[:, i], lw=0.6, alpha=0.85)
    ax.set_xlabel("frame", color=FG, fontsize=FS)

    ax = axes[1, 1]; _panel(ax, "d  spatial type purity / permutation null")
    if purity_by_res:
        res = sorted(purity_by_res)
        ax.plot(res, [purity_by_res[r] for r in res], "o-", color="#2ea043")
        ax.axhline(1.0, color=FG, lw=0.8, ls="--")
        ax.axhline(1.2, color="#cf222e", lw=0.8, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("cells per axis", color=FG, fontsize=FS)
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
        fig, ax = plt.subplots(figsize=(FIGW, FIGW), facecolor=BG)
        _panel(ax, f"state,  frame {t}")
        ax.scatter(pos[:, 0], pos[:, 1], c=v[t], cmap="coolwarm", vmin=-lim, vmax=lim, s=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(buf.copy())
        plt.close(fig)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    return path


# --------------------------------------------------------------------------------------- #
#  stage 1b -- the toy as a test bed: the field, the heterogeneity, the identifiability
# --------------------------------------------------------------------------------------- #

def field_movie(grid, path, fps=20, stride=1):
    """The coarse field over time. A wave that is not travelling, or is identically zero, is
    unmistakable here -- and both of those shipped past a scalar check on an earlier toy."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    g = np.asarray(grid)
    if g.ndim == 4:
        g = g[:, 0]
    lim = float(np.abs(g).max()) or 1.0
    frames = []
    for t in range(0, g.shape[0], stride):
        fig, ax = plt.subplots(figsize=(FIGW, FIGW * 0.95), facecolor=BG)
        _panel(ax, f"coarse field u(x,y),  frame {t}")
        ax.imshow(g[t].T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_xticks([]); ax.set_yticks([])
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    return path


def heterogeneity_map(pos, gain, node_type, path):
    """Where the heterogeneity lives: the signed per-node gain, in space and by type."""
    fig, axes = plt.subplots(1, 3, figsize=(FIGW, 2.6), facecolor=BG)
    lim = float(np.abs(gain).max()) or 1.0

    ax = axes[0]; _panel(ax, "a  signed gain in space")
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=gain, cmap="coolwarm", vmin=-lim, vmax=lim, s=9)
    ax.set_aspect("equal")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046); cb.ax.tick_params(colors="#aaaaaa", labelsize=7)

    ax = axes[1]; _panel(ax, "b  gain by type")
    for k in range(int(node_type.max()) + 1):
        m = node_type == k
        ax.scatter(np.full(m.sum(), k), gain[m], s=5, alpha=0.5)
    ax.axhline(0, color=FG, lw=0.7, ls="--")
    ax.set_xlabel("type", color=FG, fontsize=FS)
    ax.set_ylabel("gain", color=FG, fontsize=FS)

    ax = axes[2]; _panel(ax, "c  type in space (must look random)")
    ax.scatter(pos[:, 0], pos[:, 1], c=node_type, cmap="tab10", s=9)
    ax.set_aspect("equal")
    return _save(fig, path)


def identifiability_panels(stats, path):
    """The four stage-1b numbers, each as a picture rather than a scalar."""
    fig, axes = plt.subplots(2, 2, figsize=(FIGW, 5.6), facecolor=BG)

    ax = axes[0, 0]; _panel(ax, "a  per-node R2 of dv on (v, grad u)")
    ax.hist(stats["r2_rule"], bins=40, color="#2ea043")
    ax.axvline(0.90, color="#cf222e", ls=":", lw=1.0)
    ax.set_xlabel("R2", color=FG, fontsize=FS)

    ax = axes[0, 1]; _panel(ax, "b  gradient from neighbours")
    ax.hist(stats["r2_grad_nb"], bins=40, color="#58a6ff")
    ax.axvline(0.95, color="#cf222e", ls=":", lw=1.0)
    ax.set_xlabel("R2", color=FG, fontsize=FS)

    ax = axes[1, 0]; _panel(ax, "c  fitted gain vs true gain")
    ax.scatter(stats["gain_true"], stats["gain_fit"], s=6, c="#d29922", alpha=0.6)
    ax.set_xlabel("true g_i", color=FG, fontsize=FS)
    ax.set_ylabel("fitted", color=FG, fontsize=FS)

    ax = axes[1, 1]; _panel(ax, "d  |corr| between connected nodes")
    ax.hist(stats["nb_corr"], bins=40, color="#8957e5")
    ax.axvline(0.80, color="#cf222e", ls=":", lw=1.0)
    ax.set_xlabel("|Pearson r|", color=FG, fontsize=FS)
    return _save(fig, path)


def necessity_panel(r2_local, r2_neighbour, path, withheld: bool, coarse: str):
    """G26: side by side, what a node can do alone against what it can do with its neighbours.

    A test bed is only a test of a graph model if the left distribution sits well below the right
    one. On the first wave toy they coincided, because for a travelling wave du/dx = -(1/c) du/dt
    and the drive alone determines the gradient.
    """
    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 2.7))

    ax = axes[0]
    _panel(ax, "a  node-local baseline (no neighbours)")
    ax.hist(r2_local, bins=40, color="#c0504d")
    ax.axvline(0.50, color="#333333", ls=":", lw=1.0)
    ax.set_xlabel("R2 of dv from the node's own history"); ax.set_xlim(-0.05, 1.05)

    ax = axes[1]
    _panel(ax, "b  with neighbours")
    ax.hist(r2_neighbour, bins=40, color="#4f81bd")
    ax.axvline(0.90, color="#333333", ls=":", lw=1.0)
    ax.set_xlabel("R2 of dv from (v, grad u)"); ax.set_xlim(-0.05, 1.05)

    fig.suptitle(f"coarse rule: {coarse}   |   drive "
                 f"{'withheld' if withheld else 'observed'}", fontsize=FS, y=1.04)
    return _save(fig, path)
