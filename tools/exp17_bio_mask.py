"""exp17 batch 25: THE BIOLOGY INPUT MASK (Cedric, 2026-10-09, his biologist colleagues' list of each block's input
neurons). Local, model-free.

    gain       sensory  pretectum
    dots       sensory  tectum
    flash      sensory  tectum, dorsal thalamus, pretectum
    taxis      sensory  tectum, dorsal thalamus, pretectum
    turning    sensory  pretectum
    position   sensory  pretectum
    open loop  sensory  pretectum
    rotation   sensory  pretectum
    dark       motor efference  rhombomere 7

The regions are Z-Brain masks (Randlett et al. 2015) read per neuron from data/atlas_destripe.npz (tools/exp17_atlas.py,
Cedric's BigWarp registration):
  pretectum        every mask whose name contains "pretect": Pretectum, Anterior pretectum cluster of vmat2 Neurons,
                   Pretectal Gad1b Cluster, Pretectal dopaminergic cluster, Migrated Area of the Pretectum (M1)
  tectum           Mesencephalon - Tectum Stratum Periventriculare and Medial Tectal Band (the cell-body layers; the
                   neuropil has no somata)
  dorsal thalamus  Diencephalon - Dorsal Thalamus
  rhombomere 7     Rhombencephalon - Rhombomere 7
A neuron outside the Z-Brain brain (atlas `inside` false) is in no region.

THE COLUMNS. A block's feature columns that vary within the block (the `features` of 22.3's markall mask:
input_mask_destripe_bal20_markall.npz, every column not on every neuron) enter only the block's regions. The block's
condition marker (the one column constant while the block is on: f1 f3 f5 f8 f12 f17 f18 f20 f21) enters every
neuron, as in 22.3, so the per-block baseline is unchanged; open loop's only column is its marker, so its pretectum
entry routes nothing. With `--efference`: the 5 ephys columns of zapbench_destripe_ephys (swim_L, swim_R, turn_L,
turn_R, bout; tools/export_ephys_features.py) enter rhombomere 7, in every block (the mask routes per neuron, not
per frame; the fish swims in every block). With `--no-markers` (Cedric, 2026-10-09: "I do not like to learn the
offset through a sum of two learnables"): the 9 marker columns enter NO neuron -- for 24.10's law, whose rest_per_block
already gives every neuron one offset per block (rest_block), so the offset is one learnable, not rest_block plus a
marker weight. With `--permute SEED`: the same mask with its neuron rows shuffled
(non-marker columns), the control with the same count per column and the same column overlaps at random positions.

    PYTHONPATH=src:tools python tools/exp17_bio_mask.py --out input_mask_destripe_bio.npz
    PYTHONPATH=src:tools python tools/exp17_bio_mask.py --efference --out input_mask_destripe_bio_eff.npz
    PYTHONPATH=src:tools python tools/exp17_bio_mask.py --efference --permute 0 --out input_mask_destripe_bio_eff_perm.npz
-> graphs_data/zebrafish/<out>: mask [N] (any non-marker column), mask_by_input [N, 22 or 27], region_counts (json)

    PYTHONPATH=src:tools python tools/exp17_bio_mask.py --figure
-> presentation/figs/bio_mask_regions.png: the four regions on the fish, atlas frame, from above and from the side
   (tools/exp17_vrest_blocks.py's views), every other neuron grey (Cedric, 2026-10-09: "one figure regions highlighted
   in the atlas fish two view")
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
MARKALL = "input_mask_destripe_bal20_markall.npz"
BLOCK_REGIONS = {"gain": ("pretectum",), "dots": ("tectum",), "flash": ("tectum", "dorsal thalamus", "pretectum"),
                 "taxis": ("tectum", "dorsal thalamus", "pretectum"), "turning": ("pretectum",),
                 "position": ("pretectum",), "open loop": ("pretectum",), "rotation": ("pretectum",), "dark": ()}
N_EPHYS = 5


def regions():
    A = np.load(os.path.join(EXP, "data", "atlas_destripe.npz"), allow_pickle=True)
    names, R = [str(x) for x in A["names"]], A["regions"] & A["inside"][:, None]
    col = lambda n: R[:, names.index(n)]
    return {"pretectum": R[:, [i for i, n in enumerate(names) if "pretect" in n.lower()]].any(1),
            "tectum": col("Mesencephalon - Tectum Stratum Periventriculare") | col("Mesencephalon - Medial Tectal Band"),
            "dorsal thalamus": col("Diencephalon - Dorsal Thalamus"),
            "rhombomere 7": col("Rhombencephalon - Rhombomere 7")}


def build(out, efference=False, permute=None, markers=True):
    from plexus.paths import graphs_data_path
    z = np.load(graphs_data_path("zebrafish", "zapbench_destripe_recording.npz"), mmap_mode="r")
    U, off, names = z["stimulus"], z["offsets"], [str(x) for x in z["names"]]
    mk = np.load(graphs_data_path("zebrafish", MARKALL))["mask_by_input"]
    N, F = mk.shape
    marker = mk.min(0) > 0                                    # the 9 condition markers: on every neuron in 22.3
    block = np.array([next(k for k in range(len(off) - 1) if np.abs(U[off[k]:off[k + 1], j]).max() > 0) for j in range(F)])
    reg = regions()
    mbi = np.zeros((N, F + (N_EPHYS if efference else 0)), np.float32)
    free = np.r_[~marker, np.ones(N_EPHYS if efference else 0, bool)]   # the columns the regions route
    mbi[:, ~free] = 1.0 if markers else 0.0          # --no-markers: a law with rest_per_block owns the per-block offset
    counts = {r: int(m.sum()) for r, m in reg.items()}
    for j in np.where(~marker)[0]:
        rs = BLOCK_REGIONS[names[block[j]]]
        mbi[:, j] = np.any([reg[r] for r in rs], 0) if rs else 0.0
    if efference:
        mbi[:, F:] = reg["rhombomere 7"][:, None]
    if permute is not None:
        mbi[:, free] = mbi[np.random.default_rng(permute).permutation(N)][:, free]
    mask = (mbi[:, free] > 0).any(1)
    per_col = {f"f{j}" if j < F else f"ephys{j - F}": int(mbi[:, j].sum()) for j in range(mbi.shape[1])}
    np.savez(graphs_data_path("zebrafish", out), mask=mask.astype(np.float32), mask_by_input=mbi,
             region_counts=json.dumps(counts), per_column=json.dumps(per_col), block_regions=json.dumps(BLOCK_REGIONS),
             efference=efference, permute=-1 if permute is None else permute, markers=markers)
    print(f"[bio mask] {out}: {int(mask.sum()):,} of {N:,} neurons receive a non-marker column; markers "
          f"{[f'f{j}' for j in np.where(marker)[0]]} " + ("on every neuron" if markers else "on NO neuron"))
    print("   regions:", counts)
    for j in np.where(free)[0]:
        nm = names[block[j]] if j < F else "ephys"
        print(f"   {('f%d' % j) if j < F else 'ephys%d' % (j - F):7s} {nm:10s} {int(mbi[:, j].sum()):7,d} neurons")
    return mbi


REGION_COLOURS = (("tectum", "#ff9e1a", "dots, flash, taxis"), ("rhombomere 7", "#c77dff", "motor efference (25.2)"),
                  ("pretectum", "#ff3df2", "gain, flash, taxis, turning, position, rotation"),
                  ("dorsal thalamus", "#00d4ff", "flash, taxis"))                   # drawn in this order, smallest last


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = np.load(os.path.join(EXP, "data", "atlas_destripe.npz"), allow_pickle=True)
    P, ins = A["atlas_um"].astype(np.float64), A["inside"]
    reg = regions()
    xd, yd, zd = P[:, 1], (621 - 1) * 0.798 - P[:, 0], P[:, 2]      # exp17_vrest_blocks.movie's top and side views
    ext = lambda v: (float(v[ins].min()) - 5.0, float(v[ins].max()) + 5.0)                       # noqa: E731
    (x0, x1), (y0, y1), (z0, z1) = ext(xd), ext(yd), ext(zd)
    W_ = 8.0
    ht, hs = W_ * (y1 - y0) / (x1 - x0), W_ * (z1 - z0 + 40) / (x1 - x0)
    H = ht + hs + 1.0
    fig = plt.figure(figsize=(W_ + 3.4, H), facecolor="black")
    for yb, h, Y, lo, hi in (((hs + 0.55) / H, ht / H, yd, y0, y1), (0.3 / H, hs / H, zd, z0 - 40, z1)):
        ax = fig.add_axes([0.1 / (W_ + 3.4), yb, W_ / (W_ + 3.4), h])
        ax.set_facecolor("black")
        ax.scatter(xd[ins], Y[ins], s=0.6, c="0.28", lw=0, rasterized=True)
        for r, c, _ in REGION_COLOURS:
            m = reg[r]
            ax.scatter(xd[m], Y[m], s=2.2, c=c, lw=0, rasterized=True)
        ax.set_xlim(x0, x1)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.text(0.1 / (W_ + 3.4), 1 - 0.08 / H, "from above", color="white", fontsize=13, va="top")
    fig.text(0.1 / (W_ + 3.4), (hs + 0.42) / H, "from the side", color="white", fontsize=13, va="top")
    yt = 1 - 0.6 / H
    for r, c, blocks in REGION_COLOURS[::-1]:
        fig.text((W_ + 0.3) / (W_ + 3.4), yt, f"{r}, {int(reg[r].sum()):,}", color=c, fontsize=13, va="top", weight="bold")
        fig.text((W_ + 0.3) / (W_ + 3.4), yt - 0.3 / H, blocks, color="0.85", fontsize=10.5, va="top", wrap=True)
        yt -= 0.95 / H
    fig.text((W_ + 0.3) / (W_ + 3.4), yt, f"grey: the other neurons\nin the Z-Brain brain, {int(ins.sum()):,}", color="0.6",
             fontsize=10.5, va="top")
    out = os.path.join(EXP, "presentation", "figs", "bio_mask_regions.png")
    fig.savefig(out, dpi=170, facecolor="black")
    plt.close(fig)
    print("[bio mask] ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--efference", action="store_true")
    ap.add_argument("--permute", type=int, default=None)
    ap.add_argument("--no-markers", action="store_true")
    a = ap.parse_args()
    if a.figure:
        figure()
    else:
        build(a.out, a.efference, a.permute, markers=not a.no_markers)
