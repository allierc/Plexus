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
per frame; the fish swims in every block). With `--permute SEED`: the same mask with its neuron rows shuffled
(non-marker columns), the control with the same count per column and the same column overlaps at random positions.

    PYTHONPATH=src:tools python tools/exp17_bio_mask.py --out input_mask_destripe_bio.npz
    PYTHONPATH=src:tools python tools/exp17_bio_mask.py --efference --out input_mask_destripe_bio_eff.npz
    PYTHONPATH=src:tools python tools/exp17_bio_mask.py --efference --permute 0 --out input_mask_destripe_bio_eff_perm.npz
-> graphs_data/zebrafish/<out>: mask [N] (any non-marker column), mask_by_input [N, 22 or 27], region_counts (json)
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


def build(out, efference=False, permute=None):
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
    mbi[:, ~free] = 1.0
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
             efference=efference, permute=-1 if permute is None else permute)
    print(f"[bio mask] {out}: {int(mask.sum()):,} of {N:,} neurons receive a non-marker column; markers "
          f"{[f'f{j}' for j in np.where(marker)[0]]} on every neuron")
    print("   regions:", counts)
    for j in np.where(free)[0]:
        nm = names[block[j]] if j < F else "ephys"
        print(f"   {('f%d' % j) if j < F else 'ephys%d' % (j - F):7s} {nm:10s} {int(mbi[:, j].sum()):7,d} neurons")
    return mbi


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--efference", action="store_true")
    ap.add_argument("--permute", type=int, default=None)
    a = ap.parse_args()
    build(a.out, a.efference, a.permute)
