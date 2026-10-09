"""Does ParticleGraph's SMG2 salivary-gland dataset carry cell identities across frames (tracks), and what is its frame
interval? For exp 21's question of how the gland's surface cells move.

    python tools/exp21_gland_smg2_check.py

THE DATA. /workspace/ParticleGraph/graphs_data/cell/cell_gland_SMG2_smooth{2,10}/x_list_0.pt: a list of 553 tensors,
one per frame, written by ParticleGraph's mesh loader (`src/ParticleGraph/data_loaders.py`, the '3D masks meshes'
branch) from per-frame Cellpose label tables (`masks_smooth*_label_props/*.csv`, config
`config/cell/archive/cell_gland_SMG2_smooth2.yaml`). Its columns, read from that loader:
    0  N       = arange(n) -- the row index within the frame
    1-3  x, y, z, um if the label tables used cell_utils' default voxel (0.75, 0.75, 1.0) um
    4-9  zeros (velocity etc. never filled)
    10 volume, 11 surface area, 12 sphericity, 13 mean intensity, 14 intensity sd
    15 ID      = n_cells + arange(n) - 1 with n_cells fixed at 1 -- again the row index
The labels come from Cellpose run frame by frame; nothing links a label at t to a label at t + 1.

THE TEST. If row i at t and row i at t + 1 were the same cell, its displacement would be the cell's step (~1-3 um);
instead compare it with the displacement to the NEAREST detection at t + 1, with the displacement to a random row,
and count how often a row's nearest detection at t + 1 has the same index. Then ask whether nearest-neighbour
linking could make tracks: the fraction of detections whose nearest detection at t + 1 is mutual and clearly closer
than the second nearest (d1 < 0.5 d2).

Writes the section `smg2_check` into experiments/exp21_mpm_epithelium/data/wang_surface_motion.json and the figure
figs/smg2_identity_check.png (+ .txt sidecar) and rebuilds the watcher record.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                    # noqa: E402
import numpy as np                                                                 # noqa: E402
import torch                                                                       # noqa: E402
from scipy.spatial import cKDTree                                                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import exp21_gland_figures as F                                                    # noqa: E402

PG = "/workspace/ParticleGraph/graphs_data/cell"
JSON = os.path.join(ROOT, "experiments/exp21_mpm_epithelium/data/wang_surface_motion.json")


def check(name, every=25):
    xl = torch.load(os.path.join(PG, name, "x_list_0.pt"), map_location="cpu", weights_only=False)
    n = np.array([len(a) for a in xl])
    x0 = xl[0].numpy()
    out = {"path": os.path.join(PG, name, "x_list_0.pt"), "frames": len(xl), "cols": int(x0.shape[1]),
           "n_per_frame_first_median_last": [int(n[0]), int(np.median(n)), int(n[-1])],
           "col0_is_row_index_all_frames": bool(all(np.array_equal(a[:, 0].numpy(), np.arange(len(a))) for a in xl)),
           "col15_is_row_index_all_frames": bool(all(np.array_equal(a[:, 15].numpy(), np.arange(len(a))) for a in xl)),
           "cols4to9_all_zero": bool(all(float(a[:, 4:10].abs().max()) == 0 for a in xl)),
           "xyz_range": [[float(x0[:, k].min()), float(x0[:, k].max())] for k in (1, 2, 3)],
           "volume_median": float(np.median(x0[:, 10])), "volume_p10_p90": [float(np.percentile(x0[:, 10], 10)),
                                                                             float(np.percentile(x0[:, 10], 90))]}
    rows = []
    rng = np.random.default_rng(0)
    for t in range(0, len(xl) - 1, every):
        a = xl[t].numpy(); b = xl[t + 1].numpy()
        k = min(len(a), len(b))
        A, B = a[:, 1:4], b[:, 1:4]
        d_row = np.linalg.norm(A[:k] - B[:k], axis=1)
        d_rand = np.linalg.norm(A[:k] - B[rng.integers(0, len(b), k)], axis=1)
        dd, nb = cKDTree(B).query(A, k=2)
        back = cKDTree(A).query(B[nb[:, 0]])[1]
        mutual = back == np.arange(len(a))
        nn0 = cKDTree(A).query(A, k=2)[0][:, 1]
        rows.append((t, np.median(d_row), np.median(d_rand), np.median(dd[:, 0]), np.median(nn0),
                     float(np.mean(nb[:k, 0] == np.arange(k))), float(np.mean(mutual & (dd[:, 0] < 0.5 * dd[:, 1])))))
    r = np.array(rows)
    out["per_frame_sample"] = {"t": r[:, 0].astype(int).tolist(), "row_matched_disp_median": r[:, 1].tolist(),
                               "random_row_disp_median": r[:, 2].tolist(), "nearest_detection_disp_median": r[:, 3].tolist(),
                               "nn_spacing_median": r[:, 4].tolist(), "nearest_is_same_row_frac": r[:, 5].tolist(),
                               "linkable_frac_mutual_and_d1_lt_half_d2": r[:, 6].tolist()}
    out["summary"] = {"row_matched_disp_median": float(np.median(r[:, 1])),
                      "random_row_disp_median": float(np.median(r[:, 2])),
                      "nearest_detection_disp_median": float(np.median(r[:, 3])),
                      "nn_spacing_median": float(np.median(r[:, 4])),
                      "nearest_is_same_row_frac": float(np.median(r[:, 5])),
                      "linkable_frac": float(np.median(r[:, 6]))}
    return out, xl


def main():
    res = {}
    lists = {}
    for name in ("cell_gland_SMG2_smooth2", "cell_gland_SMG2_smooth10"):
        if os.path.exists(os.path.join(PG, name, "x_list_0.pt")):
            res[name], lists[name] = check(name)
            print(name, json.dumps(res[name]["summary"]), flush=True)
    keep = lists["cell_gland_SMG2_smooth2"]
    if len(lists) == 2:
        a, b = lists.values()
        dmax = np.max([np.abs(x.numpy() - y.numpy()).max(0) for x, y in zip(a, b)], 0)
        res["smooth2_vs_smooth10_columns_that_differ"] = [int(c) for c in np.flatnonzero(dmax > 0)]
    res["verdict"] = ("no cell identities: column 0 and column 15 are the row index of each frame (Cellpose labels "
                      "re-numbered per frame); row i at t and t+1 are as far apart as random rows, and a row's nearest "
                      "detection at t+1 has its index no more often than chance")
    res["frame_interval"] = ("not recorded: the config has delta_t 1 (frames) and no time stamps, and the label tables "
                             "it was built from (/groups/wang/.../SMG2-processed) are not on this machine")
    res["config"] = "/workspace/ParticleGraph/config/cell/archive/cell_gland_SMG2_smooth2.yaml (and _smooth10_1..10)"
    res["loader"] = "/workspace/ParticleGraph/src/ParticleGraph/data_loaders.py ('3D masks meshes' branch, ~l. 880-940)"
    J = json.load(open(JSON))
    J["smg2_check"] = res
    with open(JSON, "w") as fh:
        json.dump(J, fh, indent=1, default=float)
    # the figure
    s = res["cell_gland_SMG2_smooth2"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.9), gridspec_kw={"wspace": 0.35})
    a = keep[300].numpy(); b = keep[301].numpy(); k = min(len(a), len(b))
    ax = axs[0]
    dr = np.linalg.norm(a[:k, 1:4] - b[:k, 1:4], axis=1)
    dn = cKDTree(b[:, 1:4]).query(a[:, 1:4])[0]
    bins = np.logspace(-1, 3, 50)
    ax.hist(dr, bins=bins, histtype="step", color=RED_, lw=1.6, label=f"same row at t+1: median {np.median(dr):.0f}")
    ax.hist(dn, bins=bins, histtype="step", color=BLUE_, lw=1.6, label=f"nearest detection at t+1: median {np.median(dn):.1f}")
    ax.axvline(s["summary"]["nn_spacing_median"], color=F.MUTED, ls=":", lw=0.9)
    ax.text(s["summary"]["nn_spacing_median"] * 1.1, ax.get_ylim()[1] * 0.5, "nearest-neighbour\nspacing", fontsize=7)
    ax.set_xscale("log"); ax.legend(loc="upper left", fontsize=7)
    F.panel(ax, "a", "detections, frame 300 -> 301 (smooth2)", "displacement (um, default voxel 0.75 x 0.75 x 1)")
    ax = axs[1]
    ps = s["per_frame_sample"]
    ax.plot(ps["t"], ps["row_matched_disp_median"], color=RED_, marker="o", ms=3, label="same row")
    ax.plot(ps["t"], ps["random_row_disp_median"], color=F.MUTED, ls="--", label="random row")
    ax.plot(ps["t"], ps["nearest_detection_disp_median"], color=BLUE_, marker="s", ms=3, label="nearest detection")
    ax.set_yscale("log"); ax.legend(loc="center right", fontsize=7)
    F.panel(ax, "b", "median displacement t -> t+1 (um)", "frame")
    ax = axs[2]
    n = [len(x) for x in keep]
    ax.plot(n, color=F.INK, lw=1)
    F.panel(ax, "c", "detections per frame (smooth2)", "frame")
    ax.text(0.3, 0.08, f"nearest detection has the same row index: {s['summary']['nearest_is_same_row_frac']:.3f}\n"
                        f"mutual nearest with d1 < 0.5 d2: {s['summary']['linkable_frac']:.2f}", transform=ax.transAxes,
            fontsize=7.5)
    F.save(fig, "smg2_identity_check", f"""
ParticleGraph's SMG2 gland data carry no cell identities and no frame interval: rows are Cellpose labels renumbered every frame.
Layout (data_loaders.py, '3D masks meshes'): col 0 and col 15 = row index (true in all {s['frames']} frames), cols 1-3 x, y, z (um under cell_utils' default voxel 0.75 x 0.75 x 1 um), cols 4-9 zero, 10 volume (median {s['volume_median']:.0f}), 11 area, 12 sphericity, 13-14 intensity; {s['n_per_frame_first_median_last'][0]} -> {s['n_per_frame_first_median_last'][2]} detections per frame.
a-b: row i at t and at t+1 are {s['summary']['row_matched_disp_median']:.0f} um apart (median over every 25th frame), as far as random rows ({s['summary']['random_row_disp_median']:.0f} um), while the nearest detection at t+1 is {s['summary']['nearest_detection_disp_median']:.1f} um away against a {s['summary']['nn_spacing_median']:.1f}-um nearest-neighbour spacing; a row's nearest detection keeps its index in {s['summary']['nearest_is_same_row_frac']:.1%} of cases (labels follow scan order, so slightly above 1/n; no identity). Volume median {s['volume_median']:.0f} um3: the objects are nuclei (an 8-um sphere is 268 um3), 7.5 um apart. smooth10 has the same positions (only columns {res.get('smooth2_vs_smooth10_columns_that_differ')} -- surface area, sphericity of the smoothed mesh -- differ).
Tracks would need linking: {s['summary']['linkable_frac']:.0%} of detections have a mutual nearest detection at t+1 closer than half the second nearest. The frame interval is not recorded (delta_t 1, no time stamps; the source tables under /groups/wang are not on this machine).
""")


RED_, BLUE_ = F.RED, F.BLUE

if __name__ == "__main__":
    main()
