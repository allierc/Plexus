"""exp17: A LATERAL CHESS BOARD? (Cedric, 2026-10-07: "analyse the blocks for a lateral chess-board pattern: neuron pairs
that are inversely correlated"). Local, the destriped recording in the Z-Brain atlas (tools/exp17_atlas.py), no model.

MIRROR PAIRS. Each left neuron (Z-Brain x below the midline) is paired with the right neuron nearest to its mirror image
(x reflected across the midline, y and z kept), within MAX_UM; a pair only once (mutual nearest). Per stimulus block,
the correlation of the two dF/F traces over the block's frames: r < 0 the two sides alternate (a chess board),
r > 0 they move together. NULL: the same left neuron against a random right neuron of the same Z-Brain region and the
same depth band (+-10 um) -- what anatomy alone gives.
Per block and per region: the median mirror r, the share of pairs with r < -R_NEG, against the null's share.
Region-level: the left-half mean trace against the right-half mean trace, r per block and region.

    PYTHONPATH=src:tools python tools/exp17_lateral.py
-> presentation/figs/lateral.png, data/lateral.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
MAX_UM, R_NEG, MID = 6.0, 0.3, (621 - 1) * 0.798 / 2


def corr_cols(A, B):
    """Column-wise Pearson r of A [T, n] and B [T, n]."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    return (A * B).sum(0) / np.sqrt((A * A).sum(0) * (B * B).sum(0) + 1e-12)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree
    from plexus.paths import graphs_data_path
    from exp17_atlas import REGIONS
    za = np.load(os.path.join(EXP, "data", "atlas_destripe.npz"))
    A, reg, names = za["atlas_um"].astype(np.float64), za["regions"], [str(x) for x in za["names"]]
    z = np.load(graphs_data_path("zebrafish", "zapbench_destripe_recording.npz"))
    X = np.asarray(z["dff"], np.float32)
    off, bnames = z["offsets"], [str(x) for x in z["names"]]
    ok = za["inside"] & (X.std(0) > 1e-6)
    L = np.flatnonzero(ok & (A[:, 0] < MID))
    R = np.flatnonzero(ok & (A[:, 0] >= MID))
    mir = A[L].copy()
    mir[:, 0] = 2 * MID - mir[:, 0]
    tR = cKDTree(A[R])
    d, j = tR.query(mir)
    keep = d < MAX_UM
    pl, pr = L[keep], R[j[keep]]
    # mutual: the right neuron's own mirror nearest is the same left neuron
    mirR = A[pr].copy()
    mirR[:, 0] = 2 * MID - mirR[:, 0]
    _, jb = cKDTree(A[L]).query(mirR)
    mut = L[jb] == pl
    pl, pr = pl[mut], pr[mut]
    # the null partner: a random right neuron of the same most-specific region and depth band
    rlist = [(full, short) for full, short in REGIONS if full in names]
    lab = np.full(len(A), -1)
    best_n = np.full(len(A), np.inf)
    for k, (full, short) in enumerate(rlist):
        m = reg[:, names.index(full)]
        upd = m & (m.sum() < best_n)
        lab[upd], best_n[upd] = k, m.sum()
    rng = np.random.default_rng(0)
    pn = np.full(len(pl), -1)
    for k in range(len(rlist)):
        cand = R[lab[R] == k]
        sel = np.flatnonzero(lab[pl] == k)
        if len(cand) == 0 or len(sel) == 0:
            continue
        zc = A[cand, 2]
        for i in sel:
            c_ = cand[np.abs(zc - A[pl[i], 2]) < 10.0]
            if len(c_):
                pn[i] = rng.choice(c_)
    hasn = pn >= 0
    blocks = [b for b in bnames]
    doc = {"pairs": int(len(pl)), "max_um": MAX_UM, "r_neg": R_NEG, "per_block": {}, "per_region": {}}
    rm = np.zeros((len(blocks), len(pl)), np.float32)
    rn = np.zeros((len(blocks), len(pl)), np.float32)
    lr_region = np.full((len(blocks), len(rlist)), np.nan)
    for kb, b in enumerate(blocks):
        f0, f1 = int(off[kb]), int(off[kb + 1])
        Xb = X[f0:f1]
        rm[kb] = corr_cols(Xb[:, pl], Xb[:, pr])
        rn[kb, hasn] = corr_cols(Xb[:, pl[hasn]], Xb[:, pn[hasn]])
        doc["per_block"][b] = {"median_mirror_r": float(np.median(rm[kb])), "neg_share": float((rm[kb] < -R_NEG).mean()),
                               "null_neg_share": float((rn[kb, hasn] < -R_NEG).mean()),
                               "median_null_r": float(np.median(rn[kb, hasn]))}
        for k, (full, short) in enumerate(rlist):
            m = reg[:, names.index(full)]
            mL, mR = m & ok & (A[:, 0] < MID), m & ok & (A[:, 0] >= MID)
            if mL.sum() > 30 and mR.sum() > 30:
                zs = lambda v: (v - v.mean(0)) / np.maximum(v.std(0), 1e-9)      # noqa: E731
                lr_region[kb, k] = np.corrcoef(zs(Xb[:, mL]).mean(1), zs(Xb[:, mR]).mean(1))[0, 1]
    for k, (full, short) in enumerate(rlist):
        sel = lab[pl] == k
        if sel.sum() > 30:
            doc["per_region"][short] = {"pairs": int(sel.sum()),
                                        "neg_share_by_block": {b: float((rm[kb, sel] < -R_NEG).mean())
                                                               for kb, b in enumerate(blocks)},
                                        "lr_mean_r_by_block": {b: (None if np.isnan(lr_region[kb, k]) else
                                                                   float(lr_region[kb, k])) for kb, b in enumerate(blocks)}}
    json.dump(doc, open(os.path.join(EXP, "data", "lateral.json"), "w"), indent=1)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 8.6), facecolor="black")
    # left: per block, the share of anti-correlated mirror pairs against the null
    a = fig.add_axes([0.06, 0.58, 0.28, 0.36])
    xs = np.arange(len(blocks))
    a.bar(xs - 0.2, [doc["per_block"][b]["neg_share"] * 100 for b in blocks], 0.4, color="#ff4a4a",
          label="mirror pairs (left / right)")
    a.bar(xs + 0.2, [doc["per_block"][b]["null_neg_share"] * 100 for b in blocks], 0.4, color="0.55",
          label="null: same region and depth, random")
    a.set_xticks(xs)
    a.set_xticklabels(blocks, rotation=35, fontsize=8.5, ha="right")
    a.set_ylabel(f"pairs with r < -{R_NEG}, %", fontsize=10)
    a.legend(fontsize=8.5, frameon=False)
    a.set_title(f"{len(pl):,} mirror pairs (mutual nearest within {MAX_UM:g} um)", fontsize=9.5, loc="left")
    # left bottom: region x block, the left-mean / right-mean correlation
    a2 = fig.add_axes([0.12, 0.07, 0.22, 0.42])
    rows = [k for k in range(len(rlist)) if np.isfinite(lr_region[:, k]).any()]
    im = a2.imshow(lr_region[:, rows].T, aspect="auto", cmap="coolwarm_r", vmin=-1, vmax=1)
    a2.set_yticks(range(len(rows)))
    a2.set_yticklabels([rlist[k][1] for k in rows], fontsize=7)
    a2.set_xticks(xs)
    a2.set_xticklabels(blocks, rotation=35, fontsize=7.5, ha="right")
    a2.set_title("left half's mean vs right half's, r", fontsize=9, loc="left")
    cb = fig.colorbar(im, ax=a2, fraction=0.05, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    # right: in the block with the most anti-correlated pairs, where they are (both members), from above
    kb = int(np.argmax([doc["per_block"][b]["neg_share"] for b in blocks]))
    neg = rm[kb] < -R_NEG
    a3 = fig.add_axes([0.42, 0.30, 0.56, 0.64])
    xd, yd = A[:, 1], (621 - 1) * 0.798 - A[:, 0]
    a3.scatter(xd[ok], yd[ok], s=0.1, color="0.2", lw=0, rasterized=True)
    a3.scatter(xd[pl[neg]], yd[pl[neg]], s=1.5, color="#ff4a4a", lw=0, rasterized=True)
    a3.scatter(xd[pr[neg]], yd[pr[neg]], s=1.5, color="#4a7bff", lw=0, rasterized=True)
    a3.set_aspect("equal")
    a3.axis("off")
    a3.set_title(f"{blocks[kb]}: the anti-correlated mirror pairs (r < -{R_NEG}), left member red, right blue -- "
                 f"{int(neg.sum()):,} of {len(pl):,}", fontsize=9.5, loc="left")
    a4 = fig.add_axes([0.46, 0.07, 0.24, 0.17])
    a4.hist(rm[kb], bins=np.linspace(-1, 1, 81), color="#ff4a4a", alpha=0.8, label="mirror")
    a4.hist(rn[kb, hasn], bins=np.linspace(-1, 1, 81), color="0.6", alpha=0.6, label="null")
    a4.set_xlabel(f"r between the two members, {blocks[kb]}", fontsize=9)
    a4.legend(fontsize=8, frameon=False)
    a4.tick_params(labelsize=7.5)
    fig.savefig(os.path.join(EXP, "presentation", "figs", "lateral.png"), dpi=130, facecolor="black")
    plt.close(fig)
    print(json.dumps(doc["per_block"], indent=1))


if __name__ == "__main__":
    main()
