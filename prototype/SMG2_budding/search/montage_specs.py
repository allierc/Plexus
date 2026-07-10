"""montage_specs -- SEE the bootstrap morphologies: re-run representative specs and render their
density + topology (skeleton / buds / branch points) over time, so we can check whether the
'branch-like duct=1.0' records actually show ducts / budding / branching.

Rows = representative specs (top-duct per hypothesis + a fragment/cluster contrast); columns =
timepoints. Black bg, top-left labels. Re-runs are deterministic (same branch/params/seed).

  python search/montage_specs.py [--frames 300 --per_branch 1]
"""
import os, sys, json, argparse, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.feature import peak_local_max
import bootstrap as B
import smg_reward as R


def _panel(ax, P, label):
    ax.set_facecolor("black"); ax.set_xticks([]); ax.set_yticks([])
    if len(P) < 20:
        ax.text(0.5, 0.5, "collapsed", color="red", ha="center", va="center", transform=ax.transAxes)
        ax.text(0.02, 0.98, label, transform=ax.transAxes, color="white", fontsize=9, va="top"); return
    Pn = (P - P.min(0)) / (np.ptp(P, axis=0) + 1e-9)
    vox = 0.008
    n = int(1 / vox) + 1
    ix = np.clip((Pn[:, 0] / vox).astype(int), 0, n - 1); iy = np.clip((Pn[:, 1] / vox).astype(int), 0, n - 1)
    g = np.zeros((n, n), np.float32); np.add.at(g, (ix, iy), 1.0)
    dens = ndi.gaussian_filter(g, 4.5)
    occ = ndi.binary_fill_holes(dens > 0.10 * max(dens.max(), 1e-9))
    lbl, nc = ndi.label(occ); sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    body = lbl == int(sizes.argmax()) if nc else occ
    edt = ndi.distance_transform_edt(body)
    buds = peak_local_max(edt, min_distance=5, labels=body, threshold_abs=2.0)
    skel = skeletonize(body)
    ax.imshow(dens.T, origin="lower", cmap="magma")
    if skel.any():
        sy, sx = np.nonzero(skel.T); ax.scatter(sx, sy, s=0.4, c="deepskyblue", alpha=0.5, linewidths=0)
    if len(buds):
        ax.scatter(buds[:, 0], buds[:, 1], s=60, facecolors="none", edgecolors="cyan", linewidths=1.4)
    ax.text(0.02, 0.98, label, transform=ax.transAxes, color="white", fontsize=9, va="top", ha="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--per_branch", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "_bootstrap", "montage_specs.png"))
    args = ap.parse_args()
    rows = [json.loads(l) for f in glob.glob(os.path.join(HERE, "_bootstrap", "shard_*", "dataset.jsonl"))
            for l in open(f) if l.strip()]
    rows = [r for r in rows if r.get("value")]
    # select: top-duct per branch + one fragment + one cluster contrast
    sel = []
    by_branch = {}
    for r in sorted(rows, key=lambda r: -r["value"]["duct_score"]):
        by_branch.setdefault(r["branch"], []).append(r)
    for b, rs in by_branch.items():
        sel += rs[:args.per_branch]
    frag = [r for r in rows if r["failure"] == "fragment"]
    if frag:
        sel.append(sorted(frag, key=lambda r: r["value"]["duct_score"])[0])   # a clear fragment
    print(f"rendering {len(sel)} representative specs x re-run...", flush=True)

    ncol = 4
    cols = [max(0, int(round(f * (args.frames - 1)))) for f in (0.0, 0.33, 0.66, 1.0)]
    fig, axs = plt.subplots(len(sel), ncol, figsize=(ncol * 3.2, len(sel) * 3.2))
    fig.patch.set_facecolor("black"); axs = np.atleast_2d(axs)
    for i, r in enumerate(sel):
        caps, err = B.run_spec(B.mt.build_spec(r["branch"], r["params"], seed=42, frames=args.frames),
                               args.frames, max(1, args.frames // 3), 42, "cuda:0")
        v = r["value"]
        hdr = f"{r['branch']}\nrec: duct={v['duct_score']} bud={v['bud_score']} br={v['branch_count']} {r['failure']}"
        if caps is None:
            for j in range(ncol):
                axs[i, j].set_facecolor("black"); axs[i, j].axis("off")
            axs[i, 0].text(0.02, 0.98, hdr + "\nRERUN ERR", transform=axs[i, 0].transAxes,
                           color="red", fontsize=8, va="top"); continue
        aX = np.array(caps["aX"]); occ = np.array(caps["occ"]) > 0
        T = len(aX)
        for j in range(ncol):
            k = min(T - 1, int(round((j / (ncol - 1)) * (T - 1))))
            P = aX[k][occ[k]]
            _panel(axs[i, j], P, (hdr if j == 0 else f"t={k*max(1,args.frames//3)}"))
        print(f"  [{i+1}/{len(sel)}] {r['branch']:22} duct_rec={v['duct_score']}", flush=True)
    fig.subplots_adjust(left=0.003, right=0.997, top=0.997, bottom=0.003, wspace=0.02, hspace=0.06)
    fig.savefig(args.out, dpi=85, facecolor="black"); print("wrote", args.out)


if __name__ == "__main__":
    main()
