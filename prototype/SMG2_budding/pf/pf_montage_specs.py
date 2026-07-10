"""pf_montage_specs -- the phase-field analogue of the old search/montage_specs.png: representative
bootstrap specs over time with the TOPOLOGY READOUT drawn on (skeleton = ducts, cyan circles = buds,
red squares = branch points), so duct/budding/branching can be checked by eye. Rows = representative
specs (best per hypothesis + contrasts), columns = timepoints; real target on top with the same overlay.

  python pf/pf_montage_specs.py [--stride 130 --nrec 6]
"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "search"))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.feature import peak_local_max
import pf_sim, pf_tree
import smg_reward as R
from pf_bootstrap import phi_to_points, score


def overlay(ax, phi, label):
    """Draw phi + skeleton (ducts) + bud peaks + branch points -- the same readout smg_reward uses."""
    ax.set_facecolor("black"); ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(phi.T, origin="lower", cmap="magma", vmin=0, vmax=1)
    body = phi > 0.5
    lbl, nc = ndi.label(body)
    if nc:
        sizes = np.bincount(lbl.ravel()); sizes[0] = 0
        body = lbl == int(sizes.argmax())
    if body.sum() > 20:
        edt = ndi.distance_transform_edt(body)
        buds = peak_local_max(edt, min_distance=6, labels=body, threshold_abs=3.0)
        skel = skeletonize(body)
        nb = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - skel
        bp = np.argwhere(skel & (nb >= 3))                       # branch points (skeleton degree>=3)
        sy, sx = np.nonzero(skel.T)
        ax.scatter(sx, sy, s=0.5, c="deepskyblue", alpha=0.6, linewidths=0)          # ducts
        if len(buds):
            ax.scatter(buds[:, 0], buds[:, 1], s=42, facecolors="none", edgecolors="cyan", linewidths=1.1)
        if len(bp):
            ax.scatter(bp[:, 0], bp[:, 1], s=26, marker="s", facecolors="none",
                       edgecolors="red", linewidths=1.1)
    if label:
        ax.text(0.03, 0.97, label, transform=ax.transAxes, color="white", fontsize=8, va="top")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=130); ap.add_argument("--nrec", type=int, default=6)
    ap.add_argument("--data", default="_boot")     # bootstrap dir under pf/ (e.g. _ucb for the search)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(HERE, args.data, "dataset.jsonl")) if l.strip()]
    rows = [r for r in rows if r.get("value")]
    phi0 = np.load(os.path.join(HERE, "_real", "phi0.npy"))
    real = np.load(os.path.join(HERE, "_real", "targets.npz")); rphis, rframes = real["phis"], real["frames"]

    # representative selection: best (lowest td) per hypothesis branch + a no-growth failure contrast
    best = {}
    for r in sorted(rows, key=lambda r: r["target_distance"]):
        best.setdefault(r["branch"], r)
    sel = list(best.values())
    ng = [r for r in rows if r["failure"] == "no-growth"]
    if ng:
        sel.append(max(ng, key=lambda r: r["target_distance"]))   # a clear no-growth (under-developed) contrast
    print("specs:", ", ".join(f"{r['branch']}(td={r['target_distance']},{r['failure']})" for r in sel), flush=True)

    ncol = args.nrec; nrow = len(sel) + 1
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 2.5, nrow * 2.5))
    fig.patch.set_facecolor("black"); axs = np.atleast_2d(axs)
    ridx = [int(round(f * (len(rphis) - 1))) for f in np.linspace(0, 1, ncol)]
    rv = R.value_vector(R.obs_2d(phi_to_points(rphis[-1]), W=1.0))
    for j, k in enumerate(ridx):
        overlay(axs[0, j], rphis[k], f"REAL t={rframes[k]}" + (f"\nduct={rv['duct_score']} "
                f"gen={rv['generations']} bud={rv['bud_score']}" if j == ncol - 1 else ""))
        axs[0, j].texts[-1].set_color("cyan")

    for i, r in enumerate(sel, start=1):
        snaps = pf_sim.simulate(phi0, pf_tree.build_params(r["branch"], r["params"]),
                                n_record=ncol, stride=args.stride, seed=0)
        v = (score(snaps[-1]) or (None, {}, None))[1]
        for j in range(ncol):
            lab = (f"{r['branch']}\n{r['cleft_mode']}" if j == 0 else
                   (f"td={r['target_distance']}\nduct={v.get('duct_score','-')} gen={v.get('generations','-')}"
                    f"\nbud={v.get('bud_score','-')} {r['failure']}" if j == ncol - 1 else ""))
            overlay(axs[i, j], snaps[j], lab)
    fig.subplots_adjust(left=0.003, right=0.997, top=0.997, bottom=0.003, wspace=0.02, hspace=0.04)
    out = os.path.join(HERE, args.data, "montage_specs.png")
    fig.savefig(out, dpi=92, facecolor="black"); print("wrote", out)


if __name__ == "__main__":
    main()
