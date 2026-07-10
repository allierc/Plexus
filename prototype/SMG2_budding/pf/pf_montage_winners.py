"""pf_montage_winners -- re-run the best (lowest target_distance) spec per hypothesis branch and tile
phi(t) under the real target row, so we can SEE whether the phase-field loop's winners actually look
like the dense connected lobular SMG. Honest check: labels show the tightened-readout scores.

  python pf/pf_montage_winners.py [--stride 130 --nrec 6]
"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "search"))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pf_sim, pf_tree
import smg_reward as R
from pf_bootstrap import phi_to_points, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=130); ap.add_argument("--nrec", type=int, default=6)
    ap.add_argument("--data", default="_boot")     # bootstrap/search dir under pf/ (e.g. _ucb)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(HERE, args.data, "dataset.jsonl")) if l.strip()]
    rows = [r for r in rows if r.get("value")]
    phi0 = np.load(os.path.join(HERE, "_real", "phi0.npy"))
    real = np.load(os.path.join(HERE, "_real", "targets.npz"))
    rphis, rframes = real["phis"], real["frames"]

    # best spec per branch by target_distance
    best = {}
    for r in sorted(rows, key=lambda r: r["target_distance"]):
        best.setdefault(r["branch"], r)
    sel = list(best.values())
    print(f"winners: " + ", ".join(f"{r['branch']}(td={r['target_distance']})" for r in sel), flush=True)

    ncol = args.nrec; nrow = len(sel) + 1
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 2.5, nrow * 2.5))
    fig.patch.set_facecolor("black"); axs = np.atleast_2d(axs)
    ridx = [int(round(f * (len(rphis) - 1))) for f in np.linspace(0, 1, ncol)]
    ro = R.obs_2d(phi_to_points(rphis[-1]), W=1.0); rv = R.value_vector(ro)
    for j, k in enumerate(ridx):
        ax = axs[0, j]; ax.imshow(rphis[k].T, origin="lower", cmap="magma", vmin=0, vmax=1)
        ax.set_facecolor("black"); ax.set_xticks([]); ax.set_yticks([])
        lab = f"REAL t={rframes[k]}" + (f"\nduct={rv['duct_score']} gen={rv['generations']} "
                                        f"bud={rv['bud_score']}" if j == ncol - 1 else "")
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, color="cyan", fontsize=8, va="top")

    for i, r in enumerate(sel, start=1):
        snaps = pf_sim.simulate(phi0, pf_tree.build_params(r["branch"], r["params"]),
                                n_record=ncol, stride=args.stride, seed=0)
        sc = score(snaps[-1]); v = sc[1] if sc else {}
        for j in range(ncol):
            ax = axs[i, j]; ax.imshow(snaps[j].T, origin="lower", cmap="magma", vmin=0, vmax=1)
            ax.set_facecolor("black"); ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.text(0.03, 0.97, f"{r['branch']}\n{r['cleft_mode']}", transform=ax.transAxes,
                        color="white", fontsize=8, va="top")
            if j == ncol - 1:
                ax.text(0.03, 0.97, f"td={r['target_distance']}\nduct={v.get('duct_score','-')} "
                        f"gen={v.get('generations','-')}\nbud={v.get('bud_score','-')} {r['failure']}",
                        transform=ax.transAxes, color="white", fontsize=8, va="top")
    fig.subplots_adjust(left=0.003, right=0.997, top=0.997, bottom=0.003, wspace=0.02, hspace=0.04)
    out = os.path.join(HERE, args.data, "winners.png")
    fig.savefig(out, dpi=90, facecolor="black"); print("wrote", out)


if __name__ == "__main__":
    main()
