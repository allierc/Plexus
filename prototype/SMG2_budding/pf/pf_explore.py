"""pf_explore -- morphology GATE for the phase-field forward model. Run pf_sim from the REAL t=0 phi0
across a few mechanism regimes; tile phi(t) under the real target row; score each final shape with the
TIGHTENED readout (smg_reward). We decide on NUMBERS + the picture, not on hope: the model must stay a
DENSE CONNECTED mass, subdivide lobes (clefts), grow only modestly, and read branch-like / low-cluster
like the real gland -- otherwise the substrate is still wrong and we do not proceed to search.

  python pf/pf_explore.py [--stride 140 --G 256]
"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "search"))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pf_sim
import smg_reward as R

# mechanism regimes to compare (focal-ECM curvature-cleft variant; knobs = surface tension / cleft / growth)
REGIMES = {
    "curv_clean":    dict(cleft_mode="curvature", kappa=1.3, s=1.0, lam=1.0, thick_gate=0.6),
    "curv_fine":     dict(cleft_mode="curvature", kappa=1.3, s=1.6, kappa_gate=0.035),
    "big_lobes":     dict(cleft_mode="curvature", kappa=1.7),
    "turing_coarse": dict(cleft_mode="turing", kappa=1.3, feed=0.030, kill=0.062, s=1.2, lam=1.1),
    "turing_fine":   dict(cleft_mode="turing", kappa=1.3, feed=0.042, kill=0.064, s=1.2, lam=1.1),
    "turing_grow":   dict(cleft_mode="turing", kappa=1.3, growth_frac=1.6, feed=0.035, s=1.3),
}


def phi_to_points(phi, thr=0.5, max_pts=9000, rng=None):
    ys, xs = np.nonzero(phi > thr)
    if len(xs) < 20:
        return None
    P = np.c_[xs, ys].astype(float)
    if len(P) > max_pts:
        rng = rng or np.random.default_rng(0)
        P = P[rng.choice(len(P), max_pts, replace=False)]
    return (P - P.min(0)) / (np.ptp(P, 0) + 1e-9)


def score(phi):
    Pn = phi_to_points(phi)
    if Pn is None:
        return dict(duct_score=0, cluster_score=0, generations=0), "collapsed", 0.0
    o = R.obs_2d(Pn, W=1.0)
    return R.value_vector(o), R.classify(o), float((phi > 0.5).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=140)
    ap.add_argument("--nrec", type=int, default=6)
    args = ap.parse_args()
    real = np.load(os.path.join(HERE, "_real", "targets.npz"))
    phi0 = np.load(os.path.join(HERE, "_real", "phi0.npy"))
    rphis, rframes = real["phis"], real["frames"]

    names = list(REGIMES)
    nrow = len(names) + 1
    ncol = args.nrec
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 2.5, nrow * 2.5))
    fig.patch.set_facecolor("black"); axs = np.atleast_2d(axs)

    # row 0: real targets (resampled to ncol columns)
    ridx = [int(round(f * (len(rphis) - 1))) for f in np.linspace(0, 1, ncol)]
    ro = R.obs_2d(phi_to_points(rphis[-1]), W=1.0); rv = R.value_vector(ro)
    for j, k in enumerate(ridx):
        ax = axs[0, j]; ax.imshow(rphis[k].T, origin="lower", cmap="magma", vmin=0, vmax=1)
        ax.set_facecolor("black"); ax.set_xticks([]); ax.set_yticks([])
        lab = f"REAL t={rframes[k]}" + (f"\nduct={rv['duct_score']} clust={rv['cluster_score']} "
                                        f"{R.classify(ro)}" if j == ncol - 1 else "")
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, color="cyan", fontsize=8, va="top")

    for i, name in enumerate(names, start=1):
        snaps = pf_sim.simulate(phi0, REGIMES[name], n_record=ncol, stride=args.stride, seed=0)
        v, cls, area = score(snaps[-1])
        for j in range(ncol):
            ax = axs[i, j]; ax.imshow(snaps[j].T, origin="lower", cmap="magma", vmin=0, vmax=1)
            ax.set_facecolor("black"); ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.text(0.03, 0.97, name, transform=ax.transAxes, color="white", fontsize=9, va="top")
            if j == ncol - 1:
                ax.text(0.03, 0.97, f"duct={v['duct_score']} clust={v['cluster_score']}\n"
                        f"gen={v['generations']} area={area:.2f}\n{cls}", transform=ax.transAxes,
                        color="white", fontsize=8, va="top")
        print(f"{name:14} duct={v['duct_score']:.2f} cluster={v['cluster_score']:.2f} "
              f"gen={v['generations']} area={area:.3f} class={cls}", flush=True)

    fig.subplots_adjust(left=0.003, right=0.997, top=0.997, bottom=0.003, wspace=0.02, hspace=0.04)
    out = os.path.join(HERE, "_explore", "explore.png"); os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=88, facecolor="black"); print("wrote", out)


if __name__ == "__main__":
    main()
