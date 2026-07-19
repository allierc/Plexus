"""real_targets -- rasterize the REAL SMG nucleus clouds onto a grid, aspect-preserved, to give the
phase-field forward model (1) an initial condition phi0 (frame 0) and (2) a morphology TARGET sequence
(intermediate + final frames) at the SAME grid resolution the model runs on.

Per-axis [0,1] normalization DISTORTS the gland (it is ~square but not exactly), so we scale BOTH axes
by one factor and center in the grid -> the branched silhouette is preserved. Occupancy is a Gaussian
splat of the nuclei, thresholded to a smooth phi in [0,1].

  python pf/real_targets.py [--G 256]  -> writes pf/_real/{phi0.npy, phiT.npy, targets.npz, real_targets.png}
"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "search"))
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import smg_topo as st

FRAMES = [0, 92, 184, 276, 368, 460, 552]        # 7 snapshots across the movie (GT anchors at 0 and 552)


def _global_norm(all_pts, pad=0.08):
    """One scale for BOTH axes (aspect-preserved), fit to [pad, 1-pad] using the WHOLE movie's extent so
    frames are in a common frame of reference (growth is visible, not normalized away)."""
    lo = np.min([p.min(0) for p in all_pts], 0)
    hi = np.max([p.max(0) for p in all_pts], 0)
    scale = (1 - 2 * pad) / max((hi - lo).max(), 1e-9)
    ctr = 0.5 * (lo + hi)
    return lambda p: (p - ctr) * scale + 0.5


def rasterize(p, G, sigma_vox=2.0, thr=0.12):
    """Nuclei -> smooth phi in [0,1] on a GxG grid (Gaussian splat, normalized, soft-thresholded)."""
    ix = np.clip((p[:, 0] * G).astype(int), 0, G - 1)
    iy = np.clip((p[:, 1] * G).astype(int), 0, G - 1)
    g = np.zeros((G, G), np.float32); np.add.at(g, (ix, iy), 1.0)
    d = ndi.gaussian_filter(g, sigma_vox)
    d = d / max(d.max(), 1e-9)
    # soft phi: 1 well inside, smooth boundary near thr
    phi = np.clip((d - thr) / (thr + 1e-9) * 0.5 + 0.5, 0, 1)
    phi[d < 0.35 * thr] = 0.0
    return phi.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--G", type=int, default=256); args = ap.parse_args()
    xl = torch.load(st.PT_DEFAULT, map_location="cpu", weights_only=False)
    pts = [np.asarray(st.load_frame(xl, f)[:, :2], float) for f in FRAMES]
    norm = _global_norm(pts)
    phis = [rasterize(norm(p), args.G) for p in pts]
    outd = os.path.join(HERE, "_real"); os.makedirs(outd, exist_ok=True)
    np.save(os.path.join(outd, "phi0.npy"), phis[0])
    np.save(os.path.join(outd, "phiT.npy"), phis[-1])
    np.savez(os.path.join(outd, "targets.npz"), frames=np.array(FRAMES), phis=np.stack(phis))

    n = len(FRAMES)
    fig, axs = plt.subplots(1, n, figsize=(n * 2.6, 2.8)); fig.patch.set_facecolor("black")
    for j, (f, phi) in enumerate(zip(FRAMES, phis)):
        ax = axs[j]; ax.imshow(phi.T, origin="lower", cmap="magma", vmin=0, vmax=1)
        ax.set_facecolor("black"); ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.03, 0.97, f"real t={f}\narea={phi.sum()/phi.size:.3f}", transform=ax.transAxes,
                color="white", fontsize=9, va="top")
    fig.subplots_adjust(left=0.003, right=0.997, top=0.997, bottom=0.003, wspace=0.02)
    fig.savefig(os.path.join(outd, "real_targets.png"), dpi=90, facecolor="black")
    print("area fractions:", [round(float(p.sum() / p.size), 3) for p in phis])
    print("wrote", os.path.join(outd, "real_targets.png"), "+ phi0.npy phiT.npy targets.npz  G=", args.G)


if __name__ == "__main__":
    main()
