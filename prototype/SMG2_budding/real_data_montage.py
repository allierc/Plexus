"""Real-data phenomenological montage for SMG2 branching morphogenesis.

3 rows x 4 timepoints -- this is the TARGET the Plexus forward model must
reproduce (phenomenology, not a per-cell fit):

  row 1  TOPOLOGY   density MIP + detected fat-lobule buds + duct skeleton/branch points
  row 2  GROWTH     proliferation-source heat map  = d(rho)/dt + div(rho v)
                     (continuity residual: local cell PRODUCTION, migration removed)
  row 3  MIGRATION  collective-flow PIV speed (dense optical flow) + quiver

Migration uses dense optical-flow PIV on the z-projected density (the track ids
reshuffle per frame, so no per-cell velocity is available); growth is the
continuity residual of that same flow, so the two panels are a consistent
decomposition of the tissue kinematics: v (migration) + source (growth).
"""
import os
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.morphology import h_maxima, skeletonize
from skimage.registration import optical_flow_tvl1
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import smg_topo as st
import viz_topology as vz

HERE = os.path.dirname(__file__)
VOX = 4.0
TIMES = [10, 185, 360, 540]
DT = 8


def fields_2d(pts, sigma_um, bounds):
    """z-integrated density (X,Y) -- cell column count, smoothed."""
    d, lo = st.density_grid(pts, vox=VOX, sigma_um=sigma_um, bounds=bounds)
    return d.sum(axis=2), lo, d          # (X,Y) 2D + 3D grid for topology


def piv_and_growth(rho0, rho1, dt):
    """Dense optical-flow PIV (px/frame) and continuity growth source."""
    a = rho0 / max(rho0.max(), 1e-9)
    b = rho1 / max(rho1.max(), 1e-9)
    vy, vx = optical_flow_tvl1(a, b, attachment=8, num_warp=3)   # rows(y), cols(x)
    vx_um = vx * VOX / dt
    vy_um = vy * VOX / dt
    speed = np.hypot(vx_um, vy_um)
    # continuity residual: source = drho/dt + div(rho v)  (per-frame, world units)
    drdt = (rho1 - rho0) / dt
    fx = rho0 * vx / dt
    fy = rho0 * vy / dt
    div = np.gradient(fx, axis=1) + np.gradient(fy, axis=0)
    source = ndi.gaussian_filter(drdt + div, 2.0)
    return speed, vx_um, vy_um, source


def topology_overlay(ax, pts, bounds):
    d, _ = st.density_grid(pts, vox=VOX, sigma_um=10.0, bounds=bounds)
    occ, n_main = st.occupancy(d, rel_thresh=0.14)
    edt = ndi.distance_transform_edt(occ) * VOX
    hm = h_maxima(edt, 6.0); hm[~occ] = 0
    lbl, nb = ndi.label(hm > 0)
    buds = np.array(ndi.center_of_mass(hm > 0, lbl, range(1, nb + 1))) if nb else np.empty((0, 3))
    n_branch, _, _ = st.count_branches(occ, vox_um=VOX, prune_um=55.0)
    skel = skeletonize(occ)
    sk = np.argwhere(skel)
    mip = d.max(axis=2)                                    # (X,Y)
    ax.imshow(mip.T, origin="lower", cmap="magma")
    if len(sk):
        ax.scatter(sk[:, 0], sk[:, 1], s=0.4, c="deepskyblue", alpha=0.35, linewidths=0)
    if len(buds):
        ax.scatter(buds[:, 0], buds[:, 1], s=90, facecolors="none",
                   edgecolors="cyan", linewidths=1.6)
    return len(buds), n_branch, n_main


def main():
    xl = torch.load(st.PT_DEFAULT, map_location="cpu", weights_only=False)
    gb = np.load(os.path.join(HERE, "_bounds.npy")); B = (gb[0], gb[1])

    fig, axs = plt.subplots(3, 4, figsize=(20, 15)); fig.patch.set_facecolor("black")

    def label(ax, s):
        ax.text(0.02, 0.98, s, transform=ax.transAxes, color="white",
                fontsize=11, va="top", ha="left")
    # precompute a global speed/growth scale for consistent colorbars
    piv_cache = {}
    smax = gmax = 0.0
    for t in TIMES:
        r0, _, _ = fields_2d(st.load_frame(xl, t), 6.0, B)
        r1, _, _ = fields_2d(st.load_frame(xl, t + DT), 6.0, B)
        spd, vx, vy, src = piv_and_growth(r0, r1, DT)
        tissue = r0 > 0.05 * r0.max()
        piv_cache[t] = (spd, vx, vy, src, tissue)
        smax = max(smax, np.percentile(spd[tissue], 95) if tissue.any() else 1)
        gmax = max(gmax, np.percentile(np.abs(src[tissue]), 95) if tissue.any() else 1)

    for j, t in enumerate(TIMES):
        # --- row 1 topology (watershed lobules + skeleton + main tube) ---
        res = vz.analyze(st.load_frame(xl, t), B)
        vz.draw(axs[0, j], res)
        label(axs[0, j], f"{'TOPOLOGY  ' if j == 0 else ''}t={t}\n"
              f"buds {res['n_bud']}  branch {res['n_branch']}  tube {res['n_main']}")

        spd, vx, vy, src, tissue = piv_cache[t]
        # --- row 2 growth (proliferation source) ---
        g = np.where(tissue, src, np.nan)
        im2 = axs[1, j].imshow(g.T, origin="lower", cmap="RdBu_r", vmin=-gmax, vmax=gmax)
        label(axs[1, j], f"{'GROWTH (∂ρ/∂t+∇·ρv)  ' if j == 0 else ''}t={t}→{t+DT}")
        # --- row 3 migration PIV ---
        s = np.where(tissue, spd, np.nan)
        im3 = axs[2, j].imshow(s.T, origin="lower", cmap="viridis", vmin=0, vmax=smax)
        step = 6
        X, Y = np.meshgrid(np.arange(0, spd.shape[0], step), np.arange(0, spd.shape[1], step))
        m = tissue[X, Y]
        axs[2, j].quiver(X[m], Y[m], vx[X, Y][m], vy[X, Y][m],
                         color="white", scale=None, width=0.0022, alpha=0.85)
        label(axs[2, j], f"{'MIGRATION PIV  ' if j == 0 else ''}t={t}→{t+DT}")

        for i in range(3):
            axs[i, j].set_facecolor("black")
            axs[i, j].set_xticks([]); axs[i, j].set_yticks([])

    fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004, wspace=0.02, hspace=0.02)
    out = os.path.join(HERE, "real_data_montage.png")
    fig.savefig(out, dpi=95, facecolor="black"); print("wrote", out)


if __name__ == "__main__":
    main()
