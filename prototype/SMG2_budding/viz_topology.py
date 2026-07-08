"""Demonstration visualization of the SMG2 topological analysis.

Shows the FULL extracted structure so the analysis is visibly working:
  * every bud as a distinct WATERSHED lobule (colored region) -- all buds mapped
  * the duct CENTERLINE skeleton (medial axis)
  * the MAIN TUBE (longest centerline path) highlighted
  * BRANCH points (Y-junctions, persistence-pruned)
over several timepoints.
"""
import os
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import smg_topo as st

HERE = os.path.dirname(__file__)
VOX = 4.0


def analyze(pts, bounds, sigma_bud=11.0, sigma_branch=22.0, thr=0.14,
            min_dist=10, r_min_um=13.0, prune_um=110.0):
    # --- FINE scale: bud lobules (watershed on inscribed radius) ---
    d, _ = st.density_grid(pts, vox=VOX, sigma_um=sigma_bud, bounds=bounds)
    occ, _ = st.occupancy(d, rel_thresh=thr)
    edt = ndi.distance_transform_edt(occ) * VOX
    seeds = peak_local_max(edt, min_distance=min_dist, threshold_abs=r_min_um, labels=occ)
    mk = np.zeros(occ.shape, int)
    for i, s in enumerate(seeds, 1):
        mk[tuple(s)] = i
    ws = watershed(-edt, mk, mask=occ)                      # lobule segmentation
    n_bud = len(seeds)
    # --- COARSE scale: duct tree (lobules merged -> only major bifurcations) ---
    dc, _ = st.density_grid(pts, vox=VOX, sigma_um=sigma_branch, bounds=bounds)
    occ_c, n_main = st.occupancy(dc, rel_thresh=0.18)
    G, sv = st._skeleton_graph(occ_c)
    n_branch, _, H = st.count_branches(occ_c, vox_um=VOX, prune_um=prune_um)
    branch_xy = np.array([sv[n][:2] for n in H if H.degree(n) >= 3]) if H is not None else np.empty((0, 2))
    # main tube = longest weighted path in the largest skeleton component
    main_path = np.empty((0, 2))
    if G.number_of_nodes():
        comp = max(nx.connected_components(G), key=len)
        Gm = nx.Graph(G.subgraph(comp))
        for u, v in Gm.edges():
            Gm[u][v]["w"] = float(np.linalg.norm((sv[u] - sv[v]) * VOX))
        def far(src):
            dl = nx.single_source_dijkstra_path_length(Gm, src, weight="w")
            return max(dl, key=dl.get)
        a = far(next(iter(comp))); b = far(a)
        path = nx.shortest_path(Gm, a, b, weight="w")
        main_path = np.array([sv[n][:2] for n in path])
    return dict(d=d, occ=occ, ws=ws, edt=edt, seeds=seeds, sv=sv,
                branch_xy=branch_xy, main_path=main_path,
                n_bud=n_bud, n_branch=n_branch, n_main=n_main)


def draw(ax, res):
    ax.set_facecolor("black")
    d, occ, ws, edt = res["d"], res["occ"], res["ws"], res["edt"]
    mip = d.max(axis=2).T                                    # (Y,X)
    ax.imshow(mip, origin="lower", cmap="gray", alpha=0.9)
    # colored lobules: label at the medial (max-edt) z per column
    zsel = np.argmax(edt, axis=2)
    lab2d = np.take_along_axis(ws, zsel[:, :, None], axis=2)[:, :, 0].T
    occ2d = occ.any(axis=2).T
    lab2d = np.where(occ2d, lab2d, 0)
    rng = np.random.default_rng(0)
    colors = rng.permutation(plt.cm.tab20(np.linspace(0, 1, 20)))
    cmap = ListedColormap(colors)
    ax.imshow(np.ma.masked_where(lab2d == 0, lab2d % 20), origin="lower",
              cmap=cmap, alpha=0.45, interpolation="nearest")
    # skeleton (thin), main tube (thick), branch points, bud seeds
    sv = res["sv"]
    if len(sv):
        ax.scatter(sv[:, 0], sv[:, 1], s=0.5, c="white", alpha=0.5, linewidths=0)
    if len(res["main_path"]):
        ax.plot(res["main_path"][:, 0], res["main_path"][:, 1], "-",
                c="orange", lw=3, alpha=0.9, label="main tube")
    if len(res["branch_xy"]):
        ax.scatter(res["branch_xy"][:, 0], res["branch_xy"][:, 1], s=140,
                   marker="s", facecolors="none", edgecolors="red", linewidths=2.2,
                   label="branch")
    if len(res["seeds"]):
        ax.scatter(res["seeds"][:, 0], res["seeds"][:, 1], s=70, c="cyan",
                   marker="o", edgecolors="k", linewidths=0.6, label="bud")
    ax.set_xticks([]); ax.set_yticks([])


def main():
    xl = torch.load(st.PT_DEFAULT, map_location="cpu", weights_only=False)
    gb = np.load(os.path.join(HERE, "_bounds.npy")); B = (gb[0], gb[1])
    times = [0, 184, 368, 552]
    fig, axs = plt.subplots(1, len(times), figsize=(6 * len(times), 6.2))
    fig.patch.set_facecolor("black")
    for ax, t in zip(axs, times):
        res = analyze(st.load_frame(xl, t), B)
        draw(ax, res)
        ax.text(0.02, 0.98, f"t={t}\nbuds {res['n_bud']}  branch {res['n_branch']}  "
                f"tube {res['n_main']}", transform=ax.transAxes, color="white",
                fontsize=11, va="top", ha="left")
        print(f"t={t}: buds={res['n_bud']} branch={res['n_branch']} tube={res['n_main']}")
    axs[0].legend(loc="lower left", fontsize=9, framealpha=0.25, labelcolor="white")
    fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004, wspace=0.02)
    out = os.path.join(HERE, "_topo_demo.png")
    fig.savefig(out, dpi=95, facecolor="black"); print("wrote", out)


if __name__ == "__main__":
    main()
