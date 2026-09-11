"""fig_alignment -- the fibre axes of the two sheets: a line per cell, and how aligned neighbours are.

Top row: each interior cell drawn as a short line along its fitted axis phi_j at its centroid,
length proportional to g_j. Bottom left: the distribution of axis angles (mod 180 deg). Bottom right:
LOCAL alignment -- for each cell, |mean over neighbours within 3 cell radii of exp(2i phi)|, 1 = all
parallel, 0 = random -- as distributions. Healthy blue, HCM red (two sources).
"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__))
RED, BLUE, WHITE, GREY = "#e0604a", "#4a90e0", "#f2f2f2", "#8a8a8a"
plt.rcParams.update({"figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
                     "axes.edgecolor": GREY, "axes.labelcolor": WHITE, "xtick.color": WHITE, "ytick.color": WHITE,
                     "text.color": WHITE, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})
FITS = dict(healthy=("s4_live_r8_p120_d150_g2", "data"), hcm=("hcm_r3_p120_d150_g2", "data_hcm"))


def sheet(name):
    tag, dd = FITS[name]
    z = np.load(os.path.join(HERE, "out", "fits", tag, "params.npz"))
    cen = np.load(os.path.join(HERE, dd, "cell_centroids_world.npy"))
    inter = z["interior"].astype(bool)
    phi, g = z["phi"], z["g"]
    tree = cKDTree(cen); r = 3 * 0.7 / np.sqrt(len(cen))          # ~3 cell radii
    local = np.zeros(len(cen))
    for j in range(len(cen)):
        nb = [k for k in tree.query_ball_point(cen[j], r) if k != j and inter[k]]
        local[j] = np.abs(np.mean(np.exp(2j * phi[nb]))) if len(nb) >= 3 else np.nan
    return dict(cen=cen, phi=phi, g=g, inter=inter, local=local,
                order=float(np.abs(np.mean(np.exp(2j * phi[inter])))))


H, D = sheet("healthy"), sheet("hcm")
fig, ax = plt.subplots(2, 2, figsize=(12, 12), gridspec_kw=dict(height_ratios=[1.6, 1], hspace=0.25, wspace=0.25))
for a, S, col, name in ((ax[0, 0], H, BLUE, "healthy"), (ax[0, 1], D, RED, "HCM")):
    m = S["inter"]; c = S["cen"][m]; ph = S["phi"][m]; L = 0.012 * np.clip(S["g"][m] / 0.06, 0.15, 2.5)
    d = np.stack([np.cos(ph), np.sin(ph)], 1) * L[:, None]
    a.add_collection(LineCollection(np.stack([c - d, c + d], 1), colors=col, linewidths=1.6))
    a.scatter(S["cen"][~m, 0], S["cen"][~m, 1], s=4, c=GREY, lw=0)
    a.set_xlim(0.14, 0.87); a.set_ylim(0.14, 0.87); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_xlabel(f"{name}: fitted fibre axis per cell, length ~ g\n(grey dots = band cells, not fitted)\n"
                 f"global axis order {S['order']:.2f}  (0 = random, 1 = all parallel)")
a = ax[1, 0]
bins = np.linspace(0, 180, 25)
for S, col, name in ((H, BLUE, "healthy"), (D, RED, "HCM")):
    a.hist(np.degrees(S["phi"][S["inter"]] % np.pi), bins, histtype="step", lw=2, color=col, density=True,
           label=f"{name} ({int(S['inter'].sum())} cells)")
a.set_xlabel("fitted fibre axis angle (degrees, mod 180; 0 = image x axis)"); a.set_ylabel("density over interior cells")
a.legend(loc="upper right", fontsize=9)
a = ax[1, 1]
bins = np.linspace(0, 1, 21)
for S, col, name in ((H, BLUE, "healthy"), (D, RED, "HCM")):
    v = S["local"][S["inter"]]; v = v[np.isfinite(v)]
    a.hist(v, bins, histtype="step", lw=2, color=col, density=True, label=f"{name}, median {np.median(v):.2f}")
a.set_xlabel("local alignment of a cell's axis with its neighbours'\n(within ~3 cell radii; 0 = they point anywhere, 1 = all parallel)")
a.set_ylabel("density over interior cells"); a.legend(loc="upper left", fontsize=9)
for a, s in zip(ax.ravel(), "ABCD"):
    a.text(-0.02, 1.01, s, transform=a.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="right")
fig.savefig(os.path.join(HERE, "out", "figures", "fig6_alignment.png"), dpi=130, bbox_inches="tight")
print("global order healthy %.2f HCM %.2f; local median healthy %.2f HCM %.2f" % (
    H["order"], D["order"], np.nanmedian(H["local"][H["inter"]]), np.nanmedian(D["local"][D["inter"]])))
