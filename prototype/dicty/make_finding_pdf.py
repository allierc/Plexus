"""make_finding_pdf.py -- one-page PDF illustrating the headline finding:
the real data stays MULTI-MOUND, while the best simulation makes multi-mound only TRANSIENTLY
(~frame 400) then COLLAPSES to a single blob (Est #82 — no stable multi-mound attractor)."""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, ".."); sys.path.insert(0, "/workspace/Plexus/src")
import numpy as np, cv2, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scenario_schema import load
import dicty_engine as E, opt_dicty as O
from make_target import CROP, WIN
GREEN = LinearSegmentedColormap.from_list("c", ["#000", "#0a3a0a", "#1d7a1d", "#7CFC00"])
VIDEO = "/groups/saalfeld/home/allierc/GraphData/graphs_data/dicty/260228InterfaceTransferMirrorB1-DB-floor-fl2.mp4"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"

# best point-cell config, run LONG to expose the transient->collapse
sc = load("specs/dicty_good.yaml"); sc.sets["cell"]["n"] = 1000; sc.n_frames = 1600; sc.record_every = 20
_, hist = E.run(sc, device=DEV)
T = len(hist); fmax = max(float(h["field"].max()) for h in hist) or 1.0
snap = [int(x / 1600 * (T - 1)) for x in (120, 320, 640, 1100, 1600)]     # frames to show the trajectory

# real frames across the aggregation window
cap = cv2.VideoCapture(VIDEO); n = int(cap.get(7)); W = int(cap.get(3)); Hh = int(cap.get(4))
x0, x1 = int(CROP["x0"] * W), int(CROP["x1"] * W); y0, y1 = int(CROP["y0"] * Hh), int(CROP["y1"] * Hh)
reals = []
for tf in np.linspace(WIN[0], WIN[1], 5):
    cap.set(1, int(tf * (n - 1))); _, fr = cap.read(); reals.append(cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)[::-1])
cap.release()

fig, axs = plt.subplots(2, 5, figsize=(15, 6.4)); fig.patch.set_facecolor("black")
for c in range(5):
    axs[0, c].imshow(reals[c], origin="lower"); axs[0, c].set_title(f"t={np.linspace(WIN[0],WIN[1],5)[c]:.2f}", color="w", fontsize=9)
    h = hist[snap[c]]
    inner = float(O.radial_profile(O.sim_density_seq([h["pos"]])[0])[:3].sum())
    nm = O.n_mounds(O.fov_image(h["pos"]))
    axs[1, c].imshow(h["field"].T, origin="lower", extent=[0, 1, 0, 1], cmap=GREEN, vmin=0, vmax=fmax * 0.5)
    axs[1, c].scatter(h["pos"][:, 0], h["pos"][:, 1], s=2, c="#FFA500"); axs[1, c].set_xlim(0, 1); axs[1, c].set_ylim(0, 1)
    axs[1, c].set_title(f"frame {snap[c]*20}  mounds={nm}  inner={inner:.2f}", color="w", fontsize=9)
    for r in (0, 1):
        axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
axs[0, 0].set_ylabel("REAL data\n(stays multi-mound)", color="cyan", fontsize=11)
axs[1, 0].set_ylabel("BEST sim\n(transient → collapse)", color="orange", fontsize=11)
fig.suptitle("Dictyostelium — why no simulation matches the data\n"
             "The best point-cell model (sense_sat, D=0.0042) forms REAL-like multi-mounds only TRANSIENTLY (~frame 400),\n"
             "then COLLAPSES to a single blob (inner-mass→0.8+); the real aggregate stays multi-mound. No stable multi-mound\n"
             "attractor exists in 12 tested 2D mechanisms — most likely a 3-D mound viewed in projection, unreachable in 2D.",
             color="w", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("good_vs_real.png", dpi=85, facecolor="black"); plt.savefig("good_vs_real.pdf", facecolor="black")
plt.close(fig)
print("wrote good_vs_real.pdf/png | snapshots(frame,mounds,inner):",
      [(snap[c]*20, O.n_mounds(O.fov_image(hist[snap[c]]["pos"])),
        round(float(O.radial_profile(O.sim_density_seq([hist[snap[c]]["pos"]])[0])[:3].sum()),2)) for c in range(5)])
