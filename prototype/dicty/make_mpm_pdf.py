"""make_mpm_pdf.py -- render the soft-MPM dicty sim vs the real movie and save a 1-page PDF + PNG.
Rows: REAL movie | MPM soft-body particles over cAMP field | SIM density | REAL density, over time."""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, ".."); sys.path.insert(0, "/workspace/Plexus/src")
import numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scenario_schema import load
import dicty_engine_mpm as E
import opt_dicty as O
from make_target import CROP, WIN
HERE = os.path.dirname(os.path.abspath(__file__)); DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
VIDEO = "/groups/saalfeld/home/allierc/GraphData/graphs_data/dicty/260228InterfaceTransferMirrorB1-DB-floor-fl2.mp4"
GREEN = LinearSegmentedColormap.from_list("c", ["#000", "#0a3a0a", "#1d7a1d", "#7CFC00"])
SPEC = sys.argv[1] if len(sys.argv) > 1 else "dicty_mpm_base"

t_frac, target, vtarget = O._target()
sc = load(os.path.join(HERE, "specs", SPEC + ".yaml")); sc.n_frames = 240; sc.record_every = 5
_, hist = E.run(sc, device=DEV)
T = len(hist); fmax = max(float(h["field"].max()) for h in hist) or 1.0
rec = (t_frac * (T - 1)).astype(int); K = len(t_frac)
simd = O.sim_density_seq([hist[i]["pos"] for i in rec])

cap = cv2.VideoCapture(VIDEO); n = int(cap.get(7)); W = int(cap.get(3)); H_ = int(cap.get(4))
x0, x1 = int(CROP["x0"] * W), int(CROP["x1"] * W); y0, y1 = int(CROP["y0"] * H_), int(CROP["y1"] * H_)
reals = []
for tf in t_frac:
    cap.set(1, int(tf * (n - 1))); _, fr = cap.read()
    reals.append(cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)[::-1])
cap.release()

fig, axs = plt.subplots(4, K, figsize=(2.4 * K, 9.6)); fig.patch.set_facecolor("black")
for c in range(K):
    h = hist[rec[c]]
    axs[0, c].imshow(reals[c], origin="lower"); axs[0, c].set_title(f"t={t_frac[c]:.2f}", color="w", fontsize=8)
    axs[1, c].imshow(h["field"].T, origin="lower", extent=[0, 1, 0, 1], cmap=GREEN, vmin=0, vmax=fmax * 0.5)
    axs[1, c].scatter(h["ppos"][:, 0], h["ppos"][:, 1], s=0.5, c="#FFA500", edgecolors="none")  # soft-body particles
    axs[1, c].set_xlim(0, 1); axs[1, c].set_ylim(0, 1); axs[1, c].set_title(f"N={h['count']}", color="w", fontsize=8)
    axs[2, c].imshow(simd[c].T, origin="lower", cmap="inferno")
    axs[3, c].imshow(target[c].T, origin="lower", cmap="inferno")
    for r in range(4):
        axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
for r, lab in enumerate(["REAL movie", "MPM soft bodies\n(8 particles/cell)", "SIM density", "REAL density"]):
    axs[r, 0].set_ylabel(lab, color="w", fontsize=10)
inner = float(O.radial_profile(simd[-1])[:3].sum())
fig.suptitle(f"Dictyostelium: soft-MPM simulation ({SPEC}) vs real movie   |   "
             f"final N={hist[-1]['count']}, inner-mass={inner:.2f} (real {O.radial_profile(target[-1])[:3].sum():.2f})",
             color="w", fontsize=12)
plt.tight_layout()
png = os.path.join(HERE, "mpm_vs_real.png"); pdf = os.path.join(HERE, "mpm_vs_real.pdf")
plt.savefig(png, dpi=80, facecolor="black"); plt.savefig(pdf, facecolor="black")
plt.close(fig)
print("wrote", os.path.basename(png), "+", os.path.basename(pdf), "| final N", hist[-1]["count"], "inner", round(inner, 3))
