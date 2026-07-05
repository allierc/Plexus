"""audit_plot.py -- visual consolidation of the independent trajectory audit.
Reads the --eval_dump npz files; recomputes sim-vs-real loop morphology from scratch (no pipeline metric).
Figure: (A) size ratio, (B) enclosed-area ratio, (C) loopiness sim-vs-real, (D) example loop overlays.
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

REC = "/tmp/cardio_audit/wide400.npz"
Z = np.load(REC)
sim, real, mov = Z["sim_d"], Z["real_d"], Z["mov"].astype(bool)
s = sim[:, mov]; r = real[:, mov]
G, M, _ = s.shape


def stats(d):
    dc = d - d.mean(0, keepdims=True)
    peak = np.abs(dc).max(0).max(1)
    x, y = dc[..., 0], dc[..., 1]
    area = 0.5 * (x * np.roll(y, -1, 0) - np.roll(x, -1, 0) * y).sum(0)
    loopy = np.abs(area) / (np.pi * (peak / 2) ** 2 + 1e-12)
    return peak, area, loopy


sp, sa, sl = stats(s); rp, ra, rl = stats(r)
peak_ratio = sp / (rp + 1e-12)
area_ratio = np.abs(sa) / (np.abs(ra) + 1e-12)

fig = plt.figure(figsize=(16, 10))
fig.suptitle("Independent cardio-MPM audit — record model wide400 (pipeline LS=0.365)\n"
             "sim (red) vs real (green), interior moving nodes recomputed out-of-pipeline",
             fontsize=13, fontweight="bold")

axA = fig.add_subplot(2, 3, 1)
axA.hist(np.clip(peak_ratio, 0, 2), bins=60, color="#d1495b", alpha=0.85)
axA.axvline(1.0, color="k", ls="--", lw=1); axA.axvline(np.median(peak_ratio), color="navy", lw=2,
            label=f"median={np.median(peak_ratio):.2f}")
axA.set_title("A. SIZE residual: sim/real peak excursion"); axA.set_xlabel("sim peak / real peak")
axA.legend()

axB = fig.add_subplot(2, 3, 2)
axB.hist(np.clip(area_ratio, 0, 2), bins=60, color="#5b8c5a", alpha=0.85)
axB.axvline(1.0, color="k", ls="--", lw=1); axB.axvline(np.median(area_ratio), color="navy", lw=2,
            label=f"median={np.median(area_ratio):.2f}")
axB.set_title("B. AREA residual: sim/real |enclosed area|"); axB.set_xlabel("sim |area| / real |area|")
axB.legend()

axC = fig.add_subplot(2, 3, 3)
axC.scatter(rl, sl, s=3, alpha=0.15, color="#3a6ea5")
lim = np.percentile(np.concatenate([rl, sl]), 99)
axC.plot([0, lim], [0, lim], "k--", lw=1)
axC.set_xlim(0, lim); axC.set_ylim(0, lim)
axC.set_title(f"C. LOOPINESS (|area|/disc): sim under-encloses\nreal med={np.median(rl):.2f}  sim med={np.median(sl):.2f}")
axC.set_xlabel("real loopiness"); axC.set_ylabel("sim loopiness")

# D: overlay example loops — sample nodes spanning the LS range, on a shared per-panel scale
axD = fig.add_subplot(2, 1, 2)
rng = np.random.default_rng(0)
pick = rng.choice(M, 12, replace=False)
pick = pick[np.argsort(peak_ratio[pick])]
for j, m in enumerate(pick):
    sc = s[:, m] - s[:, m].mean(0); rc = r[:, m] - r[:, m].mean(0)
    sca = 1.0 / (np.abs(rc).max() + 1e-9)                 # normalise each panel by REAL extent
    ox = (j % 6) * 3.0; oy = -(j // 6) * 3.0
    axD.plot(rc[:, 0] * sca + ox, rc[:, 1] * sca + oy, color="#2e8b57", lw=1.6)
    axD.plot(sc[:, 0] * sca + ox, sc[:, 1] * sca + oy, color="#d1495b", lw=1.4)
    axD.text(ox, oy + 1.4, f"sz{peak_ratio[m]:.2f} ar{area_ratio[m]:.2f}", ha="center", fontsize=8)
axD.set_title("D. Example interior loops (green=real, red=sim), each normalised to its REAL extent — "
              "sim is smaller AND flatter (encloses less area)")
axD.set_aspect("equal"); axD.axis("off")

fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "/workspace/Plexus/prototype/cardio_mpm/audit_wide400.png"
fig.savefig(out, dpi=110, facecolor="white"); print("saved", out)
