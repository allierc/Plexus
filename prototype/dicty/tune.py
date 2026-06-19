"""Regime finder for the spring + chemotaxis dicty model: try a few scales, score
aggregation (peak coarse-grid density / mean), montage each."""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, ".."); sys.path.insert(0, "/workspace/Plexus/src")
import numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scenario_schema import load
import dicty_engine
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GREEN = LinearSegmentedColormap.from_list("c", ["#000", "#0a3a0a", "#1d7a1d", "#7CFC00"])


def agg_index(pos, bins=24):
    h, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=bins, range=[[0, 1], [0, 1]])
    h = h / max(h.sum(), 1)
    return float(h.max() / (h.mean() + 1e-9))


def run_cfg(cfg):
    sc = load("specs/dicty_aggregate.yaml")
    sc.n_frames = 300; sc.record_every = 6; sc.dt = cfg["dt"]; sc.sets["cell"]["n"] = cfg["N"]
    f = sc.fields["camp"]; f["diffusion"] = cfg["diff"]; f["decay"] = cfg["decay"]
    for o in sc.operators:
        if o.op == "spring": o.params.update(k_rep=cfg["krep"], r0=cfg["r0"], kadh=cfg["kadh"],
                                             r_on=cfg["ron"], mu_f=cfg["muf"])
        elif o.op == "secrete": o.params["rate"] = cfg["sec"]
        elif o.op == "sense": o.params["gain"] = cfg["gain"]
        elif o.op == "random_walk": o.params["strength"] = cfg["noise"]
    _, hist = dicty_engine.run(sc, device=DEV)
    return hist


CFGS = [
    # name           dt    N    krep  r0    kadh  ron   muf   diff   decay sec  gain  noise
    dict(name="S1_springonly", dt=0.3, N=600, krep=50, r0=0.05, kadh=50, ron=0.10, muf=0.05, diff=0.01, decay=0.2, sec=0.0, gain=0.0, noise=0.02),
    dict(name="S2_spring+chemo", dt=0.3, N=600, krep=50, r0=0.05, kadh=50, ron=0.10, muf=0.05, diff=0.006, decay=0.3, sec=1.0, gain=2.0, noise=0.02),
    dict(name="S3_strongadh", dt=0.3, N=600, krep=50, r0=0.04, kadh=120, ron=0.10, muf=0.06, diff=0.006, decay=0.3, sec=1.0, gain=2.0, noise=0.02),
    dict(name="S4_userparams", dt=0.5, N=400, krep=50, r0=0.10, kadh=50, ron=0.14, muf=0.05, diff=0.006, decay=0.3, sec=1.0, gain=2.0, noise=0.02),
]

results = []
for cfg in CFGS:
    hist = run_cfg(cfg)
    ai0, ai1 = agg_index(hist[0]["pos"]), agg_index(hist[-1]["pos"])
    results.append((cfg, hist, ai0, ai1))
    print(f"{cfg['name']:16s} agg {ai0:.2f}->{ai1:.2f}  fieldmax {hist[-1]['field'].max():.2f}  N {hist[-1]['count']}", flush=True)

fig, axs = plt.subplots(len(CFGS), 4, figsize=(16, 4 * len(CFGS))); fig.patch.set_facecolor("black")
for row, (cfg, hist, ai0, ai1) in enumerate(results):
    T = len(hist); fmax = max(float(h["field"].max()) for h in hist) or 1.0
    for col, i in enumerate([0, T // 3, 2 * T // 3, T - 1]):
        ax = axs[row, col]; h = hist[i]
        ax.imshow(h["field"].T, origin="lower", extent=[0, 1, 0, 1], cmap=GREEN, vmin=0, vmax=fmax * 0.5)
        ax.scatter(h["pos"][:, 0], h["pos"][:, 1], s=3, c="#FFA500", edgecolors="none")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{cfg['name']} fr{i} agg{agg_index(h['pos']):.1f}", color="white", fontsize=8)
plt.tight_layout(); plt.savefig("tune_montage.png", dpi=65, facecolor="black")
print("wrote tune_montage.png")
