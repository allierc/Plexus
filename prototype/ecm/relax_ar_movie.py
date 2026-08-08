"""The attraction_repulsion law relaxing the real membrane sheet, as a movie.

Same frozen mid-run sheet as relax_movie.py, but driven by Plexus's attraction_repulsion operator in
its purely repulsive form instead of by crosslink springs. Colour is each node's nearest-neighbour
distance against the even spacing, so an evening-out sheet goes uniformly dark; the trace is d/hex,
where 1.00 is a perfect lattice and 0.88 is blue noise.

The archived `blue` parameters are NOT used and would not work: f(0) = p0 - p2 is +0.022 for blue, an
attractive core that welds close pairs together (2D: d/hex 0.471 -> 0.242, worse than random). p0 = 0
leaves a single decaying repulsion -- the law CGI uses to scatter points evenly over a surface.
"""
import math
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.animation import FFMpegWriter
matplotlib.rcParams["animation.ffmpeg_path"] = os.path.join(
    os.path.dirname(sys.executable), "ffmpeg")
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relax_bench import DEV, load, metrics

COLS = ["#10243a", "#1f6f8b", "#3aa17e", "#8cc04f", "#e8d44d", "#f0913a", "#e0452b"]
CMAP = mc.LinearSegmentedColormap.from_list("dev", COLS)
P2, P3 = 1.6, 1.0


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "85_fixed_repel20"
    dt = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    iters, every, knn = 300, 2, 19
    X0, bi, bj, st = load(run_name, 0.5)
    q = np.random.default_rng(0).normal(size=(20000, 3)); q /= np.linalg.norm(q, axis=1)[:, None]

    X = torch.tensor(X0, dtype=torch.float32, device=DEV)
    rad = X.norm(dim=1, keepdim=True).clone()
    R = float(rad.median())
    sp = math.sqrt(4.0 * math.pi * R * R / X.shape[0])
    sig = 0.7 * sp
    rng = 3.0 * sp

    fig = plt.figure(figsize=(11.0, 5.6), facecolor="black")
    axA = fig.add_subplot(1, 2, 1, projection="3d", facecolor="black", computed_zorder=False)
    axB = fig.add_subplot(1, 2, 2, facecolor="black")
    fig.subplots_adjust(0.02, 0.02, 0.98, 0.94, wspace=0.06)
    wri = FFMpegWriter(fps=15, metadata={"title": "attraction_repulsion relaxation"})
    hist = []
    out = f"/workspace/Plexus/log/okuda_ECM/_relax_ar_{run_name}_dt{dt:g}.mp4"

    with wri.saving(fig, out, dpi=110):
        for t in range(iters + 1):
            Xn = X.detach().cpu().numpy()
            _, nb = cKDTree(Xn).query(Xn, k=knn)
            I = torch.arange(len(Xn), device=DEV).repeat_interleave(knn - 1)
            J = torch.tensor(nb[:, 1:].ravel(), dtype=torch.long, device=DEV)
            d = X[J] - X[I]
            r2 = (d * d).sum(-1)
            m = r2 < rng * rng
            f = (-P2 * torch.exp(-((r2 ** P3)) / (2.0 * sig * sig))) * m
            dp = torch.zeros_like(X).index_add_(0, I, f[:, None] * d)
            deg = torch.zeros(len(Xn), device=DEV).index_add_(0, I, m.to(X.dtype))
            X = X + dt * (dp / deg.clamp(min=1.0)[:, None])
            X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12) * rad

            if t % every:
                continue
            Xn = X.detach().cpu().numpy()
            dh, cv, gp = metrics(Xn, q)
            hist.append((t, dh))
            tree = cKDTree(Xn)
            dd, nn2 = tree.query(Xn, k=2)
            dev = np.abs(dd[:, 1] - sp) / sp
            axA.clear(); axA.set_facecolor("black"); axA.axis("off")
            axA.set_xlim(-R, R); axA.set_ylim(-R, R); axA.set_zlim(-R, R)
            axA.set_box_aspect((1, 1, 1)); axA.view_init(elev=18, azim=35)
            dv3 = np.array([np.cos(np.radians(18)) * np.cos(np.radians(35)),
                            np.cos(np.radians(18)) * np.sin(np.radians(35)), np.sin(np.radians(18))])
            far = (Xn @ dv3) > 0
            segs = np.stack([Xn[far], Xn[nn2[:, 1]][far]], axis=1)
            lc = Line3DCollection(segs, cmap=CMAP, linewidths=0.6)
            lc.set_array(np.clip(dev[far] / 0.6, 0, 1)); lc.set_clim(0, 1)
            axA.add_collection3d(lc)
            axA.text2D(0.02, 0.95, f"iteration {t}", transform=axA.transAxes, color="white", fontsize=10)

            axB.clear(); axB.set_facecolor("black")
            h = np.array(hist)
            axB.plot(h[:, 0], h[:, 1], "-", color="#8cc04f", lw=2)
            axB.set_xlim(0, iters); axB.set_ylim(0.4, 1.0)
            axB.set_xlabel("iteration", color="#bbb")
            axB.set_ylabel("d / hexagonal spacing", color="#bbb")
            axB.tick_params(colors="#bbb")
            for spine in axB.spines.values(): spine.set_color("#666")
            axB.axhline(0.88, color="#e0452b", ls="--", lw=1)
            axB.text(iters * 0.55, 0.892, "blue noise", color="#e0452b", fontsize=8)
            axB.text(0.03, 0.92, f"d/hex = {dh:.3f}   cv = {cv:.3f}",
                     transform=axB.transAxes, color="white", fontsize=11)
            wri.grab_frame()
    plt.close(fig)
    print(f"wrote {out}  ({len(hist)} frames, d/hex {hist[0][1]:.3f} -> {hist[-1][1]:.3f})")


if __name__ == "__main__":
    main()
