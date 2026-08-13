"""Render the relaxation itself: a frozen sheet, only the network acting, as a movie.

Colour is |L - L*| / L*, the deviation of each crosslink from the ONE length the network is relaxing
toward -- so a sheet that is evening out goes uniformly dark, and one that has stalled keeps its bright
edges wherever it started with them. That makes the plateau visible rather than only measurable.
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
# ffmpeg ships next to the interpreter, not on PATH -- the same thing `ecm_render._exe` handles
matplotlib.rcParams["animation.ffmpeg_path"] = os.path.join(
    os.path.dirname(sys.executable), "ffmpeg")
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relax_bench import DEV, load, metrics

COLS = ["#10243a", "#1f6f8b", "#3aa17e", "#8cc04f", "#e8d44d", "#f0913a", "#e0452b"]
CMAP = mc.LinearSegmentedColormap.from_list("dev", COLS)


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "82_mesh_restlength"
    w_rep = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    push_only = len(sys.argv) > 3 and sys.argv[3] == "push"
    iters, every = 1500, 10
    X0, bi, bj, st = load(run_name, 0.5)
    g = np.random.default_rng(0)
    q = g.normal(size=(20000, 3)); q /= np.linalg.norm(q, axis=1)[:, None]

    X = torch.tensor(X0, dtype=torch.float32, device=DEV)
    I = torch.tensor(bi, dtype=torch.long, device=DEV)
    J = torch.tensor(bj, dtype=torch.long, device=DEV)
    rad = X.norm(dim=1, keepdim=True).clone()
    R = float(rad.median())
    # the spacing an even sheet of this many nodes would have on this sphere. Every rest length is
    # this ONE value: a rest length that instead tracks the current mean chases the long bonds that
    # span the holes, so the springs pull toward a target the repulsion is trying to undo.
    sp = math.sqrt(4.0 * math.pi * R * R / X.shape[0])
    L0 = torch.full((I.numel(),), sp, device=DEV)
    NB = None

    fig = plt.figure(figsize=(11.0, 5.6), facecolor="black")
    axA = fig.add_subplot(1, 2, 1, projection="3d", facecolor="black", computed_zorder=False)
    axB = fig.add_subplot(1, 2, 2, facecolor="black")
    fig.subplots_adjust(0.02, 0.02, 0.98, 0.94, wspace=0.06)
    wri = FFMpegWriter(fps=15, metadata={"title": "membrane relaxation"})
    hist = []
    tag = "springs" if w_rep == 0 else f"springs+repulsion_w{w_rep:g}"
    out = f"/workspace/Plexus/log/okuda_ECM/_relax_{run_name}_{tag}.mp4"

    with wri.saving(fig, out, dpi=110):
        for t in range(iters + 1):
            if t % 20 == 0:                          # rewire, and refresh who is near whom
                Xn = X.detach().cpu().numpy()
                _, nb = cKDTree(Xn).query(Xn, k=7)
                NB = torch.tensor(nb[:, 1:], dtype=torch.long, device=DEV)
                if t:
                    a_ = np.repeat(np.arange(len(Xn)), 6); b_ = nb[:, 1:].ravel()
                    kp = np.unique(np.minimum(a_, b_) * (len(Xn) + 1) + np.maximum(a_, b_))
                    I = torch.tensor(kp // (len(Xn) + 1), dtype=torch.long, device=DEV)
                    J = torch.tensor(kp % (len(Xn) + 1), dtype=torch.long, device=DEV)
                    L0 = torch.full(((I.numel()),), sp, device=DEV)

            dvec = X[J] - X[I]
            L = dvec.norm(dim=1).clamp_min(1e-12)
            ext = L - L0
            if push_only:                # the attractive half deleted: a crosslink that can only push
                ext = ext.clamp_max(0.0)
            f = (5e3 * ext / 2e3)[:, None] * (dvec / L[:, None])
            F = torch.zeros_like(X).index_add_(0, I, f).index_add_(0, J, -f)
            # excluded volume: secreted protein occupies space, so nodes closer than the even spacing
            # push apart whether or not a crosslink joins them. This is the term that opens the knots
            # of short edges -- springs let them sit, because a short edge relaxes its own rest length.
            if w_rep > 0.0:
                dr = X[:, None, :] - X[NB]
                Lr = dr.norm(dim=-1).clamp_min(1e-12)
                F = F + (w_rep * 5e3 / 2e3) * (
                    (dr / Lr[..., None]) * (sp - Lr).clamp_min(0.0)[..., None]).sum(1)
            X = X + F * 4e-3
            X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12) * rad
            L0 = L0 + ((L.mean() - L0) / 60.0)

            if t % every:
                continue
            Xn = X.detach().cpu().numpy()
            Ln = (X[J] - X[I]).norm(dim=1).detach().cpu().numpy()
            dev = np.abs(Ln - Ln.mean()) / Ln.mean()
            hist.append((t, float(Ln.std() / Ln.mean())))
            for ax in (axA,):
                ax.clear(); ax.set_facecolor("black"); ax.axis("off")
                ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R, R)
                ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=18, azim=35)
            d, _, _ = (np.array([0., 0., 1.]),) * 3
            dv3 = np.array([np.cos(np.radians(18)) * np.cos(np.radians(35)),
                            np.cos(np.radians(18)) * np.sin(np.radians(35)), np.sin(np.radians(18))])
            mid = 0.5 * (Xn[bi_ := I.cpu().numpy()] + Xn[bj_ := J.cpu().numpy()])
            far = (mid @ dv3) > 0
            segs = np.stack([Xn[bi_][far], Xn[bj_][far]], axis=1)
            lc = Line3DCollection(segs, cmap=CMAP, linewidths=0.5)
            lc.set_array(np.clip(dev[far] / 0.6, 0, 1)); lc.set_clim(0, 1)
            axA.add_collection3d(lc)
            axA.text2D(0.02, 0.95, f"iteration {t}", transform=axA.transAxes, color="white", fontsize=10)

            axB.clear(); axB.set_facecolor("black")
            h = np.array(hist)
            axB.plot(h[:, 0], h[:, 1], "-", color="#8cc04f", lw=2)
            axB.set_xlim(0, iters); axB.set_ylim(0, max(0.30, h[:, 1].max() * 1.1))
            axB.set_xlabel("iteration", color="#bbb"); axB.set_ylabel("cv of crosslink length", color="#bbb")
            axB.tick_params(colors="#bbb")
            for spine in axB.spines.values(): spine.set_color("#666")
            axB.axhline(0.05, color="#e0452b", ls="--", lw=1)
            axB.text(iters * 0.55, 0.062, "a packed sheet", color="#e0452b", fontsize=8)
            axB.text(0.03, 0.92, f"cv = {h[-1,1]:.3f}", transform=axB.transAxes, color="white", fontsize=11)
            wri.grab_frame()
    plt.close(fig)
    print(f"wrote {out}  ({len(hist)} frames, cv {hist[0][1]:.3f} -> {hist[-1][1]:.3f})")


if __name__ == "__main__":
    main()
