"""exp17: a run's NEURON CLUSTERS IN 3-D (Cedric, 2026-10-02: "a slide that shows the neuron clustering, two panels:
oblique, large coloured dots; second panel iso-surfaces as already used in the watcher for exp04 ion channels").

Local analysis. The clusters are the run's own (trainer._analyse_trace's): KMeans, 8 clusters, on the neurons' learned
constants -- the rate 1/tau (softplus), the rest V and the 22 stimulus weights B -- each block z-scored and weighted
to unit total variance. Two off-screen VTK renders from the same oblique camera (the destriped 3-D movie's: over the
brain, tilted 40 degrees toward the tail):
  a  every neuron a large dot coloured by its cluster
  b  one iso-surface per cluster, the watcher's contour look (a gaussian-blurred density on a grid, contoured; matte,
     translucent, in the cluster's colour) -- contoured on the cluster's ENRICHMENT, its local share of the neurons over
     its brain-wide share, at 2: a plain density surface of the large clusters filled the whole brain
Writes experiments/exp17_zapbench_graphcast/presentation/figs/clusters_3d_<run>.png (+ a copy in png/) and
data/clusters_<run>.json (the labels' sizes).

    python tools/exp17_clusters.py zap_dm_ng_rl1lo
"""
import json
import os
import shutil
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
K = 8
COLS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22"]


def labels_of(name, k=K):
    """The run's clusters, as `_analyse_trace` makes them (k clusters), and the positions (um) of its recording."""
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    from sklearn.cluster import KMeans
    spec = T.load(name)
    out = T.out_dir(spec, None)
    fit = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location="cpu")["fitted"]
    have = [b for b in ("tau", "rest", "gain", "input") if f"neuron.{b}" in fit]
    blocks = [torch.nn.functional.softplus(fit[f"neuron.{b}"].float()) if b == "tau" else fit[f"neuron.{b}"].float()
              for b in have]
    emb = torch.cat([((b - b.mean(0)) / b.std(0).clamp(min=1e-9)) / b.shape[1] ** 0.5 for b in blocks], 1).numpy()
    lab = KMeans(k, n_init=4, random_state=0).fit_predict(emb)
    rec = TR.load(spec["task"]["reference"]["trace_recording"])
    return lab, np.asarray(rec["pos_um"], np.float64), have


def camera_horizontal(pl, pos, Q, tilt=35.0, zoom=1.9):
    """HORIZONTAL, HEAD LEFT, from above and a little from the tail (Cedric, 2026-10-02: as the run movies). zap-inr's
    anatomy frame: head -x, dorsal -z; ZAPBench's: head +y, dorsal +z."""
    ext = float(np.ptp(np.percentile(Q, [1, 99], 0), 0).max())
    r, tl = zoom * ext, np.deg2rad(tilt)
    pl.camera.focal_point = (0.0, 0.0, 0.0)
    if np.ptp(pos[:, 0]) > np.ptp(pos[:, 1]):
        pl.camera.up = (0.0, -1.0, 0.0)
        pl.camera.position = (r * np.sin(tl), 0.0, -r * np.cos(tl))
    else:
        pl.camera.up = (1.0, 0.0, 0.0)
        pl.camera.position = (0.0, -r * np.sin(tl), r * np.cos(tl))


def camera_side(pl, pos, Q, zoom=1.9):
    """FROM THE SIDE, head left -- the view Cedric asked to have the other way up (2026-10-02): zap-inr's anatomy frame
    (head -x, left-right y) seen from -y with +z up; ZAPBench's (head +y, left-right x) from +x with -z up."""
    ext = float(np.ptp(np.percentile(Q, [1, 99], 0), 0).max())
    r = zoom * ext
    pl.camera.focal_point = (0.0, 0.0, 0.0)
    if np.ptp(pos[:, 0]) > np.ptp(pos[:, 1]):
        pl.camera.up = (0.0, 0.0, 1.0)
        pl.camera.position = (0.0, -r, 0.0)
    else:
        pl.camera.up = (0.0, 0.0, -1.0)
        pl.camera.position = (r, 0.0, 0.0)


def iso_image(Q, lab, k, size, pos, only=None, cols=None):
    """One VTK render: per cluster the surface where its local share of the neurons is >= 2x its brain-wide share
    (a gaussian-blurred density ratio on a grid, contoured: the watcher's contour look), over the faint brain."""
    import pyvista as pv
    from scipy.ndimage import gaussian_filter
    cols = cols or plt_cols(k)
    lo = Q.min(0)
    ng = 110
    dx = float(np.ptp(Q, 0).max()) * 1.02 / ng
    idx = np.clip(((Q - lo) / dx).astype(int), 0, ng - 1)

    def density(m):
        d_ = np.zeros((ng, ng, ng), np.float32)
        np.add.at(d_, (idx[m, 0], idx[m, 1], idx[m, 2]), 1.0)
        return gaussian_filter(d_, sigma=2.0)
    all_ = density(np.ones(len(Q), bool))
    inside = all_ > 0.05 * all_[all_ > 0].mean()
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("black")
    if only is None:                               # the faint brain: the overview only (Cedric: not on montage panels)
        pl.add_mesh(pv.PolyData(Q[::4]), color="gray", point_size=1.5, opacity=0.25)
    for c in range(k):
        m = lab == c
        if not m.any() or (only is not None and c != only):
            continue
        if only is not None:                       # a montage panel: the cluster's own neurons too
            pl.add_mesh(pv.PolyData(Q[m]), color=cols[c], point_size=2.5, opacity=0.5)
        enr = np.where(inside, density(m) / np.maximum(all_, 1e-9) / (m.sum() / len(Q)), 0.0)
        g = pv.ImageData(dimensions=(ng + 1,) * 3, spacing=(dx, dx, dx), origin=tuple(lo))
        g["v"] = np.pad(enr, ((0, 1), (0, 1), (0, 1))).flatten(order="F")
        surf = g.contour([2.0], scalars="v")
        if surf.n_points == 0:
            continue
        surf = surf.smooth(n_iter=30, relaxation_factor=0.15)
        pl.add_mesh(surf, color=cols[c], pbr=False, specular=0.0, diffuse=0.6, ambient=0.25, opacity=0.6,
                    smooth_shading=True)
    camera_horizontal(pl, pos, Q)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def plt_cols(k):
    import matplotlib
    if k > 20:                                     # evenly spaced hues: tab20 has only 20
        cm = matplotlib.colormaps["hsv"]
        return [matplotlib.colors.to_hex(cm(i / k)) for i in range(k)]
    cm = matplotlib.colormaps["tab20" if k > 10 else "tab10"]
    return [matplotlib.colors.to_hex(cm(i % cm.N)) for i in range(k)]


def render(name, ks=(4, 8, 16, 32), size=(1300, 800)):
    """Slide: the iso-surfaces only, a 2 x 2 of panels, one per number of clusters (Cedric, 2026-10-02)."""
    import pyvista as pv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pv.OFF_SCREEN = True
    panels, sizes_all, have = [], {}, None
    for k in ks:
        lab, pos, have = labels_of(name, k)
        Q = pos - np.percentile(pos, 50, 0)
        panels.append((k, iso_image(Q, lab, k, size, pos), np.bincount(lab, minlength=k)))
        sizes_all[k] = np.bincount(lab, minlength=k).tolist()
    fig = plt.figure(figsize=(14, 9.2), facecolor="black")
    for j, (k, img, sz) in enumerate(panels):
        x0, y0 = 0.005 + 0.5 * (j % 2), 0.52 - 0.48 * (j // 2)
        ax = fig.add_axes([x0, y0 + 0.035, 0.49, 0.40])
        ax.imshow(img)
        ax.axis("off")
        fig.text(x0 + 0.005, y0 + 0.455, f"{'abcd'[j]}   {k} clusters", color="white", fontsize=13, va="top")
    path = os.path.join(EXP, "presentation", "figs", f"clusters_3d_{name}.png")
    fig.savefig(path, dpi=130, facecolor="black")
    plt.close(fig)
    shutil.copy(path, os.path.join(EXP, "png", os.path.basename(path)))
    json.dump({"run": name, "on": have, "sizes": sizes_all},
              open(os.path.join(EXP, "data", f"clusters_{name}.json"), "w"), indent=1)
    print(f"[clusters] {path}")
    return path


def dots_image(Qc, col, size, pos, Q, view="top"):
    """One cluster's neurons as dots in its colour on black: from above (the overview's camera) or from the side."""
    import pyvista as pv
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("black")
    # small dots, so a large cluster reads as a cloud and not a solid plaque (Cedric, 2026-10-02); a small cluster's
    # few dots a little larger to stay visible
    ps = 1.0 if len(Qc) > 20000 else 1.4 if len(Qc) > 5000 else 2.2
    pl.add_mesh(pv.PolyData(Qc), color=col, point_size=ps, opacity=0.7)
    (camera_horizontal if view == "top" else camera_side)(pl, pos, Q)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


MONTAGE_COLUMNS = ((4, 4), (8, 4), (16, 6), (32, 8))   # columns per k: each montage about the slide's shape


def montage(name, k, ncols, size=(700, 430)):
    """One panel per cluster of a k-clustering (Cedric, 2026-10-02: 4, 8, 16 and 32): the cluster's neurons as WHITE
    dots (one colour for every cluster: colours made the panels ambiguous), from above and, below, from the side;
    horizontal, head left; the panels SORTED by the clusters' sizes, largest first. Each view cropped to the brain and
    the grid's columns chosen so the montage has the slide's shape (it fills the slide, not a strip of it)."""
    import pyvista as pv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pv.OFF_SCREEN = True
    lab, pos, have = labels_of(name, k)
    Q = pos - np.percentile(pos, 50, 0)
    sz = np.bincount(lab, minlength=k)
    order = np.argsort(-sz)
    panels = []
    for c in order:
        top = dots_image(Q[lab == c], "white", size, pos, Q, "top")
        side = dots_image(Q[lab == c], "white", size, pos, Q, "side")
        H, Wd = top.shape[:2]
        top = top[int(0.13 * H):int(0.87 * H), int(0.07 * Wd):int(0.93 * Wd)]
        side = side[int(0.26 * H):int(0.84 * H), int(0.07 * Wd):int(0.93 * Wd)]
        panels.append(np.concatenate([top, side], 0))
    ph, pw = panels[0].shape[:2]
    nrows = int(np.ceil(k / ncols))
    lab_h = 0.09                                  # the label strip above each panel, a fraction of its height
    fig_w = 16.0
    cell_w = fig_w / ncols
    cell_h = cell_w * ph / pw * (1 + lab_h)
    fig = plt.figure(figsize=(fig_w, cell_h * nrows), facecolor="black")
    for rank, (c, img) in enumerate(zip(order, panels)):
        r_, c_ = divmod(rank, ncols)
        x0, y0 = c_ / ncols, 1 - (r_ + 1) / nrows
        ax = fig.add_axes([x0 + 0.002, y0, 1 / ncols - 0.004, (1 / nrows) / (1 + lab_h)])
        ax.imshow(img)
        ax.axis("off")
        fig.text(x0 + 0.004, 1 - r_ / nrows - 0.004, f"{rank + 1}: {sz[c]:,} neurons", color="white",
                 fontsize=9 if k > 8 else 12, va="top")
    path = os.path.join(EXP, "presentation", "figs", f"clusters_k{k}_montage_{name}.png")
    fig.savefig(path, dpi=110, facecolor="black")
    plt.close(fig)
    shutil.copy(path, os.path.join(EXP, "png", os.path.basename(path)))
    print(f"[clusters] {path}")
    return path


if __name__ == "__main__":
    for n_ in sys.argv[1:]:
        render(n_)
        for k_, nc_ in MONTAGE_COLUMNS:
            montage(n_, k_, nc_)
