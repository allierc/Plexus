"""exp17: a neuron-graph run's STRONGEST EDGES IN 3-D (Cedric, 2026-10-02: "a slide to show in 3D the edges with the
largest weight, one panel for each level").

Local analysis. The run's learned weights W_short / W_mid / W_long (one per edge, models/best.pt) on the edge sets the
law builds from its positions (state_diffuse[neuron_graph].graph: the 6 nearest neurons; the neuron nearest each point
+-mid_um and +-long_um along x, y, z). Per set, the `--top` edges of largest |W| as lines from sender to receiver,
red for W > 0, blue for W < 0, the line width by |W|, over the brain's neurons as a faint cloud; the same oblique camera
as exp17_clusters.py. Writes presentation/figs/edges_3d_<run>.png (+ png/ copy) and data/edges_<run>.json.

    python tools/exp17_edges.py zap_dm_ng_rl1lo [--top 3000]
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
SETS = ("short", "mid", "long")


def edges_of(name):
    """The run's edge sets (senders, receivers), its learned W per set, the positions (um) and the op's reaches."""
    import plexus.operators  # noqa: F401
    from plexus import trainer as T
    from plexus.models.registry import get_contract
    spec = T.load(name)
    out = T.out_dir(spec, None)
    m = yaml.safe_load(open(spec["model"]))
    op = next(o for o in m["operators"] if o.get("op") == "state_diffuse")
    if not str(op.get("model", "")).startswith("neuron_graph"):
        raise SystemExit(f"{name}: not a neuron-graph law")
    C = get_contract("state_diffuse").implementations["neuron_graph"]
    o = C({"_at": op.get("at", "neuron"), "block": op["block"], "positions": op["positions"],
           "positions_file": op["positions_file"], "inputs": 1, "short_k": op.get("short_k", 6),
           "mid_um": op.get("mid_um", 32.0), "long_um": op.get("long_um", 128.0)})
    fit = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location="cpu")["fitted"]
    W = {s: fit[f"state_diffuse.W_{s}"].float().numpy() for s in SETS if f"state_diffuse.W_{s}" in fit}
    E = {s: (o._E[s][0].cpu().numpy(), o._E[s][1].cpu().numpy()) for s in W}
    pos = np.asarray(np.load(o._pos_file[0])[o._pos_file[1]], np.float64)
    return E, W, pos, {"short": f"{o.short_k} nearest", "mid": f"+-{o.reach['mid']:.0f} um",
                       "long": f"+-{o.reach['long']:.0f} um"}


def camera_side(pl, pos, Q, zoom=1.9):
    """From the side, dorsal up, head left. zap-inr's anatomy frame: head -x, dorsal -z, left-right y (the camera on
    +y); ZAPBench's: head +y, dorsal +z, left-right x (the camera on -x)."""
    ext = float(np.ptp(np.percentile(Q, [1, 99], 0), 0).max())
    r = zoom * ext
    pl.camera.focal_point = (0.0, 0.0, 0.0)
    if np.ptp(pos[:, 0]) > np.ptp(pos[:, 1]):
        pl.camera.up = (0.0, 0.0, -1.0)
        pl.camera.position = (0.0, r, 0.0)
    else:
        pl.camera.up = (0.0, 0.0, 1.0)
        pl.camera.position = (-r, 0.0, 0.0)


def render(name, top=3000, size=(1300, 800)):
    import pyvista as pv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pv.OFF_SCREEN = True
    E, W, pos, reach = edges_of(name)
    Q = pos - np.percentile(pos, 50, 0)
    ext = float(np.ptp(np.percentile(Q, [1, 99], 0), 0).max())
    r, tl = 2.1 * ext, np.deg2rad(55.0)
    zap_inr = np.ptp(pos[:, 0]) > np.ptp(pos[:, 1])
    imgs, info = [], {}
    for s in SETS:
        if s not in W:
            continue
        snd, rcv = E[s]
        w = W[s]
        k = np.argsort(-np.abs(w))[:top]
        wmax = float(np.abs(w[k]).max())
        views = []
        for view in ("top", "side"):                  # from above, and from the side (Cedric, 2026-10-02: d e f)
            pl = pv.Plotter(off_screen=True, window_size=size)
            pl.set_background("black")
            pl.add_mesh(pv.PolyData(Q[::3]), color="gray", point_size=1.5, opacity=0.2)
            for sign, col in ((1, "#ff5a4f"), (-1, "#4f9dff")):
                kk = k[np.sign(w[k]) == sign]
                if not len(kk):
                    continue
                pts = np.concatenate([Q[snd[kk]], Q[rcv[kk]]])
                n = len(kk)
                lines = np.column_stack([np.full(n, 2), np.arange(n), np.arange(n) + n]).ravel()
                pl.add_mesh(pv.PolyData(pts, lines=lines), color=col, line_width=2.0, opacity=0.85)
            if view == "top":
                from exp17_clusters import camera_horizontal
                camera_horizontal(pl, pos, Q)         # horizontal, head left (Cedric, 2026-10-02)
            else:
                from exp17_clusters import camera_side as cs_
                cs_(pl, pos, Q)                       # the other way up (Cedric, 2026-10-02)
            views.append(pl.screenshot(return_img=True))
            pl.close()
        imgs.append((s, views))
        ln = np.linalg.norm(pos[snd[k]] - pos[rcv[k]], axis=1)
        info[s] = {"edges": int(len(w)), "shown": int(len(k)), "w_abs_cut": float(np.abs(w[k]).min()),
                   "w_abs_max": wmax, "positive_shown": int((w[k] > 0).sum()), "negative_shown": int((w[k] < 0).sum()),
                   "mean_length_um_shown": float(ln.mean()), "median_abs_w_all": float(np.median(np.abs(w)))}
    fig = plt.figure(figsize=(16, 8.0), facecolor="black")
    n = len(imgs)
    for j, (s, views) in enumerate(imgs):
        d = info[s]
        for row, img in enumerate(views):
            y0 = 0.50 - 0.45 * row
            ax = fig.add_axes([0.003 + j / n, y0 + 0.02, 1 / n - 0.006, 0.36])
            ax.imshow(img)
            ax.axis("off")
            lab = "abcdef"[j + 3 * row]
            fig.text(0.01 + j / n, y0 + 0.43, f"{lab}   {s} edges ({reach[s]}), "
                     + ("from above" if row == 0 else "from the side"), color="white", fontsize=12, va="top")
            if row == 0:
                fig.text(0.01 + j / n, y0 + 0.395, f"the {d['shown']:,} of {d['edges']:,} with the largest |W| "
                         f"(>= {d['w_abs_cut']:.3g}); {d['positive_shown']:,} > 0, {d['negative_shown']:,} < 0",
                         color="0.75", fontsize=8.5, va="top")
    fig.text(0.01, 0.01, "red: W > 0, blue: W < 0; grey: the neurons; head left", color="0.6", fontsize=9)
    fig.text(0.99, 0.03, name, color="0.6", fontsize=9, ha="right")
    path = os.path.join(EXP, "presentation", "figs", f"edges_3d_{name}.png")
    fig.savefig(path, dpi=130, facecolor="black")
    plt.close(fig)
    shutil.copy(path, os.path.join(EXP, "png", os.path.basename(path)))
    json.dump({"run": name, "top": top, "sets": info}, open(os.path.join(EXP, "data", f"edges_{name}.json"), "w"),
              indent=1)
    print(f"[edges] {path}")
    return path


if __name__ == "__main__" and "--amplitude" not in sys.argv:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--top", type=int, default=3000)
    a = ap.parse_args()
    for n_ in a.runs:
        render(n_, a.top)


def render_amplitude(name, pooled_top=9000, size=(1300, 800)):
    """THE WEIGHTS' AMPLITUDE ON ONE SCALE (Cedric, 2026-10-02): the three edge sets cut at ONE |W| threshold (the
    `pooled_top` edges of largest |W| over all three sets together, so a set with weaker weights shows fewer edges)
    and coloured by |W| through ONE colour range for all panels; top row W > 0 (reds), bottom row W < 0 (blues);
    from above, horizontal, head left."""
    import pyvista as pv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from exp17_clusters import camera_horizontal
    pv.OFF_SCREEN = True
    from matplotlib.colors import LinearSegmentedColormap
    # on black: the weakest dark, the strongest bright (matplotlib's Reds / Blues run the other way)
    CM = {"Reds": LinearSegmentedColormap.from_list("rp", ["#4a0a0a", "#d42a1f", "#ff8a6a", "#ffe2d6"]),
          "Blues": LinearSegmentedColormap.from_list("bn", ["#0a1a4a", "#1f5fd4", "#6aa8ff", "#d6e8ff"])}
    E, W, pos, reach = edges_of(name)
    Q = pos - np.percentile(pos, 50, 0)
    allw = np.concatenate([np.abs(W[s]) for s in SETS if s in W])
    cut = float(np.sort(allw)[-pooled_top])
    vmax = float(np.percentile(allw[allw >= cut], 99))
    panels, counts = {}, {}
    for s in SETS:
        if s not in W:
            continue
        snd, rcv = E[s]
        w = W[s]
        for sign, cmap in ((1, "Reds"), (-1, "Blues")):
            kk = np.where((np.abs(w) >= cut) & (np.sign(w) == sign))[0]
            kk = kk[np.argsort(np.abs(w[kk]))]                       # the strongest drawn last, on top
            counts[(s, sign)] = len(kk)
            for view in ("top", "side"):          # from above and from the side (Cedric, 2026-10-02)
                pl = pv.Plotter(off_screen=True, window_size=size)
                pl.set_background("black")
                pl.add_mesh(pv.PolyData(Q[::3]), color="gray", point_size=1.2, opacity=0.15)
                if len(kk):
                    n = len(kk)
                    pts = np.concatenate([Q[snd[kk]], Q[rcv[kk]]])
                    lines = np.column_stack([np.full(n, 2), np.arange(n), np.arange(n) + n]).ravel()
                    poly = pv.PolyData(pts, lines=lines)
                    poly.cell_data["a"] = np.abs(w[kk])
                    pl.add_mesh(poly, scalars="a", cmap=CM[cmap], clim=(cut, vmax), line_width=2.0, opacity=0.9,
                                show_scalar_bar=False)
                if view == "top":
                    camera_horizontal(pl, pos, Q)
                else:
                    from exp17_clusters import camera_side
                    camera_side(pl, pos, Q)
                img = pl.screenshot(return_img=True)
                if view == "side":                  # the side view's empty top and bottom cropped, not shrunk
                    hh = img.shape[0]
                    img = img[int(0.18 * hh):int(0.88 * hh)]
                panels[(s, sign, view)] = img
                pl.close()
    # four rows: W > 0 from above, from the side; W < 0 from above, from the side
    fig = plt.figure(figsize=(16, 12.5), facecolor="black")
    rows_ = [(1, "top", 0.245), (1, "side", 0.165), (-1, "top", 0.245), (-1, "side", 0.165)]   # + 4 gaps: ~0.95
    y = 0.99
    lab = iter("abcdefghijkl")
    ybar = {}
    for sign, view, h in rows_:
        y -= h + 0.03
        for j, s in enumerate([s for s in SETS if s in W]):
            ax = fig.add_axes([0.003 + j / 3.15, y, 1 / 3.15 - 0.006, h])
            ax.imshow(panels[(s, sign, view)])
            ax.axis("off")
            fig.text(0.01 + j / 3.15, y + h + 0.004, f"{next(lab)}   {s} edges ({reach[s]}), W {'>' if sign > 0 else '<'} 0"
                     + (f": {counts[(s, sign)]:,}, from above" if view == "top" else ", from the side"),
                     color="white", fontsize=11, va="bottom")
        ybar.setdefault(sign, []).append((y, h))
    for sign, cmap in ((1, "Reds"), (-1, "Blues")):
        (y0, h0), (y1, h1) = ybar[sign]
        cax = fig.add_axes([0.955, y1 + 0.02, 0.01, (y0 + h0) - y1 - 0.04])
        sm = plt.cm.ScalarMappable(cmap=CM[cmap], norm=plt.Normalize(cut, vmax))
        cb = fig.colorbar(sm, cax=cax)
        cb.ax.tick_params(colors="0.75", labelsize=8)
        cb.set_label("|W|", color="0.75", fontsize=9)
    fig.text(0.01, 0.01, f"one cut for the three sets: |W| >= {cut:.3g} (the {pooled_top:,} largest |W| of all "
             f"{len(allw):,} edges); one colour range {cut:.3g} .. {vmax:.3g}; grey: the neurons; head left",
             color="0.6", fontsize=9)
    path = os.path.join(EXP, "presentation", "figs", f"edges_amp_{name}.png")
    fig.savefig(path, dpi=130, facecolor="black")
    plt.close(fig)
    shutil.copy(path, os.path.join(EXP, "png", os.path.basename(path)))
    print(f"[edges] {path}: cut {cut:.3g}; counts " + ", ".join(f"{s}{'+' if g > 0 else '-'} {c}" for (s, g), c in counts.items()))
    import json
    json.dump({"run": name, "cut": cut, "vmax": vmax, "pooled_top": pooled_top, "n_edges": int(len(allw)),
               "counts": {f"{s_}{'+' if g > 0 else '-'}": int(c) for (s_, g), c in counts.items()}},
              open(path.replace(".png", ".json"), "w"), indent=1)          # the slide's table reads these
    return path


if __name__ == "__main__" and "--amplitude" in sys.argv:
    for n_ in [a for a in sys.argv[1:] if not a.startswith("--")]:
        render_amplitude(n_)
