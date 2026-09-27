#!/usr/bin/env python
"""exp06 talk movie: the REAL Utrecht sheet, each segmented cell painted by its own state, frame by frame.

    PYTHONPATH=src:tools python tools/exp06_movie.py tissue/exp06_real_healthy_s1 [--rows 0:120] [--seconds 10]
    PYTHONPATH=src:tools python tools/exp06_movie.py tissue/exp06_follow_cut_s1 --panels u,gam

The engine's `mpl2d` render of these runs draws the seed's material points coloured by cell identity,
which shows neither the wave nor the contraction. This paints the segmentation itself: cell j is label
j+1 of the run's own `label_image` field (the convention `seed_from_segmentation` seeds by), coloured by
`chem[:, 0]` (u, the excitation, 0 at rest, ~1 on the plateau) and, with `--panels u,gam`, by
`gam_prev` (the active strain's activation, what the contraction reads). Time on the title is the run's
own: model time x `units.time_s`. Writes `experiments/exp06_excitation_wave/movies/<run>.mp4`.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]


def label_image(spec):
    import tifffile
    from plexus.paths import graphs_data_path
    src = next(f["source"] for f in (spec.get("fields") or {}).values() if f.get("frame") == "label_image")
    img = tifffile.imread(src if os.path.isabs(src) else graphs_data_path(src))
    return img[..., 0] if img.ndim == 3 else img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--rows", default=None, help="a:b slice of recorded rows")
    ap.add_argument("--stride", type=int, default=1, help="every k-th row")
    ap.add_argument("--into-run", action="store_true",
                    help="write the run's movie.mp4 (the engine's particle render kept as movie_particles.mp4)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--panels", default="u")
    ap.add_argument("--down", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import imageio.v3 as iio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from exp_measures.common import open_run
    from exp_measures import exp06 as E
    T = E._on_set(open_run(a.run), "cell")
    lab = label_image(T.spec)[:: a.down, :: a.down]
    n = T.n_rows()
    r0, r1 = (int(x) for x in a.rows.split(":")) if a.rows else (0, n)
    r0, r1 = max(0, r0), min(r1, n)
    rows = list(range(r0, r1, max(1, a.stride)))
    tt = E._row_times(T)
    ts = (T.spec.get("general") or {}).get("units", {}).get("time_s", 1.0)
    panels = a.panels.split(",")
    cmaps = {"u": "inferno", "gam": "viridis"}
    titles = {"u": "excitation u (gap-junction wave)", "gam": "contraction activation"}
    vmax = {"u": 1.0, "gam": None}
    series = {}
    for p in panels:
        blk = "chem" if p == "u" else "gam_prev"
        series[p] = np.stack([np.asarray(T.state(blk, t), float)[:, 0] for t in rows])
    vmax["gam"] = float(np.nanmax(series["gam"])) if "gam" in series else None
    # junction outlines in grey: a sheet at rest (u = 0) is black in inferno, i.e. the background colour,
    # and without its outlines reads as an empty panel
    edge = np.zeros(lab.shape, bool)
    edge[1:, :] |= lab[1:, :] != lab[:-1, :]
    edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    edge = np.ma.masked_where(~edge, np.ones(lab.shape))
    frames = []
    for i, t in enumerate(rows):
        fig, axs = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5.3), facecolor="black")
        axs = np.atleast_1d(axs)
        for ax, p in zip(axs, panels):
            val = np.concatenate([[np.nan], series[p][i]])
            img = val[np.clip(lab, 0, len(val) - 1)]
            img[lab == 0] = np.nan
            ax.imshow(img, cmap=cmaps[p], vmin=0.0, vmax=vmax[p] or 1.0, interpolation="nearest")
            ax.imshow(edge, cmap="gray", vmin=0.0, vmax=2.0, interpolation="nearest")
            ax.set_axis_off()
            ax.set_title(titles[p], color="white", fontsize=12, loc="left")
        fig.suptitle(f"{os.path.basename(a.run)}   t = {tt[t] * ts * 1e3:.2f} ms", color="white", fontsize=11, x=0.02, ha="left")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    out = a.out or os.path.join(ROOT, "experiments", "exp06_excitation_wave", "movies",
                                os.path.basename(a.run) + ".mp4")
    if a.into_run:
        out = os.path.join(T.dir, "movie.mp4")
        eng = os.path.join(T.dir, "movie_particles.mp4")
        if os.path.exists(out) and not os.path.exists(eng):
            os.replace(out, eng)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fps = max(1, round(len(frames) / a.seconds))
    iio.imwrite(out, np.stack(frames), fps=fps, codec="libx264")
    print(f"[movie] {len(frames)} frames, {fps} fps -> {out}")


if __name__ == "__main__":
    main()
