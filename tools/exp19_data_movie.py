#!/usr/bin/env python
"""exp19: a movie of the streamed f338 recording itself, before any law -- is the stored grid fine enough?

    PYTHONPATH=src python tools/exp19_data_movie.py [--fps 30] [--stride 1]

Reads experiments/exp19_wholistic_graphcast/data/f338_b<B>.npy (uint16 [T, Z, C, Y, X], B x B camera pixels per
voxel) and writes graphs_data/zebrafish/wholistic_f338_data/{movie.mp4, 3d.png}, which the record links as the
md's results row D0. Six panels, each a maximum projection over the 13 planes:

    top row     brightness, log scale, each panel's own 1st-99.9th percentile -- the anatomy
    bottom row  dF/F of that projection, F0 = each pixel's median over the first 300 volumes, -0.5 .. +0.5
    columns     GCaMP7f at the stored grid (B x 0.325 um), GCaMP7f at the law's grid (4 x that, shown with
                nearest-pixel blocks), mCherry-CAAX at the stored grid (calcium-blind: what moves here is motion)

One frame per volume (3.81 s of recording), so at 30 fps the 1.69 h play in 53 s.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from plexus.paths import graphs_data_path                              # noqa: E402

SRC = os.path.join(ROOT, "experiments", "exp19_wholistic_graphcast", "data")


def mips(npy, chunk=40):
    a = np.load(npy, mmap_mode="r")
    T = a.shape[0]
    out = np.empty((T, a.shape[2], a.shape[3], a.shape[4]), np.float32)        # [T, C, Y, X]
    for t0 in range(0, T, chunk):
        out[t0:t0 + chunk] = np.asarray(a[t0:t0 + chunk]).max(axis=1)
        print(f"  projected {min(t0 + chunk, T)} / {T}", end="\r", flush=True)
    print()
    return out


def coarse(x, f):
    *lead, Y, X = x.shape
    c = x[..., :Y // f * f, :X // f * f].reshape(*lead, Y // f, f, X // f, f).mean(axis=(-3, -1))
    return np.repeat(np.repeat(c, f, -2), f, -1)                              # back to the panel's size, as blocks


def to_rgb(img, lo, hi, cmap):
    v = np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1)
    return (cmap(v)[..., :3] * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1, help="every n-th volume")
    ap.add_argument("--law", type=int, default=4, help="stored voxels per law voxel in x and y")
    a = ap.parse_args()
    import imageio.v2 as imageio
    import matplotlib
    from PIL import Image, ImageDraw, ImageFont

    npy = [p for p in glob.glob(os.path.join(SRC, "f338_b*.npy")) if not p.endswith(".done.npy")][0]
    meta = json.load(open(npy[:-4] + ".json"))
    um = meta["binned_voxel_um"]["x"]
    times = np.asarray(meta["volume_times_s"]) - meta["volume_times_s"][0]
    M = mips(npy)                                                        # [T, C, Y, X]
    Y, X = M.shape[-2:]
    Yc, Xc = Y // a.law * a.law, X // a.law * a.law
    M = M[..., :Yc, :Xc]
    panels = {"g": M[:, 0], "gl": coarse(M[:, 0], a.law), "m": M[:, 1]}
    titles = {"g": f"GCaMP7f, {um:.1f} um (stored)", "gl": f"GCaMP7f, {um * a.law:.1f} um (law grid)",
              "m": f"mCherry-CAAX, {um:.1f} um (no calcium)"}
    gray, div = matplotlib.colormaps["gray"], matplotlib.colormaps["RdBu_r"]
    lims, F0 = {}, {}
    for k, v in panels.items():
        lv = np.log(np.maximum(v[::20], 1))
        lims[k] = (np.percentile(lv, 1), np.percentile(lv, 99.9))
        F0[k] = np.median(v[:300], axis=0)
    out_dir = graphs_data_path("zebrafish", "wholistic_f338_data")
    os.makedirs(out_dir, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    w = imageio.get_writer(os.path.join(out_dir, "movie.mp4"), fps=a.fps, codec="libx264", quality=5,
                           macro_block_size=8)
    T = M.shape[0]
    still = None
    for t in range(0, T, a.stride):
        rows = []
        for top in (True, False):
            cols = []
            for k, v in panels.items():
                if top:
                    cols.append(to_rgb(np.log(np.maximum(v[t], 1)), *lims[k], gray))
                else:
                    dff = v[t] / np.maximum(F0[k], 1) - 1
                    cols.append(to_rgb(dff, -0.5, 0.5, div))
            rows.append(np.concatenate(cols, 1))
        frame = np.concatenate(rows, 0)
        im = Image.fromarray(frame)
        d = ImageDraw.Draw(im)
        for i, k in enumerate(panels):
            d.text((i * Xc + 8, 6), titles[k], fill=(255, 220, 0), font=font)
        d.text((8, Yc + 6), "dF/F of the projection, -0.5 (blue) .. +0.5 (red)", fill=(0, 0, 0), font=font)
        d.text((3 * Xc - 250, Yc - 28), f"t {times[t] / 60:6.1f} min  (vol {t + 1})", fill=(255, 220, 0), font=font)
        frame = np.asarray(im)
        frame = frame[:frame.shape[0] // 8 * 8, :frame.shape[1] // 8 * 8]
        w.append_data(frame)
        if t == T // 2 // a.stride * a.stride:
            still = frame
        if t % 200 == 0:
            print(f"  frame {t} / {T}", flush=True)
    w.close()
    Image.fromarray(still).save(os.path.join(out_dir, "3d.png"))
    print(f"-> {out_dir}/movie.mp4 ({T // a.stride} frames at {a.fps} fps), 3d.png")


if __name__ == "__main__":
    main()
