"""Contact sheet of the final frame of every run whose name starts with a prefix.

WHY A SHEET AND NOT A SET OF MOVIES. A parameter sweep is read by COMPARISON, and comparison across
twenty-two mp4s means twenty-two players and a memory of what the last one looked like. The final
frames on one page answer "which knob did what" in a glance, and the movie is then opened for the
one or two that are worth watching move.

    PYTHONPATH=src python tools/tissue_montage.py ab_02_flat_apicobasal --group tissue

`--group` is the data directory under `graphs_data/` (`tissue`, `gates`, ...). The prefix run itself
is included first and labelled BASELINE when it exists, because a sweep with no reference in the
same picture is a set of pictures rather than a comparison.

THE CROP IS THE POINT OF THE TUNING KNOBS. `3d.png` is framed on the run's own world box, so on a
specimen much smaller than its box most of the tile is black; `--crop` keeps the central fraction
of each frame. It is applied identically to every tile, so the tiles stay comparable in SCALE --
cropping each to its own content would make a collapsed patch and an intact one the same size on
the page, which is the one thing this sheet exists to show.
"""
import argparse
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix", help="run-name prefix, e.g. ab_02_flat_apicobasal")
    ap.add_argument("--group", default="tissue", help="directory under graphs_data/ (default: tissue)")
    ap.add_argument("--frame", default="3d.png", help="image inside each run dir (default: 3d.png)")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--crop", nargs=4, type=float, default=[0.18, 0.22, 0.82, 0.72],
                    metavar=("L", "T", "R", "B"), help="fractional crop, same for every tile")
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--out", default=None, help="default: <group>/<prefix>_montage.png")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from plexus.paths import graphs_data_path

    root = os.path.dirname(str(graphs_data_path(a.group, "_")))
    names = []
    if os.path.exists(os.path.join(root, a.prefix, a.frame)):
        names.append(a.prefix)                       # the reference, first and labelled BASELINE
    names += sorted(os.path.basename(os.path.dirname(p))
                    for p in glob.glob(os.path.join(root, a.prefix + "_*", a.frame)))
    if not names:
        raise SystemExit(f"no run under {root} matches {a.prefix!r} with a {a.frame}")

    n = len(names)
    ncol = min(a.cols, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.6, nrow * 2.6),
                             facecolor="black", squeeze=False)
    flat = axes.ravel()
    for ax, nm in zip(flat, names):
        im = Image.open(os.path.join(root, nm, a.frame))
        w, h = im.size
        l, t, r, b = a.crop
        im = im.crop((int(w * l), int(h * t), int(w * r), int(h * b)))
        ax.imshow(im)
        ax.set_facecolor("black")
        # THE LABEL IS THE DELTA, NOT THE RUN NAME. Every tile shares the prefix, so printing it
        # twenty-two times spends the space that tells them apart.
        ax.text(0.02, 0.97, nm[len(a.prefix) + 1:] or "BASELINE", transform=ax.transAxes,
                color="white", fontsize=9, va="top", ha="left")
    for ax in flat:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in flat[n:]:
        ax.set_visible(False)
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.995, 0.02, 0.02)
    out = a.out or os.path.join(root, f"{a.prefix}_montage.png")
    fig.savefig(out, dpi=a.dpi, facecolor="black")
    print(f"{n} tiles -> {out}")


if __name__ == "__main__":
    main()
