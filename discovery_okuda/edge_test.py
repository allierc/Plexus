#!/usr/bin/env python
"""Is the dark smear on the petals a tissue of slivers, or a stroke drawn wider than a cell?

Cedric, 12 August, zooming into `b_star/3d.png`: *"we could consider that the sliver cells is more
of a render issue... can we render them differently so that the edges in black do not create this
antialiasing defect... pronounced in the left petal, a bit on the right petal too."*

THE ARITHMETIC SAYS RENDER. `_draw` strokes every cell with a 0.25 pt opaque black edge, and a
point is a fixed length on the PAGE while a cell's projected area is not. 12,272 cells over a
7-inch figure puts a typical cell at a few pixels across; where a petal turns edge-on the
projection collapses further and the stroke does not, so past the point where the stroke is as wide
as the cell the mesh stops reading as outlined cells and becomes a dark field. That looks exactly
like a patch of slivers. Measured on this run, the genuinely elongated cells are 24 of 12,272 --
aspect above 3x the median of 1.351 -- so whatever is darkening a whole petal is not them.

THE FIFTH PANEL IS THE TEST, not the four rendering styles. `cell_shape_probe`'s aspect is computed
for every cell and the ones past the threshold are painted a single flat colour: if the dark
regions and the marked cells coincide, the smear is tissue and the render is honest; if the marked
cells are two dozen scattered specks while the dark regions are broad and smooth, the smear is the
stroke. One picture settles which, and it uses the same measurement the death experiment uses.

LEFT AND RIGHT ARE SHOWN AT THE SAME ZOOM, because that is the comparison Cedric drew: the effect
is pronounced on the left petal and mild on the right, and a difference between two petals of ONE
specimen cannot be a property of the tissue's parameters -- both petals have the same chemistry,
the same mechanics and the same history. It can only be viewing geometry, which is the hypothesis.

    python edge_test.py                  b_star, all four styles + the aspect map
    python edge_test.py --run r016_01
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(ROOT, "log", "okuda")
for _p in (HERE, os.path.join(ROOT, "prototype", "Tyssue"), os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from run_one import _scalebar, run_box                                # noqa: E402
from run_tyssue_vesicle import _draw                                  # noqa: E402
from tyssue_topology_ops3d import rings_from_flat_3d                  # noqa: E402

CAM = dict(elev=18, azim=30)

# --choose: THE SHEET TO PICK FROM, wider than the four the diagnosis needed and written into the
# run's own folder rather than _gates, because it is about this specimen's picture. Two rows,
# because the right answer is not the same at both scales and that is the whole difficulty: a
# montage tile is ~190 px wide and a 3d.png is 770, so a stroke that is invisible in one swallows
# the cell in the other. Choose by looking at BOTH rows of a column, not at one panel.
CHOICES = [
    ("black 0.25 pt   was the default until 12 Aug", dict(edge="black", edge_lw=0.25)),
    ("black 0.15 pt", dict(edge="black", edge_lw=0.15)),
    ("black 0.08 pt   CHOSEN -- run_tyssue_vesicle.EDGE_LW", dict(edge="black", edge_lw=0.08)),
    ("shaded x0.45, 0.25 pt", dict(edge="shaded", edge_lw=0.25, edge_shade=0.45)),
    ("shaded x0.45, 0.10 pt", dict(edge="shaded", edge_lw=0.10, edge_shade=0.45)),
    ("shaded x0.70, 0.25 pt", dict(edge="shaded", edge_lw=0.25, edge_shade=0.70)),
    ("no stroke", dict(edge="none")),
]
# The four candidates. `black` is what every figure in this project has used; the rest differ from
# it in the stroke ALONE, so any change between panels is the stroke and nothing else.
STYLES = [("black 0.25 pt  (the old default)", dict(edge="black", edge_lw=0.25)),
          ("black 0.08 pt  (chosen)", dict(edge="black", edge_lw=0.08)),
          ("shaded  (edge = face x 0.45)", dict(edge="shaded", edge_lw=0.25)),
          ("no stroke", dict(edge="none"))]
ASPECT_MULT = 3.0                    # the same threshold apoptosis_3d field_high uses


def last_frame(run):
    z = np.load(os.path.join(LOG, run, "traj.npz"), allow_pickle=True)
    t = sum(1 for k in z.files if k.startswith("pos_")) - 1
    mt = z[f"mesh_{t}"]
    mt = mt.item() if hasattr(mt, "item") else mt
    act = z[f"act_{t}"] if f"act_{t}" in z.files else None
    return np.asarray(z[f"pos_{t}"], float), mt, act


def aspect_of(pos, mt):
    """Ring-covariance long/short per cell -- `cell_shape_probe:aspect`, recomputed here."""
    nF = int(mt["nF"])
    es, et, ef = (np.asarray(mt[k]) for k in ("E_srce", "E_trgt", "E_face"))
    live = ef < nF
    out = np.full(nF, np.nan)
    for f, r in enumerate(rings_from_flat_3d(es[live], et[live], ef[live], nF)):
        if r is None or len(r) < 3:
            continue
        p = pos[np.asarray(r, int)]
        w = np.linalg.eigvalsh(np.cov((p - p.mean(0)).T) + 1e-15 * np.eye(3))[::-1]
        s0, s1 = np.sqrt(max(w[0], 0.0)), np.sqrt(max(w[1], 0.0))
        if s1 > 1e-9:
            out[f] = s0 / s1
    return out


def colour(act, nF):
    if act is None:
        return None
    a = np.asarray(act, float)[:nF]
    ok = np.isfinite(a)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    c = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    c[~ok] = np.nan
    return c


def choose(run, zoom=0.42):
    """`<run>/viz_options.png` -- every candidate stroke at both scales, plus its measured mean
    luminance, so the choice is made on a number as well as on the eye."""
    pos, mt, act = last_frame(run)
    nF = int(mt["nF"])
    L = run_box([(pos, mt, act, None)])
    col = colour(act, nF)
    rows = [("as a montage tile / 3d.png", L), (f"detail, {zoom:.2f} of the box", L * zoom)]
    fig = plt.figure(figsize=(4.3 * len(CHOICES), 4.6 * len(rows)))
    fig.patch.set_facecolor("black")
    lum = {}
    for ri, (rlab, box) in enumerate(rows):
        for ci, (slab, kw) in enumerate(CHOICES):
            ax = fig.add_subplot(len(rows), len(CHOICES), ri * len(CHOICES) + ci + 1,
                                 projection="3d")
            _draw(ax, pos, mt, 3.90, azim=CAM["azim"], act=col, Lbox=box, **kw)
            ax.view_init(elev=CAM["elev"], azim=CAM["azim"])
            if ri == 0:
                ax.text2D(0.02, 0.97, slab, transform=ax.transAxes, color="w", fontsize=12)
            if ci == 0:
                ax.text2D(0.02, 0.02, rlab, transform=ax.transAxes, color="w", fontsize=11)
            _scalebar(ax, box)
    out = os.path.join(LOG, run, "viz_options.png")
    fig.savefig(out, dpi=110, facecolor="black", bbox_inches="tight")
    plt.close(fig)

    # THE NUMBER UNDER THE PICTURE. Mean luminance of the specimen (background excluded) in the
    # TOP row -- the scale where the stroke does the damage. The tissue is white, so every point
    # below 255 in the body is stroke averaged into cells, not pigment.
    from PIL import Image
    im = np.asarray(Image.open(out).convert("L"), float)
    band = im[:int(im.shape[0] * 0.5)]
    print(f"\n{'option':26s}{'mean luminance of the body':>28s}{'vs current':>12s}")
    base = None
    for ci, (slab, _kw) in enumerate(CHOICES):
        p = band[:, int(band.shape[1] * ci / len(CHOICES)):
                    int(band.shape[1] * (ci + 1) / len(CHOICES))]
        body = p[p > 8]
        m = float(body.mean()) if body.size else float("nan")
        base = m if base is None else base
        print(f"  {slab:24s}{m:>28.1f}{'--' if ci == 0 else f'{m / base:.2f}x':>12s}")
    print(f"\n  -> {os.path.relpath(out, ROOT)}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="b_star")
    ap.add_argument("--zoom", type=float, default=0.42, help="detail box as a fraction of the full")
    ap.add_argument("--choose", action="store_true",
                    help="write <run>/viz_options.png -- the candidate strokes, to pick from")
    ap.add_argument("--scalebar", action="store_true",
                    help="draw the scale bar (off by default: a gallery clip carries no "
                         "printing, and every card on the site crops it away)")
    a = ap.parse_args()
    import run_one as _ro; _ro.SCALEBAR = a.scalebar

    if a.choose:
        return choose(a.run, a.zoom)

    pos, mt, act = last_frame(a.run)
    nF = int(mt["nF"])
    L = run_box([(pos, mt, act, None)])
    col = colour(act, nF)
    asp = aspect_of(pos, mt)
    ok = np.isfinite(asp)
    thr = ASPECT_MULT * float(np.median(asp[ok]))
    marked = ok & (asp > thr)
    print(f"{a.run}: {nF} cells, aspect median {np.median(asp[ok]):.3f}, "
          f"threshold {thr:.3f}, {int(marked.sum())} cells above it "
          f"({100 * marked.sum() / nF:.2f}%)")

    # THE ASPECT MAP. Every cell flat white, the marked ones flat red -- deliberately NOT the
    # activator LUT, which would let a red-tipped cell be mistaken for a marked one.
    amap = np.zeros(nF)
    amap[marked] = 1.0

    rows = [("full view", L), ("detail, left", L * a.zoom), ("detail, right", L * a.zoom)]
    fig = plt.figure(figsize=(4.6 * len(STYLES) + 4.6, 4.6 * len(rows)))
    fig.patch.set_facecolor("black")
    n_col = len(STYLES) + 1
    for ri, (rlab, box) in enumerate(rows):
        # The detail rows swing the camera to put one petal or the other broadside, at the same
        # zoom, so the two are compared under identical rendering.
        azim = CAM["azim"] + (0 if ri == 0 else (-55 if ri == 1 else 125))
        for ci, (slab, kw) in enumerate(STYLES + [("aspect > 3x median", None)]):
            ax = fig.add_subplot(len(rows), n_col, ri * n_col + ci + 1, projection="3d")
            if kw is None:
                _draw(ax, pos, mt, 3.90, azim=azim, act=amap, Lbox=box,
                      edge="shaded", edge_lw=0.08)
            else:
                _draw(ax, pos, mt, 3.90, azim=azim, act=col, Lbox=box, **kw)
            ax.view_init(elev=CAM["elev"], azim=azim)
            if ri == 0:
                ax.text2D(0.02, 0.97, slab, transform=ax.transAxes, color="w", fontsize=11)
            if ci == 0:
                ax.text2D(0.02, 0.03, rlab, transform=ax.transAxes, color="w", fontsize=11)
            _scalebar(ax, box)
    out = os.path.join(LOG, "_gates", f"edge_test_{a.run}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=115, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
