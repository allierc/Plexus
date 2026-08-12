#!/usr/bin/env python
"""`kburns.mp4` -- the finished shape, turned on the spot and zoomed into.

Cedric, 12 August: *"make a kburns_render.py to generate an mp4 starting at 3d.png (last frame) and
make a rotation of the object with a zoom in for a new mp4 to put in the same directory."*

WHAT THIS IS FOR, AND WHAT `movie.mp4` ALREADY DOES. `movie.mp4` moves through TIME with the camera
nailed down -- deliberately, because a per-frame autofit rescaled the box 4.6x over a run and hid
the growth it was supposed to show. That fixed camera is the right choice for the run and the wrong
one for the specimen: it is sized for the LAST frame, so the finished object sits small and is seen
from one angle. This moves the CAMERA with time held still. Nine arms read as nine arms when the
body turns; from `3d.png`'s single azimuth, arms pointing at the lens read as blobs.

IT STARTS ON THE 3d.png FRAME AND ITS FIRST FRAME IS THAT IMAGE. Same last frame, same activator
LUT, same `_draw` -- the reference renderer, called directly rather than approximated -- so the
movie opens on the picture you already know and moves from there. Nothing here re-derives geometry
or colour; a second renderer that drifts from the first is worse than no second renderer.

THE TWO MOVES:
    rotation  a full 360 degrees of azimuth, so the sweep closes on its own first frame and the
              clip loops seamlessly. Elevation eases from 18 to 30 and back, which is enough to
              show that an arm has a length out of the original view plane without the tumbling
              that makes a shape hard to read.
    zoom      the camera box shrinks from the run's own fixed Lbox to `--zoom` of it, on a smooth
              ease so there is no jerk at either end. The scale bar is redrawn at every frame from
              the CURRENT box, so the zoom cannot make the object look bigger than it is -- the
              number under the bar is what changes.

THE BOX IS THE RUN'S OWN, taken from `diag.json:camera_lbox` when it is there and recomputed
identically when it is not. Sharing that number is what keeps this comparable to `3d.png` and to
every other run in the batch: two specimens at the same zoom are the same size on screen.

    python kburns_render.py 'b_star*'              every star base
    python kburns_render.py --zoom 0.45 'b_star'   tighter final framing
    python kburns_render.py --seconds 8 'r013_05'
"""
import argparse
import json
import os
import sys
from fnmatch import fnmatch

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(ROOT, "log", "okuda")
for _p in (HERE, os.path.join(ROOT, "prototype", "Tyssue"), os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.animation import FFMpegWriter                        # noqa: E402

from run_one import _scalebar, run_box                               # noqa: E402
from run_tyssue_vesicle import _draw                                 # noqa: E402

OUT = "kburns.mp4"

# THE LENGTH IS THE ROTATION SPEED, because the turn is exactly one revolution however long the
# clip runs. Cedric, 12 August: *"make the rotation slower, and make it x3 longer (more frames)."*
# Those are one change: 360 degrees over 18 seconds instead of 6 is three times the frames at a
# third of the angular speed, 20 deg/s rather than 60. Spreading the same revolution over more
# frames is also the only way to slow it that keeps the clip looping -- stopping short of 360 would
# leave a jump cut back to the first azimuth.
SECONDS = 18.0


def _ease(u):
    """Smoothstep. A linear zoom reads as a jerk at both ends because the eye tracks acceleration."""
    return u * u * (3.0 - 2.0 * u)


def last_frame(run_dir):
    """The final recorded (pos, mesh, act), from the same archive `movie.mp4` is rebuilt from."""
    p = os.path.join(run_dir, "traj.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    n = sum(1 for k in z.files if k.startswith("pos_"))
    if not n:
        return None
    t = n - 1
    mt = z[f"mesh_{t}"]
    mt = mt.item() if hasattr(mt, "item") else mt
    act = z[f"act_{t}"] if f"act_{t}" in z.files else None
    return np.asarray(z[f"pos_{t}"], float), mt, act


def colour(act, mt):
    """The activator LUT `render` uses, so this clip and `3d.png` colour the same cell the same.

    The range is the FINAL frame's own, which is what `3d.png` shows. `render` takes its lo/hi over
    the whole run; on the last frame of a saturating run the two agree, and where they differ this
    one is the honest choice for a single-frame portrait -- the alternative washes out a specimen
    whose activator peaked earlier and has since decayed.
    """
    if act is None:
        return None
    a = np.asarray(act, float)[:int(mt["nF"])]
    ok = np.isfinite(a)
    if ok.sum() < 2:
        return None
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    out = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    out[~ok] = np.nan            # a cell that left the model is grey, not white
    return out


def kburns(name, run_dir, seconds=SECONDS, fps=25, zoom=0.55, dpi=110,
           edge="black", edge_lw=0.25, edge_shade=0.45, out=OUT):
    fr = last_frame(run_dir)
    if fr is None:
        return "no traj.npz"
    pos, mt, act = fr
    # THE RUN'S OWN BOX. `run_box` over a single frame is what `render` computes over all of them
    # for a specimen whose last frame is its largest, which every growing run's is.
    L0 = None
    dj = os.path.join(run_dir, "diag.json")
    if os.path.exists(dj):
        try:
            L0 = (json.load(open(dj)).get("summary") or {}).get("camera_lbox")
        except Exception:
            L0 = None
    if not L0:
        L0 = run_box([(pos, mt, act, None)])
    L0 = float(L0)

    col = colour(act, mt)
    dying = None if mt.get("apop") is None else (np.asarray(mt["apop"])[:mt["nF"]] > 0)
    n = max(2, int(round(seconds * fps)))

    fig = plt.figure(figsize=(7.0, 7.0)); fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d")
    wri = FFMpegWriter(fps=fps, codec="libx264",
                       extra_args=["-pix_fmt", "yuv420p", "-crf", "20"])
    with wri.saving(fig, os.path.join(run_dir, out), dpi=dpi):
        for i in range(n):
            u = i / (n - 1)
            # A FULL TURN, so frame n-1 lands back on frame 0's azimuth and the clip loops.
            azim = 30.0 + 360.0 * u
            elev = 18.0 + 12.0 * np.sin(np.pi * u)         # up and back down, ending where it began
            L = L0 * (1.0 - (1.0 - zoom) * _ease(u))
            ax.clear()
            _draw(ax, pos, mt, 3.90, azim=azim, act=col, Lbox=L, dying=dying,
                  edge=edge, edge_lw=edge_lw, edge_shade=edge_shade)
            ax.view_init(elev=elev, azim=azim)             # _draw hardwires elev=18 as its last act
            ax.text2D(0.02, 0.96, name, transform=ax.transAxes, color="w", fontsize=10)
            # REDRAWN FROM THE CURRENT BOX, every frame. A zoom with a frozen scale bar is a lie
            # about size, and size is the one thing separating a 2,000-cell specimen from a
            # 50,000-cell one in these pictures.
            _scalebar(ax, L)
            wri.grab_frame()
    plt.close(fig)
    return f"{n} frames, Lbox {L0:.2f} -> {L0 * zoom:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="*", default=["b_star*"])
    ap.add_argument("--seconds", type=float, default=SECONDS)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--zoom", type=float, default=0.55, help="final box as a fraction of the run's")
    ap.add_argument("--force", action="store_true")
    # THE STROKE, AS AN ARGUMENT. Measured on b_star's option sheet: the default black 0.25 pt
    # leaves the body at mean luminance 100 of 255 on a WHITE tissue, because a fixed-width stroke
    # around a few-pixel cell is most of the cell. See discovery_okuda/edge_test.py --choose.
    ap.add_argument("--edge", default="black", choices=["black", "shaded", "none"])
    ap.add_argument("--edge-lw", type=float, default=0.25)
    ap.add_argument("--edge-shade", type=float, default=0.45)
    ap.add_argument("--out", default=OUT, help="filename, so two stroke settings can coexist")
    a = ap.parse_args()

    runs = sorted(d for d in os.listdir(LOG)
                  if os.path.isdir(os.path.join(LOG, d))
                  and any(fnmatch(d, p) for p in a.patterns))
    if not runs:
        print(f"no run matches {a.patterns}")
        return 1
    done = skip = fail = 0
    for r in runs:
        d = os.path.join(LOG, r)
        if not a.force and os.path.exists(os.path.join(d, a.out)):
            print(f"  {r:22s} has {a.out} already -- --force to redo"); skip += 1; continue
        try:
            msg = kburns(r, d, seconds=a.seconds, fps=a.fps, zoom=a.zoom, out=a.out,
                         edge=a.edge, edge_lw=a.edge_lw, edge_shade=a.edge_shade)
        except Exception as e:
            msg = f"FAILED: {type(e).__name__}: {e}"
        if msg == "no traj.npz":
            print(f"  {r:22s} NO TRAJECTORY -- nothing to turn"); fail += 1
        elif msg.startswith("FAILED"):
            print(f"  {r:22s} {msg}"); fail += 1
        else:
            print(f"  {r:22s} {msg} -> {a.out}"); done += 1
    print(f"\n{done} written, {skip} already present, {fail} failed")
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
