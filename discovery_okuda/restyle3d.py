#!/usr/bin/env python
"""Re-draw a finished run's end frame with a different cell stroke, as `3d_<tag>.png`.

The stroke turned out to be worth about half the tissue's brightness -- `edge_test.py --choose`
measures the body at 100 of 255 under the default 0.25 pt black and 148 under 0.08 pt, on a tissue
that is white. Choosing between candidates means seeing them across a whole campaign and not on one
specimen, and that means re-rendering the end frame of every run, which `traj.npz` makes cheap: one
frame, no simulation.

IT WRITES A NEW FILE RATHER THAN OVERWRITING `3d.png`. Two reasons. The archive's pictures are what
every past montage, note and figure was read from, and silently restyling them would change what an
old document shows. And a comparison needs both present at once -- `montage.py --image` then tiles
whichever one is being judged.

    python restyle3d.py --tag c3 --edge black  --edge-lw 0.08          'r01[3-9]_*' 'r02*'
    python restyle3d.py --tag c5 --edge shaded --edge-lw 0.10          'b_star*'
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
for _p in (HERE, os.path.join(ROOT, "discovery_okuda", "ops"), os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from run_one import _scalebar, run_box                                # noqa: E402
from run_tyssue_vesicle import _draw, EDGE_LW                         # noqa: E402

CAM = dict(elev=18, azim=30)


def end_frame(run_dir):
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
    return np.asarray(z[f"pos_{t}"], float), mt, act, t


def draw_one(name, run_dir, out_name, **edge):
    fr = end_frame(run_dir)
    if fr is None:
        return None
    pos, mt, act, t = fr
    # THE RUN'S OWN CAMERA, so a restyled tile is the same size as the original next to it in a
    # montage. Anything else would make a stroke comparison a zoom comparison.
    L = None
    dj = os.path.join(run_dir, "diag.json")
    if os.path.exists(dj):
        try:
            L = (json.load(open(dj)).get("summary") or {}).get("camera_lbox")
        except Exception:
            L = None
    L = float(L) if L else run_box([(pos, mt, act, None)])
    nF = int(mt["nF"])
    col = None
    if act is not None:
        a = np.asarray(act, float)[:nF]
        ok = np.isfinite(a)
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        col = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
        col[~ok] = np.nan
    dying = None if mt.get("apop") is None else (np.asarray(mt["apop"])[:nF] > 0)

    fig = plt.figure(figsize=(7.0, 7.0)); fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d")
    _draw(ax, pos, mt, 3.90, azim=CAM["azim"], act=col, Lbox=L, dying=dying, **edge)
    ax.view_init(**CAM)
    ax.text2D(0.02, 0.96, f"{name}  frame {t}", transform=ax.transAxes, color="w", fontsize=10)
    _scalebar(ax, L)
    fig.savefig(os.path.join(run_dir, out_name), dpi=110, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    return nF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="*", default=["b_star*"])
    ap.add_argument("--tag", required=True, help="written as 3d_<tag>.png")
    ap.add_argument("--edge", default="black", choices=["black", "shaded", "none"])
    ap.add_argument("--edge-lw", type=float, default=EDGE_LW)
    ap.add_argument("--edge-shade", type=float, default=0.45)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--scalebar", action="store_true",
                    help="draw the scale bar (off by default: a gallery clip carries no "
                         "printing, and every card on the site crops it away)")
    a = ap.parse_args()
    import run_one as _ro; _ro.SCALEBAR = a.scalebar

    out_name = f"3d_{a.tag}.png"
    runs = sorted(d for d in os.listdir(LOG)
                  if os.path.isdir(os.path.join(LOG, d))
                  and any(fnmatch(d, p) for p in a.patterns))
    done = skip = miss = fail = 0
    for r in runs:
        d = os.path.join(LOG, r)
        if not a.force and os.path.exists(os.path.join(d, out_name)):
            skip += 1
            continue
        try:
            nF = draw_one(r, d, out_name, edge=a.edge, edge_lw=a.edge_lw, edge_shade=a.edge_shade)
        except Exception as e:
            print(f"  {r:16s} FAILED {type(e).__name__}: {str(e)[:60]}"); fail += 1; continue
        if nF is None:
            miss += 1                      # no traj.npz: counted, and named in the tally below
        else:
            done += 1
    print(f"{out_name}: {done} drawn, {skip} already there, {miss} without a trajectory, "
          f"{fail} failed  (of {len(runs)} matching runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
