#!/usr/bin/env python
"""grid_movie -- the 10x10 loop grid, drawn as it happens.

WHY A MOVIE AND NOT A STILL
================================================================================================
`grid_plot.py` draws each node's finished loop. That answers whether the SHAPE matches and says
nothing about WHEN each part of it was traced -- and timing is precisely the axis the objective was
blind to for sixty batches, the axis `coordination` was built for, and the axis on which the best
archived fit reads 0.582 where the tissue reads 0.997. A still cannot show it. A dot moving round
each loop can: if the model is coordinated with the recording, the red and green dots go round
together, and if it is not, you see them drift apart across the sheet.

Each panel is centred and scaled to its own loop, so the movie is about shape and timing, never
size. The frame colour is that node's loopscore and does not change.

    python grid_movie.py --dump _ablate/abl_none/dump.npz --out .../none.mp4 --label "as fitted"
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import metrics as M                                                 # noqa: E402
import grid_plot as GP                                              # noqa: E402


def _stitch(frame_dir, out, fps):
    # the project already knows where ffmpeg is -- next to the running interpreter in the conda
    # env, which is NOT on PATH here. shutil.which alone returns nothing and the movie silently
    # becomes a directory of pngs.
    from plexus.plot import _ffmpeg
    exe = _ffmpeg()
    if not exe:
        print("[grid_movie] no ffmpeg -- frames left in", frame_dir)
        return None
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    n = len(glob.glob(os.path.join(frame_dir, "f_*.png")))
    cmd = [exe, "-y", "-loglevel", "error", "-framerate", str(fps),
           "-i", os.path.join(frame_dir, "f_%05d.png"),
           # yuv420p subsamples chroma 2:1, so BOTH dimensions must be even -- and a 10x10 grid of
           # 1.24in panels plus a title band is 1240x1335, which is odd, so libx264 refused to open
           # the encoder at all. Pad up to the next even size rather than rescaling, which would
           # resample every panel.
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:black",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    # EXISTS IS NOT WRITTEN. ffmpeg creates the output file before it encodes anything, so a failed
    # run leaves a ZERO-BYTE mp4 that os.path.exists happily confirms -- which is how four empty
    # movies got reported as four successes. Check the size, show what ffmpeg said, and keep the
    # frames so the next attempt has something to work from.
    size = os.path.getsize(out) if os.path.exists(out) else 0
    if size < 1024:
        print(f"[grid_movie] *** ffmpeg produced {size} bytes from {n} frames *** exit={r.returncode}"
              f"\n{(r.stderr or r.stdout or '(no output)').strip()[-600:]}"
              f"\n[grid_movie] frames kept at {frame_dir}", flush=True)
        return None
    print(f"[grid_movie] {n} frames -> {size / 1e6:.1f} MB", flush=True)
    return out


def render(dump, out, label="", fps=12, loops=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    z = np.load(dump)
    sim, real = z["sim_d"].astype(np.float64), z["real_d"].astype(np.float64)
    rest = z["rest"].astype(np.float64)
    idx, rc, n = GP.grid_indices(rest)
    sc = GP.per_node_score(sim, real, idx)
    G = sim.shape[0]

    mask = np.zeros(rest.shape[0], bool); mask[idx] = True
    head = []
    for nm in ("orientation_error", "chirality_match", "coordination"):
        try:
            head.append(f"{nm} {M.REGISTRY[nm](sim, real, mask):.4f}")
        except Exception:
            pass

    cmap = mcolors.LinearSegmentedColormap.from_list("gr", ["#B3261E", "#B26B00", "#1B7F3B"])
    norm = mcolors.Normalize(vmin=-0.3, vmax=1.0)

    fig, axes = plt.subplots(n, n, figsize=(n * 1.24, n * 1.24 + 0.95), facecolor="black")
    P, Q, lines = [], [], []
    for k, (r, c) in enumerate(rc):
        ax = axes[r, c]
        p, q = real[:, idx[k]], sim[:, idx[k]]
        ctr = p.mean(0)
        p, q = p - ctr, q - ctr
        P.append(p); Q.append(q)
        lg, = ax.plot([], [], color="#22DD22", lw=1.1)
        lr, = ax.plot([], [], color="#FF3B30", lw=1.1)
        dg, = ax.plot([], [], "o", color="#22DD22", ms=3.0)
        dr, = ax.plot([], [], "o", color="#FF3B30", ms=3.0)
        lines.append((lg, lr, dg, dr))
        rad = max(np.abs(np.concatenate([p, q])).max(), 1e-12) * 1.15
        ax.set_xlim(-rad, rad); ax.set_ylim(-rad, rad)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
        for sp in ax.spines.values():
            sp.set_color(cmap(norm(sc[k]))); sp.set_linewidth(2.4)
        ax.text(0.03, 0.85, f"{sc[k]:+.2f}", color="white", fontsize=6, fontweight="bold",
                transform=ax.transAxes)

    title = fig.suptitle("", color="white", fontsize=10, y=0.995)
    fig.subplots_adjust(hspace=0.06, wspace=0.06, top=0.905, bottom=0.008, left=0.008, right=0.992)

    tmp = tempfile.mkdtemp(prefix="gridmov_")
    total = G * loops
    for f in range(total):
        t = f % G
        for k, (lg, lr, dg, dr) in enumerate(lines):
            p, q = P[k], Q[k]
            lg.set_data(p[:t + 1, 0], p[:t + 1, 1])
            lr.set_data(q[:t + 1, 0], q[:t + 1, 1])
            dg.set_data([p[t, 0]], [p[t, 1]])
            dr.set_data([q[t, 0]], [q[t, 1]])
        title.set_text(
            f"{label}    frame {t + 1}/{G} of the beat    "
            f"green = the recording, red = the model, frame colour = its loopscore\n"
            f"mean {sc.mean():+.3f}    " + "    ".join(head))
        fig.savefig(os.path.join(tmp, f"f_{f:05d}.png"), dpi=100, facecolor="black")
    plt.close(fig)

    made = _stitch(tmp, out, fps)
    if made:                       # on failure the frames stay put, as the message says they do
        for p_ in glob.glob(os.path.join(tmp, "*.png")):
            os.remove(p_)
        os.rmdir(tmp)
    return made, sc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--loops", type=int, default=3)
    a = ap.parse_args(argv)
    made, sc = render(a.dump, a.out, a.label, a.fps, a.loops)
    print(f"[grid_movie] {made}  mean {sc.mean():+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
