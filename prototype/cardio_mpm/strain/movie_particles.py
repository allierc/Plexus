"""movie_particles -- the material points themselves, moving, next to the tracking nodes.

Left: the recording's 18,769 tracking nodes (green). Right: the model's MPM particles (blue), the
fitted rollout on the same beat. Both are drawn as displacement from rest, amplified `--amplify`
times (default 10: the real motion is 1-3 px of 2048 and would not be visible), and the caption
says so. The outer band of particles is prescribed from the recording and drawn dimmer.

    python movie_particles.py --params out/fits/s4_live_r6_Eshrink0.3/params.npz --beat 1
"""
import argparse, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import model as M, recording as R
from movie_image import model_node_displacement
GREEN, BLUE, DIM, WHITE = "#3fbf6f", "#4a90e0", "#2a4a70", "#f2f2f2"
plt.rcParams.update({"figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
                     "text.color": WHITE, "axes.labelcolor": WHITE, "font.size": 11})

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--params", default=os.path.join(HERE, "out", "fits", "s4_live_r6_Eshrink0.3", "params.npz"))
ap.add_argument("--specimen", default="healthy", choices=["healthy", "hcm"]); ap.add_argument("--beat", type=int, default=1)
ap.add_argument("--device", default="cuda:0"); ap.add_argument("--per-parent", type=int, default=50)
ap.add_argument("--n-grid", type=int, default=128); ap.add_argument("--anchor", type=float, default=1e4)
ap.add_argument("--band", type=float, default=0.03); ap.add_argument("--amplify", type=float, default=10.0)
ap.add_argument("--fps", type=int, default=8)
args = ap.parse_args(); dev = args.device
rec = R.load(device=dev, specimen=args.specimen); C = rec["n_cells"]
z = np.load(args.params); mode = str(z["clock_mode"]) if "clock_mode" in z.files else "sigmoid"
P = M.Params(C, dev, clock_mode=mode, n_frames=int(z["clock"].shape[0]) if mode == "free" else 0)
P.load({k: z[k] for k in z.files if k in P.leaves() or k == "clock"})
win = R.beat_window(rec, args.beat); T = len(win["frames"])
_, ref_nodes, pos, X0, cid = model_node_displacement(args, rec, P, win)          # pos [T,N,2] particles
nodes_t = rec["pos"][win["frames"]].cpu().numpy()
band = ((X0[:, 0] < M.DOM_LO + args.band) | (X0[:, 0] > M.DOM_HI - args.band)
        | (X0[:, 1] < M.DOM_LO + args.band) | (X0[:, 1] > M.DOM_HI - args.band))
A = args.amplify
fig, ax = plt.subplots(1, 2, figsize=(14, 7.6))
s_nodes = ax[0].scatter(ref_nodes[:, 0], ref_nodes[:, 1], s=1.2, c=GREEN, lw=0)
s_band = ax[1].scatter(X0[band, 0], X0[band, 1], s=0.8, c=DIM, lw=0)
s_part = ax[1].scatter(X0[~band, 0], X0[~band, 1], s=0.8, c=BLUE, lw=0)
for a, name in zip(ax, (f"recording: {ref_nodes.shape[0]:,} tracking nodes, displacement from rest x{A:g}",
                        f"model: {X0.shape[0]:,} MPM particles, displacement from rest x{A:g} (dim = prescribed band)")):
    a.set_xlim(M.DOM_LO - 0.02, M.DOM_HI + 0.03); a.set_ylim(M.DOM_LO - 0.02, M.DOM_HI + 0.03); a.set_aspect("equal")
    a.set_xticks([]); a.set_yticks([]); a.set_xlabel(name)
txt = fig.text(0.5, 0.965, "", ha="center"); fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.07, wspace=0.04)
import imageio_ffmpeg
od = os.path.join(HERE, "out", "movies"); os.makedirs(od, exist_ok=True)
path = os.path.join(od, f"particles_{os.path.basename(os.path.dirname(args.params))}_beat{args.beat}.mp4")
fig.canvas.draw(); w, h = fig.canvas.get_width_height(); w, h = w - w % 2, h - h % 2
writer = imageio_ffmpeg.write_frames(path, (w, h), fps=args.fps, quality=7); writer.send(None)
for t in range(T):
    s_nodes.set_offsets(ref_nodes + A * (nodes_t[t] - ref_nodes))
    dp = pos[t] - X0
    s_band.set_offsets(X0[band] + A * dp[band]); s_part.set_offsets(X0[~band] + A * dp[~band])
    txt.set_text(f"{args.specimen} sheet, beat {args.beat} ({'held out' if args.beat != 3 else 'fit beat'}), "
                 f"recording frame {win['frames'][t]}, t = {t * R.DT_S:.2f} s")
    fig.canvas.draw(); writer.send(np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[:h, :w, :3]))
writer.close(); print(f"  {T} frames -> {path}")
