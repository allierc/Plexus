"""movie_cells_color -- the instance segmentation itself, deforming: one colour per cell.

Left, the recording: each of the 18,769 tracking nodes drawn at its tracked position, coloured by
the cell it belongs to. Right, the model: each of the 56,640 MPM particles at its position in the
rollout, coloured by the same cell's colour. Nothing is averaged and nothing is a heatmap -- what
moves on the screen is the segmentation, so a cell that shortens is seen to shorten and a cell that
rotates is seen to rotate.

Displacement is amplified (`--amplify`, default 10) because a beat moves the tissue 1-3 px in 2048;
the caption says so. The colours are a fixed random permutation of the hue circle, so the same cell
keeps its colour between the two panels and between runs.

    python movie_cells_color.py --continuous --params out/fits/healthy_allbeats/params.npz --seconds 6
"""
import argparse, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import model as M, recording as R
from movie_image import model_node_displacement
WHITE, GREY = "#f2f2f2", "#8a8a8a"
plt.rcParams.update({"figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
                     "text.color": WHITE, "axes.labelcolor": WHITE, "font.size": 11})

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--params", default=os.path.join(HERE, "out", "fits", "healthy_allbeats", "params.npz"))
ap.add_argument("--specimen", default="healthy", choices=["healthy", "hcm"]); ap.add_argument("--beat", type=int, default=1)
ap.add_argument("--continuous", action="store_true"); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--per-parent", type=int, default=120); ap.add_argument("--n-grid", type=int, default=128)
ap.add_argument("--anchor", type=float, default=1e4); ap.add_argument("--drag", type=float, default=150.0)
ap.add_argument("--anchor-percell", action="store_true"); ap.add_argument("--band", type=float, default=0.03)
ap.add_argument("--amplify", type=float, default=10.0); ap.add_argument("--seconds", type=float, default=6.0)
ap.add_argument("--fps", type=int, default=8); ap.add_argument("--tag", default="cont")
args = ap.parse_args(); dev = args.device
rec = R.load(device=dev, specimen=args.specimen); C = rec["n_cells"]
z = np.load(args.params); mode = str(z["clock_mode"]) if "clock_mode" in z.files else "sigmoid"
P = M.Params(C, dev, clock_mode=mode, n_frames=int(z["psi"].shape[1]) if "psi" in z.files else 0,
             n_modes=int(z["n_modes"]) if "n_modes" in z.files else 0)
P.load({k: z[k] for k in z.files if k in P.leaves() or k == "clock"})
win = R.continuous_window(rec) if args.continuous else R.beat_window(rec, args.beat)
T = len(win["frames"])
_, ref_nodes, pos, X0, cid = model_node_displacement(args, rec, P, win)     # the rollout
nodes_t = rec["pos"][win["frames"]].cpu().numpy()
lab = rec["labels"].cpu().numpy()

rng = np.random.default_rng(7)                       # one hue per cell, the same in both panels
hue = rng.permutation(C) / C
col = hsv_to_rgb(np.stack([hue, np.full(C, 0.75), np.full(C, 1.0)], 1))
c_nodes, c_part = col[lab - 1], col[cid - 1]
A = args.amplify
fig, ax = plt.subplots(1, 2, figsize=(15, 8.2))
s_rec = ax[0].scatter(ref_nodes[:, 0], ref_nodes[:, 1], s=2.6, c=c_nodes, lw=0)
s_mod = ax[1].scatter(X0[:, 0], X0[:, 1], s=1.0, c=c_part, lw=0)
for a, name in zip(ax, (f"recording: {ref_nodes.shape[0]:,} tracking nodes, one colour per cell",
                        f"model: {X0.shape[0]:,} MPM particles, the same colours")):
    a.set_xlim(M.DOM_LO - 0.02, M.DOM_HI + 0.03); a.set_ylim(M.DOM_LO - 0.02, M.DOM_HI + 0.03)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.set_xlabel(f"{name}, motion x{A:g}")
txt = fig.text(0.5, 0.965, "", ha="center")
fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.07, wspace=0.04)
import imageio_ffmpeg
od = os.path.join(HERE, "out", "movies"); os.makedirs(od, exist_ok=True)
path = os.path.join(od, f"cont_cells_colour.mp4" if args.continuous else f"cells_colour_{args.tag}.mp4")
fig.canvas.draw(); w, h = fig.canvas.get_width_height(); w, h = w - w % 2, h - h % 2
writer = imageio_ffmpeg.write_frames(path, (w, h), fps=(max(1, round(T / args.seconds)) if args.seconds > 0 else args.fps),
                                     quality=7); writer.send(None)
lab_txt = ("beats 1-3, one rollout, never reset" if args.continuous
           else f"beat {args.beat}")
for t in range(T):
    s_rec.set_offsets(ref_nodes + A * (nodes_t[t] - ref_nodes))
    s_mod.set_offsets(X0 + A * (pos[t] - X0))
    txt.set_text(f"{args.specimen} sheet, {lab_txt}, recording frame {win['frames'][t]}, t = {t * R.DT_S:.2f} s")
    fig.canvas.draw(); writer.send(np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[:h, :w, :3]))
writer.close(); print(f"  {T} frames -> {path}")
