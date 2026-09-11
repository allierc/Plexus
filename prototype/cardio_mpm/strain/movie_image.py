"""movie_image -- the raw movie next to the rest frame warped by the model's motion.

Two panels over one beat window: left, the microscope frames as recorded; right, ONE frame of the
same movie (the rest frame at the end of the window) warped by the displacement field the fitted
model predicts for that instant. If the model is right, the right panel moves like the left one:
the texture is the tissue's own, only its motion is the model's.

The model's displacement is known at the 18,769 tracking nodes (each node moves with its cell's
fitted affine map); it is interpolated linearly between nodes to every pixel, and the rest frame is
sampled backwards (I_pred(x) = I_rest(x - u(x))). Motion is 1-3 px on a 2048 px field, so
`--amplify` (default 1 = true motion) exists for when the eye needs help; the caption always says
which. Frames are downsampled 2x for the video.

    python movie_image.py --params out/fits/s4_live_r6_Eshrink0.3/params.npz --beat 1
    python movie_image.py --specimen hcm --params out/fits/hcm_r2_Eshrink0.3/params.npz --beat 2
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import tifffile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M      # noqa: E402
import recording as R  # noqa: E402

RT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data/Cardio_1"
TIFS = dict(healthy=f"{RT}/0_B_15kPa_1_MMStack_Pos0.ome.tif",
            hcm=f"{RT}/1_HCM_15kPa_MR44_W3_1_MMStack_Pos0.ome.tif")
WHITE, GREY = "#f2f2f2", "#8a8a8a"
plt.rcParams.update({"figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
                     "text.color": WHITE, "axes.labelcolor": WHITE, "font.size": 11})


def model_node_displacement(args, rec, P, win):
    """[T,N,2] model displacement at the tracking nodes, world units, from the fitted rollout."""
    dev = args.device; C = rec["n_cells"]; T = len(win["frames"])
    A_ref, u_ref = R.window_affine(rec, win)
    trunc = (win["onset"] - win["span"][0]) - R.PRE
    P.shift = -float(trunc)
    LTIF = os.path.join(HERE, "data_hcm" if rec.get("specimen") == "hcm" else "data", "cells_2560.tif")
    kw = dict(n_grid=args.n_grid, per_parent=args.per_parent, n_frames=T - 1, n_cells=C, anchor_k=args.anchor, drag_k=args.drag, anchor_percell=args.anchor_percell,
              label_tif=LTIF)
    with torch.no_grad():
        r0 = M.rollout(M.load_sim(M.build_spec(differentiable=False, name="mi_rest", **dict(kw, n_frames=0))),
                       P, dev, C, grad=False)
        X0, cid = r0["X0"], r0["cid"]
        band = ((X0[:, 0] < M.DOM_LO + args.band) | (X0[:, 0] > M.DOM_HI - args.band)
                | (X0[:, 1] < M.DOM_LO + args.band) | (X0[:, 1] > M.DOM_HI - args.band))
        out = M.rollout(M.load_sim(M.build_spec(differentiable=False, name="mi", **kw)), P, dev, C, grad=False,
                        prescribe=(band, M.band_prescription(A_ref, u_ref, X0, cid, band)), keep_pos=True)
    lab = rec["labels"].cpu().numpy(); node_cell = lab - 1
    ones = torch.ones(X0.shape[0], device=dev); idx = cid - 1
    cnt = torch.zeros(C, device=dev).index_add(0, idx, ones).clamp(min=1)
    Xbar = (torch.zeros(C, 2, device=dev).index_add(0, idx, X0) / cnt[:, None]).cpu().numpy()
    ref_nodes = win["ref"].cpu().numpy()
    dX = ref_nodes - Xbar[node_cell]
    Am, um = out["A"].cpu().numpy(), out["u"].cpu().numpy()
    disp = np.einsum("tnij,nj->tni", Am[:, node_cell] - np.eye(2), dX) + um[:, node_cell]
    return disp, ref_nodes, out["pos"], X0.cpu().numpy(), cid.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default=os.path.join(HERE, "out", "fits", "s4_live_r6_Eshrink0.3", "params.npz"))
    ap.add_argument("--specimen", default="healthy", choices=["healthy", "hcm"])
    ap.add_argument("--beat", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per-parent", type=int, default=50)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--anchor", type=float, default=1e4)
    ap.add_argument("--anchor-percell", action="store_true")
    ap.add_argument("--drag", type=float, default=30.0, help="Stokes drag k; 150 critically damps the substrate mode (step_test.py)")
    ap.add_argument("--band", type=float, default=0.03)
    ap.add_argument("--amplify", type=float, default=1.0)
    ap.add_argument("--down", type=int, default=2, help="downsampling of the 2048 px frames")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--overlay", action="store_true",
                    help="ONE panel: the recording in green and the model-warped rest frame in magenta, "
                         "superposed. Where they agree the picture is grey; a mismatch shows as a green/"
                         "magenta fringe on the side the model is late or short.")
    args = ap.parse_args()
    dev = args.device
    torch.backends.cuda.matmul.allow_tf32 = True

    rec = R.load(device=dev, specimen=args.specimen); C = rec["n_cells"]
    z = np.load(args.params)
    mode = str(z["clock_mode"]) if "clock_mode" in z.files else "sigmoid"
    P = M.Params(C, dev, clock_mode=mode, n_frames=int(z["psi"].shape[1]) if "psi" in z.files else (int(z["clock"].shape[0]) if mode == "free" else 0), n_modes=int(z["n_modes"]) if "n_modes" in z.files else 0)
    P.load({k: z[k] for k in z.files if k in P.leaves() or k == "clock"})
    win = R.beat_window(rec, args.beat); T = len(win["frames"]); lo, hi = win["span"]
    disp, ref_nodes, _, _, _ = model_node_displacement(args, rec, P, win)       # world units

    d = args.down; px = 2048.0 / 0.7 / d                                        # world -> downsampled px
    nodes_px = (ref_nodes - 0.15) * px                                          # [N,2] (x, y)
    disp_px = disp * px * args.amplify
    disp_rec_px = (rec["pos"][win["frames"]].cpu().numpy() - ref_nodes[None]) * px * args.amplify
    tri = Delaunay(nodes_px)
    n = 2048 // d
    Y, X = np.mgrid[0:n, 0:n].astype(np.float32)
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    with tifffile.TiffFile(TIFS[args.specimen]) as tf:
        rest_frame = hi - 2                                                     # in the rest tail
        I_rest = tf.pages[rest_frame].asarray().astype(np.float32)[::d, ::d]
        frames = [tf.pages[t].asarray().astype(np.float32)[::d, ::d] for t in win["frames"]]
    vmin, vmax = np.percentile(I_rest, (1, 99.5))

    def norm(im):
        return np.clip((im - vmin) / (vmax - vmin), 0, 1)
    if args.overlay:
        fig, a = plt.subplots(1, 1, figsize=(8.4, 8.9))
        rgb0 = np.stack([norm(I_rest)] * 3, -1)
        ims = [a.imshow(rgb0, origin="upper", interpolation="nearest")]
        a.set_xticks([]); a.set_yticks([])
        a.set_xlabel(f"green = recording, magenta = rest frame {rest_frame} warped by the fitted motion"
                     + (f" (x{args.amplify:g})" if args.amplify != 1 else "") + "; grey = they agree")
        ax = [a]
    else:
        fig, ax = plt.subplots(1, 2, figsize=(14, 7.6))
        ims = []
        for a, name in zip(ax, ("recording (raw frames)", f"model: rest frame {rest_frame} warped by the fitted motion"
                                                         + (f", x{args.amplify:g}" if args.amplify != 1 else ""))):
            ims.append(a.imshow(I_rest, cmap="gray", vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest"))
            a.set_xticks([]); a.set_yticks([]); a.set_xlabel(name)
    txt = fig.text(0.5, 0.965, "", ha="center")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.07, wspace=0.04)

    import imageio_ffmpeg
    od = os.path.join(HERE, "out", "movies"); os.makedirs(od, exist_ok=True)
    tag = os.path.basename(os.path.dirname(args.params))
    path = os.path.join(od, f"{'overlay' if args.overlay else 'image'}_{tag}_beat{args.beat}{'' if args.amplify == 1 else f'_x{args.amplify:g}'}.mp4")
    fig.canvas.draw(); w, h = fig.canvas.get_width_height(); w, h = w - w % 2, h - h % 2
    writer = imageio_ffmpeg.write_frames(path, (w, h), fps=args.fps, quality=7); writer.send(None)
    for t in range(T):
        f = LinearNDInterpolator(tri, disp_px[t], fill_value=0.0)
        u = f(pts).reshape(n, n, 2)
        warped = map_coordinates(I_rest, [Y - u[..., 1], X - u[..., 0]], order=1, mode="nearest")
        if args.overlay:
            # the recording drives the green channel, the model-warped frame red and blue (magenta); when the
            # warp reproduces the frame the three channels agree and the pixel is grey
            g_, m_ = norm(frames[t]), norm(warped)
            if args.amplify != 1:            # amplified motion: warp the RECORDING'S rest frame by the tracked
                fr = LinearNDInterpolator(tri, disp_rec_px[t], fill_value=0.0)(pts).reshape(n, n, 2)   # motion too,
                g_ = norm(map_coordinates(I_rest, [Y - fr[..., 1], X - fr[..., 0]], order=1, mode="nearest"))  # same scale
            ims[0].set_data(np.stack([m_, g_, m_], -1))
        else:
            ims[0].set_data(frames[t]); ims[1].set_data(warped)
        txt.set_text(f"{args.specimen} sheet, beat {args.beat} ({'held out' if args.beat != 3 else 'fit beat'}), "
                     f"recording frame {win['frames'][t]}, t = {t * R.DT_S:.2f} s; the tissue moves 1-3 px per beat"
                     + (f" (drawn x{args.amplify:g})" if args.amplify != 1 else ""))
        fig.canvas.draw()
        writer.send(np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[:h, :w, :3]))
    writer.close()
    print(f"  {T} frames -> {path}")


if __name__ == "__main__":
    main()
