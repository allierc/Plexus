"""movie -- the prediction, frame by frame, next to the recording, on a beat the fit never saw.

Four panels over one beat window (default beat 1, held out; the fit was on beat 3):
  A  recording: per-cell shortening strain at this frame, painted on the cells
  B  model:     the same, from the fitted parameters rolled out from rest
  C  displacement of every 4th tracking node, recording (green) and model (white), amplified, with
     the sheet's mean displacement of that frame subtracted (the deformation, not the drift)
  D  mean shortening over cells against time, both curves, with the current frame marked;
     the held-out R^2 of the round is printed in the corner

Repository convention: black background, no titles, white panel letters; green = recording,
white = model. Written with imageio-ffmpeg to out/movies/<tag>_beat<k>.mp4.

    python movie.py --params out/fits/s4_live_r4_delay_Efree/params.npz --beat 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M      # noqa: E402
import recording as R  # noqa: E402

GREEN, WHITE, GREY = "#3fbf6f", "#f2f2f2", "#8a8a8a"
plt.rcParams.update({"figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
                     "axes.edgecolor": GREY, "axes.labelcolor": WHITE, "xtick.color": WHITE,
                     "ytick.color": WHITE, "text.color": WHITE, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default=os.path.join(HERE, "out", "fits", "s4_live_r4_delay_Efree", "params.npz"))
    ap.add_argument("--beat", type=int, default=1)
    ap.add_argument("--specimen", default="healthy", choices=["healthy", "hcm"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per-parent", type=int, default=50)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--anchor", type=float, default=1e4)
    ap.add_argument("--anchor-percell", action="store_true")
    ap.add_argument("--drag", type=float, default=30.0, help="Stokes drag k; 150 critically damps the substrate mode (step_test.py)")
    ap.add_argument("--band", type=float, default=0.03)
    ap.add_argument("--amplify", type=float, default=12.0, help="displacement arrows are drawn this many times longer")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--stride", type=int, default=5, help="arrows on every stride-th row and column of the node lattice")
    args = ap.parse_args()
    dev = args.device
    torch.backends.cuda.matmul.allow_tf32 = True

    rec = R.load(device=dev, specimen=args.specimen); C = rec["n_cells"]
    LTIF = os.path.join(HERE, "data_hcm" if rec.get("specimen") == "hcm" else "data", "cells_2560.tif")
    z = np.load(args.params)
    mode = str(z["clock_mode"]) if "clock_mode" in z.files else "sigmoid"
    P = M.Params(C, dev, clock_mode=mode, n_frames=int(z["psi"].shape[1]) if "psi" in z.files else (int(z["clock"].shape[0]) if mode == "free" else 0), n_modes=int(z["n_modes"]) if "n_modes" in z.files else 0)
    P.load({k: z[k] for k in z.files if k in P.leaves() or k == "clock"})
    win = R.beat_window(rec, args.beat); T = len(win["frames"])
    A_ref, u_ref = R.window_affine(rec, win)
    trunc = (win["onset"] - win["span"][0]) - R.PRE
    P.shift = -float(trunc)
    kw = dict(n_grid=args.n_grid, per_parent=args.per_parent, n_frames=T - 1, n_cells=C, anchor_k=args.anchor, drag_k=args.drag, anchor_percell=args.anchor_percell)
    with torch.no_grad():
        r0 = M.rollout(M.load_sim(M.build_spec(label_tif=LTIF, differentiable=False, name="mv_rest", **dict(kw, n_frames=0))),
                       P, dev, C, grad=False)
        X0, cid = r0["X0"], r0["cid"]
        band = ((X0[:, 0] < M.DOM_LO + args.band) | (X0[:, 0] > M.DOM_HI - args.band)
                | (X0[:, 1] < M.DOM_LO + args.band) | (X0[:, 1] > M.DOM_HI - args.band))
        out = M.rollout(M.load_sim(M.build_spec(label_tif=LTIF, differentiable=False, name="mv", **kw)), P, dev, C,
                        grad=False, prescribe=(band, M.band_prescription(A_ref, u_ref, X0, cid, band)), keep_pos=True)
    A, pos = out["A"], out["pos"]                                   # [T,C,2,2], [T,N,2] numpy
    sm, sr = R.shortening(A).cpu().numpy(), R.shortening(A_ref).cpu().numpy()
    eye = torch.eye(2, device=dev)
    inter = torch.as_tensor(z["interior"], device=dev) if "interior" in z.files else torch.ones(C, dtype=torch.bool, device=dev)
    Ai, Ri = A[:, inter], A_ref[:, inter]          # cells the loss saw: band cells are dictated, not modelled
    r2A = float(1 - ((Ai - Ri) ** 2).sum() / ((Ri - eye - (Ri - eye).mean()) ** 2).sum())

    # model displacement AT the tracking nodes: each node takes its cell's affine map from the model
    ref_nodes = win["ref"].cpu().numpy(); lab = rec["labels"].cpu().numpy()
    lab_grid = np.load(os.path.join(HERE, "data_hcm" if args.specimen == "hcm" else "data", "labels_grid.npy"))
    ones = torch.ones(X0.shape[0], device=dev); idx = cid - 1
    cnt = torch.zeros(C, device=dev).index_add(0, idx, ones).clamp(min=1)
    Xbar = (torch.zeros(C, 2, device=dev).index_add(0, idx, X0) / cnt[:, None]).cpu().numpy()
    node_cell = lab - 1
    dX = ref_nodes - Xbar[node_cell]
    Am, um = A.cpu().numpy(), out["u"].cpu().numpy()
    disp_model = np.einsum("tnij,nj->tni", Am[:, node_cell] - np.eye(2), dX) + um[:, node_cell]
    disp_rec = rec["pos"][win["frames"]].cpu().numpy() - ref_nodes[None]

    ii, jj = np.divmod(np.arange(ref_nodes.shape[0]), 137)          # node n = row*137 + col
    sel = np.where((ii % args.stride == 0) & (jj % args.stride == 0))[0]   # a regular sub-lattice
    vmax = float(np.percentile(sr, 99))
    t_s = np.arange(T) * R.DT_S

    def cell_map(v):
        m = np.full(C + 1, np.nan); m[1:] = v; return m[lab_grid]

    disp_rec = disp_rec - disp_rec.mean(1, keepdims=True)       # the deformation pattern, not the drift
    disp_model = disp_model - disp_model.mean(1, keepdims=True)
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 11.8), gridspec_kw=dict(wspace=0.25, hspace=0.22))
    ax = ax.ravel()
    imA = ax[0].imshow(cell_map(sr[0]), cmap="Greens", vmin=0, vmax=vmax, origin="lower", interpolation="nearest")
    imB = ax[1].imshow(cell_map(sm[0]), cmap="Greens", vmin=0, vmax=vmax, origin="lower", interpolation="nearest")
    for a, s, name in ((ax[0], "A", "recording"), (ax[1], "B", "model, fitted on beat 3")):
        a.set_xticks([]); a.set_yticks([])
        a.set_xlabel(f"{name}: shortening strain per cell")
        a.text(-0.02, 1.01, s, transform=a.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="right")
    cb = fig.colorbar(imB, ax=ax[1], fraction=0.04, pad=0.02); cb.set_label("shortening strain (dimensionless)")
    a = ax[2]
    qr = a.quiver(ref_nodes[sel, 0], ref_nodes[sel, 1], disp_rec[0, sel, 0], disp_rec[0, sel, 1], color=GREEN,
                  angles="xy", scale_units="xy", scale=1.0 / args.amplify, width=0.004, label="recording")
    qm = a.quiver(ref_nodes[sel, 0], ref_nodes[sel, 1], disp_model[0, sel, 0], disp_model[0, sel, 1], color=WHITE,
                  angles="xy", scale_units="xy", scale=1.0 / args.amplify, width=0.0025, label="model")
    a.set_xlim(M.DOM_LO - 0.01, M.DOM_HI + 0.02); a.set_ylim(M.DOM_LO - 0.01, M.DOM_HI + 0.02); a.set_aspect("equal")
    a.set_xticks([]); a.set_yticks([])
    a.set_xlabel(f"displacement from rest of one tracking node in {args.stride} per axis, sheet mean removed, "
                 f"drawn x{args.amplify:g}\n(green = recording, white = model; the outer band is prescribed from the recording)")
    a.text(-0.02, 1.01, "C", transform=a.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="right")
    a = ax[3]
    a.plot(t_s, sr.mean(1), color=GREEN, lw=2, label="recording")
    a.plot(t_s, sm.mean(1), color=WHITE, lw=2, label="model")
    a.fill_between(t_s, np.percentile(sr, 25, 1), np.percentile(sr, 75, 1), color=GREEN, alpha=0.18, lw=0)
    marker = a.axvline(0, color=GREY, lw=1)
    a.set_xlabel(f"time in the window (s); beat {args.beat}, frames {win['span'][0]}-{win['span'][1]}"
                 f"{' -- held out' if args.beat != 3 else ' -- the fit beat'}")
    a.set_ylabel("shortening strain, mean over 472 cells\n(band = recording's 25-75% across cells)")
    a.legend(loc="upper right", fontsize=9)
    a.text(-0.02, 1.01, "D", transform=a.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="right")
    a.text(0.99, 0.02, f"per-cell strain maps, whole window, {int(inter.sum())} interior cells: R$^2$ = {r2A:.2f}", transform=a.transAxes,
           ha="right", va="bottom", fontsize=9)
    frame_txt = fig.text(0.5, 0.965, "", ha="center", fontsize=11)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.08)

    import imageio_ffmpeg
    od = os.path.join(HERE, "out", "movies"); os.makedirs(od, exist_ok=True)
    tag = os.path.basename(os.path.dirname(args.params))
    path = os.path.join(od, f"{tag}_beat{args.beat}.mp4")
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    w, h = w - w % 2, h - h % 2
    writer = imageio_ffmpeg.write_frames(path, (w, h), fps=args.fps, quality=7)
    writer.send(None)
    for t in range(T):
        imA.set_data(cell_map(sr[t])); imB.set_data(cell_map(sm[t]))
        qr.set_UVC(disp_rec[t, sel, 0], disp_rec[t, sel, 1]); qm.set_UVC(disp_model[t, sel, 0], disp_model[t, sel, 1])
        marker.set_xdata([t_s[t], t_s[t]])
        frame_txt.set_text(f"frame {win['frames'][t]} of the recording   t = {t_s[t]:.2f} s   "
                           f"mean shortening: recording {sr[t].mean():.4f}, model {sm[t].mean():.4f}")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:h, :w, :3]
        writer.send(np.ascontiguousarray(buf))
    writer.close()
    print(f"  {T} frames -> {path}   (R2 of the strain maps over the window {r2A:.3f})")
    json.dump(dict(params=args.params, beat=args.beat, window=win["span"], r2_A=r2A, frames=T, fps=args.fps),
              open(path.replace(".mp4", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
