"""layout_convergence -- does the forward converge as the particle density rises, and does the
particle layout (random vs regular lattice) matter?  Same fitted parameters, held-out beat 1.
Reports held-out R^2 against the recording and R^2 of each layout predicting the next denser one.

    python layout_convergence.py --ppc 50,80,120,200 [--lattice] [--n-grid 128]
"""
import argparse, json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import model as M, recording as R
ap = argparse.ArgumentParser(); ap.add_argument("--ppc", default="50,80,120,200"); ap.add_argument("--n-grid", type=int, default=128)
ap.add_argument("--lattice", action="store_true"); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--params", default=os.path.join(HERE, "out", "fits", "s4_live_r6_Eshrink0.3", "params.npz"))
ap.add_argument("--beat", type=int, default=1); ap.add_argument("--specimen", default="healthy")
args = ap.parse_args(); dev = args.device
rec = R.load(device=dev, specimen=args.specimen); C = rec["n_cells"]; win = R.beat_window(rec, args.beat); T = len(win["frames"])
A_rec, u_rec = R.window_affine(rec, win); z = np.load(args.params); P = M.Params(C, dev)
P.load({k: z[k] for k in ("g", "phi", "logE", "clock", "delay")}); P.shift = -float((win["onset"] - win["span"][0]) - R.PRE)
inter = torch.as_tensor(z["interior"], device=dev); eye = torch.eye(2, device=dev)
LTIF = os.path.join(HERE, "data_hcm" if args.specimen == "hcm" else "data", "cells_2560.tif")
def r2(a, b):
    return float(1 - ((a - b) ** 2).sum() / ((b - b.mean()) ** 2).sum())
out = {}; prev = None; rows = []
for pp in [int(v) for v in args.ppc.split(",")]:
    kw = dict(n_grid=args.n_grid, per_parent=pp, n_frames=T - 1, n_cells=C, anchor_k=1e4, label_tif=LTIF)
    with torch.no_grad():
        r0 = M.rollout(M.load_sim(M.build_spec(differentiable=False, name="lc0", **dict(kw, n_frames=0))), P, dev, C, grad=False,
                       lattice=args.lattice)
        X0, cid = r0["X0"], r0["cid"]; band = ((X0[:, 0] < 0.18) | (X0[:, 0] > 0.82) | (X0[:, 1] < 0.18) | (X0[:, 1] > 0.82))
        o = M.rollout(M.load_sim(M.build_spec(differentiable=False, name="lc", **kw)), P, dev, C, grad=False,
                      prescribe=(band, M.band_prescription(A_rec, u_rec, X0, cid, band)), lattice=args.lattice)
    E = (o["A"] - eye)[:, inter]; Er = (A_rec - eye)[:, inter]
    row = dict(ppc=pp, particles=int(X0.shape[0]), r2_heldout=r2(E, Er), r2_vs_previous=(r2(prev[1], E) if prev else None),
               shortening_corr_vs_previous=(float(torch.corrcoef(torch.stack([R.shortening(prev[0])[:, inter].max(0).values,
                                                                             R.shortening(o["A"])[:, inter].max(0).values]))[0, 1]) if prev else None))
    rows.append(row); prev = (o["A"], E)
    print(f"  {'lattice' if args.lattice else 'random '} {pp:4d}/cell ({X0.shape[0]:7,}): held-out R2(A) {row['r2_heldout']:.3f}"
          + (f"; vs previous density R2 {row['r2_vs_previous']:.3f}, peak-shortening corr {row['shortening_corr_vs_previous']:.4f}" if prev and row['r2_vs_previous'] is not None else ""), flush=True)
json.dump(dict(config=vars(args), rows=rows), open(os.path.join(HERE, "out", f"layout_convergence_{'lattice' if args.lattice else 'random'}_g{args.n_grid}.json"), "w"), indent=1)
