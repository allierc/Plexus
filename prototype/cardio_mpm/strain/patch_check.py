"""patch_check -- a third measurement of the beat, from the pixels, to referee two trackings that
disagree on the spatial pattern of one movie.

At `--n` nodes of the 137-lattice inside the top-left overlap, take a (2r+1)^2 patch of the REST
frame (median-of-plateau is not an image, so the single frame at the window's rest tail is used)
and find its displacement in the PEAK frame by normalised cross-correlation over a +-S px search,
refined to sub-pixel by a parabolic fit. Compare that vector with each tracking's displacement of
the same node between the same two frames. Whichever tracking the pixels agree with is the one to
believe; if neither, the spatial pattern of this movie is not measurable this way.
"""
import argparse, json, os, sys
import numpy as np, tifffile, torch
from scipy.signal import fftconvolve
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import recording as R
RT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data"
TIF = f"{RT}/Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif"

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=400)
ap.add_argument("--r", type=int, default=24); ap.add_argument("--S", type=int, default=20)
ap.add_argument("--rest", type=int, default=190); ap.add_argument("--peak", type=int, default=158)
args = ap.parse_args()
rec = R.load(); pos = rec["pos"]
h = np.load(f"{RT}/healthy.npy").astype(np.float32); T2 = h.shape[0]
p2 = 0.15 + 0.7 * h.reshape(T2, -1, 2) / 2048.0
ii, jj = np.meshgrid(np.arange(80), np.arange(80), indexing="ij"); midx = (ii * 137 + jj).reshape(-1)
px = lambda w: (w - 0.15) / 0.7 * 2048.0                         # world -> image px
with tifffile.TiffFile(TIF) as tf:
    I_rest = tf.pages[args.rest].asarray().astype(np.float32)
    I_peak = tf.pages[args.peak].asarray().astype(np.float32)
rng = np.random.default_rng(0)
cand = [k for k in range(80 * 80) if 5 <= ii.reshape(-1)[k] < 75 and 5 <= jj.reshape(-1)[k] < 75]
sel = rng.choice(cand, size=args.n, replace=False)
r, S = args.r, args.S
rows = []
for k in sel:
    n1 = midx[k]
    X = px(pos[args.rest, n1].numpy())                          # tracking 1's rest position, px (x, y)
    cx, cy = int(round(X[0])), int(round(X[1]))
    tpl = I_rest[cy - r:cy + r + 1, cx - r:cx + r + 1]
    win = I_peak[cy - r - S:cy + r + S + 1, cx - r - S:cx + r + S + 1]
    if tpl.shape != (2 * r + 1, 2 * r + 1) or win.shape != (2 * (r + S) + 1, 2 * (r + S) + 1):
        continue
    t = tpl - tpl.mean(); w = win - win.mean()
    num = fftconvolve(w, t[::-1, ::-1], mode="valid")
    # local energy of the window under the template for normalisation
    ones = np.ones_like(t)
    e = fftconvolve(w ** 2, ones, mode="valid"); e = np.sqrt(np.clip(e, 1e-9, None)) * np.sqrt((t ** 2).sum())
    ncc = num / e
    iy, ix = np.unravel_index(np.argmax(ncc), ncc.shape)
    if 0 < iy < ncc.shape[0] - 1 and 0 < ix < ncc.shape[1] - 1:      # parabolic sub-pixel
        dy = 0.5 * (ncc[iy - 1, ix] - ncc[iy + 1, ix]) / (ncc[iy - 1, ix] - 2 * ncc[iy, ix] + ncc[iy + 1, ix] + 1e-12)
        dx = 0.5 * (ncc[iy, ix - 1] - ncc[iy, ix + 1]) / (ncc[iy, ix - 1] - 2 * ncc[iy, ix] + ncc[iy, ix + 1] + 1e-12)
    else:
        dy = dx = 0.0
    d_img = np.array([ix - S + dx, iy - S + dy])                  # (x, y) px, rest -> peak
    d1 = px(pos[args.peak, n1].numpy()) - px(pos[args.rest, n1].numpy())
    d2 = px(p2[args.peak, k]) - px(p2[args.rest, k])
    rows.append(dict(node=int(n1), ncc=float(ncc[iy, ix]), d_img=d_img.tolist(), d1=d1.tolist(), d2=d2.tolist()))
D = np.array([r_["d_img"] for r_ in rows]); D1 = np.array([r_["d1"] for r_ in rows]); D2 = np.array([r_["d2"] for r_ in rows])
q = np.array([r_["ncc"] for r_ in rows]); good = q > np.percentile(q, 25)
def rep(name, A, B, m):
    cx = np.corrcoef(A[m, 0], B[m, 0])[0, 1]; cy = np.corrcoef(A[m, 1], B[m, 1])[0, 1]
    err = np.linalg.norm(A[m] - B[m], axis=1)
    print(f"  {name}: corr x {cx:+.3f} y {cy:+.3f}; median |diff| {np.median(err):.2f} px; "
          f"|A| median {np.median(np.linalg.norm(A[m],axis=1)):.2f}, |B| median {np.median(np.linalg.norm(B[m],axis=1)):.2f} px")
    return dict(corr_x=float(cx), corr_y=float(cy), median_diff_px=float(np.median(err)))
print(f"{len(rows)} nodes, rest frame {args.rest}, peak frame {args.peak}, patch {2*r+1} px, search +-{S} px; "
      f"ncc median {np.median(q):.3f}; using the best {int(good.sum())}")
out = dict(n=len(rows), rest=args.rest, peak=args.peak, patch=2 * r + 1, search=S,
           pixels_vs_tracking1=rep("pixels vs tracking 1", D, D1, good),
           pixels_vs_tracking2=rep("pixels vs tracking 2", D, D2, good),
           tracking1_vs_tracking2=rep("tracking 1 vs tracking 2", D1, D2, good))
# also with the rest frame's own displacement subtracted for tracking 2 referenced at ITS frame 0? no: both
# trackings are compared between the same two frames, so their references cancel.
json.dump(out, open(os.path.join(HERE, "out", f"patch_check_{args.rest}_{args.peak}.json"), "w"), indent=1)
