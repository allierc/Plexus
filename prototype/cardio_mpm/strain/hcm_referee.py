"""hcm_referee -- which tracking of the HCM movie moves like the pixels?

Three files claim to track `1_HCM_15kPa_MR44_W3_1`: T1 = its own `.derivatives.npy` (137-grid),
T2 = `Cardio_0/derivatives.npy` (137-grid, identical onsets and amplitudes: probably the same
measurement in another layout), T3 = `diseased.npy` (80x80 top-left). After `healthy.npy` turned out
to be broken, none of them is trusted before the pixels have spoken: normalised cross-correlation
of 49 px patches, rest frame -> peak frame, at 400 nodes, against each file's own displacement.
T2 is tried in both layouts ([t, y, x] and [t, x, y]).
"""
import json, os, sys
import numpy as np, tifffile
from scipy.signal import fftconvolve
HERE = os.path.dirname(os.path.abspath(__file__))
RT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data"
TIF = f"{RT}/Cardio_1/1_HCM_15kPa_MR44_W3_1_MMStack_Pos0.ome.tif"
T1 = np.load(f"{TIF}.derivatives.npy", mmap_mode="r"); T2 = np.load(f"{RT}/Cardio_0/derivatives.npy", mmap_mode="r")
T3 = np.load(f"{RT}/diseased.npy", mmap_mode="r")
REST, PEAK = int(sys.argv[1]) if len(sys.argv) > 1 else 110, int(sys.argv[2]) if len(sys.argv) > 2 else 67
r, S, n = 24, 24, 400
with tifffile.TiffFile(TIF) as tf:
    I_rest = tf.pages[REST].asarray().astype(np.float32); I_peak = tf.pages[PEAK].asarray().astype(np.float32)
P1 = np.asarray(T1[:, :, :, 0:2]); P2 = np.asarray(T2[:, :, :, 0:2]); P3 = np.asarray(T3)
cands = {"T1 (1_HCM derivatives)": P1, "T2 (Cardio_0) as [t,y,x]": P2,
         "T2 (Cardio_0) as [t,x,y]": np.swapaxes(P2, 1, 2)}
rng = np.random.default_rng(0)
ii, jj = np.meshgrid(np.arange(137), np.arange(137), indexing="ij")
ok = (ii >= 4) & (ii < 133) & (jj >= 4) & (jj < 133)
sel = rng.choice(np.where(ok.reshape(-1))[0], size=n, replace=False)
rows, D = [], []
for k in sel:
    i, j = divmod(k, 137)
    x, y = P1[REST, i, j]; cx, cy = int(round(x)), int(round(y))
    tpl = I_rest[cy - r:cy + r + 1, cx - r:cx + r + 1]; win = I_peak[cy - r - S:cy + r + S + 1, cx - r - S:cx + r + S + 1]
    if tpl.shape != (2 * r + 1, 2 * r + 1) or win.shape != (2 * (r + S) + 1,) * 2:
        continue
    t = tpl - tpl.mean(); w = win - win.mean()
    num = fftconvolve(w, t[::-1, ::-1], mode="valid")
    e = np.sqrt(np.clip(fftconvolve(w ** 2, np.ones_like(t), mode="valid"), 1e-9, None)) * np.sqrt((t ** 2).sum())
    ncc = num / e; iy, ix = np.unravel_index(np.argmax(ncc), ncc.shape)
    dy = dx = 0.0
    if 0 < iy < ncc.shape[0] - 1 and 0 < ix < ncc.shape[1] - 1:
        dy = 0.5 * (ncc[iy - 1, ix] - ncc[iy + 1, ix]) / (ncc[iy - 1, ix] - 2 * ncc[iy, ix] + ncc[iy + 1, ix] + 1e-12)
        dx = 0.5 * (ncc[iy, ix - 1] - ncc[iy, ix + 1]) / (ncc[iy, ix - 1] - 2 * ncc[iy, ix] + ncc[iy, ix + 1] + 1e-12)
    D.append([ix - S + dx, iy - S + dy]); rows.append((i, j, float(ncc[iy, ix])))
D = np.array(D); q = np.array([r_[2] for r_ in rows]); good = q > np.percentile(q, 25)
print(f"rest frame {REST} -> peak frame {PEAK}; {len(rows)} nodes, ncc median {np.median(q):.3f}; |pixel disp| median "
      f"{np.median(np.linalg.norm(D[good], axis=1)):.2f} px")
out = {}
for name, P in cands.items():
    d = np.array([P[PEAK, i, j] - P[REST, i, j] for i, j, _ in rows])
    cx_, cy_ = np.corrcoef(D[good, 0], d[good, 0])[0, 1], np.corrcoef(D[good, 1], d[good, 1])[0, 1]
    err = np.median(np.linalg.norm(D[good] - d[good], axis=1))
    print(f"  pixels vs {name:<28s}: corr x {cx_:+.3f} y {cy_:+.3f}; median |diff| {err:.2f} px")
    out[name] = dict(corr_x=float(cx_), corr_y=float(cy_), median_diff_px=float(err))
# T3: 80x80 top-left of the same lattice (rows i<80, cols j<80), frame offset unknown -> try 0..2 with its own frames
sub = [(i, j, k) for k, (i, j, _) in enumerate(rows) if i < 80 and j < 80]
if len(sub) > 30:
    for off in (0, -1, 1):
        d = np.array([P3[PEAK + off, i, j] - P3[REST + off, i, j] for i, j, _ in sub]); Dk = np.array([D[k] for _, _, k in sub])
        g2 = np.array([good[k] for _, _, k in sub])
        cx_, cy_ = np.corrcoef(Dk[g2, 0], d[g2, 0])[0, 1], np.corrcoef(Dk[g2, 1], d[g2, 1])[0, 1]
        print(f"  pixels vs T3 (diseased.npy, frame offset {off:+d}), {int(g2.sum())} nodes: corr x {cx_:+.3f} y {cy_:+.3f}")
        out[f"T3 offset {off}"] = dict(corr_x=float(cx_), corr_y=float(cy_), n=int(g2.sum()))
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
json.dump(dict(rest=REST, peak=PEAK, results=out), open(os.path.join(HERE, "out", f"hcm_referee_{REST}_{PEAK}.json"), "w"), indent=1)
