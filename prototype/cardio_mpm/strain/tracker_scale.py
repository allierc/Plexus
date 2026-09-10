"""tracker_scale -- at what spatial scale do two trackings of one movie agree on the strain pattern?

Per-node local affine over a (2w+1)^2-node window on the 137-lattice (w nodes ~ 15 px each), for
both trackings, on the top-left 80x80 overlap at the fit beat's peak; the correlation of the
shortening maps and the axis agreement between trackings, per window size. The scale where the
agreement becomes high is the finest scale at which a parameter map can be claimed from this
recording -- whatever the model says.
"""
import json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import recording as R
RT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data"
rec = R.load(); pos = rec["pos"]
h = np.load(f"{RT}/healthy.npy").astype(np.float32); T2 = h.shape[0]
p2 = torch.as_tensor(0.15 + 0.7 * h.reshape(T2, -1, 2) / 2048.0)
win = R.beat_window(rec, 3); lo, hi = win["span"]
ref1 = win["ref"]; ref2 = p2[hi - R.REST_TAIL:hi].median(0).values
A1w, _ = R.window_affine(rec, win); pk = int(R.shortening(A1w).mean(1).argmax()); t = win["frames"][pk]
# displacement fields on the 80x80 overlap, as [80,80,2]
d1 = (pos[t] - ref1).reshape(137, 137, 2)[:80, :80].numpy()
d2 = (p2[t] - ref2).reshape(80, 80, 2).numpy()
X1 = ref1.reshape(137, 137, 2)[:80, :80].numpy(); X2 = ref2.reshape(80, 80, 2).numpy()

def local_affine(d, X, w):
    """Least-squares gradient of d over a (2w+1)^2 window around each interior node."""
    n = d.shape[0]; out = np.full((n, n, 2, 2), np.nan)
    for i in range(w, n - w):
        for j in range(w, n - w):
            dX = (X[i-w:i+w+1, j-w:j+w+1] - X[i, j]).reshape(-1, 2)
            dd = (d[i-w:i+w+1, j-w:j+w+1] - d[i, j]).reshape(-1, 2)
            G, *_ = np.linalg.lstsq(dX, dd, rcond=None)     # dd ~ dX @ G  -> grad = G.T
            out[i, j] = G.T
    return out

def short_axis(E):
    S = 0.5 * (E + np.swapaxes(E, -1, -2)); w_, v = np.linalg.eigh(S)
    return -w_[..., 0], np.arctan2(v[..., 1, 0], v[..., 0, 0])

rows = []
for w in (1, 2, 3, 5, 8, 12):
    E1, E2 = local_affine(d1, X1, w), local_affine(d2, X2, w)
    m = ~np.isnan(E1[..., 0, 0]) & ~np.isnan(E2[..., 0, 0])
    s1, a1 = short_axis(E1[m]); s2, a2 = short_axis(E2[m])
    wt = np.clip(s1, 0, None)
    corr = np.corrcoef(s1, s2)[0, 1]; agree = (wt * np.cos(2 * (a1 - a2))).sum() / wt.sum()
    cxx = np.corrcoef(E1[m][:, 0, 0], E2[m][:, 0, 0])[0, 1]; cyy = np.corrcoef(E1[m][:, 1, 1], E2[m][:, 1, 1])[0, 1]
    cxy = np.corrcoef(E1[m][:, 0, 1] + E1[m][:, 1, 0], E2[m][:, 0, 1] + E2[m][:, 1, 0])[0, 1]
    rows.append(dict(w=w, window_px=int((2 * w + 1) * 15), n=int(m.sum()), shortening_corr=float(corr),
                     axis_agreement=float(agree), corr_xx=float(cxx), corr_yy=float(cyy), corr_shear=float(cxy),
                     median_short_1=float(np.median(s1)), median_short_2=float(np.median(s2))))
    print(f"  window {2*w+1:2d} nodes = {(2*w+1)*15:4d} px ({(2*w+1)*15*0.325:4.0f} um): shortening corr {corr:+.3f}  "
          f"axis agreement {agree:+.3f}  xx {cxx:+.3f} yy {cyy:+.3f} shear {cxy:+.3f}  "
          f"median shortening {np.median(s1):.4f} / {np.median(s2):.4f}", flush=True)
json.dump(rows, open(os.path.join(HERE, "out", "tracker_scale.json"), "w"), indent=1)
