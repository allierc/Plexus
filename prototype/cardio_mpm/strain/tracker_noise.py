"""tracker_noise -- how much of a per-cell affine map is the tracker's, not the tissue's.

VERDICT (2026-09-09, patch_check.py): the two trackings disagree on the spatial pattern of the
beat at every scale (per-cell shortening r = -0.06, node displacement r = 0.28/-0.03), yet EACH
repeats its own pattern beat to beat at r = 0.99. Normalised cross-correlation of 49 px patches on
the raw frames settles it: the pixels move as TRACKING 1 says (r = 0.97 in x and y, 0.6 px median
error on 3 px of motion) and not as `healthy.npy` says (r = 0.25 / -0.10). So `healthy.npy` is a
broken tracking, its "tracker floor" (the 0.27 spatial agreement that shaped the old loop's
premises) is that tracker's failure and not a property of the recording, and `noise_tracker.npz`
is NOT a noise model -- it is not written. `noise_rest.npz` (tracking 1's own rest frames) is the
noise model S3 uses.

Two measurements of the noise a fit must survive, both from data, both written as per-cell,
per-frame perturbations (dA [T,C,2,2], du [T,C,2]) that fit.py adds to a planted target:

  rest     the recording's own rest frames, relative to the rest reference of their beat: the
           tissue is still, so every per-cell A - I and u there is measurement. Available for all
           472 cells and ~100 frames; a window's worth is drawn per seed.
  tracker  the healthy sheet was tracked twice (the 137-grid derivatives and `healthy.npy`, an
           80x80 grid over the top-left 1185 px). Per-cell affine from each, on the cells both
           cover, over the fit beat; the difference is the tracker-vs-tracker disagreement, the
           harsher number (it includes systematic differences during motion). Cells outside the
           overlap borrow a covered cell's difference at random.

Also printed: the agreement between the two trackings at the peak -- shortening correlation and
axis agreement over the covered cells -- which is the CEILING for any per-cell claim.
"""
import json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import recording as R
RT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data"

rec = R.load(); C = rec["n_cells"]; pos = rec["pos"]; lab = rec["labels"]
rng = np.random.default_rng(0)

# ---- rest noise ----------------------------------------------------------------------------
dA_rest, du_rest = [], []
for k in range(4):
    w = R.beat_window(rec, k)
    lo, hi = w["span"]
    for t in range(w["onset"] + 26, hi + 1):            # after relaxation, up to the window's end
        A, u = R.cell_affine(pos[t], w["ref"], lab, C)
        dA_rest.append(A - torch.eye(2)); du_rest.append(u)
dA_rest, du_rest = torch.stack(dA_rest), torch.stack(du_rest)
print(f"rest frames: {dA_rest.shape[0]}; per-cell |A-I| rms {dA_rest.pow(2).mean().sqrt():.5f}, "
      f"|u| rms {du_rest.pow(2).mean().sqrt():.6f}")

# ---- the second tracking -------------------------------------------------------------------
h = np.load(f"{RT}/healthy.npy").astype(np.float32)          # [240,80,80,2] px, positions
T2 = h.shape[0]
p2 = torch.as_tensor(0.15 + 0.7 * h.reshape(T2, -1, 2) / 2048.0)    # world
# each 80-grid node takes the label of the nearest 137-grid node at rest
from scipy.spatial import cKDTree
idx = cKDTree(pos[0].numpy()).query(p2[0].numpy())[1]
lab2 = lab[idx]
cnt2 = torch.bincount(lab2, minlength=C + 1)[1:]
cnt1 = torch.bincount(lab, minlength=C + 1)[1:]
covered = (cnt2 >= 0.6 * cnt1 / (137 * 137 / (80 * 80)) * 0) & (cnt2 >= 8)   # enough nodes in both
print(f"second tracking: {p2.shape[1]} nodes over {int(covered.sum())} covered cells (>=8 nodes)")
win = R.beat_window(rec, 3)
A1, u1 = R.window_affine(rec, win)
ref2 = p2[win["span"][1] - R.REST_TAIL:win["span"][1]].median(0).values
A2s, u2s = [], []
for t in win["frames"]:
    A, u = R.cell_affine(p2[t], ref2, lab2, C); A2s.append(A); u2s.append(u)
A2, u2 = torch.stack(A2s), torch.stack(u2s)
s1, s2 = R.shortening(A1), R.shortening(A2)
pk = int(s1.mean(1).argmax())
cv = covered
corr = float(torch.corrcoef(torch.stack([s1[pk][cv], s2[pk][cv]]))[0, 1])
eye = torch.eye(2)
def axis(A):
    E = A - eye; S = 0.5 * (E + E.transpose(-1, -2)); w, v = torch.linalg.eigh(S)
    return torch.atan2(v[:, 1, 0], v[:, 0, 0]), (-w[:, 0]).clamp(min=0)
a1, w1 = axis(A1[pk]); a2, _ = axis(A2[pk])
agree = float((w1[cv] * torch.cos(2 * (a1 - a2)[cv])).sum() / w1[cv].sum())
tc = float(torch.corrcoef(torch.stack([s1.mean(1), s2.mean(1)]))[0, 1])
print(f"two trackings at the peak (frame {pk}), {int(cv.sum())} cells: shortening corr {corr:.3f}, "
      f"axis agreement {agree:.3f}, median shortening {s1[pk][cv].median():.4f} vs {s2[pk][cv].median():.4f}; "
      f"mean-curve corr over the window {tc:.4f}")
# per-component correlations
for nm, (i, j) in dict(xx=(0, 0), xy=(0, 1), yy=(1, 1)).items():
    c_ = float(torch.corrcoef(torch.stack([A1[pk][cv, i, j], A2[pk][cv, i, j]]))[0, 1])
    print(f"   A_{nm} corr {c_:.3f}")
uc = float(torch.corrcoef(torch.stack([u1[pk][cv].reshape(-1), u2[pk][cv].reshape(-1)]))[0, 1])
print(f"   centroid displacement corr {uc:.3f}")

# ---- the tracker noise field: difference on covered cells, borrowed elsewhere ----------------
dA = (A2 - A1); du = (u2 - u1)
cov_idx = torch.nonzero(cv).squeeze(1).numpy()
borrow = torch.as_tensor(rng.choice(cov_idx, size=C))
full_dA = torch.where(cv[None, :, None, None], dA, dA[:, borrow])
full_du = torch.where(cv[None, :, None], du, du[:, borrow])
print(f"tracker difference: |dA| rms {dA[:, cv].pow(2).mean().sqrt():.5f} (signal |A-I| rms at peak "
      f"{(A1[pk][cv]-eye).pow(2).mean().sqrt():.5f}), |du| rms {du[:, cv].pow(2).mean().sqrt():.6f}")
# noise_tracker.npz is deliberately NOT written: see the verdict in the docstring.
# a window's worth of rest noise: consecutive-ish frames drawn per cell independently would break the
# spatial structure, so draw FRAMES (all cells together) with replacement
sel = rng.choice(dA_rest.shape[0], size=len(win["frames"]))
np.savez(os.path.join(HERE, "data", "noise_rest.npz"), dA=dA_rest[sel].numpy(), du=du_rest[sel].numpy())
json.dump(dict(rest_frames=int(dA_rest.shape[0]), rest_A_rms=float(dA_rest.pow(2).mean().sqrt()),
               rest_u_rms=float(du_rest.pow(2).mean().sqrt()), covered_cells=int(cv.sum()),
               peak_frame=pk, shortening_corr=corr, axis_agreement=agree, curve_corr=tc,
               displacement_corr=uc, tracker_dA_rms=float(dA[:, cv].pow(2).mean().sqrt()),
               tracker_du_rms=float(du[:, cv].pow(2).mean().sqrt()),
               signal_A_rms_peak=float((A1[pk][cv]-eye).pow(2).mean().sqrt())),
          open(os.path.join(HERE, "out", "tracker_noise.json"), "w"), indent=1)
