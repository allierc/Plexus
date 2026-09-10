"""rank_ceiling -- the best any model with k fixed spatial patterns can do on held-out beats (PLAN_R2 3b).

The recording's per-cell affine maps on beat 3, E(t) = A_j(t) - I over interior cells, are
factorised by SVD into k spatial patterns B_k and k time courses a_k(t). On a held-out beat the
patterns are kept and EITHER (i) the time courses are kept too, only a whole-frame shift being
allowed (the exact analogue of the mechanical model's freedom), OR (ii) the time courses are refitted
by least squares (k x T numbers per beat: the loosest fair ceiling). R^2 of both, k = 1..4.
If the mechanical model (fixed parameters, one clock) sits near ceiling (i) for k = 1-2, what it
lacks is spatial structure of higher rank; if ceiling (i) itself is far from 0.9, no one-clock model
reaches 0.9 on this observable.
"""
import json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import recording as R
rec = R.load(); C = rec["n_cells"]
inter = np.load(os.path.join(HERE, "out", "fits", "s4_live_r7_p120_d150", "params.npz"))["interior"].astype(bool)
def E_of(beat):
    w = R.beat_window(rec, beat); A, u = R.window_affine(rec, w)
    E = (A - torch.eye(2))[:, inter].reshape(A.shape[0], -1).numpy()          # [T, cells*4]
    return E, w
def r2(pred, ref):
    return 1 - ((pred - ref) ** 2).sum() / ((ref - ref.mean()) ** 2).sum()
E3, w3 = E_of(3); U, s, Vt = np.linalg.svd(E3, full_matrices=False)
rows = []
for k in (1, 2, 3, 4, 6):
    B = Vt[:k]                                     # spatial patterns [k, cells*4]
    a3 = U[:, :k] * s[:k]                          # time courses on beat 3 [T3, k]
    row = dict(k=k, fit_beat_r2=float(r2(a3 @ B, E3)))
    for beat in (1, 2):
        E, w = E_of(beat); T = E.shape[0]
        # (i) fixed time courses, best whole-frame shift within +-3, truncated/padded at the ends
        best = -9
        for sh in range(-3, 4):
            idx = np.clip(np.arange(T) + sh, 0, a3.shape[0] - 1)
            best = max(best, r2(a3[idx] @ B, E))
        # (ii) time courses refitted by least squares on the held-out beat
        a_free = E @ np.linalg.pinv(B)
        row[f"beat{beat}_fixed_time"] = float(best); row[f"beat{beat}_free_time"] = float(r2(a_free @ B, E))
    rows.append(row)
    print(f"  k={k}: fit beat {row['fit_beat_r2']:.3f} | held-out, patterns+time courses fixed (shift only): "
          f"{row['beat1_fixed_time']:.3f} / {row['beat2_fixed_time']:.3f} | patterns fixed, time courses refitted: "
          f"{row['beat1_free_time']:.3f} / {row['beat2_free_time']:.3f}")
json.dump(rows, open(os.path.join(HERE, "out", "rank_ceiling.json"), "w"), indent=1)
