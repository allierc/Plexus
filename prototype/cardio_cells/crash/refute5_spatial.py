"""refute5_spatial.py -- ROUND 5 REFUTATION, part 1.

Round 5's headline rests on injecting a MEASURED F carrying sigma_F = 3.9e-3, drawn INDEPENDENTLY
PER PARTICLE (round5_fit.py:110, `dr(frames[0]["F0"], gm)` on a [Np,2,2] tensor).  With 100
particles per cell, an independent draw averages down by sqrt(100) = 10 inside every cell -- and the
per-cell moduli are exactly what is being estimated.  If the recording's F error is SPATIALLY
CORRELATED over a length comparable to a cell, that averaging does not happen and the effective
noise per cell is up to 10x larger.

Nothing in rounds 1-5 measured the SPATIAL structure of the recording's F noise.  This does, on the
healthy specimen only, with the same estimator real_F_check.py used for its amplitude: the second
TIME difference over the quiet stretch (frames 30..49), which annihilates any smooth signal and
leaves 6 sigma^2 of noise.  Its spatial autocorrelation is the missing number.

usage: /workspace/.conda_envs/neural-graph-linux/bin/python refute5_spatial.py
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = ("/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data/Cardio_1/"
        "0_B_15kPa_1_MMStack_Pos0.ome.tif.derivatives.npy")
QUIET = (30, 49)
FIT = (152, 201)


def main():
    D = np.load(PATH, mmap_mode="r")
    X, Y = np.asarray(D[0, :, :, 0], float), np.asarray(D[0, :, :, 1], float)
    hx = float(np.median(np.diff(X[0, :])))
    Gt = np.asarray(D[:, :, :, 2:6], float)
    u = np.asarray(D[:, :, :, 0], float) - X[None]
    v = np.asarray(D[:, :, :, 1], float) - Y[None]
    amp = np.sqrt(u ** 2 + v ** 2)
    A = amp.max(0)
    M = A > 0.2 * np.percentile(A, 99)

    out = {"path": PATH, "grid_px": hx, "n_mask": int(M.sum())}

    # ---- the noise field: second TIME difference over the quiet stretch ----------------------- #
    q0, q1 = QUIET
    d2 = Gt[q0 + 1:q1 + 1] - 2 * Gt[q0:q1] + Gt[q0 - 1:q1 - 1]          # [nt,137,137,4]
    sig = float(np.sqrt((np.linalg.norm(d2, axis=-1)[:, M] ** 2).mean() / 6.0))
    out["sigma_F_frobenius_per_node"] = sig
    out["sigma_F_per_component"] = sig / 2.0

    # ---- SPATIAL autocorrelation of that noise field, per lag, masked ------------------------- #
    def acorr(field, lag, axis):
        """Pearson r between the field and itself shifted by `lag` nodes along `axis`."""
        if axis == 0:
            a, b = field[:, :-lag or None], field[:, lag:]
            ma, mb = M[:-lag or None], M[lag:]
        else:
            a, b = field[:, :, :-lag or None], field[:, :, lag:]
            ma, mb = M[:, :-lag or None], M[:, lag:]
        m = ma & mb
        a = a[:, m].reshape(-1)
        b = b[:, m].reshape(-1)
        a = a - a.mean()
        b = b - b.mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

    lags = [1, 2, 3, 4, 6, 8, 12]
    out["spatial_acorr_of_noise"] = {}
    for ci, cname in enumerate(("dudx", "dudy", "dvdx", "dvdy")):
        f = d2[..., ci]
        out["spatial_acorr_of_noise"][cname] = {
            "along_axis0": {str(L): acorr(f, L, 0) for L in lags},
            "along_axis1": {str(L): acorr(f, L, 1) for L in lags}}
    # pooled over the four channels (isotropic summary)
    pooled = {}
    for L in lags:
        vals = [acorr(d2[..., ci], L, ax) for ci in range(4) for ax in (0, 1)]
        pooled[str(L)] = float(np.mean(vals))
    out["spatial_acorr_pooled"] = pooled

    # ---- CONTROL: the same statistic on synthetic white noise of the same shape --------------- #
    rng = np.random.default_rng(0)
    w = rng.standard_normal(d2.shape)
    out["control_white_acorr_pooled"] = {
        str(L): float(np.mean([acorr(w[..., ci], L, ax) for ci in range(4) for ax in (0, 1)]))
        for L in lags}

    # ---- CONTROL 2: a second-time-difference of white noise is MA(2) in time but white in space,
    #      so any spatial structure below is not an artefact of the differencing.
    # ---- the number that matters: variance reduction when averaging a KxK block --------------- #
    def block_var_ratio(field, k):
        """var(mean over a kxk block) / (var / k^2).  1.0 = independent, k^2 = fully correlated."""
        nt, ny, nx, _ = field.shape
        ny2, nx2 = (ny // k) * k, (nx // k) * k
        f = field[:, :ny2, :nx2].reshape(nt, ny2 // k, k, nx2 // k, k, 4)
        mm = M[:ny2, :nx2].reshape(ny2 // k, k, nx2 // k, k)
        full = mm.all(axis=(1, 3))
        bm = f.mean(axis=(2, 4))                        # [nt, ny/k, nx/k, 4]
        vb = float(bm[:, full].var())
        v1 = float(field[:, M].var())
        return vb / (v1 / (k * k))

    out["block_variance_ratio"] = {str(k): block_var_ratio(d2, k) for k in (2, 3, 4, 6)}
    out["block_variance_ratio_white"] = {str(k): block_var_ratio(w, k) for k in (2, 3, 4, 6)}

    # ---- how many nodes does one cell cover? -------------------------------------------------- #
    # 472 cells over the masked field; the sim gives each cell 100 particles with INDEPENDENT F.
    n_nodes_per_cell = float(M.sum()) / 472.0
    keff = {}
    for k in (2, 3, 4, 6):
        r = out["block_variance_ratio"][str(k)]
        keff[str(k)] = (k * k) / max(r, 1e-9)           # effective independent samples in a kxk box
    out["cell_geometry"] = {
        "n_masked_nodes": int(M.sum()), "n_cells_real": 472,
        "nodes_per_cell": n_nodes_per_cell,
        "block_side_matching_a_cell": float(np.sqrt(n_nodes_per_cell)),
        "effective_independent_samples_in_kxk": keff}

    # ---- also: is the noise correlated ACROSS the four channels at a node? -------------------- #
    cc = np.corrcoef(np.stack([d2[..., ci][:, M].reshape(-1) for ci in range(4)]))
    out["channel_corr_of_noise"] = cc.tolist()

    json.dump(out, open(os.path.join(HERE, "refute5_spatial.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
