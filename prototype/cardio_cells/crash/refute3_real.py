"""refute3_real.py -- audit the REAL-data numbers round 3 used to declare the recording unusable.

Round 3, part 3, rests on three numbers from real_F_check.py (healthy specimen only):
    sigma_F  = 0.00388     temporal noise on the derivative channels
    dF/frame = 0.00455     the signal it must resolve   => SNR 1.17
    0.0327                 "the recording's own derivative channels and a central difference of
                            its own displacement field disagree by 97% of |F-I|", called
                            "the honest single-node uncertainty on F".

Three things that number could be instead of an error, none of which round 3 tested:
  (a) a channel-ORDER or SCALE convention mismatch (du/dy vs dv/dx, per-pixel vs per-node);
  (b) a RESOLUTION mismatch -- a central difference over a 2h = 30 px baseline is the boxcar
      average of the derivative over that baseline, not its value at the node.  If the channels
      are pointwise, the two MUST differ by the field's curvature over 30 px;
  (c) a STATIC bias vs. white NOISE -- decisive for whether stacking frames helps.

Also measured: the temporal autocorrelation of the F error, which sets whether a multi-frame
estimator averages the error down as 1/sqrt(T).

usage: /workspace/.conda_envs/neural-graph-linux/bin/python refute3_real.py
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = ("/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data/Cardio_1/"
        "0_B_15kPa_1_MMStack_Pos0.ome.tif.derivatives.npy")
FIT = (152, 201)
QUIET = (30, 49)


def stat(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return {"p50": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "mean": float(a.mean())}


def main():
    D = np.load(PATH, mmap_mode="r")
    out = {"path": PATH, "shape": list(D.shape)}
    X = np.asarray(D[0, :, :, 0], float)
    Y = np.asarray(D[0, :, :, 1], float)
    hx = float(np.median(np.diff(X[0, :])))
    hy = float(np.median(np.diff(Y[:, 0])))
    out["h_px"] = [hx, hy]

    Gt = np.asarray(D[:, :, :, 2:6], float)                 # [T,137,137,4]
    u = np.asarray(D[:, :, :, 0], float) - X[None]
    v = np.asarray(D[:, :, :, 1], float) - Y[None]
    amp = np.sqrt(u ** 2 + v ** 2)
    M = amp.max(0) > 0.2 * np.percentile(amp.max(0), 99)
    out["n_mask"] = int(M.sum())

    # ---------------------------------------------------------------- central differences ------
    def cd(f):
        gx = np.zeros_like(f)
        gy = np.zeros_like(f)
        gx[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * hx)
        gy[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2 * hy)
        return gx, gy

    dudx, dudy = cd(u)
    dvdx, dvdy = cd(v)
    FD = np.stack([dudx, dudy, dvdx, dvdy], -1)
    Mi = M.copy()
    Mi[0, :] = Mi[-1, :] = Mi[:, 0] = Mi[:, -1] = False      # central diff undefined on the rim

    ts = np.arange(FIT[0], FIT[1])
    ch = Gt[ts][:, Mi]                                       # [nt, n, 4]
    fdv = FD[ts][:, Mi]

    # ---- (a) channel assignment: correlate every channel with every central difference ---------
    names_ch = ["c2", "c3", "c4", "c5"]
    names_fd = ["du/dx", "du/dy", "dv/dx", "dv/dy"]
    Cmat = np.zeros((4, 4))
    Smat = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            a, b = ch[..., i].ravel(), fdv[..., j].ravel()
            Cmat[i, j] = float(np.corrcoef(a, b)[0, 1])
            Smat[i, j] = float((a @ b) / (a @ a))             # fd ~ Smat * chan
    out["channel_audit"] = {
        "corr_channel_vs_centraldiff": {names_ch[i]: {names_fd[j]: round(Cmat[i, j], 4)
                                                      for j in range(4)} for i in range(4)},
        "ls_scale_fd_over_chan_diagonal": {names_ch[i]: round(Smat[i, i], 4) for i in range(4)},
        "best_match_per_channel": {names_ch[i]: names_fd[int(np.argmax(np.abs(Cmat[i])))]
                                   for i in range(4)},
        "assumed_order_is_argmax": bool(all(int(np.argmax(np.abs(Cmat[i]))) == i for i in range(4)))}

    # ---- (b) resolution: the central difference is the BOXCAR AVERAGE of the derivative --------
    # CD_i = (1/2h) int_{x_{i-1}}^{x_{i+1}} f' dx = average of f' over a 2h window.  Trapezoid on
    # the sample grid: (1/4, 1/2, 1/4).  Smooth the CHANNELS the same way before comparing.
    def smooth_along(a, axis):
        k = np.zeros_like(a)
        sl = [slice(None)] * a.ndim
        s0, s1, s2 = sl.copy(), sl.copy(), sl.copy()
        s0[axis] = slice(0, -2)
        s1[axis] = slice(1, -1)
        s2[axis] = slice(2, None)
        mid = list(sl)
        mid[axis] = slice(1, -1)
        k[tuple(mid)] = 0.25 * a[tuple(s0)] + 0.5 * a[tuple(s1)] + 0.25 * a[tuple(s2)]
        return k

    # channel j is differentiated along x (axis 2) for j in {0,2}, along y (axis 1) for {1,3}
    Gs = np.stack([smooth_along(Gt[..., 0], 2), smooth_along(Gt[..., 1], 1),
                   smooth_along(Gt[..., 2], 2), smooth_along(Gt[..., 3], 1)], -1)
    chs = Gs[ts][:, Mi]
    d_raw = np.linalg.norm(ch - fdv, axis=-1)
    d_sm = np.linalg.norm(chs - fdv, axis=-1)
    ref = np.linalg.norm(ch, axis=-1)
    out["resolution_test"] = {
        "disagreement_raw": stat(d_raw),
        "disagreement_after_matching_the_boxcar": stat(d_sm),
        "reduction_factor_p50": float(np.median(d_raw) / max(np.median(d_sm), 1e-12)),
        "relative_raw": stat(d_raw / (ref + 1e-9)),
        "relative_after": stat(d_sm / (ref + 1e-9)),
        "note": "if the 0.0327 were measurement error it would not care about matching the "
                "smoothing of the two estimators"}

    # ---- (c) is the disagreement STATIC (bias) or fluctuating (noise)? -------------------------
    d = ch - fdv                                            # [nt, n, 4]
    mu = d.mean(0)
    fl = d - mu[None]
    p_static = float((mu ** 2).sum() / (d ** 2).sum())
    out["bias_vs_noise"] = {
        "static_fraction_of_disagreement_power": p_static,
        "rms_total": float(np.sqrt((d ** 2).mean())),
        "rms_static": float(np.sqrt((mu ** 2).mean())),
        "rms_fluctuating": float(np.sqrt((fl ** 2).mean()))}

    # ---- (d) temporal autocorrelation of the F error in the quiet stretch ----------------------
    q = Gt[QUIET[0]:QUIET[1] + 1][:, M]                      # [nq, n, 4]
    t = np.arange(q.shape[0], dtype=float)
    Vd = np.stack([np.ones_like(t), t, t ** 2, t ** 3], 1)   # cubic detrend per node
    coef, *_ = np.linalg.lstsq(Vd, q.reshape(q.shape[0], -1), rcond=None)
    r = (q.reshape(q.shape[0], -1) - Vd @ coef)
    ac = []
    for lag in range(0, 6):
        a = r[:r.shape[0] - lag]
        b = r[lag:]
        ac.append(float((a * b).mean() / (r * r).mean()))
    out["temporal_autocorr_of_F_residual_quiet"] = {
        "lags_0_to_5": [round(x, 4) for x in ac],
        "sigma_F_from_residual": float(np.sqrt((r ** 2).mean() * 4)),   # 4 components per node
        "note": "lag-1 near 0 => the F error is temporally white => stacking T frames averages it "
                "down as 1/sqrt(T).  lag-1 near 1 => it is a static bias and stacking does not help"}

    # the same, on the fluctuating part of the channel-vs-CD disagreement
    fl2 = fl.reshape(fl.shape[0], -1)
    ac2 = []
    for lag in range(0, 6):
        a = fl2[:fl2.shape[0] - lag]
        b = fl2[lag:]
        ac2.append(float((a * b).mean() / (fl2 * fl2).mean()))
    out["temporal_autocorr_of_disagreement_fluctuation"] = [round(x, 4) for x in ac2]

    # ---- (e) how many independent frames does a beat actually offer? ---------------------------
    d1 = np.linalg.norm(Gt[FIT[0] + 1:FIT[1] + 1] - Gt[FIT[0]:FIT[1]], axis=-1)[:, M]
    out["per_frame_signal"] = {"median_dF": float(np.median(d1)),
                               "p90_dF": float(np.percentile(d1, 90)),
                               "n_frames_in_fit_beat": int(FIT[1] - FIT[0])}

    json.dump(out, open(os.path.join(HERE, "refute3_real.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
