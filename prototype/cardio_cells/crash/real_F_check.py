"""real_F_check.py -- what the REAL recording's measured deformation gradient is worth.

finject.py showed, on synthetic data, that injecting F across the substeps of a frame turns the
frame-cadence fit from med|dE/E| 0.257 into 0.008 -- and that a LINEAR INTERPOLATION between the
two measured frames (F_lerp) is as good as the substep-resolved oracle. That result rests on two
properties of the measurement which are properties of the RECORDING, not of the method:

  (1) F must not curve much within a frame, or the interpolation is wrong;
  (2) F must be measured accurately, because the recovery amplifies error ~2000x.

Both are measured here, on the HEALTHY specimen only (Cardio_1/0_B_15kPa...). The diseased
specimen is sealed and is not opened.

Channels of `...derivatives.npy` [T, 137, 137, 12]: 0,1 = X,Y of the 15-px PIV grid; 2..5 =
du/dx, du/dy, dv/dx, dv/dy -- so F = I + grad u is READ, not integrated.

usage: /workspace/.conda_envs/neural-graph-linux/bin/python real_F_check.py
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = ("/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data/Cardio_1/"
        "0_B_15kPa_1_MMStack_Pos0.ome.tif.derivatives.npy")
FIT = (152, 201)                      # split.json's fit beat
QUIET = (30, 49)                      # a diastolic stretch inside beat 0


def main():
    D = np.load(PATH, mmap_mode="r")
    T = D.shape[0]
    out = {"path": PATH, "shape": list(D.shape)}
    X, Y = np.asarray(D[0, :, :, 0], float), np.asarray(D[0, :, :, 1], float)
    hx = float(np.median(np.diff(X[0, :]))) or float(np.median(np.diff(X[:, 0])))
    hy = float(np.median(np.diff(Y[:, 0]))) or float(np.median(np.diff(Y[0, :])))
    out["grid"] = {"dx_px_axis1": float(np.median(np.diff(X[0, :]))),
                   "dy_px_axis0": float(np.median(np.diff(Y[:, 0]))),
                   "dx_px_axis0": float(np.median(np.diff(X[:, 0]))),
                   "dy_px_axis1": float(np.median(np.diff(Y[0, :])))}

    Gt = np.asarray(D[:, :, :, 2:6], float)                     # [T,137,137,4] = grad u
    u = np.asarray(D[:, :, :, 0], float) - X[None]
    v = np.asarray(D[:, :, :, 1], float) - Y[None]
    amp = np.sqrt(u ** 2 + v ** 2)
    out["displacement_px"] = {"p50": float(np.median(amp)), "p99": float(np.percentile(amp, 99)),
                              "max": float(amp.max())}
    gn = np.linalg.norm(Gt, axis=-1)                            # |F - I| per node per frame
    out["gradu_norm"] = {"p50": float(np.median(gn)), "p99": float(np.percentile(gn, 99)),
                         "max": float(gn.max())}

    # -- the eval mask: only nodes that actually move (split.json's frozen rule) -----------------
    A = amp.max(0)
    thr = 0.2 * np.percentile(A, 99)
    M = A > thr
    out["eval_mask"] = {"n_nodes": int(M.sum()), "n_total": int(M.size), "threshold_px": float(thr)}

    def stat(a):
        a = a[np.isfinite(a)]
        return {"p50": float(np.median(a)), "p90": float(np.percentile(a, 90)),
                "mean": float(a.mean())}

    # -- (1) does F curve within a frame? --------------------------------------------------------
    # the linear interpolation F_lerp neglects exactly the SECOND time difference, so its error
    # over one frame is bounded by |F(t+1) - 2F(t) + F(t-1)| / 8 (the max of the quadratic bulge).
    res = {}
    for tag, (t0, t1) in (("fit_beat", FIT), ("whole_recording", (1, T - 1))):
        s = slice(t0, t1)
        d1 = Gt[t0 + 1:t1 + 1] - Gt[t0:t1]                      # frame-to-frame change of F
        d2 = Gt[t0 + 1:t1 + 1] - 2 * Gt[t0:t1] + Gt[t0 - 1:t1 - 1]
        nF = np.linalg.norm(Gt[s], axis=-1)
        res[tag] = {
            "frame_to_frame_dF_over_F": stat(np.linalg.norm(d1, axis=-1)[..., M[None] if False else slice(None)][:, M]
                                             / (nF[:, M] + 1e-9)),
            "abs_dF_per_frame": stat(np.linalg.norm(d1, axis=-1)[:, M]),
            "abs_d2F_per_frame": stat(np.linalg.norm(d2, axis=-1)[:, M]),
            "lerp_bulge_over_dF": stat(np.linalg.norm(d2, axis=-1)[:, M] / 8.0
                                       / (np.linalg.norm(d1, axis=-1)[:, M] + 1e-12))}
    out["within_frame_curvature"] = res

    # -- (2) how noisy is the measured F? --------------------------------------------------------
    # (a) internal consistency: the measured grad u against a central difference of the measured u
    #     on the same grid. Two estimates of the same quantity; their difference bounds the error
    #     of at least one of them.
    ts = list(range(FIT[0], FIT[1], 6))
    dudx = (u[:, :, 2:] - u[:, :, :-2]) / (2 * hx)
    dudy = (u[:, 2:, :] - u[:, :-2, :]) / (2 * hy)
    dvdx = (v[:, :, 2:] - v[:, :, :-2]) / (2 * hx)
    dvdy = (v[:, 2:, :] - v[:, :-2, :]) / (2 * hy)
    Mi = M[1:-1, 1:-1]
    fd = np.stack([dudx[:, 1:-1, :], dudy[:, :, 1:-1], dvdx[:, 1:-1, :], dvdy[:, :, 1:-1]], -1)
    me = Gt[:, 1:-1, 1:-1, :]
    dif = np.linalg.norm((fd - me)[ts][:, Mi], axis=-1)
    ref = np.linalg.norm(me[ts][:, Mi], axis=-1)
    out["consistency_measured_vs_finite_difference"] = {
        "abs": stat(dif), "relative": stat(dif / (ref + 1e-9)),
        "note": "central difference of the measured displacement on the 15-px grid vs the "
                "recording's own derivative channels; a lower bound on |dF| of the two together"}

    # (b) temporal roughness in a QUIET stretch: with F nearly constant there, the second time
    #     difference is 6 sigma^2 in variance, so sigma_F ~ rms(d2)/sqrt(6).
    q0, q1 = QUIET
    d2q = Gt[q0 + 1:q1 + 1] - 2 * Gt[q0:q1] + Gt[q0 - 1:q1 - 1]
    sig = float(np.sqrt((np.linalg.norm(d2q, axis=-1)[:, M] ** 2).mean() / 6.0))
    out["noise_sigma_F_from_quiet_stretch"] = {
        "frames": [q0, q1], "sigma_F": sig,
        "sigma_F_over_p99_gradu": sig / float(np.percentile(gn, 99)),
        "note": "|F| here is the Frobenius norm of the 2x2 grad u, so sigma_F is per-node, "
                "all four components together"}

    # (c) how big is the signal it has to resolve?  the per-frame change of F, over the noise
    d1f = np.linalg.norm(Gt[FIT[0] + 1:FIT[1] + 1] - Gt[FIT[0]:FIT[1]], axis=-1)[:, M]
    out["signal_to_noise"] = {"median_dF_per_frame_over_sigma_F": float(np.median(d1f) / sig),
                              "p90_dF_per_frame_over_sigma_F": float(np.percentile(d1f, 90) / sig)}

    json.dump(out, open(os.path.join(HERE, "real_F_check.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
