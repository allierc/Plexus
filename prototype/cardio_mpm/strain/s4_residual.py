"""s4_residual -- what the fitted model does NOT explain, which is the loop's next proposal.

Rolls the fitted parameters out on the fit beat and decomposes the unexplained variance of the
per-cell affine maps by TIME (which phase of the beat), by COMPONENT (area change, shear along
the axes, shear across, rotation; centroid displacement x/y) and by CELL (a map of per-cell R^2,
and its correlation with the cell's fitted g, its size and its distance from the sheet's edge).
Also reports what the fit did to the parameters: the distribution of g, phi and E, how far E moved
from uniform, and how the fitted g compares with the amplitude read directly off the recording.

    python s4_residual.py --params out/fits/s4_live_s0/params.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M      # noqa: E402
import recording as R  # noqa: E402


def r2(pred, ref):
    return float(1.0 - ((pred - ref) ** 2).sum() / ((ref - ref.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--per-parent", type=int, default=50)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--anchor", type=float, default=1e4)
    ap.add_argument("--band", type=float, default=0.03)
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--beat", type=int, default=3)
    ap.add_argument("--specimen", default="healthy", choices=["healthy", "hcm"])
    args = ap.parse_args()
    dev = args.device
    rec = R.load(device=dev, specimen=args.specimen); C = rec["n_cells"]
    LTIF = os.path.join(HERE, "data_hcm" if rec.get("specimen") == "hcm" else "data", "cells_2560.tif")
    z = np.load(args.params)
    mode = str(z["clock_mode"]) if "clock_mode" in z.files else "sigmoid"
    P = M.Params(C, dev, nu=args.nu, clock_mode=mode, n_frames=int(z["clock"].shape[0]) if mode == "free" else 0)
    P.load({k: z[k] for k in ("g", "phi", "logE", "clock", "delay") if k in z.files})
    win = R.beat_window(rec, args.beat); T = len(win["frames"])
    A_ref, u_ref = R.window_affine(rec, win)
    kw = dict(n_grid=args.n_grid, per_parent=args.per_parent, n_frames=T - 1, n_cells=C, anchor_k=args.anchor)
    with torch.no_grad():
        r0 = M.rollout(M.load_sim(M.build_spec(label_tif=LTIF, differentiable=False, name="res_rest", **dict(kw, n_frames=0))),
                       P, dev, C, grad=False)
        X0, cid = r0["X0"], r0["cid"]
        band = ((X0[:, 0] < M.DOM_LO + args.band) | (X0[:, 0] > M.DOM_HI - args.band)
                | (X0[:, 1] < M.DOM_LO + args.band) | (X0[:, 1] > M.DOM_HI - args.band))
        out = M.rollout(M.load_sim(M.build_spec(label_tif=LTIF, differentiable=False, name="res", **kw)), P, dev, C,
                        grad=False, prescribe=(band, M.band_prescription(A_ref, u_ref, X0, cid, band)))
    A, u = out["A"], out["u"]
    eye = torch.eye(2, device=dev)
    E, Er = A - eye, A_ref - eye

    # ---- by component, in a physically meaningful basis ------------------------------------
    def parts(E):
        S = 0.5 * (E + E.transpose(-1, -2)); W = 0.5 * (E - E.transpose(-1, -2))
        return dict(area=S[..., 0, 0] + S[..., 1, 1], shear_axes=S[..., 0, 0] - S[..., 1, 1],
                    shear_diag=2 * S[..., 0, 1], rotation=W[..., 0, 1])
    pm, pr = parts(E), parts(Er)
    by_comp = {k: dict(r2=r2(pm[k], pr[k]), signal_rms=float(pr[k].pow(2).mean().sqrt()),
                       residual_rms=float((pm[k] - pr[k]).pow(2).mean().sqrt())) for k in pm}
    by_comp["u_x"] = dict(r2=r2(u[..., 0], u_ref[..., 0]), signal_rms=float(u_ref[..., 0].pow(2).mean().sqrt()),
                          residual_rms=float((u - u_ref)[..., 0].pow(2).mean().sqrt()))
    by_comp["u_y"] = dict(r2=r2(u[..., 1], u_ref[..., 1]), signal_rms=float(u_ref[..., 1].pow(2).mean().sqrt()),
                          residual_rms=float((u - u_ref)[..., 1].pow(2).mean().sqrt()))
    # ---- by time --------------------------------------------------------------------------
    res_t = ((E - Er) ** 2).mean((1, 2, 3)).sqrt(); sig_t = (Er ** 2).mean((1, 2, 3)).sqrt()
    sm, sr = R.shortening(A).mean(1), R.shortening(A_ref).mean(1)
    by_time = [dict(frame=t, signal_rms=float(sig_t[t]), residual_rms=float(res_t[t]),
                    shortening_model=float(sm[t]), shortening_rec=float(sr[t])) for t in range(T)]
    # ---- by cell --------------------------------------------------------------------------
    ss_res = ((E - Er) ** 2).sum((0, 2, 3)); ss_tot = ((Er - Er.mean()) ** 2).sum((0, 2, 3))
    r2_cell = 1 - ss_res / ss_tot
    ones = torch.ones(X0.shape[0], device=dev); idx = cid - 1
    cnt = torch.zeros(C, device=dev).index_add(0, idx, ones)
    cen = torch.zeros(C, 2, device=dev).index_add(0, idx, X0) / cnt[:, None]
    edge = torch.minimum(torch.minimum(cen[:, 0] - M.DOM_LO, M.DOM_HI - cen[:, 0]),
                         torch.minimum(cen[:, 1] - M.DOM_LO, M.DOM_HI - cen[:, 1]))
    g, E_fit, phi = P.g.detach(), P.logE.detach().exp(), P.phi.detach()
    amp_meas = R.fibre_init(A_ref)[1]
    def corr(a, b):
        return float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    by_cell = dict(r2_median=float(r2_cell.median()), r2_p10=float(r2_cell.quantile(0.1)),
                   r2_p90=float(r2_cell.quantile(0.9)),
                   corr_r2_with=dict(g=corr(r2_cell, g), logE=corr(r2_cell, E_fit.log()),
                                     edge_distance=corr(r2_cell, edge), n_particles=corr(r2_cell, cnt),
                                     measured_amp=corr(r2_cell, amp_meas)),
                   worst_cells=[int(i) + 1 for i in torch.argsort(r2_cell)[:15].tolist()])
    params = dict(g=dict(median=float(g.median()), p10=float(g.quantile(0.1)), p90=float(g.quantile(0.9)),
                         max=float(g.max()), corr_with_measured_amp=corr(g, amp_meas),
                         ratio_to_measured_median=float(g.median() / amp_meas.median())),
                  E=dict(median=float(E_fit.median()), p10=float(E_fit.quantile(0.1)), p90=float(E_fit.quantile(0.9)),
                         log_sd=float(E_fit.log().std()), corr_logE_with_g=corr(E_fit.log(), g),
                         corr_logE_with_edge=corr(E_fit.log(), edge)),
                  phi_change_from_measured_deg=float((torch.minimum((phi - R.fibre_init(A_ref)[0]) % np.pi,
                                                                      np.pi - (phi - R.fibre_init(A_ref)[0]) % np.pi))
                                                     .median() * 180 / np.pi),
                  clock=(dict(t0=float(P.clock[0]), tau_r=float(P.clock[1].exp()), dur=float(P.clock[2].exp()),
                              tau_d=float(P.clock[3].exp())) if mode == "sigmoid"
                         else dict(free=[round(float(v), 4) for v in P.gfree.detach().cpu()])))
    res = dict(params_file=args.params, r2_A=r2(E, Er), r2_u=r2(u, u_ref), by_component=by_comp,
               by_time=by_time, by_cell=by_cell, params=params)
    od = os.path.dirname(args.params)
    json.dump(res, open(os.path.join(od, "residual.json"), "w"), indent=1)
    np.savez(os.path.join(od, "residual.npz"), A=A.cpu().numpy(), u=u.cpu().numpy(), A_ref=A_ref.cpu().numpy(),
             u_ref=u_ref.cpu().numpy(), r2_cell=r2_cell.cpu().numpy(), cen=cen.cpu().numpy())
    print(f"  R2(A) {res['r2_A']:.3f}  R2(u) {res['r2_u']:.3f}")
    print("  by component (R2 | signal rms | residual rms):")
    for k, v in by_comp.items():
        print(f"    {k:<11s} {v['r2']:+.3f} | {v['signal_rms']:.5f} | {v['residual_rms']:.5f}")
    print("  by time (frame: residual/signal rms):", " ".join(f"{b['frame']}:{b['residual_rms']/max(b['signal_rms'],1e-9):.2f}"
                                                              for b in by_time[4:36:2]))
    print(f"  by cell: R2 median {by_cell['r2_median']:.3f} (p10 {by_cell['r2_p10']:.3f}, p90 {by_cell['r2_p90']:.3f}); "
          f"corr of per-cell R2 with {by_cell['corr_r2_with']}")
    print(f"  params: {json.dumps(params)}")


if __name__ == "__main__":
    main()
