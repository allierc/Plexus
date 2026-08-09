"""refute_round1.py -- ROUND 2 part 1: attack round 1's headline.

Three attacks, each with a number:

 A. TARGET MISMATCH.  `System.step` returns  a = (v_end - v_0)/(dt_sub*n_sub)  -- an ENDPOINT
    velocity difference.  `fd_accel` returns  (x_{k+1} - 2x_k + x_{k-1})/dt^2  -- a central second
    difference of positions.  At n_sub = 1 these are the same object (round 1 measured 7.6e-11).
    At n_sub = 10 they are NOT: x_{k+1}-x_k = dt_sub * sum_s v_s is an AVERAGE velocity, not the
    endpoint one.  Round 1 itself recorded fd_vs_solver_rel = 0.336 at frame cadence and then fit
    A (which predicts the endpoint difference) to b (which is the average difference) anyway.
    So the reported 51% frame error is contaminated by a definitional error that has nothing to
    do with the affine-composition problem it was meant to measure.
      A1: b_oracle = a_solver_frame - a0   (the model's own functional; isolates nonlinearity)
      A2: DISPLACEMENT observable  y(theta) = x_end(theta) - x_0, assembled and fitted
          consistently.  This is the functional real data actually delivers.

 B. GLOBAL SCALE.  theta_hat_frame has corr(E_hat, E) = 0.69 but mean E 63.3 against 128.4.  If a
    single per-block scalar repairs the rollout, the frame constraint is not "worse than nothing";
    it is right up to a gauge.

 C. NULLS ROUND 1 DID NOT RUN.  (i) an independent draw from the planting prior; (ii) per-cell
    BLIND constants deliberately mis-scaled to the same motion-energy ratio as theta_hat_frame
    (2.03x) and to a low one.  If a blind constant with the wrong amplitude scores the same as
    theta_hat_frame, then loopscore is reading amplitude, not per-cell parameters.

usage:
  PYTHONPATH=/workspace/Plexus/src python refute_round1.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ALG = "/workspace/Plexus/prototype/cardio_cells/algebraic"
DISC = "/workspace/Plexus/discovery_cardio_mpm"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/workspace/Plexus/src")
sys.path.insert(0, ALG)
sys.path.insert(0, DISC)
sys.path.insert(0, HERE)

from assemble import SUBSTEP_TOKENS, rel                       # noqa: E402
from recover import Solver, fd_accel, install_E, score         # noqa: E402
from plexus.models.entities import _lame                       # noqa: E402
import metrics as MET                                          # noqa: E402
import crash_test as CT                                        # noqa: E402


# --------------------------------------------------------------------------------------------- #
#  the DISPLACEMENT observable: same map, different read-out
# --------------------------------------------------------------------------------------------- #
def step_disp(sy, E_cell, gain_cell, n_sub):
    """Byte-for-byte System.step, except it returns x_end - x_0 instead of (v_end - v_0)/(n dt)."""
    H, p = sy.H, sy.p
    sy.restore()
    mu, la = _lame(E_cell[sy.cid])
    p.mu, p.la = mu, la
    H.zero_delta()
    H._delta["mpm_particle"] = sy.pass0 + gain_cell[sy.cid][:, None] * sy.act0
    H.sub_dt = sy.dt_sub
    for _ in range(n_sub):
        for tok in SUBSTEP_TOKENS:
            sy._tok(tok)
    H.sub_dt = None
    return (p.get("pos") - sy.x0).reshape(-1).clone()


def y_of_theta(sy, theta, n_sub):
    E = torch.zeros(sy.C + 1, device=sy.device, dtype=sy.dtype)
    gn = torch.zeros_like(E)
    E[1:], gn[1:] = theta[:sy.C], theta[sy.C:]
    return step_disp(sy, E, gn, n_sub)


def assemble_disp(sy, n_sub, sE=100.0, sg=1.0):
    t0 = time.time()
    z = torch.zeros(2 * sy.C, device=sy.device, dtype=sy.dtype)
    y0 = y_of_theta(sy, z, n_sub)
    A = torch.zeros(y0.numel(), 2 * sy.C, device=sy.device, dtype=sy.dtype)
    for j in range(2 * sy.C):
        s = sE if j < sy.C else sg
        e = z.clone()
        e[j] = s
        A[:, j] = (y_of_theta(sy, e, n_sub) - y0) / s
    torch.cuda.synchronize()
    return A, y0, time.time() - t0


# --------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="refute1")
    ap.add_argument("--skip-rollouts", action="store_true")
    a = ap.parse_args()

    # EXACTLY round 1's configuration
    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128, warmup=165,
                           window=150, dtype="float64", mode="full", e_lo=40.0, e_hi=220.0,
                           g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args)}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        sy, recA = CT.plant_and_warm(args, log)
        C, W, G = sy.C, args.warmup, args.window
        th = sy.theta_true.double()
        dx, x0 = sy.g.dx, sy.x0.clone()

        # ---- 0. is round 1 reproducible on this device? ------------------------------------
        z = np.load(os.path.join(HERE, "theta_round1.npz"))
        R["repro"] = {"theta_true_max_abs_diff_vs_round1":
                      float(np.abs(z["cand::theta_true"] - th.cpu().numpy()).max())}
        log(f"[repro] theta_true identical to round 1 to "
            f"{R['repro']['theta_true_max_abs_diff_vs_round1']:.3e}")

        # ---- reading surface, identical to round 1 -----------------------------------------
        tracers = {m: CT.tracer_indices(x0, CT.probe_points(m))
                   for m in (MET.MARGIN_SAFE, MET.MARGIN_INHERITED)}
        band = 0.06 / MET.SHEET_SPAN
        anchor = ((x0[:, 0] < band) | (x0[:, 0] > 1 - band) |
                  (x0[:, 1] < band) | (x0[:, 1] > 1 - band))
        interior = ~anchor

        # ---- reference window B -------------------------------------------------------------
        ref_full = torch.zeros(G, sy.Np, 2, device=sy.device, dtype=sy.dtype)
        sy.restore()
        install_E(sy, sy.E_true)
        for k in range(G):
            sy._outer(W + k, gain_cell=sy.gain_true)
            sy.H.sub_dt = sy.dt_sub
            for _ in range(sy.n_sub_per_frame):
                for tok in SUBSTEP_TOKENS:
                    sy._tok(tok)
            sy.H.sub_dt = None
            ref_full[k] = sy.p.get("pos")
        d_ref = ref_full - x0[None]
        dm = d_ref[:, interior].mean(0, keepdim=True)
        ss_tot = (d_ref[:, interior] - dm).pow(2).sum()
        ref_tr = {m: ref_full[:, t] for m, t in tracers.items()}

        # ---- consistency of the frame objects ----------------------------------------------
        x_fprev = recA[-2].clone()
        x_fnext = ref_full[0].clone()
        R["window_A_ends_at_x0"] = float((recA[-1] - x0).abs().max())
        nsf = sy.n_sub_per_frame

        sy.restore()
        a_solver_f = sy.step(sy.E_true, sy.gain_true, n_sub=nsf)
        x_end_true = sy.p.get("pos").clone()
        R["step_frame_reproduces_reference_frame0_max_abs"] = float((x_end_true - x_fnext).abs().max())
        log(f"[consistency] recA[-1] == x0 to {R['window_A_ends_at_x0']:.3e}; "
            f"step(n_sub=10) end position == ref_full[0] to "
            f"{R['step_frame_reproduces_reference_frame0_max_abs']:.3e}")

        a_fdf = fd_accel(x_fprev, x0, x_fnext, sy.dt)
        v0f = sy.v0.reshape(-1)
        a_endpoint = ((sy.p.get("vel") - sy.v0) / sy.dt).reshape(-1)
        R["frame_target_mismatch"] = {
            "fd_vs_solver_rel": rel(a_fdf - a_solver_f, a_solver_f),
            "a_solver_equals_endpoint_diff": rel(a_solver_f - a_endpoint, a_endpoint),
            "note": "fd_accel is (avg v over frame k - avg v over frame k-1)/dt; step() returns "
                    "(v_end - v_0)/dt. They are different functionals; at n_sub=1 they coincide."}
        log(f"[attack A] frame cadence: ||a_fd - a_step||/||a_step|| = "
            f"{R['frame_target_mismatch']['fd_vs_solver_rel']:.4f}  <-- the target round 1 fitted "
            f"was off by this much BEFORE any nonlinearity")

        # ================================================================= A. the frame solves ==
        R["solves"] = {}
        cand = {}

        # -- (a) round 1's own solve, reproduced -----------------------------------------------
        Af, a0f, tf = sy.assemble(n_sub=nsf)
        b_fd = a_fdf - a0f
        b_or = a_solver_f - a0f
        res_or = rel(Af @ sy.theta_true - b_or, b_or)
        res_fd = rel(Af @ sy.theta_true - b_fd, b_fd)
        Sf = Solver(Af, C)
        s_fd = Sf(b_fd)
        s_or = Sf(b_or)
        cand["frame_vel_fd (ROUND 1)"] = s_fd["ridge0"]
        cand["frame_vel_ORACLEb"] = s_or["ridge0"]
        R["solves"]["frame_vel_fd"] = {"residual_Atheta_vs_b": res_fd, "cond": Sf.cond,
                                       "assembly_s": tf,
                                       "scores": {k: score(v, th, C) for k, v in s_fd.items()}}
        R["solves"]["frame_vel_ORACLEb"] = {"residual_Atheta_vs_b": res_or,
                                            "scores": {k: score(v, th, C) for k, v in s_or.items()}}
        d1 = float(np.abs(cand["frame_vel_fd (ROUND 1)"].cpu().numpy()
                          - z["cand::theta_hat_frame_ridge0"]).max())
        R["repro"]["theta_hat_frame_max_abs_diff_vs_round1"] = d1
        log(f"[repro] theta_hat_frame reproduced to {d1:.3e} (abs)")
        for nm in ("frame_vel_fd", "frame_vel_ORACLEb"):
            s = R["solves"][nm]["scores"]["ridge0"]
            log(f"[solve] {nm:<20s} res {R['solves'][nm]['residual_Atheta_vs_b']:.4f}  "
                f"med|dE/E| {s['med_E']:.4f}  med|dg/g| {s['med_gain']:.4f}  l2 {s['rel_l2']:.4f}")
        Sf.free()
        del Af, Sf
        torch.cuda.empty_cache()

        # -- (b) the DISPLACEMENT observable, assembled and fitted consistently ----------------
        Ay, y0, ty = assemble_disp(sy, nsf)
        b_y = (x_fnext - x0).reshape(-1) - y0
        res_y = rel(Ay @ sy.theta_true - b_y, b_y)
        Sy = Solver(Ay, C)
        s_y = Sy(b_y)
        cand["frame_DISP"] = s_y["ridge0"]
        R["solves"]["frame_DISP"] = {"residual_Atheta_vs_b": res_y, "cond": Sy.cond,
                                     "assembly_s": ty,
                                     "scores": {k: score(v, th, C) for k, v in s_y.items()}}
        s = R["solves"]["frame_DISP"]["scores"]["ridge0"]
        log(f"[solve] {'frame_DISP':<20s} res {res_y:.4f}  med|dE/E| {s['med_E']:.4f}  "
            f"med|dg/g| {s['med_gain']:.4f}  l2 {s['rel_l2']:.4f}  cond {Sy.cond:.3e}")
        # and the same observable at SUBSTEP cadence, as the control that it is not the read-out
        Ay1, y01, _ = assemble_disp(sy, 1)
        sy.restore()
        sy.step(sy.E_true, sy.gain_true, n_sub=1)
        x1 = sy.p.get("pos").clone()
        b_y1 = (x1 - x0).reshape(-1) - y01
        Sy1 = Solver(Ay1, C)
        s_y1 = Sy1(b_y1)
        R["solves"]["substep_DISP"] = {
            "residual_Atheta_vs_b": rel(Ay1 @ sy.theta_true - b_y1, b_y1),
            "scores": {k: score(v, th, C) for k, v in s_y1.items()}}
        log(f"[solve] {'substep_DISP':<20s} res "
            f"{R['solves']['substep_DISP']['residual_Atheta_vs_b']:.3e}  med|dE/E| "
            f"{R['solves']['substep_DISP']['scores']['ridge0']['med_E']:.3e}")
        Sy.free(); Sy1.free()
        del Ay, Ay1, Sy, Sy1
        torch.cuda.empty_cache()

        # ================================================================= B. the gauge test ====
        thf = cand["frame_vel_fd (ROUND 1)"]
        kE = float((thf[:C] * th[:C]).sum() / (thf[:C] * thf[:C]).sum())
        kg = float((thf[C:] * th[C:]).sum() / (thf[C:] * thf[C:]).sum())
        resc = torch.cat([thf[:C] * kE, thf[C:] * kg])
        cand["frame_vel_fd_RESCALED(oracle k)"] = resc
        R["gauge"] = {"kE": kE, "kg": kg,
                      "corr_E": float(np.corrcoef(thf[:C].cpu(), th[:C].cpu())[0, 1]),
                      "corr_gain": float(np.corrcoef(thf[C:].cpu(), th[C:].cpu())[0, 1]),
                      "score_rescaled": score(resc, th, C)}
        log(f"[attack B] oracle per-block rescale kE {kE:.3f} kg {kg:.3f} -> med|dE/E| "
            f"{R['gauge']['score_rescaled']['med_E']:.4f} (was "
            f"{R['solves']['frame_vel_fd']['scores']['ridge0']['med_E']:.4f}); "
            f"corr(E_hat,E) {R['gauge']['corr_E']:.3f}")

        # ================================================================= C. missing nulls =====
        gp = torch.Generator(device=str(th.device)).manual_seed(4242)
        pri = torch.cat([40.0 + 180.0 * torch.rand(C, generator=gp, device=th.device,
                                                   dtype=torch.float64),
                         0.5 + 1.0 * torch.rand(C, generator=gp, device=th.device,
                                                dtype=torch.float64)])
        cand["prior_draw"] = pri
        for nm, E, g in (("blind_E40_g1", 40.0, 1.0), ("blind_E130_g2", 130.0, 2.0),
                         ("blind_E60_g1.4", 60.0, 1.4)):
            cand[nm] = torch.cat([torch.full((C,), E, device=th.device, dtype=torch.float64),
                                  torch.full((C,), g, device=th.device, dtype=torch.float64)])

        R["candidates"] = {}
        for n, t in cand.items():
            R["candidates"][n] = score(t, th, C)
            R["candidates"][n].update({"mean_E": float(t[:C].mean()),
                                       "mean_gain": float(t[C:].mean()),
                                       "n_negative_E": int((t[:C] < 0).sum())})
        np.savez(os.path.join(HERE, f"theta_{a.tag}.npz"),
                 **{f"cand::{n}": t.cpu().numpy() for n, t in cand.items()})

        # ================================================================= the crash test =======
        if not a.skip_rollouts:
            R["rollouts"] = {}
            log(f"\n[crash test] {G}-frame FREE rollouts, margin-{MET.MARGIN_SAFE} grid")
            log(f"    {'candidate':<32s} {'medE':>7s} {'medg':>7s} | {'coord':>7s} {'orient':>7s} "
                f"| {'loopsc':>8s} | {'R2':>9s} {'Eratio':>7s} {'rms/dx':>7s}")
            todo = [("theta_true", th)] + list(cand.items())
            for name, theta in todo:
                tr, _, coarse = CT.rollout(sy, theta, W, G, tracers, ref_full=ref_full,
                                           anchor=None, interior=interior, ss_tot=ss_tot,
                                           band_mask=anchor)
                m20 = CT.read_metrics(tr[MET.MARGIN_SAFE].cpu().numpy(),
                                      ref_tr[MET.MARGIN_SAFE].cpu().numpy())
                R["rollouts"][name] = {"theta_error": R["candidates"].get(
                    name, score(theta, th, C)), "margin20": m20, "coarse": coarse}
                log(f"    {name:<32s} {R['rollouts'][name]['theta_error']['med_E']:>7.4f} "
                    f"{R['rollouts'][name]['theta_error']['med_gain']:>7.4f} | "
                    f"{CT.fmt(m20['coordination'],7)} {CT.fmt(m20['orientation_error'],7)} | "
                    f"{CT.fmt(m20['loopscore'],8)} | "
                    f"{CT.fmt(coarse['R2_displacement_interior'],9)} "
                    f"{CT.fmt(coarse['motion_energy_ratio_interior'],7)} "
                    f"{CT.fmt(coarse['rms_pos_err_dx_mean'],7)}")

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json  [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()
