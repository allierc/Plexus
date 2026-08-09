"""crash_test.py -- ROUND 1. Solve for theta, put it back in the simulator, ROLL OUT, score.

THE QUESTION THE ALGEBRA DOES NOT ANSWER
====================================================================================================
`assemble.py` established that one MLS-MPM substep is exactly affine in theta = (E_1..E_C, g_1..g_C)
and `recover.py` established that least squares gets theta back to 1.7e-9 from clean data. Neither
says anything about the object the campaign actually cares about, which is a TRAJECTORY. Equation
error is not output error: a parameter vector correct to a few percent can still produce a rollout
that walks away from the reference. This script measures that gap, on synthetic data where the
answer is known, so that "the parameters are wrong" and "the parameters are right and the rollout
diverges" can be told apart.

WHAT IS RUN
----------------------------------------------------------------------------------------------------
  1. C=100 cells x 100 particles, PLANTED (E, gain) per cell (build_planted, ../algebraic/recover.py)
  2. warm up W frames, RECORDING the last 150 frames  -> reference window A (the previous beat)
  3. snapshot at tick W; free-run 150 more frames on theta_true -> reference window B (the target)
  4. solve  A theta = b  at BOTH cadences: one substep (dt_sub) and one frame (dt = 10 substeps)
  5. CRASH TEST: reinstall each theta at the tick-W snapshot and ROLL OUT 150 frames, free-running,
     twice: with the border band anchored to the reference every frame, and with nothing anchored
  6. score window B with the registry in discovery_cardio_mpm/metrics.py

THE READING SURFACE
----------------------------------------------------------------------------------------------------
The registry's grid is defined on the recording's 137x137 PIV lattice with SHEET_SPAN = 0.70; this
sheet is a unit square. The translation used here, and it is the only one used:

    node (r, c) of the 10x10 selection at margin m  ->  world (c/(137-1), r/(137-1))

so margin 20 puts the outermost probe 0.147 from the edge and margin 10 puts it 0.0735 -- the same
*fractions of the sheet* as in the campaign. The anchored band is 0.06 / SHEET_SPAN = 0.0857 wide
for the same reason, and it pins 31.3% of particles, which is what 0.06 pins on the real sheet
(1 - (0.58/0.70)^2 = 0.314). At margin 10 the probes are inside that band and score perfectly; at
MARGIN_SAFE = 20 they are outside it. Both are reported, because the difference is the whole reason
MARGIN_SAFE exists. Each probe is a TRACER: the particle nearest that world point at tick W,
followed by identity thereafter.

usage:
  PYTHONPATH=/workspace/Plexus/src python crash_test.py --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ALG = "/workspace/Plexus/prototype/cardio_cells/algebraic"
DISC = "/workspace/Plexus/discovery_cardio_mpm"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/workspace/Plexus/src")
sys.path.insert(0, ALG)
sys.path.insert(0, DISC)

from assemble import System, SUBSTEP_TOKENS, rel            # noqa: E402
from recover import Solver, fd_accel, install_E, score      # noqa: E402

import metrics as MET                                       # noqa: E402

INSTRUMENTS = ("coordination", "path_length", "peak_excursion", "orientation_error")
OBJECTIVE = "loopscore"


# --------------------------------------------------------------------------------------------- #
#  the reading surface
# --------------------------------------------------------------------------------------------- #
def probe_points(margin):
    """The registry's 10x10 selection, mapped onto this unit-square sheet. [100, 2] world."""
    idx = MET.select_grid_nodes(margin=margin)
    r, c = idx // MET.GRID_SIDE, idx % MET.GRID_SIDE
    u = np.stack([c, r], 1) / (MET.GRID_SIDE - 1.0)          # (x, y) in [0,1]
    return u


def tracer_indices(x0, pts):
    """For each probe point, the index of the nearest particle at tick W."""
    P = torch.as_tensor(pts, device=x0.device, dtype=x0.dtype)
    d = (x0[:, None, :] - P[None, :, :]).pow(2).sum(-1)      # [Np, 100]
    return d.argmin(0)


# --------------------------------------------------------------------------------------------- #
#  planting + a warm-up that records
# --------------------------------------------------------------------------------------------- #
def plant_and_warm(args, log, seed=2026):
    """build_planted, but recording the last `args.window` frames of the warm-up.

    Returns the system (snapshotted at tick W) plus:
      recA   [win, Np, 2]  x_{W-win+1} .. x_W        -- the PREVIOUS beat, the replay bar's material
      x_prev [Np, 2]       one SUBSTEP before x_W
    """
    sy = System(device=args.device, n_cells=args.cells, per_parent=args.per_parent,
                n_grid=args.n_grid, warmup=0, dtype=args.dtype, mode=args.mode)
    C = sy.C
    g = torch.Generator().manual_seed(seed)
    E = (args.e_lo + (args.e_hi - args.e_lo) * torch.rand(C, generator=g)).to(sy.device, sy.dtype)
    gn = (args.g_lo + (args.g_hi - args.g_lo) * torch.rand(C, generator=g)).to(sy.device, sy.dtype)
    sy.E_true[1:], sy.gain_true[1:] = E, gn
    sy.theta_true = torch.cat([sy.E_true[1:], sy.gain_true[1:]])
    install_E(sy, sy.E_true)

    W, win = args.warmup, args.window
    rec = torch.zeros(win, sy.Np, 2, device=sy.device, dtype=sy.dtype)
    x_prev = None
    for tick in range(W):
        sy._outer(tick, gain_cell=sy.gain_true)
        sy.H.sub_dt = sy.dt_sub
        for s in range(sy.n_sub_per_frame):
            if tick == W - 1 and s == sy.n_sub_per_frame - 1:
                x_prev = sy.p.get("pos").clone()
            for tok in SUBSTEP_TOKENS:
                sy._tok(tok)
        sy.H.sub_dt = None
        if tick >= W - win:
            rec[tick - (W - win)] = sy.p.get("pos")
    sy.warmup_frames = W
    sy._snapshot(W)
    sy.x_prev = x_prev
    log(f"[planted] C={sy.C} Np={sy.Np} grid {sy.g.nx}^2 dtype={sy.dtype} dx={sy.g.dx:.6g}")
    log(f"          E in [{float(E.min()):.1f},{float(E.max()):.1f}]  "
        f"gain in [{float(gn.min()):.3f},{float(gn.max()):.3f}]  "
        f"dt={sy.dt} dt_sub={sy.dt_sub} ({sy.n_sub_per_frame} substeps/frame)")
    log(f"          warm-up {W} frames; window A = ticks {W-win+1}..{W} recorded")
    return sy, rec


# --------------------------------------------------------------------------------------------- #
#  the crash test itself
# --------------------------------------------------------------------------------------------- #
def rollout(sy, theta, t0, G, tracers, ref_full=None, anchor=None, keep_full=False,
            interior=None, ss_tot=None, jitter=0.0, jitter_seed=9, band_mask=None):
    """Free-run G frames from the tick-t0 snapshot with parameters `theta`.

    tracers : {margin: LongTensor[M]} -- every reading surface recorded from ONE rollout, so the
              margin-10 and margin-20 numbers are the same trajectory and not two runs.
    anchor  : None, or a bool [Np] mask of particles whose position AND velocity are overwritten
              with the reference's at the END of every frame. F, C and Jp are left alone -- the
              anchor is a kinematic pin, which is all that pinning a border to a recording can be.
    jitter  : std of a Gaussian displacement applied to EVERY particle at t0. With theta = theta_true
              this separates initial-condition sensitivity from parameter error.
    """
    C = sy.C
    E = torch.zeros(C + 1, device=sy.device, dtype=sy.dtype)
    gn = torch.zeros_like(E)
    E[1:], gn[1:] = theta[:C], theta[C:]
    sy.restore()
    install_E(sy, E)
    pa, pb = sy.p.state_schema["pos"]
    va, vb = sy.p.state_schema["vel"]
    if jitter > 0:
        gj = torch.Generator(device=sy.device).manual_seed(jitter_seed)
        st = sy.p.state
        st[:, pa:pb] = st[:, pa:pb] + jitter * torch.randn(
            (sy.Np, 2), generator=gj, device=sy.device, dtype=sy.dtype)
        sy.p.state = st

    tr = {m: torch.zeros(G, t.numel(), 2, device=sy.device, dtype=sy.dtype)
          for m, t in tracers.items()}
    full = torch.zeros(G, sy.Np, 2, device=sy.device, dtype=sy.dtype) if keep_full else None
    ss_res = torch.zeros((), device=sy.device, dtype=sy.dtype)
    e_sim = torch.zeros((), device=sy.device, dtype=sy.dtype)
    e_ref = torch.zeros((), device=sy.device, dtype=sy.dtype)
    rms = torch.zeros(G, device=sy.device, dtype=sy.dtype)
    rms_b = torch.zeros(G, device=sy.device, dtype=sy.dtype)
    x_start = sy.x0.clone()

    for k in range(G):
        sy._outer(t0 + k, gain_cell=gn)
        sy.H.sub_dt = sy.dt_sub
        for tok_i in range(sy.n_sub_per_frame):
            for tok in SUBSTEP_TOKENS:
                sy._tok(tok)
        sy.H.sub_dt = None
        if anchor is not None:
            xr = ref_full[k]
            xrp = ref_full[k - 1] if k > 0 else x_start
            st = sy.p.state
            st[anchor, pa:pb] = xr[anchor]
            st[anchor, va:vb] = (xr[anchor] - xrp[anchor]) / sy.dt
            sy.p.state = st
        x = sy.p.get("pos")
        for m, t in tracers.items():
            tr[m][k] = x[t]
        if keep_full:
            full[k] = x
        if ref_full is not None and interior is not None:
            d = (x - ref_full[k])[interior]
            ss_res = ss_res + d.pow(2).sum()
            rms[k] = d.pow(2).sum(-1).mean().sqrt()
            e_sim = e_sim + (x - x_start)[interior].pow(2).sum()
            e_ref = e_ref + (ref_full[k] - x_start)[interior].pow(2).sum()
            if band_mask is not None:
                db = (x - ref_full[k])[band_mask]
                rms_b[k] = db.pow(2).sum(-1).mean().sqrt()
    coarse = {}
    if ref_full is not None and interior is not None:
        coarse = {"R2_displacement_interior": float(1.0 - ss_res / ss_tot),
                  "motion_energy_ratio_interior": float(e_sim / e_ref),
                  "rms_pos_err_world_mean": float(rms.mean()),
                  "rms_pos_err_world_final": float(rms[-1]),
                  "rms_pos_err_dx_mean": float(rms.mean() / sy.g.dx),
                  "rms_pos_err_dx_final": float(rms[-1] / sy.g.dx),
                  "max_pos_err_world": float(rms.max()),
                  "rms_pos_err_dx_per_frame": [float(v) for v in (rms / sy.g.dx).cpu()],
                  "rms_pos_err_dx_BAND_mean": float(rms_b.mean() / sy.g.dx),
                  "rms_pos_err_dx_BAND_max": float(rms_b.max() / sy.g.dx)}
    return tr, full, coarse


# --------------------------------------------------------------------------------------------- #
#  scoring
# --------------------------------------------------------------------------------------------- #
def read_metrics(sim, real):
    """The four instruments + the objective, on [G, M, 2] arrays. Nothing new is defined here."""
    out = {}
    for n in INSTRUMENTS:
        m = MET.REGISTRY[n]
        try:
            out[n] = float(m(sim, real))
        except Exception as e:
            out[n] = f"{type(e).__name__}: {e}"
    try:
        out[OBJECTIVE] = float(MET.REGISTRY[OBJECTIVE](sim, real))
    except Exception as e:
        out[OBJECTIVE] = f"{type(e).__name__}: {e}"
    for n in ("path_length", "peak_excursion"):
        m = MET.REGISTRY[n]
        out[f"{n}_reading_sim"] = m.reading(sim)
        out[f"{n}_reading_real"] = m.reading(real)
    return out


def block_diag_A(A, C):
    """Column norms of the E block and the GAIN block, and the rank of the Gram matrix.

    Round 1 crashed here first, and the reason is worth keeping: the gain multiplies the
    active_force delta, the pacemaker is a 30-frame bump on a 150-frame period, and OUTSIDE the
    bump that delta is identically zero. At such a tick every gain column of A is exactly 0, G is
    exactly singular, and Cholesky *and* LU both fail. The gain is observable only during the 20%
    of the beat when the muscle is switched on -- that is a property of the system, not a bug.
    """
    cE, cg = A[:, :C].norm(dim=0), A[:, C:].norm(dim=0)
    Ad = A.double()
    sv = torch.linalg.eigvalsh(Ad.T @ Ad).clamp(min=0).sqrt().flip(0)
    return {"colnorm_E": [float(cE.min()), float(cE.median()), float(cE.max())],
            "colnorm_gain": [float(cg.min()), float(cg.median()), float(cg.max())],
            "n_zero_gain_columns": int((cg == 0).sum()),
            "n_zero_E_columns": int((cE == 0).sum()),
            "sv_max": float(sv[0]), "sv_min": float(sv[-1]),
            "rank_1e-12": int((sv > sv[0] * 1e-12).sum())}


def safe_solver(A, C):
    """Solver, or None plus the reason. A singular G is a result, not a crash."""
    try:
        return Solver(A, C), ""
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:160]}"


def fmt(v, w=9):
    return f"{v:>{w}.4f}" if isinstance(v, float) else f"{'n/a':>{w}}"


# --------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=180, help="tick of the snapshot t0")
    ap.add_argument("--window", type=int, default=150, help="frames per scoring window (= 1 beat)")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--mode", default="full")
    ap.add_argument("--e-lo", type=float, default=40.0)
    ap.add_argument("--e-hi", type=float, default=220.0)
    ap.add_argument("--g-lo", type=float, default=0.5)
    ap.add_argument("--g-hi", type=float, default=1.5)
    ap.add_argument("--eps", default="0.01,0.03,0.1,0.3",
                    help="fractional theta perturbations for the sensitivity ladder")
    ap.add_argument("--jitter", default="0.01,0.1",
                    help="initial-position jitter, in grid cells, at theta_true")
    ap.add_argument("--tag", default="round1")
    args = ap.parse_args()

    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"argv": vars(args)}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        # ---------------------------------------------------------------- 1. plant + warm up ----
        sy, recA = plant_and_warm(args, log)
        C, W, G = sy.C, args.warmup, args.window
        th = sy.theta_true.double()
        dx = sy.g.dx
        R.update({"C": C, "Np": sy.Np, "dx": dx, "dt": sy.dt, "dt_sub": sy.dt_sub,
                  "t0_tick": W, "window_frames": G,
                  "beat_period_frames": 150, "beat_duration_frames": 30,
                  "E_planted": [float(sy.E_true[1:].min()), float(sy.E_true[1:].max())],
                  "gain_planted": [float(sy.gain_true[1:].min()), float(sy.gain_true[1:].max())]})

        clock = {t: float(np.sin(np.pi * ((t) % 150) / 30.0)) if (t % 150) < 30 else 0.0
                 for t in (W - 1, W, W + 1)}
        R["pacemaker_clock_at_t0"] = clock
        R["norm_active_force_delta_at_t0"] = float(sy.act0.norm())
        R["norm_passive_delta_at_t0"] = float(sy.pass0.norm())
        log(f"[pacemaker] clock at ticks {sorted(clock)} = "
            + ", ".join(f"{v:.4f}" for _, v in sorted(clock.items()))
            + f";  ||active_force delta|| = {R['norm_active_force_delta_at_t0']:.4g}, "
              f"||drag delta|| = {R['norm_passive_delta_at_t0']:.4g}")

        # ---------------------------------------------------------------- reading surface -------
        x0 = sy.x0.clone()
        tracers = {m: tracer_indices(x0, probe_points(m))
                   for m in (MET.MARGIN_SAFE, MET.MARGIN_INHERITED)}
        band = 0.06 / MET.SHEET_SPAN
        anchor = ((x0[:, 0] < band) | (x0[:, 0] > 1 - band) |
                  (x0[:, 1] < band) | (x0[:, 1] > 1 - band))
        interior = ((x0[:, 0] > band) & (x0[:, 0] < 1 - band) &
                    (x0[:, 1] > band) & (x0[:, 1] < 1 - band))
        R["anchor_band_world"] = band
        R["anchor_fraction_of_particles"] = float(anchor.double().mean())
        R["interior_fraction_of_particles"] = float(interior.double().mean())
        for m, t in tracers.items():
            d = (x0[t] - torch.as_tensor(probe_points(m), device=x0.device,
                                         dtype=x0.dtype)).norm(dim=1)
            R[f"tracer_snap_distance_margin{m}"] = [float(d.median()), float(d.max())]
            R[f"tracer_in_anchor_band_margin{m}"] = int(anchor[t].sum())
        log(f"[surface] anchored band {band:.4f} world = "
            f"{100*R['anchor_fraction_of_particles']:.1f}% of particles; "
            f"probes inside it: margin {MET.MARGIN_SAFE} -> "
            f"{R[f'tracer_in_anchor_band_margin{MET.MARGIN_SAFE}']}/100, "
            f"margin {MET.MARGIN_INHERITED} -> "
            f"{R[f'tracer_in_anchor_band_margin{MET.MARGIN_INHERITED}']}/100")
        log(f"          tracer snap distance (margin {MET.MARGIN_SAFE}): median "
            f"{R[f'tracer_snap_distance_margin{MET.MARGIN_SAFE}'][0]:.5f} world "
            f"({R[f'tracer_snap_distance_margin{MET.MARGIN_SAFE}'][0]/dx:.2f} dx)")

        # can the registry be CITED?  (recorded rather than assumed)
        cite_status = {}
        dummy, _ = MET.population(G=24, M=20)
        for n in INSTRUMENTS + (OBJECTIVE,):
            try:
                MET.REGISTRY[n].cite(dummy, dummy)
                cite_status[n] = "cite() permitted"
            except Exception as e:
                cite_status[n] = f"{type(e).__name__}: {str(e)[:90]}"
        R["cite_status"] = cite_status

        # ---------------------------------------------------------------- 2. the reference ------
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
        R["reference"] = {
            "max_disp_world": float(d_ref.norm(dim=-1).max()),
            "max_disp_dx": float(d_ref.norm(dim=-1).max() / dx),
            "median_peak_disp_interior_world": float(
                d_ref[:, interior].norm(dim=-1).max(0).values.median()),
            "ss_tot_interior": float(ss_tot)}
        log(f"[reference] window B = ticks {W+1}..{W+G}; max particle displacement "
            f"{R['reference']['max_disp_world']:.5f} world = "
            f"{R['reference']['max_disp_dx']:.2f} dx; median peak interior displacement "
            f"{R['reference']['median_peak_disp_interior_world']:.5f}")

        # ---------------------------------------------------------------- 3. solve for theta ----
        # (a) SUBSTEP cadence -- the constraint that is exactly affine
        sy.restore()
        a_solver = sy.step(sy.E_true, sy.gain_true, n_sub=1)
        x_next = sy.p.get("pos").clone()
        a_fd = fd_accel(sy.x_prev, x0, x_next, sy.dt_sub)
        A1, a01, t1 = sy.assemble(n_sub=1)
        b1 = a_fd - a01
        res1 = rel(A1 @ sy.theta_true - (a_solver - a01), a_solver - a01)
        diag1 = block_diag_A(A1, C)
        S1, err1 = safe_solver(A1, C)
        if S1 is None:
            raise SystemExit(f"[solve/substep] G is singular: {err1}\n  {diag1}")
        sol1 = S1(b1)
        theta_sub = sol1["ridge0"]
        R["solve_substep"] = {"assembly_s": t1, "A_shape": list(A1.shape), "blocks": diag1,
                              "fd_vs_solver_rel": rel(a_fd - a_solver, a_solver),
                              "model_residual_rel_b": res1, "cond_scaled": S1.cond,
                              "scores": {k: score(v, th, C) for k, v in sol1.items()}}
        s = R["solve_substep"]["scores"]["ridge0"]
        log(f"[solve/substep] A {tuple(A1.shape)} in {t1:.1f}s; ||A.theta_true-b||/||b|| = "
            f"{res1:.3e}; cond(G) {S1.cond:.3e}")
        log(f"                ridge0: med|dE/E| {s['med_E']:.3e}  med|dg/g| {s['med_gain']:.3e}  "
            f"p90 E {s['p90_E']:.3e}  ||dtheta||/||theta|| {s['rel_l2']:.3e}")
        log(f"                column norms: E med {diag1['colnorm_E'][1]:.3e}, gain med "
            f"{diag1['colnorm_gain'][1]:.3e}; zero gain columns "
            f"{diag1['n_zero_gain_columns']}/{C}; rank {diag1['rank_1e-12']}/{2*C}")
        S1.free()
        del A1, S1
        torch.cuda.empty_cache()

        # (b) FRAME cadence -- the constraint the proposal actually asks for
        x_fprev = recA[-2].clone()                      # x_{W-1}
        x_fnext = ref_full[0].clone()                   # x_{W+1}
        sy.restore()
        a_solver_f = sy.step(sy.E_true, sy.gain_true, n_sub=sy.n_sub_per_frame)
        a_fdf = fd_accel(x_fprev, x0, x_fnext, sy.dt)
        Af, a0f, tf = sy.assemble(n_sub=sy.n_sub_per_frame)
        bf = a_fdf - a0f
        resf = rel(Af @ sy.theta_true - (a_solver_f - a0f), a_solver_f - a0f)
        diagf = block_diag_A(Af, C)
        Sf, errf = safe_solver(Af, C)
        if Sf is None:
            raise SystemExit(f"[solve/frame] G is singular: {errf}\n  {diagf}")
        solf = Sf(bf)
        theta_frm = solf["ridge0"]
        theta_frm_r = solf["ridge0.01"]
        R["solve_frame"] = {"assembly_s": tf, "fd_vs_solver_rel": rel(a_fdf - a_solver_f,
                                                                      a_solver_f),
                            "model_residual_rel_b": resf, "cond_scaled": Sf.cond,
                            "blocks": diagf,
                            "scores": {k: score(v, th, C) for k, v in solf.items()}}
        s = R["solve_frame"]["scores"]["ridge0"]
        sr = R["solve_frame"]["scores"]["ridge0.01"]
        log(f"[solve/frame  ] A in {tf:.1f}s; ||A.theta_true-b||/||b|| = {resf:.3e}; "
            f"cond(G) {Sf.cond:.3e}")
        log(f"                ridge0:    med|dE/E| {s['med_E']:.3e}  med|dg/g| "
            f"{s['med_gain']:.3e}  l2 {s['rel_l2']:.3e}")
        log(f"                ridge1e-2: med|dE/E| {sr['med_E']:.3e}  med|dg/g| "
            f"{sr['med_gain']:.3e}  l2 {sr['rel_l2']:.3e}")
        Sf.free()
        del Af, Sf
        torch.cuda.empty_cache()

        # ---------------------------------------------------------------- candidates ------------
        nom = torch.cat([torch.full((C,), 130.0, device=th.device, dtype=torch.float64),
                         torch.full((C,), 1.0, device=th.device, dtype=torch.float64)])
        gp = torch.Generator(device=str(th.device)).manual_seed(77)
        u = torch.randn(2 * C, generator=gp, device=th.device, dtype=torch.float64)
        u = u / u.abs().max()                                   # in [-1, 1]
        pm = torch.randperm(C, device=th.device)
        shuf = torch.cat([th[:C][pm], th[C:][pm]])
        cands = [("theta_true", th, 0.0),
                 ("theta_hat_substep", theta_sub, 0.0),
                 ("theta_hat_frame_ridge0", theta_frm, 0.0),
                 ("theta_hat_frame_ridge1e-2", theta_frm_r, 0.0),
                 ("theta_const_E130_g1", nom, 0.0),
                 ("theta_shuffled_true", shuf, 0.0)]
        for e in [float(x) for x in args.eps.split(",")]:
            cands.append((f"theta_true_perturbed_{e:g}", th * (1.0 + e * u), 0.0))
        # the OTHER source of trajectory error: the initial condition, at theta_true. Separates
        # "the parameters are wrong" from "this system forgets where it started".
        for j in [float(x) for x in args.jitter.split(",") if float(x) > 0]:
            cands.append((f"theta_true_x0jitter_{j:g}dx", th, j * dx))
        R["candidates"] = {n: score(t, th, C) for n, t, _ in cands}
        # is the estimate even PHYSICAL? a least-squares fit is free to return a negative modulus.
        for n, t, _ in cands:
            R["candidates"][n]["n_negative_E"] = int((t[:C] < 0).sum())
            R["candidates"][n]["n_negative_gain"] = int((t[C:] < 0).sum())
            R["candidates"][n]["mean_E"] = float(t[:C].mean())
            R["candidates"][n]["mean_gain"] = float(t[C:].mean())
            R["candidates"][n]["signed_bias_E"] = float((t[:C] - th[:C]).mean() / th[:C].mean())
            R["candidates"][n]["signed_bias_gain"] = float((t[C:] - th[C:]).mean() / th[C:].mean())
        np.savez(os.path.join(HERE, f"theta_{args.tag}.npz"),
                 **{f"cand::{n}": t.cpu().numpy() for n, t, _ in cands})
        log(f"[candidates] mean E: true {float(th[:C].mean()):.1f}; "
            + "  ".join(f"{n.replace('theta_','')} {R['candidates'][n]['mean_E']:.1f}"
                        f"(neg {R['candidates'][n]['n_negative_E']})"
                        for n, _, _ in cands if n.startswith("theta_hat")))

        # ---------------------------------------------------------------- 4/5/6 CRASH TEST ------
        R["rollouts"] = {}
        R["margin10"] = {}
        log(f"\n[crash test] {G}-frame free rollouts from the tick-{W} snapshot, scored on the "
            f"margin-{MET.MARGIN_SAFE} grid")
        log(f"    {'candidate':<30s} {'anchor':<7s} {'medE':>7s} {'medg':>7s} | "
            f"{'coord':>7s} {'pathlen':>9s} {'peakexc':>9s} {'orient':>7s} | "
            f"{'loopsc':>7s} | {'R2':>8s} {'Eratio':>7s} {'rms/dx':>7s} {'lsM10':>7s}")
        for name, theta, jit in cands:
            for mode in ("anchored", "free"):
                t_r = time.time()
                tr, _, coarse = rollout(sy, theta, W, G, tracers, ref_full=ref_full,
                                        anchor=anchor if mode == "anchored" else None,
                                        interior=interior, ss_tot=ss_tot, jitter=jit,
                                        band_mask=anchor)
                sim = tr[MET.MARGIN_SAFE].cpu().numpy()
                real = ref_tr[MET.MARGIN_SAFE].cpu().numpy()
                rec = {"theta_error": R["candidates"][name], "x0_jitter_world": jit,
                       "margin20": read_metrics(sim, real), "coarse": coarse,
                       "seconds": time.time() - t_r}
                rec["tracer_rel_l2"] = float(
                    np.linalg.norm(sim - real) / max(np.linalg.norm(real - real.mean(0)), 1e-300))
                R["rollouts"][f"{name}|{mode}"] = rec
                m10 = read_metrics(tr[MET.MARGIN_INHERITED].cpu().numpy(),
                                   ref_tr[MET.MARGIN_INHERITED].cpu().numpy())
                R["margin10"][f"{name}|{mode}"] = m10
                m20 = rec["margin20"]
                log(f"    {name:<30s} {mode:<7s} "
                    f"{rec['theta_error']['med_E']:>7.4f} {rec['theta_error']['med_gain']:>7.4f} | "
                    f"{fmt(m20['coordination'],7)} {fmt(m20['path_length'],9)} "
                    f"{fmt(m20['peak_excursion'],9)} {fmt(m20['orientation_error'],7)} | "
                    f"{fmt(m20['loopscore'],7)} | "
                    f"{fmt(coarse['R2_displacement_interior'],8)} "
                    f"{fmt(coarse['motion_energy_ratio_interior'],7)} "
                    f"{fmt(coarse['rms_pos_err_dx_mean'],7)} {fmt(m10['loopscore'],7)}"
                    f"  band {coarse['rms_pos_err_dx_BAND_mean']:.4f}")

        # ---------------------------------------------------------------- nulls -----------------
        log("\n[nulls]")
        R["nulls"] = {}
        real20 = ref_tr[MET.MARGIN_SAFE].cpu().numpy()
        # do nothing: hold the tick-W positions
        frozen = np.repeat(x0[tracers[MET.MARGIN_SAFE]].cpu().numpy()[None], G, axis=0)
        R["nulls"]["do_nothing"] = read_metrics(frozen, real20)
        d0 = d_ref[:, interior]
        R["nulls"]["do_nothing"]["coarse"] = {
            "R2_displacement_interior": float(1.0 - d0.pow(2).sum() / ss_tot),
            "motion_energy_ratio_interior": 0.0,
            "rms_pos_err_dx_mean": float(
                (d0.pow(2).sum(-1).mean(1).sqrt()).mean() / dx)}
        # replay: window A (the previous beat) offered as a prediction of window B
        replay = recA[:, tracers[MET.MARGIN_SAFE]].cpu().numpy()
        R["nulls"]["replay_previous_beat"] = read_metrics(replay, real20)
        dA = (recA - x0[None])[:, interior]
        R["nulls"]["replay_previous_beat"]["coarse"] = {
            "R2_displacement_interior": float(1.0 - (dA - d0).pow(2).sum() / ss_tot),
            "motion_energy_ratio_interior": float(dA.pow(2).sum() / d0.pow(2).sum()),
            "rms_pos_err_dx_mean": float(
                ((recA - ref_full)[:, interior].pow(2).sum(-1).mean(1).sqrt()).mean() / dx)}
        # the identity: the reference read against itself
        R["nulls"]["identity"] = read_metrics(real20, real20)
        R["campaign_nulls"] = {"loopscore_predict_nothing": 0.0700,
                               "loopscore_replay_fit_beat": 0.851,
                               "loopscore_replay_heldout": 0.62,
                               "path_length_N0": 0.0042, "peak_excursion_N0": 0.0011,
                               "orientation_error_chance": float(np.pi / 4),
                               "coordination_N0_scrambled": 0.0778,
                               "note": "measured on the REAL recording; the paired metrics carry "
                                       "its units, so only the synthetic nulls below are "
                                       "commensurate with the rows above"}
        for k, v in R["nulls"].items():
            log(f"    {k:<24s} coord {fmt(v['coordination'],8)}  path {fmt(v['path_length'],9)}  "
                f"peak {fmt(v['peak_excursion'],9)}  orient {fmt(v['orientation_error'],8)}  "
                f"loopscore {fmt(v['loopscore'],8)}")

    R["wall_seconds"] = time.time() - t_start
    out = os.path.join(HERE, f"crash_{args.tag}.json")
    json.dump(R, open(out, "w"), indent=1, default=str)
    open(os.path.join(HERE, f"crash_{args.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {out}\n[{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()
