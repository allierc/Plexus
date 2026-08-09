"""refute_round3.py -- attack round 3's part-2/part-3 headline.

ROUND 3 CLAIMED
  (2) injecting a linearly interpolated MEASURED F across the substeps of a frame takes the
      frame-cadence recovery from med|dE/E| 0.257 to 0.0078 and the free 150-frame rollout to
      loopscore 0.9997 with no gauge -- "the per-substep problem is solved by two measured frames";
  (3) the recording's own F is not accurate enough: at sigma_F = 0.0039 the recovery returns to
      0.257, so "sigma_F <~ 1e-3, a 4x improvement on the recording" is required.

FIVE ATTACKS, each with a control
  A  reproduce finject's F_lerp row from a different script / different GPU
  B  t0 SWEEP.  Round 1 already showed frame-cadence recovery is strongly state dependent (51% at
     tick 165, 5% at tick 24).  0.0078 is ONE state.  Is it the method or the state?
  C  WRONG-F NULLS the round did not run: inject F = I; inject F_lerp shuffled across particles;
     inject the F of a WRONG model (blind E=130,g=1); inject F_lerp with the measured increment
     halved.  If any of these also lands near 0.008, the win is not coming from F's accuracy.
  D  the NOISE MODEL.  finject_noise.noisy_F draws an INDEPENDENT F error at every one of the 10
     substeps (`torch.randn(Fl.shape)`, Fl.shape[0] = n_sub).  The realizable estimator measures F
     ONCE PER FRAME and interpolates, so its error is COHERENT across the substeps of a frame.
     White-in-substep error averages down by ~sqrt(10); the round's threshold is therefore
     optimistic.  Measured here head to head.
  E  the missing estimator: MULTI-FRAME STACKING.  The recording offers 49 frames per beat and the
     round fit ONE.  refute3_real.py measures the F error's temporal autocorrelation at lag 1 =
     0.0006 (white) for the quiet-stretch noise, so stacking T frames should average it down as
     1/sqrt(T).  Run with (i) independent noise per frame and (ii) a STATIC bias reused every
     frame, which must NOT average down.
  F  the STATE oracle.  `System.restore()` puts back pos, vel, F, C and Jp from the truth.  Only
     pos and F are in the recording.  Replace C0 by the data-derivable Fdot F^-1 and v0 by a
     finite difference of frame positions and see what is left.

usage: PYTHONPATH=/workspace/Plexus/src python refute_round3.py --device cuda:1 --phases ABCDEF
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
for _p in ("/workspace/Plexus/src", ALG, DISC, HERE):
    sys.path.insert(0, _p)

from assemble import SUBSTEP_TOKENS, rel                        # noqa: E402
from recover import Solver, install_E, score                    # noqa: E402
import crash_test as CT                                         # noqa: E402
from finject import assemble_inj, y_of, record_substeps, hold, lerp   # noqa: E402


def advance(sy, t_from, t_to, xhist=None):
    # BUG FOUND THE HARD WAY: `assemble_inj` leaves p.mu/p.la at the LAST probe column (E = 0 for
    # every cell but one), and `advance` does not reinstall them.  Walking the reference trajectory
    # after a fit therefore integrates a different material.  Reinstall theta_true every frame.
    install_E(sy, sy.E_true)
    for t in range(t_from, t_to):
        sy._outer(t, gain_cell=sy.gain_true)
        sy.H.sub_dt = sy.dt_sub
        for _ in range(sy.n_sub_per_frame):
            for tok in SUBSTEP_TOKENS:
                sy._tok(tok)
        sy.H.sub_dt = None
        if xhist is not None:
            xhist[t + 1] = sy.p.get("pos").clone()


def record_substeps_theta(sy, n, E_cell, gain_cell):
    """record_substeps, but under an ARBITRARY theta -- the wrong-model F of attack C."""
    from plexus.models.entities import _lame
    H, p = sy.H, sy.p
    sy.restore()
    mu, la = _lame(E_cell[sy.cid])
    p.mu, p.la = mu, la
    H.zero_delta()
    H._delta["mpm_particle"] = sy.pass0 + gain_cell[sy.cid][:, None] * sy.act0
    H.sub_dt = sy.dt_sub
    Fs, Xs = [], []
    for _ in range(n):
        sy._tok("mpm_strain")
        Fs.append(p.F.clone())
        sy._tok("mpm_scatter")
        sy._tok("mpm_grid_update")
        sy._tok("mpm_gather")
        Xs.append(p.get("pos").clone())
    H.sub_dt = None
    return torch.stack(Fs), torch.stack(Xs)


def fit(sy, n, injF, y_obs, x0, th, C, injC=None):
    A, y0, _ = assemble_inj(sy, n, injF, injC)
    b = (y_obs - x0).reshape(-1) - y0
    cn = A.norm(dim=0)
    nzero = int((cn <= 1e-14 * float(cn.max())).sum())
    try:
        S = Solver(A, C)
        t_hat = S(b)["ridge0"]
        cond, how = S.cond, "ridge0"
        S.free()
        del S
    except Exception:                       # G is singular: the gain block is dead off-pulse
        U, sv, Vh = torch.linalg.svd(A.double(), full_matrices=False)
        keep = sv > sv.max() * 1e-10
        inv = torch.where(keep, 1.0 / sv.clamp(min=1e-300), torch.zeros_like(sv))
        t_hat = Vh.T @ (inv * (U.T @ b.double()))
        cond = float(sv.max() / sv[keep].min())
        how = f"pinv(rank {int(keep.sum())}/{2*C})"
    sc = score(t_hat, th, C)
    sc["cond"] = cond
    sc["solver"] = how
    sc["n_zero_columns"] = nzero
    sc["n_negE"] = int((t_hat[:C] < 0).sum())
    del A
    torch.cuda.empty_cache()
    return sc, t_hat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="refute3")
    ap.add_argument("--phases", default="ABCDEF")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    ap_start = 40
    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128,
                           warmup=ap_start, window=ap_start, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "phases": a.phases}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        sy, recA = CT.plant_and_warm(args, log)
        C, n = sy.C, sy.n_sub_per_frame
        th = sy.theta_true.double()
        gen = torch.Generator(device=sy.device).manual_seed(777)
        I2 = torch.eye(2, device=sy.device, dtype=sy.dtype)

        # `_snapshot` freezes whatever state the particle level happens to be in, so it may only
        # ever be called from the clean trajectory: restore -> advance -> snapshot, ticks
        # monotonically increasing.  (Getting this wrong is worth a factor 11 in the answer.)
        cur = {"tick": ap_start}
        xhist = {}

        def goto(tick):
            assert tick >= cur["tick"], (tick, cur["tick"])
            sy.restore()
            advance(sy, cur["tick"], tick, xhist)
            sy._snapshot(tick)
            cur["tick"] = tick

        SNAP_FIELDS = ("state0", "F0", "C0", "Jp0", "v0", "x0", "act0", "pass0")

        def save_snap():
            return {k: getattr(sy, k).clone() for k in SNAP_FIELDS}

        def load_snap(s):
            for k, v in s.items():
                setattr(sy, k, v.clone())

        def snap_frame(tick):
            """(x0, F0, Fs, Cs, x_next) at `tick`, the system left snapshotted there."""
            goto(tick)
            Fs, Cs, Xs = record_substeps(sy, n)
            return sy.x0.clone(), sy.F0.clone(), Fs, Cs, Xs[-1].clone()

        # ---------------------------------------------------------------- A: control -----------
        if set(a.phases) & set("ACDEF"):
            x0, F0, Fs, Cs, x_next = snap_frame(165)
            Fl = lerp(F0, Fs[-1], n)
            snap165 = save_snap()
            keep165 = (x0.clone(), F0.clone(), Fs.clone(), x_next.clone(), xhist[164].clone())
        if "A" in a.phases:
            sc, _ = fit(sy, n, Fl, x_next, x0, th, C)
            sc_none, _ = fit(sy, n, None, x_next, x0, th, C)
            R["A_control"] = {"F_lerp": sc, "none": sc_none,
                              "finject_reported": {"F_lerp_med_E": 0.0077773, "none_med_E": 0.2572}}
            log(f"[A] t0=165  none medE {sc_none['med_E']:.4f} (finject 0.2572)   "
                f"F_lerp medE {sc['med_E']:.4f} (finject 0.0078)")

        # ---------------------------------------------------------------- B: t0 sweep ----------
        if "B" in a.phases:
            R["B_t0_sweep"] = {}
            log("\n[B] t0 sweep -- is 0.0078 the method or the state?")
            log(f"    {'t0':>5s} {'|dF|frame':>10s} {'none':>9s} {'F_hold':>9s} {'F_lerp':>9s} "
                f"{'F_true':>9s} {'C_lerp':>9s} {'C_true':>9s} {'p90 Fl':>9s} {'zc':>3s}")
            for t0 in [t for t in (45, 75, 105, 120, 135, 150, 165, 180, 195)
                       if t >= cur["tick"]]:
                goto(t0)
                Fs_t, Cs_t, Xs_t = record_substeps(sy, n)
                x0t, F0t, C0t, xn_t = sy.x0.clone(), sy.F0.clone(), sy.C0.clone(), Xs_t[-1].clone()
                row = {"dF_frame_rel": float((Fs_t[-1] - F0t).norm() / F0t.norm()),
                       "dF_frame_abs": float((Fs_t[-1] - F0t).norm(dim=(-2, -1)).median()),
                       "act_norm": float(sy.act0.norm())}
                for nm, iF, iC in (("none", None, None), ("F_hold", hold(F0t, n), None),
                                   ("F_lerp", lerp(F0t, Fs_t[-1], n), None),
                                   ("F_true", Fs_t, None),
                                   ("C_lerp", None, lerp(C0t, Cs_t[-1], n)),
                                   ("C_true", None, Cs_t)):
                    s, _ = fit(sy, n, iF, xn_t, x0t, th, C, injC=iC)
                    row[nm] = s
                row["F_lerp_approx_error"] = float((Fs_t - lerp(F0t, Fs_t[-1], n)).norm()
                                                   / Fs_t.norm())
                R["B_t0_sweep"][str(t0)] = row
                log(f"    {t0:>5d} {row['dF_frame_abs']:>10.2e} {row['none']['med_E']:>9.4f} "
                    f"{row['F_hold']['med_E']:>9.4f} {row['F_lerp']['med_E']:>9.4f} "
                    f"{row['F_true']['med_E']:>9.4f} {row['C_lerp']['med_E']:>9.4f} "
                    f"{row['C_true']['med_E']:>9.4f} {row['F_lerp']['p90_E']:>9.4f} "
                    f"{row['F_lerp']['n_zero_columns']:>3d}")

        # ---------------------------------------------------------------- C: wrong-F nulls -----
        if "C" in a.phases:
            x0, F0, Fs, Cs, x_next = snap_frame(165)
            Fl = lerp(F0, Fs[-1], n)
            Eb = torch.full((C + 1,), 130.0, device=sy.device, dtype=sy.dtype)
            Eb[0] = 0.0
            gb = torch.ones(C + 1, device=sy.device, dtype=sy.dtype)
            gb[0] = 0.0
            Fw, _ = record_substeps_theta(sy, n, Eb, gb)
            perm = torch.randperm(sy.Np, generator=gen, device=sy.device)
            nulls = {
                "F_lerp (round 3)": Fl,
                "F_identity": I2.expand(n, sy.Np, 2, 2).contiguous(),
                "F_hold_only": hold(F0, n),
                "F_lerp_shuffled_particles": Fl[:, perm],
                "F_lerp_of_a_WRONG_model_E130_g1": lerp(F0, Fw[-1], n),
                "F_lerp_increment_halved": lerp(F0, F0 + 0.5 * (Fs[-1] - F0), n),
                "F_lerp_increment_doubled": lerp(F0, F0 + 2.0 * (Fs[-1] - F0), n)}
            R["C_wrongF"] = {}
            log("\n[C] wrong-F nulls the round did not run (t0=165)")
            for nm, iF in nulls.items():
                s, _ = fit(sy, n, iF, x_next, x0, th, C)
                err = float((iF - Fs).norm() / Fs.norm())
                R["C_wrongF"][nm] = {"injF_error_vs_true": err, **s}
                log(f"    {nm:<34s} |dF_inj|/|F| {err:>9.2e}  medE {s['med_E']:>8.4f}  "
                    f"p90 {s['p90_E']:>8.4f}  l2 {s['rel_l2']:>8.3f}")

        # ---------------------------------------------------------------- D: noise coherence ---
        if "D" in a.phases:
            x0, F0, Fs, Cs, x_next = snap_frame(165)
            F1 = Fs[-1].clone()
            Fl = lerp(F0, F1, n)
            sx = 0.0409 * 4.88e-4
            R["D_noise_coherence"] = {}
            log("\n[D] the noise model: round 3 redraws F error at EVERY substep; the realizable "
                "estimator measures F once per frame")
            log(f"    {'sigma_F':>9s} {'model':>22s} {'medE':>8s} {'p90E':>8s} {'l2':>9s}")
            for sF in (2e-4, 5e-4, 1e-3, 2e-3, 3.9e-3):
                for mode in ("white_per_substep", "coherent_per_frame"):
                    ms, ps, ls = [], [], []
                    for _ in range(a.reps):
                        if mode == "white_per_substep":
                            e = torch.randn(Fl.shape, generator=gen, device=Fl.device,
                                            dtype=Fl.dtype)
                            iF = Fl + (sF / 2.0) * e
                        else:
                            e0 = torch.randn(F0.shape, generator=gen, device=F0.device,
                                             dtype=F0.dtype)
                            e1 = torch.randn(F0.shape, generator=gen, device=F0.device,
                                             dtype=F0.dtype)
                            iF = lerp(F0 + (sF / 2.0) * e0, F1 + (sF / 2.0) * e1, n)
                        xn = x_next + sx * torch.randn(x_next.shape, generator=gen,
                                                       device=sy.device, dtype=sy.dtype)
                        s, _ = fit(sy, n, iF, xn, x0, th, C)
                        ms.append(s["med_E"]); ps.append(s["p90_E"]); ls.append(s["rel_l2"])
                    R["D_noise_coherence"][f"{sF:g}_{mode}"] = {
                        "sigma_F": sF, "mode": mode, "med_E": float(np.mean(ms)),
                        "p90_E": float(np.mean(ps)), "rel_l2": float(np.mean(ls)),
                        "per_rep": ms}
                    log(f"    {sF:>9.1e} {mode:>22s} {np.mean(ms):>8.4f} {np.mean(ps):>8.4f} "
                        f"{np.mean(ls):>9.3f}")

        # ---------------------------------------------------------------- E: multi-frame -------
        if "E" in a.phases:
            TMAX = 8
            sF = 3.9e-3
            sx = 0.0409 * 4.88e-4
            log(f"\n[E] MULTI-FRAME STACKING, sigma_F = {sF:g} coherent per frame, T up to {TMAX}")
            # gather the per-frame pieces once
            frames = []
            for k in range(TMAX):
                tick = 165 + k
                goto(tick)
                Fs_k, Cs_k, Xs_k = record_substeps(sy, n)
                frames.append({"tick": tick, "x0": sy.x0.clone(), "F0": sy.F0.clone(),
                               "F1": Fs_k[-1].clone(), "x_next": Xs_k[-1].clone(),
                               "snap": save_snap()})
            R["E_multiframe"] = {"sigma_F": sF, "sigma_x": sx, "T_max": TMAX, "runs": {}}
            log(f"    {'noise':>10s} {'T':>3s} {'medE':>8s} {'p90E':>8s} {'l2':>9s} {'negE':>5s} "
                f"{'cond':>10s}")
            for noise in ("none", "white_across_frames", "static_bias"):
                for rep in range(a.reps if noise != "none" else 1):
                    # one measurement error per FRAME BOUNDARY, shared by the two frames using it
                    if noise == "none":
                        es = [torch.zeros_like(frames[0]["F0"]) for _ in range(TMAX + 1)]
                    elif noise == "white_across_frames":
                        es = [(sF / 2.0) * torch.randn(frames[0]["F0"].shape, generator=gen,
                                                       device=sy.device, dtype=sy.dtype)
                              for _ in range(TMAX + 1)]
                    else:
                        e = (sF / 2.0) * torch.randn(frames[0]["F0"].shape, generator=gen,
                                                     device=sy.device, dtype=sy.dtype)
                        es = [e] * (TMAX + 1)
                    As, bs = [], []
                    for k, fr in enumerate(frames):
                        load_snap(fr["snap"])
                        iF = lerp(fr["F0"] + es[k], fr["F1"] + es[k + 1], n)
                        A, y0, _ = assemble_inj(sy, n, iF, None)
                        xn = fr["x_next"] + sx * torch.randn(fr["x_next"].shape, generator=gen,
                                                             device=sy.device, dtype=sy.dtype)
                        As.append(A)
                        bs.append((xn - fr["x0"]).reshape(-1) - y0)
                    for T in (1, 2, 4, 8):
                        if T > TMAX:
                            continue
                        Ac = torch.cat(As[:T], 0)
                        bc = torch.cat(bs[:T], 0)
                        S = Solver(Ac, C)
                        t_hat = S(bc)["ridge0"]
                        s = score(t_hat, th, C)
                        s["cond"] = S.cond
                        s["n_negE"] = int((t_hat[:C] < 0).sum())
                        key = f"{noise}_T{T}"
                        R["E_multiframe"]["runs"].setdefault(key, []).append(s)
                        S.free(); del Ac, bc, S
                        torch.cuda.empty_cache()
                    del As, bs
                    torch.cuda.empty_cache()
            for key, v in R["E_multiframe"]["runs"].items():
                log(f"    {key:>26s} medE {np.mean([x['med_E'] for x in v]):>8.4f} "
                    f"p90 {np.mean([x['p90_E'] for x in v]):>8.4f} "
                    f"l2 {np.mean([x['rel_l2'] for x in v]):>9.3f} "
                    f"negE {np.mean([x['n_negE'] for x in v]):>5.1f} "
                    f"cond {np.mean([x['cond'] for x in v]):>10.2e}")

        # ---------------------------------------------------------------- F: state oracle ------
        if "F" in a.phases:
            log("\n[F] the restored state is an ORACLE in v, C and Jp; only pos and F are measured")
            load_snap(snap165)
            x0, F0, Fs, x_next, x_fprev = keep165
            Fl = lerp(F0, Fs[-1], n)
            v_true, C_true = sy.v0.clone(), sy.C0.clone()
            # data-derivable surrogates
            v_bd = (x0 - x_fprev) / sy.dt                       # backward frame difference
            v_cd = (x_next - x_fprev) / (2 * sy.dt)             # centred, uses the next frame too
            Fdot = (Fs[-1] - F0) / sy.dt
            C_fd = Fdot @ torch.linalg.inv(F0)
            R["F_state_oracle"] = {
                "x0_matches_history": float((xhist[165] - x0).abs().max()),
                "v_rel_err_backward_fd": float((v_bd - v_true).norm() / v_true.norm()),
                "v_rel_err_centred_fd": float((v_cd - v_true).norm() / v_true.norm()),
                "C_rel_err_of_FdotFinv": float((C_fd - C_true).norm() / C_true.norm()),
                "Jp_is_trivial": float((sy.Jp0 - 1.0).abs().max())}
            log(f"    v from a frame finite difference: rel err backward "
                f"{R['F_state_oracle']['v_rel_err_backward_fd']:.3f} centred "
                f"{R['F_state_oracle']['v_rel_err_centred_fd']:.3f};  C from Fdot F^-1: rel err "
                f"{R['F_state_oracle']['C_rel_err_of_FdotFinv']:.3f};  max|Jp-1| "
                f"{R['F_state_oracle']['Jp_is_trivial']:.2e}")
            base_state = sy.state0.clone()
            va, vb = sy.p.state_schema["vel"]
            for nm, newv, newC in (("oracle_state", None, None),
                                   ("v <- backward frame fd", v_bd, None),
                                   ("v <- centred frame fd", v_cd, None),
                                   ("C <- Fdot F^-1", None, C_fd),
                                   ("both (centred v, C_fd)", v_cd, C_fd)):
                sy.state0 = base_state.clone()
                sy.C0 = C_true.clone()
                if newv is not None:
                    sy.state0[:, va:vb] = newv
                if newC is not None:
                    sy.C0 = newC
                s, _ = fit(sy, n, Fl, x_next, x0, th, C)
                R["F_state_oracle"][nm] = s
                log(f"    {nm:<32s} medE {s['med_E']:>8.4f}  p90 {s['p90_E']:>8.4f}  "
                    f"l2 {s['rel_l2']:>9.3f}")
            sy.state0 = base_state
            sy.C0 = C_true

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()
