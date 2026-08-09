"""state_derive.py -- ROUND 6, TASK 1.  Remove the LAST state oracle and validate the derivation
alone, at ZERO noise.

WHAT THE ORACLE IS
====================================================================================================
`System.restore()` (algebraic/assemble.py:225) writes back p.state (positions AND velocities), p.F,
p.C and p.Jp from the TRUE simulator snapshot.  round5_fit.py / refute5_fit.py call it, through
`assemble_inj -> y_of -> step_inj` (finject.py:72), for every one of the 401 assemblies of every
frame.  Only F was ever treated as measured; v, C and Jp came from the answer key.

A microscope gives x.  It gives F (the recording has du/dx, du/dy, dv/dx, dv/dy channels).  It does
NOT give C -- the MLS-MPM affine velocity matrix is solver bookkeeping.  So:

    v_k  <- (X_{k+1} - X_{k-1}) / (2 dt)                        centred difference of MEASURED x
    C_k  <- ((F_{k+1} - F_{k-1}) / (2 dt)) @ inv(F_k)           on the MEASURED F
    Jp_k <- 1                                                   (confirmed, not assumed: below)

`mpm_strain` integrates F <- (I + dt_sub C) F exactly, so Fdot F^-1 recovers the SUBSTEP-AVERAGED C,
not the boundary C.  That is a truncation floor and this script measures it as one.

WHAT IS MEASURED (three stages, `--stages dt18`)
----------------------------------------------------------------------------------------------------
  d : the derivation ALONE.  rel err of derived v, derived C, and Jp against the simulator's own,
      at zero noise, at every frame of the fit window; plus the same at the recording's sigma_F /
      sigma_x, to separate a TRUNCATION floor (noise-independent) from a noise effect; plus the
      control that explains the floor (Fdot F^-1 vs the substep-averaged C the strain op actually
      used).
  1 : ROUND 3'S CONTROL, single frame at t0=165, clean F, ridge0 -- the ladder
      oracle / v only / C only / both.  Must reproduce refute3_acdfe.json:F_state_oracle
      (0.007777 / 0.022707 / 0.012511 / 0.040426).
  8 : ROUND 5'S CONTROL, T=8 stacked frames, clean F, sigma_F=0, K=0, naive solve -- must reproduce
      round5_solve.json round5_norm_clean|T8|naive med|dE/E| = 0.008562 with the TRUE state (and the
      theta itself, elementwise, so the gauged loopscore 0.9980 measured in round5_score_s0.json is
      inherited rather than re-rolled), and gives the new number with the derived state.  The
      held-out one-frame residual at tick 180 (round 5's acceptance statistic, floor 0.0047) is
      reported for every theta, both with an oracle state and with a DERIVED state at the held-out
      frame -- the acceptance statistic must not be an oracle either.

Nothing in rounds 1-5 is overwritten; every output carries the `state_` prefix.

usage:
  PYTHONPATH=/workspace/Plexus/src python state_derive.py --device cuda:1 --stages d18
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

from recover import theta_scale, install_E, score                # noqa: E402
import crash_test as CT                                          # noqa: E402
from finject import lerp, assemble_inj, y_of                     # noqa: E402
from refute_round3 import advance, fit                           # noqa: E402
from round5_fit import SIGMA_F, SIGMA_X, SNAP                    # noqa: E402
from round5_solve import pstats                                  # noqa: E402


def rel(a, b):
    return float((a - b).norm() / b.norm())


# --------------------------------------------------------------------------------------------- #
#  finject.record_substeps, plus the two diagnostics it does not return: the C the strain operator
#  actually used at each substep, and the plastic ratio.  Token order is IDENTICAL, so F1/x_next
#  are the same numbers finject would have produced (asserted in stage d).
# --------------------------------------------------------------------------------------------- #
def record_substeps_diag(sy, n):
    H, p = sy.H, sy.p
    sy.restore()
    install_E(sy, sy.E_true)
    H.zero_delta()
    H._delta["mpm_particle"] = sy.pass0 + sy.gain_true[sy.cid][:, None] * sy.act0
    H.sub_dt = sy.dt_sub
    C_used, jp_dev = [], 0.0
    F_last = x_last = None
    for _ in range(n):
        C_used.append(p.C.clone())            # the C that THIS strain step integrates F with
        sy._tok("mpm_strain")
        F_last = p.F.clone()
        sy._tok("mpm_scatter")
        sy._tok("mpm_grid_update")
        sy._tok("mpm_gather")
        x_last = p.get("pos").clone()
        jp_dev = max(jp_dev, float((p.Jp - 1.0).abs().max()))
    H.sub_dt = None
    return F_last, x_last, torch.stack(C_used).mean(0), jp_dev


# --------------------------------------------------------------------------------------------- #
def collect(args, t_lo, t_hi, log):
    """Walk the reference trajectory and capture the frame-BOUNDARY series for ticks t_lo..t_hi.

    Returns (system, {tick: record}).  Each record holds the measured pair (x0, F0), the true state
    (v0, C0, Jp0), the full restore snapshot, and the end-of-frame (F1, x_next) used as frame tick's
    observation.  `plant_and_warm(warmup=t_lo)` + `advance` reproduces `plant_and_warm(warmup=t)`
    exactly (round 3 measured x0_matches_history = 0.0).
    """
    a2 = SimpleNamespace(**{**vars(args), "warmup": t_lo})
    sy, _ = CT.plant_and_warm(a2, log)
    n = sy.n_sub_per_frame
    B, cur = {}, t_lo
    for k in range(t_hi - t_lo + 1):
        if k > 0:
            sy.restore()
            advance(sy, cur, cur + 1)
            sy._snapshot(cur + 1)
            cur += 1
        F1, xn, C_eff, jp = record_substeps_diag(sy, n)
        B[cur] = {"tick": cur, "x0": sy.x0.clone(), "F0": sy.F0.clone(),
                  "v0": sy.v0.clone(), "C0": sy.C0.clone(), "Jp0": sy.Jp0.clone(),
                  "F1": F1, "x_next": xn, "C_eff": C_eff, "jp_dev_substeps": jp,
                  "snap": {kk: getattr(sy, kk).clone() for kk in SNAP}}
    return sy, B


def derived_v(B, k, dt, ex=None, stencil="c2"):
    """v at tick k from the MEASURED positions.

    c2  centred 2-point,  O(dt^2) truncation, noise gain sqrt(2)/2  = 0.707
    c4  centred 5-point,  O(dt^4) truncation, noise gain sqrt(130)/12 = 0.950
    sg5 5-point Savitzky-Golay (least-squares slope), O(dt^2) with a larger constant but noise
        gain sqrt(10)/10 = 0.316 -- the noise-optimal member of the family
    """
    def X(t):
        return B[t]["x0"] + (ex[t] if ex is not None else 0.0)

    if stencil == "c2":
        return (X(k + 1) - X(k - 1)) / (2 * dt)
    if stencil == "c4":
        return (-X(k + 2) + 8 * X(k + 1) - 8 * X(k - 1) + X(k - 2)) / (12 * dt)
    if stencil == "sg5":
        return (2 * X(k + 2) + X(k + 1) - X(k - 1) - 2 * X(k - 2)) / (10 * dt)
    raise ValueError(stencil)


def derived_state(B, k, dt, eF=None, ex=None, stencil="c2"):
    """v, C at tick k from the MEASURED boundary series.  eF/ex: {tick: error} measurement noise."""
    def F(t):
        return B[t]["F0"] + (eF[t] if eF is not None else 0.0)

    v = derived_v(B, k, dt, ex, stencil)
    C = ((F(k + 1) - F(k - 1)) / (2 * dt)) @ torch.linalg.inv(F(k))
    Cf = ((F(k + 1) - F(k)) / dt) @ torch.linalg.inv(F(k))          # round 3's forward recipe
    return v, C, Cf


def install_state(sy, snap, v=None, C=None, Jp_one=False):
    """Restore the frame snapshot, then overwrite the parts a microscope cannot supply."""
    for kk, vv in snap.items():
        setattr(sy, kk, vv.clone())
    va, vb = sy.p.state_schema["vel"]
    if v is not None:
        st = sy.state0.clone()
        st[:, va:vb] = v
        sy.state0 = st
        sy.v0 = v.clone()
    if C is not None:
        sy.C0 = C.clone()
    if Jp_one:
        sy.Jp0 = torch.ones_like(sy.Jp0)


# --------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="state_derive")
    ap.add_argument("--stages", default="d18")
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--holdout-tick", type=int, default=180)
    a = ap.parse_args()

    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128,
                           warmup=a.t0, window=150, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "stages": a.stages, "t0": a.t0, "T": a.T,
         "holdout_tick": a.holdout_tick, "sigma_F": SIGMA_F, "sigma_x": SIGMA_X}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        t_lo, t_hi = a.t0 - 2, a.holdout_tick + 2       # +/-2: the 5-point v stencils of stage h
        sy, B = collect(args, t_lo, t_hi, log)
        C, n, dt = sy.C, sy.n_sub_per_frame, sy.dt
        s = theta_scale(C, sy.device)
        th = sy.theta_true.double()
        log(f"[collect] boundaries {t_lo}..{t_hi} ({len(B)}), dt={dt} n_sub={n} "
            f"[{time.time()-t_start:.0f}s]")

        # material masks: the derivation C = Fdot F^-1 is exact per substep ONLY if mpm_strain does
        # nothing to F but F <- (I + dt C) F.  liquid / visco / snow would each break it.
        p = sy.p
        masks = {m: (None if getattr(p, m, None) is None else int(getattr(p, m).sum()))
                 for m in ("is_liquid", "is_visco", "is_snow")}
        R["material_masks"] = masks
        log(f"[spec] {masks}  (all None/0 => F <- (I + dt_sub C) F is the whole strain update)")

        # ---------------------------------------------------------------- stage d ----------- #
        if "d" in a.stages:
            cons = {"max_abs_F1_minus_next_F0": 0.0, "max_abs_xnext_minus_next_x0": 0.0}
            for k in range(t_lo, t_hi):
                cons["max_abs_F1_minus_next_F0"] = max(
                    cons["max_abs_F1_minus_next_F0"], float((B[k]["F1"] - B[k + 1]["F0"]).abs().max()))
                cons["max_abs_xnext_minus_next_x0"] = max(
                    cons["max_abs_xnext_minus_next_x0"],
                    float((B[k]["x_next"] - B[k + 1]["x0"]).abs().max()))
            R["boundary_consistency"] = cons
            log(f"[consistency] F1(k) vs F0(k+1) {cons['max_abs_F1_minus_next_F0']:.2e}; "
                f"x_next(k) vs x0(k+1) {cons['max_abs_xnext_minus_next_x0']:.2e}")

            rows = []
            log(f"\n[d] derivation alone, ZERO noise\n    {'tick':>5s} {'|v0|':>9s} {'|C0|':>9s} "
                f"{'v_ctr':>8s} {'v_back':>8s} {'C_ctr':>8s} {'C_fwd':>8s} {'Cfwd|Ceff':>9s} "
                f"{'Ceff|C0':>8s} {'maxJp-1':>9s}")
            ticks = list(range(a.t0, a.t0 + a.T)) + [a.holdout_tick]
            for k in ticks:
                v, Cc, Cf = derived_state(B, k, dt)
                v_back = (B[k]["x0"] - B[k - 1]["x0"]) / dt
                row = {"tick": k, "norm_v0": float(B[k]["v0"].norm()),
                       "norm_C0": float(B[k]["C0"].norm()),
                       "v_centred": rel(v, B[k]["v0"]), "v_backward": rel(v_back, B[k]["v0"]),
                       "C_centred": rel(Cc, B[k]["C0"]), "C_forward": rel(Cf, B[k]["C0"]),
                       "C_forward_vs_Ceff": rel(Cf, B[k]["C_eff"]),
                       "C_centred_vs_Ceff": rel(Cc, B[k]["C_eff"]),
                       "Ceff_vs_C0": rel(B[k]["C_eff"], B[k]["C0"]),
                       "Jp_max_dev_boundary": float((B[k]["Jp0"] - 1.0).abs().max()),
                       "Jp_max_dev_substeps": B[k]["jp_dev_substeps"],
                       "Jp_rel_err_of_ones": rel(torch.ones_like(B[k]["Jp0"]), B[k]["Jp0"])}
                rows.append(row)
                log(f"    {k:>5d} {row['norm_v0']:>9.3f} {row['norm_C0']:>9.1f} "
                    f"{row['v_centred']:>8.4f} {row['v_backward']:>8.4f} "
                    f"{row['C_centred']:>8.4f} {row['C_forward']:>8.4f} "
                    f"{row['C_forward_vs_Ceff']:>9.2e} {row['Ceff_vs_C0']:>8.4f} "
                    f"{row['Jp_max_dev_substeps']:>9.1e}")
            R["derivation_clean"] = rows

            # noise: is the 28% a TRUNCATION floor (noise-independent) or a noise effect?
            nz = []
            for sF, sx, seed in [(0.0, 0.0, 0), (SIGMA_F, SIGMA_X, 90210), (SIGMA_F, SIGMA_X, 555),
                                 (SIGMA_F, SIGMA_X, 777), (10 * SIGMA_F, 10 * SIGMA_X, 90210),
                                 (0.0, SIGMA_X, 90210), (SIGMA_F, 0.0, 90210)]:
                g = torch.Generator(device=sy.device).manual_seed(seed + 7)
                eF = {t: (sF / 2.0) * torch.randn(B[t]["F0"].shape, generator=g,
                                                  device=sy.device, dtype=sy.dtype) for t in B}
                ex = {t: sx * torch.randn(B[t]["x0"].shape, generator=g,
                                          device=sy.device, dtype=sy.dtype) for t in B}
                vE, cE, cF = [], [], []
                for k in range(a.t0, a.t0 + a.T):
                    v, Cc, Cf = derived_state(B, k, dt, eF, ex)
                    vE.append(rel(v, B[k]["v0"]))
                    cE.append(rel(Cc, B[k]["C0"]))
                    cF.append(rel(Cf, B[k]["C0"]))
                nz.append({"sigma_F": sF, "sigma_x": sx, "seed": seed,
                           "v_centred_mean": float(np.mean(vE)),
                           "C_centred_mean": float(np.mean(cE)),
                           "C_forward_mean": float(np.mean(cF))})
                log(f"    noise sF={sF:<8g} sx={sx:<10g} seed {seed:<6d} -> v {np.mean(vE):.4f}  "
                    f"C_centred {np.mean(cE):.4f}  C_forward {np.mean(cF):.4f}")
            R["derivation_noise"] = nz

            # ---- stage h: is v CHEAP to fix?  three stencils, truncation vs noise ------------ #
            hv = []
            for st in ("c2", "c4", "sg5"):
                clean = [rel(derived_v(B, k, dt, None, st), B[k]["v0"])
                         for k in range(a.t0, a.t0 + a.T)]
                noisy = []
                for seed in (90210, 555, 777):
                    g = torch.Generator(device=sy.device).manual_seed(seed + 7)
                    ex = {t: SIGMA_X * torch.randn(B[t]["x0"].shape, generator=g,
                                                   device=sy.device, dtype=sy.dtype) for t in B}
                    noisy += [rel(derived_v(B, k, dt, ex, st), B[k]["v0"])
                              for k in range(a.t0, a.t0 + a.T)]
                hv.append({"stencil": st, "clean_mean": float(np.mean(clean)),
                           "clean_at_t0": clean[0], "noisy_mean_sigma_x": float(np.mean(noisy))})
                log(f"    v stencil {st:<4s} clean {np.mean(clean):.5f}  "
                    f"at sigma_x {np.mean(noisy):.5f}")
            R["v_stencils"] = hv

        # ---------------------------------------------------------------- stage 1 ----------- #
        if "1" in a.stages:
            log(f"\n[1] ROUND 3 CONTROL: single frame at tick {a.t0}, clean F, ridge0")
            k = a.t0
            v, Cc, Cf = derived_state(B, k, dt)
            v4 = derived_v(B, k, dt, None, "c4")
            injF = lerp(B[k]["F0"], B[k]["F1"], n)
            ladder = [("oracle_state", None, None), ("v_centred_only", v, None),
                      ("C_forward_only(round3)", None, Cf), ("both_round3", v, Cf),
                      ("C_centred_only", None, Cc), ("both_centred", v, Cc),
                      ("v_c4_only", v4, None), ("both_centred_v_c4", v4, Cc)]
            R["stage1"] = {}
            tgt = {"oracle_state": 0.007777332098339839, "v_centred_only": 0.022707019926523922,
                   "C_forward_only(round3)": 0.01251113155512942,
                   "both_round3": 0.04042623408665236}
            log(f"    {'variant':<24s} {'medE':>8s} {'p90E':>8s} {'relL2':>8s} {'negE':>5s} "
                f"{'round3':>9s} {'d':>10s}")
            for nm, vv, CC in ladder:
                install_state(sy, B[k]["snap"], vv, CC, Jp_one=True)
                sc, t_hat = fit(sy, n, injF, B[k]["x_next"], B[k]["x0"], th, C)
                R["stage1"][nm] = sc
                d = (f"{sc['med_E'] - tgt[nm]:+.2e}" if nm in tgt else "")
                log(f"    {nm:<24s} {sc['med_E']:>8.5f} {sc['p90_E']:>8.4f} {sc['rel_l2']:>8.4f} "
                    f"{sc['n_negE']:>5d} {tgt.get(nm, float('nan')):>9.5f} {d:>10s}")
            R["stage1_round3_targets"] = tgt
            R["stage1_reproduced"] = bool(
                all(abs(R["stage1"][kk]["med_E"] - vv) < 1e-6 for kk, vv in tgt.items()))
            log(f"    round-3 ladder reproduced to <1e-6: {R['stage1_reproduced']}")

        # ---------------------------------------------------------------- stage 8 ----------- #
        if "8" in a.stages:
            log(f"\n[8] ROUND 5 CONTROL: T={a.T} stacked frames, clean F (sigma_F=0, K=0), naive")
            variants = {"oracle_state": (False, False, "c2"),
                        "derived_v_only": (True, False, "c2"),
                        "derived_C_only": (False, True, "c2"),
                        "derived_vC": (True, True, "c2")}
            if "h" in a.stages:
                variants["derived_vC_c4"] = (True, True, "c4")
                variants["derived_v_only_c4"] = (True, False, "c4")
            thetas, R["stage8"] = {}, {}
            for nm, (dv, dC, st) in variants.items():
                G0 = torch.zeros(2 * C, 2 * C, device=sy.device, dtype=sy.dtype)
                r0 = torch.zeros(2 * C, device=sy.device, dtype=sy.dtype)
                for k in range(a.t0, a.t0 + a.T):
                    v, Cc, _ = derived_state(B, k, dt, stencil=st)
                    install_state(sy, B[k]["snap"], v if dv else None, Cc if dC else None,
                                  Jp_one=True)
                    A, y0, _ = assemble_inj(sy, n, lerp(B[k]["F0"], B[k]["F1"], n), None)
                    Az = A * s[None, :]
                    b = (B[k]["x_next"] - B[k]["x0"]).reshape(-1) - y0
                    G0 += Az.T @ Az
                    r0 += Az.T @ b
                    del A, Az
                    torch.cuda.empty_cache()
                t_hat = torch.linalg.solve(G0, r0) * s
                thetas[nm] = t_hat
                ps = pstats(t_hat.cpu().numpy(), th.cpu().numpy(), C)
                R["stage8"][nm] = {"pstats": ps}
                log(f"    {nm:<16s} medE {ps['med_E']:>8.5f} p90 {ps['p90_E']:>7.4f} "
                    f"relL2 {ps['rel_l2']:>7.4f} negE {ps['n_negE']:>3d} corr {ps['corr_E']:>6.4f} "
                    f"meanratio {ps['mean_ratio_E']:>6.4f} [{time.time()-t_start:.0f}s]")

            # is the oracle-state theta the SAME VECTOR round 5 scored at gauged 0.9980?
            ref = os.path.join(HERE, "theta_round5.npz")
            if os.path.exists(ref):
                Z = np.load(ref)
                key = "round5_norm_clean|T8|naive"
                if key in Z.files:
                    t5 = torch.as_tensor(Z[key], device=sy.device, dtype=torch.float64)
                    R["stage8"]["oracle_state"]["vs_round5_clean_naive"] = {
                        "rel_l2": rel(thetas["oracle_state"], t5),
                        "max_abs": float((thetas["oracle_state"] - t5).abs().max()),
                        "round5_med_E": 0.008562}
                    log(f"    oracle_state vs round5_norm_clean|T8|naive: rel l2 "
                        f"{rel(thetas['oracle_state'], t5):.3e}")

            # ---- held-out one-frame residual at the holdout tick, clean F ------------------- #
            hk = a.holdout_tick
            injh = lerp(B[hk]["F0"], B[hk]["F1"], n)
            y_obs = (B[hk]["x_next"] - B[hk]["x0"]).reshape(-1)

            def holdout(theta, dv, dC, st="c2"):
                v, Cc, _ = derived_state(B, hk, dt, stencil=st)
                install_state(sy, B[hk]["snap"], v if dv else None, Cc if dC else None,
                              Jp_one=True)
                y = y_of(sy, theta, n, injh, None)
                return float((y - y_obs).norm() / y_obs.norm())

            log(f"\n    held-out one-frame residual, tick {hk}, clean F "
                f"(|y_obs| {float(y_obs.norm()):.4e})")
            log(f"    {'theta':<20s} {'oracleState':>12s} {'derivState':>11s} {'derivState_c4':>14s}")
            for nm in ["theta_true"] + list(variants):
                t_h = th if nm == "theta_true" else thetas[nm]
                ho = holdout(t_h, False, False)
                hd = holdout(t_h, True, True, "c2")
                h4 = holdout(t_h, True, True, "c4")
                R["stage8"].setdefault(nm, {})["holdout"] = {
                    "oracle_state": ho, "derived_state_c2": hd, "derived_state_c4": h4}
                log(f"    {nm:<20s} {ho:>12.5f} {hd:>11.5f} {h4:>14.5f}")

            np.savez(os.path.join(HERE, "state_theta_derive.npz"),
                     **{k: v.cpu().numpy() for k, v in thetas.items()},
                     theta_true=th.cpu().numpy())

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()
