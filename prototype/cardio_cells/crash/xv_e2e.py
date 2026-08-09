"""xv_e2e.py -- ROUND 6, TASK 2.  THE WINNER, END TO END: T-frame stack, realizable measurement
noise, and a free 150-frame rollout.

WHAT IS BEING RUN
====================================================================================================
Task 1's extended ladder ended with one realizable recipe:

    {F_lerp, C_lerp}   inject the MEASURED deformation gradient at all 10 substeps of a frame and
                       the DERIVED affine velocity matrix at substeps 0..8; do NOT inject x or v
                       (they are the read-out's own integrators -- injecting them deletes the
                       theta-carried signal: carried fraction 0.073 -> 0.012 for x, and under the
                       recording's own sigma_x the perfect x-injection is 5.5x WORSE than none).

and one honest qualification: no realizable rung reached the 0.0078 oracle, because a microscope
has to derive the frame-start v0 (1.0 % wrong) and C0 (13.8 % wrong) as well.  Everything here is
built on that recipe with NO state oracle anywhere in the estimator:

    x   measured at every frame boundary          (+ sigma_x)
    F   measured at every frame boundary          (+ sigma_F, SPATIALLY CORRELATED)
    v0  centred difference of the MEASURED x      -> 2.6 % wrong at sigma_x
    C0  centred (Fdot F^-1) on the MEASURED F     -> 27 % wrong at sigma_F
    Jp  = 1                                       (measured: max|Jp-1| = 0 over the window)

THE THREE THINGS TASK 2 ADDS
----------------------------------------------------------------------------------------------------
1. T FRAMES STACKED.  G = sum_k A_k' A_k, r = sum_k A_k' b_k, reported for T = 1, 2, 4, 8 out of one
   assembly pass.  Four solvers per T: naive, naive_box, eiv_snr0, eiv_box (round5_solve.py,
   imported unmodified) -- the box is the ONE prior (per-cell moduli within 25x of each other, read
   off the naive estimate, no truth) and the EIV correction is round 4's Monte-Carlo de-biasing.

2. REALIZABLE NOISE.  sigma_x = 2.00e-05 world on the positions and sigma_F = 3.9e-3 on the
   derivative channels, ONE COHERENT draw per frame BOUNDARY shared by the two frames that use it,
   and the F draw is SPATIALLY CORRELATED on a node grid (refute5_fit.NoiseF, --noise grid
   --nodes 48) so that a cell carries ~23 independent F samples, not 100.  White-per-particle F
   noise is the wrong model and is run only as a labelled control.
   The noise propagates everywhere a microscope's would: into the injected F series, into the
   DERIVED C series (both the injected one and the frame-start C0), into the DERIVED v0 through the
   position stencil, and into the observed displacement.
   Two observation conventions are carried side by side at zero extra cost (they share A):
       obs=end    b = (x_meas[k+1] - x_true[k]) - y0    round 5's convention, noise at one end
       obs=both   b = (x_meas[k+1] - x_meas[k]) - y0    honest, sqrt(2) more displacement noise
   The model is still INITIALISED at the true configuration -- that idealisation cannot be removed
   without changing the system being fitted, and it is stated rather than hidden.

3. THE CRASH TEST.  Free 150-frame rollout (anchor=None) from tick 165, read on the registry's
   10x10 grid at MARGIN_SAFE = 20, scored with discovery_cardio_mpm/metrics.py imported unmodified,
   with round 5's DETERMINISTIC fixed-budget 2-D gauge (5x5 log-grid + 2 Broyden steps, 28 rollouts
   per candidate, the loopscore spread over the gauge-satisfying cells reported as the gauge's own
   uncertainty).  Ceiling = theta_true, floor = a blind constant, plus the zero-information nulls.

ACCEPTANCE (fixed before the run)
----------------------------------------------------------------------------------------------------
  PASS   held-out one-frame residual <= 0.06,  gauged loopscore >= 0.85,  0 negative moduli
  STOP   held-out > 0.19 or med|dE/E| > 0.45  -- frame cadence carries no per-cell content on
         anything the recording can supply, and that is the most valuable outcome available.
The acceptance statistic is the HELD-OUT residual, never med|dE/E| (round 5: a vector 45 % random
has med|dE/E| = 0 by construction and scores below the zero-information band).

usage:
  PYTHONPATH=/workspace/Plexus/src python xv_e2e.py --device cuda:1 --stages ASRP
  PYTHONPATH=/workspace/Plexus/src python xv_e2e.py --device cuda:1 --stages SP --tag xv_e2e
"""
from __future__ import annotations

import argparse
import json
import math
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

from recover import theta_scale, install_E, score                 # noqa: E402
import metrics as MET                                             # noqa: E402
import crash_test as CT                                           # noqa: E402
from crash_round2 import percell_amplitude, r2_percell            # noqa: E402
from crash_round3 import scale2, t2_of                            # noqa: E402
from assemble import SUBSTEP_TOKENS                               # noqa: E402
from finject import lerp                                          # noqa: E402
from state_derive import collect, install_state                   # noqa: E402
from xv_ladder import assemble_xv, y_of_xv                        # noqa: E402
from refute5_fit import NoiseF                                    # noqa: E402
from round5_fit import SIGMA_F, SIGMA_X, SNAP                     # noqa: E402
from round5_solve import solve_box, snr_trunc, pstats             # noqa: E402
from round5_score import gauge_grid                               # noqa: E402

RUNGS = {"FC": ("F", "C"), "F": ("F",)}


# --------------------------------------------------------------------------------------------- #
#  the measurement: one coherent draw per frame BOUNDARY, shared by the frames that use it
# --------------------------------------------------------------------------------------------- #
class Measurement:
    """The microscope.  Holds the noisy boundary series and everything derived from it.

    B[t]["F1"] == B[t+1]["F0"] and B[t]["x_next"] == B[t+1]["x0"] to 0.0 (state_derive stage d), so
    a per-TICK draw is automatically coherent between the two frames that share a boundary.
    """

    def __init__(self, B, dt, NF, sigma_F, sigma_x, gen, device, dtype):
        self.B, self.dt = B, dt
        self.ticks = sorted(B)
        self.eF = {t: (sigma_F / 2.0) * NF(gen) for t in self.ticks}
        self.ex = {t: sigma_x * torch.randn(B[t]["x0"].shape, generator=gen, device=device,
                                            dtype=dtype) for t in self.ticks}
        self.F = {t: B[t]["F0"] + self.eF[t] for t in self.ticks}
        self.x = {t: B[t]["x0"] + self.ex[t] for t in self.ticks}
        self._C = {}

    def v(self, k):
        return (self.x[k + 1] - self.x[k - 1]) / (2 * self.dt)

    def C(self, k, F=None):
        if F is None:
            if k not in self._C:
                self._C[k] = self.C(k, self.F)
            return self._C[k]
        return ((F[k + 1] - F[k - 1]) / (2 * self.dt)) @ torch.linalg.inv(F[k])

    def renoise(self, k, NF, sigma_F, gen):
        """A fresh coherent F error on top of the already-noisy measurement, on the ticks frame k
        touches (k-1 .. k+2: the injected F_lerp needs k, k+1 and the derived C needs k-1 .. k+2)."""
        return {t: self.F[t] + (sigma_F / 2.0) * NF(gen) for t in range(k - 1, k + 3)}


def build_inj(M, k, n, rung, F=None):
    """The injected series for frame k under `rung`, from a (possibly re-noised) F dictionary."""
    F = M.F if F is None else F
    inj = {"F": lerp(F[k], F[k + 1], n)}
    if "C" in rung:
        inj["C"] = lerp(M.C(k, F), M.C(k + 1, F), n)
    return inj


# --------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="xv_e2e")
    ap.add_argument("--stages", default="ASRP")   # A=assemble S=solve R=rollout P=plot
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--K", type=int, default=6, help="Monte-Carlo re-noisings for the EIV Sigma")
    ap.add_argument("--holdout-tick", type=int, default=180)
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--noise", default="grid", choices=("indep", "grid", "gridsm"))
    ap.add_argument("--nodes", type=int, default=48)
    ap.add_argument("--sigma-F", type=float, default=SIGMA_F)
    ap.add_argument("--sigma-x", type=float, default=SIGMA_X)
    ap.add_argument("--seed", type=int, default=90210)
    ap.add_argument("--rungs", default="FC,F")
    ap.add_argument("--obs", default="both", choices=("both", "end"))
    ap.add_argument("--grid-gauge", type=int, default=5)
    ap.add_argument("--refine", type=int, default=2)
    ap.add_argument("--roll", default="")
    a = ap.parse_args()

    args = SimpleNamespace(device=a.device, cells=a.cells, per_parent=a.per_parent,
                           n_grid=a.n_grid, warmup=a.t0, window=a.window, dtype="float64",
                           mode="full", e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "cli": vars(a), "sigma_F": a.sigma_F, "sigma_x": a.sigma_x}
    t_start = time.time()
    torch.manual_seed(0)
    rung_names = [r for r in a.rungs.split(",") if r]
    NPZ = os.path.join(HERE, f"{a.tag}_norm.npz")
    THZ = os.path.join(HERE, f"{a.tag}_theta.npz")

    with torch.no_grad():
        t_lo, t_hi = a.t0 - 2, a.holdout_tick + 2
        sy, B = collect(args, t_lo, t_hi, log)
        C, n, dt = sy.C, sy.n_sub_per_frame, sy.dt
        th = sy.theta_true.double()
        s = theta_scale(C, sy.device)
        dev, f64 = sy.device, torch.float64
        x0_165, cid = B[a.t0]["x0"].clone(), sy.cid
        log(f"[collect] boundaries {t_lo}..{t_hi}, C={C} Np={sy.Np} n_sub={n} dt={dt} "
            f"[{time.time()-t_start:.0f}s]")

        NF = NoiseF(a.noise, sy.x0, a.nodes, dev, sy.dtype)
        gm = torch.Generator(device=dev).manual_seed(a.seed)
        gk = torch.Generator(device=dev).manual_seed(31337 + a.seed)
        M = Measurement(B, dt, NF, a.sigma_F, a.sigma_x, gm, dev, sy.dtype)

        # ---- what the noise model actually delivers per cell (refute5's control) ------------- #
        gchk = torch.Generator(device=dev).manual_seed(7)
        cstd = []
        for _ in range(24):
            e = NF(gchk).reshape(-1, 4)
            mm = torch.zeros(C + 1, 4, device=dev, dtype=sy.dtype)
            mm.index_add_(0, cid, e)
            cnt = torch.zeros(C + 1, device=dev, dtype=sy.dtype)
            cnt.index_add_(0, cid, torch.ones_like(e[:, 0]))
            cstd.append(float((mm[1:] / cnt[1:, None]).std()))
        neff = float(1.0 / np.mean(cstd) ** 2)
        R["noise_control"] = {"mode": a.noise, "nodes": a.nodes,
                              "effective_independent_F_samples_per_cell": neff}
        log(f"[noise {a.noise}{a.nodes}] effective independent F samples per cell {neff:.1f} "
            f"(white-per-particle would be 100; the recording gives 22.8)")

        # ---- how wrong the derived state is, at this noise ------------------------------------ #
        dsr = []
        for k in list(range(a.t0, a.t0 + a.T)) + [a.holdout_tick]:
            dsr.append({"tick": k,
                        "v_rel": float((M.v(k) - B[k]["v0"]).norm() / B[k]["v0"].norm()),
                        "C_rel": float((M.C(k) - B[k]["C0"]).norm() / B[k]["C0"].norm()),
                        "F_rel": float(M.eF[k].norm() / B[k]["F0"].norm()),
                        "x_rel_dx": float(M.ex[k].norm(dim=1).mean() / sy.g.dx)})
        R["derived_state_error"] = dsr
        log(f"[state] derived from the NOISY measurement: v {np.mean([d['v_rel'] for d in dsr]):.4f}"
            f"  C {np.mean([d['C_rel'] for d in dsr]):.4f}  (clean-measurement values were "
            f"0.0100 / 0.1381)   |eF|/|F| {np.mean([d['F_rel'] for d in dsr]):.2e}  "
            f"|ex| {np.mean([d['x_rel_dx'] for d in dsr]):.4f} dx")

        # ============================================================= A: ASSEMBLE ============ #
        if "A" in a.stages:
            out = {}
            log(f"\n[A] stacked normal equations, T={a.T} frames from tick {a.t0}, K={a.K} "
                f"re-noisings, rungs {rung_names}")
            for rung in rung_names:
                for j in range(a.T):
                    k = a.t0 + j
                    tj = time.time()
                    install_state(sy, B[k]["snap"], M.v(k), M.C(k), Jp_one=True)
                    A, y0 = assemble_xv(sy, n, build_inj(M, k, n, RUNGS[rung]))
                    Az = A * s[None, :]
                    d_end = (M.x[k + 1] - B[k]["x0"]).reshape(-1)
                    d_both = (M.x[k + 1] - M.x[k]).reshape(-1)
                    G = Az.T @ Az
                    out[f"{rung}|G{j}"] = G.cpu().numpy()
                    out[f"{rung}|rend{j}"] = (Az.T @ (d_end - y0)).cpu().numpy()
                    out[f"{rung}|rboth{j}"] = (Az.T @ (d_both - y0)).cpu().numpy()
                    del A, Az
                    torch.cuda.empty_cache()
                    Gm = torch.zeros_like(G)
                    rm_e = torch.zeros(2 * C, device=dev, dtype=f64)
                    rm_b = torch.zeros_like(rm_e)
                    for _ in range(a.K):
                        F2 = M.renoise(k, NF, a.sigma_F, gk)
                        install_state(sy, B[k]["snap"], M.v(k), M.C(k, F2), Jp_one=True)
                        A2, y02 = assemble_xv(sy, n, build_inj(M, k, n, RUNGS[rung], F2))
                        Az2 = A2 * s[None, :]
                        Gm += Az2.T @ Az2
                        rm_e += Az2.T @ (d_end - y02)
                        rm_b += Az2.T @ (d_both - y02)
                        del A2, Az2
                        torch.cuda.empty_cache()
                    if a.K > 0:
                        Gm, rm_e, rm_b = Gm / a.K, rm_e / a.K, rm_b / a.K
                    out[f"{rung}|Gm{j}"] = Gm.cpu().numpy()
                    out[f"{rung}|rmend{j}"] = rm_e.cpu().numpy()
                    out[f"{rung}|rmboth{j}"] = rm_b.cpu().numpy()
                    log(f"    {rung:<3s} frame {j} (tick {k}) + {a.K} re-noisings "
                        f"[{time.time()-tj:.0f}s / {time.time()-t_start:.0f}s]")
            out["theta_true"] = th.cpu().numpy()
            out["s"] = s.cpu().numpy()
            np.savez(NPZ, **out)
            log(f"    wrote {os.path.basename(NPZ)}")

        # ============================================================= S: SOLVE =============== #
        # the held-out protocol.  FIXED across every candidate so the numbers are comparable, and
        # reported in both flavours: fully realizable (noisy F, derived state, noisy observation)
        # and round-5-comparable (clean F, oracle state, clean observation, floor 0.00474).
        hk = a.holdout_tick
        yh_clean = (B[hk]["x_next"] - B[hk]["x0"]).reshape(-1)
        yh_real = (M.x[hk + 1] - M.x[hk]).reshape(-1) if a.obs == "both" else \
                  (M.x[hk + 1] - B[hk]["x0"]).reshape(-1)
        inj_clean = {"F": lerp(B[hk]["F0"], B[hk]["F1"], n)}
        inj_real = {"F": lerp(M.F[hk], M.F[hk + 1], n)}

        def holdout(theta):
            install_state(sy, B[hk]["snap"], None, None, Jp_one=True)
            yo = y_of_xv(sy, theta, n, inj_clean)
            install_state(sy, B[hk]["snap"], M.v(hk), M.C(hk), Jp_one=True)
            yr = y_of_xv(sy, theta, n, inj_real)
            return {"clean_oracle": float((yo - yh_clean).norm() / yh_clean.norm()),
                    "realizable": float((yr - yh_real).norm() / yh_real.norm())}

        if "S" in a.stages:
            Z = np.load(NPZ)
            rk = "rboth" if a.obs == "both" else "rend"
            rmk = "rmboth" if a.obs == "both" else "rmend"
            log(f"\n[S] solve.  observation convention = {a.obs}  "
                f"({'x_meas[k+1] - x_meas[k]' if a.obs=='both' else 'x_meas[k+1] - x_true[k]'})")

            # zero-information band and floor for the acceptance statistic
            gp = torch.Generator(device=dev).manual_seed(1234)
            th_perm = torch.cat([th[:C][torch.randperm(C, generator=gp, device=dev)],
                                 th[C:][torch.randperm(C, generator=gp, device=dev)]])
            th_mean = torch.cat([th[:C].mean().expand(C), th[C:].mean().expand(C)])
            R["holdout_band"] = {}
            log(f"    held-out tick {hk}: floor and zero-information band "
                f"(|y| clean {float(yh_clean.norm()):.4e})")
            for nm, tt in (("theta_true (floor)", th), ("permuted theta", th_perm),
                           ("block-mean theta", th_mean), ("theta = 0", torch.zeros_like(th))):
                hh = holdout(tt)
                R["holdout_band"][nm] = hh
                log(f"      {nm:<22s} cleanF/oracleState {hh['clean_oracle']:.5f}   "
                    f"REALIZABLE {hh['realizable']:.5f}")

            R["Tcurve"], thetas = {}, {}
            log(f"\n    {'rung':<4s} {'T':>2s} {'solver':<10s} {'medE':>7s} {'p90':>7s} "
                f"{'maxE':>8s} {'neg':>4s} {'>5x':>4s} {'relL2':>7s} {'corr':>6s} {'mE/E':>6s} "
                f"{'ho_real':>8s} {'ho_orcl':>8s}")
            for rung in rung_names:
                R["Tcurve"][rung] = {}
                for T in (1, 2, 4, 8):
                    if T > a.T:
                        continue
                    G0 = sum(torch.as_tensor(Z[f"{rung}|G{j}"], device=dev, dtype=f64)
                             for j in range(T))
                    r0 = sum(torch.as_tensor(Z[f"{rung}|{rk}{j}"], device=dev, dtype=f64)
                             for j in range(T))
                    Gb = sum(torch.as_tensor(Z[f"{rung}|Gm{j}"], device=dev, dtype=f64)
                             for j in range(T))
                    rb = sum(torch.as_tensor(Z[f"{rung}|{rmk}{j}"], device=dev, dtype=f64)
                             for j in range(T))
                    has_mc = float(Gb.abs().max()) > 0
                    Sig = (Gb - G0) if has_mc else torch.zeros_like(G0)
                    Gc = G0 - Sig
                    rc = r0 - ((rb - r0) if has_mc else torch.zeros_like(r0))
                    solv = {}
                    try:
                        solv["naive"] = torch.linalg.solve(G0, r0) * s
                    except Exception:
                        solv["naive"] = torch.linalg.lstsq(
                            G0, r0.unsqueeze(1)).solution.squeeze(1) * s
                    if has_mc:
                        solv["eiv_snr0"], _ = snr_trunc(G0, Sig, Gc, rc, s, tau=0.0)
                    nv = solv["naive"]
                    mE = float(nv[:C][nv[:C] > 0].median()) if int((nv[:C] > 0).sum()) \
                        else float(nv[:C].abs().median())
                    mg = float(nv[C:][nv[C:] > 0].median()) if int((nv[C:] > 0).sum()) \
                        else float(nv[C:].abs().median())
                    lo = torch.cat([torch.full((C,), 0.2 * mE, device=dev, dtype=f64),
                                    torch.full((C,), 0.2 * mg, device=dev, dtype=f64)])
                    hi = torch.cat([torch.full((C,), 5.0 * mE, device=dev, dtype=f64),
                                    torch.full((C,), 5.0 * mg, device=dev, dtype=f64)])
                    solv["naive_box"], _ = solve_box(G0, r0, s, lo, hi,
                                                     z0=torch.clamp(nv, lo, hi) / s)
                    if has_mc:
                        solv["eiv_box"], _ = solve_box(
                            Gc, rc, s, lo, hi, z0=torch.clamp(solv["eiv_snr0"], lo, hi) / s)
                    row = {"box_E": [0.2 * mE, 5.0 * mE], "box_g": [0.2 * mg, 5.0 * mg],
                           "min_eig_G0": float(torch.linalg.eigvalsh(G0).min()),
                           "min_eig_Gc": float(torch.linalg.eigvalsh(Gc).min()),
                           "sigma_fro_over_G_fro": float(Sig.norm() / G0.norm()), "solvers": {}}
                    for nm, t_h in solv.items():
                        ps = pstats(t_h.cpu().numpy(), th.cpu().numpy(), C)
                        hh = holdout(t_h)
                        row["solvers"][nm] = {"param": ps, "holdout": hh}
                        thetas[f"{rung}|T{T}|{nm}"] = t_h.cpu().numpy()
                        log(f"    {rung:<4s} {T:>2d} {nm:<10s} {ps['med_E']:>7.4f} "
                            f"{ps['p90_E']:>7.3f} {ps['max_E']:>8.3f} {ps['n_negE']:>4d} "
                            f"{ps['n_cells_relE_gt5']:>4d} {ps['rel_l2']:>7.3f} "
                            f"{ps['corr_E']:>6.3f} {ps['mean_ratio_E']:>6.3f} "
                            f"{hh['realizable']:>8.4f} {hh['clean_oracle']:>8.4f}")
                    R["Tcurve"][rung][f"T{T}"] = row
                log("")
            thetas["theta_true"] = th.cpu().numpy()
            np.savez(THZ, **thetas)
            R["thetas"] = sorted(thetas)
            log(f"    wrote {os.path.basename(THZ)} [{time.time()-t_start:.0f}s]")

        # ============================================================= R: ROLLOUT ============= #
        if "R" in a.stages:
            Zt = np.load(THZ)
            G = a.window
            tracers = {m: CT.tracer_indices(x0_165, CT.probe_points(m))
                       for m in (MET.MARGIN_SAFE, MET.MARGIN_INHERITED)}
            band = 0.06 / MET.SHEET_SPAN
            anchor = ((x0_165[:, 0] < band) | (x0_165[:, 0] > 1 - band) |
                      (x0_165[:, 1] < band) | (x0_165[:, 1] > 1 - band))
            interior = ~anchor
            snap0 = {kk: B[a.t0]["snap"][kk].clone() for kk in SNAP}

            ref_full = torch.zeros(G, sy.Np, 2, device=dev, dtype=sy.dtype)
            install_state(sy, snap0)
            sy.restore()
            install_E(sy, sy.E_true)
            for kf in range(G):
                sy._outer(a.t0 + kf, gain_cell=sy.gain_true)
                sy.H.sub_dt = sy.dt_sub
                for _ in range(n):
                    for tok in SUBSTEP_TOKENS:
                        sy._tok(tok)
                sy.H.sub_dt = None
                ref_full[kf] = sy.p.get("pos")
            d_ref = ref_full - x0_165[None]
            dm = d_ref[:, interior].mean(0, keepdim=True)
            ss_tot = (d_ref[:, interior] - dm).pow(2).sum()
            real20 = ref_full[:, tracers[MET.MARGIN_SAFE]].cpu().numpy()
            a_ref, _ = percell_amplitude(ref_full, x0_165, cid, C, interior)
            keep = np.isfinite(a_ref) & (a_ref > 0)
            log(f"\n[R] reference window built, {G} frames from tick {a.t0} "
                f"[{time.time()-t_start:.0f}s]")

            def scored(theta, full_out=True):
                install_state(sy, snap0)
                tr, full, coarse = CT.rollout(sy, theta, a.t0, G, tracers, ref_full=ref_full,
                                              anchor=None, interior=interior, ss_tot=ss_tot,
                                              keep_full=full_out, band_mask=anchor)
                m20 = CT.read_metrics(tr[MET.MARGIN_SAFE].cpu().numpy(), real20)
                out = {"loop": m20["loopscore"], "t1": coarse["motion_energy_ratio_interior"],
                       "t2": t2_of(m20), "R2": coarse["R2_displacement_interior"],
                       "rms_dx_mean": coarse["rms_pos_err_dx_mean"]}
                if full_out:
                    ah, _ = percell_amplitude(full, x0_165, cid, C, interior)
                    out.update({"margin20": m20, "coarse": coarse,
                                "percell": r2_percell(ah, a_ref, keep)})
                    del full
                return out

            def const(E, g):
                return torch.cat([torch.full((C,), float(E), device=dev, dtype=f64),
                                  torch.full((C,), float(g), device=dev, dtype=f64)])

            cands = []
            want = [w for w in a.roll.split(",") if w] or [
                "FC|T8|eiv_box", "F|T8|eiv_box", "FC|T8|naive_box", "FC|T1|eiv_box"]
            cands.append(("theta_true", th))
            for w in want:
                if w in Zt.files:
                    cands.append((w, torch.as_tensor(Zt[w], device=dev, dtype=f64)))
            # the floor and the zero-information bank
            mEd = float(torch.as_tensor(Zt["FC|T8|eiv_box"], device=dev, dtype=f64)[:C].median()) \
                if "FC|T8|eiv_box" in Zt.files else 130.0
            cands.append(("blind_const", const(mEd, 1.0)))
            gg = torch.Generator().manual_seed(303)
            cands.append(("null_prior_draw", torch.cat([
                (args.e_lo + (args.e_hi - args.e_lo) * torch.rand(C, generator=gg)).to(dev, f64),
                (args.g_lo + (args.g_hi - args.g_lo) * torch.rand(C, generator=gg)).to(dev, f64)])))
            gn2 = torch.Generator().manual_seed(4242)
            idx = torch.randperm(C, generator=gn2)[:45].to(dev)
            Ed = (args.e_lo + (args.e_hi - args.e_lo) * torch.rand(C, generator=gn2,
                                                                   dtype=f64)).to(dev)
            gd = (args.g_lo + (args.g_hi - args.g_lo) * torch.rand(C, generator=gn2,
                                                                   dtype=f64)).to(dev)
            nm0 = th.clone()
            nm0[idx], nm0[C + idx] = Ed[idx], gd[idx]
            cands.append(("null_med0_rand45", nm0))

            R["nulls_array_only"] = {}
            frozen = np.repeat(x0_165[tracers[MET.MARGIN_SAFE]].cpu().numpy()[None], G, axis=0)
            R["nulls_array_only"]["do_nothing"] = CT.read_metrics(frozen, real20)
            log(f"    null do_nothing loopscore "
                f"{CT.fmt(R['nulls_array_only']['do_nothing']['loopscore'],8)}")

            R["rollout"] = {}
            log(f"\n[R] free {G}-frame rollouts, anchor=None, margin-{MET.MARGIN_SAFE}, "
                f"deterministic {a.grid_gauge}x{a.grid_gauge} gauge + {a.refine} refine")
            log(f"    {'candidate':<20s} {'medE':>7s} {'neg':>4s} {'mE/E':>6s} {'ho_real':>8s} "
                f"| {'raw':>8s} {'kE':>6s} {'kg':>6s} {'gauged':>8s} {'+-':>5s} {'R2':>8s} "
                f"{'r2cell':>7s} {'rms/dx':>7s}")
            for name, theta in cands:
                tc = time.time()
                ps = pstats(theta.cpu().numpy(), th.cpu().numpy(), C)
                hh = holdout(theta)
                raw = scored(theta)

                def probe(lE, lg, theta=theta):
                    return scored(scale2(theta, math.exp(lE), math.exp(lg), C), full_out=False)

                gf = gauge_grid(probe, (raw["t1"], raw["t2"]), raw["loop"],
                                gn=a.grid_gauge, refine=a.refine)
                kE, kg = gf["k_E"], gf["k_g"]
                gau = raw if (abs(kE - 1) < 1e-12 and abs(kg - 1) < 1e-12) \
                    else scored(scale2(theta, kE, kg, C))
                R["rollout"][name] = {
                    "param": ps, "holdout": hh,
                    "param_after_gauge": pstats(scale2(theta, kE, kg, C).cpu().numpy(),
                                                th.cpu().numpy(), C),
                    "raw": raw, "gauged": gau,
                    "gauge": {kk: vv for kk, vv in gf.items() if kk != "cells"},
                    "seconds": time.time() - tc}
                log(f"    {name:<20s} {ps['med_E']:>7.4f} {ps['n_negE']:>4d} "
                    f"{ps['mean_ratio_E']:>6.3f} {hh['realizable']:>8.4f} | "
                    f"{CT.fmt(raw['loop'],8)} {kE:>6.3f} {kg:>6.3f} {CT.fmt(gau['loop'],8)} "
                    f"{gf['gauge_uncertainty']:>5.3f} {CT.fmt(gau['R2'],8)} "
                    f"{(gau['percell']['r2'] if gau['percell']['r2'] is not None else float('nan')):>7.4f} "
                    f"{gau['rms_dx_mean']:>7.4f}  [{time.time()-tc:.0f}s]")

        # ============================================================= P: PLOT ================ #
        if "P" in a.stages:
            make_figure(os.path.join(HERE, f"{a.tag}.png"), THZ, R, C, a, log)

        # ---- the verdict --------------------------------------------------------------------- #
        if "S" in a.stages:
            win = R.get("Tcurve", {}).get("FC", {}).get(f"T{a.T}", {}).get("solvers", {})
            key = "eiv_box" if "eiv_box" in win else ("naive_box" if "naive_box" in win else None)
            if key:
                w = win[key]
                ho = w["holdout"]["realizable"]
                med = w["param"]["med_E"]
                neg = w["param"]["n_negE"]
                gl = R.get("rollout", {}).get("FC|T8|eiv_box", {}).get("gauged", {}).get("loop")
                verdict = {"held_out_realizable": ho,
                           "held_out_clean_oracle": w["holdout"]["clean_oracle"],
                           "med_E": med, "n_negE": neg, "gauged_loopscore": gl,
                           "PASS_holdout": bool(ho <= 0.06), "PASS_negE": bool(neg == 0),
                           "PASS_loop": (bool(gl >= 0.85) if isinstance(gl, float) else None),
                           "STOP": bool(ho > 0.19 or med > 0.45)}
                verdict["ACCEPT"] = bool(verdict["PASS_holdout"] and verdict["PASS_negE"]
                                         and verdict["PASS_loop"] is True)
                R["verdict"] = verdict
                log(f"\n[verdict] winner FC|T{a.T}|{key}: held-out(realizable) {ho:.4f} "
                    f"(<=0.06 {verdict['PASS_holdout']}), med|dE/E| {med:.4f}, negE {neg} "
                    f"(=0 {verdict['PASS_negE']}), gauged loopscore "
                    f"{gl if gl is None else round(gl,4)} (>=0.85 {verdict['PASS_loop']}) "
                    f"-> ACCEPT {verdict['ACCEPT']}  STOP {verdict['STOP']}")

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


# --------------------------------------------------------------------------------------------- #
def make_figure(path, THZ, R, C, a, log):
    """Recovered vs planted E for the winning rung and for F alone, plus the T curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Z = np.load(THZ)
    th = Z["theta_true"]
    E = th[:C]
    panels = [("FC|T8|eiv_box", "{F_lerp, C_lerp}"), ("F|T8|eiv_box", "F_lerp only")]
    panels = [(k, lbl) for k, lbl in panels if k in Z.files]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.1 * (len(panels) + 1), 4.0))
    lim = (0, 1.05 * max(E.max(), max(float(Z[k][:C].max()) for k, _ in panels)))
    for ax, (k, lbl) in zip(axes, panels):
        Eh = Z[k][:C]
        ps = pstats(Z[k], th, C)
        ax.plot(lim, lim, color="0.35", lw=1.0, zorder=1)
        ax.scatter(E, Eh, s=22, facecolor="#3b6ea5", edgecolor="none", alpha=0.85, zorder=2)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xlabel("planted E")
        ax.set_ylabel("recovered E")
        ax.text(0.03, 0.97, f"{lbl}\nmed|dE/E| {ps['med_E']:.3f}   r {ps['corr_E']:.3f}\n"
                            f"mean ratio {ps['mean_ratio_E']:.3f}   neg {ps['n_negE']}",
                transform=ax.transAxes, va="top", ha="left", fontsize=10)
    ax = axes[-1]
    for rung, col, mk in (("FC", "#3b6ea5", "o"), ("F", "#a53b3b", "s")):
        Ts, ys = [], []
        for T in (1, 2, 4, 8):
            d = R.get("Tcurve", {}).get(rung, {}).get(f"T{T}")
            if d and "eiv_box" in d["solvers"]:
                Ts.append(T)
                ys.append(d["solvers"]["eiv_box"]["param"]["med_E"])
        if Ts:
            ax.plot(Ts, ys, marker=mk, color=col, label=f"{rung}, eiv_box")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("T frames stacked")
    ax.set_ylabel("med|dE/E|")
    ax.legend(fontsize=9, frameon=False)
    for i, ax in enumerate(axes):
        ax.text(-0.14, 1.04, "abc"[i], transform=ax.transAxes, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"[P] wrote {os.path.basename(path)}")


if __name__ == "__main__":
    main()
