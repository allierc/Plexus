"""TASK F -- adversarial probes of the TASK D 'planted recovery' claim.

Only files created: this one + refute_f/*.json.  Nothing in assemble.py / recover.py is edited.

PROBES
  P1  CIRCULARITY.  Re-run the identical pipeline but warm the trajectory up with a DECOY theta
      while theta_true (the thing 'recovered') is installed only at the frozen state.  If the
      recovery number is unchanged, the number says nothing about the trajectory.
  P2  THE FD IDENTITY.  a_fd == a_solver is claimed as evidence that 'b is built from positions
      only'.  Check whether it still holds when the warm-up used a different theta, and when the
      frozen state is deliberately corrupted -- i.e. whether it is an identity of the advection
      rule rather than a physical check.
  P3  WRONG-BUT-LINEAR MODEL.  Assemble A from a model that is wrong but still linear in theta
      (a) wrong wall model (wall_damp 0.5 -> 1.0),  (b) wrong deformation gradient F (eps rel.),
      and solve against the TRUE b.  A wrong-but-linear model that still returns theta would mean
      the fit is not testing the physics.
  P4  SCALE.  Is rel_l2 (used for every 'beats the null' comparison) actually sensitive to gain?
  P5  TAIL.  At zero noise, what fraction of cells is recovered badly?  (median hides it.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/workspace/Plexus/src")
sys.path.insert(0, "/workspace/Plexus/prototype/cardio_cells/algebraic")

from assemble import SUBSTEP_TOKENS, System, rel            # noqa: E402
from recover import Solver, fd_accel, install_E, score      # noqa: E402

OUT = "/workspace/Plexus/prototype/cardio_cells/algebraic/refute_f"


def draw(C, lo, hi, seed, dev, dt):
    g = torch.Generator().manual_seed(seed)
    return (lo + (hi - lo) * torch.rand(C, generator=g)).to(dev, dt)


def build(args, warmup, E_warm, g_warm, E_true, g_true, log):
    """Warm up with (E_warm, g_warm); freeze; declare theta_true = (E_true, g_true)."""
    sy = System(device=args.device, n_cells=args.cells, per_parent=args.per_parent,
                n_grid=args.n_grid, warmup=0, dtype=args.dtype, mode="full", real=args.real)
    C = sy.C
    sy.E_true[1:] = E_warm[:C]
    sy.gain_true[1:] = g_warm[:C]
    install_E(sy, sy.E_true)
    x_prev = None
    for tick in range(warmup):
        sy._outer(tick, gain_cell=sy.gain_true)
        sy.H.sub_dt = sy.dt_sub
        for s in range(sy.n_sub_per_frame):
            if tick == warmup - 1 and s == sy.n_sub_per_frame - 1:
                x_prev = sy.p.get("pos").clone()
            for tok in SUBSTEP_TOKENS:
                sy._tok(tok)
        sy.H.sub_dt = None
    # now DECLARE the parameters that the frozen-state constraint will be written at
    sy.E_true[1:] = E_true[:C]
    sy.gain_true[1:] = g_true[:C]
    sy.theta_true = torch.cat([sy.E_true[1:], sy.gain_true[1:]])
    install_E(sy, sy.E_true)
    sy.warmup_frames = warmup
    sy._snapshot(warmup)
    sy.x_prev = x_prev
    log(f"   built: C={sy.C} Np={sy.Np} grid {sy.g.nx}^2 warmup={warmup}")
    return sy


def run_one(sy, tag, log, extra_A=None):
    """assemble at the frozen state, b from positions, solve.  extra_A: callable -> A,a0."""
    th = sy.theta_true.double()
    C = sy.C
    a_solver = sy.step(sy.E_true, sy.gain_true, n_sub=1)
    x_next = sy.p.get("pos").clone()
    a_fd = fd_accel(sy.x_prev, sy.x0, x_next, sy.dt_sub)
    fdrel = rel(a_fd - a_solver, a_solver)
    if extra_A is None:
        A, a0, _ = sy.assemble(n_sub=1)
    else:
        A, a0 = extra_A(sy)
    b = a_fd - a0
    resid = rel(A @ sy.theta_true - b, b)
    S = Solver(A, C)
    sols = S(b)
    sc = {k: score(v, th, C) for k, v in sols.items()}
    th0 = sols["ridge0"]
    e = ((th0 - th) / th).abs()
    tail = {"frac_E_gt_1pct": float((e[:C] > 1e-2).double().mean()),
            "frac_gain_gt_1pct": float((e[C:] > 1e-2).double().mean()),
            "frac_E_gt_10pct": float((e[:C] > 1e-1).double().mean()),
            "frac_gain_gt_10pct": float((e[C:] > 1e-1).double().mean())}
    S.free()
    out = {"fd_vs_solver_rel": fdrel, "model_residual_rel_b": resid,
           "norm_b": float(b.norm()), "norm_a0": float(a0.norm()),
           "norm_a_obs": float(a_solver.norm()),
           "cond_scaled": S.cond, "ridge0": sc["ridge0"], "tail": tail,
           "best": min(sc, key=lambda k: sc[k]["rel_l2"]),
           "best_l2": min(sc[k]["rel_l2"] for k in sc)}
    log(f"[{tag}] fd-vs-solver {fdrel:.2e} | model resid {resid:.3e} | ridge0 med E "
        f"{sc['ridge0']['med_E']:.3e} med g {sc['ridge0']['med_gain']:.3e} "
        f"l2 {sc['ridge0']['rel_l2']:.3e} | best {out['best']} l2 {out['best_l2']:.3e}")
    log(f"        tail: {tail}")
    del A
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--probes", default="P1,P2,P3,P4,P5")
    ap.add_argument("--tag", default="C100")
    args = ap.parse_args()
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"argv": vars(args)}
    t0 = time.time()
    with torch.no_grad():
        dev, dt = args.device, torch.float64
        C0 = 472 if args.real else args.cells
        E_p = draw(C0, 40.0, 220.0, 2026, dev, dt)      # the TASK D planting (seed 2026, E first)
        g_p = draw(C0, 0.5, 1.5, 2026, dev, dt)
        # NB: recover.py draws E then gain from the SAME generator; reproduce that exactly
        gg = torch.Generator().manual_seed(2026)
        E_p = (40.0 + 180.0 * torch.rand(C0, generator=gg)).to(dev, dt)
        g_p = (0.5 + 1.0 * torch.rand(C0, generator=gg)).to(dev, dt)
        E_dec = torch.full((C0,), 130.0, device=dev, dtype=dt)
        g_dec = torch.ones(C0, device=dev, dtype=dt)
        gg2 = torch.Generator().manual_seed(777)
        E_alt = (40.0 + 180.0 * torch.rand(C0, generator=gg2)).to(dev, dt)
        g_alt = (0.5 + 1.0 * torch.rand(C0, generator=gg2)).to(dev, dt)

        # ---------------- P1 / P2 / P5 -------------------------------------------------- #
        if "P1" in args.probes:
            log("\n=== P1  CIRCULARITY: does the recovery depend on the warm-up theta? ===")
            log(" (a) warm-up WITH theta_true  (this is exactly what recover.py does)")
            sy = build(args, args.warmup, E_p, g_p, E_p, g_p, log)
            R["P1a_warmup_is_theta_true"] = run_one(sy, "P1a truth-warmed", log)
            xk_a = sy.x0.clone()
            del sy
            torch.cuda.empty_cache()

            log(" (b) warm-up with a DECOY theta (E=130, gain=1); theta_true installed only at "
                "the frozen state")
            sy = build(args, args.warmup, E_dec, g_dec, E_p, g_p, log)
            R["P1b_warmup_is_decoy_constant"] = run_one(sy, "P1b decoy-warmed", log)
            R["P1_xk_differs_between_a_and_b"] = float((sy.x0 - xk_a).norm() / xk_a.norm())
            del sy
            torch.cuda.empty_cache()

            log(" (c) warm-up with a DIFFERENT random theta (seed 777)")
            sy = build(args, args.warmup, E_alt, g_alt, E_p, g_p, log)
            R["P1c_warmup_is_other_random"] = run_one(sy, "P1c alt-warmed", log)
            del sy
            torch.cuda.empty_cache()

            log(" (d) warm-up 0 frames (no trajectory at all: the seeded initial configuration)")
            sy = build(args, 1, E_dec, g_dec, E_p, g_p, log)
            R["P1d_warmup_1_frame"] = run_one(sy, "P1d 1-frame", log)
            del sy
            torch.cuda.empty_cache()
            log(f" trajectories really are different: ||x_k(a)-x_k(b)||/||x_k|| = "
                f"{R['P1_xk_differs_between_a_and_b']:.3e}")

        # ---------------- P3: wrong-but-linear models ------------------------------------ #
        if "P3" in args.probes:
            log("\n=== P3  WRONG-BUT-LINEAR: A from a wrong model, b from the true one ===")
            sy = build(args, args.warmup, E_p, g_p, E_p, g_p, log)
            R["P3_ref"] = run_one(sy, "P3 reference (correct A)", log)

            # (a) wrong wall model: same operators, wall_damp 0.5 -> 1.0 (still linear in theta)
            def A_wall(s):
                obs = [ob for nm, ob, _ in s.inst if nm in ("mpm_gather", "mpm_grid_update")]
                old = [ob.wall_damp for ob in obs]
                for ob in obs:
                    ob.wall_damp = 1.0
                A, a0, _ = s.assemble(n_sub=1)
                for ob, o in zip(obs, old):
                    ob.wall_damp = o
                return A, a0
            R["P3a_wrong_wall_damp"] = run_one(sy, "P3a wall_damp 1.0 in A", log, extra_A=A_wall)

            # (b) wrong deformation gradient (F is NOT observable; how exact must it be?)
            for eps in (1e-8, 1e-6, 1e-4, 1e-2):
                def A_F(s, eps=eps):
                    F_true = s.F0.clone()
                    gF = torch.Generator(device=s.device).manual_seed(5)
                    s.F0 = F_true * (1.0 + eps * torch.randn(F_true.shape, generator=gF,
                                                             device=s.device, dtype=s.dtype))
                    A, a0, _ = s.assemble(n_sub=1)
                    s.F0 = F_true
                    return A, a0
                R[f"P3b_wrongF_{eps:g}"] = run_one(sy, f"P3b F perturbed {eps:g}", log,
                                                   extra_A=A_F)
            del sy
            torch.cuda.empty_cache()

        # ---------------- P4: is rel_l2 sensitive to the gain block? --------------------- #
        if "P4" in args.probes:
            log("\n=== P4  SCALE: rel_l2 is the metric every 'beats the null' claim uses ===")
            th = torch.cat([E_p, g_p]).double()
            C = C0
            perfect_E_wrong_gain = torch.cat([E_p, torch.ones(C, device=dev, dtype=dt)]).double()
            perfect_gain_wrong_E = torch.cat([torch.full((C,), 130.0, device=dev, dtype=dt),
                                              g_p]).double()
            allzero_gain = torch.cat([E_p, torch.zeros(C, device=dev, dtype=dt)]).double()
            R["P4"] = {
                "true_E_gain_set_to_1": score(perfect_E_wrong_gain, th, C),
                "true_E_gain_set_to_0": score(allzero_gain, th, C),
                "true_gain_E_set_to_130": score(perfect_gain_wrong_E, th, C),
            }
            for k, v in R["P4"].items():
                log(f"   {k:26s} rel_l2 = {v['rel_l2']:.4f}   med_E {v['med_E']:.3f}  "
                    f"med_gain {v['med_gain']:.3f}")

    R["wall_seconds"] = time.time() - t0
    p = os.path.join(OUT, f"probe_f_{args.tag}.json")
    json.dump(R, open(p, "w"), indent=1, default=str)
    open(os.path.join(OUT, f"probe_f_{args.tag}.log"), "w").write("\n".join(lines))
    log(f"\nwrote {p}  [{R['wall_seconds']:.1f}s]")


if __name__ == "__main__":
    main()
