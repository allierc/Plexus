"""mcv_refute.py -- adversarial audit of mcv_v.py's "model-corrected v" claim.

Five attacks, each with a control:
  R  reproduce: controls + route (a) honest, C oracle          (must match mcv_v.log)
  W  WRONG-THETA: compute the "model correction" at a deliberately wrong theta.  If the med|dE/E|
     improvement survives a wrong model, the improvement is not physics.
  N  NULL: replace the correction by a RANDOM vector of the same norm (3 seeds).  If a random
     perturbation of the same size moves med|dE/E| as much, the 9% is not a signal.
  T  OTHER FRAMES: repeat baseline/oracle/route(a) at t0 in {168,172,176}.  Is +9.1% a property of
     the method or of tick 165?
  X  CORRUPTION on the FULLY DERIVED ladder (C from the centred difference too) -- the variant that
     carries the headline 24.1%, which mcv_v never corruption-tested.

usage: PYTHONPATH=/workspace/Plexus/src python mcv_refute.py --device cuda:1 --stages RWNTX
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

from finject import lerp, y_of                                        # noqa: E402
from refute_round3 import fit                                         # noqa: E402
from state_derive import collect, install_state, rel, derived_v       # noqa: E402
from mcv_v import install2, predict, shoot, DRAG_K                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="mcv_refute")
    ap.add_argument("--stages", default="RWNTX")
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--holdout-tick", type=int, default=180)
    ap.add_argument("--other-t0", default="168,172,176")
    a = ap.parse_args()

    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128,
                           warmup=a.t0, window=150, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "stages": a.stages}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        t_lo, t_hi = a.t0 - 2, a.holdout_tick + 2
        sy, B = collect(args, t_lo, t_hi, log)
        C, n, dt = sy.C, sy.n_sub_per_frame, sy.dt
        th = sy.theta_true.double()
        k0, hk = a.t0, a.holdout_tick
        log(f"[collect] {t_lo}..{t_hi} C={C} Np={sy.Np} n={n} dt={dt} [{time.time()-t_start:.0f}s]")

        def C_cd(k):
            return ((B[k + 1]["F0"] - B[k - 1]["F0"]) / (2 * dt)) @ torch.linalg.inv(B[k]["F0"])

        injh = lerp(B[hk]["F0"], B[hk]["F1"], n)
        y_obsh = (B[hk]["x_next"] - B[hk]["x0"]).reshape(-1)
        v_cdh, C_cdh = derived_v(B, hk, dt), C_cd(hk)

        def holdout(theta, mode="oracle"):
            v, Cm = (None, None) if mode == "oracle" else (v_cdh, C_cdh)
            install2(sy, B[hk]["snap"], hk, v, Cm, mode != "oracle")
            y = y_of(sy, theta, n, injh, None)
            return float((y - y_obsh).norm() / y_obsh.norm())

        def do_fit(k, v, Cm, honest, y_obs_pos=None):
            install2(sy, B[k]["snap"], k, v, Cm, honest)
            injF = lerp(B[k]["F0"], B[k]["F1"], n)
            return fit(sy, n, injF, B[k]["x_next"] if y_obs_pos is None else y_obs_pos,
                       B[k]["x0"], th, C)

        # ------------------------------------------------------------------ stage R ---------- #
        v_cd0, C_cd0 = derived_v(B, k0, dt), C_cd(k0)
        v_true0 = B[k0]["v0"]
        inj0 = lerp(B[k0]["F0"], B[k0]["F1"], n)
        y_obs0 = (B[k0]["x_next"] - B[k0]["x0"]).reshape(-1)

        log(f"\n[R] REPRODUCTION, tick {k0}")
        sc_or, th_or = do_fit(k0, None, None, False)
        sc_b, th_b = do_fit(k0, v_cd0, None, True)
        sc_bC, th_bC = do_fit(k0, v_cd0, C_cd0, True)
        log(f"    oracle v/C            medE {sc_or['med_E']:.6f}  (mcv_v 0.007777)")
        log(f"    c2 v, oracle C honest medE {sc_b['med_E']:.6f}  (mcv_v 0.021890)")
        log(f"    c2 v, c2 C honest     medE {sc_bC['med_E']:.6f}  (mcv_v 0.020910)")
        base, floor = sc_b["med_E"], sc_or["med_E"]

        def gap(m):
            return 100.0 * (base - m) / (base - floor)

        v_a, hist = shoot(sy, B[k0]["snap"], k0, v_cd0, None, th_b, n, inj0, y_obs0, dt, True,
                          steps=1, v_true=v_true0)
        corr = v_a - v_cd0
        sc_a, th_a = do_fit(k0, v_a, None, True)
        log(f"    route (a) honest      medE {sc_a['med_E']:.6f}  (mcv_v 0.020608)  "
            f"gap {gap(sc_a['med_E']):+.1f}%  relv {rel(v_a, v_true0):.5f}  "
            f"ho(oracle) {holdout(th_a):.5f} ho(c2) {holdout(th_a,'c2'):.5f}")
        R["reproduce"] = {"oracle": sc_or["med_E"], "baseline_c2_honest": base,
                          "baseline_c2_Cc2_honest": sc_bC["med_E"], "route_a": sc_a["med_E"],
                          "route_a_gap_pct": gap(sc_a["med_E"]),
                          "corr_norm": float(corr.norm()),
                          "relv_base": rel(v_cd0, v_true0), "relv_a": rel(v_a, v_true0),
                          "ho_oracle_base": holdout(th_b), "ho_c2_base": holdout(th_b, "c2"),
                          "ho_oracle_a": holdout(th_a), "ho_c2_a": holdout(th_a, "c2"),
                          "ho_oracle_theta_true": holdout(th), "ho_c2_theta_true": holdout(th, "c2")}
        log(f"    held-out tick {hk}, HONEST c2 state (no oracle anywhere): base "
            f"{R['reproduce']['ho_c2_base']:.5f}  route_a {R['reproduce']['ho_c2_a']:.5f}  "
            f"theta_true {R['reproduce']['ho_c2_theta_true']:.5f}")
        log(f"    held-out tick {hk}, ORACLE state: base {R['reproduce']['ho_oracle_base']:.5f}  "
            f"route_a {R['reproduce']['ho_oracle_a']:.5f}  "
            f"theta_true {R['reproduce']['ho_oracle_theta_true']:.5f}")

        # ------------------------------------------------------------------ stage W ---------- #
        if "W" in a.stages:
            log("\n[W] WRONG-THETA CORRECTION.  v <- v_c2 + (obs - pred(v_c2, theta_WRONG))/dt.")
            log(f"    {'theta used for the correction':<34s} {'relTh':>7s} {'|corr|':>9s} "
                f"{'relv':>8s} {'med|dE/E|':>10s} {'gap%':>7s} {'ho(orc)':>8s}")
            g = torch.Generator(device=sy.device).manual_seed(4242)
            perm = torch.randperm(C, generator=torch.Generator().manual_seed(7)).to(sy.device)
            th_mean = torch.cat([th[:C].mean().repeat(C), th[C:].mean().repeat(C)])
            variants = {
                "theta_hat (route a, reference)": th_b,
                "0.5 * theta_hat": 0.5 * th_b,
                "2.0 * theta_hat": 2.0 * th_b,
                "theta_hat, cells shuffled": torch.cat([th_b[:C][perm], th_b[C:][perm]]),
                "flat mean E, mean gain": th_mean,
                "theta_true (planted)": th,
                "E=40 flat, gain=1 flat": torch.cat([torch.full((C,), 40.0, device=sy.device,
                                                                dtype=sy.dtype),
                                                     torch.ones(C, device=sy.device,
                                                                dtype=sy.dtype)]),
            }
            R["wrong_theta"] = {}
            for nm, tw in variants.items():
                vw, _ = shoot(sy, B[k0]["snap"], k0, v_cd0, None, tw, n, inj0, y_obs0, dt, True,
                              steps=1)
                sc_w, th_w = do_fit(k0, vw, None, True)
                row = {"rel_theta_vs_true": rel(tw, th), "corr_norm": float((vw - v_cd0).norm()),
                       "relv": rel(vw, v_true0), "med_E": sc_w["med_E"],
                       "gap_pct": gap(sc_w["med_E"]), "ho_oracle": holdout(th_w),
                       "ho_c2": holdout(th_w, "c2"), "rel_l2": sc_w["rel_l2"]}
                R["wrong_theta"][nm] = row
                log(f"    {nm:<34s} {row['rel_theta_vs_true']:>7.4f} {row['corr_norm']:>9.3f} "
                    f"{row['relv']:>8.5f} {row['med_E']:>10.5f} {row['gap_pct']:>+7.1f} "
                    f"{row['ho_oracle']:>8.5f} [{time.time()-t_start:.0f}s]")

        # ------------------------------------------------------------------ stage N ---------- #
        if "N" in a.stages:
            log("\n[N] NULL CONTROL: replace the correction by a RANDOM vector of the SAME norm.")
            log(f"    {'variant':<34s} {'relv':>8s} {'med|dE/E|':>10s} {'gap%':>7s} {'ho(orc)':>8s}")
            R["null"] = []
            for seed in (11, 22, 33):
                gg = torch.Generator(device=sy.device).manual_seed(seed)
                r = torch.randn(v_cd0.shape, generator=gg, device=sy.device, dtype=sy.dtype)
                vr = v_cd0 + corr.norm() * r / r.norm()
                sc_r, th_r = do_fit(k0, vr, None, True)
                row = {"seed": seed, "relv": rel(vr, v_true0), "med_E": sc_r["med_E"],
                       "gap_pct": gap(sc_r["med_E"]), "ho_oracle": holdout(th_r)}
                R["null"].append(row)
                log(f"    {'random step, |corr| norm, s%d' % seed:<34s} {row['relv']:>8.5f} "
                    f"{row['med_E']:>10.5f} {row['gap_pct']:>+7.1f} {row['ho_oracle']:>8.5f} "
                    f"[{time.time()-t_start:.0f}s]")

        # ------------------------------------------------------------------ stage T ---------- #
        if "T" in a.stages:
            log("\n[T] OTHER FRAMES: is +9.1% a property of the method or of tick 165?")
            log(f"    {'t0':>4s} {'oracle':>9s} {'c2 base':>9s} {'route a':>9s} {'gap%':>7s} "
                f"{'relv base':>9s} {'relv a':>8s} {'hoBase':>8s} {'hoA':>8s}")
            R["other_t0"] = []
            for k in [int(x) for x in a.other_t0.split(",") if x]:
                vk, vk_true = derived_v(B, k, dt), B[k]["v0"]
                injk = lerp(B[k]["F0"], B[k]["F1"], n)
                yk = (B[k]["x_next"] - B[k]["x0"]).reshape(-1)
                sc_o, _ = do_fit(k, None, None, False)
                sc_c, th_c = do_fit(k, vk, None, True)
                v_ak, _ = shoot(sy, B[k]["snap"], k, vk, None, th_c, n, injk, yk, dt, True, steps=1)
                sc_ak, th_ak = do_fit(k, v_ak, None, True)
                gp = 100.0 * (sc_c["med_E"] - sc_ak["med_E"]) / (sc_c["med_E"] - sc_o["med_E"])
                row = {"t0": k, "oracle": sc_o["med_E"], "c2": sc_c["med_E"],
                       "route_a": sc_ak["med_E"], "gap_pct": gp,
                       "relv_c2": rel(vk, vk_true), "relv_a": rel(v_ak, vk_true),
                       "ho_base": holdout(th_c), "ho_a": holdout(th_ak)}
                R["other_t0"].append(row)
                log(f"    {k:>4d} {row['oracle']:>9.5f} {row['c2']:>9.5f} {row['route_a']:>9.5f} "
                    f"{gp:>+7.1f} {row['relv_c2']:>9.5f} {row['relv_a']:>8.5f} "
                    f"{row['ho_base']:>8.5f} {row['ho_a']:>8.5f} [{time.time()-t_start:.0f}s]")

        # ------------------------------------------------------------------ stage X ---------- #
        if "X" in a.stages:
            log("\n[X] CORRUPTION on the FULLY DERIVED ladder (C from the centred difference).")
            th_bC_ref = th_bC
            v_aC, _ = shoot(sy, B[k0]["snap"], k0, v_cd0, C_cd0, th_bC_ref, n, inj0, y_obs0, dt,
                            True, steps=1)
            sc_aC, th_aC = do_fit(k0, v_aC, C_cd0, True)
            log(f"    clean: route_a (honest, C c2) medE {sc_aC['med_E']:.6f} (mcv_v 0.018300)  "
                f"gapC {100*(sc_bC['med_E']-sc_aC['med_E'])/(sc_bC['med_E']-sc_or['med_E']):+.1f}% "
                f"of the ORACLE-v/oracle-C gap")
            # corrupt every stored copy of v (v0, state slice, pass0) -- positions untouched
            gcor = torch.Generator(device=sy.device).manual_seed(31337)
            va, vb = sy.p.state_schema["vel"]
            for t in B:
                junk = 1e3 * torch.randn(B[t]["v0"].shape, generator=gcor, device=sy.device,
                                         dtype=sy.dtype)
                B[t]["snap"]["pass0"] = B[t]["snap"]["pass0"] + DRAG_K * (B[t]["v0"] - junk)
                B[t]["v0"] = junk
                st = B[t]["snap"]["state0"].clone()
                st[:, va:vb] = junk
                B[t]["snap"]["state0"] = st
                B[t]["snap"]["v0"] = junk.clone()
            sc_bCx, th_bCx = do_fit(k0, v_cd0, C_cd0, True)
            v_aCx, _ = shoot(sy, B[k0]["snap"], k0, v_cd0, C_cd0, th_bCx, n, inj0, y_obs0, dt,
                             True, steps=1)
            sc_aCx, _ = do_fit(k0, v_aCx, C_cd0, True)
            log(f"    corrupted v store: baseline {sc_bCx['med_E']:.6f} (clean "
                f"{sc_bC['med_E']:.6f})  route_a {sc_aCx['med_E']:.6f} (clean "
                f"{sc_aC['med_E']:.6f})  delta {sc_aCx['med_E']-sc_aC['med_E']:+.2e}")
            # ...and now ALSO corrupt C0 (the remaining state oracle in the 'C oracle' ladder)
            for t in B:
                B[t]["C0"] = B[t]["C0"]  # kept for reference; snapshot C0 is what install reads
                sn = B[t]["snap"]
                sn["C0"] = 1e3 * torch.randn(sn["C0"].shape, generator=gcor, device=sy.device,
                                             dtype=sy.dtype)
            sc_bx2, th_bx2 = do_fit(k0, v_cd0, C_cd0, True)
            sc_ox2, _ = do_fit(k0, None, None, False)
            log(f"    + C0 snapshot corrupted: c2/C_c2 baseline {sc_bx2['med_E']:.6f} "
                f"(unchanged => C_c2 install is complete), 'oracle' ladder {sc_ox2['med_E']:.6f} "
                f"(must blow up => the C-oracle ladder DOES read the simulator's C)")
            R["corruption_Cc2"] = {
                "route_a_Cc2_clean": sc_aC["med_E"], "route_a_Cc2_corrupt_v": sc_aCx["med_E"],
                "baseline_Cc2_clean": sc_bC["med_E"], "baseline_Cc2_corrupt_v": sc_bCx["med_E"],
                "delta": sc_aCx["med_E"] - sc_aC["med_E"],
                "baseline_Cc2_corrupt_vC": sc_bx2["med_E"],
                "oracle_ladder_corrupt_vC": sc_ox2["med_E"]}

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()
