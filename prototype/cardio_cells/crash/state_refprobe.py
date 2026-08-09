"""state_refprobe.py -- TASK 3 refutation probes on state_derive.py / state_combine.py.

Runs, does not read:
  1. restore-path proof: after install_state(derived), does sy.restore() put the DERIVED C into p.C?
  2. CORRUPTION control: replace the true C0/v0 in the snapshot with garbage; the derived-state
     assembly must not move at all.  Then corrupt the DERIVED C; it must move.
  3. artefact check: does frame 0 of state_norm_grid48_der_s90210.npz reproduce from the code?
  4. F identity: is the F used to derive C the same noisy F injected into the assembly?
  5. seeds: are s90210 and s555 different noise realisations?
  6. Jp: is the true Jp actually 1 (so Jp_one is not an oracle in disguise)?
"""
from __future__ import annotations

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

from recover import theta_scale                                   # noqa: E402
from finject import lerp, assemble_inj                            # noqa: E402
from round5_fit import SIGMA_F, SIGMA_X, SNAP                     # noqa: E402
from refute5_fit import NoiseF                                    # noqa: E402
import state_derive as SD                                         # noqa: E402
import state_combine as SC                                        # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
T0, T, SEED = 165, 8, 90210
R = {}
lines = []


def log(s):
    print(s, flush=True)
    lines.append(str(s))


def main():
    args = SimpleNamespace(device=DEV, cells=100, per_parent=100, n_grid=128,
                           warmup=T0, window=150, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    t_start = time.time()
    torch.manual_seed(0)
    with torch.no_grad():
        sy, B = SD.collect(args, T0 - 2, T0 + T + 1, log)
        C, n, dt = sy.C, sy.n_sub_per_frame, sy.dt
        s = theta_scale(C, sy.device)
        NF = NoiseF("grid", B[T0]["x0"], 48, sy.device, sy.dtype)
        need = sorted(set([t - 1 for t in range(T0, T0 + T)] + list(range(T0, T0 + T))
                          + [t + 1 for t in range(T0, T0 + T)]))
        eF, ex = SC.draw_noise(sy, NF, SEED, T0, T, SIGMA_F, SIGMA_X, extra_ticks=need)
        NF2 = NoiseF("grid", B[T0]["x0"], 48, sy.device, sy.dtype)
        eF2, ex2 = SC.draw_noise(sy, NF2, 555, T0, T, SIGMA_F, SIGMA_X, extra_ticks=need)
        log(f"[collect] {time.time()-t_start:.0f}s")

        k = T0
        v, Cc, _ = SD.derived_state(B, k, dt, eF, ex)
        C0t, v0t = B[k]["snap"]["C0"], B[k]["snap"]["v0"]

        # ---- 1. restore path ------------------------------------------------------------------ #
        SD.install_state(sy, B[k]["snap"], v, Cc, Jp_one=True)
        sy.restore()
        va, vb = sy.p.state_schema["vel"]
        R["restore_path"] = {
            "pC_minus_derivedC_maxabs": float((sy.p.C - Cc).abs().max()),
            "pC_minus_trueC_relnorm": float((sy.p.C - C0t).norm() / C0t.norm()),
            "pv_minus_derivedv_maxabs": float((sy.p.state[:, va:vb] - v).abs().max()),
            "pv_minus_truev_relnorm": float((sy.p.state[:, va:vb] - v0t).norm() / v0t.norm()),
            "pJp_minus_one_maxabs": float((sy.p.Jp - 1.0).abs().max()),
            "trueJp_minus_one_maxabs": float((B[k]["Jp0"] - 1.0).abs().max()),
            "pos_is_true_x0_maxabs": float((sy.p.get("pos") - B[k]["x0"]).abs().max())}
        log(f"[1] p.C == derived C to {R['restore_path']['pC_minus_derivedC_maxabs']:.2e}; "
            f"p.C vs TRUE C rel {R['restore_path']['pC_minus_trueC_relnorm']:.4f}; "
            f"p.v == derived v to {R['restore_path']['pv_minus_derivedv_maxabs']:.2e}; "
            f"p.v vs TRUE v rel {R['restore_path']['pv_minus_truev_relnorm']:.4f}")
        log(f"    true Jp max|Jp-1| = {R['restore_path']['trueJp_minus_one_maxabs']:.2e} "
            f"(Jp_one is exact, not an oracle)")

        # ---- 2. corruption control -------------------------------------------------------------- #
        F0h, F1h = B[k]["F0"] + eF[k], B[k]["F1"] + eF[k + 1]
        inj = lerp(F0h, F1h, n)
        xm1 = B[k]["x_next"] + ex[k + 1]

        def assemble_with(snap, vv, CC):
            SD.install_state(sy, snap, vv, CC, Jp_one=True)
            A, y0, _ = assemble_inj(sy, n, inj, None)
            Az = A * s[None, :]
            b = (xm1 - B[k]["x0"]).reshape(-1) - y0
            G, r = Az.T @ Az, Az.T @ b
            del A, Az
            torch.cuda.empty_cache()
            return G, r

        G_ref, r_ref = assemble_with(B[k]["snap"], v, Cc)
        log(f"[2] baseline derived-state assembly done [{time.time()-t_start:.0f}s]")

        g = torch.Generator(device=sy.device).manual_seed(1234)
        bad = {kk: vv.clone() for kk, vv in B[k]["snap"].items()}
        bad["C0"] = torch.randn(C0t.shape, generator=g, device=sy.device, dtype=sy.dtype) * \
            float(C0t.abs().max())
        badv = torch.randn(v0t.shape, generator=g, device=sy.device, dtype=sy.dtype) * \
            float(v0t.abs().max())
        st = bad["state0"].clone()
        st[:, va:vb] = badv
        bad["state0"] = st
        bad["v0"] = badv
        bad["Jp0"] = B[k]["snap"]["Jp0"] * 1.37
        G_c, r_c = assemble_with(bad, v, Cc)
        R["corrupt_snapshot"] = {
            "rel_G0": float((G_c - G_ref).norm() / G_ref.norm()),
            "rel_r0": float((r_c - r_ref).norm() / r_ref.norm()),
            "corrupt_C_rel_size": float((bad["C0"] - C0t).norm() / C0t.norm())}
        log(f"[2a] snapshot C0/v0/Jp0 CORRUPTED (rel size {R['corrupt_snapshot']['corrupt_C_rel_size']:.1f}) "
            f"-> rel dG0 {R['corrupt_snapshot']['rel_G0']:.3e}  rel dr0 "
            f"{R['corrupt_snapshot']['rel_r0']:.3e}   [{time.time()-t_start:.0f}s]")

        G_p, r_p = assemble_with(B[k]["snap"], v, Cc * 1.01)
        R["perturb_derived_C_1pct"] = {
            "rel_G0": float((G_p - G_ref).norm() / G_ref.norm()),
            "rel_r0": float((r_p - r_ref).norm() / r_ref.norm())}
        log(f"[2b] DERIVED C x1.01 -> rel dG0 {R['perturb_derived_C_1pct']['rel_G0']:.3e}  "
            f"rel dr0 {R['perturb_derived_C_1pct']['rel_r0']:.3e}")

        G_o, r_o = assemble_with(B[k]["snap"], None, None)
        R["oracle_vs_derived"] = {
            "rel_G0": float((G_o - G_ref).norm() / G_ref.norm()),
            "rel_r0": float((r_o - r_ref).norm() / r_ref.norm())}
        log(f"[2c] ORACLE state vs derived state -> rel dG0 "
            f"{R['oracle_vs_derived']['rel_G0']:.3e}  rel dr0 "
            f"{R['oracle_vs_derived']['rel_r0']:.3e}")

        # ---- 3. does the stored artefact reproduce? --------------------------------------------- #
        Z = np.load(os.path.join(HERE, "state_norm_grid48_der_s90210.npz"))
        Gz = torch.as_tensor(Z["G0"], device=sy.device, dtype=sy.dtype)
        rz = torch.as_tensor(Z["r0"], device=sy.device, dtype=sy.dtype)
        R["artefact_frame0"] = {"rel_G0": float((G_ref - Gz).norm() / Gz.norm()),
                                "rel_r0": float((r_ref - rz).norm() / rz.norm())}
        log(f"[3] frame 0 of state_norm_grid48_der_s90210.npz reproduces: rel G0 "
            f"{R['artefact_frame0']['rel_G0']:.3e}  rel r0 {R['artefact_frame0']['rel_r0']:.3e}")

        # ---- 4. F identity ---------------------------------------------------------------------- #
        Fk1_deriv = B[k + 1]["F0"] + eF[k + 1]
        R["F_identity"] = {
            "assembly_F1_minus_derivation_Fkp1_maxabs": float((F1h - Fk1_deriv).abs().max()),
            "assembly_F0_minus_derivation_Fk_maxabs": float((F0h - (B[k]["F0"] + eF[k])).abs().max()),
            "eF_rel_size_at_k": float(eF[k].norm() / B[k]["F0"].norm())}
        log(f"[4] assembly F vs derivation F: max|dF1| "
            f"{R['F_identity']['assembly_F1_minus_derivation_Fkp1_maxabs']:.2e}, max|dF0| "
            f"{R['F_identity']['assembly_F0_minus_derivation_Fk_maxabs']:.2e}")

        # ---- 5. seeds --------------------------------------------------------------------------- #
        cs = []
        for t in range(T0, T0 + T + 1):
            a1, a2 = eF[t].reshape(-1), eF2[t].reshape(-1)
            cs.append(float(torch.dot(a1, a2) / (a1.norm() * a2.norm())))
        R["seed_independence"] = {
            "corr_eF_per_tick": cs, "max_abs_corr": float(np.max(np.abs(cs))),
            "corr_ex_t0p1": float(torch.dot(ex[T0 + 1].reshape(-1), ex2[T0 + 1].reshape(-1))
                                  / (ex[T0 + 1].norm() * ex2[T0 + 1].norm()))}
        log(f"[5] seed 90210 vs 555: max |corr(eF)| over ticks "
            f"{R['seed_independence']['max_abs_corr']:.3e}, corr(ex) "
            f"{R['seed_independence']['corr_ex_t0p1']:.3e}")

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, "state_refprobe.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, "state_refprobe.log"), "w").write("\n".join(lines) + "\n")
    log(f"wrote state_refprobe.json [{R['wall_seconds']:.0f}s]")


if __name__ == "__main__":
    main()
