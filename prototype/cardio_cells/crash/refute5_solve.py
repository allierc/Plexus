"""refute5_solve.py -- round 5's solver stack, unchanged, applied to the realizable-noise fits.

Imports solve_box / snr_trunc / pstats from round5_solve.py verbatim; only the file glob and the
output names differ, so no round-5 artefact is overwritten.

usage: /workspace/.conda_envs/neural-graph-linux/bin/python refute5_solve.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ALG = "/workspace/Plexus/prototype/cardio_cells/algebraic"
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("/workspace/Plexus/src", ALG, HERE):
    sys.path.insert(0, _p)

from round5_solve import solve_box, snr_trunc, pstats            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo-f", type=float, default=0.2)
    ap.add_argument("--hi-f", type=float, default=5.0)
    ap.add_argument("--box-iters", type=int, default=4000)
    ap.add_argument("--out", default="theta_refute5")
    ap.add_argument("--tag", default="refute5_solve")
    a = ap.parse_args()
    dev = torch.device("cpu")
    R = {"box": [a.lo_f, a.hi_f], "fits": {}}
    thetas, lines = {}, []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    files = sorted(glob.glob(os.path.join(HERE, "refute5_norm_*.npz")))
    log(f"[solve] {len(files)} realizable-noise normal-equation files")
    log(f"    {'fit':<34s} {'T':>2s} {'solver':<11s} {'medE':>7s} {'p90':>7s} {'maxE':>8s} "
        f"{'neg':>4s} {'relL2':>7s} {'medE_re':>8s} {'corr':>6s} {'mr':>7s}")
    for fp in files:
        z = np.load(fp)
        name = os.path.basename(fp)[:-4]
        th = torch.as_tensor(z["theta_true"], dtype=torch.float64)
        s = torch.as_tensor(z["s"], dtype=torch.float64)
        C = th.numel() // 2
        nfr = sum(1 for k in z.files if k.startswith("G") and not k.startswith("Gm"))
        R["fits"][name] = {"n_frames": nfr, "T": {}}
        for T in (1, 8):
            if T > nfr:
                continue
            G0 = sum(torch.as_tensor(z[f"G{k}"], dtype=torch.float64) for k in range(T))
            r0 = sum(torch.as_tensor(z[f"r{k}"], dtype=torch.float64) for k in range(T))
            Gb = sum(torch.as_tensor(z[f"Gm{k}"], dtype=torch.float64) for k in range(T))
            rb = sum(torch.as_tensor(z[f"rm{k}"], dtype=torch.float64) for k in range(T))
            has_mc = float(Gb.abs().max()) > 0
            Sig = (Gb - G0) if has_mc else torch.zeros_like(G0)
            Gc, rc = G0 - Sig, r0 - (rb - r0 if has_mc else torch.zeros_like(r0))

            out = {"naive": torch.linalg.solve(G0, r0) * s}
            ex_snr = {}
            if has_mc:
                out["eiv_snr0"], ex_snr = snr_trunc(G0, Sig, Gc, rc, s, tau=0.0)
            nv = out["naive"]
            mE = float(nv[:C][nv[:C] > 0].median())
            mg = float(nv[C:][nv[C:] > 0].median())
            lo = torch.cat([torch.full((C,), a.lo_f * mE, dtype=torch.float64),
                            torch.full((C,), a.lo_f * mg, dtype=torch.float64)])
            hi = torch.cat([torch.full((C,), a.hi_f * mE, dtype=torch.float64),
                            torch.full((C,), a.hi_f * mg, dtype=torch.float64)])
            info = {}
            out["naive_box"], info["naive_box"] = solve_box(
                G0, r0, s, lo, hi, z0=torch.clamp(nv, lo, hi) / s, iters=a.box_iters)
            if has_mc:
                out["eiv_box"], info["eiv_box"] = solve_box(
                    Gc, rc, s, lo, hi, z0=torch.clamp(out["eiv_snr0"], lo, hi) / s,
                    iters=a.box_iters)
            ev = torch.linalg.eigvalsh(G0)
            row = {"box_bounds": {"E": [a.lo_f * mE, a.hi_f * mE],
                                  "gain": [a.lo_f * mg, a.hi_f * mg]},
                   "snr": ex_snr, "box_info": info,
                   "cond_G0": float(ev.max() / ev.clamp(min=1e-300).min()),
                   "min_eig_Gc": float(torch.linalg.eigvalsh(Gc).min()),
                   "sigma_fro_over_G_fro": float(Sig.norm() / G0.norm()), "solvers": {}}
            for k, t in out.items():
                p = pstats(t.numpy(), th.numpy(), C)
                row["solvers"][k] = p
                short = name.replace("refute5_norm_", "")
                thetas[f"{short}|T{T}|{k}"] = t.numpy()
                log(f"    {short:<34s} {T:>2d} {k:<11s} {p['med_E']:>7.4f} {p['p90_E']:>7.3f} "
                    f"{p['max_E']:>8.3f} {p['n_negE']:>4d} {p['rel_l2']:>7.3f} "
                    f"{p['med_E_after_rescale']:>8.4f} {p['corr_E']:>6.3f} "
                    f"{p['mean_ratio_E']:>7.3f}")
            R["fits"][name]["T"][f"T{T}"] = row
    thetas["theta_true"] = th.numpy()
    np.savez(os.path.join(HERE, f"{a.out}.npz"), **thetas)
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json and {a.out}.npz ({len(thetas)} vectors)")


if __name__ == "__main__":
    main()
