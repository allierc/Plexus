"""round5_boxwidth.py -- how much of the box-constrained result is the BOX?

The data-driven box is [0.2, 5] x the naive block median.  Because the naive fit is attenuated by
~0.44, that puts the E block's upper edge at 210-228 against a planted maximum of 216.3 -- close
enough that the prior cannot be called uninformative.  This re-solves the same normal equations
with the box widened 4x and 16x, and with positivity only, and reports what each costs.  The
resulting thetas are written to theta_round5_extra.npz for round5_score.py to roll out.

usage: PYTHONPATH=/workspace/Plexus/src python round5_boxwidth.py
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from round5_sensitivity import load                              # noqa: E402
from round5_solve import solve_box, snr_trunc, pstats            # noqa: E402

out, extra = {}, {}
for tag, sd in (("round5_norm_s90210_sF0.0039", "s90210"),
                ("round5_norm_s555_sF0.0039", "s555"),
                ("round5_norm_s777_sF0.0039", "s777")):
    G0, r0, Gb, rb, s, th = load(tag, 8)
    C = th.numel() // 2
    nv = torch.linalg.solve(G0, r0) * s
    mE = float(nv[:C][nv[:C] > 0].median())
    mg = float(nv[C:][nv[C:] > 0].median())
    Sig = Gb - G0
    Gc, rc = G0 - Sig, r0 - (rb - r0)
    e0, _ = snr_trunc(G0, Sig, Gc, rc, s, 0.0)
    for lof, hif, nm in ((0.2, 5.0, "box0.2_5"), (0.1, 10.0, "box0.1_10"),
                         (0.05, 20.0, "box0.05_20"), (0.0, 1e9, "positivity_only")):
        lo = torch.cat([torch.full((C,), lof * mE, dtype=torch.float64),
                        torch.full((C,), lof * mg, dtype=torch.float64)])
        hi = torch.cat([torch.full((C,), hif * mE, dtype=torch.float64),
                        torch.full((C,), hif * mg, dtype=torch.float64)])
        t, i = solve_box(Gc, rc, s, lo, hi, z0=torch.clamp(e0, lo, hi) / s, iters=40000)
        p = pstats(t.numpy(), th.numpy(), C)
        out[f"{sd}|{nm}"] = dict(p, n_active_bounds=i["n_active_bounds"],
                                 boxE=[lof * mE, hif * mE], planted_E_max=float(th[:C].max()))
        if sd == "s90210" and nm in ("box0.1_10", "box0.05_20"):
            extra[f"{sd}/T8/eiv_{nm}"] = t.numpy()
        print(f"{sd:>7s} {nm:<16s} E[{lof*mE:6.1f},{hif*mE:8.1f}] active {i['n_active_bounds']:>3d} "
              f"med {p['med_E']:.4f} max {p['max_E']:9.2f} relL2 {p['rel_l2']:9.3f} "
              f"corr {p['corr_E']:.3f} neg {p['n_negE']}")
json.dump(out, open(os.path.join(HERE, "round5_boxwidth.json"), "w"), indent=1, default=str)
np.savez(os.path.join(HERE, "theta_round5_extra.npz"), **extra)
print("wrote round5_boxwidth.json + theta_round5_extra.npz")
