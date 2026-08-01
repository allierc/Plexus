"""DIFFER analysis for apoptose vs oracle Death -- pooled per-macro-step death hazard.

Pure numpy; imports NO jax. Reads the two already-recorded trajectories and computes the
realization- and record-convention-independent scalar the two sides are judged on:

    p_hat = (n_active[0] - n_active[-1]) / sum(n_active[:-1])

For a pure-death rollout (no births) every decrement is a death, so p_hat is the pooled
maximum-likelihood per-macro-step Bernoulli hazard over all eligible live-cell-steps in the
recorded window. It is unbiased for the true hazard p for ANY window of a stationary process,
which is why the engine's record-after-apply off-by-one (Plexus' first recorded frame is already
post-step-1: 47506, not the pristine 50000) does not bias it.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))            # .../Atlas_jax_morph/_oracle/scripts
ATLAS = os.path.abspath(os.path.join(HERE, "..", ".."))      # .../Atlas_jax_morph
PLEXUS = os.path.abspath(os.path.join(ATLAS, ".."))          # .../Plexus

# --- oracle -------------------------------------------------------------------------------- #
oref = np.load(os.path.join(ATLAS, "_oracle", "runs", "diff_death", "reference.npz"))
with open(os.path.join(ATLAS, "_oracle", "runs", "diff_death", "summary.json")) as f:
    osum = json.load(f)
oc = np.asarray(oref["counts"], dtype=np.float64)
LAM, DT = osum["lambda"], osum["dt"]

# --- plexus -------------------------------------------------------------------------------- #
pnpz = np.load(os.path.join(PLEXUS, "log", "atlas", "death", "metrics.npz"))
pc = np.asarray(pnpz["cell__n_active"], dtype=np.float64)


def hazard(counts):
    deaths = counts[0] - counts[-1]
    eligible = counts[:-1].sum()
    p = deaths / eligible
    se = np.sqrt(p * (1.0 - p) / eligible)
    return float(p), float(se), float(deaths), float(eligible)


p_o, se_o, d_o, e_o = hazard(oc)
p_p, se_p, d_p, e_p = hazard(pc)
p_theory = float(-np.expm1(-LAM * DT))       # 1 - exp(-lambda*dt), the exact hazard both share
p_linear = float(LAM * DT)                    # the linear-approx bug this test must reject

diff = abs(p_p - p_o)
se_pooled = float(np.sqrt(se_o ** 2 + se_p ** 2))
thresh = 3.0 * se_pooled

out = {
    "metric": "|p_hat_plexus - p_hat_oracle|  (pooled per-macro-step death hazard)",
    "lambda": LAM, "dt": DT,
    "oracle": {"p_hat": p_o, "se": se_o, "deaths": d_o, "eligible_cellsteps": e_o,
               "n_frames": int(oc.size), "first": float(oc[0]), "last": float(oc[-1])},
    "plexus": {"p_hat": p_p, "se": se_p, "deaths": d_p, "eligible_cellsteps": e_p,
               "n_frames": int(pc.size), "first": float(pc[0]), "last": float(pc[-1])},
    "p_theory_exact": p_theory, "p_linear_approx_bug": p_linear,
    "diff": diff, "se_pooled": se_pooled, "threshold_3se": thresh,
    "diff_in_se": diff / se_pooled,
    "oracle_vs_theory_in_se": abs(p_o - p_theory) / se_o,
    "plexus_vs_theory_in_se": abs(p_p - p_theory) / se_p,
    "linear_bug_gap": abs(p_theory - p_linear),
    "linear_bug_gap_gt_threshold": bool(abs(p_theory - p_linear) > thresh),
    "passed": bool(diff <= thresh
                   and abs(p_o - p_theory) <= 3 * se_o
                   and abs(p_p - p_theory) <= 3 * se_p),
}
print(json.dumps(out, indent=2))
