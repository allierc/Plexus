"""Differential test for `cell_divide:volume_conserving` -- the ORACLE side.

The atlas flagged division on a single-seed count: Plexus reaches 124 live cells at frame 40
where the reference (smoke, seed 0) reaches 82. A single sample cannot separate a wrong hazard
from RNG noise -- the JAX and torch PRNG streams differ, so seed-0-vs-seed-0 matches only the
deterministic initial POSITIONS, never the stochastic division draws. This script replaces the
one-sample comparison with a DISTRIBUTION: the exact anchor composition (MechanicalRelaxation +
SaturatingCellGrowth + Division) run over M independent keys, so the division mechanism is
measured by the pooled per-macro-step hazard, which is realization- and composition-independent.

Writes reference.npz (per-seed count trajectories) + summary.json (pooled p_hat, the final-count
distribution, and the theoretical hazard) into OUT.
"""
import json, os, time
import numpy as np
import jax, jax.numpy as jnp
import jax_morph as jxm
from jax_morph.physics import MechanicalRelaxation, Morse, SaturatingCellGrowth, Division

OUT = os.environ["OUT"]
M = int(os.environ.get("M_SEEDS", "48"))
CAP, N_STEPS, DT, RATE = 512, 40, 1.0, 0.08   # CAP >> mean+4SD so overflow can never bind the count

# The authors' own proliferation composition -- identical to the smoke/anchor model.
model = jxm.Model([
    MechanicalRelaxation(Morse(epsilon=3.0, alpha=2.8), max_steps=800, f_tol=1e-3),
    SaturatingCellGrowth(max_radius=0.6),
    Division(n_space_dim=2),
])

def seed_state():
    p0 = jnp.array([[0.0, 0.0], [1.0, 0.1], [0.5, 0.9], [-0.4, 0.7]])  # the 4 founders (oracle frame)
    s = jxm.build_state_from_model(model).init_empty(capacity=CAP, n_space_dim=2, n_types=1)
    n = p0.shape[0]
    return s.update(
        alive=s.alive.at[:n].set(True), radius=s.radius.at[:n].set(0.5),
        position=s.position.at[:n].set(p0), celltype=s.celltype.at[:n, 0].set(1.0),
        growth_rate=s.growth_rate.at[:n].set(0.4), division_rate=s.division_rate.at[:n].set(RATE),
    )

def run(key):
    return jxm.simulate(model, seed_state(), n_steps=N_STEPS, dt=DT, key=key, history=True)

counts = np.zeros((M, N_STEPS + 1), dtype=np.int64)
overflow = np.zeros(M)
t0 = time.time()
for i in range(M):
    h = run(jax.random.PRNGKey(i))
    counts[i] = np.asarray(h.alive.sum(axis=1))
    overflow[i] = float(np.asarray(h.division_overflow)[-1])
    if i < 3 or i % 12 == 0:
        print(f"seed {i}: final={int(counts[i, -1])} overflow={overflow[i]:.0f} "
              f"[{time.time() - t0:.1f}s]", flush=True)

if not np.all(overflow == 0.0):
    raise SystemExit(f"division overflowed capacity on {int((overflow != 0).sum())} seeds -- "
                     f"the count hit an ARRAY BOUND, not the biology. Raise CAP.")

# pooled per-macro-step hazard: committed divisions / eligible cell-steps, over all seeds & steps.
# no death and overflow==0, so committed divisions at step t == counts[t+1]-counts[t].
divisions = int((counts[:, -1] - counts[:, 0]).sum())          # sum_seeds (final - initial)
eligible = int(counts[:, :-1].sum())                           # sum_seeds sum_{t<N} counts[t]
p_hat = divisions / eligible
p_theory = float(-np.expm1(-RATE * DT))
finals = counts[:, -1]

summary = {
    "role": "oracle", "model": "MechanicalRelaxation(Morse)+SaturatingCellGrowth+Division",
    "M_seeds": M, "n_steps": N_STEPS, "dt": DT, "capacity": CAP, "division_rate": RATE,
    "p_theory": p_theory,
    "p_hat": p_hat, "divisions_total": divisions, "eligible_cellsteps_total": eligible,
    "p_hat_se": float(np.sqrt(p_hat * (1 - p_hat) / eligible)),
    "final_count_mean": float(finals.mean()), "final_count_std": float(finals.std(ddof=1)),
    "final_count_min": int(finals.min()), "final_count_max": int(finals.max()),
    "final_counts": finals.tolist(),
    "smoke_82_in_sd": float((82 - finals.mean()) / finals.std(ddof=1)),
    "plexus_124_in_sd": float((124 - finals.mean()) / finals.std(ddof=1)),
    "wall_s": round(time.time() - t0, 1),
}
np.savez_compressed(os.path.join(OUT, "reference.npz"), counts=counts, finals=finals)
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
