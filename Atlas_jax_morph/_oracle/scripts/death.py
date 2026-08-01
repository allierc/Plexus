"""Oracle for growth:apoptose -- the reference jax-morph `Death` step, on a matched IC.

Death is a DISCRETE stochastic removal: each live cell dies over a macro-step as an independent
Bernoulli event with per-cell hazard p = 1 - exp(-clip(death_rate,0)*dt). The paper models NO
death (see the atlas entry's surprises); this is a shipped-library step, so we translate the CODE
and diff the RUNNING source.

A stochastic operator cannot be diffed bit-for-bit across the JAX/torch boundary -- the reference's
`jax.random.bernoulli` draw and Plexus's `torch.rand` draw are different RNG streams. So this run
records the observable a death process is actually judged by, the POPULATION SURVIVAL CURVE, plus
an EXACT deterministic export of the hazard map that both sides must reproduce identically:

  1. reference.npz['t'], ['counts'], ['survival']  -- a pure-death rollout of N0 = 50000 cells at a
     UNIFORM hazard lambda = 0.05, 40 macro-steps at dt = 1.0, Death the ONLY step (no
     division/growth/relaxation, so the live count can only fall). survival(t) = n_alive(t)/N0.
  2. reference.npz['hz_rates'], ['hz_dts'], ['hz_p']  -- p = Death._dist over a grid of death_rate
     (incl. NEGATIVE rates, to exercise the >=0 clip) x dt. The Plexus side recomputes the torch
     expression `apoptose` uses and must match to float precision.

Determinism is asserted (same key -> bit-identical alive history) BEFORE anything is written: a
diff against a non-deterministic reference would measure the reference's own noise.
"""
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
import jax_morph as jxm
from jax_morph.physics import Death

OUT = os.environ["OUT"]
SEED, N0, N_STEPS, DT, LAMBDA = 0, 50000, 40, 1.0, 0.05

model = jxm.Model([Death()])


def seed_state(rate, n=N0, cap=N0):
    """n live cells, all carrying a uniform per-cell death_rate = `rate`. Position is irrelevant
    (Death reads none); we leave it at the init_empty default."""
    s = jxm.build_state_from_model(model).init_empty(capacity=cap, n_space_dim=2, n_types=1)
    return s.update(
        alive=s.alive.at[:n].set(True),
        celltype=s.celltype.at[:n, 0].set(1.0),
        death_rate=s.death_rate.at[:n].set(rate),
    )


def run(key, rate=LAMBDA):
    return jxm.simulate(model, seed_state(rate), n_steps=N_STEPS, dt=DT, key=key, history=True)


# --- the oracle must be a FUNCTION, not a process ------------------------------------------- #
h1 = run(jax.random.PRNGKey(SEED))
h2 = run(jax.random.PRNGKey(SEED))
same = bool(np.array_equal(np.asarray(h1.alive), np.asarray(h2.alive)))
if not same:
    raise SystemExit("ORACLE Death IS NOT DETERMINISTIC at a fixed key -- a differential test "
                     "against it would measure the reference's own noise. Stop here.")
# and it must ACTUALLY be stochastic (a different key must move the realisation), else 'agreement'
# would be vacuous.
h3 = run(jax.random.PRNGKey(SEED + 1))
stochastic = bool(not np.array_equal(np.asarray(h1.alive), np.asarray(h3.alive)))

t = np.asarray(h1.t, dtype=float)
counts = np.asarray(h1.alive.sum(axis=1), dtype=np.int64)     # live cells per frame
survival = counts.astype(np.float64) / float(N0)
analytic = np.exp(-LAMBDA * t)                                 # S_true(t) = exp(-lambda * t)

# --- EXACT hazard map p = 1 - exp(-clip(rate,0)*dt) via the reference's own Death._dist ------ #
hz_rates = np.array([-1.0, -0.05, 0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0], dtype=np.float64)
hz_dts = np.array([0.5, 1.0, 2.0], dtype=np.float64)
step = model.steps[0]
small = jxm.build_state_from_model(model).init_empty(capacity=len(hz_rates), n_space_dim=2, n_types=1)
small = small.update(alive=small.alive.at[:].set(True),
                     death_rate=small.death_rate.at[:].set(jnp.asarray(hz_rates)))
hz_p = np.stack([np.asarray(step._dist(small, float(dt)), dtype=np.float64) for dt in hz_dts])

np.savez_compressed(
    os.path.join(OUT, "reference.npz"),
    t=t, counts=counts, survival=survival, analytic=analytic,
    hz_rates=hz_rates, hz_dts=hz_dts, hz_p=hz_p,
)

summary = {
    "seed": SEED, "N0": N0, "n_steps": N_STEPS, "dt": DT, "lambda": LAMBDA,
    "deterministic_at_fixed_key": same, "stochastic_across_keys": stochastic,
    "per_step_p": float(1.0 - np.exp(-LAMBDA * DT)),
    "cells_first": int(counts[0]), "cells_last": int(counts[-1]),
    "survival_first": float(survival[0]), "survival_last": float(survival[-1]),
    "analytic_last": float(analytic[-1]),
    "survival_vs_analytic_max": float(np.abs(survival - analytic).max()),
    "counts": counts.tolist(),
}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

if not stochastic:
    raise SystemExit("reference Death gave the SAME realisation under two keys -- not actually "
                     "sampling; a survival-curve diff would be vacuous. Stop.")

# --- one figure: the survival curve on its analytic law ------------------------------------- #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t, survival, "o-", color="tab:red", ms=4, label=f"reference Death (N0={N0})")
ax.plot(t, analytic, "-", color="black", lw=1, label=r"analytic $e^{-\lambda t}$")
ax.set_xlabel("macro-step t"); ax.set_ylabel("live fraction S(t)")
ax.set_title(f"oracle: Death, uniform lambda={LAMBDA}, dt={DT}")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT, "reference.png"), dpi=130, bbox_inches="tight")
print("wrote reference.npz, summary.json, reference.png")
