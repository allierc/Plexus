"""odecontroller -- the reference trajectory for the `regulate` differential test.

Integrates jax-morph's GeneNetworkConnectionist (the concrete ODEController subclass, the closest
shipped circuit to the paper's genetic regulatory interactions) over 21 macro-steps and records
the per-cell gene trajectory, so the Plexus `regulate` operator can be diffed against it on the
SAME initial condition and the SAME circuit.

The circuit is a 3-gene connectionist network -- a hidden_size=1 latent regulator plus a width-2
output gene (reference field order [gene_hidden, g_out0, g_out1]) -- driven by a width-2 frozen
input, with a deliberately ASYMMETRIC W_gene (zero diagonal, mixed-sign off-diagonal cross-talk),
a mixed-sign W_in, a per-gene bias b, and gamma=1.0 so the fixed points sit at O(1) and the
transient from g=0 is fully exercised over the interval. Reaction law (source, ode.py:261):

    dg/dt = sigma( g @ W_gene^T + u @ W_in^T + b ) - gamma * g      (u frozen over [0, dt])

with sigma the algebraic rescaled sigmoid. This exercises cross-regulation (off-diagonal W_gene),
the frozen drive INSIDE the sigmoid (the paper-vs-code fork; source wins), the bias, linear
degradation, the self-solved adaptive Dopri5 increment, and the hidden/output split.

Two runs are recorded:
  * UNIFORM   -- one shared drive for all 4 cells. This is the run the Plexus spec (config/atlas_jax/
                 odecontroller.yaml, seeded by the uniform `seed_state` harness op) reproduces, so
                 it is the run the scalar diff_metric is computed on.
  * DISTINCT  -- a different frozen drive per cell. All parameters are shared; only the per-cell
                 drive differs, so the four cells follow four different trajectories. If our
                 operator leaked any cell-to-cell coupling (a violation of the maps=[] intracellular
                 identity that separates `regulate` from `signal`), this run would diverge where the
                 uniform run could not. Corroboration only; not the primary metric.

The oracle must be a FUNCTION, not a process: the uniform run is executed twice from the same key
and asserted bit-identical before anything is written down (a deterministic model has no key, but
we check anyway -- the differential test is meaningless if the reference is not reproducible).
"""
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
import jax_morph as jxm
from jax_morph.control import GeneNetworkConnectionist
from jax_morph.core.state import StateFieldSpec

OUT = os.environ["OUT"]
CAP, N_LIVE, N_STEPS, DT = 8, 4, 21, 1.0

# --- the shared circuit ----------------------------------------------------------------------- #
W_GENE = [[0.0, 0.8, -0.5], [-0.6, 0.0, 0.7], [0.4, -0.3, 0.0]]   # zero diagonal, mixed-sign cross-talk
W_IN = [[1.2, 0.0], [0.0, -0.9], [0.5, 0.5]]                      # each gene reads the 2 drivers differently
B = [0.1, -0.2, 0.0]                                             # per-gene basal drive
GAMMA = 1.0                                                      # fixed points at O(1); transient fully seen at dt=1

U_UNIFORM = [0.8, 0.3]                                           # one frozen drive, all cells
U_DISTINCT = [[0.8, 0.3], [-0.5, 1.0], [0.2, -0.7], [1.5, 0.5]]  # a different frozen drive per cell

INPUT_SPECS = (StateFieldSpec("drive", shape=(2,)),)             # in_size = 2
OUTPUT_SPECS = (StateFieldSpec("g_out", shape=(2,)),)           # out_size = 2
HIDDEN = 1                                                       # -> n_gene = 3, y = [gene_hidden, g_out0, g_out1]


def controller():
    return GeneNetworkConnectionist(
        INPUT_SPECS, OUTPUT_SPECS, hidden_size=HIDDEN,
        W_gene=jnp.asarray(W_GENE), W_in=jnp.asarray(W_IN), b=jnp.asarray(B),
        gamma=GAMMA, tag="gene",
    )


def seed(model, drive):
    """An empty state with N_LIVE cells alive, genes at 0, and a per-cell frozen `drive`.

    `drive` is either a length-2 vector (broadcast to every cell) or an (N_LIVE, 2) array.
    """
    StateCls = jxm.build_state_from_model(model)
    s = StateCls.init_empty(capacity=CAP, n_space_dim=2, n_types=1)
    u = jnp.broadcast_to(jnp.asarray(drive, dtype=s.g_out.dtype), (N_LIVE, 2))
    return s.update(
        alive=s.alive.at[:N_LIVE].set(True),
        celltype=s.celltype.at[:N_LIVE, 0].set(1.0),
        drive=s.drive.at[:N_LIVE].set(u),
        g_out=s.g_out.at[:N_LIVE].set(0.0),
        gene_hidden=s.gene_hidden.at[:N_LIVE].set(0.0),
    )


def rollout(drive, key):
    model = jxm.Model([controller()])
    h = jxm.simulate(model, seed(model, drive), n_steps=N_STEPS, dt=DT, key=key, history=True)
    # y = concat(hidden, outputs) in the reference's declaration order -> [N_STEPS+1, CAP, 3].
    y = np.concatenate([np.asarray(h.gene_hidden), np.asarray(h.g_out)], axis=-1)
    return y, np.asarray(h.alive)


# --- the reference must be reproducible ------------------------------------------------------- #
y_uniform, alive = rollout(U_UNIFORM, jax.random.PRNGKey(0))
y_uniform2, _ = rollout(U_UNIFORM, jax.random.PRNGKey(0))
same = bool(np.array_equal(y_uniform, y_uniform2))
if not same:
    raise SystemExit("ORACLE IS NOT DETERMINISTIC at a fixed key -- a differential test against it "
                     "would measure the reference's own noise. Stop here.")

y_distinct, alive_d = rollout(U_DISTINCT, jax.random.PRNGKey(0))

# gene order is load-bearing for the diff; write it down next to the arrays.
GENE_ORDER = ["gene_hidden", "g_out0", "g_out1"]

np.savez_compressed(
    os.path.join(OUT, "reference.npz"),
    y_uniform=y_uniform.astype(np.float32),           # [22, 8, 3] = initial ++ after 1..21 steps
    y_distinct=y_distinct.astype(np.float32),          # [22, 8, 3]
    alive=alive.astype(bool),                          # [22, 8]
    W_gene=np.asarray(W_GENE, np.float32), W_in=np.asarray(W_IN, np.float32),
    b=np.asarray(B, np.float32), gamma=np.float32(GAMMA),
    u_uniform=np.asarray(U_UNIFORM, np.float32), u_distinct=np.asarray(U_DISTINCT, np.float32),
    n_steps=np.int64(N_STEPS), dt=np.float32(DT), n_live=np.int64(N_LIVE),
    hidden_size=np.int64(HIDDEN), gene_order=np.asarray(GENE_ORDER),
)

live = alive[-1].astype(bool)
summary = {
    "circuit": "GeneNetworkConnectionist, n_gene=3 (hidden=1 ++ out=2), n_in=2, gamma=1.0",
    "n_steps": N_STEPS, "dt": DT, "n_live": N_LIVE, "capacity": CAP,
    "hidden_size": HIDDEN, "gene_order": GENE_ORDER,
    "deterministic_at_fixed_key": same,
    "reaction_law": "dg/dt = sigma(g@W_gene^T + u@W_in^T + b) - gamma*g  (u frozen; algebraic sigmoid)",
    "gene_first_uniform": np.asarray(y_uniform[0][live]).mean(0).tolist(),   # all 0 (shared init)
    "gene_last_uniform": np.asarray(y_uniform[-1][live]).mean(0).tolist(),   # fixed-point approach, cell-mean
    "gene_last_uniform_percell": np.asarray(y_uniform[-1][:N_LIVE]).tolist(),
    "gene_last_distinct_percell": np.asarray(y_distinct[-1][:N_LIVE]).tolist(),
    "distinct_spread_last": float(
        np.asarray(y_distinct[-1][:N_LIVE]).max(0).max() - np.asarray(y_distinct[-1][:N_LIVE]).min(0).min()),
}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
print("wrote reference.npz (y_uniform, y_distinct, alive, params) and summary.json")
