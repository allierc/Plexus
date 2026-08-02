"""Oracle for `regulate:mwc` -- the reference GeneNetworkMWC vector field, on matched state.

The three ODEController siblings (connectionist / MWC / neural-ODE) share the same integration and
I/O and differ ONLY in the per-cell vector field f. So the differential test is built on f, not on
an integrated trajectory (which would conflate f with the Axis-A integrator: the engine's explicit
Euler vs the reference's adaptive Dopri5). This script emits, for ONE fixed parameter set / initial
gene state / fixed drivers:

  * G_ref[t] : the reference vector field integrated with an explicit-Euler step (dt, g0, u0), the
               SAME integrator the Plexus engine applies -- so a later trajectory diff isolates f.
  * dopri_delta : the reference's OWN adaptive Dopri5 one-macro-step delta y(dt)-y0 (the true
               __call__ behaviour), reported as an integration-gap diagnostic, not scored.
  * dg_ref1 : f on an adversarial batch (negatives / zeros / large positives) with the MAIN params.
  * dg_ref2 : f on a SECOND, extreme parameter set (log_K at its float32 underflow clip, mixed-sign
               H) that forces g/K -> +inf, exercising the overflow guard and the +inf/-inf -> NaN
               it prevents.

Everything (params, g0, u0, both adversarial batches, both trajectories) is written to
reference.npz so the Plexus-side differ reads ONE artefact and the two runs cannot drift apart.
"""
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
import diffrax

from jax_morph.control.ode import GeneNetworkMWC
from jax_morph.core.state import StateFieldSpec

OUT = os.environ["OUT"]

# JAX default is float32 -- MATCH the Plexus engine's float32 state; do NOT enable x64.
assert jnp.zeros(1).dtype == jnp.float32, "oracle must run in float32 to match torch"

N_GENE, N_IN, DT, N_FRAMES, N_CELLS = 4, 2, 1.0, 24, 4
rng = np.random.default_rng(20260731)


def f32(x):
    return np.asarray(x, dtype=np.float32)


# --- MAIN parameters: rich but well-scaled (tau ~ [0.8,1.6] => dt/tau < 2 => stable Euler sweep) -
P = dict(
    log_rho=f32(rng.normal(0.0, 0.3, N_GENE)),
    log_tau=f32(rng.normal(0.2, 0.25, N_GENE)),
    F0=f32(rng.normal(0.0, 0.5, N_GENE)),
    H_gene=f32(rng.normal(0.0, 0.8, (N_GENE, N_GENE))),      # signed couplings, incl. diagonal
    log_K_gene=f32(rng.normal(0.0, 0.5, (N_GENE, N_GENE))),  # K ~ O(1)
    H_in=f32(rng.normal(0.0, 0.8, (N_GENE, N_IN))),
    log_K_in=f32(rng.normal(0.0, 0.5, (N_GENE, N_IN))),
)
g0 = f32([0.0, 2.0, 0.5, 3.0])                                # varied initial concentrations
u0 = f32([1.0, 0.3])                                          # fixed drivers, held over dt

in_spec = StateFieldSpec("drive", shape=(N_IN,))
out_spec = StateFieldSpec("gene", shape=(N_GENE,))


def build(params):
    return GeneNetworkMWC((in_spec,), (out_spec,), hidden_size=0,
                          **{k: jnp.asarray(v) for k, v in params.items()})


model = build(P)


def vf(m, Y, U):
    """Reference per-cell vector field dg/dt on batched state, as float32 numpy."""
    return np.asarray(m.vector_field(0.0, jnp.asarray(f32(Y)), jnp.asarray(f32(U))), np.float32)


# --- (P) reference EULER trajectory, matching the engine (gene_{t+1} = gene_t + dt*f) ------------
#     engine records AFTER each tick's integrate, so G_ref[t] = state after (t+1) Euler steps.
Y = np.tile(g0, (N_CELLS, 1))                                 # N_CELLS identical rows (as the seed does)
U = np.tile(u0, (N_CELLS, 1))
G_ref = np.zeros((N_FRAMES + 1, N_CELLS, N_GENE), np.float32)
for t in range(N_FRAMES + 1):
    Y = f32(Y + DT * vf(model, Y, U))
    G_ref[t] = Y

# --- Dopri5 one-macro-step delta (the reference's real __call__ integrator), diagnostic only -----
term = diffrax.ODETerm(lambda t, y, args: model.vector_field(t, y, jnp.asarray(u0)[None, :]))
sol = diffrax.diffeqsolve(term, diffrax.Dopri5(), t0=0.0, t1=DT, dt0=DT, y0=jnp.asarray(g0)[None, :],
                          stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
                          saveat=diffrax.SaveAt(t1=True))
dopri_delta = np.asarray(sol.ys[-1] - jnp.asarray(g0)[None, :], np.float32)   # [1, N_GENE]
euler_delta = f32(DT * vf(model, g0[None, :], u0[None, :]))                   # [1, N_GENE]
dopri_vs_euler = float(np.max(np.abs(dopri_delta - euler_delta)))

# --- (A) adversarial batch on the MAIN params: negatives / zeros / large positives --------------
Y1 = f32([
    [0.0, 0.0, 0.0, 0.0],     # inert baseline: dg = rho*sigmoid(F0) - 0
    [-0.5, -2.0, -0.1, -5.0],  # negatives: occupancy clamps to 0, DECAY uses the raw negative
    [5.0, 8.0, 6.0, 7.0],     # large positives: saturation + large log-occupancy
    [0.5, -1.0, 3.0, 0.0],    # mixed signs across genes
    [1e-3, 2e-3, 0.0, 1e-2],  # near-zero positives
    [2.0, 0.5, 4.0, 1.5],     # ordinary
])
U1 = f32([
    [0.0, 0.0],
    [1.0, 3.0],
    [0.0, 2.0],
    [-1.0, 0.5],              # negative driver: clamps to 0 inside occupancy
    [4.0, 0.0],
    [0.7, 0.3],
])
dg_ref1 = vf(model, Y1, U1)

# --- (A) EXTREME params: log_K at the underflow clip + mixed-sign large H => overflow guard ------
P2 = dict(
    log_rho=f32([0.1, -0.2, 0.0, 0.3]),
    log_tau=f32([0.2, 0.3, 0.1, 0.25]),
    F0=f32([0.0, -0.5, 0.5, 0.2]),
    # log_K far below log(finfo.tiny) -> _positive_from_log clips to tiny -> K ~ 1.18e-38 -> g/K
    # overflows float32; the guard (min with finfo.max) must keep log1p finite on BOTH sides.
    log_K_gene=f32(np.full((N_GENE, N_GENE), -1000.0)),
    log_K_in=f32(np.full((N_GENE, N_IN), -1000.0)),
    # mixed-sign LARGE couplings: without the guard a row sums (+inf)+(-inf) = NaN.
    H_gene=f32([[10.0, -10.0, 8.0, -8.0],
                [-6.0, 6.0, -9.0, 9.0],
                [7.0, -7.0, 5.0, -5.0],
                [-4.0, 4.0, -3.0, 3.0]]),
    H_in=f32([[10.0, -10.0], [-8.0, 8.0], [6.0, -6.0], [-5.0, 5.0]]),
)
model2 = build(P2)
Y2 = f32([
    [5.0, 6.0, 5.0, 7.0],     # all > 4 => g/K overflows with tiny K
    [8.0, 4.5, 9.0, 5.5],
    [4.2, 10.0, 6.0, 4.1],
])
U2 = f32([[5.0, 6.0], [7.0, 4.5], [4.3, 8.0]])                # drivers > 4 => input overflow too
dg_ref2 = vf(model2, Y2, U2)

# --- save everything the differ needs (single source of truth) ----------------------------------
np.savez_compressed(
    os.path.join(OUT, "reference.npz"),
    n_gene=N_GENE, n_in=N_IN, dt=DT, n_frames=N_FRAMES, n_cells=N_CELLS,
    g0=g0, u0=u0,
    **{f"P_{k}": v for k, v in P.items()},
    G_ref=G_ref, dopri_delta=dopri_delta, euler_delta=euler_delta,
    Y1=Y1, U1=U1, dg_ref1=dg_ref1,
    **{f"P2_{k}": v for k, v in P2.items()},
    Y2=Y2, U2=U2, dg_ref2=dg_ref2,
)

summary = {
    "n_gene": N_GENE, "n_in": N_IN, "dt": DT, "n_frames": N_FRAMES, "n_cells": N_CELLS,
    "g0": g0.tolist(), "u0": u0.tolist(),
    "G_ref_first": G_ref[0, 0].tolist(),
    "G_ref_last": G_ref[-1, 0].tolist(),
    "G_ref_abs_max": float(np.max(np.abs(G_ref))),
    "dopri_delta": dopri_delta[0].tolist(),
    "euler_delta": euler_delta[0].tolist(),
    "dopri_vs_euler_one_step": dopri_vs_euler,
    "dg_ref1_abs_max": float(np.max(np.abs(dg_ref1))),
    "dg_ref2_abs_max": float(np.max(np.abs(dg_ref2))),
    "dg_ref1_finite": bool(np.isfinite(dg_ref1).all()),
    "dg_ref2_finite": bool(np.isfinite(dg_ref2).all()),
}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
if not (summary["dg_ref1_finite"] and summary["dg_ref2_finite"]):
    raise SystemExit("reference produced non-finite dg -- the overflow guard did not hold; stop.")
print("wrote reference.npz, summary.json")
