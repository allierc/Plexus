"""Differential test for `adhere:hertzian` -- the ORACLE side (jax-morph Hertzian).

Hertzian is a pure FORCE law (a pair-interaction ENERGY autodiffed to a force), so this isolates
the force, exactly as the saturating-growth differ isolated the radius ODE. Three artefacts, all on
initial conditions the Plexus side reproduces bit-for-bit:

  H  HETEROGENEOUS FORCE FIELD (the primary contract test). One fixed 7-cell overlapping cluster
     with UNEQUAL radii (0.40-0.70) and PER-CELL epsilon. F_ref = Hertzian(epsilon).forces(state)
     -- the autodiff force of E = 0.5*sum_{i!=j alive} (2/5)eps(1-r/sigma)^(5/2). Exercises the
     additive contact distance sigma = r_i + r_j (size-consistency) and the arithmetic-mean per-cell
     epsilon mix -- the two combining rules that distinguish `adhere` from `attraction_repulsion`.
     Also SoftSphere(epsilon).forces on the SAME state: the exponent-2-vs-2.5 NEGATIVE CONTROL.

  S  2-CELL RADIAL SCAN vs ANALYTIC. Confirms the reference truly implements the Hertzian law
     f(r) = (eps/sigma)(1 - r/sigma)^(3/2), compact on [0, sigma) -- so the differ is scored against
     the real potential, not a mis-configured reference (guards the oracle itself).

  U  UNIFORM DETERMINISTIC TRAJECTORY (the dynamical / engine-composition test). BrownianDynamics(
     Hertzian, kT=0, gamma=1) is overdamped forward Euler Dx = dt*forces -- byte-identical to the
     Plexus engine's EMIT=velocity path x += dt*v at mobility=1. Same 7-cell cluster, UNIFORM radius
     0.5, epsilon 2.0, dt=0.1, T=40 macro-steps; the cluster relaxes from overlapping to
     just-touching, sweeping the whole force curve.

Writes reference.npz + summary.json + reference.png into OUT.
"""
import os, json
import numpy as np
import jax, jax.numpy as jnp
import jax_morph as jxm
from jax_morph.physics import Hertzian, SoftSphere, BrownianDynamics
from jax_morph.core.state import StateFieldSpec

OUT = os.environ["OUT"]

# origin-centred 7-cell cluster (the Plexus engine spec is the same geometry translated by +[20,20];
# forces and centroid-aligned trajectories are translation-invariant).
POS7 = np.array([[0.0, 0.0], [0.9, 0.1], [0.45, 0.8], [-0.6, 0.5],
                 [0.3, -0.7], [1.2, 0.7], [-0.3, -0.5]], np.float32)
RAD_HET = np.array([0.70, 0.50, 0.60, 0.40, 0.55, 0.45, 0.65], np.float32)
EPS_HET = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 1.2, 2.2], np.float32)
RAD_UNI, EPS_UNI, DT, T = 0.5, 2.0, 0.1, 40

eps_spec = StateFieldSpec('epsilon', heritable=True)
MODEL_FIELD = jxm.Model([BrownianDynamics(Hertzian(epsilon=eps_spec), n_space_dim=2, kT=0.0)])
MODEL_UNI = jxm.Model([BrownianDynamics(Hertzian(epsilon=EPS_UNI), n_space_dim=2, kT=0.0, gamma=1.0)])


def state_percell_eps(model, pos, radius, eps):
    pos = jnp.asarray(np.asarray(pos, np.float32))
    N = pos.shape[0]
    s = jxm.build_state_from_model(model).init_empty(capacity=N, n_space_dim=2, n_types=1)
    return s.update(alive=s.alive.at[:N].set(True),
                    radius=s.radius.at[:N].set(jnp.asarray(np.asarray(radius, np.float32))),
                    position=s.position.at[:N].set(pos), celltype=s.celltype.at[:N, 0].set(1.0),
                    epsilon=s.epsilon.at[:N].set(jnp.asarray(np.asarray(eps, np.float32))))


def state_shared_eps(model, pos, radius):
    pos = jnp.asarray(np.asarray(pos, np.float32))
    N = pos.shape[0]
    s = jxm.build_state_from_model(model).init_empty(capacity=N, n_space_dim=2, n_types=1)
    return s.update(alive=s.alive.at[:N].set(True),
                    radius=s.radius.at[:N].set(jnp.asarray(np.asarray(radius, np.float32))),
                    position=s.position.at[:N].set(pos), celltype=s.celltype.at[:N, 0].set(1.0))


# ---------------------------------------------------------------------------------------------- #
#  H -- heterogeneous force field (per-cell radius + per-cell epsilon), + the SoftSphere control
# ---------------------------------------------------------------------------------------------- #
stH = state_percell_eps(MODEL_FIELD, POS7, RAD_HET, EPS_HET)
F_ref_H = np.asarray(Hertzian(epsilon=eps_spec).forces(stH))          # [7,2]
F_ss_H = np.asarray(SoftSphere(epsilon=eps_spec).forces(stH))         # [7,2] wrong-exponent control
# per-pair overlap census (how many pairs actually interact -- the test must exercise the law)
sig = RAD_HET[:, None] + RAD_HET[None, :]
rij = np.linalg.norm(POS7[:, None] - POS7[None, :], axis=-1)
overlapping = int(((rij < sig) & ~np.eye(7, dtype=bool)).sum() // 2)
nc_softsphere_rel = float(np.abs(F_ss_H - F_ref_H).max() / max(1e-12, np.abs(F_ref_H).max()))

# ---------------------------------------------------------------------------------------------- #
#  S -- 2-cell radial scan vs analytic f(r) = (eps/sigma)(1-r/sigma)^1.5
# ---------------------------------------------------------------------------------------------- #
r_i, r_j, eps_s = 0.5, 0.5, 2.0
sigma_s = r_i + r_j
seps = np.linspace(0.3, 1.3, 21).astype(np.float32)
f_ref_scan, f_an_scan = [], []
for r in seps:
    st = state_shared_eps(MODEL_UNI, [[0.0, 0.0], [float(r), 0.0]], [r_i, r_j])
    F = np.asarray(Hertzian(epsilon=eps_s).forces(st))
    f_ref_scan.append(float(F[1, 0]))
    ov = max(0.0, 1.0 - float(r) / sigma_s)
    f_an_scan.append((eps_s / sigma_s) * ov ** 1.5 if r < sigma_s else 0.0)
f_ref_scan = np.array(f_ref_scan); f_an_scan = np.array(f_an_scan)
scan_ref_vs_analytic_max = float(np.abs(f_ref_scan - f_an_scan).max())

# ---------------------------------------------------------------------------------------------- #
#  U -- uniform deterministic overdamped-Euler trajectory (kT=0)
# ---------------------------------------------------------------------------------------------- #
s0 = state_shared_eps(MODEL_UNI, POS7, np.full(7, RAD_UNI, np.float32))
hU = jxm.simulate(MODEL_UNI, s0, n_steps=T, dt=DT, key=jax.random.PRNGKey(0), history=True)
posU = np.asarray(hU.position)[:, :7, :]                              # [T+1,7,2]
aliveU = np.asarray(hU.alive)[:, :7]
# determinism at a second key (kT=0 must be key-independent), per the oracle contract
hU2 = jxm.simulate(MODEL_UNI, s0, n_steps=T, dt=DT, key=jax.random.PRNGKey(123), history=True)
det_U = float(np.abs(posU - np.asarray(hU2.position)[:, :7, :]).max())
if det_U != 0.0:
    raise SystemExit(f"BrownianDynamics(kT=0) is not deterministic across keys ({det_U}); "
                     "a differential against it would measure the reference's own noise. Stop.")
# first Euler step must equal dt*forces(s0) (confirms Dx = dt*forces)
F_uni0 = np.asarray(Hertzian(epsilon=EPS_UNI).forces(s0))[:7]
first_step_resid = float(np.abs((posU[1] - posU[0]) - DT * F_uni0).max())

summary = {
    "role": "oracle", "potential": "Hertzian",
    "x64_enabled": bool(jax.config.read("jax_enable_x64")), "dtype": str(F_ref_H.dtype),
    "H_n_cells": 7, "H_overlapping_pairs": overlapping,
    "H_force_maxabs": float(np.abs(F_ref_H).max()),
    "H_radii": RAD_HET.tolist(), "H_epsilon": EPS_HET.tolist(),
    "negative_control_softsphere_vs_hertzian_rel": nc_softsphere_rel,
    "scan_ref_vs_analytic_max": scan_ref_vs_analytic_max,
    "U_dt": DT, "U_T": T, "U_radius": RAD_UNI, "U_epsilon": EPS_UNI,
    "U_deterministic_across_keys": det_U,
    "U_first_step_vs_dt_forces_max": first_step_resid,
    "U_gyration_first": float(np.sqrt(((posU[0] - posU[0].mean(0)) ** 2).sum(1).mean())),
    "U_gyration_last": float(np.sqrt(((posU[-1] - posU[-1].mean(0)) ** 2).sum(1).mean())),
}

np.savez_compressed(
    os.path.join(OUT, "reference.npz"),
    pos7=POS7, rad_het=RAD_HET, eps_het=EPS_HET, F_ref_H=F_ref_H, F_ss_H=F_ss_H,
    seps=seps, f_ref_scan=f_ref_scan, f_an_scan=f_an_scan,
    posU=posU, aliveU=aliveU, F_uni0=F_uni0,
    rad_uni=np.float32(RAD_UNI), eps_uni=np.float32(EPS_UNI), dt=np.float32(DT), T=np.int32(T),
)
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

# --- one sanity figure: the 2-cell scan (ref vs analytic) + trajectory gyration -------------- #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(seps, f_an_scan, "k-", lw=2, label="analytic (eps/sigma)(1-r/sigma)^1.5")
ax1.plot(seps, f_ref_scan, "r.", ms=8, label="jax-morph Hertzian.forces")
ax1.axvline(sigma_s, ls="--", c="gray", label="contact sigma=1.0")
ax1.set_xlabel("separation r"), ax1.set_ylabel("radial force"), ax1.legend(fontsize=8)
ax1.set_title("S: reference vs analytic Hertzian force")
gyr = [float(np.sqrt(((posU[t] - posU[t].mean(0)) ** 2).sum(1).mean())) for t in range(T + 1)]
ax2.plot(np.arange(T + 1) * DT, gyr, "b-")
ax2.set_xlabel("t"), ax2.set_ylabel("gyration radius")
ax2.set_title("U: uniform cluster relaxing (kT=0)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "reference.png"), dpi=120)
print("wrote reference.npz, summary.json, reference.png")
