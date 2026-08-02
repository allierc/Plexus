"""Differential test for `agitate` (BrownianDynamics thermal leg) -- the ORACLE side.

`agitate` is the constitutive core of the reference `BrownianDynamics` step with everything the
frozen Plexus language already owns stripped away: the deterministic drift F/gamma is delegated to
a separate pluggable pair-potential, so what is left is a temperature-controlled thermal (Brownian)
bath -- an isotropic Gaussian kick of per-cell displacement std sqrt(2 kT dt / gamma) per alive
cell per macro-step, giving the Einstein free-diffusion constant D = kT/gamma with the Wiener
sqrt(dt) DISPLACEMENT scaling. On the reference side this is EXACTLY BrownianDynamics with
potential=None (NoForce): forces()==0, so dx = std*xi, a free Brownian gas.

Because the JAX and torch PRNG streams differ, a pathwise trajectory can never match sample by
sample -- the right invariant to diff a stochastic bath on is a REALIZATION-INDEPENDENT statistic.
We use the free-diffusion constant read from the radius of gyration of the cloud:

    D_hat(dt) = Rg(T)^2 / (2 * n_dim * T),
    Rg(T)^2   = mean over alive cells of |r_i(T) - c(T)|^2   (c = live-cell centroid),

for a pure bath of N cells all seeded at ONE point, in free space, at total physical time
T = n_steps*dt held FIXED as dt varies. D is dt-INVARIANT iff the noise carries the Wiener
sqrt(dt) scaling (per-step per-dim variance 2 kT dt/gamma over T/dt steps -> T-total variance
2 kT T/gamma, dt-free); a noise that instead scaled with dt would give D = dt*kT/gamma and BREAK
the invariance -- so measuring at several dt is what makes this test bite on the one feature that
distinguishes `agitate` from the frozen operators' bolt-on `noise*randn` jitter.

Writes reference.npz (per-dt Rg^2 and MSD trajectories, D_hat, per-step increment variance) and
summary.json (D_ref per dt, theory kT/gamma, sampling error) into OUT. Asserts the oracle is a
function (deterministic at a fixed key) and that D_ref matches theory within the sampling band --
if either fails, the reference is not a usable oracle and we stop.
"""
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
import jax_morph as jxm
from jax_morph.physics import BrownianDynamics

OUT = os.environ["OUT"]

# JAX default is float32 -- MATCH the Plexus torch engine's float32 state; do NOT enable x64.
assert jnp.zeros(1).dtype == jnp.float32, "oracle must run in float32 to match torch"

SEED = 0
N = 20000                 # cells: sets the sampling error, rel std of Rg^2 = sqrt(2/(n_dim*N))
KT = 0.1                  # thermal energy (reference default)
GAMMA = 1.0               # translational drag (reference default); D_theory = kT/gamma = 0.1
NDIM = 2
T_TOTAL = 40.0            # physical end time, HELD FIXED across dt (the invariance is at fixed T)
DTS = [1.0, 0.5, 0.25]    # n_steps = T_TOTAL/dt = 40 / 80 / 160
D_THEORY = KT / GAMMA

# A pure thermal bath: BrownianDynamics with potential=None -> NoForce (forces()==0, no drift).
model = jxm.Model([BrownianDynamics(potential=None, n_space_dim=NDIM, gamma=GAMMA, kT=KT)])
State = jxm.build_state_from_model(model)


def seed_state():
    """N cells all at the origin (a single point) in free space; radius/celltype set for validity.

    The bath is translation-invariant and Rg removes the centroid, so the absolute seed point is
    irrelevant -- the Plexus side seeds the identical N-at-one-point cloud at its world centre.
    """
    s = State.init_empty(capacity=N, n_space_dim=NDIM, n_types=1)
    return s.update(
        alive=s.alive.at[:].set(True),
        position=s.position.at[:].set(0.0),      # every cell at (0, 0)
        radius=s.radius.at[:].set(0.5),
        celltype=s.celltype.at[:, 0].set(1.0),
    )


def run(dt, n_steps, key):
    return jxm.simulate(model, seed_state(), n_steps=n_steps, dt=dt, key=key, history=True)


def rg2_series(pos, alive):
    """Rg^2(t) = mean over alive cells of |r - centroid|^2, per recorded frame. pos [T+1,N,D]."""
    out = np.empty(pos.shape[0])
    for t in range(pos.shape[0]):
        live = alive[t].astype(bool)
        p = pos[t][live]
        c = p.mean(0)
        out[t] = float(((p - c) ** 2).sum(1).mean())
    return out


def msd_series(pos, alive):
    """MSD(t) = mean over alive cells of |r(t) - r(0)|^2 (from the common seed point)."""
    r0 = pos[0]
    out = np.empty(pos.shape[0])
    for t in range(pos.shape[0]):
        live = alive[t].astype(bool)
        d = pos[t][live] - r0[live]
        out[t] = float((d ** 2).sum(1).mean())
    return out


# --- the oracle must be a FUNCTION, not a process (deterministic at a fixed key) -------------- #
h_a = run(1.0, int(T_TOTAL / 1.0), jax.random.PRNGKey(SEED))
h_b = run(1.0, int(T_TOTAL / 1.0), jax.random.PRNGKey(SEED))
same = bool(np.array_equal(np.asarray(h_a.position), np.asarray(h_b.position)))
if not same:
    raise SystemExit("BrownianDynamics is NOT deterministic at a fixed key -- a differential test "
                     "against it would measure the reference's own noise. Stop here.")

# --- sweep dt at fixed total time; measure D from Rg(T) ------------------------------------- #
save = {}
per_dt = {}
for dt in DTS:
    n_steps = int(round(T_TOTAL / dt))
    h = run(dt, n_steps, jax.random.PRNGKey(SEED))
    pos = np.asarray(h.position)          # [n_steps+1, N, D]
    alive = np.asarray(h.alive)           # [n_steps+1, N]
    rg2 = rg2_series(pos, alive)
    msd = msd_series(pos, alive)
    T = float(np.asarray(h.t)[-1])        # == n_steps*dt == T_TOTAL
    D_hat = rg2[-1] / (2.0 * NDIM * T)
    # empirical per-dim increment variance, pooled over cells & steps: must be 2 kT dt / gamma.
    incr = np.diff(pos, axis=0)           # [n_steps, N, D]
    var_incr = float((incr ** 2).mean())  # per-dim variance (zero-mean noise)
    key = f"dt{dt}".replace(".", "p")
    save[f"{key}_rg2"] = rg2.astype(np.float32)
    save[f"{key}_msd"] = msd.astype(np.float32)
    save[f"{key}_t"] = np.asarray(h.t, np.float32)
    per_dt[str(dt)] = {
        "dt": dt, "n_steps": n_steps, "T": T,
        "D_hat": float(D_hat),
        "rg2_final": float(rg2[-1]),
        "msd_final": float(msd[-1]),
        "var_incr_per_dim": var_incr,
        "var_incr_theory_2kTdt_over_gamma": float(2.0 * KT * dt / GAMMA),
        "n_live_final": int(alive[-1].sum()),
    }
    print(f"dt={dt:<5} steps={n_steps:<4} Rg(T)^2={rg2[-1]:.4f} D_hat={D_hat:.5f} "
          f"(theory {D_THEORY:.5f})  var_incr/dim={var_incr:.5f} "
          f"(theory {2*KT*dt/GAMMA:.5f})", flush=True)

# sampling band on D_hat: endpoint Rg^2 over N iid cells -> rel std sqrt(2/(n_dim*N)).
rel_se = float(np.sqrt(2.0 / (NDIM * N)))
D_vals = np.array([per_dt[str(dt)]["D_hat"] for dt in DTS])

summary = {
    "role": "oracle", "step": "BrownianDynamics(potential=None) = free Brownian gas",
    "seed": SEED, "N_cells": N, "kT": KT, "gamma": GAMMA, "n_space_dim": NDIM,
    "T_total": T_TOTAL, "dts": DTS,
    "D_theory_kT_over_gamma": D_THEORY,
    "deterministic_at_fixed_key": same,
    "per_dt": per_dt,
    "D_hat_by_dt": {str(dt): per_dt[str(dt)]["D_hat"] for dt in DTS},
    "D_hat_mean": float(D_vals.mean()),
    "D_hat_spread_over_dt": float(D_vals.max() - D_vals.min()),  # ~0 iff dt-invariant (Wiener)
    "rel_sampling_error_per_side": rel_se,
    "D_rel_err_vs_theory": {str(dt): float(abs(per_dt[str(dt)]["D_hat"] - D_THEORY) / D_THEORY)
                            for dt in DTS},
}
np.savez_compressed(os.path.join(OUT, "reference.npz"),
                    dts=np.array(DTS, np.float32),
                    D_hat=D_vals.astype(np.float32),
                    D_theory=np.float32(D_THEORY), **save)
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

# --- guard: the oracle itself must match theory, else it is not a usable reference ----------- #
band = 5.0 * rel_se                      # 5 sigma: a loud guard on the reference, not the test bound
bad = {dt: r for dt, r in summary["D_rel_err_vs_theory"].items() if r > band}
if bad:
    raise SystemExit(f"reference D_hat departs from kT/gamma by > 5 sigma ({band:.3%}) at dt={bad} "
                     f"-- the oracle does not reproduce its own Einstein relation; stop.")
if not np.all(np.isfinite(D_vals)):
    raise SystemExit("reference produced a non-finite diffusion constant; stop.")
print(f"\noracle OK: D_ref dt-invariant at {D_vals.mean():.5f} (theory {D_THEORY}); "
      f"spread over dt {summary['D_hat_spread_over_dt']:.5f}, 1-sigma sampling {rel_se:.3%}")
print("wrote reference.npz, summary.json")
