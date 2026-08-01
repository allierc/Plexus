"""Differential test for `reorient` -- the ORACLE side.

`reorient` is the ONE uncovered leg of jax-morph's ActiveBrownianDynamics2D: the rotational
diffusion of the persistent heading, dtheta = sqrt(2 D_r dt) * xi, applied to the orientation.
(The other two legs -- the passive drift F/gamma and the self-propulsion v0*e + translational
noise -- already alias registered contracts: a pair potential under an overdamped mobility, and
`glide` + its `noise`.) So this test ISOLATES the rotational leg of the reference and measures the
one observable it controls: the orientational decorrelation of the heading.

Isolation recipe (examples/04_physics_examples.ipynb, the textbook active-Brownian particle):

    ActiveBrownianDynamics2D(None, n_space_dim=2, kT=0.0, rot_diffusion=D_r)

potential = None -> NoForce (zero drift); kT = 0 -> no translational noise; self-propulsion
active_speed = v0 (only moves positions -- irrelevant to the heading metric, matched so the two
runs are the SAME free ABP). The heading channel is then driven by NOTHING but the rotational
term. The metric is the ensemble orientational autocorrelation C(t) = <e(t).e(0)>, which decays
as exp(-D_r t dt) in 2-D and is the physical signature of the rotational-diffusion leg (it sets
the walker's persistence length and its ballistic->diffusive crossover).

The comparison is DISTRIBUTIONAL, not pathwise: JAX and torch draw independent rotational-noise
streams, so a seed-0-vs-seed-0 per-cell match is impossible; the ensemble average over N cells is
the realization-independent invariant (exactly as the division differ pooled a per-step hazard).

Writes reference.npz (the heading trajectory + the C / MSD curves) and summary.json into OUT.
"""
import json, os, time
import numpy as np
import jax, jax.numpy as jnp
import jax_morph as jxm
from jax_morph.physics import ActiveBrownianDynamics2D

OUT = os.environ["OUT"]
N, N_STEPS, DT = 20000, 40, 1.0
D_R, V0 = 0.1, 0.3            # rotational diffusion rate; self-propulsion speed (heading-metric-independent)
SEED = 0

# The isolated rotational-diffusion leg: a free active-Brownian gas.
model = jxm.Model([ActiveBrownianDynamics2D(None, n_space_dim=2, kT=0.0, rot_diffusion=D_R)])


def seed_state(head_key):
    """N cells at a common origin (positions dynamically irrelevant under NoForce; a point start
    just makes the MSD cross-check clean), each with a self-propulsion speed v0 and a heading drawn
    UNIFORMLY on the circle -- the matched initial heading distribution."""
    s = jxm.build_state_from_model(model).init_empty(capacity=N, n_space_dim=2, n_types=1)
    theta0 = jax.random.uniform(head_key, (N,), minval=0.0, maxval=2.0 * jnp.pi)
    return s.update(
        alive=s.alive.at[:].set(True),
        radius=s.radius.at[:].set(0.15),
        position=s.position.at[:].set(0.0),
        celltype=s.celltype.at[:, 0].set(1.0),
        active_speed=s.active_speed.at[:].set(V0),
        active_heading=s.active_heading.at[:].set(theta0),
    )


HEAD_KEY = jax.random.PRNGKey(1234)          # FIXED initial-heading draw (reproducible IC)


def run(sim_key):
    return jxm.simulate(model, seed_state(HEAD_KEY), n_steps=N_STEPS, dt=DT, key=sim_key,
                        history=True)


# --- the oracle must be a function, not a process -------------------------------------------- #
t0 = time.time()
h1 = run(jax.random.PRNGKey(SEED))
h2 = run(jax.random.PRNGKey(SEED))
same = bool(np.array_equal(np.asarray(h1.active_heading), np.asarray(h2.active_heading)))
if not same:
    raise SystemExit("ORACLE IS NOT DETERMINISTIC at a fixed key -- a differential test against "
                     "it would measure the reference's own noise. Stop here.")

theta = np.asarray(h1.active_heading)                       # [T+1, N] unwrapped scalar angle
pos = np.asarray(h1.position)                              # [T+1, N, 2]
alive = np.asarray(h1.alive)                               # [T+1, N] (all True here)
e = np.stack([np.cos(theta), np.sin(theta)], axis=-1)      # [T+1, N, 2] unit heading vectors

# --- the metric: ensemble orientational autocorrelation C(t) = <e(t).e(0)> ------------------- #
e0 = e[0]                                                   # heading at frame 0 (0 reorient steps)
C = np.einsum("tnd,nd->tn", e, e0).mean(axis=1)            # [T+1], averaged over cells

# --- corroborating invariants ---------------------------------------------------------------- #
dtheta = theta[1:] - theta[:-1]                            # [T, N] per-step increment (unwrapped)
var_dtheta = float(dtheta.var())                          # pooled; theory = 2 D_r dt
mean_dtheta = float(dtheta.mean())                        # theory 0 (zero-drift symmetry)

tt = np.arange(N_STEPS + 1) * DT                          # physical time per frame
sig = C > 0.05                                             # frames where -ln C is well defined
sig[0] = False                                            # drop t=0 (x=0,y=0 carries no slope info)
x, y = tt[sig], -np.log(C[sig])
D_r_eff = float((x * y).sum() / (x * x).sum())            # LS slope through origin: C ~ exp(-D_r_eff t)

msd = ((pos - pos[0]) ** 2).sum(-1).mean(1)               # [T+1] mean-squared displacement
# ABP theory: MSD(t) = (2 v0^2/D_r^2)(D_r t - 1 + exp(-D_r t)); ballistic (t^2) -> diffusive (t)
msd_theory = (2 * V0 ** 2 / D_R ** 2) * (D_R * tt - 1 + np.exp(-D_R * tt))

np.savez_compressed(os.path.join(OUT, "reference.npz"),
                    t=tt, C=C, active_heading=theta.astype(np.float32),
                    msd=msd, msd_theory=msd_theory)

summary = {
    "role": "oracle", "model": "ActiveBrownianDynamics2D(NoForce, kT=0, rot_diffusion=D_r)",
    "N": N, "n_steps": N_STEPS, "dt": DT, "rot_diffusion": D_R, "active_speed": V0, "seed": SEED,
    "deterministic_at_fixed_key": same,
    "C": C.tolist(),
    "C_theory_final": float(np.exp(-D_R * N_STEPS * DT)),
    "C_final": float(C[-1]),
    "D_r_eff": D_r_eff, "D_r_input": D_R,
    "var_dtheta": var_dtheta, "var_dtheta_theory": 2 * D_R * DT,
    "mean_dtheta": mean_dtheta,
    "msd_final": float(msd[-1]), "msd_theory_final": float(msd_theory[-1]),
    "wall_s": round(time.time() - t0, 1),
}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps({k: v for k, v in summary.items() if k != "C"}, indent=2))
print(f"C(t) sampled: t0={C[0]:.4f} t5={C[5]:.4f} t10={C[10]:.4f} t20={C[20]:.4f} "
      f"t40={C[-1]:.4f}  (theory exp(-0.1 t): "
      f"{np.exp(-0.1*5):.4f} {np.exp(-0.1*10):.4f} {np.exp(-0.1*20):.4f} {np.exp(-0.1*40):.4f})")
