"""Probe the jax-morph Hertzian API before writing the full differential.

Confirms: (1) dtype (float32 vs x64), (2) Hertzian(epsilon).forces(state) works and equals the
analytic radial force f(r)=(eps/sigma)(1-r/sigma)^1.5 on a 2-cell config, (3) per-cell epsilon
mixes by the arithmetic mean, (4) BrownianDynamics(Hertzian, kT=0) is deterministic forward Euler
Dx = dt*forces, (5) SoftSphere vs Hertzian differ (the negative control is real).
"""
import os, json
import numpy as np
import jax, jax.numpy as jnp
import jax_morph as jxm
from jax_morph.physics import Hertzian, SoftSphere, BrownianDynamics

OUT = os.environ["OUT"]
out = {}
out["x64_enabled"] = bool(jax.config.read("jax_enable_x64"))

# --- build a Hertzian-only model + state ---------------------------------------------------- #
model = jxm.Model([BrownianDynamics(Hertzian(epsilon=2.0), n_space_dim=2, kT=0.0, gamma=1.0)])
pot = Hertzian(epsilon=2.0)

def state_of(pos, radius, n_types=1):
    pos = jnp.asarray(np.asarray(pos, np.float32))
    radius = jnp.asarray(np.asarray(radius, np.float32))
    N = pos.shape[0]
    s = jxm.build_state_from_model(model).init_empty(capacity=N, n_space_dim=2, n_types=1)
    return s.update(alive=s.alive.at[:N].set(True), radius=s.radius.at[:N].set(radius),
                    position=s.position.at[:N].set(pos), celltype=s.celltype.at[:N, 0].set(1.0))

# --- 2-cell radial scan: forces vs analytic ------------------------------------------------- #
r_i, r_j, eps = 0.5, 0.5, 2.0
sigma = r_i + r_j
seps = np.linspace(0.3, 1.3, 11)          # from deep overlap to beyond contact
f_ref, f_an = [], []
for r in seps:
    st = state_of([[0.0, 0.0], [float(r), 0.0]], [r_i, r_j])
    F = np.asarray(pot.forces(st))         # [2,2]
    f_ref.append(float(F[1, 0]))           # x-force on cell 1 (pushed +x if repulsive)
    overlap = max(0.0, 1.0 - r / sigma)
    f_an.append((eps / sigma) * overlap ** 1.5 if r < sigma else 0.0)
f_ref = np.array(f_ref); f_an = np.array(f_an)
out["dtype_forces"] = str(F.dtype)
out["scan_seps"] = seps.tolist()
out["scan_f_ref"] = f_ref.tolist()
out["scan_f_analytic"] = f_an.tolist()
out["scan_ref_vs_analytic_max"] = float(np.abs(f_ref - f_an).max())

# --- per-cell epsilon arithmetic-mean mix: eps=(1,3) at contact-frac should act like eps=2 --- #
from jax_morph.core.state import StateFieldSpec
# (only probe the shared-scalar path here; per-cell needs a field spec -- checked in the real run)

# --- SoftSphere vs Hertzian on a fixed overlapping config (the negative control) ------------ #
pos7 = [[0, 0], [0.8, 0], [0.4, 0.7], [-0.5, 0.4], [0.3, -0.6], [1.1, 0.5], [-0.2, -0.9]]
rad7 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
st7 = state_of(pos7, rad7)
Fh = np.asarray(Hertzian(epsilon=2.0).forces(st7))
Fs = np.asarray(SoftSphere(epsilon=2.0).forces(st7))
out["hertzian_force_maxabs"] = float(np.abs(Fh).max())
out["softsphere_vs_hertzian_rel"] = float(np.abs(Fs - Fh).max() / max(1e-12, np.abs(Fh).max()))

# --- BrownianDynamics kT=0 deterministic forward Euler? ------------------------------------- #
s0 = state_of(pos7, rad7)
h1 = jxm.simulate(model, s0, n_steps=5, dt=0.1, key=jax.random.PRNGKey(0), history=True)
h2 = jxm.simulate(model, s0, n_steps=5, dt=0.1, key=jax.random.PRNGKey(999), history=True)
p1 = np.asarray(h1.position); p2 = np.asarray(h2.position)
out["bd_kT0_deterministic_across_keys"] = float(np.abs(p1 - p2).max())
# first Euler step should equal dt*forces(s0)
step_pred = 0.1 * np.asarray(BrownianDynamics(Hertzian(epsilon=2.0), n_space_dim=2, kT=0.0).potential.forces(s0))
step_act = p1[1] - p1[0]
out["bd_first_step_vs_dt_forces_max"] = float(np.abs(step_act - step_pred).max())

with open(os.path.join(OUT, "probe.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
