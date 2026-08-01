"""Probe the per-cell epsilon StateFieldSpec path for Hertzian."""
import os, json
import numpy as np
import jax, jax.numpy as jnp
import jax_morph as jxm
from jax_morph.physics import Hertzian, BrownianDynamics
from jax_morph.core.state import StateFieldSpec

OUT = os.environ["OUT"]
out = {}

eps_spec = StateFieldSpec('epsilon', heritable=True)
model = jxm.Model([BrownianDynamics(Hertzian(epsilon=eps_spec), n_space_dim=2, kT=0.0)])
pot = Hertzian(epsilon=eps_spec)

# two cells overlapping at r=0.6, sigma=1.0; per-cell eps=(1,3) mixes by mean -> 2
pos = jnp.asarray([[0.0, 0.0], [0.6, 0.0]], jnp.float32)
radius = jnp.asarray([0.5, 0.5], jnp.float32)
s = jxm.build_state_from_model(model).init_empty(capacity=2, n_space_dim=2, n_types=1)
s = s.update(alive=s.alive.at[:2].set(True), radius=s.radius.at[:2].set(radius),
             position=s.position.at[:2].set(pos), celltype=s.celltype.at[:2, 0].set(1.0),
             epsilon=s.epsilon.at[:2].set(jnp.asarray([1.0, 3.0], jnp.float32)))
F_percell = np.asarray(pot.forces(s))

# shared eps=2 reference
model2 = jxm.Model([BrownianDynamics(Hertzian(epsilon=2.0), n_space_dim=2, kT=0.0)])
s2 = jxm.build_state_from_model(model2).init_empty(capacity=2, n_space_dim=2, n_types=1)
s2 = s2.update(alive=s2.alive.at[:2].set(True), radius=s2.radius.at[:2].set(radius),
               position=s2.position.at[:2].set(pos), celltype=s2.celltype.at[:2, 0].set(1.0))
F_shared2 = np.asarray(Hertzian(epsilon=2.0).forces(s2))

out["percell_eps_supported"] = True
out["F_percell_1_3"] = F_percell.tolist()
out["F_shared_2"] = F_shared2.tolist()
out["percell_mean_eq_shared_max"] = float(np.abs(F_percell - F_shared2).max())
with open(os.path.join(OUT, "probe_eps.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
