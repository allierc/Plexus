import time
import numpy as np
import jax, jax.numpy as jnp
import jax_morph as jxm
from jax_morph.physics import Division

CAP, N_STEPS, DT = 512, 40, 1.0
model = jxm.Model([Division(n_space_dim=2)])   # DIVISION ONLY
def seed_state():
    p0 = jnp.array([[0.0,0.0],[1.0,0.1],[0.5,0.9],[-0.4,0.7]])
    s = jxm.build_state_from_model(model).init_empty(capacity=CAP, n_space_dim=2, n_types=1)
    n = p0.shape[0]
    return s.update(alive=s.alive.at[:n].set(True), radius=s.radius.at[:n].set(0.5),
                    position=s.position.at[:n].set(p0), celltype=s.celltype.at[:n,0].set(1.0),
                    division_rate=s.division_rate.at[:n].set(0.08))
def run(key): return jxm.simulate(model, seed_state(), n_steps=N_STEPS, dt=DT, key=key, history=True)
for i in range(3):
    t=time.time(); h=run(jax.random.PRNGKey(i)); c=np.asarray(h.alive.sum(axis=1))
    print(f"div-only seed {i}: final={int(c[-1])} overflow={float(np.asarray(h.division_overflow)[-1])} wall={time.time()-t:.2f}s", flush=True)
