import os, json
import numpy as np
import jax, jax.numpy as jnp
from jax_morph.core.state import StateFieldSpec, build_state_from_model
from jax_morph.core.step import Model
from jax_morph.control.ode import NeuralODE

# --- build a NeuralODE: n_in=2, out=2 (vector), hidden=2 -> n_gene=4, MLP in=6 out=4 ---
u_spec = StateFieldSpec('u', shape=(2,))
g_spec = StateFieldSpec('gout', shape=(2,))
hidden = 2
key = jax.random.PRNGKey(0)
mlp = NeuralODE.make_mlp((u_spec,), (g_spec,), hidden, key=key, width=8, depth=2)
step = NeuralODE((u_spec,), (g_spec,), hidden, mlp=mlp, tag='ode')
print("mlp in_size", mlp.in_size, "out_size", mlp.out_size, "width", mlp.width_size, "depth", mlp.depth)
print("layers:", [(l.weight.shape, None if l.bias is None else l.bias.shape) for l in mlp.layers])
print("activation:", mlp.activation, "final:", mlp.final_activation, "use_bias:", mlp.use_bias)

# --- build a state and drive the step directly ---
model = Model([step])
State = build_state_from_model(model)
N = 4
s = State.init_empty(capacity=N, n_space_dim=2, n_types=1)
rng = np.random.default_rng(1)
u0 = jnp.asarray(rng.normal(size=(N,2)))
g0 = jnp.asarray(rng.normal(size=(N,2)))
h0 = jnp.asarray(rng.normal(size=(N,2)))
s = s.update(
    alive=s.alive.at[:N].set(True),
    position=s.position.at[:N].set(jnp.asarray(rng.normal(size=(N,2)))),
    celltype=s.celltype.at[:N,0].set(1.0),
    u=s.u.at[:N].set(u0),
    gout=s.gout.at[:N].set(g0),
    ode_hidden=s.ode_hidden.at[:N].set(h0),
)
print("state fields:", sorted(s.specs))
dt = 1.3
delta = step(s, dt=dt, key=key)
print("delta gout:", np.asarray(delta.gout))
print("delta ode_hidden:", np.asarray(delta.ode_hidden))
print("default jax dtype:", jnp.zeros(1).dtype)
# sanity: vector_field at t=0
vf = step.vector_field(0.0, jnp.concatenate([h0, g0], axis=1), u0)
print("vf shape", vf.shape, "vf[0]", np.asarray(vf[0]))
