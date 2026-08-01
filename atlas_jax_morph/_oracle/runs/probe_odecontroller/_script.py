import os, json
import numpy as np
import jax, jax.numpy as jnp
import jax_morph as jxm
from jax_morph.control import GeneNetworkConnectionist
from jax_morph.core.state import StateFieldSpec

print("x64:", jax.config.jax_enable_x64, "default dtype:", jnp.zeros(1).dtype)

inp = (StateFieldSpec('drive', shape=(2,)),)
out = (StateFieldSpec('g_out', shape=(2,)),)
W_gene = jnp.array([[0.0,0.8,-0.5],[-0.6,0.0,0.7],[0.4,-0.3,0.0]])
W_in   = jnp.array([[1.2,0.0],[0.0,-0.9],[0.5,0.5]])
b      = jnp.array([0.1,-0.2,0.0])
ctrl = GeneNetworkConnectionist(inp, out, hidden_size=1, W_gene=W_gene, W_in=W_in, b=b, gamma=1.0, tag='gene')
print("hidden field:", ctrl._hidden, "in_size", ctrl.in_size, "out_size", ctrl.out_size)
print("state_writes:", [(s.name, s.shape) for s in ctrl.state_writes()])
print("state_reads:", [(s.name, s.shape) for s in ctrl.state_reads()])

model = jxm.Model([ctrl])
StateCls = jxm.build_state_from_model(model)
print("schema fields:", sorted(StateCls('dummy') if False else model.state_requires(), key=lambda s:s.name) and [s.name for s in model.state_requires()])
CAP=8
s = StateCls.init_empty(capacity=CAP, n_space_dim=2, n_types=1)
n=4
u = jnp.array([[0.8,0.3],[-0.5,1.0],[0.2,-0.7],[1.5,0.5]])
alive = s.alive.at[:n].set(True)
s = s.update(alive=alive,
             drive=s.drive.at[:n].set(u),
             g_out=s.g_out.at[:n].set(0.0),
             gene_hidden=s.gene_hidden.at[:n].set(0.0))
# one macro-step delta directly
delta = ctrl(s, dt=1.0, key=jax.random.PRNGKey(0))
print("delta gene_hidden[:n]:", np.asarray(delta.gene_hidden)[:n].ravel())
print("delta g_out[:n]:\n", np.asarray(delta.g_out)[:n])
# full simulate history
h = jxm.simulate(model, s, n_steps=20, dt=1.0, key=jax.random.PRNGKey(0), history=True)
y = np.concatenate([np.asarray(h.gene_hidden), np.asarray(h.g_out)], axis=-1)  # [T+1, CAP, 3]
print("history shape:", y.shape, "y[-1,:n]:\n", y[-1,:n])
