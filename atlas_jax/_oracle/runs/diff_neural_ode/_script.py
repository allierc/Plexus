"""Oracle for regulate:neural_ode -- the reference NeuralODE controller, on a matched IC.

NeuralODE is a DYNAMIC per-cell ODE whose ENTIRE behaviour is its MLP vector field. There is no
morphogenesis trajectory to diff (the paper's only ODE controller is the gene network; NeuralODE
appears in no reference composition), and -- decisively -- a free-form MLP cannot cross the
JAX/torch boundary through a YAML spec. So the ONLY way to give the reference and the Plexus
operator the SAME operator is to build the MLP here, integrate the reference controller over one
macro-step on a fixed per-cell initial condition, and EXPORT the exact weights + the reference
increment. The Plexus side loads those weights verbatim into the torch operator and must
reproduce the endpoint.

This script writes, for two circuit shapes and three macro-step sizes:
  * the exact per-layer MLP weights/biases (so torch can rebuild the identical field),
  * the matched initial condition g0 (= concat(hidden0, outputs0)), u0,
  * the reference one-step endpoint y_ref(dt) = y0 + (increment ODEController.__call__ returns),
  * a HIGH-ACCURACY ground-truth endpoint y_true(dt) (diffrax at rtol=1e-10) of the same field,
  * a probe (input matrix + MLP output) so the Plexus side can verify it rebuilt the same net.
Determinism is checked (a fixed key must give a fixed endpoint) before anything is recorded.
"""
import os
import json

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # compare INTEGRATION SEMANTICS in float64, so float32
import jax.numpy as jnp                       # roundoff (~1e-3 at dt=2) cannot masquerade as a defect;
import diffrax                                # the native-float32 path is exercised by run_spec.py

from jax_morph.core.state import StateFieldSpec, build_state_from_model
from jax_morph.core.step import Model
from jax_morph.control.ode import NeuralODE

OUT = os.environ["OUT"]

N = 16                      # cells in the batch
N_IN = 2                   # driver width (u)
WIDTH, DEPTH = 8, 2        # MLP shape (make_mlp defaults are 64/2; 8 keeps the .npz small)
DTS = [0.5, 1.0, 2.0]      # exercise the /dt mean-rate convention (dt != 1 makes it visible)

# two circuit shapes: (tag, hidden_size, out_size) -> n_gene = hidden + out
CONFIGS = [("A", 0, 3), ("B", 2, 2)]


def mlp_layers(mlp):
    """Per-layer (weight, bias) as float64 numpy; weight is (out, in), torch's convention too."""
    out = []
    for lin in mlp.layers:
        w = np.asarray(lin.weight, dtype=np.float64)
        b = np.zeros(w.shape[0], np.float64) if lin.bias is None else np.asarray(lin.bias, np.float64)
        out.append((w, b))
    return out


def endpoint_true(step, y0, u, dt):
    """A high-accuracy integration of the SAME vector field -> the true ODE endpoint y(dt).
    Same ODETerm as ODEController.__call__, but rtol/atol driven to ~machine so this is the
    reference-free ground truth both sides can be scored against."""
    term = diffrax.ODETerm(lambda t, y, args: step.vector_field(t, y, u))
    sol = diffrax.diffeqsolve(
        term, diffrax.Dopri5(), t0=0.0, t1=dt, dt0=dt, y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-12),
        saveat=diffrax.SaveAt(t1=True), max_steps=100000,
    )
    return np.asarray(sol.ys[-1], dtype=np.float64)


arrays = {}
summary = {"N": N, "n_in": N_IN, "width": WIDTH, "depth": DEPTH, "dts": DTS,
           "configs": {}, "determinism_ok": True, "reference_vs_truth_max": 0.0}

rng = np.random.default_rng(7)
ref_truth_max = 0.0

for tag, hidden, n_out in CONFIGS:
    n_gene = hidden + n_out
    u_spec = StateFieldSpec("u", shape=(N_IN,))
    g_spec = StateFieldSpec("gout", shape=(n_out,))
    key = jax.random.PRNGKey({"A": 11, "B": 22}[tag])
    mlp = NeuralODE.make_mlp((u_spec,), (g_spec,), hidden, key=key, width=WIDTH, depth=DEPTH)
    step = NeuralODE((u_spec,), (g_spec,), hidden, mlp=mlp, tag=tag)

    # --- export the exact field so torch can rebuild it -------------------------------------- #
    layers = mlp_layers(mlp)
    for i, (w, b) in enumerate(layers):
        arrays[f"{tag}__W{i}"] = w
        arrays[f"{tag}__b{i}"] = b
    # net-equality probe: 5 random inputs of width in_size and the MLP's exact outputs
    P = rng.standard_normal((5, N_IN + n_gene)).astype(np.float64)
    arrays[f"{tag}__probe_in"] = P
    arrays[f"{tag}__probe_out"] = np.asarray(jax.vmap(mlp)(jnp.asarray(P)), np.float64)

    # --- the matched initial condition ------------------------------------------------------- #
    u0 = rng.standard_normal((N, N_IN)).astype(np.float64)
    out0 = rng.standard_normal((N, n_out)).astype(np.float64)
    hid0 = rng.standard_normal((N, hidden)).astype(np.float64) if hidden else np.zeros((N, 0), np.float64)
    y0 = np.concatenate([hid0, out0], axis=1)            # == ODEController y0 = concat(hidden, outputs)
    arrays[f"{tag}__g0"] = y0.astype(np.float64)         # Plexus gene block == this
    arrays[f"{tag}__u0"] = u0

    State = build_state_from_model(Model([step]))
    base = State.init_empty(capacity=N, n_space_dim=2, n_types=1)
    upd = dict(
        alive=base.alive.at[:N].set(True),
        position=base.position.at[:N].set(jnp.asarray(rng.standard_normal((N, 2)), jnp.float64)),
        celltype=base.celltype.at[:N, 0].set(1.0),
        u=base.u.at[:N].set(jnp.asarray(u0)),
        gout=base.gout.at[:N].set(jnp.asarray(out0)),
    )
    if hidden:
        upd[f"{tag}_hidden"] = getattr(base, f"{tag}_hidden").at[:N].set(jnp.asarray(hid0))
    state = base.update(**upd)

    cfg = {"hidden": hidden, "n_out": n_out, "n_gene": n_gene}
    for dt in DTS:
        d1 = step(state, dt=dt, key=jax.random.PRNGKey(0))
        d2 = step(state, dt=dt, key=jax.random.PRNGKey(999))   # key is unused -> must be identical
        inc_out = np.asarray(d1.gout, np.float64)
        inc_out2 = np.asarray(d2.gout, np.float64)
        if not np.array_equal(inc_out, inc_out2):
            summary["determinism_ok"] = False
        if hidden:
            inc_hid = np.asarray(getattr(d1, f"{tag}_hidden"), np.float64)
            increment = np.concatenate([inc_hid, inc_out], axis=1)
        else:
            increment = inc_out
        y_ref = (y0 + increment).astype(np.float64)          # reference macro-step endpoint
        y_true = endpoint_true(step, jnp.asarray(y0), jnp.asarray(u0), dt)
        arrays[f"{tag}__y_ref_dt{dt}"] = y_ref
        arrays[f"{tag}__y_true_dt{dt}"] = y_true
        arrays[f"{tag}__inc_dt{dt}"] = increment.astype(np.float64)
        rt = float(np.abs(y_ref - y_true).max())
        ref_truth_max = max(ref_truth_max, rt)
        cfg[f"ref_vs_true_dt{dt}"] = rt
    summary["configs"][tag] = cfg

summary["reference_vs_truth_max"] = ref_truth_max
if not summary["determinism_ok"]:
    raise SystemExit("reference NeuralODE not deterministic at fixed inputs -- a diff would "
                     "measure the reference's own noise. Stop.")

np.savez_compressed(os.path.join(OUT, "reference.npz"), **arrays)
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
print("wrote reference.npz (%d arrays), summary.json" % len(arrays))
