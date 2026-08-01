"""Oracle for `morphogen:free_space_greens_function` -- the reference FreeScreenedDiffusion field.

`morphogen` is a QUASISTATIC pure state->field map: `__call__` ignores dt/key and OVERWRITES the
per-cell `chemical` block with the t=infinity screened-diffusion solution. There is no integrator
to conflate and no trajectory to score (the field is recomputed from scratch each step and moves
no cell). So the differential test is the analog of the gene-network `metric_A`: evaluate the
reference field DIRECTLY on a set of FIXED matched states, chosen to exercise every regime of the
dimension-selected finite-source Green's-function kernel, and save the input state + output field
so the Plexus-side differ reads ONE artefact and the two runs cannot drift apart.

Configs (each a distinct kernel regime the CODE has and the paper's graph-Laplacian solve does not):
  C_anchor (2-D disk): the anchor's 4 founders, uniform radius 0.5, uniform unit secretion, D=K=1
    -- the exact geometry the Plexus ENGINE run seeds.
  C2 (2-D disk, adversarial): 12 slots / 9 live, per-cell varied radius, an OVERLAPPING pair
    (r<a -> r_eff=max(r,a) self/near clamp), 2 species with per-species (D,K), secretion including
    zeros and a BIG nonzero on a DEAD slot (a source-mask bug would be loud).
  C3 (3-D sphere, adversarial): 12 slots / 9 live, species 0 UNSCREENED (K=0, kappa=0, the
    deliberately-unfloored exact branch) + species 1 screened, an overlapping pair, dead slots.
  C1 (1-D segment, adversarial): 8 slots / 6 live, D=1, K=0.7.

float32 throughout -- MATCH the Plexus torch engine; do NOT enable x64.
"""
import json
import os

import numpy as np
import jax
import jax.numpy as jnp

from jax_morph.core.step import Model
from jax_morph.core.state import build_state_from_model
from jax_morph.physics import FreeScreenedDiffusion

OUT = os.environ["OUT"]

# JAX default is float32 -- MATCH the Plexus engine's float32 state; do NOT enable x64.
assert jnp.zeros(1).dtype == jnp.float32, "oracle must run in float32 to match torch"


def f32(x):
    return np.asarray(x, dtype=np.float32)


def run_ref(pos, radius, alive, secretion, n_space_dim, diffusion, degradation):
    """Reference steady field c_ref [capacity, n_species] for one fixed matched state (float32)."""
    pos = f32(pos)
    radius = f32(radius).reshape(-1)
    alive = np.asarray(alive, dtype=bool).reshape(-1)
    secretion = f32(secretion)
    if secretion.ndim == 1:
        secretion = secretion[:, None]
    cap, n_species = pos.shape[0], secretion.shape[1]
    assert pos.shape[1] == n_space_dim

    diff = diffusion if np.ndim(diffusion) == 0 else jnp.asarray(f32(diffusion))
    degr = degradation if np.ndim(degradation) == 0 else jnp.asarray(f32(degradation))
    step = FreeScreenedDiffusion(n_field_species=n_species, n_space_dim=n_space_dim,
                                 diffusion=diff, degradation=degr)
    model = Model([step])
    State = build_state_from_model(model)
    s = State.init_empty(capacity=cap, n_space_dim=n_space_dim, n_types=1)
    s = s.update(
        alive=jnp.asarray(alive),
        radius=jnp.asarray(radius),
        position=jnp.asarray(pos),
        celltype=s.celltype.at[:, 0].set(jnp.where(jnp.asarray(alive), 1.0, 0.0)),
        secretion_rate=jnp.asarray(secretion),
    )
    out1 = step(s, dt=1.0, key=jax.random.PRNGKey(0))
    out2 = step(s, dt=2.0, key=jax.random.PRNGKey(7))   # dt/key IGNORED -> must be identical
    c1 = np.asarray(out1[step.field_name], np.float32)
    c2 = np.asarray(out2[step.field_name], np.float32)
    if not np.array_equal(c1, c2):
        raise SystemExit("FreeScreenedDiffusion is NOT invariant to dt/key -- it is not "
                         "quasistatic as claimed; a differential test would be meaningless. Stop.")
    return c1


# --------------------------------------------------------------------------------------------- #
#  the fixed configs (chosen BEFORE any run; a fixed rng seeds the non-hand-placed cells)
# --------------------------------------------------------------------------------------------- #
rng = np.random.default_rng(20260731)
configs = {}

# --- C_anchor: the anchor's 4 founders in their local frame (world - [20,20]); the ENGINE geometry
anchor_pos = f32([[0.0, 0.0], [1.0, 0.1], [0.5, 0.9], [-0.4, 0.7]])
cap = 8                                                # 4 live + 4 dead (capacity > n, like the spec)
pos = np.zeros((cap, 2), np.float32); pos[:4] = anchor_pos
radius = np.zeros(cap, np.float32); radius[:4] = 0.5
alive = np.zeros(cap, bool); alive[:4] = True
S = np.zeros((cap, 1), np.float32); S[:4, 0] = 1.0     # uniform unit secretion
configs["C_anchor"] = dict(pos=pos, radius=radius, alive=alive, secretion=S,
                           n_space_dim=2, diffusion=1.0, degradation=1.0)

# --- C2: 2-D disk, adversarial. 12 slots, 9 live (0..8), 3 dead (9..11). 2 species.
cap = 12
pos = np.zeros((cap, 2), np.float32)
pos[0] = [0.0, 0.0]
pos[1] = [0.03, 0.0]        # OVERLAPPING with cell 0: r=0.03 < a=0.05 -> r_eff clamp
pos[2] = [2.0, 0.0]         # far -> small screened field (relative error near FLOOR)
pos[3] = [0.4, 0.7]
pos[4] = [-0.6, 0.3]
pos[5] = [1.1, -0.5]
pos[6] = [-0.2, -0.9]
pos[7] = [0.8, 1.2]
pos[8] = [-1.3, -0.4]
pos[9:] = rng.normal(0, 1.0, (3, 2))                   # dead slots: positions must not matter
radius = np.zeros(cap, np.float32)
radius[:9] = f32([0.05, 0.05, 0.04, 0.06, 0.03, 0.07, 0.05, 0.02, 0.08])
alive = np.zeros(cap, bool); alive[:9] = True
S = np.zeros((cap, 2), np.float32)
S[:9, 0] = f32([1.0, 0.0, 2.0, 0.5, 0.0, 1.5, 0.3, 0.0, 2.5])   # species 0, incl. zeros
S[:9, 1] = f32([0.2, 1.0, 0.0, 1.3, 0.7, 0.0, 2.0, 0.4, 1.1])   # species 1
S[9, 0] = 99.0; S[9, 1] = 77.0                         # BIG source on a DEAD slot: must emit nothing
configs["C2"] = dict(pos=pos, radius=radius, alive=alive, secretion=S,
                     n_space_dim=2, diffusion=[1.0, 2.0], degradation=[0.5, 1.5])

# --- C3: 3-D sphere, adversarial. 12 slots, 9 live, 3 dead. species 0 UNSCREENED (K=0).
cap = 12
pos = np.zeros((cap, 3), np.float32)
pos[0] = [0.0, 0.0, 0.0]
pos[1] = [0.03, 0.0, 0.0]  # OVERLAPPING with cell 0
pos[2] = [2.0, 0.0, 0.0]   # far
pos[3] = [0.4, 0.7, -0.2]
pos[4] = [-0.6, 0.3, 0.5]
pos[5] = [1.1, -0.5, 0.3]
pos[6] = [-0.2, -0.9, -0.6]
pos[7] = [0.8, 1.2, 0.1]
pos[8] = [-1.3, -0.4, 0.9]
pos[9:] = rng.normal(0, 1.0, (3, 3))
radius = np.zeros(cap, np.float32)
radius[:9] = f32([0.05, 0.05, 0.04, 0.06, 0.03, 0.07, 0.05, 0.02, 0.08])
alive = np.zeros(cap, bool); alive[:9] = True
S = np.zeros((cap, 2), np.float32)
S[:9, 0] = f32([1.0, 0.0, 2.0, 0.5, 0.0, 1.5, 0.3, 0.0, 2.5])
S[:9, 1] = f32([0.2, 1.0, 0.0, 1.3, 0.7, 0.0, 2.0, 0.4, 1.1])
S[10, 0] = 99.0; S[10, 1] = 77.0                       # BIG source on a DEAD slot
configs["C3"] = dict(pos=pos, radius=radius, alive=alive, secretion=S,
                     n_space_dim=3, diffusion=[1.0, 1.5], degradation=[0.0, 1.0])

# --- C1: 1-D segment, adversarial. 8 slots, 6 live, 2 dead. single species.
cap = 8
pos = np.zeros((cap, 1), np.float32)
pos[:6, 0] = f32([0.0, 0.03, 0.5, 1.4, -0.7, 2.2])     # overlapping 0&1 (r=0.03 < a=0.05)
pos[6:, 0] = rng.normal(0, 1.0, 2)
radius = np.zeros(cap, np.float32)
radius[:6] = f32([0.05, 0.05, 0.04, 0.06, 0.03, 0.07])
alive = np.zeros(cap, bool); alive[:6] = True
S = np.zeros((cap, 1), np.float32)
S[:6, 0] = f32([1.0, 0.0, 2.0, 0.5, 1.5, 0.0])
S[6, 0] = 99.0                                         # BIG source on a DEAD slot
configs["C1"] = dict(pos=pos, radius=radius, alive=alive, secretion=S,
                     n_space_dim=1, diffusion=1.0, degradation=0.7)

# --------------------------------------------------------------------------------------------- #
#  run the reference, save one artefact
# --------------------------------------------------------------------------------------------- #
save = {}
summary = {"configs": list(configs), "float32": True}
for name, cfg in configs.items():
    c_ref = run_ref(**cfg)
    save[f"{name}_pos"] = f32(cfg["pos"])
    save[f"{name}_radius"] = f32(cfg["radius"])
    save[f"{name}_alive"] = np.asarray(cfg["alive"], bool)
    save[f"{name}_secretion"] = f32(cfg["secretion"])
    save[f"{name}_diffusion"] = f32(cfg["diffusion"])
    save[f"{name}_degradation"] = f32(cfg["degradation"])
    save[f"{name}_n_space_dim"] = np.int64(cfg["n_space_dim"])
    save[f"{name}_c_ref"] = c_ref
    live = np.asarray(cfg["alive"], bool)
    dead = ~live
    dead_max = float(np.abs(c_ref[dead]).max()) if dead.any() else 0.0
    summary[name] = {
        "n_space_dim": int(cfg["n_space_dim"]),
        "capacity": int(c_ref.shape[0]),
        "n_live": int(live.sum()),
        "n_species": int(c_ref.shape[1]),
        "c_ref_live_max": float(np.abs(c_ref[live]).max()),
        "c_ref_live_min_nonzero": float(np.abs(c_ref[live][c_ref[live] != 0]).min())
                                  if (c_ref[live] != 0).any() else 0.0,
        "dead_slots_field_max_abs": dead_max,   # must be exactly 0.0 (twice-applied alive mask)
        "finite": bool(np.isfinite(c_ref).all()),
    }

np.savez_compressed(os.path.join(OUT, "reference.npz"), **save)
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

bad = [n for n in configs if not summary[n]["finite"]]
if bad:
    raise SystemExit(f"reference produced non-finite field in {bad}; stop.")
dead_leak = [n for n in configs if summary[n]["dead_slots_field_max_abs"] != 0.0]
if dead_leak:
    raise SystemExit(f"reference left nonzero field on DEAD slots in {dead_leak}; the alive mask "
                     f"did not hold on the reference side -- stop.")
print("wrote reference.npz, summary.json")
