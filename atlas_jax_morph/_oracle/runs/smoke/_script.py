
import json, os
import numpy as np
import jax, jax.numpy as jnp
import jax_morph as jxm
from jax_morph.physics import MechanicalRelaxation, Morse, SaturatingCellGrowth, Division

OUT = os.environ["OUT"]
SEED, CAP, N_STEPS, DT = 0, 140, 40, 1.0

model = jxm.Model([
    MechanicalRelaxation(Morse(epsilon=3.0, alpha=2.8), max_steps=800, f_tol=1e-3),
    SaturatingCellGrowth(max_radius=0.6),
    Division(n_space_dim=2),
])

def seed_state():
    p0 = jnp.array([[0.0, 0.0], [1.0, 0.1], [0.5, 0.9], [-0.4, 0.7]])
    s = jxm.build_state_from_model(model).init_empty(capacity=CAP, n_space_dim=2, n_types=1)
    n = p0.shape[0]
    return s.update(
        alive=s.alive.at[:n].set(True),
        radius=s.radius.at[:n].set(0.5),
        position=s.position.at[:n].set(p0),
        celltype=s.celltype.at[:n, 0].set(1.0),
        growth_rate=s.growth_rate.at[:n].set(0.4),
        division_rate=s.division_rate.at[:n].set(0.08),
    )

def run(key):
    return jxm.simulate(model, seed_state(), n_steps=N_STEPS, dt=DT, key=key, history=True)

h1 = run(jax.random.PRNGKey(SEED))
h2 = run(jax.random.PRNGKey(SEED))

# --- the oracle must be a function, not a process -------------------------------------------- #
same = bool(np.array_equal(np.asarray(h1.position), np.asarray(h2.position)) and
            np.array_equal(np.asarray(h1.alive), np.asarray(h2.alive)))
if not same:
    raise SystemExit("ORACLE IS NOT DETERMINISTIC at a fixed key -- a differential test against "
                     "it would measure the reference's own noise. Stop here.")

counts = np.asarray(h1.alive.sum(axis=1))
pos, rad, alive = np.asarray(h1.position), np.asarray(h1.radius), np.asarray(h1.alive)
np.savez_compressed(os.path.join(OUT, "reference.npz"),
                    t=np.asarray(h1.t), position=pos, radius=rad, alive=alive, counts=counts)

# --- a few numbers the Plexus side will have to match ---------------------------------------- #
def gyration(p, a):
    q = p[a.astype(bool)]
    return float(np.sqrt(((q - q.mean(0)) ** 2).sum(1).mean()))

summary = {
    "seed": SEED, "n_steps": N_STEPS, "dt": DT, "capacity": CAP,
    "deterministic_at_fixed_key": same,
    "cells_first": int(counts[0]), "cells_last": int(counts[-1]),
    "counts": counts.tolist(),
    "radius_mean_last": float(rad[-1][alive[-1].astype(bool)].mean()),
    "radius_max_last": float(rad[-1][alive[-1].astype(bool)].max()),
    "gyration_first": gyration(pos[0], alive[0]),
    "gyration_last": gyration(pos[-1], alive[-1]),
    "division_overflow": float(np.asarray(h1.division_overflow)[-1]),
}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

if summary["division_overflow"] != 0.0:
    raise SystemExit("division overflowed the capacity -- the run hit an ARRAY BOUND, not a "
                     "biological limit. (This is exactly the 1778-cell error, in their units.)")

# --- one figure, on a common scale ------------------------------------------------------------ #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

live = alive[-1].astype(bool)
m = float(rad[-1].max()) + 0.6
lims = ((pos[-1][live][:, 0].min() - m, pos[-1][live][:, 0].max() + m),
        (pos[-1][live][:, 1].min() - m, pos[-1][live][:, 1].max() + m))
picks = [0, 8, 16, 24, 32, 40]
fig, axes = plt.subplots(1, len(picks), figsize=(2.6 * len(picks), 3.0))
for ax, t in zip(axes, picks):
    st = jax.tree_util.tree_map(lambda x: x[t], h1)
    jxm.viz.draw(st, ax=ax, color="tab:blue", lims=lims)
    ax.set_title(f"t = {float(h1.t[t]):g}   {int(st.alive.sum())} cells", fontsize=9)
fig.suptitle("oracle: MechanicalRelaxation(Morse) + SaturatingCellGrowth + Division", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "reference.png"), dpi=130, bbox_inches="tight")
print("wrote reference.npz, summary.json, reference.png")
