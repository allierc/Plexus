"""Oracle for grow_radius / SaturatingCellGrowth -- ISOLATED per-cell radius growth trajectories.

Growth is isolated from relaxation and division so the reference and the Plexus operator can be put
on the SAME per-cell initial condition and diffed cell-for-cell, frame-for-frame. With division
present the two sides reach different cell COUNTS (124 vs 82 on the anchor) and the radius arrays
cannot be aligned; Division's hazard p = 1 - exp(-division_rate*dt) never reads radius (division.py
_dist), so growth is not what makes the counts diverge, and isolating it is the faithful test of
THIS operator's contract -- the per-cell radius ODE.

Two scenarios, both max_radius R = 0.6, dt = 2.0 (dt != 1 makes the mean-rate convention
observable), T = 20 macro-steps:
  A UNIFORM -- 4 cells, r0 = 0.30, k = 0.40      (matches config/atlas_jax/saturating_cell_growth.yaml)
  B GRID    -- 36 cells, r0 x k over a 6x6 grid   (spans k=0 no-op, saturation, r0==R, r0>R)

The reference is jax-morph SaturatingCellGrowth run ALONE via jxm.simulate(history=True); its
s_0..s_T radius history is exported. An analytic float64 recurrence r_{t+1} = R + (r_t - R)*
exp(-k*dt/R) is computed to confirm the reference truly implements the exact von-Bertalanffy flow
(not forward Euler, not a min-clamp). Growth carries no RNG, but determinism at a fixed key is
checked before anything is recorded, per the oracle contract.
"""
import os
import json

import numpy as np
import jax
import jax.numpy as jnp
import jax_morph as jxm
from jax_morph.physics import SaturatingCellGrowth

OUT = os.environ["OUT"]
R, DT, T = 0.6, 2.0, 20

model = jxm.Model([SaturatingCellGrowth(max_radius=R)])


def build(r0, k):
    """A capacity-N state with the given per-cell radius r0 and growth_rate k; positions are
    arbitrary (no force acts, so they never move)."""
    r0 = np.asarray(r0, np.float32)
    k = np.asarray(k, np.float32)
    N = r0.shape[0]
    s = jxm.build_state_from_model(model).init_empty(capacity=N, n_space_dim=2, n_types=1)
    pos = jnp.asarray(np.stack([np.arange(N), np.zeros(N)], 1).astype(np.float32))
    return s.update(
        alive=s.alive.at[:N].set(True),
        radius=s.radius.at[:N].set(jnp.asarray(r0)),
        position=s.position.at[:N].set(pos),
        celltype=s.celltype.at[:N, 0].set(1.0),
        growth_rate=s.growth_rate.at[:N].set(jnp.asarray(k)),
    )


def sim(state, key=0):
    return jxm.simulate(model, state, n_steps=T, dt=DT, key=jax.random.PRNGKey(key), history=True)


def analytic(r0, k):
    """Exact-flow recurrence r_{t+1} = R + (r_t - R)*exp(-k*dt/R) in float64 -> [T+1, N]."""
    r0 = np.asarray(r0, np.float64)
    k = np.asarray(k, np.float64)
    decay = np.exp(-k * DT / R)
    out = [r0.copy()]
    r = r0.copy()
    for _ in range(T):
        r = R + (r - R) * decay
        out.append(r.copy())
    return np.stack(out)


arrays, summary = {}, {"R": R, "dt": DT, "T": T}

# ---------------------------------------------------------------------------------------------- #
#  Scenario A -- uniform IC (mirrors the run_spec.py spec)
# ---------------------------------------------------------------------------------------------- #
rA0 = np.full(4, 0.30, np.float32)
kA = np.full(4, 0.40, np.float32)
hA = sim(build(rA0, kA))
radA = np.asarray(hA.radius)                       # [T+1, 4]
aliveA = np.asarray(hA.alive)
detA = bool(np.array_equal(radA, np.asarray(sim(build(rA0, kA), key=999).radius)))
refA_vs_analytic = float(np.abs(radA.astype(np.float64) - analytic(rA0, kA)).max())

# ---------------------------------------------------------------------------------------------- #
#  Scenario B -- heterogeneous 6x6 grid (r0 x k), the discriminating case
# ---------------------------------------------------------------------------------------------- #
r0_vals = np.array([0.05, 0.25, 0.45, 0.59, 0.60, 0.75], np.float32)
k_vals = np.array([0.0, 0.05, 0.15, 0.40, 1.00, 2.50], np.float32)
RR, KK = np.meshgrid(r0_vals, k_vals, indexing="ij")
rB0, kB = RR.flatten(), KK.flatten()               # 36 cells
hB = sim(build(rB0, kB))
radB = np.asarray(hB.radius)                       # [T+1, 36]
aliveB = np.asarray(hB.alive)
detB = bool(np.array_equal(radB, np.asarray(sim(build(rB0, kB), key=999).radius)))
refB_vs_analytic = float(np.abs(radB.astype(np.float64) - analytic(rB0, kB)).max())

if not (detA and detB):
    raise SystemExit("SaturatingCellGrowth is not deterministic at a fixed key -- a differential "
                     "test against it would measure the reference's own noise. Stop here.")

summary.update({
    "deterministic_at_fixed_key": bool(detA and detB),
    "state_dtype": str(radA.dtype),
    "A": {"n": 4, "r0": 0.30, "k": 0.40,
          "radius_first": float(radA[0, 0]), "radius_last": float(radA[-1, 0]),
          "ref_vs_analytic_max": refA_vs_analytic},
    "B": {"n": int(rB0.shape[0]), "r0_vals": r0_vals.tolist(), "k_vals": k_vals.tolist(),
          "ref_vs_analytic_max": refB_vs_analytic,
          # the k=0 column must be a byte no-op (radius held == r0)
          "k0_noop_max_drift": float(np.abs(radB[:, kB == 0.0] - rB0[kB == 0.0][None, :]).max()),
          # r0 > R column must relax DOWN to R (no clamp): last radius should approach R from above
          "above_R_last_min": float(radB[-1, rB0 > R].min())},
    "ref_vs_analytic_max": max(refA_vs_analytic, refB_vs_analytic),
})

np.savez_compressed(
    os.path.join(OUT, "reference.npz"),
    radiusA=radA, aliveA=aliveA, r0A=rA0, kA=kA,
    radiusB=radB, aliveB=aliveB, r0B=rB0, kB=kB,
    r0_vals=r0_vals, k_vals=k_vals,
    R=np.float32(R), dt=np.float32(DT), T=np.int32(T),
)
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

# --- one sanity figure: scenario-B radius trajectories on a common scale --------------------- #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
tt = np.arange(T + 1) * DT
for j in range(rB0.shape[0]):
    ax.plot(tt, radB[:, j], lw=0.8, alpha=0.7)
ax.axhline(R, ls="--", c="k", lw=1, label=f"R = {R}")
ax.set_xlabel("t"), ax.set_ylabel("radius"), ax.set_title("reference SaturatingCellGrowth (grid B)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "reference.png"), dpi=120)
print("wrote reference.npz, summary.json, reference.png")
