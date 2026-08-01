"""Probe the DUMBBELL IC for the LJ differential: independent 2-cell pairs, each relaxing to contact.

A purely-adhesive 19-cell blob is geometrically frustrated -> collapses into the stiff r^-12 core
-> explodes (see _probe_lj.py). Instead use N well-SEPARATED pairs (centres >2.5*sigma apart, so
NO cross-pair force), each at a different initial separation that sweeps core -> adhesive tail ->
cutoff ramp -> beyond-cutoff. Each pair relaxes monotonically to the LJ equilibrium at CONTACT
(sigma), overdamped, with NO collapse and NO deep-core excursion. This isolates the LJ-DISCRIMINATING
features (adhesive tail, well-at-contact, smooth cutoff) that the repulsion-only siblings lack.

Checks: no NaN; min pair separation stays out of the deep core (>~0.9*sigma); each pair converges
toward contact; the SoftSphere negative-control trajectory (no adhesion) DIVERGES from LJ.
"""
import os, json
import numpy as np
import jax, jax.numpy as jnp
import jax_morph as jxm
from jax_morph.physics import LennardJones, SoftSphere, BrownianDynamics

OUT = os.environ["OUT"]
R0, EPS, DT, NST = 0.5, 1.0, 0.01, 100
SIGMA = 2 * R0                              # 1.0
SEPS = [1.15, 1.25, 1.35, 1.50, 1.70, 2.60]   # pair separations (sigma units): tail -> ramp -> beyond-cutoff
XC = [4.0, 9.0, 14.0, 19.0, 24.0, 29.0]       # pair centres, 5 apart (>2.5*sigma -> no cross-pair force)

# build the live dumbbells (each pair split symmetrically in y about its centre) + 4 dead slots
live = []
for xc, s in zip(XC, SEPS):
    live.append([xc, 20.0 + s / 2]); live.append([xc, 20.0 - s / 2])
P_live = np.round(np.array(live, np.float64), 6)          # [12,2]
N = len(P_live)

model = jxm.Model([BrownianDynamics(LennardJones(epsilon=EPS), n_space_dim=2, kT=0.0, gamma=1.0)])
mss = jxm.Model([BrownianDynamics(SoftSphere(epsilon=EPS), n_space_dim=2, kT=0.0, gamma=1.0)])


def seed(m, P):
    n = P.shape[0]
    s = jxm.build_state_from_model(m).init_empty(capacity=n, n_space_dim=2, n_types=1)
    return s.update(alive=s.alive.at[:n].set(True), radius=s.radius.at[:n].set(R0),
                    position=s.position.at[:n].set(jnp.asarray(P, jnp.float32)),
                    celltype=s.celltype.at[:n, 0].set(1.0))


hLJ = jxm.simulate(model, seed(model, P_live), n_steps=NST, dt=DT, key=jax.random.PRNGKey(0), history=True)
hSS = jxm.simulate(mss, seed(mss, P_live), n_steps=NST, dt=DT, key=jax.random.PRNGKey(0), history=True)
posLJ = np.asarray(hLJ.position); posSS = np.asarray(hSS.position)

# per-pair separation over the LJ trajectory
def pair_seps(pos):
    return np.array([[np.linalg.norm(pos[t, 2 * k] - pos[t, 2 * k + 1]) for k in range(N // 2)]
                     for t in range(pos.shape[0])])                # [NST+1, npairs]
sepsLJ = pair_seps(posLJ)

# min pairwise separation over the WHOLE trajectory (must stay out of the deep r^-12 core)
mins = []
for t in range(NST + 1):
    dd = np.linalg.norm(posLJ[t][:, None] - posLJ[t][None], axis=-1); np.fill_diagonal(dd, np.inf)
    mins.append(float(dd.min()))

out = {
    "N_live": N, "seps_initial": SEPS, "dt": DT, "NST": NST,
    "any_nan_LJ": bool(np.isnan(posLJ).any()),
    "traj_min_separation_ever": float(min(mins)),
    "pair_sep_initial": sepsLJ[0].tolist(),
    "pair_sep_final": sepsLJ[-1].tolist(),                         # should approach 1.0 (except the 2.60 frozen pair)
    "max_cell_displacement_LJ": float(np.linalg.norm(posLJ[-1] - posLJ[0], axis=-1).max()),
    "LJ_vs_SoftSphere_traj_maxdiff_over_sigma": float(np.abs(posLJ - posSS).max() / SIGMA),
    "SoftSphere_max_cell_displacement": float(np.linalg.norm(posSS[-1] - posSS[0], axis=-1).max()),
    "force_ic_max_LJ": float(np.abs(np.asarray(LennardJones(epsilon=EPS).forces(seed(model, P_live)))).max()),
}
with open(os.path.join(OUT, "probe_lj2.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
