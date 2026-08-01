"""Probe the jax-morph LennardJones API + pick a STABLE, adhesive IC before writing the full differ.

LJ is the r_min 12-6 well: U = eps*((sigma/r)^12 - 2(sigma/r)^6), min -eps EXACTLY at contact
sigma = r_i + r_j, times a sigma-relative smooth cutoff on [1.5, 2.5]*sigma. Unlike the
repulsion-only SoftSphere/Hertzian it has an ADHESIVE tail, and a HARD r^-12 core that becomes
numerically violent under deep overlap (~1e8) -- so the differential IC must sit in the ADHESIVE
window (NN just BEYOND contact) and gently contract to contact, never diving into the stiff core.

Confirms: (1) dtype; (2) LennardJones.forces == analytic radial LJ force on a 2-cell scan, with
the well AT CONTACT (f(sigma)=0, the r_min discriminator) and f=0 beyond 2.5*sigma; (3) the
SoftSphere negative control differs (adhesion is real); (4) BrownianDynamics(LJ, kT=0) is
deterministic forward Euler dx = dt*forces; (5) a candidate 19-cell adhesive sunflower IC is
STABLE under the reference integrator (no NaN, min pair separation stays out of the deep core,
cluster CONTRACTS toward contact).
"""
import os, json
import numpy as np
import jax, jax.numpy as jnp
import jax_morph as jxm
from jax_morph.physics import LennardJones, SoftSphere, BrownianDynamics

OUT = os.environ["OUT"]
out = {}
out["x64_enabled"] = bool(jax.config.read("jax_enable_x64"))

model = jxm.Model([BrownianDynamics(LennardJones(epsilon=1.0), n_space_dim=2, kT=0.0, gamma=1.0)])


def state_of(pos, radius):
    pos = jnp.asarray(np.asarray(pos, np.float32))
    radius = jnp.asarray(np.asarray(radius, np.float32))
    N = pos.shape[0]
    s = jxm.build_state_from_model(model).init_empty(capacity=N, n_space_dim=2, n_types=1)
    return s.update(alive=s.alive.at[:N].set(True), radius=s.radius.at[:N].set(radius),
                    position=s.position.at[:N].set(pos), celltype=s.celltype.at[:N, 0].set(1.0))


# --- (2) 2-cell radial scan: LennardJones.forces vs analytic --------------------------------- #
# analytic radial force (component pushing cell-1 toward +x): for r < r_on=1.5*sigma the cutoff
# S=1, S'=0, so f = -dU/dr = 12*eps*((sigma/r)^12 - (sigma/r)^6)/r.  Positive => repulsive (+x).
r_i, r_j, eps = 0.5, 0.5, 1.0
sigma = r_i + r_j                         # 1.0
seps = np.linspace(0.85, 2.7, 38).astype(np.float32)   # deep-ish core -> adhesive tail -> past cutoff
f_ref, f_an = [], []
for r in seps:
    st = state_of([[0.0, 0.0], [float(r), 0.0]], [r_i, r_j])
    F = np.asarray(LennardJones(epsilon=eps).forces(st))
    f_ref.append(float(F[1, 0]))          # x-force on cell 1
    x = sigma / r
    if r < 1.5 * sigma:                   # S = 1 region (analytic without cutoff-derivative)
        f_an.append(12.0 * eps * (x**12 - x**6) / r)
    else:
        f_an.append(np.nan)               # cutoff-ramp region compared separately below
f_ref = np.array(f_ref); f_an = np.array(f_an)
core_mask = seps < 1.5 * sigma
out["dtype_forces"] = str(F.dtype)
out["scan_ref_vs_analytic_max_below_ron"] = float(np.nanmax(np.abs(f_ref[core_mask] - f_an[core_mask])))
# well AT contact: force ~ 0 at r = sigma (the r_min discriminator vs the textbook 4-eps form)
st_c = state_of([[0.0, 0.0], [1.0, 0.0]], [r_i, r_j])
out["force_at_contact"] = float(np.asarray(LennardJones(epsilon=eps).forces(st_c))[1, 0])
# adhesive (negative/attractive -x force on cell1) just beyond contact
st_a = state_of([[0.0, 0.0], [1.25, 0.0]], [r_i, r_j])
out["force_at_1p25_sigma"] = float(np.asarray(LennardJones(epsilon=eps).forces(st_a))[1, 0])
# exactly 0 beyond the cutoff end
st_z = state_of([[0.0, 0.0], [2.6, 0.0]], [r_i, r_j])
out["force_beyond_cutoff_2p6"] = float(np.asarray(LennardJones(epsilon=eps).forces(st_z))[1, 0])

# --- (3) SoftSphere negative control: purely repulsive, ZERO on a non-overlapping adhesive pair #
Fss_adh = np.asarray(SoftSphere(epsilon=eps).forces(st_a))[1, 0]
out["softsphere_force_at_1p25_sigma"] = float(Fss_adh)   # ~0 (no overlap) -> LJ adhesion is discriminating

# --- (4) deterministic forward Euler? -------------------------------------------------------- #
pos7 = [[0, 0], [1.15, 0], [0.55, 1.0], [-0.6, 0.55], [0.35, -0.65], [1.3, 0.6], [-0.25, -0.9]]
rad7 = [0.5] * 7
s0 = state_of(pos7, rad7)
h1 = jxm.simulate(model, s0, n_steps=5, dt=0.02, key=jax.random.PRNGKey(0), history=True)
h2 = jxm.simulate(model, s0, n_steps=5, dt=0.02, key=jax.random.PRNGKey(999), history=True)
p1 = np.asarray(h1.position); p2 = np.asarray(h2.position)
out["bd_kT0_deterministic_across_keys"] = float(np.abs(p1 - p2).max())
F0 = np.asarray(LennardJones(epsilon=1.0).forces(s0))
out["bd_first_step_vs_dt_forces_max"] = float(np.abs((p1[1] - p1[0]) - 0.02 * F0).max())

# --- (5) candidate 19-cell adhesive sunflower IC: STABILITY under the reference --------------- #
def sunflower(n, scale, center):
    ga = np.pi * (3.0 - np.sqrt(5.0))
    k = np.arange(n)
    r = scale * np.sqrt(k + 0.5)
    th = k * ga
    return np.round(np.stack([center[0] + r*np.cos(th), center[1] + r*np.sin(th)], 1), 6)

for SCALE in (0.58, 0.62, 0.66, 0.70):
    N = 19
    P = sunflower(N, SCALE, np.array([20.0, 20.0])).astype(np.float64)
    d = np.linalg.norm(P[:, None] - P[None], axis=-1); np.fill_diagonal(d, np.inf)
    nn = d.min(1)
    st = state_of(P, [0.5]*N)
    DT, NST = 0.02, 120
    h = jxm.simulate(model, st, n_steps=NST, dt=DT, key=jax.random.PRNGKey(0), history=True)
    pos = np.asarray(h.position)          # [NST+1, N, 2]
    anyNaN = bool(np.isnan(pos).any())
    # min pairwise separation over the whole trajectory (must stay OUT of the deep r^-12 core)
    mins = []
    for t in range(NST + 1):
        dd = np.linalg.norm(pos[t][:, None] - pos[t][None], axis=-1); np.fill_diagonal(dd, np.inf)
        mins.append(float(dd.min()))
    def gyr(p):
        c = p.mean(0); return float(np.sqrt(((p - c) ** 2).sum(1).mean()))
    Fic = np.asarray(LennardJones(epsilon=1.0).forces(st))
    out[f"IC_scale_{SCALE}"] = {
        "ic_min_nn": float(nn.min()), "ic_median_nn": float(np.median(nn)), "ic_max_nn": float(nn.max()),
        "ic_force_max": float(np.abs(Fic).max()),
        "traj_min_separation_ever": float(min(mins)), "any_nan": anyNaN,
        "gyration_first": gyr(pos[0]), "gyration_last": gyr(pos[-1]),
        "max_cell_displacement": float(np.linalg.norm(pos[-1] - pos[0], axis=-1).max()),
    }

with open(os.path.join(OUT, "probe_lj.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
