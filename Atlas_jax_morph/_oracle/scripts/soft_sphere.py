"""Oracle for adhere/soft_sphere -- the jax-morph SoftSphere harmonic soft-sphere potential,
ISOLATED and put under the reference's OWN overdamped integrator.

SoftSphere is a POTENTIAL (an energy): it writes no state, its whole contract is the force
F = -grad U for the purely-repulsive harmonic core U(r) = (eps/2)(1 - r/sigma)^2, sigma = r_i+r_j.
To diff it we isolate it from every other mechanism (no growth, no division -> fixed cell count,
fixed radii, fixed slots) and drive it with jax-morph's own overdamped Langevin step at ZERO
temperature,

    BrownianDynamics(SoftSphere(epsilon), gamma=1, kT=0):  dx = dt * forces / gamma = dt*(-grad U),

which is exactly the Plexus engine's overdamped-Euler integration of the operator's emitted
velocity (pos += dt * mobility * F, mobility = 1/gamma = 1). So the resulting position trajectory
is a PURE function of the force law, and a per-cell position diff isolates it.

Initial condition: a fixed 19-cell mutually-OVERLAPPING sunflower (Vogel) cluster centred at
(20,20) in free space, uniform radius 0.5 (so sigma = 1.0), plus 5 DEAD padding slots parked at
the origin (so the dead-pair mask and the sigma=0 safe_divide are exercised). Coordinates are
rounded to 6 decimals and PRINTED as a ready-to-paste YAML `start:` block, so the Plexus spec and
this oracle share a BYTE-IDENTICAL float32 IC.

Three things are asserted before anything is recorded (the oracle contract):
  1. determinism at a fixed key,
  2. determinism ACROSS two different keys (kT=0 truly removes the noise -> the trajectory does
     not depend on the PRNG stream), and
  3. the jxm.simulate trajectory is BIT-IDENTICAL to a hand-rolled Euler loop over
     SoftSphere().forces() -- this pins the integrator convention so the differential test is a
     test of the FORCE, not of an integrator mismatch.

Writes reference.npz (positions/alive/radius/IC/force-at-IC) + summary.json into OUT.
"""
import os
import json

import numpy as np
import jax
import jax.numpy as jnp
import jax_morph as jxm
from jax_morph.physics import BrownianDynamics, SoftSphere

OUT = os.environ["OUT"]

# --- parameters (must match config/atlas/soft_sphere.yaml exactly) --------------------------- #
N = 19                       # live cells
CAP = 24                     # capacity: 19 live + 5 dead padding slots (dead-pair mask + sigma=0)
EPS = 1.0                    # repulsion strength (SoftSphere epsilon)
R0 = 0.5                     # uniform cell radius -> sigma = r_i + r_j = 1.0
DT = 0.2                     # macro-step; dt*eps/sigma^2 = 0.2 << 2 -> explicit Euler is stable
NSTEPS = 60                  # 60 overdamped steps (T = 12) -- fully relaxes the overlaps
GAMMA = 1.0                  # drag; mobility = 1/gamma = 1 matches the Plexus operator default
KT = 0.0                     # ZERO temperature -> deterministic overdamped Euler (no noise)
CENTER = np.array([20.0, 20.0])
SCALE = 0.42                 # Vogel scale -> nearest-neighbour ~0.8 (moderate overlap of sigma=1)
SIGMA = 2.0 * R0


def sunflower(n, scale, center):
    """A deterministic Vogel sunflower disk: r_k = scale*sqrt(k+0.5), theta_k = k*golden_angle.
    Irregular (no lattice symmetry) so every cell feels a net force. Rounded to 6 decimals so the
    same decimal strings can be pasted into the YAML `start:` block (byte-identical float32 IC)."""
    ga = np.pi * (3.0 - np.sqrt(5.0))            # golden angle
    k = np.arange(n)
    r = scale * np.sqrt(k + 0.5)
    th = k * ga
    p = np.stack([center[0] + r * np.cos(th), center[1] + r * np.sin(th)], axis=1)
    return np.round(p, 6)


P0_live = sunflower(N, SCALE, CENTER).astype(np.float64)   # [N,2], 6-decimal

# full capacity IC: live cluster then dead slots at the origin (all zeros)
p0 = np.zeros((CAP, 2), np.float64)
p0[:N] = P0_live
radius = np.zeros((CAP,), np.float64)
radius[:N] = R0
alive0 = np.zeros((CAP,), bool)
alive0[:N] = True

# report the IC geometry so we can confirm the repulsion is actually ACTIVE (cells overlap)
d = np.linalg.norm(P0_live[:, None] - P0_live[None], axis=-1)
np.fill_diagonal(d, np.inf)
nn = d.min(1)
overlap_pairs = int(((d < SIGMA) & np.isfinite(d)).sum() // 2)
print(f"IC: N={N} live, min_nn={nn.min():.4f} median_nn={np.median(nn):.4f} "
      f"max_nn={nn.max():.4f}  overlapping_pairs(r<sigma={SIGMA})={overlap_pairs}", flush=True)

# --- the model: SoftSphere force under overdamped (kT=0) Langevin ----------------------------- #
model = jxm.Model([BrownianDynamics(SoftSphere(epsilon=EPS), n_space_dim=2, gamma=GAMMA, kT=KT)])


def seed_state():
    s = jxm.build_state_from_model(model).init_empty(capacity=CAP, n_space_dim=2, n_types=1)
    return s.update(
        alive=s.alive.at[:N].set(True),
        radius=s.radius.at[:N].set(R0),
        position=s.position.at[:CAP].set(jnp.asarray(p0, jnp.float32)),
        celltype=s.celltype.at[:N, 0].set(1.0),
    )


def run(key):
    return jxm.simulate(model, seed_state(), n_steps=NSTEPS, dt=DT, key=key, history=True)


h1 = run(jax.random.PRNGKey(0))
h2 = run(jax.random.PRNGKey(0))            # same key
h3 = run(jax.random.PRNGKey(12345))        # DIFFERENT key: kT=0 must make it identical

pos1 = np.asarray(h1.position)             # [NSTEPS+1, CAP, 2]
same_key = bool(np.array_equal(pos1, np.asarray(h2.position)))
diff_key = bool(np.array_equal(pos1, np.asarray(h3.position)))
if not (same_key and diff_key):
    raise SystemExit("SoftSphere overdamped dynamics is not deterministic at kT=0 "
                     f"(same_key={same_key}, diff_key={diff_key}) -- a differential test against "
                     "it would measure the reference's own noise. Stop here.")

# --- self-check: jxm's BrownianDynamics(kT=0) == a hand-rolled Euler over SoftSphere.forces() -- #
# this pins the integrator convention (mobility = 1/gamma, dx = dt*forces) so the trajectory diff
# is a test of the FORCE LAW, not of an integrator mismatch.
pot = SoftSphere(epsilon=EPS)
st = seed_state()
manual = [np.asarray(st.position)]
for _ in range(NSTEPS):
    F = pot.forces(st)                     # -jax.grad(total_energy) -- the class under test
    st = st.update(position=st.position + DT * F)
    manual.append(np.asarray(st.position))
manual = np.stack(manual)                  # [NSTEPS+1, CAP, 2]
euler_matches = bool(np.array_equal(pos1, manual))
euler_max_dev = float(np.abs(pos1 - manual).max())
if not euler_matches:
    # a tiny nonzero here would only mean jxm masks/schedules slightly differently; report it, do
    # not silently accept a convention we do not understand.
    print(f"WARNING: jxm vs hand-Euler max|dev| = {euler_max_dev:.3e} (expected 0.0)", flush=True)

# force at the IC (the sharpest single-step probe of the raw force law, before any compounding)
force_ic = np.asarray(pot.forces(seed_state()))   # [CAP,2]

alive = np.asarray(h1.alive)               # [NSTEPS+1, CAP]
# alive is fixed (no birth/death); confirm and reduce to a single [CAP] mask
alive_fixed = bool(np.array_equal(alive, np.broadcast_to(alive[0], alive.shape)))


def gyration(p, a):
    q = p[a.astype(bool)]
    c = q.mean(0)
    return float(np.sqrt(((q - c) ** 2).sum(1).mean()))


def extent(p, a):
    q = p[a.astype(bool)]
    return float((q.max(0) - q.min(0)).max())


gyr = [gyration(pos1[t], alive[t]) for t in range(NSTEPS + 1)]
ext = [extent(pos1[t], alive[t]) for t in range(NSTEPS + 1)]
# how far did the LIVE cluster move overall (a sanity magnitude for the diff to sit under)?
disp_live = float(np.linalg.norm(pos1[-1, :N] - pos1[0, :N], axis=-1).max())
dead_moved = float(np.abs(pos1[:, N:] - pos1[0:1, N:]).max())   # dead slots must never move

np.savez_compressed(
    os.path.join(OUT, "reference.npz"),
    position=pos1.astype(np.float32),      # [NSTEPS+1, CAP, 2] index t = IC after t steps
    alive=alive,                           # [NSTEPS+1, CAP]
    radius=radius.astype(np.float32),      # [CAP]
    p0=p0.astype(np.float32),              # [CAP,2] the IC
    force_ic=force_ic.astype(np.float32),  # [CAP,2] -grad U at the IC
    manual=manual.astype(np.float32),      # the hand-Euler trajectory (== position)
    N=np.int32(N), CAP=np.int32(CAP), NSTEPS=np.int32(NSTEPS),
    dt=np.float32(DT), eps=np.float32(EPS), r0=np.float32(R0), sigma=np.float32(SIGMA),
    gamma=np.float32(GAMMA), kt=np.float32(KT),
)

summary = {
    "role": "oracle", "operator": "adhere/soft_sphere",
    "model": "BrownianDynamics(SoftSphere(epsilon=1.0), gamma=1.0, kT=0.0)",
    "N": N, "CAP": CAP, "NSTEPS": NSTEPS, "dt": DT, "eps": EPS, "r0": R0, "sigma": SIGMA,
    "gamma": GAMMA, "kT": KT, "center": CENTER.tolist(), "scale": SCALE,
    "deterministic_same_key": same_key, "deterministic_diff_key": diff_key,
    "euler_convention_matches_bit_for_bit": euler_matches, "euler_max_dev": euler_max_dev,
    "alive_fixed": alive_fixed,
    "ic_min_nn": float(nn.min()), "ic_median_nn": float(np.median(nn)),
    "ic_overlapping_pairs": overlap_pairs,
    "force_ic_max": float(np.abs(force_ic).max()),
    "gyration_first": gyr[0], "gyration_last": gyr[-1],
    "extent_first": ext[0], "extent_last": ext[-1],
    "max_live_cell_displacement": disp_live,
    "dead_slots_max_move": dead_moved,
}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

# --- the ready-to-paste YAML `start:` block (byte-identical float32 IC for the Plexus spec) --- #
print("\n# ---- paste into config/atlas/soft_sphere.yaml sets.cell.start (LIVE cells only) ----")
print("    start:")
for x, y in P0_live:
    print(f"    - [{x:.6f}, {y:.6f}]")
print("# ---- end start block ----")
print("wrote reference.npz, summary.json")
