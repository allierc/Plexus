"""Oracle for relax -- the jax-morph MechanicalRelaxation quasistatic equilibration, ISOLATED.

MechanicalRelaxation is a QUASISTATIC step: in ONE macro-step it drives cell positions to a
mechanical equilibrium x* of an interaction potential (grad_x U(x*) = 0), run by FIRE (Bitzek 2006)
to a genuine force tolerance f_tol -- a real force balance, not a fixed step count. To diff the
SOLVER (and the Plexus operator's quasistatic emit (x*-x0)/dt) we isolate it from every other
mechanism -- no growth, no division, fixed cell count / radii / slots -- and relax ONE rigid,
single-basin cluster under a Morse potential, whose genuine minimum makes the equilibrium a sharp
isolated attractor both float32 solvers must funnel into.

WHY A STIFF (over-constrained) CLUSTER, NOT A LOOSE ONE. A pilot found that a loose 7-cell hexagon
has soft ring-shear modes: FIRE creeps along a shallow valley where float32 gradient noise
dominates, so its residual STALLS at ~1.4e-4..4e-4 and never reaches a tight tolerance -- comparing
two solvers stopped in a floppy valley would measure noise, not the equilibrium. The initial
condition here is a 4-cell RIGID DIAMOND: two equilateral triangles sharing an edge (a centre pair
plus an up-apex and a down-apex), compressed to lattice spacing a = 0.85 sigma (a 15% compression
Morse relaxes back out), with a small DETERMINISTIC per-cell jitter (no RNG) that breaks the
symmetry so the relaxation moves cells in genuine 2D. It is over-constrained (5 strong bonds + a
weak far-diagonal Morse tail within the cutoff, vs 5 physical DOF) hence RIGID with NO floppy
modes: FIRE converges to |grad U|_inf ~ 9e-5 (well under the f_tol below), and the equilibrium is
unique up to the rigid-body gauge (which FIRE, starting at v = 0 with a conserved centroid, does
not excite). Centred at (20, 20) in free space, uniform radius 0.5 (sigma = 1.0). Four DEAD padding
slots are parked at the origin (state zeros, radius 0), so the dead-pair mask and the sigma = 0
safe_divide / safe_norm guards are exercised. Coordinates are rounded to 6 decimals and PRINTED as a
ready-to-paste YAML `start:` block, so the Plexus spec and this oracle share a BYTE-IDENTICAL
float32 IC.

Morse: epsilon = 3.0, alpha = 2.8, r_onset_frac = 1.5, r_cutoff_frac = 2.5 (the library defaults,
== the anchor jax_morph_proliferation.yaml). FIRE: max_steps = 2000, f_tol = 2e-4 (comfortably above
the ~9e-5 float32 force floor of this cluster, so BOTH the JAX reference and the torch operator
genuinely converge and stop at the tolerance rather than timing out; sharp enough that the
f_tol/kappa equilibrium shell is ~4e-6 sigma). All float32 (no x64), matching the Plexus torch
engine.

Asserted before anything is recorded (the oracle contract):
  1. determinism at a fixed key AND across two different keys (relax ignores the key and is
     deterministic -- a differential against it must not measure a PRNG stream);
  2. FIRE genuinely converged -- |grad U|_inf at x* is <= f_tol (a real equilibrium, not a
     max_steps timeout);
  3. the quasistatic STEP (one simulate macro-step) lands EXACTLY on the free-function
     relax_equilibrium x* (the step is the solver, nothing else);
  4. the equilibrium is a FIXED POINT -- relaxing again is a no-op (frames 2..N == frame 1),
     the property the Plexus operator must also show;
  5. dead padding slots never move.

Writes reference.npz (history positions/alive/radius/IC/x*/force-at-IC/residual) + summary.json.
"""
import os
import json

import numpy as np
import jax
import jax.numpy as jnp
import jax_morph as jxm
from jax_morph.physics import Morse, MechanicalRelaxation, relax_equilibrium

OUT = os.environ["OUT"]

# --- parameters (must match config/atlas_jax/mechanical_relaxation.yaml exactly) ------------------ #
N = 4                         # live cells (a rigid 4-cell diamond: two equilateral triangles)
CAP = 8                       # capacity: 4 live + 4 dead padding slots (dead-pair mask + sigma=0)
EPS = 3.0                     # Morse well depth (anchor value)
ALPHA = 2.8                   # Morse well steepness (anchor value)
R0 = 0.5                      # uniform cell radius -> sigma = r_i + r_j = 1.0
SIGMA = 2.0 * R0
A_LATT = 0.85                 # compressed lattice spacing (0.85 sigma -> 15% compression)
JITTER = 0.02                 # deterministic per-cell jitter amplitude (breaks the symmetry)
CENTER = np.array([20.0, 20.0])
MAX_STEPS = 2000              # FIRE fallback bound (NOT the stop test)
F_TOL = 2.0e-4                # force tolerance -> genuine equilibrium (above the float32 floor)
NSTEPS = 6                    # 6 quasistatic macro-steps: frame 1 = x*, frames 2..6 = fixed point


def diamond_cluster():
    """A rigid 4-cell diamond (two equilateral triangles sharing an edge), compressed + jittered.

    Cells: a centre pair (0, 1) one lattice spacing apart, an up-apex (2) and a down-apex (3), so
    five nearest-neighbour bonds sit at A_LATT and the far diagonal (2-3) at A_LATT*sqrt(3) falls
    inside the Morse cutoff as a weak tail -- an over-constrained (rigid, no floppy modes) cluster.
    The small index-keyed jitter (sin/cos of the cell index, NO RNG) tilts every cell off the
    symmetry axes so the relaxation moves cells in genuine 2D. Rounded to 6 decimals so the same
    decimal strings paste into the YAML `start:` block (byte-identical float32 IC)."""
    h = A_LATT * np.sqrt(3.0) / 2.0
    p = np.array([
        [0.0, 0.0],                 # 0: centre-left
        [A_LATT, 0.0],              # 1: centre-right (shares the 0-1 edge)
        [A_LATT / 2.0, h],          # 2: up-apex
        [A_LATT / 2.0, -h],         # 3: down-apex
    ])
    k = np.arange(N)
    jit = JITTER * np.stack([np.cos(2.1 * k), np.sin(1.4 * k + 0.5)], axis=1)  # deterministic
    p = p + jit + CENTER
    return np.round(p, 6)


P0_live = diamond_cluster().astype(np.float64)        # [N, 2], 6-decimal

# full-capacity IC: live cluster then dead slots at the origin (all zeros)
p0 = np.zeros((CAP, 2), np.float64)
p0[:N] = P0_live
radius = np.zeros((CAP,), np.float64)
radius[:N] = R0
alive0 = np.zeros((CAP,), bool)
alive0[:N] = True

# report the IC geometry so we can confirm the relaxation is actually ACTIVE (cells are compressed)
d = np.linalg.norm(P0_live[:, None] - P0_live[None], axis=-1)
np.fill_diagonal(d, np.inf)
nn = d.min(1)
print(f"IC: N={N} live, min_nn={nn.min():.4f} median_nn={np.median(nn):.4f} "
      f"max_nn={nn.max():.4f}  (sigma={SIGMA}; compressed pairs r<sigma = "
      f"{int(((d < SIGMA) & np.isfinite(d)).sum() // 2)})", flush=True)

# --- the model: MechanicalRelaxation(Morse) run ALONE ---------------------------------------- #
pot = Morse(epsilon=EPS, alpha=ALPHA)                 # r_onset_frac=1.5, r_cutoff_frac=2.5 defaults
model = jxm.Model([MechanicalRelaxation(pot, max_steps=MAX_STEPS, f_tol=F_TOL)])


def seed_state():
    s = jxm.build_state_from_model(model).init_empty(capacity=CAP, n_space_dim=2, n_types=1)
    return s.update(
        alive=s.alive.at[:N].set(True),
        radius=s.radius.at[:N].set(R0),
        position=s.position.at[:CAP].set(jnp.asarray(p0, jnp.float32)),
        celltype=s.celltype.at[:N, 0].set(1.0),
    )


s0 = seed_state()
print(f"position dtype = {s0.position.dtype}  (expect float32, matching the torch engine)",
      flush=True)

# --- (2) FIRE genuinely converged: the free function's x* is a real force balance ------------- #
x_star = relax_equilibrium(pot, s0, max_steps=MAX_STEPS, f_tol=F_TOL)
s_star = s0.update(position=x_star)
residual_at_xstar = float(jnp.max(jnp.abs(pot.forces(s_star))))    # |grad U|_inf at x*
converged = bool(residual_at_xstar <= F_TOL)
if not converged:
    raise SystemExit(f"FIRE did NOT converge: |grad U|_inf at x* = {residual_at_xstar:.3e} > "
                     f"f_tol = {F_TOL:.1e}. The reference is off-equilibrium -- a differential "
                     "against it would compare a solver timeout, not an equilibrium. Stop here.")


def run(key):
    return jxm.simulate(model, seed_state(), n_steps=NSTEPS, dt=1.0, key=key, history=True)


h1 = run(jax.random.PRNGKey(0))
h2 = run(jax.random.PRNGKey(0))            # same key
h3 = run(jax.random.PRNGKey(12345))        # DIFFERENT key: relax ignores it -> must be identical

pos1 = np.asarray(h1.position)             # [NSTEPS+1, CAP, 2]
same_key = bool(np.array_equal(pos1, np.asarray(h2.position)))
diff_key = bool(np.array_equal(pos1, np.asarray(h3.position)))
if not (same_key and diff_key):
    raise SystemExit("MechanicalRelaxation is not deterministic (same_key="
                     f"{same_key}, diff_key={diff_key}) -- a differential test against it would "
                     "measure the reference's own PRNG. Stop here.")

# --- (3) the quasistatic STEP lands exactly on the free-function x* --------------------------- #
step_vs_func = float(np.abs(pos1[1] - np.asarray(x_star)).max())   # frame 1 vs relax_equilibrium x*

# --- (4) the equilibrium is a FIXED POINT: relaxing again is a no-op -------------------------- #
plateau_drift = float(np.abs(pos1[2:] - pos1[1:2]).max()) if NSTEPS >= 2 else 0.0

# --- (5) dead slots never moved -------------------------------------------------------------- #
dead_moved = float(np.abs(pos1[:, N:] - pos1[0:1, N:]).max())

# raw force at the IC (a single-step probe of the force that drives the first FIRE iteration)
force_ic = np.asarray(pot.forces(seed_state()))       # [CAP, 2]

alive = np.asarray(h1.alive)               # [NSTEPS+1, CAP]
alive_fixed = bool(np.array_equal(alive, np.broadcast_to(alive[0], alive.shape)))


def gyration(p, a):
    q = p[a.astype(bool)]
    c = q.mean(0)
    return float(np.sqrt(((q - c) ** 2).sum(1).mean()))


def mean_nn(p, a):
    q = p[a.astype(bool)]
    dd = np.linalg.norm(q[:, None] - q[None], axis=-1)
    np.fill_diagonal(dd, np.inf)
    return float(dd.min(1).mean())


gyr = [gyration(pos1[t], alive[t]) for t in range(NSTEPS + 1)]
mnn = [mean_nn(pos1[t], alive[t]) for t in range(NSTEPS + 1)]
# how far did the LIVE cluster move overall (the magnitude the diff must sit far under)?
disp_live = float(np.linalg.norm(pos1[1, :N] - pos1[0, :N], axis=-1).max())

np.savez_compressed(
    os.path.join(OUT, "reference.npz"),
    position=pos1.astype(np.float32),      # [NSTEPS+1, CAP, 2] index t = IC after t macro-steps
    alive=alive,                           # [NSTEPS+1, CAP]
    radius=radius.astype(np.float32),      # [CAP]
    p0=p0.astype(np.float32),              # [CAP, 2] the IC
    x_star=np.asarray(x_star).astype(np.float32),   # [CAP, 2] the relaxed equilibrium (free fn)
    force_ic=force_ic.astype(np.float32),  # [CAP, 2] -grad U at the IC
    N=np.int32(N), CAP=np.int32(CAP), NSTEPS=np.int32(NSTEPS),
    eps=np.float32(EPS), alpha=np.float32(ALPHA), r0=np.float32(R0), sigma=np.float32(SIGMA),
    max_steps=np.int32(MAX_STEPS), f_tol=np.float32(F_TOL),
    residual_at_xstar=np.float32(residual_at_xstar),
)

summary = {
    "role": "oracle", "operator": "relax (MechanicalRelaxation)",
    "model": f"MechanicalRelaxation(Morse(epsilon={EPS}, alpha={ALPHA}), "
             f"max_steps={MAX_STEPS}, f_tol={F_TOL})",
    "N": N, "CAP": CAP, "NSTEPS": NSTEPS, "eps": EPS, "alpha": ALPHA, "r0": R0, "sigma": SIGMA,
    "a_latt": A_LATT, "jitter": JITTER, "center": CENTER.tolist(),
    "max_steps": MAX_STEPS, "f_tol": F_TOL,
    "position_dtype": str(s0.position.dtype),
    "deterministic_same_key": same_key, "deterministic_diff_key": diff_key,
    "fire_converged": converged, "residual_at_xstar": residual_at_xstar,
    "step_equals_relax_equilibrium_x_star": step_vs_func,
    "equilibrium_fixed_point_plateau_drift": plateau_drift,
    "alive_fixed": alive_fixed,
    "ic_min_nn": float(nn.min()), "ic_median_nn": float(np.median(nn)),
    "force_ic_max": float(np.abs(force_ic).max()),
    "gyration_first": gyr[0], "gyration_equilibrium": gyr[1], "gyration_last": gyr[-1],
    "mean_nn_first": mnn[0], "mean_nn_equilibrium": mnn[1],
    "max_live_cell_displacement_first_step": disp_live,
    "dead_slots_max_move": dead_moved,
}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

# --- the ready-to-paste YAML `start:` block (byte-identical float32 IC for the Plexus spec) --- #
print("\n# ---- paste into config/atlas_jax/mechanical_relaxation.yaml sets.cell.start (LIVE only) ----")
print("    start:")
for x, y in P0_live:
    print(f"    - [{x:.6f}, {y:.6f}]")
print("# ---- end start block ----")
print("wrote reference.npz, summary.json")
