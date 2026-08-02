"""Oracle for adhere/lennard_jones -- the jax-morph LennardJones potential, ISOLATED and driven by
the reference's OWN overdamped integrator.

LennardJones is a POTENTIAL (an energy): it writes no state, its whole contract is the force
F = -grad U for the r_min 12-6 well U(r) = eps((sigma/r)^12 - 2(sigma/r)^6), min -eps EXACTLY at
contact sigma = r_i+r_j, its adhesive tail truncated by a sigma-relative smooth C1 cutoff on
[1.5, 2.5]*sigma. To diff it we isolate it from every other mechanism (fixed cell count, fixed
radii) and drive it with jax-morph's own overdamped Langevin step at ZERO temperature,

    BrownianDynamics(LennardJones(epsilon), gamma=1, kT=0):  dx = dt * forces / gamma = dt*(-grad U),

which is exactly the Plexus engine's overdamped-Euler integration of the operator's emitted
velocity (pos += dt*mobility*F, mobility = 1/gamma = 1). So the position trajectory is a PURE
function of the LJ force law, and a per-cell position diff isolates it.

INITIAL CONDITION -- the key design choice for THIS member. A purely-adhesive many-cell blob is
geometrically frustrated: it collapses into the stiff r^-12 core and the explicit integrator
explodes (verified in _probe_lj.py). So the IC is SIX well-separated dumbbell pairs (centres
5 apart, >2.5*sigma, so NO cross-pair force), each at a different initial separation that sweeps
the ADHESIVE tail, the cutoff RAMP, and BEYOND the cutoff: [1.15, 1.25, 1.35, 1.50, 1.70, 2.60]*sigma.
Each pair relaxes MONOTONICALLY to contact under overdamped dynamics -- no frustration, no
collapse, no deep-core excursion -- so the trajectory isolates the LJ-DISCRIMINATING feature (the
adhesive tail + the equilibrium AT contact) that the repulsion-only siblings SoftSphere/Hertzian
lack. Plus 4 DEAD padding slots at the origin (radius 0 -> sigma 0) to exercise the dead-pair mask
and the sigma=0 safe_divide/cutoff guards, exactly as the soft_sphere oracle does.

Asserted before anything is recorded (the oracle contract):
  1. determinism at a fixed key, and ACROSS two different keys (kT=0 truly removes the noise);
  2. jxm.simulate is BIT-IDENTICAL to a hand-rolled Euler loop over LennardJones().forces()
     (pins the integrator convention: the diff tests the FORCE, not an integrator mismatch);
  3. the trajectory has no NaN and never dips into the deep core (min separation stays near contact).

Also writes a 2-cell radial SCAN of LennardJones.forces vs the analytic r_min LJ force (well AT
contact, =0 beyond 2.5*sigma) -- a guard on the REFERENCE itself -- and a SoftSphere trajectory on
the SAME IC as the negative control (adhesion off -> pairs frozen).

Writes reference.npz + summary.json + reference.png into OUT.
"""
import os
import json

import numpy as np
import jax
import jax.numpy as jnp
import jax_morph as jxm
from jax_morph.physics import BrownianDynamics, LennardJones, SoftSphere

OUT = os.environ["OUT"]

# --- parameters (must match config/atlas_jax/lennard_jones.yaml exactly) -------------------------- #
R0 = 0.5                      # uniform cell radius -> sigma = r_i + r_j = 1.0
SIGMA = 2.0 * R0
EPS = 1.0                     # LJ well depth
DT = 0.01                     # macro-step; small so the tail relaxation is smooth (no core-slam)
NSTEPS = 100                  # 100 overdamped steps: every tail pair reaches contact
GAMMA = 1.0                   # drag; mobility = 1/gamma = 1 matches the Plexus operator default
KT = 0.0                      # ZERO temperature -> deterministic overdamped Euler
CENTER_Y = 20.0
XC = [4.0, 9.0, 14.0, 19.0, 24.0, 29.0]              # pair centres, 5 apart (>2.5*sigma)
SEPS = [1.15, 1.25, 1.35, 1.50, 1.70, 2.60]          # initial pair separations (sigma units)
N = 2 * len(XC)               # 12 live cells (6 dumbbells)
CAP = N + 4                   # + 4 dead padding slots at the origin (dead-pair mask + sigma=0)

# --- byte-identical float32 IC (6-decimal so it pastes into the YAML `start:` block) ---------- #
live = []
for xc, s in zip(XC, SEPS):
    live.append([xc, CENTER_Y + s / 2.0])
    live.append([xc, CENTER_Y - s / 2.0])
P0_live = np.round(np.array(live, np.float64), 6)     # [N,2]

p0 = np.zeros((CAP, 2), np.float64)
p0[:N] = P0_live
radius = np.zeros((CAP,), np.float64)
radius[:N] = R0
alive0 = np.zeros((CAP,), bool)
alive0[:N] = True

model = jxm.Model([BrownianDynamics(LennardJones(epsilon=EPS), n_space_dim=2, gamma=GAMMA, kT=KT)])
model_ss = jxm.Model([BrownianDynamics(SoftSphere(epsilon=EPS), n_space_dim=2, gamma=GAMMA, kT=KT)])


def seed_state(m):
    s = jxm.build_state_from_model(m).init_empty(capacity=CAP, n_space_dim=2, n_types=1)
    return s.update(
        alive=s.alive.at[:N].set(True),
        radius=s.radius.at[:N].set(R0),
        position=s.position.at[:CAP].set(jnp.asarray(p0, jnp.float32)),
        celltype=s.celltype.at[:N, 0].set(1.0),
    )


def run(m, key):
    return jxm.simulate(m, seed_state(m), n_steps=NSTEPS, dt=DT, key=key, history=True)


# --- determinism (same key + across keys: kT=0 must remove the PRNG dependence) --------------- #
h1 = run(model, jax.random.PRNGKey(0))
h2 = run(model, jax.random.PRNGKey(0))
h3 = run(model, jax.random.PRNGKey(12345))
pos1 = np.asarray(h1.position)                        # [NSTEPS+1, CAP, 2]
same_key = bool(np.array_equal(pos1, np.asarray(h2.position)))
diff_key = bool(np.array_equal(pos1, np.asarray(h3.position)))
if not (same_key and diff_key):
    raise SystemExit("LennardJones overdamped dynamics is not deterministic at kT=0 "
                     f"(same_key={same_key}, diff_key={diff_key}) -- a differential against it would "
                     "measure the reference's own noise. Stop here.")

# --- self-check: jxm BrownianDynamics(kT=0) == hand-rolled Euler over LennardJones.forces() ---- #
pot = LennardJones(epsilon=EPS)
st = seed_state(model)
manual = [np.asarray(st.position)]
for _ in range(NSTEPS):
    F = pot.forces(st)                                # -jax.grad(total_energy) -- the class under test
    st = st.update(position=st.position + DT * F)
    manual.append(np.asarray(st.position))
manual = np.stack(manual)
euler_matches = bool(np.array_equal(pos1, manual))
euler_max_dev = float(np.abs(pos1 - manual).max())
if not euler_matches:
    print(f"WARNING: jxm vs hand-Euler max|dev| = {euler_max_dev:.3e} (expected 0.0)", flush=True)

any_nan = bool(np.isnan(pos1).any())
# min pairwise separation over the whole live trajectory (must stay OUT of the deep r^-12 core)
mins = []
for t in range(NSTEPS + 1):
    q = pos1[t, :N]
    dd = np.linalg.norm(q[:, None] - q[None], axis=-1); np.fill_diagonal(dd, np.inf)
    mins.append(float(dd.min()))
traj_min_sep = float(min(mins))
if any_nan or traj_min_sep < 0.85 * SIGMA:
    raise SystemExit(f"trajectory left the tame regime (any_nan={any_nan}, "
                     f"min_sep={traj_min_sep:.4f} < 0.85*sigma): a stiff-core excursion would make the "
                     "float32 diff meaningless. Stop and re-scale the IC.")

# force at the IC (the sharpest single-step probe of the raw force law before compounding)
force_ic = np.asarray(pot.forces(seed_state(model)))  # [CAP,2]

# per-pair separation trajectory (physics sanity + a corroborator vs the Plexus side)
def pair_seps(pos):
    return np.array([[float(np.linalg.norm(pos[t, 2 * k] - pos[t, 2 * k + 1]))
                      for k in range(len(XC))] for t in range(pos.shape[0])])   # [NSTEPS+1, npairs]
seps_traj = pair_seps(pos1)

alive = np.asarray(h1.alive)                          # [NSTEPS+1, CAP]
alive_fixed = bool(np.array_equal(alive, np.broadcast_to(alive[0], alive.shape)))

# --- SoftSphere negative control on the SAME IC (adhesion off -> pairs frozen) ---------------- #
pos_ss = np.asarray(run(model_ss, jax.random.PRNGKey(0)).position)
lj_vs_ss = float(np.abs(pos1 - pos_ss).max() / SIGMA)
ss_max_disp = float(np.linalg.norm(pos_ss[-1, :N] - pos_ss[0, :N], axis=-1).max())

# --- 2-cell radial scan: LennardJones.forces vs analytic r_min force (guard the REFERENCE) ----- #
def scan_state(r):
    s = jxm.build_state_from_model(model).init_empty(capacity=2, n_space_dim=2, n_types=1)
    return s.update(alive=s.alive.at[:2].set(True), radius=s.radius.at[:2].set(R0),
                    position=s.position.at[:2].set(jnp.asarray([[0.0, 0.0], [float(r), 0.0]], jnp.float32)),
                    celltype=s.celltype.at[:2, 0].set(1.0))

r_scan = np.linspace(0.90, 2.70, 46).astype(np.float32)
f_ref_scan, f_an_scan = [], []
for r in r_scan:
    F = np.asarray(LennardJones(epsilon=EPS).forces(scan_state(r)))
    f_ref_scan.append(float(F[1, 0]))
    x = SIGMA / float(r)
    # analytic radial LJ force (S=1 region, r < 1.5*sigma): f = -dU/dr = 12 eps (x^12 - x^6)/r
    f_an_scan.append(12.0 * EPS * (x**12 - x**6) / float(r) if r < 1.5 * SIGMA else np.nan)
f_ref_scan = np.array(f_ref_scan); f_an_scan = np.array(f_an_scan)
below_ron = r_scan < 1.5 * SIGMA
scan_ref_vs_analytic_max = float(np.nanmax(np.abs(f_ref_scan[below_ron] - f_an_scan[below_ron])))
force_at_contact = float(np.asarray(LennardJones(epsilon=EPS).forces(scan_state(SIGMA)))[1, 0])
force_beyond_cutoff = float(np.asarray(LennardJones(epsilon=EPS).forces(scan_state(2.6 * SIGMA)))[1, 0])

dead = ~alive0
dead_moved = float(np.abs(pos1[:, dead] - pos1[0:1, dead]).max()) if dead.any() else 0.0

np.savez_compressed(
    os.path.join(OUT, "reference.npz"),
    position=pos1.astype(np.float32),                 # [NSTEPS+1, CAP, 2]  index t = IC after t steps
    alive=alive,                                      # [NSTEPS+1, CAP]
    radius=radius.astype(np.float32),                 # [CAP]
    p0=p0.astype(np.float32),                         # [CAP,2] the IC
    force_ic=force_ic.astype(np.float32),             # [CAP,2] -grad U at the IC
    manual=manual.astype(np.float32),                 # hand-Euler trajectory (== position)
    seps_traj=seps_traj.astype(np.float32),           # [NSTEPS+1, npairs] per-pair separation
    pos_ss=pos_ss.astype(np.float32),                 # SoftSphere negative-control trajectory
    r_scan=r_scan, f_ref_scan=f_ref_scan, f_an_scan=f_an_scan,
    N=np.int32(N), CAP=np.int32(CAP), NSTEPS=np.int32(NSTEPS),
    dt=np.float32(DT), eps=np.float32(EPS), r0=np.float32(R0), sigma=np.float32(SIGMA),
    gamma=np.float32(GAMMA), kt=np.float32(KT),
)

summary = {
    "role": "oracle", "operator": "adhere/lennard_jones",
    "model": "BrownianDynamics(LennardJones(epsilon=1.0), gamma=1.0, kT=0.0)",
    "N": N, "CAP": CAP, "NSTEPS": NSTEPS, "dt": DT, "eps": EPS, "r0": R0, "sigma": SIGMA,
    "gamma": GAMMA, "kT": KT, "pair_centres_x": XC, "pair_seps_initial": SEPS,
    "x64_enabled": bool(jax.config.read("jax_enable_x64")), "dtype": str(force_ic.dtype),
    "deterministic_same_key": same_key, "deterministic_diff_key": diff_key,
    "euler_convention_matches_bit_for_bit": euler_matches, "euler_max_dev": euler_max_dev,
    "alive_fixed": alive_fixed, "any_nan": any_nan, "traj_min_separation_ever": traj_min_sep,
    "force_ic_max": float(np.abs(force_ic).max()),
    "pair_sep_final": seps_traj[-1].tolist(),         # tail pairs -> ~1.0 (contact); 2.60 pair frozen
    "max_live_cell_displacement": float(np.linalg.norm(pos1[-1, :N] - pos1[0, :N], axis=-1).max()),
    "negative_control_lj_vs_softsphere_over_sigma": lj_vs_ss,
    "negative_control_softsphere_max_disp": ss_max_disp,
    "scan_ref_vs_analytic_max_below_ron": scan_ref_vs_analytic_max,
    "force_at_contact": force_at_contact, "force_beyond_cutoff_2p6sigma": force_beyond_cutoff,
    "dead_slots_max_move": dead_moved,
}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

# --- ready-to-paste YAML `start:` block (byte-identical float32 IC for the Plexus spec) -------- #
print("\n# ---- paste into config/atlas_jax/lennard_jones.yaml sets.cell.start (LIVE cells only) ----")
print("    start:")
for x, y in P0_live:
    print(f"    - [{x:.6f}, {y:.6f}]")
print("# ---- end start block ----")

# --- one sanity figure: the scan (ref vs analytic) + the pair-separation trajectories --------- #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(r_scan, np.where(below_ron, f_an_scan, np.nan), "k-", lw=2, label="analytic 12eps(x^12-x^6)/r")
ax1.plot(r_scan, f_ref_scan, "r.", ms=7, label="jax-morph LennardJones.forces")
ax1.axvline(SIGMA, ls="--", c="gray", label="contact sigma=1.0")
ax1.axvline(2.5 * SIGMA, ls=":", c="green", label="cutoff 2.5 sigma")
ax1.axhline(0, ls="-", c="0.8", lw=0.8)
ax1.set_xlabel("separation r"), ax1.set_ylabel("radial force"), ax1.legend(fontsize=7)
ax1.set_title("scan: reference vs analytic r_min LJ (well AT contact)")
for k in range(len(XC)):
    ax2.plot(np.arange(NSTEPS + 1) * DT, seps_traj[:, k], label=f"init {SEPS[k]:.2f}")
ax2.axhline(SIGMA, ls="--", c="gray", label="contact")
ax2.set_xlabel("t"), ax2.set_ylabel("pair separation"), ax2.legend(fontsize=7)
ax2.set_title("dumbbells relaxing to contact (kT=0)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "reference.png"), dpi=120)
print("wrote reference.npz, summary.json, reference.png")
