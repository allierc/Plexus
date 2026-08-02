"""Oracle for adhere/harmonic -- the jax-morph Harmonic finite-range shifted-spring potential,
ISOLATED and put under the reference's OWN overdamped integrator.

Harmonic is a POTENTIAL (an energy): it writes no state, its whole contract is the force
F = -grad U for the shifted harmonic well U(r) = 0.5*k*[(r-sigma)^2 - (r_c-sigma)^2] on r < r_c,
sigma = r_i + r_j, r_c = r_cutoff_frac*sigma. Its radial force is f(r) = k*(sigma - r), truncated
at r_c: REPULSIVE for r < sigma (excluded volume) and ADHESIVE for sigma < r < r_c (the down-shift
makes the well minimum negative at contact). That adhesive tail is the ONLY thing that
distinguishes Harmonic from its already-validated purely-repulsive siblings SoftSphere/Hertzian, so
this oracle is built to EXERCISE it.

To diff it we isolate it from every other mechanism (no growth, no division -> fixed cell count,
fixed radii, fixed slots) and drive it with jax-morph's own overdamped Langevin step at ZERO
temperature,

    BrownianDynamics(Harmonic(k, r_cutoff_frac), gamma=1, kT=0):  dx = dt*forces/gamma = dt*(-grad U),

which is exactly the Plexus engine's overdamped-Euler integration of the operator's emitted
velocity (pos += dt * mobility * F, mobility = 1/gamma = 1). So the position trajectory is a PURE
function of the force law (repulsive core + adhesive tail + hard C0 cutoff), and a per-cell diff
isolates it.

Initial condition: a fixed 19-cell Vogel sunflower cluster centred at (20,20), uniform radius 0.5
(sigma = 1.0), scale 0.5 so nearest neighbours OVERLAP (r < sigma, repulsion) while the many
second/third neighbours sit in the adhesive band sigma < r < r_c = 2.5 -- BOTH regimes present at
t=0. Plus 5 DEAD padding slots parked at the origin (dead-pair mask + sigma=0 safe_divide).
Coordinates are rounded to 6 decimals and PRINTED as a ready-to-paste YAML `start:` block, so the
Plexus spec and this oracle share a BYTE-IDENTICAL float32 IC.

Guards before anything is recorded (the oracle contract):
  1. determinism at a fixed key,
  2. determinism ACROSS two different keys (kT=0 truly removes the noise),
  3. jxm.simulate BIT-IDENTICAL to a hand-rolled Euler over Harmonic().forces() (pins dx=dt*forces),
  4. a 2-cell radial scan of Harmonic.forces vs the analytic f(r)=k(sigma-r)|_{r<r_c} (proves the
     reference implements the law, adhesion negative in (sigma,r_c), exactly zero beyond r_c),
  5. NEGATIVE CONTROL: with k=eps=1, sigma=1 the SoftSphere/Harmonic repulsive cores are IDENTICAL
     (f=1-r for r<sigma), so on the SAME IC they differ ONLY in the adhesive tail -- the
     single-step force gap and the SoftSphere-driven trajectory both isolate the adhesion.

Writes reference.npz + summary.json + reference.png into OUT.
"""
import os
import json

import numpy as np
import jax
import jax.numpy as jnp
import jax_morph as jxm
from jax_morph.physics import BrownianDynamics, Harmonic, SoftSphere

OUT = os.environ["OUT"]

# --- parameters (must match config/atlas_jax/harmonic.yaml exactly) ------------------------------ #
N = 19                       # live cells
CAP = 24                     # capacity: 19 live + 5 dead padding slots (dead-pair mask + sigma=0)
K = 1.0                      # spring stiffness (Harmonic k) -- also the SoftSphere epsilon (matched core)
RCF = 2.5                    # r_cutoff_frac -> r_c = 2.5*sigma (the source default; long adhesive range)
R0 = 0.5                     # uniform cell radius -> sigma = r_i + r_j = 1.0
DT = 0.03                    # macro-step; small so the stiff dense adhesive network relaxes smoothly
NSTEPS = 160                 # 160 overdamped steps (T = 4.8) -- reaches the adhesive equilibrium
GAMMA = 1.0                  # drag; mobility = 1/gamma = 1 matches the Plexus operator default
KT = 0.0                     # ZERO temperature -> deterministic overdamped Euler (no noise)
CENTER = np.array([20.0, 20.0])
SCALE = 0.5                  # Vogel scale -> nearest-neighbour ~0.8 (overlap) + adhesive shells
SIGMA = 2.0 * R0             # 1.0
R_C = RCF * SIGMA            # 2.5


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

# --- IC regime census: the test MUST exercise BOTH the repulsive core and the adhesive tail --- #
d = np.linalg.norm(P0_live[:, None] - P0_live[None], axis=-1)
np.fill_diagonal(d, np.inf)
nn = d.min(1)
overlap_pairs = int(((d < SIGMA) & np.isfinite(d)).sum() // 2)             # r < sigma  (repulsion)
adhesive_pairs = int(((d >= SIGMA) & (d < R_C)).sum() // 2)                # sigma <= r < r_c (adhesion)
beyond_pairs = int(((d >= R_C) & np.isfinite(d)).sum() // 2)               # r >= r_c   (no force)
print(f"IC: N={N} live, min_nn={nn.min():.4f} median_nn={np.median(nn):.4f} "
      f"max_nn={nn.max():.4f}  overlapping(r<{SIGMA})={overlap_pairs}  "
      f"adhesive({SIGMA}<=r<{R_C})={adhesive_pairs}  beyond(r>={R_C})={beyond_pairs}", flush=True)
if overlap_pairs < 5 or adhesive_pairs < 20:
    raise SystemExit("IC does not exercise both regimes (need repulsive overlaps AND many adhesive "
                     f"pairs): overlap={overlap_pairs}, adhesive={adhesive_pairs}. Stop.")

# --- the models: Harmonic (under test) and SoftSphere (matched-core negative control) --------- #
model = jxm.Model([BrownianDynamics(Harmonic(k=K, r_cutoff_frac=RCF), n_space_dim=2,
                                    gamma=GAMMA, kT=KT)])
model_ss = jxm.Model([BrownianDynamics(SoftSphere(epsilon=K), n_space_dim=2,
                                       gamma=GAMMA, kT=KT)])  # adhesion OFF, same core (eps=k, sigma=1)


def seed_state(mdl):
    s = jxm.build_state_from_model(mdl).init_empty(capacity=CAP, n_space_dim=2, n_types=1)
    return s.update(
        alive=s.alive.at[:N].set(True),
        radius=s.radius.at[:N].set(R0),
        position=s.position.at[:CAP].set(jnp.asarray(p0, jnp.float32)),
        celltype=s.celltype.at[:N, 0].set(1.0),
    )


def run(mdl, key):
    return jxm.simulate(mdl, seed_state(mdl), n_steps=NSTEPS, dt=DT, key=key, history=True)


h1 = run(model, jax.random.PRNGKey(0))
h2 = run(model, jax.random.PRNGKey(0))            # same key
h3 = run(model, jax.random.PRNGKey(12345))        # DIFFERENT key: kT=0 must make it identical

pos1 = np.asarray(h1.position)                     # [NSTEPS+1, CAP, 2]
same_key = bool(np.array_equal(pos1, np.asarray(h2.position)))
diff_key = bool(np.array_equal(pos1, np.asarray(h3.position)))
if not (same_key and diff_key):
    raise SystemExit("Harmonic overdamped dynamics is not deterministic at kT=0 "
                     f"(same_key={same_key}, diff_key={diff_key}) -- a differential test against "
                     "it would measure the reference's own noise. Stop here.")

# --- self-check: jxm's BrownianDynamics(kT=0) == a hand-rolled Euler over Harmonic.forces() ---- #
pot = Harmonic(k=K, r_cutoff_frac=RCF)
st = seed_state(model)
manual = [np.asarray(st.position)]
for _ in range(NSTEPS):
    F = pot.forces(st)                             # -jax.grad(total_energy) -- the class under test
    st = st.update(position=st.position + DT * F)
    manual.append(np.asarray(st.position))
manual = np.stack(manual)                          # [NSTEPS+1, CAP, 2]
euler_matches = bool(np.array_equal(pos1, manual))
euler_max_dev = float(np.abs(pos1 - manual).max())
if not euler_matches:
    print(f"WARNING: jxm vs hand-Euler max|dev| = {euler_max_dev:.3e} (expected 0.0)", flush=True)

# force at the IC (the sharpest single-step probe of the raw force law, before any compounding)
force_ic = np.asarray(pot.forces(seed_state(model)))          # [CAP,2]

# --- 2-cell radial scan vs analytic f(r) = k*(sigma - r) for r < r_c, else 0 (oracle guard) ---- #
def scan_state(mdl, r):
    s = jxm.build_state_from_model(mdl).init_empty(capacity=2, n_space_dim=2, n_types=1)
    return s.update(alive=s.alive.at[:2].set(True), radius=s.radius.at[:2].set(R0),
                    position=s.position.at[:2].set(jnp.asarray([[0.0, 0.0], [float(r), 0.0]],
                                                               jnp.float32)),
                    celltype=s.celltype.at[:2, 0].set(1.0))


sr = np.linspace(0.3, 3.0, 55).astype(np.float32)   # spans repulsion (r<1), adhesion (1<r<2.5), off (>2.5)
f_ref_scan, f_an_scan = [], []
for r in sr:
    F = np.asarray(Harmonic(k=K, r_cutoff_frac=RCF).forces(scan_state(model, r)))
    f_ref_scan.append(float(F[1, 0]))               # radial force on cell 1: +apart / -together
    f_an_scan.append(K * (SIGMA - float(r)) if r < R_C else 0.0)
f_ref_scan = np.array(f_ref_scan); f_an_scan = np.array(f_an_scan)
scan_ref_vs_analytic_max = float(np.abs(f_ref_scan - f_an_scan).max())
scan_min_adhesion = float(f_ref_scan.min())         # most negative (deepest adhesive pull) < 0
scan_force_at_cutoff = float(np.abs(f_ref_scan[sr >= R_C]).max()) if (sr >= R_C).any() else 0.0

# --- NEGATIVE CONTROL: SoftSphere (adhesion OFF, matched core) on the SAME IC ----------------- #
F_ss_ic = np.asarray(SoftSphere(epsilon=K).forces(seed_state(model_ss)))          # [CAP,2]
nc_force_rel = float(np.abs(F_ss_ic - force_ic).max() / max(1e-12, np.abs(force_ic).max()))
h_ss = run(model_ss, jax.random.PRNGKey(0))
pos_ss = np.asarray(h_ss.position)                  # [NSTEPS+1, CAP, 2] purely-repulsive trajectory
live0 = alive0
nc_traj_Dpos = float((np.linalg.norm(pos_ss[:, live0] - pos1[:, live0], axis=-1) / SIGMA).max())

alive = np.asarray(h1.alive)                        # [NSTEPS+1, CAP]
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
gyr_ss = [gyration(pos_ss[t], alive[t]) for t in range(NSTEPS + 1)]
disp_live = float(np.linalg.norm(pos1[-1, :N] - pos1[0, :N], axis=-1).max())
dead_moved = float(np.abs(pos1[:, N:] - pos1[0:1, N:]).max())     # dead slots must never move

np.savez_compressed(
    os.path.join(OUT, "reference.npz"),
    position=pos1.astype(np.float32),      # [NSTEPS+1, CAP, 2] index t = IC after t steps
    alive=alive,                           # [NSTEPS+1, CAP]
    radius=radius.astype(np.float32),      # [CAP]
    p0=p0.astype(np.float32),              # [CAP,2] the IC
    force_ic=force_ic.astype(np.float32),  # [CAP,2] -grad U at the IC (Harmonic)
    force_ic_ss=F_ss_ic.astype(np.float32),  # [CAP,2] SoftSphere force at IC (negative control)
    position_ss=pos_ss.astype(np.float32),   # [NSTEPS+1, CAP, 2] SoftSphere trajectory (control)
    manual=manual.astype(np.float32),      # the hand-Euler trajectory (== position)
    sr=sr, f_ref_scan=f_ref_scan, f_an_scan=f_an_scan,
    N=np.int32(N), CAP=np.int32(CAP), NSTEPS=np.int32(NSTEPS),
    dt=np.float32(DT), k=np.float32(K), r0=np.float32(R0), sigma=np.float32(SIGMA),
    r_cutoff_frac=np.float32(RCF), r_c=np.float32(R_C),
    gamma=np.float32(GAMMA), kt=np.float32(KT),
)

summary = {
    "role": "oracle", "operator": "adhere/harmonic",
    "model": "BrownianDynamics(Harmonic(k=1.0, r_cutoff_frac=2.5), gamma=1.0, kT=0.0)",
    "N": N, "CAP": CAP, "NSTEPS": NSTEPS, "dt": DT, "k": K, "r0": R0, "sigma": SIGMA,
    "r_cutoff_frac": RCF, "r_c": R_C, "gamma": GAMMA, "kT": KT,
    "center": CENTER.tolist(), "scale": SCALE,
    "deterministic_same_key": same_key, "deterministic_diff_key": diff_key,
    "euler_convention_matches_bit_for_bit": euler_matches, "euler_max_dev": euler_max_dev,
    "alive_fixed": alive_fixed,
    "ic_min_nn": float(nn.min()), "ic_median_nn": float(np.median(nn)),
    "ic_overlapping_pairs": overlap_pairs, "ic_adhesive_pairs": adhesive_pairs,
    "ic_beyond_pairs": beyond_pairs,
    "force_ic_max": float(np.abs(force_ic).max()),
    "scan_ref_vs_analytic_max": scan_ref_vs_analytic_max,
    "scan_min_adhesion_force": scan_min_adhesion, "scan_force_at_cutoff": scan_force_at_cutoff,
    "negctrl_softsphere_vs_harmonic_force_rel": nc_force_rel,
    "negctrl_softsphere_traj_Dpos": nc_traj_Dpos,
    "gyration_first": gyr[0], "gyration_last": gyr[-1],
    "gyration_ss_last": gyr_ss[-1],
    "extent_first": ext[0], "extent_last": ext[-1],
    "max_live_cell_displacement": disp_live,
    "dead_slots_max_move": dead_moved,
}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

# --- the ready-to-paste YAML `start:` block (byte-identical float32 IC for the Plexus spec) --- #
print("\n# ---- paste into config/atlas_jax/harmonic.yaml sets.cell.start (LIVE cells only) ----")
print("    start:")
for x, y in P0_live:
    print(f"    - [{x:.6f}, {y:.6f}]")
print("# ---- end start block ----")

# --- one sanity figure: the 2-cell scan (ref vs analytic) + gyration (adhesion is bounded) ---- #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.axhline(0, ls=":", c="gray")
ax1.plot(sr, f_an_scan, "k-", lw=2, label="analytic k*(sigma-r) | r<r_c")
ax1.plot(sr, f_ref_scan, "r.", ms=7, label="jax-morph Harmonic.forces")
ax1.axvline(SIGMA, ls="--", c="green", label="contact sigma=1.0 (force=0)")
ax1.axvline(R_C, ls="--", c="purple", label="cutoff r_c=2.5 (force->0)")
ax1.set_xlabel("separation r"), ax1.set_ylabel("radial force on cell 1")
ax1.legend(fontsize=8), ax1.set_title("S: reference vs analytic (repel | adhere | off)")
tt = np.arange(NSTEPS + 1) * DT
ax2.plot(tt, gyr, "b-", label="Harmonic (adhesion ON)")
ax2.plot(tt, gyr_ss, "r--", label="SoftSphere (adhesion OFF, neg-ctrl)")
ax2.set_xlabel("t"), ax2.set_ylabel("gyration radius")
ax2.legend(fontsize=8), ax2.set_title("U: relaxation -- adhesion holds the cluster bounded")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "reference.png"), dpi=120)
print("wrote reference.npz, summary.json, reference.png")
