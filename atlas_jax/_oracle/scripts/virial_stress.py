"""Oracle for mechanosense / VirialStress -- the jax-morph per-cell Irving-Kirkwood virial
pressure, ISOLATED. Unlike a force law (soft_sphere/harmonic/hertzian), VirialStress MOVES
NOTHING: it is a pure quasistatic SENSOR that writes each live cell's virial pressure into the
transient ``stress`` field. So there is no trajectory to diff -- the observable IS the written
scalar, evaluated on ONE frozen configuration. That makes the differential test CONFOUND-FREE:
no integrator, no mechanics, nothing to mismatch; both sides just reduce the same pair law over
the same live-non-self neighbour map on the same positions.

We build one RICH configuration that exercises every branch of

    p_i = -(1 / (2 d V_i)) sum_{j != i, j alive} r_ij (dU/dr)(r_ij)

with the paper's Morse mechanics (epsilon=3.0, alpha=2.8, smooth cutoff 1.5*sigma..2.5*sigma;
uniform radius 0.5 so sigma = r_i + r_j = 1.0, well minimum at r = 1.0):

  * FIVE isolated pairs (centres 6 apart, > cutoff, so each cell feels ONLY its partner) at fixed
    separations r = 0.7 / 1.0 / 1.3 / 2.0 / 3.0 -- clean single-neighbour analytic points spanning
    COMPRESSION (r<1, p>0), the WELL MINIMUM (r=1, p~0), full TENSION (1<r<1.5, p<0), TAPERED
    tension inside the cutoff window (1.5<r<2.5, p<0), and BEYOND CUTOFF (r=3, p=0);
  * one dense 8-cell sunflower cluster -- the multi-neighbour reduction;
  * 4 DEAD padding slots (occ=0) -- the dead-source / dead-receiver masking and the V_i=0
    safe_divide.

Everything is float32 and the LIVE positions are printed as a ready-to-paste YAML ``start:`` block
so the Plexus spec shares a BYTE-IDENTICAL IC. We compute the reference stress two ways -- the
VirialStress STEP (state.stress) and PairwisePotential.virial_pressure directly -- and assert they
agree (the step is a thin delegator). We also assert key-independence (the step takes a PRNG key
and must ignore it) -- the sensor's analogue of the kT=0 determinism gate.

For breadth we ALSO compute the reference virial pressure on this SAME IC for the other four pair
laws (soft_sphere / hertzian / harmonic / lennard_jones, each at its default coupling), and on a
SECOND small UNEQUAL-RADII configuration for all five laws -- the arrays the Plexus-side supplement
cross-checks (the sigma=r_i+r_j and per-cell d-ball V_i paths the uniform-radius engine run cannot
reach). Writes reference.npz + summary.json into OUT.
"""
import os
import json

import numpy as np
import jax
import jax.numpy as jnp
import jax_morph as jxm
from jax_morph.physics import (VirialStress, Morse, SoftSphere, Hertzian, Harmonic, LennardJones)

OUT = os.environ["OUT"]

# --- parameters (must match config/atlas_jax/virial_stress.yaml exactly) ------------------------- #
R0 = 0.5                       # uniform cell radius -> sigma = r_i + r_j = 1.0
EPS = 3.0                      # Morse well depth (the paper / anchor mechanics)
ALPHA = 2.8                    # Morse well steepness (the paper / anchor mechanics)
SIGMA = 2.0 * R0
PAIR_SEPS = [0.7, 1.0, 1.3, 2.0, 3.0]   # compression / well-min / tension / tapered / beyond-cutoff
PAIR_X = [0.0, 6.0, 12.0, 18.0, 24.0]   # pair centres, 6 apart (> 2.5*sigma cutoff -> isolated)
CLUSTER_CENTER = np.array([12.0, 12.0])
CLUSTER_N = 8
CLUSTER_SCALE = 0.45
N_DEAD = 4
SHIFT = np.array([2.0, 2.0])            # keep every coordinate positive / inside the world box

# the 5 laws and their Plexus-matching defaults (mechanosense: morse eps=3.0, others 1.0)
LAWS = {
    "morse": Morse(epsilon=EPS, alpha=ALPHA),
    "soft_sphere": SoftSphere(epsilon=1.0),
    "hertzian": Hertzian(epsilon=1.0),
    "harmonic": Harmonic(k=1.0),
    "lennard_jones": LennardJones(epsilon=1.0),
}


def sunflower(n, scale, center):
    """Deterministic Vogel sunflower disk, rounded to 6 decimals (byte-identical float32 paste)."""
    ga = np.pi * (3.0 - np.sqrt(5.0))
    k = np.arange(n)
    r = scale * np.sqrt(k + 0.5)
    th = k * ga
    return np.round(np.stack([center[0] + r * np.cos(th), center[1] + r * np.sin(th)], axis=1), 6)


# --- build the primary (uniform-radius) IC --------------------------------------------------- #
live = []
regime = []                                            # a human label per live slot (for the summary)
for cx, sep in zip(PAIR_X, PAIR_SEPS):
    live.append([cx, -sep / 2.0]); live.append([cx, +sep / 2.0])
    tag = ("compression" if sep < SIGMA else "well_min" if sep == SIGMA else
           "tension_full" if sep < 1.5 * SIGMA else "tension_tapered" if sep < 2.5 * SIGMA else "beyond_cutoff")
    regime += [f"pair_r{sep}_{tag}"] * 2
cluster = sunflower(CLUSTER_N, CLUSTER_SCALE, CLUSTER_CENTER)
live += cluster.tolist()
regime += ["cluster"] * CLUSTER_N

P_live = (np.round(np.array(live, np.float64), 6) + SHIFT)     # [N,2]
N = P_live.shape[0]
CAP = N + N_DEAD
p0 = np.zeros((CAP, 2), np.float64)
p0[:N] = P_live
p0[N:] = SHIFT + np.array([30.0, 30.0])                # dead slots parked far away (masked anyway)
radius = np.zeros((CAP,), np.float64); radius[:N] = R0
alive0 = np.zeros((CAP,), bool); alive0[:N] = True

# report the IC geometry so the regimes are visibly present
d = np.linalg.norm(P_live[:, None] - P_live[None], axis=-1); np.fill_diagonal(d, np.inf)
print(f"IC: N={N} live, CAP={CAP}, min_nn={d.min():.4f}, "
      f"interacting_pairs(r<2.5*sigma={2.5*SIGMA})={int(((d < 2.5*SIGMA)).sum()//2)}", flush=True)


def make_state(model, p, rad, alv):
    s = jxm.build_state_from_model(model).init_empty(capacity=p.shape[0], n_space_dim=2, n_types=1)
    nlive = int(alv.sum())
    return s.update(
        alive=s.alive.at[:].set(jnp.asarray(alv)),
        radius=s.radius.at[:].set(jnp.asarray(rad, jnp.float32)),
        position=s.position.at[:].set(jnp.asarray(p, jnp.float32)),
        celltype=s.celltype.at[:nlive, 0].set(1.0),
    )


# --- reference stress via the VirialStress STEP, cross-checked against virial_pressure -------- #
step = VirialStress(Morse(epsilon=EPS, alpha=ALPHA))
model = jxm.Model([step])
st = make_state(model, p0, radius, alive0)

stress_step_k0 = np.asarray(step(st, dt=1.0, key=jax.random.PRNGKey(0)).stress)
stress_step_k1 = np.asarray(step(st, dt=1.0, key=jax.random.PRNGKey(12345)).stress)
stress_direct = np.asarray(Morse(epsilon=EPS, alpha=ALPHA).virial_pressure(st))

key_independent = bool(np.array_equal(stress_step_k0, stress_step_k1))
step_eq_direct = bool(np.array_equal(stress_step_k0, stress_direct))
if not key_independent:
    raise SystemExit("VirialStress depends on the PRNG key -- a sensor must not. Stop here.")
if not step_eq_direct:
    raise SystemExit("VirialStress step != PairwisePotential.virial_pressure -- delegation broken. Stop.")
stress_morse = stress_step_k0

# analytic single-neighbour Morse pressure for the in-window pairs (r < 1.5*sigma), by hand
def morse_dudr(r, s=SIGMA, eps=EPS, a=ALPHA):
    e = 1.0 - np.exp(-a * (r - s))
    return eps * 2.0 * e * (a * np.exp(-a * (r - s)))     # S(r)=1 for r < 1.5*sigma
def morse_p_analytic(r, r_cell=R0, d_dim=2):
    V = np.pi * r_cell ** 2
    return -(1.0 / (2.0 * d_dim * V)) * r * morse_dudr(r)
analytic = {sep: float(morse_p_analytic(sep)) for sep in PAIR_SEPS if sep < 1.5 * SIGMA}
# match reference against the hand value on the first cell of each in-window pair
an_max_dev = 0.0
for i, (cx, sep) in enumerate(zip(PAIR_X, PAIR_SEPS)):
    if sep < 1.5 * SIGMA:
        an_max_dev = max(an_max_dev, abs(float(stress_morse[2 * i]) - analytic[sep]))
print(f"reference vs hand-analytic (in-window Morse pairs) max|dev| = {an_max_dev:.3e}", flush=True)

# --- the other four laws on the SAME uniform-radius IC (breadth) ------------------------------ #
stress_laws = {}
for name, pot in LAWS.items():
    stress_laws[name] = np.asarray(pot.virial_pressure(make_state(jxm.Model([VirialStress(pot)]), p0, radius, alive0)))

# --- a SECOND, UNEQUAL-RADII config (sigma=r_i+r_j and per-cell V_i paths) -------------------- #
vr_radii_pairs = [(0.5, 0.3, 0.6), (0.7, 0.4, 1.3), (0.6, 0.5, 0.9)]   # (r_a, r_b, separation)
vr_live, vr_rad = [], []
for k, (ra, rb, sep) in enumerate(vr_radii_pairs):
    vr_live.append([10.0 * k, 0.0]); vr_rad.append(ra)
    vr_live.append([10.0 * k, sep]);  vr_rad.append(rb)
vr_N = len(vr_live); vr_CAP = vr_N + 2
vr_p0 = np.zeros((vr_CAP, 2), np.float64); vr_p0[:vr_N] = np.round(vr_live, 6)
vr_p0[vr_N:] = [99.0, 99.0]
vr_radius = np.zeros((vr_CAP,), np.float64); vr_radius[:vr_N] = vr_rad
vr_alive = np.zeros((vr_CAP,), bool); vr_alive[:vr_N] = True
vr_stress_laws = {}
for name, pot in LAWS.items():
    vr_stress_laws[name] = np.asarray(pot.virial_pressure(make_state(jxm.Model([VirialStress(pot)]), vr_p0, vr_radius, vr_alive)))

# --- save --------------------------------------------------------------------------------------- #
np.savez_compressed(
    os.path.join(OUT, "reference.npz"),
    p0=p0.astype(np.float32), radius=radius.astype(np.float32), alive=alive0,
    stress_morse=stress_morse.astype(np.float32),
    **{f"stress_{k}": v.astype(np.float32) for k, v in stress_laws.items() if k != "morse"},
    N=np.int32(N), CAP=np.int32(CAP), r0=np.float32(R0), eps=np.float32(EPS),
    alpha=np.float32(ALPHA), sigma=np.float32(SIGMA),
    pair_seps=np.asarray(PAIR_SEPS, np.float32),
    vr_p0=vr_p0.astype(np.float32), vr_radius=vr_radius.astype(np.float32), vr_alive=vr_alive,
    vr_N=np.int32(vr_N), vr_CAP=np.int32(vr_CAP),
    **{f"vr_stress_{k}": v.astype(np.float32) for k, v in vr_stress_laws.items()},
)

live_stress = stress_morse[:N]
summary = {
    "role": "oracle", "operator": "mechanosense / VirialStress",
    "model": f"VirialStress(Morse(epsilon={EPS}, alpha={ALPHA}))",
    "N": int(N), "CAP": int(CAP), "n_dead": int(N_DEAD), "r0": R0, "sigma": SIGMA,
    "eps": EPS, "alpha": ALPHA,
    "key_independent": key_independent, "step_eq_virial_pressure": step_eq_direct,
    "ref_vs_analytic_morse_max_dev": an_max_dev,
    "stress_morse_live_min": float(live_stress.min()),
    "stress_morse_live_max": float(live_stress.max()),
    "stress_morse_live_absmax": float(np.abs(live_stress).max()),
    "stress_morse_n_positive": int((live_stress > 1e-6).sum()),
    "stress_morse_n_negative": int((live_stress < -1e-6).sum()),
    "stress_morse_n_zero": int((np.abs(live_stress) <= 1e-6).sum()),
    "dead_slots_stress_absmax": float(np.abs(stress_morse[N:]).max()),
    "pair_stress": {f"r={s}": [float(stress_morse[2 * i]), float(stress_morse[2 * i + 1])]
                    for i, s in enumerate(PAIR_SEPS)},
    "analytic_in_window": {f"r={s}": v for s, v in analytic.items()},
    "other_laws_live_absmax": {k: float(np.abs(v[:N]).max()) for k, v in stress_laws.items()},
    "vr_N": int(vr_N),
    "vr_stress_morse": [float(x) for x in vr_stress_laws["morse"][:vr_N]],
}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

# --- the ready-to-paste YAML `start:` block (byte-identical float32 IC for the Plexus spec) ---- #
print("\n# ---- paste into config/atlas_jax/virial_stress.yaml sets.cell.start (LIVE cells only) ----")
print("    start:")
for x, y in P_live:
    print(f"    - [{x:.6f}, {y:.6f}]")
print("# ---- end start block ----")
print("wrote reference.npz, summary.json")
