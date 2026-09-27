# Cilia campaign — 100 iterations

Record: `builder/exp_01_memiopsis`, watched at http://127.0.0.1:8799/watch.
Every run goes through `tools/gui_drive.py opencycle`. No exceptions.

## Where this starts

The working point, reached 2026-09-22 and not to be re-derived:

| thing | value | where it came from |
|---|---|---|
| rod | OVERDAMPED, `emit: velocity` on rod_elastic / rod_base / rod_motor | step 0343 — a real cilium is overdamped by 6.1e6; an inertial rod rings and spins |
| nodes | 8, segment 2.86 um | the limit `dt < 2/(16 k_bend)` goes as h^4; 8 puts it at 3.0e-5 s |
| `k_bend` | 4150 /s | `B/(zeta_perp h^4)`, B = 580 pN um^2 (Sartori 2016), zeta_perp from Gray & Hancock |
| `k_stretch` | 3.0e4 /s | binding CFL, 6.7x margin at dt 1e-5 |
| clamp / pin | 64.4 / 100 | rates are the SQUARE of these; clamp^2 matched to k_bend |
| motor | 0.20 rad/um, 10 Hz, wavelength = L | Sartori's C0 0.232 rad/um; Poon 2025 beat 10 +- 2 Hz |
| `zeta` | 4.794e4 sim-mass/s | `zeta_perp h` / (1 sim mass = 1.25e-13 kg) |
| `force_nN` | 6.25e-9 | derived from the water's own particle mass |
| coupling | **0.1** | 2,357 nm water displacement, beat intact |
| dt | 1e-5, 15000 frames = 0.15 s | 1.5 beats |

Baselines to compare against, one for one:
- **0347** `cil_s29_overdamped2` — rod in vacuum. base 28.31, tip 35.88, t/b 1.27, 10.00 Hz, tip 250 um/s.
- **0356** `cil_t00_baseline` — same rod + water, coupling 0. Reproduces 0347; water untouched (0.0 nm).
- **0362/0367** `cil_t01_c0p1` / `c0p01` — coupling on. 2,357 nm and 927 nm displacement.

## Gates — every run, before any claim

1. `momentum residual` near 1e-08 AND thousands of fluid particles touched.
2. rod INSIDE the box (`max|pos| < 1.05`).
3. beat matches the baseline: base 28.3 +- 1 deg, frequency 10.00 Hz.
4. worst segment strain < 10%.
5. arc length within 2% of 20.00 um.

A run failing any of these is a failed run, and that is the headline, not a caveat.
Look at a still AND read the caption AND print the invariants before writing a word.

## Known trap, not yet fixed

The rod is seeded STRAIGHT while the motor commands full amplitude at frame 1: startup velocity
135 world/s against a steady 2.5, 54x. `zeta` turns that into a 6.5e6 kick. This is why coupling
1.0 diverged. **Fix before P4**: ramp `rod_motor` amplitude from 0 over the first beat.

## Two rig defects found and fixed 2026-09-23 (runs 0449-0455, all void)

Both were invisible in a 1.5-beat window and in a single cilium on a line, which is everything
before P3. Both were measured on the coupling-0 CONTROL 0451, which flipped in beat 3 with no
water forces at all -- so the beat-2 flip of 0449 was never the water.

1. **`anchor: cell` pulled every base to the cell's centre.** The pin target was the mean of the
   body points, and a point cell IS its centre: five bases seeded 2 um from it were at 0.03 um
   within 0.01 s (pin rate 1e4/s). Every multi-cilia picture so far had the filaments fanning out
   of one point inside the sphere. Now the base keeps the offset it was laid at
   (`rod_base`, `self._off`). `cilia_row` column `base_r` / gate OFF-CELL (10 % of the radius).
2. **The base clamp held the direction only WITHIN the stroke plane.** It was a scalar torque
   about the beat normal; tilt out of the plane was a zero mode, and `rod_motor`, itself a couple
   about that normal, loses its arm as the segment lines up with it. Every failing rod left its
   plane at azimuth -90 deg, tilted 0 -> 78 deg over 0.19-0.32 s and FROZE there with the motor
   still commanding a full beat. The clamp is now a vector torque
   `-(clamp^2) theta khat - 2 zeta_c clamp omega` (`rod_base`), identical to the old law for
   in-plane tilt -- `tests/test_rod_base.py` states both. `cilia_row` column `oop` / gate OOP
   (15 deg; the vacuum working point 0347 reads 0.0).

The ramp (0453/0455) delayed the flip by exactly one beat and changed nothing else -- the first
sign it was not the startup kick. `cilia_row` also read every rod against one global frame;
each rod is now read in its own (`rod_layout`, shared with `rod_seed`).

3. **`rod_motor`'s couple form is torque-free only for a rod IN its plane** (found on 0457, the
   control re-run with 1+2 fixed: no flip, bases on the sphere, but every rod settled 19 deg out
   of its plane at the base, 57 at the tip, beating inside that tilt). Running the operator on
   0457's recorded shapes: net force 1e-16, net torque 0.000 of |F| L while planar, 0.15-0.26
   once tilted, same sign on every rod, about the in-plane axis -- it tilts the rod further, and
   only the clamp bounds it. The motor is now the exact gradient of `E = sum M_i theta_i`, theta
   the bend about the beat normal CARRIED from the basal body along the filament by parallel
   transport (twist-free frame, Bergou 2008). Identical to the couple form for a planar rod; zero
   torque about every axis perpendicular to the rest direction (4e-16 vs 6e-4..1.2e-2); the twist
   about the rest direction is the basal body's, and stays. `tests/test_rod_motor.py`. Cost: the
   motor is autograd, ~7 ms/frame on top of the water's 7 (measured on cuda:1, five rods) --
   perf item, not a physics one. The dead `frame:` knob of `rod_motor` was removed.

---

## P1 — patterns of oscillation (iterations 1-20)

One knob at a time off `cil_t01_c0p1`. Measure amplitude along the rod, tip/base, frequency,
and water displacement for each.

1-4    `wavelength` 0.2, 0.4, 0.8, 1.6 world (L/2, L, 2L, 4L). Short = more waves on the filament.
5-8    `omega` 2pi x {5, 10, 20, 40} Hz. Poon: 7-16 Hz resting, 18.6 with serotonin.
9-12   `amplitude` 5, 10, 15, 20 /world (0.1-0.4 rad/um). Above 25 it coils — measured at 0331.
13-15  `s0` 0.0, 0.1, 0.2 — silence the motor near the base, which real flagella roughly do.
16-18  standing vs travelling: set `wavelength` huge so phase is uniform.
       ~~THE CONTROL THAT MATTERS: a standing wave is reciprocal and must pump nothing.~~
       **WRONG, and step 0416 shows why.** A uniform COMMAND on a flexible filament still gives a
       propagating SHAPE: measured lag 7, 17, 32, 53, 78, 101 degrees from base to tip. Bending
       against drag introduces the lag (Machin 1958), so a flexible thing driven reciprocally does
       not MOVE reciprocally -- which is exactly why sperm swim. It pumped the most of any
       pattern, and that is correct rather than a failure.
       THE REAL RECIPROCITY CONTROL is a RIGID filament (k_bend enormous, no lag possible), or
       drag anisotropy 1. Run the rigid one before trusting any propulsion number.
19-20  `ratio` on the anisotropy once drag carries it — 1.0 (no anisotropy) vs 2.0.
       `ratio: 1` is a paddle that cannot swim however it waves. Second control.

Deliverable: one builder figure, amplitude envelope vs arclength for the whole family, and a
table of water displacement per pattern. The two controls (standing, ratio 1) must give ~0.

## P2 — anchor the cilium to a cell (21-35)

21-24  a `cell` set: 1 element, a sphere of radius 2 um at (0.5, 0.5, 0.28), rendered as a sphere
       (`plotting` needs a second subject; check `render_vtk`/`live_movie` for a sphere glyph).
25-28  `rod_base [anchor: cell]` instead of `fixed`. The operator ALREADY supports this — it
       finds the `n_body` nearest cell points and returns the reaction as a delta on that set.
       With the cell held still this must reproduce 0362 exactly. That is the gate.
29-32  cell held by a stiff pin; check the reaction the rod applies to it is equal and opposite.
33-35  render: cell sphere + filament + water slice, one picture that reads.

Deliverable: 0362's beat reproduced with the base on a cell rather than on nothing.
DONE at step 0411 (28.31 / 37.64 against 28.30 / 37.60).

## P3 — synchrony must be EARNED, not granted (36-60)

The biologically honest programme, and it has no phase operator in it on purpose. Real cilia are
not driven synchronous by any signal: they start wherever they start and organise through coupling
with their neighbours -- hydrodynamic (Taylor 1951; Brumley et al. 2012 measured it in *Volvox*)
and, at the real 0.57 um spacing, steric (Poon 2025, on this very animal: frequency is a property
of the CELL, uniform within one and discontinuous across cells; phase is not commanded and forms a
dexioplectic wave of wavelength ~L/2). Kuramoto is the phenomenological reduction of that coupling.
An operator that imposed it would put the answer in by hand and produce a synchronous movie with no
finding in it.

So: **frequency set per cell; each cilium started at a random phase; no phase operator at all;
measure whether the fluid coupling we already have synchronises them.** The Kuramoto order
parameter `r = |mean_k exp(i theta_k)|` is the thing to MEASURE, never to impose. Kuramoto itself
is kept as the BASELINE (Cedric): `tools/cilia_sync.py` fits the K that best matches each run's
emergent `r(t)` and overlays it dashed -- the fitted K is comparable to Brumley 2012, and a
curve Kuramoto cannot fit is a finding about the coupling's form. Reduction in the analysis,
never in the registry. Two outcomes,
both informative: `r -> 1` means hydrodynamic coupling suffices at this spacing; `r` stays low
means it does not -- which is Poon's claim, and would say steric contact is needed.

36-38  cilia ON A SPHERE, not on a line. `rod_seed sphere_radius`/`cap_deg`: base is the cell's
       centre, each rod on the surface along the outward normal, on a ring at `cap_deg`. DONE,
       unit-tested (radius exact, direction.normal 1.000, beat axis re-projected, anchors on
       the surface). Cell drawn at its WORLD radius by `plotting.spheres`.
39-40  `p3s_n5_sync` -- five cilia from one clock. Synchronous by construction: the REFERENCE,
       not a result. `p3s_n5_jitter` -- same, `rod_motor phase_jitter: true`, random start.
41-45  measure `r(t)` for both (`tools/cilia_sync.py`): instantaneous phase per rod from the
       base-joint angle by Hilbert transform, then `r`. sync must sit at ~1 throughout; jitter
       starts near `1/sqrt(5)` and either climbs or does not.
46-50  THE CONTROL: `p3s_n5_jitter_c0`, identical with `coupling: 0`. Phases MUST NOT converge.
       If `r` climbs with no fluid, something other than the fluid is coupling them -- a shared
       buffer, the pin, the renderer -- and every synchrony claim is void until it is found.
51     THE GATE: `p3s_n5_jitter_c1` -- coupling 1.0 with `rod_motor ramp 0.1`. This is the first
       time the full physical exchange is reached. If it holds (rod in box, strain < 10%), every
       later rung runs at 1.0; if it diverges the queue stops.
       ~~coupling sweep 0.01, 0.1, 1.0~~ DROPPED: `coupling` has meaning at exactly two values,
       1 (the derived zeta) and 0 (the null). Anything between is a fudge, and sweeping a fudge
       is a phenomenological reduction. The 0.1 used in 0435 and the first jitter run was a
       convenience from before `ramp` existed; those two stay as a record, not as results.
52-55  the AMPLITUDE axis instead, which is physical (Cedric): the P1 patterns of 0389 / 0391 /
       0393 (10 / 20 / 40 Hz: growing, flat, decaying wave; increasing water motion) and 0395
       (0.10 rad/um, the gentlest), each on the sphere with the same random phases at coupling
       1.0. Does synchrony depend on how hard the cilia stir the water?
56-58  spacing: `cap_deg` 15, 30, 45 puts the bases 0.6, 1.2, 1.7 um apart on a 2 um cell, at
       coupling 1.0. Coupling should strengthen as they close. Real spacing 0.57 um is below the
       1.56 um grid cell.
59-60  two cells at different frequencies (9 and 11 Hz): the discontinuity across cell
       boundaries Poon measured. Frequency is a CELL parameter, so this is one `omega` per set.

Deliverable: `r(t)` for sync / jitter / jitter-coupling-0 on one figure, then `r` vs frequency,
amplitude and spacing -- all at coupling 1.0.
The claim under test: does the fluid alone synchronise five cilia 1.2 um apart?

### P3 finding, 2026-09-23 (runs 0461 control, 0464 s1, 0466 s1r, 0468 s2; figures 0463, 0469)

Five cilia on a 2 um cell now hold five beats as five copies of the working point: base 28 deg
peak to peak on every rod in every beat, tip 36 (vacuum) / 37 (coupling 0.1), 10.1 Hz, out-of-
plane tilt 0.0, bases 2.00 um from the centre, gate PASS on all four. Water at 0.1: 63-65 um/s
at the 99.9th percentile, 840-1,070 nm net drift near the tuft over five beats. The ramped runs
reach full amplitude in beat 2 and are otherwise identical to the unramped one.

**`r(t)` is flat and IDENTICAL for coupling 0 and 0.1: seed 1 reads 0.053 after the first beat
and 0.054 at the end, on the control, on s1 and (0.055/0.056) on s1r; seed 2 reads 0.747 ->
0.743 -- a different constant, set by its draw, equally flat.** Not "no synchrony at this
coupling" -- no synchrony is POSSIBLE on this rig, and
that is the finding. `rod_motor` prescribes each rod's phase as a function of time
(`omega t + phi_r`); the fluid can shift the LAG between command and shape by a fixed amount but
can never accumulate a phase change, so `r` is set by the seed and stays there whatever the
coupling. The docstring said the wave was prescribed and a stated limitation; P3 is where the
limitation binds. The Kuramoto baseline fits K = 0 by construction.

**What P3 needs before it can be asked again -- a decision, not tonight's:** a drive whose phase
is a STATE of the filament rather than a clock, i.e. a self-sustained oscillator that the fluid
can entrain. The mechanistic candidates (none is a reduction; each is a model of dynein
regulation): (a) curvature control, Sartori et al. 2016 -- the motor moment responds to
curvature and its rate, `M_i = chi kappa_i` with complex chi (their fit: purely imaginary,
beta'' = -6.5 nN, R^2 95 percent), which oscillates on its own above a threshold; (b) sliding
control, Camalet & Julicher 2000 -- moment responds to inter-doublet sliding; (c) the geometric
clutch, Lindemann. (a) is the one measured on this animal's relative and the one the motor's
docstring already names; it replaces the sine, it does not sit beside it. Its own gates before
any synchrony claim: a single cilium must beat at ~10 Hz and ~30 deg with NO clock in the spec,
and the beat must survive coupling 0 -> 0.1 unchanged in frequency.

Until then the rig is a validated five-cilia-on-a-cell platform: P4 (release the cell) and P5
(propulsion) do not need a free phase and can proceed on the prescribed wave.

### P3b free-phase drive, small steps from the working point (2026-09-23)

Brokaw 1972 read in full (`exp_02_bacterium/papers/Brokaw_1972_BiophysJ_12_564.pdf`). His model is
NOT a threshold relay (my first guess, `switching`, removed) but curvature-controlled SLIDING: the
active moment at a joint is set INSTANTANEOUSLY by the distal shear (integral of curvature from the
joint to the free tip); it oscillates from the spatial-integral phase shift, no clock, no delay,
and bends run BASE to tip. `rod_motor [control: curvature]`, gain = m0. Amplitude capped by a tanh
saturation (his cubic-elastic / two-state role); wavelength set by internal viscous bending
resistance C_B, without which the wavelength -> 0 (his Fig 7).

- **0476 `p3b_curv01`** -- the 0347 vacuum working point, ONE change: motor prescribed -> curvature,
  gain 10. It self-oscillates and stays intact (base 2.1 deg, 172 Hz, arc 100.1%, strain 0.35%, in
  box), but tiny and fast: no wavelength stabiliser yet, so it sits at the shortest wavelength, the
  still is a near-straight rod. Exactly the bench prediction and Brokaw's no-C_B case. Gate PASS
  (no commanded freq to fail).
- **DIAGNOSIS of 0476 (per-joint, no new run):** joint bends alternate sign
  [-2.4,+2.1,-0.6,-0.4,+2.7,-3.0] -- adjacent joints beat in antiphase, the shortest wavelength,
  so no coherent bend forms and the tip stays at 2 deg. Brokaw's collapse without C_B.
- **The 8-node rod cannot host a self-organised Brokaw beat -- shown three ways, then stopped
  (Cedric: small sure steps, no rabbit hole):** (1) `zeta_bend` (his C_B) NaNs even at 0.005 -- the
  overdamped emit:velocity dashpot feeds the curvature of the stored velocity back into the emitted
  velocity, a x16 positive feedback; (2) more nodes hits dt < 2/(16 k_bend) with
  k_bend = B/(zeta h^4), so dt goes as nodes^-4 and 16/24 nodes NaN at 1e-5, cost ~nodes^5; (3)
  smoothing the active moment to kill the zigzag (finite spatial extent of dynein) removes ALL
  growth even at gain 40 -- the coherent long-wavelength mode is simply NOT unstable at 8 nodes,
  which is exactly why Brokaw needed >=15-25 segments. The `smooth` knob was tried and reverted
  (subtract, don't add). CONCLUSION: the free-phase beat is blocked on RESOLUTION, and the
  resolution is capped by the MPM cell that forced 8 nodes. Not a failure -- the mechanism is right
  (0476 oscillates base->tip); the mesh is too coarse. FORK for Cedric: keep the prescribed wave for
  P4/P5 (unblocked), or make a deliberate representation change (finer grid + more nodes, or a
  sub-grid drag so a thin rod resolves) before returning to free-phase.

## P4 — let the cell go (61-80)

### P4 anisotropic drag = the thrust the rig lacked (runs 0478/0480, 2026-09-23)

The P4 finding (0473) was that a released cell does not swim because the overdamped rod's mobility
was ISOTROPIC -- one zeta on every rate -- so a periodic beat pumps no net fluid. The fix is
Gray-Hancock anisotropy: a slender filament resists NORMAL motion ~2x harder than TANGENTIAL
(ratio = zeta_perp/zeta_par ~ 2), and the two half-strokes then push unequal fluid. The `drag`
operator already had a `ratio` param used only in its inertial `along: chain` branch; the OVERDAMPED
branch ignored it. Now the overdamped branch reduces the tangential component of the rod->water
reaction by `ratio`, using the local tangent off the segment table, and ONLY when `along: chain` is
declared -- so every existing overdamped spec (no flag) is byte-identical, and `ratio: 1` with the
flag is the paddle control. Momentum is conserved by construction (the reaction is re-oriented, its
equal-and-opposite still lands on the same water).

Fixed-base test on the 0362 rig (cil_t01_c0p1, coupling 0.1, 1.5 beats):
- **0478 ratio 1 (isotropic control):** reproduces 0362 -- base 28.28, tip 37.29, t/b 1.32.
- **0480 ratio 2 (anisotropic):** beat identical (28.30 / 37.12), but net water transport near the
  cilium 647 -> 1152 nm (+78%) and its DIRECTION turned from sideways [-0.8,0,-0.6] to along the
  cilium axis [-0.37,0,+0.93]. No runaway (rod |pos| 0.70, water COM drift 77 nm). The thrust
  appears exactly where it was missing. NEXT: ratio 2 on the free cell (0473 rig) -> run 0481+.

### P4 preliminary, 2026-09-23 (Cedric: "unfix the cell in 0468")

Run 0471 (`p4_free_s2`: 0468's spec, cell released, `rod_base zeta_node/zeta_body` = 4.794e4 /
3.016e5 so the pin reaction becomes a Stokes velocity of a 2 um sphere, plus a `drag` on the
cell for advection + its reaction on the water): **the cell flew along its cilia at 1.7 mm/s,
735 um in 0.5 s, out of the box with the rods in tow -- a rod_base bug, found by reading the
operator.** The body was handed `-(a0 - a1)`: the pin reaction MINUS the clamp arm force, i.e.
+a1 on top; the clamp couple is already internal (+a1 node 1, -a1 node 0), so rod + cell carried
a net force a1 ~ clamp^2 theta |e| ~ 60 world/s x 0.159 = 9 world/s = 470 um/s, the order
measured. Pinned, the cell absorbed it unseen (0461-0468, all still valid). Fixed: the body takes
`-a0` only; `tests/test_rod_base.py` asserts rod + cell force-free with the clamp active. Re-run
queued as the next record. Stated expectation: isotropic node mobility -> zero mean base force
over a beat -> only the recoil through the induced flow, ~0.2 um/s, ~100 nm in five beats.
`cilia_row` now reports `cell um` (net displacement over whole beats) and `cos` (against the cap
axis; -1 is a swimmer, +1 is being carried the way the cilia point).

### P4 ANSWERED 2026-09-23: the cell swims, and only with anisotropic drag (figure 0488).

The 0473 finding was right -- an isotropic overdamped rod pumps no net fluid over a periodic beat.
The fix is Gray-Hancock anisotropy, wired into `motion_ops.drag`'s overdamped branch and gated by
`along: chain` (every earlier isotropic run unchanged; `ratio: 1` with the flag is the paddle
control). Same rig both runs (five cilia, released 2 um cell, seed 2, ramped, coupling 0.1, all
gates PASS), only the rod drag `ratio` differs:
- **0484/0485 `p4_free_iso` (ratio 1):** reproduces 0473 -- net transport after the ramp beat
  -41,-10,-2,-11 nm, no swim. The control that must not swim.
- **0486/0487 `p4_free_aniso` (ratio 2):** steady +51,+78,+84,+73 nm/beat along the cilia, cell at
  1.06 um / 0.75 um/s / cos +0.81 at 0.5 s. The power stroke (broadside, full zeta) and recovery
  (edge-on, zeta/2) push unequal fluid -- the thrust of a cilium.

Momentum conserved by construction (scatter/gather is a partition of unity, so re-orienting the
reaction cannot create net momentum; the water's forward momentum is balanced by the cell's).
STATED CAVEAT: coupling 0.1 is an unresolved-grid correction, so 0.75 um/s is not calibrated -- the
RESULT is the contrast, iso 0 vs aniso steady. Next (P5): sweep `ratio` (1.0/1.5/2.0) and the beat
amplitude to see how swim speed scales, and add a second cell; all through the record.


**Result (run 0473, figure 0474): the released cell does not swim, as predicted.** All gates PASS
(base 26 deg, tip 41, 10.1 Hz, oop 0.2, bases 2.00 um). The cell oscillates +-0.8 um in the stroke
plane at the beat (rms 40 um/s) -- recoil of the bases -- and its net transport per whole beat is
+746 nm along the cilia in the ramp beat, then -41, -10, -2, -11 nm: nothing. Isotropic node
mobility gives zero mean base force over a periodic beat; the thrust of a cilium IS Gray-Hancock's
zeta_par/zeta_perp ~ 2, and the overdamped rod has one zeta in every rate. **P4 proper = anisotropic
node mobility** on the rod's own force sum (v = f_par/zeta_par + f_perp/zeta_perp): the engine sums
per-operator velocities, so this needs the rod operators to emit forces and one mobility step, or
the engine to apply a per-set mobility tensor before integration -- a design choice for Cedric.
The paddle control (`ratio 1`) is then exactly run 0473. Note the driver of 0473 died silently at
frame 13633 (the server finished; the record step was taken by hand as 0473).

Do the startup ramp FIRST (see the trap above), then:

61-65  release the cell: no pin, cell + cilium free to translate. Does it move?
66-70  TWO readings, not one. (a) net translation of the cell. (b) the far field must fall from
       r^-1 to r^-2, because a free swimmer exerts ZERO net force on the fluid and the Stokeslet
       term vanishes. The second is much harder to fake. `tools/` already has the radial probe.
71-75  sweep `coupling` 0.01-1.0 and `eta` for a nominal trajectory; expect Taylor's
       `U ~ (1/2) b^2 k omega` for small amplitude.
76-80  is the trajectory smooth? Speed should ripple at 2 omega (thrust peaks twice a cycle)
       with a small mean. A symmetric planar beat swims straight; nonzero mean curvature circles.

## P5 — propulsion by several cilia (81-100)

81-85  N cilia on a free cell, N = 1..5. Speed vs N.
86-90  synchronous vs metachronal at the same N — which swims faster, and by how much.
91-95  the controls again on the free swimmer: standing wave, and `ratio: 1`. Both must give
       zero net swimming. If either swims, something is wrong and the whole campaign is void.
96-100 the summary figure, and an honest list of what is still phenomenological — above all
       that `rod_motor` COMMANDS the beat rather than the beat emerging from dynein
       (Sartori's curvature-control law is the next thing, and it is not in this plan).

---

## Running notes

- **P1 done**, steps 0377-0409, figure at 0413. Wavelength below L makes the wave DECAY (t/b 0.80
  at L/2) and above L it grows. Frequency damps it: t/b 1.33, 1.01, 0.86 at 10, 20, 40 Hz, which
  is `Sp ~ omega^(1/4)`. Amplitude ceiling is sharp between 0.20 and 0.30 rad/um -- and Sartori's
  measured C0 is 0.232, so the animal sits just under this rig's limit. Raising it is a timestep
  problem.
- **The standing-wave control was ill-posed**, step 0416. See P1 16-18 above. Replace with a rigid
  filament before any propulsion claim in P4/P5.
- **`excursion` is not `drift`** and I reported the first as the second for a while. Peak excursion
  is inflated by pure oscillation; net drift is the mean displacement VECTOR over whole beats.
  `tools/cilia_row.py` now prints both, and `--no-water` skips the 200 MB water array when only
  the gates are wanted.
- **P2 done**, step 0411, PASSES its gate: anchoring the base to a cell reproduces the fixed-base
  beat to three digits (28.31 vs 28.30, tip 37.64 vs 37.60). No new operator -- `rod_base` already
  took `anchor: <set>`.
- **Built for P3-P5**: `rod_seed.spacing`/`spacing_axis` (every rod was being laid at the same
  base), the chain renderer reading the segment table instead of `per` (confirmed live: "7
  segments from the table"), and `rod_motor.ramp` for P4's startup transient.
- **P3 first rungs, 2026-09-23.** `p3s_n5_sync` 0435 and `p3s_n5_jitter` 0437, five cilia on a
  2 um cell, coupling 0.1, 1.5 beats. Both pass the gates (strain 5.2-5.3%, in box, 10.00 Hz).
  Two measurement faults found and fixed BEFORE reading them: (1) the phase axis per rod was
  "direction of largest sway" with an arbitrary SIGN, which read the one-clock reference as two
  groups 180 degrees apart -- it is now `b = n x d` per rod, exactly rod_motor's stroke direction,
  and the reference reads `r` = 1.000; (2) 1.5 beats is too short to see synchronisation, one beat
  of which is the transient -- every coupling-1.0 rung is now 0.5 s = 5 beats. Result so far: the
  seed-1 draw is nearly equispaced by chance (`r0` = 0.053 from the seed itself), the run's phases
  match the seeded ones at 1 beat, and at 0.1 coupling they have barely moved by 1.5 beats
  (`r` 0.039 -> 0.200 at the very end, rods 1-2 drifting toward rod 0). Not a result yet.
  **Control 0439 (coupling 0): `r` 0.033 -> 0.070, indistinguishable from the coupled run's
  0.034 -> 0.077.** So at coupling 0.1 and 1.5 beats the fluid did NOTHING measurable, and the
  small late rise is the transient, present in both. Figure 0441. This is the null the whole
  question rests on, and it is why the control ran before any result was read.
  TODO: a second seed (`phase_seed 2`) for robustness once the c1 series is in.
- **The coupling-1.0 gate FAILED (0442) even with the ramp**: max |pos| 1.4e25, no NaN, so the
  run never stopped itself. Diagnosis: EXPLICIT COUPLING INSTABILITY, not the startup. A grid
  cell holds ~8 water particles = 0.03 sim-mass; a rod node's force lands on a 3x3x3 stencil of
  ~0.25 mass, so `zeta*dt/m_local` is ~2 at coupling 1.0 (unstable) and ~0.2 at 0.1 (stable) --
  which is exactly why 0.1 "worked". Fix: the IMPLICIT drag exchange per step, force to the fluid
  `zeta(v-u)/(1+zeta*dt/m_local)`, unconditionally stable, equal to explicit when the ratio is
  small. Not a fudge: it is implicit Euler on the two-body exchange. Then re-run the gate.
- **Gate re-run 0444 with the implicit exchange: STABLE (box 0.90, strain 2.87%, exact residual)
  but NOT A BEAT.** Base 234 deg, tip 330: the cilia lie FLAT, swept by a jet of outward flow at
  the cell. `r` 0.47 -> 0.71 against the control's 0.03 -> 0.07 is therefore five rods carried
  together, not phase locking -- and my gate passed it because it never asked whether the thing
  was still a cilium. `cilia_row` now fails any run with base > 150 deg ("FLAIL"). Open: is this
  the 0.1 s ramp transient (the still is at 0.086 s) or the steady regime at full coupling? The
  5-beat `p3s_c1_sync` decides; if the cilia are still flat at 0.4 s the queue stops and the
  coupling model is wrong somewhere -- candidates: advection `u` applied to node 0 against the
  pin, or the local water simply too light for this zeta at this grid.
- **COUPLING 1.0 IS A DIFFERENT REGIME, figure 0447 -- the c1 queue was STOPPED.** The five-beat
  one-clock run at full coupling ends with three fates from one command: rod 0 sweeping 352 deg,
  rods 1/4 with tips going full circle, rods 2/3 nearly dead at 17 (0.1 gives five at 35 +- 1).
  The implicit factor 0.05 says why: the water under a node's stencil co-moves with the rod within
  a step, the elastic force has nothing to push against, the rod+water blob drifts and wraps.
  Gray-Hancock's zeta is a FAR-FIELD drag (RFT contains the near field analytically); applying it
  against the stencil-local grid velocity on a 1.56 um cell around a 0.2 um filament
  double-counts the near field. **The 0.1 that gave every good picture is an unresolved-grid
  correction wearing a dial's name.** Consistent fixes, NOT decided tonight (a modelling choice
  for Cedric): (a) far-field exchange -- zeta against u averaged over a filament-length region,
  reaction spread over the same; (b) grid refinement near the cilia; (c) derive the effective
  zeta for a body of size dx rather than a (ratio only ~1.85, so not enough alone).
- **Disk: today's runs wrote 954 GB of trajectories** (`save_data: true` overrides `record_cap`
  and records EVERY frame: 29 GB per 15,000-frame run). All queued specs switched to
  `save_data: null, record_cap: 301` (~200 MB). The 33 big npz files are reproducible from their
  specs and their measurements are in the record; deleting them is Cedric's call.
- **What P3 can still answer tonight, in the regime the rig supports:** the synchrony test at
  coupling 0.1 over FIVE beats -- random phases (seed 1 and seed 2) against the coupling-0
  control. Stated caveat: 0.1 is a correction, so a positive result is "the far field couples
  them", not "the physical drag does".
- **Five beats at coupling 0.1 (p3s5_jitter_s1, no ramp) ALSO FLAIL -- from beat 2.** Beat 1 is
  clean, five rods at 30-33 deg; in beat 2 rods 0/2/3 flip and stay flipped; beats 3-5 read
  [353, 69, 20, 16, 61] -- the SAME fates as coupling 1.0. Water: startup burst peaking at 0.15 s
  (87 um/s next to the cilia, 27 um/s bulk drift), decaying to 6 um/s by 0.45 s. So the tuft's own
  jet flattens its downstream members during beat 2. Transient + closed box, not steady
  circulation. The 1.5-beat windows never reached beat 2, which is why 0.1 "worked". `r` = 0.93
  on that run is rods carried together, gate FAIL (freq 8.99, FLAIL). Tests queued: seed 1 and
  seed 2 WITH `ramp 0.1` at coupling 0.1. If they hold five beats at ~30 deg, P3 proceeds on
  (0.1, ramp); if not, the boundary (jet accumulating in a closed 50 um box) is the next lever.
- **The five-beat flip was the RIG, not the water: control 0451 (coupling 0) flipped too.** See
  "Two rig defects" above: bases pulled to the cell centre, and an out-of-plane zero mode of the
  base clamp. Runs 0449-0455 are void. The four P3 five-beat runs (control, s1, s1r, s2) are
  re-queued on the fixed rig; the control must hold ~30 deg on all five rods with `oop` < 1 deg
  and `base_r` = 2.0 um before s1/s2 mean anything.
- **Kuramoto is the BASELINE, in the analysis only** (`tools/cilia_sync.py` fits K and overlays it
  dashed). See the P3 header.
- **P4 has an unresolved design point**: a released cell needs its own mobility. `rod_base`'s
  reaction arrives as a delta on the cell set, but the cell is only integrated if something emits
  on it, and the reaction is not scaled by the CELL's drag coefficient (Stokes 6*pi*eta*a =
  1.51e5 in sim units for a 2 um sphere) which differs from a rod node's. Decide this deliberately
  rather than at speed -- two unit slips today came from moving fast.
