# Protocol: characterising an eye for the oculomotor controller

For the Plexus `prototype/eye` session. Produces the measurements needed to
fit a differentiable 6-muscle plant that the connectome controller is trained
through. Written to be run unchanged on any eye variant — F, G, whatever comes
next — so nothing here should need re-deriving per eye.

## Three things that are easy to get wrong

Each is detailed in its own stage below; they are collected here because each
has already cost us a dataset or a fit.

**Hold length is derived per eye** from stage 0's measured settling time,
`max(2.0 s, 1.5 × settling)` — it is not fixed at 2.0 s. A fixed constant is
exactly how eyes A–E ended up fitted entirely from transients: their holds ran
0.19 s, 1.27 s and 0.24 s against a settling time of 1.28 s, so not one of them
had stopped moving when it was sampled, and the resulting "static curve" had a
negative slope at the origin.

**Keep the raw runs.** The `curves.npz` behind eyes A–E were deleted and that
fit is no longer reproducible — the coefficients survive in `plant.npz`, but
nothing can be re-derived, re-plotted or re-checked from them.

**Stage-1 levels are `{0.10, 0.25, 0.50, 1.00}`**, weighted to the low end
rather than evenly spaced, because eye F's nonlinearity is strongly convex:
LR's local gain rose 6.9× from `u = 0` to `u = 1`. The low end carries the
shape, and it is also where a tracking controller actually operates.

## The model this feeds, in one paragraph

The eye is taken to be a **static map followed by linear mechanics**. Six
muscle drives `m ∈ [0,1]⁶` produce an equilibrium pose
`(θ∞, φ∞, ψ∞) = g(m)` — horizontal, vertical, torsion, in degrees — and the
globe then swings to that equilibrium under second-order dynamics. So the
static map is measured with **holds** (drive constant, wait until it stops
moving, record where it stopped) and the mechanics with **steps** (drive
jumps, record the whole swing). Those are the only two kinds of run needed.

Torsion is recorded throughout, not discarded. On eye F the lateral rectus
produced 3.86° of torsion at full drive against 6.17° of horizontal — 63 % —
so a two-angle description of this plant is not tenable.

## Task requirement — check this first

The controller has to track a target spanning **25° horizontal and 10°
vertical**. An eye that cannot reach that cannot do the task, and no amount of
characterisation fixes it.

| eye | horizontal span | vertical span | verdict |
|---|---|---|---|
| C `eye_p3a_length` | 30° (−15.0 / +15.1) | 18° | passes |
| F | **7.9°** (−1.75 / +6.17) | 6.3° (−4.35 / +1.99) | **fails, 3.2× short** |
| G (Blender anatomy) | **16.4°** (−8.5 / +7.9) | 27.6° (−15.4 / +12.2) | **fails, 1.5× short** |
| H (G, globe ×1.2) | 4.8° | 14.2° | fails, 5× short |

Span is quoted from single-muscle extremes. Allowing full co-activation of
every muscle in the helpful direction — which also swings the other two axes,
so it is a generous bound rather than a usable range — eye F reaches 10.8°
horizontal and 11.4° vertical, so horizontal still fails by 2.3×.

Two other stage-0 outcomes on eye F are worth recording as failure modes this
protocol is meant to catch, because neither is visible once the data is
pooled:

- **SR diverged** — 69.2° vertical with 13.5° peak-to-peak. That is a
  simulation blow-up, not a measurement, and it has to be diagnosed rather
  than excluded, because stage 2 drives muscles in pairs and whatever caused
  it will reappear there.
- **Torsion is the largest axis**: 14.9° range against 7.9° horizontal, with
  IO alone producing −11.08°. A plant that twists twice as far as it looks
  sideways has its straps pulling tangentially rather than in the plane of
  rotation. Related: SO elevates (+1.99) and IO depresses (−4.35), which is
  the reverse of the textbook action of both. It is not a sign convention —
  IR depresses correctly — so check the two obliques' insertions in
  `eye_anatomy.MUSCLES` before anything else is fitted.

Eye G is the first eye whose **vertical** passes, and it passes by 2.8×; its
horizontal is the binding axis and it is 1.5× short rather than F's 3.2×. Two
of its numbers say which lever to reach for:

- **Globe size is the strong lever, and bigger is worse.** Eye H is eye G with
  the globe scaled ×1.2 and every direction collapses — H/G is 0.57 up, 0.47
  down, 0.39 nasal and **0.19 temporal**. The lateral rectus is the most
  size-sensitive muscle in the plant, and horizontal is the axis that needs
  the help, so the obvious test is the other direction: globe ×0.85, one
  stage-0-lite run, one job.
- **Drive is not a lever; it is already at its ceiling.** At ×1.5 (amplitude
  67 → 100.5) the globe loses 6.4 % of its radius, peak shortening goes 26 % →
  74 %, `strain_p99` returns NaN and two of the four synergies fail. At ×2 the
  radius loss is 16 % and three of four fail. The working point is as hard as
  this geometry can be driven.

If stage 0-lite fails the gate, stop and change the eye rather than
continuing. The other known lever is strap geometry: B → C widened the muscle
gap 0.020 → 0.042 and the strap fraction 0.55 → 0.95, and travel went 3.4° →
15.0°.

## Conventions, fixed for every eye

- **Muscle order** is `[m["key"] for m in eye_anatomy.MUSCLES]`. Write that
  list into every output file rather than assuming it; do not hardcode an
  order in the analysis.
- **Pose** is the Kabsch fit `eye_pose` → `(h, v, t)` in degrees, recorded
  relative to the pose at rest at the start of that run.
- **Drive** `u ∈ [0,1]` per muscle, one-sided: muscles pull, they do not push.
- **Settled** means peak-to-peak of all three angles ≤ **0.05°** over the
  averaging window. Every hold carries this flag; unsettled holds are kept in
  the file and excluded by the fit, never silently dropped.

## Stage 0-lite — the synergy gate (ONE run)

Before the per-muscle sweep, drive the four cardinal **synergies** in turn in a
single simulation: SR+SO up, IR+IO down, LR temporal, MR nasal. This is
`probe_groups.py` with `muscle_probe [groups]`, and
`archive/eye_G/pairs_long_spec.yaml` is a working instance of it —
`groups: [[1,4],[3,5],[0],[2]]`, `a_hi 1.0`, `tonic 0.14`, `hold 200`,
`rest 160`, four phases in 1540 frames.

It answers two questions the per-muscle sweep cannot, at a cost of one job:

- **Does the geometry do what the anatomy claims?** No single extraocular
  muscle moves a fish eye along a cardinal axis, so only a synergy can be
  scored against a direction written down in advance. On eye G all four came
  out correct, which is what says the scanned insertions are right.
- **What is the span?** The union of the four excursions is the usable
  workspace, and it is what the gate below tests.

Use `probe_groups.py`, not `run_synergies.py`. The latter still carries
`PHASES = [("up", ["LR", "SO"], …)]` — the *lateral* rectus, not the superior
— while listing LR again as the temporal phase. `probe_groups.py` has SR+SO.
Fix or delete the stale one before anyone runs it.

## Stage 0 — per-muscle span and settling time (6 runs)

Full drive `u = 1.0`, one muscle at a time, from rest, held long enough to
stop moving (start with 3 s; if it has not settled, say so rather than
shortening the analysis window).

Report per muscle: settled pose `(h, v, t)`, and the settling time — the first
moment the trace stays inside ±0.05° of its final value.

Two numbers come out of this and both are used everywhere below:

- **span** = `max h − min h` over the six muscles, and likewise for `v`.
  Gate: horizontal ≥ 25°, vertical ≥ 10°.
- **hold length** `T_hold = max(2.0 s, 1.5 × slowest settling time)`. Derived
  per eye, not a constant — a softer eye settles more slowly and a fixed
  2.0 s would quietly measure transients again, which is the error that
  invalidated eyes A–E.

## Stage 1 — the six single-muscle curves (24 holds)

Each muscle alone at `u ∈ {0.10, 0.25, 0.50, 1.00}`. From rest each time:
ramp over 0.2 s, hold `T_hold`, average the last 25 %.

Four levels, weighted to the low end rather than evenly spaced, because the
nonlinearity on eye F is **strongly convex** — LR's local gain rose 6.9× from
`u = 0` to `u = 1` — so the shape lives near the origin, which is also where a
tracking controller spends most of its time. Four suffice because each curve
is a quadratic through the origin: two coefficients, four points, two degrees
of freedom left to see the fit fail.

This gives the six *marginal* curves `φᵢ(u)`, which are most of the static map.

## Stage 2 — which muscle pairs are worth testing (15 + ~16 holds)

The reduction you asked for. The additive prediction for two muscles driven
together is `φᵢ(uᵢ) + φⱼ(uⱼ)`. It is either right or it is not, and **one hold
per pair decides it**.

**2a — screen (15 holds).** Every unordered pair, both at `u = 0.5`. Compute
the residual against the additive prediction, per angle. Flag the pair if any
component exceeds **0.20°** (four times the settled tolerance).

**2b — grid, flagged pairs only (4 holds each).** For each flagged pair,
`uᵢ, uⱼ ∈ {0.35, 0.75}`. That is enough for a bilinear interaction term, which
is the order the screen can justify; anything richer is not identifiable from
a screen at one point. Unflagged pairs get nothing further: their interaction
has been measured and found to be below the noise that matters.

Expect roughly 3–5 pairs to flag, so ~16 holds. If more than 8 flag, stop and
report — that would mean the eye is not close to additive and the sampling
plan needs to change rather than expand.

Report the full 15-row residual table either way. A pair that does *not*
interact is a result, and it is what keeps the next eye cheap.

## Stage 3 — the mechanics (3 runs, because the rest is already recorded)

The earlier version of this protocol spent about 25 runs on transients. It
does not need to. **Every hold in stages 1 and 2 is already a step from rest
followed by a release**, recorded at full rate in its own `curves.npz`, so the
39 holds above contain 39 step responses across the whole reachable workspace.
Fit `C` and `K` from those. Requiring separate step runs was double-paying for
data the protocol already writes, and it is where most of the saving below
comes from.

Only one measurement is genuinely missing, because no hold varies it:

- **3 co-contraction pairs**: two commands matched to give the *same* final
  pose but different total drive `Σ mᵢ` — e.g. `(LR, MR) = (0.5, 0.0)` against
  `(0.9, 0.4)` tuned to land at the same horizontal angle. If the second
  settles faster or rings less, co-contraction is stiffening the eye and the
  model needs `C(s)`, `K(s)`. If not, it does not, and the block is dropped.

If the holds turn out to be too heavily damped to identify `ζ` — they will not
be at `ζ ≈ 0.2–0.3`, but check — add single-muscle chirps, `u(t) = 0.5 + 0.4
sin(2π f(t) t)` sweeping 0.2 → 5 Hz over 8 s, one per muscle. That is the only
circumstance in which stage 3 costs more than three runs.

## Output format

One directory `characterise_<eye>/` containing:

```
holds.npz       muscles  (6,)   str, the muscle order used
                m        (n,6)  commands
                pose     (n,3)  settled mean (h, v, t) in deg
                p2p      (n,3)  peak-to-peak over the averaging window
                settled  (n,)   bool
                stage    (n,)   '0' | '1' | '2a' | '2b'
                T_hold   ()     seconds, from stage 0

transients.npz  one group per run: t, m(t) (T,6), pose(t) (T,3), dt

report.json     span_h, span_v, gate_pass, settling_time per muscle,
                noise_floor, pair_residuals (15 rows), flagged_pairs
```

Plus, per run, whatever spec/movie the prototype already writes — the
`curves.npz` behind eyes A–E were deleted and their fit is no longer
reproducible, so **keep the raw runs for this one**.

## Cost

| stage | runs | note |
|---|---|---|
| 0-lite synergy gate | **1** | four directions in one simulation; stop here if it fails |
| 0 per-muscle span | 6 | also sets `T_hold` |
| 1 marginals | 24 | six muscles, four levels |
| 2a pair screen | 15 | decides 2b |
| 2b flagged grids | ~16 | 4 per flagged pair, `{0.35, 0.75}²` |
| 3 mechanics | 3 | the step responses come from stages 1 and 2 |
| **total** | **~65** | against 200 blind Sobol points |

Three compressions, in order of how much they save. Stage 3 stops re-running
steps it already has. Stage 2 screens all fifteen pairs at one point each and
grids only the ones that interact — a pair below threshold becomes a recorded
measurement rather than an untested assumption, at a cost of one run. And
stage 0-lite answers the gate in a single job, so a plant that cannot do the
task is rejected for one run rather than six.

## Sanity checks to run before handing the data over

- **Noise floor**: repeat 5 stage-1 holds at the end of the session. Spread
  across repeats is the resolution of everything above; if it exceeds 0.05°
  the settled criterion has to move with it.
- **Hysteresis**: revisit 10 stage-1 points in a different order. A systematic
  offset means the plant has memory the static map cannot represent, and that
  has to be known before anything is fitted.
- **Rest drift**: confirm the pose returns to within the noise floor of zero
  between runs.
