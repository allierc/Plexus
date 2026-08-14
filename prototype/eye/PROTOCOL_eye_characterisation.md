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

**Stage-1 levels are `{0.10, 0.25, 0.50, 0.75, 1.00}`**, not four evenly
spaced ones, because eye F's nonlinearity is strongly convex: LR's local gain
rose 6.9× from `u = 0` to `u = 1`. The low end carries the shape, and it is
also where a tracking controller actually operates.

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
| F | **7.9°** (−1.75 / +6.17) | not yet measured | **fails, 3.2× short** |

If stage 0 below fails the gate, stop and change the eye rather than
continuing. The known lever is strap geometry: B → C widened the muscle gap
0.020 → 0.042 and the strap fraction 0.55 → 0.95, and travel went 3.4° → 15.0°.
Drive amplitude is a weaker lever (D → E raised it 60 → 67 and travel fell).

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

## Stage 0 — span gate and settling time (6 runs)

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

## Stage 1 — the six single-muscle curves (30 holds)

Each muscle alone at `u ∈ {0.10, 0.25, 0.50, 0.75, 1.00}`. From rest each
time: ramp over 0.2 s, hold `T_hold`, average the last 25 %.

Five levels rather than four because the nonlinearity on eye F is **strongly
convex** — LR's local gain rose 6.9× from `u = 0` to `u = 1` — so the low end
is where the shape is, and it is also where a tracking controller spends most
of its time.

This gives the six *marginal* curves `φᵢ(u)`, which are most of the static map.

## Stage 2 — which muscle pairs are worth testing (15 + ~36 holds)

The reduction you asked for. The additive prediction for two muscles driven
together is `φᵢ(uᵢ) + φⱼ(uⱼ)`. It is either right or it is not, and **one hold
per pair decides it**.

**2a — screen (15 holds).** Every unordered pair, both at `u = 0.5`. Compute
the residual against the additive prediction, per angle. Flag the pair if any
component exceeds **0.20°** (four times the settled tolerance).

**2b — grid, flagged pairs only (9 holds each).** For each flagged pair, the
3×3 interior `uᵢ, uⱼ ∈ {0.25, 0.50, 0.75}`. Unflagged pairs get nothing
further: their interaction has been measured and found to be below the noise
that matters.

Expect roughly 3–5 pairs to flag, so ~36 holds. If more than 8 flag, stop and
report — that would mean the plant is not close to additive and the sampling
plan needs to change rather than expand.

Report the full 15-row residual table either way. A pair that does *not*
interact is a result, and it is what keeps the next eye cheap.

## Stage 3 — the mechanics (about 25 runs)

Steps, not holds; record the whole trajectory at full rate.

- **12 steps** from rest to a random `m` (use the stage-1 and stage-2 points
  already known to be reachable), so the swing is observed from many
  directions.
- **6 single-muscle chirps**, `u(t) = 0.5 + 0.4 sin(2π f(t) t)`, `f` sweeping
  0.2 → 5 Hz over 8 s. This is what pins down the damping.
- **6 co-contraction pairs**: two runs matched to give the *same* final pose
  but different total drive `Σ mᵢ` — e.g. `(LR, MR) = (0.5, 0.0)` against
  `(0.9, 0.4)` tuned to land at the same horizontal angle. If the second
  settles faster or rings less, co-contraction is stiffening the plant and the
  model needs a term for it. If not, it does not. Three such matched pairs.

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
| 0 span gate | 6 | stop here if it fails |
| 1 marginals | 30 | |
| 2a pair screen | 15 | decides 2b |
| 2b flagged grids | ~36 | 9 per flagged pair |
| 3 mechanics | ~25 | trajectories, not holds |
| **total** | **~112** | against 200 blind Sobol points |

The saving is entirely in stage 2: screening all 15 pairs at one point each
costs 15 runs and removes the need to grid the ones that do not interact.

## Sanity checks to run before handing the data over

- **Noise floor**: repeat 5 stage-1 holds at the end of the session. Spread
  across repeats is the resolution of everything above; if it exceeds 0.05°
  the settled criterion has to move with it.
- **Hysteresis**: revisit 10 stage-1 points in a different order. A systematic
  offset means the plant has memory the static map cannot represent, and that
  has to be known before anything is fitted.
- **Rest drift**: confirm the pose returns to within the noise floor of zero
  between runs.
