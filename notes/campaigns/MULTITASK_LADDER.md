# One connectome, several functions, selected by context

*Plan. Five rungs, each one thing harder than the last, on `${CLUSTER_QUEUE_PREFIX}l4`.*

## The question

The 285-cell zebrafish oculomotor connectome has been fitted to seven filter laws — an integrator,
a delay, a resonator, a Butterworth/Bessel bank — **one circuit per law**, seven separate fits. It
succeeds at all of them, and that is close to a foregone conclusion: 5,798 free parameters over
285 nodes with 285 available poles will fit a four-pole filter. The comparison is not even
flattering. `zf285` has 4.5× more units than the free 64-unit ctRNN and is 2.6–10× WORSE on six of
the seven, so the topological constraint costs more than the extra units buy.

The one result not explained by parameter count is that the connectome **wins** on
`t1_integrator_perfect` — 0.00064 against the free network's 0.00080. That is the task these cells
exist for: the velocity-to-position integrator that holds gaze, a pole at the origin, a line
attractor.

**The stronger statement, and the one this campaign tests: can ONE parameter set hold several
different function TYPES at once, selected by a context input?** Not a family of one law — that is
already shown, `t1_integrator_tau_sweep` is one fit holding four time constants at 0.0047 with its
slowest pole at −0.0298 /s against ground truth's −0.0312 — but an integrator *and* a delay *and* a
high-pass in the same weights.

**If it fails, WHERE it fails is the result.** Which pair of laws the circuit cannot hold
simultaneously is a statement about the connectome, not a null.

## What has to change first

A task spec declares **one** `teacher:` plus an optional `conditions:` grid over that teacher's
PARAMETERS. A grid cannot express "cell 0 is an integrator, cell 1 is a 100 ms delay, cell 2 is a
4th-order Butterworth" — those laws take different parameters and no cross product reaches them.

So the task language gains `teachers:`, a LIST of named cells each with its own law:

```yaml
teachers:
  - {name: integrate,  law: integrate}
  - {name: delay,      law: delay, seconds: 0.1}
  - {name: lowpass,    law: lti, family: butter, band: lowpass, order: 4, cutoff_hz: 1.0}
```

`teacher:` + `conditions:` stays exactly as it is — a grid over one law is the common case and its
specs must not move. The two forms produce the SAME thing downstream: one condition-cell index per
trial, one `per_cell` record in `teacher.pt`, and the one-hot context channel the trainers already
append. So the trainer needs no change at all; only the generator and the schema do.

**One plot per function.** `plot_trainer` currently writes one movie and one figure per run. With
`n_cond > 1` it writes one of each PER CELL, named by the cell, plus the aggregate — because "it
holds the integrator and loses the delay" is invisible in a pooled number.

## The ladder

Each rung is a task spec in `config/task/` and a run spec in `config/run/`, fitted with the
zebrafish circuit (`config/neural/zf_circuit_285.yaml`) and, as the control, the free 64-unit
ctRNN. Results land in `log/task/<run>/` as every other fit does.

| | task | cells | what it settles |
|---|---|---|---|
| **1** | `m1_integrate` | 1 | **the working point.** One law, no context channel. Must reproduce `t1_integrator_perfect` — 0.0008 ctRNN / 0.00064 zf285 — or the ladder is measuring something else |
| **2** | `m2_integrate_ctx` | 1 | **the one-hot is inert.** The same law with a context channel that carries no information (one cell, so the one-hot is a constant 1). Any change from rung 1 is the CHANNEL, not the task |
| **3** | `m3_int_delay` | 2 | integrator + 100 ms delay. The first genuinely different pair: one needs a pole at the origin, the other a flat magnitude and a linear phase |
| **4** | `m4_int_delay_lowpass` | 3 | + a 4th-order Butterworth low-pass |
| **5** | `m5_six` | 6 | + high-pass, resonator, leaky differentiator. The full claim |

Rung 2 is not a formality. The context one-hot enters through the same afferent map as the
stimulus, so it competes for the same 41 AF5 cells; a constant extra input that shifts the
operating point would degrade rung 1's number for a reason that has nothing to do with multitasking
and would be invisible at rung 3 onwards.

### The six laws

| name | law | why it is here |
|---|---|---|
| `integrate` | `integrate` | pole at the origin — the circuit's own function |
| `delay` | `delay`, 0.1 s | flat magnitude, linear phase — no pole at all |
| `lowpass` | `lti` butter, lowpass, order 4, 1 Hz | four poles on an arc |
| `highpass` | `lti` butter, highpass, order 4, 1 Hz | the same arc with zeros at the origin |
| `resonator` | `laplace` ω₀ = 2π rad/s, ζ = 0.4 | a complex pair — ringing |
| `differentiate` | `laplace` num [1, 0] den [0.05, 1] | a leaky differentiator; the pure one is improper |

## How it is judged

1. **Per cell, always.** `mse_per_cell` already exists in the tester; the ladder's whole content is
   how it degrades as cells are added. A pooled number can hide one dead function among five good
   ones.
2. **Poles per cell.** Output error says the answer is close; the linearised poles say the law was
   acquired. For a multi-cell fit the circuit is ONE system and the ground truths are several, so
   the question becomes whether the union of the circuit's poles contains every cell's — which
   `max Re` cannot ask and a per-cell plot can show.
3. **Against the free ctRNN at every rung**, because the interesting quantity is the CONSTRAINT's
   cost, not the absolute error.
4. **Against rung 1**, because a rung that degrades has to be read against the same circuit doing
   one thing, not against zero.

## Running it

`${CLUSTER_QUEUE_PREFIX}l4`, one `bsub` per run, the pattern in `tools/submit_specs.py`: relative paths after a `cd`,
`cluster.cpath` for the mount translation, `conda run -n` for the environment. Nothing runs on a
login node — `python`, `conda` and `rsync` are watched there and the admins kill them.

The corpora are generated locally first (seconds) and only the fits are submitted.
