# Dicty Model-Discovery Loop — agent-in-the-loop, knowledge over score

## What this is

A hypothesis-driven scientific loop (not a numeric optimizer) for discovering **which Plexus
mechanisms are necessary and sufficient to reproduce *Dictyostelium* aggregation** from the real
movie. You are the scientist. Each batch you run **256 simulations = 16 single-parameter sweeps ×
16 values** (sims are cheap; this amortizes your reasoning call). You *look at each sweep's response
figure + morphology strip*, read the metrics, decide which hypotheses are supported or falsified,
**update the knowledge ledger**, and design the next 16 sweeps. The score adjudicates hypotheses —
it is never the objective. A clean falsification is a success.

Unlike a parameter optimizer, **you may change the MODEL itself**: edit operator code and the
engine to add/modify mechanisms (e.g., a cAMP relay-wave field), then test whether the new
mechanism is necessary/sufficient. The parallel UCB run (`opt_dicty_continue.py`) is the
"parameter-tuning-alone" control arm — your job is the structural/mechanistic science it cannot do.

## The system you are studying

`dicty_engine.py` — a flat point-cell engine (one `cell` Level over a dormant buffer `H.c_active`/
`H.c_w`) + one diffusing/decaying `camp` GridField. Overdamped integration: `pos += dt * Σ(cell
velocities)`, capped at `vmax`. Operators (in `dicty_ops.py` unless noted, all registered):

| op | kind | role |
|---|---|---|
| `secrete` (chemotaxis.py) | exchange | cells deposit cAMP into `camp` |
| `sense` (chemotaxis.py) | exchange | chemotaxis up the cAMP gradient |
| `spring` | lateral | repulsion (volume exclusion) + sigmoid-gated adhesion `[k_rep,r0,kadh,r_on,delta,mu_f]` |
| `inflow` | structural | SOURCE: cells enter the volume at independent positions, `rate`/frame |
| `divide` | structural | proliferation (daughters next to a mother) — biologically wrong here, kept for ablation |
| `random_walk` (random_walk.py) | lateral | undirected motility |

Schedule tokens: operator names, `<field>.diffuse`, `integrate`. The engine applies any
registered cell-level op returning `{"cell": <Nx2 velocity>}`, and structural ops that mutate
`H.c_active`/positions and return `{}`.

### Adding or changing a mechanism (operator code)

To add an operator, in `dicty_ops.py`:
```python
@register_operator("relay", level="cell", kind="exchange")   # or a field-update hook
class Relay(...):
    REQUIRES_PARAMS = [...]
    def forward(self, H, mask=None):
        ...                      # read H.level("cell"), H.fields["camp"], H.c_active
        return {"cell": accel}   # a velocity contribution; or {} if it only mutates state/field
```
For an **excitable/relay cAMP field** you will likely edit `dicty_engine.py` (the field `.step()`
or add a new field with FitzHugh–Nagumo-style dynamics: an activator that fires when a threshold
is crossed and a slow refractory variable), then reference it in the spec schedule. Keep edits
minimal and reversible; ablation = a spec that does not schedule the new op.

## The data + how you are scored

Real movie: `260228...mp4` — scattered amoebae stream into a few compact mounds. Targets are
pre-extracted (`make_target.py`, `make_velocity_target.py`); **do not regenerate them**.

- **loss** (primary): late-weighted **(1 − SSIM of the full-FOV cell IMAGE)** + 0.2·**g(r)**
  autocorrelation MSE + 0.3·**mound-count** penalty + 0.5·velocity MSE. Lower = closer. SSIM-on-image
  is sound *because the sim is seeded from the real frame-0 cell positions* (`init_npz`) — mound
  locations should be reproducible, so absolute layout is meaningful. g(r) + mound-count are the
  translation/rotation-invariant structure terms (they preserve multi-mound structure; a COM radial
  profile destroys it).
- **ssim** (↑ better, 1 = identical) and **n_mounds** (target: real has **~8 mounds**) — the two
  scalars to actually move. Raise SSIM and match n_mounds≈8.
- **inner_mass / radial profile**: DIAGNOSTICS ONLY (gameable by a single over-tight blob — do not
  optimize). Real≈0.61 but matching it without the right morphology is meaningless.
- **N growth** (n0→n_final): should rise (influx), matching the real 767→~1400.

## VISUAL INSPECTION IS MANDATORY (primary evidence)

Every batch, **Read every sweep figure** `sweep_<i>_<param>.png` (a response curve — inner_mass &
loss vs the swept value, with REAL inner-mass=0.61 dashed — plus a strip of the final SIM density at
each of the 16 values, REAL density at the end) **and** `best_montage.png` (the single best config
over time, REAL vs SIM). Numbers can match for the wrong morphology, so each sweep's log entry MUST
include a one-line **morphological observation** from the strip, e.g. "many small mounds at all
values", "merges into a strand above kadh=120", "single central blob — overshoots", "no effect —
saturated". This qualitative read is the science.

## Method (every batch)

1. **Read** `dicty_knowledge.md` (ledger) and `user_input.md` (act on pending items, then move them
   to Acknowledged with a timestamp).
2. **Observe**: read `sweep_results.json` AND look at all sweep figures + `best_montage.png`. For
   each sweep record: response shape (monotone / peaked / flat), best value, and the morphology
   from the strip.
3. **Adjudicate**: was each hypothesis supported / falsified / inconclusive? Update the ledger:
   - mechanism-level result seen ≥3× → **Established Principles**
   - new/uncertain → **Open Questions**
   - contradicted → **Falsified Hypotheses** (state original, evidence, lesson)
4. **Design the next batch**:
   - Set `specs/dicty_loop_base.yaml` to the **best config so far** (the parent/control). Every sweep
     varies ONE parameter around this parent, so each effect is attributable.
   - Choose 16 sweeps: refine ranges around the optima you found, drop saturated parameters, add new
     ones. To test a **mechanism**, edit `dicty_ops.py`/`dicty_engine.py` to add it, put it in the
     base spec, and include a sweep over its strength **plus keep strength=0 in the range as the
     ablation** (necessity + sufficiency in one sweep).
5. **Edit** `sweep_plan.json` (and the base spec + operator/engine code as needed). Append your
   per-sweep entries to `dicty_loop_log.md` and update `dicty_knowledge.md`.

## Files you maintain

- `dicty_knowledge.md` — the ledger (Established / Falsified / Open). Read+update every batch. THE deliverable.
- `dicty_loop_log.md` — append-only full log (per-slot entries + batch summaries).
- `user_input.md` — read every batch; acknowledge pending instructions.
- `sweep_plan.json` — the experiment design (16 sweeps × 16 values) you rewrite each batch.
- `specs/dicty_loop_base.yaml` — the parent/control config; set it to the current best each batch.

**Do NOT read** `dicty_cli_transcript.md` — it is a human-only record of raw CLI output written by
the loop; reading it wastes your time budget.

## Log entry format (per sweep, append to dicty_loop_log.md)

```
## Batch B Sweep i — <param>  [supported/falsified/inconclusive]
Hypothesis: "..."
Response: <monotone up / peaked at X / flat>; best value=<v> (loss=<l>, inner_mass=<m>)
Morphology (from strip): <one line of what you SAW across the 16 values>
Verdict: <supported/falsified/inconclusive> — <why>
Knowledge update: <what this adds to / removes from the ledger>
```
Also append a one-line **batch summary**: new parent config + best loss/inner_mass + key insight.

## The central scientific question (current frontier)

The model {AR + self-generated chemotaxis + influx} is **falsified as sufficient** for the
few-compact-mound morphology (inner_mass ≤0.38 vs real 0.61; many small mounds). Leading
hypothesis: a **long-range cAMP relay-wave (excitable medium)** mechanism is necessary to merge
streams into few large mounds. Design experiments to test its **necessity and sufficiency**, and
record the verdict — that verdict, not a lower loss, is the result.

## Start (when prompted `DICTY START`)

- Read this file, the ledger, and the base spec `specs/dicty_loop_base.yaml`.
- Design the first `sweep_plan.json`: 16 sweeps × 16 values around the influx baseline (the base
  spec is the control). A reasonable first batch sweeps the core parameters (inflow.rate, sense.gain,
  secrete.rate, camp.diffusion, camp.decay, spring.{k_rep,r0,kadh,r_on,mu_f}, random_walk.strength,
  dt, cell.n, vmax) to map the response surface before committing to a mechanism change.
- Write the planned hypotheses to `dicty_knowledge.md` before editing `sweep_plan.json`.
  (A seed `sweep_plan.json` with 16 sweeps already exists — you may refine it.)
