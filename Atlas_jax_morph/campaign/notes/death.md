<!-- Death -- append below; the driver merges this into campaign/analysis.md -->

## Death (physics/death.py:L26) -- inspected

Read the whole `Death` class, its `StochasticStep` base (`core/step.py`), the AD primitives it
calls (`sample_bernoulli_st`, `bernoulli_logp` in `core/ad_utils.py`), and the sibling `Division`
it must compose with. It is Division's mirror image: a per-cell Bernoulli hazard
`p = 1 - exp(-death_rate*dt)`, forward-exact with an identity straight-through surrogate, replayed
HARD (flip `alive`, write a float `death` record) and scored by a masked score-function `logp`.

**Biggest surprise -- a code-vs-paper contradiction, and it runs the other direction from the usual
one.** The paper has NO death mechanism at all: "death"/"apoptosis"/"necrosis" occur ZERO times
(grep of the extracted text). The forward model's capability list (p. 2) is exactly "division,
growth, mechanical stress sensing, and morphogen excretion and detection." The paper's only cell
removal is an *external* robustness ablation -- deleting a random fraction of an already-finished
cluster to measure loss sensitivity (p. 8; Fig. caption p. 21), not a dynamic step. So `Death` is a
library step the source ships but the paper never describes. Recorded per rule 5 (source wins).

**Other traps I flagged:** `died` (ephemeral scored trace) vs `death` (persistent float record,
re-derived in replay by `(died>0.5) & alive`, and OVERWRITTEN not accumulated each step); the
Division-before-Death ordering and deferred slot reuse that keep `reconstruct_lineage` correct; the
`death_rate` clip to >=0 that stops a negative rate NaN-ing `bernoulli_logp`; and that the tunable
hazard is a STATE field, not a constructor arg (the only ctor param is `score_by_default`).

**Did NOT establish:** I did not run the oracle (paper ships no death config, and jax is
deliberately absent from this env), so I have no numerical trajectory confirming the hazard/lineage
behaviour -- purely a source read. I also did not trace whether any assembled model in the library's
examples/guides actually *includes* `Death` in a pipeline, so its intended composition partners
beyond the documented Division pairing are unverified. Verdict/contract left for the normalizer.

## Death -- normalized

**Verdict: `new`.** Contract `apoptose` (kind `structural`, family `growth`, set `cell`; reads
`death_rate`, writes `alive`+`death`). The frozen baseline has no operator that removes a set
member: `cell_divide` and `cell_grow` are the only structural/growth/cell contracts, and both ADD
matter (a daughter, or volume). `apoptose` is `cell_divide`'s biological inverse -- it retires a
live slot rather than waking a dormant one -- so no existing contract covers it and widening
`cell_divide` to also destroy cells would conflate mitosis with apoptosis (the paper itself lists
division as a capability and never death). I gave it `cell_divide`'s exact typing on purpose: same
kind/family/set, so the record shows a birth/death PAIR of structural growth operators, not a lone
outlier.

**Strongest argument AGAINST `new` (the alternative I had to defeat).** Death and Division are so
tightly coupled at the implementation level -- identical hazard `p = 1 - exp(-rate*dt)`, identical
`>=0` clip, identical straight-through discrete draw, identical `{action}_eligible` masking, the
same DISCRETE phase, and a *mandatory* divide-then-die ordering -- that one could argue they are two
IMPLEMENTATIONS (or two directions) of a single abstract contract: a Bernoulli-hazard toggle of cell
occupancy, `occ 0->1` for division and `1->0` for death. Under that reading Death is an alias of
`cell_divide` (or the pair is one operator with a sign), and calling it `new` inflates the atlas's
yield by counting a sign flip as a new contract -- exactly the failure `record.py`'s R5 exists to
catch. I rejected it because Plexus fixes contract identity by what an operator DOES to the state
(its writes and biology), not by the noise law it borrows: division writes position/radius/lineage
and conserves volume across a new inherited slot, while death writes only `alive`+`death` and frees
nothing that step; sharing a random-timing law makes them no more one contract than `diffuse` and
`decay` are for both scaling by `dt`. But the coupling is real, and if Plexus ever adds a
`population_turnover`/occupancy-toggle abstraction, `apoptose` and `cell_divide` would be its first
two implementations -- worth revisiting then.


## Death -- implemented

Operator `apoptose` at `src/plexus/operators/candidates/jax_morph_death.py`; test at
`tests/test_jax_morph_death.py` (6 properties, all pass). Written as `cell_divide`'s literal
inverse and modelled on it line-for-line: same `Structural` base, `EMIT=None`, the same
`getattr(lvl, "<rate>", None)`-else-scalar-`rate` fallback (`death_rate` buffer here, `div_rate`
there), the same `getattr(H, "rng", None)` draw. The whole effect is `lvl.occ[die] = 0.0` -- occ
IS the Plexus analog of the source's `alive` mask, and retiring it (rather than parking/zeroing
mass) is the faithful minimal translation, because jax-morph cells are single particles with no
MPM child to retire. Reuse of the freed slot is deferred to a LATER macro-step exactly as the
source intends: `cell_divide` (which allocates `occ==0` slots) runs earlier in the pipeline under
divide-then-die, so a slot freed this step survives for lineage reconstruction.

Deliberate translations of the source's subtleties:
- Hazard `p = -expm1(-clip(rate,0)*dt)` copied verbatim (the `-expm1` form, and the `clamp(min=0)`
  guard so a negative controller output gives p=0, not a NaN score).
- `die` is re-AND'd with the LIVE mask via the eligibility set (`elig = live & at-mask`), so an
  already-dormant slot can never be marked "newly dead" -- the source's `(died>0.5) & alive` guard.
- `death` is a lazily-registered per-node FLOAT buffer, zeroed then set each step (OVERWRITTEN, not
  accumulated), dtype float so it is summable/differentiable while occ stays the boolean liveness.
- The `at:` mask gates eligibility (the source's `die_eligible`); a test confirms masked-out live
  cells survive.

NOT modelled (and why): the trace/`logp` score-function (REINFORCE) layer. Plexus's engine runs the
forward EFFECT only (the source's `replay`), so `died`/`die_eligible` traces and `Death.logp` have
no engine counterpart -- same scope as `cell_divide`, which realises the forward proliferation event
without its scoring term. If a scoring/inverse-design driver lands, that layer is the follow-up.

Tests are reference-free by construction (limits, sign, conservation): rate-0 no-op; negative-rate
clip -> no death; huge-rate -> certain death of every live cell (seed-independent, tests the hazard
form + the retire); dormant slots never die/revive and live-count is monotone non-increasing
(apoptose is strictly a remover); the float `death` record equals the exact occ 1->0 flip mask; and
the eligibility mask restricts who may die. No oracle run -- the paper ships no death config and jax
is absent from this env by design -- so `evidence.oracle_run` stays null for the differ/curator.

Name note: `apoptose`, not `death`, so no clash with the pre-existing efflux-boundary `death`
operator (candidates/death.py, kind `lateral`) -- a geometric exit-line sink, an unrelated
mechanism. Candidates are not auto-imported, so nothing registers until the differ/tests import it.


## Death -- validated (DIFFER)

**PASS. `status: validated`.** The implemented `apoptose` reproduces the reference `Death` hazard.

**Why not a bit-for-bit trajectory diff.** Death is a discrete Bernoulli event and the JAX
(`jax.random.bernoulli`) and torch (`torch.rand`) streams are different RNGs, so seed-0-vs-seed-0
matches nothing stochastic. The paper also ships no death config, and there is no shared IC in the
smoke/anchor run (which has no death). So the differential is run on a purpose-built PURE-DEATH
rollout with a matched IC, and measures the one thing a hazard is defined by -- the pooled
per-macro-step death probability -- which is realization- and record-convention-independent.

**Metric (pre-registered).** `diff_metric = |p_hat_plexus - p_hat_oracle|`, pooled per-macro-step
death hazard, with `p_hat = (n_active[0] - n_active[-1]) / sum(n_active[:-1])` over the 41 recorded
frames (no births => every live-count decrement is one committed death => MLE Bernoulli hazard over
every eligible live-cell-step). `threshold = 3*SE_pooled`, `SE_pooled = sqrt(SE_o^2 + SE_p^2)`,
`SE = sqrt(p_hat(1-p_hat)/eligible)`. Both decided before reading the result; the band is justified
not just as a sampling interval but by discrimination -- it is tighter than the gap between the exact
hazard `1-exp(-lambda*dt)=0.048771` and the linear-approx bug `lambda*dt=0.05` (gap 0.00123), so an
operator that wrote `p=rate*dt` instead of the source's `-expm1(-rate*dt)` would FAIL this test.

**Matched IC.** Oracle `Model([Death])` and Plexus `config/atlas/death.yaml` both start N0=50000
live cells at uniform `death_rate = lambda = 0.05`, `dt=1.0`, 40 macro-steps, death the only
operator. (Death reads no position, so scattered positions in the Plexus box are irrelevant to the
survival metric.) One caveat handled explicitly: Plexus records AFTER applying each op, so its first
recorded frame is already post-step-1 (47506, not 50000; the acted ledger shows apoptose acted 41x
over 41 frames). A frame-index-aligned survival diff would be off-by-one and wrong; the pooled
hazard is stationary and unbiased on any consecutive-frame window, so it sidesteps this cleanly.

**Numbers.**

| side | p_hat | SE | deaths | eligible cell-steps | vs exact 0.048771 |
|------|-------|-----|--------|---------------------|-------------------|
| oracle (jax-morph Death) | 0.048928 | 2.29e-4 | 43236 | 883662 | 0.69 SE |
| Plexus (apoptose)        | 0.048985 | 2.35e-4 | 41169 | 840445 | 0.91 SE |

`diff = 5.66e-05 = 0.17 * SE_pooled  <=  threshold 0.000986`  ->  **PASS**. Both sides also agree
with the exact hazard within 1 SE. Acted ledger: `apoptose` calls 41 / acted 41 / inert []
=> `valid_evidence: true` (checked before any metric).

**Paths.** oracle run `Atlas_jax_morph/_oracle/runs/diff_death/` (script
`_oracle/scripts/death.py`); Plexus evidence `log/atlas/death/` (spec `config/atlas/death.yaml`);
differential computed by `_oracle/scripts/_analyze_death.py`.

**Scope (unchanged from implemented).** This validates the forward EFFECT (the source's `replay`):
the exact hazard form, the `>=0` clip, the retire (occ 1->0), and monotone non-increasing counts.
It does NOT exercise the trace/`logp` score-function (REINFORCE) layer -- Plexus's engine has no
counterpart, same as `cell_divide`. The `death` lineage float / divide-then-die ordering are covered
by the reference-free unit tests, not by this population-level differential.
