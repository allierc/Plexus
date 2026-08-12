# Epistemic audit — the definitions

The taxonomy, the detectors and the constants that `epistemic_audit.py` reads. **Edit this file, not
the script.** The framework is Allier & Saalfeld 2026, *"Understanding: an experiment-LLM-memory
experiment"* (`/workspace/NeuralGraph/instructions_epistemic_analysis.md`); what is new here is that
Okuda's reasoning is *structured rather than prose*, so most modes are computed from
`records.jsonl` instead of tagged by hand.

WHY A SPEC FILE AND NOT CONSTANTS IN THE SCRIPT. The same reason `crew/basis.yaml` and
`crew/flow.yaml` exist: a number that shapes a conclusion belongs where the person reading the
conclusion can see it. The seed floors below decide whether three quarters of this campaign's
predictions count as measurable at all, and that is not a decision to bury in a function.

WHAT IS COMPUTED AND WHAT IS READ. Six modes are computable from the record because the loop
records them as fields — a prediction is a field, an outcome is a field, descent is a field. Six are
not: they live as prose in `knowledge.md` and `analysis.md`, and the script counts *candidate*
occurrences by marker and says so rather than pretending to have measured them. A marker count is a
lower bound on a mode, never a measurement of it.

```yaml
corpus:
  records: campaign/records.jsonl
  knowledge: campaign/knowledge.md
  analysis: campaign/analysis.md
  # a run is EVIDENCE only if it produced metrics; the rest are timeouts (see the note on
  # PG_ROUND_CAP in cluster.py) and counting them as reasoning would flatter every rate below.
  require_metrics: true
```

## The seed floor

Measured, not assumed: every `intent: replicate` run re-runs its parent's composition at a fresh
seed, so the spread between a replicate and its parent IS this substrate's reproducibility. Median
absolute relative difference over 21 replicate/parent pairs, r001–r022.

A prediction that asks for a change smaller than the floor of its own metric is not a prediction.
It is a coin toss with a number on it, and the audit reports it as such.

```yaml
seed_floor:                 # median |Δ| / parent, MEASURED over replicate pairs
  protrusion_aspect_max: 0.41
  n_tips: 0.36
  grip: 0.30
  n_cells: 0.29
  cells: 0.29
  n_tubes: 0.20
  act_cv: 0.17
  spot_spacing_cells: 0.15
  corr_act_rad: 0.15
  n_spots: 0.13
  gyr_prolate: 0.08
  invagination: 0.08
  act_max: 0.06
  protr: 0.02
  reduced_volume: 0.02
  _default: 0.20            # used, and named in the output, when a metric has no measured floor
```

## Modes

`detect` is how the script finds an instance. `computed` modes read structured fields; `marker`
modes count regex hits in the named prose files and are reported as candidates.

```yaml
modes:
  - name: Deduction
    kind: computed
    detect: has_prediction
    validated: true
    note: >-
      a slot that commits to one metric, one direction and one number BEFORE the run, scored
      automatically after it. This is the mode Okuda is built out of.

  - name: Falsification
    kind: computed
    detect: outcome_refuted
    validated: false
    note: >-
      a prediction contradicted. Popper's asymmetry makes this the strongest evidence there is --
      but only if something CHANGES because of it, which `refutation_followed_up` measures.

  - name: Boundary probing
    kind: computed
    detect: is_sweep
    validated: false
    note: Route A's ladders -- a parameter walked across its range on a fixed base.

  - name: Replication
    kind: computed
    detect: is_replicate
    validated: true
    note: >-
      not in the original taxonomy and added because this substrate is stochastic. It is what
      measures the seed floor, and its validation rate is the campaign's reliability.

  - name: Analogy/Transfer
    kind: computed
    detect: cross_lineage_edit
    validated: true
    note: >-
      the same edit (operator + parameter) applied on a parent from a DIFFERENT lineage than the
      one it was learned on. The framework's strongest evidence class; see `confidence`.

  - name: Surprise-chasing
    kind: computed
    detect: chases_set
    validated: false
    note: >-
      a slot following up a result nobody predicted, named by run id in the `chases` field. The
      only route by which an unplanned result becomes an experiment.

  - name: Induction
    kind: marker
    files: [knowledge]
    pattern: '(?im)^\s*[-*]\s+\*\*L\d+|scales with|consistently|in every run|across (all|both)'
    note: a pattern generalised over several runs. In Okuda these are the STANDING LAWS.

  - name: Abduction
    kind: marker
    files: [analysis, knowledge]
    pattern: '(?i)likely because|suggests that|consistent with .* mechanism|would explain|attributable to'

  - name: Causal chain
    kind: marker
    files: [analysis]
    pattern: '(?i)because .*,? which (causes|drives|makes|leads)|-> .* -> '

  - name: Regime recognition
    kind: marker
    files: [analysis, knowledge]
    pattern: '(?i)different regime|morphotype|qualitatively different|phase (transition|boundary)'

  - name: Meta-reasoning
    kind: marker
    files: [analysis, knowledge]
    pattern: '(?i)this metric (lies|is blind)|the instrument|strategy|we keep|round after round'

  - name: Uncertainty
    kind: marker
    files: [analysis, knowledge]
    pattern: '(?i)not reproducible|seed|variance|inconclusive|cannot be read|undefined'
```

## Confidence

The framework's formula, unchanged, so a claim scored here is comparable to one scored in the
NeuralGraph campaign. `n_blocks` is the number of distinct LINEAGES a claim has been tested on —
Okuda's nearest equivalent of a block, and the term that dominates the score.

```yaml
confidence:
  base: 0.30
  per_confirmation: 0.05      # x log2(n+1)
  per_alt_rejected: 0.10      # x log2(n+1)
  per_block: 0.15             # x n, linear -- cross-context is the strongest evidence
  cap: 1.00
  levels: {very_high: 0.90, high: 0.75, medium: 0.60}
```

## Priors — what was GIVEN and must not be counted as learned

```yaml
priors:
  - "the operator vocabulary: 13 operators, their families, slots and legal links"
  - "the parameter boxes (lo, hi, default) in composition_space.OPERATORS"
  - "the basis: 25 declared members and the axes that separate them"
  - "the four target morphologies and the metrics each seat is scored on"
  - "the premises P1-P13 and the Critic's static rejections"
  - "the metric bank: 24 admitted quantities and what each measures"
```
