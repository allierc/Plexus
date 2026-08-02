# Atlas — CompuCell3D: status

*Written 2026-08-02, at the point where Phases 0–2 are done and everything downstream is deferred
by programme policy. Programme-level view: `../ATLAS_STATUS.md`. Narrative: `atlas_note.pdf`
(11 pages).*

---

## 1. Why this target

`atlas_jax/atlas_note.pdf` nominated it before any of this work, on the grounds that it is *the
biggest step away from everything we have*: a cell is not a point with a radius, it is a **set of
lattice sites** sharing an id. Volume is a site count, surface a boundary count, position a derived
centroid. Time advances by attempting to copy one site's id into a neighbour, accepted with
probability 1 if the total energy falls and `exp(−ΔE/T)` otherwise.

That is the point of the exercise. Every measurement in the first atlas rested on cells being
points and a differential test being a subtraction. **If the procedure is general it survives this;
if it is secretly a particle-framework procedure, this is where it breaks.**

**Target:** CompuCell3D 4.10.0 (conda channel `compucell3d`, py312), Swat et al., *Multi-Scale
Modeling of Tissues Using CompuCell3D*, Methods Cell Biol 110:325–366 (2012).

---

## 2. Where it stands

| phase | state | what it produced |
|---|---|---|
| 0 — instruments + oracle | **done** | 5 of 7 instruments transferred with zero edits; oracle installs, imports, isolates, runs headless, is deterministic |
| 1 — read every mechanism | **done** | 35 mechanisms at `inspected`, 25 excavator calls, 0 reverted |
| 2 — normalize + skeptic | **done** | 14 `new` claims challenged, 5 refuted; NEW 8, yield 0.33 |
| 3 — implement | **deferred** | policy: extract many → catalog → promote. See `../ATLAS_STATUS.md` §3 |
| 4 — differential validation | deferred | needs implementations |
| 5 — forward figure | deferred | |
| 6 — differentiability | **likely N/A** | a Metropolis accept/reject has no pathwise derivative; saying so is a finding, not a gap |
| 7 — promotion | not started | |

`atlas.py status`: 0 candidates, 34 normalized, **0 validator violations**, 4 blocked (all logged).

---

## 3. The measurement

35 mechanisms → **8 genuinely new contracts**, 6 implementations, 4 aliases, 6 refinements,
11 out-of-scope. **Yield 0.33 new contracts per scored mechanism**, against jax-morph's 0.44 — the
direction the saturation hypothesis predicts, and not guaranteed. One data point, not a trend.

New here: `occupy`, `volume_elasticity`, `membrane_tension`, `elongate`, `stiffen`, `bond`,
`stay_connected`, `react`.

**Five mechanisms landed on contracts jax-morph had already proposed** — the four contact plugins
on `adhere`, and `SteadyStateDiffusionSolver` on `morphogen`. Those are the measurement; the rest
is bookkeeping.

---

## 4. Findings worth carrying forward

- **Every CPM mechanism is an energy term, not an update.** Nothing writes state; each only changes
  the probability that a proposed pixel copy is accepted. A Plexus operator returns a delta the
  engine integrates. Whether that gap is expressible in the algebra is the open question, and the
  normalizer prompt tells agents not to paper over it.
- **Five architectural mechanisms no scan can see** — `cell_as_lattice_domain`,
  `metropolis_acceptance`, `energy_sum_composition`, `mcs_time_unit`, `pixel_neighbourhood`. Only
  the first is `new` (`occupy`); the other three came back **out of scope**, the same verdict
  jax-morph's four architectural entries received. *Two very different architectures, both scoring
  zero new biological vocabulary.*
- **`growth_mitosis` is the direct comparison point** with jax-morph's `cell_divide`: same biology,
  structurally different mechanism (target-volume ramp + volume threshold + geometric bisection,
  versus a Bernoulli draw on a per-cell rate + volume-conserving offset).
- **Observables must be distributional.** A Potts model is a Metropolis chain, so a matched
  trajectory is not a meaningful object — the six reference runs in `log/atlas_cc3d/` already use
  population/topology statistics, which is the currency Phase 4 will need.

---

## 5. What exists on disk

```
oracle.py          CC3D in its own env, with provenance; `verify` = isolation + headless + determinism
inventory.py       seeds the record from cc3d.core.PyCoreSpecs (30) + 5 architectural, by hand
demos.py           one spec-builder per mechanism, each with its ABLATION
evidence.py        runs both arms and renders strip/movie/metrics into log/atlas_cc3d/
phase1_fill.py     the 11 entries substantiated before the excavator ran
atlas_record.yaml  35 mechanisms, all normalized
campaign/notes/    25 mechanism notes
log/atlas_cc3d/    6 mechanisms × {on,off} with strip.png, movie.mp4, metrics.png, metrics.json
```

**These are REFERENCE runs, not Plexus runs** — every artefact says so on its face. No Plexus
operator has been written for CompuCell3D.

---

## 6. Six mechanisms, each beside its ablation

| mechanism | observable | ON | ABLATED |
|---|---|---|---|
| `contact_adhesion` | heterotypic boundary | 290 → 167 (−42%) | 290 → 388 (+34%) |
| `volume_constraint` | mean volume | 25 → 59.4 (+138%) | 25 → 0 (−100%) |
| `surface_constraint` | mean perimeter | 20 → 19.3 (−3%) | 20 → 22.8 (+14%) |
| `chemotaxis` | mean x (source at wall) | 15 → 18 (+20%) | 15 → 14.8 (−1%) |
| `growth_mitosis` | live cells | 37 → 296 (+700%) | 37 → 37 (0%) |
| `external_potential` | mean x | 32 → 57.6 (+80%) | 32 → 32.2 (+0.5%) |

Several ablations *reverse* rather than merely weaken: sorting becomes mixing, perimeter grows
instead of shrinking, and without a volume constraint the cells dissolve entirely.

---

## 7. Open, and honest about it

- **No verdict here has been validated against the reference.** Phase 4 is deferred, so the 8 new
  contracts are claims about our reading of the source.
- **4 mechanisms are blocked** (`_state/blocked.json`), each with its reason; 8 earlier blocks were
  cleared with evidence logged to `_state/unblocked.jsonl`.
- **`steadystatediffusionsolver`** was the last to normalize and is worth re-reading: it is the
  strongest cross-repository match in the whole programme.
- The three upstream CompuCell3D defects (`service_cc3d` specs-list, steppable globals, output
  directory) are routed around in `oracle.py`, not fixed upstream.
