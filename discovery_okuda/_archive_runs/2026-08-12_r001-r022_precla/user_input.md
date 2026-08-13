# User input

Written by Cedric, read fresh by the Proposer and the Analyst at the top of every round.
Edit this file mid-campaign -- the next round picks it up, no relaunch needed.

*Note, 10 August 2026: for the first 28 rounds that sentence was false. `crew/flow.yaml` declared
this file as an input to both roles and neither `crew/proposer.py` nor `crew/analyst.py` ever read
it, so nothing written here reached anyone. It is wired now, and `load_flow` refuses to start if a
declared input is not genuinely read -- so this channel cannot go silent again without the loop
failing loudly. The previous contents are archived in
`_archive_runs/2026-08-10_r001-r029_campaign/user_input.md`.*

---

## 1. The objective: FOUR morphologies at once, not one

Explore **tube forming, budding, branching and complex shape together.**

This is a change of objective and it is deliberate. The previous campaign ranked candidate parents
by `grip_peak` then `protr_peak` -- one scalar, greedy, no diversity term. One scalar can only
climb one hill. Measured over r001-r029: **18 distinct compositions across 196 structural records,
one composition proposed 87 times, `r020_06` the parent of 33 slots**, and for four consecutive
rounds the top six parents were five clones of one result at protr_peak 1.595.

The parent set is now a **portfolio with a reserved seat per target** (`crew/flow.yaml`,
`parents.args.targets`), plus a lineage cap so no family can take the table:

| target | what it is | scored on |
|---|---|---|
| `tube` | a sustained finger | `protrusion_aspect_max_peak`, `n_tubes_peak` |
| `bud` | a BULGE that is not yet a finger -- aspect counts AGAINST it | `protr_peak`, minus `protrusion_aspect_max_peak` |
| `branched` | more than one tip on a sustained protrusion | `n_tips_peak`, `protr_peak` |
| `complex` | undulation: many protrusions gripping, no dominant finger | `grip_peak`, `invagination_peak` |

**A run is not a failure for being the wrong shape.** A bud is not a bad tube. Each target holds its
own seat and is judged by its own figure of merit, so a lone branched specimen is a parent even when
every high-grip run is a sphere.

`morphology` is now shown in the parent block. It is `admitted = False` -- do NOT rest a prediction
on a classifier's label, rest it on a number -- but use it to see which target a parent serves.

## 2. Start from scratch

Nothing is inherited. `records.jsonl` is empty, `knowledge.md` is empty, and the parent set seeds
from the 16-member basis pool. The old ledger, its knowledge and the graph that produced it are in
`_archive_runs/2026-08-10_r001-r029_campaign/`, including `flow.yaml.asrun` -- that campaign's
behaviour was a property of its graph, and reading the records without it would blame the agents.

## 3. There is no push force any more, by construction

`rd_interface_tension` carried two terms under one name: `K_purse`, an ordinary line tension on the
activator interface, and `K_extrude`, an energy that FALLS as activated cells move outward -- the
answer written into the objective. They are two operators now:

- `interface_tension` -- the purse-string. **IN** the vocabulary. It has no `K_extrude` key
  at all, so the forcing is absent rather than defaulted to zero.
- `interface_push` -- the forcing alone, under a name that says so. **NOT** in
  `composition_space`. No edit you can write reaches it.

**Do not spend slots arguing about whether a result was forced.** It cannot have been. `K_extrude`
measured 0.0 in all 78 specs that ever carried the old operator -- nothing this project ever ran was
forced -- and the Grounder still called r028 "the same extrude-forced star for a fourth round", on
runs whose specs contain no such operator at all. That same verdict was already retracted once, for
`r017_07`, in section 3 of the previous version of this file.

The measured stand-in (`mech_p_ratio > 2.0`) is **off**. It demoted 11 runs to last place
permanently and not one was ever used as a parent -- among them `r019_07`, protr_peak **1.817**,
grip_peak **0.294**, the highest of both in the whole campaign, against a control ceiling of
1.595 / 0.262 that four rounds could not beat. The best result the loop ever produced was
structurally unreachable.

## 4. Cell death is in the vocabulary AND in the basis

`cell_die` is the Die family -- the first mechanism here that deforms the sheet **inward**
(Monier et al. 2015: apoptotic force drives epithelial folding). Every other operator pushes
outward, which is why invagination was never reached.

It was injected before round 25 and **never once chosen in five rounds**, while sitting on the menu
and being named by `coverage` as the only operator never exercised. Route A could not reach it
either -- `_build_sweep` refuses a base that does not already carry the operator. So it is now in
the **basis**: one death twin per Route A base (`b_*_death`), which makes `mode` and
`max_mark_frac` sweepable ladders.

`max_mark_frac` is the knob that matters. It caps how much of the tissue may be under sentence at
once, so the mode chooses WHO dies and the cap chooses HOW FAST. Uncapped, five of six modes
destroyed the best parents (r020_00_ctrl: protr 1.513 -> 1.131, grip 0.228 -> 0.049, 1,660 of 7,424
cells dead). At 0.005 all six finish within a few percent of the parent with no premise broken.

## 5. What the loop still cannot do -- named so you do not mistake these for your own failures

- **No serendipity organ.** Nothing notices a result that was interesting but not what the round was
  testing. `hypothesis.py` implements one -- surprise rate, novelty yield, an intent-mix band -- and
  is imported by nothing.
- **No global hypothesis.** Every proposal is `(parent, edit, one metric, one threshold)`. A claim
  spanning runs -- "grip is set by the diffusion RATIO, not by `d_a`" -- cannot be posed or scored.
- **No quota.** `intent` is free text and nothing counts it. If you label five replicates
  `adversarial`, nothing objects.
- **Prior knowledge is one-way.** `_premises_raw.md` holds 70 literature-sourced facts; 11 became
  gates and 59 are unread. Among them: *tissues stop growing when compressed; a growth law that
  reads only a chemical signal has no mechanism by which it can ever stop* -- a direct diagnosis of
  this project's 30,743-cell overshoot.

## 6. Two habits the record says to break

- **Duplication is currently free.** A refused duplicate is re-admitted at a fresh seed, relabelled
  `replicate`, and scored against the copied prediction -- so re-proposing a known experiment
  survives and inflates the confirm rate. It was 5 of 7 Route B slots in r028 and 4 of 5 in r027.
  Replicates bound the seed floor; two per round is plenty.
- **The structural slot is `set_impl` now, not `add_op`.** `add_op` fired 30 times in the previous
  campaign and 20 in this one, and all 50 added the same operator. That is not laziness: every
  operator in the vocabulary has now been exercised, so there is nothing left for `add_op` to add,
  and `coverage.operators_never_exercised` is empty — which made the coverage block read as
  *satisfied* while `set_impl` had fired **once** in 196 runs against `add_op`'s 27, and 11 of 25
  implementations had never run. `coverage.the_untried_edit` now says which of the two is live.

## 7. The star, and the levers that were sitting unused behind it

`r013_05` is the only run in 196 with **arms** rather than lobes: 11 tubes, `reduced_volume` 0.285
(deepest on file, 1.0 being a sphere), `grip` 0.273 and `invagination` 0.617 (both campaign
records), activator at the tips 7.4× the cell mean. Two edits from `b_gs_shaping_soft_lo`: `K_V` 80
makes a cell nearly incompressible so growth has to buckle the sheet instead of inflating one cell,
and `cell_chem_react.rate` 0.5 halves the reaction so spots stay 4.04 cells apart and each gets its own
arm. It is now a **basis member** (`b_star`), with eight variations, because a Route B leaf can
carry no ladder — `_build_sweep` refuses a base that is not in the basis, so no sweep had ever been
run on the one composition that makes the shape the campaign is looking for.

**It scored zero on the seat meant for it.** `protrusion_aspect_max_final` reads 0.0 and
`n_tips_final` 0 on eleven visible arms, and the `tube` seat's floor was `aspect_final ≥ 0.4`, so
r013_05 could not take it and got in through `complex`, on grip. That seat is keyed on `n_tubes`
now, which counts the arms at 11 — the same number the eye reports. This is standing law L3, filed
since r006: *"the metric zeroes tips the picture shows."*

**What had never been run**, all now in the basis and all reachable by `set_impl`:

| mechanism | had run | why it is a different shape |
|---|---|---|
| `cell_divide:orient_iface` | 0 of 282 specs | Hertwig's rule makes compact daughters and therefore a lobe; orienting the septum along the bud axis stacks daughters ALONG the arm. This is how an arm becomes a tube. |
| `cell_chem_from_shape:tension` / `:pressure` / `:apical_area` | 0 of 275 | the chemistry has only ever read **curvature**. A tissue that patterns where it is stretched and one that patterns where it is bent are not the same object. |
| `cell_chem_from_shape.beta < 0` | 0 | the operator's own docstring: "the sign is a real hypothesis, not a convention, and both must be swept". Negative = pattern AVOIDS the deformation, so it should dimple where it patterns. |
| `cell_chem_seed:patch` / `:noise` | unreachable | two of the engine's four seeding modes that the vocabulary did not offer at all. `patch` starts from ONE domain; `noise` seeds nothing and lets the instability pick the wavelength, which is the only setting in which "the chemistry chose this many spots" is a result rather than a setting. |
| `cell_chem_seed:cones` | 3 of 308, none here | Okuda's own Fig 5 setup: N fixed radial cones, so "how many tubes" becomes a controlled variable. |

## 8. Two threshold bugs, and the class they belong to

Every threshold on a chemistry field in this substrate is relative to that field's own maximum —
`interface_tension.a_sw`, `interface_push.a_sw`, `cell_die` chem_low. Two were
not, and both silently weakened a mechanism rather than failing:

- **`cell_divide`** built its bud axis from the top `orient_asw` **fraction** of the field and then
  tested each cell against `orient_asw` as an **absolute** value. Different sets; and on any run
  whose activator peaks below `orient_asw` the axis exists while no cell passes, so the operator
  was behaviourally `hertwig` with nothing saying so. Fixed — `b_star_oriented` is the first run of
  the operator as written.
- **`cell_grow.a_sw`**, the campaign's main growth gate, is **absolute**, while `crew/basis.yaml` and
  the inhibitor branch beside it both describe it as a fraction of the maximum. Measured over 154
  runs the gate sat at 0.24 of the field on the strongest and above all of it on 16. Not silently
  redefined — both readings are real mechanisms (absolute stops growing when the chemistry dies;
  relative means the same thing in every run, which is what a sweepable lever must do), so
  `a_sw_rel` makes the choice explicit and `b_star_relgate` is the member that measures it.

The boxes were wrong too. `K_V`, `Lambda`, `l_th_frac` and `cell_grow.rate` were flagged
`out_of_range` on **all 196 runs and every basis member**, because each box excluded the value the
project actually runs at — so the sweep menu, which takes its two points from the box, could not
propose the region the best result lives in. Widened, with the sampling basins pinned to the old
widths so no robustness claim changes meaning. `out_of_range` now means something again.
