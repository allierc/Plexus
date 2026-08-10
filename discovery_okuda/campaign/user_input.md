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

- `interface_line_tension_3d` -- the purse-string. **IN** the vocabulary. It has no `K_extrude` key
  at all, so the forcing is absent rather than defaulted to zero.
- `extrusion_forcing_3d` -- the forcing alone, under a name that says so. **NOT** in
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

`apoptosis_3d` is the Die family -- the first mechanism here that deforms the sheet **inward**
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
- **`add_op` has fired 30 times in this project's history and all 30 added the same operator**, none
  since round 24. An operator nothing has exercised answers a question no retune can.
