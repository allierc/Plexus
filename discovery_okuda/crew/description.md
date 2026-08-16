# The DESCRIPTION SLOTS

Two roles fill this form and neither may see the other's answer: the **Forecaster** fills it from the
spec before the job is launched, the **Eye** fills it from the frames after the job has landed.
`foresight.py` compares them slot by slot.

This file is the schema. Editing it changes what both roles write and what the comparison scores —
which is the only reason the schema lives outside both of them: so they cannot drift apart.

## Why a form, and why a short one

A paragraph scores as one number and that number says nothing: two descriptions come back 0.72
similar and the loop learns neither what it got right nor what it got wrong. In slots, a miss is
*located* — `count` predicted 6 and saw 11 names the mechanism that was misunderstood, and `chem`
right with `form` wrong is a different failure from the reverse.

The word limits are not a courtesy to the reader; they force a commitment. A slot that may run to
thirty words will hedge across every outcome and score well against all of them — the linguistic
version of an effect smaller than the seed floor. A claim that cannot fail is not a claim.

## How the anchors work

Each slot carries an **anchors:** line. Those words, and only those words, are what `foresight.py`
compares by: it takes the anchors each answer names and scores the overlap divided by the larger of
the two sets. Everything else in your sentence is for the human reading it.

They are a MENU, NOT A CLOSED SET. A body the campaign has never seen will not have a word here, and
forcing it into one repeats the morphology classifier's mistake — it returned `sphere` for most of a
campaign, including an eleven-armed star. If none fits, write the phrase that does and accept that
the slot scores on raw word overlap instead.

Naming three anchors where the other role named one scores 0.33, not 1.00. Write the words you mean.

## The form

Seven lines, in this order, each `key: value`. Nothing else — no preamble, no bullets, no closing
remark. A line over its limit is truncated, not rejected: the discipline is on the writer.

```
form:     <= 8 words    the body's overall shape at the END of the run
topology: <= 8 words    how the surface is ARRANGED -- in/out, connected, holed, symmetric
count:    an integer, or a range like 6-9, or 0    how many of the salient repeated feature
surface:  <= 6 words    the texture of the shell
chem:     <= 8 words    where the activator sits RELATIVE TO the geometry
time:     <= 10 words   when the shape appears, and whether it stops
free:     <= 25 words   anything else. NOT SCORED.
```

### form

**anchors:** `sphere` `lobed` `star` `tube` `branched` `folded` `collapsed` `sheet` `elongated` `flat`

THE SILHOUETTE ONLY, since `topology` was split out below: what the outline looks like, not how the
surface is arranged.

### topology

**anchors:** `convex` `concave` `invaginated` `evaginated` `budding` `tube` `finger` `lobe` `bud` `branched` `bifurcated` `undulating` `rippled` `pinched` `constricted` `necked` `detached` `fragmented` `holed` `fenestrated` `open` `closed` `sealed` `radial` `bilateral` `asymmetric` `irregular`

WHAT IT TAKES OFF `form`, which was carrying two questions and answering neither cleanly: the body's
shape (round or long) and its arrangement (in or out, one piece or several). A slot answering two
questions matches on either, and scored 1.00 everywhere as a result. `form` is the OUTLINE; this is
the STRUCTURE. Four groups matter for a closed epithelial shell:

- **which way the sheet went** — `convex`, `concave`, `invaginated`, `evaginated`, `budding`. Not the
  same axis as concave/convex, and this is the one Okuda's paper is about: a deep inward pocket has
  convex walls all the way down, so curvature and direction-of-travel are independent answers.
- **what the outgrowths are** — `tube`, `finger`, `lobe`, `bud`, `branched`, `bifurcated`,
  `undulating`, `rippled`. `branched` means an arm that itself divides; many arms from one centre is
  `radial`, not branched.
- **connectivity, the real topology** — `pinched`, `constricted`, `necked`, `detached`, `fragmented`,
  `holed`, `fenestrated`, `open`, `closed`, `sealed`. A through-hole is the one genuine invariant
  here; `pinched` is the state just before a piece leaves; `open` and `closed` separate a lumen that
  reaches outside from a blind sac — different organs from the same operators.
- **symmetry** — `radial`, `bilateral`, `asymmetric`, `irregular`.

Write two or three. A sphere's honest answer is `convex, closed, radial`, and that is a measurement.

### count

**anchors:** none — this slot is not compared by words.

**Of the feature named in `form`** — arms if star, lobes if lobed, branches if branched. `0` if there
is no repeated feature, which is the correct answer for a sphere and must be written rather than left
blank. A range is allowed and is not a hedge: eleven arms of which three are stubby is honestly
`8-11`, and both roles are better served by that than by a false integer.

### surface

**anchors:** `smooth` `ruffled` `creased` `dimpled` `pitted` `ragged`

The shell's texture, not its shape: a sphere can be ruffled and a star can be smooth.

### chem

**anchors:** `tips` `troughs` `crests` `spots` `stripes` `patches` `uniform` `banded` `absent` `scattered` `pole` `first` `second` `co-located` `anti-correlated` `complementary` `segregated` `overlapping`

Where the red is, **relative to the shape** — not how much of it there is. The relation is the
informative part: activator at the tips of arms and activator in the troughs between them are
opposite mechanisms producing the same amount of red.

**THE ANCHORS NAME NO COLOUR, AND THAT IS THE POINT.** They are `first` and `second` — the species,
not the paint. Which colour a species is drawn in is a RENDERER decision that has changed before;
the relation between two morphogens is a fact about the tissue. Binding the schema to a colour would
mean re-mapping the palette silently re-scores every run in the campaign.

**GREEN IS NOT A CHEMICAL** — on the tissue it marks a cell that has RECENTLY DIVIDED. Do not put
green in `chem`; a green cell belongs in `time` (division is still happening) or in `free`.

No second slot for a second species, deliberately: a slot that does not apply yet is one both roles
fill with "absent", agree on for free, and inflate the score with. With one species, say where it is;
with two, say where each is and **how they relate**, because that relation is the whole content of a
two-species system.

    chem: at the tips, one spot per arm
    chem: first at the tips, second in the troughs, anti-correlated
    chem: first scattered spots, second uniform

A second morphogen needs its own ROW in the render, not its own hue — the hue circle on this
artefact is already full. See `crew/strip.md`.

### time

**anchors:** `early` `halfway` `late` `throughout` `grows` `holds` `arrests` `collapses` `oscillates` `never`

When the shape arrives and whether it arrests: `grows throughout`, `arrests near the end`, `appears
halfway then holds`, `collapses after forming`. A shape that is transient is a different result from
one that survives, and a peak read as a final value has misled this campaign before.

### free

**anchors:** none — this slot is not compared by words.

Unscored, and **mandatory**. Where an observation the schema has no slot for goes: *"like a flower"*,
*"a piece detached and floated away"*, *"the cross-section inset is a rendering artifact"*. Each of
those was worth more than the run's metrics on the day it was written, and none fits a form.

Scoring it would destroy it — a scored free slot is one the writer games toward whatever the other
role is likely to say. It is carried onto the record, read by the Analyst, and never compared.

## Worked example

```
form:     eleven thin arms from a central body
topology: evaginated, radial, closed, unbranched
count:    11
surface:  smooth
chem:     at the tips
time:     appears halfway, still growing at the end
free:     like a flower; the arms are all in one plane, which may be the seeding
```

## How the comparison works

`count` compares as numbers, with overlap allowed. `form`, `topology`, `surface`, `chem` and `time`
compare by anchor-word overlap divided by the larger of the two sets. `free` does not compare.

Neither role is scored on being *agreeable*. The Eye is not shown the forecast and the Forecaster is
not shown the run, so there is no way to converge except by both being right about the tissue —
which is the entire point. A forecast that hedges toward the average run matches a sphere and misses
everything worth finding; an Eye that reports the average run scores well while seeing nothing.
