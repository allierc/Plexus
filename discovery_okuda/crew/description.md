# The DESCRIPTION SLOTS

Two roles fill this form and neither may see the other's answer: the **Forecaster** fills it from
the spec before the job is launched, the **Eye** fills it from the frames after the job has landed.
`foresight.py` compares them slot by slot.

This file is the schema. Editing it changes what both roles write and what the comparison scores —
so the two can never drift apart, which is the only reason the schema lives outside both of them.

## Why a form and not a paragraph

Because a paragraph scores as one number and that number says nothing. Two descriptions of the same
run come back 0.72 similar and the loop learns neither what it got right nor what it got wrong.
Filled in slots, a miss is *located*: `count` predicted 6 and saw 11 names the mechanism that was
misunderstood, and `chem` right while `form` wrong is a different failure from the reverse.

## Why the form is short

Because a long description is a wide net and a wide net always catches something. The word limits
are not a courtesy to the reader — they are what forces a commitment. A slot that may run to thirty
words will hedge across every outcome and score well against all of them, which is the linguistic
version of the objection R7 raises against an effect smaller than the seed floor: a claim that
cannot fail is not a claim.

## How the anchors work

Each slot below carries an **anchors:** line. Those words, and only those words, are what
`foresight.py` compares the two answers by: it takes the anchors each answer names, and scores the
overlap divided by the larger of the two sets. Everything else in your sentence is for the human
reading it.

They are a MENU, NOT A CLOSED SET. A body this campaign has never seen will not have a word here,
and forcing it into one would repeat the morphology classifier's mistake -- it returned `sphere` for
314 of 416 runs, including an eleven-armed star. If none of them fits, write the phrase that does
and accept that the slot scores on raw word overlap instead.

Naming three anchors where the other role named one scores 0.33, not 1.00. Hedging across several
answers used to match all of them; it no longer does. Write the words you mean.

ONLY THE `anchors:` LINE IS READ, which is why it exists: the parser used to take every backticked
word in a section, so `form` -- named in the topology section's explanation of what it took off
`form` -- became an anchor of `topology`, and any answer containing the word "form" matched.

## The form

Six lines, in this order, each `key: value`. Nothing else — no preamble, no bullet list, no closing
remark. A line over its word limit is truncated at the limit, not rejected, so the discipline is on
the writer rather than on the parser.

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
surface is arranged. The suggested words are `sphere`, `lobed`, `star`, `tube`, `branched`,
`folded`, `collapsed`, `sheet`. They are a starting vocabulary and not a closed set — a body this campaign has not seen
before will not have a word here, and forcing it into one would be the classifier's mistake all over
again (`morphology` returned `sphere` for 314 of 416 runs, including an eleven-armed star). If none
of the words fits, write the phrase that does.

### topology


**anchors:** `convex` `concave` `invaginated` `evaginated` `budding` `tube` `finger` `lobe` `bud` `branched` `bifurcated` `undulating` `rippled` `pinched` `constricted` `necked` `detached` `fragmented` `holed` `fenestrated` `open` `closed` `sealed` `radial` `bilateral` `asymmetric` `irregular`

Added 13 August, at Cedric's request: *"I would like also overall topology of the spheroid:
branches, ondulation, tubes, concave/convex what more?"*

WHAT IT TAKES OFF `form`. `form` was carrying two questions and answering neither cleanly -- the
body's shape (is it round, is it long) and its arrangement (does it go in or out, is it one piece).
Both roles were putting the same words in one slot, which is why `form` read 1.00 on all six basis
pairs before the set-overlap fix: a slot answering two questions matches on either.

So `form` is now the OUTLINE -- what the silhouette looks like -- and this slot is the STRUCTURE.

The anchors, in the four groups that matter for a closed epithelial shell:

- **which way the sheet went** -- `convex`, `concave`, `invaginated`, `evaginated`, `budding`.
  NOT the same axis as concave/convex, and this is the one Okuda's paper is about: a deep inward
  pocket has convex walls all the way down, so local curvature and direction-of-travel are
  independent answers and both are worth a word.
- **what the outgrowths are** -- `tube`, `finger`, `lobe`, `bud`, `branched`, `bifurcated`,
  `undulating`, `rippled`. `branched` means an arm that itself divides; a body with eleven arms
  from one centre is not branched, it is radial, and the campaign has never once produced the
  former.
- **connectivity, which is the real topology** -- `pinched`, `constricted`, `necked`, `detached`,
  `fragmented`, `holed`, `fenestrated`, `open`, `closed`, `sealed`. A through-hole is the one
  genuine invariant in this list; `pinched` is the state just before a piece leaves; `open` and
  `closed` distinguish a lumen that reaches the outside from a blind sac, which are different
  organs built by the same operators.
- **symmetry** -- `radial`, `bilateral`, `asymmetric`, `irregular`. The flower is eleven-fold
  radial and nothing in the schema could say so.

Write two or three of them. A sphere's honest answer is `convex, closed, radial` and that is a
measurement, not a non-answer.

### count


**anchors:** none — this slot is not compared by words.

**Of the feature named in `form`** — arms if star, lobes if lobed, branches if branched. `0` if the
body has no repeated feature, which is the correct answer for a sphere and must be written rather
than left blank. A range is allowed and is not a hedge: eleven arms of which three are stubby is
honestly `8-11`, and both roles are better served by that than by a false integer.

### surface


**anchors:** `smooth` `ruffled` `creased` `dimpled` `pitted` `ragged`

`smooth`, `ruffled`, `creased`, `dimpled`, `pitted`, `ragged`. This is the shell's texture, not its
shape: a sphere can be ruffled and a star can be smooth.

### chem


**anchors:** `tips` `troughs` `crests` `spots` `stripes` `patches` `uniform` `banded` `absent` `scattered` `pole` `first` `second` `co-located` `anti-correlated` `complementary` `segregated` `overlapping`

Where the red is, **relative to the shape** — `at the tips`, `in the troughs`, `uniform`, `banded`,
`one spot`, `absent`. Not how much of it there is. The relation is the informative part: activator
at the tips of arms and activator in the troughs between them are opposite mechanisms that produce
the same amount of red.

### A SECOND MORPHOGEN, WHEN THERE IS ONE

Cedric, 13 August: *"at some point we will add a green morphogen, prepare it already"* -- and then,
correcting it the same day: *"note the greens are the cell division, so I'm messing up with color...
we will change when the two morphogens color are set up."*

**SO THE ANCHORS NAME NO COLOUR, AND THAT IS THE POINT.** They are `first` and `second` — the
species, not the paint. The colour a species is drawn in is a RENDERER decision that has already
changed once and is about to change again; the relation between two morphogens is a fact about the
tissue. Binding the schema to a colour would mean that re-mapping the palette silently re-scores
every run in the campaign, and the anchors would still look correct.

**GREEN IS NOT A CHEMICAL.** On the tissue, a green tint marks a cell that has RECENTLY DIVIDED --
`caption_wave.py`'s layout constant has said so all along, and the Eye has never been shown it. Do
not put green in `chem`. A green cell belongs in `time` (division is still happening) or in `free`.

No second slot for the second species, deliberately. A slot that does not apply yet is a slot both
roles fill with "absent", agree on for free, and inflate the score with — so `chem` describes
whatever chemistry is visible, one species or two, in one line. With one, say where it is. With two,
say where each is and **how they relate**, because that relation is the whole content of a
two-species system: Gray-Scott's activator and inhibitor are anti-correlated by construction, and a
run where they come out co-located has broken something the campaign believes.

    chem: at the tips, one spot per arm
    chem: first at the tips, second in the troughs, anti-correlated
    chem: first scattered spots, second uniform

WHEN THE COLOURS ARE SETTLED, this section gets one line naming which colour is which species, and
the anchors do not change.

AND THE RENDER AUDIT OF 13 AUGUST SAYS THE COLOURS CANNOT SIMPLY BE CHOSEN. The hue circle on this
artefact is full: red-to-maroon is the activator ramp, amber and yellow are the branch and tip
classes, green is recent division, teal is growth inhibition, blue is marked-to-die and body-class,
magenta is the alarm, black is background and cell edges, white is low activator. The only
unclaimed band is a narrow deep violet, confused with the maroon top of the activator ramp at low
luminance and with the alarm magenta at high. So **a second morphogen needs its own ROW, not its own
hue** -- and the thing that has to change to give it one is the `[:3]` slice at `run_one.py:1389`,
which currently discards `act_b` on every render call. See `crew/strip.md`.

### time


**anchors:** `early` `halfway` `late` `throughout` `grows` `holds` `arrests` `collapses` `oscillates` `never`

When the shape arrives and whether it arrests: `grows throughout`, `arrests near the end`,
`appears halfway then holds`, `collapses after forming`. The campaign's runs are 1800 frames and a
shape that is transient is a different result from one that survives — standing law L3 exists
because a peak and a final value were read as the same thing for six rounds.

THIS SLOT HAD NO ANCHORS UNTIL 13 AUGUST and every one of its phrases
was multi-word, so it carried no vocabulary at all and fell through to raw word overlap: measured
across six basis pairs it scored 0.00, 0.08, 0.09, 0.10, 0.14 and 0.18 -- a slot that read as near-
total disagreement on every run in the campaign, including the ones where both roles plainly agreed.

### free


**anchors:** none — this slot is not compared by words.

Unscored, and **mandatory**. This is where an observation that the schema has no slot for goes:
*"like a flower"*, *"a piece detached and floated away"*, *"the cross-section inset is a rendering
artifact"*. Every one of those was worth more than the run's metrics on the day it was written, and
none of them fits a form.

Scoring it would destroy it — a scored free slot is a slot the writer games toward whatever the
other role is likely to say. It is carried onto the record, read by the Analyst, and never compared.

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

## How the comparison works, so both roles know what they are being read for

`count` compares as numbers with overlap allowed. `form`, `topology`, `surface`, `chem` and `time`
compare as short phrases -- by how much their ANCHOR WORDS overlap, divided by the larger of the two
sets. `free` does not compare.

That divisor is why naming three anchors where the other role named one scores 0.33 and not 1.00:
hedging across several answers used to match all of them. Write the words you mean.

Neither role is scored on being *agreeable*. The Eye is not shown the forecast and the Forecaster is
not shown the run, so there is no way to converge except by both being right about the tissue —
which is the entire point. A forecast that hedges toward the campaign's average run will match a
sphere and miss everything worth finding, and an Eye that reports the campaign's average run will
score well while seeing nothing.
