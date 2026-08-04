# Campaign memory

## Abstract

This campaign builds the causal lever-map of Okuda's mechanism space and, as a by-product,
seeks his four morphologies (protrusion/tube/branch/bud) from operator compositions on a
blastula shell. Six rounds (~40 runs) have established the two obvious drivers are dead ends —
UNIFORM GROWTH either stretches the confluent sheet (P7) or explodes it past protr_peak 2 into
an invalid non-tissue MECHANICALLY (base-independent), CHEMISTRY is wholly INERT for shape,
and SLOW growth stays valid but still returns a sphere. The block is PROCEDURAL and the loop is
DEGENERATING: the one untried cell — a LOCALISED, anisotropic force driver — has been the named
frontier for 4 rounds and is STILL UNRUN; R5 AND R6 each spent their single run on the sphere
control, buying no science two rounds running.

## What is ESTABLISHED
- "Uniform growth explosion is mechanical, not chemical" — SUPPORTED by R2/R3 add_op
  vesicle_growth uniform_ramp on plain/gierer-gray/reaction-swapped bases, protr_peak=2.266
  identical across all. Specimen INVALID on P7/P11/P5b every time. Falsifiable by: a base
  whose kinetics change the growth endpoint.
- "Chemistry is inert for shape on the mechanics shell" — SUPPORTED by every set_impl
  react/diffuse/seed and every chemistry remove_op, protr_peak=1.006, mech_p_tube=0, R2–R3;
  includes remove cell_rd_seed0 → sphere (seeding not necessary). Falsifiable by: any
  chemistry-only edit reaching protr_peak>1.2 or mech_p_ratio>0.
- "divide_3d hertwig deforms the shell BASE-INDEPENDENTLY" — SUPPORTED by R3,
  interface_weighted ≈ graph_laplacian ≈ gierer+gray, same relief-path response. Falsifiable
  by: a base that changes the division deformation.
- "Growth-only edits do NOT protrude — stretch or self-intersect" — SUPPORTED, P7 broke 4/7
  R3, 2/12 R2, 1/8 R4 (slow rate cuts the break rate but not to a valid protrusion).
  Falsifiable by: a growth-only edit at protr_peak>1.3 with P7 and P11 intact.

## What is OPEN
- Whether ANY edit bends the shell into a VALID protrusion (protr_peak>1.2 / mech_p_ratio>0,
  specimen valid): every localised/anisotropic-force edit is still UNRUN — never measured.
  Unsettled because it has never been PROPOSED, not because it was tried and failed.
- Whether the operator bank even CONTAINS a localised/anisotropic-force op (polar protrusion,
  apical constriction, one-sided tension). If it does not, that absence is the round's finding
  and must be reported — not papered over with another growth/chemistry re-run.
- wk_* growth family past frame ~304: still UNMEASURABLE while reservoir-censored; needs the
  destination-sized reservoir before its endpoint means anything.

## Known traps
- protr_peak read alone LIES: high protr_peak with an INVALID specimen (P7/P11/P5b) is an
  explosion, not morphology (R3 uniform_ramp 2.266). Guard: always apply the specimen gate.
- Uniform area/volume growth for protrusion → P7 stretch or P7+P11+P5b explosion (R2/R3).
  Guard: localised anisotropic force, not global growth; do not vary the chemical base under
  a growth op (inert).
- Growth that inflates rest-volume WITHOUT conserving solute → P4: activator diluted,
  chemistry non-physically quenched (broke 1/8 R4, first seen). Guard: conserve concentration
  under any growth op, or read shape before dilution erases the pattern.
- Chemistry-only edit expecting shape → inert, protr_peak=1.006 (R2/R3). Guard: don't propose.
- RE-RUNNING a known trap buys nothing: vesicle_growth uniform_ramp ran 3× (R2/R3/R3),
  divide_3d hertwig 4× (R2/R3/R3/R4) — all inconclusive on invalid specimens. Do not re-run.
- Growing volumes against a FROZEN shell target radius → P11 self-intersection. Guard: let the
  target radius track growth, or drop the radial spring.
- Growth faster than relaxation → P5b residual force climbs. Guard: slow the rate.
- Division ceiling below the trigger → P3b volume drift, divisions on timeout (2/7 R3, 1/8
  R4). Guard: vth_frac > factor; size the reservoir for the DESTINATION count.
- Prediction `unstated` → scores nothing. Guard: a clause `<metric> <op> <value>`.

## Frontier and parent
Breed from C138f409dbe0 (clean mechanics-only sphere baseline; protr_peak=1.006, uncensored).
The growth-driver AND chemistry families are EXHAUSTED for protrusion. The next parent must
carry a LOCALISED/anisotropic force operator — not more growth, not a chemistry swap. Division
deforms but base-independently — not a shape lever on its own.

## Stability envelope
Non-growing sphere holds at n=2000, genus 0, ray-crossing 1.0, shape-index min 3.659 (floor
3.545), activator finite ≥0 (R1–R3). A valid growth edit must keep residual-force ratio ~1
(P5b), rays crossing exactly once (P11), no long stretched shape-index tail (P7), and solute
conserved (P4). Uniform growth violates the first three once rate is high; slow growth (R4)
holds them but still returns a sphere. wk_* interpretable only before saturation (~304/401).

## Track A — the map
Chemistry (react/diffuse/seed impls, all variants) = INERT for shape. Uniform growth =
necessary for area but produces only stretch or explosion, mechanically/base-independently;
slow growth stays valid but does not protrude (R4). divide_3d = deforms base-independently
(relief-path), P3b-fragile. Localised/anisotropic force = UNTESTED, the one open cell. Every
VALID-protrusion cell of the map is still blank.

## Track B — the figure
0 of 4 Okuda morphologies achieved. Protrusion/tube/branch/bud: attempted-and-failed via
uniform growth (stretch/explode), via chemistry (inert), and via slow growth (valid but flat,
R4); a localised-force driver has NOT been attempted.

## Next action
FIRST confirm whether a LOCALISED-force operator (polar/oriented protrusion, apical
constriction, one-sided tension) EXISTS in the bank. If it does: propose ONE such edit — NOT
uniform growth, NOT a chemistry swap, NOT another control — reservoir sized for target, rate
slow so P5b/P11 hold, solute conserved so P4 holds, scorable prediction (protr_peak>1.3 or
mech_p_ratio>1) AND a required valid specimen. If NO such operator exists, REPORT THAT ABSENCE
as the finding — it ends the campaign cleanly. Do not spend a third round on a control or a
re-run. Changes once a first valid non-sphere lands, the frontier is falsified, or its absence
is confirmed.

HEADLINE: 6 rounds, ~40 runs, 0 morphologies — R5+R6 both single control runs; frontier still unrun, loop degenerating
