# User input

Written by Cedric, read fresh by the Proposer and the Analyst at the top of every round.
Edit this file mid-campaign -- the next round picks it up, no relaunch needed.

## Pending Instructions

### 1. Your K_purse conclusion is WITHDRAWN. The operator never ran.

You reported `rd_interface_tension.K_purse` as inert. It is not inert -- it has never fired, in
this campaign or the last one, and every statement resting on it is void rather than revised.

The gate is `red = a > a_sw * amax`, a fraction of the activator's OWN maximum. The operator's
default `a_sw` was **1.0** -- cells strictly ABOVE the maximum, the empty set by construction --
and the four `*_shaping` basis specs omitted `a_sw`, so they took that default. Route A then swept
K_purse over [0, 0.25, 3, 6] and got four runs identical to FOUR SIGNIFICANT FIGURES (grip 0.04216,
protr 1.087, 21 spots, 3267 cells) with `acted = 0` on every one. Four measurements of nothing.

This is the SECOND write-off of the same operator without running it; the first is in section 3
below, where `a_sw` was an absolute value against a field whose median maximum is 0.000. That fix
made it a fraction and left a default that is a fraction of one.

WHAT K_purse IS, because it matters to your open problem. rd_interface_tension carries
`E = K_purse * sum_interface(edge length) - K_extrude * sum_red(a*r)`. K_purse is a LINE TENSION on
the red/white boundary -- a purse-string that contracts the ring around an activated patch. It is
how an epithelium necks a bud into a tube, and your standing result is that you make fat buds and
not tubes. K_extrude is the other half, an energy that falls as red cells move outward; it stays at
zero and a run carrying it above zero is a control, not evidence.

REPAIRED, and both places, because either alone leaves the trap: the operator's default is now 0.6
and the four `*_shaping` bases write `a_sw: 0.6` explicitly. Their runs have been re-measured, so
the basis you build on now has a purse-string that fires. The four void sweep records are deleted
from `records.jsonl` (backup in `campaign/_archive/`), so the ladder is OPEN and Route A will offer
it again. Sweep it as if for the first time, because it is.

### 2. You now have an instrument that can see INWARD: `invagination`.

Every shape metric you have measures outward excursion -- `protr_peak`, `protrusion_aspect_max`,
`n_tubes`, `act_at_tip` -- or the whole body -- `reduced_volume`, `gyr_prolate`. Invagination is one
of Okuda's three morphologies and you had no way to detect one. Twenty rounds could have produced a
pit and reported a sphere.

`invagination` is how far the deepest dimple sits below the tissue's own radius, as a fraction of
it, measured on cell radii smoothed over ring neighbours so one squeezed cell is not a dimple.

READ IT AGAINST A CONTROL, NOT ABSOLUTELY. A quiet 2,000-cell vesicle reads about 0.019 after 600
frames -- that is mesh roughness, not morphology. Measured on the apoptosis smoke runs, ordered by
how many cells were removed: control 0.0189, one death 0.0194, a 76-cell cap 0.0297, a 278-cell
band 0.0307, nine bands 0.0461.

### 3. `mech_p_ratio` is NOT a forcing test when no forcing operator is present. Retract the verdict.

For four rounds your headline has been "protr and grip past the wall are bought with forcing, not
growth", and for four rounds the run it rests on has carried no forcing term at all.

**`r017_07` has no `rd_interface_tension` operator in its spec.** There is no `K_extrude` in it to
set, at any value. The structural test agrees: `_is_forced(r017_07)` returns False, and the same
for `r014_01` and `r017_02`. Nothing is pushing these runs. `mech_p_ratio` is a PROXY -- the ratio
of pressure in the protrusion to pressure in the body -- and round.py's own note says why it is
kept only alongside the structural test: "the proxy did not fire" on a run that WAS forced, and it
can equally fire on a run that is not.

So two of your standing conclusions are wrong as stated:

- **"the grown wall stands, no unforced run beats protr 1.204"** -- `r017_07` at protr 1.588 and
  grip 0.198 IS unforced. The wall you have been reporting for fifteen rounds does not exist at
  1.204; it is at least 1.588.
- **"forcing stays linear past protr 1.5"** -- the correlation you fitted (ratio 2.284/2.025/1.958
  -> protr 1.588/1.268/1.453) is between two consequences of something else, not cause and effect.

### 4. What that something else is, and the control it needs.

`r017_07` differs from its parent `r014_01` (protr 1.453, n_tubes 0) by ONE edit:
`remove_op reconnect_t1_3d0`. T1 transitions are gone, so cells cannot exchange neighbours. With
topology frozen, pressure has no way to equilibrate between a bud and the body -- which raises
`p_tube/p_body` mechanically, with no external force anywhere. That is the likelier reading of
`mech_p_ratio` 2.284, and it is testable.

**Do not credit any protrusion measured without `reconnect_t1_3d` until this is controlled.** A
protrusion on a mesh that cannot rearrange may be a stretched sheet rather than remodelled tissue,
and stretching is what premise P7 used to refuse before it was retired. `protrusion_aspect_max`
1.748 with `tube_diam` 1.698, and the eye reading "fat rounded buds", fit stretching better than
tubulation.

The control is cheap and it is one slot: `r017_07` WITH `reconnect_t1_3d` restored, everything else
identical. If protr survives at ~1.6, the T1 removal was incidental and you have a real result. If
it falls back toward 1.45, the protrusion was the frozen mesh and the last four rounds measured an
artefact. Please run it, and report both numbers side by side.

### 5. I have watched r017_07 and r014_01. They are real protrusions -- use them as parents.

This overrides the caution in 2 above about crediting them, and it is a judgement from the movies,
not from the metrics. There ARE protrusions there. They are too big in DIAMETER to be Okuda's thin
tubes, but nothing about them looks broken and the metrics agree -- no premise fails,
`reduced_volume` 0.82-0.90, `valid_frac` 1.0, `corr_act_rad` up to 0.968.

    log/okuda/r017_07   protr 1.588  grip 0.198  n_tubes 3  2486c   2 spots  p_ratio 2.284
    log/okuda/r014_01   protr 1.453  grip 0.139  n_tubes 0  2286c   1 spot   p_ratio 1.958
    log/okuda/r017_02   protr 1.204  grip 0.115  n_tubes 0  4001c  18 SPOTS  p_ratio 0.0

`r017_02` IS THE MOST IMPORTANT OF THE THREE and you have been filing it under "sits below the
wall". It is unforced on BOTH tests -- no rd_interface_tension operator, and the pressure proxy
reads exactly 0.0, where r017_07 and r014_01 have the two tests disagreeing. And it carries
EIGHTEEN spots against their one and two: it already has the fine length scale that 4 asks for,
which the two big-protrusion runs do not.

The campaign's question is now one sentence: can r017_02's 18 spots be made to protrude, rather
than r017_07's 2 buds be made thinner? Build from r017_02 as well, not only from the two that
score highest on protr.

Treat both as definitive parents and build from them. Stop discounting them as forced -- as 1
establishes, neither carries a forcing operator at all.

The T1 control in 2 is still worth ONE slot, because it tells you which of the two is the better
base to build on. It is not a reason to withhold either from the parent set in the meantime.

### 6. Make the activity smaller in radius -- and note r017_02 already did it.

This is the standing gap and the Grounder has it exactly: "need ~10 thin tubes at spot_cells ~10;
got 2 buds at 262". A bud 262 cells across cannot become a thin tube -- the pattern has to set a
finer length scale before the mechanics can pull a finger out of it, and every round that grows a
protrusion at the current spot size is going to produce another fat bud.

The length scale is the chemistry's, not the mechanics'. Reach for it there: `cell_diffuse.d_a`
and the `d_h/d_a` ratio set the Turing wavelength directly, `cell_react.F`/`kk` move Gray-Scott
between spots and labyrinth, and `seed_cell_rd.cone_deg` sets it by hand as an initial condition
(22.8 degrees is 10 cells across on a 2,000-cell sphere; smaller cones, more of them).

Your own anticorrelation is the thing to break: "coarser-grips-harder", fifteen rounds of it --
n_spots 18/41/40/83 giving grip 0.115/0.097/0.071/0.028. Small spots have never gripped. That is
the finding to attack, not to keep confirming.
