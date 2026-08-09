# User input

Written by Cedric, read fresh by the Proposer and the Analyst at the top of every round.
Edit this file mid-campaign -- the next round picks it up, no relaunch needed.

## Pending Instructions

### 1. `mech_p_ratio` is NOT a forcing test when no forcing operator is present. Retract the verdict.

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

### 2. What that something else is, and the control it needs.

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

### 3. I have watched r017_07 and r014_01. They are real protrusions -- use them as parents.

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

### 4. Make the activity smaller in radius -- and note r017_02 already did it.

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
