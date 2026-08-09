# Integrins as MPM fibres, anchored at both ends

The design to build next. Written down rather than half-coded, because the last session ended here.

## What it fixes, and why it is not just another route

Two things force it, one mechanical and one biological.

**One integrator.** In the direct-force hybrid (120-123) `mpm_gather` advances particle positions ~20
times per frame, and the engine then applies the adhesion delta ONCE per frame on top, computed from
positions at frame start that MPM has since changed twenty times. Neither knows about the other. At
k/gamma = 25 the engine's share is small and it is benign (121: coverage 1.000, strain 2.25); at 125 the
per-frame jump is large enough that the two disagree and the sheet collapses (122: coverage 0.002,
strain -0.56, i.e. compressed). If the integrin is MPM MATERIAL, everything is integrated by MPM and
the failure mode is gone by construction rather than by staying under a ratio.

**The right object.** An integrin linkage is a fibre anchored at both ends -- cytoskeleton inside the
cell, laminin in the basement membrane. As material rather than a spring constant it carries load,
yields and RUPTURES, so detachment becomes a material failure that is measured instead of a `detach`
threshold that is set. A beta1 knockout is then fewer fibres, which is what the experiment does.

## The structure

    set        integrin        MPM particles, a short fibre each: inner end + outer end
    inner end  KINEMATIC       position prescribed from the surface map (the epithelium is a replay
                               in pass 2 and cannot be an MPM body)
    outer end  MATERIAL        couples to basement_membrane_particle through the shared grid, the
                               way any two bonded MPM bodies do

The inner end is the only prescribed thing, and this is the important difference from
`mpm_tissue_boundary`: that operator imposed the tissue on EVERY grid node, which smeared the
constraint over the B-spline stencil and produced a standoff of ~1.5 cells set by the stencil width
rather than by the sheet. Here the constraint touches ~2,000 integrin particles, and the sheet feels
them through ordinary MPM contact. Local constraint, global consequence.

## Operators

| operator | kind | what it does |
|---|---|---|
| `integrin_seed` | seed | place fibres on the basal surface, uniform per unit area (cells tile the surface) |
| `integrin_track` | structural | ride the inner ends on the current surface -- prescribed, may write state |
| `integrin_rupture` | rewire | fibre fails past a stretch; the relation it carried disappears |
| `integrin_form` | structural | new fibres appear where the surface is bare -- turnover, minutes not frames |

The material response of the fibre itself needs no new operator: it is `mpm_strain` on its own set with
its own Young's modulus, and it reaches the membrane through `mpm_scatter[accumulate]` into the shared
grid, which is the pattern the stroma and the sheet already use.

## What to check first, before building all of it

1. Does a two-particle fibre transmit at all at this grid? A fibre shorter than `dx` = 0.0208 is
   sub-cell, and the membrane already sits ~0.004 from the surface. If the two ends land in the same
   grid cell the fibre transmits nothing, and the answer is a longer fibre or a finer grid.
2. Does the inner-end constraint reintroduce the stencil-width standoff on a small scale? It should
   not -- it acts on 2,000 particles, not on the grid -- but it is the same class of mistake and it
   should be measured, not assumed.
3. Rupture threshold in units of what? Fibre stretch, not sheet displacement, so it is a material
   property and comparable across runs.

## What this does not fix

The standoff. 121 sits 0.0082 INSIDE the surface, and a basement membrane is basal -- it belongs just
outside, never inside. Nothing in this design addresses that directly; it is worth knowing whether a
fibre pulling from a prescribed inner end holds the sheet at the fibre's rest length, which would make
the standoff a material property (fibre length) instead of a tuning parameter. That would be the first
thing this buys, and it should be measured on the first run.
