<!-- Division -- append below; the driver merges this into campaign/analysis.md -->

## Division -- normalized

**Verdict: `refinement` of `cell_divide`.** Same biological contract as the registered
`cell_divide` (structural/growth/cell, tags proliferation|mitosis|growth): identical Bernoulli
hazard `p = 1 - exp(-rate*dt)`, a daughter waking a free buffer slot beside the mother and
inheriting her per-cell state, capacity as a hard wall. Not `new` (proliferation already has a
home; widening does no violence to mitotic biology). Not a bare `alias`, because the two do NOT
fully agree: `cell_divide`'s promoted "default" implementation is ISOTROPIC and RADIUS-PRESERVING
(random-jitter placement, daughter radius cloned so two full-size cells stand where one did),
whereas Division (a) reads a per-cell `division_axis`+`orientation_snr` and places the daughter
ORIENTED along `s*a_hat + xi` (spindle-axis / Hertwig's-rule division) and (b) is VOLUME-CONSERVING
(mother shrinks to `r*2^(-1/d)`, daughters just-touching). The contract must widen its `reads`
(+`division_axis`) and its radius/position write SEMANTICS, and gains `mother`/`division_overflow`.
The cost a refinement must name: existing callers get isotropic full-size daughters, so enabling
volume conservation halves every cell's radius on division (changes packing/contact/virial stress)
and requiring an axis forces a default -- a real breaking change, hence a costed refinement, not a
free alias.

**Strongest argument AGAINST `refinement` (the alternative I had to defeat).** Plexus's registry
explicitly supports MULTIPLE implementations under one contract (`cell_divide` already carries an
`implementations` list), and each implementation may declare its own reads/writes -- so one could
file Division as simply a SECOND implementation of `cell_divide` (oriented + volume-conserving),
leaving the contract signature untouched. That is exactly the Morse/SoftSphere/Hertzian pattern the
campaign celebrates as convergence (and how the three `ODEController` siblings collapsed to one
`regulate`): same biological job, different internal recipe, `implementation_of: cell_divide`,
verdict `alias`. Under that reading nothing "breaks" (the default impl is untouched; you just add
another), the contract count is unchanged either way, and calling it a refinement over-reports
"language incomplete." I rejected it because oriented placement requires READING a `division_axis`
field that NO promoted operator reads today -- the capability genuinely does not exist in the frozen
language, so an alias would flatter it in precisely the way `registry_view.py`'s docstring warns
against ("record an alias without ever asking whether the two contracts actually agree"). The
Morse-family siblings share one I/O signature and differ only in a force's functional form; Division
and `cell_divide` differ in their declared I/O (an orientation input, a volume-conserving radius
write). If Plexus later promotes the widened signature, the current isotropic `cell_divide` becomes
its first implementation and Division the oriented/volume-conserving second -- but until then the
gap is real and belongs on the record as a refinement.



## Division -- implemented

Operator registered as `cell_divide` with `implementation="volume_conserving"` at
`src/plexus/operators/candidates/jax_morph_division.py`; test at
`tests/test_jax_morph_division.py` (10 properties, all pass). This is the faithful realization of
the `refinement` / `implementation_of: cell_divide` verdict: the registry keys operators by
`(name, implementation)` and only enforces that co-implementations share the contract `kind`
(structural == structural), so importing the candidate ADDS `volume_conserving` alongside the
promoted isotropic `default` on the SAME `cell_divide` contract -- the widened read/write set lives
declaratively in the atlas entry, the registry does not re-validate it. Default stays the promoted
mass-doubling impl, so the widening is additive for current callers exactly as the entry claims; the
candidate is not auto-imported, so nothing registers until the differ/tests pull it in. Same
multi-implementation pattern as `diffuse` (finite_difference/spectral) and `regulate`
(connectionist/mwc/neural_ode).

State representation (the reason this needed care). Plexus has NO standard `radius` state block --
`radius` is only a spawn-time scalar in `engine.py`, never carried per-cell -- so `radius`,
`division_rate`, and `division_axis` are modelled as per-cell BUFFERS the operator reads/lazily
provisions, mirroring `apoptose`'s `death_rate` convention. `born` is a float buffer, `mother` a long
buffer with the -1 founder sentinel, and `division_overflow` a 0-dim scalar buffer.

Deliberate translations of the source's subtleties:
- Volume factor `m = 2^(-1/d)` reads the live world dim `H.dim` (not a hardcoded 1/2), so both
  daughters take `r*m` and conserve the mother's d-volume (~0.707 in 2D, ~0.794 in 3D).
- The offset uses the NEW radius `(r*m)*dir`; mother moves to `x+offset` and daughter to `x-offset`,
  so the pair is centred on the mother's pre-division position and sits exactly touching (centre gap
  `2*r*m` = sum of radii). I capture `x_old`/`r_old` BEFORE any write so the two placements can't read
  each other back.
- Oriented direction `normalize(orientation_snr*a_hat + xi/sqrt(d))` with `a_hat = axis/(||axis||+1e-12)`;
  a zero axis or `orientation_snr=0` collapses to pure isotropic via the same 1e-12 guard (a test
  confirms the isotropic fallback still conserves volume and just-touches). With no `division_axis`
  buffer at all the direction is pure `xi`.
- Capacity is a hard wall: I draw the movers FIRST, then allocate `cap = min(movers, free)` and add
  `movers - cap` to `division_overflow` -- so the surplus is counted even when the buffer is
  completely full (`cap == 0`), matching the source's `sum(divide) - sum(committed)`. `born`/`mother`
  reset to their defaults every macro-step (a per-step lineage record); `division_overflow` is GLOBAL
  and accumulates (a test checks 4 dropped then 4+8=12 over two steps, deterministic at a huge rate).

One chosen divergence from the source, noted for the curator: jax-morph resets a daughter's
NON-heritable cell fields to their spec default and inherits only heritable ones; I inherit EVERY
per-cell buffer from the mother (like the promoted `cell_divide`'s spawn), then reset `born`/`mother`
explicitly. The heritable drivers (`division_rate`, `division_axis`, `celltype`) inherit correctly
either way; the only difference is that a recycled dead slot's stale non-heritable buffers are
overwritten with the parent's value rather than a default -- arguably safer, and consistent with the
sibling `cell_divide`. Plexus's own `Level.lineage`/`birth` buffers already record parent-slot
provenance, so `mother` is partly redundant with the container, but I write it explicitly because
`reconstruct_lineage` reads that exact field.

NOT modelled (same scope exclusion as `cell_divide`/`apoptose`): the straight-through / pathwise
differentiability and the `logp` score-function term. Plexus's engine runs the forward EFFECT only
(the source's `replay`); the `divided`/`divide_eligible`/`division_dir` traces and `Division.logp`
have no engine counterpart. If a scoring/inverse-design driver lands, that layer is the follow-up.

Tests are reference-free by construction (limits, sign, conservation, symmetry): rate-0 and
negative-rate (clip) no-ops; d-volume conservation `r_m^d + r_d^d == r_old^d` in 2D and 3D; the
just-touching centre-distance = sum-of-radii geometry; the symmetric split centred on the mother;
lineage (`born=1`, `mother=parent`, defaults elsewhere) with live-count growth; the overflow cap +
GLOBAL accumulation; the large-`orientation_snr` limit aligning the split with the axis; the
isotropic fallback; and the `at:` mask gating who divides. No oracle run -- jax is deliberately
absent from this env -- so `evidence.oracle_run` stays null for the differ/curator.
