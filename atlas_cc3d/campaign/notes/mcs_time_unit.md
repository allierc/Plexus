<!-- mcs_time_unit -- append below; the driver merges this into campaign/analysis.md -->

**Verdict: `out_of_scope`.** The Monte Carlo Step is not an operator over sets or fields; it is
the engine's *temporal-integration contract* -- the definition of the clock itself. It computes no
force, energy term, flux, or division, and none of the seven Plexus kinds describes a scheduler. Its
worth to the atlas is the contradiction it pins down, the sharpest in the record: the promoted Plexus
engine advances state by integrating operator deltas against a real-valued `dt`; CC3D has no `dt` and
no integrator -- time is one attempted pixel copy per lattice site, and a rate must be re-expressed as
a per-MCS acceptance *probability*. The two frameworks disagree on the meaning of the time axis, not
its units. The Plexus algebra can express the energy *terms* of a Potts model but the engine cannot
express its *time*.

**Strongest argument against.** One could call this `new` rather than `out_of_scope`: the campaign
measures whether the language is complete, and here is a genuine capability the promoted vocabulary
lacks -- a discrete, dt-free, per-site stochastic clock. If "the language" is read to include the
engine's execution model and not just the 42 operator contracts, then MCS is precisely a missing
piece and marking it out-of-scope hides a real incompleteness behind "it's plumbing." I keep
`out_of_scope` because the frozen baseline `new` is measured against is a set of *operators* with a
typed `set/inputs/outputs/reads/writes/maps` signature, and a time unit has no set to act on and
writes only the step counter -- promoting it as an operator would corrupt the very saturation metric
the atlas exists to protect. But I record the incompleteness explicitly in `why:` so the measurement
is not lost: it is a gap in the *engine*, logged where it belongs, not smuggled into the operator count.
