<!-- metropolis_acceptance -- append below; the driver merges this into campaign/analysis.md -->

**metropolis_acceptance -> out_of_scope (forced-fit contract `metropolis_step`, structural/growth).**
This is the CPM's core modified-Metropolis Monte Carlo integrator, declared on PottsCore (the
simulation root), not a plugin: it reads the summed dE from every enabled plugin and one T, then
accepts a proposed pixel copy with P=1 if dE<=0 else exp(-(dE+offset)/T). It carries no biology
of its own -- the biology is entirely in the per-plugin dE terms already catalogued -- so it is
framework mechanics, the counterpart to the Plexus engine's integration loop, which sits outside
the operator algebra. Recording it as an operator (`new` or a `cell_divide` alias) would inflate
the yield with the integrator itself and corrupt the saturation measurement.

**Strongest argument AGAINST out_of_scope:** the hand-added surprise says the discrete, stochastic,
accept/reject dynamics "with no pathwise derivative" is itself the finding -- and a gap in the
language is exactly what `new` is meant to flag. If Plexus genuinely cannot express energy-plus-
accept/reject dynamics, one could argue that incapacity is a missing contract, not out-of-scope
plumbing. I reject this because `new` in this loop means a missing *operator* (a typed map over
sets/fields returning a delta), and the Metropolis rule is the integrator that *consumes* deltas/
energies, not one that produces them -- it maps to Plexus's engine, which registers no operator
either. The orthogonality of the two integration paradigms (energies+stochastic accept/reject vs
deltas+deterministic integration) is real and is recorded verbatim in `why:` and `surprises:`; that
is a measurement result about the engine, not a new entry in the operator vocabulary. The honest
move is to state the gap loudly under an out_of_scope verdict rather than mint a fake operator to
represent it.
