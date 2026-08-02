<!-- chemotaxis -- append below; the driver merges this into campaign/analysis.md -->

# chemotaxis

**Verdict: alias of `chemotax`** (implementation_of: chemotax). CC3D's ChemotaxisPlugin is
gradient-following with a per-type sensitivity lambda_chemo — the exact biology of the registered
Keller-Segel `chemotax` contract. It is a second implementation, realized as a Potts energy term
dE = -lambda*(c(x_new)-c(x_old)) that biases pixel-copy acceptance, rather than a velocity delta
the engine integrates.

**Strongest argument against:** if you take `writes` seriously as part of the typed signature, these
are NOT the same contract. Plexus `chemotax` writes a velocity on `pos`; CC3D's plugin writes
nothing and only enters a Hamiltonian. That gap could justify a `refinement` (widen `chemotax` with a
third `emit: energy` / acceptance-bias routing beside `velocity` and `mpm_acceleration`) or even a
`new` energy-term kind the algebra lacks. Calling it an alias risks burying the single most
interesting thing CC3D contributes here — that gradient following is a probability bias on a discrete
move, not a drift. I keep the verdict at alias because the biology is identical and the
energy-vs-update divide is a systemic property of the Potts paradigm shared by nearly every plugin;
minting it per-mechanism would inflate the yield and destroy the measurement. The routing gap is
flagged in `state_io` and `why` so it is recorded once, not counted many times.
