<!-- curvature -- append below; the driver merges this into campaign/analysis.md -->

## curvature (EXCAVATOR)

Read the actual C++ (found on disk at `papers/CompuCell3D/.../plugins/Curvature/CurvaturePlugin.cpp`,
`.h`, `CurvatureTracker.h`), cross-checked against the installed binary `libCC3DCurvature.so`
(demangled symbols) and the shipped `Curvature_test_generate` XML. Web fetch/search were blocked,
so the on-disk source is the sole evidence — but it is the real thing, not a guess.

What it does: a **Cellular-Potts energy term** that penalizes bending of chains of compartmental
cells linked by junctions it maintains itself. Per triple of consecutive linked cells it adds
`lambda * kappa` where `kappa` is the **Menger curvature (1/circumradius)** of the three centers
of mass — zero when straight. `changeEnergy` returns the dE over affected triples (COMs recomputed
+/-1 volume); it writes no cell state. It ALSO grows a junction graph (activation_energy biases a
bond-forming move; the bond is committed in the `field3DChange` watcher on accepted moves).

Surprises worth the record:
- The function `calculateInverseCurvatureSquare` is a **misnomer** — it returns the plain curvature
  (2*sin(theta)/|chord| = 1/R_circ), neither inverted nor squared. Trust the name and you invert
  the physics.
- Half the plugin is **dead code**: `potentialFunction` (harmonic spring), `targetDistance`,
  `maxDistance`, and all three `diffEnergy*` (return 0). Only `lambda_curve` and `activation_energy`
  matter. Clearly cloned from FocalPointPlasticity and left half-gutted.
- Junctions are the plugin's OWN state (within-cluster only), NOT FPP's — the two are separate
  plugins that merely co-occur in the demo. So Curvature is a **hybrid**: energy term + stateful
  bond-graph watcher. This is the interesting bit for the algebra: `changeEnergy` fits "return a
  delta the engine integrates," but `field3DChange` is a genuine write-on-accept side effect that
  the pure-energy framing does not cover.
- Apparent BUG: the volume-1->0 branch adds three curvature terms WITHOUT the `lambda` factor
  (L674/L682/L687). A faithful port must reproduce it to match the oracle.

Could NOT establish: no oracle/ablation run exists for curvature under `log/atlas_cc3d/` (not one
of the six evidenced mechanisms), so the dynamical magnitudes (does lambda=1000 in the demo
actually straighten the chain, how strong is the activation-energy bias vs Temperature=10) are
unmeasured here — inferred from the formula only. I did not confirm the exact set of "affected
triples" is complete for every geometry; I read the five triple-blocks in each branch but did not
prove they exhaust all triples touched by a COM shift. Left `verdict`/`contract` unset (normalizer's
call); set `status: inspected`.

## curvature — NORMALIZED

Verdict `new`, contract **`stiffen`** (lateral/interaction, set cell) — a BENDING STIFFNESS: an
angular energy over an ORDERED chain of junction-linked compartmental cells that penalises
curvature (Menger `kappa = 2 sin(theta)/|chord| = 1/R_circ`, linear, zero for straight) and drives
a linked filament toward straight. `implementation_of` left null. Consistent with the excavator's
read, this is a HYBRID: the primary term is the bending stiffness (a Potts `changeEnergy`, writes
nothing); a SECOND stateful half GROWS the junction graph (biases a bond-forming move by
`activation_energy`, commits the bond in the accepted-move watcher) — a topology rewire near
[[radius_graph]] that builds the chain the stiffness acts on. That rewire half is recorded in
writes:/maps:/surprises, not the verdict.

STRONGEST ARGUMENT AGAINST: this should be `implementation_of: stillinger_weber`, not a new
contract. SW is the registry's only three-body angular term, `(cos - cos0)^2`; set `cos0 = -1`
(theta0 = 180°) and SW *is* a straightness/bending penalty — same "angular energy, preferred
angle" idea, and the Menger-vs-cosine functional difference is "just an implementation." Rebuttal
(why I still chose `new`): SW builds its OWN isotropic min-image neighbour list and sums over ALL
geometric triples within a cutoff — it has no ordered backbone. Filament bending requires selecting
only CONSECUTIVE triples along a MAINTAINED 1D bond graph (left/mid/right, ≤3 hops each side);
SW-with-cos0=-1 on a blob would penalise every bent geometric triple, not straighten a defined
chain. Covering Curvature forces SW to take an EXTERNAL ordered bond graph it does not maintain — a
new required input that breaks every current SW user (mW water, silicon) and changes its output
from an integrated Newtonian force to a Metropolis accept/reject energy. That is a signature-
breaking refinement, not a free implementation slot, so `stiffen` is the honest call. (If a
reviewer rules SW's contract already means "any three-body angular stiffness," the fallback is
`implementation_of: stillinger_weber` with `cos0=-1` — and the disagreement is exactly the
maintained-topology axis this campaign exists to surface.)

Not a second sighting of any jax-morph-proposed contract (adhere, agitate, apoptose, mechanosense,
morphogen, regulate, relax, reorient) — none is an angular/bending stiffness. All excavator caveats
(no ablation run, no paper text, C++ read-not-run) carry forward unchanged.

