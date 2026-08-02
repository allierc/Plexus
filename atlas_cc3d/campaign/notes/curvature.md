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

