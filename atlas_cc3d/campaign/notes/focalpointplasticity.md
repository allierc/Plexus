<!-- focalpointplasticity -- append below; the driver merges this into campaign/analysis.md -->

## FocalPointPlasticity (order 16) — excavated 2026-08-02

Read: `PyCoreSpecs.py` L4512-4908 — the Python spec layer (`LinkConstituentLaw`,
`FocalPointPlasticityParameters`, `FocalPointPlasticityPlugin`), the twedit ML generator
(`CC3DMLGeneratorBase.py` L859-915), and the shipped test XML
(`tests/.../FocalPointPlasticity.xml`). The compiled FPP core is NOT readable in this env, so the
energy form is inferred, not read line-by-line — flagged in the entry.

What it does: keeps a dynamic set of pairwise **junctions** (focal-point links) between cells of a
type pair, and adds per link a spring energy `lambda*(d - target_distance)^2` on the distance `d`
between the two cells' **centers of mass**. Links form on contact (both cells below max_junctions),
pay a one-time ActivationEnergy at formation, and break when `d > max_distance`. So it is an energy
term that ALSO carries persistent inter-cell state — the interesting bit for the algebra: not a
stateless Potts plugin, it mutates a link registry between steps.

Surprised me: (1) `d` is CoM-to-CoM, long-range — target 7 / break 20 on volume-25 cells, i.e. links
span well beyond cell contact. (2) ActivationEnergy is XML-only, explicitly NOT runtime-steerable
even though targetDistance/lambda/maxDistance are (generator warning L882-884) — a one-time
formation threshold, not a per-step energy. (3) default energy comes from `LinkConstituentLaw`, which
is user-overridable with an arbitrary formula string over Lambda/Length/TargetLength.

Could NOT determine: the exact compiled energy/lifecycle code (inferred from the default
LinkConstituentLaw formula + Swat et al.); the full variable set bindable in a custom
LinkConstituentLaw and how the core parses/evaluates the formula string; whether link formation is
scanned every MCS or only on boundary-changing copies (I assume the latter from the neighbor_order
contact semantics, but did not confirm in core). No ablation run exists for this mechanism yet.

### Verification pass (re-excavation)
Re-checked the two specific citations in this entry against source, both hold:
- ActivationEnergy-XML-only warning is verbatim at `CC3DMLGeneratorBase.py:L882-884`.
- The `Lambda 10 / ActivationEnergy -50 / TargetDistance 7 / MaxDistance 20 / MaxNumberOfJunctions 1`
  defaults are real, but they come from the **twedit generator template** (`CC3DMLGeneratorBase.py`
  L894-898), NOT from the `FocalPointPlasticity.xml` test — that test file has no FPP block at all,
  only Volume(target 25)/Contact/ExternalPotential. So "target 7 / max 20 on volume-25 cells"
  combines two different co-named sources; entry surprise #3 was corrected to say so. The negative
  ActivationEnergy default (-50) corroborates "negative promotes link formation."

### Normalization — verdict `new` → contract `bond`
Verdict **`new`** against the frozen 42. Contract `bond`: a persistent, plastic, load-ruptured
cell-cell link network — junctions self-assemble on contact under a per-cell coordination cap
(paying a one-time ActivationEnergy), persist as identified per-pair state, and rupture when their
CoM-CoM distance exceeds a break length. Classified `rewire`/`topology`/set `cell`. NOT
`implementation_of` anything: it is distinct from `adhere` (continuum surface-contact energy; the
CC3D Contact/AdhesionFlex mechanisms are `adhere`), because `bond` is a discrete centroid-pair link
graph, not a boundary-site energy. The restoring spring `lambda*(d-target)^2` is charged separately
to the registered `squared_law`, so only ONE new contract is credited — the plastic topology, not
the spring.

**Strongest argument AGAINST `new`:** FPP may be nothing but a COMPOSITION of two things already in
hand — `radius_graph` (proximity edges) + `squared_law` (a quadratic pair spring) — with no new
atomic contract at all; on that reading it should be recorded as two existing contracts, and minting
`bond` inflates the yield. The rebuttal I rest on: `radius_graph` is deliberately memoryless and
symmetric-threshold (it rebuilds every edge from scratch each tick), so it cannot produce FPP's
hysteresis (form within ~1–2 contact sites, break only past distance ~20), its persistent per-link
identity/attributes, its per-cell coordination cap, or its once-paid formation energy — the
composition genuinely fails to reproduce the dynamics. If that rebuttal is wrong (e.g. a stateful
variant of `radius_graph` is considered fair game to widen into), `bond` collapses to a
`refinement` of `radius_graph` and the honest record is one fewer new contract.

### Re-read pass (LinkConstituentLaw variables)
Resolved part of the `law` UNKNOWN by reading `LinkConstituentLaw` (`PyCoreSpecs.py:L4512`) more
closely: Lambda/Length/TargetLength are the built-in default variables, and ARBITRARY extra
variables are bindable via the `variable[name] = value` accessor (L4550), each emitted as a
`<Variable Name=.. Value=..>` child of `<LinkConstituentLaw>` (L4545). Updated the `law` param role to
say so. Still unread: how the compiled core parses/evaluates the Formula string — only the XML
emission is visible from the Python layer, not the evaluator.
