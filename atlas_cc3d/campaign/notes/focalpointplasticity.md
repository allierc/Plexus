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

### Re-read pass (LinkConstituentLaw variables)
Resolved part of the `law` UNKNOWN by reading `LinkConstituentLaw` (`PyCoreSpecs.py:L4512`) more
closely: Lambda/Length/TargetLength are the built-in default variables, and ARBITRARY extra
variables are bindable via the `variable[name] = value` accessor (L4550), each emitted as a
`<Variable Name=.. Value=..>` child of `<LinkConstituentLaw>` (L4545). Updated the `law` param role to
say so. Still unread: how the compiled core parses/evaluates the Formula string — only the XML
emission is visible from the Python layer, not the evaluator.
