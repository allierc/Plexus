<!-- PairwisePotential -- append below; the driver merges this into campaign/analysis.md -->

# pairwise_potential (PairwisePotential, base class @ potentials.py:L128)

**What I read.** The whole `potentials.py`: the `Potential` protocol (L56, `forces = -jax.grad(total_energy)`),
`NoForce` (L108), the `PairwisePotential` base (L128), the five concrete subclasses (Morse, SoftSphere,
Hertzian, Harmonic, LennardJones), and the two module helpers `_compact_repulsion` / `_smooth_cutoff`. Then
the primitives it leans on: `core/geometry.py` (`pairwise_displacements`, the dense `neighbor_sum` alive/self
mask) and `core/ad_utils.py` (`safe_norm`, `safe_divide`). This entry is the ABSTRACT base only -- Morse,
SoftSphere, Hertzian, Harmonic, LennardJones, NoForce, MechanicalRelaxation, and VirialStress are each their
own record entry (confirmed in atlas_record.yaml), so I scoped this to the contract they inherit: energy ->
force (autodiff) -> virial, over live non-self pairs, with couplings resolved scalar-or-per-cell-field.

**Line/anchor checks.** L128 is exactly `class PairwisePotential` -- code_path unchanged. Paper anchor: the
Morse energy `V_ij(r)` on p. 9 ("Mechanical interactions", also fig. 1d, SI p. 14). The base class itself is
never described in the paper; only its concrete Morse instance is (recorded that gap).

**What surprised me.** (1) SOURCE vs PAPER: the paper says JAX-MD ran the mechanics (Methods, p. 9), but the
source's module docstring is explicit that forces come from autodiff "with no jax-md dependency." Source wins.
(2) Two different halves that look identical: the `0.5` in `total_energy` is a double-count dedup (neighbor_sum
sums both (i,j) and (j,i)); the `1/2` in `virial_pressure` is the Irving-Kirkwood bond split. (3) `mix` is the
ARITHMETIC mean (deliberate -- finite grad, no sqrt), not the geometric/Lorentz-Berthelot mean I expected from
MD. (4) Couplings are hard-restricted to a shared scalar OR a per-cell scalar field; per-type/array couplings
are rejected at `__check_init__`, so the paper's heterotypic cadherin matrix can only be expressed as a per-cell
field mixed to the pair mean -- a real expressiveness limit worth flagging to the normalizer.

**What I did NOT establish.** I did not run the oracle -- no numeric confirmation of the energy/force/virial
values (evidence left null; this is an EXCAVATOR pass). I read `virial_pressure`'s d-ball volume and IK sign
convention off the code/docstring but did not verify the sign against a compression test. I did not trace which
wrapping steps actually call `virial_pressure` vs `forces` (VirialStress / MechanicalRelaxation /
BrownianDynamics are separate entries) -- I only asserted from the signatures that the base writes no state
itself. Whether `mix` is ever overridden by a shipped subclass I did not check (none of the five in this file
override it).

---

**NORMALIZER.** Verdict: **alias of `attraction_repulsion`** (implementation_of: attraction_repulsion). The
abstract pair-potential base is the energy-form statement of the pairwise attraction-repulsion contract Plexus
already registers: a radial cell-cell force with a repulsive core (excluded volume) and an adhesive tail — the
Morse well IS attraction_repulsion's "long-range pull minus short-range push", and both are the same
conservative radial pair force (attraction_repulsion's `f(r)·(pos_j−pos_i)` is itself the gradient of a radial
potential). Morse/SoftSphere/Hertzian/Harmonic/LennardJones are interchangeable force-law shapes over this one
contract (the pattern the registry is built to hold), and the NoForce normalizer already named exactly this
landing. Plexus's own criterion clinches it: the registered `stillinger_weber` docstring says every prior
interaction op is *pairwise* and only its *three-body* term earns a new contract — PairwisePotential is strictly
two-body, and SW is itself an energy-defined, autodiff-force interaction, so neither "pairwise" nor
"energy+autodiff" is novel here. **Strongest argument AGAINST (and why it loses):** the energy formulation is a
genuine capability attraction_repulsion lacks — a scalar potential U(r) unlocks two things the D'Orsogna
velocity law cannot give: a virial *pressure* readout and relaxation to a differentiable mechanical
*equilibrium* (FIRE / gradient descent). If the atlas counts "carries a conservative energy whose equilibrium is
itself differentiable" as contract-level content, this is at least a **refinement** (widen attraction_repulsion
to carry an energy, a pressure output, and an equilibrium-minimiser consumer), not a drop-in alias. It loses
because (a) attraction_repulsion's radial pair force is *already* conservative, so storing-and-differentiating an
energy vs hand-coding the force is a representation strategy, not new biology; (b) the family already spans both
integration modes (attraction_repulsion EMITs velocity, squared_law/stillinger_weber EMIT acceleration) and both
pair topologies (squared_law's all_pairs option; SW's dense min-image list), so none of that discriminates a
contract; and (c) the pressure and the equilibrium-relaxation are separate *registered* concerns (the VirialStress
and MechanicalRelaxation entries), not outputs of this interaction — folding them in would double-count. The one
residual is honest and recorded as a surprise, not a contract: the scalar/per-cell coupling restriction cannot
express the paper's type×type cadherin matrix — an expressiveness weakness of this implementation, not a wider
contract.
