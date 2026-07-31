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
