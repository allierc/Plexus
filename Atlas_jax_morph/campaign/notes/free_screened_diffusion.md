<!-- FreeScreenedDiffusion -- append below; the driver merges this into campaign/analysis.md -->

# free_screened_diffusion (FreeScreenedDiffusion)

**Read:** `jax_morph/physics/diffusion.py` in full — the `FreeScreenedDiffusion` class (L92), its
`_kernel`, `__call__`, and the module helpers `_k0`/`_k1` (Abramowitz & Stegun polynomial
approximations of modified Bessel K0/K1). Base contract in `core/step.py` (`SimulationStep`,
`StepType.QUASISTATIC`), geometry in `core/geometry.py` (`pairwise_displacements`, free vs periodic
space), `safe_norm` in `core/ad_utils.py`, base field specs in `core/state.py`. Paper Diffusion
section read at p. 15 (M&M) plus the main-text Fig. 1c mention (p. 3).

**Central surprise (paper vs source, source wins):** the paper and code solve the *same* steady
screened-diffusion PDE (`D grad^2 c - K c + S = 0`) by completely different numerics. Paper: a
**graph-Laplacian** lattice solve `c = (K I - D L)^{-1} S` with `L = deg(A) - A` and *explicit*
boundary handling (closed/reflecting `A_ij = 1/dist`, or permeable via a heuristic **ghost sink
node** wired to detected boundary cells). Code: **analytic free-space Green's-function
superposition** — open boundaries automatic, no adjacency, no Laplacian, no ghost node, and a
**finite source radius `a`** (with Bessel/exponential kernels) that has *no* paper counterpart. A
reader trusting the paper would build a different solver and get different boundary behavior.

**Also surprised me:** the self/near field is *included* (r clamped to the source surface, so a cell
reads its own secretion — not the usual self-excluded neighbor sum); three easy-to-miss finite-ness
guards (`a`, `kappa` floors) matter only under `jax_debug_nans`/traced-kappa optimization; and
`state_reads()` declares only `secretion_rate`, silently also consuming `position`/`radius`/`alive`.

**Did NOT establish (open):**
- I did **not** run the oracle or any differential check — no numerical evidence the code and paper
  agree (or how far they diverge, especially near the cluster boundary). That is the validator's job.
- I confirmed by grep that `FreeScreenedDiffusion` is the **sole** diffusion step in the library
  (no graph-Laplacian sibling), so this is a wholesale method replacement, not one of two variants —
  but I did **not** check which example/guide configs actually instantiate it, with what
  `n_space_dim`/`degradation`, or whether any example still expects the paper's closed-system field.
- I recorded the 2-D disk kernel `K0(kappa r)/(2 pi D a kappa K1(kappa a))` verbatim from source but
  did **not** independently derive that it is the correct finite-disk screened Green's function, nor
  verify the `_k0`/`_k1` rational approximations against a reference over the full argument range.

## Normalizer

**Verdict: `new` — contract `morphogen`** (kind `exchange`, family `fields`, set `cell`;
`cell -> cell`, reads `secretion_rate`+geometry, writes `chemical`, map `pairwise`), with
`implementation_of: morphogen`. This is the steady-state secreted-signal FIELD — a source `S`, a
degradation/screening `K`, and diffusion `D` fused into one quasistatic constraint that OVERWRITES a
per-cell concentration each macro-step, computed as an all-pairs Green's-function superposition on the
mesh-free cell set. The closest registered contract, `diffuse`, is a different contract, not a
widening: it is `set: field` (a GRID stencil), it is source- and sink-free by design (source is
`deposit`, sink is `decay`, and reaction-diffusion is BUILT by composing the three atoms), and it is
a dt-time-stepper — whereas this is mesh-free CELL state, bundles S/K/D, and returns the t=infinity
equilibrium. The code's free-space Green's-function solve and the paper's graph-Laplacian inverse
`c = (K I - D L)^{-1} S` are two SIBLING implementations of this one contract, which is the
convergence result worth recording.

**Single strongest argument AGAINST:** that `morphogen` is not new at all but merely the fixed point
of `deposit + diffuse + decay` — a COMPOSITION of three registered atoms run to steady state — so the
algebra already covers it and inventing a name inflates the yield. I reject it because the three atoms
iterate a LOCAL update on a GRID (a different discretization, with grid-dependent boundaries) and must
be run to convergence; you cannot wire them to reproduce an EXACT, mesh-free, all-pairs analytic
equilibrium solve on the particle set in a single quasistatic step. If a future contributor showed
that iterating the grid atoms to convergence reproduces this field to within the oracle threshold on a
real config, the honest move would be to demote `morphogen` to that composition — the entry stands or
falls on that unrun differential check.
