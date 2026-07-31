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

---

## Implementation (IMPLEMENTER)

**Built:** `src/plexus/operators/candidates/jax_morph_free_screened_diffusion.py` —
`MorphogenFreeSpace(Exchange)`, registered as the `morphogen` contract with
`implementation="free_space_greens_function"` (kind=exchange / family=fields / set=cell, matching
the normalized contract). This is the `diffuse` `finite_difference`/`spectral` pattern applied to
`morphogen`: the paper's graph-Laplacian inverse `c = (K I - D L)^{-1} S` can later register
`implementation="graph_laplacian"` on the SAME contract — two numerical methods, one biological
operator, which is the convergence the ledger is meant to record. `get_contract("morphogen")` now
resolves; `implementations = {"free_space_greens_function"}`.

**The load-bearing decomposition choice — an OVERWRITE, not a delta.** Unlike the gene-network
siblings (`regulate:*` return `dg/dt` for the engine to integrate), this is a QUASISTATIC CONSTRAINT
SOLVE: `dt` is meaningless, the field is the `t=infinity` equilibrium, so the operator OVERWRITES the
`chemical` block each step rather than incrementing it. It is therefore a *derived readout* in exactly
the `aggregate` (centroid) sense: `EMIT=None`, `MAY_MUTATE_INTEGRATED_STATE=True`, mutate state via
clone-and-assign of the `chemical` columns, return `{}`. `MAY_MUTATE_...=True` is required because the
engine's frame-0 integration guard clones the WHOLE `state` tensor and would otherwise flag the
in-place block write (it is the derived-readout exemption, not a force). Verified `pos` is invariant
across a forward (a test), so the exemption is honest — it only ever writes `chemical`.

**Faithful details carried over (diffusion.py:92-264):**
- Dimension-selected kernel by `pos.shape[1]` — 1-D segment `exp(-kappa(r_eff-a))/(2 D kappa)`, 2-D
  disk `K0(kappa r_eff)/(2 pi D a kappa K1(kappa a))`, 3-D sphere
  `exp(-kappa(r_eff-a))/(4 pi D r_eff (1+kappa a))`. The reference's static `n_space_dim` assert is
  preserved as an OPTIONAL param that raises on mismatch (surprise kept reproducible).
- `r_eff = max(r_ij, a_j)` surface clamp → the `i==j` diagonal contributes the on-surface value (SELF
  field included, tested). Source radius `a` is the EMITTER's, broadcast over receiver rows.
- Alive-masking applied TWICE and asymmetrically: sources masked over columns j (`* occ[None,:]`),
  receivers over rows i (`* occ[:,None]`) — tested both directions with a dead big-source slot.
- The three finiteness guards: `a <- max(a,1e-12)`, `kappa <- max(kappa,1e-12)` for 1-D/2-D only, 3-D
  left exact (bounded at kappa=0). The low-dimension screening requirement (`degradation>0` in 1-D/2-D)
  is raised at forward (where the spatial dim is known), matching the reference's constructor check.
- `safe_norm` ported verbatim (value+grad zero at the zero vector — the `where` trick, not a bare
  sqrt), so the diagonal r=0 does not poison gradients. Minimum-image applied if the world is periodic,
  with the documented caveat that free-space kernel + periodic box is a modeling error the user owns.
- JAX `vmap(in_axes=(0,0,1))` over species → a Python loop over the small static species axis;
  `diffusion`/`degradation` broadcast to `(n_species,)` (scalar or per-species). Multi-species
  independence tested.

**Bessel port (the one non-mechanical carry-over).** `_k0`/`_k1` are the reference's Abramowitz &
Stegun 9.8.5–9.8.8 rational/series approximations, ported to torch on `torch.special.i0`/`i1`. I did
NOT use torch's built-in `torch.special.modified_bessel_k0/k1` because those carry no autograd
backward (they would break `DIFFERENTIABLE=True`, the whole point of a jax-morph translation). I
cross-checked my port against torch's builtin K0/K1 over x in [1e-3, 20]: max relative error ~1e-7
(the A&S series precision) — so the disk kernel is faithful, not eyeballed. This also closes the
normalizer's open item "did not verify `_k0`/`_k1` against a reference over the full range."

**One deliberate robustness add (flagged, not hidden).** The reference always has `state.radius`; a
Plexus cell set may not carry a `radius` block. `_radii` reads a per-cell `radius` block/buffer if
present, else falls back to a UNIFORM default (the engine's `spawn radius` default 0.02). For the
oracle differential the source set carries a real per-cell radius, so this is inert there; it only
keeps the operator runnable on a radius-less set. Also: the contract's `READS` lists
`pos`/`radius`/`alive` explicitly — the reference `state_reads()` under-declares them (declares only
`secretion_rate`), the surprise the normalizer flagged.

**Test:** `tests/test_jax_morph_free_screened_diffusion.py` (7 pass). Headline property (reference-free):
SUPERPOSITION — the map `S -> c` is LINEAR (`c(3S)=3c(S)`, `c(S1+S2)=c(S1)+c(S2)`), the defining
property of a Green's-function steady-state solve, tested in 2-D so it exercises the Bessel path. Plus:
non-negative sources → non-negative field (kernel positivity); stronger `K` shortens the range (a
distant cell reads less); the self/diagonal field is included; a dead cell neither emits nor carries;
`pos` invariance under the solve; per-species independence. No oracle numbers hard-coded.

**FLAG for the curator's differential run (not a translation bug).** (1) Use a FREE (non-periodic)
world — the kernel is open-boundary; a periodic oracle would need the paper's method, not this. (2)
This is an OVERWRITE and quasistatic, so it must run BEFORE any step that reads `chemical` in the same
macro-step (the reference's `quasistatic -> dynamic -> discrete` phase order); schedule it first. (3)
Match `n_space_dim`/`diffusion`/`degradation` and per-cell `radius` to the oracle config; the finite
radius `a` has no paper counterpart, so agreement is a code-vs-code check, not code-vs-paper.

**Not done (next role):** the differential run against the oracle — `evidence.*` stays null,
`status: implemented`.
