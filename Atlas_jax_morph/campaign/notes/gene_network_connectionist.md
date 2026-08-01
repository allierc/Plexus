<!-- GeneNetworkConnectionist -- append below; the driver merges this into campaign/analysis.md -->

# gene_network_connectionist (NORMALIZER)

**Verdict: `new`, contract `regulate` (exchange/fields, set=cell); `implementation_of: regulate`.**
The mechanism is a per-cell INTERNAL regulatory dynamical system — a heritable gene-state vector
whose autonomous per-cell ODE (dense recurrent gene→gene coupling through a saturating
nonlinearity + linear decay + a learnable drive from *sensed external fields*) is integrated over
the macro-step to emit the action-setting outputs that division/secretion/adhesion read. Nothing in
the frozen 42 covers cell-internal regulatory computation: the motion/interaction/mechanics/mpm
families move or couple agents; the fields family sources/reads fields; growth/topology change the
set. The one real neighbour is `signal`. Crucially, this is also the anti-inflation call: the three
`ODEController` subclasses (this connectionist linear-drive form, `GeneNetworkMWC`'s thermodynamic
log-occupancy drive, and `NeuralODE`'s black-box MLP) are interchangeable *implementations* of the
same `regulate` contract — different vector fields under one integration/IO contract — so they
collapse to **one** new contract, not three. That is the convergence the ledger exists to record.

**Strongest argument AGAINST `new` (and why it loses).** `signal` is already a registered
recurrent, nonlinear, first-order ODE network (its own tags say "recurrent"), with the identical
−decay + bias + saturated-weighted-drive skeleton; one could argue `regulate` is merely a
`refinement` of `signal` — widen `set` to `cell`, make `edge_set`/`MAPS` optional, add a
field-input term — and that "gene regulation vs. connectome" is just parameterization of one
"recurrent-ODE-network" contract. This is the counterargument I had to defeat, and it is the
tempting one because the *math* really does rhyme. It loses on the *signature*, not the math:
`signal`'s recurrence runs BETWEEN nodes across a fixed connectome edge-set, and its typed
signature is load-bearing on exactly that topology (`INPUTS ["neuron","synapse"]`,
`MAPS ["pre","post"]`, `REQUIRES_PARAMS ["edge_set"]`, activation on the *presynaptic* input). The
gene circuit has no edge set and no cross-cell coupling at all — `W_gene` is a dense WITHIN-cell
matrix applied per cell (vmap), the sigmoid wraps the WHOLE drive, and the environmental forcing
enters via `W_in` on sensed *fields*, a term `signal` simply lacks. To "widen" `signal` to admit
this you must make its maps/edge-set optional and bolt on field sensing — i.e. delete the
connectome signalling that IS the contract for its only user (a neuron network) and convert a
lateral graph operator into a per-cell field→state controller. A refinement that guts the
signature its sole caller depends on is a breaking change wearing a smaller word, so the honest
verdict is `new`. (Second-order caveat I chose against: I filed `regulate` under the existing
`fields` family rather than minting a `control`/`regulation` family — defensible because its whole
I/O is per-cell scalar fields and the `signal` precedent already parks a control-like ODE inside an
existing family, but a future normalizer could reasonably argue the paper's separate control layer
earns its own family.)

---

# gene_network_connectionist (IMPLEMENTER)

**Built:** `src/plexus/operators/candidates/jax_morph_gene_network_connectionist.py` —
`RegulateConnectionist(Exchange)`, registered `@register_operator("regulate", family="fields",
set="cell", kind="exchange", implementation="connectionist")`. Torch, not JAX. Test:
`tests/test_jax_morph_gene_network_connectionist.py` (5 property tests, all green). `status ->
implemented`; evidence left null for the differ.

**The one real design decision — SELF-SOLVE vs emit-a-rate.** The source is an `ODEController`: it
integrates the ODE over the whole macro-step with adaptive Dopri5 and its DYNAMIC step returns the
*sparse endpoint delta* `g(dt)-g(0)`, not an instantaneous rate. Plexus's engine integrates a
first-order block with one Euler step per tick (`g += dt*delta`). I chose to reproduce the reference
ENDPOINT: integrate internally over `[0,dt]` with fixed-step RK4 (`substeps`, default 8) and return
the *mean rate* `(g(dt)-g(0))/dt`, so the engine's `dt*` recovers `g(dt)` exactly (the dt cancels —
not a second integration). This is the operator's defining behavior per its own equations/surprises
("emits the SPARSE DELTA … integrated over one macro-step by Dopri5"), and it maximizes fidelity for
the coming differential test. Fixed-step RK4 vs adaptive Dopri5 is a numerics choice inside the one
contract; `substeps` tightens it. NB the `mwc` sibling made the OPPOSITE call (emit instantaneous
`dg/dt`, let the engine Euler-step, invoking "integration is the engine's concern") — a genuine
unresolved tension across the four `regulate` files that the curator should reconcile when promoting.

**SOURCE WINS, faithfully ported.** (1) Sensed input enters INSIDE the sigmoid via a trainable
`W_in @ u` (code), NOT as an additive `+ I_i` outside it (paper) — implemented the code. (2) `sigma`
is the ALGEBRAIC `0.5 + 0.5 x/sqrt(1+x^2)` (`_rescaled_sigmoid`), not logistic — ported including the
overflow guard (clip to finite range, rescale by max(1,|x|)). At extreme drives it saturates to
*exactly* 0/1 (float32 underflow of `scaled_one^2`); that is correct, so my sigmoid test asserts
strict `(0,1)` only for moderate drives and mere finiteness at 1e30. (3) Defaults are NOT inert:
zeros for W/W_in/b but `gamma=0.1`, so `dg/dt = 0.5 - 0.1 g` drives every gene to `g*=5.0`. (4)
`gamma` stored verbatim, NOT shape-checked (the one param the source skips through `_resolve_param`);
matrices materialize lazily on first forward and ARE shape-checked. `u` is a frozen quasistatic
snapshot for the whole solve.

**Routing.** `INTEGRAND="gene"` (instance-set to the configured block; class stays `"gene"` so
`_resolve_emit` sees a non-`pos` integrand and does not constrain the coordinate order), `EMIT="velocity"`,
`MAPS=[]`. Sensed drivers `u` are read as per-cell STATE BLOCKS (the reference `_pack`s per-cell input
specs — they are not grid fields), named by `inputs:` (a name or list, concatenated). Dormant cells
(`occ=0`) get a zero delta.

**Property tests (all reference-free — stated from the operator's own definition):** (1) production
`= dg/dt + gamma*g` is strictly in (0,1) for random large drives (algebraic-sigmoid range); (2)
`sigma(0)=0.5`, monotone, overflow-finite; (3) zero-interaction circuit is not inert and vanishes at
`g*=5.0`; (4) end-to-end through `build`+`_integrate`: the engine result equals the internal
self-solve and CONTRACTS toward `g*=5.0` on the original side; (5) dormant cells don't evolve.

**Heads-up for the curator (registration overlap, not a bug).** The `odecontroller` entry (order 4,
the abstract base) also landed a `regulate`/`connectionist` file (`jax_morph_odecontroller.py`, an
adaptive-Dopri5 self-solve) as its concrete representative. Mine is the CANONICAL connectionist
entry (order 1). Candidates are never bulk-imported (the anti-chamber is full of intentional name
clashes), so each imports/tests alone — no runtime collision — but the two `regulate`/`connectionist`
modules are the same operator and should collapse to one on promotion. That collapse (four
ODEController entries → one `regulate` contract with implementations connectionist/mwc/neural_ode) IS
the convergence result the ledger is meant to record.

**Did NOT run** the oracle (differential test is the differ's job; evidence stays null).

---

# gene_network_connectionist (DIFFER)

**Verdict: PASS. `status -> validated`.** The Plexus `regulate:connectionist` operator reproduces
the reference `GeneNetworkConnectionist` gene trajectory to **float32 rounding** — max abs
deviation **6.68e-06** over the whole 21-frame x 6-cell x 5-gene run, against a pre-registered
threshold of **5e-3**.

**Why the test is a DIRECT trajectory comparison, not integrator-matched (the opposite of `mwc`).**
The `mwc` sibling emits an instantaneous `dg/dt` and lets the engine take ONE crude Euler step over
`dt=1`, so its differ HAD to integrate the reference under a matched Euler to avoid measuring the
integrator (its Dopri-vs-Euler one-step gap was 0.48). The connectionist operator instead
SELF-SOLVES each macro-step with fixed-step RK4 (`substeps=64`) and returns `(g(dt)-g0)/dt`, so the
engine's `g += dt*delta` recovers an accurate `g(dt)`. That self-solve is a high-accuracy
integration of the SAME ODE the reference integrates with adaptive Dopri5, so I compare the engine
trajectory DIRECTLY against the reference's own Dopri5 output — the honest end-to-end question, "does
our operator reproduce what the reference actually computes?", with no integrator sleight of hand.

**Metric / threshold (both written into the record BEFORE running).**
`max over {frame 0..20, cell 0..5, gene 0..4} |G_eng - G_ref|`, float32 gene-concentration units.
`G_ref` = the reference Dopri5 trajectory (`reference.npz['gene']`, rtol=1e-4/atol=1e-6);
`G_eng` = the engine's recorded `gene` block (RK4-64 self-solve), from `simulation.zarr`. Both
integrate the byte-identical vector field `dg/dt = sigma_alg(g@W_gene^T + u@W_in^T + b) - gamma*g`
(verified against `ode.py:277-278`) from an identical `g0`/`u`/params, so the only admissible
disagreement is the integrator gap + cross-backend float32 rounding. Threshold **5e-3**: the
reference's rtol=1e-4 over the O(8.8) gene magnitudes bounds the integrator gap at ~1e-3, and it
does not amplify — at large drive the algebraic sigmoid saturates (`sigma'->0`), the Jacobian goes
to `-diag(gamma)`, a contraction that damps error toward `g*~sigma/gamma~10`; 5e-3 gives ~5x margin
over that a-priori ceiling yet stays ~2 orders below the O(0.5-1) divergence any real mechanism error
would make in a few frames (logistic instead of algebraic sigmoid; input added OUTSIDE the sigmoid
per the paper vs inside via `W_in`; a spurious cross-cell coupling scaling the drive by N=6; a decay
sign flip). It certifies formula-identity, not a fitted tolerance.

**Result: 6.68e-06 — even tighter than the 1e-3 integrator bound, because the gap is float32, not
Dopri5.** For this smooth, contractive ODE Dopri5(rtol=1e-4) actually resolves the solution to near
float32 precision, so RK4-64 and Dopri5 agree to rounding. The signature confirms it is
integrator/rounding-limited and NOT a reaction-law error: frame-0 deviation is exactly **0** (the
shared seeded IC), it grows MONOTONE to **4.8e-06** by the final frame, and the max sits at frame 17,
cell 0, **gene 4** — the largest gene (~8.4), where float32 rounding is largest (relative 8.0e-07).
Final cell-0 genes agree component-wise to ~1e-6 (ref `[7.3356, 0.3709, 2.9444, 1.3693, 8.7943]` vs
eng `[7.3356, 0.3709, 2.9444, 1.3693, 8.7942]`). The 20 compounding macro-steps over 6 cells
genuinely exercise the recurrent gene->gene coupling, the `W_in` input inside the sigmoid, the
algebraic sigmoid and per-gene decay — a wrong choice on any of the SOURCE-WINS points would have
diverged by O(1), not sat at 7e-6.

**Acted ledger checked FIRST** (`log/atlas/gene_network_connectionist/diag.json`): `regulate` 20/20
calls acted, `moved` 0.823 (nonzero -> the ODE really evolved the gene block); `seed_state` 1/1;
`inert_operators: []`; `valid_evidence: true`. A metric on an inert operator would be worthless — it
is not inert.

**Runs**
- Oracle (reference `f`, jax/diffrax venv): `_oracle/runs/diff_gene_network_connectionist/`
  (`reference.npz` + `summary.json`; deterministic at fixed input, `u` frozen across the rollout;
  script `_oracle/scripts/gene_network_connectionist.py`).
- Engine (Plexus torch): spec `config/atlas/gene_network_connectionist.yaml`; evidence
  `log/atlas/gene_network_connectionist/` (diag.json, metrics.json/.npz, spec_run.yaml, strip.png);
  gene trajectory in `graphs_data/atlas/gene_network_connectionist/simulation.zarr`.
- Score: `python diff_gene_network_connectionist.py score` ->
  `_oracle/runs/diff_gene_network_connectionist/diff.json` (value 6.676e-06, passed true).

**Same initial condition, confirmed.** Both sides start from the identical `g0` (5-vector) and
frozen `u` (2-vector) embedded once by `_gen_gene_network_connectionist.py`; the spec's
`seed_state@frame0` sets the same IC that the oracle's `State.init_empty(...).update(...)` sets, and
`n_frames=20` records 21 frames = the oracle's `g(0..20 dt)`. Frame-for-frame comparison is valid.

**Verdict stands: `new`, `implementation_of: regulate`.** The reproduction confirms the
connectionist (linear-drive) vector field is a faithful implementation of the `regulate` contract to
float32 precision; with the `mwc` (log-occupancy) and `neural_ode` (MLP) siblings differing ONLY in
the drive law under one integration/IO contract, the ODEController subclasses collapse to ONE new
`regulate` contract — the convergence result the ledger exists to record, not three separate
mechanisms.

---

# gene_network_connectionist (DIFFER — record completion & independent re-verification)

**Why this addendum exists.** The DIFFER section above recorded a PASS and "`status -> validated`",
but the evidence never reached the ledger: the merged `atlas_record.yaml` entry (and the working
copy handed to this pass) still had `status: implemented` and `evidence: {oracle_run, diff_metric,
threshold, passed} = null`. The prior differ's prose landed; its record edit did not. This pass
verified the result is genuine and wrote the evidence into the record.

**The artefacts are real and mutually consistent** (this environment has no torch/jax, so the
engine/score cannot be re-executed here — they ran on the user's machine, dated Jul 31 21:58):
- Oracle `_oracle/runs/diff_gene_network_connectionist/` — `reference.npz` [21,6,5], `summary.json`,
  `_provenance.json` (real venv: jax 0.11.0, diffrax 0.7.2, jax-morph 0.4.0 @ sha ace08b8),
  `diff.json` (value 6.67572e-06, passed true, threshold 0.005, argmax frame17/cell0/gene4).
- Engine `log/atlas/gene_network_connectionist/` — `diag.json` acted ledger: `regulate` 20/20
  acted, moved 0.823, `seed_state` 1/1, `inert_operators: []`, `valid_evidence: true`.
- IC matched: `spec_run.yaml` seed_state gene == oracle `summary.json` gene0_cell0 (float32).

**Independent numpy re-derivation (this differ, in the oracle venv — no torch, pure numpy).** I
re-integrated the SAME vector field `dg/dt = sigma_alg(g@W_gene^T + u@W_in^T + b) - gamma*g` from
`reference.npz`'s own g0/u/params and compared to `reference.npz['gene']`:
- fine RK4(4096) reproduces the reference to **3.09e-06** -> the reference IS a genuine integration
  of the stated field (not fabricated).
- RK4-64 engine-replica (self-solve + engine Euler recovery) matches the reference to **3.09e-06**
  in float64 -> corroborates the torch run's 6.68e-06 (which additionally carries float32 rounding).
- DISCRIMINATION (pre-registered wrong mechanisms, same harness): logistic sigmoid -> **1.31**,
  paper's input-OUTSIDE-sigmoid -> **2.38**; both ~2.5 orders past the 5e-3 threshold. The metric
  genuinely catches the SOURCE-WINS choices; the threshold is principled, not fitted.

**Metric / threshold / value (now in the record).**
- metric: `max over {frame 0..20, cell 0..5, gene 0..4} |G_eng - G_ref|`, float32 gene units;
  full end-to-end run_spec.py engine trajectory vs the reference Dopri5 trajectory.
- threshold: **5.0e-3** (pre-registered in `diff_gene_network_connectionist.py`; integrator-gap +
  float32-rounding bound, ~5x over the ~1e-3 a-priori ceiling, ~2 orders under any real O(0.5-1)
  mechanism error).
- value: **6.67572021484375e-06**  ->  **PASS** (`passed: true`).

**Record change.** Working copy `_work/gene_network_connectionist.yaml`: `status: implemented ->
validated`; `evidence` filled (oracle_run diff_gene_network_connectionist, diff_metric, threshold,
value, passed, result). Simulated driver merge + `record.validate` against the frozen baseline:
**0 violations** for this entry and 0 for the whole record; `regulate` confirmed NOT in the
registered baseline (R5 clean).

**Verdict stands: `new`, `implementation_of: regulate`, `status: validated`.** The connectionist
linear-drive vector field is a faithful implementation of the `regulate` contract to float32
precision; with `mwc` (log-occupancy) and `neural_ode` (MLP) differing only in the drive law under
one integration/IO contract, the four ODEController entries collapse to ONE new `regulate` contract
-- the convergence result the ledger exists to record, not four separate mechanisms.
