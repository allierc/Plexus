# SMG Branching Morphogenesis — Mechanism-Search Loop (design for external review)

This replaces the first "LLM-proposes-the-next-batch" loop with a **search system that has memory and a
learned surrogate** — AlphaZero-lite (policy prior + value head + UCB tree) + Bayesian optimization
**over a tree of MECHANISMS, not raw YAML**. Below is the architecture, the discipline that gates it,
and the current status. Feedback especially wanted on: the value/reward design, the mechanism tree, and
the bootstrap-before-UCB protocol.

---

## 1. Why not a free LLM loop
SMG branching is a **deceptive landscape**: many parameter sets make plausible **clusters**, but clusters
are **not on the causal path** to buds/ducts/branches. A free LLM loop re-discovers clusters — empirically,
a 16-batch run did rigorous mechanism-falsification (division = repack, uniform growth = inflate-no-bud,
Keller–Segel = no intrinsic wavelength …) yet plateaued at "clusters that rearrange," because (a) the
operator palette could not express a boundary to cleft against, and (b) the scalar score was partly
gameable. The fix is a search operator suited to a deceptive landscape + a reward that is *anchored to the
real data* and penalizes clusters even when they look organized.

## 2. Forward model
- **Phenomenological Plexus composition** of operators — NOT a per-cell fit.
- **Initialized from the real SMG state** (`x_list` t=0); the model must reproduce the *dynamics*
  (migration + growth + budding) from the real initial condition — a well-posed inverse problem.
- **2D first** (~2–3 min/sim; a 200–500-spec bootstrap runs overnight on 1–2 GPUs); the *winning* branch
  is later escalated to 3D.

## 3. Observables — measured by identical code on real data and every sim
geometry · **topology** (branch GRAPH + genealogy: main duct → branch → subbranch generations) ·
velocity (dense optical-flow PIV) · growth map (continuity residual ∂ρ/∂t+∇·ρv) · density.

## 4. Value vector — what the surrogate predicts and UCB scores
`cluster_score · bud_score · duct_score · branch_count · branch_length_ratio · tip_growth_localization ·
migration_coherence · target_distance`. **Key design point:** a cluster and a nascent duct differ in
*branch-genealogy + elongation + tip-localized growth*, **not in density** — so `cluster_score` and
`duct_score` come from the same locked topology readout.

## 5. Reward — anchored to real SMG, penalizes clusters (`smg_reward.py`)
- `duct_score`/`cluster_score` are **normalized against the real gland** (`BLR_REF=1.83, GEN_REF=5.0`
  from 5 real frames → `_calib.json`), so real anchors at duct≈0.9, cluster≈0 — not an absolute heuristic.
- **Automatic failure taxonomy** (the surrogate learns the *kind* of failure, not just pass/fail):
  `unstable · no-growth · overgrowth · fragment · blob · sheet · cluster · branch-like`.
- **Stage rewards** (early stages penalize clusters hard): e.g. `connect: −1.5·cluster + duct + 0.3·migr`.
- **CALIBRATION GATE (must pass before any search):** real SMG must strongly separate from cluster/blob in
  the *value vector*, not just the class label. **Current status — PASS:**

  | case | cluster | bud | duct | class |
  |---|:--:|:--:|:--:|---|
  | REAL SMG | **0.00** | 0.24 | **0.88** | branch-like |
  | cluster (6 blobs) | 0.62 | 0.03 | 0.05 | fragment |
  | blob | 0.29 | 0.05 | 0.05 | no-growth |
  | branch (Y) | 0.14 | 0.49 | 0.49 | sheet |

  Gate checks all PASS: real duct>0.6, cluster<0.2, class=branch-like, duct ≫ cluster/blob duct.

## 6. Mechanism tree — a tree over BIOLOGICAL HYPOTHESES (`mechanism_tree.py`)
7 branches, each activating **one biological hypothesis** (not an operator list) so the search is
scientifically interpretable, not just combinatorial; UCB decides which hypothesis gets compute, specs
are sampled *within* a branch, all built on the real-init substrate:
`baseline_migration` (null control) · **differential_adhesion** (cell-cell vs cell-matrix, Wang–Yamada)
· **chemotaxis** · **ecm_guidance** (deformable boundary) · **growth_instability** (differential growth
buckles) · **reaction_diffusion** (Turing prepattern) · **mechanical_buckling** (stiffness heterogeneity).
A structured **encoder** (47-D: operators-present multi-hot, normalized scalar params, field-type /
growth-law / branch one-hots) turns any (branch, params) into the surrogate's input.

## 7. Operators — palette gaps, grounded in the vendored repos
A branch can only be reached if its operators exist. Missing operators, mapped to source algorithms:
| operator | source | mechanism |
|---|---|---|
| **`ecm_boundary`** | Chaste `immersed_boundary` + `ImmersedBoundaryLinearDifferentialAdhesionForce` | basement membrane as virtual Lagrangian nodes (spring+curvature stiffness) with **cell-matrix vs cell-cell** differential adhesion = **Wang–Yamada's SMG budding mechanism** |
| `growth_gate`/`growth_field` | Chaste `CellwiseSourceParabolicPde` + `gray_scott` (vendored) | morphogen field gates local `cell_grow` |
| `stiffness_field` | CompuCell3D `Elasticity`/`LengthConstraint` | spatial Young's modulus (soft ducts) |
| `slow_field` | — | slowly time-modulated growth field |
These are built + unit-tested (register + one step on the base substrate) **before** the bootstrap.

## 8. Search algorithm
1. **Bootstrap the failure manifold first** — 200–500 specs (stratified over branches) → run on GPU →
   dataset (`encoding → value vector → failure class`). Sampling is **biased 50% random / 25%
   hand-plausible / 25% perturb-best** (not pure-uniform) so the dataset contains failures AND
   "almost-working" tissue — otherwise it is ~95% fragments = an "everything fails" value function.
2. **Train an ensemble surrogate** — `spec → MORPHOLOGY VECTOR` (`{duct, cluster, bud, branch_count,
   migration, growth_ratio, …}`), **not a scalar reward**; reward is computed *from* the predicted
   vector (richer supervision, smoother landscape, interpretable errors, reusable). RandomForest first;
   **ensemble variance = epistemic uncertainty** for exploration.
3. **Gate the surrogate** — held-out R² and failure-class separability before trusting it.
4. **UCB over branches** (shallow bandit first; MCTS with add-mechanism edges later) → surrogate
   **pre-screens** many candidate specs → run only high-value / high-uncertainty candidates on GPU →
   add results → retrain. Real simulator = expensive ground truth; surrogate makes it tractable.

## 9. Components + status
| file | role | status |
|---|---|---|
| `smg_reward.py` | value vector · failure taxonomy · stage reward · **calibration gate** | **built; gate PASSES** |
| `mechanism_tree.py` | 7 hypothesis branches · spec generator · 47-D encoder | **built; all branches build+run** |
| `operators_smg.py` | `ecm_boundary`·`growth_field`·`slow_field`·`growth_gate`·`chemotax_field`·`stiffness_field` | **built + tested** (ecm 3-test contract passes) |
| `bootstrap.py` | biased (50/25/25) stratified → dataset | **built + validated** (all 7 branches run) |
| `surrogate.py` | ensemble `spec → morphology vector` + uncertainty | **next** |
| `ucb_loop.py` | UCB over branches + surrogate pre-screen | queued |

## 10. Open questions for the reviewer
- Reward: is normalizing `duct`/`cluster` to the real gland (vs a fixed heuristic) the right anchor?
  Should `branch_count` be duct-grade-pruned harder (currently ~5 on real; `duct_score` doesn't depend on it)?
- Tree: are the 8 branches the right mechanism decomposition? Missing a differential-adhesion-only branch?
- Protocol: 200–500 bootstrap specs enough to learn the failure manifold in a 48-D encoding? RF vs GP surrogate?
- 2D→3D: escalate only the winning branch, or keep a 3D control from the start?
