# A GraphCast-style fitting model as Plexus2 operators — clean restart

## Context

The first pass built the container and a toy, then discovered — three times — that the *data* did
not pose the problem it claimed to. Each failure cost a training run: a circuit at a fixed point,
a stimulus field identically zero, and a travelling wave whose gradient is recoverable node-locally
so the graph was never needed. The container, the gate machinery and the two-scale rule are sound
and worth keeping; the ordering, the toy design and the reporting standard were wrong.

This is a restart in place. Git preserves the old work (`prototype/graphcast-plexus`, three
commits); `prototype/graphcast/` is rebuilt so every requirement below holds from the first commit
rather than being retrofitted.

**Goal:** a model whose four architectural choices are config options, expressed as Plexus2 sets,
fields and operators, tested on three toys with ground truth, then ZAPBench, then redox.

---

## 1. Requirements — every one of these is binding

| # | requirement |
|---|---|
| R1 | Four **options**, not forks: `encoder_decoder` off/on, `message` simple/graphcast, `n_passes` 1…16, `embedding` none/free/multires — see "Why 24" below |
| R2 | `a_i` carries the **heterogeneity** and is central; it must be injected into the GraphCast form, not bolted on |
| R3 | **Spatial coupling first.** No connectome set; that is a later addition |
| R4 | Expressed as Plexus2 **sets / fields / operators**, respecting composition — learnable operators |
| R5 | **Nothing hardcoded.** All of it in config: `data:` and `training:` sections added to the Plexus spec |
| R6 | Specs declare **units** (`plexus.units`), and every measurement threshold is in phenomenon units |
| R7 | **One entry script**: `engine.py -o <task> <config>`, tasks generate/train/test/plot/gates, mirroring `GNN_Main.py` |
| R8 | **Reuse, don't rewrite**: metrics, plots, tests, training schemes already in the workspace |
| R9 | **Gates before implementation**, thresholds fixed in advance, three tiers |
| R10 | Gates archived in **git** *and* in a dedicated **PDF** |
| R11 | **Every gate points to a PNG and, where the quantity is a trajectory or field, an MP4** |
| R12 | The PDF **embeds** the PNGs and **links** the MP4s |
| R13 | Gate figures must show the **field** and the **heterogeneity map** |
| R14 | **Data is gated before any training** |
| R15 | Archive: `graphs_data/<dataset>/viz/` PNG+MP4; `log/<run>/{tmp_training,test,plot}/` PNG+MP4 |
| R16 | **Figure standard**: white background; panel labels **above** the axes, **not bold**; one *read* font size across all figures |
| R17 | **Publication quality.** Re-read the rendered PDF before shipping, every time |
| R18 | Build on the workspace's **training schemes**: per-group LRs, t+1 vs recurrent, regularisers, batching |

---

## 2. Reuse map — read, do not reimplement

**Plexus operators** (`src/plexus/operators/`): `mpm_scatter`/`p2g` (`mpm_ops.py:342`),
`mpm_gather`/`g2p` (`:1168`) — the transfer pair for the encoder/decoder option; `neuron_update`,
`neuron_signal`, `neuron_field_input`, `pacemaker`, `activation_pulse`.

**Plexus infrastructure**: `plexus.schema.load` (ignores unknown top-level keys — this is what lets
`data:`/`training:` live in the same yaml); `plexus.units.Units`; `plexus.models.base`
(`Operator.tunable()` keeps a knob learnable through construction); `plexus.models.registry`;
`plexus.engine.run` (`grad=True` keeps the tape); `edges_file:` already resolved at `engine.py:441`.

**connectome-gnn**: `metrics.py` (`compute_r_squared_lin_fit`, `recovery_param_metrics`,
`compute_activity_stats`); `sparsify.py:202` `clustering_evaluation`; `plot.py` (`plot_embedding`,
`plot_g_phi`); `GNN_Main.py:110-384` for the `-o` dispatch.

**Training schemes** — from `config/fly/flyvis_noise_005_calib_nominal_l4.yaml` and the weekend
benchmark:

| scheme | source | value |
|---|---|---|
| three learning rates | production config | `lr_W 0.0009`, `lr 0.0018`, `lr_embedding 0.002325` |
| `W` sparsity | production config | `coeff_W_L1 1.5e-4`, `coeff_W_L2 1.5e-6` |
| message smoothness | production config | `coeff_g_phi_diff 750` — the largest term there |
| weight L1/L2 | production config | on `g_phi` and `f_theta` matrices |
| group lasso | weekend benchmark §2 | **+0.153 R²_W on 5/5 folds**, by removing a degeneracy |
| t+1 vs rollout | weekend benchmark §4 | plain t+1 wins; `pushforward` −0.082, `last` −0.028 |
| curriculum | GraphCast suppl. §4.5 | 96% of updates at K=1, then a short tail at LR/3300 |
| target norm | GraphCast suppl. §4.2 | inverse variance of the **increment**, not the state |
| optimiser | GraphCast suppl. §4.4 | AdamW β₂=0.95, global grad-clip 32, cosine **to zero** |

Two corrections already paid for and carried in: weight decay must **not** touch `W`, `a`, `b` —
they are the scientific output; and `W` is identifiable only up to a **per-sender gauge**, since
`W_e·g_φ(v_j,a_j)` is invariant under `W_e → c_j W_e`.

---

## 3. The toy: two scales, two rules, uncoupled

*Rewritten after the node-based toys were abandoned. The three earlier variants — `toy_counter`,
`toy_withhold`, `toy_envelope` — differed only in the coarse rule and all three failed G26 at
about 1.000: a linear per-node ODE driven by a smooth quasi-periodic field is autoregressively
predictable from its own history whatever drives it, so no choice of coarse rule could rescue a
fine rule that was locally solvable. Their configs remain in `config/` as the record of that.*

The toy is now a **spatial PDE with different rules at different scales**, and no neurons at all.
Two mechanisms, at different resolutions and different rates, **not coupled to each other**:

| | coarse | fine |
|---|---|---|
| rule | transport, `∂u/∂t + c ∂u/∂x = 0` | Kuramoto, `∂φ/∂t = ω(x) + K Σ_nbr sin(φ_j − φ_i)`, observable `v = sin φ` |
| order | first in time, first in space | first in time; the coupling is a *saturating* second spatial difference |
| resolution | 256² / 64³ | 1024² / 256³ |
| support | the whole domain | four discs (2-D) / four **tubes** (3-D) |
| heterogeneity | none — one rule, no freedom | **`ω(x)`, per region and per pixel** — this is what `a_i` has to carry |

**They must not couple, and the split runs prove they do not.** `u` never reads `v` and `v` never
reads `u`, so running each spec alone gives a trajectory bit-identical to running both and looking
at one channel — verified: `std(u) = 0.7071067` and `std(v) = 0.2778887` agree to seven digits
between the split and combined runs, as do the autocorrelations. The model's task is therefore to
**separate two mechanisms**, not to trace a cascade from one into the other.

**Different in kind, not the same rule twice.** Transport moves a fixed profile at a fixed speed and
has no attractor; Kuramoto synchronises, and where it fails to synchronise it makes phase defects —
spiral cores and travelling target patterns. Neither behaviour resembles the other, so a model that
captures both has captured two mechanisms rather than one rule at two settings. **No second time
derivative anywhere**: the original sketch was a wave, and a wave is second order in time.

### The six runs

Three specs per dimension, each a complete standalone Plexus spec writing its own zarr into its own
run directory — so "the coarse field" is an archived object, not a channel someone has to remember
to slice out.

| spec | fields | run directory |
|---|---|---|
| `toy2d_coarse.yaml` / `toy3d_coarse.yaml` | `u` only | `log/toy{2,3}d_coarse_noed_simple_p1_none/` |
| `toy2d_fine.yaml` / `toy3d_fine.yaml` | `v` only | `log/toy{2,3}d_fine_noed_simple_p1_none/` |
| `toy2d.yaml` / `toy3d.yaml` | both, plus the sum | `log/toy{2,3}d_noed_simple_p1_none/` |

The sum is **not** written to disk: it is `interpolate(u) + v` exactly, both terms are already
archived, and a third copy — 4.3 GB at 256³×64 — could only drift out of step with the two it
derives from. The MP4 is the artifact; the recipe is one line.

### Measured, on the data as generated

| | 2-D | 3-D |
|---|---|---|
| coarse lag-1 autocorrelation | 0.998 | 0.977 |
| fine lag-1 autocorrelation | 0.818 | 0.797 |
| **rate separation** (coarse traverse / fine period) | **20×** | **10×** |
| samples per fine period | 10 | 6.4 |
| fine mask fraction | 0.154 | 0.104 |
| fine spatial correlation half-length | 8 px of 1024 | — |

**Lag-1 autocorrelation is in the table because getting it wrong cost a day.** A fine rule tuned to
`ω = 0.45` rad/substep was sampled every 6 frames, so consecutive records were ~8 rad apart: the
movie showed noise, and the "1 − autocorrelation ≈ 0.83" that was read as *fast* was measuring
*undersampled*. An aliased oscillation and a fast one are indistinguishable by eye. Near 1 means
resolved; near 0 means aliased; the number is now written by the generator into `summary.json`
before any gate runs.

Two knobs, and they are independent — moving both at once was the second error of the same day:

- **`ω` sets the rate.** `ω = 0.035` rad/substep × 12 substeps × dt 0.25 → fine period ≈ 60 frames.
- **`K` sets the domain size.** `K/ω_spread` = 75; at `K = 0.11` the phase domains were 2 px across
  and the render was sub-Nyquist, which looks exactly like "no pattern".

### Why the 3-D fine rule lives in tubes

A ball of radius 0.1 at the centre of a 256³ volume cannot be seen from outside: a ray cast
integrates through everything in front of it. A **tube spanning the box meets two faces**, so the
fine pattern is legible on the outside of the cube without cutting it open. Four tubes, two along
`z` and one each along `x` and `y`, so all six faces carry some. Same rule, same mask fraction,
visible geometry.

### Rendering: VTK in both dimensions

Both dimensions go through `plexus.render_vtk` (`vtk_toy.py`), so a 2-D panel and a 3-D panel are
the same picture at different D — one colour map (**viridis**), one background, one caption line,
one movie container. The 3-D path *is* `render_vtk.evolve_volume`, reached by writing the
`trajectory.npz` layout it reads, not a reimplementation of it.

The one genuine difference is stated rather than hidden: a volume render shows only what its
opacity transfer function passes, so 3-D ray-casts `|field|`, while a plane has no such integral
and 2-D shows the **signed** field. The transfer function is chosen by measuring the field's own
zero fraction — `sigmoid_5` when it fills the box, a zero-floored ramp when it is masked. That
matters: on the masked fine field the sigmoid's non-zero alpha at `|v| = 0`, integrated over ~256
empty voxels, rendered the vacuum as purple fog, and saturated the tube walls opaque so the pattern
showed only at the faces.

---

### Why 24

The switches are `encoder_decoder` (off/on), `message` (simple/graphcast), `n_passes`, and
`embedding` (none/free/multires). That looks like it should multiply out bigger, but `simple`
carries no edge state, so running it twice isn't "the same model, deeper" — it's a different model
wearing the same name. The schema refuses that combination outright. What's left is four legal
`(message, n_passes)` pairs — `(simple,1)`, `(graphcast,1)`, `(graphcast,4)`, `(graphcast,16)` —
times 2 encoder/decoder times 3 embeddings: **24**.

That count is what G1 and G1b check, and the enumeration is balanced by construction: 6 configs per
`(message, n_passes)` pair, 12 per encoder/decoder setting, 8 per embedding.

## 4. Plexus decomposition

**Sets** — `node` (the fine scale). **Fields** — `u` (coarse), `mesh` (only when
`encoder_decoder: on`).

| operator | kind | learnable | gated by |
|---|---|---|---|
| `wave_field` | `field` | no | always (three variants by `impl:`) |
| `gradient_gain` | `exchange` | no (generator only) | always |
| `gc_message` | `lateral` | **ψ, φ, W** | `message`, `n_passes` |
| `gc_embedding` | `broadcast` | **a_i** | `embedding` |
| `gc_scatter` / `gc_gather` | `exchange` | no — wraps `mpm_scatter`/`mpm_gather` | `encoder_decoder` |
| `gc_stimulus` | `field`→`exchange` | **b_i** | always |

`a_i` enters the **message input** (`[v_j, s_j, a_j]`) and the update, so heterogeneity acts at the
point of interaction — the placement the GraphCast form otherwise lacks.

---

## 5. Config, entry point, archiving

One yaml per dataset: `general:` (+`units:`), `sets:`, `fields:`, `operators:`, `schedule:` (the
Plexus half, unchanged), plus `data:`, `model:`, `training:`. `spec_schema.py` parses the additions
and validates the options as enums. YAML 1.1 coerces bare `off`/`on` to booleans — handled in the
schema, not by requiring quotes.

```
python engine.py -o generate config/toy_counter.yaml
python engine.py -o train    config/toy_counter.yaml
python engine.py -o test     config/toy_counter.yaml [checkpoint]
python engine.py -o plot     config/toy_counter.yaml
python engine.py -o gates    config/toy_counter.yaml
```

Archiving (R15): `graphs_data/<dataset>/viz/` gets the summary PNGs and a state MP4;
`log/<run>/tmp_training/` gets periodic PNGs during training; `log/<run>/test/` and `plot/` get the
test figures and assembled MP4s.

---

## 6. Figure and document standard (R16, R17)

Enforced in `viz.py`, not per figure:
- **white** background, black text;
- panel labels **above the axes** (`set_title(loc="left")`), never inside the data area, and
  **not bold** — regular weight at the same size as the rest of the figure;
- **one figure width** (`FIGW`) for every figure, because all are included at `\linewidth` — equal
  point sizes only *read* equal if the scale factor is equal. Heights may vary;
- one base point size, everything derived from it.

Document: fixed column widths in the gate table (an unbounded `l` column crushes its neighbours),
every PNG embedded with a caption naming gate/threshold/measured/outcome, every MP4 hyperlinked.
**The rendered PDF is read page by page before every commit** — the table was shipped unreadable
once, and the figures were shipped stale once.

---

## 7. Gates — pre-registered, three tiers

Thresholds are literals in `gates.py`, never config. An artifact is a **condition of passing**.

### Tier 1 — bookkeeping (does the code do what the operator says?)

| id | gate | threshold | status |
|---|---|---|---|
| G1 | every option combination can be READ | 24 of 24 option combinations load | done |
| G1b | every option combination can be RUN | 24 of 24 option combinations run one forward step | pending |
| G2a | no dataset identity appears as a VALUE in the code | 0 offending constants outside config/ | done |
| G2b | ONE pipeline actually runs on all three datasets | 3 of 3 datasets complete generate/train/test with only the config changed | pending |
| G3 | the transfer pair returns what it was given | < 1e-6 of the field value | pending |
| G4 | the transfer conserves what it moves | \|sum(w) - 1\| < 1e-6 | done |
| G5 | the simple option IS the existing model, arithmetically | < 1e-5 of the voltage range | pending |
| G6 | depth is an option, not a different model | bit-identical (max \|delta\| == 0) | pending |
| G7 | the spec is allowed to carry a unit | units declared, and no measurement threshold in mesh units | done |
| G7b | a measurement result is REPORTED in the declared unit | every tier-3 measured value carries its declared unit through the conversion | pending |
| G8 | one-step accuracy is not stability | state norm stays < 2x the ground-truth norm | pending |

`status` is not `outcome`. **Outcome** is what the number did against the threshold;
**status** is whether the gate has been walked through — definition written, estimator
sanity-checked, negative control run, result read by a human. A gate can pass
mechanically and still be pending review, and that difference is the difference between
a number and evidence.

#### What each gate is for

**G1 — every option combination can be READ.** The model has four switches, and the whole premise is that any setting of them is a legal model -- options, not forks. This gate checks that literally: every combination must be readable. It takes the reference config, overwrites only the model block with each of the 24 settings, and calls the loader. Why 24: `simple` carries no edge state, so running it twice is a different model wearing the same name and the schema refuses it, which leaves four legal (message, n_passes) pairs times 2 encoder/decoder times 3 embeddings. It catches an option that exists only in the documentation, and a combination that silently falls back to a default instead of erroring.

**G1b — every option combination can be RUN.** The other half of G1. Reading a config proves the vocabulary is right; it does not prove the model can be built from it. This gate constructs the model object for each of the 24 settings and pushes one forward pass through it. It is split from G1 because the two become available at different stages -- G1 needs only the schema, G1b needs a model -- and a row that goes green having checked half its claim reads as an endorsement it has not earned. It catches shape mismatches that appear only when two particular options meet.

**G2a — no dataset identity appears as a VALUE in the code.** A scan of the prototype's own Python on the ABSTRACT SYNTAX TREE rather than as text, checking every string and numeric constant except docstrings. That distinction is the point: naming a dataset in prose is documentation, while the same name used as a value is a hardcoded path. Scanning the text would flag the first; skipping strings entirely would miss the second, because a path IS a string. Reviewing it found a hole -- the patterns were word-bounded, so a name inside a filename slipped through and a planted-violation check caught only 2 of 3. The boundaries are gone and all 4 are caught, while docstring prose and innocent constants still are not. Note what this does NOT establish: it is a necessary condition, not the claim. See G2b.

**G2b — ONE pipeline actually runs on all three datasets.** The claim G2a only gestures at. The point of forbidding dataset identity in the code is that the same trainer should run on a toy, on a point-cloud recording and on a field recording with nothing changing but the yaml. That can only be checked once all three loaders exist and all three have been run end to end, which is stage 8. Until then the scanner is passing partly because the most likely place to hardcode a path -- the ZAPBench and redox loaders -- has not been written yet.

**G3 — the transfer pair returns what it was given.** The encoder/decoder option moves state onto a background grid and back. If it is sound, depositing a constant and gathering it again returns the constant. This is the end-to-end version of G4 and it catches what G4 cannot: a transfer pair that is not each other's adjoint, an off-by-one in the stencil, a normalisation applied on one side only. The threshold is a FRACTION of the field value, so it does not depend on what the field is.

**G4 — the transfer conserves what it moves.** The local half of G3. Each transfer spreads a node's value over the corners of the grid cell it sits in, and those weights must sum to one, or the transfer quietly changes the total amount of stuff every time it is applied. Summing to one is also exactly the condition that makes interpolation reproduce a constant, which is why G3 tests the same property from the outside. Dimensionless by construction. Its stage is 0, not 5, because it tests `mpm_ops.bspline` directly -- the transfer the encoder/decoder option wraps already exists, so no model and no wiring are needed. Measured 2.4e-07 over 2-D and 3-D at three resolutions, which is float32 machine precision, so the 1e-6 threshold is about eight epsilons: sharp enough to catch a real error, loose enough not to fail on rounding. The negative control -- dropping the middle B-spline lobe -- reads 9.4e-01.

**G5 — the simple option IS the existing model, arithmetically.** The pivot of the whole prototype. With `simple`, one pass and no encoder/decoder, this model is meant to be connectome-gnn's NeuralGNN term for term. The gate copies NeuralGNN's weights across and requires the two to produce the same numbers. If it passes, everything downstream is a controlled variation on a model already known to reach R^2_W around 0.97. If it fails, no later result can be interpreted at all, because a new model's failure and a reimplementation bug are indistinguishable.

**G6 — depth is an option, not a different model.** Both message-passing MLPs have their final layer initialised to zero, so every residual block is EXACTLY the identity before training. It follows that one pass and sixteen passes must give the same numbers at step zero. The threshold is exactly zero with no tolerance, because this is an algebraic identity rather than a numerical one. It catches a residual that is not a residual -- a missing skip connection, or an initialisation that makes the stack a different model at every depth.

**G7 — the spec is allowed to carry a unit.** Two halves, both with teeth. The spec must declare a units block, because plexus/units.py is explicit that a model without one is dimensionless and no result from it may be quoted with a unit -- and every measurement-tier gate is a comparison against a quantity. And no measurement threshold may be denominated in grid cells, voxels or steps; that half is the lesson from the ecm study, where a penetration of 0.82 grid cells sounded small and was 15 microns, nearly two cell diameters. A threshold in the mesh's own currency is the easiest one to pass. Both halves were checked against negative controls: a spec with no units block is refused, a spec declaring a DERIVED unit is refused, and poisoning one measurement gate's unit to 'grid cells' makes this gate fail and name the offender. WHAT IT DOES NOT ESTABLISH: it compares a unit LABEL against a blocklist, so it verifies that the declaration is honest in form, not that the number is really in that unit; and it says nothing about whether any result has actually been converted. See G7b.

**G7b — a measurement result is REPORTED in the declared unit.** The half G7 cannot reach. Declaring length_um = 100 does not convert anything; it only makes a conversion possible. Whether a measured tier-3 value is actually reported in seconds or micrometres rather than in frames or cells can only be checked once a tier-3 gate has run, which is stage 7. Until then G7 establishes that the spec is ALLOWED to carry a unit, and nothing about whether it does.

**G8 — one-step accuracy is not stability.** A model can predict the next increment almost perfectly and still blow up when it is fed its own output twenty times over, because a one-step fit never sees its own error compound. This runs a 20-step rollout and requires the state to stay bounded. The threshold is a RATIO to the ground-truth norm, so it is dimensionless and means the same thing on the toy and on real data.

### Tier 2a — the toy is a valid test bed (DATA only, before any training)

Added after three toys failed for reasons that had nothing to do with any model, and in every
case training had been run before the data was known to pose the problem it claimed to. These
need no model. Thresholds are principled rather than fitted: a deterministic rule is
recoverable at R² > 0.95 by definition, and 0.80 excludes collinearity rather than describing
what was seen.

**G26 is currently failing on all three toys and is what blocks the design.**

| id | gate | threshold | status |
|---|---|---|---|
| G16 | the types cannot be read off position | spatial-cell purity within 20% of a label-permutation null | pending |
| G21 | the coarse field is the rule it claims | phase speed within 5% of lambda/period | pending |
| G22 | the fine rule is recoverable from state and gradient | minimum per-node R^2 > 0.90 | pending |
| G23 | the gradient is reconstructible from neighbours | R^2 > 0.95, else the graph cannot carry the fine rule | pending |
| G24 | the heterogeneity is linearly readable | corr(fitted gain, true g_i) > 0.90 | pending |
| G25 | connected nodes are not collinear | mean \|corr\| between connected nodes < 0.80 | pending |
| G26 | the graph is NECESSARY: a node-local baseline cannot fit | node-local R^2 < 0.50 while (v, grad u) exceeds 0.90 | pending |

#### What each gate is for

**G16 — the types cannot be read off position.** G11 asks whether the embedding recovers the node types. That is only a real test if the types cannot be read off position, because position is free information the model already has. The toy assigns types by a permutation independent of position and this measures that it worked, as the purity of a spatial cell against a LABEL-PERMUTATION null so that 1.0 is chance at any resolution. The null is empirical rather than 1/n_types, and that matters: at 32 cells per axis there are 1024 cells for 1024 nodes, so almost every occupied cell holds one node and its purity is 1.0 by construction. The first version read 6.1x chance and meant nothing but 'the grid is finer than the sampling'.

**G21 — the coarse field is the rule it claims.** The spec says a wave travels left to right at lambda over T. This checks that the field actually written does. It projects the recorded field onto the known wavelength, unwraps the phase and measures the drift per frame. It catches a field that is static, identically zero, or moving at the wrong speed -- an earlier toy ran three generations with a stimulus field of exactly zero, because the operator read a clock that nothing was writing. The estimator had to be sharpened twice: argmax on a 128-cell grid quantises to whole cells while the wave moves half a cell per frame, and an integer FFT bin then biased the speed by exactly 6.67/7. An estimator has to be sharper than the threshold it is judged against.

**G22 — the fine rule is recoverable from state and gradient.** Before asking a model to learn dv from the state and the field gradient, check that dv IS a function of them. A per-node linear regression of dv on (v, grad u), reporting the WORST node rather than the mean -- because a mean of 0.98 can hide a third of the nodes at zero, and that is exactly what happened when 58.8% of nodes sat outside the field domain where sampling clamps and the gradient is identically zero.

**G23 — the gradient is reconstructible from neighbours.** The model has to build the field gradient out of its neighbours; this checks that doing so is possible at all. It regresses the true gradient at each node on the differences between its neighbours' states and its own. If it fails, no message-passing model can learn the fine rule and nothing downstream of it means anything.

**G24 — the heterogeneity is linearly readable.** The signed gain g_i is what the embedding must carry, so it has to be present in the data before any model is asked to find it. From the same per-node regression as G22, the coefficient on the gradient IS dt times g_i; this correlates it against the truth. The fitted-to-true ratio should be 1.0 and reads 1.03, the residual being the finite-difference step against the sampled field rather than the analytic one.

**G25 — connected nodes are not collinear.** If a node's neighbours are near-copies of it, their states carry nothing it does not already have and the graph is decoration. This measures the mean absolute correlation between the time series of connected nodes. It caught a real defect: at wavelength 0.5 a twelve-neighbour ball spans only 0.5 radians of phase, the measure read 0.84, and that is why an earlier fit drove the loss to 0.005 while recovering none of the mechanism. Shortening to 0.15 moved it to 0.61.

**G26 — the graph is NECESSARY: a node-local baseline cannot fit.** The strongest of the data gates and the one that would have caught the travelling-wave defect directly. A deliberately generous node-local baseline -- four lags of the node's own state, plus its own drive where observed, and NO neighbour -- must FAIL where the neighbour-informed fit succeeds. It is generous on purpose: the gate is only informative if the thing it rules out was given every chance. It catches a test bed whose fine rule is solvable without the graph at all, which is the case for any u = f(x - ct), since there du/dx = -(1/c) du/dt. It is currently FAILING at about 1.000 on all three coarse rules, and the reason is more general than the coarse rule: a linear per-node ODE driven by a smooth quasi-periodic field is autoregressively predictable from its own history whatever drives it, so no choice of field can rescue a fine rule that is locally solvable.

### Tier 2b — closed form (does the fit reproduce the physics it was given?)

These need a trained model and so are unavailable until stage 3. G12, which scored the
embedding against 65 types on a flyvis-scale toy, was removed when that toy was dropped.

| id | gate | threshold | status |
|---|---|---|---|
| G9 | the message becomes a gradient operator | R^2 > 0.90 against the true field gradient | pending |
| G10 | recover the per-node time constant | R^2 > 0.95 against the known tau | pending |
| G11 | the embedding recovers the types | ARI > 0.70 against the true type labels | pending |
| G13 | recover the per-node SIGNED GAIN (the heterogeneity) | R^2 > 0.90 against the true g_i | pending |
| G14 | encoder/decoder is a genuine option | \|delta R^2(gradient)\| < 0.03 | pending |
| G15 | graphcast vs simple is RESOLVED, either way | \|delta\| reported against a 3-seed floor; below it is UNRESOLVED, not ranked | pending |
| G27 | which coarse rule forces the graph | the three toys ranked by G26; reported, not tuned | pending |
| G28 | known-ODE recovers the coarse speed c from the coarse field | \|c_hat - c\| / c < 0.01 | pending |
| G29 | known-ODE recovers K and omega_i from the fine field | \|K_hat - K\| / K < 0.05 AND R^2(omega_hat, omega) > 0.90 | pending |
| G30 | known-ODE recovers BOTH rules from the SUM alone | c within 5%, K within 10%, R^2(omega_hat, omega) > 0.80 | pending |

#### What each gate is for

**G9 — the message becomes a gradient operator.** On this toy the fine rule IS a spatial derivative, so 'did the model recover the interaction' and 'did the aggregated message become du/dx' are the same question -- and the second can be measured directly against ground truth rather than through a proxy such as an edge-weight correlation.

**G10 — recover the per-node time constant.** Read off the trained operator's own Jacobian: d(dv_i)/dv_i is -1/tau_i for a leaky unit. Taken from the OPERATOR rather than from a named parameter, so the same measurement works for both message forms and does not assume the model wrote tau down anywhere.

**G11 — the embedding recovers the types.** The headline scientific readout, scored the way connectome-gnn scores it: cluster the embedding and take the adjusted Rand index against the true labels. The 0.70 threshold is the flyvis Ward-tree reference, which reaches 0.702 against 65 cell types. It is only meaningful because G16 established that the types cannot be read off position instead.

**G13 — recover the per-node SIGNED GAIN (the heterogeneity).** The heterogeneity itself rather than a proxy for it, read as d(dv_i)/d(msg_i) from the trained operator. Signed matters: a model that recovers the magnitude and flips the sign fails, and it should, because an inverted gain is a different claim about the mechanism, not a small error.

**G14 — encoder/decoder is a genuine option.** On a toy where the node set already IS the computation set, routing through a background grid should change the answer very little. The 0.03 threshold is twice the 0.015 run-to-run resolution floor measured on flyvis_A in the weekend benchmark. It catches an option that silently changes the model rather than the route it takes.

**G15 — graphcast vs simple is RESOLVED, either way.** Not 'graphcast wins'. The weekend benchmark's discipline: report the difference against a floor measured from three seeds, and call anything below that floor UNRESOLVED rather than ranking it. It catches the temptation to read a 0.006 gap as a result -- which is how that benchmark found that four of its seven rollout arms were indistinguishable.

**G27 — which coarse rule forces the graph.** Ranks the three coarse rules by G26 and reports the spread. Explicitly not a tuning target: the point is to learn which rule makes the graph necessary, and a rule that only passes after being adjusted until it passes has told us nothing. G12, which scored the embedding against 65 types on a flyvis-scale toy, was removed when that toy was dropped from the plan.

### The two equations the known-ODE gates fit

Written once here, referenced by G28–G30. Both are the generator's own rules with every constant
replaced by a learnable parameter — the `connectome-gnn` `known_ode.py` construction, which is a
model with no network in it at all.

**Coarse — pure transport, one unknown scalar.**

```
du/dt = -c du/dx                                                              (C1)
```

| symbol | is | unit | true value |
|---|---|---|---|
| `u(x,t)` | the coarse field | dimensionless | — |
| `c` | phase speed, THE ONLY LEARNABLE | domain widths per frame | 0.000833333 (one traverse in 1,200 frames) |
| `du/dx` | centred difference on the periodic axis | per domain width | — |

**Fine — coupled phase oscillators, one scalar and one field of unknowns.**

```
dphi_i/dt = omega_i + K SUM_{j in N(i)} sin(phi_j - phi_i)                    (F1)
v_i       = sin(phi_i) * m_i                                                  (F2)
```

`phi` is *not observed*, and (F2) is many-to-one, so (F1) cannot be fitted to `v` as written. In
the quadrature pair `v = sin phi`, `w = cos phi` the identity `sin(phi_j - phi_i) = v_j w_i - w_j v_i`
closes the system in the observables:

```
r_i     = omega_i + K SUM_{j in N(i)} ( v_j w_i - w_j v_i )                   (F3)
dv_i/dt =  w_i r_i m_i                                                        (F4)
dw_i/dt = -v_i r_i m_i                                                        (F5)
```

| symbol | is | unit | true value |
|---|---|---|---|
| `K` | coupling strength, one scalar for every pixel | rad per unit time per neighbour | 0.90 |
| `omega_i` | natural frequency, **one per pixel — the heterogeneity** | rad per unit time | region mean 0.6/0.95/1.3/1.65 × 0.035, plus per-pixel ±0.012 |
| `m_i` | region mask (discs in 2-D, tubes in 3-D) | — | known, not fitted |
| `N(i)` | lattice nearest neighbours | — | 4 in 2-D, 6 in 3-D |

**(F3) is already a message-passing layer**, and that is why this toy is worth fitting rather than a
trick for making it fittable: `K` is the edge weight, `omega_i` the additive node embedding, and
`(w_i, -v_i)` the receiver-side gauge applied to the aggregate. A GNN that recovers `K` as an edge
weight and `omega_i` as an embedding has recovered the Kuramoto rule exactly.

**What this costs the generator:** a run that stores only `sin phi` has thrown `w` away and nothing
can be fitted from it. `kuramoto_field` therefore takes `emit: quadrature`, which writes `cos phi`
into a second channel. The phase itself is still never written out.

**G28 — known-ODE recovers the coarse speed c from the coarse field.** THE EQUATION FITTED IS  du/dt = -c du/dx  (C1), one unknown scalar: c, the phase speed in DOMAIN WIDTHS PER FRAME, true value 0.000833333 -- one full traverse of the domain in 1,200 frames. The model is the equation itself with c as an nn.Parameter, no network, exactly as connectome-gnn's known_ode.py replaces every constant of the true ODE with a parameter and learns nothing else. (C1) is LINEAR in c, so the batch least-squares answer c* = -<du/dt, du/dx> / <du/dx, du/dx> is available in closed form, and the gate is really asking whether the trainer lands on a number it could have computed. That is the point: it is the cheapest check that the training loop is wired correctly, and it cannot be passed by a lucky architecture.

**G29 — known-ODE recovers K and omega_i from the fine field.** THE EQUATIONS FITTED ARE the Kuramoto rule written in the observables: r_i = omega_i + K SUM_j (v_j w_i - w_j v_i)  (F3), dv_i/dt = w_i r_i m_i (F4), dw_i/dt = -v_i r_i m_i  (F5), where v = sin(phi), w = cos(phi) and m is the known region mask. TWO UNKNOWNS OF DIFFERENT KIND: K, one coupling shared by every pixel, true value 0.90; and omega_i, ONE NATURAL FREQUENCY PER PIXEL -- this is the heterogeneity, the thing a_i exists to carry, drawn as a per-region mean (0.6/0.95/1.3/1.65 x 0.035 rad per unit time) plus a per-pixel offset of half-width 0.012. K is scored by relative error because it is one number; omega_i by R^2 because it is a field of a million, and a map that is right in pattern and off by a constant has still found the heterogeneity. (F3) IS ALREADY A MESSAGE-PASSING LAYER -- K is the edge weight, omega_i the additive node embedding, and (w_i, -v_i) the receiver gauge -- so a GNN that passes this has recovered a graph rule, not fitted a curve.

**G30 — known-ODE recovers BOTH rules from the SUM alone.** The only one of the three that asks the prototype's real question. The model sees s = u + v, one field, and must fit (C1) and (F3)-(F5) TOGETHER without being told which part of the signal belongs to which rule. Thresholds are deliberately looser than G28/G29 -- 5%, 10%, 0.80 against 1%, 5%, 0.90 -- because separation is a strictly harder problem than recovery and a gate that demanded the same numbers would be measuring the difficulty of the decomposition as if it were a defect. What makes the separation possible at all is that the two rules DO NOT COUPLE and live at different resolutions and rates: the coarse traverse is 1,200 frames and the fine period is about 30, a 40x separation, verified in the generator's summary.json rather than assumed. If G30 fails while G28 and G29 pass, the finding is that the sum is not identifiable at this rate ratio, and the ratio is a config knob.

### Tier 3 — measurement (does it agree with something observed?)
| id | gate | threshold |
|---|---|---|
| G17 | ZAPBench held-out d(ΔF/F)/dt | R² > 0.268, the parameter-free kNN pool |
| G18 | learned `b_i` is spatially structured | Moran's I > 0.2 vs a permutation null |
| G19 | fitted calcium decay | 0.5–2 s (GCaMP6) |
| G20 | redox washout response | threshold fixed from `Development_Time_Trend.xlsx` **before** the run |

**Estimator rule, learned twice:** an estimator must be sharper than the threshold it is judged
against. `argmax` on a 128-cell grid and an integer FFT bin both failed this.

---

## 8. Stages

| stage | build | gates |
|---|---|---|
| 0 | container: `spec_schema`, `engine`, `gates`, `viz`, tests | G1, G2, G7 |
| 1 | three toys generated via `plexus.engine.run` | G16 |
| 2 | **data gates — no model** | G21–G26 |
| 3 | `simple`, 1 pass, `free`; train/test/plot | G1b, **G5**, G9, G10, G11, G13 |
| 4 | `graphcast` message, `n_passes` 1→16 | G6, G8, G15 |
| 5 | encoder/decoder via `mpm_scatter`/`mpm_gather` | G3, G4, G14 |
| 6 | compare the three coarse rules | G27 |
| 7 | ZAPBench | G17, G18, G19 |
| 8 | redox, field-only | G20 |

Stage 2 gates the data before stage 3 touches training. Stage 3's G5 is the pivot: it establishes
that `simple` is arithmetically the existing `NeuralGNN`, so everything after is a controlled
variation on something known to work.

---

## 9. Verification

- `python engine.py -o gates <config>` runs the table, writes `gates.csv` and `gates_table.tex`.
- `pytest prototype/graphcast/tests/` — one test per tier-1 gate.
- Each stage: regenerate the PDF, **read it**, then commit `gates.csv` + PDF + figures.
- Cross-checks against known numbers: G11's 0.70 is the flyvis Ward-tree figure; G17's 0.268 the
  measured kNN baseline; G14's 0.03 twice the measured run-to-run floor.

## 10. Files

`prototype/graphcast/`: `engine.py`, `spec_schema.py`, `gates.py`, `viz.py`, `ops_graphcast.py`,
`toy.py`, `model.py`, `train.py`, `config/toy_{counter,withhold,envelope}.yaml`,
`config/{zapbench,redox}.yaml`, `tests/`, `note_graphcast_plexus.tex` → `.pdf`.

Nothing is written into `src/plexus/`; nothing is promoted.

---

## 11. The trainer as an operator schedule

*Added after the two-scale toys were generated. This is the design for the fitting half, and it is
the same design as the forward half rather than a second one beside it.*

### The claim

`plexus.engine.run` takes a `schedule:` — a list of named operators, each with its params — and
applies them in order to a hierarchy. That is not a simulation-specific idea; it is a way to write
a composition down so that every term is named, parameterised from the file, and separately
inspectable. **A training step has exactly the same shape.** So the `training:` section stops being
a bag of scalars and becomes a second schedule, run by the same kind of loop:

```
forward   schedule:  advect_field -> kuramoto_field                (what the world does)
fitting   schedule:  knn_graph -> predict -> loss -> regularize -> step   (what we do about it)
```

The word for the second list is **trainer**. A trainer is a list of operators with params, exactly
as a simulation is.

### Why this is worth the trouble, in one line each

- **A loss stops being a hard-coded line in a train loop.** `coeff_g_phi_diff: 750` in the
  production config is the largest term in the objective and it is a number with no operator
  attached. As `- op: smoothness_loss` with `coeff: 750` it has a name, a signature, and a place in
  a schedule that can be printed.
- **The residual becomes attributable to a mechanism.** This is the whole reason the prototype is
  in Plexus (`plexus2.tex`, mechanistic inverse modelling). It only holds if each learnable thing
  is its own operator.
- **`known_ode` and `gnn` become two implementations of one role**, not two code paths. Both are
  `role: predict`; they differ in what they hold and nothing else sees the difference.
- **Ablations become edits to a list.** Dropping the regulariser is deleting a line, not a flag.

### The five roles

The simulation `KINDS` (`lateral`/`aggregate`/`broadcast`/`exchange`/`field`/`rewire`/`structural`/
`seed`) describe how an operator moves data *inside a hierarchy during a step*. A trainer operator
does something else — it consumes a trajectory and produces a scalar or a parameter update — so it
gets its own small vocabulary rather than an overload of that one. Overloading would have made
`kind` mean two things at once, and `kind` is the thing the registry dispatches on.

| role | signature | implementations |
|---|---|---|
| `graph` | positions → edge set | `knn_graph`, `radius_graph`, `multimesh` |
| `predict` | state, graph, t → state at t+1 (or its increment) | **`known_ode`**, **`gnn`** |
| `loss` | prediction, target → scalar | `mse`, `increment_mse`, `rollout_mse` |
| `regularize` | parameters → scalar | `l1`, `l2`, `group_lasso`, `smoothness` |
| `step` | scalars, parameters → updated parameters | `adamw`, with param groups and a schedule |

`graph` is the exception and deliberately so: **building a graph is already a Plexus kind**. A kNN
graph over positions changes the edge set, which is `kind: rewire`, so `knn_graph` is a plain
registered operator that can sit in a *forward* schedule too. It is listed here because the trainer
needs it, not because it is new vocabulary.

### What a config looks like

Two specs, same shape, differing only in the `predict` line. Nothing about the dataset appears in
either (G2), and every number is in the file (R5).

```yaml
trainer:
  schedule: [knn_graph, predict, loss, regularize, step]
  operators:
    - op: knn_graph                   # role: graph   (kind: rewire, a normal Plexus operator)
      at: probe
      k: 16

    - op: known_ode                   # role: predict -- FIT A KNOWN EQUATION
      equation: transport             #   du/dt = -c du/dx
      learn: [c]                      #   the ONLY unknown
      init: {c: 0.0}

    - op: mse_loss                    # role: loss
      on: increment
      norm: increment_variance        # GraphCast suppl. 4.2: 1/Var[x_{t+1} - x_t]

    - op: adamw                       # role: step
      groups:
        - {params: [c], lr: 1.0e-3}
      betas: [0.9, 0.95]              # GraphCast suppl. 4.4
      grad_clip: 32.0
      scheduler: cosine_to_zero
      n_iter: 2000
```

and the general model is one line different:

```yaml
    - op: gnn                         # role: predict -- A GENERAL LEARNABLE MODEL
      message: graphcast
      n_passes: 4
      embedding: multires
      encoder_decoder: on
```

with the extra groups the workspace's schemes require, which are now visibly *groups* rather than
three scalars that happen to be named `lr`, `lr_W`, `lr_embedding`:

```yaml
    - op: adamw
      groups:
        - {params: [W],         lr: 0.0009,   weight_decay: 0.0}   # scientific output: NO decay
        - {params: [a],         lr: 0.002325, weight_decay: 0.0}   # scientific output: NO decay
        - {params: [g_phi, f_theta], lr: 0.0018, weight_decay: 0.1}
    - op: l1_regularizer
      params: [W]
      coeff: 1.5e-4
```

### The ladder this buys

`known_ode` first is not a warm-up, it is the control. It has one scalar and a closed-form answer,
so if the trainer does not find it, nothing measured on a network afterwards means anything.

| rung | `predict` | unknowns | answer known in closed form? |
|---|---|---|---|
| 0 | `known_ode`, `equation: transport` | 1 — the speed `c` | **yes**, least squares |
| 1 | `known_ode`, `equation: kuramoto` | 2 — `K`, `omega` | no, but the truth is in the spec |
| 2 | `gnn`, `simple`, 1 pass | ψ, φ, W, a | no — but G5 says it equals `NeuralGNN` |
| 3 | `gnn`, `graphcast`, 4–16 passes | the same, deeper | no |

### G28 — the new gate, defined before the code

**What it asks.** Whether the trainer, given the equation and one unknown, lands on the number it
could have computed.

The fitted equation is transport, `du/dt = -c du/dx` (C1), one unknown scalar: `c`, the phase speed
in **domain widths per frame**, true value `0.000833333` — one full traverse of the domain in 1,200
frames. The `predict` operator is the equation itself with `c` as an `nn.Parameter`, no network,
exactly as `connectome-gnn/src/connectome_gnn/models/known_ode.py` replaces every constant of the
true ODE with a parameter and learns nothing else.

(C1) is **linear in `c`**, so the batch least-squares answer

```
c* = - <du/dt, du/dx> / <du/dx, du/dx>
```

is available without any optimiser at all. The gate is therefore not asking whether the model can
represent the physics — it is asking whether the **training loop is wired correctly**: whether the
loss is on the right quantity, the gradient reaches the parameter, the schedule converges, and the
sign conventions agree. It cannot be passed by a lucky architecture, because there is no
architecture.

**Measured on the generated coarse field before writing the trainer**, so the gate is known to be
answerable and its threshold is set against a real number rather than a hope:

| quantity | value |
|---|---|
| closed form `c*` | `0.00083054` domain widths / frame |
| true `c` | `0.00083333` |
| relative error of the closed form | **0.335 %** |
| R² of (C1) at `c*` | **0.9632** |
| cells advanced per recorded step | 1.280 |

The 3.7 % of `du/dt` that (C1) does not explain is the transport operator's **integer-cell roll**:
the field shifts by a whole number of cells, so a step of 1.280 cells is delivered as alternating
1s and 2s. That is a known, documented property of the operator, not a defect, and it sets the
floor the trainer is judged against.

| id | gate | threshold | unit | status |
|---|---|---|---|---|
| **G28a** | the data supports the true speed | closed form within **1 %** of the true `c` | fraction of `c` | **new** |
| **G28** | the trainer recovers the speed | learned `c` within **1 %** of the true `c` | fraction of `c` | pre-registered, unchanged |

**G28's threshold is not moved.** It was fixed at 1 % before any of this was measured, and the
measurement came out at 0.335 % — so the gate is answerable and the number stands. Retuning a
pre-registered threshold *after* seeing the data is the one thing the gate discipline exists to
prevent, and the temptation to relax it to "within 2 % of `c*`" is exactly the shape of that
mistake.

What is added is **G28a**, a new precondition, because G28 alone confounds two failures that want
different fixes: **G28a failing means the toy or the estimator is wrong; G28 failing with G28a
passing means the training loop is wrong.** Note the margin is tight — 1 % is only 3× the
closed-form error — so a trainer that merely gets close is not enough; it has to land on the least
squares answer.

Artifacts (R11): a PNG of the loss curve and of `c` against iteration with `c*` and `c_true` drawn
as horizontal lines, and an MP4 of the fitted field beside the observed one.

---

## 12. The two-resolution partition — the phenomenon, not a defect

*Added after the coarse grid was dropped to 64² and the consequence was measured.*

### What was found

The coarse rule advances by **whole cells** (`AdvectField`: a fractional accumulator per axis, an
integer `torch.roll` when it crosses one — a permutation, so amplitude is preserved to the bit). Its
motion per recorded frame is therefore

```
cells per record  =  |v| × resolution × record_stride
```

At fixed velocity and stride, coarsening the mesh drives that below one, and when it goes below one
**most consecutive records are bit-identical and `du/dt` is exactly zero on them**:

| coarse grid | cells per record | consecutive records with no change |
|---|---|---|
| 256² | 1.28 | ~0 |
| **64²** | **0.29** | **~0.7** |

### Why this is the point of the prototype and not a bug

The first instinct is to fix it — lengthen the stride, raise the speed, refine the mesh. That
instinct is wrong, and naming why is the reason this section exists.

**There is no timestep at which a 64² transport is a smooth motion.** The field is genuinely static
for twenty frames and then jumps a whole cell. A single-resolution fit sees a field that mostly does
not move and occasionally teleports, and no amount of tuning the observation cadence changes that,
because the discreteness is in the *state*, not in the sampling. Any model forced to integrate this
level on the observation's clock is being asked to represent a step function as a rate.

**A multi-resolution model is exactly the thing that does not have to.** Each level carries its own
clock: the coarse level is integrated at the cadence at which it actually moves, and the fine level
at its own, much faster one. That is the same statement as the NGP/one-sided note's — a level
represents what its band can represent and nothing else — expressed in time rather than in space.

So the under-resolved coarse grid is the **target phenomenon**. It is what a real dataset looks
like when a slow process is observed at a fast rate, and it is what the whole multi-level structure
is being built to handle.

### The partition

Two coarse datasets per dimension, identical in every respect except the mesh — same velocity, same
wavevector, same profile, same recording stride — so the difference between a fit on one and a fit
on the other is attributable to the resolution and to nothing else:

| dimension | resolved | under-resolved | fine |
|---|---|---|---|
| 2-D | `toy2d_coarse256` (256², 1.28 cells/record) | `toy2d_coarse64` (64², 0.29) | `toy2d_fine` (1024²) |
| 3-D | `toy3d_coarse64` (64³) | `toy3d_coarse32` (32³) | `toy3d_fine` (256³) |

`still_pair_fraction` is written into every run's `summary.json` beside the autocorrelation, so
which side of the partition a dataset is on is a recorded property of the data rather than
something a reader has to infer.

**Why 3-D stops at 32³ and not 16³.** The coarse profile's highest harmonic (k = 3, wavevector
component 2) has 6 cycles across the domain. At 32³ that is 5.3 cells per cycle — chunky, which is
wanted. At 16³ it is 2.7, which is below the ~4 needed for a wave to read as a wave and close to
outright aliasing: the grid would no longer represent the rule it claims to, and the dataset would
be testing the wrong thing. **The partition is about the observation cadence, not about destroying
the field.**

### What this changes downstream

- **G28/G28a run on BOTH coarse datasets.** On 256² the closed form is well conditioned (measured
  R² 0.9632 for (C1)). On 64² the least-squares estimate stays **unbiased** — the accumulator
  guarantees the mean velocity exactly — but the per-pair residual is dominated by the ~70% of
  pairs with zero increment, so the R² will be far lower. **That gap is a result to report, not a
  threshold to relax:** it is the quantitative statement of what the coarse level costs a
  single-resolution fit.
- The `predict` role acquires a per-level cadence. A level whose `still_pair_fraction` is high must
  be integrated on its own clock; this is the first concrete requirement the multi-level model has
  that a single-level one does not.

---

## 13. The trainer engine, and what G28 found

*Written after the first fit ran end to end. Files: `model_hierarchy.py`, `trainer.py`,
`ops_trainer.py`, `config/fit_transport256.yaml`.*

### The model is a hierarchy, not a function

The decision that mattered: `predict` does not wrap a `nn.Module`. The `model:` section of a fit
spec **is a Plexus spec fragment** — fields, operators, schedule — loaded by `plexus.schema.load`
and built by `plexus.engine.build`/`seed`, and one prediction is *running its schedule*:

```
load the observed frame into the hierarchy  ->  step its schedule K times  ->  read it back
```

A `forward()` would have kept the arithmetic and discarded all four of the paper's compositions.
Keeping the hierarchy keeps two things a function cannot have: the learnable parameters live **on
the operators**, so a residual is attributable to a named mechanism; and the schedule carries
`every:`, so a level whose motion is slower than the observation cadence is integrated on **its own
clock** — the multi-rate composition, which is the whole reason the 64² dataset exists.

`known_ode` and `gnn` are therefore not two code paths. They are two lists of operators in a file.

### What the trainer engine does and does not know

`trainer.py` contains no loss expression, no coefficient, no learning rate, no optimiser choice, no
clip value, no dataset name. It reads a batch out of a recorded zarr and runs two lists in order.
Everything else is an operator with a role: `predict` / `loss` / `regularize` / `step`, plus
`graph`, which is deliberately **not** a new kind — a kNN graph changes the edge set, which is
already Plexus's `rewire`.

### Three bugs the first run found, all in the same family

Each was a quantity in the wrong unit, and each looked like a converged fit to the wrong answer.

1. **The tape survived the iteration.** A hierarchy is persistent, so `fld.grid` still carried the
   previous iteration's autograd history; writing into it in place made iteration two walk a freed
   graph. The fix is a modelling statement, not a `retain_graph=True`: **the observation is data**,
   so it enters as a leaf and only the operators' parameters carry gradient.
2. **The stride was counted twice.** The model steps once at its own `dt`, so its increment is
   already per sim-frame; only the *target* spans `stride` of them. Dividing both scaled the
   recovered velocity by the stride exactly.
3. **The velocity vector was not identifiable at all.** See below — the largest of the three.

### G28's real finding: a single plane wave cannot determine a velocity

For `u = f(m·x)`, `∇u = m f'(m·x)`, so the partial derivatives are **exactly proportional**. The
least-squares system for `v` in (C1) is then singular in every direction but one. Measured on the
original profile (harmonics 1 and 3 of one wavevector):

| | value |
|---|---|
| condition number of the normal matrix | **5.0 × 10⁶** |
| recovered `v` | `[0.00320, −0.00455]` — **568%** wrong as a vector |
| recovered component **along** the wavevector | **0.542%** — right |
| recovered component **perpendicular** | `−0.0055` against a true `0` — pure invention |

The data determined the phase speed and said nothing about the perpendicular drift, so the fit put
whatever it liked there. **This is precisely the failure the known-ODE stage exists to catch:** the
parameter was underdetermined *by the data*, not by the model, and no network would have done
better. Adding one non-parallel wavevector `[1,−3]` to the profile drops the condition number to
**5.34** and makes the whole vector identifiable. The transport rule is untouched — it carries any
profile.

### G28a's finding: the estimator's own truncation error

With the vector identifiable, the closed form still sat 1.06% from the truth — outside G28a's 1%.
The cause is that (C1) approximates a finite shift by its first derivative, so the error grows with
the displacement per recorded pair:

| displacement per pair | speed error |
|---|---|
| 1.28 cells | 1.062% |
| 2.56 cells | 2.688% |
| 3.84 cells | 5.309% |
| 7.68 cells | 18.126% |

The threshold was **not** moved. The *sampling* was: recording the coarse datasets at stride 3
rather than 6 puts the displacement at 0.64 cells and the closed form at **0.443%** — inside the
gate. Both coarse specs moved together, so the mesh remains the only difference between them.

### The diagnostic pair worked exactly as designed

At that point G28a passed (0.443%) while G28 failed (1.482%), and *trainer vs closed form* was
1.933%. By the gates' own logic that says the fault is the **training loop**, and it was: with 43%
of pairs stationary at the finer stride, a batch of 8 sits on the gradient-noise floor. Raising the
batch to 64:

| | vx | vy | speed | vs true |
|---|---|---|---|---|
| closed form `v*` | 0.000742704 | 0.000369722 | 0.000829640 | 0.443% |
| **trainer** | 0.000742843 | 0.000369511 | **0.000829671** | **0.439%** |
| true `v` | 0.000745356 | 0.000372678 | 0.000833333 | — |

**G28b PASS (cond 5.34 < 100). G28a PASS (0.443% < 1%). G28 PASS (0.439% < 1%).**

And the number that mattered was never any of those: ***trainer vs closed form = 0.004%***. The
training loop lands on the least-squares answer to four decimal places, which is the only thing a
one-scalar fit can tell us and exactly the reason it is worth running before anything harder. The
0.44% that both share is the estimator's own truncation error against the truth — a property of the
data and the model class, not of the optimiser, and the two gates separate those cleanly.

---

## 14. Why K was not identifiable, and what fixed it

*The longest-running open number in this prototype. `fit_kuramoto2d` recovered K = 0.42 against a
true 0.90 while its own loss fell monotonically. Three explanations were proposed and the first two
were wrong; recording that is the point of this section.*

### The rule, and what K is

```
r_i      = ω_i + K Σ_{j∈N(i)} (v_j w_i − w_j v_i)          (F3)
dv_i/dt  =  w_i r_i m_i ,   dw_i/dt = −v_i r_i m_i          (F4,F5)
```

`K` is one coupling shared by every pixel; `ω_i` is one natural frequency per pixel — 1,048,576 of
them. `(v,w) = (sin φ, cos φ)`.

### Three explanations, in order

**1. Capacity — WRONG.** "ω free per pixel can absorb K·coupling at any instant." Predicted that
constraining ω would free K. Tested by replacing ω with a hash encoding (~90k correlated
parameters): ω's *pattern* improved a great deal — Pearson 0.269 → **0.732** — and K got **worse**,
0.42 → 0.017. Right about ω, wrong about K.

**2. Optimisation — WRONG.** "The joint fit hasn't converged." Tested by freezing ω **at the truth**
and fitting K alone, one parameter: K → 0.014. Nothing to do with ω.

**3. Identifiability, and it depends on WHICH FRAMES.** With ω at the truth, the loss minimum in K:

| starts used | minimum at |
|---|---|
| early 0–9 | **K = 0.9** ← the truth |
| spread 0–216 (step 24) | K = 0.3 |
| late 130–238 | K = 0.1 |

**A coupling is identifiable only while it is still doing something.** Early on K is actively
synchronising oscillators from random phases. Once domains lock, neighbours are in phase, the
coupling term is ≈ 0 inside a domain, and any K > 0 only adds error at the walls. A loss over
equilibrated frames asks the data about a parameter that has stopped mattering.

### Neither learning rate nor batch size can substitute

K fitted alone, ω frozen at the truth. True K = 0.90:

| | all frames (0–287) | | | transient (0–19) | | |
|---|---|---|---|---|---|---|
| lr \ batch | 1 | 4 | 8 | 1 | 4 | 8 |
| 0.005 | 0.032 | 0.026 | 0.022 | 0.272 | 0.429 | 0.525 |
| 0.020 | 0.035 | 0.021 | 0.033 | 0.601 | 0.610 | **0.689** |
| 0.050 | 0.027 | 0.018 | 0.031 | **0.656** | 0.519 | 0.675 |

On all frames every cell lands between 0.018 and 0.035 — a 2× spread around a value 26× too small.
On the transient the same two knobs take K from 0.27 to 0.69. **A knob only helps once the signal
exists**, and no amount of either creates one.

### Horizon, on the transient, is what finishes it

The minimum moves onto the truth as the rollout lengthens, and sharpens
(`loss(0.1)/loss(0.9)` rises from 1.6× to 2.2×):

| horizon | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| minimum at K | 0.7 | 0.7 | 0.7 | **0.9** | **0.9** |
| K fitted (ω frozen) | — | 0.723 | 0.752 | **0.888** | 0.879 |

**Horizon 8 recovers K to 1.4%**, from 3% of true on all frames. 16 buys nothing for twice the tape.

### What is still open

`fit_kuramoto_transient` learns **both** K and ω, and gets K = 0.687 — better than 0.42, short of
0.888. The reason is a genuine tension rather than a tuning failure: **K needs the transient and ω
needs the equilibrated frames**, and a hard cut gives one what it needs by taking it from the other.
Its 100-step rollout scores R² −0.543 with 9 usable steps, worse than the all-frames fit's 20,
because ω is now fitted on 20 frames instead of 290.

So the next move is a frame **weighting** rather than a split — the two parameters want different
parts of the trajectory, and the objective should say so. This spec is the control that would be
measured against.

---

## 15. Stage 4 — the GraphCast operator

### Where a variant lands in Plexus, and two of R1's "options" are not options

Plexus has exactly three places a variant can live, and choosing wrongly is how a switch ends up in
code instead of in a file:

| mechanism | means | selected by | precedent |
|---|---|---|---|
| `model=` on `@register_operator` | a **different rule** under one operator name | `model:` | `wave_field` — travelling / counter / envelope |
| `implementation=` | the **same rule, different numerics** | `impl:` | MPM's `warp` vs `default` |
| ordinary params | a **knob** of one rule | the operator line | `n_passes`, `hidden_dim` |

Mapping R1's four switches onto that changes two of them:

- **`message: simple | graphcast`** → **`model=`**. A different rule, not a knob: GraphCast carries
  a residually-updated **edge latent** across layers and `simple` has no edge state at all.
- **`n_passes: 1…16`** → a **param**. Same rule, deeper.
- **`embedding: none | free`** → a **param**; but **`ngp` is a schedule composition**,
  `[hash_encoding, gnn_message]`, with the encoder writing a field the rule reads. Better than a
  param value because it keeps the residual attributable to a named mechanism.
- **`encoder_decoder: off | on`** → also a **schedule composition**,
  `[gc_scatter, gnn_message, gc_gather]` versus not.

So the four families land as: `known_ode` and `gnn_message` are separate **operators**; `simple` vs
`graphcast` is a **`model=` variant** of the second; the hashtable and the encoder/decoder are
**schedule entries**. Fewer switches, more composition — the direction the whole restructure went.

**Consequence for G1, and it is a threshold change that must be recorded rather than absorbed.** The
pre-registered count of 24 assumed four switches multiplying out. If `encoder_decoder` and the
hashtable become schedule compositions, the enumerated matrix is `2 message-models × 4 n_passes ×
2 embedding params = 16`, and the compositions are separate specs. **G1's 24 has not been changed
here** — it will be, in the commit that makes the compositions real, and not before.

### What the operator must reproduce

From `papers/weathernext/weathernext/utils/`:

- `InteractionNetwork` with `update_edge_fn` and `update_node_fn`, each an MLP **followed by**
  LayerNorm with the residual added after (`dense.py:131` — post-norm, not pre-norm)
- `use_edge_residuals=True`: the edge latent **persists and accumulates** across layers
- `include_sent_messages_in_node_update=False`: the node update sees only **received** messages
- layers built per index (`processor_edges_{index}_`), so parameters are **unshared** — a stack of
  16 distinct layers, not one layer applied 16 times

### Gates, fixed before the code (R9)

Four already pre-registered and unchanged: **G5** (`simple` ≡ `NeuralGNN` at copied weights, < 1e-5),
**G6** (1 vs 16 passes at init bit-identical), **G8** (K=20 rollout < 2× the GT norm), **G15**
(graphcast vs simple against a 3-seed floor).

Four are new, and each names one published feature so that a degenerate implementation cannot pass
by tying with `simple`:

| id | what | threshold |
|---|---|---|
| **G31** | the edge latent **persists** — zeroing it between layers changes the output | relative change > 0.10 |
| **G32** | layers are **unshared** — parameters linear in depth | `params(16)/params(1)` ∈ [15.2, 16.8] |
| **G33** | the node update ignores **sent** messages | permutation leaves output bit-identical |
| **G34** | graphcast **contains** simple — edge latent off, depth 1, they agree | < 1e-5 |

**G31 and G34 are the pair that matters.** G5 says `simple` is the model we already trust; G34 says
`graphcast` *contains* it; G31 says the extra state is *actually running*. Without G31, an
implementation that quietly degenerates would **pass G15 by tying**, and we would conclude "the
extra machinery does not pay" about machinery that was never exercised.

### Order of work

1. G31–G34 into `gates.py` — **done**, before any operator code.
2. `ops_graphcast.py`: `gnn_message` gains `model="graphcast"` — the edge-latent interaction
   network. Reuses `Lateral`, `radius_graph`, `lin_edge`/`lin_phi` naming.
3. G32, G33, G34 first: they need no training and no data, only a built operator.
4. G31 next: needs one forward pass on real data.
5. Only then a fit, and only then G15.

**A dependency to settle before step 3.** G5 compares against `connectome-gnn`'s `NeuralGNN` at
copied weights. That class is in another repo with its own config object, so G5 is either an
import across repos or a re-derivation here — and a re-derivation is not the gate. This is the one
stage-4 gate whose cost is not yet known.
