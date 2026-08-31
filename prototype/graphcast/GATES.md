# Gate report

8 pass, 2 fail, 18 not yet run (of 28: 9 bookkeeping, 15 closed form, 4 measurement)

`status` is not `outcome`. **Outcome** is what the number did against the threshold;
**status** is whether the gate has been walked through — definition written, estimator
sanity-checked, negative control run, result read. A gate can pass mechanically and
still be pending review.

## Tier 1 — bookkeeping: does the code do what the operator says?

| id | gate | threshold | measured | outcome | status | figures |
|---|---|---|---|---|---|---|
| G1 | every option combination can be READ | 24 of 24 option combinations load | 24 | **PASS** | done | [G1_options.png](log/toy_counter_noed_simple_p1_free/G1_options.png) |
| G1b | every option combination can be RUN | 24 of 24 option combinations run one forward step | — | · | pending | — |
| G2 | no dataset identity anywhere in the code | 0 offending literals outside config/ | 0 | **PASS** | pending | [G2_scan.png](log/toy_counter_noed_simple_p1_free/G2_scan.png) |
| G3 | the transfer pair returns what it was given | < 1e-6 of the field value | — | · | pending | — |
| G4 | the transfer conserves what it moves | \|sum(w) - 1\| < 1e-6 | — | · | pending | — |
| G5 | the simple option IS the existing model, arithmetically | < 1e-5 of the voltage range | — | · | pending | — |
| G6 | depth is an option, not a different model | bit-identical (max \|delta\| == 0) | — | · | pending | — |
| G7 | the spec is allowed to carry a unit | 1 = declared and checked | 1 | **PASS** | pending | [G7_units.png](log/toy_counter_noed_simple_p1_free/G7_units.png) |
| G8 | one-step accuracy is not stability | state norm stays < 2x the ground-truth norm | — | · | pending | — |

### What each bookkeeping gate is for

**G1 — every option combination can be READ.** The model has four switches, and the whole premise is that any setting of them is a legal model -- options, not forks. This gate checks that literally: every combination must be readable. It takes the reference config, overwrites only the model block with each of the 24 settings, and calls the loader. Why 24: `simple` carries no edge state, so running it twice is a different model wearing the same name and the schema refuses it, which leaves four legal (message, n_passes) pairs times 2 encoder/decoder times 3 embeddings. It catches an option that exists only in the documentation, and a combination that silently falls back to a default instead of erroring.

**G1b — every option combination can be RUN.** The other half of G1. Reading a config proves the vocabulary is right; it does not prove the model can be built from it. This gate constructs the model object for each of the 24 settings and pushes one forward pass through it. It is split from G1 because the two become available at different stages -- G1 needs only the schema, G1b needs a model -- and a row that goes green having checked half its claim reads as an endorsement it has not earned. It catches shape mismatches that appear only when two particular options meet.

**G2 — no dataset identity anywhere in the code.** One model has to serve three datasets, and it only does so if none of them has left a trace in the code. This scans the prototype's own Python on the ABSTRACT SYNTAX TREE rather than as text, checking every string and numeric constant except docstrings. That distinction is the point: naming a dataset in prose is documentation, while the same name used as a value is a hardcoded path. Scanning the text would flag the first; skipping strings entirely would miss the second, because a path IS a string.

**G3 — the transfer pair returns what it was given.** The encoder/decoder option moves state onto a background grid and back. If it is sound, depositing a constant and gathering it again returns the constant. This is the end-to-end version of G4 and it catches what G4 cannot: a transfer pair that is not each other's adjoint, an off-by-one in the stencil, a normalisation applied on one side only. The threshold is a FRACTION of the field value, so it does not depend on what the field is.

**G4 — the transfer conserves what it moves.** The local half of G3. Each transfer spreads a node's value over the corners of the grid cell it sits in, and those weights must sum to one, or the transfer quietly changes the total amount of stuff every time it is applied. Summing to one is also exactly the condition that makes interpolation reproduce a constant, which is why G3 tests the same property from the outside. Dimensionless by construction.

**G5 — the simple option IS the existing model, arithmetically.** The pivot of the whole prototype. With `simple`, one pass and no encoder/decoder, this model is meant to be connectome-gnn's NeuralGNN term for term. The gate copies NeuralGNN's weights across and requires the two to produce the same numbers. If it passes, everything downstream is a controlled variation on a model already known to reach R^2_W around 0.97. If it fails, no later result can be interpreted at all, because a new model's failure and a reimplementation bug are indistinguishable.

**G6 — depth is an option, not a different model.** Both message-passing MLPs have their final layer initialised to zero, so every residual block is EXACTLY the identity before training. It follows that one pass and sixteen passes must give the same numbers at step zero. The threshold is exactly zero with no tolerance, because this is an algebraic identity rather than a numerical one. It catches a residual that is not a residual -- a missing skip connection, or an initialisation that makes the stack a different model at every depth.

**G7 — the spec is allowed to carry a unit.** Two halves. The spec must declare a units block, because plexus/units.py is explicit that a model without one is dimensionless and no result from it may be quoted with a unit -- and every measurement-tier gate is a comparison against a quantity. And no measurement threshold may be denominated in grid cells, voxels or steps. That second half is the lesson from the ecm study: a penetration of 0.82 grid cells sounded small and was 15 microns, nearly two cell diameters. A threshold in the mesh's own currency is the easiest one to pass.

**G8 — one-step accuracy is not stability.** A model can predict the next increment almost perfectly and still blow up when it is fed its own output twenty times over, because a one-step fit never sees its own error compound. This runs a 20-step rollout and requires the state to stay bounded. The threshold is a RATIO to the ground-truth norm, so it is dimensionless and means the same thing on the toy and on real data.

## Tier 2 — closed form: does it reproduce the physics it was given?

| id | gate | threshold | measured | outcome | status | figures |
|---|---|---|---|---|---|---|
| G9 | the message becomes a gradient operator | R^2 > 0.90 against the true field gradient | — | · | pending | — |
| G10 | recover the per-node time constant | R^2 > 0.95 against the known tau | — | · | pending | — |
| G11 | the embedding recovers the types (small toy) | ARI > 0.70 against the 6 true types | — | · | pending | — |
| G12 | the embedding recovers the types (large toy) | ARI > 0.70 against the 65 true types | — | · | pending | — |
| G13 | recover the per-node SIGNED GAIN (the heterogeneity) | R^2 > 0.90 against the true g_i | — | · | pending | — |
| G14 | encoder/decoder is a genuine option: on vs off | \|delta R^2(gradient)\| < 0.03 | — | · | pending | — |
| G15 | graphcast vs simple message is RESOLVED, either way | \|delta\| reported against the 3-seed floor; below it is UNRESOLVED, not ranked | — | · | pending | — |
| G16 | types are spatially mixed by construction | spatial-cell type purity within 20% of chance (1/n_types) | 1.131 | **PASS** | pending | [G16_toy.png](log/toy_counter_noed_simple_p1_free/G16_toy.png), [G16_state.mp4](log/toy_counter_noed_simple_p1_free/G16_state.mp4) |
| G21 | the coarse field is a travelling wave, cyclic left to right | phase drift per frame within 5% of lambda/period | 0.01241 | **PASS** | pending | [G21_field.mp4](log/toy_counter_noed_simple_p1_free/G21_field.mp4) |
| G22 | the fine rule is exactly recoverable from (v, grad u) | minimum per-node R^2 > 0.90 | 0.638 | **FAIL** | pending | [G22_identifiability.png](log/toy_counter_noed_simple_p1_free/G22_identifiability.png) |
| G23 | the gradient is reconstructible from neighbours' states | R^2 > 0.95, else the graph cannot carry the fine rule | 1 | **PASS** | pending | [G22_identifiability.png](log/toy_counter_noed_simple_p1_free/G22_identifiability.png) |
| G24 | the heterogeneity is linearly readable | corr(fitted gain, true g_i) > 0.90 | 0.9959 | **PASS** | pending | [G24_heterogeneity.png](log/toy_counter_noed_simple_p1_free/G24_heterogeneity.png) |
| G25 | connected nodes are not collinear | mean \|corr\| between connected nodes < 0.80 | 0.576 | **PASS** | pending | [G22_identifiability.png](log/toy_counter_noed_simple_p1_free/G22_identifiability.png) |
| G26 | the graph is NECESSARY: a node-local baseline cannot fit | node-local R^2 < 0.50 while (v, grad u) exceeds 0.90 | 1 | **FAIL** | pending | [G26_necessity.png](log/toy_counter_noed_simple_p1_free/G26_necessity.png) |
| G27 | which coarse rule forces the graph | the three toys ranked by G26; reported, not tuned | — | · | pending | — |

## Tier 3 — measurement: does it agree with something observed?

| id | gate | threshold | measured | outcome | status | figures |
|---|---|---|---|---|---|---|
| G17 | ZAPBench held-out prediction of d(dF/F)/dt | R^2 > 0.268, the parameter-free kNN spatial pool | — | · | pending | — |
| G18 | the learned stimulus gain b_i is spatially structured | Moran's I > 0.2 over the soma graph, against a permutation null | — | · | pending | — |
| G19 | fitted calcium decay time constant | 0.5 - 2 s (GCaMP6) | — | · | pending | — |
| G20 | redox field fit reproduces the washout response | THRESHOLD TO BE FIXED from Development_Time_Trend.xlsx, before the run | — | · | pending | — |

