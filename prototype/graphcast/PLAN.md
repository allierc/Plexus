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
| R1 | Four **options**, not forks: `encoder_decoder` off/on, `message` simple/graphcast, `n_passes` 1…16, `embedding` none/free/multires |
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

## 3. The three toys

All are the same two-scale spatial PDE and differ **only in the coarse rule**, so the comparison is
controlled. The fine rule is always

```
dv_i/dt = −v_i/τ_i + g_i · ∂u/∂x(r_i)        g_i SIGNED — this is the heterogeneity
```

| toy | coarse rule | why |
|---|---|---|
| `toy_counter` | two counter-propagating waves, different λ | `u` no longer fixes `∂u/∂x`, so the graph is required **even when the drive is observed** — closest to the real datasets, where the stimulus is known |
| `toy_withhold` | one travelling wave, drive **not** given to the model | phase can only come from the spatial pattern of neighbours; simplest, and the variant already measured |
| `toy_envelope` | `u = A(x,y)·sin(2π(x/λ − t/T))` | the sinusoid's gradient is local but `∂A/∂x` is not; a graded case between the two |

Which coarse rule best forces the graph is then **an experiment the gates decide**, not a guess.
No flyvis-scale toy.

**Toy design constraints, learned the hard way and now pre-conditions:**
- nodes placed with `spawn: random` so they lie **inside** the field domain (58.8% were outside);
- `λ` short enough that a k-neighbourhood spans real phase (`λ=0.5` gave neighbour correlation 0.84);
- the finite-difference step `δ ≪ λ`;
- types assigned independently of position, verified against a **permutation null**.

---

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
| id | gate | threshold |
|---|---|---|
| G1 | all option combinations parse | 24 of 24 |
| G1b | all option combinations build and take one step | 24 of 24 |
| G2 | no dataset path, name or dimension in code | 0, AST-scanned, docstrings exempt |
| G3 | `scatter→gather` round trip on a constant field | < 1e-6 of the field value |
| G4 | transfer weights are a partition of unity | \|Σw − 1\| < 1e-6 |
| G5 | `simple`+1 pass+no enc/dec ≡ `NeuralGNN` at copied weights | < 1e-5 of the voltage range |
| G6 | residual blocks start at identity: 1 vs 16 passes at init | bit-identical |
| G7 | units declared; no measurement threshold in mesh units | pass/fail |
| G8 | a K=20 rollout does not diverge | state norm < 2× the GT norm |

### Tier 2a — the toy is a valid test bed (DATA only, **before any training**)
| id | gate | threshold |
|---|---|---|
| G21 | the coarse field is the rule it claims | phase speed within 5% of λ/T |
| G22 | the fine rule is recoverable from `(v, ∇u)` | **min** per-node R² > 0.90 |
| G23 | `∇u` is reconstructible from neighbours | R² > 0.95 |
| G24 | the heterogeneity is linearly readable | corr(fitted, true `g_i`) > 0.90 |
| G25 | connected nodes are not collinear | mean \|corr\| < 0.80 |
| G26 | **the graph is necessary**: a node-local baseline cannot fit | node-local R² < 0.50 while `(v, ∇u)` > 0.90 |
| G16 | types are spatially mixed | purity / permutation null < 1.2 |

**G26 is new and is the gate that would have caught the travelling-wave defect directly.**

### Tier 2b — closed form (does the fit reproduce the physics it was given?)
| id | gate | threshold |
|---|---|---|
| G9 | the message becomes a gradient operator | R² > 0.90 against `∂u/∂x` |
| G10 | recover the per-node time constant | R² > 0.95 against known `τ` |
| G11 | the embedding recovers the types | ARI > 0.70 |
| G13 | recover the per-node **signed gain** | R² > 0.90 against true `g_i` |
| G14 | encoder/decoder is a genuine option | \|Δ R²\| < 0.03 (2× the measured floor) |
| G15 | `graphcast` vs `simple` is resolved either way | Δ reported against a 3-seed floor |
| G27 | which coarse rule forces the graph | rank the three toys by G26; report, do not tune |

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
