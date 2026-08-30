# ZAPBench in Plexus — a fitting prototype

**Status: design note only. No code yet, nothing promoted.**

This is the first Plexus prototype whose object is *fitting* rather than forward simulation.
The point is not to port a GNN into the repo — it is to express the fit as a composition of
typed operators, so that the parameters stay attached to named mechanisms and the residual
stays interpretable (plexus2.tex §"Mechanistic inverse modelling").

---

## 0. Data — built, verified, on disk

```
/groups/saalfeld/home/allierc/GraphData/graphs_data/zebrafish/zapbench/zapbench.zarr   (1.8 GB)
```

| array | shape | what |
|---|---|---|
| `traces` | (7870, 71721) f32 | ΔF/F, chunked (32, N) — all neurons at a few times |
| `positions` | (71721, 3) f64 | soma centroids |
| `neuron_ids` | (71721,) i32 | |
| `stim_frames` | (7870,) f32 | **external stimulus, already at frame rate** |
| `condition` | (7870,) i8 | 0–8, the nine task blocks |
| `onsets_img` / `onsets_ephys` | (9,) | block onsets, frame and 6 kHz sample |
| `plane_time_ephys` | (7872, 74) i64 | **acquisition sample of every plane of every volume**, −1 padded |
| `plane_count` | (7872,) i32 | planes per volume (mode 72, range 57–74) |

Assembled offline from local files (`zapbench_dff_full.npy`, `fishFuncEM/data/functional/*`);
the GCS bucket is **not reachable** from this container, so `download_zapbench.py` cannot run here.

### The acquisition fact that shapes the design

Measured from `plane_time_ephys`, at the 6 kHz ephys clock:

- **12.0 ms** between planes, **72 planes** per volume
- **0.914 s** between volumes
- **0.851 s** to acquire one volume — **93% of the frame interval**

So neurons at different `z` are **not sampled simultaneously**, and the offset is nearly a
whole frame. Per-neuron time series are fine (the offset is constant per neuron); what does
not exist is a simultaneous network state — which is exactly what a message-passing operator
needs. `plane_time_ephys` carries the exact timing, so this is correctable *as data*, not as
a modelling assumption. **Open:** the `z` → plane-index mapping is not documented upstream and
must be established before use; `traces` has 7870 frames against 7872 marker volumes, and the
alignment of the extra two is unverified.

---

## 1. The Plexus objects

### Sets

| set | N | state |
|---|---|---|
| `neuron` | 71,721 | `dff`, `v` (latent voltage), `a` (embedding), position |
| `synapse` | — | deferred: no connectome exists for these neurons (~481 EM-matched of 71,721) |

### Fields

| field | discretization | role |
|---|---|---|
| `mesh_l` | background grid, one per level ℓ | the computation substrate the neurons are transferred onto |
| `stim` | 0-dimensional in space, playback in time | the external drive |

### Maps

| map | direction | implementation |
|---|---|---|
| `π_{neuron→mesh_l}` | set → field | **MPM P2G** — quadratic B-spline, 27-node stencil |
| `π_{mesh_l→neuron}` | field → set | **MPM G2P** |

Reusing the MPM warp transfers is the whole reason this is a Plexus prototype and not a fresh
GNN: `P` and `I` already exist, are differentiable, and are fast (4.29 ns/particle-substep at
570k particles; ZAPBench is 8× smaller). The B-spline is also C¹, which the multilinear and
smoothstep interpolants used elsewhere are not — so `∂/∂t` through it is continuous.

---

## 2. The operators, and which are learnable

Kinds are the registry's own (`models/registry.py`): `lateral` / `aggregate` / `broadcast` /
`exchange` move a set's state and return a delta; `field` is a field's own dynamics
(diffuse / decay / **playback**); `rewire` / `structural` change the graph or the set.

| operator | kind | signature | learnable |
|---|---|---|---|
| `stim_playback` | `field` | → `stim` | no — it is data |
| `stim_drive` | `broadcast` | `stim` → `neuron` | **yes**: per-neuron gain `b_i` |
| `scatter_l` | `aggregate` | `neuron` → `mesh_l` | no — fixed B-spline |
| `mesh_dynamics_l` | `field` | `mesh_l` → `mesh_l` | **yes**: message ψ_ℓ, update φ_ℓ |
| `gather_l` | `broadcast` | `mesh_l` → `neuron` | no — fixed B-spline |
| `embedding` | `broadcast` | table → `neuron` | **yes**: the multiresolution `a_i` |
| `calcium` | `lateral` (self only) | `v` → `dff` | **yes**: rise/decay time constants |
| `synaptic` | `lateral` | `neuron` → `neuron` over `synapse` | **yes**, deferred with the set |

`stim_drive`, `mesh_dynamics_l`, `embedding` and `calcium` are the learnable operators. Each
owns its own parameters, so a residual can be attributed to a *mechanism* rather than to a
model. That is the property the whole exercise is for.

### Composition

```
dv/dt  =  Σ_ℓ  gather_ℓ ∘ mesh_dynamics_ℓ ∘ scatter_ℓ  (v, a)     +     stim_drive(stim, b)
dff    =  calcium(v)
```

Two things to note. The sum over ℓ is a sum of *contributions*, not of independent
predictions, and it is one-sided as written — see `connectome-gnn/docs/multilevel_one_sided_note.pdf`
for why a band-pass is needed and what it requires (`P I = Id`, which the mass-weighted MPM
P2G does **not** satisfy — it needs wrapping in a CG solve). Start without it; add it only
once a single level works.

---

## 3. How the external stimulus enters

This is the question the prototype has to answer cleanly, and both this dataset and `redox`
have one.

**GraphCast's answer is `forcing_variables`** (`weathernext/utils/task.py:26`, e.g. incident
solar radiation): quantities known for all time, appended to the node inputs at every
autoregressive step, never predicted. That is the right pattern, and in Plexus it is already
a named thing — a **`field` operator with playback dynamics** feeding a **`broadcast`** into
the set.

The one design choice: ZAPBench's stimulus is a single global scalar `s(t)`, but its *effect*
is not global — a visual stimulus drives tectum and retina, not hindbrain. So the drive is

```
stim_drive:   Δv_i  =  b_i · s(t)
```

with `b_i` learnable per neuron. **`b_i` is then a scientific readout, not a nuisance
parameter** — it is a stimulus-responsiveness map over the whole brain, and it can be checked
against known anatomy without any connectome. That makes it the cheapest real deliverable in
the whole plan, and it is available before any dynamics work at all.

`connectome-gnn` already has this slot: `NeuralGNN.forward` concatenates an `excitation`
column into `f_theta`'s input (`neural_gnn.py:685`). Here it becomes `b_i · s(t)` with `b_i`
learned. For `redox` the same operator applies unchanged — the washout is a global scalar in
time with a per-cell response.

---

## 4. The objective

plexus2.tex is explicit that the loss should be **local in space and time**: a subset of
entities over a short rollout, because reverse-mode differentiation of a whole-system
trajectory scales as entities × steps and is what makes long fits intractable. That applies
directly here — 71,721 neurons × 7,870 frames is not a loss to unroll.

```
L  =  Σ_{i ∈ R}  Σ_{t ∈ W}   w_i ( dff_i(t) − dff_i^obs(t) )²
```

with `R` a region of interest and `W` a short window. Two things carried over from the
GraphCast reading and the weekend benchmark (`connectome-gnn/papers/graphcast_for_recurrent_training.md`):

- **Weight by the inverse variance of the *increment***, `w_i = 1/Var_t(dff_i(t+1) − dff_i(t))`,
  not of the state. This is GraphCast's `s_j` (supplement §4.2) and it is what makes one loss
  weight transfer across neurons of very different activity.
- **Train one-step, fine-tune on rollout.** GraphCast spends 96% of updates at K=1 with cosine
  decay to zero, then 3.5% at K=2→12 at a learning rate 3,300× below peak. The weekend grid
  found rollout-as-objective loses; this is the arm it did not test.

---

## 5. Staged plan, with what each stage has to show

| stage | build | gate |
|---|---|---|
| 0 | locate and verify the MPM `scatter`/`gather` operators; confirm they take grid resolution as a parameter, not baked in | a P2G→G2P round trip on the 71,721 soma positions reproduces a smooth test field |
| 1 | `stim_playback` + `stim_drive` alone, no dynamics | `b_i` map is spatially structured and matches known stimulus-responsive anatomy — *this is a result on its own* |
| 2 | single level: `scatter → mesh_dynamics → gather`, one-step fit | beats a parameter-free k-nearest-neighbour spatial pool, which already reaches held-out R² 0.268 on dΔF/dt |
| 3 | `calcium` observation operator | fitted rise/decay land in the published GCaMP range, or we learn that ΔF/F is being modelled as the dynamics |
| 4 | multi-level, additive | per-level leakage measured; band-pass only if leakage is large |
| 5 | rollout fine-tune | final checkpoint agrees with the trailing median |

Stage 1 is deliberately first: it is cheap, it needs no dynamics, and it produces something
checkable against biology.

---

## 6. Not decided

- The `z` → plane-index mapping, hence the exact per-neuron acquisition offset.
- Whether to deskew the traces or **condition on the offset** (put `Δt_ij` on the transfer,
  as GraphCast puts relative position on the edge). The second is more honest — the traces are
  already sub-Nyquist for GCaMP dynamics at 0.914 s, so resampling cannot recover what was
  never sampled.
- Whether `mesh_dynamics` should be a graph operator at all on a regular lattice, where a
  translation-invariant GNN is exactly a convolution (verified to 1.19e-6). The graph framing
  earns its keep only on the irregular neuron set, which argues for the *synaptic* operator
  and against the mesh one — but ZAPBench has no connectome, so the mesh route is what is
  available.
- Connectivity ZAPBench: deferred by agreement, and it is what would make the `synapse` set
  real.
