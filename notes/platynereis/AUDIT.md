# What already exists, and must be reused rather than rewritten

Compiled 2026-09-19 from three sweeps: the Plexus operator registry and its circuit/eye campaigns,
the `prototype/microswimmer` and MPM-cardio prototypes, and `/workspace/connectome-gnn`.

The single most useful sentence in it: **the whole chain motoneuron → effector → contraction →
water already exists as separate registered pieces, and nothing in the repo chains them.** One
operator is missing, and it is small. Everything else is composition.

---

## 1. The chain, rung by rung, with the operator that does it

| rung | what it needs | the operator that already does it | where |
|---|---|---|---|
| R1 anatomy | 4,117 somata at measured positions | **`neural_seed`** | `neuron_ops.py:535` |
| | their arbours | `morphology_seed` | `neuron_ops.py:697` |
| | the body surface | `plotting.static_mesh` / `plotting.meshes` | `live_movie.py` |
| R2 fall | gravity on an MPM body | `gravity`, the four MPM operators | `motion_ops.py:468`, `mpm_ops.py` |
| R3 connectome | 4,664 edges | `edges_file:` on an `edge_set: true` set | a set property, not an operator |
| R4 activity | membrane law | **`neuron_update`** (`model: leaky_tanh`) | `neuron_ops.py:125` |
| | synaptic transfer | **`neuron_signal`** (`shared`/`type_pre`/`type_pairwise`) | `neuron_ops.py:360-394` |
| | or both fused | `signal` | `field_ops.py:780` |
| | external drive | `neuron_drive` (3-D), `neuron_field_input` (2-D only) | `neuron_ops.py:494`, `:418` |
| | the viz | `plotting.color_field: voltage`; `paint_children`; `voxelize` | |
| R5 motor → cilia | circuit drives an effector | **`readout`** | `relation_ops.py:196` |
| | per-cilium phase | **`phase_clock`** | `cell_ops.py:1418` |
| | a beat as stress | **`polar_active_stress`** | `cell_ops.py:1495` |
| | a beat as rest length | `active_strain` | `contractile_ops.py:81` |
| | a metachronal wave | `activation_pulse` with `delay_from:` | `field_ops.py:676` |
| R6 water | cilia push fluid | **two particle sets sharing one `mpm_grid`** | `mpm_ops.py:717` |
| | water behaving as water | `mpm_viscosity`, `bulk_modulus` on a liquid type | `mpm_ops.py:1764` |
| R7 tune | fit the drive | `tasks/spec_trainer.py` + `learnables/` | already batched and differentiable |

## 2. The four things that are NOT there

1. **No cilium or flagellum operator. No beat, no metachrony.** `prototype/microswimmer` is a
   *squirmer*: the cilia are never objects, their effect is a prescribed tangential slip on a
   sphere and the flow outside is the closed-form Stokes series, evaluated, never solved. Its own
   `FlowField.step()` is `pass`. It also cannot run today — it imports `scenario_schema`, which is
   no longer in the tree, and it predates `src/plexus/{schema,engine,operators}`. Archaeology.
2. **No spiking neuron model in Plexus.** Every neural operator is a graded rate/voltage law.
   (connectome-gnn does have AdEx, `flyvis_adex_ode.py` — the only model in either repo with real
   units: C 200 pF, g_L 10 nS, V_rest −65 mV, τ_w 500 ms. Translating it is a project of its own.)
3. **Nothing anywhere turns neural activity into a mechanical quantity.** I looked in both repos.
   connectome-gnn stops at membrane voltage in every one of its four biomodels — the zebrafish
   abducens motor neurons have their outgoing edges *zeroed*, the Drosophila-larva motor pool has
   nothing downstream, and the CX's `wout` is a heading readout, not a plant. In Plexus every
   neural→mechanical spec stops at the reduced 3-DOF eye plant.
4. **The one missing link, stated exactly.** `active_strain`'s activation γ comes from a CLOCK
   (`contractile_ops.py:175`, `gamma()`), not from a state block, and `active_stress`
   (`mpm_ops.py:4060`) reads a FIELD and is 2-D only. What R5 needs is a contraction operator that
   reads an `act` block **on a parent set** — the way `prototype/eye/muscle_ops.py:418`
   `muscle_contract` does — registered in `src` and writing the per-set buffer `p.act_stress`,
   which stock `mpm_scatter` already sums in (`mpm_ops.py:628-633`). That is the whole gap.

## 3. Load-bearing facts, each of which would cost a day if learned late

**The connectome joins by NAME, and most cells have none.** The manifest says it outright: *"edges
join the geometry by CELL NAME, the only key the adjacency matrix and the compendium share. A cell
without a name has no edges rather than a guessed one, so degree 0 means unjoined, not unwired."*
Only **1,196 of 4,117** cells carry a connectome name. Most of the 840 muscle cells are almost
certainly among the 2,921 unjoined. **Check the per-class degree before designing R5 around
motoneuron→muscle edges that may not exist in the file.**

**All 291 cell types in the existing spec share one placeholder parameter vector**,
`p: [1.0, 0.0, 1.2, 0.5, 1.0, 0.0]`. No per-type dynamics have been fitted. `p = [a, b, g, s, w, h]`
= leak, offset, gain, self-coupling, width, threshold (`neuron_ops.py:96`).

**`neural_seed` checks units and count, and refuses both.** `general.units.length_um` must equal
the manifest's `side_um` — **195.9** for this animal — and `per_parent` must equal 4,117. It
refuses rather than truncating, and it is right to: seeding the first N rows of a different animal
is indistinguishable, in the output, from seeding the right one.

**A shared MPM grid is the stock behaviour, not an extension.** `mpm_scatter` zeroes the grid only
on the substep's *first* scatter (`mpm_ops.py:717-719`, tagged `shared_grid_accumulate`), so
several particle sets exert force on each other through one grid with no special implementation.
`config/cell/adh_base.yaml` runs fifteen. `prototype/eye/muscle_ops.py:541` registers an
`implementation: accumulate` that predates this and is no longer needed in `src`.

**`active_strain` goes at the END of the schedule, after the substep block.** The engine calls
`on_frame` at the bottom of the tick loop; at the head of the schedule it applies one step early,
and the prototype measured the cost: **23% of the per-cell strain signal**.

**`readout` masks properly.** `at: cilium[type=prototroch]` writes only those elements
(`relation_ops.py:164-172`), so one pool per band composes instead of clobbering. Its two
nonlinearities are not interchangeable: `send:` is the sender's rate code, `activation:` is the
receiver's ("a muscle pulls or does nothing" — `softplus`).

**`phase_clock` exists because `pacemaker` is the wrong object for a band.** `pacemaker` publishes
ONE scalar per tick to `H.signals`; every cell would beat in lockstep. `phase_clock` gives each
cell its own angle with rate `jitter`, and `cell_ops._gate`'s `offset` puts two operators at
different points of the same cycle — which is the mechanism a metachronal wave needs.

**`polar_active_stress` is sign-reversing and deviatoric**: `σ_act = A·cos(φ+offset)·(n nᵀ − I/3)`,
so it extends then retracts over a cycle and preserves volume, and its amplitude can be declared
as a target strain (`amplitude_frac = A/(λ+2μ)`). The body-force version of the same idea needed a
drive **47× larger** before its deformation was visible — that is recorded in its docstring.

**`active_force` and `active_stress` are 2-D only.** For 3-D contraction: `active_strain`,
`polar_active_stress`, or the new operator of §2.4.

**`src/plexus/operators/candidates/` is not auto-imported** and nothing in it is reachable from a
spec. Neither is anything in `prototype/`, unless a prototype runner imports it.

**`library/*.qmd` is stale in places** — several pages name operators that no longer exist under
those names (`eye_mechanics`, `muscle_gaze_map`, `agent_to_mpm`, the `integrin_*` trio). Trust the
registry, not the library pages.

**`paint_children` raises the integration invariant on every run.** Pre-existing; reproducible with
the repo's own `config/neural/hemi_soma_skeleton_1000.yaml`. Do not adopt it as mine.

## 4. The reference spec to copy from, for each rung

| rung | spec |
|---|---|
| anatomy + activity | `config/neural/platynereis_larva.yaml` — 4,117 neurons, 291 types, connectome, `neural_seed`, `neuron_update`, `neuron_signal`. It already exists and already runs. |
| arbours | `config/neural/platynereis_larva_arbours.yaml` — adds `morphology_seed` + `paint_children`, `subject: morphology`, `color_field: paint`. |
| the fall | `config/neural/platynereis_drop.yaml` — `shape: obj`, 60,000 MPM particles, warp implementations. |
| two bodies in one grid | `config/studio/platynereis_a3_plankton.yaml` — the water template for R6. |
| circuit drives an effector | `config/neural/zf_eyeG_285.yaml:92-141` — `schedule: [project, neuron_update, neuron_signal, readout, readout, muscle_pose_map, organ_mechanics]`, with one masked `readout` per effector group over its own `edges_file`. **This is the R5 template.** |
| a beating cell | `config/cell/adh_stress_f25.yaml` — `phase_clock` (ω 12.6, jitter 0.15) + `polar_active_stress` (`amplitude_frac: 0.25`, `deviatoric: true`). |
| contraction in tissue | `config/tissue/cardio_strain_healthy.yaml` and `..._3beats.yaml` — `seed_state_from_file` → `material_from_cell` → `active_strain`, bit-identical to the prototype in float32. |
| edge-file builder | `tools/make_zf_eye_edges.py` — reads the index ranges out of the spec's own `types:` table so the two cannot drift. **Copy this for the motoneuron→cilium junction.** |

## 5. The leaky integrator, written out

The law both repos share, in Plexus's own form (`neuron_ops.py:8`):

    dx_i/dt = -x_i/tau_i + s_i*phi(x_i) + g_i*Omega_i(t) * sum_j W_ij*psi_ij(x_j) + eta_i(t)

`x_i` is neuron i's membrane state (dimensionless here — these are not millivolts); `tau_i` its
membrane time constant in the run's own time unit; `s_i` a self-coupling; `phi` the local
nonlinearity; `g_i` a gain; `Omega_i` an optional multiplicative field; `W_ij` the connectome
weight from j to i; `psi_ij` the synaptic transfer function; `eta_i` noise of declared standard
deviation per step. `neuron_update` is the `phi` half, `neuron_signal` the `psi` half.

connectome-gnn's flyvis form is the same object with `s = 0`:

    dv_i/dt = ( -v_i + sum_j W_ij*ReLU(v_j) + I_i + V_rest_i ) / tau_i

with `tau_i ≈ 0.019` s and `V_rest_i` from the flyvis node bias. Set `a_i = 1/tau_i` and
`b_i = V_rest_i/tau_i` and Plexus's `leaky_tanh` with `s = 0` is exactly it. **Units are arbitrary
in every model in either repo except AdEx — do not paste mV/ms onto these numbers.**

Weight normalisation differs by dataset and matters: spectral radius 0.9 on max(Re λ) for
zebrafish/CX/C. elegans; row-normalised by total input count for zebrafish *before* that; for
FlyWire `W_ij = (0.01/mean_n_syn(type→type)) · log(n_syn_ij) · sign_j`. **Raw contact counts are
never used directly anywhere.**
