# Building *Mnemiopsis leidyi* in Plexus

A staged plan for a model of the comb jelly's aboral organ and the swimming animal it steers,
built as Plexus sets, fields and activities, with the hierarchy and composition the design paper
(`paper/plexus2.tex`) requires.

The target is the loop the animal actually closes: **gravity tilts a statolith → the statolith
presses harder on one group of balancer cilia → that balancer's beat frequency changes → a
ciliated groove conducts the change to two comb rows → those rows beat differently from the
others → the body turns → the tilt changes.** On top of that, a second sensory channel: the
animal feels the density of food particles drifting past it and turns toward them.

Every stage below ends in something that runs and something that can be looked at. Nothing is
"designed then built"; each rung is a working animal with less of it missing than the last.

---

## 0. What is on disk, and what it is worth

Everything is under `graphs_data/ctenophore_ao/`. Three datasets, three different animals, three
coordinate frames — **never merge their coordinates**.

| Source | What it gives | Status |
|---|---|---|
| **Jokura et al. 2026, eLife** — `jokura2026_elife/` | A CATMAID reconstruction of the aboral organ: **903 cells in the organ** (880 with a named cell type, 23 untyped) out of 1,037 traced skeletons, 1,132 synapses with 3D coordinates, 997 cilia with measured lengths and axoneme type, per-cell mitochondria and centriole counts, a quadrant label on 738 cells, and 67 high-speed recordings of balancer ciliary beating each paired with a body tilt angle | **The spine of the model.** Frozen as a Plexus region by `build_region.py` → `graphs_data/neural_regions/mnemiopsis_ao_271/`, 271 cells with positions and 303 directed synaptic edges |
| **Sachkova et al. 2021** — `Sachkova2021_TrakEM2_project.xml` | 63 hand-traced 3D objects at 20 × 20 × 100 nm: a nerve-net neuron with its nucleus, mitochondria, Golgi, dense-core and clear vesicles and a synapse; 34 comb-row cells with 7 traced macrocilia bundles; 8 ciliated-groove cells with 2 cilia bundles; a sensory neuron with its cilium and synapse | **The only traced geometry.** Extracted to point clouds by `extract_trakem2.py` → `trakem2_geometry/`, 585,471 points, scene 59 × 83 × 133 µm |
| **Ferraioli et al. 2026, Sci. Adv.** — the paper the work started from | 17 cell types and their counts (890 cells), the organ's architecture, the nerve net's condensation around it, and the proposed signal path | **Prose and counts, no geometry.** Its figshare deposit (71 GB) is five RAW unaligned SBFSEM stacks with no segmentation and no meshes |

The SBFSEM stacks *are* downloaded (`sbfsem/`), and animal1 — 5,784 × 4,962 px × 697 sections at
12 nm/px and 100 nm/section, a 69 × 60 × 70 µm volume — covers the whole aboral organ. It is
browsable at `python graphs_data/ctenophore_ao/em_viewer.py` (a slice viewer on port 8801) and
downsampled to a near-isotropic 96 nm overview in `sbfsem_overview/`. It is **unaligned**: the
paper aligned in Fiji and deposited only raw sections, so `align_stack.py` estimates one
translation per section by phase correlation before anything is measured off it.

**The counts, stated once so they are not confused again.** `stats_master.csv` has **1,037 rows**,
which is traced skeletons, not cells of the organ: 134 are named `outside*` by the tracer and lie
outside it, leaving **903 in the organ**, of which 880 carry a named cell type. (Jokura's own
abstract says 1,011 reconstructed cells, which matches neither and cannot be reconciled from the
deposit; Ferraioli counts 890 in a younger animal.) Of all of them only **271 have coordinates**,
because a synapse site is the only position the shared tables record, and those coordinates are
the centroid of each cell's own synapses, not its soma. The rest of the organ's
geometry has to be built, and stage R1 is where that happens.

### The authors' own analysis repository, and the one thing still out of reach

`jokura2026_repo/` is a clone of <https://github.com/JekelyLab/Jokura_2024_ctenophore_apical_organ>,
which holds much more than the figure source data:

- **67 raw ciliary-beat-frequency recordings** of the balancers, each a time series in light or in
  dark, at frame rates of 100.07 to 100.75 frames per second
  (`analysis/data/balancer_CBF_Pearson_correlation_analysis/csv/`), plus the rolling left-right
  Pearson correlation at five window lengths. This is the dataset R4 fits the beat dynamics to,
  and it is far richer than the two recordings that reached the paper's figures.
- **The authors' own 3D renderings of the organ, per cell type**
  (`manuscript/pictures/3d_plot/`), including `all_cells_3_views_alt.png` — every cell as a sphere
  with its neurites, in three views. The top view shows the **four balancer groups at 90°**
  radiating from the centre with the lithocytes among them. **This is the target picture for R1**;
  build against it.
- The R analysis scripts, including `contactome.R` and `synapse_positions.R`, which are the
  authors' own definition of every quantity used above.

**What is still out of reach.** The reconstruction's skeletons — the full morphology of every one
of all 903 cells of the organ, which would give R1 real somata and end the positioning problem outright —
live on the Jékely lab CATMAID server at <https://catmaid.jekelylab.ex.ac.uk/>, and the
electron-microscopy volume behind it is EMPIAR-13030. Both are **blocked by this devcontainer's
firewall** (`.devcontainer/init-firewall.sh` whitelists a fixed set of hosts; figshare and GitHub
are reachable, these are not). Fetching them from a machine with open network access, via the
`rcatmaid` helpers in `jokura2026_repo/rcatmaid_functions_library/`, would be the single highest-value
thing to do before R1 — it would replace the whole procedural-geometry step with measurement.

### What is measured that the model must reproduce

These are the validation targets. They come from the data, not from the modeller.

- **The wiring is a hub and a fan.** Three subepithelial-nerve-net cells make 164, 121 and 67
  outgoing synapses and drive nonciliated (210 reconstructed edges), monociliated (126), bridge
  (105) and balancer (84) cells. They receive almost only from bridge cells.
- **The steering circuit is quadrant-wired.** Nerve-net cell ANN Q1Q2 contacts balancers Q1 (14
  contacts) and Q2 (15); ANN Q3Q4 contacts balancers Q3 (2) and Q4 (3); a third cell, ANN Q1-4,
  contacts all four (10, 12, 9, 19). Bridge cells feed the net (57 and 30 contacts). A per-quadrant
  nerve-net cell driving a per-quadrant balancer **is** differential thrust.
- **Arrest is plane-dependent, re-beat is not — and the wiring explains both.** The left and right
  balancers of a **sagittal** pair begin to arrest within 0.159 s of each other (range 0.020–0.263,
  n = 5 larvae); a **tentacular** pair is offset by 1.726 s (range 0.650–2.163). Re-beat, by
  contrast, is near-simultaneous in *both* planes (0.273 s and 0.227 s). The authors' explanation
  uses only the connectivity already in hand: a sagittal pair shares one nerve-net cell (ANN Q1Q2
  or ANN Q3Q4) so its balancers stop together, a tentacular pair is driven by two *different*
  nerve-net cells so it need not, and ANN Q1-4 — the one cell contacting all four balancers —
  restarts them together. **This is the primary fitting target of R3.** A model wired from the
  measured contact matrix should reproduce the sign and the ordering with nothing added.
- **Balancer beat frequency, over 65 larvae and 5,073 recording-seconds** (`balancer_cbf.yaml`,
  from all 67 recordings in the authors' repository rather than the two in their figures):
  9.47 Hz in the sagittal plane against 8.31 Hz in the tentacular. The left–right correlation is
  **0.17 sagittal against 0.48 tentacular** — tentacular pairs nearly three times more coordinated,
  which is the paper's own result (t-test on their rolling statistic, p = 0.014) recovered here
  from an independent whole-record statistic. The mean absolute left-minus-right difference is
  8.70 Hz and 6.15 Hz; that difference is the steering signal.
- **Every recording is paired with a body tilt angle** (`balancer_info.csv`), so the stimulus and
  the response of the geotaxis reflex sit in one table. That is what R3 fits against.
- **The authors propose synapse SIGNS.** Their circuit figure marks arrest-inducing synapses
  separately from re-beat- and frequency-increasing ones, so the nerve net acts in both directions
  on its targets. Nothing measures the sign, but the model has to choose one, and their assignment
  is the hypothesis to start from.
- **The paper's own conclusion is that the nerve net is coordinating, not sensory-motor.** A model
  that makes it a relay from statolith to comb rows contradicts the data it is built on.

---

## 1. The categorical decisions, and why

`plexus2.tex` makes five hierarchy rules load-bearing. The decisions below follow them; the
reasoning is quoted where it decides something.

**A set is a kind, and a kind is one state schema** (`plexus2.tex:213`). Two things needing
different state are two sets, never two types of one set. A balancer carrying a beat drive and a
lithocyte carrying mineral mass do not share a schema.

**A level is not a kind** (`:228`). The aboral organ's level legitimately holds balancers, a
statolith, a nerve net, a contact relation and a field at once.

**Containment is declared, and π is a function** (`:264`, `:297`). Aggregate (Σ along π) and
Broadcast (π*) are the *only* two families that cross a level.

**Many-to-many needs no new primitive** (`:307`). A relation *is* an entity with two maps. One
statolith resting on four balancers is an edge-set, not a special case.

### The hierarchy — one root, two branches

```
organism                              pose, heading, gravity vector
├── aboral_organ                      the tilt estimate
│   ├── statolith                     position, velocity, mass
│   ├── lithocyte      (parent: statolith)
│   ├── balancer                      beat drive, beat phase
│   ├── ao_cell                       the other organ cells, by type
│   ├── nerve_net_cell                voltage
│   └── contact  (edge-set: statolith × balancer)   load
└── comb_row              × 8         gain from its balancer, metachronal phase
    └── comb_plate        × ~20       beat phase, paddle angle, thrust
```

Both branches hang off one root. π stays a function, so this is legal and needs nothing new.

### The decision table

| Thing | Verdict | Why |
|---|---|---|
| organism | **Set**, `n: 1` | Carries pose and heading, which its own activities change (`:269`) |
| aboral organ | **Set**, `parent: organism` — a level | Has its own state and its own activity; not a parameter of the body |
| balancer, lithocyte, groove cell, nerve-net cell | **separate Sets** | Different state schemas (`:213`). Only cells differing purely by a *parameter* are types |
| the 17 anatomical classes in the connectome region | **Types within one `neuron` set** | Here they genuinely share one schema — voltage — and differ by parameters. This is what the GUI tab already does |
| cilium | **Set** if the beat phase is resolved per cilium; a **State** on the cell otherwise | `:269` decides it: a level must have its own rate. Resolve it at R4, not before |
| statolith | **Set** with pos/vel, plus a `lithocyte` child set | Chosen over a bare 3-vector state because forces act *on* it, and over a pure aggregate because lithocyte count is data (8 in Ferraioli, 5 in Jokura) |
| nerve net | **Set** `nerve_net_cell` + **edge-set** `synapse` with `pre`/`post` | `:307` exactly. **Flagged:** the real net is partly syncytial, which is a continuum, not a population of pairs. R3 models the synaptic net; the syncytial reading is a separate arm, and the two are not interchangeable |
| statolith load on a balancer | **State `load` on the `contact` edge-set**, delivered to the balancer by **Aggregate** | One statolith on four balancers is not a function, and `:300` says a many-to-many relation admits neither Aggregate nor Broadcast. Give the contact entity status and both legs become ordinary maps |
| comb row | **Set**, `parent: organism, per_parent: 8` | It carries the metachronal phase travelling down the groove. Without it that phase has nowhere to live |
| water | **Field** (velocity) at R5; a **particle Set** plus a grid Field if MPM is used at R6 | Not merely a numerical choice: it retypes every activity that touches water. Decide once, at the top of the spec |
| food particle | **Set** | Discrete, counted, eaten. Advection is an Exchange; being eaten is a Die |
| ciliary beat frequency | **State** on the comb plate, **written by a Broadcast** from its row | A number that never changes is a type parameter; this one is imposed from above, which is precisely Broadcast (`:454`) |
| sensed particle density | **State** on the sensor, written by an **Exchange** | Particles deposit into a `food` field; the sensor gathers it back. Two Exchanges and one field, as `config/slime/` already does |

**The two genuinely open questions**, flagged rather than forced: whether the nerve net is a
population of pairs or a syncytial continuum, and whether water is a field or particles. Both must
be decided in writing at the top of the spec, because neither is reversible downstream.

---

## 2. What does not exist yet and must be written

Found by inventorying the live registry. These are the real engineering costs.

1. **Broadcast has ZERO implementations.** The registry holds 60 lateral, 17 exchange, 29 seed, 21
   structural, 7 rewire, 7 field, 6 aggregate, 1 divide, 1 die — and no operator registered
   `kind="broadcast"`. Organ→cell and row→plate top-down control is exactly what this model needs,
   so **the first registered Broadcast operator in Plexus gets written here.** (`paint_children` is
   tagged `aggregate`, subclasses `Exchange`, and its body is π* — a Broadcast wearing two wrong
   labels. Do not use it as the precedent; read `forward()`, not `kind=`.)
2. **`Field.sample` and `Field.grad_at` are 2D-only** (`models/base.py:424`). A 3D read-back of a
   transmitter or food field does not exist and must be written. `neuron_drive` advertises
   `SUPPORTED_DIMS = [2, 3]` while calling the 2D sampler, so a 3D spec loads and dies on the first
   tick.
3. **There is no "sample the fluid velocity where I am".** `Field.sample` on an MPM grid returns
   **mass**, not velocity; the nodal velocity is read by exactly one line in the repo, inside
   `mpm_gather`.
4. **There is no orientation counterpart to `aggregate_centroid`.** A body's position is recovered
   from its material points; its *orientation* is not. Differential thrust produces a torque, and
   nothing currently integrates one.
5. **`voxelize` rebuilds the grid every tick rather than accumulating**, so the obvious
   "splat the transmitter and let it diffuse" wiring silently discards the previous tick's pool.
6. **The microswimmer prototype does not import today** — `prototype/scenario_schema.py` was
   deleted in commit `b01958a8`. Its four operators (`squirmer_flow`, `slip`, `swim`, `absorb`)
   are worth porting into the main registry rather than reviving its private engine; note its
   `swim` has no torque at all, and its `axis` buffer is written by nothing.

Two traps that cost an afternoon each if unknown: **declare sets parent-first** (the engine walks
them in file order and reports a child's declared parent as undeclared), and **give any edge-set
not literally named `synapse` an explicit `entity: connection`**, or its weights are silently
discarded. Avoid the registered set names `cell`, `organ`, `neuron`, `membrane`, `synapse`,
`muscle`, `nucleus`, `particle`, `eye` unless you want their reserve factors — a set named `cell`
with `per_parent: 3` allocates a buffer of 16.

---

## 3. The stages

Each stage names what it adds, what it needs written, and the picture that says it worked.

### R0 — the measured circuit, running *(done)*

The connectome as a Plexus region, rendered in the web interface, integrating as continuous-time
rate neurons over the measured contact counts.

- `graphs_data/ctenophore_ao/build_region.py` → `neural_regions/mnemiopsis_ao_271/`
- `src/plexus/gui/tabs/mnemiopsis.py`, one line in `tabs/__init__.py`, one route in `server.py`
- **Look at:** `http://127.0.0.1:8799/?tab=mnemiopsis` — three red nerve-net cells at the centre,
  blue balancers ringing them, magenta bridge cells in a band. Switch `colour by` to activity.

**What it does not yet do:** the dynamics are a placeholder. No membrane time constant has been
measured in any ctenophore cell, so the slow/fast split across the 17 classes is a choice. Fitting
it is R3's job, not a detail to tune now.

### R1 — the organ's body: geometry for the 619 cells without coordinates

The region holds 271 cells with positions, against 903 in the organ. The other ~630 need them, and there are two honest routes,
to be used together:

- **Procedural, from the anatomy.** Biradial symmetry about the oral–aboral axis; four balancer
  groups at 90°, separated by the bridge; 8 lithocytes resting on them; two polar fields on the
  tentacular plane; four ciliated grooves leaving toward eight comb rows; a dome enclosing it all.
  Counts from Ferraioli's table.
- **Anchored, from the EM.** Register `sbfsem_overview/` with `align_stack.py`, then blob-detect
  nuclei in the aligned overview to get real cell centres at ~96 nm resolution. It does not need to
  be precise; it needs to put the dome above the floor and the balancers under the statolith.

The TrakEM2 point clouds give the comb rows and grooves their real shapes, at their own scale.

- **Write:** a `ctenophore_seed` operator (`kind: seed`), reading the layout from
  `graphs_data/ctenophore_ao/`. An operator and a spec, not a script.
- **Look at:** the organ from three angles beside Fig. 2B–C of the paper. The balancers must sit in
  four discrete groups under the statolith; the dome must enclose the rest.

### R2 — the statolith and the first Broadcast

The mechanical half of the reflex, with no neurons in it yet.

- `statolith` set with pos/vel/mass; `lithocyte` children; a `contact` edge-set carrying `load`.
- Gravity as a **Lateral** on the statolith; `load` computed per contact; delivered to each
  balancer by an **Aggregate** along the contact's `post` leg.
- The balancer's drive reaches its comb row by the **first registered Broadcast operator**.
- **Look at:** tilt the organism by hand through ±30° and plot `load` per balancer group against
  tilt angle. It must be a clean, monotone, four-phase pattern. If it is not, the contact geometry
  is wrong and nothing downstream can be trusted.

### R3 — the circuit in the loop, and fitting it

Join R0's measured wiring to R2's mechanics: balancer load is the circuit's input, nerve-net
output is the comb rows' gain.

- Sensory → circuit → effector is already expressible with `project` and `readout`
  (`relation_ops.py`); `config/neural/zf_circuit_285.yaml` is the working template.
- **Fit** the unmeasured cell parameters against the measured behaviour. The target is the
  plane asymmetry, not a single latency: left-right arrest offset 0.159 s sagittal against 1.726 s
  tentacular, while re-beat offset stays near 0.25 s in both planes; and left-right beat
  correlation 0.17 sagittal against 0.48 tentacular. Four numbers, one wiring, no free choice about
  which cell drives which — that is as strong a constraint as a connectome model gets.
- **Decide and state**: synaptic net, or syncytial continuum. If syncytial, the net becomes one
  fused set with a Lateral continuity term, or a field on a graph — not an edge-set.
- **Look at:** the neural panel with the measured connectome, and arrest latency by plane against
  the measured range.

### R4 — cilia that beat

Until now "beat frequency" is a number. Here it becomes a phase.

- `comb_plate` set with a beat phase; metachrony as a **Lateral** along the row with a phase lag;
  the row's gain arriving by **Broadcast**.
- Calibrate against the measured left/right beat counts per frame, converted to hertz using the
  frame rate recorded in the source file names.
- **Look at:** a kinograph of plate phase down one row. Metachronal waves must travel, and the
  left-minus-right frequency difference must match the recorded spread.

### R5 — thrust, torque and a body that turns

- Thrust per plate from its beat; summed to the row by **Aggregate**; row thrusts summed to the
  organism by **Aggregate**.
- **Write:** the orientation counterpart to `aggregate_centroid` — a torque from off-axis thrust,
  integrated into the body's pose. This is the missing piece that makes differential beating a turn.
- Water as an analytic Stokes flow first (port the microswimmer's `squirmer_flow`), which is cheap
  and sufficient to show steering.
- **Look at:** release the animal tilted by 20° and watch it right itself. The loop closes here:
  the righting must be produced by the circuit, not prescribed.

### R6 — feeding the particles in, and a microswimmer that turns toward food

The stage the whole thing was for.

- Water as **MPM particles** (`mpm_ops.py` has the full MLS-MPM solver) carrying floating food.
- **Write:** a fluid-velocity gather (sampling an MPM grid's velocity at a point does not exist),
  and a 3D field sampler for the food field.
- Particles deposit into a `food` field; the organ's sensory cells gather it (two Exchanges and one
  field, the `config/slime/` pattern); the asymmetry in what the quadrants sense biases the
  nerve-net drive; the body turns up the gradient.
- **Look at:** a movie of the animal in a patchy particle field, turning toward density. Plot
  heading against the sensed left-right difference — the control law should fall out as a line.

---

## 4. Order of work, and what to decide before writing any of it

Two decisions must be made in writing before R2, because neither is reversible:

1. **Is the nerve net synaptic or syncytial?** The measured connectome is synaptic, so R3 starts
   there — but Ferraioli's central claim is that the net is a *fused continuous* network, and the
   dense-core vesicles traced along all 225 sections of `neu1` are a volume-transmission substrate,
   not a synaptic one. Modelling both is a real result; modelling one while describing the other is
   the failure mode.
2. **Is water a field or particles?** R5 can run on an analytic flow. R6 needs particles. Declaring
   it at R5 and switching at R6 retypes every activity that touches water, so declare particles
   from R5 and accept the cost, or accept a rewrite.

R0 is done. R1 is the next rung and the only one blocking everything else, because every later
stage needs the organ to have a body.

---

## 5. Where things live

```
graphs_data/ctenophore_ao/        the dataset: EM stacks, overview, figures, traced geometry,
                                  the Jokura source data, and the scripts that made each
graphs_data/neural_regions/mnemiopsis_ao_271/    the frozen connectome region
src/plexus/gui/tabs/mnemiopsis.py the web interface tab
config/neural/mnemiopsis_ao_circuit.yaml         the R0 spec, run by Plexus_Main.py -o generate
notes/ctenophore/                 this plan
```

Run the page with `PYTHONPATH=src python Plexus_gui.py --host 0.0.0.0 --tab mnemiopsis`; the
`--host 0.0.0.0` matters, because a loopback-only bind is invisible to the editor's port
forwarding and the page simply fails to load.
