# Transfer plan: promoting the Tyssue AVM operators into core Plexus

*Planning only — no core-registry edits yet. Guided by plexus2 App. "Implementing new operators",
"Promotion", and "Refactoring guidelines": **prototype freely, promote biological mechanisms
conservatively.** A prototype op is promoted only when it has a stable contract, at least one
validated implementation, clear provenance, and evidence it is reusable beyond this prototype.*

---

## 1. The one real blocker: the mesh representation

Every mechanics/topology operator here reads/writes a **prototype-specific stash**, `lvl._mesh`, a
Python dict holding the half-edge table (`E_srce/E_trgt/E_face` tensors), a numpy face-ring list
(`faces`), and per-face buffers (`A0_np`, `alive_np`, `pin`, `Nv`). This is not a Plexus primitive
— it is an ad-hoc attribute on a `Level`. **Nothing should be promoted until this becomes a
first-class runtime object**, because the promoted operators' typed signatures must name real
Maps/state, not `getattr(lvl, "_mesh")`.

Two options, in order of preference:

- **(A) A `HalfEdgeTopology` runtime object** carried by the Hierarchy (like `Field`), holding
  `srce/trgt/face` index buffers + a `face`-ring view, with methods `rebuild()`, `neighbours()`,
  `add_vertices()`. Operators traverse it through named maps (`srce`, `trgt`, `face`) exactly as
  plexus2 describes ("maps are index buffers, named by the operators that traverse them"). This
  matches tyssue's own architecture (a single half-edge Level) and the extraction in
  `tyssue_atlas.yaml`. **Recommended.**
- **(B) Two Levels (`vertex`, `face`) + edge Maps.** Cleaner on paper but the `face` Level carries
  no *integrated* state (area/perimeter are aggregates), so it buys little and doubles bookkeeping.
  Reject for now.

Until (A) lands, the operators stay in the prototype. The `cell` **set** (a0/chem/ctype — genuine
integrated biological state) is already a proper Level and does **not** need this refactor.

---

## 2. Promotability triage

| prototype operator | core contract | status | note |
|---|---|---|---|
| `shape_energy` | `area_elasticity` + `perimeter_elasticity`/`contractility` + `line_tension` (existing, per atlas) | **decompose, then promote** | it is a *composition*, not one contract; split into the three effector contracts + the bounded-Euler *solver*. The bounded step is the reusable novelty. |
| bounded Euler step (`cap_frac`) | `integrate` / `relax` implementation | **promote** (small, general) | a differentiable, displacement-capped overdamped step; reusable by any overdamped mechanics. Independent of the mesh. |
| `t1_transition` | `reconnect` (new) | **needs (A) first** | contract is clean (rewire on the edge set); the impl (simple-polygon + non-overlap accept, insertion-side search) is validated. Promote the *contract* once maps are first-class. |
| `face_divide_line` | `divide` (existing family) | **needs (A) first** | straight-line/edge-midpoint split, decidable angle, area-conserving. Sound. Needs the vertex-buffer/occ story generalised. |
| `apoptosis` / `face_extrude` | `die` / `extrude` (existing family) | **needs (A) first** | apoptosis is a *behaviour* (shrink+T1+collapse), not a primitive — promote `extrude`/`collapse` as the primitive, keep the behaviour as a schedule. |
| `cell_geometry` | `aggregate` (existing family) | **promote after (A)** | genuine vertices→cell aggregate; trivial once maps are first-class. |
| `cell_morphogen`, `morphogen_growth`, `cell_differentiate` | `field` / `growth` / `differentiate` | **keep prototyping** | `differentiate` (French-flag threshold) is a clean new contract worth promoting later; the imposed-bump morphogen is a demo scaffold, not a mechanism. |
| `cell_diffuse`, `cell_react` (RD) | `diffuse` / `react` (exist in main registry already) | **do NOT re-promote** | the core already has `diffuse`/`react`; these are forks. Reconcile, don't duplicate. |
| `seed_mesh`, `seed_cell`, `cell_adjacency`, `topo_snapshot` | seed / rewire / tooling | **stay prototype** | `seed_mesh` bootstraps via a Voronoi (prototype convenience); `topo_snapshot` is a render helper. |

---

## 3. What to promote, in order

1. **The bounded-Euler step** — smallest, mesh-independent, reusable. A capability/param on the
   existing overdamped integration path (`EMIT=velocity` + `cap_frac`). No new contract.
2. **`HalfEdgeTopology`** (§1A) — the enabling refactor. Not an operator; a runtime primitive +
   the `srce/trgt/face` maps. Everything topological depends on it.
3. **`reconnect` (T1)** contract + the validated 2D implementation.
4. **`divide` / `extrude`** primitives (the family already exists; add the vertex-model impls).
5. **`aggregate` geometry** (`cell_geometry`) and **`differentiate`** (French-flag).
6. The apoptosis/division/growth *behaviours* as **schedules** composing the above — not as core
   operators (they are compositions; plexus2 keeps behaviours in specs, not the engine).

---

## 4. Validation gates before each promotion (plexus2 App)

For every operator moved to `src/plexus/operators/`:
- passes `tools/audit_operator_registry.py` (valid `family`/`kind`, canonical `EMIT`,
  `SUPPORTED_DIMS ⊆ {2,3}`);
- a minimal `spec.yaml` using it loads through `schema.py`;
- a test reproduces the prototype numbers (rigidity transition; T1 self-test 0 invalid; division
  118→159; apoptosis 118→117; growth 88→176);
- provenance in the docstring (Bi 2015 / Farhadifar 2007 / Okuda RNR / Monier 2015 / tyssue).

## 5. Explicitly NOT promoting yet

- The `_mesh` dict (replaced by §1A first).
- `seed_mesh`'s Voronoi bootstrap (prototype convenience; core should take an explicit mesh).
- The imposed-bump morphogen (a demo, not a mechanism) — promote `differentiate`, not the bump.
- Anything 3D (IH/HI, monolayer) — not yet prototyped here (Goal 1bis).

## 6. Net

The **mechanics** contracts are already in the core algebra (the atlas showed saturation); the
genuinely new, promotable material is: the **bounded differentiable Euler step**, the **`reconnect`
(T1)** contract, the vertex-model **`divide`/`extrude`** implementations, and **`differentiate`** —
all gated behind making the half-edge topology a first-class runtime object. Until that refactor,
promotion is premature; the prototype is the right home.
