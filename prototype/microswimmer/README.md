# Microswimmer prototype — a ciliate squirmer in Plexus

A faithful Plexus port of the hydrodynamic feeding model in

> Jingyi Liu, John H. Costello, Eva Kanso,
> **"Flow physics of nutrient transport drives functional design of ciliates"**,
> *Nature Communications* **16**, 4154 (2025). doi:[10.1038/s41467-025-59413-x](https://doi.org/10.1038/s41467-025-59413-x)
> (`papers/Flow-Physics-drives-functional-design-of-microswimmers/`)

A single-celled ciliate is a **sphere** whose beating cilia impose a tangential
**slip velocity** ("wavy surface velocities") on its surface. That slip drives an
**analytical Stokes flow** (the squirmer multipole series) through the surrounding
**water**, which advects a **nutrient concentration field** toward a localized
absorbing **mouth**. The paper's question is *where to put the mouth* — and it
answers it with the dimensionless feeding flux (Sherwood number).

This prototype reproduces that physics as **one more spec over the same registry**
the `water/`, `ant/` and maze sims use — validated by the same
`scenario_schema.py`. Nothing in the generic engine changed; the simulation is one
new field type (an *analytic* field) and four operators.

---

## 1. The categorical question (do this first)

The user's instinct is right: before any code, decide **what each piece of the
physics *is*** in the sets / fields / operators algebra. Every quantity answers
exactly one of Plexus's orthogonal questions (`plexus.tex` §"Categorical
discipline").

| Physics | Plexus category | Why | Home |
|---|---|---|---|
| the ciliate body (a swimming sphere) | **Set** `organism` (top-level $S$) | a *kind of thing* with its own state (pose, slip modes) | `sets:` |
| the beating surface / wavy slip | **Set** `surface_node` (`role: membrane`, `parent: organism`) | a *different kind* (lives on the sphere, carries slip) → a new sort, not a type | `sets:` |
| mouth (absorbing cap) vs bare cilia | **Type** within `surface_node` | same state + operators, differ only in *whether they absorb* → the $\tau$-colouring | `types:` (geometric) |
| sessile vs motile life strategy | **Type** within `organism` (`lifestyle`) | same state, different *parameter* (swim on/off, flow basis) | `types:` |
| the analytical Stokes flow $\mathbf u(\mathbf r)$ | **Field** `flow` (vector, *analytic*) | a continuum on $\Omega$ coupled to a set — but its update is a closed form, not a PDE | `fields:` |
| the nutrient concentration $c$ | **Field** `chemical` (advection–diffusion) | a continuum with its own PDE | `fields:` |
| evaluate the squirmer solution on the grid | **Exchange** `squirmer_flow` (organism → flow) | scatter object state into a field — but the "kernel" is the Stokes solution | operator |
| place nodes on the sphere + set their slip | **Broadcast** `slip` (organism → surface_node) | lift a parent decision to its children along $\pi$ | operator |
| swim at $U=\tfrac23B_1$ | **Lateral** `swim` (organism self-propel) | within-set self-propulsion (1st-order, overdamped Stokes) | operator |
| mouth uptake (the feeding) | **Exchange** `absorb` (surface_node[mouth] ← chemical) | gather field onto objects + accumulate the objective | operator |
| advect + diffuse the nutrient (Péclet) | **builtin** `chemical.diffuse` | the field's own PDE; reads `flow` | `schedule:` |
| food parcels carried by the fluid | **Set** `tracer` (passive particles) | a new sort, advected by `flow` | `sets:` |
| a parcel swept into the fluid | **Exchange** `advect_particles` (tracer ← flow) | gather the fluid velocity onto the parcel | operator |
| a parcel eaten at the mouth | **Structural** `capture` (the Die/eat primitive) | membership change (occupancy → 0) + count | operator |

The flow field, the swim speed, and the mouth coverage are exactly the
inverse-problem targets — and they live on the **operator** / **type**, never on a
set's geometry. *Thing → set; process → operator.*

### The one genuinely new idea: an *analytic* field

The single non-obvious call is the **flow field**. It is unmistakably a `Field`
(a continuum on $\Omega$, coupled to the organism, sampled by the chemical's
advection) — but it does **not diffuse**. Its value is a *closed-form function of
the organism's pose and slip modes*, recomputed every tick. That is the field
analogue of the paper's existing `navigation: geodesic` (a field whose value is
*computed*, not *integrated*). So:

* `FlowField.step()` is a **no-op** (it is never integrated);
* the `squirmer_flow` Exchange operator **rewrites** it each tick from the organism.

This is the only thing the simulation forces the vocabulary to grow by, and it
grows *monotonically* — `flow` is just a new `@register_field`, every earlier spec
still runs.

---

## 2. What's here

```
squirmer.py          analytic squirmer flow + slip-mode design (port of Visual_flow.m)
swimmer_fields.py    FlowField (analytic vector) + ChemField (advect-diffuse, axisymmetric)
ops_swim.py          squirmer_flow / slip / swim / absorb           (the four core operators)
                     + advect_particles / capture                  (food parcels eaten)
swimmer_engine.py    generic engine: build sets + fields, run the schedule
sessile.yaml         a 20%-mouth attached ciliate (body frame)
motile.yaml          a swimming ciliate (lab frame, depleted wake)
feeding.yaml         a ciliate in a current eating drifting food parcels (phagotrophy)
render_swim.py       flow panels + concentration gifs + feeding gif (mode: flow|conc|feed)
viz_swim.py          flow-visualization gallery (PIV / pathlines / streak / quiver /
                     vorticity / LIC / metachronal slip wave / dye) -> viz_gallery.png
run_swim.py          the exhaustive suite + Sherwood curves
results.md           auto-written metrics
```

Run:
```bash
cd prototype/microswimmer
PYTHONPATH=../../src python render_swim.py flow sessile motile   # flow fields
PYTHONPATH=../../src python render_swim.py conc sessile          # concentration gif
PYTHONPATH=../../src python run_swim.py                          # full suite
```

## 3. The exhaustive test families (mirroring `water/`)

| family | sweeps | intent check |
|---|---|---|
| `flow_*` | sessile / motile-body / motile-lab | analytic streamlines match `Visual_flow.m` |
| `pe_*` | Péclet 0…200 | boundary layer thins, **Sh rises monotonically** |
| `cover_*` | mouth coverage 5…100 % | **Sh vs feeding area** (paper Fig. 4) |
| `place_*` | mouth front / side / rear | uptake depends on placement vs the feeding current |
| `swim_*` | swim speed | a motile cell sweeps a symmetric depleted wake |

**Sherwood number** $\mathrm{Sh}=\dfrac{\text{uptake rate}(Pe)}{\text{uptake rate}(Pe{=}0)}$ —
the advective feeding enhancement over pure diffusion, the paper's efficiency metric.

### Flow-visualization gallery (`viz_swim.py` → `viz_gallery.png`)

Because the flow is analytic, a rich gallery of flow viz needs no solver — one
squirmer eval + numpy tracer integration. 21 gifs across 8 styles:

| style | what it shows |
|---|---|
| `piv_*` | PIV-style tracer streaks (motion-blur) revealing the feeding current |
| `pathlines_*` | integrated pathlines coloured by speed (incl. flow-past-a-cell) |
| `streak_*` | streaklines from an upstream rake in an ambient current |
| `quiver_*` | animated velocity arrows coloured by \|u\| (swimming cell) |
| `vorticity_*` | curl of the flow — the **puller vs pusher** counter-rotating lobes |
| `lic_*` | animated line-integral-convolution "smoke" texture along streamlines |
| `slipwave_*` | the **wavy surface velocity**: tangential slip with a travelling metachronal wave |
| `dye_*` | a passive dye blob advected + stretched by the feeding current |

Each is sessile / motile-body / motile-lab where meaningful; `vorticity_puller`
(small cap) vs `vorticity_pusher` (large cap) contrast the two swimming gaits.

### Aligning with the authors' viz (axisymmetry)

The paper never solves the full 2-D field: `Visual_concentration.m` solves the
axisymmetric half-plane $(r,\theta)$ and **mirrors** it (`x_extend=[x,-x];
c_extend=[c,c]`), and `recover_full_u` mirrors the flow. A full-grid solver
accrues asymmetric numerical error, so `ChemField` **re-imposes the axisymmetry**
each tick (average $c$ with its reflection about the swim axis $y=0.5$). The
motile wake is then mirror-symmetric, as the physics requires.

---

## 4. What it takes to implement this in the framework — the honest answer

**It takes one new field kind, four operators, and two small grammar extensions —
no engine change.** Concretely:

1. **A new sort** `surface_node` (`role: membrane`) hanging off `organism` by a
   parent map. Free: the containment graph already allows many child sorts.
2. **One new *field flavour*** — an **analytic field** whose `step()` is a no-op
   and whose value is written by an operator. The `Field` contract already
   supports this (`navigation: geodesic` is the precedent); we only added a
   vector-valued grid.
3. **Four operators**, one per category already in the algebra
   (Exchange ×2, Broadcast, Lateral). Each is ~30 lines, declares its
   `REQUIRES_*`, and honours its selector mask.
4. **The chemical field's PDE gained an advection term** that reads another field.
   Field PDE coefficients (diffusion, Péclet) stay *on the field*, consistent with
   how `diffusion`/`decay` already live there.

### Two genuine framework gaps this surfaced

* **Field-reads-field coupling.** Advection is *one field driven by another*
  (`chemical` advected by `flow`). Plexus's Exchange is set↔field; field↔field has
  no first-class operator. We bound `flow` into `ChemField` at build and let its
  `step()` read it. A clean framework would add a **field-coupling** declaration
  (`chemical: {advected_by: flow}`) the engine wires up, instead of an ad-hoc
  reference.
* **Geometric selectors.** The mouth is a *geometric* subpopulation (a polar cap),
  not a random fraction. The `types:` mechanism assigns membership by random
  `fraction`; we coloured the cap at build from `feeding_area` + angle. This is
  exactly the `cell[x<0.5]`-style **geometric selector** the paper flags as a
  natural grammar extension (`set[θ<θ_cap]`). Until it lands, geometric types are
  engine-assigned, not spec-expressible.

Everything else — the sphere, the wavy slip, swimming, the chemical field,
absorption, the schedule, validation, determinism, the gallery — is **just a spec**
over machinery that already existed.
