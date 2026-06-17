# Ant colony in Plexus — what it takes to implement Lague's *Ant-Simulation*

This prototype ports Sebastian Lague's
[**Ant-Simulation**](../../papers/Ant-Simulation) (`papers/Ant-Simulation`, Unity)
into the Plexus framework defined in [`paper/plexus.tex`](../../paper/plexus.tex):
the *sets + fields + operators* container, the *spec → registry → engine* contract,
and the categorical separation of operators (Lateral / Aggregate / Broadcast /
Exchange / Rewire / Structural).

It is **self-contained** in `prototype/ant/` (specs, render driver, suite, gifs) and
reuses the existing engine + operator registry. It does **not** build on the
removed `forage` prototype.

---

## The headline: it takes *one* new operator + a spec

Lague's ant model is, in framework terms:

> two diffusing pheromone fields, a carrying-state machine, and three-sensor
> trail-following — where **which** field an ant follows/deposits depends on its
> state (searching vs. carrying food).

Mapping that onto the existing Plexus vocabulary, **everything except the
state machine already existed**. The only new code is one registered operator,
[`ops/colony.py`](../ops/colony.py) (~110 lines). This is exactly the framework's
Part III claim — *new behaviour is one new operator + a spec, the engine
untouched* — exercised once more.

### Categorical map (Lague → Plexus)

| Lague `Ant.cs` concept            | Plexus construct                                   | new code? |
|-----------------------------------|----------------------------------------------------|-----------|
| ant agent                         | a `cell` (a small disc of `mpm` `particle`s)       | reused    |
| forward motion + random steer     | `motility` (Lateral) — heading + rotational noise  | reused    |
| 3-sensor pheromone steering       | `trail` (Exchange) — Physarum L/C/R sense + turn   | reused    |
| deposit pheromone (recency-faded) | `secrete` (Exchange) with `runout` (reads clock)   | reused    |
| home / food pheromone maps        | two `grid` fields (`home_pher`, `food_pher`)       | reused    |
| walls / obstacles                 | one obstacle mask, reused as **MPM + field BC**    | reused    |
| **State: Searching ↔ Returning**  | **`colony` (Lateral): pickup/dropoff, perception** | **NEW**   |
| "only state X does Y"             | the `cell[loaded=0/1]` **selector** (engine native)| reused    |

The dual-pheromone rule is then *pure spec* — a selector on each `trail`/`secrete`:

```
searching ant  (loaded=0):  secrete -> home_pher,  trail <- food_pher
returning ant  (loaded=1):  secrete -> food_pher,  trail <- home_pher
```

### Why the pheromones form the right gradients (the recruitment mechanism)

`secrete(runout=R)` fades a marker over `R` ticks since the ant last left a goal
(`colony` maintains the `cell.t_since_goal` clock). So **home pheromone is densest
at the nest** (laid strongest just after leaving it) and **food pheromone is
densest at the food**. `trail` climbs toward the strongest sensor, so a searching
ant climbs food pheromone *to the food* and a returning ant climbs home pheromone
*to the nest*. No global path-finding — the gradient is built by the colony itself
(stigmergy).

---

## What did NOT need new code

* **Engine** — untouched except one generic line: record *all* fields (multi-field
  sims need both pheromones), not just the first. No ant-specific branch.
* **Schema / validator** — untouched. `colony` declares `REQUIRES_PARAMS=[home,food]`,
  so a spec missing them fails up front like any other operator.
* **Selectors** — `cell[loaded=0/1]` was already resolved by the engine; `colony`
  just sets `cell.loaded`.
* **Walls / food / home** — no pathfinding code: walls are the existing obstacle
  mask (MPM zero-velocity interior + field no-flux); food/home are circles the
  `colony` operator reads.

## Categorical placement (thing → set, process → operator)

Following the paper's *Categorical discipline* (§"what belongs where"), every key
in an ant spec sits in exactly one category:

| key | category | home |
|---|---|---|
| `boundary`, `obstacles` | **World** — the shared space | top level |
| `n`, `start`, `types: {ant: {youngs}}` | **Set** — what a node *is* | `sets:` |
| `youngs` (per-type stiffness) | **Set** type prop, *read by* `mpm` | `sets.cell.types` |
| `speed/rot`, `turn/sensor_*`, `rate/runout`, MPM knobs | **Operator** — the process | `operators:` |
| `colony.home`, `colony.food`, `perception`, `food_stock` | **Operator** — the foraging *task* | `operators:` |
| order / multi-rate firing | **Schedule** | `schedule:` |
| pheromone colours, ant colours | **Style** (render only) | `render_ant.py` |

The one judgement call is the `home`/`food` **goal regions** on the `colony`
operator. They are *spatial*, which tempts the World category, but they define the
foraging **process's** targets (where it picks up / drops off) — so they live with
the operator that consumes them, exactly mirroring the existing convention that a
field's `source:` region lives on the *field*, not the world. `youngs` is the
textbook correct case: a per-*type* law parameter owned by the set's `types:` and
merely *read* by `mpm` along the containment chain.

## The one new operator (`colony`)

A Lateral operator @ `cell` that, each tick:
1. **pick up** — a searching ant inside a food disc → `loaded=1`, turn around,
   (optionally) debit that source's finite stock, reset the recency clock;
2. **drop off** — a returning ant inside the home disc → `loaded=0`, turn around,
   `H.food_delivered += 1` (the colony objective), reset the clock;
3. **perceive** — within a short `perception` radius, steer heading straight at the
   target (Lague's `OverlapCircle`), so the last leg is reliable while long-range
   navigation stays emergent;
4. advance `cell.t_since_goal` (the recency clock `secrete` reads).

It mutates state and returns `{}` (no acceleration) — uniform with every other
structure-touching operator.

---

## Running it

```bash
cd prototype/ant
PYTHONPATH=../../src python render_ant.py ant_colony     # one scenario -> gif + montage
PYTHONPATH=../../src python suite.py                     # the full suite (resumable)
```

* `specs/*.yaml` — one self-contained spec per variant (readable, diffable).
* `<name>.gif`, `<name>_montage.png` — the rollout (food pheromone red, home
  pheromone blue, searching ants pale, carrying ants yellow; nest = orange ring,
  food = green rings, walls = grey).
* `results.md`, `gallery.png` — the suite table (delivered + trail coverage +
  intent check) and thumbnail grid.

## The suite (exhaustive, like `water/`)

Twelve variants over the **same registry**, only spec parameters changed —
spanning food layout, diffusion/decay, colony size, sensor reach, food depletion,
and walls, plus a control:

| variant | what it probes |
|---|---|
| `ant_colony` | baseline: central nest, three food discs |
| `ant_highway` | corner nest → one far food: a single reinforced trail |
| `ant_many_food` | five sources: several coexisting trails |
| `ant_sharp` | low diffusion + decay → crisp, long-lived highways |
| `ant_volatile` | high decay → trails evaporate; constant re-recruitment |
| `ant_big_colony` | 240 ants → faster discovery, denser network |
| `ant_far_food` | far source → fragile recruitment (honest about regime) |
| `ant_wide_sensor` | long reach + wide cone → exploratory, fuzzy trails |
| `ant_depleting` | finite food stock → trails shift as sources empty |
| `ant_wall` | barrier + one gap → trail routes through (walls as BC) |
| `ant_maze` | barrier + two gaps → colony commits to a route |
| `ant_no_trail` | **ablation**: trail off → recruitment fails (the control) |

**Intent check** (it ran ≠ it worked): a variant *works* iff `delivered > 0`.

**Honesty about regime (what the numbers do and don't show).** The runs are
single-seed and the bottleneck is the *stochastic first discovery* of food. Once
recruitment ignites it is vigorous — `ant_colony_s1`=213, `ant_many_food`=191,
`ant_food1`=126 delivered — but the scalar `delivered` for any one variant is
dominated by discovery luck, not the swept parameter (the three baseline seeds give
17 / 213 / 97). Two consequences stated plainly rather than hidden:

* The per-parameter *signal* is the **trail morphology in the gifs** (crisp vs.
  blurry, tight vs. loose, one highway vs. many), not the noisy delivered count.
* In this easy central arena the `ant_no_trail` ablation still delivers (21) by
  chance + short-range perception, so it is **not** a clean on/off control —
  trail-following *multiplies* throughput (~10× at a lucky seed) rather than being
  strictly necessary for any delivery. A clean ablation needs a hard layout where
  random discovery is rare (e.g. food behind a wall), which is also where a single
  seed often fails to ignite at all (`ant_wall`, `ant_maze` = 0).
