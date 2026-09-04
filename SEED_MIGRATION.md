# SEED_MIGRATION -- what the deprecation warning was asking for, and who cannot give it

`schema.load` has warned for months:

    [warn] deprecated: ['mesh_seed'] declared in operators: with kind="seed"
           -- move to the seed: section (see SEED_MIGRATION.md)

**and `SEED_MIGRATION.md` did not exist.** This is that file, written when the migration was
actually run on 4 September.

---

## The two spellings

A seed establishes $x_0$. It runs **once**, before the trajectory exists.

```yaml
# THE INTENDED SPELLING -- a top-level section, and the op is in NEITHER operators: NOR schedule:
seed:
  - {op: mesh_seed, at: vertex, cell_set: cell, n_cells: 200, radius: 5.0}
operators:
  - {op: cell_geometry, at: cell}
schedule: [cell_geometry]
```

```yaml
# THE LEGACY SPELLING -- inside the tick loop, suppressed after frame 0 by a window
operators:
  - {op: mesh_seed, at: vertex, before_frame: 1, cell_set: cell, n_cells: 200, radius: 5.0}
  - {op: cell_geometry, at: cell}
schedule: [mesh_seed, cell_geometry]
```

`Spec.seed_ops` puts it in one line: *"the `seed:` section (x_0), **NOT a schedule**"*. The legacy
form runs the seed through the engine's seed-window inside `_run`; the modern one runs it in
`engine.seed()` before the loop.

---

## It is byte-identical, and that was measured rather than argued

Every seed operator declares `MAY_MUTATE_INTEGRATED_STATE`, writes the level's state directly and
returns `{}`. So `engine.seed()` has **no delta to integrate**, and "run it before tick 0" is the
same instruction sequence as "run it first *inside* tick 0" -- which is where it already was, since
the seed is the first schedule entry in 1,503 of the 1,512 legacy specs.

Measured on `gate_00_spheroid` at 8 frames, 25,584 vertices, with `junction_myosin`,
`cell_mechanics`, `edge_flip`, `cell_divide` and `junction_sync` in the loop:

| comparison | `max abs delta` on vertex positions |
|---|---|
| **migrated vs legacy** | **0.000e+00** |
| `mesh_seed.jitter` 0.18 -> 0.180001 | 1.107e-03 |
| `cell_mechanics.Lambda` 3.0 -> 3.000001 | 3.268e-03 |

690,768 floats compared. **The two positive controls are the point**: a perturbation in the sixth
decimal of a seed parameter moves the trajectory by 1.1e-3, so the comparison detects changes four
orders of magnitude smaller than the one being tested. A zero from a harness that has not been shown
to produce a non-zero is not evidence.

---

## What was migrated, and what was refused

`python tools/migrate_seed_section.py --dry-run` / `--apply`, over `config/**/*.yaml`:

| outcome | count |
|---|---|
| migrated | **191** |
| already on `seed:` | 23 |
| no seed operator | 429 |
| **refused** | **1,413** |
| unreadable | 3 |

### The 1,399 that cannot migrate: `before_frame: 3`

`seed_cell_chem` / `cell_chem_seed` is declared with a **three-frame window** in 1,399 specs, so it
re-runs on ticks 0, 1 and 2. **A `seed:` block has no window to put that in.** The deprecation
warning asks these specs to move somewhere that cannot express what they do.

And the window is load-bearing, not vestigial. The operator's own docstring records why `mode: tip`
was removed: *"re-applying it every frame overwrites BOTH chemistry channels, so no operator that
writes to `chem` can accumulate anything."* Three frames of re-seeding therefore **suppress
`cell_chem_diffuse` and `cell_chem_react` for the first three frames** of every one of those runs.
Whether that is intended or a template copied 1,399 times is a real question, and it is not this
migration's to answer: moving them would change the trajectory, so the tool refuses.

### The 14 that cannot migrate: ordering

Eleven specs run `ecm_seed` **after** `aggregate` in the schedule and three run `cell_chem_seed`
after `cell_geometry`. Moving the seed to `seed:` would run it earlier and reorder the tick.
`config/gates/gate_02_ecm_block.yaml` is one of them, so **one of the four gates stays on the legacy
spelling** -- declared here rather than quietly migrated.

---

## The tool

`tools/migrate_seed_section.py`. Three things about it are deliberate:

* **The seed names come from the registry**, not a literal list -- `{n for n, c in REG.items() if
  c.KIND == "seed"}`. A hard-coded list is how the per-face carry came to miss `myo_med`.
* **It refuses rather than migrates** anything with a window wider than one frame, a seed absent
  from the schedule, or a seed that runs after a non-seed. Every refusal prints its reason.
* **It edits in place through `ruamel.yaml`**, and deletes list items with `.remove()` rather than
  rebuilding the list. Rebuilding assigns a plain python list, and ruamel keeps a `CommentedSeq`'s
  comments in a side table keyed by item **index** -- the first version dropped
  `gate_00_spheroid.yaml` from 94 comment lines to 77, and the seventeen lost were the ones carrying
  the argument: why `junction_myosin` precedes `cell_mechanics`, why `lam`/`gam` are 0.0, why
  `max_flips: 30` is a rail. In this repo the comments are the evidence.

## What did not move

All three frozen gate hashes are unchanged -- `00_spheroid` `c648b915791284fa`, `01_junction`
`09278b16dc861705`, `02_ecm_block` `78ff4878e7afe8f7` -- because the migration touches `operators:`
and `schedule:` and never the `_gate:` block that `--freeze-reference` hashes.
