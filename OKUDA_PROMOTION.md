# OKUDA_PROMOTION -- the 50 operators, where they go, and what is redundant

Phase A of the promotion plan. Every registered okuda operator, its contract, its target module in
`src/plexus/operators/`, and a verdict. The table is GENERATED FROM THE REGISTRY, not typed: the
family / kind / set / implementation columns are what `register_operator` actually recorded, so a
row cannot drift from the class it describes. Regenerate with the snippet at the bottom.

    50 canonical names (52 with the two aliases) across 16 files  ->  6 modules
    53 core names across 45 one-operator files                    ->  5 modules  (Phase E)

## Why a module and not a file per operator

`src/plexus/operators/` today is 45 files holding 43 names, which is a directory nobody can read: to
find out what acts on an MPM particle you open eight files, and the eight cannot be compared because
they are never on the screen together. The two implementations of `cell_mechanics` -- the AVM in
`mesh_ops.py` and the monolayer in `monolayer_ops.py` -- are the case in point: they are one
contract, and reading them side by side is how anyone can tell which one a spec is getting.

## The naming rule, and why the okuda names win

`paper/plexus2.tex`: an operator declares `family` / `kind` / `set` / `implementation`, and DIMENSION
IS AN IMPLEMENTATION CAPABILITY, not part of the name. So `cell_chem_diffuse`, not `graph_diffuse_3d`;
`seed_mesh`, not `seed_mesh_3d`. The okuda spellings already follow it and 461 specs use them.

Two names carry an alias, and both aliases are load-bearing rather than courtesy: the corpus is split
324/137 between `seed_mesh` and `mesh_seed` and 320/141 between `seed_cell_chem` and
`cell_chem_seed`, because the specs were migrated to the post-seed-refactor `seed_<noun>` convention
and the classes were not. Until 30e70e5e the larger half of the corpus did not load at all.

| target module | okuda file | operator | family | kind | set | implementations | alias |
|---|---|---|---|---|---|---|---|
| `contact_ops.py` | `bm_sense_ops.py` | **bm_sense** | signalling | structural | vertex | -- | -- |
| `contact_ops.py` | `load_ops.py` | **ecm_gate_growth** | population | structural | vertex | -- | -- |
| `contact_ops.py` | `load_ops.py` | **ecm_load** | mechanics | structural | vertex | -- | -- |
| `contact_ops.py` | `mesh_contact_ops.py` | **mesh_contact** | boundary | lateral | particle | -- | -- |
| `contact_ops.py` | `mesh_contact_ops.py` | **mesh_inside** | hierarchy | lateral | particle | -- | -- |
| `contact_ops.py` | `plate_ops.py` | **plate_confine** | boundary | structural | vertex | -- | -- |
| `contact_ops.py` | `surface_ops.py` | **surface_track** | hierarchy | structural | particle | -- | -- |
| `diffusion_reaction.py` | `chem_ops.py` | **cell_chem_diffuse** | fields | lateral | cell | graph_laplacian, interface_weighted | -- |
| `diffusion_reaction.py` | `chem_ops.py` | **cell_chem_react** | fields | lateral | cell | brusselator, gierer_meinhardt, gray_scott | -- |
| `diffusion_reaction.py` | `chem_ops.py` | **cell_geometry** | hierarchy | aggregate | cell | -- | -- |
| `diffusion_reaction.py` | `chem_ops.py` | **cell_grow** | population | structural | vertex | balance, default, sizer, timer | -- |
| `diffusion_reaction.py` | `chem_ops.py` | **cell_neighbours** | topology | rewire | cell | -- | -- |
| `diffusion_reaction.py` | `chem_ops.py` | **interface_push** | mechanics | lateral | vertex | -- | -- |
| `diffusion_reaction.py` | `chem_ops.py` | **interface_tension** | mechanics | lateral | vertex | -- | -- |
| `diffusion_reaction.py` | `chem_ops.py` | **seed_cell_chem** | seed | seed | cell | -- | cell_chem_seed |
| `diffusion_reaction.py` | `shape_chem_ops.py` | **cell_chem_from_shape** | fields | lateral | cell | apical_area, curvature, pressure, tension | -- |
| `diffusion_reaction.py` | `shape_probe_ops.py` | **cell_shape_probe** | hierarchy | lateral | cell | aspect, shape_index | -- |
| `ecm_ops.py` | `block_ops.py` | **block_seed** | seed | seed | particle | -- | -- |
| `ecm_ops.py` | `block_ops.py` | **block_stress** | hierarchy | lateral | particle | -- | -- |
| `ecm_ops.py` | `ecm_ops.py` | **cell_exclude** | boundary | structural | particle | -- | -- |
| `ecm_ops.py` | `ecm_ops.py` | **ecm_from_cell** | mechanics | lateral | particle | replay, sphere | -- |
| `ecm_ops.py` | `ecm_ops.py` | **ecm_seed** | seed | seed | particle | -- | -- |
| `ecm_ops.py` | `ecm_ops.py` | **ecm_stress** | hierarchy | lateral | particle | -- | -- |
| `junction_ops.py` | `junction_ops.py` | **junction_myosin** | mechanics | structural | vertex | default, two_pool | -- |
| `junction_ops.py` | `junction_ops.py` | **junction_sync** | mechanics | rewire | vertex | -- | -- |
| `junction_ops.py` | `medioapical_ops.py` | **cytokinetic_ring** | mechanics | structural | vertex | -- | -- |
| `junction_ops.py` | `medioapical_ops.py` | **medioapical_myosin** | mechanics | lateral | cell | -- | -- |
| `membrane_ops.py` | `integrin_ops.py` | **integrin_pull** | mechanics | lateral | particle | -- | -- |
| `membrane_ops.py` | `integrin_ops.py` | **integrin_seed** | seed | structural | particle | -- | -- |
| `membrane_ops.py` | `integrin_ops.py` | **integrin_track** | mechanics | structural | particle | -- | -- |
| `membrane_ops.py` | `membrane_ops.py` | **adhesion_pull** | mechanics | exchange | particle | -- | -- |
| `membrane_ops.py` | `membrane_ops.py` | **adhesion_seed** | seed | seed | particle | -- | -- |
| `membrane_ops.py` | `membrane_ops.py` | **adhesion_turnover** | topology | rewire | particle | -- | -- |
| `membrane_ops.py` | `membrane_ops.py` | **bm_bond** | mechanics | lateral | particle | -- | -- |
| `membrane_ops.py` | `membrane_ops.py` | **bm_contact** | boundary | lateral | particle | -- | -- |
| `membrane_ops.py` | `membrane_ops.py` | **bm_crosslink** | topology | rewire | particle | -- | -- |
| `membrane_ops.py` | `membrane_ops.py` | **bm_remodel** | population | lateral | particle | -- | -- |
| `membrane_ops.py` | `membrane_ops.py` | **bm_repel** | boundary | lateral | particle | -- | -- |
| `membrane_ops.py` | `membrane_ops.py` | **bm_secrete** | population | structural | particle | -- | -- |
| `membrane_ops.py` | `membrane_ops.py` | **bm_seed** | seed | seed | particle | -- | -- |
| `membrane_ops.py` | `membrane_ops.py` | **bm_strain** | mpm | lateral | particle | -- | -- |
| `membrane_ops.py` | `membrane_ops.py` | **bm_unbond** | topology | rewire | particle | -- | -- |
| `membrane_ops.py` | `membrane_ops.py` | **integrin_adhesion** | mechanics | lateral | particle | -- | -- |
| `membrane_ops.py` | `membrane_ops.py` | **mpm_boundary** | boundary | field | field | -- | -- |
| `vertex_ops.py` | `mesh_ops.py` | **cell_die** | population | structural | vertex | -- | -- |
| `vertex_ops.py` | `mesh_ops.py` | **cell_divide** | population | structural | vertex | default, doubler, timer | -- |
| `vertex_ops.py` | `mesh_ops.py` | **cell_mechanics** | mechanics | lateral | vertex | default, monolayer | -- |
| `vertex_ops.py` | `mesh_ops.py` | **seed_mesh** | seed | seed | vertex | -- | mesh_seed |
| `vertex_ops.py` | `mesh_ops.py` | **topo_record** | harness | structural | vertex | -- | -- |
| `vertex_ops.py` | `t1_ops.py` | **edge_flip** | topology | rewire | vertex | -- | -- |

50 canonical okuda operator names across 16 files -> 6 modules.

## Redundancy: the four merge candidates, settled by reading them

**`interface_tension` + `interface_push` -- DO NOT MERGE.** The plan listed these as two terms of one
energy written as two operators, which is true and is exactly why they must stay apart. They WERE one
operator, `rd_interface_tension`, carrying

    E = K_purse * sum_iface l_e   -   K_extrude * sum_red a*r
        [___ ordinary physics ___]     [_ the answer written into the objective _]

The second term does not model a force; it pays the tissue to produce the morphology the campaign was
searching for. One name over both cost four rounds: `K_extrude` measured 0.0 in all 78 specs that ever
carried the operator, so nothing was ever forced, and the Grounder still reported a round as
"the same extrude-forced star for a fourth round" on three runs whose specs contain no such operator.
The split on 10 August is the fix. Merging them back would restore the exact condition -- a reader
who sees a plausible name cannot check a term that is not in front of them.

**`ecm_stress` / `block_stress` -- MERGE (Phase D).** Same body: `|J-1|` (or the deviatoric or von
Mises variant) divided by a full-scale, banded into an integer palette index. They differ in two
defaults -- `scale` 0.15 for the matrix against 0.004 for the block, which is right, since the block
is ~130x stiffer and reads as uniformly unstrained at the matrix's scale -- and in WHICH MODULE-LEVEL
LIST they append to. That list is the whole reason the duplicate exists: `ecm_ops.STRESS_HISTORY` is a
single module global, so a second `ecm_stress` instance would interleave two sets' rows and the
renderer would colour each set with the other's numbers, silently. Move the history onto the Level it
belongs to and one operator serves both sets; `block_stress` survives as an alias so no spec changes.

**`ecm_seed` / `block_seed` -- DO NOT MERGE.** Not two spellings of one thing. `ecm_seed` fills the
box MINUS A CAVITY (a disc, anisotropic on purpose) with aligned fibres; `block_seed` fills two slabs
beyond a free gap with a jittered lattice. Same family, same module, different geometry. They look
alike only in that both rewrite every position once at frame 0, which is what `kind: seed` means.

**`integrin_adhesion` vs `integrin_pull` / `integrin_seed` / `integrin_track` -- DO NOT MERGE, and say
so in the module.** The shared prefix is the problem, not the code. `integrin_adhesion` tethers the
BASEMENT MEMBRANE to the EPITHELIAL SURFACE -- each particle keeps the direction `u0` it was seeded on
and is pulled back toward it, so a surface whose radius triples stretches the bonds by ~R, which is
the loading a real basement membrane feels under growth. The `integrin_*` trio is the MATRIX FIBRE
model: fibres seeded outward from the surface, each bound at its tip to the nearest membrane particle,
with the cell end prescribed. One is membrane->epithelium, the other matrix->membrane. Both are needed
and they are one hop apart in the same physical chain, so they go in the same module with that
sentence at the top of it.

## Carried over from `ops/AUDIT.md`: two operators are NOT promoted

**`mpm_boundary` -- rejected.** It overwrites grid-node velocity, so the constraint is kinematic:
momentum is not conserved and the reaction is discarded. Its standoff is set by the B-spline stencil
width rather than by anything physical -- sweeping `recover` 0 / 2 / 6 / 20 gave 46.6% / 3.8% /
11.5% / 13.9% of the sheet inside the tissue against standoffs +0.0006 / +0.0124 / +0.0088 / +0.0069,
never reaching the 0 -> +0.002 that would mean "just touching". `integrin_track` is the constraint it
should have been: it prescribes `n_fibres` PARTICLES and lets the sheet feel them through ordinary
MPM contact.

**`bm_strain` -- rejected.** AUDIT's verdict is "not a mechanism". It is promoted nowhere and its
name is not kept as an alias, because an alias would let a spec ask for it and get silence.

What replaces both is the explicit membrane<->matrix coupling, promoted WITH ITS RESOLUTION LIMIT IN
THE METADATA: at `n_grid 48`, `dx = 0.021` against a 0.002-thick sheet, so one grid cell holds ~16
membrane particles and the coupling strength was set by grid resolution, not by a measured adhesion.

## One contract, two implementations -- the collisions that are already handled

| contract | implementations | selected by |
|---|---|---|
| `cell_mechanics` | `default` (3D AVM, `mesh_ops`), `monolayer` (`monolayer_ops`) | `implementation:` on the op line |
| `junction_myosin` | `default` (per-junction), `two_pool` (`medioapical_ops`) | `tissue.py`'s `myo_model` |
| `cell_divide` | `default`, `doubler`, `timer` | `implementation:` |
| `cell_grow` | `default`, `balance`, `sizer`, `timer` | `implementation:` |
| `cell_chem_react` | `brusselator`, `gierer_meinhardt`, `gray_scott` | `implementation:` |
| `cell_chem_diffuse` | `graph_laplacian`, `interface_weighted` | `implementation:` |
| `cell_chem_from_shape` | `apical_area`, `curvature`, `pressure`, `tension` | `implementation:` |
| `cell_shape_probe` | `aspect`, `shape_index` | `implementation:` |
| `ecm_from_cell` | `replay`, `sphere` | `implementation:` |

Nine contracts carry 24 implementations between them. None needs merging: this IS the merge, in the
form the paper prescribes -- one name, one contract, several bodies.

## Open findings, recorded rather than guessed at

* **`morphogen_growth_3d`** is named by `config/okuda/mini_coral_nodilute.yaml` and registered
  nowhere. It is the last unresolved name in the corpus (460 of 461 specs load). It predates the
  `cell_grow` implementations and is most likely one of them, but "most likely" is not a migration;
  it is left failing loudly until someone who ran that spec says which.
* **`load_mesh_3d`** is emitted by `translate.py:210` and registered nowhere -- the same class of
  break as `seed_mesh` was, waiting for the first spec that uses the checkpoint path.
* **`integrin_seed`** declares `family="seed"` with `kind="structural"`, which the registry warns
  about on every import: a seed that masquerades as a dynamics kind skips the seed lifecycle
  guarantees. Fixed when `integrin_ops` moves, not before -- it is a behaviour change.
* **`topo_record` stores its history INSIDE the mesh** (`m.setdefault("hist", [])`), so the table the
  engine now owns grows without bound for the length of the run. Phase 0 step 5 moved the recording
  to the engine (`rec_mesh` + `MeshTable.snapshot`); the operator still dual-writes `hist`, and
  retires when its readers move over.
* **`config/material/material_cell_grow_aniso.yaml` and its sibling name `cell_grow` in a 2D spec.**
  They have never worked: `cell_grow` was not registered in core before this promotion, so they
  failed with *not in registry*, and now they fail with *supports dims [3], not dim=2* -- a better
  message for the same broken spec. They almost certainly mean `agent_grow`, the core's 2D
  growth operator, but "almost certainly" is not a migration and whoever wrote them should say.
* **`plexus.operators.<one-operator-module>` is imported by name from five prototype scripts** --
  `prototype/eye/muscle_ops.py`, three files under `prototype/cardio_cells/`, and
  `prototype/inverse_slime/operators.py` reach for `mpm_grid`, `deposit` and `diffuse`. That is why
  the core regrouping leaves re-export shims behind instead of deleting the files.

## Regenerating the table

```
cd discovery_okuda && PYTHONPATH=../src:ops:. python - <<'EOF'
import inspect, os
import plexus.operators, mesh_ops, chem_ops, t1_ops, monolayer_ops, shape_chem_ops, shape_probe_ops
import ecm_ops, membrane_ops, integrin_ops, load_ops, mesh_contact_ops, bm_sense_ops
import plate_ops, surface_ops, block_ops, junction_ops, medioapical_ops
from plexus.models.registry import _OPERATOR_REGISTRY as REG, _OP_CONTRACTS as CON
for n in sorted(REG):
    c = REG[n]
    if "/src/plexus/" in (inspect.getsourcefile(c) or ""): continue
    print(n, getattr(c,"FAMILY",None), getattr(c,"KIND",None), getattr(c,"SET",None),
          sorted(CON[n].implementations))
EOF
```
