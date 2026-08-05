# The biological hierarchy, and the operators on it

Written at the level asked for -- **ecm / cells / junction / basement_membrane** -- and no finer. The
finer decomposition (cortex, cytoplasm, nucleus; laminin vs collagen IV as separate sets) is a real
direction but it is one level down, and every set added is a set whose state has to survive T1s,
division and death.

The correction that produced this file: **the basement membrane is not the ECM, it is a specialised
ECM.** An earlier version of these comments called `mpm_particle` "the ECM", which put the basement
membrane outside the thing it is a part of.

```
ECM                              extracellular matrix -- the parent of both layers below
├── basement_membrane            a SPECIALISED ECM sheet, on the epithelium's basal side
│                                laminin network + collagen IV network, linked by nidogen + perlecan
│                                thin (~100-300 nm), stiff, crosslinked; the load-bearing layer
└── interstitial_ecm             collagen I/III, fibronectin, elastin, hyaluronan
                                 bulk, soft, fibrous -- what this prototype has always modelled

EPITHELIUM
├── cells                        the sheet's mechanical degrees of freedom
└── junction                     cell-cell contacts (adherens / tight / desmosome / gap)
                                 carries MYOSIN; an internal relation of the epithelium,
                                 not a free-standing body

FIELD
└── mpm_grid                     the shared exchange grid. Not decoration: every particle body
                                 scatters into and gathers from this one grid, and that sharing
                                 IS the mechanical coupling between them.
```

Reading order, outside in: `interstitial_ecm` -> `basement_membrane` -> `junction` -> `cells`, with
the epithelium's basal face against the membrane. Our topology is a gland/acinus (basal outward), so
the membrane is on the OUTER surface with the interstitial matrix beyond it.

## How that maps onto what is actually implemented

| hierarchy entity | set in the spec | status |
|---|---|---|
| `cells` | `vertex` (+ `cell` for per-cell state) | 3D AVM, real |
| `junction` | a **relation** on the vertex mesh, not a declared Level | myosin implemented, keyed by vertex pair |
| `basement_membrane` | `basement_membrane_particle` | MPM + explicit bonds, real |
| `interstitial_ecm` | `mpm_particle` | MPM fibres, real. **Misnamed** -- it is the interstitial matrix, not "the ECM" |
| `mpm_grid` | field | real, shared by all particle bodies |

Two honesty notes about the spec, because the YAML does not say them:

* **`parent: cell` on the particle sets is not biology.** MPM's provision needs a parent to hang
  `per_parent` counts off. Cells *secrete* the basement membrane and *adhere* to it
  (`integrin_adhesion`); neither matrix layer is a child of a cell.
* **`junction` is not a Level.** The half-edge table already exists on the mesh, and myosin is keyed
  by the unordered vertex pair. Promoting it to a Level would buy per-junction provisioning and cost a
  reconciliation operator for every topology change; the keying gets the same result for free.

## Operators, by the entity they act on

Status: **[x]** implemented and tested, **[~]** implemented, not yet validated, **[ ]** designed only.

```
cells / vertex
  [x] seed_mesh_3d              build the epithelial shell
  [x] cell_geometry_3d          per-cell area / perimeter / volume from the mesh
  [x] shape_energy_3d           the 3D AVM: area, perimeter, volume, line tension, radial
  [x] morphogen_growth_3d       per-cell target growth
  [x] divide_3d                 division on volume doubling                      (structural)
  [x] reconnect_t1_3d           T1 rearrangement                                (rewire)
  [x] topo_snapshot_3d          record the mesh for analysis and rendering
  [x] ecm_growth_gate_3d        matrix stress suppresses the cell cycle
  [~] ecm_load_3d               matrix pressure pushes the surface inward
  [ ] cell_extrude_3d           death / extrusion                               (structural)
  [ ] cell_polarity             apical-basal axis -- currently ABSENT, which is why the
                                membrane has to be placed by hand rather than by polarity

junction
  [x] junction_myosin           per-junction myosin, recruited by tension; survives T1 /
                                division / death by topological keying
  [ ] junction_turnover         adherens-junction assembly / disassembly
  [ ] junction_rupture          junction failure -> a gap                       (rewire)

basement_membrane
  [x] basement_membrane_seed    lay the sheet on the epithelium's basal surface
  [x] basement_membrane_bond    crosslinks: the collagen IV network
  [x] basement_membrane_bond_break   fragmentation                              (rewire)
  [~] integrin_adhesion         anchor the sheet to the epithelium. WITHOUT THIS THE SHEET SLIDES
                                and never stretches -- see the note below
  [ ] basement_membrane_stiffen collagen IV deposition -> stiffening
  [ ] basement_membrane_degrade MMP proteolysis -> local softening

interstitial_ecm
  [x] ecm_seed                  fibres, with alignment and a density field
  [x] ecm_stress                strain / von Mises colouring
  [x] cell_to_ecm               the epithelium's contact force on the matrix
  [x] cell_exclude_3d           non-penetration                                 (structural)
  [ ] ecm_crosslink             connectivity for the interstitial matrix too
  [ ] ecm_remodel               fibre realignment under load

field / coupling
  [x] mpm_strain, mpm_scatter, mpm_gather, mpm_grid_update
  [x] mpm_scatter[accumulate]   the second and third bodies into the shared grid
  [ ] basement_to_ecm           an explicit membrane-to-matrix link (collagen VII);
                                today they couple only through the grid
```

## The gap that matters most

`cell_polarity` is absent, and everything about the membrane's placement currently depends on knowing
which side is basal by hand. Worse, the epithelium has no basal side to know about: cells are
**wedges from the sphere centre** (`tyssue_ops3d.face_geometry_3d`), and the monolayer prism in the
movies is drawn by `_draw`, not simulated. "apical (outer) / basal (inner)" is a rendering label.

`integrin_adhesion` is marked `[~]` and not `[x]` deliberately. The test battery caught both states it
has been in:

* **without it**, mean bond strain stayed 0.0000 for the whole run at every bond stiffness -- the sheet
  slid over the epithelium instead of being stretched, so `59`/`60`'s fragmentation numbers describe the
  wrong loading path and are discarded;
* **with it at k = 2e4**, 69,428 of 70,129 bonds broke within 40 frames at a strain of 0.95 -- the anchor
  is far stiffer than the crosslinks, so it tears the sheet apart before the tissue has grown into it.

The physics needs `k_adhesion` and `k_bond` to be comparable, and neither number is yet calibrated
against the other. Until they are, no fragmentation result from this membrane means anything.
