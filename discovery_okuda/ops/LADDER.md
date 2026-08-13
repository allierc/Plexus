# The continuum ladder, 88-107

**One step per folder, one change per step.** Nominal plan; adapted as results come in.

## Why the spring line stopped

Runs 82-87 and the deleted 88-97 chased the holes with crosslink springs, excluded volume, and the
`attraction_repulsion` law. Best was 86 at `d/hex 0.733`. Across all 16 runs,
`corr(d/hex, mean_degree_z) = -0.68`: **every mechanism that improved packing did so by breaking
crosslinks**, with no counterexample. Best packing with an intact network (`z >= 6`) was 0.581.

That is a defect of the *object*, not of the tuning. Holes, `lcc` and `z` are properties of a bond
network. An MPM continuum has none — particles are quadrature points, the response comes from the
deformation gradient through the grid, and a gap between particles is not a gap in the material. The
metrics stop being meaningful rather than becoming good.

## The one real risk

The sheet is 0.002 thick; the grid cell is `1/64 = 0.0156`. The BM is **1/8 of a cell**
through-thickness, so the grid smears it over ~8x its thickness. In-plane it is well resolved
(spacing 0.0015 at `reserve=0`, ~10 particles per cell edge). **Steps 89 and 92 exist to measure how
much of the baseline is this**, before anything is built on top of it.

Consequence to watch: MPM separates material that thins past the grid's support, so "tearing" may be
set by `dx` rather than by stress. A tear that moves when the grid is refined is numerical.

---

## M1 — does the continuum hold, and grow with the spheroid? (88-92)

| # | name | change | question |
|---|---|---|---|
| 88 | `mpm_bare` | no bonds, no anchor, no secretion | does an MPM shell survive 402 frames and get pushed outward at all? |
| 89 | `mpm_anchor` | integrin anchor ON | **give the sheet a FORCE.** See 88's result below |
| 90 | `grid128` | `n_grid` 64 -> 128 | how much of the baseline is resolution? |
| 91 | `stiff10` | `membrane_youngs` x10 | does stiffness change the outcome or only the timescale? |
| 92 | `thick3` | thickness 0.002 -> 0.006 | a shell the grid can actually resolve |

### 88 result: the sheet grows with the spheroid, but is mechanically INERT

`R: 0.0875 -> 0.2985`, tracking the tissue exactly, `coverage 1.0`, sheet intact. So the geometric
half of milestone 1 passes. The mechanical half does not: `sigma_max(F) - 1` reaches only **7e-4**
against a true in-plane stretch of ~3.4x.

The cause is in the log -- **18,134 membrane particles per frame projected by `cell_exclude`**.
That is a hard positional constraint: it repositions particles without touching `F`, so the sheet is
carried outward as a decal and the material never learns it was stretched. A body at zero strain
cannot tear, cannot resist growth, and cannot load the stroma, so 90-92 would have measured nothing --
stiffness is irrelevant to a body carrying no stress.

The membrane receives no force at all. `ecm_from_cell`, the stroma's soft push, is an analytic growing
sphere bound to `mpm_particle`; its geometry does not match the vertex tissue, so it cannot simply be
repointed. The one operator coupling the membrane to the real surface mechanically is
`integrin_adhesion` -- it emits a force, which becomes grid momentum, which is what makes `C` non-zero
and `F` integrate. **So the anchor is not milestone 3; it is the prerequisite for milestone 1**, and
M3 below becomes about anchor STRENGTH and turnover rather than about whether to have one.

**M1 passes if** the sheet stays a shell (`coverage` high, on-shell fraction ~100%) and the strain
rises as the spheroid grows. If it vanishes or passes through, the continuum route fails here.

## M2 — material supply, and the breaking point (93-98)

| # | name | change | question |
|---|---|---|---|
| 93 | `secrete` | secretion on, nominal rate | does added material keep the sheet intact? |
| 94 | `secrete_fast` | rate x3 | is the sheet supply-limited? |
| 95 | `starved` | rate /3 | the tear demo: too little material |
| 96 | `no_secrete_long` | none, full growth | **locate the breaking point**: stretch at which it tears |
| 97 | `visco_slow` | viscoelastic, long `tau` | does relaxing shear replace secretion? |
| 98 | `visco_fast` | viscoelastic, short `tau` | as above, faster |

**The demo the whole ladder is for:** 95/96 tear, 93 does not. Tearing must be shown to be
stress-driven, not grid-driven — cross-check the tear onset against 89's refinement.

## M3 — anchoring (99-102)

| # | name | change | question |
|---|---|---|---|
| 99 | `anchor` | integrin on, nominal | does the sheet stop sliding? does strain rise? |
| 100 | `anchor_stiff` | `k_adh` x5 | does a stiff anchor tear the sheet instead? |
| 101 | `anchor_turnover` | `tau_adh` finite | focal adhesions turn over in minutes |
| 102 | `anchor_secrete` | best of M2 + anchor | the combined working point |

Note: with no anchor the sheet is held only by normal contact through the grid. An isotropically
growing sphere gives no tangential drive, so 88 should be stable without one — 99 tests whether the
anchor is needed at all, rather than assuming it.

## M4 — the three-way interaction (103-107)

| # | name | change | question |
|---|---|---|---|
| 103 | `bm_on_tissue` | BM present vs absent | **BM -> spheroid**: does the sheet constrain growth (aspect, volume)? |
| 104 | `no_stroma` | ECM absent | **BM <-> ECM**: isolate the stroma's contribution |
| 105 | `stroma_stiff` | ECM `youngs` x10 | does a stiff stroma load the BM? |
| 106 | `stroma_soft` | ECM `youngs` /10 | other side |
| 107 | `nominal` | anchor + secretion + stroma | the figure run |

ECM<->BM needs no new code: both sets already scatter into `mpm_grid` with `implementation:
accumulate`, so the coupling exists the moment the springs stop masking it. 104-106 measure it.

---

## What each step must report

`coverage` (fraction of 16x32 solid-angle bins holding a particle) is the measure that means the same
thing for a continuum and for a spring network, so it is the one to compare across the whole history.
Plus: on-shell fraction, `sigma_max(F) - 1` (mean and p99), radial position vs the tissue surface, and
particle count.

`d/hex`, `gap`, `lcc`, `mean_degree_z` are **not** reported for continuum runs. They describe a bond
network that no longer exists.
