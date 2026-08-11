# Paste-ready prompt to open the 06 session

Continue the ECM/BM/spheroid work in `/workspace/Plexus/prototype/ecm`, runs in
`/workspace/Plexus/log/okuda_ECM`. Python is
`/workspace/.conda_envs/neural-graph-linux/bin/python` with `PYTHONPATH=/workspace/Plexus/src`.
Last commit `65620e7f`. Two GPUs: cuda:0, cuda:1.

## Where things stand

The goal is 06: the vertex-model spheroid, the basement membrane and the fibre matrix on one
replayed tissue, with **no BM-ECM interaction** (that is deliberate, not a shortcut).

**Both halves work and are certified against the same driver:**

- `06_spheroid_ecm` — vertex model (6,380 cells) + fibre matrix, 2x2 panel movie. This is 04d's
  configuration renamed.
- `06c_real_driver` — the sheet on the real tissue. 401 frames, lam_geo 3.713 against an apical
  AREA ratio of 3.766 (1.40%), momentum 1.2e-16, 100% of bonds held.

**The key lesson, learned expensively — do not undo it.** `test_06_three_bodies.py` failed badly
(worst triangle quality 0.000, lam_geo 9.46, the sheet collapsed to a third of the epithelium's
radius) because it invented a new adhesion: one bond per tissue VERTEX, a per-plaque rest length,
turnover retargeting. 05b's own law — **bind to a FACE with barycentric weights, one shared rest
length** — handles the real tissue on the first try. The failure was the three additions, not the
tissue. Change ONE variable at a time.

**`RealDriver`** (in `test_05l_supply.py`) is the mixin that does the swap, applied at three heights
of one class chain, each time changing only `_epi_anchor`:

| rig | base | adds | folder |
|---|---|---|---|
| `Rig06c` | 05b | adhesion | `06c_real_driver` |
| `Rig05l` | 05f | refinement, mass balance, tear | `05l_G43_secrete`, `05l_G44_refine` |
| `Rig05m` | 05h1 | MT1/proMMP2/TIMP chemistry | `05m_protease` |

`05m` breaches the real epithelium: 2,167 of 5,120 faces torn in 300 frames, rho_min on
rho_crit 0.350.

Three bugs the swap surfaced, all state sized to the OLD epithelium — expect more of this kind:
`ct_face` still indexing a 5,120-triangle icosphere (device-side assert far from its cause);
the receptor pool `N_f` is per epithelial CELL so it resizes with the mesh; and `sheet.live` masks
the whole reservoir, so `(~live).sum()` reported -13,109,760 faces torn (the breach is live faces
LOST since seeding).

## Gates

`05k_G40_stiffness` .. `05k_G46_standoff`, one folder per gate, each with `gate.png`,
`metrics.json`, `movie.mp4`. Rig: `test_05k_gates.py` (numbers) and `test_05k_folders.py` (folders).

- G40 lam_geo vs kappa_n over 8x: **pass 4.72%** — but monotone (3.641 -> 3.817), so the bond's
  compliance is leaking into a growth measure. Passed, mechanism present.
- G41 lam_geo vs sqrt(A_ep(T)/A_ep(0)): **pass 1.40%**
- G42 worst triangle quality > 0.2: **pass 0.587**
- G43 rho within 10%, `bm_secrete` ON: **pass 0.70%** (401 frames)
- G44 mean edge in [0.8,1.7]x, `bm_refine` ON: **pass 0.929x** — the sheet refined twice,
  5,120 -> 20,480 -> 81,920 faces. Without refinement the same run reaches 3.63x, which is what
  the demand for `bm_refine` measures.
- G46 signed standoff: **fail -2.7e-3**, but the measure compares MEAN RADII, which is biased on a
  bumpy surface. Needs 05b's per-node signed standoff on the tissue's faces before it means anything.

## Do this next

1. Update `note_sheet.tex`: G43 and G44 now pass on the real surface (0.70%, 0.929x) and the table
   still shows G43 pending and G44 failing. Figures are in `05l_G43_secrete/`, `05l_G44_refine/`.
2. Wire `bm_panel.py` into the 2x2's bottom-left slot, which is currently blank. `draw_bm(...)` is
   written and previewed: `mode="lam"` colours the surface by lam_geo, `mode="mt1"` by MT1-MMP
   expression, both with red plaque dots at their ATTACHMENT POINT on the epithelium and a zoomed
   cross-section inset. Prerequisite: the rigs must save per-frame `mt1` and the plaque attachment
   points in their kept-frame tuple — a few lines each.
3. Run 06 properly: replay the tissue once, run the matrix and the sheet against it, render the 2x2
   with the BM panel filled. No coupling, so this is a render job, not new physics.
4. Then, if it holds, inject 05m's chemistry into 06 for the breach version.

Still open beyond that: the penalty contact's compliance (S13/S14/S15/S18/U1, wants a barrier
formulation, BFEMP); `fibril_pull` (B5, the sheet->stroma arrow) does not exist anywhere; B9 is
one-way by construction.

## Conventions to keep

- **One gate = one folder** with `gate.png` and `movie.mp4`. A gate you cannot watch is a number
  someone has to trust.
- Thresholds are decided BEFORE the run, and a threshold belongs in the unit of the phenomenon, not
  of the mesh.
- Figures: white background, no box (top/right spines off), **no titles at all**, bold letter
  top-left, verdict in green when passed and red when failed.
- Notes follow the 4-part template: coarse mechanism/biology with schematic, equations and symbols,
  Plexus implementation (hierarchy/sets/operators/state/field), gate tests. `note_sheet.tex` is the
  reference for the gate-table style (tabularx, page width, \scriptsize).
- Every model declares `general.units:` (length_um, time_s, force_nN); without it no result may be
  quoted with a unit.
- A constant across a sweep is a rail; a limit in a comment stops describing the code.
