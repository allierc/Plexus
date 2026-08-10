# Prompt to open the 05 session: the basement membrane as a Plexus2 sheet

Paste everything below the line into a fresh session, working directory `/workspace/.devcontainer`.

---

We are building an epithelial spheroid growing inside an extracellular matrix, as Plexus2 operators,
in `/workspace/Plexus/prototype/ecm` with runs in `/workspace/Plexus/log/okuda_ECM`. Four pieces are
finished and each was built on its own testbench first, measured against a control, written into its
own note, and only then combined. **Your job is the fifth piece: the basement membrane.** Build it on
its own testbench. Do not touch `04` until the sheet stands up on its own.

## Read first, in this order

- `discovery_okuda/note_spheroid_bm_ecm.pdf` — §6 is the archived sheet (nodes + crosslinks) and why
  it stalled; §9 is the literature the sheet has to answer to; **§10 is the set/operator table you
  are implementing**. §8 is sixty-six archived runs and what survived them.
- `discovery_okuda/note_surface.pdf` — §6 is the argument that the sheet must be **codimensional**
  and its adhesion an explicit relation; §7 is `04`, the run you will eventually feed.
- `discovery_okuda/note_junction.pdf` §8 and `note_fibre.pdf` §9 — the two specs that already fix the
  schedule, the substep and the matrix around you.
- `prototype/ecm/HANDOVER.md` — the corrections section at the top. Every claim in it was checked
  against the engine and several did not survive; read it as a list of ways this exact problem has
  already produced convincing wrong answers.

## What is already settled, with numbers, and is not yours to re-litigate

- **The sheet cannot be an MPM continuum.** ~100 nm on a ~200 µm spheroid is one part in 2000 of the
  radius; resolving it needs Δx ≈ 3×10⁻⁴ and a grid of 3500³. It is codim-1: **thickness enters as a
  rest length, never as resolved geometry.** Its adhesion is an explicit relation, because below one
  grid cell the MPM grid welds the two bodies — runs 142–151 measured that ten times.
- **An MPM fibre cannot act as a rope.** `mpm_strain` advects **F** by the *grid's* velocity gradient
  and never measures the distance between two particles, so two ends more than a cell apart are two
  bodies. Any load path you build must be an explicit edge set.
- **A positional update launders deformation.** A particle moved by the engine delta never passes
  through the grid, so **F** misses the motion: run 130 reported 0.31 of a true stretch of 2.44 (13%)
  while run 121, routed through the grid, reported 2.25 of 2.30 (98%).
- **The interface exists and conserves momentum.** `mesh_contact` (`prototype/ecm/mesh_contact_ops.py`)
  does particle-to-surface contact against the spheroid's own moving, re-meshed surface: momentum
  residual 1.2×10⁻⁷ max / 1.5×10⁻⁸ median (float32 precision), penetration 0.82 grid cells, and its
  lookup is certified against brute force by `selftest()` before any run uses it.
- **The matrix around you** is `02h`: 20 particles per strand, 8 substeps at Δt_sub = 4×10⁻⁴,
  `drag: 8` (undamped it rings at 12.9 Hz for the whole run), `store_stress: true` with
  `measure: vonmises` (|J−1| is 250× smaller and cannot see a shear), stroma `youngs: 15`.
- **A frame has a duration.** 200 → 6380 cells is 5.1 doublings, so at a 12–24 h cycle the 401 frames
  are 2.5–5 days and one frame is 9–18 minutes. Collagen IV turnover (half-life 3–10 h) is therefore
  13–93 frames — resolved, and the reason the mass balance is worth having.

## Three properties of the substrate, re-checked against the engine on 2026-08-10

`HANDOVER.md` is a record of what was true when it was written, and one of its three named defects has
since been fixed. Do not compensate for a bug that is gone, and do not assume the other two are.

- **The substep force-accumulation defect is FIXED.** `engine.py:838–856` now snapshots the outer
  delta and restores it at the head of every substep, so an operator emitting a force from inside a
  `substep_dt` block is recomputed at that substep's own positions and applied ONCE — a flat 20/20/20/20
  where it used to ramp 20/40/60/80 over four substeps. A frame-level force (gravity) still persists
  across the loop unchanged. So `plaque_pull` and `fibril_pull` can go inside the block as designed,
  and HANDOVER's "every run with `membrane_contact_k > 0` applied a contact ramping 1× to 20×" describes
  runs 110–123, not what you will get.
- **`youngs` is still read from the PARENT set, never from the particle set's own.**
  `src/plexus/models/entities.py:63` — `types = getattr(parent, "types_raw", None)`. This is how the
  basement membrane carried the stroma's E = 15 instead of the 400 its spec declared, in every run up
  to 148, and how run 146 came back bit-identical to 144 at ten times the stiffness. If anything you
  build hangs MPM particles off a shared parent, its modulus reaches nothing and the framework says so
  only as `[warn] property 'youngs' on mpm_particle.t is read by no operator`. Give each MPM body its
  own parent set, and PROVE the material you asked for is the material you got before quoting any run.
- **Trajectory sets can be strided independently.** `mpos` kept 202 of 403 frames while `pos`,
  `stress` and `mstrain` kept every one, so indexing them with the same integer paired a sheet at frame
  402 with an epithelium at 201. Read each set's own frame list; do not assume two sets share an axis.

## What to build

The five rows of `note_spheroid_bm_ecm` §10 that have no counterpart in the archived line, in this
order, each its own operator with its own falsifier:

1. **`bm` as a surface mesh** — nodes with `pos`, **F**, areal density ρ, age; `bm_edge` carrying
   in-plane elasticity. Massless and overdamped, so its stiffness costs drag rather than substeps
   (the bound is Δt(z k_b + κ)/γ < 2, a rate, not a wave speed).
2. **`bm_secrete` / `bm_degrade` as a MASS BALANCE**, not a fixed particle budget:
   Dρ/Dt = s − ρ/τ_bm − ρ·(Ȧ/A), with τ_bm = 4–14 h. The two removal terms are the same size here,
   which is the whole reason a sheet can enclose a tripling radius at all.
   *The archived failure to beat:* run 128's secretion added 45,000 particles that never joined the
   sheet — they sat at standoff −0.19, a second shell in the lumen, and halved the reported strain.
3. **`bm_stiffness`** — tangent modulus rising with the principal stretch, E(λ) = E₀[1 + β(λ−1)],
   which for the λ ≈ 2.4 a growing spheroid drives it to is a stiffening of ≈ 7.5× the sheet's OWN
   low-strain modulus at β = 5 — the 0.4 → 3 MPa Candiello (2007) measured on native basement
   membrane. Note that MPa is not a quantity this simulation has: the box is dimensionless and the
   stroma's `youngs: 15` is a number, not a pressure. So the two falsifiable statements are
   DIMENSIONLESS — the stiffening ratio E(λ_max)/E₀ ≈ 7.5, and E_bm/E_ecm ∈ [10, 100] against the
   stroma's 15. Fix the mapping from box units to Pa once, in the note, and state it; do not carry
   MPa into a spec.
4. **`plaque` as an edge set `cell → bm`** — normal spring at rest length ℓ₀ ≈ 0.3 T, tangential
   friction ξ against *relative velocity* (the sheet slides; it is not pinned to a frozen direction),
   at the measured density Σ⁻², Σ ≈ 7 T. **One operator call returns a delta to both endpoints**, the
   way `mesh_contact` does, so the reaction cannot be half-implemented.
5. **`fibril` as an edge set `bm → mpm_particle`** — the stroma is loaded *through* the sheet
   (Keene 1987), which is what lets the epithelium stop pushing the stroma directly.

## The testbench, which is the point of this session

Not the spheroid. Build the smallest rig that can falsify each operator, in the spirit of
`test_03_mesh_contact.py` (a flat patch pressed into a block, three numbers, each of which kills the
method if it fails). A prescribed spherical surface that expands on a schedule, with the sheet on it
and an MPM block or shell outside it, gets you every measurement below at a fraction of the cost —
and it has a closed form to check against, which the spheroid does not.

Each operator needs a measurement that can come back wrong:

| operator | what its test measures | what a failure looks like |
|---|---|---|
| `bm_edge`, `bm_stiffness` | the stiffening ratio E(λ_max)/E₀ over the run, ≈ 7.5, and E_bm/E_ecm | one modulus wearing two names; a declared modulus that reached nothing (see `youngs`, above) |
| `bm_secrete` | ρ held constant while A(t) triples | material that is not part of the sheet (128) |
| `bm_degrade` | a hole opens only where protease is | proteolysis everywhere |
| `plaque_pull` | standoff = ℓ₀; **momentum conserved to machine precision** | a one-way coupling passes everything else |
| `plaque_rupture` | fraction bound vs applied stress | a threshold above the load (127 was a null for this reason, not because it was unwired) |
| `fibril_pull` | stroma displacement with the sheet present vs absent | the epithelium still pushing the stroma directly |

## What NOT to reuse

`membrane_ops.py` (134 KB) is the archived line — `seed_basement_membrane`,
`basement_membrane_bond`, `integrin_adhesion`, `adhesion_pull`, `basement_membrane_secrete` and the
rest are the operators of the sixty-six runs in §8, i.e. the sheet that stalled. Do not import them
and do not extend them. Start a new `bm_ops.py`; take names and lessons from that file, not code.
`integrin_ops.py` is the MPM-fibre integrin refuted by runs 142–151 and is likewise a record, not a
dependency. What you DO build on: `mesh_contact_ops.py` (the certified interface),
`ecm_ops.py`/`ecm_spec.py`/`ecm_render.py` (the matrix, the spec builder, the renderer),
`junction_ops.py` + `medioapical_ops.py` (the tissue `04` replays).

## How to work

- **Certify before you run.** The lookup in `mesh_contact_ops.py` is checked against a test on every
  triangle (`selftest()`: coverage 1.00000, zero disagreement) before any 200-frame run is trusted to
  it. Anything with a spatial query, an index map or an inheritance rule gets the same treatment —
  those are the failures that are silent.
- **Every number gets a control.** The fibre reorientation in `04` looks like a fibre effect until you
  carry the frame-0 strands through the run's own displacement field and get the same answer to three
  decimals. A measurement without its control is an impression.
- **Read every quantity against what it is a quantity OF.** `04`'s tissue-wide mean fibre stretch is
  1.031 and the shell nearest the tissue is at 1.81; the mean was the far field's answer wearing the
  whole matrix's name. Bin by radius, by entity, by class.
- **Finish and verify each stage before starting the next.** Do not roll on.
- Report into a new `discovery_okuda/note_sheet.tex` (compile it: `pdflatex` twice), one section per
  operator, in the register the other notes use — the mechanism, the number, and what would falsify
  it. **Then stop and wait.**

## Environment

```bash
PY=/workspace/.conda_envs/neural-graph-linux/bin/python      # default python3 has no torch
export PYTHONPATH=/workspace/Plexus/src
cd /workspace/Plexus/prototype/ecm
$PY mesh_contact_ops.py --device cuda:0                      # the certification pattern to copy
```

The convention every step here follows: one file `test_05_*.py` per rig, taking
`--device / --frames / --name`, writing to `log/okuda_ECM/05_<name>/` with `spec.yaml` (the spec as
run, so the run is reproducible from its own folder), `metrics.json`, `metrics.png`, `movie.mp4`,
`strip.png`, `3d.png`. The module docstring states what the rig combines, what it replaces, what it
measures — each measurement able to come back wrong — and what it is NOT. Copy `test_03`'s.

One path trap: a `spec.yaml` written by a cluster run holds `/groups/saalfeld/home/allierc/Graph/Plexus/...`
paths (see `04c_spheroid_fibres/spec.yaml`), and that tree is NOT visible from inside the devcontainer,
where the same files are `/workspace/Plexus/...`. Do not copy a saved spec's `tissue:` path verbatim
into a local run.

Two local RTX A6000s (`cuda:0`, `cuda:1`). The cluster is up and faster for anything long:
`/workspace` is the same filesystem as `/groups/saalfeld/home/allierc/Graph`, submit with

```bash
ssh allierc@login1 "bsub -n 8 -gpu num=1 -q gpu_a100 -W 240 -J pg_05 -o <log>.out -e <log>.err bash -l <script>.sh"
```

with the job script `cd`-ing to the `/groups/...` path and running `conda run -n connectome-gnn python ...`.
See `log/okuda_ECM/_cluster/04c.sh` for a working example.

## What you hand back

An operator set that drops into `04`'s schedule as a change to the spec rather than to the
architecture: `bm_stiffness → bm_elastic → plaque_pull → fibril_pull` inside the substep block,
`bm_secrete`/`bm_degrade`/`plaque_seed`/`plaque_rupture` at frame level. Plus the note, plus the
numbers, plus an explicit statement of what the sheet does **not** yet do.
