# 09 — one live run: tissue, basement membrane and matrix on one clock

## Why 05–08 has to be rebuilt

Every body in the okuda_ECM series is a **replay**, so none can react to another:

- the epithelium is solved once by `tissue.load_or_build` into a cache and replayed by
  `RealDriver` (`test_05l_supply.py`);
- the basement membrane is a Python **rig** (`Rig05b → Rig05l → Rig07c → Rig07i → Rig08a`), not
  engine operators, so it never enters a spec;
- the matrix is a `traj.npz` from `06_spheroid_ecm`, whose `mesh_contact` ran against the
  **original reference tissue**.

Measured consequence: `08b_s1_finger`'s tissue is R_med 7.41 with a finger tip at 16.51, while the
replayed matrix cavity sits at r ≈ 16.6–17.2 — the reference tissue's final R_med 17.58. The bud
carved into the tissue is invisible to the matrix, and the ~10-unit gap is the whole error.

## What already exists (verified in the source, not assumed)

`test_06_three_bodies.py` is **already one engine run**:

```python
H, out = engine_run(S, device=dev, on_frame=on_frame)      # :492
```

with the sheet's substep loop inside the engine's frame loop (`:413-427`), `n_sub` re-measured from
`sheet.spectral_rate` every 10 frames. And the membrane→tissue force is already computed:

```python
fbs, fes = plq.scatter(f_n, sheet.x, x_ep)                  # :424
mom = max(mom, float((fbs.sum(0) + fes.sum(0)).norm()) / ...)
```

`fes` is the tissue's share, and it is used **for nothing but the momentum residual**. The coupling
exists in the code and is discarded, because `on_frame` is a hook that cannot return a delta.

`VertexPlaques` (`:104-175`) already binds a sheet node to an epithelial **vertex index**, and its
header says why: "`cell_divide` re-indexes faces, so a plaque bound to face k at frame 0 is bound to
a different piece of tissue at frame 200. Vertices are only ever APPENDED." The 07/08 chain binds to
`ct_face`, a cell id, which is correct only against a cache. **The fused run builds on the 06
lineage, not the 08 one.**

## The shape of the fix

Neither extreme. Do **not** promote the membrane to ~30 operators (21 of 31 rig capabilities have no
registered operator and 4 are genuinely new), and do **not** invert ownership so a rig drives the
engine. Instead:

- tissue and matrix are engine sets and operators in one spec;
- the membrane is **one** operator, `bm_solve` (`at: vertex`, `EMIT="velocity"`), owning
  `bm_ops.Sheet`, the plaque edge set and its own adaptive substep loop as internal state, and
  returning `{vertex: fes/mu_v}` — the arrow that is currently thrown away.

This keeps G14–G83 valid by construction: same file, same float64, same autograd force. Only the
*driver* changes, which is the thing being changed on purpose.

Two constraints force the loop-owning shape rather than a substep block:

- the engine takes `count = round(sim.dt / H.sub_dt)` from a **spec constant**, so an adaptively
  measured `n_sub` has nowhere to live, and nested substep blocks hard-fail;
- at the tissue's `dt = 1.0` an MPM substep block would either integrate past the certified 4e-4 CFL
  or need ~2,500 substeps a frame (~18 h). So the matrix gets `mpm_relax`, one operator owning its
  eight substeps, the same pattern `mpm_gather` already uses.

## Milestones

**M1 — `09a_fused_bm`.** Tissue + membrane, live, in the tissue's own 50-unit box. No MPM, no
rescale, no npz anywhere. `bm_solve` last in the schedule so the sheet sees the frame's final
topology and the vertices `cell_divide` just appended. Ships with a **`kn: 0` control** that must
reproduce the uncoupled tissue's `n_cells(t)` and `r_apical(t)` exactly — otherwise the coupling is
not the only thing that changed.

**M2 — `09b_fused_all`.** Add the matrix: `MPMGrid` gains `origin`/`span`, the tissue is rescaled
into box units by an exact similarity transform, `mesh_contact` gains a live mode reading
`lvl._mesh`, and `mesh_reaction` returns the contact's summed impulse to the vertices — the
matrix→tissue arrow, which today does not exist at all (`ecm_load` cannot read the contact run's
output, by key and by absence).

## What re-opens, and it is not the sheet's own gates

`Sheet`, `Clutch` and `protease_ops` are imported verbatim and stay float64, so their gates are
re-run as a smoke test rather than re-derived. What genuinely re-opens is **everything measured
against the `k_drive` shadow epithelium**: `x_epi` obeyed `f_epi = fe + k_drive·(a_epi − x_epi)` with
`k_drive = 50`, so the membrane's entire effect on the tissue was a bounded offset `|fe|/50` that the
cache erased every frame. Live, that term is gone. Re-measure standoff, λ_geo against R_ep, slip,
`epi_track`, and the plaque momentum residual.

## Two defects found in `mesh_contact` on the way

1. **Fixed** (`0ce71582`): the bin grid sized a face's angular extent as `max_edge / rmin`, the
   *minimum* corner radius, so one vertex near the centroid sent it to 6.19 rad — 355° for one face —
   collapsing the grid to its 4-row floor (~5,800 triangles per bin). Now uses the median corner
   radius; a no-op on the reference tissue, which is what makes it safe.
2. **Open**: the contact assumes the tissue is **star-shaped about its centroid** — the ray goes from
   the centroid and takes the outermost hit. Exact for a spheroid, approximate for a bud with a
   waist, silently wrong for an overhang. This matters directly for anisotropic growth in fibres.
