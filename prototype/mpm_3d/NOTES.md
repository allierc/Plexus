# 3D MLS-MPM in Plexus — prototype + the operator modifications

This prototype brings the Plexus MLS-MPM solver to 3D. The decomposed 2D operators
(`mpm_grid`, `mpm_strain`, `p2g`, `mpm_grid_update`, `g2p`) were made **dimension-
generic** in the main codebase (not copied to `*_3d` variants), so the *same* operators
run 2D and 3D — `general.dim: 3` is the only switch. The 2D path is **bit-identical**
to before (`check_consistency.py` asserts it); the 3D path mirrors the reference
`MPM_pytorch/.../MPM_3D_P2G.py` + `MPM_3D_step.py`.

## What changed in the codebase (all gated so D==2 is unchanged)

| file | change |
|---|---|
| `operators/mpm_grid.py` | field is `dim`-D (`[nx,ny]` / `[nx,ny,nz]`); `stencil_offsets(D)` (3^D); `bspline()` D-generic over `g.shape`, row-major flatten |
| `operators/mpm_strain.py` | F is D×D; `J` = analytic 2D / `det` 3D; **liquid reset `F:=J^(1/D) I`** (cube-root in 3D — volume-correct, see below); snow SVD + the MPM_3D proper-rotation sign fix in 3D |
| `operators/p2g.py` | D-stencil scatter; polar rotation R = analytic (2D) / **SVD `U Vh`** (3D); `a_ext`/momentum are D-vectors |
| `operators/mpm_grid_update.py` | 2D body kept verbatim (CSF + obstacle walls); 3D adds reflective walls + tangential `wall_damp` on all D box faces. CSF / rasterized obstacles are 2D-only |
| `operators/g2p.py` | D-generic gather, affine-C outer product, wall-contact test and advection clamp (box = `world_size`, axis 0 = width, others 1) |
| `models/entities.py` (`mpm_particle.provision`) | `D`-generic block-fill (`[lo…,hi…]`, 4 vals 2D / 6 vals 3D); `p_vol` = disc πr² (2D) / sphere 4/3·πr³ (3D); `F=I_D`, `C∈R^{D×D}` |
| `operators/gravity.py`, `operators/aggregate.py` | emit D-vectors |
| `engine.py` | pass-2 (child) sets use `_dim_schema(H.dim)` like top-level sets; `_start_centers` D-generic |
| all six mpm-stack ops | `SUPPORTED_DIMS = [2, 3]` |

## Consistency notes (2D vs 3D vs the reference)

- **2D is byte-for-byte unchanged** — verified on a liquid spec and a snow spec
  (`check_consistency.py`: pos/vel/F/C/Jp all `torch.equal`).
- **Liquid volume reset.** The reference `MPM_3D_step.py` resets liquid `F := sqrt(J)·I`
  in 3D (a literal port of the 2D line); that makes `det(F)=J^{3/2}≠J` and mis-evolves
  the liquid volume. We use the dimension-correct `F := J^{1/D}·I` (so `det(F)=J` in any
  D); in 2D `J^{1/2} == sqrt(J)`, i.e. identical to both Plexus-2D and the 2D reference.
- **Polar rotation.** 2D keeps the analytic `cs/sn` rotation; 3D uses the SVD polar
  `R = U Vh` with the same proper-rotation sign fix the reference applies in its SVD.
- **Snow clamp** uses Plexus-2D's `[1-2.5e-2, 1+7.5e-3]` in both D (the reference 3D uses
  `+4.5e-3`); kept for 2D bit-identicalness. Jp is clamped `[0.6, 20]` as in Plexus-2D.
- **Boundaries.** Plexus uses reflective walls (clamp the normal grid velocity) + a
  tangential `wall_damp` friction; the reference multiplies the normal component by a
  `friction` factor. Same qualitative "settles in a box" behaviour; we keep Plexus's.

## Specs (`specs/`)

| spec | what it tests |
|---|---|
| `mpm3d_cube_drop` | elastic (jelly) cube, block-fill, gravity + bounce — minimal smoke |
| `mpm3d_ball_drop`  | elastic ball, **no** block (3D ball seeding + sphere p_vol) |
| `mpm3d_water_drop` | liquid block collapse (3D liquid branch, `J^{1/3}` reset, mu=0) |
| `mpm3d_snow_block` | snow cube crumple (3D snow SVD + sign fix + plastic Jp) |
| `mpm3d_multimaterial` | jelly + water + snow side by side (multi-object, à la `multimaterial_1_3D`) |

## Run

```bash
# conda env + PYTHONPATH=src, from the repo root
python prototype/mpm_3d/run_mpm3d.py                # all specs -> 3D turntable mp4
python prototype/mpm_3d/run_mpm3d.py cube_drop      # substring filter
python prototype/mpm_3d/check_consistency.py        # assert 2D bit-identical
```

Outputs: `$GNN_OUTPUT_ROOT/graphs_data/mpm_3d/<name>/movie_mpm_particle.mp4`
(+ evolution / final PNGs). Rendering is the existing 3D gaussian-splat turntable
(`plotting.render_3d: tight`), which auto-engages for any `dim: 3` set.
