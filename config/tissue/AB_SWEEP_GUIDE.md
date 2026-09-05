# `cell_mechanics[model: apicobasal]`, one parameter at a time

22 variants of `ab_02_flat_apicobasal`, each changing exactly ONE key of the operator and nothing
else. Same seed, same 60-cell flat disc, same 20 frames. Movies in
`graphs_data/tissue/ab_02_flat_apicobasal_<param>_<lo|hi>/movie.mp4`; final frames side by side in
`graphs_data/tissue/ab_02_flat_apicobasal_montage.png`.

The baseline is `k_v: 4.0, kappa_s: 0.2, gamma: 0.0, Lambda: 0.0, K_R: 0.0, mu: 1.0,
relax_iters: 30, eta: 0.08, cap_frac: 0.12, mono_k: 1.2, plane_axis: 2,
rest_calibration: volume_only, sep_mu: 0.0`.

**shrink** = mean junction length at the last frame over the first (below 1 = the patch contracted).
**cv** = coefficient of variation of cell area at the last frame (how unequal the cells ended up).
**h** = median cell thickness `2|sep|`, seeded at 0.4.

| variant | value | shrink | cv | h | what it is |
|---|---|---|---|---|---|
| BASELINE | -- | 0.5955 | 0.224 | 0.400 | |
| `k_v_lo` | 0.4 | 0.2356 | 1.761 | 0.400 | volume elasticity |
| `k_v_hi` | 40.0 | 0.9954 | 0.037 | 0.400 | |
| `mono_k_lo` | 0.6 | 0.2544 | 1.334 | 0.400 | target-volume multiplier |
| `mono_k_hi` | 2.4 | 1.0933 | 0.051 | 0.400 | |
| `kappa_s_lo` | 0.02 | 0.9757 | 0.037 | 0.400 | surface tension |
| `kappa_s_hi` | 2.0 | 0.3460 | 1.582 | 0.400 | |
| `gamma_lo` | 0.5 | 0.0920 | 1.562 | 0.400 | perimeter contractility (OFF at baseline) |
| `gamma_hi` | 2.0 | 0.0621 | 1.711 | 0.400 | |
| `Lambda_lo` | 1.0 | 0.4681 | 1.393 | 0.400 | line tension (OFF at baseline) |
| `Lambda_hi` | 5.0 | 0.5304 | 1.598 | 0.400 | |
| `K_R_hi` | 0.5 | 1.1794 | 1.263 | 0.400 | radial spring (OFF at baseline) |
| `mu_lo` | 0.1 | 0.8306 | 0.248 | 0.400 | mobility |
| `mu_hi` | 10.0 | 0.5423 | 0.645 | 0.400 | |
| `eta_lo` | 0.01 | 0.8137 | 0.242 | 0.400 | gradient step |
| `eta_hi` | 0.4 | 0.5049 | 0.573 | 0.400 | |
| `relax_iters_lo` | 5 | 0.7907 | 0.226 | 0.400 | steps per frame |
| `relax_iters_hi` | 120 | 0.1875 | 1.740 | 0.400 | |
| `cap_frac_lo` | 0.02 | 0.5834 | 0.222 | 0.400 | per-step displacement cap |
| `cap_frac_hi` | 1.0 | 0.5955 | 0.224 | 0.400 | |
| `rest_calibration_fb` | force_balance | 0.9981 | 0.035 | 0.400 | how `V_eq` is set |
| `plane_axis_off` | none | 0.5955 | 0.224 | 0.400 | in-plane constraint |
| `sep_mu_hi` | 1.0 | 0.6022 | 0.240 | **0.525** | apico-basal mobility |

---

## Read them in four groups

### 1. The two knobs that set the cell's target size

`k_v` and `mono_k` are the volume term `(1/2) k_v (V - V_eq)^2` with `V_eq = mono_k * V0f`. One sets
how hard the cell defends a volume, the other sets which volume.

* **`k_v_hi` (40) and `mono_k_hi` (2.4)** are the two biggest, most orderly patches on the sheet.
  Cells stay near-identical (cv 0.04-0.05). `mono_k_hi` is the only variant besides `K_R_hi` where
  the patch **grows** (1.09): the target volume is doubled, so the tissue inflates to reach it.
* **`k_v_lo` (0.4) and `mono_k_lo` (0.6)** collapse to a quarter of their edge length and the mesh
  goes ragged. With nothing defending volume, surface tension wins outright.

### 2. The knobs that pull the tissue in

* **`kappa_s`** is the surface tension, and it is the only inward force at baseline. At 0.02 the
  patch barely moves (0.976) -- **that pairing with `k_v_hi` is the control worth noticing: two
  different parameters, same "nothing happens" picture, because both make the same term irrelevant.**
  At 2.0 it contracts to a third and the cells scramble.
* **`gamma` (perimeter contractility) is the most violent thing in the sweep.** It is 0 at baseline
  and the term is `(1/2) gamma * P^2` with **no target perimeter**, so it is pure contraction with
  nothing to balance it: at 0.5 the patch collapses to 9% of its edge length, at 2.0 to 6%. In the
  montage these two are barely more than slivers. If you want to see the tissue survive `gamma`,
  it needs a target perimeter or a much bigger `k_v`.
* **`Lambda` (line tension)** is also 0 at baseline, also pure contraction, and it **is not
  monotone**: 1.0 shrinks MORE (0.468) than 5.0 (0.530). That is the displacement cap biting -- at
  Lambda 5 the per-step force is so large that `cap_frac` truncates it every step, so the tissue
  moves LESS per frame despite a stronger force. Worth knowing before reading any Lambda sweep as
  a dose-response.
* **`K_R` (radial spring)** pins each vertex to the seed radius. It is the only variant that both
  **expands** the patch (1.18) and scrambles it (cv 1.26) -- the montage shows a jagged star. On a
  flat disc this term has no physical meaning; it is here so you can see what it does when someone
  leaves it on.

### 3. The solver knobs -- `mu`, `eta`, `relax_iters`, `cap_frac`

`mu`, `eta` and `relax_iters` multiply together into ONE quantity: **how far the relaxation travels
per frame**. The sweep shows exactly that, as a single ordered sequence in `shrink`:

```
relax_iters 5   0.791        eta 0.01   0.814        mu 0.1    0.831
   BASELINE     0.596        eta 0.08   0.596        mu 1.0    0.596
relax_iters 120 0.188        eta 0.4    0.505        mu 10     0.542
```

**None of these runs is at convergence**, so these three do not change the physics -- they change
how far down the same path you are at frame 20. That is why `relax_iters_hi` (0.188) looks so much
like `k_v_lo` (0.236): one collapsed because volume was cheap, the other because it had four times
as long to collapse. **The pictures alone cannot tell those two apart**, which is the honest limit
of this sweep and the reason the shrink number is printed next to each.

**`cap_frac` is the quiet one, and its result matters.** At 1.0 -- the cap effectively removed -- the
run is 9.2e-4 of an edge from the baseline, i.e. **the cap almost never binds at the baseline 0.12**,
so it is not silently shaping the reference. At 0.02 it does bind, and the run moves 3.6e-2 away.

### 4. The structural switches

* **`rest_calibration: force_balance`** is the "do nothing" control, and it is the cleanest picture
  on the sheet: shrink 0.9981, cv 0.035. It solves for the `V_eq` offset that puts the SEEDED state
  at rest, so the tissue starts balanced and stays there. `volume_only` (the baseline) does not, so
  the baseline patch has a genuine collapse to watch.
* **`plane_axis: none` is BIT-IDENTICAL to the baseline** -- `|dpos| = 0.000e+00`, not merely close.
  The constraint zeroes the z-component of the step; on a patch that is exactly planar and
  symmetric, the z-force is already exactly zero, so the projection has nothing to remove. **Every
  one of the 22 variants ends with a z-extent of exactly 0.0000.** The constraint is inert on this
  specimen, which is worth knowing before trusting it on a curved one.
* **`sep_mu: 1.0` is the only variant that touches what this promotion added.** At baseline the
  apico-basal separation is frozen, so `h` is 0.4000 in all 21 other runs. Here the cell thickens
  to 0.525 while the patch contracts by about the same amount as the baseline (0.602 against
  0.596): the cell keeps its volume and trades width for height. **That trade is the doubled degree
  of freedom doing its job, and no other row of this table can show it.**

---

## What the montage cannot show you

The camera sits at 18 degrees elevation and this is a flat sheet, so everything is seen obliquely
and thickness is invisible. `h` is a column in the table above for that reason. Neither renderer
exposes a per-spec camera elevation, so a top-down view of these would need a change to
`render_vtk.CAM` or to `live_movie`'s camera, which is shared by every other run in the repo.
