# CODEMAP — per-cell parameter → particle acceleration in `material_cardio_cells`

Task B. Every claim below is read off the source (file:line) and, where it is a claim about
*behaviour*, backed by a number from a probe in this directory. Probes are read-only
(`probe_clamps.py`, `probe_affine.py`, `probe_affine2.py`, `probe_sigma.py`), all run on
`cuda:1` against `config/material/material_cardio_cells.yaml`, N = 236 000 particles,
C = 472 cells, 128² grid, `dt = 2e-3`, `substep_dt = 2e-4` → **10 substeps per frame**
(`engine.py:835-841`).

---

## 0. Headline

| claim | verdict | number |
|---|---|---|
| one **substep** map θ=(E₁..E₄₇₂) → X is exactly affine | **TRUE** in the interior | rel. superposition defect **5.4e-12** at ±300 % spread (float64), i.e. machine precision |
| … in the wall band | **kinked but tiny** | rel. defect **7.2e-5** at ±300 %, **9.1e-8** at ±30 % |
| one **frame** (10 substeps) map is affine | **FALSE** | rel. defect **1.5e-2** at ±30 %, **9.7e-2** at ±300 %; scales ∝ spread, and is a **bulk** effect (interior 0.0147 vs overall 0.0147), not a boundary one |
| `a_max: 200` clamp is active | **NO — dead code here** | `H.delta('cell')` is identically **0.0** over 410 substeps |
| CFL velocity cap is active | **NO** | max speed **0.575** vs vmax **15.625** |
| per-cell **gain** exists in this spec | **NO** | `active_force` has no gain; `lvl.gain` is absent (`gain_buffer: false`) |
| a per-cell E column is observable | **NO, by ~3 orders** | +10 % on one cell moves the sheet by **0.0016 px** (1 substep) / **0.036 px** (1 frame) max, on a 1024² image |

---

## 1. Chain A — per-cell Young's modulus E_c → particle position

### A1. per-cell E → per-particle E → Lamé (LINEAR, exactly)
- `segmentation_seed.py:114-138` `_cell_values` → one E per cell (measured beat amplitude,
  inverse-mapped into `[youngs_min, youngs_max] = [40, 220]`).
- `segmentation_seed.py:209` `p_y = y_all[cid.clamp(0, n_cells)]` — broadcast cell→particle;
  every particle of a cell shares the value **exactly** (this is what makes θ per-cell).
- `segmentation_seed.py:211` → `entities.py:36-40`
  ```python
  def _lame(E, nu=_NU):        # _NU = 0.2  (entities.py:33)
      mu = E / (2 * (1 + nu));  la = E * nu / ((1 + nu) * (1 - 2 * nu))
  ```
  **Linear, homogeneous** in E. Measured: `mu/E = 0.4166667`, `la/E = 0.2777778` (probe_clamps).
- `segmentation_seed.py:212-214` zeroes `mu` for liquid particles — **inactive**
  (`is_liquid_any: false`).
- `segmentation_seed.py:215` `lvl.mu, lvl.la = mu, la`; `:216-220` also registers `youngs`
  and `cell_id` per-particle buffers. **`cell_id` is the cell↔particle map you need to build A.**

### A2. Lamé → Kirchhoff stress (LINEAR in E, given F)
- `mpm_scatter.py:95` `mu, la = p.mu, p.la`.
- `mpm_scatter.py:96-100` snow hardening `mu ← mu·exp(...)` — **INACTIVE** (`is_snow_any: false`);
  gated by `snow.any()` at `:97`.
- `mpm_scatter.py:86-88` prestress `F ← F @ F_res_inv` — **absent** (`F_res_inv: false`).
- `mpm_scatter.py:101-105` polar rotation R (2D analytic `cs/sn`, `+1e-9` in the denominator at
  `:103`). R depends on F only — **no θ dependence**.
- `mpm_scatter.py:112-113`
  ```python
  stress = 2*mu[:,None,None] * ((F - R) @ F.transpose(-2,-1)) + eye * (la*J*(J-1))[:,None,None]
  ```
  With `mu = c₁E`, `la = c₂E`: **τ_p = E_p · K_p(F_p)**, exactly homogeneous of degree 1.
  Verified numerically: `‖τ(1.7E) − 1.7·τ(E)‖/‖1.7τ‖ = 6.8e-8` (float32 eps) — `probe_sigma.py`.
- `mpm_scatter.py:118-120` `+= H.active_stress` — **skipped** in this spec
  (`act_stress_present: 0.0`; the spec uses `active_force`, not `active_stress`).
- `mpm_scatter.py:121-131` `store_stress: true` caches **Cauchy** `sigma = stress / |J|.clamp_min(1e-9)`
  *after* any active stress and *before* the dt rescale at `:132`. `J ∈ [0.73, 1.07]` so the
  `abs()`/`clamp_min` are no-ops. Verified: `‖p.sigma·|J| − τ_recomputed‖/‖τ‖ = 1.2e-8`.

### A3. stress → affine momentum matrix (LINEAR; the `mass·C` term is the offset b)
- `mpm_scatter.py:132` `stress = (-dt*4*inv_dx*inv_dx) * p_vol * stress` — scalar rescale, linear.
- `mpm_scatter.py:133` `affine = stress + mass[:,None,None]*C` — **`mass·C` is θ-independent → b**.

### A4. scatter to the grid (LINEAR)
- `mpm_grid.py:74-96` `bspline(X, ...)`: `base = floor(X·inv_dx − 0.5)` (`:79`), quadratic weights
  (`:81-83`), grid indices `clamp(0, shape-1)` (`:92`). **Functions of X only** — known from the
  measurement, no θ.
- `mpm_scatter.py:138-140` weights masked by `occ` — identity (`occ_all_one: true`).
- `mpm_scatter.py:142` `mom = mass·V + affine @ dpos_phys` — linear in `affine`; `mass·V` → b.
- `mpm_scatter.py:143-145` `index_add_` of `weight·mass` (→ `gm`, θ-independent) and
  `weight·mom` (→ `gmv`, affine in θ). `mpm_scatter.py:151` publishes `g.m, g.mv, g.c`.

### A5. grid solve (LINEAR except at the walls)
- `mpm_grid_update.py:96` `gv = gmv / gm.clamp(min=1e-10)` — division by a **θ-independent**
  mass ⇒ still linear. (All 16384 cells are massful, `grid_massful: 16384`.)
- `:100` CSF surface tension — **INACTIVE** (`surface_tension` defaults to 0, spec doesn't set it).
- `:113-125` reflective walls, `bnd = 3` (`:115`), active because `boundary: wall` ⇒ `not periodic`:
  - `:118-119` `gv[lox,:,0].clamp(min=0)` / `.clamp(max=0)` (+ the y pair) — **NON-LINEAR KINK**.
    **ACTIVE**: up to **46** (x) + **43** (y) massful cells actually clipped per substep.
  - `:121-122` `torch.where(gl > 0, gl*wd, gl)` with `wd = wall_damp = 0.5` — **NON-LINEAR KINK**
    (sign-conditional). **ACTIVE**: up to **257** massful cells per substep.
  - `:123-124` `gv[:, loy, 0] *= wd`, `gv[:, hiy, 0] *= wd` — unconditional ⇒ **LINEAR**.
    Touches up to **640** massful cells.
- `:126-135` obstacle walls — **INACTIVE** (`obstacles: []`, so `walls` is all-False and `:135` is
  the identity).
- `:160` `g.v = gv`.

### A6. gather back to particles (LINEAR; one kink, inactive)
- `mpm_gather.py:52-56` `new_V = Σ w·gv`, `new_C = 4·inv_dx·Σ w·gv⊗dpos` — linear in `g.v`.
  Note the gather **overwrites** V (pure PIC): the particle's own previous velocity does not
  survive, which is why "acceleration" here means `ΔX = dt_sub · new_V`, not `v_new − v_old`.
- `:57`, `:70`, `:71` `nan_to_num` — identity on finite values.
- `:58-66` inelastic wall contact, `wall_contact = 0.06`, `wall_damp = 0.5`: `near` is computed
  from **X only** (`:62`) ⇒ this is a **frozen diagonal scaling, still LINEAR in θ**. But it is
  large: **26.4 %** of particles are in the band (the segmentation fills the whole unit square —
  the label tif has nonzero labels on 100 % of its 1024² pixels).
- `:67-69` CFL cap `vmax = min(1e9, 0.4·dx/dt) = 15.625` — **NON-LINEAR** but **INACTIVE**:
  max speed observed **0.575**, zero particles at the cap over 410 substeps.
  (`sp.clamp(min=1e-9)` at `:67` is also a no-op: it makes the ratio exactly 1 below 1e-9.)
- `:71` `Xn = X + dt·new_V` — the position update. **This is the whole "acceleration" step**;
  there is no engine integration (see A7).
- `:75` `Xn[:,k].clamp(2*dx, box[k] − 2*dx)` = clamp to `[0.015625, 0.984375]` — **NON-LINEAR KINK**
  and **ACTIVE**: **9.45 %** of particles sit on this bound (they are seeded there, because the
  segmentation covers the full domain out to the pixel edge). For a pinned particle the clamped
  coordinate has an identically-zero row in A.
- `:78-83` occ freeze — identity. `:84-88` writes `pos`/`vel`; `p.C = new_C`.

### A7. there is NO engine integration for this set
- `engine.py:278-303` `_resolve_emit` skips `EMIT` values that are not `velocity`/`acceleration`.
  Here `active_force.EMIT = "mpm_acceleration"` (`active_force.py:33`) and `drag` is given
  `emit: mpm_acceleration` in the spec, so **`H.emit_order == {}`** (measured).
- `engine.py:578` `_integrate` iterates `H.emit_order` ⇒ **does nothing** for `mpm_particle`.
  Positions are advected **only** by `mpm_gather.py:71`. Confirmed empirically.

---

## 2. Chain B — the "gain": what actually exists, and what does not

**The cardio_cells spec has no per-cell gain, and `active_force` cannot express one.**

- Spec line: `{amplitude: 20.0, at: mpm_particle, from: activation, mode: inward, op: active_force}`.
- `active_force.py:69-70`: `grad = fld.grad_at(pos, ...)`; `acc = self.sign * self.amplitude * grad`.
  `self.amplitude` is a **global python float** (`:43`); `self.sign = +1` for `inward` (`:48`).
  There is **no read of `lvl.gain` anywhere in this file**. Probe: `gain_buffer: false`.
- `active_force.py:72` `acc *= lvl.occ` (identity); `:78` returns `{mpm_particle: acc}`.
- The engine adds it to the accumulator: `base.py:475-490 add_delta`, read back at
  `base.py:496-501 delta`.

The operator that *does* carry a gain is `active_stress` (what `discovery_cardio_mpm/train.py:693`
uses, `force_ops = ["active_stress", "drag"]`, with `train.py:175 lvl.gain = gain_p`):
- `active_stress.py:64` `gate = (a * lvl.occ).clamp(min=0.0)` — the clamp is on the **activation**,
  *before* gain, so it is θ-independent.
- `active_stress.py:65-67` `gate = gate * gain` — per-particle gain, **LINEAR**.
- `active_stress.py:70-74` Frank-Starling `gate *= (1 + β(λ−1)).clamp(min=0)` — θ-independent
  factor (λ = |F n|), so still linear in gain; **OFF by default** (`stretch_activation = 0`).
- `active_stress.py:79` `sigma = (amplitude * gate) * nn`; `:81` `H.active_stress = sigma`,
  consumed at `mpm_scatter.py:118-120` — i.e. gain enters through the **stress**, exactly like E,
  additively.
- **`active_stress` needs `direction_from:` (`:35`, `:46-48`) — a vector field the cardio_cells
  spec does not declare.** Switching specs is not free.

If instead you add a per-cell gain to `active_force`, it enters the **body force** path, which is
also affine but through a *different* term:
- `mpm_scatter.py:75` `a_ext += torch.nan_to_num(H.delta(p.name))` (no clamp — see §3),
- `mpm_scatter.py:76` `V = V + dt*(a_ext − self.drag*V)` with `self.drag = 0.0` from the spec,
- `mpm_scatter.py:142` `mom = mass·V + …` ⇒ the gain column lands in the `mass·V` term.
Both routes are linear; **they are not the same column structure** (stress column ∝ `∂mom/∂affine`,
gain column ∝ `∂mom/∂V`), which matters for the Gram matrix's conditioning.

### drag: confirmed linear, and confirmed a **frozen** offset within a frame
- `drag.py:37` `acc = -k * vel * occ`, `k = 30`. Linear in velocity, **θ-free**.
- Crucially, `drag` and `active_force` sit in the **outer** schedule, and `engine.py:828`
  `H.zero_delta()` runs once per tick, so `H.delta('mpm_particle')` is computed **once per frame
  from the frame-start velocity** and held **constant across all 10 substeps**. Within one
  substep it is a pure constant → it goes into b. Measured `‖a_ext‖∞ = 96.3`.

---

## 3. Complete list of clamps / max / min / conditionals on the path

| file:line | thing | linear in θ? | active at cardio settings? |
|---|---|---|---|
| `mpm_scatter.py:66` | `nan_to_num(a_cell, ±a_max).clamp(±200)` on the **parent (cell)** delta | non-linear | **NO — `H.delta('cell')` ≡ 0.0** (nothing emits a cell delta; `aggregate.py:18 EMIT=None`). `a_max: 200` in the yaml is inert. |
| `mpm_scatter.py:75` | `nan_to_num(H.delta(particle))` — the **particle** body force | linear | active but a no-op (finite). **Note the asymmetry: the particle body force is NOT clamped by `a_max`.** |
| `mpm_scatter.py:97` | snow hardening `exp((10(1−Jp)).clamp(±6))` | non-linear | **NO** (`is_snow_any: false`) |
| `mpm_scatter.py:103` | `r = sqrt(cs²+sn²) + 1e-9` | θ-free | active, harmless |
| `mpm_scatter.py:128` | `J.abs().clamp_min(1e-9)` (store_stress only) | θ-free | no-op (`J ∈ [0.73, 1.07]`) |
| `mpm_grid.py:92` | grid index `clamp(0, shape-1)` | θ-free (X only) | active, harmless |
| `mpm_grid_update.py:96` | `gm.clamp(min=1e-10)` | θ-free | no-op (all cells massful) |
| `mpm_grid_update.py:100` | CSF gate `surf > 0` | — | **NO** (`surface_tension = 0`) |
| `mpm_grid_update.py:118-119` | reflective wall `clamp(min=0)/clamp(max=0)` | **NON-LINEAR** | **YES**, ≤ 46 (x) + 43 (y) massful cells/substep |
| `mpm_grid_update.py:121-122` | `where(gl > 0, gl*wd, gl)` | **NON-LINEAR** | **YES**, ≤ 257 massful cells/substep |
| `mpm_grid_update.py:123-124` | `gv *= wd` unconditional | linear | YES, ≤ 640 cells |
| `mpm_grid_update.py:127,135` | obstacle friction / no-slip | — | **NO** (`obstacles: []`) |
| `mpm_gather.py:62-66` | `near` band → `new_V *= 0.5` | **linear** (mask from X) | **YES**, 26.4 % of particles |
| `mpm_gather.py:67-69` | CFL cap `vmax = 15.625` | non-linear | **NO** (max speed 0.575) |
| `mpm_gather.py:75` | `Xn.clamp(2dx, box−2dx)` | **NON-LINEAR** | **YES**, 9.45 % of particles pinned |
| `mpm_gather.py:81-83`, `mpm_strain.py:83-86`, `mpm_scatter.py:138-140` | `occ` masks | linear | identity (`occ_all_one: true`) |
| `mpm_strain.py:47-77` | liquid reset / visco SVD relax / snow SVD clamp | non-linear | **NO** (all three masks empty) |
| `active_stress.py:64,74` | `clamp(min=0)` on activation / Frank-Starling | θ-free | n/a (operator not in this spec) |

---

## 4. Measured verdict on the affine claim

`probe_affine2.py`, midpoint superposition defect
`d = ‖g(½θₐ+½θ_b) − ½g(θₐ) − ½g(θ_b)‖ / ‖g(θ_b) − g(θₐ)‖`, θ_b = θₐ·(1 + s·u), u ~ U(0,1)ᶜ fixed.

**float64** (the honest measurement — see the float32 warning below):

| s | 1 substep, all | 1 substep, interior | 1 substep, wall band | 1 frame, all | 1 frame, interior |
|---|---|---|---|---|---|
| 0.03 | 6.2e-10 | 5.4e-10 | 3.3e-9 | 1.56e-3 | 1.56e-3 |
| 0.30 | 8.2e-9 | **5.4e-11** | 9.1e-8 | 1.47e-2 | 1.47e-2 |
| 3.00 | 6.5e-6 | **5.4e-12** | 7.2e-5 | 9.7e-2 | 9.7e-2 |

Read it as: the interior one-substep defect is **constant in absolute terms at the fp floor**
(rel ∝ 1/s ⇒ the numerator does not grow with the parameter spread) — the one-substep map is
**exactly affine** in the bulk. All of the residual one-substep non-affinity lives in the wall
band (`defect_frac_in_wallband = 0.99996` at s = 0.3) and is ≤ 1e-4 relative even at ±300 %.

The one-frame defect, by contrast, grows **∝ s** (rel goes 1.6e-3 → 1.5e-2 → 9.7e-2 for
s = 0.03 → 0.3 → 3.0), is identical in the interior and overall, and only 0.13 % of it sits in the
wall band. **That is the F/C feedback**: `mpm_strain.py:41 F = (I + dt·C)·F` makes F θ-dependent
from substep 2 onward, and `mpm_gather.py:88 p.C = new_C` does the same for C. So:

> **the algebraic constraint is exact for ONE SUBSTEP (dt = 2e-4), not for one FRAME
> (dt = 2e-3). Injecting measured positions at every frame leaves a ~1.5 % relative
> non-linearity at a ±30 % parameter spread.**

**FLOAT32 WARNING (a failure worth reporting).** The same test in float32 reports a
3.2 % one-substep "defect" — which is *entirely* float32 quantisation of the absolute position
(`X ≈ 0.5`, `ulp = 6e-8`, whole parameter signal ≈ 1e-6). Its absolute defect is flat at
1.0e-5 across s = 0.03…3.0 (rel ∝ 1/s), the 1/s signature of rounding, not physics.
**Any algebraic fit must difference in float64 or must compute ΔX directly rather than
subtracting two O(0.5) positions.** The float32 *frame* numbers are fine (0.0151 vs 0.0147).

---

## 5. The cheapest correct way to get one column of A

### The three options, priced

One full MLS-MPM substep over 236 000 particles: **13.7 ms** measured (`cuda:1`, float32).
θ has **944** components (472 E + 472 gain), so one "column-per-parameter" sweep is 944 units.

**(i) forward step with a one-hot parameter (finite difference).**
Cost: 1 baseline + 944 perturbed substeps = **945 forward passes ≈ 12.9 s per frame**
(≈ 1.9 h for a 530-frame trajectory), plus a second full sweep if you use central differences.
Correct *in principle* — because the map is exactly affine, a finite difference is the exact
derivative, with any step size. **But it fails in practice at float32:** the response of one cell
at ΔE/E = +10 % is `max ‖ΔX‖ = 2.0e-7 … 1.6e-6` world units (1 substep, measured for cells
7/200/400), versus `ulp(0.5) = 6e-8` — an SNR of **3 to 27**. In float64 it works but costs 2×.

**(ii) reuse the cached Cauchy stress and re-scatter only cell c's particles. ← RECOMMENDED**
The identity that makes it work, verified numerically to 2.6e-8 (`probe_sigma.py`):
```
∂τ_p/∂E_c = 1[p ∈ c] · τ_p / E_p = 1[p ∈ c] · p.sigma_p · |J_p| / p.youngs_p
```
so **one** baseline substep with `store_stress: true` (already on, `mpm_scatter.py:121-131`) hands
you the stress sensitivity for **all 472 E columns at once** — no re-evaluation of
`(F − R)F^T`, no SVD, no polar decomposition. The same trick gives all 472 gain columns from the
cached `H.delta('mpm_particle')` (`∂a_ext_p/∂g_c = 1[p∈c]·a_ext_p/g_c`).
What remains per column is the *linear* propagation, which is **strictly local**: a grid node sees
only particles within 1.5 dx, and a particle reads only its 3×3 stencil, so column c is supported
on cell c's 500 particles plus a one-stencil halo. **Measured support: 1648–1902 particles
(0.70–0.81 % of N) for a one-substep column** (float64; float32 reported only 714–1391 because it
quantised the small responses away). Cost per column ≈ 0.8 % of a substep ⇒
**944 columns ≈ 7.6 forward-pass equivalents ≈ 0.10 s per frame — ~130× cheaper than (i)**, and
exact (no differencing, no cancellation).
The one thing you must get right is the **frozen masks**: record, in the baseline pass,
(a) which grid cells `mpm_grid_update.py:118-119` clipped and which `:121-122` damped,
(b) the `near` mask at `mpm_gather.py:62`, (c) which coordinates `mpm_gather.py:75` pinned; then
apply the same masks (zero / ×0.5 / zero-row) to the perturbation. That is the linearisation, and
it is what makes the column a true directional derivative at a kink.

**(iii) autograd Jacobian.** `torch.func.jvp` on the one-substep map with a one-hot tangent gives
one exact column per call at ≈ 2.5× forward ⇒ **≈ 2400 forward-equivalents ≈ 32 s per frame**,
~250× worse than (ii) in FLOPs. `jacfwd`/`vmap` batching improves wall-clock but not FLOPs and
costs 944× state memory unless chunked. Its virtue is that it gets the frozen-mask bookkeeping
right for free.

**(iv) worth knowing:** if you only want a Gauss-Newton step (not A itself), go matrix-free —
one `jvp` + one `vjp` per CG iteration ≈ 5 forwards; 50 CG iterations ≈ 3.4 s/frame, cheaper than
(i) and it never materialises A (dense A is 472 000 × 944 = 1.8 GB fp32).

### Recommendation

**Use (ii), verified against (iii) on ~5 columns.** Concretely:
1. one baseline substep in **float64** with `store_stress: true`, recording `p.sigma`, `p.youngs`,
   `p.cell_id`, `H.delta('mpm_particle')`, `g.m`, and the three boundary masks;
2. build A as a **sparse** matrix (≈ 944 × 1900 × 2 ≈ 3.6 M nonzeros — trivial) by pushing
   `∂τ/∂E_c` through bspline-scatter → `/g.m` → frozen masks → bspline-gather → `× dt_sub`;
3. assert `‖A[:,c] − jvp_column(c)‖/‖jvp_column(c)‖ < 1e-10` for c ∈ {7, 200, 400} before trusting
   G = AᵀA. Budget: **≈ 8 forward-pass equivalents + 5 jvps ≈ 0.3 s per frame**.

---

## 6. Two things that will sink the downstream effort if not faced first

**(a) F and C are NOT observable from the injected positions.** The proposal says "positions are
observed, so F, R, J and the p2g weights are all KNOWN". The weights are (`mpm_grid.py:74-96`
reads X only) — but **F is not**. F is a path integral, `mpm_strain.py:41 F = (I + dt·C)F`, carried
across every substep, and C is set by the previous gather (`mpm_gather.py:88`). At frame 15 the
state is well away from the reference: `J ∈ [0.73, 1.07]`, `‖τ‖∞ = 20.8`. Two consequences:
- if you inject positions but keep the solver's F/C, then A depends on the *θ-history*, so the
  system is not "algebraic in the parameters" — it is a fixed point, and needs to be solved as one;
- if you reset F = I at injection, the elastic stress is identically zero and the E columns vanish.
  **Neither is what the proposal assumes.** The honest fix is an *independent, data-only* estimator
  of F (e.g. moving-least-squares fit of neighbour offsets against a reference configuration) and
  of C (the MLS velocity gradient) — that estimator does not exist anywhere in this repo and its
  error will propagate straight into b.

**(b) The per-cell columns are far below any plausible measurement noise.** A +10 % change in one
cell's Young's modulus moves the sheet by (max over 236 000 particles, float64):

| | max ‖ΔX‖ (world) | max, in px of the 1024² image | RMS over moved particles (px) | particles moved |
|---|---|---|---|---|
| 1 substep | 2.0e-7 … 1.6e-6 | 0.0002 … 0.0016 | 5e-5 … 5.5e-4 | 1648 – 1902 |
| 1 frame | 6.2e-6 … 3.5e-5 | 0.0064 … 0.036 | 1.0e-3 … 6.8e-3 | 7405 – 9887 |

Cell tracking on this data is optimistically good to ~0.1 px. **A single frame's algebraic residual
therefore carries ~3 orders of magnitude less signal than the noise for a single cell's E.** The
Gram matrix will be numerically fine and physically meaningless unless the residual is accumulated
over many frames — and ideal √N averaging over all 530 frames buys only ~23×, still ~0.8 px-
equivalent short. This should be measured properly (Task C/D) before anyone builds the solver;
it is the number that decides whether the whole approach is viable.

---

## 7. Probes in this directory

| file | what it measures |
|---|---|
| `probe_clamps.py` | which clamps/branches are live over 410 substeps; the once-only facts (masks, `emit_order`, mu/E, p_vol) |
| `probe_affine.py` | boundary breakdown (clamp vs conditional damp vs unconditional damp); float32 affinity; column support; substep timing |
| `probe_affine2.py` | the spread-scaling discriminator (rounding vs kink vs smooth NL), float32 and `--f64` |
| `probe_sigma.py` | that `p.sigma·\|J\|` reproduces τ, and that τ is exactly degree-1 homogeneous in E |

Run: `cd /workspace/Plexus && PYTHONPATH=/workspace/Plexus/src /workspace/.conda_envs/neural-graph-linux/bin/python prototype/cardio_cells/algebraic/<probe>.py --device cuda:1`
