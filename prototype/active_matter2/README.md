# active_matter2 — communicating active matter

A Plexus reproduction of **Ziepke, Maryshev, Aranson & Frey, "Multi-scale
organization in communicating active matter", Nat. Commun. 13:6727 (2022)**:
self-propelled agents that **emit and chemotax toward an excitable chemical signal**,
self-organizing into a hierarchy of collective states — directed streams, ring-like
streams, active droplets, vortices (spiral-wave sources), and polar bands.

## The model (their Eqs 1–5)

```
dr_i/dt   = v0 n_i + Σ_j f_ij                                        (1) self-propel + repel
dφ_i/dt   = −Γ Σ_j sin(φ_i−φ_j)/r_ij + ω sin(φ_c−φ_i) + ξ_i          (2) align + chemotax + noise
∂_t c     = Dc ∇²c − α c + β Σ_i f(|r−r_i|)(1−s_i)Θ(c−c_th)          (3-4) excitable emission
ds_i/dt   = ε (c − s_i)                                              (5) internal (refractory) state
```

## Operators

Reused from `src/plexus/operators` (the framework of `paper/plexus.tex`):
`glide` (v0 n_i), `diffuse` (Dc∇²), `decay` (−αc), `radius_graph` (the interaction
graph), and the `grid` ScalarField (one shared channel `c`, `components: 1`).

**New** (in [`am2_ops.py`](am2_ops.py), all dimension-generic — 2D & 3D):

| operator      | paper term                     | what it does |
|---------------|--------------------------------|--------------|
| `polar_align` | Eq 2 term 1 + ξ                | heading Vicsek alignment (1/r-weighted) + angular noise |
| `chemotax`    | Eq 2 term 2 `ω sin(φ_c−φ)`     | rotate the heading toward the chemical **gradient** |
| `relay`       | Eqs 3–4 `β(1−s)Θ(c−c_th)`      | excitable Schmitt-trigger emission, Gaussian source |
| `adapt`       | Eq 5 `ε(c−s)`                  | per-agent internal state (refractoriness) |
| `repel`       | `f_ij`                         | short-range hard-core repulsion (first-derivative) |

The stock `alignment`/`chemotaxis` are *velocity* laws; these steer the unit
**heading** the heading-kinematic sibling `glide` moves along (like slime `sense`).

## The six collective states

One nominal agent-based spec per state (`specs/`), plus the continuum theory:

| spec | regime | paper Fig. 1 state |
|------|--------|--------------------|
| `am2_streams`      | moderate ω, moderate β        | directed particle streams (g,l) |
| `am2_rings`        | strong Γ + ω, slow ε, fast Dc | ring-like "whispering-gallery" streams (e,j) |
| `am2_droplets_v2`  | strong ω, medium-range c, low ρ | active droplets — compact motile blobs (f,k) |
| `am2_vortex`       | strong ω + β, slow ε          | vortices with trapped spiral waves (h,m) |
| `am2_bands`        | strong Γ, weak ω              | Vicsek polar bands (i,n) |
| `am2_aggregation`  | Γ = 0, pure chemotaxis        | aggregation / coarsening (Suppl. Note 2) |
| `am2_vortex_3d`    | 3D, strong ω + β              | 3D aggregation (dimension-generic check) |

(`am2_droplets` is the first, gas-like attempt; `am2_droplets_v2` is the
figure-reproduction variant that condenses into blobs.)

**Hydrodynamic PDE model** (Eqs 6–9) — [`am2_hydro.py`](am2_hydro.py), a standalone
periodic finite-difference integrator of ρ, p=(pₓ,p_y), s, c. Presets `nominal` /
`vortex` (a polarization-vortex lattice with chemical peaks) / `bands`.

## Run

```bash
# repo root, conda env + PYTHONPATH=src
python prototype/active_matter2/run_am2.py                 # all agent specs
python prototype/active_matter2/run_am2.py streams vortex  # substring filter
python prototype/active_matter2/am2_hydro.py --all         # the 3 hydrodynamic presets
python prototype/active_matter2/am2_figure.py              # rebuild the Fig.1 montage
```

All data is written **inside this prototype** at `data/graphs_data/active_matter2/<name>/`
(gitignored): `movie_cell.mp4` (oriented triangles), `movie_chemical.mp4`,
`movie_overlay_chemical.mp4` (agents over the field), the 3D turntable, `movie_hydro.mp4`,
and `fig1_reproduction.png` (the paper-style montage: orientation over chemical, one
column per state). Override the location with `AM2_DATA_ROOT`.

## Notes

- Emission bootstraps with `c_th < 0` (always relay, gated by the refractory `s`),
  giving self-sustained excitable/oscillatory kinetics from `c ≡ 0`. A genuine
  quiescent threshold (`c_th > 0`) would need an initial seed pulse in the field.
- Periodic (toroidal) world: agents wrap and the chemical field wraps with them.
