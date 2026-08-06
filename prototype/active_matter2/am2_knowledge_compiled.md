# active_matter2 — COMPILED KNOWLEDGE (self-contained recipe layer)

Distilled from 3×10 agentic batches (knowledge_fig{1,2,3}.md). This file is
SELF-SUFFICIENT: given only this + the worker `am2_job.py`, you can regenerate any
state of Ziepke et al. (2022) Figs 1–2 with NO conversation history. Paper: communicating
active matter — self-propelled agents emit + chemotax an excitable chemical `c`.

## How to run (the worker interface)
`python am2_job.py --outdir archive/<name> --kind <agent|hydro> --device cuda:0 <--flag val ...>`
Writes `panel.png` (top: orientation-coloured field; bottom: `c`) + `progress.txt`
(P=global polar order, Nc=cluster count, contrast, signal). Recipes below are the flags.

- **agent** (Fig 1, microscopic): glide+polar_align+chemotax+relay(excitable)+adapt+repel.
  Defaults: n8000 move_speed0.006 radius0.03 res200 frames1000 beta0.16 c_th-0.001 sigma1.2
  eps0.05 diffuse0.16(=Dc) decay0.02(=alpha) gamma0.15(align) align_noise0.04 omega0.38(chemotax)
  repel0.015 r00.010. Extra medium knobs: c_base, seed_spiral, rf_tau, rf_gain, rf_th (see VORTEX).
- **hydro** (Fig 2/3, continuum ρ,p,s,c). Base preset `fig`: sigma0.7 delta0.6 chi0.5 Drho0.5
  Dp0.6 Dc1.1 Q0.5 alpha0.42 beta0.6 eps0.045 c_th-1 v00.6 omega1.8 rho0 1.2. Overridable flags:
  v0 omega sigma alpha beta eps Dc chi Q delta Drho Dp rho0 L N nsteps seed mode(snapshot|coarsen).

## AGENT recipes — Fig 1 (5/6 states SOLVED; judge by MORPHOLOGY, P is a weak global metric)
| state | flags (delta from defaults) | morphology |
|---|---|---|
| streams     | `--gamma 0.35 --omega 0.38` | directed rivers; c de-foams to sinuous wave-front lanes |
| ring-streams| `--gamma 0.50 --omega 0.45 --diffuse 0.22 --decay 0.014 --eps 0.035 --n 9000` | closed azimuthal loops, annular c fronts |
| active-droplets | `--gamma 0.55 --omega 0.30 --eps 0.012 --n 4500 --move_speed 0.002` | compact blob, coherent core + migration tail |
| polar-bands | `--gamma 0.42 --omega 0.06 --move_speed 0.007 --beta 0.08 --decay 0.03` | straight coherent travelling stripe, empty bg |
| aggregation | `--gamma 0.0 --omega 0.85 --marker dot` | Keller-Segel coarsening; separated bright blobs |
| **vortex (proxy)** | `--move_speed 0.002 --omega 0.55 --n 20000` | field of small FILLED rainbow pinwheels (each a mini rotating disk) |
| **spiral vortex** | proxy + `--c_th 0.05 --c_base 0.05 --seed_spiral 1 --rf_tau 40 --rf_gain 0.10 --rf_th 0.5` | sustained rotating spiral (see VORTEX below) |

## HYDRO recipes — Fig 2 (all 6 states placed)
| state | flags | note |
|---|---|---|
| vortices (lattice) | `--v0 0.6 --omega 1.8` | +1 polarization defects (HSV wheels) + spiral/target c. Fewer/larger: `--Dc 2.2` |
| droplets (few-large)| `--v0 1.0 --omega 1.0 --rho0 1.02 --chi 0.0` | isolated compact blobs; rho0 sets COUNT |
| droplets (many-small)| `--v0 1.0 --omega 1.8 --chi 0.0` | chi→0 = coherent single-hue; high v0 fragments |
| rings | `--v0 0.6 --omega 1.4 --chi 0.0 --Dc 4.0 --rho0 1.05` | FRAGILE box: Dc≈4 exact (3=arcs,5=no pattern); rho0-down = more sites |
| bands (silent) | `--v0 2.0 --omega 0.0 --rho0 1.10 --Drho 0.15` | clean parallel stripes. Alt network route: `--rho0 1.05 --Dp 0.2` |
| bands (signalling) | `--v0 2.0 --omega 0.5 --rho0 1.10 --Drho 0.15` | silent recipe + omega ON (c co-travels on stripes) |
| streams | `--v0 1.8 --omega 0.6 --rho0 1.05 --Dp 0.2` | low-Dp network + weak omega + high v0 → directed rivers |

## LEVERS (causal — what each knob DOES)
AGENT: **gamma = master morphology axis** (gas→streams→loops) and c de-foam. **move_speed(v0)
DOWN = fill** (the hollow is a MILL; slow agents fall into the well → disk/blob). **omega** =
chemotactic collapse + nucleation COUNT (up→more,smaller). eps localizes the well (eps=0 →
saturated fat channels). c_th/c_base = medium excitability/ignition; diffuse = wavelength.
HYDRO: **omega = condensation switch** (sharp onset ~1.5, ~v0-independent; >1.5 condensed). **v0**
= motility+sharpener (up→more/smaller vortices; straightens bands/streams; SATURATES ~2.0, stable
to 2.5). **rho0** = band onset + nucleation count (>1.05 bands; down→more sites). **chi** =
pinwheel↔coherent-blob (→0 for droplets & ring precursors). **Dc** = c wavelength (ring optimum
~4, NON-monotonic). **Dp/Drho DOWN** = release the band modulation (growth v0·sigma·p0 vs damping
(Dp+Drho)q²). **alpha** = c decay (up→fewer/larger compact wells).

## THE VORTEX — the one hard state (KEY FINDING, Fig-1 loop batches 5–10)
Reproducing a single big rotating **spiral vortex** with the AGENT model is a MEDIUM problem,
not a parameter problem:
- Transport knobs are exhausted: consolidation RE-HOLLOWS (mill), omega-up FRAGMENTS, slow-decay →
  LABYRINTH, v0-up → filaments. The honest proxy is a FIELD OF MINI-PINWHEELS (fill works, scale fails).
- ROOT CAUSE: recovery `s` (Eq 5) is carried by MOBILE agents → they advect into their own emitted
  front and scramble the refractory tail → the medium cannot HOLD a phase singularity → a seeded
  broken front (`seed_spiral`) WASHES OUT (seed == no-seed at the final frame).
- THE FIX (the model needs a continuum inhibitor): the `refract` op maintains a SPACE-FIXED per-voxel
  refractory FIELD `fld._rf` (∂ₜrf = rf_gain·Θ(c−c_th) − rf/rf_tau; relay blocked where rf>rf_th),
  turning the single-scalar relay into a FitzHugh-Nagumo excitable medium. A broken front then leaves
  a fixed wake the next front can't re-invade → it winds into a SUSTAINED spiral. **rf_tau = spiral
  core size.** Default-off (rf_th=2.0). NB this op was authored+reviewed in the loop but may be
  UNVERIFIED (in-loop exec was gated) — running the `spiral vortex` recipe verifies it.
- In HYDRO the excitable c (c_th<0, omega>1.5) already makes spiral/target waves around each aster
  core — so `--kind hydro --v0 0.6 --omega 1.8` is the reliable spiral-vortex field.

## STATUS
SOLVED: streams, rings, droplets, bands(silent+signalling), aggregation — both agent & hydro.
OPEN/PROXY: a single large agent spiral vortex (mini-pinwheel proxy; `refract` continuum route to verify).
Fig 3 (coarsening): the droplet→stream→vortex cascade + v0-graded endgame are chemotaxis-INDEPENDENT
(pure Toner-Tu flocking+pressure+advection at omega=0); chemotaxis only controls PINNING (crystal vs not).
