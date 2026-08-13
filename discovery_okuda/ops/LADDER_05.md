# The sheet ladder

**One step per folder, one change per step, one gate per operator.**
Revised 2026-08-10 after the adhesion audit -- see "The plan, REVISED" below, which supersedes the
step numbering in the first half of this file.

A gate is a number with a threshold decided BEFORE the run and a stated consequence if it fails.
`note_sheet.tex` §4 holds the table; this file holds the plan.

## Done

| # | folder | what | verdict |
|---|---|---|---|
| 05a | `05a_sheet` | codim-1 StVK membrane, tethered to the recorded surface | **passes G1-G8** |
| 05b | `05b_plaque` | the plaque as an edge set `cell -> bm` | **G9, G10 pass at 1e-16; G11 marginal, G12, G13 FAIL** |

05a's result that drives everything below: **remodelling is what recovers the standoff.** At `tau_r = 25`
frames the standoff improves 1026x (-1.52e-3 -> -1.48e-6) while `lambda_geo` stays at 3.38, i.e. the
sheet still goes where the tissue puts it and simply stops fighting. Strain stiffening (`beta = 5`) makes
it 8.5x worse, which is the same statement from the other side.

## The blocker both remaining mechanisms share

The sheet has a FIXED mesh. Over the run its area goes x11.1 on the same 5120 triangles: mean edge
length 0.00305 -> 0.0102 box units, areal density to 1/11.1. Secretion cannot add material to a set
whose size is fixed, and tearing cannot remove it. **So the reservoir comes first, and it is one
mechanism for three operators.**

---

## 05c -- the reservoir and remeshing (`bm_refine`)

Nodes and faces are allocated to their maximum at seeding and carry `occ`, exactly as
`engine.py:453`'s `grow_reserve` does for MPM particles. Refinement, secretion and tearing are then all
flips of `face_occ` and never reallocations.

Global 1->4 midpoint refinement, triggered when the mean live edge length exceeds a threshold.
Conforming by construction (every edge is split, so there are no hanging nodes), and vectorised.

**The trap this step exists to avoid.** A newly split triangle must inherit its parent's REFERENCE
frame, or splitting silently resets `lambda` to 1 and the sheet forgets everything it has been
stretched by -- the mesh version of the laundering that made run 130 report 13% of its true stretch.
The four children of a midpoint split are related to the parent by a constant 2x2 map `S_k` in material
coordinates, so `Dm_inv_child = S_k^-1 Dm_inv_parent`, `A0_child = A0_parent/4`, `C0` and `Y2`
inherited unchanged, and `F_geo` is then IDENTICAL for parent and child. That identity is the gate.

| gate | threshold | measured | what it kills |
|---|---|---|---|
| G14 refining an UNLOADED sheet changes `lambda` | `< 1e-6` (isotropic: sqrt(eps), see G1/G2) | **6.2e-9** | a reference frame rebuilt from the current shape |
| G15 refining a LOADED sheet changes `lambda`, area, energy | `< 1e-12`, `< 1e-9`, `< 1e-6` rel. | **2.3e-14, 0.0, 0.0** | the same, but where it actually matters |
| G16 no hanging nodes: every live interior edge has exactly 2 live faces | exact | **0 bad, 0 rim** | non-conforming refinement |
| G17 edge length stays inside a band as R triples | `[0.8, 1.7]` x seeded | 05c | refinement that never fires, or fires forever |
| G18 the run's own measures are unchanged by refinement | `lambda_geo` to `< 1%` vs the fixed-mesh 05a | 05c | refinement that changes the physics it was meant to resolve |

G14 sits at 1e-9 rather than 1e-14 for the same reason G1 and G2 do: on an isotropically stretched
triangle the discriminant `sqrt(tr^2/4 - det)` is a cancellation, so it carries `sqrt(eps)`. The loaded
case G15 has well-separated eigenvalues, nothing cancels, and it comes back at 2.3e-14 with the total
area and the total elastic energy **bit-identical** before and after the split. That is the strongest
form the gate can take: refinement is not approximately invisible to the mechanics, it is exactly
invisible.

Note the midpoints are placed on the CHORD, not projected onto the sphere. Projecting would smooth the
surface and change `lambda`, which would break G14 by construction -- the smoothing has to come from the
dynamics, not from the remesher.

## RETIRED: the old 05d -- tearing (`bm_degrade`). Ran, passed G19-G23, archived.
_Its result is kept in `note_sheet`; the step number is reused below for the adhesion clutch._

A face dies when a criterion is met; a hole is then a region with no live faces and a free rim.
Criterion candidates, in order of preference: areal density `rho` below critical (the biology: a sheet
fails where it is not resupplied, which couples the tear to 05e's mass balance), the second Piola
stress, or `lambda_el` past a break strain.

The archived MPM sheet separated when it thinned past the grid's support, which `LADDER.md:24` flagged
at the outset -- "a tear that moves when the grid is refined is numerical" -- and the refinement
cross-check (run 90, `n_grid` 64 -> 128) was planned and never ran. It also ran on a sheet 1/8 of a grid
cell thick carrying the stroma's `youngs` 15 instead of the 400 its spec declared. So the visually
convincing tearing is, as of today, untested.

| gate | threshold | what it kills |
|---|---|---|
| G19 **the tear is in the same place at the same `lambda` under refinement** | position within one seeded edge length, onset `lambda` within 5% | a tear set by the discretisation -- the test MPM never passed |
| G20 onset scales with the CRITERION, not with element size | onset `lambda` vs threshold, monotone; vs `subdiv`, flat | the same, from the other side |
| G21 a hole stays open; no face is re-woken by the remesher | exact | a remesher that heals what a mechanism killed |
| G22 control: criterion above the maximum load -> zero faces die | exact | run 127's null, which was a threshold above the load |
| G23 momentum is still conserved with a hole present | `< 10 eps` | a rim that leaks force |

## The audit, 2026-08-10, and what it changed

An external review of `note_sheet` made one correction and one demand, and both are adopted:

- **A remesh is NUMERICAL ONLY.** It changes the triangulation and nothing else -- not the physical
  surface, not the material state, not the mass. Calling refinement, secretion and tearing "the same
  mechanism seen three times" (the first draft did) is exactly the confusion that lets a discretisation
  choice be read as a result, which is the error the MPM sheet made when its tearing was set by `dx`.
  They share the `occ` reservoir as an implementation substrate and nothing else.
- **What a remesh must carry, and it is now gated:** `C0` (the remodelling state, intensive, inherited
  identically), `rho` (intensive -- inherited; the MASS `rho*A0` divides with `A0`, so
  `sum_f rho_f A0_f` is invariant), and the plaque attachments (node indices survive, but the plaques
  must be RE-SEEDED to hold the areal density `Sigma^-2`).
- **Keep secretion, remodelling and degradation as the biological operators**; remesh is maintenance of
  the discretisation when the area changes.

Two defects follow immediately and are 05e's content: `rho` is currently a DERIVED quantity
(`rho0 / J`) and not a state, so nothing conserves mass across a split; and `plaque_seed` takes a
FRACTION OF NODES, so the plaque areal density falls 4x at every refinement.

And one gate was written wrong. G18 said "the refined run must match the fixed-mesh run to 1%"; they
came back 1.12% apart. **The fixed mesh is not the truth** -- a coarse triangle chords a curved,
stretching surface and under-reports its stretch, so refinement moved the fidelity against the APPLIED
stretch from 0.9873 to 0.9985 and the standoff by 15x. G18 as phrased would have rejected an
improvement; it is restated against the applied stretch.

---

# The plan, REVISED 2026-08-10 after the adhesion audit

## What changed, and why the old 05d-05f are gone

The units work forced a recount and the recount forced a rewrite. A `plaque` was one integrin
NANOCLUSTER -- 20-50 integrins, 35-100 nm across -- and clusters sit at a measured ~555 nm
centre-to-centre spacing. On this sheet that is **~1.03 million clusters against 40,962 nodes: 25
clusters and ~3000 integrins per NODE.** Above one object per element the discrete description is the
approximation and a density is exact, so 05b's 1166 discrete plaques were not a sparse sample of
adhesion, they were the wrong object by a factor of 883.

**The new scheme (`note_sheet` §2.3):** free receptor `N_f` on the CELL's basal face (well-mixed, so a
per-cell ODE and NO diffusion PDE), bound bonds `N_b` on the `plaque` edge (one per sheet node, from
the `bm_contact` map, carrying a bond DENSITY), ligand on `bm_face` (it is already rho). Traction
proportional to `n_b`, reaction unchanged, Bell's `k_off(f)` for unbinding.

**It closes three open defects at once,** which is why it replaces steps rather than adding one:
G12 (standoff 3.43 l0 -- 96% of nodes were unanchored and the sheet sagged between plaques),
G13 (slip -- a fixed barycentric anchor is a tangential PIN; slip IS bond turnover), and rupture
as a clock (a fixed threshold on a rising load is reached by everyone eventually; `k_off(f)` gives a
steady-state bound fraction that depends on LOAD).

**The old 05d/05e/05f are retired from the plan.** Their results stand and are archived under
`log/okuda_ECM/_archive_05def/`, and what survives from them is carried in `note_sheet`:

| retired | what it was | what survives |
|---|---|---|
| old 05d `tear` | tearing on a stretch criterion | **G19 passed**: the tear is at the same lambda (0.52% apart) at 4x resolution -- the test MPM's version never faced. Kept. |
| old 05e `conserve` | the reservoir's conservation | **G18b-e passed**, mass bit-identical; `bm_contact` was found here. Folded into 05c. |
| old 05f `secrete` | the mass balance | **G24-G26 passed, G27 FAILED** (turnover dominates dilution 4.2x, contradicting note S9). The `bm_assemble` lateral-spreading finding. Becomes the new 05f. |

---

# The sequential plan, 05d -- 05n

One step at a time, each read before the next is written. Every step names the result that would change
the plan.

| # | folder | the question | headline gate | what a surprise would change |
|---|---|---|---|---|
| **05d** | `05d_adhesion` | **the clutch: receptor on the cell, bonds on the edge** | G30 receptor conserved; G31 bound fraction vs LOAD | if the density model does not reproduce the discrete one at high density (G33), the continuum step is wrong, not the count |
| 05e | `05e_slip` | does bond turnover produce slip? | G32 slip monotone in `k_off`, zero at 0 | if a turning-over bond still pins, adhesion needs a sliding contact, not a relation |
| 05f | `05f_secrete` | the mass balance and the supply-driven tear | G24-G27 (re-run on the new adhesion) | G27 already failed once; the tau sweep says where the note's claim holds |
| 05g | `05g_degrade` | proteolysis: a hole only where protease is | G27' hole follows dose, not frame | if onset moves with resolution, the criterion is wrong |
| 05h | `05h_assemble` | `bm_assemble`: lateral polymerisation of laminin | spreading length vs Sigma | it is currently a Jacobi stand-in; if the length matters, it needs to be a real operator |
| 05i | `05i_bending` | can a hole have a shape? | rim curvature converges under refinement | without it, holes are polygonal artefacts |
| 05j | `05j_flip` | edge flips under anisotropic stretch (reuse the tested T1) | triangle quality floor | if flips are needed early, Rivara bisection comes forward |
| 05k | `05k_vertex` | the real vertex-model epithelium | momentum on the re-meshed tissue | `N_f` becomes per real cell; T1/divide must not break the map |
| 05l | `05l_fibril` | is the stroma loaded THROUGH the sheet? | stroma displacement, sheet present vs absent | if not, Fig 3b of the design note is wrong |
| 05m | `05m_three` | all three bodies, one pass | wall clock per frame | first place the cost could stop being affordable |
| 05n | `05n_inject` | 04 with a membrane | 04's own measures | the deliverable |

## 05d -- adhesion as a bond density (`plaque_bind`, `integrin_turnover`)

The step the recount forces. Three states on three entities (`note_sheet` §2.3, Fig. 3):

    cell.N_f      free receptor on the basal face   dN_f/dt = s_i - N_f/tau_i - bind + unbind
    plaque.N_b    bound bonds, one edge per node    dN_b/dt = k_on n_f rho_L (1 - n_b/n_max) - k_off(f) N_b
    bm_face.rho_L ligand                            already there, it is rho

with `k_off(f) = k_off0 exp(f/f_b)` (Bell 1978) and traction proportional to `n_b`.

**No diffusion operator, and that is a result rather than a simplification.** The plasma membrane is
continuous within a cell and interrupted at junctions, so integrins do not diffuse between cells. Within
one cell, sqrt(4 D t) = 15.5 um at D = 0.1 um^2/s over a 600 s frame, against a ~10 um cell: the basal
face is WELL MIXED at our timestep. That removes the Laplacian and the `dt D / h^2 < 1/2` bound, which
would have cost ~14 substeps and, unlike the sheet's rate bound, would have TIGHTENED at every
refinement. The assumption is stated, not silent: resolve a cell with more than one face and it returns.

| gate | threshold |
|---|---|
| G30 receptor conserved: `N_f + sum N_b` moves only by `s_i`, `1/tau_i` | exact, and across a remesh |
| G31 bound fraction depends on LOAD | monotone in applied stress, plateaus; NOT a ramp in frame |
| G32 slip rate monotone in `k_off`, zero at `k_off = 0` | any separation (this is G13, at last) |
| G33 the discrete-plaque model converges to the density model as `Sigma^-2` rises | within 5% |
| G12' the standoff returns to `l0` | `l/l0` in [0.9, 1.1] |

**If G33 fails, stop.** A density that does not reproduce the discrete model in the limit where the
discrete model is valid is not a coarse-graining, it is a different model wearing its name.



---

# Gates for the field pair (05g), and why each one exists

Every gate below was written AFTER a specific failure, and each names the failure it would have caught.
A field is harder to gate than a force: a force that is wrong moves something visibly, whereas a field
that is wrong just looks smooth.

## Certification, before any physics (`protease_ops.selftest()`)

A sphere has eigenfunctions, so a surface Laplacian on one can be checked against a closed form rather
than against plausibility -- the treatment the strain measure got in G1-G4 and the field had never had.

| gate | threshold | measured |
|---|---|---|
| **G39** `lap Y_l = -l(l+1)/R^2` for l = 1, 2 | `< 2%`, and the SAME error for both l | **0.65% and 0.60%** -- consistent across l, i.e. a discretisation error and not a bug |
| **G40** the solve conserves the field with no source or sink | `< 1e-12` | **1.3e-15** |
| **G41** a point source with a sink decays as `exp(-r/sqrt(D/k))` | `< 5%` once resolved | **14%** at `sqrt(D/k) = 0.05` against a mean edge of 0.076 -- see G42 |
| **G42** the length scale is RESOLVED: `sqrt(D/k) >= 2h` | hard requirement | G41's 14% is this gate failing, not the operator being wrong. A field whose own structure is finer than the mesh is not being solved, it is being averaged. |

## Structure: the three failures that shipped

These are the gates that would have caught what the first 05g did, and each is stated so that the
failure returns a number rather than a picture that looks fine.

| gate | threshold | why it exists |
|---|---|---|
| **G43** the two fields are not the same field: `corr(c_M, c_T) < 0.99` AND `max abs(c_M - c_T)/mean(c_M) > 0.1` | both | The first run had **corr = 1.000000 and a maximum difference of exactly 0**. Equal sources, equal D and a 1:1 sink make the two fields solve the same equation, so they were one field drawn twice. |
| **G44** the field HAS structure: `CV(c_M) > 0.05` | | The first run had **CV = 1.9e-16** -- uniform to machine precision, and not because it diffused: a spatially uniform source is uniform at D = 0 too. The run could not test the diffusion argument it was built to test. |
| **G45** the structure comes from the SOURCE, not the initial condition: two different initial conditions reach the same steady state | `< 1%` | This is the question "is it just initialisation?" turned into a measurement. It is NOT initialisation: subtracting the two equations, the 1:1 reaction cancels identically, so `d(c_M - c_T)/dt = D lap (c_M - c_T) + (s_M - s_T)` and the difference has no source. Any initial difference diffuses flat. Only a SOURCE difference can separate them. |

## The mechanism: does the field do anything the source could not

| gate | threshold | why it exists |
|---|---|---|
| **G46** the breach is LARGER than its source, and the excess tracks `sqrt(D/k)` | ratio > 1, and monotone in `sqrt(D/k)` | The first run's holes were **exactly the 317-face MT1 stencil** -- killed in one pass, no propagation. The breach had no emergent size; it was a rubber stamp. If the halo adds nothing, the diffusible species is decoration and the model should say so and delete it. |
| **G47** control, no sink (no TIMP): the field fills the sheet and the breach loses its size | breach radius >> with sink | Proves the SINK is what sets the length, not the diffusion. |
| **G48** control, no protease at all: nothing dies | exact | Passed already: 0 dead. |
| **G49** a hole opens only where protease REACHES -- deaths beyond the source footprint but within a few `sqrt(D/k)` | no deaths at > 5 `sqrt(D/k)` | The tethered-only arm passed the weaker version of this (317/317 under MT1). The stronger version is what distinguishes a halo from a stencil. |

## The mesh moving underneath it

The field lives on faces, and faces are created by refinement and destroyed by tearing. Both are gates,
and they are the G18b treatment applied to a field.

| gate | threshold | why it exists |
|---|---|---|
| **G50** `sum_f c_f A_f` invariant across a 1->4 split | exact | A split must inherit the parent's concentration (intensive), so the integral is conserved. The mass gate caught this class of error for material; the field has the same exposure. |
| **G51** field on a torn face is ACCOUNTED, not dropped | reported, and the balance closes | A dying face carries its protease with it. Silently discarding it is a sink nobody declared -- which is exactly the kind of thing that makes a rate look right for the wrong reason. |
| **G52** the steady profile is mesh-independent | `< 5%` between subdiv 4 and subdiv 4 + one refinement | The G19 treatment for the field: a length scale that moves when the mesh is refined is the mesh's, not the model's. |

---

# 05g / 05h results, 2026-08-11

## 05g -- proteolysis: only the tethered enzyme can make a hole

| arm | dead | under MT1 | elsewhere | breached |
|---|---|---|---|---|
| both arms | 317 | 317 | 0 | 6.2% |
| soluble only | **0** | 0 | 0 | 0% |
| tethered only | 317 | 317 | 0 | 6.2% |
| no protease | 0 | 0 | 0 | 0% |
| **no inhibitor** | **5120** | 317 | **4803** | **100%** |

The last row is the field argument as an experiment: remove TIMP and the sheet dissolves entirely,
4803 of the deaths nowhere near an MT1-expressing cell. **A soluble protease has no length scale
unless something quenches it.**

**Two defects the runs found:** (1) neither field had a CLEARANCE term, so TIMP accumulated without
bound and no steady state existed -- and the `sqrt(D/k)` that justifies solving the field IS that k;
(2) a five-point cluster sweep of the source ratio over 100x returned *identically* 317 dead at every
point, because the direct tethered cut was set 350x larger than the soluble arm could contribute. The
sweep axis had no effect on the answer.

## 05h -- TIMP-2 as adaptor AND inhibitor: the bell, confirmed

Closed form (Karagiannis & Popel 2004; Sato & Takino 2010):
`act ~ T^2 c_pro (c_T/K)/(1 + c_T/K)^2`, peaking at `c_T = K`.

| TIMP source | peak activation | breach | control (pure inhibitor) |
|---|---|---|---|
| 1e-5 | 3.42e-2 | 1.6% | 0% |
| 3e-5 | 5.50e-2 | 6.2% | 0% |
| 1e-4 | 8.42e-2 | 6.2% | 0% |
| **3e-4** | **8.44e-2** | 1.6% | 0% |
| 1e-3 | 6.19e-2 | 0% | 0% |
| 3e-3 | 2.38e-2 | 0% | 0% |
| 1e-2 | 7.53e-3 | 0% | 0% |
| 3e-2 | 2.55e-3 | 0% | 0% |

| gate | result |
|---|---|
| G54 breach/activation non-monotonic | **PASS** -- a clean bell over two decades |
| G55 peak at `c_T ~ K` | **MARGINAL**: peak at 2.03 K against a 2.0x threshold. Reported as a fail, not rounded down; the peak is broad and the sweep is half-decade. |
| G56 the pure-inhibitor control is monotone | **PASS** -- so the bell is the mechanism, not the tuning |
| G57 every field steady | **PASS** -- drift 0.0 after the clearance fix |

**Two measurement errors corrected before the curve could be read**, both worth recording because
neither was a modelling error: activation had to be made RATE-LIMITING (`k_act` = 0.5, not 4e4 --
at large `k_act` the whole zymogen pool converts every frame whatever TIMP does and saturation erases
the dose response); and the rate must be read WHILE THE SUBSTRATE STILL EXISTS, since at the TIMP
levels where the mechanism works best it destroys every MT1-bearing face, and a late-window average
then reported exactly 0.0 at the peak and displaced it by 7x.
