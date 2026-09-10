# Epithelium + basement membrane + extracellular matrix: what was done, and the road to one simulation

Written 2026-09-10 from an audit of `log/okuda_ECM` (rungs 0u..08b), the rig scripts in
`discovery_okuda/ops/`, the operator registry in `src/plexus/operators/`, the paper
(`paper/plexus2.tex`), the minisite page "Cell tissue + basement membrane + extracellular matrix",
and one comment from a biologist reading that page. Numbers are quoted from the archive's own
`metrics.json` / `spec.yaml` files; the file paths are given so each can be checked.

---

## 1. The big picture

**The target** (paper, `plexus2.tex:1133`): a growing epithelial tissue that deforms a basement
membrane embedded in an extracellular matrix, with molecular mechanisms regulating the interface.
Three bodies, three physics, one spec.

**What the archive achieved** (June to September 2026): each body and each interface was built and
measured on its own ladder rung, and the payoff rung (`08b_s1_finger`) showed that *a hole in the
basement membrane decides where the epithelium grows out*: bud excess +0.975 on the hole's axis,
-0.055 off-axis, and the mirror control with the hole rotated 180 degrees gives +0.956 on the
rotated axis (`discovery_okuda/BUDDING_08.md:11-14`; noise floor +-0.04 over 80 tissues).

**Why it is not yet one simulation.** Every body in the archive is a *replay* of another body's
recording (`discovery_okuda/FUSED_09.md`): the epithelium is a cached trajectory
(`log/okuda_ECM/_tissue/cellfix_B_new_*.npz`) replayed frame by frame; the membrane is a Python rig
(`Rig05b -> Rig08a` in `discovery_okuda/ops/`) that never entered a spec; the matrix is a
`traj.npz` produced against the *original* reference tissue. So nothing reacts to anything: the
matrix pushed out by the tissue cannot push back, the membrane's pull on the tissue is computed and
thrown away (`test_06_three_bodies.py:424`, `fes` used for nothing but a momentum residual), and
a bud carved into the tissue is invisible to a matrix whose cavity sits ten units away (the
measured error, `FUSED_09.md`). The biologist's remark on the minisite is the same finding read
from outside: "the epithelium grows almost identically across all three conditions, so currently
there is no rule set for mechanical constraint to influence growth." There is not, because no
arrow points back at the tissue.

**What changed this week.** The apico-basal tissue is now a live engine model with a working point
under regression protection (`cvd2_adder_tension`, `tests/regression/`), and the tissue + matrix
pair runs live in one spec (`config/tissue/spheroid_ecm_ab.yaml`: 801 frames, the tissue tracks
its reference exactly, the matrix is pushed out to 0.95 of the shell radius). That is the first
rung of the archive rebuilt without a replay. The membrane and the chemistry are next.

**The road, in one line each** (details in section 4):

| milestone | what it delivers | the test that closes it |
|---|---|---|
| M1 tissue + membrane, live | `bm_solve`: the rig's sheet, plaques and refinement as ONE operator on the vertex set, returning the pull on the tissue | `kn: 0` control reproduces the uncoupled tissue's fingerprint; plaque momentum residual < 1e-12 |
| M2 matrix -> tissue | `mesh_reaction`: the contact's `VERTEX_FORCE` returned to the vertices; MPM substeps inside the tissue frame | momentum residual; growth slows measurably under a stiffer matrix (the missing "mechanical constraint") |
| M3 a fibrous matrix | a transversely isotropic stress term reading `ecm_seed`'s fibre directions | 02i's stiffening ratio 1.13 -> 10-100; 04's cos2 above the affine prediction |
| M4 the protease chemistry as operators | six operators on the membrane's face set: secrete, diffuse, activate, inhibit, degrade, tear | archived gates G24-G27, G53-G57 re-run as regression fingerprints |
| M5 the interface regulates growth | `bm_sense` -> `cell.chem` -> `cell_grow` on the LIVE tissue | 08b's bud (+0.975 on axis, +-0.04 off) reproduced with a reacting matrix |
| M6 the molecular corset | anisotropic junction tension in the apico-basal energy, circumferential fibres in the matrix | an elongating spheroid: aspect ratio > 1.3 against 1.02 for the isotropic control |

---

## 2. The top view: entities, hierarchy, operators

### 2.1 The entities and their hierarchy

The paper's own words (`plexus2.tex:172`): "a tissue contains cells, a basement membrane and an
extracellular matrix. Each of those is a distinct set with its own state and its own operators, and
all of them share a parent." Containment is a function `pi` from children to parent, and a
many-to-many relation (a vertex in three cells, a plaque between a cell and a membrane face) is
"a set `R` together with two functions `R -> A` and `R -> B`" (`:252-268`), which is what the
half-edge table and the plaque edge set are. The minisite states the realised form: *three levels,
four sets; aggregates up, broadcasts down; no set gives up its own representation to take part.*

    level 2   tissue
                |
    level 1   epithelium ........ basement membrane ........ extracellular matrix
              cell  (set)         bm_face (set)              mpm_particle (set)
              vertex (set,        bm_node (set)              mpm_grid (field)
                mesh: half_edge)  proteins: state blocks
                                    on bm_face and per cell
                |                     |
    level 0   half_edge (relation   plaque (relation
              vertex -> cell)       cell/vertex -> bm_face)

| set | representation | own state | in `src/` today |
|---|---|---|---|
| `vertex` | apico-basal vertex mesh, `mesh: half_edge` | pos, sep (apical-basal separation), vel | yes (`cell_mechanics[model: apicobasal]`, `vertex_ops.py:4793`) |
| `cell` | contained in the tissue through the half-edge relation | area, volume targets, cycle, `chem` (morphogens, MT1-MMP, receptors) | yes |
| `bm_node` / `bm_face` | St Venant-Kirchhoff membrane, one constant-strain triangle per face, massless, overdamped; refines from a reservoir as the tissue grows | node pos; per face: rest metric, density rho, proMMP2, MMP2, TIMP2, TIMP3, ligand | **no**: lives in `discovery_okuda/ops/bm_ops.py`, `bm_refine_*.py`; `src/plexus/operators/membrane_ops.py` holds the abandoned MPM-particle membrane (fidelity 0.127 against 0.994, `05a_sheet/metrics`) |
| `plaque` | edge set cell -> bm_face: a normal spring at rest length 0.3 T plus tangential friction; later a bond density with receptor and ligand | l0, bound bonds N_b, per-plaque lineage | **no**: `discovery_okuda/ops/adhesion_ops.py` |
| `mpm_particle` | MLS-MPM fibre matrix (20 particles per strand), fixed-corotated | pos, vel, F, C, type | yes (`mpm_ops.py`, `seed_ecm` at `:4803`) |
| `mpm_grid` | the matrix's background grid, a field | mass, momentum | yes |

### 2.2 The operators, by arrow

Arrows are what the roadmap is about, so the operators are listed by which way they point.

| arrow | operator | kind | status |
|---|---|---|---|
| epithelium, within | `cell_mechanics[apicobasal]`, `cell_grow`, `cell_divide`, `edge_flip`, `cell_die`, `junction_myosin`, `medioapical_myosin` | lateral / structural / rewire | live, protected by `tests/regression` |
| epithelium -> matrix | `mesh_contact` (ICFEMP particle-to-surface, `contact_ops.py:87`) | lateral on particles, `EMIT mpm_acceleration` | live, one-way; penetration 0.18 units, 1,113 of 380,000 particles behind the surface at frame 801 |
| **matrix -> epithelium** | `mesh_reaction` (FUSED_09 M2) | lateral on vertex, `EMIT velocity` | **missing**: `VERTEX_FORCE` is computed barycentrically (`contact_ops.py:522-527`) and read by nothing |
| matrix, within | `mpm_strain`, `mpm_scatter`, `mpm_grid_update`, `mpm_gather` (+ `ecm_stress` as the readout) | lateral / exchange / field | live; **isotropic** (`mpm_ops.py:441-452`) |
| epithelium -> membrane | plaque spring and friction (`Rig05b`) | lateral on the plaque relation, delta to both ends | rig only; momentum conserved to 1.4e-16 (`05b_plaque/metrics`) |
| **membrane -> epithelium** | the same operator's other half | | computed and discarded (`FUSED_09.md`) |
| membrane, within | sheet elasticity, `bm_refine`, `bm_secrete`, `bm_tear` | lateral / structural / rewire | rig only; gates G14-G18, G24-G27, G43-G44 passed |
| membrane chemistry | secrete, diffuse, activate, inhibit, degrade, tear (the minisite's six) | lateral on bm_face + rewire | rig only (`protease_ops.py`); G53-G57 passed |
| membrane -> epithelium (signal) | `bm_sense` (`contact_ops.py:777`) writes the ligation deficit into `cell.chem` | structural on vertex | live, but reads a *recorded* map |
| membrane <-> matrix | none needed: "both sets scatter into `mpm_grid` with `implementation: accumulate`" (`ops/AUDIT.md`) | | deferred by design (`PROMPT_06.md:10`) |

The two abandoned routes, for the record: `ecm_from_cell[replay]` and `cell_exclude` ("a projection
backstop, which makes non-penetration unmeasurable", `04_spheroid_ecm/what.yaml`), and the MPM
membrane of `membrane_ops.py` ("mechanically inert: sigma_max(F)-1 reaches 7e-4 against a true
stretch of 3.4x", `ops/LADDER.md`).

---

## 3. What the archive established, rung by rung

| stage | question | answer, with the number | file |
|---|---|---|---|
| 0u | box <-> um, frame <-> s | 10 um per tissue unit, 600 s per frame, force NOT declared | `0u_units/spec.yaml:29-38` |
| 00-01c | the epithelium alone; myosin pools; the newborn junction | 200 -> 6076 cells, r 4.65 -> 16.56; two-pool myosin keeps its setpoint through division (drift 0.956 vs 0.815); cytokinetic ring x1 chosen as the reference tissue | `00_spheroid/metrics.json`, `01b_*`, `01c_*` |
| 02b-02h | the matrix alone: a ball dropped on fibres | stress measure matters 100x (`|J-1|` 0.378 vs von Mises 45.8); 02h lean adopted | `02c_ecm_block_vonmises/metrics.json` |
| 02i | confined compression: is it a fibre network? | **no**: stiffening ratio 1.13 (a network gives 10-100), Poisson 0.216 (a solid); "the constitutive law reads the strand direction nowhere" | `02i_compress/spec.yaml`, `gates:` |
| 03-03g | the contact law | penetration falls with the penalty (1.92 -> 1.27 cells over 4x k); momentum residual 1.5e-7; shear and breach loadings | `03d_gates/gates.json` |
| 04, 04c-e | spheroid loading the matrix, one way | density peak 1.26 at r 0.158; displacement exponent -1.26; **cos2 measured 0.284 = affine 0.283**: the strands turn exactly as passive advection predicts, no more | `04c_spheroid_ecm/metrics.json` |
| 05a | a mesh sheet instead of an MPM membrane | carries 99.4% of the applied stretch (MPM: 12.7%) | `05a_sheet/metrics.json` |
| 05b | adhesion as a relation | momentum conserved to 1.4e-16; standoff at its own rest length | `05b_plaque/metrics.json` |
| 05c, 05l | the sheet refines as the spheroid grows | refinement changes nothing (dlambda 6e-9), mean edge stays 0.93x seeded with refinement vs 3.63x without | `05c_remesh`, `05l_G44_refine` |
| 05d, 05e | adhesion as bond density; slip from bond turnover | receptor conserved; slip monotone in k_off, zero at k_off = 0 | `05d_adhesion`, `05e_slip` |
| 05f | mass balance and a supply-driven tear | rho settles at 1.00 / 0.50 / 0.36 for fed / half / starved; tear at rho < 0.35, not at a strain | `05f_secrete/metrics.json` |
| 05g-05i | the protease network | steady-state drift 0; TIMP-2 bell over two decades (G54); TIMP-3 must be immobile to hold a pattern (sqrt(4Dt) = 268 um per frame) | `05h_ternary`, `05i_phase` |
| 06 | the three bodies drawn together | "two solvers, one tissue, no force between them"; detaching the plaques drops the sheet stretch from 3.71 to 1.02 | `06_spheroid_bm_ecm/what.yaml`, `06_detach` |
| 06_breach*, 06_hole* | one source vs many; degradation unopposed | 54 openings from six random modes; 6 from one cap; k_deg 300 tears the sheet apart (diverged at frame 323); the deciding variable is the INITIALISATION of MT1-MMP, not a rate | `06_hole_tiny/hole_metrics.json` |
| 07 | plaque identity through refinement | G44 passes; G78 (no interval grows faces by > 20%) fails by 0.03 in every full run: refinement is right in aggregate, lumpy in time | `07l_nominal/metrics.json` |
| 08b | a hole decides where the bud grows | +0.975 on axis, -0.055 off; rotated control +0.956; shape is a taper, no waist | `BUDDING_08.md` |

Two measurement defects the archive logged and did not fix, both of which the rebuild inherits:
`mesh_contact` assumes the tissue is star-shaped about its centroid ("silently wrong for an
overhang", `FUSED_09.md`), and the tear metric does not survive refinement (`06_hole_stable`:
`faces_torn 0` beside `verdict: 46 holes`).

---

## 4. The milestones, with steps

### M0 (done 2026-09-10): the tissue is sound and protected

The apico-basal working point was lost on 2026-09-09 (a default flip re-targeted every spec) and
recovered; 14 working points are fingerprinted and rerun nightly (`tests/regression/`). The
tissue + matrix pair runs live (`spheroid_ecm_ab`). Everything below builds on a tissue that
cannot silently change again.

### M1: tissue + membrane, live, in the tissue's own box

**What M1 delivers.** One spec runs the apico-basal tissue and its basement membrane together,
frame by frame, with the membrane pulling on the tissue. Today the membrane only exists in a
separate Python program (a "rig") that reads a recorded tissue and cannot push back.

**The design: one operator, not thirty.** The membrane becomes a single engine operator called
`bm_solve`. It acts on the vertex set of the tissue. It returns a velocity for each tissue vertex:
the force the membrane exerts on that vertex, divided by the vertex's friction coefficient. Inside,
the operator keeps everything the rig kept: the elastic sheet, the adhesion springs (called
plaques) that tie the sheet to the tissue, the pool of spare triangles the sheet refines from as
the tissue grows, and its own small time loop, because the sheet needs many small steps for each
tissue frame. The sheet's internals were certified together by the archive's gates G14 to G18
(refinement changes neither the stretch nor the energy, and no bad triangles appear). Splitting
those internals into separate engine operators would reopen every one of those gates and would not
change the model, so they stay together.

**Steps.**

1. Move the rig's three modules unchanged into the engine's model folder: the elastic sheet
   (`discovery_okuda/ops/bm_ops.py`, class `Sheet`), the plaques (`adhesion_ops.py`) and the
   local refinement (`bm_refine_local.py`). `bm_solve` wraps them. The membrane's three sets are
   declared in the spec (its nodes, its triangular faces, and the plaque relation between a tissue
   face and a membrane node), so the engine records and renders their state like any other set.
2. Attach the plaques to the basal surface of the apico-basal tissue. This is the payoff the
   apico-basal promotion note announced: the archive attached the membrane to a recorded
   mid-surface because the old tissue had no inner surface. The live tissue has one, but no
   operator can bind to it yet. **Open finding (2026-09-10):** on the archived spheroids the
   apical ring is the outer one, so the basal surface faces the lumen, while a basement membrane
   needs the basal side facing out. The seed option `apical: in` flips the rings but does not
   reproduce the working point (r 4.85 instead of 4.53 at frame 0, no division by frame 75, cell
   area 30 percent off): something in the seeded targets or the growth readers takes the apical
   ring as the outer one. This has to be found before a membrane is attached outside. The
   integrin demo (`config/tissue/spheroid_integrins.yaml`) uses the model's basal surface as it is.
3. Run the control with the plaque stiffness set to zero. The tissue must then reproduce the
   cvd2_adder_tension working point exactly, cell count and shell radius at every checkpoint.
   Once the coupled run is looked at and accepted, register it as a working point.
4. Re-measure what the old rig could not. The archived membrane runs held the tissue in place
   with an artificial restoring force of stiffness 50 toward the recorded positions, so the
   membrane's effect on the tissue was erased every frame. With that gone, measure: the gap
   between the sheet and the tissue surface (the standoff), the sheet's stretch against the growth
   of the basal radius, how much the sheet slides over the tissue (the slip), and the sum of the
   forces exchanged through the plaques, which must be zero to round-off.

### M2: the matrix pushes back

**What M2 delivers.** The matrix pushes back on the tissue. Today the contact operator computes
the force the matrix exerts on each tissue vertex, records it, and nothing reads it.

**The design.** A new operator, `mesh_reaction`, acting on the tissue's vertex set. It adds up the
recorded force on each vertex over the small matrix steps of one tissue frame and returns it as a
velocity: force divided by the vertex's friction coefficient. The force is already distributed
correctly over the three corners of each tissue face, and the total force on tissue and matrix
together is already zero to round-off, so no new mechanics is needed.

**One decision to make.** The matrix needs about 2,500 small steps per tissue frame to stay stable
at the tissue's time step of 1.0, and the engine reads the number of small steps from a constant in
the spec. Either the matrix keeps a fixed block of small steps inside each tissue frame with an
explicit stability check (what `spheroid_ecm_ab` does today: 20 steps of 0.05), or the matrix is
relaxed to rest between tissue frames instead of integrated in time.

**Tests.** The total force exchanged stays zero to round-off. And the number the biologist asked
for: growth in a stiff matrix is slower than growth in a soft one. Run the matrix stiffness (its
Young's modulus, the spec key `youngs`) at 15, 150 and 1500 and record the cell count and the shell
radius over time. The run with the coupling switched off must come out identical to
`spheroid_ecm_ab`.

### M3: a matrix that is a fibre network

**What M3 delivers.** A matrix that behaves like a network of fibres, not like a solid drawn in
the shape of fibres. The archive measured the difference: under compression the matrix stiffened
by a factor 1.13, where a fibre network stiffens by 10 to 100; and the fibres near the growing
tissue turned exactly as much as a passively advected continuum would, no more.

**The design.** The matrix seeder already stores the direction of the strand each particle
belongs to, in anticipation of exactly this. Add to the stress law a term that resists stretch
along that direction and does nothing across it, engaging only under tension, of the standard
Holzapfel-Gasser-Ogden form: the energy per unit volume is
`psi_f = (k1 / (2 k2)) * (exp(k2 * (I4 - 1)^2) - 1)` when `I4 > 1`, where `I4` is the squared
stretch along the fibre direction, `k1` is the fibre stiffness and `k2` sets how sharply it
stiffens. The fibre stretch is already computed for the cardiac active-stress operator, so the
kinematics exist.

**Tests.** The archived compression test's stiffening ratio moves from 1.13 into the 10 to 100
range and its lateral bulging drops toward that of a network; the fibre alignment near the tissue
exceeds the passive prediction; the pressure difference between along-fibre and across-fibre
directions exceeds the 1.5 times measured today.

### M4: the protease chemistry as operators on the membrane

**What M4 delivers.** The protease chemistry of the minisite as engine operators on the membrane's
face set, so a spec can switch it on. Six operators, one per step of the chemistry: secretion of
membrane material, diffusion of the soluble species over the sheet, activation of the enzyme
(the matrix metalloproteinase MMP-2 is activated by the membrane-bound enzyme MT1-MMP through a
three-molecule complex with its own inhibitor TIMP-2, which is why activation rises and then
falls as the inhibitor increases), inhibition (TIMP-2 diffuses, TIMP-3 stays bound and is the only
inhibitor that can hold a pattern), degradation of membrane material by the free enzyme, and
tearing (a face is removed when its material density falls below a threshold). The diffusion uses
the rig's implicit solver on the face graph. The archived gates for these steps (G24 to G27 for the
mass balance, G53 to G57 for the chemistry) become regression fingerprints. Also fix the tear
count so it survives refinement: one archived run reported zero torn faces while 46 holes were
visible.

### M5: the interface regulates growth on the live tissue

**What M5 delivers.** The membrane regulates growth on the live tissue, which is the archive's
payoff rebuilt without replays. The sensing operator `bm_sense` reads today a recorded map of how
much membrane sits under each cell. Make it read the live sheet through the plaque relation. Then
the chain runs live: the enzyme degrades and tears the membrane; each cell reads how much ligand it
has lost underneath; that deficit is written into the cell's chemical state; growth rate rises with
it; division orients along the interface. This is integrin signalling read as a brake on the cell
cycle: a cell anchored to an intact membrane divides less.

**Test.** The bud follows the hole, as in the archive (+0.975 along the hole's axis, within
+-0.04 elsewhere), now with a matrix that feels the bud. Two defects the archive logged are fixed
on the way: the smoothing of the membrane map that flattened it (it left a growth contrast of only
1.01 to 1.10 times), and a division cap that was declared in specs but read by nothing.

### M6: the molecular corset (the biologist's demo)

**What M6 delivers.** The demo the biologist suggested. In the fly egg chamber (Bilder's work)
the tissue elongates because growth is constrained in one direction: collagen fibrils in the
basement membrane run around the circumference and the cells' own tension is polarised, so
together they act as a corset and the tissue grows along the free axis. Two ingredients, each
building on a milestone above:

1. **Direction-dependent junction tension within the tissue.** Give the junction-myosin operator a
   new mode in which the tension of an edge depends on its orientation relative to a declared
   axis, so circumferential junctions pull harder. This first needs the apico-basal energy to read
   myosin at all: today only the mid-surface model does. Add that and certify the gradient against
   automatic differentiation, as the warp gradient is certified.
2. **A corset in the matrix.** Seed the fibres aligned around the polar axis in a ring around the
   spheroid, make them stiff along their length (M3), and let the tissue feel them (M2).

**Test.** The spheroid's aspect ratio at frame 800 against the isotropic control (1.02 in the
archive). When the corset axis is rotated, the elongation axis must rotate with it, the same control
design as the archive's rotated hole.

---

## 5. The specifics: the equations each body obeys

### 5.1 The epithelium (live)

Apico-basal vertex model (`vertex_ops.py:4508`), per cell j with apical and basal rings and
separation `sep`:

    U = sum_j [ 1/2 k_v (V_j - V_eq_j)^2 + kappa_s S_j + 1/2 gamma P_j^2 ] + Lam sum_e l_e
        + K_R sum_i (|x_i - c| - R0)^2

`V_j` the cell polyhedron volume, `V_eq_j` its target (wedge convention by default, polyhedron when
declared), `S_j` its surface, `P_j` the ring perimeter, `l_e` edge lengths, `c` the tissue
centroid. Overdamped relaxation, 30 iterations per frame, capped step. Growth: `cell_grow` raises
`V_eq` at `rate` per frame; division by the adder rule on the cell's own volume; T1 by `edge_flip`.

### 5.2 The basement membrane (rig, to be promoted at M1)

St Venant-Kirchhoff membrane, one constant-strain triangle per face, thickness T = 0.002 box
(2.34 um, 23x the real 0.1 um: "the proportions are right, the absolute scale is not",
`05f_secrete/spec.yaml:5-9`), E = 400, nu = 0.3, massless, overdamped explicit Euler
`x += dt M f`, adaptive substeps. Refinement: global 1 -> 4 midpoint split with
`Dm_inv_child = S_k^-1 Dm_inv_parent`, triggered when the mean live edge exceeds 1.45x seeded;
tear is a flip of `face_occ`.

Plaque (`05b`): normal spring `f = k_n (l - l0) n` with `l0 = 0.3 T` (integrin layer ~30 nm under
a ~350 nm plaque, Kanchanawong 2010), tangential friction `xi` on the relative velocity, reaction
distributed barycentrically to the three vertices of the epithelial face.

Bond density (`05d`, `05e`): receptor `N_f` per cell (an ODE, no diffusion: sqrt(4Dt) = 15.5 um
exceeds a cell), bound bonds `N_b` on the edge, ligand `rho_L` on the face:

    dN_b/dt = k_on N_f rho_L - k_off0 exp(f / f_bell) N_b        (Bell slip bond)

with slip a consequence of turnover: monotone in `k_off`, zero at `k_off = 0` (G32).

Mass balance (`05f`): `d rho/dt = s(theta, phi) - rho / tau_bm - (dilution by area growth)`,
with `tau_bm = 40` frames; tear at `rho < rho_crit = 0.35`.

### 5.3 The protease network (rig, to be promoted at M4)

Species on `bm_face`: `proMMP2`, `MMP2`, `TIMP2` (diffusing, D > 0), `TIMP3` (bound, D = 0,
"the only inhibitor that can hold a pattern"); `MT1_MMP` tethered per cell, its field the deciding
initialisation (six random modes -> 54 openings; one 20-degree cap -> 6). Reference Karagiannis &
Popel (2004) J Biol Chem 279:39105, inhibitor held in the matrix after Yu et al. (2000).

    dc/dt = D Lap_face c + R(c)                          semi-implicit: (I + dt D L) c = c_old, CG
    activation  a = k T^2 c_pro (c_T / K) / (1 + c_T / K)^2      (TIMP-2 bridges, then inhibits:
                                                                  a bell over two decades, G54)
    inhibition  MMP2 + TIMP -> complex, K_i = 0.16 nM (MT1-MMP:TIMP-2)
    degradation d rho/dt = - k_deg [MMP2_free] rho                 k_deg 100: a hole; 300: torn apart
    tear        face removed when rho < rho_crit = 0.35             (structural, rim loops counted)

### 5.4 The matrix (live; M3 adds the fibre term)

MLS-MPM, fixed-corotated (`mpm_ops.py:441-452`):

    tau_p = 2 mu_p (F_p - R_p) F_p^T + I lambda_p J_p (J_p - 1)
    (mv)_n += sum_p w_n m_p (v_p + C_p (x_n - x_p)) - (4 dt / dx^2) w_n V_p tau_p (x_n - x_p)

Contact (`contact_ops.py:88-127`): for a particle at depth `d_i` behind the nearest face,
`a_n = k d_i n`, `a_t = -mu |a_n| v_t / (|v_t| + eps)`, `k = (k_frac / dt_sub)^2`, reaction
returned to the face's vertices barycentrically. M3 adds, per particle with seeded direction `n_p`:

    I4 = n_p . C_p n_p ,   psi_f = (k1 / 2 k2) (exp(k2 (I4 - 1)^2) - 1)  for I4 > 1

### 5.5 The interface as a signal (live map today, live sheet at M5)

`bm_sense`: `def_f = clamp(1 - L_f / p_ref, 0, 1)^sharp` from the ligand mass under the cell,
written into one channel of `cell.chem`; `cell_grow` reads `rate (rho + Hill(a))`; `cell_divide`
orients along the interface. This is integrin ligation read as a brake on the cycle
(`BUDDING_08.md:20-30`).

---

## 6. Order and effort

M1 and M2 are independent and each is a week: M1 is mostly moving certified code under one operator
and binding it to the basal surface; M2 is one operator plus an integration decision. M3 is the
one genuinely new piece of physics (a few days for the term, a week for the gates). M4 is a
promotion of certified code with its gates turned into fingerprints. M5 needs M1, M2 and M4. M6
needs M2, M3 and one energy-core change; the in-surface half alone (anisotropic junction tension
on the live tissue, no matrix) can be shown right after M0 and is the cheapest demo on the list.

Every milestone ends the way this week's work did: an accepted run registered as a working point,
so the next default flip fails a test.

## 7. Open renderer defects met on the way (2026-09-10)

- `plotting.zoom` on a REPLAYED free-boundary run renders every frame black (the live pass with
  zoom renders; the replay without zoom renders; the replay with zoom rendered only the late
  frames of the first integrin run). The framing centre of a replayed free run under zoom is the
  suspect. Until fixed, specs that declare curve panels (which force the replay) do not use zoom.

