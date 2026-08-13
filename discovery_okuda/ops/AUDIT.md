# Audit of the nominal (91_gridbc_band): Plexus2 form, and biology first

The operators active in 91, and whether each is a biological mechanism or a numerical device wearing
one's clothes.

| operator | kind | verdict |
|---|---|---|
| `ecm_seed`, `bm_seed` | seed | initial condition, fine |
| `ecm_from_cell` | mechanics | replay of the recorded surface pushing the stroma -- fine |
| `cell_exclude` (on the ECM) | mechanics | **positional projection** -- see 3 below |
| `mpm_strain / scatter / grid_update / gather` | mpm | the solver, fine |
| `mpm_boundary` | field | **the hack** -- see 1 |
| `bm_strain` | lateral | **not a mechanism** -- see 2 |

## 1. `mpm_boundary` is a numerical device, not a mechanism

The biology it stands for is real and simple: *the epithelium is impenetrable, and it grows*. The
implementation is not. It overwrites grid-node velocity, so:

- it is **kinematic**: we inject whatever momentum holds the prescribed motion. Momentum is not
  conserved at the boundary, and the reaction the tissue would feel is recorded and then discarded.
- `band` and `recover` have **no biological referent**. Nothing in a cell corresponds to "how many grid
  cells the constraint reaches" or "how many frames an overlap takes to clear".
- worst, the standoff it produces is set by the **B-spline stencil width**, not by anything physical:
  each particle smears over +-1.5 cells, so a correctly placed sheet always has part of its footprint
  inside R and is expelled until the whole footprint clears. That is why sweeping `recover` over
  0 / 2 / 6 / 20 traded 46.6% / 3.8% / 11.5% / 13.9% of the sheet lying inside against standoffs of
  +0.0006 / +0.0124 / +0.0088 / +0.0069 and never approached the target of 0 to +0.002.

Biology is unambiguous about that target: a basement membrane sits ON the basal plasma membrane --
integrin a6b4 and dystroglycan bind laminin directly, and the lamina lucida of classical TEM is read
today as a fixation artefact. Any visible gap is numerical.

**Replaced by** `bm_contact` (runs 110/111): penetration measured per PARTICLE against
the surface, force proportional to depth. The standoff then emerges from stiffness instead of being
dialled, and 111 (k x5) is the honest check -- a real penalty contact has standoff proportional to 1/k,
so if it does not scale, the force is just another tuning knob.

## 2. `bm_strain` is measurement registered as an operator

`EMIT = None`, and its whole body appends to a module-level Python list that the renderer reads. It
changes no state. In Plexus2 an operator is a rule that changes state; a quantity computed for looking
at is a diagnostic. It also writes to a module global, which is not part of the state at all, so it
cannot be replayed, cached or checkpointed with the run.

Not urgent -- it is honest about being a colour channel -- but it should not be in the schedule.

## 3. `cell_exclude` still launders deformation, now only for the stroma

We proved this for the membrane: a hard positional projection repositions particles without touching F,
so the material never learns it was deformed (88 tracked the spheroid perfectly at a peak strain of
7e-4 against a true stretch of 3.4x). It is switched off for the membrane in 91 and still ON for the
ECM, so **the stroma's strain field has the same defect** -- 18,134 particles per frame were being
projected in the membrane's case, and the matrix is larger. Its stress colouring and `strained_frac`
are therefore partly non-physical. Same fix applies: a penalty contact rather than a projection.

## 4. The reserve is a stand-in for synthesis

Cells SYNTHESISE basement membrane; they do not activate particles from a pre-existing pool. `reserve`
and `park` are a fixed-budget discretisation of that, and the consequence is real: the sheet cannot
secrete more material than the pool holds. Defensible, but it should be stated rather than assumed --
and it is why 92 exhausted 43,950 of 45,000 particles.

## 5. What 91 does NOT contain, and should not be described as containing

Adhesion is off (`membrane_adhesion=0`), secretion is off, there is no BM anisotropy, and no proteolytic
remodelling. 91 is a mechanical skeleton -- the sheet grows with the spheroid and carries the right
strain -- not a biological model of a basement membrane. That is the right nominal to build on, but the
biology arrives with 112-114 (adhesion, and the no-stroma control that decides whether we are modelling
adhesion or merely squeezing).

## 6. One clean thing worth keeping

BM<->ECM momentum exchange needs no operator at all: both sets scatter into `mpm_grid` with
`implementation: accumulate`, so the stroma resists the sheet's expansion for free. That is the shared
grid doing exactly what it should, and it is the part of this prototype that is least hacky.
