"""Reaction-diffusion ON THE CELL GRAPH, and the two couplings between chemistry and shape.

    seed_cell_chem (alias cell_chem_seed)  the initial morphogen field
    cell_chem_diffuse                      graph_laplacian | interface_weighted
    cell_chem_react                        gray_scott | brusselator | gierer_meinhardt
    cell_neighbours                        the cell adjacency the Laplacian runs on
    cell_geometry                          per-cell area / perimeter / centroid / volume
    cell_grow                              default | balance | sizer | timer
    cell_chem_from_shape                   shape -> chemistry: apical_area | curvature | pressure | tension
    cell_shape_probe                       aspect | shape_index, published for a discriminator
    interface_tension                      a purse-string line tension on the red/white interface
    interface_push                         and the term that is NOT physics -- kept separate on purpose

THE DIFFUSION IS NOT ON A GRID. The cells are the nodes, `cell_neighbours` is the graph, and the
Laplacian is over shared faces -- so the domain grows and rewires as the tissue divides, which a
fixed lattice cannot do. That is why these are `set=cell` operators rather than `field` ones.

INTERFACE_TENSION AND INTERFACE_PUSH ARE TWO OPERATORS AND MUST STAY TWO. They were one,
`rd_interface_tension`, carrying `K_purse * sum l_e` (ordinary vertex-model physics) MINUS
`K_extrude * sum a*r` (an energy that falls as red cells move outward -- it pays the tissue to
produce the morphology the search was looking for). One name over both cost four campaign rounds of
verdicts about a term that measured 0.0 in all 78 specs that ever carried it. See OKUDA_PROMOTION.md.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch
from plexus.models.base import Aggregate, Lateral, Rewire, Structural
from plexus.models.registry import register_operator
from plexus.models.base import Lateral
from plexus.operators.vertex_ops import face_geometry_3d


def _chan(params, who, n_species=2):
    """The FIRST COLUMN of the contiguous species span this instance owns.

    `chan` IS A COLUMN INDEX, NOT A SPECIES INDEX, and it reads like one. A reaction model occupies
    `n_species` ADJACENT columns of `chem` -- Gray-Scott two (a, u), May-Leonard three (u, v, w),
    a coupled pair of Gray-Scott systems four -- so the second two-species system starts at
    `chan: 2`, not `chan: 1`. `chan: 1` is the natural thing to write and would have put the second
    system's activator ON THE FIRST SYSTEM'S SUBSTRATE: both would run, both would look alive, and
    they would be driving one shared column through a coupling nobody wrote.

    THE OLD RULE WAS "chan MUST BE EVEN" and it was wrong the moment a three-species model existed:
    May-Leonard tiles at 0, 3, 6, and an even-only guard rejects a correct spec while accepting
    `chan: 2` for a three-species system, which overlaps. The rule is now "a multiple of this
    model's own width", which is the actual tiling, and it degrades to the even rule for the
    two-species models that are all there used to be.

    The BOUNDS check cannot happen here -- `chem`'s width is not known until forward -- so it is
    `_span` that raises on a span running off the end of the buffer.
    """
    c = int(params.get("chan", 0))
    n = int(n_species)
    if c < 0 or (n > 0 and c % n):
        ok = [i * n for i in range(4)]
        raise ValueError(
            f"{who}: chan={c} does not start a {n}-species span. `chan` is a COLUMN index, not a "
            f"species index -- this model owns {n} adjacent columns of `chem`, so its systems tile "
            f"at {ok}... A chan that is not a multiple of {n} overlaps the neighbouring system and "
            f"silently couples the two through a shared column.")
    return c


def _span(chem, chan, n, who):
    """The `n` columns starting at `chan`, or a loud failure -- the bounds half of `_chan`."""
    if chan + n > chem.shape[1]:
        raise ValueError(
            f"{who}: needs columns {chan}..{chan + n - 1} of `chem` but the block is only "
            f"{chem.shape[1]} wide. Widen `sets.<set>.state.chem.width` to at least {chan + n}.")
    return [chem[:, chan + k] for k in range(n)]


def _emit(chem, chan, terms, rate, occ):
    """A delta that is ZERO IN EVERY COLUMN THIS INSTANCE DOES NOT OWN.

    That is what makes two reaction instances additive rather than mutually overwriting: the engine
    sums operator deltas, so an instance which wrote its own columns and left the others UNSET
    would be fine, but one that wrote zeros over another's work would silently erase it. Building
    the full-width zero tensor and filling only this span is the cheap way to be certain.
    """
    out = torch.zeros_like(chem)
    for k, t in enumerate(terms):
        out[:, chan + k] = rate * t
    return out * occ


@register_operator("cell_geometry", set="cell", kind="aggregate", family="hierarchy")
class CellGeometry3D(Aggregate):
    """AGGREGATE the 3D vertex mesh -> per-cell centroid + area (the cross-scale readout the RD needs:
    the activator spot is seeded by centroid, and the pattern is rendered per cell). Reads the stashed
    half-edge table on the vertex Level; writes cell.cen (+ cell.area) via scatter-add over half-edges."""
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    INPUTS = ["vertex"]; OUTPUTS = ["cell"]; READS = ["pos"]; WRITES = ["area", "cen"]
    MECHANISM_TAGS = ["aggregate", "cell_geometry", "cross_scale"]
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell"); self.vat = params.get("vertex_set", "vertex")

    def forward(self, H, mask=None):
        from plexus.operators.vertex_ops import face_geometry_3d
        clvl = H.level(self.at); vlvl = H.level(self.vat); m = getattr(vlvl, "_mesh", None)
        if m is None:
            return {}
        pos = vlvl.get("pos")[:m["Nv"]]
        area, _, cen, _ = face_geometry_3d(pos, m["E_srce"], m["E_trgt"], m["E_face"], m["nF"])
        nF = m["nF"]; st = clvl.state.clone(); sch = clvl.state_schema
        if "cen" in sch:
            i0, i1 = sch["cen"]; st[:nF, i0:i1] = cen.detach()
        if "area" in sch:
            i0, i1 = sch["area"]; st[:nF, i0:i1] = area.detach()[:, None]
        clvl.state = st
        if getattr(clvl, "occ", None) is not None:
            occ = torch.zeros(clvl.state.shape[0], device=clvl.state.device); occ[:nF] = 1.0; clvl.occ = occ
        return {}


@register_operator("cell_neighbours", set="cell", kind="rewire", family="topology")
class CellAdjacency(Rewire):
    """Two cells are neighbours iff they share a mesh edge. Build that graph from the half-edge
    table on the vertex Level and store it as `edge_index` on the cell Level -- the graph the
    reaction-diffusion runs on. (Rebuilt each call so it tracks T1/division if they run.)"""
    SUPPORTED_DIMS = [2, 3]; DIFFERENTIABLE = False
    MECHANISM_TAGS = ["cell_neighbours", "neighbour_graph"]
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell"); self.vat = params.get("vertex_set", "vertex")

    def forward(self, H, mask=None):
        clvl = H.level(self.at); vlvl = H.level(self.vat); m = getattr(vlvl, "_mesh", None)
        dev = clvl.state.device
        if m is None:
            clvl.edge_index = torch.zeros(2, 0, dtype=torch.long, device=dev); return {}
        es = m["E_srce"].cpu().numpy(); et = m["E_trgt"].cpu().numpy(); ef = m["E_face"].cpu().numpy()
        byedge = defaultdict(list)
        for k in range(len(ef)):
            byedge[(min(int(es[k]), int(et[k])), max(int(es[k]), int(et[k])))].append(int(ef[k]))
        pairs = set()
        for faces in byedge.values():
            for a in range(len(faces)):
                for b in range(a + 1, len(faces)):
                    x, y = faces[a], faces[b]
                    if x != y:
                        pairs.add((min(x, y), max(x, y)))
        if not pairs:
            clvl.edge_index = torch.zeros(2, 0, dtype=torch.long, device=dev); return {}
        e = np.array(sorted(pairs)).T
        ei = np.concatenate([e, e[::-1]], axis=1)                # symmetric (both directions)
        clvl.edge_index = torch.as_tensor(ei, dtype=torch.long, device=dev)
        return {}


# CANONICAL `seed_cell_chem`, ALIAS `cell_chem_seed` -- see `mesh_ops.SeedMesh3D` for why both
# spellings must resolve: 320 specs use the first and the rest use the second.
@register_operator("seed_cell_chem", "cell_chem_seed", set="cell", kind="seed", family="seed")
class CellRDSeed(Structural):
    N_SPECIES = 2
    """Gray-Scott initial condition on the cell set: substrate u=1 everywhere, activator a=0 except
    a central spot (a=0.5, u=0.25) that nucleates the pattern. chem = [a, u].

    `mode: tip` WAS REMOVED, 6 August. It re-activated a fixed-size cap at the current outermost
    cell EVERY FRAME, so the activation chased the advancing tip and forced a constant-diameter
    extension. Two reasons it had to go, and the second is the one that matters:

      * it is a moving BOUNDARY CONDITION, not an initial condition. Where the activity sits is
        then our answer rather than the simulation's, and the campaign's question -- does the
        chemical pattern grip the shape? -- was being asked of a pattern pinned to the shape's own
        outermost point. `corr_act_rad` was partly measuring the seeding rule against itself;
      * re-applying it every frame overwrites BOTH chemistry channels, so no operator that writes
        to `chem` can accumulate anything. `cell_chem_from_shape` writes to channel 1 and `tip` sets that
        channel to exactly 1.0, which is why 8 same-seed `beta` edits across 13 rounds moved the
        trajectory by exactly zero and were each recorded as a refuted hypothesis.

    AN UNKNOWN MODE NOW RAISES. It used to fall through to the `else`, which means deleting the
    branch alone would have turned 265 archived `mode: tip` specs into SCATTER runs that still
    load, still finish and describe a different mechanism. A spec that can no longer be run is a
    correct outcome; a spec that quietly runs something else is the failure this whole phase is
    about.
    """
    SUPPORTED_DIMS = [2, 3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["initial_condition", "gray_scott"]
    MODES = ("scatter", "noise", "patch", "cones", "simplex")
    REFERENCE = "Plexus (this work); cone seeding after Okuda, S. et al. (2018). Sci. Rep. 8:2386."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell"); self.vat = params.get("vertex_set", "vertex")
        self.seed = int(params.get("seed", 0))
        # WHICH SPECIES THIS SEEDER FILLS: 0 (columns 0,1) by default; 2 for a second RD system.
        # HOW MANY COLUMNS THIS SEEDER FILLS. Two unless told otherwise, so nothing archived moves.
        self.n_species = int(params.get("n_species", self.N_SPECIES))
        # (`simplex`) the total the species are normalised to. 1.0 is the May-Leonard simplex.
        self.p0 = float(params.get("p0", 1.0))
        self.chan = _chan(params, type(self).__name__, self.n_species)
        self.mode = params.get("mode", "scatter")               # "noise" | "scatter" | "patch" | "cones"
        if self.mode not in self.MODES:
            raise ValueError(
                f"cell_chem_seed: unknown mode {self.mode!r}; known: {list(self.MODES)}. "
                + ("`tip` was removed on 6 August -- it re-seeded every frame, which makes it a "
                   "moving boundary condition and annihilates every operator that writes to "
                   "`chem`. Use `scatter` with `before_frame: 3`." if self.mode == "tip" else ""))
        self.seed_frac = float(params.get("seed_frac", 0.06))   # (scatter) fraction of strong activator seeds
        self.A = float(params.get("A", 1.0)); self.B = float(params.get("B", 3.0))   # (noise) steady state (A, B/A)
        self.noise = float(params.get("noise", 0.04))
        self.patch_z = float(params.get("patch_z", 0.6))        # (patch) activate cells with cen_z > patch_z x z_max
        self.n_spots = int(params.get("n_spots", 5))            # (cones) number of fixed radial activation foci
        self.cone_deg = float(params.get("cone_deg", 18.0))     # (cones) half-angle of each activation cone
        self.seed_dir = params.get("seed_dir", None)            # (cones, n_spots=1) override the cone axis to a fixed
        #   direction -> aim the tube where we want it (e.g. FRONT of the render camera at elev18/azim30 ~ (.82,.48,.31))

    def _cone_dirs(self):
        """`n_spots` spread unit directions on the sphere (Fibonacci) -> fixed radial tube axes (Fig 5). A given
        `seed_dir` overrides the axis (used for n_spots=1 to point the single tube at the camera)."""
        if self.seed_dir is not None and self.n_spots == 1:
            v = np.asarray(self.seed_dir, float); return (v / (np.linalg.norm(v) + 1e-12))[None, :]
        i = np.arange(self.n_spots) + 0.5
        phi = np.arccos(1 - 2 * i / self.n_spots); theta = np.pi * (1 + 5 ** 0.5) * i
        return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)

    def forward(self, H, mask=None):
        clvl = H.level(self.at)
        # A MESH IS NOT REQUIRED, AND ASKING FOR ONE BY NAME WAS FATAL. This read
        # `H.level(self.vat)` unconditionally, so a spec with no `vertex` set died on
        # `KeyError: 'vertex'` -- and `SUPPORTED_DIMS` says [2, 3], so a flat 2D run is supposed to
        # be legal. All this operator wants from the mesh is nF, THE NUMBER OF CELLS; `scatter` and
        # `noise` use no geometry whatever, and `patch`/`cones` already guard on `"cen" in
        # state_schema` and fall back to a uniform 0.02 without it. On a mesh-free set the cell
        # level IS the population, so its own occupancy answers the only question being asked.
        # `in` rather than `.get`: `H.levels` is an `nn.ModuleDict`, which has no `.get` -- the
        # exact trap that silently disabled `renumber_set`.
        vlvl = H.levels[self.vat] if self.vat in H.levels else None
        m = getattr(vlvl, "_mesh", None) if vlvl is not None else None
        if "chem" not in clvl.state_schema:
            return {}
        if m is not None:
            nF = m["nF"]
        else:
            occ = getattr(clvl, "occ", None)
            nF = int(occ.sum().item()) if occ is not None else int(clvl.state.shape[0])
            if nF <= 0:
                return {}
        dev = clvl.state.device
        g = torch.Generator(device="cpu"); g.manual_seed(self.seed)
        if self.mode == "patch":                                # localized activation source (a bud/tube driver)
            a = torch.full((nF,), 0.02, device=dev)
            if "cen" in clvl.state_schema:
                ci0, ci1 = clvl.state_schema["cen"]; zc = clvl.state[:nF, ci0 + 2]
                a = torch.where(zc > self.patch_z * float(zc.max()), torch.ones(nF, device=dev), a)
            u = torch.ones(nF, device=dev)
        elif self.mode == "cones":                              # N FIXED radial activation cones (Fig 5 multi-tube):
            a = torch.full((nF,), 0.02, device=dev)             # each cone's tip stays activated as it extends ->
            if "cen" in clvl.state_schema:                      # N radial tubes. Re-seeded every frame (tracks tips).
                ci0, ci1 = clvl.state_schema["cen"]; cen = clvl.state[:nF, ci0:ci0 + 3]
                d = cen / (cen.norm(dim=1, keepdim=True) + 1e-9)
                dirs = torch.as_tensor(self._cone_dirs(), dtype=cen.dtype, device=dev)
                cosmax = (d @ dirs.T).max(dim=1).values
                a = torch.where(cosmax > float(np.cos(np.radians(self.cone_deg))), torch.ones(nF, device=dev), a)
            u = torch.ones(nF, device=dev)
        elif self.mode == "simplex":
            # A SYMMETRIC START FOR A COMPETITION MODEL. Every other mode here is a Gray-Scott
            # initial condition: activator in column 0, SUBSTRATE = 1 in column 1. Feed that to
            # May-Leonard, where column 1 is just the second competitor, and the run begins with
            # v at 0.9 against u and w at 0.07 -- not a competition, a landslide. Measured: v
            # dominates immediately, w cyclically beats v, and the field collapses to w = 1
            # everywhere with zero spatial variance. No motif can emerge from that and none did.
            #
            # `simplex` gives all n species the same random field and then NORMALISES SO THEY SUM
            # TO `p0`, which is what ParticleGraph's RD_RPS does (`init_mesh`, case 'RD_Mesh':
            # `node_value = rand(n,3)` then `node_value[:,k] /= sum`). p0 = 1 is not a detail: the
            # logistic factor is (1 - p - a*rival), so starting at p = 0.175 leaves (1-p) = 0.83 of
            # net growth for EVERY species and the run spends itself climbing to the simplex
            # instead of competing on it. The heteroclinic cycle that produces spirals lives AT
            # p = 1, so an initial condition below it postpones the whole phenomenon.
            a = None
            u = None
        elif self.mode == "noise":                              # Brusselator: homogeneous steady state + noise
            a = (self.A + self.noise * torch.randn(nF, generator=g)).to(dev)
            u = (self.B / self.A + self.noise * torch.randn(nF, generator=g)).to(dev)
        else:                                                   # "scatter" -- and ONLY scatter. The mode is validated
            # in __init__, so this branch can no longer be reached by a typo or by a mode that was
            # deleted out from under an archived spec.
            # (a central spot is 2D-disk logic -- on a sphere every cell is equidistant, so scatter/noise)
            a = (0.04 * torch.rand(nF, generator=g)).to(dev)
            u = torch.ones(nF, device=dev)
            nucl = (torch.rand(nF, generator=g) < self.seed_frac).to(dev)
            a = torch.where(nucl, torch.full_like(a, 0.5), a)
            u = torch.where(nucl, torch.full_like(u, 0.25), u)
        # SEEDED INTO THIS SPECIES' OWN COLUMNS. `chan` offsets from the schema's base, so a
        # second seeder writes the second pair and leaves the first alone.
        h0, h1 = clvl.state_schema["chem"]
        base = h0 + self.chan
        st = clvl.state.clone()
        # SIMPLEX WRITES EVERY SPECIES AND RETURNS -- it has no activator/substrate pair to write,
        # which is the whole point of it, so it must run BEFORE the `a`/`u` write rather than after.
        if self.mode == "simplex":
            _cols = [k for k in range(self.n_species) if h1 - base > k]
            _r = torch.rand(nF, len(_cols), generator=g).to(dev)
            _r = self.p0 * _r / _r.sum(dim=1, keepdim=True).clamp(min=1e-9)
            for _i, _k in enumerate(_cols):
                st[:nF, base + _k:base + _k + 1] = _r[:, _i:_i + 1]
            clvl.state = st
            return {}
        st[:nF, base:base + 1] = a[:, None]
        if h1 - base > 1:
            st[:nF, base + 1:base + 2] = u[:, None]
        # A SPAN WIDER THAN TWO gets the remaining species seeded the same way the first was, from
        # the SAME generator, so a three-species start is three independent random fields rather
        # than one field and two zeros -- two zeros is extinction, not a neutral start.
        for _k in range(2, self.n_species):
            if h1 - base > _k:
                _v = (0.04 * torch.rand(nF, generator=g)).to(dev)
                _nu = (torch.rand(nF, generator=g) < self.seed_frac).to(dev)
                st[:nF, base + _k:base + _k + 1] = torch.where(
                    _nu, torch.full_like(_v, 0.5), _v)[:, None]
        clvl.state = st
        return {}


@register_operator("cell_chem_diffuse", set="cell", kind="lateral", family="fields", implementation="graph_laplacian")
class CellDiffuse(Lateral):
    N_SPECIES = 2
    """`graph_laplacian` implementation of cell_chem_diffuse: PURELY COMBINATORIAL diffusion of the two
    morphogens between neighbouring cells (forked from Turing_vertex `graph_diffuse`). `norm=True` uses
    the degree-normalised Laplacian (eigenvalues in [-2,0]) so an explicit step is stable at any cell
    degree. First-order -> EMIT=velocity into chem.

    CAVEAT (why the `interface_weighted` sibling exists): this forward reads ONLY `chem` and
    `edge_index` -- no geometry at all. Two cells sharing a thin sliver exchange exactly as much as two
    sharing a broad face, and a cell stretched to twice its volume dilutes as if it had not stretched,
    so mesh DEFORMATION IS INVISIBLE to the chemistry (the pattern rides on the tissue like a decal).
    That is the right numerics for a pure Turing-on-a-graph study and it is what every calibrated
    round_* preset was tuned against, so it stays the contract DEFAULT; select
    `implementation: interface_weighted` for the Okuda shape<->chemistry two-way coupling.
    The name is not new: discovery/composition_space.py has always listed this impl as
    "graph_laplacian" -- it was registered as the anonymous "default", so the two disagreed."""
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["chi"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]; MAPS = ["edge_index"]
    MECHANISM_TAGS = ["diffusion", "graph_laplacian", "turing"]
    REFERENCE = "Fick, A. (1855). Ueber Diffusion. Ann. Phys. 170:59-86; Turing, A. M. (1952). Phil. Trans. R. Soc. B 237:37-72."
    PARAM_ROLES = {"d_a": "activator_diffusivity", "d_h": "substrate_diffusivity", "chi": "spatial_scale"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        # A DIFFUSIVITY PER SPECIES, because two is not the only width. `d_a`/`d_h` name the roles
        # of the two Gray-Scott species and cannot express three (May-Leonard) or four (a coupled
        # pair), and FitzHugh-Nagumo needs one of them to be exactly ZERO -- only `u` diffuses
        # there, `v` has no Laplacian at all. `d: [..]` is the general spelling and its LENGTH
        # declares the span; `d_a`/`d_h` remain the two-species one and every archived spec keeps
        # working unchanged.
        _d = params.get("d")
        if _d is not None:
            self.d = [float(x) for x in _d]
        elif "d_a" in params and "d_h" in params:
            self.d = [float(params["d_a"]), float(params["d_h"])]
        else:
            raise ValueError(f"{type(self).__name__}: needs either `d: [..]` (one diffusivity per "
                             f"species) or both `d_a` and `d_h` (the two-species spelling).")
        self.d_a, self.d_h = self.d[0], (self.d[1] if len(self.d) > 1 else self.d[0])
        self.N_SPECIES = len(self.d)          # instance attribute: shadows the class default of 2
        self.chi = float(params["chi"])
        self.norm = bool(params.get("norm", True))
        # WHICH SPECIES THIS INSTANCE OWNS: 0 is the first pair (chem columns 0,1) and is the
        # default, so every existing spec is unchanged. A second RD system lives at chan 2.
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)


    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        ei = getattr(lvl, "edge_index", None)
        if ei is None or ei.numel() == 0:
            return {self.at: torch.zeros_like(chem)}
        i, j = ei[0], ei[1]; N = chem.shape[0]
        agg = torch.zeros_like(chem).index_add_(0, i, chem[j])
        deg = torch.zeros(N, device=chem.device, dtype=chem.dtype).index_add_(0, i, torch.ones_like(i, dtype=chem.dtype))
        lap = (agg / deg.clamp(min=1)[:, None] - chem) if self.norm else (agg - deg[:, None] * chem)
        # A PER-COLUMN COEFFICIENT, so a second SPECIES can live in the same `chem` buffer at its
        # own columns. This was `tensor([d_a, d_h])` -- exactly two entries -- which both assumed a
        # width-2 chem and diffused whatever happened to be in those two columns. With `chan` the
        # operator says WHICH pair it owns, and writes zeros everywhere else, so two instances at
        # chan 0 and chan 2 are two independent reaction-diffusion systems that cannot leak into
        # one another through the diffusion step.
        _span(chem, self.chan, len(self.d), type(self).__name__)      # bounds, loudly
        coef = torch.zeros(chem.shape[1], device=chem.device, dtype=chem.dtype)
        for _k, _dk in enumerate(self.d):
            coef[self.chan + _k] = _dk * self.chi
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: (coef[None, :] * lap) * occ}


@register_operator("cell_chem_diffuse", set="cell", kind="lateral", family="fields", model="interface_weighted")
class CellDiffuseInterfaceWeighted(Lateral):
    """`interface_weighted` MODEL of cell_chem_diffuse -- the OKUDA finite-volume form, and the
    MISSING HALF of the chemistry<->shape coupling.

    `model:`, NOT `implementation:`, AND THE TEST IS THE CONTINUUM LIMIT. Two implementations must be
    two ways of computing the SAME equation -- plexus2 allows them to differ in numerical assumptions,
    spatial representation, dimension or differentiability "while preserving the same biological
    semantics". These do not preserve it. This operator is a finite-volume discretisation of
    div(D grad c) on the tissue's own geometry; `graph_laplacian` is an unweighted graph average, and
    refining the mesh does NOT make it converge to the diffusion equation on a non-uniform tissue --
    it converges to something else. An unweighted Laplacian is not a coarser scheme for a
    finite-volume operator, it is a different constitutive law.
    "Transport is limited by the wall two cells share and diluted by the receiving cell's volume"
    versus "it is not" is a claim ABOUT THE TISSUE, so it belongs on the axis that carries claims.

    IT IS NOT A NEW OPERATOR EITHER: the biological transformation is the same one -- a signalling
    molecule moves between neighbouring cells, set=cell, kind=lateral, family=fields, reads chem,
    writes chem -- and plexus2 reserves a new contract for a DISTINCT biological transformation.
    The precedent is `cell_mechanics[model: monolayer]` against its `default`: one name, one
    contract, two hypotheses about what a cell is.

    (This docstring used to open "Same contract as `graph_laplacian` ... only the numerics differ",
    and then argue the opposite three paragraphs down. A reader could quote whichever half suited.)

        dc_i/dt = D * kappa * ( sum_j A_ij (c_j - c_i) ) / v_i

    i.e. the flux from cell j into cell i is weighted by the wall they SHARE (A_ij) and diluted by the
    RECEIVING cell's volume (v_i). Deformation therefore feeds back into the chemistry: a sliver wall
    passes proportionally less morphogen than a broad one, and a cell inflated to twice its volume
    dilutes what arrives twice as much. `graph_laplacian` has neither term (it is a plain unweighted
    neighbour average), which is the defect this implementation exists to remove.

    WHICH GEOMETRIC QUANTITIES ARE USED, AND WHY (read this before trusting the numbers)
      * A_ij -- NOT a true 3D interface area, because the mesh does not carry one. This substrate is an
        APICAL-SURFACE representation: a cell IS a face of a closed shell, so two neighbouring cells
        meet along a shared mesh EDGE (a 1-D segment), not along a stored 2-D lateral wall -- there is
        no basal sheet and no thickness field anywhere in `_mesh`. We therefore use the sanctioned
        proxy A_ij = l_ij * h, the shared-edge length times a notional epithelial thickness h. h is a
        single global constant and CANCELS EXACTLY against the kappa normalisation below, so it is not
        exposed as a parameter: the operator is driven by shared-edge LENGTH. Two cells that share
        several edges get all of them summed, which is the correct total interface.
      * v_i -- the per-cell WEDGE volume v_f = (1/3)(cen_f . N_f) from face_geometry_3d, the pyramid
        from the shell centre out to the cell. This is the model's own definition of cell volume (it is
        exactly what mesh_seed stores as the target V0f and what cell_mechanics's K_V term
        controls), so the chemistry dilutes by the same volume the mechanics conserves. It is
        origin-referenced, so it is only meaningful while the shell stays star-shaped about the origin
        -- true for the vesicle/bud/tube runs this campaign is about.
      * kappa = 1 / mean_i(S_i / v_i), with S_i = sum_j A_ij the cell's total shared interface. A
        mesh-wide scalar that non-dimensionalises the finite-volume operator. It is what makes this a
        DROP-IN for `graph_laplacian`: on a mesh whose walls are all equal and whose volumes are all
        equal, kappa*S_i/v_i = 1 and the expression collapses ALGEBRAICALLY to mean_j(c_j) - c_i, the
        degree-normalised graph Laplacian -- so d_a/d_h/chi keep the meaning every round_* preset
        calibrated them with, and only the DEVIATION from uniformity acts. (It also means a uniform
        inflation of the whole vesicle is normalised away; global dilution under growth is already
        handled structurally by cell_grow's conserve_amount, so applying it here too would
        double-count it.)

    STABILITY / STENCIL GAIN (derived, then measured -- do not re-guess it). The operator is -L for a
    weighted graph Laplacian whose row sums are row_i = kappa*S_i/v_i, mean 1 by construction but
    UNBOUNDED ABOVE: a cell squashed thin (small v_i, perimeter unchanged) acquires a large row weight
    and blows up an explicit Euler step that was safe for graph_laplacian's [-2,0] spectrum. `w_cap`
    clamps row_i, so by Gershgorin the spectrum lies in [-2*max_i(row_i), 0] subset [-2*w_cap, 0], i.e.

        stencil_gain(interface_weighted) = w_cap * stencil_gain(graph_laplacian)   -- worst case

    and the CFL bound dt*chi*max(d_a,d_h)*gain <= 1 tightens by that factor. MEASURED on an 80-cell
    vesicle (eigenvalues of the assembled matrix): pristine gain 1.25, budded 1.33, and only a violent
    40%-vertex-jitter mesh reaches 2.82 -- the w_cap=4 default is slack on every realistic mesh (it
    binds on 0/80 cells pristine/budded/15%-jitter, 1/80 at 40% jitter) yet still caps the tail: on
    that violent mesh the spectrum is -1.40 / -2.23 / -4.14 / -5.34 for w_cap = 1 / 2 / 4 / uncapped.
    `vol_floor` guards the other end -- a wedge volume that has collapsed or INVERTED (v_i <= 0 after a
    bad T1 / cap inversion) would divide by ~0 or flip the sign of the Laplacian, turning diffusion
    into anti-diffusion; the floor keeps it a diffusion."""
    SUPPORTED_DIMS = [3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["d_a", "d_h", "chi"]
    INPUTS = ["cell", "vertex"]; OUTPUTS = ["cell"]; READS = ["chem", "pos"]; WRITES = ["chem"]
    MAPS = ["E_srce", "E_trgt", "E_face"]
    MECHANISM_TAGS = ["diffusion", "finite_volume", "interface_weighted", "turing", "cross_scale"]
    REFERENCE = ("Okuda, S. et al. (2018). Combining Turing and 3D vertex models reproduces autonomous "
                 "multicellular morphogenesis of the tissue. Sci. Rep. 8:2386 (Appendix A: inter-cellular "
                 "flux ~ shared area / cell volume); Eymard, R., Gallouet, T. & Herbin, R. (2000). "
                 "Finite volume methods. Handb. Numer. Anal. 7:713-1018.")
    PARAM_ROLES = {"d_a": "activator_diffusivity", "d_h": "substrate_diffusivity", "chi": "spatial_scale",
                   "vol_floor": "collapsed_cell_volume_floor", "w_cap": "max_row_weight_vs_mesh_mean"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell"); self.vat = params.get("vertex_set", "vertex")
        self.d_a = float(params["d_a"]); self.d_h = float(params["d_h"]); self.chi = float(params["chi"])
        self.norm = bool(params.get("norm", True))
        # floor on v_i as a fraction of the median LIVE positive wedge volume -- see STABILITY above
        self.vol_floor = float(params.get("vol_floor", 0.05))
        # cap on a cell's row weight relative to the mesh mean (1.0) -- see STABILITY above
        self.w_cap = float(params.get("w_cap", 4.0))

    def forward(self, H, mask=None):
        from plexus.operators.vertex_ops import face_geometry_3d, ShapeEnergy3D
        lvl = H.level(self.at); vlvl = H.level(self.vat)
        chem = lvl.get("chem")
        m = getattr(vlvl, "_mesh", None)
        if m is None:
            # NOT a silent geometry fallback: the mesh simply does not exist yet (mesh_seed has not
            # run). Matching graph_laplacian's no-adjacency path -- emit nothing rather than guess.
            return {self.at: torch.zeros_like(chem)}
        nF = int(m["nF"]); Nv = int(m["Nv"]); dev = chem.device; dt = chem.dtype
        es = torch.as_tensor(m["E_srce"], device=dev, dtype=torch.long)   # robust to numpy after division
        et = torch.as_tensor(m["E_trgt"], device=dev, dtype=torch.long)
        ef = torch.as_tensor(m["E_face"], device=dev, dtype=torch.long)
        if nF == 0 or es.numel() == 0:
            return {self.at: torch.zeros_like(chem)}
        pos = vlvl.get("pos")[:Nv].to(dtype=dt)
        twin = ShapeEnergy3D._twin_faces(es, et, ef, Nv)      # cell on the far side of each shared edge
        shared = (twin != ef).to(dt)                          # 0 on an unpaired (boundary) half-edge
        w = (pos[et] - pos[es]).norm(dim=-1) * shared         # A_ij / h : the SHARED-WALL weight

        _, _, _, vf = face_geometry_3d(pos, es, et, ef, nF)   # per-cell wedge volume = the model's own v_i
        alive = m["alive"][:nF].to(device=dev, dtype=dt) if "alive" in m else torch.ones(nF, device=dev, dtype=dt)
        live_pos = vf[(vf > 0) & (alive > 0)]
        med = live_pos.median() if live_pos.numel() else vf.new_tensor(1.0)
        v = vf.clamp(min=float(self.vol_floor * med.clamp(min=1e-12)))   # collapsed/inverted-cell guard

        c = chem[:nF]
        agg = torch.zeros(nF, c.shape[1], device=dev, dtype=dt).index_add_(0, ef, w[:, None] * c[twin])
        S = torch.zeros(nF, device=dev, dtype=dt).index_add_(0, ef, w)   # total shared interface of cell i
        r = S / v                                              # per-cell conductance/volume [1/length]
        live = alive > 0
        rbar = r[live].mean() if int(live.sum()) else r.mean()
        row = r / rbar.clamp(min=1e-12)                        # RELATIVE row weight, mean 1 by construction
        kappa = 1.0 / rbar.clamp(min=1e-12)                    # h and the mesh length scale cancel here
        if not self.norm:                                      # parity with graph_laplacian's norm=False
            z = torch.zeros(nF, device=dev, dtype=dt).index_add_(0, ef, shared)   # shared-edge degree
            kappa = kappa * (z[live].mean() if int(live.sum()) else z.mean())
        lap_c = kappa * (agg - S[:, None] * c) / v[:, None]
        # The cap is measured on `row` (the mean-1 RELATIVE weight), never on kappa*r. Folding the
        # norm=False degree factor into the capped quantity made the clamp bind at w_cap/zbar on EVERY
        # cell of a perfectly uniform mesh -- a silent global 0.8x on the dodecahedron -- so norm=False
        # no longer reproduced graph_laplacian. Keeping the cap relative makes it deformation-triggered
        # only, and makes `norm` a pure change of overall scale exactly as it is in graph_laplacian.
        lap_c = lap_c * torch.clamp(self.w_cap / row.clamp(min=1e-12), max=1.0)[:, None] * alive[:, None]

        lap = torch.zeros_like(chem); lap[:nF] = lap_c
        coef = torch.tensor([self.d_a, self.d_h], device=dev, dtype=dt) * self.chi
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: (coef[None, :] * lap) * occ}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="gray_scott")
class CellReactGrayScott(Lateral):
    N_SPECIES = 2
    """Gray-Scott autocatalysis on the cell set (forked from Turing_vertex `react`), chem = [a, u]:
        da/dt =  u a^2 - (F + kk) a      (a = activator / autocatalyst)
        du/dt = -u a^2 + F (1 - u)       (u = substrate)
    `rate` time-scales the whole RD (pair it with `chi`) so the pattern develops in fewer frames."""
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["F", "kk"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]; MAPS = []
    MECHANISM_TAGS = ["reaction", "autocatalysis", "turing", "gray_scott"]
    PARAM_ROLES = {"F": "feed_rate", "kk": "kill_rate", "rate": "reaction_time_scale"}
    REFERENCE = "Gray, P. & Scott, S. K. (1984). Chem. Eng. Sci. 39:1087-1097; Pearson, J. E. (1993). Science 261:189-192."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.F = float(params["F"]); self.kk = float(params["kk"]); self.rate = float(params.get("rate", 1.0))
        # WHICH SPECIES THIS INSTANCE OWNS: 0 is the first pair (chem columns 0,1) and is the
        # default, so every existing spec is unchanged. A second RD system lives at chan 2.
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        # `chan` NAMES THE PAIR THIS INSTANCE OWNS. It was hard-wired to columns 0 and 1, so a
        # second species in the same buffer was unreachable: two `cell_chem_react` operators would both
        # have driven the same two columns and the second would simply have overwritten the first.
        c = self.chan
        a = chem[:, c]; u = chem[:, c + 1]
        uaa = u * a * a
        da = uaa - (self.F + self.kk) * a
        du = -uaa + self.F * (1.0 - u)
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        # zero in every column but this species', so the delta is additive with the other species'
        out = torch.zeros_like(chem)
        out[:, c] = self.rate * da
        out[:, c + 1] = self.rate * du
        return {self.at: out * occ}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields",
                   model="rock_paper_scissor")
class CellReactRPS(Lateral):
    """May-Leonard cyclic competition -- THREE species, each suppressing the next. chem = [u, v, w]:

        p = u + v + w
        du/dt = u (1 - p - a v)
        dv/dt = v (1 - p - a w)
        dw/dt = w (1 - p - a u)

    Every species is limited by the TOTAL population `p` (shared resource) and additionally
    suppressed by ONE named rival, cyclically: u loses to v, v to w, w to u. Nothing dominates, so
    the fixed point is unstable and the field breaks into travelling domains -- spirals on a
    2D sheet -- rather than settling. That is the qualitative difference from Gray-Scott, whose
    pattern is stationary once formed.

    `a` IS THE ASYMMETRY AND IT IS THE WHOLE MODEL. At a = 0 the three species merely compete for
    the shared resource `p` and the outcome is neutral coexistence; the cyclic term is what makes
    the dynamics non-transitive, and its size sets how fast domains invade one another.

    THIS IS NOT A COUPLING OPERATOR. The cyclic term is intrinsic to May-Leonard, not a cross term
    bolted onto three independent logistic species, so it belongs in the model rather than in a
    separate `cell_chem_couple`. Selecting it is a biological decision, which is why it is a
    `model=` and not an `implementation=`.
    """
    N_SPECIES = 3
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]; MAPS = []
    MECHANISM_TAGS = ["reaction", "competition", "cyclic_dominance", "non_transitive",
                      "may_leonard", "rock_paper_scissor"]
    PARAM_ROLES = {"a": "cyclic_suppression", "rate": "reaction_time_scale"}
    REFERENCE = ("May, R. M. & Leonard, W. J. (1975). SIAM J. Appl. Math. 29:243-253; "
                 "Reichenbach, T., Mobilia, M. & Frey, E. (2007). Nature 448:1046-1049.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        # 0.6 is the value the ParticleGraph `RD_RPS` generator ran, kept so the two agree.
        self.a = float(params.get("a", 0.6))
        self.rate = float(params.get("rate", 1.0))
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        u, v, w = _span(chem, self.chan, 3, type(self).__name__)
        p = u + v + w
        terms = (u * (1.0 - p - self.a * v),
                 v * (1.0 - p - self.a * w),
                 w * (1.0 - p - self.a * u))
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: _emit(chem, self.chan, terms, self.rate, occ)}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields",
                   model="gray_scott_coupled")
class CellReactGrayScottCoupled(Lateral):
    """TWO Gray-Scott systems that compete for each other's activator. chem = [a1, u1, a2, u2]:

        da1/dt =  u1 a1^2 - (F1 + k1) a1 - g a1 a2
        du1/dt = -u1 a1^2 + F1 (1 - u1)
        da2/dt =  u2 a2^2 - (F2 + k2) a2 - g a2 a1
        du2/dt = -u2 a2^2 + F2 (1 - u2)

    AT g = 0 THIS IS EXACTLY TWO INDEPENDENT SYSTEMS, term for term, and that is the test: a run at
    g = 0 must reproduce a pair of `gray_scott` instances bit for bit. Anything else means the
    refactor moved something.

    WHY ONE OPERATOR AND NOT TWO PLUS A CROSS TERM. A cross term reads columns the instance does
    not own, which breaks the rule that makes two reaction instances additive -- each writes zeros
    outside its own span. So a coupled model owns the whole four-column span. The consequence for a
    spec is that `cell_chem_react` is named ONCE in the schedule here, where the uncoupled
    two-species specs name it twice: the engine binds the i-th occurrence to the i-th instance, so
    naming it twice with one instance declared would run this operator twice and double its delta.

    THE COUPLING IS ACTIVATOR-ACTIVATOR, the mildest of the three plausible choices (the others
    being a shared substrate, and B inhibiting A's autocatalysis). It is symmetric and it is a
    LOSS to both -- `-g a1 a2` in each -- so it removes activator where the two patterns overlap
    and leaves them alone where they do not. The visible consequence is exclusion: the two motifs
    stop being able to occupy the same cells, which is exactly what superposing two independent
    systems cannot show.
    """
    N_SPECIES = 4
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["F", "kk"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]; MAPS = []
    MECHANISM_TAGS = ["reaction", "autocatalysis", "turing", "gray_scott", "competition", "coupled"]
    PARAM_ROLES = {"F": "feed_rate", "kk": "kill_rate", "F2": "feed_rate_2", "kk2": "kill_rate_2",
                   "gamma": "cross_suppression", "rate": "reaction_time_scale"}
    REFERENCE = "Gray, P. & Scott, S. K. (1984). Chem. Eng. Sci. 39:1087-1097 (coupling: this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.F = float(params["F"]); self.kk = float(params["kk"])
        # SYSTEM B FALLS BACK TO SYSTEM A's PARAMETERS. Two identical systems that differ only by
        # their coupling is the control this model exists to be compared against.
        self.F2 = float(params.get("F2", self.F)); self.kk2 = float(params.get("kk2", self.kk))
        self.gamma = float(params.get("gamma", 0.0))
        self.rate = float(params.get("rate", 1.0))
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        a1, u1, a2, u2 = _span(chem, self.chan, 4, type(self).__name__)
        x = self.gamma * a1 * a2
        terms = (u1 * a1 * a1 - (self.F + self.kk) * a1 - x,
                 -u1 * a1 * a1 + self.F * (1.0 - u1),
                 u2 * a2 * a2 - (self.F2 + self.kk2) * a2 - x,
                 -u2 * a2 * a2 + self.F2 * (1.0 - u2))
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: _emit(chem, self.chan, terms, self.rate, occ)}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="gierer_meinhardt")
class CellReactGiererMeinhardt(Lateral):
    """Gierer-Meinhardt activator(a)-inhibitor(h) -- the RD OKUDA uses (ref 37). chem = [a, h]:
        da/dt = gm_rho * a^2/h - mu_a * a + a0     (SELF-ENHANCING activator: the a^2/h AUTOCATALYSIS is the
        dh/dt = gm_rho * a^2   - mu_h * h            amplification feedback that self-maintains a localised PEAK)
    Paired (in cell_chem_diffuse) with a FAST inhibitor (d_h >> d_a via chi) -> lateral inhibition -> a stable
    localised activator peak WITH A GRADIENT (Okuda's tip spot), unlike Brusselator (decays the seed) or
    Gray-Scott (substrate-depletion). `rate` time-scales the reaction; a0 is a small basal activator source."""
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]; MAPS = []
    MECHANISM_TAGS = ["reaction", "autocatalysis", "self_enhancing", "turing", "gierer_meinhardt"]
    PARAM_ROLES = {"gm_rho": "production", "mu_a": "activator_decay", "mu_h": "inhibitor_decay", "a0": "basal_source"}
    REFERENCE = "Gierer, A. & Meinhardt, H. (1972). A theory of biological pattern formation. Kybernetik 12:30-39."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.gm_rho = float(params.get("gm_rho", 1.0)); self.mu_a = float(params.get("mu_a", 1.0))
        self.mu_h = float(params.get("mu_h", 1.0)); self.a0 = float(params.get("a0", 0.01))
        self.rate = float(params.get("rate", 1.0))
        # Meinhardt SATURATION kappa: a^2/(h(1+kappa a^2)) bounds the activator peak so self-enhancement can't
        # run away under growth (keeps the red spot CONFINED -> red_over_tip ~1 instead of flooding). 0 = off.
        self.sat = float(params.get("sat", 0.0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        a = chem[:, 0].clamp(min=0.0); h = chem[:, 1].clamp(min=1e-3)   # h>0: the a^2/h autocatalysis stays finite
        auto = a * a / (h * (1.0 + self.sat * a * a))                   # SATURATED autocatalysis (bounded peak)
        da = self.gm_rho * auto - self.mu_a * a + self.a0
        dh = self.gm_rho * a * a - self.mu_h * h
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: self.rate * torch.stack([da, dh], dim=1) * occ}


# `cell_grow`, AND THE OLD NAME IS GONE RATHER THAN ALIASED. Cedric, 8 August: "I always found
# cell_grow misleading -- is it morphogen, is it growth?" and then, on the alias:
# "I'm not a fan of alias and backward compatibility, this makes everything intricated and not
# readable. I prefer modifying prior spec files. Simplicity needs erasing here."
#
# It is growth. It does not produce a morphogen, it READS one and uses it as a per-cell rate: the
# morphogen is a GATE on this operator, and the composition space already declares that gate as an
# optional slot. With the gate open (`a_sw = 0`) the same operator is plain uniform growth. Naming
# the gate in the operator made the optional half look mandatory, and made the sibling pair
# unreadable -- `cell_grow` / `cell_divide` says what the schedule actually does.
@register_operator("cell_grow", set="vertex", kind="structural", family="population")
class Grow3D(Structural):
    # READS one species' activator; the span it points into is two wide because a Gray-Scott
    # system is. It never writes chem, so this only has to name the right column.
    N_SPECIES = 2
    """Cell growth on the vesicle: each cell's targets (A0 / P0 / v_eq) grow at a per-cell rate,
    and the per-cell volume elasticity in cell_mechanics then inflates the cell by force balance.
    This operator moves no vertex itself -- it raises what the cells ASK for, and the mechanics
    decides whether they get it.

    THE RATE IS `rate * (rho + Hill(activator))`, which is one operator covering both regimes:
      rho = 1, a_sw = 0   uniform growth, every cell at the same rate (what `cell_grow` did)
      rho = 0, a_sw > 0   growth only where the activator is high -> self-organised budding/coral
      in between          a baseline everywhere plus an activator-driven excess at the tips
    A cross-set coupling: reads cell.chem, writes the vertex mesh targets."""
    # MAY_MUTATE_INTEGRATED_STATE was declared False and the operator mutates it anyway: the
    # `conserve_amount` branch rescales cell.chem in place (c_j <- c_j * (v_old/v_new)) when the
    # cell's target volume grows. The engine's integration invariant caught it and refused to run
    # -- which is why the Turing x vertex (coral) movie on the site's front page could not be
    # regenerated at all, while the plain grow+divide movie could. Plain has no RD, so the
    # activator is identically zero and the branch is never reached; the tag only lies once
    # chemistry is present, i.e. exactly in the composition the campaign is about.
    #
    # The DECLARATION is what was wrong, not the behaviour. The rescale is a change of variable
    # forced by a volume change, not a dynamics delta -- the operator is registered
    # kind="structural", which is precisely the category the invariant exempts -- and the comment
    # at the branch records it as load-bearing (Okuda's intra-domain gradients come from it).
    # Returning it as an integrated delta would change the physics; declaring it honestly does not.
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["growth", "morphogen_driven", "budding", "cross_scale"]
    REFERENCE = "Okuda, S. et al. (2018). Combining Turing and 3D vertex models reproduces autonomous multicellular morphogenesis of the tissue. Sci. Rep. 8:2386."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex"); self.cat = params.get("cell_set", "cell")
        self.rate = float(params.get("rate", 0.01)); self.a_sw = float(params.get("a_sw", 0.20))
        # WHICH SPECIES GATES GROWTH: 0 (chem columns 0,1) by default, so existing specs are
        # unchanged; 2 reads a second RD system living in the same buffer.
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)
        # IS `a_sw` A VALUE OF THE ACTIVATOR OR A FRACTION OF ITS MAXIMUM? Default False = the
        # absolute reading every run in this project's history used, so no archived spec changes
        # meaning. See the gate itself in `forward` for why both are real mechanisms. `a_live` is
        # the floor below which a field counts as dead and the relative gate refuses to open --
        # 1e-3 against activators that reach 0.6-1.5 when alive and 1e-9 when they have collapsed.
        self.a_sw_rel = bool(params.get("a_sw_rel", False))
        self.a_live = float(params.get("a_live", 1e-3))
        # THE INHIBITOR: None = off, so every existing spec is unchanged. `inhib_sw` is a fraction
        # of the inhibitor's OWN maximum and `inhib_hill` its sharpness, mirroring a_sw/hill.
        _ic = params.get("inhib_chan", None)
        self.inhib_chan = None if _ic is None else int(_ic)
        self.inhib_sw = float(params.get("inhib_sw", 0.35))
        self.inhib_hill = float(params.get("inhib_hill", 4.0))
        self._inhib_applied = 0.0
        self.hill = float(params.get("hill", 3.0)); self.cap = float(params.get("cap", 2.5))
        from plexus.operators.vertex_ops import _engine_owns_clock
        self.every = _engine_owns_clock(params); self._k = 0
        # OKUDA uniform-cell mode: growth rate lambda = rate*(rho + Hill(a)); rho = baseline so ALL cells
        # cycle (the activator sets the RATE, not the size), and v_eq is capped at vth_frac*v_ref so every
        # cell oscillates in [~2/3, vth_frac]*v_ref -> uniform. rho=0 (default) = legacy activator-only bulge.
        self.rho = float(params.get("rho", 0.0)); self.vth_frac = float(params.get("vth_frac", 1.35))
        # OKUDA Appendix A: the morphogen is the AMOUNT m_j (conserved within the cell); the concentration
        # c_j=m_j/v_j is only READ by the kinetics. We store c_j, so growing v_j must DILUTE c_j to conserve
        # amount (else we silently CREATE mass each step -> spuriously feeds the tip). On (default) = correct.
        self.conserve_amount = bool(params.get("conserve_amount", True))

    def _advance(self, s_prev, hillv, m, v_ref):
        """The RATE LAW, and the only thing a `model=` variant of cell_grow changes.

        Returns the new per-cell linear scale. Volume is V0f_init * s**3, so a multiplicative step
        on `s` is exponential growth in volume -- which is the default and is deliberate: it is
        what Okuda's growth term does. Ginzberg, Kafri & Kirschner (Science 2015) name the
        consequence exactly: "with exponential growth, larger cells grow faster than do smaller
        cells, amplifying any existing size disparities". Measured on this campaign's own basis,
        the coefficient of variation of cell volume climbs 0.160 at seed to 0.33-0.53 by frame 900
        in every run. The variants below are the mechanisms that review says real cells use to
        stop that, written so the search can put them side by side.
        """
        return s_prev * (1.0 + self.rate * (self.rho + hillv))

    def forward(self, H, mask=None):
        vlvl = H.level(self.at); m = getattr(vlvl, "_mesh", None)
        if m is None:
            return {}
        self._k += 1                    # monotonic tick only -- D1: the engine owns the period
        clvl = H.level(self.cat)
        nF = m["nF"]
        dev = m["V0f"].device
        # NO CHEMISTRY IS UNIFORM GROWTH, NOT NO GROWTH. This used to `return {}` when the cell set
        # carried no `chem` state, which made a growth operator silently inert on any composition
        # without an RD pair -- and hid a dependency that is not real: the activator is a GATE, and
        # with the gate open (a_sw = 0, rho = 1) the rate does not consult it at all. That guard is
        # also why `cell_grow` had to exist as a separate operator. a = 0 is the honest reading
        # of "there is no activator here"; the Hill term evaluates to 0 and the rho baseline stands.
        if "chem" in clvl.state_schema:
            # `chan` PICKS THE SPECIES THAT GATES GROWTH. With two RD systems in one buffer this is
            # what lets them do different jobs -- species 0 driving growth while species 1 drives
            # death -- instead of both operators reading the same activator, which would be one
            # mechanism wearing two names.
            h0, _h1 = clvl.state_schema["chem"]
            a = clvl.state[:nF, h0 + self.chan].detach().to(dev)  # per-cell activator
        else:
            a = torch.zeros(nF, device=dev, dtype=m["V0f"].dtype)
        if "mg_scale" not in m or m["mg_scale"].shape[0] != nF:  # per-cell cumulative linear scale (capped)
            m["mg_scale"] = torch.ones(nF, device=dev, dtype=m["V0f"].dtype)
            m["A0_init"] = m["A0"].clone(); m["P0_init"] = m["P0"].clone(); m["V0f_init"] = m["V0f"].clone()
        # THE GATE'S HALF-POINT, AND WHAT IT IS A FRACTION OF.
        #
        # `a_sw` is ABSOLUTE by default: the Hill half-point sits at a fixed value of the activator
        # and does not move when the field does. Every other threshold on a chemistry field in this
        # substrate is relative to that field's own maximum -- `interface_tension.a_sw`,
        # `interface_push.a_sw`, `cell_die` chem_low, `cell_divide.orient_asw` -- and two
        # comments in this project (crew/basis.yaml on `a_sw_gated`, and the inhibitor branch
        # twenty lines below) already describe THIS one as a fraction of the maximum. They were
        # describing an intention, not the code.
        #
        # Both are defensible and they are not the same mechanism, so this is a switch and not a
        # correction:
        #   absolute  the gate stops opening when the chemistry dies. A field that decays to 1e-9
        #             drives no growth, and the tissue goes static -- which is what the 16 runs
        #             with act_max < a_sw did, honestly.
        #   relative  the gate always selects the same TOP FRACTION of cells, so `a_sw` means the
        #             same thing in a run peaking at 0.6 and one peaking at 1.5. That is what a
        #             sweepable lever has to do; measured over 154 runs the absolute gate sat
        #             anywhere from 0.24 of the field to above all of it.
        # The floor is what keeps `relative` honest: without it, a field decayed to noise still has
        # cells "above 35% of the maximum" and the tissue would grow on numerical dust forever.
        thr = self.a_sw
        if self.a_sw_rel:
            amax = float(a.max()) if a.numel() else 0.0
            thr = self.a_sw * amax if amax > self.a_live else float("inf")   # inf -> Hill term 0
        hillv = a ** self.hill / (thr ** self.hill + a ** self.hill + 1e-12)   # Hill activation in [0,1]
        # A SECOND MORPHOGEN THAT STOPS GROWTH. Cedric, 11 August: "make variants where the blue
        # morphogen stops cell growth, so that we see the blue and only red spots growing."
        #
        # Every growth law this campaign has run is purely ACTIVATING: `rate * (rho + hill(a))`, so
        # the only thing a morphogen can do is make a cell grow FASTER, and the slowest a cell can
        # grow is the rho baseline -- which is why six rounds produced broad lobes and never a
        # finger. A bulge sharpens into a finger when the tissue grows at the tip AND STOPS at the
        # flanks, and no single activating field can say "stop": it has no zero to reach.
        #
        # `inhib_chan` names a species whose HIGH values switch growth off, multiplicatively:
        #
        #     growth  <-  rate * (rho + hill(a_act)) * (1 - hill(a_inhib))
        #
        # so where the inhibitor is saturated the cell does not grow at all, baseline included.
        # This is lateral inhibition, and it is a different mechanism from a sharper gate: `hill`
        # narrows the shoulder of the activating curve, this puts a floor of ZERO under it.
        if self.inhib_chan is not None and "chem" in clvl.state_schema:
            _h0, _ = clvl.state_schema["chem"]
            b = clvl.state[:nF, _h0 + self.inhib_chan].detach().to(dev).clamp(min=0.0)
            # RELATIVE TO THE INHIBITOR'S OWN MAXIMUM, for the same reason a_sw is: an absolute
            # threshold against a field whose scale the chemistry sets is either always on or
            # always off, and this project has paid for that twice (rd_interface_tension, chem_low).
            bmax = float(b.max()) if b.numel() else 0.0
            if bmax > 1e-9:
                bn = b / bmax
                inh = bn ** self.inhib_hill / (self.inhib_sw ** self.inhib_hill
                                               + bn ** self.inhib_hill + 1e-12)
                m["inhib_frac"] = inh                     # recorded, so the renderer can show it
                hillv = hillv * (1.0 - inh)
                self._inhib_applied = float(inh.mean())
        s_prev = m["mg_scale"]                                    # per-cell scale BEFORE this tick (for the dilution rate)
        v_ref = float(m.get("v_ref", 1.0))                        # SEED-TIME MEDIAN cell volume (mesh_ops:220)
        s = self._advance(m["mg_scale"], hillv, m, v_ref)         # <-- the rate law; models override THIS only
        if self.rho > 0:                                             # OKUDA uniform-cell mode: cap v_eq per cell at
            s_cap = (self.vth_frac * v_ref / m["V0f_init"].clamp(min=1e-9)) ** (1.0 / 3.0)
            s = torch.minimum(s, s_cap.clamp(min=1.0))
        else:
            s = torch.clamp(s, max=self.cap)                        # legacy: activator-only bulge to `cap`
        m["mg_scale"] = s
        m["A0"] = m["A0_init"] * (s * s)                         # keep A0/P0/v_eq consistent (area~R^2, vol~R^3)
        m["P0"] = m["P0_init"] * s
        m["V0f"] = m["V0f_init"] * (s ** 3)
        m["V0"] = float(m["V0f"].sum())
        # THE SHELL RADIUS MUST GROW WITH THE CELLS. cell_mechanics carries a radial spring,
        #     E += K_R * sum_i (|x_i| - R0)^2                 (mesh_ops.py:85)
        # and R0 is set once at seeding (:217). `cell_grow` rescales it (:409); this operator
        # never did. So with K_R = 0.4 the mechanics pinned the shell at the seed radius while the
        # cells' target volumes grew sixteenfold, and the sheet had nowhere to put the extra area
        # but through itself. Measured on mini_grow_divide_bigger: rays cast from the tissue
        # centroid cross the surface exactly once at frame 384 (100% of them) and 13 times at
        # frame 423. The genus check reported "sphere (as built)" at every one of those frames --
        # Euler characteristic is combinatorial and cannot see a shell folded through its own
        # centre, which is why this survived until premise 11 was written.
        #
        # The radius of the sphere enclosing the current target volume, NOT the measured mean
        # radius: R0 must express what the cells are ASKING for. Setting it from |x| would make
        # the spring chase the shell's own excursions and quietly penalise a growing bud, which is
        # the one shape this campaign exists to produce.
        if "R0" in m:
            m["R0"] = float((3.0 * max(m["V0"], 1e-12) / (4.0 * np.pi)) ** (1.0 / 3.0))
        if self.conserve_amount and "chem" in clvl.state_schema:
            # conserve molecule AMOUNT: c_j <- c_j * (v_old/v_new) = c_j * (s_prev/s)^3 as v_eq grows ~ s^3.
            # Makes dilution STRUCTURAL (no continuum -c(div.v) term); it is LOAD-BEARING (Okuda's intra-domain
            # gradients come from it) -> keep it, don't cancel. The flood is a gamma (rate) problem, fixed elsewhere.
            g_vol = (s / s_prev.clamp(min=1e-9)) ** 3
            cst = clvl.state.clone()
            # THE ACTIVATOR COLUMN ONLY. Diluting BOTH columns extinguishes Gray-Scott outright:
            # measured, 1% loss per step kills the pattern within 250 steps, while the undiluted
            # one reaches 53% coverage by step 250 and holds indefinitely. Diluting either column
            # alone survives (a_max at t=60: 0.704 / 0.686); both together does not (0.047),
            # because the substrate's own feed term F(1-u) is what pulls u back up and diluting u
            # fights it directly. The activator has no source at all except its own autocatalysis,
            # which is QUADRATIC, so it is the one that genuinely loses material when a cell grows.
            # Correct physics on a fragile mechanism is still a broken model; this keeps the
            # physics where it belongs and stops it destroying the pattern it is meant to shape.
            cst[:nF, h0:h0 + 1] = cst[:nF, h0:h0 + 1] / g_vol.clamp(min=1e-9)[:, None]
            clvl.state = cst
        return {}


# =========================================================== cell_grow: the size-control models
# Three mechanisms from Ginzberg, Kafri & Kirschner, "On being the right (cell) size", Science 348
# (2015) and Ginzberg et al., eLife 7:e26957 (2018). They are `model=` variants, not replacements,
# because each is a different biological hypothesis at the same slot and the search should put
# them side by side. The default `cell_grow` above has NO size control at all -- its rate reads the
# morphogen and never the cell's own size -- which by the review's Eq. 2 means cell-size variance
# can only ever increase, and in this campaign's basis it does: vol_cv 0.160 -> 0.53.
#
# WHY THIS MATTERS BEYOND TIDINESS. The review's Fig 2 sets a healthy mammary epithelium, uniform
# in cell size, beside a pleomorphic tumour that is not, and states that "pleomorphism ... is a
# histological characteristic of many malignant lesions". This project is trying to grow an
# epithelial TUBE -- a coherent structure -- and its tissue drifts toward the second picture.


@register_operator("cell_grow", model="sizer", set="vertex", kind="structural", family="population")
class Grow3DSizer(Grow3D):
    """Growth rate falls with the cell's own size: small cells grow faster, large ones slower.

    The review's Fig 3B, and the eLife paper measures it directly -- twice per cell cycle the
    correlation between size and subsequent growth rate goes negative, and cell-size variance
    falls while the cells are still growing. The rate is multiplied by

        f = (v_ref / v_now) ** size_gain,  clamped to [1/f_max, f_max]

    so a cell at the reference volume is unchanged, one at half it grows 2**size_gain faster, and
    one at twice it grows that much slower. size_gain = 0 recovers the default exactly, which is
    the null this variant must be run against.
    """
    MECHANISM_TAGS = ["growth", "size_control", "sizer", "negative_feedback"]
    REFERENCE = ("Ginzberg, M.B., Kafri, R. & Kirschner, M.W. (2015). On being the right (cell) "
                 "size. Science 348:1245075; Ginzberg et al. (2018) eLife 7:e26957.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.size_gain = float(params.get("size_gain", 1.0))
        self.f_max = float(params.get("f_max", 4.0))

    def _advance(self, s_prev, hillv, m, v_ref):
        v_now = m["V0f"].clamp(min=1e-9)
        f = (v_ref / v_now) ** self.size_gain
        f = torch.clamp(f, 1.0 / max(self.f_max, 1e-9), self.f_max)
        return s_prev * (1.0 + self.rate * (self.rho + hillv) * f)


@register_operator("cell_grow", model="balance", set="vertex", kind="structural", family="population")
class Grow3DBalance(Grow3D):
    """Size emerges from a synthesis/degradation balance, with no size sensor anywhere.

        dV/dt = k_syn * (rho + Hill(a))  -  k_deg * V

    The review, on how a cell could regulate size without measuring it: "If, for example, cells
    synthesise proteins at a fixed rate but degrade them at a rate that is proportional to their
    total cell size, net growth would slow as cell size increases." It then flags the non-trivial
    part, which is why this is a separate model and not a tweak: "this requires that degradation
    depend on the total AMOUNT of protein in the cell, rather than the concentration."

    The steady state is V* = k_syn * (rho + Hill(a)) / k_deg -- so the morphogen sets the TARGET
    SIZE here, not the rate, which is a genuinely different hypothesis from every other variant.
    """
    MECHANISM_TAGS = ["growth", "size_control", "synthesis_degradation_balance", "homeostasis"]
    REFERENCE = ("Ginzberg, M.B., Kafri, R. & Kirschner, M.W. (2015). On being the right (cell) "
                 "size. Science 348:1245075.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        # k_syn is expressed as a multiple of v_ref per unit time, so the balance point lands near
        # v_ref at k_deg = rate and the parameter has the same meaning at any mesh scale.
        self.k_syn = float(params.get("k_syn", 1.0))
        self.k_deg = float(params.get("k_deg", 1.0))

    def _advance(self, s_prev, hillv, m, v_ref):
        v_now = (m["V0f_init"] * s_prev ** 3).clamp(min=1e-9)
        dv = self.rate * (self.k_syn * v_ref * (self.rho + hillv) - self.k_deg * v_now)
        v_new = (v_now + dv).clamp(min=1e-9)
        return (v_new / m["V0f_init"].clamp(min=1e-9)) ** (1.0 / 3.0)


@register_operator("cell_grow", model="timer", set="vertex", kind="structural", family="population")
class Grow3DTimer(Grow3D):
    """Grow at whatever rate lands the cell on its target size after `cycle_frames` frames.

    The partner of `cell_divide model: timer`: if division fires on the clock, growth has to be the
    thing that guarantees the cell is the right size when it does. This is the review's Fig 3B
    taken to its limit -- the rate is set entirely by the size deficit rather than modulated by it:

        per-frame volume factor = (v_target / v_now) ** (1 / cycle_frames)

    which is a proportional controller with time constant `cycle_frames`, so a cell far below
    target grows fast and one at target holds. Together with a clock-driven division that gives
    every cell the same age AND the same size at division, which is the strongest size homeostasis
    of the four and therefore the right upper bound to measure the others against.

    The morphogen still decides WHERE: the target is scaled by (rho + Hill(a)) / (rho + 1), so a
    red cell aims for the full `vth_frac * v_ref` and a white one for the rho fraction of it.
    """
    MECHANISM_TAGS = ["growth", "size_control", "timer", "target_size", "proportional_control"]
    REFERENCE = ("Ginzberg, M.B., Kafri, R. & Kirschner, M.W. (2015). On being the right (cell) "
                 "size. Science 348:1245075.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.cycle_frames = float(params.get("cycle_frames", 100.0))

    def _advance(self, s_prev, hillv, m, v_ref):
        v_now = (m["V0f_init"] * s_prev ** 3).clamp(min=1e-9)
        share = (self.rho + hillv) / max(self.rho + 1.0, 1e-9)   # the morphogen sets WHERE, as a target
        v_tgt = (self.vth_frac * v_ref * share).clamp(min=1e-9)
        g = (v_tgt / v_now) ** (1.0 / max(self.cycle_frames, 1.0))
        return s_prev * g ** (1.0 / 3.0)


@register_operator("interface_tension", set="vertex", kind="lateral", family="mechanics")
class InterfaceLineTension3D(Lateral):
    """A PURSE-STRING line tension on the RED/WHITE activator interface -- and NOTHING ELSE.

    SPLIT FROM `rd_interface_tension` ON 10 AUGUST, and the split is the point. That operator carried
    two terms under one name:

        E = K_purse * Sigma_iface l_e   -   K_extrude * Sigma_red a*r
            [___ ordinary physics ___]       [_ the answer written into the objective _]

    The first is a line tension on the interface ring: real vertex-model mechanics, the same kind of
    term `cell_mechanics` already charges for, and how a purse-string actually works. The second is
    an energy that FALLS as red cells move outward -- it does not model a force, it pays the tissue
    to produce the morphology the campaign is searching for. A run carrying it can only be a control.

    ONE NAME OVER BOTH TERMS COST FOUR ROUNDS. `K_extrude` measured 0.0 in all 78 specs that have
    ever carried this operator, so nothing the campaign ran was ever forced -- and the Grounder
    still reported r028 as "the same extrude-forced star for a fourth round", on three runs
    (`r028_00`, `03`, `06`) whose specs contain no such operator at all. `user_input.md` section 3
    had already told it to retract exactly that verdict about `r017_07`. A reader who sees a
    plausible name cannot check a term that is not in front of them, so the terms are now two
    operators: this one, and `interface_push` below, which the loop vocabulary does not
    contain. There is no longer a setting of anything in the search space that pushes.

    Cross-set: reads cell.chem, forces on vertices; EMIT velocity (the engine integrates it beside
    shape_energy). Bounded-Euler substeps of -grad E."""
    SUPPORTED_DIMS = [3]; EMIT = "velocity"; DIFFERENTIABLE = True
    INPUTS = ["vertex", "cell"]; OUTPUTS = ["vertex"]; READS = ["pos", "chem"]; WRITES = ["pos"]
    MAPS = ["E_srce", "E_trgt", "E_face"]
    MECHANISM_TAGS = ["interface_tension", "purse_string", "tube", "oriented", "cross_scale"]
    PARAM_ROLES = {"K_purse": "interface_line_tension", "a_sw": "red_threshold"}
    REFERENCE = "Plexus (this work); purse-string / apical-constriction tubulation after Okuda, S. et al. (2018). Sci. Rep. 8:2386."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex"); self.cat = params.get("cell_set", "cell")
        self.K_purse = float(params.get("K_purse", 1.0))
        # 0.6, AND IT WAS 1.0 -- A DEFAULT THAT CANNOT FIRE. The gate below is
        # `red = a > a_sw * amax`, so a_sw = 1.0 asks for cells STRICTLY ABOVE the maximum: the
        # empty set, by construction, at every operating point and for every value of K_purse.
        #
        # This is the SECOND time this operator has been written off as inert without ever having
        # run. The first was an absolute threshold against a field whose median maximum is 0.000,
        # fixed by making a_sw a fraction -- and the fix left a default that is a fraction of one.
        # Route A then swept K_purse [0, 0.25, 3, 6] on b_gs_gated_shaping, whose spec omits a_sw,
        # and got four runs identical to four significant figures with `acted = 0` on all of them.
        # Reported as "K_purse is inert"; nothing was measured. 0.6 is the composition space's own
        # declared default and means "the top 40% of the field is red".
        self.a_sw = float(params.get("a_sw", 0.6)); self.eta = float(params.get("eta", 0.05))
        self.cap_frac = float(params.get("cap_frac", 0.10)); self.iters = int(params.get("iters", 4))

    def forward(self, H, mask=None):
        from plexus.operators.vertex_ops import ShapeEnergy3D
        vlvl = H.level(self.at); m = getattr(vlvl, "_mesh", None); clvl = H.level(self.cat)
        if m is None or "chem" not in clvl.state_schema:
            return {}
        nF = int(m["nF"]); Nv = int(m["Nv"]); dev = vlvl.state.device; dt = vlvl.state.dtype
        es = torch.as_tensor(m["E_srce"], device=dev, dtype=torch.long)      # robust to numpy/tensor after division
        et = torch.as_tensor(m["E_trgt"], device=dev, dtype=torch.long)
        ef = torch.as_tensor(m["E_face"], device=dev, dtype=torch.long)
        h0, _ = clvl.state_schema["chem"]
        a = clvl.state[:nF, h0].detach().to(dev)
        # `a_sw` IS A FRACTION OF THE FIELD'S OWN MAXIMUM, NOT AN ABSOLUTE VALUE.
        #
        # It used to be `a > self.a_sw` with a_sw declared (0.2, 6.0) and defaulted to 0.5, while
        # the activator's own ceiling is whatever the chemistry produces -- measured across 78
        # campaign runs, act_max_final has a MEDIAN OF 0.000 and a maximum of 1.541. So the entire
        # declared range sat above the field, `red.sum()` was 0, and the operator returned {} on
        # every one of 800 scheduled frames. The acted-ledger recorded `rd_interface_tension: 0`
        # and the Analyst reported it "inert" for two rounds without being able to say why. Nine
        # edits, 10% of a campaign, on an operator that could not fire at any legal setting.
        #
        # Fixing the RANGE would have been a patch: the ceiling moves with the chemistry, so the
        # next parent could put it out of reach again. `cell_chem_from_shape` in this same repo
        # standardises its feature for exactly this reason -- so `beta` means one thing whatever
        # the units -- and its docstring names the alternative as finding F009. A threshold
        # relative to the field cannot be outside the field.
        amax = float(a.max()) if a.numel() else 0.0
        red = (a > self.a_sw * amax).to(dt) if amax > 0 else torch.zeros_like(a)
        twin = ShapeEnergy3D._twin_faces(es, et, ef, Nv)        # neighbour cell across each half-edge
        iface = (red[ef] != red[twin]).to(dt)                   # 1 on the red/white interface half-edges (the ring)
        if float(red.sum()) < 1.0 or float(iface.sum()) < 1.0:  # no red spot / no interface yet -> nothing to do
            return {}
        px0, px1 = vlvl.state_schema["pos"]
        x0 = vlvl.state[:, px0:px1].detach()
        x = x0[:Nv].clone()
        cap = self.cap_frac * float((x0[et] - x0[es]).norm(dim=-1).mean().clamp(min=1e-3))
        for _ in range(self.iters):              # DIRECT forces (no autograd): the purse-string alone
            force = torch.zeros(Nv, 3, device=dev, dtype=dt)
            d = x[et] - x[es]; u = d / (d.norm(dim=-1, keepdim=True) + 1e-9)   # interface edge shortens
            f = (self.K_purse * iface)[:, None] * u
            force.index_add_(0, es, f); force.index_add_(0, et, -f)
            x = x + (self.eta * force).clamp(-cap, cap)
        vel = torch.zeros_like(x0)
        vel[:Nv] = (x - x0[:Nv]) / max(float(getattr(H.config, "dt", 1.0)), 1e-6)
        occ = vlvl.occ[:, None] if getattr(vlvl, "occ", None) is not None else 1.0
        return {self.at: vel * occ}


@register_operator("interface_push", set="vertex", kind="lateral", family="mechanics")
class ExtrusionForcing3D(Lateral):
    """THE DISQUALIFIED TERM, ON ITS OWN AND UNDER ITS OWN NAME. A run carrying this is a control.

    Split out of `rd_interface_tension` on 10 August. The energy is

        E = - K_extrude * Sigma_red a * r

    -- it FALLS as activator-high cells move outward, so the tissue is paid, per frame, to do the
    thing the campaign is searching for. That is not a mechanism; it is the answer written into the
    objective. Growth, division, adhesion and line tension are hypotheses about what cells DO. This
    is a hypothesis about what the experimenter WANTS, and any protrusion it produces is evidence
    about the term and not about the tissue.

    IT IS NOT IN THE LOOP VOCABULARY, deliberately -- `composition_space.OPERATORS` does not contain
    it, so no `add_op` or `set_param` the Proposer can write will reach it. It exists so that the
    forcing CAN be run when a control genuinely calls for one, and so that running it is an explicit
    act that the record shows as such. Keeping it inside the tension operator made forcing a
    parameter of a sound mechanism, which is how "extrude-forced" was reported for four rounds
    across runs whose K_extrude was 0.0 -- in fact whose specs held no such operator at all.

    Same gate as the tension operator: `red = a > a_sw * amax`, a fraction of the field's own
    maximum, because a threshold relative to the field cannot be outside the field."""
    SUPPORTED_DIMS = [3]; EMIT = "velocity"; DIFFERENTIABLE = True
    INPUTS = ["vertex", "cell"]; OUTPUTS = ["vertex"]; READS = ["pos", "chem"]; WRITES = ["pos"]
    MAPS = ["E_srce", "E_trgt", "E_face"]
    MECHANISM_TAGS = ["extrusion", "forcing", "control_only", "disqualified"]
    PARAM_ROLES = {"K_extrude": "normal_extrusion_forcing", "a_sw": "red_threshold"}
    REFERENCE = "Plexus (this work) -- a forcing term retained only as an explicit control."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex"); self.cat = params.get("cell_set", "cell")
        self.K_extrude = float(params.get("K_extrude", 0.5))
        self.a_sw = float(params.get("a_sw", 0.6)); self.eta = float(params.get("eta", 0.05))
        self.cap_frac = float(params.get("cap_frac", 0.10)); self.iters = int(params.get("iters", 4))

    def forward(self, H, mask=None):
        from plexus.operators.vertex_ops import face_geometry_3d
        vlvl = H.level(self.at); m = getattr(vlvl, "_mesh", None); clvl = H.level(self.cat)
        if m is None or "chem" not in clvl.state_schema:
            return {}
        nF = int(m["nF"]); Nv = int(m["Nv"]); dev = vlvl.state.device; dt = vlvl.state.dtype
        es = torch.as_tensor(m["E_srce"], device=dev, dtype=torch.long)
        et = torch.as_tensor(m["E_trgt"], device=dev, dtype=torch.long)
        ef = torch.as_tensor(m["E_face"], device=dev, dtype=torch.long)
        h0, _ = clvl.state_schema["chem"]
        a = clvl.state[:nF, h0].detach().to(dev)
        amax = float(a.max()) if a.numel() else 0.0
        red = (a > self.a_sw * amax).to(dt) if amax > 0 else torch.zeros_like(a)
        if float(red.sum()) < 1.0:
            return {}
        px0, px1 = vlvl.state_schema["pos"]
        x0 = vlvl.state[:, px0:px1].detach()
        x = x0[:Nv].clone()
        cap = self.cap_frac * float((x0[et] - x0[es]).norm(dim=-1).mean().clamp(min=1e-3))
        redpush = (self.K_extrude * a.clamp(min=0.0) * red)          # per-cell outward magnitude
        for _ in range(self.iters):
            force = torch.zeros(Nv, 3, device=dev, dtype=dt)
            _, _, cen, _ = face_geometry_3d(x, es, et, ef, nF)
            cdir = cen / (cen.norm(dim=-1, keepdim=True) + 1e-9)
            force.index_add_(0, es, (redpush[ef])[:, None] * cdir[ef] / 3.0)
            x = x + (self.eta * force).clamp(-cap, cap)
        vel = torch.zeros_like(x0)
        vel[:Nv] = (x - x0[:Nv]) / max(float(getattr(H.config, "dt", 1.0)), 1e-6)
        occ = vlvl.occ[:, None] if getattr(vlvl, "occ", None) is not None else 1.0
        return {self.at: vel * occ}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="brusselator")
class CellReactBrusselator(Lateral):
    """`brusselator` implementation of cell_chem_react (transposed verbatim from Turing_vertex fig4_coral),
    chem = [a, h] (activator, inhibitor):
        da/dt = gamma ( A - (B+1) a + a^2 h )
        dh/dt = gamma ( B a - a^2 h )
    Homogeneous steady state (a,h) = (A, B/A); Turing-unstable for B > 1 + A^2. Pair with the noise
    seed (steady state + noise) and a fast-inhibitor diffusion ratio (d_h >> d_a) -> smooth coral."""
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["gamma", "A", "B"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]; MAPS = []
    MECHANISM_TAGS = ["reaction", "activator_inhibitor", "turing", "brusselator"]
    REFERENCE = "Prigogine, I. & Lefever, R. (1968). Symmetry breaking instabilities in dissipative systems. J. Chem. Phys. 48:1695-1700."
    PARAM_ROLES = {"gamma": "reaction_rate", "A": "feed", "B": "conversion"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.gamma = float(params["gamma"]); self.A = float(params["A"]); self.B = float(params["B"])

    def forward(self, H, mask=None):
        lvl = H.level(self.at); chem = lvl.get("chem")
        a = chem[:, 0]; h = chem[:, 1]; a2h = a * a * h
        da = self.gamma * (self.A - (self.B + 1.0) * a + a2h)
        dh = self.gamma * (self.B * a - a2h)
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: torch.stack([da, dh], dim=1) * occ}



# ==========================================================================================================
# FROM `discovery_okuda/ops/shape_chem_ops.py` -- #!/usr/bin/env python
# ==========================================================================================================
F_CEIL = 0.11


# --------------------------------------------------------------------------- shared machinery
def _cell_adjacency(es, et, ef, nF):
    """(src, dst) cell pairs: two cells are neighbours iff they share a mesh edge."""
    key = np.minimum(es, et).astype(np.int64) * (int(max(es.max(), et.max())) + 1) \
        + np.maximum(es, et)
    o = np.argsort(key, kind="stable")
    k, f = key[o], ef[o]
    src, dst = [], []
    i = 0
    while i < len(k):
        j = i
        while j + 1 < len(k) and k[j + 1] == k[i]:
            j += 1
        if j > i:
            for a in range(i, j + 1):
                for b in range(a + 1, j + 1):
                    if f[a] != f[b]:
                        src += [f[a], f[b]]; dst += [f[b], f[a]]
        i = j + 1
    return np.asarray(src, np.int64), np.asarray(dst, np.int64)


def _np(x):
    """Mesh arrays are torch tensors ON THE GPU in a real run and numpy in the self-test. Assuming
    numpy crashed the first end-to-end launch on cuda -- `can't convert cuda:0 device type tensor
    to numpy` -- after the CPU tests had all passed."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _standardise(phi, alive):
    """Median-centred, MAD-scaled, clipped. See the module docstring: without this, `beta` means a
    different physical quantity in each implementation and the sweep axis is meaningless."""
    ok = np.isfinite(phi) & (alive > 0)
    if ok.sum() < 8:
        return np.zeros_like(phi)
    x = phi[ok]
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    if mad < 1e-12:
        return np.zeros_like(phi)                      # a uniform field carries no signal
    out = np.zeros_like(phi)
    out[ok] = np.clip((x - med) / mad, -4.0, 4.0)      # clip: one spike must not drive the feed
    return out


class _ShapeToChemBase(Lateral):
    """The contract. Subclasses supply `_feature(...) -> per-cell scalar` and nothing else."""
    SUPPORTED_DIMS = [3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = False
    INPUTS = ["cell", "vertex"]; OUTPUTS = ["cell"]; READS = ["chem", "pos"]; WRITES = ["chem"]
    MAPS = ["E_srce", "E_trgt", "E_face"]
    REQUIRES_PARAMS = ["beta"]
    MECHANISM_TAGS = ["shape_to_chemistry", "mechanochemical_feedback", "cross_scale", "closes_the_loop"]
    REFERENCE = ("Okuda, S. et al. (2018). Sci. Rep. 8:2386 (the shape-chemistry loop this closes); "
                 "Dupont, S. et al. (2011). Nature 474:179-183 (YAP/TAZ mechanotransduction); "
                 "Pearson, J. E. (1993). Science 261:189-192 (F selects the Gray-Scott morphology).")
    PARAM_ROLES = {"beta": "shape_feedback_strength", "F0": "baseline_feed_rate"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell"); self.vat = params.get("vertex_set", "vertex")
        self.beta = float(params["beta"])
        self.F0 = float(params.get("F0", 0.055))       # match cell_chem_react's feed, or it fights it
        self.rate = float(params.get("rate", 1.0))     # same time-scaling as cell_chem_react

    def _feature(self, pt, m, es, et, ef, nF):
        raise NotImplementedError

    def forward(self, H, mask=None):
        clvl = H.level(self.at); vlvl = H.level(self.vat)
        m = getattr(vlvl, "_mesh", None)
        if m is None or "chem" not in clvl.state_schema:
            return {}
        chem = clvl.get("chem")
        if self.beta == 0.0:
            return {self.at: torch.zeros_like(chem)}   # the NULL, and it must remain runnable
        nF = int(m["nF"])
        es = _np(m["E_srce"]); et = _np(m["E_trgt"]); ef = _np(m["E_face"])
        live = ef < nF
        es, et, ef = es[live], et[live], ef[live]
        pt = vlvl.get("pos")[:int(m["Nv"])].detach().cpu().numpy().astype(np.float64)
        alive = _np(m["alive"])[:nF] if "alive" in m else np.ones(nF)
        phi = self._feature(pt, m, es, et, ef, nF)
        if phi is None:                                # precondition absent: no-op, never a guess
            return {self.at: torch.zeros_like(chem)}
        w = _standardise(np.asarray(phi, float), alive)
        dev, dt = chem.device, chem.dtype
        wt = torch.zeros(chem.shape[0], device=dev, dtype=dt)
        wt[:nF] = torch.as_tensor(w, device=dev, dtype=dt)
        # F_j = F0 (1 + beta phihat_j). The Gray-Scott feed acts on the SUBSTRATE: du/dt += F(1-u).
        # We contribute only the DIFFERENCE from the baseline feed cell_chem_react already applies, so
        # the two operators compose instead of double-counting.
        u = chem[:, 1]
        # A FEED RATE CANNOT BE NEGATIVE, and letting it go negative is not merely unphysical --
        # it is unstable. The substrate obeys du/dt = F (1 - u); with F < 0 and u < 1 the term is
        # negative, u falls, (1 - u) grows, and the whole thing diverges exponentially. Measured
        # before the clamp: `tension` at beta = 1.5 reached act_max 1.4e16 in forty frames, and
        # `apical_area` overflowed to NaN. The multiplier is clamped at zero, which caps the
        # feedback at "this cell is not fed at all" rather than "this cell is drained".
        # THE MODULATED FEED MUST STAY INSIDE THE GRAY-SCOTT REGIME, not merely stay positive.
        # Clamping only at zero was not enough: with phihat clipped at +/-4 and beta = 1.5 the
        # multiplier reached 7, so F rose to 0.385 -- far outside Pearson's diagram, which is
        # explored for F <~ 0.11. Measured consequence, in order: the activator climbed past 1.6
        # (Gray-Scott lives near 0.4), then u a^2 drained the substrate NEGATIVE at frame 15, and
        # the explicit step diverged to +/-inf by frame 25. A feedback strong enough to leave the
        # model's own parameter region is not a mechanism, it is a blow-up.
        F = torch.clamp(self.F0 * (1.0 + self.beta * wt), min=0.0, max=F_CEIL)
        dF = F - self.F0
        out = torch.zeros_like(chem)
        out[:, 1] = self.rate * dF * (1.0 - u)
        occ = clvl.occ[:, None] if getattr(clvl, "occ", None) is not None else 1.0
        return {self.at: out * occ}


# --------------------------------------------------------------------------- implementations
@register_operator("cell_chem_from_shape", set="cell", kind="lateral", family="fields",
                   model="curvature")
class ShapeToChemCurvature(_ShapeToChemBase):
    """The chemistry listens to CURVATURE -- the feedback Okuda's framing implies.

    Discrete mean curvature on the CELL graph: how far a cell's centroid sits from the mean of its
    neighbours' centroids, projected on its own outward normal, divided by the squared spacing.
    Positive where the sheet bulges outward, negative in a dimple, and ~1/R on a sphere of radius R.
    A proxy rather than the cotangent-Laplacian curvature, which is why it is certified against
    spheres of known radius in the self-test below rather than asserted.
    """
    MECHANISM_TAGS = _ShapeToChemBase.MECHANISM_TAGS + ["curvature_sensing"]

    def _feature(self, pt, m, es, et, ef, nF):
        area, _, cen, _ = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(es),
                                           torch.as_tensor(et), torch.as_tensor(ef), nF)
        cen = cen.numpy()
        nrm = np.zeros((nF, 3))                        # Newell normal per cell, outward
        for a, b, f in zip(es, et, ef):
            nrm[f] += np.cross(pt[a] - cen[f], pt[b] - cen[f])
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = nrm / np.maximum(ln, 1e-12)
        src, dst = _cell_adjacency(es, et, ef, nF)
        if not len(src):
            return None
        deg = np.bincount(src, minlength=nF).astype(float)
        nb = np.zeros((nF, 3))
        for d in range(3):
            nb[:, d] = np.bincount(src, weights=cen[dst][:, d], minlength=nF)
        nb /= np.maximum(deg, 1)[:, None]
        delta = nb - cen                                # umbrella vector
        # Divide by the NEIGHBOUR SPACING squared, not by |delta|^2. On a sphere the tangential
        # parts of the umbrella cancel, so |delta| is itself only ~L^2/2R -- dividing by it gives
        # 2R/L^2, which GROWS with radius. That reads as 1/R only if you hold the cell count fixed
        # so that L scales with R, which is exactly how the first version of this passed its own
        # test. With the spacing: delta.n = -L^2/2R, so H = 2 (delta.n) / L^2 = 1/R. Correct, and
        # now independent of how finely the sphere is meshed.
        sp = np.zeros(nF)
        np.add.at(sp, src, np.linalg.norm(cen[dst] - cen[src], axis=1))
        L = sp / np.maximum(deg, 1)
        return -2.0 * (delta * nrm).sum(1) / np.maximum(L ** 2, 1e-12)


@register_operator("cell_chem_from_shape", set="cell", kind="lateral", family="fields",
                   model="tension")
class ShapeToChemTension(_ShapeToChemBase):
    """The chemistry listens to CORTICAL TENSION -- mechanotransduction.

    tension_j = 2 kP (P_j - P0_j) + Gamma P_j + Lambda, the same quantity analyze_forces.cell_mechanics
    reports. This is the best-evidenced feedback in real epithelia: YAP/TAZ translocates to the
    nucleus under tension and Piezo1 is a stretch-gated channel, so "tense cells signal differently"
    is not a modelling convenience.

    NEEDS the mechanical targets P0, which exist only once a mechanics operator has run.
    """
    MECHANISM_TAGS = _ShapeToChemBase.MECHANISM_TAGS + ["mechanotransduction", "tension_sensing"]

    def _feature(self, pt, m, es, et, ef, nF):
        if "P0" not in m:
            return None                                 # precondition absent -> no-op, not a guess
        _, perim, _, _ = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(es),
                                          torch.as_tensor(et), torch.as_tensor(ef), nF)
        P = perim.numpy()
        P0 = np.asarray(_np(m["P0"])[:nF], float)
        mech = m.get("mech", {}) or {}
        kP = float(mech.get("K_P", 1.0)); Gam = float(mech.get("Gam", mech.get("Gamma", 0.0)))
        Lam = float(mech.get("Lam", mech.get("Lambda", 0.0)))
        return 2.0 * kP * (P - P0) + Gam * P + Lam


@register_operator("cell_chem_from_shape", set="cell", kind="lateral", family="fields",
                   model="apical_area")
class ShapeToChemApicalArea(_ShapeToChemBase):
    """The chemistry listens to APICAL AREA -- crowding and density sensing.

    The most direct reading of "am I stretched or am I crowded", and the cheapest: no mechanical
    targets required, only geometry. Reported relative to the cell's own target area A0 when that
    exists, so a uniformly-scaled tissue reads as unstretched; absolute area otherwise.
    """
    MECHANISM_TAGS = _ShapeToChemBase.MECHANISM_TAGS + ["crowding_sensing", "density_sensing"]

    def _feature(self, pt, m, es, et, ef, nF):
        area, _, _, _ = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(es),
                                         torch.as_tensor(et), torch.as_tensor(ef), nF)
        a = area.numpy()
        if "A0" in m:                                   # strain, not size: a uniformly scaled
            A0 = np.asarray(_np(m["A0"])[:nF], float)      # tissue is not stretched
            return a / np.maximum(A0, 1e-12) - 1.0
        return a


@register_operator("cell_chem_from_shape", set="cell", kind="lateral", family="fields",
                   model="pressure")
class ShapeToChemPressure(_ShapeToChemBase):
    """The chemistry listens to VOLUME-ELASTIC PRESSURE.

    pressure_j = 2 kV (V0_j - v_j): positive when a cell is BELOW its target volume, i.e. squeezed.
    The quantity that would have flagged finding F004's compression phase in real time, had anything
    been reading it.

    NEEDS the mechanical targets V0f.
    """
    MECHANISM_TAGS = _ShapeToChemBase.MECHANISM_TAGS + ["pressure_sensing", "compression_sensing"]

    def _feature(self, pt, m, es, et, ef, nF):
        if "V0f" not in m:
            return None
        _, _, _, vf = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(es),
                                       torch.as_tensor(et), torch.as_tensor(ef), nF)
        V0 = np.asarray(_np(m["V0f"])[:nF], float)
        mech = m.get("mech", {}) or {}
        kV = float(mech.get("K_V", 1.0))
        return 2.0 * kV * (V0 - vf.numpy())


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, "/workspace/Plexus/discovery_okuda/ops")
    from plexus.operators.vertex_ops import build_sphere_mesh
    fails = []

    def chk(c, what, extra=""):
        print(f"  [{'ok ' if c else 'FAIL'}] {what}{('  ' + extra) if extra else ''}")
        if not c:
            fails.append(what)

    print("CERTIFYING the shape features against shapes whose answer is known\n")

    # --- curvature must read ~1/R on a sphere, and must HALVE when the radius doubles
    op = ShapeToChemCurvature({"beta": 0.3})
    for R in (2.5, 5.0, 10.0):
        v, es, et, ef, nF = build_sphere_mesh(500, R, 0.0, 0)
        m = dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=v.shape[0])
        h = op._feature(v, m, es, et, ef, nF)
        print(f"        sphere R={R:<5} mean curvature {np.median(h):7.4f}   1/R = {1.0/R:.4f}")
        chk(abs(float(np.median(h)) - 1.0 / R) < 0.25 / R,
            f"sphere R={R} reads curvature ~1/R", f"{np.median(h):.4f} vs {1.0/R:.4f}")
        if R == 5.0:
            h5 = float(np.median(h))
    v, es, et, ef, nF = build_sphere_mesh(500, 10.0, 0.0, 0)
    m = dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=v.shape[0])
    h10 = float(np.median(op._feature(v, m, es, et, ef, nF)))
    chk(0.35 < h10 / max(h5, 1e-9) < 0.65, "curvature halves when the radius doubles",
        f"ratio {h10/max(h5,1e-9):.3f}")

    # --- curvature must be POSITIVE on a bump and NEGATIVE in a dimple
    v, es, et, ef, nF = build_sphere_mesh(600, 5.0, 0.0, 0)
    m = dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=v.shape[0])
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    cap = u[:, 2] > 0.86
    # A LOCALIZED GAUSSIAN DOME, not a scaled cap. Scaling a spherical cap outward leaves it on a
    # sphere of LARGER radius, i.e. genuinely flatter -- the first version of this test demanded a
    # positive curvature from a shape that is objectively less curved, and the operator was right
    # to disagree with it.
    for tag, amp in (("bump", +1.2), ("dimple", -1.2)):
        g = np.exp(-((u[:, 2] - 1.0) ** 2) / (2 * 0.05 ** 2))
        w = v + amp * g[:, None] * u
        h = op._feature(w, m, es, et, ef, nF)
        _, _, cen, _ = face_geometry_3d(torch.as_tensor(w), torch.as_tensor(es),
                                        torch.as_tensor(et), torch.as_tensor(ef), nF)
        top = cen.numpy()[:, 2] > 0.90 * np.linalg.norm(cen.numpy(), axis=1)
        d = float(np.median(h[top]) - np.median(h[~top]))
        print(f"        {tag:7} curvature at the feature minus elsewhere: {d:+.4f}")
        chk((d > 0) if amp > 0 else (d < 0), f"a {tag} reads the right SIGN")

    # --- standardisation must make beta mean the same thing whatever the units
    for scale in (1.0, 1000.0):
        w = _standardise(np.arange(200.0) * scale, np.ones(200))
        print(f"        feature scaled x{scale:<8g} -> standardised spread {w.std():.4f}")
    a = _standardise(np.arange(200.0), np.ones(200))
    b = _standardise(np.arange(200.0) * 1000.0, np.ones(200))
    chk(np.allclose(a, b, atol=1e-9), "standardisation is invariant to the feature's units")
    chk(np.allclose(_standardise(np.full(50, 7.0), np.ones(50)), 0.0),
        "a UNIFORM feature carries no signal (all zeros, not noise)")

    # --- a single spike must not drive the feed
    x = np.r_[np.random.default_rng(0).normal(0, 1, 199), [1e6]]
    chk(abs(_standardise(x, np.ones(200))).max() <= 4.0 + 1e-9,
        "one extreme cell is clipped, not allowed to set the scale")

    # ----------------------------------------------------------------- everything above is CHECK 0
    # It certifies the ARITHMETIC INSIDE the operator, and every line of it passes. It is also the
    # whole reason `beta` spent 13 rounds and 25 GPU-runs contributing nothing: the operator whose
    # curvature reads 1/R on spheres of known radius, whose bump is positive and whose dimple is
    # negative, never reached the state at all. What used to stand here was
    #
    #     chk(True, "beta = 0 returns zeros by construction (see forward)")
    #
    # -- the single check about `forward`, hardcoded to pass. Below are the three that were missing.
    import torch

    print("\n  CHECK 0b -- the null, actually executed rather than asserted:")
    v, es, et, ef, nF = build_sphere_mesh(500, 5.0, 0.0, 0)
    m = dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=v.shape[0])

    class _Lvl:                                    # the smallest thing `forward` will accept
        def __init__(self, st, sch, mesh=None):
            self.state, self.state_schema, self._mesh, self.occ = st, sch, mesh, None

        def get(self, k):
            a, b = self.state_schema[k]
            return self.state[:, a:b]

    class _H:
        def __init__(self, d):
            self.levels = d

        def level(self, n):
            return self.levels[n]

    chem = torch.zeros(nF, 2)
    chem[:, 0] = 0.3
    chem[:, 1] = 0.6                                # u != 1, or (1 - u) zeroes the emission anyway
    H = _H({"cell": _Lvl(chem, {"chem": (0, 2)}),
            "vertex": _Lvl(torch.as_tensor(v, dtype=torch.float32), {"pos": (0, 3)}, m)})
    out0 = ShapeToChemCurvature({"beta": 0.0}).forward(H)["cell"]
    chk(float(out0.abs().max()) == 0.0, "beta = 0 emits exactly zero (executed)",
        f"max |emission| {float(out0.abs().max()):.3e}")

    print("\n  CHECK 1 -- is the emission set by BETA, or by the clamp?")
    # I expected zero here, on the reasoning that a sphere has uniform curvature so `_standardise`
    # returns zero. The operator disagreed and it was right: a FIBONACCI sphere is discrete, its
    # per-cell curvature carries mesh noise, and the MAD-scaling turns that noise into a full-range
    # standardised field. So the operator fires on a sphere -- driven by discretisation, not shape.
    #
    # The number that matters is not that it fires but WHAT SETS ITS SIZE. dF is clamped into
    # [0 - F0, F_CEIL - F0], so the largest correction the operator can ever emit is
    # (F_CEIL - F0) * (1 - u), independent of beta and of geometry. If the measured maximum equals
    # that bound, beta is not a strength -- it only chooses WHICH cells sit at the ceiling.
    outs = {b: ShapeToChemCurvature({"beta": b}).forward(H)["cell"] for b in (-2.0, -4.0)}
    bound = (F_CEIL - 0.055) * (1.0 - 0.6)
    for b, o in outs.items():
        print(f"        beta={b:<6} max |emission| {float(o.abs().max()):.4e}"
              f"   clamp bound {bound:.4e}")
    chk(all(abs(float(o.abs().max()) - bound) < 1e-6 for o in outs.values()),
        "the emission is PINNED TO THE CLAMP CEILING, so beta sets no magnitude",
        f"{float(outs[-2.0].abs().max()):.4e} vs bound {bound:.4e}")

    print("\n  CHECK 1b -- and on a shape that HAS curvature variation?")
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    g = np.exp(-((u[:, 2] - 1.0) ** 2) / (2 * 0.05 ** 2))
    w = (v + 1.2 * g[:, None] * u).astype(np.float32)
    H.levels["vertex"] = _Lvl(torch.as_tensor(w), {"pos": (0, 3)}, m)
    e2 = {b: float(ShapeToChemCurvature({"beta": b}).forward(H)["cell"].abs().max())
          for b in (-2.0, -4.0)}
    for b, mx in e2.items():
        print(f"        beta={b:<6} max |emission| {mx:.3e}")
    chk(all(x > 0 for x in e2.values()), "a bumped sphere emits a nonzero feed correction")
    # THE SATURATION, measured rather than reasoned about. F = clamp(F0 (1 + beta phihat), 0,
    # F_CEIL) with phihat clipped at +/-4: at beta = -2 the bracket already leaves [0, F_CEIL] on
    # most cells, so DOUBLING beta must not double the emission. If it does, the clamp is not
    # binding and this comment is wrong.
    ratio = e2[-4.0] / max(e2[-2.0], 1e-30)
    print(f"        doubling beta multiplies the emission by {ratio:.3f} (2.000 = unsaturated)")
    chk(ratio < 1.6, "the F clamp SATURATES -- beta is not proportional",
        f"ratio {ratio:.3f}")

    print("\n  CHECK 2+3 -- does each parameter reach the STATE, and does a gradient reach it?")
    ck = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "fixtures", "coral_gate_div_f400.npz")
    if not os.path.exists(ck):
        print(f"        SKIPPED: no fixture at {ck}")
        print(f"        build it with:  python op_probe.py --build-fixture")
    else:
        import yaml
        import op_probe as P
        spec = yaml.safe_load(open("/workspace/Plexus/log/okuda/coral_gate_div/spec_run.yaml"))
        rows = P.selftest(spec, ck, {"cell_chem_from_shape": {"beta": [-2.0, -4.0],
                                                       "F0": [0.0275, 0.11],
                                                       "rate": [0.5, 2.0]}}, frames=50)
        P.report(rows)
        chk(not any(r["verdict"] in ("DEAD", "UNREAD") for r in rows),
            "every cell_chem_from_shape parameter reaches the state on this fixture")

    print("\n  " + ("ALL SHAPE FEATURES CERTIFIED" if not fails else f"{len(fails)} FAILURES"))
    raise SystemExit(1 if fails else 0)


# ==========================================================================================================
# FROM `discovery_okuda/ops/shape_probe_ops.py` -- A measurement, as an operator: per-cell shape descriptors published on the mesh.
# ==========================================================================================================
# `_np` is defined identically in shape_chem_ops.py above; the duplicate from shape_probe_ops.py is dropped.


class _ShapeProbeBase(Lateral):
    """Compute one scalar per cell and publish it on the mesh under `field`. No state is touched."""
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False
    INPUTS = ["cell", "vertex"]; OUTPUTS = []; READS = ["pos"]; WRITES = []
    MAPS = ["E_srce", "E_trgt", "E_face"]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["measurement", "cell_shape", "publishes_field"]
    REFERENCE = ("Bi, D. et al. (2015). Nat. Phys. 11:1074-1079 (the shape index as the tissue's "
                 "own order parameter, rigid below 3.81 and fluid above).")
    PARAM_ROLES = {"field": "published_field_name"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell"); self.vat = params.get("vertex_set", "vertex")
        # THE NAME IS THE WIRING. Whatever this is called is what a Die operator asks for.
        self.field = str(params.get("field", "elong"))

    def _measure(self, pos, m, es, et, ef, nF):
        raise NotImplementedError

    def forward(self, H, mask=None):
        vlvl = H.level(self.vat)
        m = getattr(vlvl, "_mesh", None)
        if m is None:
            return {}
        nF = int(m["nF"])
        es = _np(m["E_srce"]); et = _np(m["E_trgt"]); ef = _np(m["E_face"])
        live = ef < nF
        pos = _np(vlvl.get("pos"))[:int(m["Nv"])].astype(np.float64)
        val = self._measure(pos, m, es[live], et[live], ef[live], nF)
        if val is None:
            # A PRECONDITION IS ABSENT, so nothing is published -- rather than publishing zeros,
            # which a Die reading `field_high` would score as "no cell is elongated" and a Die
            # reading `field_low` would score as "every cell is". An absent field is undefined;
            # zero is a measurement. This substrate has paid for that distinction twice.
            m.pop(self.field, None)
            return {}
        v = np.asarray(val, float)
        v[~np.isfinite(v)] = np.nan          # a degenerate cell is UNMEASURED, not zero
        m[self.field] = v
        return {}


@register_operator("cell_shape_probe", set="cell", kind="lateral", family="hierarchy",
                   model="shape_index")
class ShapeIndexProbe(_ShapeProbeBase):
    """P / sqrt(A) per cell -- the quantity the vertex model itself minimises towards `p0`."""

    def _measure(self, pos, m, es, et, ef, nF):
        pt = torch.as_tensor(pos)
        area, perim, _cen, _vf = face_geometry_3d(
            pt, torch.as_tensor(es), torch.as_tensor(et), torch.as_tensor(ef), nF)
        a = _np(area)[:nF]; p = _np(perim)[:nF]
        out = np.full(nF, np.nan)
        ok = a > 1e-12
        out[ok] = p[ok] / np.sqrt(a[ok])
        return out


@register_operator("cell_shape_probe", set="cell", kind="lateral", family="hierarchy",
                   model="aspect")
class AspectProbe(_ShapeProbeBase):
    """Longest over shortest axis of the cell ring -- "thin and elongated" as a number.

    THE EIGENVALUES ARE OF THE RING'S COVARIANCE IN 3D and the ratio is taken between the FIRST TWO,
    not the first and the last. A cell on a curved shell is a nearly-flat patch, so its third
    eigenvalue is the sheet's thickness and is small for every cell, elongated or not; using it
    would report the whole tissue as extreme and rank nothing.
    """

    def _measure(self, pos, m, es, et, ef, nF):
        rings = rings_from_flat_3d(es, et, ef, nF)
        out = np.full(nF, np.nan)
        for f, r in enumerate(rings):
            if r is None or len(r) < 3:
                continue
            p = pos[np.asarray(r, int)]
            c = p.mean(0)
            w = np.linalg.eigvalsh(np.cov((p - c).T) + 1e-15 * np.eye(3))[::-1]
            s0, s1 = np.sqrt(max(w[0], 0.0)), np.sqrt(max(w[1], 0.0))
            if s1 > 1e-9:
                out[f] = s0 / s1
        return out
