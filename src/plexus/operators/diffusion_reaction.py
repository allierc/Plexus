"""Reaction-diffusion ON THE CELL GRAPH, and the two couplings between chemistry and shape.

The diffusion is not on a grid. The CELLS are the nodes, `cell_neighbours` is the graph, and the
Laplacian runs over shared faces -- so the domain grows and rewires as the tissue divides, which a
fixed lattice cannot do. That is why these are `set=cell` operators rather than `field` ones.

In the order they appear below:

    cell_geometry         aggregate  vertex mesh -> per-cell centroid, area, perimeter, volume
    cell_neighbours       rewire     the cell adjacency the Laplacian runs on
    seed_cell_chem        seed       the initial morphogen field
    cell_chem_diffuse     lateral    morphogen exchange between neighbouring cells
    cell_chem_react       lateral    the local reaction: the pattern-forming nonlinearity
    cell_grow             lateral    morphogen -> growth: the chemistry-to-shape coupling
    interface_tension     lateral    a purse-string line tension on the activator interface
    interface_push        lateral    the term that is NOT physics, kept separate on purpose
    cell_chem_from_shape  lateral    shape -> chemistry: the other half of the loop
    cell_shape_probe      lateral    one shape scalar per cell, published for a discriminator

then the models -- different hypotheses in one slot, not different arithmetic:

    seed_cell_chem       scatter (default), noise, patch, cones, simplex, uniform
    cell_chem_diffuse    graph_laplacian (default implementation), interface_weighted (a model)
    cell_chem_react      gray_scott, brusselator, gierer_meinhardt, rock_paper_scissor,
                         gray_scott_coupled, source_decay (a
                         sustained source), balaskas (a gene circuit reading a signal)
    cell_grow            default, sizer, balance, timer, stretch
    cell_chem_from_shape apical_area, curvature, pressure, tension
    cell_shape_probe     shape_index, aspect

`interface_tension` and `interface_push` are two operators and must stay two. Written as one, they
carried a purse-string line tension -- ordinary vertex-model physics -- MINUS an energy that falls
as activator-high cells move outward. The second term does not model a force: it pays the tissue to
produce the morphology a search is looking for, so a run carrying it can only ever be a control.
One name over both makes that impossible to see in a schedule.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch
from plexus.models.base import Aggregate, Lateral, Rewire, Structural
from plexus.models.registry import register_operator
from plexus.models.base import Lateral
from plexus.operators.vertex_ops import face_geometry_3d, resolve_cell_set, wedge_apex


def _chan(params, who, n_species=2):
    """The FIRST COLUMN of the contiguous species span this instance owns.

    `chan` is a COLUMN index, not a species index, though it reads like one. A reaction model occupies
    `n_species` ADJACENT columns of `chem` -- Gray-Scott two (a, u), May-Leonard three (u, v, w),
    a coupled pair of Gray-Scott systems four -- so the second two-species system starts at
    `chan: 2`, not `chan: 1`. `chan: 1` is the natural thing to write and would have put the second
    system's activator ON THE FIRST SYSTEM'S SUBSTRATE: both would run, both would look alive, and
    they would be driving one shared column through a coupling nobody wrote.

    AN EVEN-ONLY RULE ON `chan` IS WRONG the moment a three-species model exists: May-Leonard
    tiles at 0, 3, 6, and an even-only guard rejects a correct spec while accepting `chan: 2` for a
    three-species system, which overlaps. The rule is "a multiple of this model's own width", which
    is the actual tiling, and it reduces to the even rule for two-species models.

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


def cap_target_rate(ds, s_prev, V0f, V0f_init, cap_v, dt, dims=3):
    """The growth rate on the linear scale `ds`, gated so the TARGET `V0f` stops at `cap_v`.

    The target's own rate is dV0f/dt = dims V0f_init s^(dims-1) ds (the chain rule `cell_grow` emits), so the
    largest ds that keeps V0f <= cap_v this step is (cap_v - V0f) / (dt dims V0f_init s^(dims-1)), and never below
    zero: a gate, not a pull -- a cell already over the cap holds, it is not shrunk."""
    dvds = (dims * V0f_init * s_prev ** (dims - 1)).clamp(min=1e-12)
    room = (cap_v - V0f).clamp(min=0.0) / max(float(dt), 1e-12) / dvds
    return torch.minimum(ds, room)


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


@register_operator("cell_geometry", set="cell", kind="aggregate", family="hierarchy", title="Cell geometry from the mesh",
                   equation=r"""$$\mathbf{cen}_f=\frac{1}{n_f}\sum_{e\in f}\mathbf x_{\mathrm{srce}(e)},\qquad A_f=\tfrac12\left\lVert\sum_{e\in f}\mathbf x_{\mathrm{srce}(e)}\times\mathbf x_{\mathrm{trgt}(e)}\right\rVert$$""")
class CellGeometry3D(Aggregate):
    """Aggregate the 3D vertex mesh into per-cell scalars: the cross-scale readout the
    reaction-diffusion runs on.

    vertex -> cell: reads the half-edge table stashed on the vertex set and writes the cell set's
    centroid `centroid` and area, by scatter-add over half-edges.

        cen_f = (1/n_f) sum_{e in f} x_srce(e)          the face centroid
        A_f   = (1/2) | sum_{e in f} x_srce(e) x x_trgt(e) |

    the second being the magnitude of the summed cross products around the face ring, which is
    twice the area of a planar polygon in 3D and needs no projection onto a normal.

    It is an aggregate because it is a many-to-one map across the containment relation: many
    vertices, one cell. Every operator here that speaks of a cell's position or size depends on it,
    so it is scheduled first.

    Reference: none -- a geometric readout, not a mechanism. Plexus (this work).
    """
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    INPUTS = ["vertex"]; OUTPUTS = ["cell"]; READS = ["pos"]; WRITES = ["area", "centroid", "volume"]
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
        area, _, centroid, vol = face_geometry_3d(pos, m["E_srce"], m["E_trgt"], m["E_face"], m["nF"])
        nF = m["nF"]; st = clvl.state.clone(); sch = clvl.state_schema
        # `volume` -- THE CELL'S SIZE AS THE SIZE RULES READ IT, written only when the set declares
        # the block, so every existing spec is unchanged. The convention is `cell_size`'s: the
        # polyhedron between the apical and basal caps whenever the vertices carry a separation
        # `sep`, the origin-referenced wedge otherwise. Computed here rather than through
        # `cell_size`, which caches the run's reference volume as a side effect. Recording it is
        # what lets a ruler read a run's cell sizes instead of rebuilding every cell's polyhedron at
        # every frame -- minutes per run on CPU (exp 3, 2026-09-25).
        if "volume" in sch:
            if "sep" in getattr(vlvl, "state_schema", {}):
                from plexus.operators.vertex_ops import apicobasal_geometry_3d
                vol = apicobasal_geometry_3d(pos, vlvl.get("sep")[:m["Nv"]], m["E_srce"], m["E_trgt"],
                                             m["E_face"], nF)[0]
            i0, i1 = sch["volume"]; st[:nF, i0:i1] = vol.detach().to(st.dtype)[:, None]
        if "centroid" in sch:
            i0, i1 = sch["centroid"]; st[:nF, i0:i1] = centroid.detach()
        if "area" in sch:
            i0, i1 = sch["area"]; st[:nF, i0:i1] = area.detach()[:, None]
        clvl.state = st
        if getattr(clvl, "occ", None) is not None:
            occ = torch.zeros(clvl.state.shape[0], device=clvl.state.device); occ[:nF] = 1.0; clvl.occ = occ
        return {}


@register_operator("cell_neighbours", set="cell", kind="rewire", family="topology", title="Who neighbours whom",
                   equation=r"""$$E=\big\{(f,g)\ :\ \text{some mesh edge is shared by faces } f \text{ and } g\big\}$$""")
class CellAdjacency(Rewire):
    """The relation the reaction-diffusion runs on: two cells are neighbours if and only if
    they share a mesh edge.

    vertex -> cell: reads the half-edge table, writes the cell set's `edge_index`.

        E = { (f, g) : some mesh edge is shared by face f and face g }

    Rebuilt on every call, so it follows a T1 or a division the moment one happens -- which is the
    whole reason the chemistry lives on this graph rather than on a fixed lattice. A grid cannot
    grow or rewire; a tissue does both.

    Reference: none -- the adjacency of a mesh, not a mechanism. Plexus (this work).
    """
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


@register_operator("cell_neighbours", set="cell", kind="rewire", family="topology", model="point_contact",
                   title="Who neighbours whom",
                   equation=r"""$$E=\big\{(c,d)\ :\ \exists\, i\in c,\ j\in d,\ |\mathbf x_i-\mathbf x_j|<r_c\big\},\qquad w_{cd}=\#\{(i,j)\}$$""")
class CellAdjacencyPointContact(Rewire):
    """`point_contact` MODEL of cell_neighbours -- the cells are BODIES OF MATERIAL POINTS (MPM cells), so two cells
    touch when some point of one lies within `contact` of some point of the other.

    particle -[containment]-> cell: reads the point set's positions and its parent map (`points`), writes the cell
    set's `edge_index` (both directions) and `edge_weight` (the number of touching point pairs, a contact AREA in
    point pairs, for `cell_chem_diffuse[model: contact_weighted]`), every `every` calls (default 1: the cells move).

        E    = { (c, d) : some point i of c and j of d are closer than r_c }
        w_cd = the number of such (i, j)

    r_c is `contact` (world) -- the reach of the contact law the cells meet through (`pair_potential` sigma, plus its
    adhesive shoulder). WHY A MODEL: the contract is `cell_neighbours`' own (-> cell `edge_index`); what differs is
    where the contact comes from: a body of points, not a half-edge mesh or a label image. A gap junction needs two
    membranes touching, which is exactly a contact between two cells' material.

    Reference: none -- the adjacency of touching bodies, not a mechanism. Plexus (this work).
    """
    SUPPORTED_DIMS = [2, 3]; DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["points", "contact"]
    MECHANISM_TAGS = ["cell_neighbours", "neighbour_graph", "contact"]
    PARAM_ROLES = {"points": "the_material_point_set", "contact": "touch_distance_world", "every": "calls_between_rebuilds"}
    PARAM_UNITS = {"contact": "length"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.points = str(params["points"])
        self.contact = float(params["contact"])
        self.every = max(1, int(params.get("every", 1)))
        self._n = 0

    def forward(self, H, mask=None):
        self._n += 1
        clvl = H.level(self.at)
        if (self._n - 1) % self.every and getattr(clvl, "edge_index", None) is not None:
            return {}
        P = H.level(self.points)
        X = P.get("pos").detach()
        par = H.lift_index(self.points, self.at).long()
        live = (P.occ > 0.5) & torch.isfinite(X).all(1)
        X, par = X[live], par[live]
        dev = clvl.state.device
        from scipy.spatial import cKDTree
        pr = cKDTree(X.cpu().double().numpy()).query_pairs(self.contact, output_type="ndarray")
        if len(pr) == 0:
            clvl.edge_index = torch.zeros(2, 0, dtype=torch.long, device=dev)
            clvl.edge_weight = torch.zeros(0, device=dev)
            return {}
        pc = par.cpu().numpy()
        a, b = pc[pr[:, 0]], pc[pr[:, 1]]
        m = a != b
        e = np.sort(np.stack([a[m], b[m]], 1), axis=1)
        if e.shape[0] == 0:
            clvl.edge_index = torch.zeros(2, 0, dtype=torch.long, device=dev)
            clvl.edge_weight = torch.zeros(0, device=dev)
            return {}
        u, c = np.unique(e, axis=0, return_counts=True)
        ei = np.concatenate([u.T, u.T[::-1]], axis=1)
        clvl.edge_index = torch.as_tensor(ei, dtype=torch.long, device=dev)
        clvl.edge_weight = torch.as_tensor(np.concatenate([c, c]).astype(np.float32), device=dev)
        return {}


@register_operator("cell_neighbours", set="cell", kind="rewire", family="topology", model="label_image",
                   title="Who neighbours whom",
                   equation=r"""$$E=\big\{(i,j)\ :\ \text{a pixel of label } i+1 \text{ shares an edge with a pixel of label } j+1\big\}$$""")
class CellAdjacencyLabelImage(Rewire):
    """`label_image` MODEL of cell_neighbours -- the cells are a MEASURED segmentation, so who touches
    whom is read from the segmentation itself, not from a mesh the sheet does not have.

    label_image field -> cell: reads the integer label map named by `from:`, writes the cell set's
    `edge_index`, once (the labels do not move).

        E = { (i, j) : some pixel of label i+1 shares a pixel edge with a pixel of label j+1 }

    Cell i is label i+1, the convention `seed_from_segmentation` seeds by. Label 0 is background
    and touches nobody. A pixel edge is enough: in the Utrecht segmentations the cells tile the
    tissue with no gap (1,314 touching pairs over 472 cells on the healthy sheet, median 6
    neighbours, none isolated), so the graph is the junctional contact graph of the real sheet --
    what a gap junction needs -- and not a Delaunay or radius guess.

    Why a model and not a new operator: the contract is `cell_neighbours`' own (cell -> cell,
    writes `edge_index`); what differs is where the contact comes from, a measured image instead
    of a half-edge mesh. The default is untouched. It also writes `edge_weight`, each junction's
    contact length in pixels, which only `cell_chem_diffuse[model: contact_weighted]` reads.

    Reference: none -- the adjacency of a measured segmentation, not a mechanism. Plexus (this work).
    """
    SUPPORTED_DIMS = [2]; DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["from"]
    MECHANISM_TAGS = ["cell_neighbours", "neighbour_graph", "instance_segmentation"]
    PARAM_ROLES = {"from": "label_field"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell"); self.field_name = params["from"]
        self._ei = None

    @staticmethod
    def pairs(grid, counts=False):
        """[m, 2] unique label pairs (a < b) that share a pixel edge; background 0 excluded. With
        `counts`, also [m] the number of pixel edges each pair shares -- its contact length in pixels."""
        out = []
        for a, b in ((grid[:-1, :], grid[1:, :]), (grid[:, :-1], grid[:, 1:])):
            m = (a > 0) & (b > 0) & (a != b)
            out.append(torch.stack([torch.minimum(a[m], b[m]), torch.maximum(a[m], b[m])], 1))
        u, c = torch.unique(torch.cat(out, 0), dim=0, return_counts=True)
        return (u, c) if counts else u

    def forward(self, H, mask=None):
        clvl = H.level(self.at)
        if self._ei is None:
            grid = H.fields[self.field_name].grid[0]
            p, c = self.pairs(grid, counts=True)
            p = p - 1                                                    # label -> cell index
            keep = (p < clvl.n).all(1)
            p, c = p[keep], c[keep]
            e = p.T.to(torch.long)
            self._ei = torch.cat([e, e.flip(0)], 1).to(clvl.state.device)
            # the contact length of each junction, in pixels, aligned with edge_index (both directions);
            # read by `cell_chem_diffuse[model: contact_weighted]`, ignored by every other reader
            self._ew = torch.cat([c, c]).to(clvl.state.device, torch.get_default_dtype())
        clvl.edge_index = self._ei
        clvl.edge_weight = self._ew
        return {}


# CANONICAL `seed_cell_chem`, ALIAS `cell_chem_seed` -- see `mesh_ops.SeedMesh3D` for why both
# spellings must resolve: 320 specs use the first and the rest use the second.
@register_operator("seed_cell_chem", "cell_chem_seed", set="cell", kind="seed", family="seed", title="Initial morphogen",
                   species=(("a", "activator"), ("u", "substrate")),
                   equation=r"""$$u_j=1\ \ \text{everywhere},\qquad a_j=0\ \ \text{but on the seeded patch}$$""")
class CellRDSeed(Structural):
    """The initial morphogen field on the cell set, written once at the opening of the trajectory.
    The default `scatter` model is the Gray-Scott initial condition.

    cell -> cell: writes the `chem` block, once.

        u_j = 1 everywhere,   a_j = 0 everywhere
        except a central spot, where (a, u) = (0.5, 0.25)

    a is the activator and u the substrate, both dimensionless concentrations; chem = [a, u]. The
    substrate starts full and the activator absent, so nothing happens anywhere except at the spot
    -- which is what makes the pattern's origin a declared initial condition rather than an
    accident of the numerics.

    A SEED, NOT A BOUNDARY CONDITION, and the kind enforces it: the runtime confines every seed to
    the opening frames. A rule re-applied every frame is a moving boundary condition, and it makes
    the answer to "does the chemical pattern grip the shape?" a property of the rule rather than of
    the simulation. It also overwrites both chemistry channels every tick, so no other operator
    writing to `chem` -- `cell_chem_from_shape`, for one -- can accumulate anything at all, and its
    parameters become inert while still appearing to be under test.

    An unrecognised model raises rather than falling through to the default. A specification that
    can no longer be run is a correct outcome; one that quietly runs something else is not.

    Reference: Plexus (this work); cone seeding after Okuda, S. et al. (2018). Sci. Rep. 8:2386.
    """

    N_SPECIES = 2
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
        # How a pattern nucleates is on the `model:` axis, not a `mode:` setting. Five ways of
        # writing x_0 are five claims about HOW PATTERNING NUCLEATES -- from random fluctuation,
        # from a placed patch, from N fixed foci -- and seeding it in cones puts part of the answer
        # in by hand. That is a hypothesis, and it belongs where the registry can see it.
        #
        # It also matters because the variants READ DIFFERENT THINGS: `scatter` and `noise` use no
        # geometry at all, while `patch` and `cones` read the cell centroids. On the `model:` axis
        # each carries its own typed signature, so "may this seed be scheduled ahead of
        # `cell_geometry`?" is answerable from the registry rather than from a hard-coded list.
        if "mode" in params:
            raise ValueError(
                "seed_cell_chem: `mode` is gone -- write `model:`. How a pattern nucleates is a "
                "hypothesis, not a setting. "
                + ("`tip` is gone: it re-seeded every frame, which makes it a moving boundary "
                   "condition and annihilates every operator that writes to `chem`. Use "
                   "`model: scatter`." if params.get("mode") == "tip" else ""))
        self.mode = getattr(type(self), "NUCLEATION", "scatter")
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
        # `noise` use no geometry whatever, and `patch`/`cones` already guard on `"centroid" in
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
            if "centroid" in clvl.state_schema:
                ci0, ci1 = clvl.state_schema["centroid"]; zc = clvl.state[:nF, ci0 + 2]
                a = torch.where(zc > self.patch_z * float(zc.max()), torch.ones(nF, device=dev), a)
            u = torch.ones(nF, device=dev)
        elif self.mode == "cones":                              # N FIXED radial activation cones (Fig 5 multi-tube):
            a = torch.full((nF,), 0.02, device=dev)             # each cone's tip stays activated as it extends ->
            if "centroid" in clvl.state_schema:                      # N radial tubes. Re-seeded every frame (tracks tips).
                ci0, ci1 = clvl.state_schema["centroid"]; centroid = clvl.state[:nF, ci0:ci0 + 3]
                d = centroid / (centroid.norm(dim=1, keepdim=True) + 1e-9)
                dirs = torch.as_tensor(self._cone_dirs(), dtype=centroid.dtype, device=dev)
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


@register_operator("seed_cell_chem", "cell_chem_seed", set="cell", kind="seed", family="seed",
                   model="noise", title="Initial morphogen")
class CellRDSeedNoise(CellRDSeed):
    """`noise` MODEL of seed_cell_chem -- the homogeneous steady state plus NOISE -- patterning from fluctuation alone, the strictest test that the pattern is emergent.

    Geometry: none. That is not a footnote -- a seed reading no geometry may be scheduled ahead of
    `cell_geometry`, and one reading the cell centroids may not.
    """
    NUCLEATION = "noise"


@register_operator("seed_cell_chem", "cell_chem_seed", set="cell", kind="seed", family="seed",
                   model="patch", title="Initial morphogen")
class CellRDSeedPatch(CellRDSeed):
    """`patch` MODEL of seed_cell_chem -- a LOCALIZED activation source, placed by hand -- a bud/tube driver.

    Geometry: reads the cell centroids. That is not a footnote -- a seed reading no geometry may be
    scheduled ahead of `cell_geometry`, and one reading the centroids may not.
    """
    NUCLEATION = "patch"


@register_operator("seed_cell_chem", "cell_chem_seed", set="cell", kind="seed", family="seed",
                   model="cones", title="Initial morphogen")
class CellRDSeedCones(CellRDSeed):
    """`cones` MODEL of seed_cell_chem -- N FIXED radial activation cones (Okuda Fig 5's multi-tube) -- the strongest hand in the answer, and the honest place to declare it.

    Geometry: reads the cell centroids. That is not a footnote -- a seed reading no geometry may be
    scheduled ahead of `cell_geometry`, and one reading the centroids may not.
    """
    NUCLEATION = "cones"


@register_operator("seed_cell_chem", "cell_chem_seed", set="cell", kind="seed", family="seed",
                   model="simplex", title="Initial morphogen")
class CellRDSeedSimplex(CellRDSeed):
    """`simplex` MODEL of seed_cell_chem -- three species normalised to a simplex -- the May-Leonard initial condition for cyclic competition.

    Geometry: none. That is not a footnote -- a seed reading no geometry may be scheduled ahead of
    `cell_geometry`, and one reading the cell centroids may not.
    """
    NUCLEATION = "simplex"


@register_operator("seed_cell_chem", "cell_chem_seed", set="cell", kind="seed", family="seed",
                   model="uniform", title="Initial state of a regulatory circuit")
class CellRDSeedUniform(CellRDSeed):
    """`uniform` MODEL of seed_cell_chem -- every cell the same, `values: [v0, v1, ...]` written into the
    columns from `chan` on. The initial state of a gene-regulatory circuit, which starts from its
    no-signal state everywhere (Balaskas et al. 2012, Table S2: Pax6 P = 3, Olig2 O = Nkx2.2 N = 0),
    not the nucleus of a pattern -- so nothing here is random and `seed` is unused.

    WHY A SEED AND NOT LETTING THE CIRCUIT START AT ZERO. From P = 0 a cell that meets a high signal
    before Pax6 has risen skips the Pax6 repression of Nkx2.2 altogether; the paper's temporal
    sequence (Pax6, then Olig2, then Nkx2.2, Fig. 4C) starts from P = 3, and a circuit with hysteresis
    remembers where it started.

    Geometry: none.
    """
    NUCLEATION = "uniform"

    def __init__(self, params, device="cpu"):
        vals = params.get("values")
        if not vals:
            raise ValueError("seed_cell_chem[uniform]: needs `values: [..]`, one per column from `chan`")
        params = dict(params, n_species=len(vals))
        super().__init__(params, device)
        self.values = [float(v) for v in vals]

    def forward(self, H, mask=None):
        """Its own body, so the base class's is untouched: the cell count as `CellRDSeed` reads it
        (the mesh's nF, else the set's live count), then the declared values into the columns."""
        clvl = H.level(self.at)
        if "chem" not in clvl.state_schema:
            return {}
        vlvl = H.levels[self.vat] if self.vat in H.levels else None
        m = getattr(vlvl, "_mesh", None) if vlvl is not None else None
        if m is not None:
            nF = int(m["nF"])
        else:
            occ = getattr(clvl, "occ", None)
            nF = int(occ.sum().item()) if occ is not None else int(clvl.state.shape[0])
        h0, h1 = clvl.state_schema["chem"]
        _span(clvl.state[:, h0:h1], self.chan, len(self.values), type(self).__name__)
        st = clvl.state.clone()
        for _k, _v in enumerate(self.values):
            st[:nF, h0 + self.chan + _k] = float(_v)
        clvl.state = st
        return {}


@register_operator("seed_cell_chem", "cell_chem_seed", set="cell", kind="seed", family="seed",
                   model="lognormal", title="Cell-to-cell variability around a common state")
class CellRDSeedLognormal(CellRDSeedUniform):
    """`lognormal` MODEL of seed_cell_chem -- `uniform`'s declared state with CELL-TO-CELL VARIABILITY:
    column k of cell j is values[k] exp(s_k z_jk - s_k^2 / 2), z_jk standard normal from `seed`, s_k =
    sqrt(ln(1 + cv[k]^2)), so each column keeps its declared MEAN and has coefficient of variation
    cv[k] (standard deviation over mean). cv 0 is `uniform` exactly -- the no-variability control.

    Log-normal because the quantity is a concentration: positive, and its spread multiplicative
    (a cell with twice the YAP is as far from the mean as one with half). Serra et al. 2019 measure the
    spread this seeds -- nuclear YAP varies from cell to cell before the first Paneth cell appears
    (ED Fig. 7a, Fig. 5f) -- without naming a distribution; the log-normal is the model's choice.

    Reference: Serra, D. et al. (2019). Nature 569:66-72 (the variability); Plexus (this work).
    """
    NUCLEATION = "lognormal"

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        cv = params.get("cv", 0.0)
        cv = [float(cv)] * len(self.values) if not isinstance(cv, (list, tuple)) else [float(c) for c in cv]
        if len(cv) != len(self.values):
            raise ValueError(f"seed_cell_chem[lognormal]: `cv` has {len(cv)} entries, `values` {len(self.values)}")
        self.cv = cv

    def forward(self, H, mask=None):
        super().forward(H, mask)
        clvl = H.level(self.at)
        if "chem" not in clvl.state_schema or not any(self.cv):
            return {}
        vlvl = H.levels[self.vat] if self.vat in H.levels else None
        m = getattr(vlvl, "_mesh", None) if vlvl is not None else None
        if m is not None:
            nF = int(m["nF"])
        else:
            occ = getattr(clvl, "occ", None)
            nF = int(occ.sum().item()) if occ is not None else int(clvl.state.shape[0])
        g = torch.Generator(device="cpu"); g.manual_seed(self.seed)
        h0, _ = clvl.state_schema["chem"]
        st = clvl.state.clone()
        for _k, _c in enumerate(self.cv):
            if _c <= 0:
                continue
            sg = float(np.sqrt(np.log1p(_c * _c)))
            z = torch.randn(nF, generator=g, dtype=torch.float64)
            fac = torch.exp(sg * z - 0.5 * sg * sg).to(device=st.device, dtype=st.dtype)
            st[:nF, h0 + self.chan + _k] = st[:nF, h0 + self.chan + _k] * fac
        clvl.state = st
        return {}


@register_operator("seed_cell_chem", "cell_chem_seed", set="cell", kind="seed", family="seed",
                   model="lattice_sites", title="Individuals on lattice sites, at random")
class CellRDSeedLatticeSites(CellRDSeed):
    """`lattice_sites` MODEL of seed_cell_chem -- every cell is a lattice SITE, and each is given, at
    random and independently, one individual of species k with probability `fractions[k]`, or left
    empty with the remaining probability. An individual is a one-hot row of `chem` (columns from
    `chan`); an empty site is a row of zeros.

        P(site holds species k) = f_k,     P(empty) = 1 - sum_k f_k

    The initial condition of both individual-based rock-paper-scissors lattices: Reichenbach et al.
    2007 start "well mixed, equal density of individuals of each species and of empty sites"
    (f = 1/4 each, Suppl. Movies 1-2); Kerr et al. 2002 assign "one of the following states:
    occupation by a C, S or R cell or the empty state" at random (Box 1).

    WHY A MODEL OF `seed_cell_chem`: it writes the initial `chem` of a cell set, the contract's one
    job; what differs from `simplex` (the densities' own initial condition) is the biology of what a
    column means -- an individual that is there or not, not a concentration.

    Reference: Reichenbach, T., Mobilia, M. & Frey, E. (2007). Nature 448:1046; Kerr, B. et al. (2002).
    Nature 418:171, Box 1.
    """
    NUCLEATION = "lattice_sites"

    def __init__(self, params, device="cpu"):
        fr = params.get("fractions")
        if not fr:
            raise ValueError("seed_cell_chem[lattice_sites]: needs `fractions: [..]`, one per species")
        if sum(float(v) for v in fr) > 1.0 + 1e-9:
            raise ValueError(f"seed_cell_chem[lattice_sites]: fractions {fr} sum above 1")
        params = dict(params, n_species=len(fr))
        super().__init__(params, device)
        self.fractions = [float(v) for v in fr]

    def forward(self, H, mask=None):
        clvl = H.level(self.at)
        if "chem" not in clvl.state_schema:
            return {}
        h0, h1 = clvl.state_schema["chem"]
        _span(clvl.state[:, h0:h1], self.chan, len(self.fractions), type(self).__name__)
        n = clvl.state.shape[0]
        g = torch.Generator(device="cpu"); g.manual_seed(self.seed)
        u = torch.rand(n, generator=g, dtype=torch.float64)
        edges = torch.cumsum(torch.tensor([0.0] + self.fractions, dtype=torch.float64), 0)
        st = clvl.state.clone()
        for k in range(len(self.fractions)):
            st[:, h0 + self.chan + k] = ((u >= edges[k]) & (u < edges[k + 1])).to(st.dtype).to(st.device)
        clvl.state = st
        return {}


@register_operator("cell_chem_diffuse", set="cell", kind="lateral", family="fields", implementation="graph_laplacian", title="Morphogen diffusion between neighbouring cells",
                   equation=r"""$$\frac{dc_i}{dt}=D_s\sum_{j\sim i}\frac{c_j-c_i}{\deg(i)}$$""")
class CellDiffuse(Lateral):
    """Morphogen exchange between neighbouring cells, as a purely combinatorial graph diffusion:
    every neighbour counts the same, whatever the geometry between them.

    cell -[cell adjacency]-> cell: reads chem and edge_index, emits d(chem)/dt.

        dc_i/dt = D_s sum_{j ~ i} (c_j - c_i) / deg(i)        norm: true  (the default)
        dc_i/dt = D_s sum_{j ~ i} (c_j - c_i)                 norm: false

    c is the concentration of one species, dimensionless, and the sum runs over the cells sharing a
    mesh edge with i. D_s is that species' diffusion coefficient, in inverse time, since the graph
    carries no length. `chi` sets the RATIO between the two species' coefficients, which is the
    Turing condition -- a pattern needs the inhibitor to spread faster than the activator, and chi
    is where that claim lives. Dividing by the degree gives the normalised Laplacian, whose
    eigenvalues lie in [-2, 0], so an explicit step stays stable at any cell degree; without it, a
    cell with many neighbours can overshoot.

    THE GEOMETRY IS INVISIBLE HERE, which is why the `interface_weighted` sibling exists. This
    reads only `chem` and `edge_index`: two cells sharing a thin sliver exchange exactly as much as
    two sharing a broad face, and a cell stretched to twice its volume dilutes as though it had not
    stretched. The pattern therefore rides on the tissue like a decal. That is the right numerics
    for a pure Turing-on-a-graph study, and it stays the contract default; `interface_weighted` is
    the model to select where shape must feed back into chemistry.

    Reference: Fick, A. (1855). Ueber Diffusion. Ann. Phys. 170:59-86; Turing, A. M. (1952). The
    chemical basis of morphogenesis. Phil. Trans. R. Soc. B 237:37-72.
    """

    N_SPECIES = 2
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["chi"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]
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


# ============================================================================================
#  individuals on lattice sites: the shared machinery of the lattice variants
# ============================================================================================
# A SITE is a cell of the set; it holds one individual (a one-hot row of `chem` over the species'
# columns) or none (a row of zeros). The lattice variants below (`cell_chem_diffuse[lattice_exchange]`,
# `cell_chem_react[rps_lattice]`, `cell_chem_react[kerr_csr]`) each draw discrete events for one tick.
#
# HOW THEY COMBINE WITH THE REST OF THE SCHEDULE -- read before writing another. The engine SUMS the
# deltas of every operator writing `chem` and integrates once per tick (`Hierarchy.add_delta`,
# `engine._integrate`), so two lattice operators that each flipped sites of the SAME state could
# both empty one site, or swap an individual another had just killed, and the sum would no longer be
# one-hot. Each variant therefore reads the PROVISIONAL state -- the state plus the `chem` delta the
# operators before it added this tick -- draws its events on it, and returns exactly the increment
# that takes the provisional state to its result. The sum is then the operators applied one after
# the other in schedule order (a first-order operator splitting of the paper's Gillespie scheme, exact
# as dt -> 0), and every site stays empty or one individual.
def _lattice_provisional(H, at, chem):
    # WHERE THE ENGINE KEEPS THE `chem` DELTA DEPENDS ON THE SET: when `chem` is the set's coordinate
    # block (a lattice of sites whose `pos` is not integrated) it accumulates in `H._delta[at]`, else in
    # `H._delta_blocks[at]["chem"]` (`Hierarchy.add_delta`). Reading only the second made two lattice
    # operators blind to each other on exactly the sets they run on -- sites holding two individuals.
    co = H.level(at).state_schema.coordinate
    if co is not None and co.name == "chem":
        d = (getattr(H, "_delta", {}) or {}).get(at)
    else:
        d = (getattr(H, "_delta_blocks", {}) or {}).get(at, {}).get("chem")
    return chem if d is None else chem + float(getattr(H, "dt", 1.0)) * d


def _lattice_sites(x, chan, ns):
    """(occupied [N] bool, species [N] long) of the provisional state's species columns."""
    cols = x[:, chan:chan + ns]
    return cols.max(1).values > 0.5, cols.argmax(1)


def _lattice_onehot(occ, s, ns):
    t = torch.zeros(occ.shape[0], ns, device=occ.device, dtype=torch.float32)
    t[occ, s[occ]] = 1.0
    return t


def _lattice_neighbour(op, lvl, gen):
    """One uniformly random neighbour per site, from the set's relation (`edge_index`, receiver first).
    The CSR table is rebuilt only when the relation object changes."""
    ei = getattr(lvl, "edge_index", None)
    if ei is None or ei.numel() == 0:
        raise ValueError(f"{type(op).__name__}: the set has no neighbour relation -- schedule a "
                         f"`radius_graph` (e.g. `model: periodic_tiles`) before it")
    if getattr(op, "_csr_of", None) is not ei:
        n = lvl.state.shape[0]
        o = torch.argsort(ei[0], stable=True)
        op._col = ei[1][o]
        op._deg = torch.bincount(ei[0], minlength=n)
        op._off = torch.cumsum(op._deg, 0) - op._deg
        op._csr_of = ei
    u = torch.rand(op._deg.shape[0], generator=gen, device=op._deg.device)
    k = torch.clamp((u * op._deg).long(), max=(op._deg - 1).clamp(min=0))
    return op._col[(op._off + k).clamp(max=op._col.numel() - 1)]


def _lattice_one_per_target(tgt, gen):
    """Indices into `tgt` keeping ONE event per target site, chosen at random -- two individuals
    reaching for one site in one tick: one succeeds, the other waits a tick."""
    if tgt.numel() == 0:
        return tgt
    key = torch.rand(tgt.shape, generator=gen, device=tgt.device, dtype=torch.float64)
    o = torch.argsort(tgt.double() * 2.0 + key)
    ts = tgt[o]
    last = torch.ones_like(ts, dtype=torch.bool)
    last[:-1] = ts[1:] != ts[:-1]
    return o[last]


def _lattice_gen(op, dev):
    if getattr(op, "_gen", None) is None or op._gen.device != torch.device(dev):
        op._gen = torch.Generator(device=dev)
        op._gen.manual_seed(int(getattr(op, "seed", 0)))
    return op._gen


@register_operator("cell_chem_diffuse", set="cell", kind="lateral", family="fields", implementation="lattice_exchange",
                   title="Morphogen diffusion between neighbouring cells",
                   equation=r"""$$P\big(i\leftrightarrow j\ \text{in}\ \Delta t\big)=\varepsilon\,\Delta t/\deg(i),\qquad \varepsilon=\tfrac12\,d\,\chi$$""")
class CellDiffuseLatticeExchange(Lateral):
    """`lattice_exchange` IMPLEMENTATION of cell_chem_diffuse -- the same mobility, computed on
    INDIVIDUALS: each occupied site swaps its content with a uniformly chosen neighbour at rate eps,
    hopping onto the neighbour's site if it is empty (Reichenbach et al. 2007's exchange, Fig. 1).

        P(i <-> j in dt) = eps dt / deg(i),      eps = d chi / 2

    THE SAME EQUATION, IN EXPECTATION -- which is what makes this an implementation. On a fully
    occupied lattice a site loses its content to a neighbour it initiates with (eps dt) and to each
    neighbour that initiates with it (eps dt / deg), so E[dc_i/dt] = 2 eps (mean_j c_j - c_i): the
    default `graph_laplacian` with d chi = 2 eps (tested), whose continuum limit on the unit lattice of N
    sites is D = eps / (2 N). REICHENBACH ET AL.'S CONVENTION: their mobility enters as "M Delta"
    (Suppl. Notes), i.e. M = D = eps / (2 N), and their M = 2 eps_paper / N makes eps_paper = eps / 4 --
    a rate per NEIGHBOUR PAIR on the 4-neighbour lattice. A spec at the paper's M therefore declares
    `d: [4 M N, ...]` (exp 15, Finding 17: the extinction probabilities and wavelengths agree with the
    paper only under this reading).
    Empty sites do not initiate (an exchange is "an individual" moving), as in the paper; the species'
    `d` must be equal, since what moves is the whole individual.

    Two swaps claiming one site in one tick are resolved in rounds: a swap is kept when it holds the
    smallest random key among every swap touching either of its two sites, and the losers draw again
    among the still-free sites (three rounds), so conflicts cost almost nothing at eps dt <= 0.05.

    CARRIED BLOCKS, `carry: [trait, ...]` (exp 15, direction 4): per-individual blocks that are not
    integrated (e.g. the attack trait of `cell_chem_react[rps_lattice]`'s `trait:`) swap with the
    individual -- for every kept swap (i, j), b_i <-> b_j for each listed block b, the same pairs as
    chem, written in place into the set's state (the operator then declares MAY_MUTATE_INTEGRATED_STATE;
    absent, it does not). A PARAM AND NOT A NEW OPERATOR because it is the same exchange: what moves is
    the whole individual, and a trait is part of it. Absent, nothing else changes (tested); the swaps
    and their draws are identical with or without it.

    Reads the provisional state (see the section comment above). Reference: Reichenbach, T.,
    Mobilia, M. & Frey, E. (2007). Nature 448:1046, Fig. 1 and Methods.
    """

    N_SPECIES = 3
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["chi"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]
    MECHANISM_TAGS = ["diffusion", "mobility", "exchange", "lattice", "individual_based"]
    REFERENCE = "Reichenbach, T., Mobilia, M. & Frey, E. (2007). Nature 448:1046."
    PARAM_ROLES = {"d": "per_species_diffusivity", "chi": "spatial_scale", "seed": "rng_seed",
                   "carry": "per_individual_blocks_moving_with_it"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.carry = [str(b) for b in (params.get("carry") or [])]
        if "chem" in self.carry:
            raise ValueError("cell_chem_diffuse[lattice_exchange]: `carry` lists blocks OTHER than chem -- "
                             "chem already moves, through the returned increment")
        if self.carry:
            self.MAY_MUTATE_INTEGRATED_STATE = True                     # instance only: absent => the check stands
        d = params.get("d")
        if d is None:
            raise ValueError("cell_chem_diffuse[lattice_exchange]: needs `d: [..]`, one per species")
        self.d = [float(v) for v in d]
        if max(self.d) - min(self.d) > 1e-12 * max(1.0, max(self.d)):
            raise ValueError("cell_chem_diffuse[lattice_exchange]: the species' `d` must be equal -- an "
                             "individual moves whole")
        self.N_SPECIES = len(self.d)
        self.chi = float(params["chi"])
        self.eps = 0.5 * self.d[0] * self.chi
        self.seed = int(params.get("seed", 0))
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)
        self.rounds = int(params.get("rounds", 3))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        ns, c0 = self.N_SPECIES, self.chan
        _span(chem, c0, ns, type(self).__name__)
        dt = float(getattr(H, "dt", 1.0))
        dev = chem.device
        gen = _lattice_gen(self, dev)
        x = _lattice_provisional(H, self.at, chem)
        occ, s = _lattice_sites(x, c0, ns)
        tgt = _lattice_onehot(occ, s, ns)
        n = tgt.shape[0]
        free = torch.ones(n, dtype=torch.bool, device=dev)
        # THE SWAPS ARE DRAWN ONCE, at rate eps; the rounds only settle conflicts among them. Drawing
        # afresh in every round tripled the exchange rate (tests/test_lattice_variants.py, identity
        # against the graph Laplacian, caught it).
        j = _lattice_neighbour(self, lvl, gen)
        a = occ & (torch.rand(n, generator=gen, device=dev) < self.eps * dt)
        ai, aj = torch.nonzero(a).flatten(), j[a]
        keep = ai != aj
        ai, aj = ai[keep], aj[keep]
        won_i, won_j = [], []
        for _ in range(self.rounds):
            live = free[ai] & free[aj]
            ai, aj = ai[live], aj[live]
            if ai.numel() == 0:
                break
            key = torch.rand(ai.shape, generator=gen, device=dev)
            m = torch.full((n,), 2.0, device=dev)
            m.scatter_reduce_(0, ai, key, reduce="amin")
            m.scatter_reduce_(0, aj, key, reduce="amin")
            win = (m[ai] == key) & (m[aj] == key)
            wi, wj = ai[win], aj[win]
            ti, tj = tgt[wi].clone(), tgt[wj].clone()
            tgt[wi], tgt[wj] = tj, ti
            free[wi] = False; free[wj] = False
            won_i.append(wi); won_j.append(wj)
            ai, aj = ai[~win], aj[~win]
        if self.carry:
            # The kept swaps are disjoint across rounds (`free`), so applying them all at once is the
            # same as applying them round by round.
            st = lvl.state.clone()
            wi = torch.cat(won_i) if won_i else torch.zeros(0, dtype=torch.long, device=dev)
            wj = torch.cat(won_j) if won_j else torch.zeros(0, dtype=torch.long, device=dev)
            for blk in self.carry:
                b0, b1 = lvl.state_schema[blk]
                vi, vj = st[wi, b0:b1].clone(), st[wj, b0:b1].clone()
                st[wi, b0:b1], st[wj, b0:b1] = vj, vi
            lvl.state = st
        out = torch.zeros_like(chem)
        out[:, c0:c0 + ns] = (tgt - x[:, c0:c0 + ns].to(tgt.dtype)).to(chem.dtype) / dt
        return {self.at: out}


@register_operator("cell_chem_diffuse", set="cell", kind="lateral", family="fields", implementation="closed_junctions",
                   title="Morphogen diffusion between neighbouring cells",
                   equation=r"""$$\frac{dc_i}{dt}=D_s\sum_{j\sim i}w_{ij}\frac{c_j-c_i}{\deg(i)},\qquad w_{ij}=0\ \text{across the closed line, else }1$$""")
class CellDiffuseClosedJunctions(CellDiffuse):
    """`closed_junctions` IMPLEMENTATION of cell_chem_diffuse -- the graph Laplacian of `graph_laplacian`
    with every junction that crosses a declared line shut:

        dc_i/dt = D_s sum_{j ~ i} w_ij (c_j - c_i) / deg(i)
        w_ij = 0 when the centroids of i and j lie on opposite sides of the plane, else 1

        closed: {point: [x, y, z], normal: [nx, ny, nz]}      required
        closed: {..., half_length: L}                         optional: a SEGMENT, not a whole line --
            only junctions whose midpoint lies within L of `point` along the line's in-plane tangent
            (-ny, nx) are shut: a scar with two free ends, around which a wave can turn

    This is the conduction-block experiment (Kleber & Rudy 2004): a line of closed gap junctions, and
    nothing else changed. A shut junction still counts in deg(i) -- closing one junction removes its
    current, it does not hand its conductance to the cell's other junctions -- so a cell away from the
    line sees exactly `graph_laplacian`'s arithmetic, and a line placed outside the sheet reproduces it
    to round-off (the test). An implementation and not a model, per the registry's axis for "the same
    coupling law, some edges switched off"; `graph_laplacian` itself is untouched.
    """
    READS = ["chem", "centroid"]
    PARAM_ROLES = {**CellDiffuse.PARAM_ROLES, "closed": "closed_junction_line"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        c = params.get("closed")
        if not c or "point" not in c or "normal" not in c:
            raise ValueError("cell_chem_diffuse[closed_junctions]: needs `closed: {point: [..], normal: [..]}` "
                             "-- without a line this is `graph_laplacian`, and should say so.")
        self.closed = c

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        ei = getattr(lvl, "edge_index", None)
        if ei is None or ei.numel() == 0:
            return {self.at: torch.zeros_like(chem)}
        i, j = ei[0], ei[1]; N = chem.shape[0]
        deg = torch.zeros(N, device=chem.device, dtype=chem.dtype).index_add_(0, i, torch.ones_like(i, dtype=chem.dtype))
        cen = lvl.get("centroid" if "centroid" in lvl.state_schema else "pos")
        p = torch.as_tensor([float(x) for x in self.closed["point"]], dtype=cen.dtype, device=cen.device)
        nrm = torch.as_tensor([float(x) for x in self.closed["normal"]], dtype=cen.dtype, device=cen.device)
        side = ((cen[:, :len(p)] - p) @ nrm) > 0
        shut = side[i] != side[j]
        if self.closed.get("half_length") is not None:
            tan = torch.stack([-nrm[1], nrm[0]])
            mid = 0.5 * (cen[i, :2] + cen[j, :2]) - p[:2]
            shut = shut & ((mid @ tan).abs() <= float(self.closed["half_length"]))
        w = (~shut).to(chem.dtype)[:, None]
        flux = torch.zeros_like(chem).index_add_(0, i, w * (chem[j] - chem[i]))
        lap = flux / deg.clamp(min=1)[:, None] if self.norm else flux
        _span(chem, self.chan, len(self.d), type(self).__name__)
        coef = torch.zeros(chem.shape[1], device=chem.device, dtype=chem.dtype)
        for _k, _dk in enumerate(self.d):
            coef[self.chan + _k] = _dk * self.chi
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: (coef[None, :] * lap) * occ}


@register_operator("cell_chem_diffuse", set="cell", kind="lateral", family="fields", model="contact_weighted",
                   title="Morphogen diffusion between neighbouring cells",
                   equation=r"""$$\frac{dc_i}{dt}=D_s\sum_{j\sim i}\frac{\ell_{ij}}{\bar\ell}\,\frac{c_j-c_i}{\deg(i)}$$""")
class CellDiffuseContactWeighted(CellDiffuse):
    """`contact_weighted` MODEL of cell_chem_diffuse -- a junction's conductance proportional to the
    length of membrane the two cells share, as gap-junction plaques are distributed along it:

        dc_i/dt = D_s sum_{j ~ i} (l_ij / l_bar) (c_j - c_i) / deg(i)

    l_ij is the contact length of the pair (the set's `edge_weight`, written by
    `cell_neighbours[model: label_image]` from the segmentation), l_bar its mean over the sheet's
    junctions, so the sheet's AVERAGE coupling is the default's and only its distribution changes: a
    long shared edge couples more, a corner contact almost not at all. With every contact the same
    length it IS `graph_laplacian` (the test). A model and not an implementation: it is a different
    claim about where the conductance sits.

    Reference: gap junctions at the intercalated disc scale with contact area: Kleber, A. G. & Rudy, Y.
    (2004). Physiol. Rev. 84:431-488 (section IV); Plexus (this work).
    """
    READS = ["chem"]

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        ei = getattr(lvl, "edge_index", None)
        w = getattr(lvl, "edge_weight", None)
        if ei is None or ei.numel() == 0:
            return {self.at: torch.zeros_like(chem)}
        if w is None or w.numel() != ei.shape[1]:
            raise ValueError("cell_chem_diffuse[contact_weighted]: the cell set has no `edge_weight` "
                             "aligned with its edge_index -- use cell_neighbours[model: label_image].")
        i, j = ei[0], ei[1]; N = chem.shape[0]
        deg = torch.zeros(N, device=chem.device, dtype=chem.dtype).index_add_(0, i, torch.ones_like(i, dtype=chem.dtype))
        wn = (w / w.mean()).to(chem.dtype)[:, None]
        flux = torch.zeros_like(chem).index_add_(0, i, wn * (chem[j] - chem[i]))
        lap = flux / deg.clamp(min=1)[:, None] if self.norm else flux
        _span(chem, self.chan, len(self.d), type(self).__name__)
        coef = torch.zeros(chem.shape[1], device=chem.device, dtype=chem.dtype)
        for _k, _dk in enumerate(self.d):
            coef[self.chan + _k] = _dk * self.chi
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: (coef[None, :] * lap) * occ}


@register_operator("cell_chem_diffuse", set="cell", kind="lateral", family="fields", model="fibre_anisotropic",
                   title="Morphogen diffusion between neighbouring cells",
                   equation=r"""$$\frac{dc_i}{dt}=D_s\sum_{j\sim i}\frac{a_{ij}}{\bar a}\,\frac{c_j-c_i}{\deg(i)},\quad a_{ij}=\tfrac12\sum_{k\in\{i,j\}}\big(\kappa\cos^2(\theta_{ij}-\phi_k)+\sin^2(\theta_{ij}-\phi_k)\big)$$""")
class CellDiffuseFibreAnisotropic(CellDiffuse):
    """`fibre_anisotropic` MODEL of cell_chem_diffuse -- cardiac conduction is faster along the fibre than
    across it, because the junctions sit mostly at the cells' ends (intercalated discs). Each
    junction's conductance follows the angle between it and the two cells' fibre axes:

        a_ij = 1/2 sum_{k in {i, j}} ( kappa cos^2(theta_ij - phi_k) + sin^2(theta_ij - phi_k) )
        dc_i/dt = D_s sum_{j ~ i} (a_ij / a_bar) (c_j - c_i) / deg(i)

    theta_ij is the direction from cell i's centroid to cell j's (`centroid`, else `pos`), phi_k the
    cell's fibre angle (its `phi` block -- the axis the cardio fit recovers), kappa = `ratio`, the
    along/across conductance ratio. Speed goes as the square root of conductance, so a speed
    anisotropy of 2.1 (Kleber & Rudy 2004 p. 441: from 10 in the crista terminalis to 2.1 in the
    ventricles) is kappa = 4.41, the default. a_bar is the sheet's mean, so the average coupling is the
    default's; with kappa = 1 it IS `graph_laplacian` (the test).

    Reference: Kleber, A. G. & Rudy, Y. (2004). Physiol. Rev. 84:431-488, section IV.A.
    """
    READS = ["chem", "phi"]
    PARAM_ROLES = {**CellDiffuse.PARAM_ROLES, "ratio": "along_across_conductance_ratio"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.ratio = float(params.get("ratio", 4.41))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        ei = getattr(lvl, "edge_index", None)
        if ei is None or ei.numel() == 0:
            return {self.at: torch.zeros_like(chem)}
        if "phi" not in lvl.state_schema:
            raise ValueError("cell_chem_diffuse[fibre_anisotropic]: the cell set has no `phi` block (the "
                             "fibre angle); seed it, e.g. seed_state_from_file from a cardio fit.")
        i, j = ei[0], ei[1]; N = chem.shape[0]
        cen = lvl.get("centroid" if "centroid" in lvl.state_schema else "pos")
        dx = cen[j, :2] - cen[i, :2]
        th = torch.atan2(dx[:, 1], dx[:, 0])
        phi = lvl.get("phi")[:, 0]
        k = self.ratio
        a = 0.5 * sum(k * torch.cos(th - phi[q]) ** 2 + torch.sin(th - phi[q]) ** 2 for q in (i, j))
        wn = (a / a.mean()).to(chem.dtype)[:, None]
        deg = torch.zeros(N, device=chem.device, dtype=chem.dtype).index_add_(0, i, torch.ones_like(i, dtype=chem.dtype))
        flux = torch.zeros_like(chem).index_add_(0, i, wn * (chem[j] - chem[i]))
        lap = flux / deg.clamp(min=1)[:, None] if self.norm else flux
        _span(chem, self.chan, len(self.d), type(self).__name__)
        coef = torch.zeros(chem.shape[1], device=chem.device, dtype=chem.dtype)
        for _k, _dk in enumerate(self.d):
            coef[self.chan + _k] = _dk * self.chi
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: (coef[None, :] * lap) * occ}


@register_operator("cell_chem_diffuse", set="cell", kind="lateral", family="fields", model="interface_weighted", title="Morphogen diffusion between neighbouring cells")
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
        degree-normalised graph Laplacian -- so d_a, d_h and chi keep the meaning they have in the
        default implementation, and only the DEVIATION from uniformity acts. (It also means a uniform
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

        _, _, _, vf = face_geometry_3d(pos, es, et, ef, nF, apex=wedge_apex(m, pos))   # per-cell wedge volume = the model's own v_i
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


# `cell_chem_diffuse[steady_uptake]` -- a nutrient supplied at the tissue's free edge and consumed by the
# living cells, at steady state every frame (exp12_tumor_invasion). Moved here from `nutrient_ops.py`
# on 2026-09-27 (experiments/INSTRUCTION.md: a variant lives beside its base operator).
def steady_uptake(pos, es, et, ef, nF, consume, c_init, uptake, K_m=0.01, supply=1.0,
                  newton_iters=40, tol=1e-6):
    """The nutrient at steady state on a cell mesh: the numerical core of
    `cell_chem_diffuse[steady_uptake]`, a pure numpy function so it can be tested on a hand-built mesh.

        0 = sum_j T_ij (c_j - c_i) + sum_b T_b (supply - c_i) - uptake * A_i * consume_i * c_i / (K_m + c_i)

    A two-point finite-volume balance on each cell i: T_ij = l_ij / |x_i - x_j| is the transmissibility
    of the wall cell i shares with j (l_ij the shared edge length, x the face centroids), T_b = l_b /
    |x_i - m_b| that of a FREE edge b (no twin) to the medium at its midpoint m_b, where the nutrient is
    held at `supply`. A_i is the cell's area (the norm of its vector area), `uptake` = a / D, the
    zero-order consumption over the diffusivity, in concentration / length^2, and K_m the Michaelis
    constant; both concentrations are in the units of `supply`, so lowering the supply at fixed
    uptake thins the fed rim as sqrt(supply). Solved by Newton on the uptake term from `c_init`; every
    iterate is clipped at 0. Returns (c [nF], newton iterations used, final max |dc|).
    """
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    pos = np.asarray(pos, float); es = np.asarray(es, np.int64); et = np.asarray(et, np.int64)
    ef = np.asarray(ef, np.int64)
    Nv = int(max(es.max(initial=0), et.max(initial=0))) + 1
    cnt = np.bincount(ef, minlength=nF).astype(float)
    cen = np.zeros((nF, 3)); np.add.at(cen, ef, pos[es]); cen /= np.maximum(cnt, 1)[:, None]
    va = np.zeros((nF, 3)); np.add.at(va, ef, 0.5 * np.cross(pos[es] - cen[ef], pos[et] - cen[ef]))
    A = np.linalg.norm(va, axis=1)
    key, tkey = es * Nv + et, et * Nv + es
    o = np.argsort(key); ks = key[o]
    j = np.clip(np.searchsorted(ks, tkey), 0, len(ks) - 1)
    has = ks[j] == tkey
    twin = np.where(has, ef[o[j]], -1)
    L = np.linalg.norm(pos[et] - pos[es], axis=1)
    inn = twin >= 0
    fi, fj = ef[inn], twin[inn]
    T = L[inn] / np.maximum(np.linalg.norm(cen[fi] - cen[fj], axis=1), 1e-12)
    bnd = ~inn
    mid = 0.5 * (pos[es[bnd]] + pos[et[bnd]])
    Tb = L[bnd] / np.maximum(np.linalg.norm(cen[ef[bnd]] - mid, axis=1), 1e-12)
    diag = np.bincount(fi, T, minlength=nF) + np.bincount(ef[bnd], Tb, minlength=nF)
    rhs0 = supply * np.bincount(ef[bnd], Tb, minlength=nF)
    M0 = coo_matrix((np.concatenate([diag + 1e-12, -T]), (np.concatenate([np.arange(nF), fi]),
                     np.concatenate([np.arange(nF), fj]))), shape=(nF, nF)).tocsr()
    q = float(uptake) * A * np.asarray(consume, float)
    K = float(K_m)
    c = np.clip(np.asarray(c_init, float), 0.0, None)
    dc, it = np.inf, 0
    for it in range(1, int(newton_iters) + 1):
        f = q * c / (K + c)
        fp = q * K / (K + c) ** 2
        J = M0 + coo_matrix((fp, (np.arange(nF), np.arange(nF))), shape=(nF, nF)).tocsr()
        cn = np.clip(spsolve(J.tocsc(), rhs0 - f + fp * c), 0.0, None)
        dc = float(np.max(np.abs(cn - c))) if nF else 0.0
        c = cn
        if dc < tol:
            break
    return c, it, dc


@register_operator("cell_chem_diffuse", set="cell", kind="lateral", family="fields", model="steady_uptake",
                   title="A nutrient supplied at the tissue edge, consumed by the cells",
                   equation=r"""$$0=D\nabla^{2}c-a\,\frac{c}{K_m+c}\,[\text{cell consumes}],\qquad c=c_{\text{supply}}\ \text{at the free edge}$$""")
class CellDiffuseSteadyUptake(Lateral):
    """`steady_uptake` MODEL of cell_chem_diffuse: a nutrient (oxygen, glucose) supplied by the medium
    at the tissue's free edge, diffusing through the cells and consumed by the living ones, AT ITS
    STEADY STATE EVERY FRAME.

    cell <- vertex mesh: reads the mesh (walls, areas, the free edge), `alive` and `apop_flag`, and
    WRITES chem[:, chan] in place with the solution of

        0 = D lap c - a c / (K_m + c)     in every consuming cell
        c = c_supply                      at the free edge (the medium)

    discretised as a two-point finite-volume balance (`steady_uptake` above): the flux between two
    cells goes through the wall they share over the distance between their centroids, so the field is
    the diffusion equation on the tissue's own geometry, not a graph average. a is the zero-order
    consumption per volume and D the diffusivity; only their ratio sets a steady field, so the one
    parameter is `uptake` = a / D, in concentration / length^2, concentrations in the units of
    `supply` (1.0 = the medium's normal level). Its one derived length is Grimes' minimum rim,
    r_m = sqrt(2 D c_supply / a) = sqrt(2 supply / uptake) (their eq. 2.10), so `uptake` = 2 supply / r_m^2
    calibrates the operator to a measured spheroid. K_m is the Michaelis constant, in the same units,
    and keeps c >= 0 where the tissue runs out.

    WHY A MODEL OF cell_chem_diffuse AND NOT A NEW OPERATOR. It is the same transformation -- a
    molecule moving between neighbouring cells, set=cell, lateral, reads and writes chem -- under two
    further claims, each a claim about the tissue: (1) the cells CONSUME it, with no feed but the
    edge, and (2) it equilibrates much faster than the tissue changes. Oxygen crosses a 580 um
    spheroid in R^2 / D ~ (5.8e-4 m)^2 / 2e-9 m^2/s ~ 3 minutes, a cell cycles in about a day, and
    Grimes et al. 2014 solve exactly this steady state (their eq. 2.2) to read spheroid layers. The
    explicit `graph_laplacian` step moves the field one cell per frame, so crossing a 45-cell radius
    takes ~ 45^2 x 4 = 8,000 frames: the nutrient would lag the tissue by weeks. Solving the steady
    state is the equilibrium the explicit form would reach, reached at once.

    WHO CONSUMES. A cell with `alive` > 0 and no `apop_flag`: a cell `cell_die` has marked is necrotic
    and consumes nothing, which is Grimes' anoxic core (no consumption inside r_n). It still conducts:
    the nutrient diffuses through dead tissue.

    WHAT IT DOES NOT MODEL: the medium's own boundary layer (the edge is held at c_supply, as in
    Grimes' p_o at r_o), and any source inside the tissue.

    Reference: Grimes, D. R. et al. (2014). A method for estimating the oxygen consumption rate in
    multicellular tumour spheroids. J. R. Soc. Interface 11:20131124 (eqs. 2.2-2.10);
    Mueller-Klieser, W. (1984). Biophys. J. 46:343-348; Eymard, Gallouet & Herbin (2000), finite
    volume two-point flux.
    """
    N_SPECIES = 1
    SUPPORTED_DIMS = [3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["uptake"]
    INPUTS = ["cell", "vertex"]; OUTPUTS = ["cell"]; READS = ["chem", "pos"]; WRITES = ["chem"]
    MECHANISM_TAGS = ["diffusion", "finite_volume", "nutrient", "consumption", "quasi_steady"]
    REFERENCE = ("Grimes, D. R. et al. (2014). J. R. Soc. Interface 11:20131124; Mueller-Klieser, W. "
                 "(1984). Biophys. J. 46:343-348.")
    PARAM_UNITS = {"K_m": "fraction", "supply": "fraction"}
    PARAM_ROLES = {"uptake": "consumption_over_diffusivity", "supply": "edge_concentration",
                   "K_m": "michaelis_constant", "newton_iters": "solver_iterations", "tol": "solver_tolerance"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell"); self.vat = params.get("vertex_set", "vertex")
        self.uptake = float(params["uptake"])
        self.supply = float(params.get("supply", 1.0))
        self.K_m = float(params.get("K_m", 0.01))
        self.newton_iters = int(params.get("newton_iters", 40))
        self.tol = float(params.get("tol", 1e-6))
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)
        self._warned = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at); vlvl = H.level(self.vat)
        chem = lvl.get("chem")
        m = getattr(vlvl, "_mesh", None)
        if m is None or int(m["nF"]) == 0:
            return {self.at: torch.zeros_like(chem)}
        _span(chem, self.chan, 1, type(self).__name__)
        nF = int(m["nF"]); Nv = int(m["Nv"])
        npy = lambda a: a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
        pos = npy(vlvl.get("pos")[:Nv]).astype(np.float64)
        consume = np.ones(nF)
        if m.get("alive") is not None:
            consume *= (npy(m["alive"])[:nF] > 0)
        flag = m.get("apop_flag")
        if flag is not None and len(flag) >= nF:
            consume *= (npy(flag)[:nF] <= 0)
        c0 = npy(chem[:nF, self.chan]).astype(np.float64)
        if not np.any(c0 > 0):                      # first call: start from the supply, not from zero
            c0 = np.full(nF, self.supply)
        c, it, dc = steady_uptake(pos, npy(m["E_srce"]), npy(m["E_trgt"]), npy(m["E_face"]), nF, consume,
                                  c0, self.uptake, self.K_m, self.supply, self.newton_iters, self.tol)
        if dc >= self.tol and not self._warned:
            print(f"[cell_chem_diffuse[steady_uptake]] Newton stopped at {it} iterations with max |dc| "
                  f"{dc:.2e} > tol {self.tol:.0e}; the field is not at its steady state.", flush=True)
            self._warned = True
        chem[:nF, self.chan] = torch.as_tensor(c, dtype=chem.dtype, device=chem.device)
        return {self.at: torch.zeros_like(chem)}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="gray_scott", title="Autocatalytic reaction",
                   species=(("a", "activator"), ("u", "substrate")),
                   equation=r"""$$\frac{da}{dt}=r\big(u a^{2}-(F+k)a\big),\qquad \frac{du}{dt}=r\big(-u a^{2}+F(1-u)\big)$$""")
class CellReactGrayScott(Lateral):
    """Gray-Scott autocatalysis: an activator that makes more of itself by consuming a substrate
    the system slowly replenishes. Pattern by substrate DEPLETION rather than by inhibition.

    cell -> cell: reads chem, emits d(chem)/dt. No relation is traversed -- the reaction is local.

        da/dt = r ( u a^2 - (F + k) a )        a = activator, and its own autocatalyst
        du/dt = r ( -u a^2 + F (1 - u) )       u = substrate, fed toward 1

    a and u are dimensionless concentrations. F is the feed rate, in inverse time: it both tops the
    substrate back up toward 1 and removes activator. k is the extra kill rate on the activator, in
    the same units, so the activator decays at F + k while the substrate is replenished at F -- and
    the whole morphology of the model is set by where (F, k) sits in that two-dimensional plane.
    r is `rate`, a dimensionless time rescaling of the whole reaction, so a pattern can be made to
    develop in fewer frames; it must be raised together with the diffusion for the two to stay in
    proportion.

    The a^2 u term is cubic, which is what makes the model bistable: a small activator
    perturbation dies, and a large enough one grows until it exhausts its local substrate and
    splits. That is why Gray-Scott makes spots that replicate, where an activator-inhibitor model
    makes a stationary peak.

    Reference: Gray, P. & Scott, S. K. (1984). Autocatalytic reactions in the isothermal,
    continuous stirred tank reactor. Chem. Eng. Sci. 39:1087-1097; Pearson, J. E. (1993). Complex
    patterns in a simple system. Science 261:189-192 (the (F, k) morphology map).
    """

    N_SPECIES = 2
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["F", "kk"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]
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
                   model="rock_paper_scissor", title="Autocatalytic reaction in cyclic competition",
                   species=(("u", "competitor 1"), ("v", "competitor 2"), ("w", "competitor 3")),
                   equation=r"""$$\begin{aligned}\frac{du}{dt}&=r\,u\,(1-p-a\,v)\\ \frac{dv}{dt}&=r\,v\,(1-p-a\,w)\\ \frac{dw}{dt}&=r\,w\,(1-p-a\,u)\end{aligned}\qquad p=u+v+w$$""")
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
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]
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


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", implementation="rps_lattice",
                   title="Autocatalytic reaction in cyclic competition",
                   equation=r"""$$AB\xrightarrow{\ \sigma\ }A\varnothing,\qquad A\varnothing\xrightarrow{\ \mu\ }AA,\qquad \sigma=r\,a,\ \mu=r$$""")
class CellReactRPSLattice(Lateral):
    """`rps_lattice` IMPLEMENTATION of cell_chem_react -- the May-Leonard cyclic competition of
    `rock_paper_scissor`, computed on INDIVIDUALS on lattice sites (Reichenbach et al. 2007, Fig. 1):

        selection     a site's individual picks a random neighbour at rate sigma = r a; if the
                      neighbour is its prey, the neighbour dies (the site is emptied)
        reproduction  it picks a random neighbour at rate mu = r; if that site is empty, it is
                      filled with an individual of its own species

    Columns [u, v, w] from `chan`, as in `rock_paper_scissor`: v kills u, w kills v, u kills w.

    THE SAME BIOLOGY, IN EXPECTATION -- which is what makes it an implementation of the same reaction.
    On an uncorrelated lattice with densities (u, v, w) and p = u + v + w, u gains mu u (1 - p) per unit
    time from births into empty neighbours and loses sigma u v to its predator: du/dt = r u (1 - p -
    a v), exactly `rock_paper_scissor` with the same `r` and `a` (tested). The paper runs sigma = mu = 1,
    i.e. `rate: 1, a: 1`. Individuals, empty sites and neighbours make it the stochastic lattice whose
    extinction probability against mobility is the paper's Fig. 2b; the density model is its mean field.

    CROSS-FEEDING, `feed: {block: food, matrix: F}` (exp 15, direction 3; Momeni et al. 2013, Han et al.
    2021): an individual of species i reproduces at mu_i = mu (1 + sum_k F[i][k] f_k), f_k its site's
    column k of the `food` block (a metabolite concentration, written by `readout` from a medium set),
    F a 3 x 3 matrix of benefits per unit concentration, row = consumer, column = metabolite. Absent (or
    F = 0) every mu_i = mu and the draws are the same, so the model above is unchanged (tested).

    HERITABLE FEEDING PREFERENCE, `feed: {..., heritable: {block: pref, mutation: m_p, budget: B, init:
    "matrix" | [p0, p1, p2], seed: k}}` (exp 15, directions 3 x 4: does selection favour feeding on the
    predator's metabolite?): every individual carries its own preference vector p_i over the 3
    metabolite columns of `food`, in a width-3 block of the set (`integration: none`; 0 on an empty
    site), and it replaces the species' row of F:

        mu_i = mu (1 + sum_k p_ik f_k),      p_ik >= 0,   sum_k p_ik = B
        daughter:  p_daughter = rescale_B( max(0, p_i + m_p xi) ),  xi ~ N(0, I_3)    a killed site: p = 0

    p_ik is a benefit per unit concentration of metabolite k, in the units of F (so p_i = F[s_i] is the
    matrix model); f_k is the site's column k of the `food` block, as above; B (default 1) is the total
    benefit an individual spreads across the three metabolites -- a choice between them, not a free
    gain; rescale_B multiplies a vector by B / (its sum), and a vector clamped to all zeros becomes B/3
    in every column; m_p is the standard deviation of each component of the daughter's p around the
    parent's, in the units of p, before the clamp and rescale (at m_p = 0 the daughter copies the
    parent's p exactly). `init` (default "matrix") fills p on every occupied site on the first call if
    the block is still all zeros: "matrix" gives each individual its species' row of F rescaled to sum
    B, a 3-list gives every individual that list rescaled to sum B; F is used for nothing else (and may
    be omitted with a list). Sites the provisional state holds empty are reset to p = 0 on every call.
    The mutation noise comes from its own generator seeded `heritable.seed` (default `seed` + 3), so
    the event draws and the trait and defence draws are never shifted by it. At init "matrix",
    m_p = 0 and B = every row's sum, the events are the matrix feed's exactly (tested).
    `cell_chem_diffuse[lattice_exchange]`'s `carry: [pref]` moves p with its individual.

    ARMS RACE, `trait: {block: trait, mutation: m, cost: c, seed: k, init: t0}` (exp 15, direction 4):
    every individual carries a heritable attack trait t_i >= 0 in a width-1 block of the set
    (`integration: none`; 0 on an empty site). It scales the individual's own selection rate and
    taxes its reproduction:

        P(i kills its prey neighbour in dt) = sigma dt t_i
        P(i reproduces into an empty neighbour in dt) = mu dt max(0, 1 - c t_i)   (x the feed factor)
        daughter:  t_daughter = max(0, t_i + m xi),  xi ~ N(0, 1)      a killed site: t = 0

    t_i is dimensionless -- a multiple of the base selection rate sigma = r a -- so t_i = 1 is the
    model above; c is the fractional loss of the base reproduction rate mu = r per unit of trait
    (c t_i = 1 stops the individual reproducing); m is the standard deviation of the daughter's trait
    around the parent's, in the same units of t. The trait is NOT integrated: it is written in place
    into the set's state (the individual-based bookkeeping of who carries what), so the operator
    declares MAY_MUTATE_INTEGRATED_STATE only when `trait` is given. `init: t0` (optional) sets t = t0 on
    every occupied site on the first call if the block is still all zeros -- `seed_state_random` would
    fill empty sites too. Sites the provisional state holds empty are reset to t = 0 on every call.
    A PARAM OF THIS IMPLEMENTATION AND NOT A NEW OPERATOR because the events, the draws and the chem
    increment are exactly this operator's; the trait only rescales two of its probabilities per
    individual. t = 1, m = 0, c = 0 draws the same events as the model without it (tested). The
    mutation draw uses its own generator (`seed`), so the event draws are never shifted by it.
    `cell_chem_diffuse[lattice_exchange]`'s `carry: [trait]` moves the trait with its individual.

    COEVOLVING DEFENCE, `trait: {..., defence: {block: defence, mutation: m_d, cost: c_d, init: d0}}`
    (exp 15, direction 4, attack vs defence): every individual also carries a heritable defence
    d_i >= 0 in its own width-1 block (`integration: none`; 0 on an empty site). It divides the
    selection rate of any attacker reaching it and taxes the carrier's own reproduction:

        P(i kills its prey neighbour j in dt) = sigma dt t_i / (1 + d_j)
        P(i reproduces into an empty neighbour in dt) = mu dt max(0, 1 - c t_i - c_d d_i)   (x feed)
        daughter:  d_daughter = max(0, d_i + m_d zeta),  zeta ~ N(0, 1)   a killed site: d = 0

    d_j is dimensionless: d_j = 1 halves the rate at which any attacker kills j, d = 0 is the attack-only
    model above exactly (tested). c_d is the fractional loss of the base reproduction rate mu = r per
    unit of defence, summed with the attack cost c t_i; m_d is the standard deviation of the daughter's
    defence around the parent's, in units of d. `init: d0` and the empty-site reset behave as the
    trait's. The defence noise comes from a SECOND generator seeded `defence.seed` (default `seed` + 2),
    so the attack mutation draws are never shifted by it, at any m_d. `carry: [trait, defence]`.

    One event per target site per tick (two individuals reaching for one site: one succeeds); an
    individual acts at most once per tick. Reads the provisional state (see the lattice section above
    `cell_chem_diffuse[lattice_exchange]`). Reference: Reichenbach, T., Mobilia, M. & Frey, E. (2007).
    Nature 448:1046; May, R. M. & Leonard, W. J. (1975). SIAM J. Appl. Math. 29:243.
    """
    N_SPECIES = 3
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = False
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]
    MECHANISM_TAGS = ["reaction", "competition", "cyclic_dominance", "may_leonard", "rock_paper_scissor",
                      "lattice", "individual_based"]
    PARAM_ROLES = {"a": "cyclic_suppression", "rate": "reaction_time_scale", "seed": "rng_seed",
                   "feed": ("cross_feeding_benefit_matrix_and_food_block"
                            "_with_optional_heritable_preference_block_mutation_sd_budget_and_init"),
                   "trait": ("heritable_attack_trait_block_mutation_sd_and_reproduction_cost"
                             "_with_optional_defence_block_dividing_the_attackers_kill_rate")}
    REFERENCE = ("Reichenbach, T., Mobilia, M. & Frey, E. (2007). Nature 448:1046; "
                 "May, R. M. & Leonard, W. J. (1975). SIAM J. Appl. Math. 29:243-253.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.a = float(params.get("a", 0.6))
        self.rate = float(params.get("rate", 1.0))
        self.seed = int(params.get("seed", 0))
        fd = params.get("feed")
        self.feed_block = None if not fd else str(fd.get("block", "food"))
        hr = None if not fd else fd.get("heritable")
        mat = None if not fd else fd.get("matrix")
        if fd and mat is None and (hr is None or hr.get("init", "matrix") == "matrix"):
            raise ValueError("cell_chem_react[rps_lattice]: `feed` needs `matrix` (or `heritable.init` as a 3-list)")
        self.feed_F = None if mat is None else [[float(v) for v in row] for row in mat]
        self.feed_on = bool(fd)
        self.pref_block = None if hr is None else str(hr.get("block", "pref"))    # `heritable: {}` = all defaults
        if hr is not None:
            self.pref_mut = float(hr.get("mutation", 0.0))             # sd of each daughter component, units of p
            self.pref_budget = float(hr.get("budget", 1.0))            # B = sum_k p_ik, in the units of F
            ini = hr.get("init", "matrix")
            if isinstance(ini, str):
                if ini != "matrix":
                    raise ValueError(f"cell_chem_react[rps_lattice]: heritable.init {ini!r} is 'matrix' or a 3-list")
                self.pref_init = "matrix"
            else:
                self.pref_init = [float(v) for v in ini]
                if len(self.pref_init) != 3:
                    raise ValueError("cell_chem_react[rps_lattice]: heritable.init as a list has 3 entries")
            self.pref_seed = int(hr.get("seed", self.seed + 3))
            self._pref_gen = None
            self._pref_first = True
            self.MAY_MUTATE_INTEGRATED_STATE = True                    # instance only: absent => the check stands
        tr = params.get("trait")
        self.trait_block = None if tr is None else str(tr.get("block", "trait"))   # `trait: {}` = all defaults
        self.def_block = None
        if tr is not None:
            self.trait_mut = float(tr.get("mutation", 0.0))            # sd of the daughter's trait, units of t
            self.trait_cost = float(tr.get("cost", 0.0))               # fraction of mu lost per unit of t
            self.trait_seed = int(tr.get("seed", self.seed + 1))
            self.trait_init = None if tr.get("init") is None else float(tr["init"])
            self._trait_gen = None
            self._trait_first = True
            df = tr.get("defence")
            self.def_block = None if df is None else str(df.get("block", "defence"))
            if df is not None:
                self.def_mut = float(df.get("mutation", 0.0))          # sd of the daughter's defence, units of d
                self.def_cost = float(df.get("cost", 0.0))             # fraction of mu lost per unit of d
                self.def_seed = int(df.get("seed", self.seed + 2))
                self.def_init = None if df.get("init") is None else float(df["init"])
                self._def_gen = None
            self.MAY_MUTATE_INTEGRATED_STATE = True                    # instance only: absent => the check stands
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)

    def _pref_rescale(self, p):
        """Each row times B / (its sum); a row summing to 0 becomes B/3 in every column."""
        tot = p.sum(1, keepdim=True)
        B = self.pref_budget
        return torch.where(tot > 0, p * (B / tot.clamp(min=1e-300)), torch.full_like(p, B / 3.0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        c0 = self.chan
        _span(chem, c0, 3, type(self).__name__)
        dt = float(getattr(H, "dt", 1.0))
        dev = chem.device
        gen = _lattice_gen(self, dev)
        x = _lattice_provisional(H, self.at, chem)
        occ, s = _lattice_sites(x, c0, 3)
        tgt = _lattice_onehot(occ, s, 3)
        n = occ.shape[0]
        sig, mu = self.rate * self.a * dt, self.rate * dt
        writes = self.trait_block is not None or self.pref_block is not None
        st = lvl.state.clone() if writes else None                       # in-place writer: a fresh buffer
        if self.pref_block is not None:
            pb0, pb1 = lvl.state_schema[self.pref_block]
            if pb1 - pb0 != 3:
                raise ValueError(f"cell_chem_react[rps_lattice]: preference block {self.pref_block!r} must be width 3")
            if self._pref_first and not torch.any(st[:, pb0:pb1] != 0):
                if self.pref_init == "matrix":                           # each individual: its species' row of F
                    rows = torch.as_tensor(self.feed_F, dtype=torch.float64, device=dev)[s.clamp(min=0).long()]
                else:                                                    # every individual: the given list
                    rows = torch.as_tensor(self.pref_init, dtype=torch.float64, device=dev).expand(n, 3)
                st[occ, pb0:pb1] = self._pref_rescale(rows[occ]).to(st.dtype)
            self._pref_first = False
            st[~occ, pb0:pb1] = 0.0                                      # an empty site carries no preference
            pref = st[:, pb0:pb1].clone()                                # p_i at this tick's start
            food = lvl.get(self.feed_block).to(torch.float64)            # [n, 3] metabolite at the site
            mu = (mu * (1.0 + (food * pref.to(torch.float64)).sum(1))).to(chem.dtype)   # mu (1 + sum_k p_ik f_k)
        elif self.feed_on:
            food = lvl.get(self.feed_block).to(torch.float64)                # [n, 3] metabolite at the site
            F = torch.as_tensor(self.feed_F, dtype=torch.float64, device=dev)
            mu = (mu * (1.0 + (food @ F.T).gather(1, s.clamp(min=0).long()[:, None])[:, 0])).to(chem.dtype)
        if self.trait_block is not None:
            tb0, tb1 = lvl.state_schema[self.trait_block]
            if tb1 - tb0 != 1:
                raise ValueError(f"cell_chem_react[rps_lattice]: trait block {self.trait_block!r} must be width 1")
            if self._trait_first and self.trait_init is not None and not torch.any(st[:, tb0] != 0):
                st[occ, tb0] = self.trait_init                           # `init`: occupied sites only
            if self.def_block is not None:
                db0, db1 = lvl.state_schema[self.def_block]
                if db1 - db0 != 1:
                    raise ValueError(f"cell_chem_react[rps_lattice]: defence block {self.def_block!r} must be width 1")
                if self._trait_first and self.def_init is not None and not torch.any(st[:, db0] != 0):
                    st[occ, db0] = self.def_init                         # `init`: occupied sites only
                st[~occ, db0] = 0.0                                      # an empty site carries no defence
                dfn = st[:, db0].to(chem.dtype)                          # d_i, divides attackers' sigma
            self._trait_first = False
            st[~occ, tb0] = 0.0                                          # an empty site carries no trait
            t = st[:, tb0].to(chem.dtype)                                # t_i, a multiple of sigma
            sig = sig * t                                                # per individual: sigma dt t_i
            if self.def_block is None:
                mu = mu * (1.0 - self.trait_cost * t).clamp(min=0.0)     # mu dt max(0, 1 - c t_i) (x feed)
            else:                                                        # mu dt max(0, 1 - c t_i - c_d d_i)
                mu = mu * (1.0 - self.trait_cost * t - self.def_cost * dfn).clamp(min=0.0)
        u = torch.rand(n, generator=gen, device=dev)
        j = _lattice_neighbour(self, lvl, gen)
        prey = (s + 2) % 3                                               # v (1) kills u (0), w kills v, u kills w
        # The victim's defence divides the attacker's selection probability: sigma dt t_i / (1 + d_j).
        # Births need an empty j, where d_j = 0, so the birth window [sig, sig + mu) keeps the attacker's sig.
        sig_k = sig if self.def_block is None else sig / (1.0 + dfn[j])
        kill = occ & (u < sig_k) & occ[j] & (s[j] == prey) & (j != torch.arange(n, device=dev))
        birth = occ & (u >= sig) & (u < sig + mu) & ~occ[j]
        ai = torch.nonzero(kill | birth).flatten()
        if ai.numel():
            keep = _lattice_one_per_target(j[ai], gen)
            ai = ai[keep]
            aj = j[ai]
            k = kill[ai]
            tgt[aj[k]] = 0.0                                             # selection: the prey's site empties
            b = ~k
            tgt[aj[b]] = 0.0
            tgt[aj[b], s[ai[b]]] = 1.0                                   # reproduction: its species fills it
            if self.trait_block is not None:
                # Parents' traits read from `t` (this tick's start), so an individual killed while it
                # reproduces still passes on its own trait; kill and birth targets are disjoint (a birth
                # needs an empty site, a kill an occupied one).
                child = t[ai[b]].to(st.dtype)
                if self.trait_mut != 0.0 and child.numel():
                    if self._trait_gen is None or self._trait_gen.device != torch.device(dev):
                        self._trait_gen = torch.Generator(device=dev)
                        self._trait_gen.manual_seed(self.trait_seed)
                    child = child + self.trait_mut * torch.randn(child.shape, generator=self._trait_gen,
                                                                 device=dev, dtype=st.dtype)
                st[aj[b], tb0] = child.clamp(min=0.0)                    # daughter: parent's t + m xi, >= 0
                st[aj[k], tb0] = 0.0                                     # the killed site carries none
                if self.def_block is not None:
                    dchild = dfn[ai[b]].to(st.dtype)                     # parent's d at this tick's start
                    if self.def_mut != 0.0 and dchild.numel():
                        if self._def_gen is None or self._def_gen.device != torch.device(dev):
                            self._def_gen = torch.Generator(device=dev)
                            self._def_gen.manual_seed(self.def_seed)
                        dchild = dchild + self.def_mut * torch.randn(dchild.shape, generator=self._def_gen,
                                                                     device=dev, dtype=st.dtype)
                    st[aj[b], db0] = dchild.clamp(min=0.0)               # daughter: parent's d + m_d zeta, >= 0
                    st[aj[k], db0] = 0.0                                 # the killed site: no defence
            if self.pref_block is not None:
                pchild = pref[ai[b]]                                     # parent's p at this tick's start
                if self.pref_mut != 0.0 and pchild.numel():
                    if self._pref_gen is None or self._pref_gen.device != torch.device(dev):
                        self._pref_gen = torch.Generator(device=dev)
                        self._pref_gen.manual_seed(self.pref_seed)
                    pchild = pchild.to(torch.float64) + self.pref_mut * torch.randn(
                        pchild.shape, generator=self._pref_gen, device=dev, dtype=torch.float64)
                    pchild = self._pref_rescale(pchild.clamp(min=0.0)).to(st.dtype)   # >= 0, sum B
                st[aj[b], pb0:pb1] = pchild                              # m_p = 0: the parent's p exactly
                st[aj[k], pb0:pb1] = 0.0                                 # the killed site carries none
        if writes:
            lvl.state = st
        out = torch.zeros_like(chem)
        out[:, c0:c0 + 3] = (tgt - x[:, c0:c0 + 3].to(tgt.dtype)).to(chem.dtype) / dt
        return {self.at: out}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="kerr_csr",
                   title="Colicin producer, sensitive and resistant strains on a lattice",
                   species=(("C", "colicin producer"), ("S", "sensitive"), ("R", "resistant")),
                   equation=r"""$$\varnothing\xrightarrow{\ f_i\ }i,\qquad i\xrightarrow{\ \Delta_i\ }\varnothing,\qquad \Delta_S=\Delta_{S,0}+\tau f_C$$""")
class CellReactKerrCSR(Lateral):
    """`kerr_csr` MODEL of cell_chem_react -- Kerr et al. 2002's C-S-R community on a lattice (Box 1).
    Columns [C, S, R] from `chan`: C makes colicin (and pays for it), S is killed by it, R resists it.

        an empty site, updated, is filled with strain i with probability f_i
        a site of strain i, updated, dies with probability Delta_i,   Delta_S = Delta_S,0 + tau f_C

    f_i is the fraction of the site's NEIGHBOURHOOD occupied by i: `neighbourhood: local` reads the
    set's relation (the 8 surrounding sites with `radius_graph[periodic_tiles]`, radius 1.5); `global`
    reads every other site of its lattice (the relation `tile_index`, else the whole set) -- the
    well-mixed flask. Every site is updated at rate `rate` per unit time (1: one update per site per
    EPOCH, the paper's N updates), so a probability p per update becomes p rate dt per tick. The
    paper's values: Delta_C = 1/3, Delta_S,0 = 1/4, Delta_R = 10/32, tau = 3/4 (Fig. 1 caption).

    WHY A MODEL OF `cell_chem_react`: it is the reaction step of a three-species community on the
    cell graph -- who replaces whom -- under the same contract as `rock_paper_scissor`, but a different
    biology (killing by a diffusing toxin within a neighbourhood, costs as death rates), so a model
    and not an implementation. Reads the provisional state (lattice section above).

    Reference: Kerr, B., Riley, M. A., Feldman, M. W. & Bohannan, B. J. M. (2002). Local dispersal
    promotes biodiversity in a real-life game of rock-paper-scissors. Nature 418:171, Box 1.
    """
    N_SPECIES = 3
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = False
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]
    REQUIRES_PARAMS = ["delta_c", "delta_s0", "delta_r", "tau"]
    MECHANISM_TAGS = ["reaction", "competition", "cyclic_dominance", "colicin", "allelopathy", "lattice",
                      "individual_based"]
    PARAM_ROLES = {"delta_c": "death_prob_C", "delta_s0": "death_prob_S_without_C", "delta_r": "death_prob_R",
                   "tau": "toxicity", "neighbourhood": "local_or_global", "rate": "updates_per_unit_time",
                   "seed": "rng_seed"}
    REFERENCE = "Kerr, B., Riley, M. A., Feldman, M. W. & Bohannan, B. J. M. (2002). Nature 418:171, Box 1."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.dc, self.ds0 = float(params["delta_c"]), float(params["delta_s0"])
        self.dr, self.tau = float(params["delta_r"]), float(params["tau"])
        self.nb = str(params.get("neighbourhood", "local"))
        if self.nb not in ("local", "global"):
            raise ValueError(f"cell_chem_react[kerr_csr]: neighbourhood {self.nb!r} is not local or global")
        self.rate = float(params.get("rate", 1.0))
        self.seed = int(params.get("seed", 0))
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        c0 = self.chan
        _span(chem, c0, 3, type(self).__name__)
        dt = float(getattr(H, "dt", 1.0))
        dev = chem.device
        gen = _lattice_gen(self, dev)
        x = _lattice_provisional(H, self.at, chem)
        occ, s = _lattice_sites(x, c0, 3)
        oh = _lattice_onehot(occ, s, 3)
        n = occ.shape[0]
        if self.nb == "local":
            ei = getattr(lvl, "edge_index", None)
            if ei is None or ei.numel() == 0:
                raise ValueError("cell_chem_react[kerr_csr]: `local` needs a neighbour relation -- schedule "
                                 "`radius_graph` (model periodic_tiles, radius 1.5) before it")
            agg = torch.zeros_like(oh).index_add_(0, ei[0], oh[ei[1]])
            deg = torch.bincount(ei[0], minlength=n).clamp(min=1).to(oh.dtype)
            f = agg / deg[:, None]
        else:
            tile = getattr(lvl, "tile_index", None)
            tile = torch.zeros(n, dtype=torch.long, device=dev) if tile is None else tile.to(dev)
            nt = int(tile.max()) + 1
            tot = torch.zeros(nt, 3, device=dev, dtype=oh.dtype).index_add_(0, tile, oh)
            size = torch.bincount(tile, minlength=nt).to(oh.dtype)
            f = (tot[tile] - oh) / (size[tile] - 1).clamp(min=1)[:, None]    # every OTHER site of its lattice
        upd = torch.rand(n, generator=gen, device=dev) < self.rate * dt       # this site is updated this tick
        u = torch.rand(n, generator=gen, device=dev)
        cf = torch.cumsum(f, 1)
        new_s = (u[:, None] >= cf).sum(1)                                     # 0,1,2, or 3 = stays empty
        fill = upd & ~occ & (new_s < 3)
        delta_i = torch.stack([torch.full((n,), self.dc, device=dev),
                               self.ds0 + self.tau * f[:, 0],
                               torch.full((n,), self.dr, device=dev)], 1)
        die = upd & occ & (u < delta_i.gather(1, s[:, None]).squeeze(1))
        tgt = oh.clone()
        tgt[die] = 0.0
        tgt[fill, new_s[fill]] = 1.0
        out = torch.zeros_like(chem)
        out[:, c0:c0 + 3] = (tgt - x[:, c0:c0 + 3].to(tgt.dtype)).to(chem.dtype) / dt
        return {self.at: out}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields",
                   model="gray_scott_coupled", title="Two autocatalytic reactions, coupled",
                   species=(("a1", "activator 1"), ("u1", "substrate 1"), ("a2", "activator 2"), ("u2", "substrate 2")),
                   equation=r"""$$\begin{aligned}\frac{da_1}{dt}&=r\big(u_1a_1^{2}-(F_1+k_1)a_1-g\,a_1a_2\big), & \frac{du_1}{dt}&=r\big(-u_1a_1^{2}+F_1(1-u_1)\big)\\ \frac{da_2}{dt}&=r\big(u_2a_2^{2}-(F_2+k_2)a_2-g\,a_2a_1\big), & \frac{du_2}{dt}&=r\big(-u_2a_2^{2}+F_2(1-u_2)\big)\end{aligned}$$""")
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
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]
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


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="gierer_meinhardt", title="Activator-inhibitor reaction",
                   species=(("a", "activator"), ("h", "inhibitor")),
                   equation=r"""$$\frac{da}{dt}=r\Big(\rho\,\frac{a^{2}}{h}-\mu_a a+a_0\Big),\qquad \frac{dh}{dt}=r\big(\rho\,a^{2}-\mu_h h\big)$$""")
class CellReactGiererMeinhardt(Lateral):
    """Gierer-Meinhardt activator(a)-inhibitor(h) -- the RD OKUDA uses (ref 37). chem = [a, h]:
        da/dt = gm_rho * a^2/h - mu_a * a + a0     (SELF-ENHANCING activator: the a^2/h AUTOCATALYSIS is the
        dh/dt = gm_rho * a^2   - mu_h * h            amplification feedback that self-maintains a localised PEAK)
    Paired (in cell_chem_diffuse) with a FAST inhibitor (d_h >> d_a via chi) -> lateral inhibition -> a stable
    localised activator peak WITH A GRADIENT (Okuda's tip spot), unlike Brusselator (decays the seed) or
    Gray-Scott (substrate-depletion). `rate` time-scales the reaction; a0 is a small basal activator source."""
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]
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


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="source_decay",
                   title="A sustained morphogen source, and first-order decay",
                   species=(("c", "morphogen"),),
                   equation=r"""$$\frac{dc_i}{dt}=r\big(p\,\mathbb{1}[i\in S]-k\,c_i\big),\qquad S=\{i:\ x_i-x_{\min}<f\,(x_{\max}-x_{\min})\}$$""")
class CellReactSourceDecay(Lateral):
    """A morphogen made by a SOURCE population every frame and lost everywhere at a first-order rate --
    the floor plate secreting Shh into the neural tube. chem = [c], one column:

        dc_i/dt = r ( p 1[i in S] - k c_i )
        S = the cells whose centroid lies within the fraction f of the sheet's current extent along
            `axis`, from its `low` (or `high`) edge

    c is the morphogen's concentration, dimensionless. p is `production`, the source cells' synthesis
    rate, in concentration per unit time; k is `decay`, the first-order loss rate, in inverse time,
    applied to every cell including the source; r is `rate`, a time rescaling of the whole reaction.
    With `cell_chem_diffuse` (graph_laplacian, norm: true) on the same column the steady profile away
    from the source is c ~ exp(-x / lambda), lambda = sqrt(D_eff / k), D_eff = d chi <l^2> / 4 for a
    neighbour set that is isotropic, l the distance between neighbouring centroids. Decay length and
    amplitude are thus separate dials: k (with d) sets lambda, p sets the amplitude.

    WHY THIS AND NOT A SEED. Every seed runs only in the opening frames (the runtime confines them), so
    `seed_cell_chem[cones]` or `seed_type_by_axis` gives an initial pulse that diffuses and decays
    away, never a steady gradient. A source is a term of the dynamics, so it lives in a reaction model.
    (Audited 2026-09-26 for exp 7; `decay` is a field-grid operator that removes a constant amount,
    not dc/dt = -k c, and does not act on the cell set's `chem`.)

    THE SOURCE IS A FRACTION OF THE CURRENT EXTENT, re-read every frame from the centroids, so it grows
    with the sheet: the neural tube's floor plate keeps 0.06-0.085 of the dorso-ventral length from 30
    to 90 hours post headfold (Kicheva et al. 2014, Fig. 1E). `production: 0` is the no-source control,
    one line. `source: {axis: 0, side: low, frac: 0.07}`.

    Reference: Crick, F. (1970). Diffusion in embryogenesis. Nature 225:420-422 (source, diffusion,
    decay); Kicheva, A. et al. (2014). Science 345:1254927 (the floor plate's share of the length).
    """
    N_SPECIES = 1
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["production", "decay"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem", "centroid"]; WRITES = ["chem"]
    MECHANISM_TAGS = ["reaction", "morphogen", "source", "decay", "gradient"]
    PARAM_ROLES = {"production": "source_synthesis_rate", "decay": "first_order_decay_rate",
                   "source": "source_region", "rate": "reaction_time_scale"}
    REFERENCE = "Crick, F. (1970). Nature 225:420-422; Kicheva, A. et al. (2014). Science 345:1254927."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.p = float(params["production"]); self.k = float(params["decay"])
        self.rate = float(params.get("rate", 1.0))
        src = params.get("source") or {}
        self.axis = int(src.get("axis", 0)); self.side = str(src.get("side", "low"))
        self.frac = float(src.get("frac", 0.07))
        if self.side not in ("low", "high"):
            raise ValueError(f"source_decay: source.side must be low or high, got {self.side!r}")
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)

    def source_mask(self, lvl):
        cen = lvl.get("centroid")
        x = cen[:, self.axis]
        occ = getattr(lvl, "occ", None)
        live = (occ > 0) if occ is not None else torch.ones_like(x, dtype=torch.bool)
        big = torch.finfo(x.dtype).max
        lo = torch.where(live, x, torch.full_like(x, big)).min()
        hi = torch.where(live, x, torch.full_like(x, -big)).max()
        L = hi - lo
        s = (x - lo) if self.side == "low" else (hi - x)
        return (s < self.frac * L) & live

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        (c,) = _span(chem, self.chan, 1, type(self).__name__)
        dc = self.p * self.source_mask(lvl).to(chem.dtype) - self.k * c
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: _emit(chem, self.chan, (dc,), self.rate, occ)}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="source_ptch",
                   title="A morphogen source, sequestered by the receptor it induces (Hedgehog / PTCH)",
                   species=(("c", "morphogen"), ("R", "receptor")),
                   equation=r"""$$\frac{dc_i}{dt}=r\big(p\,\mathbb{1}[i\in S]-k\,c_i-s\,R_i\,c_i\big),\qquad \frac{dR_i}{dt}=r\Big(b_0+b_1\frac{c_i^{\,n}}{K^n+c_i^{\,n}}-k_R\,R_i\Big)$$""")
class CellReactSourcePtch(CellReactSourceDecay):
    """`source_decay` with the morphogen also removed by a RECEPTOR THE MORPHOGEN INDUCES -- Shh bound
    and sequestered by PTCH, whose own expression Hedgehog signalling raises (Li et al. 2018). chem =
    [c, R], two adjacent columns from `chan`:

        dc_i/dt = r ( p 1[i in S] - k c_i - s R_i c_i )
        dR_i/dt = r ( b0 + b1 c_i^n / (K^n + c_i^n) - kR R_i )

    c is the morphogen (dimensionless, as in `source_decay`; p, k, S and r are that model's). R is the
    receptor level of cell i, dimensionless; s is `sequester`, the removal rate per unit receptor per
    unit time; b0 is `r_basal`, the receptor's basal synthesis; b1 is `r_induced`, the extra synthesis
    at full signalling, a Hill function of c with half-point K (`r_half`, in units of c) and exponent
    n (`r_hill`); kR is `r_decay`, the receptor's own turnover rate. R is membrane-bound: nothing
    diffuses it, so `cell_chem_diffuse` must name only c's column.

    b1 = 0 is Li's OPEN LOOP: R relaxes to b0 / kR everywhere and c sees one extra uniform decay,
    k + s b0 / kR -- exactly `source_decay` at that decay (tested), so amplitude and decay length are
    set by the ratio of morphogen to receptor production (Li 2018 Fig. 2D-E). b1 > 0 CLOSES THE LOOP:
    where c is high R rises and clears c faster, so raising p raises the clearance with it and the
    profile moves less (Li 2018 Fig. 3B-C: amplitude and length scale robust to SHH production). In one
    cell with no diffusion the steady c* solves c* (k + s R*(c*)) = p; open loop c* is linear in p,
    closed loop sublinear (tested).

    WHY A MODEL AND NOT A FLAG ON `source_decay`: a second species with its own dynamics is a different
    reaction, and experiments do not edit an operator (INSTRUCTION.md); the source region is inherited
    unchanged.

    Reference: Li, P. et al. (2018). Morphogen gradient reconstitution reveals Hedgehog pathway design
    principles. Science 360:543-548 (Figs. 2D-E open loop, 3B-C feedback).
    """
    N_SPECIES = 2
    MECHANISM_TAGS = CellReactSourceDecay.MECHANISM_TAGS + ["receptor", "sequestration", "feedback", "ptch"]
    PARAM_ROLES = dict(CellReactSourceDecay.PARAM_ROLES, sequester="receptor_removal_rate",
                       r_basal="receptor_basal_synthesis", r_induced="receptor_induced_synthesis",
                       r_half="receptor_induction_half_point", r_hill="receptor_induction_hill",
                       r_decay="receptor_turnover_rate")
    REFERENCE = "Li, P. et al. (2018). Science 360:543-548."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.s = float(params.get("sequester", 0.0))
        self.b0 = float(params.get("r_basal", 0.0)); self.b1 = float(params.get("r_induced", 0.0))
        self.K = float(params.get("r_half", 1.0)); self.n = float(params.get("r_hill", 1.0))
        self.kR = float(params.get("r_decay", 1.0))

    def rates(self, c, R, src):
        cp = c.clamp(min=0.0)
        dc = self.p * src - self.k * c - self.s * R * c
        dR = self.b0 + self.b1 * cp ** self.n / (self.K ** self.n + cp ** self.n) - self.kR * R
        return dc, dR

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        c, R = _span(chem, self.chan, 2, type(self).__name__)
        src = self.source_mask(lvl).to(chem.dtype)
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: _emit(chem, self.chan, self.rates(c, R, src), self.rate, occ)}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="stripe_decay",
                   title="A morphogen made by a central stripe",
                   equation=r"""$$\frac{dc_i}{dt}=r\big(p\,\mathbf 1[\,|x_i-\bar x|<\tfrac f2 L\,]-k\,c_i\big)$$""")
class CellReactStripeDecay(CellReactSourceDecay):
    """`source_decay` with the source a STRIPE THROUGH THE MIDDLE of the tissue instead of a band at one
    edge -- Dpp made along the anterior-posterior compartment boundary of the wing-disc pouch.

        dc_i/dt = r ( p 1[|x_i - xbar| < f L / 2] - k c_i )

    x_i is the cell centroid's coordinate along `source.axis`, xbar the live cells' mean on it and L
    their current extent along it, so the stripe is centred on the tissue and keeps the fraction f of
    its width as it grows; p, k, r and the diffusion that shapes the two flanks are `source_decay`'s.

    WHY A MODEL AND NOT A `side` VALUE of `source_decay`: experiments do not edit an operator, and a
    central source is a different geometry of the same dynamics -- the one every wing-disc paper of exp
    13 draws: LeGoff et al. 2013 Fig. 1A ("the two sources of morphogens, Dpp and Wg, abutting the
    compartment boundaries"), Aegerter-Wilmsen et al. 2012 Fig. 1B (Dpp highest along the AP boundary,
    lowest laterally). A two-sided exponential c ~ exp(-|x - xbar| / lambda) results.

    `source.width` w (world units, off by default) fixes the stripe instead: |x_i - xbar| < w / 2 whatever
    the tissue's extent -- a source that does NOT scale with the pouch, Aegerter-Wilmsen et al. 2007's
    third assumption ("the growth factor ... has a fixed range"), under which a growing pouch outruns its
    growth factor.

    Reference: Aegerter-Wilmsen, T. et al. (2012). Development 139:3221 (Fig. 1B); Crick, F. (1970).
    Nature 225:420-422 (source, diffusion, decay).
    """
    MECHANISM_TAGS = CellReactSourceDecay.MECHANISM_TAGS + ["stripe", "dpp"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        w = (params.get("source") or {}).get("width")
        self.width = None if w is None else float(w)

    def source_mask(self, lvl):
        cen = lvl.get("centroid")
        x = cen[:, self.axis]
        occ = getattr(lvl, "occ", None)
        live = (occ > 0) if occ is not None else torch.ones_like(x, dtype=torch.bool)
        big = torch.finfo(x.dtype).max
        lo = torch.where(live, x, torch.full_like(x, big)).min()
        hi = torch.where(live, x, torch.full_like(x, -big)).max()
        xbar = (x * live.to(x.dtype)).sum() / live.sum().clamp(min=1)
        half = 0.5 * self.width if self.width is not None else 0.5 * self.frac * (hi - lo)
        return ((x - xbar).abs() < half) & live


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="clock_turnover",
                   title="A histone mark turned over by a clock-gated deacetylase that a microbial signal induces",
                   species=(("A", "acetylation"),),
                   equation=r"""$$\frac{dA_i}{dt}=r\Big(k_{in}-k_{out}\big(1+g\,\tfrac{m_i}{K+m_i}\big)\,\gamma(\phi_i)\,A_i\Big),\quad m_i=\sum_s w_s\,x_{i,s}$$""")
class CellReactClockTurnover(Lateral):
    """A HOST cell's histone acetylation A, written on at a constant rate and erased by a deacetylase
    (HDAC3) whose activity follows the cell's clock and is raised by a microbial signal. chem = [A]:

        dA_i/dt = r ( k_in - k_out (1 + g s_i) gamma(phi_i) A_i )
        s_i     = m_i / (K + m_i),      m_i = sum_s w_s x_{i,s}
        gamma   = f + (1 - f) (1 + cos(phi_i + offset)) / 2

    A is the acetylation level, dimensionless (1 = the germ-free steady level at k_in = k_out). k_in is
    the acetylation (writing) rate and k_out the germ-free deacetylation rate, both per unit time; g is
    how many times the germ-free deacetylation the microbial signal adds at saturation; s_i in [0, 1) is
    that signal's saturating read-out, K its half-saturation, in the units of m. m_i is the microbial
    signal the cell sees: the weighted sum of its `signal:` block's columns x_{i,s} (one column per
    strain, written by `readout` across the community-host relation), w_s = `weights:`, one per strain
    -- `[1, 0, 0]` when strain 0 alone makes the metabolite. gamma(phi) is the clock's gate on HDAC3,
    read from the `phase` block `phase_clock` advances, with the `gate:` keys `_gate` gives every
    phase-gated operator: `floor` f in [0, 1] (the gate's minimum, the enzyme's residual activity) and
    `offset` (radians, where on the cycle HDAC3 peaks). r is `rate`, a time rescaling.

    TWO GATES, WHEN THE MICROBES' SHARE OF THE ENZYME KEEPS ITS OWN CLOCK. `gate_signal:` (same keys)
    gates the INDUCED term separately from the basal one:

        dA_i/dt = r ( k_in - k_out ( gamma_0(phi_i) + g s_i gamma_1(phi_i) ) A_i )

    gamma_0 = `gate` (the basal, germ-free enzyme), gamma_1 = `gate_signal` (the part the microbes add).
    Absent, gamma_1 = gamma_0 and the equation is the one above. Kuang's Fig. 2A mechanism is this
    split -- the microbiota make HDAC3's recruitment rhythmic -- and it is what lets a mark whose
    germ-free rhythm is weak (a high basal floor) gain a strong rhythm from the microbes without a
    matching fall in its mean (exp 15, rig 4, Finding 26: H3K27ac).

    WHAT IT REPRODUCES (Kuang et al. 2019, Fig. 1C-D and Fig. 2A). Germ-free (m = 0): deacetylation
    runs at k_out gamma, so A sits high and follows the clock weakly; with a community (m > 0) the
    total turnover k_out (1 + g s) is faster, so A's mean falls by roughly (1 + g s) and its daily
    swing, relative to its mean, grows -- the fast-turnover limit tracks gamma fully, the slow one
    averages it away. That is the direction of Kuang's Fig. 1C (reads higher in germ-free) and
    Fig. 1D (amplitude lower in germ-free). One enzyme and one clock give both; the ratio of the two is
    this model's prediction, not a separate dial.

    WHY A MODEL OF `cell_chem_react` AND NOT A NEW OPERATOR: it is a reaction on a set's `chem`, read
    per cell and integrated by the clock, like `source_decay` beside it; the clock is `phase_clock`,
    the signal arrives by `readout` across a relation, and the gate is `_gate`'s. Nothing but the
    reaction's right-hand side is new. `g: 0` (or an all-zero signal) and `floor: 1` is `source_decay`
    with the source everywhere: dA/dt = k_in - k_out A.

    Reference: Kuang, Z. et al. (2019). The intestinal microbiota programs diurnal rhythms in host
    metabolism through histone deacetylase 3. Science 365:1428-1434 (Fig. 1C-D, Fig. 2A).
    """
    N_SPECIES = 1
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["k_in", "k_out"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem", "phase", "signal"]; WRITES = ["chem"]
    MECHANISM_TAGS = ["reaction", "histone_acetylation", "hdac3", "circadian", "microbial_signal", "host"]
    PARAM_ROLES = {"k_in": "acetylation_rate", "k_out": "germ_free_deacetylation_rate",
                   "g": "signal_induced_deacetylation_at_saturation_x_germ_free", "K": "signal_half_saturation",
                   "weights": "per_strain_signal_weight", "signal": "signal_block", "gate": "clock_gate",
                   "gate_signal": "clock_gate_of_the_induced_term", "rate": "reaction_time_scale"}
    REFERENCE = "Kuang, Z. et al. (2019). Science 365:1428-1434."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.cell_set = self.at                                   # `_gate` reads the clock from this set
        self.k_in = float(params["k_in"]); self.k_out = float(params["k_out"])
        self.g = float(params.get("g", 0.0)); self.K = float(params.get("K", 1.0))
        self.w = params.get("weights")
        self.signal = str(params.get("signal", "signal"))
        self.gate = dict(params.get("gate") or {"block": "phase", "floor": 1.0})
        gs = params.get("gate_signal")
        self.gate_signal = None if gs is None else dict(gs)
        self.rate = float(params.get("rate", 1.0))
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)

    def forward(self, H, mask=None):
        import types
        from plexus.operators.cell_ops import _gate
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        (A,) = _span(chem, self.chan, 1, type(self).__name__)
        if self.g != 0.0:
            if self.signal not in lvl.state_schema:
                raise ValueError(f"clock_turnover: set {self.at!r} has no {self.signal!r} block -- declare it "
                                 f"(integration none) and fill it with `readout ... into: {self.signal}`")
            x = lvl.get(self.signal)
            w = torch.ones(x.shape[1], dtype=x.dtype, device=x.device) if self.w is None else \
                torch.as_tensor([float(v) for v in self.w], dtype=x.dtype, device=x.device)
            if w.shape[0] != x.shape[1]:
                raise ValueError(f"clock_turnover: {w.shape[0]} weights for a {x.shape[1]}-wide {self.signal!r} block")
            m = (x * w).sum(1).clamp(min=0.0)
            s = m / (self.K + m)
        else:
            s = torch.zeros_like(A)
        gam = _gate(self, H, slice(None))
        gam1 = gam if self.gate_signal is None else \
            _gate(types.SimpleNamespace(gate=self.gate_signal, cell_set=self.at), H, slice(None))
        dA = self.k_in - self.k_out * (gam + self.g * s * gam1) * A
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: _emit(chem, self.chan, (dA,), self.rate, occ)}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="notch_delta",
                   title="Notch-Delta lateral inhibition gated by YAP, and the Wnt its winners secrete",
                   species=(("N", "Notch activity"), ("D", "Delta (Dll1)"), ("Y", "nuclear YAP"), ("W", "Wnt")),
                   equation=r"""$$\begin{aligned}\frac{dN_i}{dt}&=r\Big(\frac{\bar D_i^{k}}{a+\bar D_i^{k}}-N_i\Big),\quad \bar D_i=\langle D_j\rangle_{j\sim i}\\ \frac{dD_i}{dt}&=r\,v\Big(\frac{Y_i}{1+bN_i^{h}}-D_i\Big)\\ \frac{dY_i}{dt}&=r\,k_y\Big(\frac{y_{\max}W_i^{n}}{K_w^{n}+W_i^{n}}-Y_i\Big)\ \ \text{after } t_{\rm Wnt}\\ \frac{dW_i}{dt}&=r\Big(p\,\frac{D_i^{q}}{\theta^{q}+D_i^{q}}-k_wW_i\Big)\end{aligned}$$""")
class CellReactNotchDelta(Lateral):
    """Serra et al. 2019's symmetry breaking as a reaction: YAP variability biases Notch-Delta lateral
    inhibition, the winners (DLL1+ -> Paneth) secrete Wnt, and Wnt is what the crypt becomes.
    chem = [N, D, Y, W] from `chan`, four columns:

        dN_i/dt = r ( Dbar_i^k / (a + Dbar_i^k) - N_i )             Dbar_i = mean D over the neighbours
        dD_i/dt = r v ( Y_i / (1 + b N_i^h) - D_i )
        dY_i/dt = r k_y ( y_max W_i^n / (K_w^n + W_i^n) - Y_i )     only from frame `wnt_off` on
        dW_i/dt = r ( p D_i^q / (theta^q + D_i^q) - k_w W_i )

    N is the cell's Notch activity and D its Delta (DLL1), both dimensionless; the first two lines are
    Collier et al. 1996's lateral inhibition (Eq. 2.1-2.2: f(x) = x^k/(a + x^k), g(x) = 1/(1 + b x^h),
    a = 0.01, b = 100, k = h = 2, v = 1), with Delta's production scaled by Y, the cell's nuclear YAP --
    Serra's DLL1+ cells carry ~2.8x the nuclear YAP of DLL1- ones (Fig. 5f), and DLL1 is a YAP target.
    Y does not change while exogenous Wnt is in the medium; from frame `wnt_off` (Serra: Wnt for the
    first three days, Fig. 1a) YAP relaxes at rate k_y to the level the cell's own Wnt sets (half at
    W = K_w) -- YAP falls after Wnt removal (Serra ED Fig. 7c) and an organoid with no Paneth cell by
    then has none to hold it: an enterocyst. A TARGET, not a decay damped by Wnt: a damped decay
    (-k_y (1 - W/(W + K_w)) Y) never reaches zero, so every organoid, Paneth cell or not, leaks to an
    enterocyst given time; the target form has a held state (Wnt-high cells keep YAP near y_max, default 1).

    A YAP THRESHOLD ON DELTA, opt-in (`y_th`, default None: Delta's production is v Y, Collier's scaled by
    YAP): with `y_th` set, the production is v Y^m / (y_th^m + Y^m) (`m`, default 8) -- only a cell whose
    nuclear YAP clears y_th makes enough Delta to win, as Serra's DLL1+ cells are the YAP-high ones (Fig.
    5f, x2.8). Three things follow from it and from YAP being inherited at division: an organoid none of
    whose cells clears y_th picks nobody (an enterocyst, the population's budding fraction); the high-YAP
    cell's clone is a CLUSTER, so its winners are neighbours' neighbours rather than a salt-and-pepper
    field; and after withdrawal only cells whose Wnt holds YAP above y_th (`y_max` sets the level Wnt
    holds) can still win, so the chosen territory stays one patch. Without it (C1e, finding 32) winners
    were scattered over the whole organoid and, dividing into identical pairs that inhibit each other,
    died out.

    EXPRESSION NOISE ON DELTA, opt-in (`sigma_d`, default 0): the production is multiplied by
    max(0, 1 + sigma_d xi), xi standard normal per cell per call from the operator's own `seed`. A division
    gives two IDENTICAL daughters that are neighbours -- the lateral-inhibition pair's unstable symmetric
    state, which a deterministic model leaves to roundoff while their Wnt and YAP decay (C1e: a winner that
    divided lost both daughters). Gene expression noise is what breaks such a tie in a cell. W is the Wnt a DLL1-high cell secretes (Hill in D, half at
    theta), lost at k_w; its SPREAD is `cell_chem_diffuse` on the same column, and the crypt region is
    where it lands -- `cell_mechanics[apicobasal_region]` reads it as the fate (Serra ED Fig. 10h: the
    canonical Wnt response in the Paneth cell's neighbours).

    THE NEIGHBOURS ARE THE MESH'S, from `cell_neighbours` (`edge_index`), read the way the graph
    Laplacian reads them: juxtacrine, contact only. With no edge a cell sees Dbar = 0. The delta is zero
    outside the four columns (`_emit`), so it adds to `cell_chem_diffuse` on W rather than overwriting.

    IDENTITY: with every Y equal and no initial difference in N and D, nothing breaks the symmetry and
    no cell wins -- the control of the experiment (tested).

    Reference: Collier, J. R., Monk, N. A. M., Maini, P. K. & Lewis, J. H. (1996). Pattern formation by
    lateral inhibition with feedback. J. Theor. Biol. 183:429-446; Serra, D. et al. (2019). Nature
    569:66-72 (YAP variability -> Notch/DLL1 -> the first Paneth cell).
    """
    N_SPECIES = 4
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem", "edge_index"]; WRITES = ["chem"]
    MECHANISM_TAGS = ["reaction", "lateral_inhibition", "notch_delta", "juxtacrine", "symmetry_breaking",
                      "wnt_source"]
    PARAM_ROLES = {"a": "notch_activation_threshold", "b": "delta_repression_strength", "k": "notch_hill",
                   "h": "delta_hill", "v": "delta_rate_over_notch_rate", "p": "wnt_production",
                   "theta": "delta_level_for_wnt", "q": "wnt_hill", "k_w": "wnt_decay",
                   "k_y": "yap_relaxation_after_wnt_withdrawal", "K_w": "wnt_that_holds_yap_half", "n": "yap_wnt_hill",
                   "y_th": "yap_threshold_for_delta", "m": "yap_delta_hill", "y_max": "yap_level_wnt_holds",
                   "sigma_d": "delta_expression_noise", "wnt_off": "frame_exogenous_wnt_ends",
                   "rate": "reaction_time_scale"}
    REFERENCE = ("Collier, J. R. et al. (1996). J. Theor. Biol. 183:429-446; "
                 "Serra, D. et al. (2019). Nature 569:66-72.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.a = float(params.get("a", 0.01)); self.b = float(params.get("b", 100.0))
        self.k = float(params.get("k", 2.0)); self.h = float(params.get("h", 2.0))
        self.v = float(params.get("v", 1.0))
        self.p = float(params.get("p", 0.0)); self.theta = float(params.get("theta", 0.5))
        self.q = float(params.get("q", 4.0)); self.k_w = float(params.get("k_w", 1.0))
        self.k_y = float(params.get("k_y", 0.0)); self.K_w = float(params.get("K_w", 0.1))
        self.n = float(params.get("n", 2.0))
        _yt = params.get("y_th", None)
        self.y_th = None if _yt is None else float(_yt)
        self.m = float(params.get("m", 8.0))
        self.y_max = float(params.get("y_max", 1.0))
        self.sigma_d = float(params.get("sigma_d", 0.0))
        self._gen = torch.Generator(device="cpu"); self._gen.manual_seed(int(params.get("seed", 0)))
        _wo = params.get("wnt_off", None)
        self.wnt_off = None if _wo is None else float(_wo)
        self.rate = float(params.get("rate", 1.0))
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        N, D, Y, W = _span(chem, self.chan, 4, type(self).__name__)
        ei = getattr(lvl, "edge_index", None)
        if ei is None or ei.numel() == 0:
            Dbar = torch.zeros_like(D)
        else:
            i, j = ei[0], ei[1]
            agg = torch.zeros_like(D).index_add_(0, i, D[j])
            deg = torch.zeros_like(D).index_add_(0, i, torch.ones_like(D[j]))
            Dbar = agg / deg.clamp(min=1)
        Dp = Dbar.clamp(min=0) ** self.k
        dN = Dp / (self.a + Dp) - N
        if self.y_th is None:
            prod = Y
        else:
            Ym = Y.clamp(min=0) ** self.m
            prod = Ym / (self.y_th ** self.m + Ym)
        if self.sigma_d > 0:
            xi = torch.randn(prod.shape[0], generator=self._gen, dtype=torch.float64)
            prod = prod * (1.0 + self.sigma_d * xi).clamp(min=0).to(device=prod.device, dtype=prod.dtype)
        dD = self.v * (prod / (1.0 + self.b * N.clamp(min=0) ** self.h) - D)
        fr = getattr(H, "frame", None)
        if self.wnt_off is not None and self.k_y > 0 and fr is not None and float(fr) >= self.wnt_off:
            Wn = W.clamp(min=0) ** self.n
            dY = self.k_y * (self.y_max * Wn / (self.K_w ** self.n + Wn) - Y)
        else:
            dY = torch.zeros_like(Y)
        Dq = D.clamp(min=0) ** self.q
        dW = self.p * Dq / (self.theta ** self.q + Dq) - self.k_w * W
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: _emit(chem, self.chan, (dN, dD, dY, dW), self.rate, occ)}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="balaskas",
                   title="The Pax6-Olig2-Nkx2.2 circuit reading a Shh-Gli input",
                   species=(("P", "Pax6"), ("O", "Olig2"), ("N", "Nkx2.2")),
                   equation=r"""$$\begin{aligned}\frac{dP}{dt}&=r\Big(\frac{\alpha}{1+(N/N_{cP})^{h_1}+(O/O_{cP})^{h_2}}-k_1P\Big)\\ \frac{dO}{dt}&=r\Big(\frac{\beta\,G^{n}}{1+G^{n}}\,\frac{1}{1+(N/N_{cO})^{h_3}}-k_2O\Big)\\ \frac{dN}{dt}&=r\Big(\frac{\gamma\,G^{m}}{1+G^{m}}\,\frac{1}{1+(O/O_{cN})^{h_4}+(P/P_{cN})^{h_5}}-k_3N\Big)\end{aligned}$$""")
class CellReactBalaskas(Lateral):
    """The three-gene cross-repressive circuit that reads the Shh gradient in the neural tube into three
    domains -- Pax6 (P, no signal), Olig2 (O, pMN), Nkx2.2 (N, p3) -- Balaskas et al. 2012, eqs. 1-3.
    chem[chan : chan + 3] = [P, O, N]:

        dP/dt = r ( alpha / (1 + (N/NcritP)^h1 + (O/OcritP)^h2) - k1 P )
        dO/dt = r ( beta G^n/(1 + G^n) / (1 + (N/NcritO)^h3) - k2 O )
        dN/dt = r ( gamma G^m/(1 + G^m) / (1 + (O/OcritN)^h4 + (P/PcritN)^h5) - k3 N )

    P, O, N are the three factors' levels, dimensionless (the paper calls a factor HIGH above 1). G is
    the Shh-Gli input, G = `g_gain` x chem[`g_col`], the morphogen column this model READS and never
    writes -- so it stays additive with the source model that owns that column. alpha, beta, gamma are
    the maximal synthesis rates; h1..h5 the Hill coefficients of the five repressions (N -| P, O -| P,
    N -| O, O -| N, P -| N); the *crit are the levels at which each repression is half-maximal; k1..k3
    the degradation rates; n, m the cooperativity of G on O and on N. r is `rate`, the circuit's time
    scale against the frame clock (the paper's time unit is 1 / k).

    EVERY DEFAULT IS TABLE S2 of the paper's supplement (mmc1): alpha 3, beta 5, gamma 5, h1 6, h2 2,
    h3 5, h4 1, h5 1, k1 = k2 = k3 = 1, every crit 1, n = m = 1 (and P starts at 3: `seed_cell_chem`
    model `uniform`). Integrated to t = 20 from (3, 0, 0), the fate (the highest factor) switches
    P -> O at G = 0.35 and O -> N at G = 2.15; with alpha = 0 (Pax6-/-) N takes over at G = 0.85, with
    beta = 0 (Olig2-/-) at G = 1.3 -- the paper's Fig. 4B reads 0.3, 2.2, 0.8, 1.33
    (tests/test_cell_chem_balaskas.py).

    THE MUTANTS ARE ONE LINE EACH, as in the paper's own simulations (Fig. 4B): Pax6-/- is `alpha: 0`,
    Olig2-/- is `beta: 0`. Nothing else is retuned.

    Reference: Balaskas, N. et al. (2012). Gene regulatory logic for reading the Sonic Hedgehog
    signaling gradient in the vertebrate neural tube. Cell 148:273-284.
    """
    N_SPECIES = 3
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["g_col"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]
    MECHANISM_TAGS = ["gene_regulatory_network", "cross_repression", "morphogen_readout", "balaskas"]
    PARAM_ROLES = {"alpha": "pax6_max_rate", "beta": "olig2_max_rate", "gamma": "nkx22_max_rate",
                   "h1": "N_represses_P", "h2": "O_represses_P", "h3": "N_represses_O",
                   "h4": "O_represses_N", "h5": "P_represses_N", "g_col": "signal_column",
                   "g_gain": "signal_scale", "rate": "reaction_time_scale"}
    REFERENCE = "Balaskas, N. et al. (2012). Cell 148:273-284 (eqs. 1-3, Table S2)."
    DEFAULTS = dict(alpha=3.0, beta=5.0, gamma=5.0, h1=6.0, h2=2.0, h3=5.0, h4=1.0, h5=1.0,
                    k1=1.0, k2=1.0, k3=1.0, NcritP=1.0, OcritP=1.0, NcritO=1.0, OcritN=1.0, PcritN=1.0,
                    n=1.0, m=1.0)

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        for k, v in self.DEFAULTS.items():
            setattr(self, k, float(params.get(k, v)))
        self.g_col = int(params["g_col"]); self.g_gain = float(params.get("g_gain", 1.0))
        self.rate = float(params.get("rate", 1.0))
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)
        if self.chan <= self.g_col < self.chan + 3:
            raise ValueError(f"balaskas: g_col={self.g_col} lies inside this model's own span "
                             f"{self.chan}..{self.chan + 2} -- the signal is read from another column")

    def rates(self, P, O, N, G):
        """(dP, dO, dN) before `rate`, on tensors; the levels are clipped at 0 before any power."""
        P, O, N, G = P.clamp(min=0.0), O.clamp(min=0.0), N.clamp(min=0.0), G.clamp(min=0.0)
        Gn = G ** self.n / (1.0 + G ** self.n)
        Gm = G ** self.m / (1.0 + G ** self.m)
        dP = self.alpha / (1.0 + (N / self.NcritP) ** self.h1 + (O / self.OcritP) ** self.h2) - self.k1 * P
        dO = self.beta * Gn / (1.0 + (N / self.NcritO) ** self.h3) - self.k2 * O
        dN = self.gamma * Gm / (1.0 + (O / self.OcritN) ** self.h4 + (P / self.PcritN) ** self.h5) - self.k3 * N
        return dP, dO, dN

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        P, O, N = _span(chem, self.chan, 3, type(self).__name__)
        if self.g_col >= chem.shape[1]:
            raise ValueError(f"balaskas: g_col={self.g_col} but `chem` is {chem.shape[1]} wide")
        G = self.g_gain * chem[:, self.g_col]
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: _emit(chem, self.chan, self.rates(P, O, N, G), self.rate, occ)}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="aliev_panfilov",
                   title="Excitable reaction: a threshold, an action potential, a refractory tail",
                   species=(("u", "excitation"), ("v", "recovery")),
                   equation=r"""$$\frac{du}{dt}=r\big(-k\,u(u-a)(u-1)-u\,v+I_{\mathrm{stim}}\big),\qquad \frac{dv}{dt}=r\,\varepsilon(u,v)\big(-v-k\,u(u-a-1)\big),\qquad \varepsilon=\varepsilon_0+\frac{\mu_1 v}{u+\mu_2}$$""")
class CellReactAlievPanfilov(Lateral):
    """Aliev-Panfilov excitable kinetics: the smallest model with a threshold, a full-size action
    potential and a refractory tail whose length depends on the pacing -- which none of the five
    pattern-forming models here has, and which a conducted wave needs. chem = [u, v]:

        du/dt = r ( -k u (u - a)(u - 1) - u v + I_stim )
        dv/dt = r eps(u, v) ( -v - k u (u - a - 1) ),      eps = eps0 + mu1 v / (u + mu2)

    u is the excitation (the transmembrane potential, E = 100 u - 80 mV), v the recovery variable,
    both dimensionless; one unit of model time is 12.9 ms (Aliev & Panfilov 1996, eq. 2, scaled so
    the free pulse's APD90 is 330 ms). k = 8 sets the upstroke, a = 0.15 is the threshold as a
    fraction of the pulse, eps0 = 0.002 the slow recovery, and mu1 = 0.2, mu2 = 0.3 were fitted by the
    paper to the canine restitution curve (1/apd = 1.016 + 1.059/cl, Fig. 3). Every default is the
    paper's (eq. 1 and p. 296). The `u v` term, where FitzHugh-Nagumo has `v`, keeps u from going
    below rest; the quadratic v-nullcline and the u,v-dependent eps give the restitution.

    THE COUPLING IS NOT HERE. The paper's d_ij Laplacian term is `cell_chem_diffuse` along
    `cell_neighbours`, with `d: [D, 0]` -- only u spreads, as in eq. 1 -- which is a gap junction's
    current law: current through a shared edge proportional to the difference. This operator is the
    membrane; the junction is the graph.

    THE STIMULUS IS A MEMBRANE CURRENT OF THIS MODEL, NOT AN OPERATOR. FitzHugh's z (1961, eq. 1) and
    the paper's pacing are a current injected into the u equation, so it lives in the equation:

        stim: {times: [t1, t2, ...], duration: d, amp: A, center: [x, y, z], radius: R}
        stim: {times: [...], duration: d, amp: A, below: {axis: 0, value: x0}}
        stim: {times: [...], duration: d, amp: A, box: [[x_lo, y_lo], [x_hi, y_hi]]}
        stim: [{...}, {...}]      several stimuli, each with its own site and times (S1 here, S2 there)

    I_stim = A on the cells whose centroid lies inside the disc (or below x0 along the axis, a strip
    that launches a planar front; or inside an axis-aligned box, the S2 stripe that breaks a wave
    into a vortex, Aliev & Panfilov Fig 4) while the model time t = frame x dt is inside any
    [t_k, t_k + d). The position is the set's `centroid` block, or its `pos` when it has no
    centroid -- a cell set seeded from a segmentation carries `pos`, the label's pixel centroid.
    The existing sources were audited first: `pacemaker` publishes a periodic clock and cannot give
    two stimuli at an arbitrary S1-S2 interval, `activation_pulse` paints a grid field, and
    `phase_clock` is a phase -- none of them reaches a cell-set species. The time is read from
    `H.frame_t`, the device tensor, so a captured CUDA graph does not replay frame 1's stimulus.

    Reference: Aliev, R. R. & Panfilov, A. V. (1996). A simple two-variable model of cardiac
    excitation. Chaos Solitons Fractals 7:293-301.
    """

    N_SPECIES = 2
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem", "centroid"]; WRITES = ["chem"]
    MECHANISM_TAGS = ["reaction", "excitable", "refractory", "action_potential", "aliev_panfilov"]
    PARAM_ROLES = {"k": "upstroke_rate", "a": "threshold", "eps0": "recovery_rate",
                   "mu1": "restitution_1", "mu2": "restitution_2", "rate": "reaction_time_scale",
                   "stim": "stimulus_current"}
    REFERENCE = "Aliev, R. R. & Panfilov, A. V. (1996). Chaos Solitons Fractals 7:293-301."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.k = float(params.get("k", 8.0)); self.a = float(params.get("a", 0.15))
        self.eps0 = float(params.get("eps0", 0.002))
        self.mu1 = float(params.get("mu1", 0.2)); self.mu2 = float(params.get("mu2", 0.3))
        self.rate = float(params.get("rate", 1.0))
        self.chan = _chan(params, type(self).__name__, self.N_SPECIES)
        s = params.get("stim") or {}
        self.stims = [self._site(x) for x in (s if isinstance(s, (list, tuple)) else [s])]
        first = self.stims[0]
        self.stim_times, self.stim_dur, self.stim_amp = first["times"], first["dur"], first["amp"]

    @staticmethod
    def _site(s):
        st = dict(times=[float(t) for t in (s.get("times") or [])], dur=float(s.get("duration", 0.5)),
                  amp=float(s.get("amp", 0.0)), center=s.get("center"), radius=float(s.get("radius", 0.0)),
                  below=s.get("below"), box=s.get("box"))
        if (st["times"] and st["amp"] and st["center"] is None and st["below"] is None
                and st["box"] is None):
            raise ValueError("aliev_panfilov: `stim` needs a region -- `center` + `radius`, "
                             "`below: {axis, value}` or `box`. A stimulus on every cell is not a stimulus site.")
        return st

    def _stim(self, H, lvl, like):
        out = torch.zeros_like(like)
        if not any(s["times"] and s["amp"] for s in self.stims):
            return out
        cen = lvl.get("centroid" if "centroid" in lvl.state_schema else "pos")
        ft = getattr(H, "frame_t", None)
        t = (ft if ft is not None else torch.tensor(float(getattr(H, "frame", 0)))).to(like) * float(H.dt)
        for s in self.stims:
            if not (s["times"] and s["amp"]):
                continue
            if s["below"] is not None:
                where = cen[:, int(s["below"].get("axis", 0))] < float(s["below"]["value"])
            elif s["box"] is not None:
                lo = torch.as_tensor([float(x) for x in s["box"][0]], dtype=cen.dtype, device=cen.device)
                hi = torch.as_tensor([float(x) for x in s["box"][1]], dtype=cen.dtype, device=cen.device)
                k = len(lo)
                where = ((cen[:, :k] >= lo) & (cen[:, :k] <= hi)).all(1)
            else:
                c = torch.as_tensor([float(x) for x in s["center"]], dtype=cen.dtype, device=cen.device)
                where = (cen[:, :len(c)] - c).norm(dim=1) <= s["radius"]
            t0 = torch.as_tensor(s["times"], dtype=like.dtype, device=like.device)
            on = ((t >= t0) & (t < t0 + s["dur"])).any().to(like.dtype)
            out = out + s["amp"] * on * where.to(like.dtype)
        return out

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        u, v = _span(chem, self.chan, 2, type(self).__name__)
        eps = self.eps0 + self.mu1 * v / (u + self.mu2)
        du = -self.k * u * (u - self.a) * (u - 1.0) - u * v + self._stim(H, lvl, u)
        dv = eps * (-v - self.k * u * (u - self.a - 1.0))
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: _emit(chem, self.chan, (du, dv), self.rate, occ)}


# `cell_grow`, AND THE OLD NAME IS GONE RATHER THAN ALIASED. An alias makes two names for one
# thing and leaves a reader unable to tell which a specification meant; prior specifications are
# migrated instead.
#
# It is growth. It does not produce a morphogen, it READS one and uses it as a per-cell rate: the
# morphogen is a GATE on this operator, and the composition space already declares that gate as an
# optional slot. With the gate open (`a_sw = 0`) the same operator is plain uniform growth. Naming
# the gate in the operator made the optional half look mandatory, and made the sibling pair
# unreadable -- `cell_grow` / `cell_divide` says what the schedule actually does.
@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="balaskas_schedule",
                   title="The Pax6-Olig2-Nkx2.2 circuit with a scheduled change of its Shh-Gli input",
                   species=(("P", "Pax6"), ("O", "Olig2"), ("N", "Nkx2.2")),
                   equation=r"""$$G(\mathbf x,t)=g\,c(\mathbf x,t)\,s(t),\qquad s(t)=s_k\ \ \text{for}\ t_k\le t<t_{k+1}$$""")
class CellReactBalaskasSchedule(CellReactBalaskas):
    """`balaskas` whose Gli input is multiplied by a piecewise-constant factor in circuit time -- a drug
    added to the dish at a declared time: Dessaud et al. 2007 (Fig. 2c-e) add cyclopamine to neural
    plate explants that carry their own floor plate after 12 h, halve GLI activity (9.2 -> 4.1, Fig. 2c)
    and find NKX2.2 at 7 % of the untreated explants' at 18 h while OLIG2 rises to 157 % (Fig. 2e).

        G(x, t) = g_gain c(x, t) s(t),   `schedule: [[t_1, s_1], [t_2, s_2], ...]`, s = 1 before t_1

    t is the circuit's own time, `rate` x frame x dt. Empty schedule = `balaskas`, term for term
    (tested). Everything else, parameters and mutants, is `balaskas`; the class is a subclass.

    Reference: Dessaud, E. et al. (2007). Nature 450:717-720 (Fig. 2c-e); Balaskas, N. et al. (2012).
    Cell 148:273-284.
    """
    PARAM_ROLES = dict(CellReactBalaskas.PARAM_ROLES, schedule="input_gain_schedule")
    REFERENCE = ("Dessaud, E. et al. (2007). Nature 450:717-720; Balaskas, N. et al. (2012). "
                 "Cell 148:273-284.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        sch = sorted([(float(t), float(f)) for t, f in (params.get("schedule") or [])])
        self.sched_t = [t for t, _ in sch]
        self.sched_f = [f for _, f in sch]

    def factor(self, t):
        """s(t) on a 0-d tensor: the factor of the last step at or before t, 1 before the first."""
        s = torch.ones_like(t)
        for tk, fk in zip(self.sched_t, self.sched_f):
            s = torch.where(t >= tk, torch.full_like(t, fk), s)
        return s

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        P, O, N = _span(chem, self.chan, 3, type(self).__name__)
        if self.g_col >= chem.shape[1]:
            raise ValueError(f"balaskas_schedule: g_col={self.g_col} but `chem` is {chem.shape[1]} wide")
        ft = getattr(H, "frame_t", None)
        frame = (ft if ft is not None else torch.tensor(float(getattr(H, "frame", 0)))).to(chem)
        G = self.g_gain * chem[:, self.g_col] * self.factor(self.rate * frame * float(H.dt))
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: _emit(chem, self.chan, self.rates(P, O, N, G), self.rate, occ)}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="balaskas_adapt",
                   title="The Pax6-Olig2-Nkx2.2 circuit reading an ADAPTING Shh-Gli input",
                   species=(("P", "Pax6"), ("O", "Olig2"), ("N", "Nkx2.2")),
                   equation=r"""$$G(\mathbf x,t)=g\,c(\mathbf x,t)\,\Big(\frac{t}{t_p}\Big)^{2}e^{2(1-t/t_p)},\qquad t=r\,n\,\Delta t$$""")
class CellReactBalaskasAdapt(CellReactBalaskas):
    """`balaskas` with the paper's OWN time course of Gli activity: the intracellular input rises, peaks
    and retracts while the morphogen itself does not (Balaskas et al. 2012, Fig. 7A; Fig. S7A):

        G(x, t) = g_gain c(x, t) f(t),   f(t) = (t / t_p)^2 e^(2 (1 - t / t_p))

    f is the paper's G(t) = a t^2 e^(-b t) normalised to 1 at its peak, t_p = 2 / b; b = 0.16 at both
    the p3 and the pMN position in Fig. S7A, so t_p = 12.5 circuit units (`t_peak`). The position
    enters through c, the morphogen column -- the amplitude a of the paper's profiles. t is the
    circuit's own time, `rate` x frame x dt, from the start of the run. Everything else, parameters
    and mutants, is `balaskas` (this is a subclass; `balaskas` is untouched).

    WHY THE ADAPTATION IS HERE AND NOT IN THE SOURCE. In vivo the Shh PROTEIN gradient keeps rising
    while Gli activity (the Tg(GBS-GFP) reporter, Ptch1) peaks at 16-30 hours post headfold and
    declines (Balaskas Fig. 1; Chamberlain 2008): the decline is the cells' desensitisation (Dessaud
    et al. 2007), the transduction, so the morphogen column keeps its source and f scales the input.

    Tested (tests/test_cell_chem_balaskas.py): one cell under the three Fig. S7A profiles ends, at t =
    20, Nkx2.2-high / Olig2-high with low Pax6 / Pax6-high, as the paper's panels iv / iii / ii; the
    same profile continued to t = 60 returns the p3 cell to Pax6 (the circuit's hysteresis holds only
    while G stays above Nkx2.2's maintenance level) -- a prediction beyond the paper's t <= 20.

    Reference: Balaskas, N. et al. (2012). Cell 148:273-284 (Figs. 7A, S7A); Dessaud, E. et al.
    (2007). Nature 450:717-720.
    """
    PARAM_ROLES = dict(CellReactBalaskas.PARAM_ROLES, t_peak="signal_peak_time")
    REFERENCE = ("Balaskas, N. et al. (2012). Cell 148:273-284 (eqs. 1-3, Table S2, Fig. S7A); "
                 "Dessaud, E. et al. (2007). Nature 450:717-720.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.t_peak = float(params.get("t_peak", 12.5))
        if self.t_peak <= 0:
            raise ValueError("balaskas_adapt: t_peak must be > 0 (circuit time units)")

    def profile(self, t):
        """f(t) = (t / t_p)^2 e^(2 (1 - t / t_p)): 1 at t = t_p, 0 at t = 0; a tensor or a float."""
        x = t / self.t_peak
        return x * x * torch.exp(2.0 * (1.0 - x)) if torch.is_tensor(x) else x * x * float(np.exp(2.0 * (1.0 - x)))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        P, O, N = _span(chem, self.chan, 3, type(self).__name__)
        if self.g_col >= chem.shape[1]:
            raise ValueError(f"balaskas_adapt: g_col={self.g_col} but `chem` is {chem.shape[1]} wide")
        # THE DEVICE CLOCK, as aliev_panfilov reads it, so a captured CUDA graph does not replay frame 1.
        ft = getattr(H, "frame_t", None)
        frame = (ft if ft is not None else torch.tensor(float(getattr(H, "frame", 0)))).to(chem)
        G = self.g_gain * chem[:, self.g_col] * self.profile(self.rate * frame * float(H.dt))
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: _emit(chem, self.chan, self.rates(P, O, N, G), self.rate, occ)}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="balaskas_commit",
                   title="The Pax6-Olig2-Nkx2.2 circuit reading an adapting input, then committed",
                   species=(("P", "Pax6"), ("O", "Olig2"), ("N", "Nkx2.2")),
                   equation=r"""$$\frac{d(P,O,N)}{dt}=\mathbb{1}[t<t_c]\;\big(\text{balaskas\_adapt}\big)$$""")
class CellReactBalaskasCommit(CellReactBalaskasAdapt):
    """`balaskas_adapt` until the circuit time `t_commit`, then NOTHING: every cell keeps its P, O, N
    (inherited on division) -- the second phase of neural-tube patterning, in which progenitor
    identities are fixed (Kicheva et al. 2014: Olig2 respecification 0.034 -> 0.004 per hour after
    ~40 hours post headfold, Fig. 5C; 92 % of clones of one type).

        d(P, O, N)/dt = 1[t < t_commit] x (the balaskas_adapt rates)

    t_commit = 20 circuit units by default: Balaskas et al. 2012's own simulations end at t = 20, about
    39 hours post headfold by its reporter's timing -- Kicheva's phase boundary. The late phase's
    domain-specific differentiation is NOT here; after the commitment the boundaries move only with
    the tissue's own growth.

    WHY A COMMITMENT AND NOT A SIGNAL FLOOR. With the paper's own adapting input the circuit forms the
    domains and then loses them (exp 7 finding 33): its hysteresis needs G above Nkx2.2's maintenance
    level, and in vivo the Gli reporter is gone by 55 hours post headfold while the domains persist.
    A floor on G would be a number no figure gives; the commitment is what Kicheva measures.

    Reference: Balaskas, N. et al. (2012). Cell 148:273-284; Kicheva, A. et al. (2014). Science
    345:1254927 (the two-phase model).
    """
    PARAM_ROLES = dict(CellReactBalaskasAdapt.PARAM_ROLES, t_commit="commitment_time")
    REFERENCE = ("Balaskas, N. et al. (2012). Cell 148:273-284; Kicheva, A. et al. (2014). "
                 "Science 345:1254927 (identities fixed after ~40 hph).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.t_commit = float(params.get("t_commit", 20.0))

    def forward(self, H, mask=None):
        out = super().forward(H, mask)
        ft = getattr(H, "frame_t", None)
        frame = ft if ft is not None else torch.tensor(float(getattr(H, "frame", 0)))
        live = (self.rate * frame * float(H.dt) < self.t_commit)
        return {k: v * live.to(v) for k, v in out.items()}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="balaskas_commit_schedule",
                   title="The two-phase Pax6-Olig2-Nkx2.2 circuit with a scheduled change of its input",
                   species=(("P", "Pax6"), ("O", "Olig2"), ("N", "Nkx2.2")),
                   equation=r"""$$G=g\,c\,f(t)\,s(t),\qquad \frac{d(P,O,N)}{dt}=\mathbb{1}[t<t_c]\,(\dots)$$""")
class CellReactBalaskasCommitSchedule(CellReactBalaskasCommit):
    """`balaskas_commit` -- the adapting input (Fig. S7A) and the commitment -- with the drug of
    `balaskas_schedule` applied on top: G = g_gain c f(t) s(t). Dessaud et al. 2007's explants ADAPT
    (GLI activity decays over the culture, Fig. 2a) before cyclopamine halves what is left at 12 h
    (Fig. 2c-e), so the in-silico explant of the model of record is this one, not the constant-input
    `balaskas_schedule`. Empty schedule = `balaskas_commit` (tested).

    Reference: Dessaud, E. et al. (2007). Nature 450:717-720; Balaskas, N. et al. (2012). Cell 148:273.
    """
    PARAM_ROLES = dict(CellReactBalaskasCommit.PARAM_ROLES, schedule="input_gain_schedule")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        sch = sorted([(float(t), float(f)) for t, f in (params.get("schedule") or [])])
        self.sched_t = [t for t, _ in sch]
        self.sched_f = [f for _, f in sch]

    factor = CellReactBalaskasSchedule.factor

    def profile(self, t):
        base = CellReactBalaskasAdapt.profile(self, t)
        return base * self.factor(t) if torch.is_tensor(t) else base * float(self.factor(torch.tensor(float(t))))


@register_operator("cell_grow", set="vertex", kind="lateral", family="population", title="Growth where the morphogen is high",
                   equation=r"""$$\frac{ds_j}{dt}=s_j\,\text{rate}\big(\rho+\mathrm{Hill}(a_j)\big),\qquad \mathrm{Hill}(a)=\frac{a^{n}}{a^{n}+a_{sw}^{n}}$$""")
class Grow3D(Lateral):
    """Chemistry-to-shape: the morphogen decides where the tissue grows. Each cell's mechanical
    TARGETS are raised, and the mechanics then inflates the cell by force balance.

    cell -> vertex: reads the cell set's chem, writes the mesh's target area A0, perimeter P0,
    volume V0f and radius R0. It moves no vertex itself -- it raises what the cells ASK for, and
    `cell_mechanics` decides whether they get it.

        ds_j/dt = s_j rate ( rho + Hill(a_j) )
        Hill(a) = a^n / (a^n + a_sw^n)
        s_j                                the cumulative per-cell growth scale, integrated by the engine

    rate is the growth rate in inverse units of simulation time -- the fraction of itself a cell
    adds per unit time, so a spec's own `dt` sets how much of it lands in one frame. rho is the dimensionless baseline: the fraction of
    the full rate a cell grows at with no activator present. a_sw is the activator concentration at
    which the switch is half open, and n is `hill`, its dimensionless sharpness. The two regimes
    are the ends of one operator:

        rho = 1, a_sw = 0   uniform growth, every cell at the same rate
        rho = 0, a_sw > 0   growth only where the activator is high: self-organised budding
        in between          a baseline everywhere plus an activator-driven excess at the tips

    `a_sw_rel` decides whether a_sw is an ABSOLUTE activator concentration or a fraction of the
    field's own running maximum. Both are real mechanisms -- an absolute threshold is a receptor
    with a fixed affinity, a relative one is a cell comparing itself with its neighbours -- and the
    default is absolute. `a_live` is the floor below which the field counts as dead and the
    relative gate refuses to open, so a collapsed activator cannot have its own noise rescaled into
    a threshold crossing.

    Reference: Okuda, S. et al. (2018). Combining Turing and 3D vertex models reproduces
    autonomous multicellular morphogenesis of the tissue. Sci. Rep. 8:2386.
    """

    # Reads one species' activator; the span it points into is two wide because a Gray-Scott
    # system is. It never writes chem, so this only has to name the right column.
    N_SPECIES = 2
    # MAY_MUTATE_INTEGRATED_STATE IS STILL TRUE, AND GROWTH IS NO LONGER WHY. The four targets --
    # `mg_scale`, `A0`, `P0`, `V0f` -- are returned as deltas now and the engine integrates them,
    # so the dynamics this operator exists for writes nothing. Three things it does BESIDE the
    # dynamics still land in place on the cell set, and each is a different kind of write:
    #
    #   the baseline re-take   `A0_init` / `P0_init` / `V0f_init` are re-seeded whenever `nF` moves.
    #                          That is a RESET, not a rate, and it has no delta form. (The policy of
    #                          re-taking the whole population's baseline because some cell divided is
    #                          wrong and is its own parked rung -- see the note at the test below.)
    #   `conserve_amount`      c_j <- c_j * (v_old/v_new) when the cell's target volume grows: a
    #                          change of VARIABLE forced by a volume change, not a dynamics delta.
    #                          Returning it as an integrated delta would change the physics.
    #   `inhib_frac`           a readout of the inhibitor gate, recorded so the renderer can show it.
    #
    # None of the three is reached on every composition -- the rescale needs chemistry present, the
    # inhibitor needs `inhib_chan` set -- so a declaration of False would look correct on most runs
    # and fail on the ones that matter.
    # THE EXPONENT OF VOLUME IN THE LINEAR SCALE s: 3 = isotropic growth, every model here; the
    # `timer_planar` model sets 2 (see it). A class constant because it names the variant.
    GROWTH_DIMS = 3
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    # `rate` IS PER UNIT TIME. Applied once per call it would mean "fraction of itself a cell adds
    # per FRAME" and the spec's own `dt` would never enter; the engine integrates `s += dt * ds`
    # instead. Declaring `1/T` is therefore a statement the code
    # honours, and it is the one this whole vocabulary exists to be able to make.
    #
    # `a_sw` IS DELIBERATELY ABSENT. It is an absolute activator concentration when `a_sw_rel` is
    # false and a fraction of the field's own running maximum when it is true -- one parameter with
    # two dimensions, selected by another parameter. Either declaration would be wrong half the
    # time, and UNDECLARED is silent, which is the honest state until the two are separated.
    PARAM_UNITS = {"rate": "rate", "rho": "fraction", "hill": "fraction", "cap": "fraction",
                   "vth_frac": "fraction", "a_live": "fraction", "inhib_sw": "fraction",
                   "inhib_hill": "fraction", "size_gain": "fraction", "f_max": "fraction",
                   "k_syn": "fraction", "k_deg": "fraction", "cycle_frames": "time"}
    # `V0f` IS UNDECLARED FOR THE SAME REASON `cell_die`'s IS: growth SCALES the seed's target and
    # never chooses its convention. `vth_frac` is a multiple of whichever `v_ref` matches it, which
    # `cell_size` now supplies. The pair the checker should compare is the SEED against the ENERGY,
    # and those two both declare.
    BLOCK_UNITS = {"A0": "area[midsurface]", "P0": "length"}
    MECHANISM_TAGS = ["growth", "morphogen_driven", "budding", "cross_scale"]
    REFERENCE = "Okuda, S. et al. (2018). Combining Turing and 3D vertex models reproduces autonomous multicellular morphogenesis of the tissue. Sci. Rep. 8:2386."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex"); self._cat = params.get("cell_set")
        self.rate = float(params.get("rate", 0.01)); self.a_sw = float(params.get("a_sw", 0.20))
        # `rate_decay_T` (exp 11, 2026-09-27; default 0 = none, unchanged): the growth rate falls as rate / (1 + t / T),
        # t the simulated time. Exponential growth -- the law below at a fixed rate -- multiplied a gland's area x20 in
        # 53 h and ran it into the world's walls; Wang et al. 2021 Fig 1H measured the surface outline growing LINEARLY in
        # time in all 13 glands (r^2 0.80-0.99, 10-34 h), i.e. area A0 (1 + rho t)^2 with a relative rate 2 rho / (1 + rho t):
        # T = 1 / rho. Every model of cell_grow that calls `_law` or reads `self._decay` inherits it.
        self.rate_decay_T = float(params.get("rate_decay_T", 0.0))
        self._decay = 1.0
        self.hold_in_m = bool(params.get("hold_in_m", False))
        self.ceiling_with_divider = bool(params.get("ceiling_with_divider", False))
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
        self._said_no_cap = False       # the "ceiling not applied" note is printed once per run
        # `target_cap: c` (exp 11, 2026-09-28; default 0 = off): the TARGET `V0f` stops growing at c x the seed's median
        # target (`V0f_init` -- target against target, one convention), whether or not a `cell_divide` is scheduled.
        # The growth writes the target while `cell_cycle`'s sizer reads the MEASURED volume, so a squeezed cell that
        # never reached the checkpoint kept growing its target: 218 of the healthy gland's 1,006 cells above 2x the
        # median target at row 160, the largest 8.4x (exp 11 Finding 119). With `cell_cycle` `size_from: target` the
        # cap sits just above the checkpoint (`g1_size`) and a cell that cannot divide waits there.
        self.target_cap = float(params.get("target_cap", 0.0) or 0.0)
        # `body_cap: b` (exp 11, 2026-09-28; default 0 = off): the target may run at most b x ahead of the cell's own
        # MEASURED volume -- V0f <= b x k x v, k the seed's median V0f / median measured volume (the two conventions'
        # ratio, taken once on the first call). A squeezed cell stops growing its target instead of banking it (exp 11
        # Finding 119: targets up to 12.7x the seed median while the body sat at a tenth), its push on its neighbours
        # is bounded, and a cell with room is not held under a fixed ceiling (Finding 125: `target_cap` halved growth).
        self.body_cap = float(params.get("body_cap", 0.0) or 0.0)

    def _rate(self, s_prev, hillv, m, v_ref):
        """The RATE LAW, and the only thing a `model=` variant of cell_grow changes.

        Returns ds/dt: the per-UNIT-TIME rate of change of the per-cell linear growth scale `s`,
        which the engine integrates as `s <- s + dt * ds`. Volume is V0f_init * s**3, so a constant
        proportional rate on `s` is exponential growth in volume -- which is the default and is
        deliberate: it is what Okuda's growth term does. Ginzberg, Kafri & Kirschner (Science 2015)
        name the consequence exactly: "with exponential growth, larger cells grow faster than do
        smaller cells, amplifying any existing size disparities". Measured on this campaign's own
        basis, the coefficient of variation of cell volume climbs 0.160 at seed to 0.33-0.53 by
        frame 900 in every run. The variants below are the mechanisms that review says real cells
        use to stop that, written so the search can put them side by side.

        A RATE, NOT A PER-FRAME FACTOR. The law `s <- s (1 + rate (rho + Hill(a)))` applied once
        per CALL would make `rate` mean "fraction of itself a cell adds per frame", with the
        simulation's own `dt` nowhere in it. That is the same number only at `dt = 1`: every
        `config/tissue` spec is `dt: 1.0` and reads identically, while a spec at `dt: 0.0032` would
        run its growth 312x faster per
        unit of simulated time than the same `rate` now means, so those specs carry `rate / dt` to
        preserve what they did.
        """
        return s_prev * self.rate * self._decay * (self.rho + hillv)

    def forward(self, H, mask=None):
        if self.rate_decay_T > 0:
            # the clock of the RECORDED run: the engine's unrecorded warm-up (`H.warmup` ticks) is not growth time
            _t = max(float(getattr(H, "frame", 0) or 0) - float(getattr(H, "warmup", 0) or 0), 0.0) \
                * float(getattr(H, "dt", 1.0) or 1.0)
            self._decay = 1.0 / (1.0 + _t / self.rate_decay_T)
        # THE PAIRING IS READ FROM THE SET, ONCE PER CALL -- see `resolve_cell_set`.
        self.cat = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
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
        # THE TEST WAS DOING TWO JOBS AND THE MOVE SEPARATES THEM. `"mg_scale" not in m or
        # m["mg_scale"].shape[0] != nF` meant BOTH "this operator has not run yet" AND "the cell
        # count changed since it last did" -- and on the cell set the name is always present with
        # length exactly `nF`, so the test can never fire, `s` stays 0 and every target collapses.
        #
        # `mg_scale_nF` IS THE SECOND MEANING, WRITTEN DOWN. It is a scalar on the mesh table, like
        # `n_div` and `div_blocked`, because it is one number per run and not one per cell. The
        # behaviour is exactly what it was: re-baseline whenever the face count moves, which with
        # `cell_divide` every four frames is most frames.
        #
        # THAT POLICY IS WRONG AND IS NOT CHANGED HERE. Re-taking the whole population's baseline
        # because SOME cell divided is why `mg_scale` stops meaning "how much this cell has grown"
        # -- measured, it climbs 1.0035 -> 1.0459 before the first division and then never exceeds
        # 1.0139 -- and `contact_ops.ecm_gate_growth` read it as cumulative and spent 400 frames
        # correcting nothing. Fixing it moves numbers, so it is its own rung with its own evidence;
        # this one only moves the storage, and is byte-identical because of that.
        if int(m.get("mg_scale_nF", -1)) != int(nF):
            m["mg_scale_nF"] = int(nF)
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
        # A SECOND MORPHOGEN THAT STOPS GROWTH, so that only the activator-high spots grow.
        #
        # The growth law above is purely ACTIVATING: `rate * (rho + hill(a))`, so
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
        # THE REFERENCE IN THE CONVENTION THE CELL IS ACTUALLY MEASURED IN -- polyhedron where the
        # run carries a separation, wedge where it does not. See `vertex_ops.cell_size`. `vth_frac`
        # is a multiple of this; measured in the wedge convention on an apicobasal run, the growth
        # ceiling would be stated in a volume the cell does not have.
        # ONE READER: the cell's ACTUAL volume and the reference, both from `cell_size`, so the
        # size-dependent growth rule compares what the cell has with what a typical cell had --
        # it used to compare the TARGET `V0f` with the reference, two different quantities.
        from plexus.operators.vertex_ops import cell_size
        _v_now, v_ref = cell_size(vlvl, m, nF, H=H, at=self.at)
        self._v_now = torch.as_tensor(np.asarray(_v_now, np.float64), device=dev,
                                      dtype=m["V0f"].dtype)
        dt = float(getattr(H, "dt", 1.0))
        ds = self._rate(s_prev, hillv, m, v_ref)                  # <-- the rate law; models override THIS only
        # `hold_in_m: true` (default false = unchanged): a cell in M (`phase` >= 3 on the cell set) does not grow.
        # exp 11's Type I loop: a division-ready cell WAITS in M for its turn to leave the layer (the dive cap), and
        # growing there it inflated its targets 5x and stretched the layer 4x over them (Finding 102). A mitotic cell
        # stops growing (Mitchison 1971); every cell that is not waiting is untouched.
        if self.hold_in_m:
            from plexus.operators.vertex_ops import cell_block
            _ph = cell_block(H, self.cat, "phase", nF) if H is not None else None
            if _ph is not None and len(_ph) == nF:
                ds = ds * torch.as_tensor((np.asarray(_ph, np.float64) < 3.0).astype(np.float64),
                                          device=ds.device, dtype=ds.dtype)
        # THE WARM-UP HOLDS GROWTH (2026-09-28). While the engine's unrecorded warm-up runs (H.frame < H.warmup, the
        # seed's settle window) the cell cycle and division idle; growing the targets there handed every cell about
        # twice its age's size at frame 0 -- the large, young cells and the first division wave of exp 11 (Finding 130).
        # `general.warmup: 0` turns the warm-up, and with it this hold, off.
        _wu = int(getattr(H, "warmup", 0) or 0)
        if _wu > 0 and int(getattr(H, "frame", 0) or 0) < _wu:
            ds = ds * 0.0
        # THE CEILING IS FOR A TISSUE WITH NO DIVIDER, AND ONLY FOR ONE.
        #
        # `vth_frac` is Okuda's uniform-cell mode: cap `v_eq` under `vth_frac * v_ref` so every cell
        # oscillates in a band and the population stays uniform WITHOUT anything resetting it. Once
        # `cell_divide` is in the schedule, division is what resets size -- a cell doubles, splits,
        # and each daughter starts at half -- so the ceiling is no longer size control. It is a lid,
        # and if it sits below the division threshold the tissue can never reach that threshold.
        #
        # IT DID. `divide_growing_ball` caps at `vth_frac 2.5 * v_ref 2.5433` = 6.36 in WEDGE units,
        # which plateaus the polyhedron volume -- the one `cell_divide`'s trigger reads -- at 1.987
        # against a reference of 1.229. That is 1.62x, and `factor: 2.0` needs 2x. Not one division
        # fired in 401 frames and the cell count sat flat at 200. The two conventions are the deeper
        # problem (AB_R7R8_TODO section 0a) but they are not what makes this spec dead: two
        # mechanisms were doing one job.
        #
        # A CEILING ON A RATE IS A GATE, NOT A CLAMP. When the operator wrote `s` itself the lid was
        # `s <- min(s, s_cap)`; an operator that only emits ds/dt cannot clamp a value it does not
        # own, so the lid is applied to the RATE -- a cell at or above its ceiling grows at zero.
        # Same fixed point (`s` settles on `s_cap`), and it no longer overshoots-then-truncates
        # within a step, so the band is entered rather than snapped to.
        _has_divider = "cell_divide" in getattr(H, "scheduled_ops", frozenset())
        # `ceiling_with_divider: true` (default false = unchanged): keep the ceiling although a `cell_divide` is
        # scheduled -- for a layer whose `cell_divide`s do NOT reset its cells' size in place (exp 11's Type I loop:
        # `reinsert` and `cell_divide[model: mpm]`; every division of a surface cell leaves the layer), where the
        # uncapped targets grew x5 and blew the shell up past its own membrane (Finding 103).
        if self.ceiling_with_divider:
            _has_divider = False
        if self.rho > 0 and not _has_divider:                        # OKUDA uniform-cell mode
            s_cap = ((self.vth_frac * v_ref
                      / m["V0f_init"].clamp(min=1e-9)) ** (1.0 / 3.0)).clamp(min=1.0)
            ds = torch.where(s_prev + dt * ds > s_cap, (s_cap - s_prev) / max(dt, 1e-12), ds)
        elif self.rho > 0:
            if not self._said_no_cap:
                self._said_no_cap = True
                print(f"[cell_grow] `cell_divide` is scheduled, so the `vth_frac` ceiling is not "
                      f"applied: division is what resets cell size here, and a ceiling below the "
                      f"division threshold would stop the tissue reaching it.", flush=True)
        else:
            # legacy: activator-only bulge, gated to a stop at `cap` for the same reason as above
            cap = torch.as_tensor(self.cap, device=ds.device, dtype=ds.dtype)
            ds = torch.where(s_prev + dt * ds > cap, (cap - s_prev) / max(dt, 1e-12), ds)
        # THE FOUR TARGETS ARE ONE INTEGRAND SEEN FOUR WAYS, and the chain rule is what keeps them
        # one. `s` is what grows; A0 = A0_init s^2, P0 = P0_init s, V0f = V0f_init s^3 are algebraic
        # functions of it, so their rates are its rate times the derivative of each map:
        #
        #     dA0/dt = 2 A0_init s (ds/dt)      dP0/dt = P0_init (ds/dt)      dV0f/dt = 3 V0f_init s^2 (ds/dt)
        #
        # WHAT THIS COSTS, said plainly, because it is a real difference and not a reassociation.
        # Writing `A0 = A0_init s_new^2` was exact in `s`; integrating its derivative is exact only
        # to first order, and leaves A0 short of A0_init s^2 by A0_init (dt ds)^2 per step. On this
        # campaign's own growth rate (5.78e-04 per unit time) that is 3.3e-07 of A0 per frame. The
        # alternative -- emitting the exact chord (s_new^2 - s^2)/dt -- would reproduce today's
        # numbers bit for bit but requires the operator to integrate `s` itself, which is the engine's
        # job and the thing this rung exists to give back to it.
        if self.body_cap > 0:
            _n = ds.shape[0]
            _vb = self._v_now[:_n].to(ds.dtype)
            if getattr(self, "_k_body", None) is None:
                self._k_body = float(torch.median(m["V0f"][:_n])) / max(float(torch.median(_vb)), 1e-12)
            ds = cap_target_rate(ds, s_prev, m["V0f"][:_n], m["V0f_init"][:_n],
                                 self.body_cap * self._k_body * _vb.clamp(min=0.0), dt, self.GROWTH_DIMS)
        if self.target_cap > 0:
            # THE REFERENCE IS TAKEN ONCE, on the first call (the seed's targets). `V0f_init` cannot serve: this
            # operator re-baselines it to the current `V0f` whenever the face count moves, so a cap on its median
            # floated with the population and held nothing (exp 11 mc_e_tc_ctl, first build).
            if getattr(self, "_v0_seed", None) is None:
                self._v0_seed = float(torch.median(m["V0f"][:ds.shape[0]]))
            _v0 = m["V0f"][:ds.shape[0]]
            _cap = self.target_cap * self._v0_seed
            ds = cap_target_rate(ds, s_prev, _v0, m["V0f_init"][:ds.shape[0]], _cap, dt, self.GROWTH_DIMS)
        s_next = s_prev + dt * ds                                # what `s` will be once the engine integrates
        deltas = {(self.cat, "mg_scale"): ds,
                  (self.cat, "A0"): 2.0 * m["A0_init"] * s_prev * ds,
                  (self.cat, "P0"): m["P0_init"] * ds,
                  (self.cat, "V0f"): (3.0 * m["V0f_init"] * s_prev * s_prev * ds if self.GROWTH_DIMS == 3
                                      else 2.0 * m["V0f_init"] * s_prev * ds)}   # the isotropic form verbatim: bit-identical
        s = s_next                                               # the scalars below summarise the POST-tick tissue
        m["V0"] = float((m["V0f_init"] * s ** self.GROWTH_DIMS).sum())
        # THE SHELL RADIUS MUST GROW WITH THE CELLS. cell_mechanics carries a radial spring,
        #     E += K_R * sum_i (|x_i| - R0)^2
        # and R0 is set once at seeding. An operator that grows the cells without rescaling R0
        # leaves the mechanics pinning the shell at the SEED radius while the
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
            g_vol = (s / s_prev.clamp(min=1e-9)) ** self.GROWTH_DIMS
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
        # THE LIVE CELLS ARE A PREFIX OF THE BUFFER, and a block delta has to be the whole buffer:
        # `_integrate` adds it to `lvl.state[:, cx0:cx1]` with no mask, so a short tensor would not
        # broadcast and a full one must be zero past `nF` or the dead tail of the buffer would grow.
        N = int(clvl.state.shape[0])
        out = {}
        for key, d in deltas.items():
            z = torch.zeros(N, 1, device=clvl.state.device, dtype=clvl.state.dtype)
            z[:nF, 0] = d.to(device=clvl.state.device, dtype=clvl.state.dtype)
            out[key] = z
        return out


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


@register_operator("cell_grow", model="sizer", set="vertex", kind="lateral", family="population", title="Growth where the morphogen is high")
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

    def _rate(self, s_prev, hillv, m, v_ref):
        v_now = self._v_now.clamp(min=1e-9)                     # actual volume, from `cell_size`
        f = (v_ref / v_now) ** self.size_gain
        f = torch.clamp(f, 1.0 / max(self.f_max, 1e-9), self.f_max)
        return s_prev * self.rate * (self.rho + hillv) * f


@register_operator("cell_grow", model="balance", set="vertex", kind="lateral", family="population", title="Growth where the morphogen is high")
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

    def _rate(self, s_prev, hillv, m, v_ref):
        # `dv` IS ALREADY dV/dt -- the docstring's own equation is written that way -- so the only
        # conversion needed is from a volume rate to a rate on the linear scale. With V = V0f_init s^3,
        # dV/dt = 3 V0f_init s^2 (ds/dt), hence ds/dt = (dV/dt) * s / (3 V).
        v_now = (m["V0f_init"] * s_prev ** 3).clamp(min=1e-9)
        dv = self.rate * (self.k_syn * v_ref * (self.rho + hillv) - self.k_deg * v_now)
        return dv * s_prev / (3.0 * v_now)


@register_operator("cell_grow", model="timer", set="vertex", kind="lateral", family="population", title="Growth to a target size",
                   equation=r"""$$\frac{d\ln V_j}{dt}=\frac{1}{T}\ln\frac{V_{\mathrm{target}}}{V_j}$$""")
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

    def _rate(self, s_prev, hillv, m, v_ref):
        # THE PROPORTIONAL CONTROLLER, WRITTEN AS THE RATE IT ALWAYS WAS. The per-frame factor
        # (v_tgt/v_now)^(1/cycle_frames) is exp(ln(v_tgt/v_now)/cycle_frames), so the underlying law is
        # d(ln V)/dt = ln(v_tgt/v_now) / cycle_frames -- first-order relaxation of log-volume with
        # time constant `cycle_frames`. On the linear scale that is a third of it, since s = V^(1/3).
        # `cycle_frames` is now a DURATION IN SIMULATION TIME rather than a count of calls; at the
        # `dt: 1.0` every spec using this model runs at, the two are the same number.
        v_now = (m["V0f_init"] * s_prev ** self.GROWTH_DIMS).clamp(min=1e-9)
        share = (self.rho + hillv) / max(self.rho + 1.0, 1e-9)   # the morphogen sets WHERE, as a target
        v_tgt = (self.vth_frac * v_ref * share).clamp(min=1e-9)
        return s_prev * torch.log(v_tgt / v_now) / (self.GROWTH_DIMS * max(self.cycle_frames, 1.0))


# =========================================================== cell_grow[stretch]: growth read off the cell's stretch
# `model: stretch`, written for experiment 13 (`experiments/exp13_mechanical_size.md`), whose Stage 0 audit
# found that no `cell_grow` model reads the cell's mechanical state. `model=` and not `implementation=`, by
# the axis test: a different growth law, not the same one computed differently. Its three helpers
# (`face_neighbour_mean`, `polar_gf`, `clone_cap`) sit with it. Folded here from its own module
# `growth_feedback.py` on 2026-09-27 (INSTRUCTION.md: every variant lives beside its base operator).


def face_neighbour_mean(values, es, et, ef, nF, rings=1):
    """Each face's value averaged with its edge-neighbours', `rings` times: (v_j + sum_nb v_k) / (1 + deg_j).
    Two faces touch when half-edge (a, b) of one has its twin (b, a) in the other."""
    es, et, ef = (torch.as_tensor(a).long().to(values.device) for a in (es, et, ef))
    live = (ef >= 0) & (ef < nF)
    es, et, ef = es[live], et[live], ef[live]
    if len(es) == 0 or rings <= 0:
        return values
    big = int(torch.maximum(es.max(), et.max())) + 1
    key, twin = es * big + et, et * big + es
    ks, order = torch.sort(key)
    j = torch.searchsorted(ks, twin).clamp(max=len(ks) - 1)
    ok = ks[j] == twin
    fa, fb = ef[ok], ef[order[j[ok]]]
    keep = fa != fb
    fa, fb = fa[keep], fb[keep]
    deg = torch.zeros(nF, dtype=values.dtype, device=values.device).index_add_(
        0, fa, torch.ones_like(fa, dtype=values.dtype))
    v = values
    for _ in range(int(rings)):
        v = (v + torch.zeros_like(v).index_add_(0, fa, v[fb])) / (1.0 + deg)
    return v


def polar_gf(centroids, axis, angle_deg, width_deg):
    """A growth factor in [0, 1] highest around `axis`: 1 / (1 + exp((theta - angle) / width)), theta the
    angle between `axis` and the cell's centroid seen from the tissue's centroid, in degrees. Set by the
    ANGLE, so it scales with the tissue as it grows -- Aegerter-Wilmsen 2007's third assumption, "the
    Dpp activity gradient and that of the other growth factor are scaled"."""
    c0 = torch.as_tensor(centroids)
    c = c0.to(torch.float64)
    # THE AXIS ON THE CENTROIDS' DEVICE: live, they are CUDA tensors, and a CPU axis made batch 6's four
    # jobs die in their first tick (exp 13, 2026-09-26) -- a CPU smoke run cannot see it.
    a = torch.as_tensor(axis, dtype=torch.float64, device=c.device)
    a = a / a.norm().clamp(min=1e-12)
    x = c - c.mean(0)
    cos = (x @ a) / x.norm(dim=1).clamp(min=1e-12)
    theta = torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))
    return torch.sigmoid((float(angle_deg) - theta) / max(float(width_deg), 1e-9)).to(c0.dtype)


def isotropic_stress(pos, es, et, ef, nF, A, A0, P0, mech, myo=None):
    """Each face's isotropic (Batchelor) stress on a vertex sheet, over 2 K_A A0 -- dimensionless:

        s_f = [ 2 K_A (A_f - A0_f) + (1 / (4 A_f)) sum_{h in f} T_h |l_h| ] / (2 K_A A0_f)
        T_h = tau_h + tau_twin,   tau_h = Lambda m_h + 2 K_P (P_f - P0_f) + Gamma P_f

    half the trace of sigma_f = -Pi_f I + (1 / A_f) sum_h (T_h / 2) l_h l_h / |l_h| for the default
    `cell_mechanics` energy (K_A, K_P, Gamma, Lambda read from the mesh's `mech`). Tensile positive; 0 for a
    cell at force balance of ANY size, which is what a size-free stretch must be. m_h is the half-edge's
    myosin multiplier on Lambda (`junction_myosin` writes it as m["myo"], `cell_mechanics` uses it); 1 when
    `myo` is None or does not match the half-edge count, so a run without myosin reads as before."""
    es, et, ef = (torch.as_tensor(a).long().to(pos.device) for a in (es, et, ef))
    l = pos[et] - pos[es]
    L = l.norm(dim=1)
    dt = pos.dtype
    P = torch.zeros(nF, dtype=dt, device=pos.device).index_add_(0, ef, L)
    KA, KP, G, Lam = (float(mech.get(k, 0.0)) for k in ("K_A", "K_P", "Gamma", "Lambda"))
    mh = torch.ones_like(L) if myo is None or myo.shape[0] != L.shape[0] else myo.to(pos.device, dt).reshape(-1)
    tau = Lam * mh + (2 * KP * (P - P0.to(dt)) + G * P)[ef]            # per half-edge
    big = int(torch.maximum(es.max(), et.max())) + 1
    key, twin = es * big + et, et * big + es
    ks, order = torch.sort(key)
    j = torch.searchsorted(ks, twin).clamp(max=len(ks) - 1)
    has = ks[j] == twin
    T = tau + torch.where(has, tau[order[j]], torch.zeros_like(L))
    iso_t = torch.zeros(nF, dtype=dt, device=pos.device).index_add_(0, ef, T * L) / (4 * A.to(dt).clamp(min=1e-12))
    return (2 * KA * (A.to(dt) - A0.to(dt)) + iso_t) / (2 * max(KA, 1e-12) * A0.to(dt).clamp(min=1e-12))


def clone_disk(centroids, point, frac):
    """Indices of the `frac` of cells nearest the point centroid + `point` x R, R the largest distance of
    a cell centroid from the tissue's centroid -- a compact clone INSIDE a tissue (on a flat disc the
    polar cap of `clone_cap` is the free rim). At least one cell when frac > 0."""
    c = torch.as_tensor(centroids).to(torch.float64)
    g = c.mean(0)
    R = (c - g).norm(dim=1).max()
    target = g + torch.as_tensor(point, dtype=torch.float64, device=c.device) * R
    n = max(1, int(round(float(frac) * len(c)))) if frac > 0 else 0
    return torch.argsort((c - target).norm(dim=1))[:n] if n else torch.zeros(0, dtype=torch.long)


def hinge_ring(centroids, frac):
    """Indices of the `frac` of cells whose centroids lie FARTHEST from the tissue's centroid -- the outer
    annulus of a disc, the hinge that rings the wing pouch. At least one cell when frac > 0."""
    c = torch.as_tensor(centroids, dtype=torch.float64)
    n = c.shape[0]
    k = max(1, int(round(float(frac) * n))) if frac > 0 else 0
    if k == 0:
        return torch.zeros(0, dtype=torch.long)
    r = (c - c.mean(0)).norm(dim=1)
    return torch.argsort(r, descending=True)[:k]


def clone_cap(centroids, axis, frac):
    """Indices of the `frac` of cells whose centroid lies farthest along `axis` from the tissue's
    centroid -- a polar cap, the compact patch a clone is (Shraiman 2005 Fig. 1). At least one cell
    when frac > 0."""
    c = torch.as_tensor(centroids).to(torch.float64)
    a = torch.as_tensor(axis, dtype=torch.float64, device=c.device)
    a = a / a.norm().clamp(min=1e-12)
    proj = (c - c.mean(0)) @ a
    n = max(1, int(round(float(frac) * len(c)))) if frac > 0 else 0
    return torch.argsort(proj)[len(c) - n:] if n else torch.zeros(0, dtype=torch.long)


@register_operator("cell_grow", model="stretch", set="vertex", kind="lateral", family="population",
                   title="Growth where the cell is stretched",
                   equation=r"""$$\frac{ds_j}{dt}=s_j\,\text{rate}\big(\rho+\mathrm{Hill}(a_j)\big)\,m_j\,\mathrm{clip}\big(1+g(\sigma_j-1),0,f_{max}\big),\qquad \sigma_j=\frac{V_j/V^0_j}{\mathrm{median}_k\,V_k/V^0_k}$$""")
class Grow3DStretch(Grow3D):
    """Growth rate rises when the cell is stretched and falls when it is compressed: MECHANICAL
    FEEDBACK on growth, the Hippo/YAP mechanism (Shraiman 2005; Aegerter-Wilmsen et al. 2007, 2012;
    Pan et al. 2016 measured its readout in fast-growing clones).

        ds_j/dt  = s_j rate (rho + Hill(a_j)) m_j f_j
        f_j      = clip(1 + gain (sigma_j - 1), 0, f_max)
        sigma_j  = (V_j / V0f_j) / median_k (V_k / V0f_k)

    WHY THIS MODEL EXISTS: none of the four others reads the cell's mechanical state. `balance` is
    dV/dt = rate (k_syn v_ref (rho + Hill) - k_deg V) with V the TARGET volume V0f_init s^3, never the
    actual one; `sizer` reads the actual volume but against the population's `v_ref`, which is size,
    not stress; `timer` reads the target again (exp 13's Stage 0 audit, 2026-09-26).

    WHAT "STRETCH" IS HERE. The apico-basal energy has ONE per-cell target, 1/2 k_v (V_j - V0f_j)^2,
    so V_j / V0f_j is the cell's volumetric stretch and k_v (V0f_j - V_j) its pressure. V_j is the
    cell's actual volume from `cell_size` (the convention `k_v` defends). sigma_j divides that
    stretch by the TISSUE'S MEDIAN, for two reasons:
      1. Shraiman's Eq. 1: the pressure a patch feels integrates its growth rate MINUS the tissue's
         average, so uniform growth is stress-free; the feedback reads a cell against the tissue.
      2. The recorded V / V0f is 0.53 in EVERY cell at every frame of exp 3's base (`g1_sizer_s1`,
         median 0.531-0.543 from frame 0 to 1600) -- the polyhedron/wedge ratio 1.3508/2.5433 that
         `ApicoBasalShapeEnergy3D.BLOCK_UNITS` documents. A convention offset, not a stress; the
         median divides it out. The cost: a GLOBAL compression cannot be read, so this model alone
         cannot arrest a tissue that grows uniformly.

    `readout: area` READS THE STRETCH IN THE PLANE instead: sigma_j = (A_j / V_j^(2/3)) / median, A_j
    the cell's area from `cell_geometry` -- a size-free flatness, below 1 for a cell squeezed in the
    plane and made taller. Shraiman's own footnote says this is where the pressure shows: "this 2D
    pressure would be the uniaxial stress in the cell layer corresponding to the modulation of layer
    thickness and cell (apical) area", and Pan 2016 Fig. 1E sees fast clones widen apico-basally. On
    this tissue it is also where it is MEASURABLE: `k_v` defends the volume, so a 2x clone's V/V0f
    stayed within 1.5 % of the tissue's (and flipped sign), while its A/V^(2/3) fell to 0.91-0.87 of
    it (exp 13 batch 1, clone_k1/k2 s1, frames 500-1400). `volume` stays the default so batch 1 reads
    the same.

    `readout: thickness` IS THE IN-PLANE STRETCH OF A MONOLAYER: sigma_j = (A_j / V_j) / median, the
    inverse of the cell's height. A / V^(2/3) is size-free for an isotropic body, NOT for a sheet of
    set thickness, where A ~ V and A / V^(2/3) ~ V^(1/3): it reads a big cell as a stretched one.
    Measured on the no-feedback `exp13_g0_s1`, corr(log V, log A/V^(2/3)) = +0.60 to +0.71 across cells
    (frames 400-1200), so `area` feeds size back POSITIVELY -- the no-clone probe at gain 5 grew with a
    within-cell exponent beta = 1.78 against exp 3's 0.9. A / V reads -0.25 to -0.39 there (bigger
    cells slightly taller), and it carries the clone's compression more strongly: the runaway clone's
    cells read 0.85 / 0.88 / 0.77 of the rest's A / V at frames 400 / 800 / 1200, against 1.00 / 0.99 /
    0.92 for A / V^(2/3) (`exp13_clone_g0_s1`).

    `smooth: n` AVERAGES sigma OVER EACH CELL'S EDGE-NEIGHBOURS, n rings (`face_neighbour_mean`). Shraiman's
    finite-thickness correction does exactly this to the pressure -- a term w^2 laplacian(dp/dt) that
    "smoothen[s] out any spatial variation of the pressure", w comparable with the cell size -- and on
    this tissue it is what makes a useful gain stable: the per-cell scatter of sigma (interquartile
    0.1-0.2, from division and relaxation) is larger than a clone's signal (~0.08), and a gain of 12 on
    the raw sigma broke the clone into shards (exp 13 batch 2, clone_a3_s1, wrecked at frame 1060).

    `readout: a0` IS THE STRETCH OF A FLAT SHEET: sigma_j = (A_j / A0_j) / reference, the cell's area over
    its target area. A flat vertex sheet at z = 0 has no wedge volume (V and V0f are 0 -- exp 13 Phase 2's
    survey of exp01_v8 and exp07), so the three readouts above divide by zero there; the apical area
    against its target is Aegerter-Wilmsen et al. 2012's own measure of compression ("a weighted average
    of the area of a cell and its surroundings", with `smooth`), and `cell_grow` raises A0 as s^2.

    `readout: stress` READS THE CELL'S MECHANICAL STRESS, not a shape: sigma_j = 1 + s_j, s_j the isotropic
    Batchelor stress of the default vertex energy over 2 K_A A0 (`isotropic_stress`), 0 at force balance
    for a cell of any size -- Shraiman 2005's pressure, AW 2012's compression. WHY: every shape readout
    tried is size-confounded on some tissue. On exp 13 Phase 2's flat pouch (exp07's energy, line tension
    and contractility) cells sit at A / A0 ~0.36 and it RISES with cell size (0.36 -> 0.44 by frame 200 with
    no feedback at all), so `a0` against the settled tissue fed size back positively and the pouch exploded
    (x41,811 in area, the cell buffer full by frame 600, auditor wrecked at 220). `stretch_ref: tissue`
    subtracts the tissue's median stress (Shraiman's "minus the average").

    `and_chan: k` MAKES THE DRIVE AN AND GATE of two morphogens: rho + Hill(a) Hill_k(c_k), Hill_k relative
    to c_k's own maximum (`and_sw`, `and_hill`). Aegerter-Wilmsen 2007 (p. 319): "Growth is induced if both
    Dpp and the second growth factor are present" -- Dpp from the A/P stripe times Wg from the D/V
    boundary, "a tent, with highest activity in the center of the disc" (Fig. 1); AW 2012 Fig. 1B. A stripe
    alone drives growth along a line and the pouch elongates (exp 13 Phase 2, P5/P8); the tent peaks at
    the centre, which LeGoff's concentric stress pattern needs.

    `gain` is dimensionless, the fractional change of the growth rate per unit of relative stretch.
    gain = 0 is f = 1 and the multiplication is skipped: the default law, bit for bit -- the identity
    this model is run against. f is clipped at 0 (a compressed cell stops; dying is `cell_die`'s job)
    and at `f_max`, a multiple of the undisturbed rate.

    THE CLONE. m_j = `clone_mult` for cells whose `clone_block` reads above 0.5, else 1: a patch
    driven to grow faster, the experiment the papers predict from (Shraiman Fig. 4a). With
    `clone_frac` > 0 the operator marks it at its first call: the `clone_frac` of live cells farthest
    along `clone_axis` from the tissue's centroid (`clone_cap`). It is marked HERE and not by a seed
    operator because a mesh cell has no position at x_0 -- `cell_geometry` writes the `centroid`
    block on the first tick, just before this operator -- and `seed_state` reads a `pos` the cell set
    of a mesh run does not carry. The block is a declared cell-set column, which `cell_divide` copies
    onto both daughters, so the clone is a lineage. If the set declares a `stretch` block, sigma_j is
    written into it every call (a readout, like `inhib_frac`), for the rulers and the movie.
    `clone_point: [x, y, z]` places the clone INSIDE the tissue instead (`clone_disk`): the `clone_frac` of
    cells nearest the tissue's centroid plus `clone_point` times its radius -- on a flat disc the cap is
    the free rim, and Pan et al. 2016's fast clones sit in the pouch (Fig. 1D).

    AEGERTER-WILMSEN 2007's LAW (all off by default). Relative stretch cannot stop a tissue that grows
    uniformly -- the tissue's own median is always 1 (exp 13 findings 11 and 19). AW 2007's arrest needs
    three things this model then adds, each a parameter:
      `drive: polar`          the growth factor only in a cap around `gf_axis` (`polar_gf`, half-width
                              `gf_angle_deg`, edge `gf_width_deg`), multiplying the default drive;
      `stretch_ref: settle_median`  with `readout: stress`, the tissue's median stress at `ref_frame` is
                              subtracted as a fixed offset (a shape readout treats it as `settle`);
      `stretch_ref: settle`   sigma read against the tissue's median at `ref_frame` (the settle frame),
                              then FIXED -- an absolute stretch, so a global compression can be read;
      `stretch_growth` k_s, `stretch_threshold` theta_s
                              growth induced by stretch above the threshold, anywhere:
                              + s rate k_s max(sigma - 1 - theta_s, 0).
    So ds/dt = s rate [(rho + Hill) gf m f(sigma) + k_s max(sigma - 1 - theta_s, 0)]: "growth is induced
    if both [growth factors] are present", "compression ... inhibits net growth and ... stretching
    stimulates it", "stretching is assumed to only induce growth above a certain threshold" (AW 2007
    p. 319). The quantitative law is in AW's supplement, which is not on disk; these forms are ours.

    Reference: Shraiman, B.I. (2005). Mechanical feedback as a possible regulator of tissue growth.
    PNAS 102:3318; Aegerter-Wilmsen, T. et al. (2007). Mech. Dev. 124:318.

    THE HINGE (off by default). `hinge_frac` > 0 marks, at the first call, that fraction of live cells
    farthest from the tissue's centroid (`hinge_ring`) into the declared `hinge_block`, and multiplies
    their rate by `hinge_mult` (default 0: they do not grow). A non-growing ring of CELLS around the
    pouch, as the hinge surrounds the wing pouch (Aegerter-Wilmsen et al. 2012, Development 139:3221):
    unlike a wall (`plate_confine[sphere]`), it is elastic and stretches when the pouch pushes, so a
    pouch that does not read its stress keeps growing into it instead of folding over itself (exp 13
    Phase 2 P23, P28). Like the clone, the block is copied onto both daughters.
    """
    MECHANISM_TAGS = ["growth", "size_control", "mechanical_feedback", "stretch", "clone"]
    REFERENCE = ("Shraiman, B.I. (2005). PNAS 102:3318; Aegerter-Wilmsen, T., Aegerter, C.M., Hafen, E. "
                 "& Basler, K. (2007). Mech. Dev. 124:318.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.gain = float(params.get("gain", 0.0))
        self.f_max = float(params.get("f_max", 4.0))
        self.clone_mult = float(params.get("clone_mult", 1.0))
        self.clone_frac = float(params.get("clone_frac", 0.0))
        self.clone_axis = [float(v) for v in (params.get("clone_axis") or [0.0, 0.0, 1.0])]
        self.clone_block = str(params.get("clone_block", "clone"))
        self.hinge_frac = float(params.get("hinge_frac", 0.0))
        self.hinge_mult = float(params.get("hinge_mult", 0.0))
        self.hinge_block = str(params.get("hinge_block", "hinge"))
        self._hinge_marked = False
        _ac = params.get("and_chan", None)
        self.and_chan = None if _ac is None else int(_ac)
        self.and_sw = float(params.get("and_sw", 0.3))
        self.and_hill = float(params.get("and_hill", 2.0))
        self._and = None
        _cp = params.get("clone_point", None)
        self.clone_point = None if _cp is None else [float(v) for v in _cp]
        self.readout = str(params.get("readout", "volume")).lower()
        self.smooth = int(params.get("smooth", 0))
        # AEGERTER-WILMSEN 2007's LAW, every option off by default (see the class docstring's last part)
        self.drive_kind = str(params.get("drive", "uniform")).lower()
        if self.drive_kind not in ("uniform", "polar"):
            raise ValueError(f"cell_grow[stretch]: drive is uniform or polar, got {self.drive_kind!r}")
        self.gf_axis = [float(v) for v in (params.get("gf_axis") or [0.0, 0.0, 1.0])]
        self.gf_angle = float(params.get("gf_angle_deg", 60.0))
        self.gf_width = float(params.get("gf_width_deg", 10.0))
        self.stretch_ref = str(params.get("stretch_ref", "tissue")).lower()
        if self.stretch_ref not in ("tissue", "settle", "settle_median"):
            raise ValueError(f"cell_grow[stretch]: stretch_ref is tissue, settle or settle_median, got {self.stretch_ref!r}")
        self._off = None
        self.ref_frame = int(params.get("ref_frame", 60))
        self.stretch_growth = float(params.get("stretch_growth", 0.0))
        self.stretch_threshold = float(params.get("stretch_threshold", 0.0))
        self._ref = None
        self._gf = None
        if self.readout not in ("volume", "area", "thickness", "a0", "stress"):
            raise ValueError(f"cell_grow[stretch]: readout is volume, area, thickness, a0 or stress, got {self.readout!r}")
        self._marked = False
        self._drive = None

    def _clone_drive(self, H):
        """m_j per live cell, or None when there is no clone (so the default path multiplies nothing)."""
        if self.clone_frac <= 0 and self.clone_mult == 1.0:
            return None
        from plexus.operators.vertex_ops import cell_block_t, set_cell_block_t
        cat = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
        m = getattr(H.level(self.at), "_mesh", None)
        if m is None:
            return None
        nF = int(m["nF"])
        k = cell_block_t(H, cat, self.clone_block, nF)
        if k is None:
            raise ValueError(f"cell_grow[stretch]: a clone needs `sets.{cat}.state.{self.clone_block}: "
                             f"{{width: 1}}` declared on the cell set")
        if not self._marked and self.clone_frac > 0:
            clvl = H.level(cat)
            if "centroid" not in clvl.state_schema:
                raise ValueError(f"cell_grow[stretch]: marking a clone needs the `centroid` block on "
                                 f"`{cat}` (written by `cell_geometry`, scheduled before this operator)")
            vals = torch.zeros(nF, dtype=clvl.state.dtype, device=clvl.state.device)
            _c = clvl.get("centroid")[:nF].detach().cpu()
            vals[clone_disk(_c, self.clone_point, self.clone_frac) if self.clone_point is not None
                 else clone_cap(_c, self.clone_axis, self.clone_frac)] = 1.0
            set_cell_block_t(H, cat, self.clone_block, vals, nF)
            self._marked = True
            k = cell_block_t(H, cat, self.clone_block, nF)
        one = torch.ones((), dtype=k.dtype, device=k.device)
        return torch.where(k > 0.5, one * self.clone_mult, one)

    def _hinge_drive(self, H):
        """`hinge_mult` for hinge cells, 1 elsewhere; None without a hinge (the default path)."""
        if self.hinge_frac <= 0:
            return None
        from plexus.operators.vertex_ops import cell_block_t, set_cell_block_t
        cat = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
        m = getattr(H.level(self.at), "_mesh", None)
        if m is None:
            return None
        nF = int(m["nF"])
        k = cell_block_t(H, cat, self.hinge_block, nF)
        if k is None:
            raise ValueError(f"cell_grow[stretch]: a hinge needs `sets.{cat}.state.{self.hinge_block}: "
                             f"{{width: 1}}` declared on the cell set")
        if not self._hinge_marked:
            clvl = H.level(cat)
            if "centroid" not in clvl.state_schema:
                raise ValueError(f"cell_grow[stretch]: marking a hinge needs the `centroid` block on `{cat}`")
            vals = torch.zeros(nF, dtype=clvl.state.dtype, device=clvl.state.device)
            vals[hinge_ring(clvl.get("centroid")[:nF].detach().cpu(), self.hinge_frac)] = 1.0
            set_cell_block_t(H, cat, self.hinge_block, vals, nF)
            self._hinge_marked = True
            k = cell_block_t(H, cat, self.hinge_block, nF)
        one = torch.ones((), dtype=k.dtype, device=k.device)
        return torch.where(k > 0.5, one * self.hinge_mult, one)

    def forward(self, H, mask=None):
        self._drive = self._clone_drive(H)
        hd = self._hinge_drive(H)
        if hd is not None:
            self._drive = hd if self._drive is None else self._drive * hd.to(self._drive.device)
        self._H = H
        self._gf = self._polar_gf(H) if self.drive_kind == "polar" else None
        self._and = self._and_gate(H) if self.and_chan is not None else None
        return super().forward(H, mask)

    def _polar_gf(self, H):
        """The growth factor of each live cell, a logistic cap around `gf_axis` (`polar_gf`); written into
        a declared `gf` block for the movie and the rulers."""
        cat = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
        m = getattr(H.level(self.at), "_mesh", None)
        clvl = H.level(cat)
        if m is None or "centroid" not in clvl.state_schema:
            raise ValueError("cell_grow[stretch] drive: polar needs the `centroid` block (cell_geometry)")
        nF = int(m["nF"])
        gf = polar_gf(clvl.get("centroid")[:nF].detach(), self.gf_axis, self.gf_angle, self.gf_width)
        if "gf" in clvl.state_schema:
            from plexus.operators.vertex_ops import set_cell_block_t
            set_cell_block_t(H, cat, "gf", gf, nF)
        return gf

    def _and_gate(self, H):
        """Hill(c) of the `and_chan` morphogen, relative to its own maximum (`and_sw` of it, sharpness
        `and_hill`) -- the second growth factor of an AND gate (the class docstring)."""
        cat = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
        m = getattr(H.level(self.at), "_mesh", None)
        clvl = H.level(cat)
        if m is None or "chem" not in clvl.state_schema:
            return None
        nF = int(m["nF"])
        h0, _ = clvl.state_schema["chem"]
        c = clvl.state[:nF, h0 + self.and_chan].detach().clamp(min=0.0)
        cmax = float(c.max()) if c.numel() else 0.0
        if cmax <= 1e-9:
            return torch.zeros_like(c)
        thr = self.and_sw * cmax
        return c ** self.and_hill / (thr ** self.and_hill + c ** self.and_hill)

    def _rate(self, s_prev, hillv, m, v_ref):
        if self._and is not None:
            hillv = hillv * self._and[:int(s_prev.shape[0])].to(hillv.dtype).to(hillv.device)
        base = super()._rate(s_prev, hillv, m, v_ref)
        nF = int(s_prev.shape[0])
        r = base
        if self._gf is not None:
            r = r * self._gf[:nF].to(r.dtype).to(r.device)
        if self._drive is not None:
            r = r * self._drive[:nF].to(r.dtype).to(r.device)
        need_sigma = self.gain != 0.0 or self._stretch_declared() or self.stretch_growth > 0
        if not need_sigma:
            return r
        if self.readout == "stress":
            mech = m.get("mech")
            vlvl = self._H.level(self.at)
            if mech is None or not all(k in m for k in ("E_srce", "E_trgt", "E_face")):
                return r                                       # the mechanics has not run yet: no signal
            from plexus.operators.vertex_ops import cell_block_t
            a = cell_block_t(self._H, self.cat, "area", nF)
            pos = vlvl.get("pos")[:int(m["Nv"])].detach().to(r.dtype)
            sig = 1.0 + isotropic_stress(pos, m["E_srce"], m["E_trgt"], m["E_face"], nF, a.to(r.device),
                                         m["A0"][:nF], m["P0"][:nF], mech, myo=m.get("myo")).to(r.device)
            if self.stretch_ref == "tissue":
                sig = sig - sig.median() + 1.0
            elif self.stretch_ref == "settle_median":
                # the tissue's median stress at `ref_frame`, subtracted from then on as a FIXED offset: a
                # settled pouch reads 1 (its line tension's standing compression removed), and a later
                # global compression or stretch is still seen -- unlike `tissue`, which removes it every frame
                # -- and NO signal before it (sigma = 1): an unsettled mesh's stress spans -0.7..1.8, which a
                # stretch-growth term turns into x14 area in 30 frames (the batch-9 CPU smoke)
                if self._off is None and getattr(self, "_k", 0) >= self.ref_frame:
                    self._off = (sig.median() - 1.0).detach()
                sig = sig - self._off if self._off is not None else torch.ones_like(sig)
            if self.smooth > 0:
                sig = face_neighbour_mean(sig, m["E_srce"], m["E_trgt"], m["E_face"], nF, self.smooth)
            if self._stretch_declared():
                from plexus.operators.vertex_ops import set_cell_block_t
                set_cell_block_t(self._H, self.cat, "stretch", sig.detach(), nF)
            if self.gain != 0.0:
                r = r * (1.0 + self.gain * (sig - 1.0)).clamp(0.0, self.f_max)
            if self.stretch_growth > 0:
                r = r + s_prev.to(r.dtype) * self.rate * self.stretch_growth * (sig - 1.0 - self.stretch_threshold).clamp(min=0.0)
            return r
        v = self._v_now[:nF].to(r.dtype).clamp(min=1e-12)
        if self.readout == "a0":
            from plexus.operators.vertex_ops import cell_block_t
            a = cell_block_t(self._H, self.cat, "area", nF)
            if a is None:
                raise ValueError("cell_grow[stretch] readout: a0 needs the `area` block (cell_geometry)")
            ratio = a.to(r.dtype).to(r.device).clamp(min=1e-12) / m["A0"][:nF].to(r.dtype).clamp(min=1e-12)
        elif self.readout in ("area", "thickness"):
            from plexus.operators.vertex_ops import cell_block_t
            a = cell_block_t(self._H, self.cat, "area", nF)
            if a is None:
                raise ValueError("cell_grow[stretch] readout: area needs the `area` block on the cell set "
                                 "(written by `cell_geometry`)")
            a = a.to(r.dtype).to(r.device).clamp(min=1e-12)
            ratio = a / v ** (2.0 / 3.0) if self.readout == "area" else a / v
        else:
            ratio = v / m["V0f"][:nF].to(r.dtype).clamp(min=1e-12)
        ref = ratio.median().clamp(min=1e-12)
        if self.stretch_ref in ("settle", "settle_median"):
            if self._ref is None and getattr(self, "_k", 0) >= self.ref_frame:
                self._ref = ref.detach().clone()
            ref = self._ref if self._ref is not None else ref
        sigma = ratio / ref
        if self.smooth > 0:
            sigma = face_neighbour_mean(sigma, m["E_srce"], m["E_trgt"], m["E_face"], nF, self.smooth)
        if self._stretch_declared():
            from plexus.operators.vertex_ops import set_cell_block_t
            set_cell_block_t(self._H, self.cat, "stretch", sigma.detach(), nF)
        if self.gain != 0.0:
            r = r * (1.0 + self.gain * (sigma - 1.0)).clamp(0.0, self.f_max)
        if self.stretch_growth > 0:
            # growth INDUCED by stretch above a threshold, wherever the cell is (AW 2007: the periphery,
            # with no growth factor, grows only when stretched), so it is s * rate, not the drive.
            r = r + s_prev.to(r.dtype) * self.rate * self.stretch_growth * (
                sigma - 1.0 - self.stretch_threshold).clamp(min=0.0)
        return r

    def _stretch_declared(self):
        H = getattr(self, "_H", None)
        if H is None or getattr(self, "cat", None) is None:
            return False
        return "stretch" in getattr(H.level(self.cat), "state_schema", {})



@register_operator("cell_grow", model="timer_planar", set="vertex", kind="lateral", family="population",
                   title="Growth to a target size",
                   equation=r"""$$\frac{d\ln V_j}{dt}=\frac{1}{T}\ln\frac{V_{\mathrm{target}}}{V_j},\qquad V\propto A\propto s^{2}$$""")
class Grow3DTimerPlanar(Grow3DTimer):
    """`timer`, for a MONOLAYER: a cell grows in area at constant height, so its volume target
    scales as s**2 like its area, not s**3.

    WHY IT MATTERS OVER MANY GENERATIONS. `cell_divide` halves both the area and the volume targets;
    isotropic regrowth that restores the volume by 2 restores the area by only 2**(2/3) = 1.59, so
    the area target falls by 0.79 a generation -- invisible over the few generations of a growth
    run, fatal over fifty of turnover. Measured on a 60-cell shell (exp 14, Finding 3): isotropic,
    the area target's median went 1.49 -> 0.22 in five cycles and extruding cells stalled; planar,
    both targets and the shell radius stayed put over 7.6 cycles.

    `hold_block: <block>` (exp 14 Phase 2): a cell whose width-1 cell block is above 0.5 relaxes, with
    the same time constant, to the MEAN VOLUME OF THE CELLS THAT ARE NOT HELD instead of to `vth_frac`
    times the reference -- a committed (B) basal cell, `fate` 1, which never divides again (Clayton et
    al. 2007) and is the size of its cycling neighbours. Measured on an 80-cell copy (600 frames): held
    at zero growth, B cells kept the half size they were born with and turnover halved the shell's
    radius; relaxed to their seeded size (scale 1), their target area still fell 1.46 -> 0.89 against
    ~1.35 for the cycling cells, whose targets run above seed through the cycle. A held cell SENTENCED
    TO DIE (`apop_flag` > 0) does not grow at all: `cell_die` is shrinking its targets, and this operator
    re-reads every cell's reference (`A0_init`, `V0f_init`) from the current targets whenever the cell
    count changes, so the pull back towards the cycling size started from a shrunken reference and drove
    one cell's area target to 26 x the median before it was removed (exp 14 P2.1, seed 1, row 941).
    Absent, the law is untouched.

    `size_block: <block>`, `size_factor: f` (exp 14 Phase 3): a cell whose block is above 0.5 aims for f
    times the volume it would otherwise aim for, on either path -- a mutant that is simply bigger, with no
    advantage in its fate or its cycle, so that any takeover is mechanical: it crowds its neighbours.
    The law adds ln(f) / (GROWTH_DIMS x cycle_frames) x s to the rate; a sentenced cell still does not grow.
    """
    GROWTH_DIMS = 2
    MECHANISM_TAGS = ["growth", "size_control", "timer", "target_size", "monolayer"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.hold_block = params.get("hold_block")
        self.size_block = params.get("size_block")
        self.size_factor = float(params.get("size_factor", 1.0))
        self._hold = None
        self._dying = None
        self._big = None

    def forward(self, H, mask=None):
        self._hold = self._dying = self._big = None
        if self.size_block and self.size_factor != 1.0:
            from plexus.operators.vertex_ops import cell_block, resolve_cell_set
            m = getattr(H.level(self.at), "_mesh", None)
            if m is not None:
                b = cell_block(H, resolve_cell_set(H, self.at, getattr(self, "_cat", None)),
                               self.size_block, int(m["nF"]))
                if b is not None:
                    self._big = torch.as_tensor(b > 0.5)
                    d = cell_block(H, resolve_cell_set(H, self.at, getattr(self, "_cat", None)),
                                   "apop_flag", int(m["nF"]))
                    self._dying = torch.as_tensor(d > 0) if d is not None else None
        if self.hold_block:
            from plexus.operators.vertex_ops import cell_block, resolve_cell_set
            m = getattr(H.level(self.at), "_mesh", None)
            if m is not None:
                nF = int(m["nF"])
                cat = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
                b = cell_block(H, cat, self.hold_block, nF)
                if b is not None:
                    self._hold = torch.as_tensor(b > 0.5)
                    d = cell_block(H, cat, "apop_flag", nF)
                    self._dying = torch.as_tensor(d > 0) if d is not None else None
        return super().forward(H, mask)

    def _rate(self, s_prev, hillv, m, v_ref):
        ds = super()._rate(s_prev, hillv, m, v_ref)
        if self._hold is not None and self._hold.shape[0] == ds.shape[0]:
            hold = self._hold.to(ds.device)
            v_now = (m["V0f_init"] * s_prev ** self.GROWTH_DIMS).clamp(min=1e-9)
            if bool((~hold).any()):
                # the same law with v_tgt = the cycling cells' mean volume
                v_tgt = v_now[~hold].mean()
                back = s_prev * torch.log(v_tgt / v_now) / (self.GROWTH_DIMS * max(self.cycle_frames, 1.0))
                ds = torch.where(hold, back, ds)
            if self._dying is not None and self._dying.shape[0] == ds.shape[0]:
                ds = torch.where(hold & self._dying.to(ds.device), torch.zeros_like(ds), ds)
        if self._big is not None and self._big.shape[0] == ds.shape[0]:
            big = self._big.to(ds.device)
            if self._dying is not None and self._dying.shape[0] == ds.shape[0]:
                big = big & ~self._dying.to(ds.device)
            extra = s_prev * float(np.log(self.size_factor)) / (self.GROWTH_DIMS * max(self.cycle_frames, 1.0))
            ds = torch.where(big, ds + extra, ds)
        return ds


@register_operator("interface_tension", set="vertex", kind="lateral", family="mechanics", title="Purse-string line tension",
                   equation=r"""$$E=K_{\mathrm{purse}}\!\!\sum_{e\in\mathrm{interface}}\!\!\ell_e,\qquad \mathbf f_v=-\frac{\partial E}{\partial\mathbf x_v}=-K_{\mathrm{purse}}\sum_{e\ni v}\frac{\mathbf x_v-\mathbf x_{\mathrm{other}}}{\ell_e}$$""")
class InterfaceLineTension3D(Lateral):
    """A purse-string line tension on the activator interface -- and nothing else.

    vertex -> vertex: reads pos and the cell set's chem, emits a velocity on the interface
    vertices, which the engine integrates beside the shape energy.

        E   = K_purse sum_{e in interface} l_e
        f_v = -dE/dx_v = -K_purse sum_{e ~ v} (x_v - x_other) / l_e

    The interface is the set of mesh edges separating an activator-high cell from a low one, with
    "high" meaning a > a_sw * a_max: a fraction of the field's own running maximum, because a
    threshold relative to the field cannot fall outside the field. l_e is the edge length in world
    units and K_purse the line tension, in force per unit length -- the same kind of term
    `cell_mechanics` already charges for. Shortening the ring costs less energy, so the ring
    closes, which is how a purse-string actually contracts.

    It carries only that term. An energy falling as activator-high cells move outward would not
    model a force; it would pay the tissue to produce the morphology a search is looking for. That
    term exists under its own name as `interface_push`, which the search vocabulary does not
    contain, so no setting of anything in the search space pushes.

    Reference: Plexus (this work); purse-string apical constriction after Okuda, S. et al. (2018).
    Sci. Rep. 8:2386.
    """
    SUPPORTED_DIMS = [3]; EMIT = "velocity"; DIFFERENTIABLE = True
    INPUTS = ["vertex", "cell"]; OUTPUTS = ["vertex"]; READS = ["pos", "chem"]; WRITES = ["pos"]
    MECHANISM_TAGS = ["interface_tension", "purse_string", "tube", "oriented", "cross_scale"]
    PARAM_ROLES = {"K_purse": "interface_line_tension", "a_sw": "red_threshold"}
    REFERENCE = "Plexus (this work); purse-string / apical-constriction tubulation after Okuda, S. et al. (2018). Sci. Rep. 8:2386."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex"); self._cat = params.get("cell_set")
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
        # THE PAIRING IS READ FROM THE SET, ONCE PER CALL -- see `resolve_cell_set`.
        self.cat = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
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
        # An ABSOLUTE threshold cannot work here, because the activator's own ceiling is whatever
        # the chemistry produces and that varies by orders of magnitude between parameter sets. A
        # threshold declared above the field selects nothing, the operator returns an empty delta
        # on every scheduled frame, and the ledger records
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


@register_operator("interface_push", set="vertex", kind="lateral", family="mechanics",
                   equation=r"""$$E=-K_{\mathrm{ext}}\!\!\sum_{j\,:\,a_j>a_{sw}}\!\! a_j\,r_j,\qquad \mathbf f_v=+K_{\mathrm{ext}}\,a_j\,\mathbf u_v$$""")
class ExtrusionForcing3D(Lateral):
    """The disqualified term, on its own and under its own name. A run carrying this is a control.

    vertex -> vertex: reads pos and the cell set's chem, emits an outward velocity on
    activator-high cells.

        E   = -K_extrude sum_{activator-high j} a_j r_j
        f_v = +K_extrude a_j u_v                       outward, along the radial direction

    a_j is the cell's activator concentration, dimensionless, and r_j the distance of its centroid
    from the tissue centre in world units; K_extrude is a force per unit concentration. The energy
    FALLS as activator-high cells move outward, so the tissue is paid, per frame, to do the thing a
    search is looking for.

    That is not a mechanism. Growth, division, adhesion and line tension are hypotheses about what
    cells DO; this is a hypothesis about what the experimenter WANTS, and any protrusion it
    produces is evidence about the term rather than about the tissue.

    It is deliberately absent from the search vocabulary, so no proposed edit can reach it. It
    exists so the forcing CAN be run when a control genuinely calls for one, and so that running it
    is an explicit act the record shows as such. Kept inside the tension operator, forcing would be
    a parameter of a sound mechanism, and a reader seeing a plausible name cannot check a term that
    is not in front of them.

    Same gate as the tension operator: a > a_sw * a_max.

    Reference: Plexus (this work) -- a forcing term retained only as an explicit control.
    """
    SUPPORTED_DIMS = [3]; EMIT = "velocity"; DIFFERENTIABLE = True
    INPUTS = ["vertex", "cell"]; OUTPUTS = ["vertex"]; READS = ["pos", "chem"]; WRITES = ["pos"]
    MECHANISM_TAGS = ["extrusion", "forcing", "control_only", "disqualified"]
    PARAM_ROLES = {"K_extrude": "normal_extrusion_forcing", "a_sw": "red_threshold"}
    REFERENCE = "Plexus (this work) -- a forcing term retained only as an explicit control."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex"); self._cat = params.get("cell_set")
        self.K_extrude = float(params.get("K_extrude", 0.5))
        self.a_sw = float(params.get("a_sw", 0.6)); self.eta = float(params.get("eta", 0.05))
        self.cap_frac = float(params.get("cap_frac", 0.10)); self.iters = int(params.get("iters", 4))

    def forward(self, H, mask=None):
        # THE PAIRING IS READ FROM THE SET, ONCE PER CALL -- see `resolve_cell_set`.
        self.cat = resolve_cell_set(H, self.at, getattr(self, "_cat", None))
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
            _, _, centroid, _ = face_geometry_3d(x, es, et, ef, nF)
            cdir = centroid / (centroid.norm(dim=-1, keepdim=True) + 1e-9)
            force.index_add_(0, es, (redpush[ef])[:, None] * cdir[ef] / 3.0)
            x = x + (self.eta * force).clamp(-cap, cap)
        vel = torch.zeros_like(x0)
        vel[:Nv] = (x - x0[:Nv]) / max(float(getattr(H.config, "dt", 1.0)), 1e-6)
        occ = vlvl.occ[:, None] if getattr(vlvl, "occ", None) is not None else 1.0
        return {self.at: vel * occ}


@register_operator("cell_chem_react", set="cell", kind="lateral", family="fields", model="brusselator", title="Activator-inhibitor reaction",
                   species=(("a", "activator"), ("h", "inhibitor")),
                   equation=r"""$$\frac{da}{dt}=\gamma\big(A-(B+1)a+a^{2}h\big),\qquad \frac{dh}{dt}=\gamma\big(B\,a-a^{2}h\big)$$""")
class CellReactBrusselator(Lateral):
    """The Brusselator: the textbook activator-inhibitor system, and the one whose Turing
    condition is exactly solvable, so whether a pattern is possible can be checked before running.

    cell -> cell: reads chem, emits d(chem)/dt. Local; no relation traversed. chem = [a, h].

        da/dt = gamma ( A - (B + 1) a + a^2 h )      a = activator
        dh/dt = gamma ( B a - a^2 h )                h = inhibitor

    a and h are dimensionless concentrations. A is the constant feed of activator and B the rate at
    which activator is converted into inhibitor, both in inverse time; gamma is a dimensionless
    rescaling of the whole reaction. The homogeneous steady state is (a, h) = (A, B/A), and it is
    Turing-unstable for B > 1 + A^2 -- so the two parameters say in advance whether a pattern is
    possible at all, which is what makes this the right model to certify a diffusion operator
    against.

    Pair it with the `noise` seed, which starts exactly at that steady state and perturbs it, and
    with a diffusion ratio putting the inhibitor well above the activator. Patterning then comes
    from fluctuation alone, which is the strictest test that it is emergent.

    Reference: Prigogine, I. & Lefever, R. (1968). Symmetry breaking instabilities in dissipative
    systems II. J. Chem. Phys. 48:1695-1700.
    """
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["gamma", "A", "B"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]
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
                   model="curvature", title="Chemistry driven by curvature",
                   equation=r"""$$H_j=\frac{2\big(\overline{\mathbf x}_k-\mathbf x_j\big)\cdot\mathbf n_j}{d_j^{2}}$$""")
class ShapeToChemCurvature(_ShapeToChemBase):
    """The chemistry listens to CURVATURE: the tissue's shape telling its cells where they are.

        H_j = 2 (mean_k(x_k) - x_j) . n_j / d_j^2

    over the neighbours k of cell j, where x is a cell centroid, n_j the cell's own outward normal
    and d_j the mean centroid spacing, all in world units. H is therefore in inverse world units:
    positive where the sheet bulges outward, negative in a dimple, and about 1/R on a sphere of
    radius R.

    A proxy for the mean curvature rather than the cotangent-Laplacian form, which is why it is
    certified against spheres of known radius in the self-test below rather than asserted.

    Reference: Okuda, S. et al. (2018). Sci. Rep. 8:2386 (the shape-chemistry loop this closes).
    """
    MECHANISM_TAGS = _ShapeToChemBase.MECHANISM_TAGS + ["curvature_sensing"]

    def _feature(self, pt, m, es, et, ef, nF):
        area, _, centroid, _ = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(es),
                                           torch.as_tensor(et), torch.as_tensor(ef), nF)
        centroid = centroid.numpy()
        nrm = np.zeros((nF, 3))                        # Newell normal per cell, outward
        for a, b, f in zip(es, et, ef):
            nrm[f] += np.cross(pt[a] - centroid[f], pt[b] - centroid[f])
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = nrm / np.maximum(ln, 1e-12)
        src, dst = _cell_adjacency(es, et, ef, nF)
        if not len(src):
            return None
        deg = np.bincount(src, minlength=nF).astype(float)
        nb = np.zeros((nF, 3))
        for d in range(3):
            nb[:, d] = np.bincount(src, weights=centroid[dst][:, d], minlength=nF)
        nb /= np.maximum(deg, 1)[:, None]
        delta = nb - centroid                                # umbrella vector
        # Divide by the NEIGHBOUR SPACING squared, not by |delta|^2. On a sphere the tangential
        # parts of the umbrella cancel, so |delta| is itself only ~L^2/2R -- dividing by it gives
        # 2R/L^2, which GROWS with radius. That reads as 1/R only if you hold the cell count fixed
        # so that L scales with R, which is exactly how the first version of this passed its own
        # test. With the spacing: delta.n = -L^2/2R, so H = 2 (delta.n) / L^2 = 1/R. Correct, and
        # now independent of how finely the sphere is meshed.
        sp = np.zeros(nF)
        np.add.at(sp, src, np.linalg.norm(centroid[dst] - centroid[src], axis=1))
        L = sp / np.maximum(deg, 1)
        return -2.0 * (delta * nrm).sum(1) / np.maximum(L ** 2, 1e-12)


@register_operator("cell_chem_from_shape", set="cell", kind="lateral", family="fields",
                   model="tension", title="Chemistry driven by curvature")
class ShapeToChemTension(_ShapeToChemBase):
    """The chemistry listens to CORTICAL TENSION: mechanotransduction.

        tension_j = 2 kP (P_j - P0_j) + Gamma P_j + Lambda

    the same quantity `cell_mechanics` charges for. P_j is the cell's perimeter and P0_j its
    target, both in world units; kP is the perimeter elasticity, Gamma the contractility and
    Lambda the line tension, so tension is a force. It is the best-evidenced feedback in real
    epithelia -- YAP/TAZ translocates to the nucleus under tension and Piezo1 is a stretch-gated
    channel -- so "tense cells signal differently" is not a modelling convenience.

    Needs the mechanical target P0, which exists only once a mechanics operator has run.

    Reference: Dupont, S. et al. (2011). Role of YAP/TAZ in mechanotransduction. Nature
    474:179-183.
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
                   model="apical_area", title="Chemistry driven by curvature")
class ShapeToChemApicalArea(_ShapeToChemBase):
    """The chemistry listens to APICAL AREA: crowding and density sensing.

        feature_j = A_j / A0_j        where a target area exists
        feature_j = A_j               otherwise

    A_j is the cell's apical area in world units squared and A0_j its target. Reporting the RATIO
    where a target exists is what makes a uniformly grown tissue read as unstretched; an absolute
    area would rise everywhere as the tissue grows and report growth as crowding.

    The most direct reading of "am I stretched or am I crowded", and the cheapest: no mechanical
    targets are required, only geometry.

    Reference: Okuda, S. et al. (2018). Sci. Rep. 8:2386 (the shape-chemistry loop this closes).
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
                   model="pressure", title="Chemistry driven by curvature")
class ShapeToChemPressure(_ShapeToChemBase):
    """The chemistry listens to VOLUME-ELASTIC PRESSURE.

        pressure_j = 2 kV (V0_j - v_j)

    positive when a cell is BELOW its target volume, i.e. squeezed. v_j is the cell's volume and
    V0_j its target, both in world units cubed, and kV the volume elasticity. It is the quantity
    that says whether a tissue is in compression, which a movie of positions does not show.

    Needs the mechanical target V0f.

    Reference: Okuda, S. et al. (2018). Sci. Rep. 8:2386 (the shape-chemistry loop this closes).
    """
    MECHANISM_TAGS = _ShapeToChemBase.MECHANISM_TAGS + ["pressure_sensing", "compression_sensing"]

    def _feature(self, pt, m, es, et, ef, nF):
        if "V0f" not in m:
            return None
        _, _, _, vf = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(es),
                                       torch.as_tensor(et), torch.as_tensor(ef), nF,
                                       apex=wedge_apex(m, torch.as_tensor(pt)))
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
        _, _, centroid, _ = face_geometry_3d(torch.as_tensor(w), torch.as_tensor(es),
                                        torch.as_tensor(et), torch.as_tensor(ef), nF)
        top = centroid.numpy()[:, 2] > 0.90 * np.linalg.norm(centroid.numpy(), axis=1)
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
    # It certifies the ARITHMETIC INSIDE the operator, and every line of it passes -- which is
    # exactly why it is not enough on its own. An operator whose curvature reads 1/R on spheres of
    # known radius, whose bump is positive and whose dimple negative, can still never reach the
    # state at all, and every check above would still be green. The three below test `forward`.
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


# `_np` is defined once, above, and shared by both the shape-to-chemistry and the shape-probe
# operators.


class _ShapeProbeBase(Lateral):
    """Compute one scalar per cell and publish it on the mesh under `field`. No state is touched."""
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False
    INPUTS = ["cell", "vertex"]; OUTPUTS = []; READS = ["pos"]; WRITES = []
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
                   model="shape_index", title="Shape index", probe=True,
                   equation=r"""$$q_j=\frac{P_j}{\sqrt{A_j}}$$""")
class ShapeIndexProbe(_ShapeProbeBase):
    """The dimensionless shape index: what the vertex model itself minimises towards p0, and
    the tissue's own order parameter for rigid against fluid.

        q_j = P_j / sqrt(A_j)

    P_j is the cell's perimeter and A_j its area, both in world units, so q is dimensionless and
    reads about 3.72 for a regular hexagon. Below roughly 3.81 the tissue is rigid and above it
    fluid -- the rigidity transition -- so this one number says which phase a simulated epithelium
    is in.

    Reference: Bi, D., Lopez, J. H., Schwarz, J. M. & Manning, M. L. (2015). A density-independent
    rigidity transition in biological tissues. Nat. Phys. 11:1074-1079.
    """

    def _measure(self, pos, m, es, et, ef, nF):
        pt = torch.as_tensor(pos)
        area, perim, _cen, _vf = face_geometry_3d(
            pt, torch.as_tensor(es), torch.as_tensor(et), torch.as_tensor(ef), nF, apex=wedge_apex(m, pt))
        a = _np(area)[:nF]; p = _np(perim)[:nF]
        out = np.full(nF, np.nan)
        ok = a > 1e-12
        out[ok] = p[ok] / np.sqrt(a[ok])
        return out


@register_operator("cell_shape_probe", set="cell", kind="lateral", family="hierarchy",
                   model="aspect", title="Shape index", probe=True)
class AspectProbe(_ShapeProbeBase):
    """"Thin and elongated" as a number: the aspect ratio of the cell's own vertex ring.

        aspect_j = sqrt(lambda_1 / lambda_2)

    lambda_1 >= lambda_2 >= lambda_3 are the eigenvalues of the covariance of the ring's vertex
    positions in 3D, so the ratio is dimensionless and equals 1 for a circular cell.

    The ratio is taken between the FIRST TWO eigenvalues, not the first and the last. A cell on a
    curved shell is a nearly flat patch, so its third eigenvalue is the sheet's thickness and is
    small for every cell, elongated or not; using it would report the whole tissue as extreme and
    rank nothing.

    Reference: none -- a shape descriptor, not a mechanism. Plexus (this work).
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
