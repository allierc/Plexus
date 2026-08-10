"""tyssue_rd_ops -- live Turing reaction-diffusion ON the cell set (Goal 2). Forked from the
Turing_vertex prototype so the two can evolve separately, and adapted to run on the tyssue AVM cell
set: the morphogen `chem`=[a,u] (activator, substrate) lives as CELL state, diffuses on the
cell-cell adjacency (two cells are neighbours iff they share a mesh edge), and reacts by Gray-Scott.
This replaces the IMPOSED morphogen bump (cell_morphogen) with a SELF-ORGANISED activator -- the
substrate for morphogen-driven growth -> budding.

Operators:
  cell_adjacency (rewire)     -- build the cell-cell neighbour graph from the half-edge table
  seed_cell_rd   (structural) -- Gray-Scott initial condition (substrate=1, a central activator spot)
  cell_diffuse   (lateral)    -- graph-Laplacian diffusion of chem on the cell adjacency (EMIT=velocity)
  cell_react     (lateral)    -- Gray-Scott autocatalysis (EMIT=velocity)
  cell_geometry_3d (aggregate)-- vertices -> per-cell centroid/area
  grow_3d (growth)-- grow each cell's target volume by a Hill fn of its activator -> BUDDING
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from plexus.models.base import Aggregate, Lateral, Rewire, Structural
from plexus.models.registry import register_operator


@register_operator("cell_geometry_3d", set="cell", kind="aggregate", family="hierarchy")
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
        from tyssue_ops3d import face_geometry_3d
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


@register_operator("cell_adjacency", set="cell", kind="rewire", family="topology")
class CellAdjacency(Rewire):
    """Two cells are neighbours iff they share a mesh edge. Build that graph from the half-edge
    table on the vertex Level and store it as `edge_index` on the cell Level -- the graph the
    reaction-diffusion runs on. (Rebuilt each call so it tracks T1/division if they run.)"""
    SUPPORTED_DIMS = [2, 3]; DIFFERENTIABLE = False
    MECHANISM_TAGS = ["cell_adjacency", "neighbour_graph"]
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


@register_operator("seed_cell_rd", set="cell", kind="seed", family="growth")
class CellRDSeed(Structural):
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
        to `chem` can accumulate anything. `shape_to_chem` writes to channel 1 and `tip` sets that
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
    MODES = ("scatter", "noise", "patch", "cones")
    REFERENCE = "Plexus (this work); cone seeding after Okuda, S. et al. (2018). Sci. Rep. 8:2386."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell"); self.vat = params.get("vertex_set", "vertex")
        self.seed = int(params.get("seed", 0))
        self.mode = params.get("mode", "scatter")               # "noise" | "scatter" | "patch" | "cones"
        if self.mode not in self.MODES:
            raise ValueError(
                f"seed_cell_rd: unknown mode {self.mode!r}; known: {list(self.MODES)}. "
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
        clvl = H.level(self.at); vlvl = H.level(self.vat); m = getattr(vlvl, "_mesh", None)
        if m is None or "chem" not in clvl.state_schema:
            return {}
        nF = m["nF"]; dev = clvl.state.device
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
        h0, h1 = clvl.state_schema["chem"]
        st = clvl.state.clone(); st[:nF, h0:h0 + 1] = a[:, None]
        if h1 - h0 > 1:
            st[:nF, h0 + 1:h0 + 2] = u[:, None]
        clvl.state = st
        return {}


@register_operator("cell_diffuse", set="cell", kind="lateral", family="fields", implementation="graph_laplacian")
class CellDiffuse(Lateral):
    """`graph_laplacian` implementation of cell_diffuse: PURELY COMBINATORIAL diffusion of the two
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
    REQUIRES_PARAMS = ["d_a", "d_h", "chi"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]; MAPS = ["edge_index"]
    MECHANISM_TAGS = ["diffusion", "graph_laplacian", "turing"]
    REFERENCE = "Fick, A. (1855). Ueber Diffusion. Ann. Phys. 170:59-86; Turing, A. M. (1952). Phil. Trans. R. Soc. B 237:37-72."
    PARAM_ROLES = {"d_a": "activator_diffusivity", "d_h": "substrate_diffusivity", "chi": "spatial_scale"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.d_a = float(params["d_a"]); self.d_h = float(params["d_h"]); self.chi = float(params["chi"])
        self.norm = bool(params.get("norm", True))

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
        coef = torch.tensor([self.d_a, self.d_h], device=chem.device, dtype=chem.dtype) * self.chi
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: (coef[None, :] * lap) * occ}


@register_operator("cell_diffuse", set="cell", kind="lateral", family="fields", implementation="interface_weighted")
class CellDiffuseInterfaceWeighted(Lateral):
    """`interface_weighted` implementation of cell_diffuse -- the OKUDA finite-volume form, and the
    MISSING HALF of the chemistry<->shape coupling. Same contract as `graph_laplacian` (set=cell,
    kind=lateral, family=fields, EMIT=velocity into chem); only the numerics differ.

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
        exactly what seed_mesh_3d stores as the target V0f and what shape_energy_3d's K_V term
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
        handled structurally by grow_3d's conserve_amount, so applying it here too would
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
        from tyssue_ops3d import face_geometry_3d, ShapeEnergy3D
        lvl = H.level(self.at); vlvl = H.level(self.vat)
        chem = lvl.get("chem")
        m = getattr(vlvl, "_mesh", None)
        if m is None:
            # NOT a silent geometry fallback: the mesh simply does not exist yet (seed_mesh_3d has not
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


@register_operator("cell_react", set="cell", kind="lateral", family="fields", model="gray_scott")
class CellReactGrayScott(Lateral):
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

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        a = chem[:, 0]; u = chem[:, 1]
        uaa = u * a * a
        da = uaa - (self.F + self.kk) * a
        du = -uaa + self.F * (1.0 - u)
        occ = lvl.occ[:, None] if getattr(lvl, "occ", None) is not None else 1.0
        return {self.at: self.rate * torch.stack([da, du], dim=1) * occ}


@register_operator("cell_react", set="cell", kind="lateral", family="fields", model="gierer_meinhardt")
class CellReactGiererMeinhardt(Lateral):
    """Gierer-Meinhardt activator(a)-inhibitor(h) -- the RD OKUDA uses (ref 37). chem = [a, h]:
        da/dt = gm_rho * a^2/h - mu_a * a + a0     (SELF-ENHANCING activator: the a^2/h AUTOCATALYSIS is the
        dh/dt = gm_rho * a^2   - mu_h * h            amplification feedback that self-maintains a localised PEAK)
    Paired (in cell_diffuse) with a FAST inhibitor (d_h >> d_a via chi) -> lateral inhibition -> a stable
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


# `grow_3d`, AND THE OLD NAME IS GONE RATHER THAN ALIASED. Cedric, 8 August: "I always found
# grow_3d misleading -- is it morphogen, is it growth?" and then, on the alias:
# "I'm not a fan of alias and backward compatibility, this makes everything intricated and not
# readable. I prefer modifying prior spec files. Simplicity needs erasing here."
#
# It is growth. It does not produce a morphogen, it READS one and uses it as a per-cell rate: the
# morphogen is a GATE on this operator, and the composition space already declares that gate as an
# optional slot. With the gate open (`a_sw = 0`) the same operator is plain uniform growth. Naming
# the gate in the operator made the optional half look mandatory, and made the sibling pair
# unreadable -- `grow_3d` / `divide_3d` says what the schedule actually does.
@register_operator("grow_3d", set="vertex", kind="structural", family="growth")
class Grow3D(Structural):
    """Cell growth on the vesicle: each cell's targets (A0 / P0 / v_eq) grow at a per-cell rate,
    and the per-cell volume elasticity in shape_energy_3d then inflates the cell by force balance.
    This operator moves no vertex itself -- it raises what the cells ASK for, and the mechanics
    decides whether they get it.

    THE RATE IS `rate * (rho + Hill(activator))`, which is one operator covering both regimes:
      rho = 1, a_sw = 0   uniform growth, every cell at the same rate (what `grow_3d` did)
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
        self.hill = float(params.get("hill", 3.0)); self.cap = float(params.get("cap", 2.5))
        from tyssue_ops3d import _engine_owns_clock
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
        """The RATE LAW, and the only thing a `model=` variant of grow_3d changes.

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
        # also why `grow_3d` had to exist as a separate operator. a = 0 is the honest reading
        # of "there is no activator here"; the Hill term evaluates to 0 and the rho baseline stands.
        if "chem" in clvl.state_schema:
            h0, _h1 = clvl.state_schema["chem"]
            a = clvl.state[:nF, h0].detach().to(dev)             # per-cell activator
        else:
            a = torch.zeros(nF, device=dev, dtype=m["V0f"].dtype)
        if "mg_scale" not in m or m["mg_scale"].shape[0] != nF:  # per-cell cumulative linear scale (capped)
            m["mg_scale"] = torch.ones(nF, device=dev, dtype=m["V0f"].dtype)
            m["A0_init"] = m["A0"].clone(); m["P0_init"] = m["P0"].clone(); m["V0f_init"] = m["V0f"].clone()
        hillv = a ** self.hill / (self.a_sw ** self.hill + a ** self.hill + 1e-12)   # Hill activation in [0,1]
        s_prev = m["mg_scale"]                                    # per-cell scale BEFORE this tick (for the dilution rate)
        v_ref = float(m.get("v_ref", 1.0))                        # SEED-TIME MEDIAN cell volume (tyssue_ops3d:220)
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
        # THE SHELL RADIUS MUST GROW WITH THE CELLS. shape_energy_3d carries a radial spring,
        #     E += K_R * sum_i (|x_i| - R0)^2                 (tyssue_ops3d.py:85)
        # and R0 is set once at seeding (:217). `grow_3d` rescales it (:409); this operator
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


# =========================================================== grow_3d: the size-control models
# Three mechanisms from Ginzberg, Kafri & Kirschner, "On being the right (cell) size", Science 348
# (2015) and Ginzberg et al., eLife 7:e26957 (2018). They are `model=` variants, not replacements,
# because each is a different biological hypothesis at the same slot and the search should put
# them side by side. The default `grow_3d` above has NO size control at all -- its rate reads the
# morphogen and never the cell's own size -- which by the review's Eq. 2 means cell-size variance
# can only ever increase, and in this campaign's basis it does: vol_cv 0.160 -> 0.53.
#
# WHY THIS MATTERS BEYOND TIDINESS. The review's Fig 2 sets a healthy mammary epithelium, uniform
# in cell size, beside a pleomorphic tumour that is not, and states that "pleomorphism ... is a
# histological characteristic of many malignant lesions". This project is trying to grow an
# epithelial TUBE -- a coherent structure -- and its tissue drifts toward the second picture.


@register_operator("grow_3d", model="sizer", set="vertex", kind="structural", family="growth")
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


@register_operator("grow_3d", model="balance", set="vertex", kind="structural", family="growth")
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


@register_operator("grow_3d", model="timer", set="vertex", kind="structural", family="growth")
class Grow3DTimer(Grow3D):
    """Grow at whatever rate lands the cell on its target size after `cycle_frames` frames.

    The partner of `divide_3d model: timer`: if division fires on the clock, growth has to be the
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


@register_operator("interface_line_tension_3d", set="vertex", kind="lateral", family="mechanics")
class InterfaceLineTension3D(Lateral):
    """A PURSE-STRING line tension on the RED/WHITE activator interface -- and NOTHING ELSE.

    SPLIT FROM `rd_interface_tension` ON 10 AUGUST, and the split is the point. That operator carried
    two terms under one name:

        E = K_purse * Sigma_iface l_e   -   K_extrude * Sigma_red a*r
            [___ ordinary physics ___]       [_ the answer written into the objective _]

    The first is a line tension on the interface ring: real vertex-model mechanics, the same kind of
    term `shape_energy_3d` already charges for, and how a purse-string actually works. The second is
    an energy that FALLS as red cells move outward -- it does not model a force, it pays the tissue
    to produce the morphology the campaign is searching for. A run carrying it can only be a control.

    ONE NAME OVER BOTH TERMS COST FOUR ROUNDS. `K_extrude` measured 0.0 in all 78 specs that have
    ever carried this operator, so nothing the campaign ran was ever forced -- and the Grounder
    still reported r028 as "the same extrude-forced star for a fourth round", on three runs
    (`r028_00`, `03`, `06`) whose specs contain no such operator at all. `user_input.md` section 3
    had already told it to retract exactly that verdict about `r017_07`. A reader who sees a
    plausible name cannot check a term that is not in front of them, so the terms are now two
    operators: this one, and `extrusion_forcing_3d` below, which the loop vocabulary does not
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
        from tyssue_ops3d import ShapeEnergy3D
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
        # next parent could put it out of reach again. `shape_to_chem` in this same repo
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


@register_operator("extrusion_forcing_3d", set="vertex", kind="lateral", family="mechanics")
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
        from tyssue_ops3d import face_geometry_3d
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


@register_operator("cell_react", set="cell", kind="lateral", family="fields", model="brusselator")
class CellReactBrusselator(Lateral):
    """`brusselator` implementation of cell_react (transposed verbatim from Turing_vertex fig4_coral),
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

