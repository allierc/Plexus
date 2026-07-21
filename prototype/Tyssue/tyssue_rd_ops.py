"""tyssue_rd_ops -- live Turing reaction-diffusion ON the cell set (Goal 2). Forked from the
Turing_vertex prototype so the two can evolve separately, and adapted to run on the tyssue AVM cell
set: the morphogen `chem`=[a,u] (activator, substrate) lives as CELL state, diffuses on the
cell-cell adjacency (two cells are neighbours iff they share a mesh edge), and reacts by Gray-Scott.
This replaces the IMPOSED morphogen bump (cell_morphogen) with a SELF-ORGANISED activator -- the
substrate for morphogen-driven growth -> budding.

Operators:
  cell_adjacency (rewire)     -- build the cell-cell neighbour graph from the half-edge table
  cell_rd_seed   (structural) -- Gray-Scott initial condition (substrate=1, a central activator spot)
  cell_diffuse   (lateral)    -- graph-Laplacian diffusion of chem on the cell adjacency (EMIT=velocity)
  cell_react     (lateral)    -- Gray-Scott autocatalysis (EMIT=velocity)
  cell_geometry_3d (aggregate)-- vertices -> per-cell centroid/area
  morphogen_growth_3d (growth)-- grow each cell's target volume by a Hill fn of its activator -> BUDDING
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


@register_operator("cell_rd_seed", set="cell", kind="structural", family="growth")
class CellRDSeed(Structural):
    """Gray-Scott initial condition on the cell set: substrate u=1 everywhere, activator a=0 except
    a central spot (a=0.5, u=0.25) that nucleates the pattern. chem = [a, u]."""
    SUPPORTED_DIMS = [2, 3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["initial_condition", "gray_scott"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell"); self.vat = params.get("vertex_set", "vertex")
        self.seed = int(params.get("seed", 0))
        self.mode = params.get("mode", "scatter")               # "noise" | "scatter" | "patch" (localized source)
        self.seed_frac = float(params.get("seed_frac", 0.06))   # (scatter) fraction of strong activator seeds
        self.A = float(params.get("A", 1.0)); self.B = float(params.get("B", 3.0))   # (noise) steady state (A, B/A)
        self.noise = float(params.get("noise", 0.04))
        self.patch_z = float(params.get("patch_z", 0.6))        # (patch) activate cells with cen_z > patch_z x z_max

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
        elif self.mode == "noise":                              # Brusselator: homogeneous steady state + noise
            a = (self.A + self.noise * torch.randn(nF, generator=g)).to(dev)
            u = (self.B / self.A + self.noise * torch.randn(nF, generator=g)).to(dev)
        else:                                                   # Gray-Scott: substrate=1 + scattered activator nuclei
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


@register_operator("cell_diffuse", set="cell", kind="lateral", family="fields")
class CellDiffuse(Lateral):
    """Graph-Laplacian diffusion of the two morphogens between neighbouring cells (forked from
    Turing_vertex `graph_diffuse`). `norm=True` uses the degree-normalised Laplacian (eigenvalues in
    [-2,0]) so an explicit step is stable at any cell degree. First-order -> EMIT=velocity into chem."""
    SUPPORTED_DIMS = [2, 3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["d_a", "d_h", "chi"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["chem"]; WRITES = ["chem"]; MAPS = ["edge_index"]
    MECHANISM_TAGS = ["diffusion", "graph_laplacian", "turing"]
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


@register_operator("cell_react", set="cell", kind="lateral", family="fields", implementation="gray_scott")
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


@register_operator("morphogen_growth_3d", set="vertex", kind="structural", family="growth")
class MorphogenGrowth3D(Structural):
    """Morphogen-driven growth: each cell's targets grow at a per-cell rate set by a Hill function of
    its activator (read from the cell set), so the shell BULGES where the Turing activator is high ->
    self-organised budding/coral. A cross-set coupling (reads cell.chem, writes the vertex mesh targets
    A0/P0/v_eq); the per-cell volume elasticity then inflates the activated cells by force balance.
    Uniform vesicle_growth is the a_sw->0 limit."""
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = False
    MECHANISM_TAGS = ["growth", "morphogen_driven", "budding", "cross_scale"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex"); self.cat = params.get("cell_set", "cell")
        self.rate = float(params.get("rate", 0.01)); self.a_sw = float(params.get("a_sw", 0.20))
        self.hill = float(params.get("hill", 3.0)); self.cap = float(params.get("cap", 2.5))
        self.every = int(params.get("every", 1)); self._k = 0

    def forward(self, H, mask=None):
        vlvl = H.level(self.at); m = getattr(vlvl, "_mesh", None)
        if m is None:
            return {}
        self._k += 1
        if self._k % self.every != 0:
            return {}
        clvl = H.level(self.cat)
        if "chem" not in clvl.state_schema:
            return {}
        nF = m["nF"]; h0, _ = clvl.state_schema["chem"]
        dev = m["V0f"].device
        a = clvl.state[:nF, h0].detach().to(dev)                 # per-cell activator
        if "mg_scale" not in m or m["mg_scale"].shape[0] != nF:  # per-cell cumulative linear scale (capped)
            m["mg_scale"] = torch.ones(nF, device=dev, dtype=m["V0f"].dtype)
            m["A0_init"] = m["A0"].clone(); m["P0_init"] = m["P0"].clone(); m["V0f_init"] = m["V0f"].clone()
        hillv = a ** self.hill / (self.a_sw ** self.hill + a ** self.hill + 1e-12)   # Hill activation in [0,1]
        s = torch.clamp(m["mg_scale"] * (1.0 + self.rate * hillv), max=self.cap)     # grow while activated, capped
        m["mg_scale"] = s
        m["A0"] = m["A0_init"] * (s * s)                         # keep A0/P0/v_eq consistent (area~R^2, vol~R^3)
        m["P0"] = m["P0_init"] * s
        m["V0f"] = m["V0f_init"] * (s ** 3)
        m["V0"] = float(m["V0f"].sum())
        return {}


@register_operator("cell_react", set="cell", kind="lateral", family="fields", implementation="brusselator")
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
