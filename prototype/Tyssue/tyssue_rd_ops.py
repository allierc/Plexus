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


@register_operator("cell_rd_seed", set="cell", kind="structural", family="growth")
class CellRDSeed(Structural):
    """Gray-Scott initial condition on the cell set: substrate u=1 everywhere, activator a=0 except
    a central spot (a=0.5, u=0.25) that nucleates the pattern. chem = [a, u]."""
    SUPPORTED_DIMS = [2, 3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["initial_condition", "gray_scott"]
    REFERENCE = "Plexus (this work); cone seeding after Okuda, S. et al. (2018). Sci. Rep. 8:2386."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell"); self.vat = params.get("vertex_set", "vertex")
        self.seed = int(params.get("seed", 0))
        self.mode = params.get("mode", "scatter")               # "noise" | "scatter" | "patch" (localized source)
        self.seed_frac = float(params.get("seed_frac", 0.06))   # (scatter) fraction of strong activator seeds
        self.A = float(params.get("A", 1.0)); self.B = float(params.get("B", 3.0))   # (noise) steady state (A, B/A)
        self.noise = float(params.get("noise", 0.04))
        self.patch_z = float(params.get("patch_z", 0.6))        # (patch) activate cells with cen_z > patch_z x z_max
        self.n_spots = int(params.get("n_spots", 5))            # (cones) number of fixed radial activation foci
        self.cone_deg = float(params.get("cone_deg", 18.0))     # (cones) half-angle of each activation cone
        self.seed_dir = params.get("seed_dir", None)            # (cones, n_spots=1) override the cone axis to a fixed
        #   direction -> aim the tube where we want it (e.g. FRONT of the render camera at elev18/azim30 ~ (.82,.48,.31))
        self.tip_radius = float(params.get("tip_radius", 2.0))  # (tip mode) 3D radius of the tip-tracking activation cap

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
        elif self.mode == "tip":                                # TIP-TRACKING: a fixed-SIZE cap riding the advancing
            a = torch.full((nF,), 0.02, device=dev)             # tip -> CONSTANT-diameter extension (a fixed-angle cone
            if "cen" in clvl.state_schema:                      # widens with radius -> fat lobe; a fixed 3D-size cap
                ci0, ci1 = clvl.state_schema["cen"]; cen = clvl.state[:nF, ci0:ci0 + 3]
                if self.seed_dir is not None:                   # position ALONG the bud axis; tip = the furthest cell
                    sd = np.asarray(self.seed_dir, float); sd = sd / (np.linalg.norm(sd) + 1e-12)
                    proj = cen @ torch.as_tensor(sd, dtype=cen.dtype, device=dev)
                else:
                    proj = cen.norm(dim=1)
                tip = cen[int(torch.argmax(proj))]              # current outermost cell (the advancing tip)
                d3 = (cen - tip).norm(dim=1)                     # 3D distance from the tip (FIXED size, not angle)
                a = torch.where(d3 < self.tip_radius, torch.ones(nF, device=dev), a)
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


@register_operator("cell_react", set="cell", kind="lateral", family="fields", implementation="gierer_meinhardt")
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


@register_operator("morphogen_growth_3d", set="vertex", kind="structural", family="growth")
class MorphogenGrowth3D(Structural):
    """Morphogen-driven growth: each cell's targets grow at a per-cell rate set by a Hill function of
    its activator (read from the cell set), so the shell BULGES where the Turing activator is high ->
    self-organised budding/coral. A cross-set coupling (reads cell.chem, writes the vertex mesh targets
    A0/P0/v_eq); the per-cell volume elasticity then inflates the activated cells by force balance.
    Uniform vesicle_growth is the a_sw->0 limit."""
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = False
    MECHANISM_TAGS = ["growth", "morphogen_driven", "budding", "cross_scale"]
    REFERENCE = "Okuda, S. et al. (2018). Combining Turing and 3D vertex models reproduces autonomous multicellular morphogenesis of the tissue. Sci. Rep. 8:2386."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex"); self.cat = params.get("cell_set", "cell")
        self.rate = float(params.get("rate", 0.01)); self.a_sw = float(params.get("a_sw", 0.20))
        self.hill = float(params.get("hill", 3.0)); self.cap = float(params.get("cap", 2.5))
        self.every = int(params.get("every", 1)); self._k = 0
        # OKUDA uniform-cell mode: growth rate lambda = rate*(rho + Hill(a)); rho = baseline so ALL cells
        # cycle (the activator sets the RATE, not the size), and v_eq is capped at vth_frac*v_ref so every
        # cell oscillates in [~2/3, vth_frac]*v_ref -> uniform. rho=0 (default) = legacy activator-only bulge.
        self.rho = float(params.get("rho", 0.0)); self.vth_frac = float(params.get("vth_frac", 1.35))
        # OKUDA Appendix A: the morphogen is the AMOUNT m_j (conserved within the cell); the concentration
        # c_j=m_j/v_j is only READ by the kinetics. We store c_j, so growing v_j must DILUTE c_j to conserve
        # amount (else we silently CREATE mass each step -> spuriously feeds the tip). On (default) = correct.
        self.conserve_amount = bool(params.get("conserve_amount", True))

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
        nF = m["nF"]; h0, h1 = clvl.state_schema["chem"]
        dev = m["V0f"].device
        a = clvl.state[:nF, h0].detach().to(dev)                 # per-cell activator
        if "mg_scale" not in m or m["mg_scale"].shape[0] != nF:  # per-cell cumulative linear scale (capped)
            m["mg_scale"] = torch.ones(nF, device=dev, dtype=m["V0f"].dtype)
            m["A0_init"] = m["A0"].clone(); m["P0_init"] = m["P0"].clone(); m["V0f_init"] = m["V0f"].clone()
        hillv = a ** self.hill / (self.a_sw ** self.hill + a ** self.hill + 1e-12)   # Hill activation in [0,1]
        s_prev = m["mg_scale"]                                    # per-cell scale BEFORE this tick (for the dilution rate)
        s = m["mg_scale"] * (1.0 + self.rate * (self.rho + hillv))    # lambda = rate*(rho baseline + activator)
        if self.rho > 0:                                             # OKUDA uniform-cell mode: cap v_eq per cell at
            v_ref = float(m.get("v_ref", 1.0))                       # vth_frac*v_ref so it cycles (divides), not bulge
            s_cap = (self.vth_frac * v_ref / m["V0f_init"].clamp(min=1e-9)) ** (1.0 / 3.0)
            s = torch.minimum(s, s_cap.clamp(min=1.0))
        else:
            s = torch.clamp(s, max=self.cap)                        # legacy: activator-only bulge to `cap`
        m["mg_scale"] = s
        m["A0"] = m["A0_init"] * (s * s)                         # keep A0/P0/v_eq consistent (area~R^2, vol~R^3)
        m["P0"] = m["P0_init"] * s
        m["V0f"] = m["V0f_init"] * (s ** 3)
        m["V0"] = float(m["V0f"].sum())
        if self.conserve_amount:
            # conserve molecule AMOUNT: c_j <- c_j * (v_old/v_new) = c_j * (s_prev/s)^3 as v_eq grows ~ s^3.
            # Makes dilution STRUCTURAL (no continuum -c(div.v) term); it is LOAD-BEARING (Okuda's intra-domain
            # gradients come from it) -> keep it, don't cancel. The flood is a gamma (rate) problem, fixed elsewhere.
            g_vol = (s / s_prev.clamp(min=1e-9)) ** 3
            cst = clvl.state.clone()
            cst[:nF, h0:h1] = cst[:nF, h0:h1] / g_vol.clamp(min=1e-9)[:, None]
            clvl.state = cst
        return {}


@register_operator("rd_interface_tension", set="vertex", kind="lateral", family="mechanics")
class RDInterfaceTension(Lateral):
    """The RD-INTERFACE tube mechanism (user hypothesis): a PURSE-STRING line tension on the RED/WHITE
    interface ring + a NORMAL (outward) EXTRUSION of the red cells. This turns a localized activator disk
    into a CYLINDER instead of a ball -- the interface cells reorient into the tube neck (holding the
    diameter), while the red interior is expelled outward (Okuda: constant-diameter tube from a maintained
    activator spot). Cross-set: reads cell.chem activator, forces on the vertices; EMIT velocity (the engine
    integrates it alongside shape_energy). K_purse = ring line tension, K_extrude = outward push, a_sw = red
    threshold. Runs a few bounded-Euler substeps of -grad E per frame, E = K_purse*Sigma_iface l_e - K_extrude*Sigma_red a*r."""
    SUPPORTED_DIMS = [3]; EMIT = "velocity"; DIFFERENTIABLE = True
    INPUTS = ["vertex", "cell"]; OUTPUTS = ["vertex"]; READS = ["pos", "chem"]; WRITES = ["pos"]
    MAPS = ["E_srce", "E_trgt", "E_face"]
    MECHANISM_TAGS = ["interface_tension", "purse_string", "extrusion", "tube", "oriented", "cross_scale"]
    PARAM_ROLES = {"K_purse": "interface_line_tension", "K_extrude": "normal_extrusion", "a_sw": "red_threshold"}
    REFERENCE = "Plexus (this work); purse-string / apical-constriction tubulation after Okuda, S. et al. (2018). Sci. Rep. 8:2386."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex"); self.cat = params.get("cell_set", "cell")
        self.K_purse = float(params.get("K_purse", 1.0)); self.K_extrude = float(params.get("K_extrude", 0.5))
        self.a_sw = float(params.get("a_sw", 1.0)); self.eta = float(params.get("eta", 0.05))
        self.cap_frac = float(params.get("cap_frac", 0.10)); self.iters = int(params.get("iters", 4))

    def forward(self, H, mask=None):
        from tyssue_ops3d import face_geometry_3d, ShapeEnergy3D
        vlvl = H.level(self.at); m = getattr(vlvl, "_mesh", None); clvl = H.level(self.cat)
        if m is None or "chem" not in clvl.state_schema:
            return {}
        nF = int(m["nF"]); Nv = int(m["Nv"]); dev = vlvl.state.device; dt = vlvl.state.dtype
        es = torch.as_tensor(m["E_srce"], device=dev, dtype=torch.long)      # robust to numpy/tensor after division
        et = torch.as_tensor(m["E_trgt"], device=dev, dtype=torch.long)
        ef = torch.as_tensor(m["E_face"], device=dev, dtype=torch.long)
        h0, _ = clvl.state_schema["chem"]
        a = clvl.state[:nF, h0].detach().to(dev)
        red = (a > self.a_sw).to(dt)                             # per-cell red state
        twin = ShapeEnergy3D._twin_faces(es, et, ef, Nv)        # neighbour cell across each half-edge
        iface = (red[ef] != red[twin]).to(dt)                   # 1 on the red/white interface half-edges (the ring)
        if float(red.sum()) < 1.0 or float(iface.sum()) < 1.0:  # no red spot / no interface yet -> nothing to do
            return {}
        px0, px1 = vlvl.state_schema["pos"]
        x0 = vlvl.state[:, px0:px1].detach()
        x = x0[:Nv].clone()
        cap = self.cap_frac * float((x0[et] - x0[es]).norm(dim=-1).mean().clamp(min=1e-3))
        redpush = (self.K_extrude * a.clamp(min=0.0) * red)                  # per-cell outward magnitude
        for _ in range(self.iters):                              # DIRECT forces (no autograd): purse-string + extrusion
            force = torch.zeros(Nv, 3, device=dev, dtype=dt)
            d = x[et] - x[es]; u = d / (d.norm(dim=-1, keepdim=True) + 1e-9)   # PURSE-STRING: interface edge shortens
            f = (self.K_purse * iface)[:, None] * u
            force.index_add_(0, es, f); force.index_add_(0, et, -f)
            _, _, cen, _ = face_geometry_3d(x, es, et, ef, nF)  # EXTRUSION: red cells pushed radially OUTWARD
            cdir = cen / (cen.norm(dim=-1, keepdim=True) + 1e-9)
            force.index_add_(0, es, (redpush[ef])[:, None] * cdir[ef] / 3.0)   # push each cell's src vertices out
            x = x + (self.eta * force).clamp(-cap, cap)
        vel = torch.zeros_like(x0)
        vel[:Nv] = (x - x0[:Nv]) / max(float(getattr(H.config, "dt", 1.0)), 1e-6)
        occ = vlvl.occ[:, None] if getattr(vlvl, "occ", None) is not None else 1.0
        return {self.at: vel * occ}


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
