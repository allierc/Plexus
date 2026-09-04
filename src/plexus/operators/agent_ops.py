"""Agents in a material: the two-way coupling, the population, and the scale maps.

    agent_scatter (agent_to_mpm)  agent -> grid: the agent deforms the material
    agent_gather  (mpm_to_agent)  grid -> agent: the material drags and confines the agent
    agent_remodel                 agent -> material stiffness: cells soften or rigidify tissue
    agent_divide / agent_grow     the population, on a fixed buffer with occupancy
    polarity_align (heading_align) / polarity_flow_align (flow_align)   where an agent points
    active_force / active_stress  a pulse becomes a contraction or a stress
    aggregate / broadcast         the two SCALE maps: child -> parent, parent -> child
    seed_from_segmentation        a measured instance segmentation becomes the cell level

THE PAIR THAT MATTERS is `agent_scatter` / `agent_gather`: they are the same coupling read in two
directions, and they must be scheduled together. Splitting them across files is how a spec comes to
push on a material that never pushes back.
"""
from __future__ import annotations
import torch
from plexus.models.base import Structural
from plexus.models.registry import register_operator
from plexus.models.base import Exchange
from plexus.operators.mpm_ops import stencil_offsets, bspline
from plexus.models.base import Aggregate
from plexus.models.base import Broadcast
import json
import os
from plexus.models.base import Field, Seed
from plexus.models.registry import register_field, register_operator
from plexus.paths import graphs_data_path


# ==========================================================================================================
# FROM `discovery_okuda/ops/agent_divide.py` -- agent_divide (agent set, structural): proliferation on a fixed buffer via occupancy.
# ==========================================================================================================
@register_operator("agent_divide", family="population", set="cell", kind="structural")
class CellDivide(Structural):
    EMIT = None                                       # structural: wakes dormant slots, mutates occ+state in place; returns {} — no integrable delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []                              # no required params — `rate` falls back to per-type div_rate else 0
    MECHANISM_TAGS = ["proliferation", "mitosis", "growth"]
    PARAM_ROLES = {"rate": "division_rate", "max_occ": "homeostatic_ceiling"}
    REFERENCE = "Okuda, S. et al. (2015). Reversible network reconnection model for simulating large deformation in dynamic tissue morphogenesis. Biomech. Model. Mechanobiol. 12:627-644."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "agent")
        self.rate = float(params.get("rate", 0.0))        # fallback rate if no per-type div_rate
        self.offset = float(params.get("offset", 0.006))  # daughter placement jitter (world units)
        self.max_occ = float(params.get("max_occ", 0.98)) # stop when this fraction of the buffer is live

    def forward(self, H, mask=None):
        lvl = H.level(self.at); dev = lvl.state.device
        dt = float(getattr(H.config, "dt", 1.0))
        occ = lvl.occ
        live = occ > 0
        nlive = int(live.sum())
        buf = occ.shape[0]
        free = (~live).nonzero(as_tuple=True)[0]
        if nlive == 0 or free.numel() == 0 or nlive >= int(self.max_occ * buf):
            return {}

        rate = getattr(lvl, "div_rate", None)
        rate = rate if rate is not None else torch.full((buf,), self.rate, device=dev)
        p = (1.0 - torch.exp(-rate.clamp(min=0.0) * dt)) * live.float()
        draw = torch.rand(buf, generator=getattr(H, "rng", None), device=dev)
        movers = (draw < p).nonzero(as_tuple=True)[0]
        if movers.numel() == 0:
            return {}
        cap = min(movers.numel(), free.numel(), int(self.max_occ * buf) - nlive)
        if cap <= 0:
            return {}
        parents = movers[:cap]; slots = free[:cap]

        # inherit EVERY per-node buffer (state, heading, node_type, speeds, div_rate, s, ...)
        D = lvl.state.shape[1]
        lvl.state[slots] = lvl.state[parents].clone()
        for name, b in list(lvl.named_buffers()):
            if b is None or b.dim() == 0 or b.shape[0] != buf:
                continue
            b[slots] = b[parents].clone()
        # place the daughter beside the mother
        px0, px1 = lvl.state_schema["pos"]
        jitter = (torch.rand(cap, H.dim, generator=getattr(H, "rng", None), device=dev) - 0.5) * (2 * self.offset)
        lvl.state[slots, px0:px1] = lvl.state[parents, px0:px1] + jitter
        lvl.occ[slots] = 1.0
        if hasattr(lvl, "birth"):
            lvl.birth[slots] = 1.0
        return {}


# ==========================================================================================================
# FROM `discovery_okuda/ops/agent_grow.py` -- agent_grow (cell, structural): CELL rest-VOLUME growth -- the biological growth primitive.
# ==========================================================================================================
@register_operator("agent_grow", family="population", set="cell", kind="structural")
class CellGrow(Structural):
    EMIT = None                                           # structural: advances `grow_V` + wakes reserve child particles in place; returns {} — no integrable delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []                                  # no required params — `rate`<=0 is a no-op; all knobs optional
    MECHANISM_TAGS = ["tissue_growth", "rest_volume_growth", "anisotropic_growth", "budding"]
    PARAM_ROLES = {"rate": "growth_rate", "target": "size_inhibition",
                   "aniso": "growth_anisotropy", "tip": "tip_localization",
                   "stress_gain": "mechano_inhibition"}
    REFERENCE = "Okuda, S. et al. (2015). Biomech. Model. Mechanobiol. 12:627-644."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.child = params.get("child", "mpm_particle")      # the MPM discretisation of this cell
        self.rate = float(params.get("rate", 0.0))            # specific growth rate (per unit time)
        self.target = float(params.get("target", 4.0))        # logistic ceiling on grow_V (size inhibition)
        # `mode: isotropic | anisotropic | tip` BECAME THE `model:` AXIS, 4 September. Three claims
        # about HOW a cell grows -- outward in every direction, preferentially along an axis, or only
        # at the leading edge -- are three hypotheses, not three settings, and `cell_grow`'s sibling
        # triggers are already carried on `model:`. Nothing in `config/` selected it (0 specs), so
        # this rename cost nothing. See AXES.md.
        if "mode" in params:
            raise ValueError(
                "agent_grow: `mode` is gone -- write `model: isotropic | anisotropic | tip`. "
                "How a cell grows is a hypothesis, not a setting. See AXES.md.")
        self.mode = getattr(type(self), "GROWTH", "isotropic")
        self.aniso = float(params.get("aniso", 0.0))          # 0 round .. 1 fully along `axis`
        self.tip = float(params.get("tip", 0.0))              # 0 uniform .. large = seed only the leading edge
        self.offset = float(params.get("offset", 0.01))       # placement distance from the seed (world units)
        # prestretch: woken-particle rest state F = prestretch * I. 1.0 = the old density-only realization
        # (F=I -> zero corotated stress -> new material co-locates, adds DENSITY not VOLUME: byte-identical
        # no-op). prestretch < 1 inserts each woken particle PRE-COMPRESSED (J = prestretch^D < 1) so the
        # fixed-corotated law gives it outward pressure -> it relaxes toward rest by pushing neighbours out
        # -> the envelope genuinely INFLATES. This is what realizes cell rest-VOLUME growth (b57 proved the
        # F=I realization cannot: grow_ratio pinned at 1.000 across the whole offset/reserve ladder).
        self.prestretch = float(params.get("prestretch", 1.0))
        self.stress_gain = float(params.get("stress_gain", 0.0))  # >0 slows growth in deformed tissue
        ax = params.get("axis", None)
        self.axis = [float(a) for a in ax] if ax is not None else None

    # --- backend realisation: wake `k` dormant particles of ONE cell, near its tissue, per the law ---
    def _realize_cell(self, part, livep, free, k, rng, dev):
        px0, px1 = part.state_schema["pos"]; D = px1 - px0
        X = part.state[:, px0:px1]
        Xl = X[livep]
        a = None
        if self.axis is not None and self.mode in ("anisotropic", "tip"):
            a = torch.tensor(self.axis[:D], device=dev, dtype=X.dtype)
            a = a / a.norm().clamp(min=1e-9)
        # seed selection: leading edge along axis when tip>0, else uniform
        if a is not None and self.tip > 0 and self.mode == "tip":
            proj = Xl @ a
            w = torch.softmax(self.tip * (proj - proj.max()) / proj.std().clamp(min=1e-6), dim=0)
            sel = torch.multinomial(w, k, replacement=True, generator=rng)
        else:
            sel = torch.randint(0, int(livep.numel()), (k,), generator=rng, device=dev)
        seeds = livep[sel]; slots = free[:k]
        # placement direction: blend axis (anisotropic) with random (isotropic)
        rnd = torch.randn(k, D, generator=rng, device=dev)
        rnd = rnd / rnd.norm(dim=1, keepdim=True).clamp(min=1e-9)
        dirv = (self.aniso * a[None, :] + (1.0 - self.aniso) * rnd) if a is not None else rnd
        dirv = dirv / dirv.norm(dim=1, keepdim=True).clamp(min=1e-9)
        jit = 0.25 * self.offset * torch.randn(k, D, generator=rng, device=dev)
        newpos = X[seeds] + self.offset * dirv + jit
        # inherit the seed's material; the fresh material starts PRE-COMPRESSED (F=prestretch*I, C=0) so it
        # carries outward pressure and inflates the envelope (prestretch=1 -> F=I -> old density-only no-op)
        buf = part.occ.shape[0]
        part.state[slots] = part.state[seeds].clone()
        for _n, b in list(part.named_buffers()):
            if b is None or b.dim() == 0 or b.shape[0] != buf:
                continue
            b[slots] = b[seeds].clone()
        part.state[slots, px0:px1] = newpos
        if hasattr(part, "F") and part.F is not None:
            part.F[slots] = (self.prestretch * torch.eye(D, device=dev)).expand(k, D, D).clone()
        if hasattr(part, "C") and part.C is not None:
            part.C[slots] = 0.0
        part.occ[slots] = 1.0
        if hasattr(part, "birth"):
            part.birth[slots] = 1.0

    def forward(self, H, mask=None):
        if self.rate <= 0.0:
            return {}
        cell = H.level(self.at); dev = cell.state.device
        part = H.level(self.child)
        rng = getattr(H, "rng", None)
        dt = float(getattr(H.config, "dt", 1.0))
        ncell = cell.occ.shape[0]
        par = part.parent

        # --- biological growth STATE on the cell (lazy provision) ---
        if getattr(cell, "grow_V", None) is None:
            cell.register_buffer("grow_V", torch.ones(ncell, device=dev))
        if getattr(cell, "grow_base", None) is None:           # birth-size live count per cell (grow_V==1)
            base = torch.zeros(ncell, device=dev)
            base.index_add_(0, par, (part.occ > 0).float())
            cell.register_buffer("grow_base", base.clamp(min=1.0))

        # --- 1. BIOLOGY: advance grow_V by the logistic growth law (+ optional mechano-inhibition) ---
        V = cell.grow_V
        live_cell = (cell.occ > 0).float()
        # `at:` may restrict growth to a SUBSET of cells (e.g. cell[type=bud]): the engine
        # passes that per-cell boolean mask, so gate the growth law by it -> only the selected
        # cells advance grow_V and wake reserve; the rest stay at birth size. For a plain
        # `at: cell` the mask is all-live -> live_cell unchanged -> byte-identical to before.
        if mask is not None:
            live_cell = live_cell * mask.to(live_cell.dtype)
        smod = torch.ones(ncell, device=dev)
        if self.stress_gain > 0.0 and getattr(part, "F", None) is not None:
            eye = torch.eye(part.F.shape[-1], device=dev)
            defo = (part.F - eye).reshape(part.F.shape[0], -1).norm(dim=1) * (part.occ > 0).float()
            cdefo = torch.zeros(ncell, device=dev); cdefo.index_add_(0, par, defo)
            cnt = torch.zeros(ncell, device=dev); cnt.index_add_(0, par, (part.occ > 0).float())
            smod = 1.0 / (1.0 + self.stress_gain * (cdefo / cnt.clamp(min=1.0)))
        cell.grow_V = V + self.rate * V * (1.0 - V / self.target) * dt * smod * live_cell

        # --- 2. REALIZATION: wake dormant child particles so live count ~ grow_base * grow_V ---
        live_mask = part.occ > 0
        cur = torch.zeros(ncell, device=dev); cur.index_add_(0, par, live_mask.float())
        deficit = ((cell.grow_base * cell.grow_V) - cur).floor().clamp(min=0).long()
        if int(deficit.sum().item()) == 0:
            return {}
        for c in torch.nonzero(deficit > 0, as_tuple=True)[0].tolist():
            free = ((par == c) & (part.occ == 0)).nonzero(as_tuple=True)[0]
            livep = ((par == c) & live_mask).nonzero(as_tuple=True)[0]
            k = min(int(deficit[c].item()), int(free.numel()))
            if k <= 0 or livep.numel() == 0:
                continue
            self._realize_cell(part, livep, free, k, rng, dev)
        return {}


# ==========================================================================================================
# FROM `discovery_okuda/ops/agent_scatter.py` -- agent_scatter (was agent_to_mpm) (agent set -> mpm_grid): the agents deform the material.
# ==========================================================================================================
@register_operator("agent_grow", family="population", set="cell", kind="structural",
                   model="anisotropic")
class CellGrowAnisotropic(CellGrow):
    """`anisotropic` MODEL of agent_grow -- new material is placed preferentially ALONG `axis:`.

    A claim about the cell, not a setting on it: `aniso` blends the axis with a random direction, so
    the default (`isotropic`) is the aniso=0 end of the same blend -- but WHICH claim is being made
    is the thing a reader needs from the spec, and a float buried in the parameters does not say it.
    """
    GROWTH = "anisotropic"


@register_operator("agent_grow", family="population", set="cell", kind="structural", model="tip")
class CellGrowTip(CellGrow):
    """`tip` MODEL of agent_grow -- new material is seeded only at the LEADING EDGE along `axis:`.

    Distinct from `anisotropic` in WHERE the parent is chosen, not in which direction the daughter
    is placed: the seed is drawn from a softmax over the projection onto the axis, so growth extends
    a front rather than thickening a body. That is the difference between a tube and an ellipsoid,
    which is a morphological claim.
    """
    GROWTH = "tip"


@register_operator("agent_scatter", "agent_to_mpm", family="coupling", set="cell", kind="exchange")
class AgentScatter(Exchange):              # (alias `agent_to_mpm`, one migration cycle)
    EMIT = None                               # writes the grid; consumed by the MPM substep
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["to"]
    MECHANISM_TAGS = ["agent_to_grid", "active_stress_source"]
    PARAM_ROLES = {"agent_mass": "effective_agent_mass", "k": "push_gain"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "agent")
        self.to = params.get("to", "mpm_grid")
        self.agent_mass = float(params.get("agent_mass", 1e-4))
        self.k = float(params.get("k", 1.0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at); g = H.field(self.to); dev = lvl.state.device
        X = lvl.get("pos")
        D = X.shape[1]
        periodic = bool(getattr(H, "periodic", False))
        offsets = stencil_offsets(D, dev)
        _, weight, flat = bspline(X, g.inv_dx, offsets, g.shape, periodic)

        h = getattr(lvl, "heading", None)
        if h is not None:
            v_agent = lvl.move_speed[:, None] * h                      # propulsion velocity, like glide
        else:
            v_agent = lvl.get("vel")                                   # fallback: whatever velocity it carries
        m_eff = (self.agent_mass * self.k) * lvl.occ                   # [N] effective mass deposit
        if mask is not None:
            m_eff = m_eff * mask.float()
        mom_pp = m_eff[:, None] * v_agent                             # [N,D] per-agent momentum

        g.m.index_add_(0, flat, (weight * m_eff[:, None]).reshape(-1))
        g.mv.index_add_(0, flat, (weight[..., None] * mom_pp[:, None, :]).reshape(-1, D))
        return {}


# ==========================================================================================================
# FROM `discovery_okuda/ops/agent_gather.py` -- agent_gather (was mpm_to_agent) (mpm_grid -> agent set): the material drags the agents, and the fluid's
# ==========================================================================================================
@register_operator("agent_gather", "mpm_to_agent", family="coupling", set="cell", kind="exchange")
class AgentGather(Exchange):               # (alias `mpm_to_agent`, one migration cycle)
    EMIT = "velocity"                 # emits an advection velocity; engine integrates pos
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["from"]
    MECHANISM_TAGS = ["grid_to_agent", "fluid_drag", "surface_confinement"]
    PARAM_ROLES = {"k": "fluid_drag_gain", "confine": "surface_tension_confinement"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "agent")
        self.frm = params.get("from", "mpm_grid")
        self.k = float(params.get("k", 1.0))           # fluid advection gain (1 = fully carried)
        self.confine = float(params.get("confine", 0.0))  # inward drift up grad(density); 0 = off
        self.field = params.get("field", "mass")       # "mass" (g.m, universal) or "colour" (g.c, liquid)

    def forward(self, H, mask=None):
        lvl = H.level(self.at); g = H.field(self.frm); dev = lvl.state.device
        X = lvl.get("pos")
        D = X.shape[1]
        periodic = bool(getattr(H, "periodic", False))
        offsets = stencil_offsets(D, dev); S = offsets.shape[0]
        _, weight, flat = bspline(X, g.inv_dx, offsets, g.shape, periodic)

        # --- fluid drag: B-spline gather of the solved grid velocity (same as g2p) ---
        gvn = g.v[flat].view(lvl.n, S, D)
        v_fluid = torch.nan_to_num((weight[..., None] * gvn).sum(1))   # [N,D]
        vel = self.k * v_fluid

        # --- confinement: drift up the MATERIAL-DENSITY gradient (inward), holding cells in the
        # blob. Uses the grid mass g.m (high inside the material, ~0 outside) so it works for a
        # fully-elastic disc too, not only a liquid one; `field: colour` switches to the liquid
        # colour g.c (true surface-tension interface) when a liquid skin is present.
        if self.confine != 0.0:
            src = g.c if (self.field == "colour" and bool((g.c > 0).any())) else g.m
            dens = (src / src.max().clamp(min=1e-9)).view(g.shape)     # normalised 0..1 density
            grad = torch.stack([                                       # central diff * 0.5*inv_dx per axis
                (torch.roll(dens, -1, k) - torch.roll(dens, 1, k)) * (0.5 * g.inv_dx)
                for k in range(D)], dim=-1).reshape(g.n_cells, D)      # [n_cells, D]
            gcn = grad[flat].view(lvl.n, S, D)
            grad_at = torch.nan_to_num((weight[..., None] * gcn).sum(1))
            vel = vel + self.confine * grad_at                        # +grad(density) points inward

        m = (mask.float() if mask is not None else torch.ones(lvl.n, device=dev)) * lvl.occ
        return {self.at: vel * m[:, None]}


# ==========================================================================================================
# FROM `discovery_okuda/ops/agent_remodel.py` -- agent_remodel (agent set -> mpm_particle stiffness): cells remodel the tissue.
# ==========================================================================================================
@register_operator("agent_remodel", family="coupling", set="cell", kind="exchange")
class AgentRemodel(Exchange):
    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["to", "target"]
    MECHANISM_TAGS = ["tissue_remodelling", "stiffening", "fluidisation"]
    PARAM_ROLES = {"gain": "remodel_gain", "rate_attr": "per_type_remodel_rate"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "agent")
        self.to = params.get("to", "mpm_grid")            # grid used only for the transfer geometry
        self.target = params.get("target", "mpm_particle")
        self.gain = float(params.get("gain", 1.0))
        self.rate_attr = params.get("rate_attr", "remodel_rate")
        self.mu_min = float(params.get("mu_min", 0.5)); self.mu_max = float(params.get("mu_max", 1.0e4))
        self.la_min = float(params.get("la_min", 0.5)); self.la_max = float(params.get("la_max", 1.0e4))

    def forward(self, H, mask=None):
        lvl = H.level(self.at); g = H.field(self.to); tgt = H.level(self.target)
        rate = getattr(lvl, self.rate_attr, None)
        if rate is None or float(rate.abs().max()) == 0.0:
            return {}                                     # no remodellers -> nothing to do
        dt = float(getattr(H.config, "dt", 1.0))
        periodic = bool(getattr(H, "periodic", False))
        dev = lvl.state.device
        offs = stencil_offsets(tgt.get("pos").shape[1], dev); S = offs.shape[0]

        # scatter cells' remodel rate + presence onto the grid (density-normalised mean rate)
        Xa = lvl.get("pos"); _, wa, fa = bspline(Xa, g.inv_dx, offs, g.shape, periodic)
        occ_a = lvl.occ if mask is None else lvl.occ * mask.float()
        R = torch.zeros(g.n_cells, device=dev); Wt = torch.zeros(g.n_cells, device=dev)
        R.index_add_(0, fa, (wa * (rate * occ_a)[:, None]).reshape(-1))
        Wt.index_add_(0, fa, (wa * occ_a[:, None]).reshape(-1))
        Rn = R / Wt.clamp(min=1e-6)                       # mean remodel rate per cell

        # gather at material points and update their stiffness multiplicatively
        Xp = tgt.get("pos"); _, wp, fp = bspline(Xp, g.inv_dx, offs, g.shape, periodic)
        Rp = (wp * Rn[fp].view(tgt.n, S)).sum(1)          # [Np]
        factor = torch.exp((self.gain * dt) * Rp)
        tgt.mu = (tgt.mu * factor).clamp(self.mu_min, self.mu_max)
        tgt.la = (tgt.la * factor).clamp(self.la_min, self.la_max)
        return {}


# ==========================================================================================================
# FROM `discovery_okuda/ops/polarity_align.py` -- polarity_align (was heading_align) (agent set -> agent heading): FIRST-ORDER Vicsek polar alignment.
# ==========================================================================================================
@register_operator("polarity_align", "heading_align", family="polarity", set="cell", kind="exchange")
class PolarityAlign(Exchange):                   # (alias `heading_align`, one migration cycle)
    EMIT = None                                 # writes `heading` in place (Vicsek steering); returns {} — not an integrable delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []                        # no required params — gain/noise optional (defaults in __init__)
    MECHANISM_TAGS = ["vicsek", "polar_alignment", "collective_motion", "flocking"]
    PARAM_ROLES = {"gain": "alignment_rate", "noise": "orientation_noise"}
    REFERENCE = "Vicsek, T. et al. (1995). Phys. Rev. Lett. 75:1226-1229."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "agent")
        self.gain = float(params.get("gain", 1.0))
        self.noise = float(params.get("noise", 0.0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        h = lvl.heading; occ = lvl.occ; dev = h.device
        N, D = h.shape
        dt = float(getattr(H.config, "dt", 1.0))
        ei = getattr(lvl, "edge_index", None)
        if ei is None or ei.numel() == 0:
            return {}
        i, j = ei[0], ei[1]                                       # row0 receiver i, row1 neighbour j
        w = occ[j].to(h.dtype)                                    # mask dormant neighbours
        hbar = torch.zeros(N, D, device=dev, dtype=h.dtype).index_add_(0, i, h[j] * w[:, None])
        deg = torch.zeros(N, device=dev, dtype=h.dtype).index_add_(0, i, w)
        hbar = hbar / deg.clamp(min=1.0)[:, None]                 # mean neighbour heading [N,D]
        hmag = hbar.norm(dim=-1, keepdim=True)
        hhat = hbar / hmag.clamp(min=1e-9)
        perp = hhat - (hhat * h).sum(-1, keepdim=True) * h        # component of n_hat perp to n
        new_h = h + (self.gain * dt) * perp
        if self.noise > 0.0:                                      # Vicsek angular noise: order vs disorder
            new_h = new_h + self.noise * torch.randn(N, D, generator=getattr(H, "rng", None), device=dev)
        new_h = new_h / new_h.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        keep = (occ > 0) & (deg > 0) & (hmag[:, 0] > 1e-7)        # keep heading where no coherent neighbour signal
        if mask is not None:
            keep = keep & (mask > 0)
        lvl.heading = torch.where(keep[:, None], new_h, h)
        return {}


# ==========================================================================================================
# FROM `discovery_okuda/ops/polarity_flow_align.py` -- polarity_flow_align (was flow_align) (mpm_grid -> agent heading): polarity-velocity alignment to the tissue FLOW.
# ==========================================================================================================
@register_operator("polarity_flow_align", "flow_align", family="polarity", set="cell", kind="exchange")
class PolarityFlowAlign(Exchange):               # (alias `flow_align`, one migration cycle)
    EMIT = None                                 # writes `heading` in place (flow-alignment steering); returns {} — not an integrable delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["from"]
    MECHANISM_TAGS = ["polarity_velocity_alignment", "flow_alignment"]
    PARAM_ROLES = {"gain": "flow_alignment_rate"}
    REFERENCE = "Toner, J. & Tu, Y. (1995). Long-range order in a two-dimensional dynamical XY model. Phys. Rev. Lett. 75:4326-4329."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "agent")
        self.frm = params.get("from", "mpm_grid")
        self.gain = float(params.get("gain", 1.0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at); g = H.field(self.frm); dev = lvl.state.device
        h = lvl.heading; X = lvl.get("pos"); occ = lvl.occ
        D = X.shape[1]
        dt = float(getattr(H.config, "dt", 1.0))
        periodic = bool(getattr(H, "periodic", False))
        offsets = stencil_offsets(D, dev); S = offsets.shape[0]
        _, weight, flat = bspline(X, g.inv_dx, offsets, g.shape, periodic)
        vg = (weight[..., None] * g.v[flat].view(lvl.n, S, D)).sum(1)     # gathered flow velocity [N,D]
        vg = torch.nan_to_num(vg)
        vmag = vg.norm(dim=-1, keepdim=True)
        vhat = vg / vmag.clamp(min=1e-9)
        perp = vhat - (vhat * h).sum(-1, keepdim=True) * h               # component of v_hat perp to n
        new_h = h + (self.gain * dt) * perp
        new_h = new_h / new_h.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        keep = (occ > 0) & (vmag[:, 0] > 1e-7)
        if mask is not None:
            keep = keep & (mask > 0)
        lvl.heading = torch.where(keep[:, None], new_h, h)
        return {}


# ==========================================================================================================
# FROM `discovery_okuda/ops/active_force.py` -- active_force -- the FORCE constitutive law: an activation field -> per-particle MPM body force.
# ==========================================================================================================
@register_operator("active_force", "pulse_to_contraction", family="mechanics", set="particle", kind="exchange")
class ActiveForce(Exchange):                     # (alias `pulse_to_contraction` for one migration cycle)
    EMIT = "mpm_acceleration"           # a body accel the MPM substep consumes as a_ext, not engine-integrated
    SUPPORTED_DIMS = [2]                 # 2D — reads a 2-vector activation gradient / direction field
    REQUIRES_PARAMS = ["from"]                # the activation field to read
    MECHANISM_TAGS = ["active_contraction", "field_gradient_force", "directed_active_stress"]
    PARAM_ROLES = {"amplitude": "contraction_strength", "mode": "gradient_or_directional"}
    REFERENCE = "Marchetti, M. C. et al. (2013). Hydrodynamics of soft active matter. Rev. Mod. Phys. 85:1143-1189."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("from")
        self.amplitude = float(params.get("amplitude", 50.0))
        self.channel = int(params.get("channel", 0))
        # `mode:` SPLIT ONTO ITS TWO REAL AXES, 4 September. It carried three words doing two
        # different jobs: `inward`/`outward` chose a SIGN on one rule, `directional` chose a
        # DIFFERENT RULE. One key, two axes, and nothing in the registry could tell them apart.
        # `along:` is the value; `model: directional` is the hypothesis. See AXES.md.
        if "mode" in params:
            raise ValueError(
                "active_force: `mode` is gone. `inward`/`outward` are now `along:` (a value on this "
                "model -- the sign of the gradient), and `directional` is now "
                "`model: directional` (a different rule: a prescribed direction field). See AXES.md.")
        self.along = str(params.get("along", "inward"))
        if self.along not in ("inward", "outward"):
            raise ValueError(f"active_force: along must be inward|outward, got {self.along!r}")
        self.sign = 1.0 if self.along == "inward" else -1.0
        self.at = params.get("_at", "particle")

    def _accel(self, H, lvl, pos, fld):
        """THE HYPOTHESIS, and the only thing a `model=` variant of active_force changes.

        Default: the contraction follows the ACTIVATION GRADIENT -- direction = +/- grad(a), with
        `along:` choosing the sign. `inward` pulls up the gradient, toward higher activation.
        """
        grad = fld.grad_at(pos, self.channel, periodic=getattr(H, "periodic", False))   # [N, 2]
        return self.sign * self.amplitude * grad                          # inward for sign>0

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        fld = H.fields[self.field_name]
        acc = self._accel(H, lvl, pos, fld)
        acc = acc * lvl.occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        # return a per-particle force delta; the engine sums it (with drag's) into
        # H.delta(mpm_particle), which p2g consumes as the MPM body force. EMIT=None,
        # so the engine never integrates the particle set (g2p owns advection).
        return {self.at: acc}


@register_operator("active_force", "pulse_to_contraction", family="mechanics", set="particle",
                   kind="exchange", model="directional")
class ActiveForceDirectional(ActiveForce):
    """`directional` MODEL of active_force -- the contraction follows a PRESCRIBED DIRECTION FIELD.

    A DIFFERENT HYPOTHESIS, NOT A DIFFERENT SIGN, which is why it is a `model:` and the old
    `mode: inward|outward|directional` could not say so. The default reads the activation's GRADIENT
    and lets the field decide where to push; this one reads the activation only for HOW MUCH and
    takes WHERE from a separate unit-vector field -- the active-stress orientation map. On a uniform
    activation the default produces no force at all and this one produces its full magnitude, so
    they are not two ways of computing one thing.

        F_i = amplitude * a(x_i) * d(x_i)

    It also READS A SECOND FIELD, which the typed signature now records per variant (R1(c)).
    """
    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.direction_from = params.get("direction_from")
        if self.direction_from is None:
            raise ValueError("active_force[model: directional] needs `direction_from:` "
                             "(a vector_grid field giving the contraction direction)")

    def _accel(self, H, lvl, pos, fld):
        a = fld.sample(pos, self.channel)                                 # [N] activation: HOW MUCH
        d = H.fields[self.direction_from].sample(pos)                     # [N, 2] direction: WHERE
        d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-9)
        return self.amplitude * a[:, None] * d


# ==========================================================================================================
# FROM `discovery_okuda/ops/active_stress.py` -- active_stress -- the STRESS constitutive law: an activation field -> per-particle active stress.
# ==========================================================================================================
@register_operator("active_stress", "pulse_to_active_stress", family="mechanics", set="particle", kind="exchange")
class ActiveStress(Exchange):                    # (alias `pulse_to_active_stress` for one migration cycle)
    EMIT = None                         # stress is consumed by the MPM substep, not integrated
    SUPPORTED_DIMS = [2]                 # 2D — contraction axis n and n n^T are 2-vectors / 2x2
    REQUIRES_PARAMS = ["from", "direction_from"]
    MECHANISM_TAGS = ["active_contraction", "active_stress_tensor", "directed_active_stress"]
    PARAM_ROLES = {"amplitude": "active_stress_gain", "direction_from": "contraction_axis_field"}
    REFERENCE = "Simha, R. A. & Ramaswamy, S. (2002). Phys. Rev. Lett. 89:058101; Marchetti, M. C. et al. (2013). Rev. Mod. Phys. 85:1143."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("from")
        self.amplitude = float(params.get("amplitude", 50.0))
        self.channel = int(params.get("channel", 0))
        self.direction_from = params.get("direction_from")
        if self.direction_from is None:
            raise ValueError("active_stress needs `direction_from:` "
                             "(a vector_grid field giving the contraction axis n)")
        self.at = params.get("_at", "particle")
        # FRANK-STARLING (length-dependent tension, NHS/Niederer form): scale contraction by local fibre
        # stretch lambda -> T *= 1 + stretch_activation*(lambda-1). 0 = OFF (byte-identical). Real cardiomyocytes
        # contract HARDER when stretched; a stretch-REGULATED size lever (bigger loops without the runaway
        # overshoot of raw amplitude/gain), aimed at the size<->direction frontier.
        self.stretch_activation = float(params.get("stretch_activation", 0.0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        fld = H.fields[self.field_name]

        a = fld.sample(pos, self.channel)                                         # [N] activation a(x)
        n = H.fields[self.direction_from].sample(pos)                             # [N, 2] contraction axis
        n = n / n.norm(dim=1, keepdim=True).clamp(min=1e-9)                        # unit
        gate = (a * lvl.occ).clamp(min=0.0)                                       # only inactive=0 particles off
        gain = getattr(lvl, "gain", None)                                         # optional per-particle gain map
        if gain is not None:                                                      # (apply_material_map target=gain)
            gate = gate * gain                                                    # spatially-structured contraction gain
        if mask is not None:
            gate = gate * mask.float()
        if self.stretch_activation != 0.0:                                        # FRANK-STARLING length-dependent tension
            F = getattr(lvl, "F", None)                                           # per-particle deformation gradient [N,2,2]
            if F is not None:
                lam = torch.bmm(F, n[:, :, None]).squeeze(-1).norm(dim=1).clamp(min=1e-6)   # fibre stretch lambda = |F n|
                gate = gate * (1.0 + self.stretch_activation * (lam - 1.0)).clamp(min=0.0)  # T *= 1+beta*(lambda-1)
        nn = n[:, :, None] * n[:, None, :]                                        # [N, 2, 2]  n n^T
        # Active TENSION along the fibre axis n (cardiac convention sigma_a = +T n n^T): added to the
        # elastic stress it SHORTENS the tissue along n. (The p2g scaling carries the MPM sign; this
        # sign is fixed empirically so axis n => contraction ALONG n, see active_stress_test.)
        sigma = (self.amplitude * gate)[:, None, None] * nn                        # +A a n n^T
        # side-channel for p2g (same idiom as H.part_accel); overwritten each frame, read every substep.
        H.active_stress = sigma
        return {}                                                                 # no body-force delta


# ==========================================================================================================
# FROM `discovery_okuda/ops/aggregate.py` -- aggregate -- children -> parent. Reduce a contained set onto its parent.
# ==========================================================================================================
@register_operator("aggregate", family="hierarchy", set="cell", kind="aggregate")
class Centroid(Aggregate):
    """children -> parent: the occupancy-weighted mean of a child block, along the `parent` map.

    THE READOUT IS ONE STEP BEHIND THE ROW IT IS RECORDED IN, and that is a property of the
    schedule rather than of this operator, so it is worth stating where someone reading the
    output will look. `run` evaluates the whole schedule, THEN integrates, THEN records
    (`engine.py`: `_integrate(H, sim.dt)` followed by `rec_index.get(tick)`). An operator inside
    the schedule therefore sees the state at the START of the tick. Measured on
    `config/neural/ctrnn_assemblies.yaml`: `assembly.activity[t]` equals the mean of
    `neuron.voltage[t-1]` to 1e-6 (float32), and differs from the mean at row `t` by up to 0.22.
    Neither number is wrong; reading the first as the second is.
    """

    EMIT = None                                    # readout: writes parent `pos` in place (MAY_MUTATE_INTEGRATED_STATE); returns {} — no integrable delta
    # typed signature (Plexus2 sec. 2.1): children -> parent along the `parent` map.
    INPUTS = ["particle"]                          # the contained (child) set
    OUTPUTS = ["cell"]                             # the parent set it writes
    READS = ["pos"]
    WRITES = ["pos"]                               # parent centroid position (a derived readout)
    MAPS = ["parent"]                              # reduce along the parent (containment) map
    SUPPORTED_DIMS = [2, 3]                         # occupancy-weighted mean is dimension-generic
    REQUIRES_PARAMS = []                            # no required params — `child` defaults to the first contained set
    MECHANISM_TAGS = ["centroid", "reduction", "hierarchical_readout"]
    PARAM_ROLES = {"child": "source_child_set", "block": "aggregated_child_block",
                   "into": "target_parent_block"}
    REFERENCE = "Battaglia, P. W. et al. (2018). Relational inductive biases, deep learning, and graph networks. arXiv:1806.01261."
    MAY_MUTATE_INTEGRATED_STATE = True             # writes the parent's derived position (a readout)

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.child = params.get("child")           # optional: which contained set (default: first child)
        # WHICH BLOCK IS AGGREGATED. `pos` by default, so every existing spec is unchanged and the
        # operator is still the centroid it was named for. The mechanism was never about position:
        # it is an occupancy-weighted mean of a child block along the `parent` map, and hard-coding
        # `pos` is what stopped a `neural_assembly` reading the mean voltage of its neurons. Same
        # contract, same kind, one more knob -- a refinement of the signature, not a new operator.
        #
        # TWO NAMES, because the block often means something different one scale up: a neuron's
        # `voltage` aggregates into an assembly's `activity`. `into` defaults to `block`, so the
        # centroid case stays a single word.
        self.block = params.get("block", "pos")            # the CHILD block being reduced
        self.into = params.get("into", self.block)         # the PARENT block it is written to

    def forward(self, H, mask=None):
        parent = H.level(self.at)
        kids = H.children(self.at)
        if not kids:
            return {}
        child = H.level(self.child) if self.child else H.level(kids[0])
        pidx = child.parent                        # [Nc] parent slot per child
        if pidx.numel() == 0:
            return {}
        dev = parent.state.device
        if self.block not in child.state_schema:
            raise ValueError(
                f"aggregate: child set {child.name!r} has no state block {self.block!r} to reduce "
                f"(it has: {', '.join(child.state_schema)}).")
        if self.into not in parent.state_schema:
            raise ValueError(
                f"aggregate: parent set {parent.name!r} has no state block {self.into!r} to write "
                f"(it has: {', '.join(parent.state_schema)}).")
        px0, px1 = parent.state_schema[self.into]
        cw0, cw1 = child.state_schema[self.block]
        if (px1 - px0) != (cw1 - cw0):
            raise ValueError(
                f"aggregate: {child.name}.{self.block} is {cw1 - cw0} wide but "
                f"{parent.name}.{self.into} is {px1 - px0}.")
        D = px1 - px0
        cpos = child.get(self.block); cocc = child.occ
        # ONE PARENT MEANS THE INDEX CARRIES NO INFORMATION -- SO DO NOT PAY TO SCATTER BY IT.
        #
        # `index_add_(0, pidx, v)` asks "add each row of v into the slot pidx says". When the parent
        # set has ONE slot, every row of `pidx` is 0 and the answer is just the sum of all the rows;
        # the index lookup, and the machinery that makes concurrent writes to the same slot safe, are
        # both pure overhead. `sum(0)` is that answer computed directly, as a tree reduction.
        #
        # Measured on an RTX A6000, [570760, 3] reduced into [1, 3]:
        #
        #     index_add_, deterministic  140.98 ms      index_add_, atomics  1.00 ms
        #     sum(0)                       0.016 ms
        #
        # 63x against plain atomics, 8,800x against the deterministic form the engine used to force
        # on every run. On si_waterfall the two calls below were 214 ms of a 255 ms frame -- 84% of
        # the simulation spent taking the centroid of a set with one member, while the four MPM
        # operators together took 41 ms. Replacing them at n == 1 takes that frame to 58.2 ms.
        #
        # SAFE WITHOUT ANY ASSUMPTION ABOUT `pidx`: with one parent slot there is nowhere else a
        # child could be summed to, so the sum over all children IS the segment sum. No host sync,
        # no cached predicate, nothing to invalidate when `agent_divide` rewrites parenthood.
        #
        # IT IS ALSO THE MORE ACCURATE OF THE TWO, which was not the expected result and is the
        # reason this is a fix and not a trade. The segment scan adds 570,760 terms one after
        # another, so the running total is ~1e5 while each new term is ~0.25 and every addition
        # rounds off a piece of it: the error grows with N. `sum` reduces as a tree -- pairs, then
        # pairs of pairs -- so both operands stay the same size and the error grows with log N.
        # Against a float64 evaluation of the same sum on frame 6 of si_waterfall (true 2.7546e5):
        #
        #     sum(0)       error 2.1e-2   = 7.7e-8 relative, one float32 ulp
        #     index_add_   error 5.9e+2   = 2.2e-3 relative, 28,000x worse
        #
        # -- so the centroid moves by up to 1.6e-3 m on a 0.5 m box when this path is taken, and it
        # moves TOWARDS the right answer. It is also still bit-reproducible run to run, which is what
        # the determinism flag was for; deterministic was never the same claim as correct.
        if parent.n == 1:
            s = (cpos * cocc[:, None]).sum(0, keepdim=True)
            w = cocc.to(s.dtype).sum(0, keepdim=True)
        else:
            s = torch.zeros(parent.n, D, device=dev).index_add_(0, pidx, cpos * cocc[:, None])
            w = torch.zeros(parent.n, device=dev).index_add_(0, pidx, cocc)
        centroid = s / w.clamp(min=1.0)[:, None]
        # IN PLACE. `new = parent.state.clone(); parent.state = new` gave the parent level a NEW
        # state tensor on every tick, which is invisible in eager and fatal for a captured CUDA
        # graph: the graph holds the address it saw at capture, so a replay would keep reading the
        # tensor the run had stopped writing. `aggregate` sits OUTSIDE the substep block, which is
        # why it survived the sweep that put the four MPM operators in place -- the guard that
        # compares buffer addresses each tick is what caught it, firing at frame 2 of cell_13 and
        # dropping the graph for the whole run.
        #
        # Legal for the same reason the clone was: this operator declares
        # MAY_MUTATE_INTEGRATED_STATE, so the engine's tick-0 integration-invariant guard does not
        # apply, and `centroid` is fully computed from the CHILD set before the parent is touched.
        parent.state[:, px0:px1] = torch.where(parent.occ[:, None] > 0, centroid,
                                               parent.state[:, px0:px1])
        return {}


# ==========================================================================================================
# FROM `discovery_okuda/ops/broadcast.py` -- broadcast -- parent -> children. Lift a parent quantity onto its children.
# ==========================================================================================================
@register_operator("broadcast", family="hierarchy", set="particle", kind="broadcast")
class BroadcastLift(Broadcast):
    EMIT = "velocity"            # emits a velocity; the engine integrates
    # typed signature (Plexus2 sec. 2.1): parent -> children along the `parent` map.
    INPUTS = ["cell"]                           # the parent set
    OUTPUTS = ["particle"]                       # the child set it lifts onto
    READS = ["pos"]
    WRITES = ["pos"]                             # velocity delta pulling each child toward its parent
    MAPS = ["parent"]                            # lift along the parent (containment) map
    SUPPORTED_DIMS = [2, 3]                     # dimension-generic: the lift is `stiffness*(parent_pos - child_pos)` in N-D
    REQUIRES_PARAMS = ["stiffness"]
    MECHANISM_TAGS = ["containment", "hierarchical_coupling", "spring"]
    PARAM_ROLES = {"stiffness": "containment_strength"}
    REFERENCE = "Battaglia, P. W. et al. (2018). arXiv:1806.01261 (graph-network broadcast)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.k = float(params.get("stiffness", 1.0))
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        child = H.level(self.at)
        dev = child.state.device
        pname = getattr(child, "parent_name", None)
        if pname is None:
            return {self.at: torch.zeros_like(child.get("pos"))}   # no parent -> zero delta (matches pos dim, 2D/3D)
        parent = H.level(pname)
        ppos = parent.get("pos")[child.parent]     # each child's parent position
        vel = self.k * (ppos - child.get("pos")) * child.occ[:, None]
        if mask is not None:
            vel = vel * mask[:, None].float()
        return {self.at: vel}


# ==========================================================================================================
# FROM `discovery_okuda/ops/segmentation_seed.py` -- segmentation_seed -- a measured instance segmentation becomes the CELL level of the hierarchy.
# ==========================================================================================================
@register_field("label_image", frame="label_image")
class LabelImageField(Field):
    """An integer instance map read from a TIFF. NOT normalised, NEVER interpolated.

    The one job it has that `image` cannot do: return the id that is actually there. Bilinear
    weights between label 7 and label 12 are a number that means nothing and points at a cell that
    may not exist, so `sample_label` indexes rather than interpolates.
    """

    def __init__(self, name, source=None, res=None, width=1.0, device="cpu", **kw):
        super().__init__(name)
        if source is None:
            raise ValueError(f"label_image field {name!r} needs a `source:` (path to a label .tif)")
        import tifffile
        path = source if os.path.isabs(source) else graphs_data_path(source)
        img = tifffile.imread(path)
        if img.ndim == 3:
            img = img[..., 0]
        img = img[::-1, :].copy()                       # image-top -> domain-top, as ImageField
        v = torch.tensor(img.astype("int64"), device=device).permute(1, 0).contiguous()
        self.C = 1
        self.nx, self.ny = int(v.shape[0]), int(v.shape[1])
        self.width = float(width)
        self.R = self.nx / self.width
        self.register_buffer("grid", v[None])           # [1, nx, ny] int64 labels
        self.n_labels = int(v.max())

    def sample_label(self, pos):
        """[N,2] world positions -> [N] integer label, nearest neighbour."""
        x = pos[:, 0].clamp(0, self.width - 1e-6) / self.width * self.nx
        y = pos[:, 1].clamp(0, self.width - 1e-6) / self.width * self.ny
        gx = x.long().clamp(0, self.nx - 1)
        gy = y.long().clamp(0, self.ny - 1)
        return self.grid[0][gx, gy]


@register_operator("seed_from_segmentation", family="seed", set="particle", kind="seed")
class SeedFromSegmentation(Seed):
    """Populate tissue -> cell -> particle from a measured instance segmentation. Runs once.

    Was `kind="exchange"` (an `Exchange` subclass reusing the field-sampling machinery for
    its numerics) with a `family="seed"` tag that already said what it actually was; the
    mismatch let it masquerade as ordinary dynamics and skip the seed lifecycle guarantees
    (never scheduled, runs once, before frame 0) -- exactly the case `Seed` exists to rule
    out. The numerics (reading a field, scattering onto particles) are unchanged; only the
    lifecycle classification is corrected.
    """

    EMIT = None
    # It establishes the configuration -- where the cells are and which particles belong to
    # them -- and writes the state buffer directly to do it. The engine's integration
    # invariant forbids that for a dynamics operator, correctly; a Seed is exempted because
    # establishing x_0 IS writing the state buffer (see base.Seed).
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["from"]
    SUPPORTED_DIMS = [2]
    MECHANISM_TAGS = ["instance_segmentation", "cell_identity", "heterogeneous_material"]
    PARAM_ROLES = {"youngs_min": "param_lo", "youngs_max": "param_hi",
                   "from": "label_field", "cell_set": "middle_level"}
    REFERENCE = "instance segmentation measured from the beat; see prototype/cardio_cells"

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("from")
        self.at = params.get("_at", "mpm_particle")
        self.cell_set = params.get("cell_set", "cell")
        self.y_lo = float(params.get("youngs_min", 40.0))
        self.y_hi = float(params.get("youngs_max", 220.0))
        self.props = params.get("props")               # optional measured per-cell json
        self.jitter = float(params.get("jitter", 0.0))
        self._done = False

    def _cell_values(self, n_cells, device):
        """Per-cell Young's modulus: from the MEASURED beat when a props file is given, else a
        deterministic spread so the tissue is heterogeneous but reproducible.

        Measured is the interesting case. A cell that moved little in the recording is either stiff
        or weakly contractile; mapping amplitude to stiffness INVERSELY is one hypothesis about
        which, it is stated here rather than hidden, and the alternative (amplitude -> contraction
        gain) is the same one line the other way round.
        """
        if self.props:
            path = self.props if os.path.isabs(self.props) else graphs_data_path(self.props)
            if os.path.exists(path):
                d = json.load(open(path))
                amp = torch.tensor([d.get(str(k), {}).get("amp", float("nan"))
                                    for k in range(1, n_cells + 1)], device=device)
                good = torch.isfinite(amp)
                if good.any():
                    a = amp.clone()
                    a[~good] = a[good].median()
                    lo, hi = torch.quantile(a, 0.05), torch.quantile(a, 0.95)
                    u = ((a - lo) / (hi - lo + 1e-9)).clamp(0, 1)
                    return self.y_lo + (1.0 - u) * (self.y_hi - self.y_lo), "measured beat amplitude"
        g = torch.Generator(device="cpu").manual_seed(12345)
        u = torch.rand(n_cells, generator=g).to(device)
        return self.y_lo + u * (self.y_hi - self.y_lo), "deterministic spread (no props file)"

    def forward(self, H, mask=None):
        if self._done:
            return {}
        self._done = True
        lvl = H.level(self.at)
        fld = H.fields[self.field_name]
        dev = lvl.state.device
        px0, px1 = lvl.state_schema["pos"]
        n_cells = int(fld.n_labels)

        # ---- where each label lives, in world coordinates ---------------------------------
        gridl = fld.grid[0]                                     # [nx,ny] int64
        nx, ny = gridl.shape
        gx, gy = torch.meshgrid(torch.arange(nx, device=dev), torch.arange(ny, device=dev),
                                indexing="ij")
        flat = gridl.reshape(-1)
        wx = (gx.reshape(-1).double() + 0.5) / nx * fld.width
        wy = (gy.reshape(-1).double() + 0.5) / ny * fld.width
        inside = flat > 0
        lab_in, wx_in, wy_in = flat[inside], wx[inside], wy[inside]
        cnt = torch.bincount(lab_in, minlength=n_cells + 1).clamp(min=1)
        cx = torch.bincount(lab_in, weights=wx_in, minlength=n_cells + 1) / cnt
        cy = torch.bincount(lab_in, weights=wy_in, minlength=n_cells + 1) / cnt

        # ---- the CELL level moves onto its own segmented cell -----------------------------
        moved_cells = 0
        if self.cell_set in H.levels:
            cl = H.level(self.cell_set)
            cx0, cx1 = cl.state_schema["pos"]
            m = min(cl.n, n_cells)
            st = cl.state.clone()
            st[:m, cx0] = cx[1:m + 1].float()
            st[:m, cx0 + 1] = cy[1:m + 1].float()
            cl.state = st
            moved_cells = m
            if cl.n != n_cells:
                print(f"  [seed_from_segmentation] the {self.cell_set!r} set has {cl.n} entities "
                      f"and the map has {n_cells} cells -- seeding {m}. Declare "
                      f"per_parent: {n_cells} to use all of them.", flush=True)

        # ---- each particle is placed INSIDE its own cell's mask ---------------------------
        # ordering by label makes the members of one cell contiguous, so a particle can be given a
        # pixel of its OWN cell by index arithmetic instead of a python loop over 472 cells
        order = torch.argsort(lab_in)
        lab_s, wx_s, wy_s = lab_in[order], wx_in[order], wy_in[order]
        start = torch.cumsum(torch.bincount(lab_s, minlength=n_cells + 1), 0) - \
            torch.bincount(lab_s, minlength=n_cells + 1)

        pidx = lvl.parent if lvl.parent is not None else torch.zeros(lvl.n, dtype=torch.long,
                                                                     device=dev)
        pcell = (pidx % n_cells) + 1 if lvl.parent is not None else None
        if pcell is None or moved_cells == 0:
            # no declared cell level: assign each particle the label it already sits on
            pos = lvl.state[:, px0:px1]
            cid = fld.sample_label(pos)
        else:
            cid = pcell.clamp(1, n_cells)
            g = torch.Generator(device="cpu").manual_seed(777)
            u = torch.rand(lvl.n, generator=g).to(dev)
            k = (start[cid] + (u * cnt[cid].float()).long().clamp(max=0 + cnt[cid] - 1))
            k = k.clamp(0, lab_s.numel() - 1)
            newpos = torch.stack([wx_s[k].float(), wy_s[k].float()], 1)
            if self.jitter > 0:
                newpos = newpos + (torch.rand_like(newpos) - 0.5) * self.jitter
            st = lvl.state.clone(); st[:, px0:px1] = newpos; lvl.state = st

        # ---- one material per cell, shared exactly by its particles -----------------------
        yc, how = self._cell_values(n_cells, dev)
        y_all = torch.cat([yc[:1], yc])                          # index 0 = background, unused
        p_y = y_all[cid.clamp(0, n_cells)]
        from plexus.models.entities import _lame
        mu, la = _lame(p_y)
        liquid = getattr(lvl, "is_liquid", None)
        if liquid is not None:
            mu = torch.where(liquid, torch.zeros_like(mu), mu)
        lvl.mu, lvl.la = mu, la
        for nm, val in (("youngs", p_y), ("cell_id", cid.float())):
            if nm in getattr(lvl, "_buffers", {}):
                setattr(lvl, nm, val)
            else:
                lvl.register_buffer(nm, val)

        print(f"  [seed_from_segmentation] {n_cells} cells from {self.field_name!r}; "
              f"{lvl.n} particles ({lvl.n / max(n_cells,1):.0f} per cell); "
              f"youngs {float(yc.min()):.0f}-{float(yc.max()):.0f} from {how}; "
              f"cell centres seeded: {moved_cells}", flush=True)
        return {}
