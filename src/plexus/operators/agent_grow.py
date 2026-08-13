"""agent_grow (cell, structural): CELL rest-VOLUME growth -- the biological growth primitive.

Growth belongs to the CELL, not to the material points. Each cell carries a biological growth STATE
-- a rest-volume multiplier `grow_V` (1.0 = birth size) -- that `agent_grow` advances by a GROWTH LAW.
A backend REALIZATION then discretises that state by waking dormant reserve MPM particles (declare
`grow_reserve:` on the mpm_particle child) so the live material tracks the target volume. The biology
(rest-volume growth) is the invariant; particle addition is merely the current numerical realization
-- swap the discretisation (adaptive sampling, another continuum) and the growth LAW stays identical.

This is a change of ONTOLOGY, not just another behaviour: earlier operators answer "how do cells
move?"; growth answers "how does biological material itself increase?". So the abstraction is CELL
GROWTH, not particle activation.

GROWTH LAWS (`mode`):
  isotropic    logistic rest-volume growth; new material added in all directions -> rounds by elasticity
  anisotropic  same law, new material biased along `axis` (`aniso`) -> a finger / bud
  tip          growth localised at the leading edge along `axis` (`tip`) -> tip-driven elongation
`rate` is the specific growth rate (dV/dt = rate*V*(1-V/`target`), logistic -> `target` = size/contact
inhibition). `stress_gain`>0 couples the rate to local deformation (mechano-inhibition: growth slows in
compressed tissue). An anisotropic bud ROUNDS on its own once growth isotropises / relaxes (emergent,
not scripted). Separates MASS INCREASE (agent_grow) from TOPOLOGY CHANGE (`agent_divide`, which then
REPOPULATES the grown volume).

`rate<=0` is a byte-identical no-op (the reserve is inert -- p2g/g2p mask by occ). kind=structural;
mutates `cell.grow_V` + the child occ/state, returns {}.
"""
from __future__ import annotations

import torch

from plexus.models.base import Structural
from plexus.models.registry import register_operator


@register_operator("agent_grow", family="growth", set="cell", kind="structural")
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
        self.mode = str(params.get("mode", "isotropic"))      # isotropic | anisotropic | tip
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
