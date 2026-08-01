"""Stillinger--Weber interaction: a two-body well + a THREE-BODY angular penalty.

Promoted from the ice-nucleation prototype (``prototype/ice``), where decomposing the mW
monatomic-water model surfaced the registry's first genuinely *many-body* force. Every prior
interaction operator (attraction_repulsion, squared_law, cohesion, ...) is pairwise; the
Stillinger--Weber three-body term penalises non-tetrahedral bond angles and so cannot be
expressed by any pairwise law without changing its meaning -- exactly the plexus2 criterion for
a new operator contract. It is broadly reusable: the SAME functional form models the whole
family of tetrahedral liquids, the substance selected by two parameters (the tetrahedral
strength ``lam`` and preferred cosine ``cos0``):

    monatomic water (mW)  lam=23.15  cos0=-1/3      Molinero & Moore (2009)
    silicon               lam=21.0   cos0=-1/3      Stillinger & Weber (1985)
    germanium             lam=20.0   cos0=-1/3
    a 2D honeycomb analog lam~23     cos0=-1/2       (120 deg)

Energy (reduced units, lengths in ``sigma``, energy in ``eps``):

    E = A eps sum_{i<j, r<a} (B r^-p - 1) exp(1/(r-a))                           [two-body]
      + lam eps sum_i sum_{j<k} (cos_jik - cos0)^2 exp(gamma/(r_ij-a)) exp(gamma/(r_ik-a))

with the canonical SW form/scale constants ``A=7.049556277, B=0.6022245584, p=4, gamma=1.2``
and cutoff ``a=1.8`` (all terms and forces vanish smoothly at ``r = a sigma``). The force is one
autograd pass over ``E`` (``DIFFERENTIABLE``), over a minimum-image neighbour list the operator
builds each tick (the potential's cutoff ``a sigma`` IS the neighbour cutoff -- an implicit
per-tick rewire, so no separate ``radius_graph`` is required). Newtonian: ``EMIT=acceleration``.

Cost is O(N^2) in the neighbour build (dense min-image), which is fine at MD scale
(N up to a few thousand); a cell-list implementation is a future numerical variant of this same
contract. This is a self-contained port of the validated ``mw_forces`` prototype operator.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator

# canonical Stillinger--Weber form/scale constants (Stillinger & Weber 1985; mW: Molinero 2009)
_A, _B, _P = 7.049556277, 0.6022245584, 4.0


@register_operator("stillinger_weber", set="particle", kind="lateral", family="interaction",
                   implementation="autograd")
class StillingerWeber(Lateral):
    EMIT = "acceleration"                       # d2x/dt2 = force / m  (Newtonian, engine-integrated)
    SUPPORTED_DIMS = [2, 3]                      # dimension-generic (reads D = pos.shape[-1])
    DIFFERENTIABLE = True                        # force = -grad E by autograd
    INPUTS = ["particle"]; OUTPUTS = ["particle"]
    READS = ["pos"]; WRITES = ["vel"]
    MAPS = []                                    # builds its own min-image neighbour list (implicit rewire)
    REQUIRES_PARAMS = []                         # all params default to mW water; none mandatory
    MECHANISM_TAGS = ["tetrahedral_network", "three_body", "angular_interaction",
                      "stillinger_weber", "monatomic_water", "directional_bonding",
                      "mechanical_interaction"]
    PARAM_ROLES = {"lam": "tetrahedral_strength", "cos0": "preferred_cos_angle",
                   "gamma": "three_body_range", "a": "cutoff_over_sigma", "eps": "energy_scale",
                   "maxnb": "neighbour_buffer"}
    REFERENCE = ("Stillinger, F. H. & Weber, T. A. (1985). Phys. Rev. B 31:5262; "
                 "mW water: Molinero, V. & Moore, E. B. (2009). J. Phys. Chem. B 113:4008. "
                 "Promoted from plexus prototype/ice (mw_forces).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.lam = float(params.get("lam", 23.15))          # mW water default
        self.cos0 = float(params.get("cos0", -1.0 / 3.0))   # tetrahedral (109.47 deg)
        self.gamma = float(params.get("gamma", 1.2))
        self.a = float(params.get("a", 1.8))                # cutoff / sigma
        self.eps = float(params.get("eps", 1.0))            # overall energy scale
        self.maxnb = int(params.get("maxnb", 40))           # padded neighbour count

    # --- geometry ---------------------------------------------------------- #
    def _box(self, H, D):
        ws = getattr(H, "world_size", None)
        if ws is None:
            return None, False
        return ws[:D], bool(getattr(H, "periodic", False))

    def _min_image(self, dv, box, periodic):
        return dv - box * torch.round(dv / box) if (periodic and box is not None) else dv

    def _neighbours(self, pos, box, periodic):
        with torch.no_grad():
            d = self._min_image(pos[:, None, :] - pos[None, :, :], box, periodic)
            r = d.norm(dim=-1); r.fill_diagonal_(1e9)
            k = min(self.maxnb, pos.shape[0] - 1)
            rr, idx = torch.topk(r, k, largest=False)
            return idx, rr < self.a

    # --- energy (two-body + three-body) ------------------------------------ #
    def _energy(self, pos, nb, valid, box, periodic):
        a = self.a
        d = self._min_image(pos[:, None, :] - pos[nb], box, periodic)     # [N,k,D]
        r = d.norm(dim=-1)
        inside = valid & (r < a) & (r > 1e-4)
        rr = r.clamp(min=1e-4)
        arg2 = torch.where(inside, 1.0 / (rr - a), torch.full_like(rr, -1e9))
        phi2 = _A * (_B * rr.pow(-_P) - 1.0) * torch.exp(arg2)
        E2 = 0.5 * (phi2 * inside).sum()                                  # symmetric list -> halve
        u = d / rr.unsqueeze(-1)
        cos = torch.einsum("nmc,nkc->nmk", u, u)                          # [N,k,k]
        arg3 = torch.where(inside, self.gamma / (rr - a), torch.full_like(rr, -1e9))
        h = torch.exp(arg3)                                              # ~0 outside cutoff
        k = h.shape[1]
        triu = torch.triu(torch.ones(k, k, device=pos.device), diagonal=1).bool()
        pair = (h[:, :, None] * h[:, None, :]) * (cos - self.cos0) ** 2
        E3 = self.lam * (pair * triu[None]).sum()
        return self.eps * (E2 + E3)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        N, D = pos.shape[0], pos.shape[-1]
        box, periodic = self._box(H, D)
        if box is not None:
            box = box.to(pos.device)
        nb, valid = self._neighbours(pos, box, periodic)
        with torch.enable_grad():                                        # engine runs under no_grad
            p = pos.detach().requires_grad_(True)
            E = self._energy(p, nb, valid, box, periodic)
            grad, = torch.autograd.grad(E, p)
        acc = torch.nan_to_num(-grad) * lvl.occ[:, None]                 # force / m (m = 1); dormant -> 0
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}
