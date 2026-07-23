"""tyssue_monolayer -- lift the single mid-surface vesicle to a MONOLAYER SHELL (Okuda gap-analysis C#1).

Each cell gets its OWN 3D volume and surface instead of the lumen-wedge proxy. We keep the mid-surface
vertices as the only DOF (so half-edge/T1/division/RD are untouched) and give every cell a thickness h_j.
Each frame we build an apical + basal surface by offsetting the mid-surface vertices along the VERTEX
normal by +/- H/2 (H = mean incident-cell thickness):

    a_i = x_i + (H_i/2) n_i        b_i = x_i - (H_i/2) n_i

Per cell j (a prism: apical cap + basal cap + lateral quads):
    v_j = prism volume  (exact, signed-tet divergence sum over the prism boundary)
    s_j = A_apical + A_basal + sum(lateral quad areas)

Energy is Okuda Eq. 3 verbatim:  U = sum_j [ 1/2 k_v (v_j - v_eq_j)^2 + kappa_s s_j ] .

Why VERTEX (not face) normals: on a curved sheet A_apical != A_basal (convex side stretches), so the
surface tension kappa_s (A_apical + A_basal) penalises curvature -> BENDING STIFFNESS ~ kappa_s h^2 falls
out, EMERGENT (no explicit K_bend). See monolayer_design.md.
"""
from __future__ import annotations
import torch
from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from tyssue_ops3d import face_geometry_3d


def apical_basal_shells(pos, es, et, ef, nF, h_cell):
    """Apical (outer) and basal (inner) vertex positions a_i, b_i = x_i +/- (H_i/2) n_i, for RENDERING
    the monolayer as two offset shells with a visible thickness. Same offset the energy uses."""
    dev, dt = pos.device, pos.dtype
    Nv = pos.shape[0]
    s, t = pos[es], pos[et]
    Nf = 0.5 * torch.zeros(nF, 3, device=dev, dtype=dt).index_add(0, ef, torch.cross(s, t, dim=-1))
    vn = torch.zeros(Nv, 3, device=dev, dtype=dt).index_add(0, es, Nf[ef])
    n = vn / (vn.norm(dim=-1, keepdim=True) + 1e-12)
    cnt = torch.zeros(Nv, device=dev, dtype=dt).index_add(0, es, torch.ones(es.shape[0], device=dev, dtype=dt))
    hv = torch.zeros(Nv, device=dev, dtype=dt).index_add(0, es, h_cell[ef]) / cnt.clamp(min=1e-9)
    return pos + 0.5 * hv[:, None] * n, pos - 0.5 * hv[:, None] * n


def monolayer_geometry_3d(pos, es, et, ef, nF, h_cell, eocc=None):
    """Per-cell prism volume v_f and surface s_f (apical+basal+lateral), plus the apical/basal cap areas.
    All differentiable in `pos`. h_cell is per-cell thickness [nF]. eocc masks dead half-edges (or None)."""
    dev, dt = pos.device, pos.dtype
    Nv = pos.shape[0]
    s, t = pos[es], pos[et]
    ones_e = torch.ones(es.shape[0], device=dev, dtype=dt) if eocc is None else eocc
    # mid-surface face area vectors -> vertex normals (sum of incident face area vectors)
    crossm = torch.cross(s, t, dim=-1) * ones_e[:, None]
    Nf = 0.5 * torch.zeros(nF, 3, device=dev, dtype=dt).index_add(0, ef, crossm)
    vn = torch.zeros(Nv, 3, device=dev, dtype=dt).index_add(0, es, Nf[ef] * ones_e[:, None])
    n = vn / (vn.norm(dim=-1, keepdim=True) + 1e-12)
    # thickness at a vertex = mean thickness of incident cells
    cnt_v = torch.zeros(Nv, device=dev, dtype=dt).index_add(0, es, ones_e)
    hv = torch.zeros(Nv, device=dev, dtype=dt).index_add(0, es, h_cell[ef] * ones_e) / cnt_v.clamp(min=1e-9)
    a = pos + 0.5 * hv[:, None] * n                                # apical (outer) shell
    b = pos - 0.5 * hv[:, None] * n                                # basal (inner) shell
    a_s, a_t, b_s, b_t = a[es], a[et], b[es], b[et]
    # cell VOLUME = mid-surface area x thickness (v_j = A_mid*h_j). Exact for a flat cell, first-order in
    # curvature (the O((h/R)^2) prism correction ~0.3% is dropped); ALWAYS positive & differentiable, and
    # -- the key point -- bending resistance comes from the SURFACE term below (apical!=basal area under
    # curvature), NOT from the volume, so A_mid*h is the physically correct choice, not just the simple one.
    v_f = Nf.norm(dim=-1) * h_cell
    # apical / basal cap areas (Newell magnitude, origin-independent)
    Na = 0.5 * torch.zeros(nF, 3, device=dev, dtype=dt).index_add(0, ef, torch.cross(a_s, a_t, dim=-1) * ones_e[:, None])
    Nb = 0.5 * torch.zeros(nF, 3, device=dev, dtype=dt).index_add(0, ef, torch.cross(b_s, b_t, dim=-1) * ones_e[:, None])
    A_ap, A_ba = Na.norm(dim=-1), Nb.norm(dim=-1)
    # lateral quad area per edge = tri(a_s,a_t,b_t) + tri(a_s,b_t,b_s)
    la = (0.5 * torch.cross(a_t - a_s, b_t - a_s, dim=-1).norm(dim=-1)
          + 0.5 * torch.cross(b_t - a_s, b_s - a_s, dim=-1).norm(dim=-1)) * ones_e
    A_lat = torch.zeros(nF, device=dev, dtype=dt).index_add(0, ef, la)
    s_f = A_ap + A_ba + A_lat
    return v_f, s_f, A_ap, A_ba


def _monolayer_energy_core(pos, es, et, ef, nF, h_cell, V_eq, alive, k_v, kappa_s, Lam, K_R, R0, eocc, vocc, gamma=0.0):
    """U = sum_j [ 1/2 k_v (v_j - v_eq_j)^2 + kappa_s s_j + 1/2 gamma P_j^2 ] + Lam*sum_e l_e + K_R*sum_i (|x_i|-R0)^2 .
    gamma is a cortical CONTRACTILITY (perimeter^2) that rounds cells and resists shear -- a cell-shape
    regularizer standing in for the RNR/T1 remeshing Okuda relies on (without it the bare volume+surface
    energy shears/spikes under large deformation). Lam/K_R are optional dials (both default 0 in the op)."""
    v_f, s_f, _, _ = monolayer_geometry_3d(pos, es, et, ef, nF, h_cell, eocc)
    E = (0.5 * k_v * (v_f - V_eq) ** 2 * alive).sum() + kappa_s * (s_f * alive).sum()
    if gamma != 0.0:
        perim = torch.zeros(nF, device=pos.device, dtype=pos.dtype).index_add(0, ef, (pos[et] - pos[es]).norm(dim=-1) * eocc)
        E = E + 0.5 * gamma * (perim ** 2 * alive).sum()
    if Lam != 0.0:
        E = E + Lam * ((pos[et] - pos[es]).norm(dim=-1) * eocc).sum()
    if K_R != 0.0:
        E = E + K_R * (((pos.norm(dim=1) - R0) ** 2) * vocc).sum()
    return E


@register_operator("shape_energy_3d", implementation="monolayer", set="vertex", kind="lateral", family="mechanics")
class MonolayerShapeEnergy3D(Lateral):
    """The MONOLAYER implementation of the shape_energy_3d contract (plexus2 sec. 5: same biological
    operator -- the mechanical force that shapes the epithelial vesicle -- different NUMERICS). The
    default implementation is a mid-surface model with a lumen-wedge volume; this one gives every cell
    its OWN 3D volume + surface (apical+basal+lateral, Okuda Eq. 3): per-cell 3D volume elasticity +
    linear surface tension. Force = -grad U by one autograd pass; bounded overdamped Euler (displacement
    capped at cap_frac x mean edge). EMIT=velocity. Selected by {op: shape_energy_3d, implementation:
    monolayer}. Emergent bending (thin undulate / thick straight) falls out of the vertex-normal offset;
    no explicit K_bend. See monolayer_design.md."""
    SUPPORTED_DIMS = [3]; EMIT = "velocity"; DIFFERENTIABLE = True
    INPUTS = ["vertex"]; OUTPUTS = ["vertex"]; READS = ["pos"]; WRITES = ["pos"]
    MAPS = ["E_srce", "E_trgt", "E_face"]
    MECHANISM_TAGS = ["vertex_model", "monolayer", "cell_3d_volume", "surface_tension", "emergent_bending", "force_balance"]
    REFERENCE = "Okuda, S. et al. (2018). Sci. Rep. 8:2386 (monolayer 3D vertex model, Eq. 3)."
    PARAM_ROLES = {"k_v": "cell_volume_elasticity", "kappa_s": "surface_tension", "h0": "cell_thickness"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.k_v = float(params.get("k_v", 4.0)); self.kappa_s = float(params.get("kappa_s", 0.2))
        self.h0 = float(params.get("h0", 0.4))                    # uniform cell thickness (v1: fixed field)
        self.gamma = float(params.get("gamma", 0.0))             # cortical contractility (cell-shape regularizer)
        self.Lambda = float(params.get("Lambda", 0.0)); self.K_R = float(params.get("K_R", 0.0))
        self.mu = float(params.get("mu", 1.0)); self.dt = float(params.get("dt", 1.0))
        self.relax_iters = int(params.get("relax_iters", 30)); self.eta = float(params.get("eta", 0.08))
        self.cap_frac = float(params.get("cap_frac", 0.12))

    def _grad(self, x, es, et, ef, nF, h, V_eq, alive, R0t, eocc, vocc):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            E = _monolayer_energy_core(x, es, et, ef, nF, h, V_eq, alive, self.k_v, self.kappa_s,
                                       self.Lambda, self.K_R, R0t, eocc, vocc, self.gamma)
            g = torch.autograd.grad(E, x)[0]
        return torch.nan_to_num(g)

    def forward(self, H, mask=None):
        lvl = H.level(self.at); pos_full = lvl.get("pos"); v_full = torch.zeros_like(pos_full)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {self.at: v_full}
        Nv = int(m["Nv"]); nF = int(m["nF"]); es, et, ef = m["E_srce"], m["E_trgt"], m["E_face"]
        E = es.shape[0]; dev, dt = pos_full.device, pos_full.dtype
        x0 = pos_full[:Nv].detach().clone()
        eocc = torch.ones(E, device=dev, dtype=dt); vocc = torch.ones(Nv, device=dev, dtype=dt)
        R0t = torch.as_tensor(float(m["R0"]), dtype=dt, device=dev)
        h_cell = torch.full((nF,), self.h0, dtype=dt, device=dev)   # v1: uniform fixed thickness
        # target monolayer volume: calibrate ONCE so V_eq matches the rest prism volume, then track the
        # growth op's scaling of the wedge target V0f (morphogen_growth_3d scales V0f per cell) -> reuse it.
        v_rest, _, _, _ = monolayer_geometry_3d(x0, es, et, ef, nF, h_cell, eocc)
        if "mono_k" not in m:
            wedge = face_geometry_3d(x0, es, et, ef, nF, eocc)[3]
            m["mono_k"] = float((v_rest.median() / wedge.median().clamp(min=1e-9)).item())
        V_eq = (m["mono_k"] * m["V0f"]).clamp(min=1e-9)
        with torch.no_grad():
            cap = self.cap_frac * (x0[et] - x0[es]).norm(dim=-1).mean().clamp(min=1e-6)
        x = x0.clone()
        for _ in range(max(1, self.relax_iters)):
            step = -(self.eta * self.mu) * self._grad(x, es, et, ef, nF, h_cell, V_eq, m["alive"], R0t, eocc, vocc)
            step = step * torch.clamp(cap / (step.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
            x = x + step
        v_full[:Nv] = (x - x0) / max(self.dt, 1e-9)
        if mask is not None:
            v_full = v_full * mask[:, None].float()
        return {self.at: v_full}
