"""Integrins on the basal surface of an epithelium: a contained set of point entities, one per
receptor cluster, that ride the tissue, are born and retire at the cell's own rates, and are
shared between daughters at division.

THE OBJECT. An integrin cluster is a POINT with a position and a parent cell. It is not matter:
no deformation gradient, no grid, no mass -- so no MPM operator ever touches this set. It is a
contained set (`parent: cell`, `parent_pos: centroid`) with a dormant reserve, exactly the
mechanism every other cardinality-changing set uses (`occ`, `spawn`, `kill`).

THREE OPERATORS, EACH ANSWERING ONE QUESTION.
  integrin_seed      where do they start: on each cell's basal face, uniformly by area (a seed;
                     runs once).
  integrin_project   how do they stay on the tissue: every frame each particle is moved to the
                     nearest point of its parent's basal face. The face is the corral -- a
                     particle pushed past the cell's edge by the repulsion between integrins is
                     put back on the boundary, not handed to the neighbour. A division hands the
                     daughter exactly half of the mother's particles, the half nearest the
                     daughter's basal centroid; a T1 swap changes no count. (structural on the
                     integrin set: it writes positions and parents in place.)
  integrin_express   how many there are: per cell, birth at `integrin_s` per unit basal area per
                     unit time and retirement with probability dt / `integrin_tau` per particle,
                     the rate law of archive rung 05d (`dN/dt = s_i - N/tau_i`, receptor
                     conserved under binding) with the receptor now countable. Writes the count
                     back as `n_integrin` on the cell. (structural on the integrin set.)

THE CELL CARRIES THE RATES, `integrin_s` and `integrin_tau` (state blocks, width 1), so they can
differ per cell and per region and be written later by a signalling operator; the particle
carries nothing but its position. Both operators read the parent's blocks through the
containment map (`lvl.parent`), the engine's broadcast, and write the count through the
aggregate (a scatter-add of `occ` along the same map).

WHICH SURFACE IS BASAL. `basal = pos - sep` on an apico-basal tissue, the model's own convention
(`vertex_ops.apicobasal_geometry_3d`). On the archived spheroids the apical ring is the outer
one, so this is the surface facing the lumen; `seed_mesh[apicobasal] apical: in` flips it and
does NOT yet reproduce the working point (r 4.85 vs 4.53 at frame 0, no division by frame 75:
something in the seeded targets reads the apical ring as the outer one). Until that is
settled, this module reads whichever surface the tissue says is basal.

Reference: the receptor rate law and its conservation gate, discovery_okuda/ops/adhesion_ops.py
(rung 05d, G30); integrin cluster spacing ~555 nm, Changede & Sheetz (2017) Dev. Cell 40:1.
"""
from __future__ import annotations

import numpy as np
import torch

from plexus.models.base import Seed, Structural
from plexus.models.registry import register_operator
from plexus.operators.vertex_ops import resolve_cell_set

# DORMANT SLOTS ARE PARKED FAR OFF-DOMAIN. A neighbour graph over the buffer (`radius_graph`) sees
# every slot; a million dormant integrins left at one point are a million neighbours of each
# other and of nothing else, and the pairwise block mask alone was 29 GB. Parked here they
# neighbour nothing; a slot that wakes is placed on its face before anything reads it.
PARK = -1.0e6


# ---------------------------------------------------------------------------------------------
# the basal surface of a tissue, as fans of triangles: (basal centroid, b[es], b[et]) per half-edge
# ---------------------------------------------------------------------------------------------
def _basal_fans(H, tissue_set):
    """The tissue's basal surface as one triangle per live half-edge: corners A (the face's basal
    centroid), B (the source vertex), C (the target vertex), and the face each triangle belongs
    to. Returns None when the tissue carries no mesh yet."""
    lvl = H.level(tissue_set)
    m = getattr(lvl, "_mesh", None) or getattr(lvl, "mesh", None)
    if m is None or not int(m.get("Nv", 0) or 0):
        return None
    Nv, nF = int(m["Nv"]), int(m["nF"])
    pos = lvl.get("pos")[:Nv]
    sep = lvl.get("sep")[:Nv] if "sep" in lvl.state_schema else torch.zeros_like(pos)
    b = pos - sep                                            # the basal ring, the model's convention
    es, et, ef = m["E_srce"], m["E_trgt"], m["E_face"]
    eocc = m.get("eocc", None)
    live = (ef >= 0) & (ef < nF) & (es >= 0) & (et >= 0)
    if eocc is not None:
        live = live & (eocc[: es.shape[0]] > 0)
    es, et, ef = es[live], et[live], ef[live]
    cnt = torch.bincount(ef, minlength=nF).clamp_min(1).to(b.dtype)
    cen = torch.zeros(nF, 3, device=b.device, dtype=b.dtype).index_add_(0, ef, b[es]) / cnt[:, None]
    A, B, C = cen[ef], b[es], b[et]
    return dict(A=A, B=B, C=C, face=ef, nF=nF, cen=cen,
                area=0.5 * torch.cross(B - A, C - A, dim=1).norm(dim=1))


def _face_area(fans):
    """Basal area per face, summed over its fan."""
    a = torch.zeros(fans["nF"], device=fans["A"].device, dtype=fans["A"].dtype)
    return a.index_add_(0, fans["face"], fans["area"])


def _sample_on_faces(fans, faces, gen):
    """One uniform point per requested face index, on its basal fan (triangle picked by area)."""
    dev = fans["A"].device
    nF = fans["nF"]
    # CSR of triangles per face
    order = torch.argsort(fans["face"])
    f_sorted = fans["face"][order]
    starts = torch.searchsorted(f_sorted, torch.arange(nF, device=dev))
    ends = torch.searchsorted(f_sorted, torch.arange(nF, device=dev), right=True)
    out = torch.empty(faces.shape[0], 3, device=dev, dtype=fans["A"].dtype)
    tri_choice = torch.empty(faces.shape[0], dtype=torch.long, device=dev)
    # pick a triangle per particle with probability proportional to its area (loop over distinct
    # fan sizes is avoided by a padded gather: fans have at most ~12 triangles)
    deg = (ends - starts)
    maxdeg = int(deg.max().item()) if deg.numel() else 0
    idx = starts[faces][:, None] + torch.arange(maxdeg, device=dev)[None, :]
    valid = torch.arange(maxdeg, device=dev)[None, :] < deg[faces][:, None]
    idx = torch.where(valid, idx, torch.zeros_like(idx))
    tri = order[idx]                                          # [n, maxdeg] triangle ids
    w = torch.where(valid, fans["area"][tri], torch.zeros_like(fans["area"][tri]))
    w = w / w.sum(1, keepdim=True).clamp_min(1e-20)
    u = torch.rand(faces.shape[0], generator=gen, device=dev)
    pick = (torch.cumsum(w, 1) < u[:, None]).sum(1).clamp(max=maxdeg - 1)
    tri_choice = tri[torch.arange(faces.shape[0], device=dev), pick]
    r1 = torch.rand(faces.shape[0], generator=gen, device=dev)
    r2 = torch.rand(faces.shape[0], generator=gen, device=dev)
    s = torch.sqrt(r1)
    w0, w1, w2 = 1.0 - s, s * (1.0 - r2), s * r2               # uniform barycentric on a triangle
    A, B, C = fans["A"][tri_choice], fans["B"][tri_choice], fans["C"][tri_choice]
    out = w0[:, None] * A + w1[:, None] * B + w2[:, None] * C
    return out


def _project_to_faces(fans, faces, x):
    """The nearest point of each particle's own face (its basal fan) to `x`. Padded over the
    fan's triangles; the closest triangle wins."""
    dev = x.device
    nF = fans["nF"]
    order = torch.argsort(fans["face"])
    f_sorted = fans["face"][order]
    starts = torch.searchsorted(f_sorted, torch.arange(nF, device=dev))
    ends = torch.searchsorted(f_sorted, torch.arange(nF, device=dev), right=True)
    deg = ends - starts
    maxdeg = int(deg.max().item()) if deg.numel() else 0
    idx = starts[faces][:, None] + torch.arange(maxdeg, device=dev)[None, :]
    valid = torch.arange(maxdeg, device=dev)[None, :] < deg[faces][:, None]
    idx = torch.where(valid, idx, torch.zeros_like(idx))
    tri = order[idx]                                          # [n, maxdeg]
    A, B, C = fans["A"][tri], fans["B"][tri], fans["C"][tri]  # [n, maxdeg, 3]
    P = x[:, None, :]
    # closest point on a triangle (Ericson, Real-Time Collision Detection, 5.1.5), vectorised
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = P - B
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = P - C
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    denom = (va + vb + vc).clamp_min(1e-30)
    v = (vb / denom)[..., None]; w = (vc / denom)[..., None]
    q = A + ab * v + ac * w                                   # inside
    q = torch.where(((d1 <= 0) & (d2 <= 0))[..., None], A, q)
    q = torch.where(((d3 >= 0) & (d4 <= d3))[..., None], B, q)
    q = torch.where(((d6 >= 0) & (d5 <= d6))[..., None], C, q)
    vv = (d1 / (d1 - d3).where((d1 - d3) != 0, torch.ones_like(d1)))[..., None]
    q = torch.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], A + ab * vv, q)
    ww = (d2 / (d2 - d6).where((d2 - d6) != 0, torch.ones_like(d2)))[..., None]
    q = torch.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], A + ac * ww, q)
    den = ((d4 - d3) + (d5 - d6)).where(((d4 - d3) + (d5 - d6)) != 0, torch.ones_like(d4))
    ww2 = ((d4 - d3) / den)[..., None]
    q = torch.where(((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0))[..., None], B + (C - B) * ww2, q)
    dist = (q - P).norm(dim=-1)
    dist = torch.where(valid, dist, torch.full_like(dist, float("inf")))
    best = dist.argmin(dim=1)
    return q[torch.arange(x.shape[0], device=dev), best]


# ---------------------------------------------------------------------------------------------
@register_operator("integrin_seed", family="seed", set="particle", kind="seed")
class IntegrinSeed(Seed):
    """Place each cell's seeded integrins uniformly on its basal face, once.

    `density` (per unit basal area, in the run's length units) decides how many of the
    `per_parent` slots each cell wakes; the rest stay dormant for `integrin_express`. Any count
    above the cell's block is refused, printed, and the block is filled -- a spec that wants more
    declares a larger `per_parent`."""
    SUPPORTED_DIMS = (3,)                                 # a basal surface is a surface in space
    REQUIRES_PARAMS = ["tissue"]
    PARAM_ROLES = {"tissue": "tissue_set", "density": "receptor_density"}
    READS = ["pos"]
    WRITES = ["pos"]
    MECHANISM_TAGS = ["seed", "receptor_placement"]
    REFERENCE = "Plexus (this work); cluster spacing after Changede & Sheetz (2017) Dev. Cell 40:1."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "integrin")
        self.tissue = str(params["tissue"])
        self.density = float(params.get("density", 2.0))
        self.seed = int(params.get("seed", 0))
        # THE CELL'S RATES START HERE. `integrin_s` (births per unit basal area per unit time) and
        # `integrin_tau` (mean lifetime) are cell state so a later operator can vary them per cell;
        # this seed writes the spec's starting values into every cell slot, live or dormant, so a
        # daughter born later inherits them through the carry.
        self.s0 = float(params.get("s", 0.0))
        self.tau0 = float(params.get("tau", float("inf")))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        fans = _basal_fans(H, self.tissue)
        if fans is None:
            raise ValueError(f"integrin_seed: {self.tissue!r} carries no mesh at seed time; put seed_mesh before it")
        dev = lvl.state.device
        gen = torch.Generator(device=dev).manual_seed(self.seed)
        par = lvl.parent
        if par is None:
            raise ValueError(f"integrin_seed: set {self.at!r} declares no parent; it needs `parent: cell`")
        area = _face_area(fans)
        nF = fans["nF"]
        want = torch.round(area * self.density).long()               # per live face
        # every slot starts dormant; wake `want[f]` slots of each face's block
        lvl.occ[:] = 0.0
        slots = torch.arange(par.shape[0], device=dev)
        blk = torch.bincount(par, minlength=int(par.max().item()) + 1)
        offset = torch.cumsum(blk, 0) - blk
        p0, p1 = lvl.state_schema["pos"]
        wake = []
        for f in range(nF):
            k = int(min(want[f].item(), blk[f].item()))
            if k:
                wake.append(slots[offset[f]: offset[f] + k])
        if not wake:
            print(f"[integrin_seed] density {self.density} put no integrin on any of {nF} faces", flush=True)
            return {}
        wake = torch.cat(wake)
        lvl.occ[wake] = 1.0
        lvl.state[:, p0:p1] = PARK                                  # every slot parked ...
        lvl.state[wake, p0:p1] = _sample_on_faces(fans, par[wake], gen)
        if "vel" in lvl.state_schema:
            v0, v1 = lvl.state_schema["vel"]; lvl.state[wake, v0:v1] = 0.0
        cs = resolve_cell_set(H, self.tissue, None)
        clvl = H.level(cs)
        for name, val in (("integrin_s", self.s0), ("integrin_tau", self.tau0)):
            if name in clvl.state_schema:
                c0, _c1 = clvl.state_schema[name]
                clvl.state[:, c0] = val
        refused = int((want[:nF] - blk[:nF]).clamp_min(0).sum().item())
        print(f"[integrin_seed] {int(wake.numel())} integrins on {nF} basal faces "
              f"({self.density:g} per unit area; {refused} refused by the per-cell block)", flush=True)
        _write_count(H, lvl, self.tissue)
        return {}


def _write_count(H, lvl, tissue_set):
    """n_integrin on the cell set: live children per parent (the aggregate along the map)."""
    cs = resolve_cell_set(H, tissue_set, None)
    try:
        clvl = H.level(cs)
    except Exception:                                        # noqa: BLE001
        return
    if "n_integrin" not in clvl.state_schema:
        return
    c0, c1 = clvl.state_schema["n_integrin"]
    n = torch.bincount(lvl.parent[lvl.occ > 0], minlength=clvl.state.shape[0]).to(clvl.state.dtype)
    clvl.state[:, c0] = n[: clvl.state.shape[0]]


# ---------------------------------------------------------------------------------------------
@register_operator("integrin_project", family="mechanics", set="particle", kind="structural")
class IntegrinProject(Structural):
    """Keep every integrin on its parent cell's basal face; share them at division.

    Each frame, after the engine has integrated whatever velocity the repulsion emitted, each
    live particle is moved to the nearest point of its parent's basal fan. A particle whose
    parent is no longer live (the cell died or was refused) is retired.

    DIVISION. A cell that divided this frame is recognised by its `age` block dropping to zero
    with a live sibling born the same frame (the daughter is the face whose slot was dormant
    on the previous frame). The mother's particles are split: the half nearest the daughter's
    basal centroid go to the daughter. Exactly half, by count -- not by area -- as agreed."""
    READS = ["pos"]
    WRITES = ["pos"]
    SUPPORTED_DIMS = (3,)                                 # a basal surface is a surface in space
    REQUIRES_PARAMS = ["tissue"]
    PARAM_ROLES = {"tissue": "tissue_set"}
    MECHANISM_TAGS = ["surface_constraint", "inheritance_at_division"]
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "integrin")
        self.tissue = str(params["tissue"])
        self._prev_live = None                                # cell occupancy on the previous frame
        self._prev_age = None                                 # cell age on the previous frame

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        fans = _basal_fans(H, self.tissue)
        if fans is None:
            return {}
        dev = lvl.state.device
        cs = resolve_cell_set(H, self.tissue, None)
        clvl = H.level(cs)
        nF = fans["nF"]
        live_cells = clvl.occ[:nF] > 0 if getattr(clvl, "occ", None) is not None else torch.ones(nF, dtype=torch.bool, device=dev)
        par = lvl.parent
        occ = lvl.occ > 0
        # ---- division: newborn faces this frame, and their mothers ---------------------------
        # `cell_divide` records no lineage; it resets `age` on BOTH daughters and the newborn is the
        # face whose slot was dormant a frame ago. The mother is therefore the face, live a frame
        # ago, whose age just dropped, nearest the newborn's basal centroid.
        # the age `cell_divide` writes lives on the mesh table (it resets daughter A in place and
        # appends daughter B, so nF grows); the previous frame's arrays are padded to the new nF
        _m = getattr(H.level(self.tissue), "_mesh", None)
        _age = _m.get("age") if _m is not None else None
        age = (torch.as_tensor(_age, device=dev).reshape(-1)[:nF] if _age is not None
               else (clvl.get("age")[:nF, 0] if "age" in clvl.state_schema else None))
        if self._prev_live is not None and age is not None and self._prev_age is not None:
            def _pad(v, fill):
                out = torch.full((nF,), fill, dtype=v.dtype, device=dev); k = min(nF, v.shape[0]); out[:k] = v[:k]; return out
            prev_live = _pad(self._prev_live, False); prev_age = _pad(self._prev_age, 0.0)
            born = live_cells & ~prev_live
            reset = live_cells & prev_live & (age < prev_age)
            if bool(born.any()) and bool(reset.any()):
                cand = torch.nonzero(reset).flatten()
                for d in torch.nonzero(born).flatten().tolist():
                    dist = (fans["cen"][cand] - fans["cen"][d]).norm(dim=1)
                    mother = int(cand[dist.argmin()].item())
                    if mother == d:
                        continue
                    mine = torch.nonzero(occ & (par == mother)).flatten()
                    if mine.numel() < 2:
                        continue
                    p0, p1 = lvl.state_schema["pos"]
                    x = lvl.state[mine, p0:p1]
                    dd = (x - fans["cen"][d]).norm(dim=1)
                    k = mine.numel() // 2
                    give = mine[torch.argsort(dd)[:k]]              # the half nearest the daughter
                    par[give] = d
        self._prev_live = live_cells.clone()
        self._prev_age = age.clone() if age is not None else None
        # ---- retire orphans, project the rest -----------------------------------------------
        orphan = occ & ((par >= nF) | (~live_cells[par.clamp(max=nF - 1)]))
        if bool(orphan.any()):
            lvl.kill(torch.nonzero(orphan).flatten(), park=torch.full((3,), PARK, device=dev, dtype=lvl.state.dtype))
            occ = lvl.occ > 0
        idx = torch.nonzero(occ).flatten()
        if idx.numel():
            p0, p1 = lvl.state_schema["pos"]
            lvl.state[idx, p0:p1] = _project_to_faces(fans, par[idx], lvl.state[idx, p0:p1])
            if "vel" in lvl.state_schema:
                v0, v1 = lvl.state_schema["vel"]; lvl.state[idx, v0:v1] = 0.0
        _write_count(H, lvl, self.tissue)
        return {}


# ---------------------------------------------------------------------------------------------
@register_operator("integrin_express", family="population", set="particle", kind="structural")
class IntegrinExpress(Structural):
    """Birth and retirement of integrins at the cell's own rates: `dN/dt = s_i A_i - N / tau_i`.

    Per cell, `integrin_s` (births per unit basal area per unit time) and `integrin_tau` (mean
    lifetime, in the run's time unit) are read off the cell set through the containment map.
    Births wake dormant slots and place them on the cell's basal fan; retirements kill live
    slots with probability dt / tau. Fractional births carry over between frames so a slow rate
    is honoured on average. The count is written back as `n_integrin`."""
    READS = ["pos"]
    WRITES = ["pos"]
    SUPPORTED_DIMS = (3,)                                 # a basal surface is a surface in space
    REQUIRES_PARAMS = ["tissue"]
    PARAM_ROLES = {"tissue": "tissue_set"}
    MECHANISM_TAGS = ["synthesis", "turnover", "receptor_number"]
    REFERENCE = ("The receptor rate law of discovery_okuda/ops/adhesion_ops.py (rung 05d, gate G30: "
                 "receptor conserved under binding).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "integrin")
        self.tissue = str(params["tissue"])
        self.seed = int(params.get("seed", 0))
        self.s_default = float(params.get("s", 0.0))         # used when the cell set carries no integrin_s
        self.tau_default = float(params.get("tau", float("inf")))
        self._carry = None
        self._gen = None

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        fans = _basal_fans(H, self.tissue)
        if fans is None:
            return {}
        dev = lvl.state.device
        if self._gen is None:
            self._gen = torch.Generator(device=dev).manual_seed(self.seed)
        dt = float(getattr(H, "dt", 1.0) or 1.0)
        cs = resolve_cell_set(H, self.tissue, None)
        clvl = H.level(cs)
        nF = fans["nF"]
        n_cells = clvl.state.shape[0]
        s = clvl.get("integrin_s")[:, 0] if "integrin_s" in clvl.state_schema else torch.full((n_cells,), self.s_default, device=dev)
        tau = clvl.get("integrin_tau")[:, 0] if "integrin_tau" in clvl.state_schema else torch.full((n_cells,), self.tau_default, device=dev)
        par = lvl.parent
        occ = lvl.occ > 0
        # ---- retirement: each live particle with probability dt / tau of its parent ----------
        p_die = (dt / tau.clamp_min(1e-9))[par].clamp(0.0, 1.0)
        u = torch.rand(par.shape[0], generator=self._gen, device=dev)
        die = occ & (u < p_die)
        n_died = int(die.sum().item())
        if n_died:
            lvl.kill(torch.nonzero(die).flatten(), park=torch.full((3,), PARK, device=dev, dtype=lvl.state.dtype))
            occ = lvl.occ > 0
        # ---- birth: s_i * A_i * dt per live cell, with fractional carry-over ------------------
        area = _face_area(fans)
        live_cells = clvl.occ[:nF] > 0 if getattr(clvl, "occ", None) is not None else torch.ones(nF, dtype=torch.bool, device=dev)
        want = (s[:nF] * area * dt) * live_cells.to(area.dtype)
        if self._carry is None or self._carry.shape[0] != n_cells:
            self._carry = torch.zeros(n_cells, device=dev, dtype=area.dtype)
        want = want + self._carry[:nF]
        k = torch.floor(want).long()
        self._carry[:nF] = want - k.to(want.dtype)
        total = int(k.sum().item())
        n_born = 0
        if total:
            free = lvl.free_slots(total)
            n_born = int(free.numel())
            if n_born:
                parents = torch.repeat_interleave(torch.arange(nF, device=dev), k)[:n_born]
                par[free] = parents
                lvl.occ[free] = 1.0
                p0, p1 = lvl.state_schema["pos"]
                lvl.state[free, p0:p1] = _sample_on_faces(fans, parents, self._gen)
                if "vel" in lvl.state_schema:
                    v0, v1 = lvl.state_schema["vel"]; lvl.state[free, v0:v1] = 0.0
            if n_born < total and not getattr(self, "_said_full", False):
                print(f"[integrin_express] buffer exhausted: {total - n_born} births refused this frame "
                      f"(declare a larger grow_reserve)", flush=True)
                self._said_full = True
        _write_count(H, lvl, self.tissue)
        return {}
