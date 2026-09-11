"""Proteins as one contained set of point clusters, the species as its types.

THE OBJECT. A cluster is a POINT with a position and a parent (a cell today; a half-edge for a
junction species, next). It is not matter: no deformation gradient, no grid, no mass, so no MPM
operator ever touches the set. One set, `protein`, holds every species as a `types:` entry:

    protein:
      parent: cell
      parent_pos: centroid
      per_parent: 20
      grow_reserve: 40
      state: {pos, vel, activated, folded, bound, age}
      types:
        integrin: {region: basal,  density: 3.0, s: 0.02, tau: 300}
        myosin:   {region: apical, density: 3.0, s: 0.02, tau: 300}

so a new species is a new line, a conversion (pro-enzyme to enzyme, free to engaged) is a state
flip and not a second buffer, and the reservoir is sized once for the total. The species'
starting numbers are type properties; the CELL carries the running rates as blocks of one column
per species (`protein_s`, `protein_tau`, in the order of `types:`) so a signalling operator can
vary them per cell later, and the count comes back the same way (`n_protein`).

`region:` names where a species is confined: `apical` and `basal` are the two caps of an
apico-basal cell, `mid` the mid-surface, `interior` the cell's polyhedron between the caps.
`lateral` (the walls) and `junction` (a half-edge-parented species on its edge) are declared
and refused until the half-edge parent lands.

THREE OPERATORS, EACH ANSWERING ONE QUESTION, looping over species inside:
  protein_seed      where they start: on each cell's face of the species' region, uniformly by
                    area at the species' `density`; writes the cell's starting rates (a seed).
  protein_project   how they stay on the tissue: every frame each cluster is moved to the nearest
                    point of its parent's region. The region is the corral: a cluster pushed past
                    the cell's edge by the repulsion between clusters is put back on the boundary,
                    not handed to the neighbour. A division hands the daughter exactly half of the
                    mother's clusters of each species, the half nearest the daughter's centroid;
                    a T1 swap changes no count. Orphans (a parent that died) are retired.
  protein_express   how many there are: per cell and species, birth at `s` per unit area of the
                    region per unit time and retirement with probability dt / `tau` per cluster,
                    the rate law of archive rung 05d (`dN/dt = s_i - N/tau_i`, receptor conserved
                    under binding) with the receptor countable. `age` counts frames since birth.

Dormant slots are parked far off-domain (`PARK`): a neighbour graph over the buffer sees every
slot, and a million dormant clusters left at one point are a million neighbours of each other.

Reference: the receptor rate law and its conservation gate, discovery_okuda/ops/adhesion_ops.py
(rung 05d, G30); integrin cluster spacing ~555 nm, Changede & Sheetz (2017) Dev. Cell 40:1;
medioapical and junctional myosin pools, Munjal et al. (2015) Nature 524:351.
"""
from __future__ import annotations

import numpy as np
import torch

from plexus.models.base import Seed, Structural
from plexus.models.registry import register_operator
from plexus.operators.vertex_ops import resolve_cell_set

PARK = -1.0e6
REGIONS = ("apical", "basal", "mid", "interior", "lateral", "junction")


# ---------------------------------------------------------------------------------------------
# the tissue's caps as fans of triangles: (cap centroid, cap[es], cap[et]) per live half-edge
# ---------------------------------------------------------------------------------------------
def _fans(H, tissue_set):
    """Mid-surface fan corners with each corner's separation vector, so every region's geometry
    is one expression: cap = mid + k * sep with k = +1 (apical), -1 (basal), 0 (mid), and the
    interior is the segment between k = -1 and k = +1. Returns None while the tissue has no mesh."""
    lvl = H.level(tissue_set)
    m = getattr(lvl, "_mesh", None) or getattr(lvl, "mesh", None)
    if m is None or not int(m.get("Nv", 0) or 0):
        return None
    Nv, nF = int(m["Nv"]), int(m["nF"])
    pos = lvl.get("pos")[:Nv]
    sep = lvl.get("sep")[:Nv] if "sep" in lvl.state_schema else torch.zeros_like(pos)
    es, et, ef = m["E_srce"], m["E_trgt"], m["E_face"]
    eocc = m.get("eocc", None)
    live = (ef >= 0) & (ef < nF) & (es >= 0) & (et >= 0)
    if eocc is not None:
        live = live & (eocc[: es.shape[0]] > 0)
    es, et, ef = es[live], et[live], ef[live]
    cnt = torch.bincount(ef, minlength=nF).clamp_min(1).to(pos.dtype)
    z = lambda: torch.zeros(nF, 3, device=pos.device, dtype=pos.dtype)   # noqa: E731
    cen = z().index_add_(0, ef, pos[es]) / cnt[:, None]
    csep = z().index_add_(0, ef, sep[es]) / cnt[:, None]
    return dict(A=cen[ef], B=pos[es], C=pos[et], SA=csep[ef], SB=sep[es], SC=sep[et],
                face=ef, nF=nF, cen=cen, csep=csep)


def _cap(fans, k):
    """The three corners of every fan triangle on cap `k` (+1 apical, -1 basal, 0 mid)."""
    A = fans["A"] + k * fans["SA"]; B = fans["B"] + k * fans["SB"]; C = fans["C"] + k * fans["SC"]
    return A, B, C


_K = {"apical": 1.0, "basal": -1.0, "mid": 0.0, "interior": 0.0}


def _face_area(fans, region):
    A, B, C = _cap(fans, _K[region])
    area = 0.5 * torch.cross(B - A, C - A, dim=1).norm(dim=1)
    return torch.zeros(fans["nF"], device=A.device, dtype=A.dtype).index_add_(0, fans["face"], area)


def _csr(fans):
    dev = fans["A"].device
    nF = fans["nF"]
    order = torch.argsort(fans["face"])
    f_sorted = fans["face"][order]
    starts = torch.searchsorted(f_sorted, torch.arange(nF, device=dev))
    ends = torch.searchsorted(f_sorted, torch.arange(nF, device=dev), right=True)
    return order, starts, ends - starts


def _tri_of(fans, faces):
    """[n, maxdeg] triangle ids of each requested face's fan, and the validity mask."""
    order, starts, deg = _csr(fans)
    dev = fans["A"].device
    maxdeg = int(deg.max().item()) if deg.numel() else 0
    idx = starts[faces][:, None] + torch.arange(maxdeg, device=dev)[None, :]
    valid = torch.arange(maxdeg, device=dev)[None, :] < deg[faces][:, None]
    return order[torch.where(valid, idx, torch.zeros_like(idx))], valid


def _sample(fans, faces, region, gen):
    """One uniform point per requested face, on the region's fan (triangle by area); the interior
    picks a uniform depth between the two caps as well."""
    dev = fans["A"].device
    k = _K[region]
    A, B, C = _cap(fans, k)
    area = 0.5 * torch.cross(B - A, C - A, dim=1).norm(dim=1)
    tri, valid = _tri_of(fans, faces)
    w = torch.where(valid, area[tri], torch.zeros_like(area[tri]))
    w = w / w.sum(1, keepdim=True).clamp_min(1e-20)
    u = torch.rand(faces.shape[0], generator=gen, device=dev)
    pick = (torch.cumsum(w, 1) < u[:, None]).sum(1).clamp(max=max(tri.shape[1] - 1, 0))
    t = tri[torch.arange(faces.shape[0], device=dev), pick]
    r1 = torch.rand(faces.shape[0], generator=gen, device=dev)
    r2 = torch.rand(faces.shape[0], generator=gen, device=dev)
    s = torch.sqrt(r1)
    w0, w1, w2 = (1.0 - s)[:, None], (s * (1.0 - r2))[:, None], (s * r2)[:, None]
    x = w0 * A[t] + w1 * B[t] + w2 * C[t]
    if region == "interior":
        depth = (2.0 * torch.rand(faces.shape[0], generator=gen, device=dev) - 1.0)[:, None]
        x = x + depth * (w0 * fans["SA"][t] + w1 * fans["SB"][t] + w2 * fans["SC"][t])
    return x


def _closest_on_triangles(P, A, B, C, valid):
    """Closest point on each row's set of triangles (Ericson, Real-Time Collision Detection 5.1.5),
    vectorised over [n, maxdeg, 3]; the nearest valid triangle wins. Returns (point, weights)."""
    P = P[:, None, :]
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = P - B; d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = P - C; d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    denom = (va + vb + vc)
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    v = vb / denom; w = vc / denom
    q = A + ab * v[..., None] + ac * w[..., None]
    wt = torch.stack([1.0 - v - w, v, w], -1)
    def put(mask, point, weights):
        nonlocal q, wt
        q = torch.where(mask[..., None], point, q); wt = torch.where(mask[..., None], weights, wt)
    one = torch.ones_like(d1); zero = torch.zeros_like(d1)
    put((d1 <= 0) & (d2 <= 0), A, torch.stack([one, zero, zero], -1))
    put((d3 >= 0) & (d4 <= d3), B, torch.stack([zero, one, zero], -1))
    put((d6 >= 0) & (d5 <= d6), C, torch.stack([zero, zero, one], -1))
    e1 = torch.where((d1 - d3) != 0, d1 - d3, one); vv = d1 / e1
    put((vc <= 0) & (d1 >= 0) & (d3 <= 0), A + ab * vv[..., None], torch.stack([1 - vv, vv, zero], -1))
    e2 = torch.where((d2 - d6) != 0, d2 - d6, one); ww = d2 / e2
    put((vb <= 0) & (d2 >= 0) & (d6 <= 0), A + ac * ww[..., None], torch.stack([1 - ww, zero, ww], -1))
    e3 = (d4 - d3) + (d5 - d6); e3 = torch.where(e3 != 0, e3, one); ww2 = (d4 - d3) / e3
    put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), B + (C - B) * ww2[..., None], torch.stack([zero, 1 - ww2, ww2], -1))
    dist = (q - P).norm(dim=-1)
    dist = torch.where(valid, dist, torch.full_like(dist, float("inf")))
    best = dist.argmin(dim=1)
    ar = torch.arange(P.shape[0], device=P.device)
    return q[ar, best], wt[ar, best], best


def _project(fans, faces, region, x):
    """The nearest point of each cluster's own face in its region. The interior projects onto the
    mid fan and keeps the depth along the local separation, clamped to the two caps."""
    k = _K[region]
    A, B, C = _cap(fans, k)
    tri, valid = _tri_of(fans, faces)
    q, wt, best = _closest_on_triangles(x, A[tri], B[tri], C[tri], valid)
    if region != "interior":
        return q
    ar = torch.arange(x.shape[0], device=x.device)
    t = tri[ar, best]
    sep = wt[:, :1] * fans["SA"][t] + wt[:, 1:2] * fans["SB"][t] + wt[:, 2:3] * fans["SC"][t]
    n2 = (sep * sep).sum(1, keepdim=True).clamp_min(1e-20)
    depth = (((x - q) * sep).sum(1, keepdim=True) / n2).clamp(-1.0, 1.0)
    return q + depth * sep


# ---------------------------------------------------------------------------------------------
def _species(lvl, params):
    """The species table: names in the order of `types:`, and their per-type properties."""
    names = list(getattr(lvl, "type_names", []) or [])
    if not names:
        raise ValueError(f"protein set {getattr(lvl, 'name', '?')!r} declares no `types:`; the species are its types")
    table = list(getattr(lvl, "_type_table", []) or [{} for _ in names])
    out = []
    for i, n in enumerate(names):
        t = dict(table[i]) if i < len(table) else {}
        region = str(t.get("region", params.get("region", ""))).lower()
        if region not in REGIONS:
            raise ValueError(f"protein species {n!r}: region must be one of {REGIONS}, not {region!r}")
        if region in ("lateral", "junction"):
            raise NotImplementedError(f"protein species {n!r}: region {region!r} needs the half-edge parent, not yet available")
        out.append(dict(id=i, name=n, region=region, density=float(t.get("density", params.get("density", 0.0))),
                        s=float(t.get("s", params.get("s", 0.0))), tau=float(t.get("tau", params.get("tau", float("inf"))))))
    return out


def _cell_block(clvl, name, n_species):
    if name not in clvl.state_schema:
        return None
    c0, c1 = clvl.state_schema[name]
    if c1 - c0 != n_species:
        raise ValueError(f"cell block {name!r} has width {c1 - c0}; the protein set declares {n_species} species, "
                         f"declare `{name}: {{width: {n_species}}}`")
    return c0


def _write_count(H, lvl, tissue_set, species):
    cs = resolve_cell_set(H, tissue_set, None)
    try:
        clvl = H.level(cs)
    except Exception:                                        # noqa: BLE001
        return
    c0 = _cell_block(clvl, "n_protein", len(species))
    if c0 is None:
        return
    live = lvl.occ > 0
    for sp in species:
        sel = live & (lvl.node_type == sp["id"])
        n = torch.bincount(lvl.parent[sel], minlength=clvl.state.shape[0]).to(clvl.state.dtype)
        clvl.state[:, c0 + sp["id"]] = n[: clvl.state.shape[0]]


def _set_block(lvl, name, idx, value):
    if name in lvl.state_schema:
        a, b = lvl.state_schema[name]
        lvl.state[idx, a:b] = value


# ---------------------------------------------------------------------------------------------
@register_operator("protein_seed", family="seed", set="particle", kind="seed")
class ProteinSeed(Seed):
    """Place every species' starting clusters on its region, once, and write the cell's rates."""
    REQUIRES_PARAMS = ["tissue"]
    PARAM_ROLES = {"tissue": "tissue_set"}
    SUPPORTED_DIMS = (3,)
    READS = ["pos"]
    WRITES = ["pos"]
    MECHANISM_TAGS = ["seed", "protein_placement"]
    REFERENCE = "Plexus (this work); cluster spacing after Changede & Sheetz (2017) Dev. Cell 40:1."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "protein")
        self.tissue = str(params["tissue"])
        self.seed = int(params.get("seed", 0))
        self.params = dict(params)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        fans = _fans(H, self.tissue)
        if fans is None:
            raise ValueError(f"protein_seed: {self.tissue!r} carries no mesh at seed time; put seed_mesh before it")
        if lvl.parent is None:
            raise ValueError(f"protein_seed: set {self.at!r} declares no parent; it needs `parent: cell`")
        species = _species(lvl, self.params)
        dev = lvl.state.device
        gen = torch.Generator(device=dev).manual_seed(self.seed)
        nF = fans["nF"]
        par = lvl.parent
        blk = torch.bincount(par, minlength=int(par.max().item()) + 1)
        offset = torch.cumsum(blk, 0) - blk
        p0, p1 = lvl.state_schema["pos"]
        lvl.occ[:] = 0.0
        lvl.state[:, p0:p1] = PARK
        for name in ("activated", "folded", "bound", "age"):
            if name in lvl.state_schema:
                a, b = lvl.state_schema[name]; lvl.state[:, a:b] = -1.0 if name == "bound" else 0.0
        used = torch.zeros(nF, dtype=torch.long, device=dev)      # slots taken in each face's block so far
        total = 0; refused = 0
        for sp in species:
            want = torch.round(_face_area(fans, sp["region"]) * sp["density"]).long()
            room = (blk[:nF] - used).clamp_min(0)
            take = torch.minimum(want, room)
            refused += int((want - take).sum().item())
            if int(take.sum().item()) == 0:
                continue
            faces = torch.repeat_interleave(torch.arange(nF, device=dev), take)
            within = torch.cat([torch.arange(int(k), device=dev) for k in take.tolist() if k]) if take.sum() else torch.zeros(0, dtype=torch.long, device=dev)
            slots = offset[faces] + used[faces] + within
            used = used + take
            lvl.occ[slots] = 1.0
            lvl.node_type[slots] = sp["id"]
            lvl.state[slots, p0:p1] = _sample(fans, faces, sp["region"], gen)
            _set_block(lvl, "vel", slots, 0.0)
            total += int(slots.numel())
            print(f"[protein_seed] {int(slots.numel())} {sp['name']} clusters on {nF} {sp['region']} faces "
                  f"({sp['density']:g} per unit area)", flush=True)
        if refused:
            print(f"[protein_seed] {refused} clusters refused by the per-cell block; declare a larger per_parent", flush=True)
        # the cell's rates, one column per species, from the species' starting values
        cs = resolve_cell_set(H, self.tissue, None); clvl = H.level(cs)
        for key, field in (("protein_s", "s"), ("protein_tau", "tau")):
            c0 = _cell_block(clvl, key, len(species))
            if c0 is not None:
                for sp in species:
                    clvl.state[:, c0 + sp["id"]] = sp[field]
        _write_count(H, lvl, self.tissue, species)
        return {}


# ---------------------------------------------------------------------------------------------
@register_operator("protein_project", family="mechanics", set="particle", kind="structural")
class ProteinProject(Structural):
    """Keep every cluster in its species' region of its parent cell; share them at division."""
    REQUIRES_PARAMS = ["tissue"]
    PARAM_ROLES = {"tissue": "tissue_set"}
    SUPPORTED_DIMS = (3,)
    READS = ["pos"]
    WRITES = ["pos"]
    MECHANISM_TAGS = ["region_constraint", "inheritance_at_division"]
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "protein")
        self.tissue = str(params["tissue"])
        self.params = dict(params)
        self._prev_live = None
        self._prev_age = None

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        fans = _fans(H, self.tissue)
        if fans is None:
            return {}
        species = _species(lvl, self.params)
        dev = lvl.state.device
        cs = resolve_cell_set(H, self.tissue, None); clvl = H.level(cs)
        nF = fans["nF"]
        live_cells = clvl.occ[:nF] > 0 if getattr(clvl, "occ", None) is not None else torch.ones(nF, dtype=torch.bool, device=dev)
        par = lvl.parent
        occ = lvl.occ > 0
        p0, p1 = lvl.state_schema["pos"]
        # ---- division: the newborn face and its mother (age reset this frame, nearest) ---------
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
                    for sp in species:                       # exactly half of each species
                        mine = torch.nonzero(occ & (par == mother) & (lvl.node_type == sp["id"])).flatten()
                        if mine.numel() < 2:
                            continue
                        dd = (lvl.state[mine, p0:p1] - fans["cen"][d]).norm(dim=1)
                        give = mine[torch.argsort(dd)[: mine.numel() // 2]]
                        par[give] = d
        self._prev_live = live_cells.clone()
        self._prev_age = age.clone() if age is not None else None
        # ---- orphans retire, the rest are projected into their region -----------------------
        orphan = occ & ((par >= nF) | (~live_cells[par.clamp(max=nF - 1)]))
        if bool(orphan.any()):
            lvl.kill(torch.nonzero(orphan).flatten(), park=torch.full((3,), PARK, device=dev, dtype=lvl.state.dtype))
            occ = lvl.occ > 0
        for sp in species:
            idx = torch.nonzero(occ & (lvl.node_type == sp["id"])).flatten()
            if idx.numel():
                lvl.state[idx, p0:p1] = _project(fans, par[idx], sp["region"], lvl.state[idx, p0:p1])
                _set_block(lvl, "vel", idx, 0.0)
        _write_count(H, lvl, self.tissue, species)
        return {}


# ---------------------------------------------------------------------------------------------
@register_operator("protein_express", family="population", set="particle", kind="structural")
class ProteinExpress(Structural):
    """Birth and retirement per species at the cell's rates: `dN/dt = s_i A_i - N / tau_i`."""
    REQUIRES_PARAMS = ["tissue"]
    PARAM_ROLES = {"tissue": "tissue_set"}
    SUPPORTED_DIMS = (3,)
    READS = ["pos"]
    WRITES = ["pos"]
    MECHANISM_TAGS = ["synthesis", "turnover", "cluster_number"]
    REFERENCE = ("The receptor rate law of discovery_okuda/ops/adhesion_ops.py (rung 05d, gate G30: "
                 "receptor conserved under binding).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "protein")
        self.tissue = str(params["tissue"])
        self.seed = int(params.get("seed", 0))
        self.params = dict(params)
        self._carry = None
        self._gen = None

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        fans = _fans(H, self.tissue)
        if fans is None:
            return {}
        species = _species(lvl, self.params)
        dev = lvl.state.device
        if self._gen is None:
            self._gen = torch.Generator(device=dev).manual_seed(self.seed)
        dt = float(getattr(H, "dt", 1.0) or 1.0)
        cs = resolve_cell_set(H, self.tissue, None); clvl = H.level(cs)
        nF = fans["nF"]; n_cells = clvl.state.shape[0]
        cs0 = _cell_block(clvl, "protein_s", len(species)); ct0 = _cell_block(clvl, "protein_tau", len(species))
        par = lvl.parent
        p0, p1 = lvl.state_schema["pos"]
        if self._carry is None or self._carry.shape != (n_cells, len(species)):
            self._carry = torch.zeros(n_cells, len(species), device=dev, dtype=lvl.state.dtype)
        live_cells = clvl.occ[:nF] > 0 if getattr(clvl, "occ", None) is not None else torch.ones(nF, dtype=torch.bool, device=dev)
        if "age" in lvl.state_schema:
            a, b = lvl.state_schema["age"]; lvl.state[lvl.occ > 0, a:b] += dt
        park = torch.full((3,), PARK, device=dev, dtype=lvl.state.dtype)
        for sp in species:
            i = sp["id"]
            s = clvl.state[:, cs0 + i] if cs0 is not None else torch.full((n_cells,), sp["s"], device=dev)
            tau = clvl.state[:, ct0 + i] if ct0 is not None else torch.full((n_cells,), sp["tau"], device=dev)
            occ = lvl.occ > 0
            mine = occ & (lvl.node_type == i)
            # retirement
            p_die = (dt / tau.clamp_min(1e-9))[par].clamp(0.0, 1.0)
            u = torch.rand(par.shape[0], generator=self._gen, device=dev)
            die = mine & (u < p_die)
            if bool(die.any()):
                lvl.kill(torch.nonzero(die).flatten(), park=park)
            # birth, with a fractional carry per cell
            want = (s[:nF] * _face_area(fans, sp["region"]) * dt) * live_cells.to(lvl.state.dtype) + self._carry[:nF, i]
            k = torch.floor(want).long()
            self._carry[:nF, i] = want - k.to(want.dtype)
            total = int(k.sum().item())
            if total:
                free = lvl.free_slots(total)
                n_born = int(free.numel())
                if n_born:
                    parents = torch.repeat_interleave(torch.arange(nF, device=dev), k)[:n_born]
                    par[free] = parents
                    lvl.occ[free] = 1.0
                    lvl.node_type[free] = i
                    lvl.state[free, p0:p1] = _sample(fans, parents, sp["region"], self._gen)
                    _set_block(lvl, "vel", free, 0.0)
                    _set_block(lvl, "age", free, 0.0)
                    _set_block(lvl, "activated", free, 0.0)
                    _set_block(lvl, "bound", free, -1.0)
                if n_born < total and not getattr(self, "_said_full", False):
                    print(f"[protein_express] {sp['name']}: buffer exhausted, {total - n_born} births refused this "
                          f"frame (declare a larger grow_reserve)", flush=True)
                    self._said_full = True
        _write_count(H, lvl, self.tissue, species)
        return {}
