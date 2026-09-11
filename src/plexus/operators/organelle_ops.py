"""Organelles as one contained set of pieces with a radius, the species as its types.

THE OBJECT. A piece is a POINT with a position, a radius and a parent cell: one nucleus, one
mitochondrion, one Golgi. It is the twin of a protein cluster (`protein_ops.py`) with two
differences a biologist would insist on: its number is stated PER CELL (`count`), not per unit
area, and it has a size that must fit inside the cell. One set holds every species as a
`types:` entry, and the engine tiles the counts per cell and assigns `node_type` at build:

    organelle:
      entity: organelle
      parent: cell
      parent_pos: centroid
      per_parent: 31                       # the counts below add up to it
      types:
        nucleus:      {count: 1,  radius: 0.30, region: basal_side, on_divide: duplicate}
        mitochondria: {count: 30, radius: 0.06, region: interior,   on_divide: halve}

`region:` is where inside the cell's prism the piece lives: `interior` (anywhere between the
two caps), `apical_side` / `basal_side` (the half nearer that cap). The piece's radius is a
MARGIN on the region: its centre stays at least one radius from both caps and from the cell's
side walls, so the drawn sphere is inside the cell, not through it. `on_divide` is the rule at a
cell division: `duplicate` (each daughter has one afterwards: nucleus, centrosome), `halve` (the
mother's pieces are shared, nearest half to each daughter), `none` (the daughter that holds a
piece keeps it).

What a piece is MADE OF (nothing, a mesh, or MPM particles) is a child set of this one and is
not this file's business; see notes/organelles/ORGANELLE_PLAN.md section 2a. Here the piece is
prescribed: it follows its cell and is kept in its region.

THREE OPERATORS:
  organelle_seed     where they start: every live piece, at a uniform point of its species'
                     region of its cell, at least two radii from any other piece of the cell
                     (a few attempts, then the last draw); writes `n_organelle` on the cell.
  organelle_project  how they stay: every frame each piece is carried by its cell's centroid
                     displacement and then put back to the nearest point of its region, radius
                     margin included. A division (detected as `protein_project` does: the
                     newborn face and the reset-age mother nearest to it) assigns each of the
                     mother's pieces to the daughter whose centroid is nearer, then applies
                     `on_divide`; orphans (a parent that died) are retired.

  organelle_express  how many there are: per cell and species with a `tau`, the number relaxes
                     to the declared `count` with that time constant, `dN/dt = (count - N) / tau`
                     (biogenesis after a division halved the mitochondria, retirement after an
                     excess), with a fractional carry per cell so slow rates still act. A species
                     without `tau` keeps whatever number division left it.

Dormant slots are parked far off-domain (`PARK`), as protein clusters are.

Reference: piece counts per cell after the cell atlas (`figures/cell_atlas.png`); nuclear
positioning in columnar epithelia, Norden (2017) J. Cell Sci. 130:1861 (interkinetic nuclear
migration, the O5 refinement of `on_divide: duplicate`).
"""
from __future__ import annotations

import torch

from plexus.models.base import Seed, Structural
from plexus.models.registry import register_operator
from plexus.operators.protein_ops import PARK, _closest_on_triangles, _fans, _set_block, _tri_of
from plexus.operators.vertex_ops import resolve_cell_set

REGIONS = ("interior", "apical_side", "basal_side")
DIVIDE_RULES = ("duplicate", "halve", "none")
_DEPTH = {"interior": (-1.0, 1.0), "apical_side": (0.0, 1.0), "basal_side": (-1.0, 0.0)}


# ---------------------------------------------------------------------------------------------
def _species(lvl, params):
    """The species table in `types:` order: count per cell, radius, region, division rule."""
    names = list(getattr(lvl, "type_names", []) or [])
    if not names:
        raise ValueError(f"organelle set {getattr(lvl, 'name', '?')!r} declares no `types:`; the species are its types")
    table = list(getattr(lvl, "_type_table", []) or [{} for _ in names])
    out = []
    for i, n in enumerate(names):
        t = dict(table[i]) if i < len(table) else {}
        region = str(t.get("region", params.get("region", "interior"))).lower()
        if region not in REGIONS:
            raise ValueError(f"organelle species {n!r}: region must be one of {REGIONS}, not {region!r}")
        rule = str(t.get("on_divide", params.get("on_divide", "none"))).lower()
        if rule not in DIVIDE_RULES:
            raise ValueError(f"organelle species {n!r}: on_divide must be one of {DIVIDE_RULES}, not {rule!r}")
        out.append(dict(id=i, name=n, region=region, rule=rule, count=int(t.get("count", 0)),
                        radius=float(t.get("radius", params.get("radius", 0.0))),
                        tau=float(t.get("tau", params.get("tau", float("inf"))))))
    return out


def _mid(fans):
    return fans["A"], fans["B"], fans["C"]


def _in_region(fans, faces, region, radius, x):
    """The nearest point of the region to `x`, one radius inside it.

    In the plane of the cell: the closest point on the mid fan is written in the fan triangle's
    barycentric weights (w0 on the centroid, w1 and w2 on the ring); `1 - w0` is how far along
    the spoke from centroid to ring edge the point sits, so the wall margin is a cap on it at
    `1 - radius / spoke`, with the spoke length measured on that triangle. Along the axis: the
    depth is the signed fraction of the local separation vector (+1 apical cap, -1 basal), and
    the caps' margin shrinks the species' depth window by `radius / |sep|` on each side. A piece
    larger than the cell's half-thickness sits on the mid-surface and is reported once."""
    A, B, C = _mid(fans)
    tri, valid = _tri_of(fans, faces)
    q, wt, best = _closest_on_triangles(x, A[tri], B[tri], C[tri], valid)
    ar = torch.arange(x.shape[0], device=x.device)
    t = tri[ar, best]
    a, b, c = A[t], B[t], C[t]
    # the wall margin, along the spoke from the centroid through q to the ring edge
    edge = 0.5 * (b + c)
    spoke = (edge - a).norm(dim=1).clamp_min(1e-9)
    frac = (1.0 - wt[:, 0]).clamp(0.0, 1.0)                      # 0 at the centroid, 1 on the ring
    cap = (1.0 - radius / spoke).clamp(0.0, 1.0)
    scale = torch.where(frac > cap, cap / frac.clamp_min(1e-9), torch.ones_like(frac))
    q = a + (q - a) * scale[:, None]
    # the depth window, shrunk by the radius on both caps
    sep = wt[:, :1] * fans["SA"][t] + wt[:, 1:2] * fans["SB"][t] + wt[:, 2:3] * fans["SC"][t]
    n2 = (sep * sep).sum(1, keepdim=True).clamp_min(1e-20)
    m = (radius / n2.sqrt()).clamp(0.0, 1.0)
    lo, hi = _DEPTH[region]
    lo_t = torch.full_like(m, lo) + m
    hi_t = torch.full_like(m, hi) - m
    mid_t = 0.5 * (lo_t + hi_t)
    lo_t, hi_t = torch.minimum(lo_t, mid_t), torch.maximum(hi_t, mid_t)
    depth = ((x - q) * sep).sum(1, keepdim=True) / n2
    depth = torch.maximum(torch.minimum(depth, hi_t), lo_t)
    return q + depth * sep


def _sample(fans, faces, region, radius, gen):
    """One uniform point per requested face in the species' region, then pulled one radius
    inside it by `_in_region` (a point already inside is left where it is)."""
    dev = fans["A"].device
    A, B, C = _mid(fans)
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
    lo, hi = _DEPTH[region]
    depth = (lo + (hi - lo) * torch.rand(faces.shape[0], generator=gen, device=dev))[:, None]
    x = x + depth * (w0 * fans["SA"][t] + w1 * fans["SB"][t] + w2 * fans["SC"][t])
    return _in_region(fans, faces, region, radius, x)


def _spread(fans, faces, region, radius, x, others, tries, gen):
    """Redraw the pieces that sit closer than the sum of the radii to another piece of the SAME
    cell, a few times; the ones still overlapping keep their last draw (a crowded cell is
    reported by the seed, not refused). `others` is (positions, faces, radii) of the pieces
    already placed. Pairs are tested per cell on a padded [cells, k, k] block, never over the
    whole set, so a 2,048-cell tissue costs 2 M distances and not 4 G."""
    n = x.shape[0]
    if n == 0:
        return x
    dev = x.device
    nF = fans["nF"]
    for _ in range(int(tries)):
        P = torch.cat([x, others[0]], 0); F = torch.cat([faces, others[1]], 0)
        R = torch.cat([torch.full((n,), radius, device=dev, dtype=x.dtype), others[2]], 0)
        cnt = torch.bincount(F, minlength=nF)
        k = int(cnt.max().item())
        order = torch.argsort(F, stable=True)
        start = torch.cumsum(cnt, 0) - cnt
        slot = torch.arange(F.shape[0], device=dev) - start[F[order]]
        Pp = torch.full((nF, k, 3), 0.0, device=dev, dtype=x.dtype)
        Rp = torch.full((nF, k), -1e9, device=dev, dtype=x.dtype)       # a padded slot never overlaps
        Pp[F[order], slot] = P[order]; Rp[F[order], slot] = R[order]
        D = (Pp[:, :, None, :] - Pp[:, None, :, :]).norm(dim=-1)
        lim = Rp[:, :, None] + Rp[:, None, :]
        eye = torch.eye(k, device=dev, dtype=torch.bool)[None]
        bad_pad = ((D < lim) & ~eye).any(-1)                          # [nF, k]
        bad = torch.zeros(F.shape[0], dtype=torch.bool, device=dev)
        bad[order] = bad_pad[F[order], slot]
        bad = bad[:n]
        if not bool(bad.any()):
            break
        idx = torch.nonzero(bad).flatten()
        x[idx] = _sample(fans, faces[idx], region, radius, gen)
    return x


def _write_count(H, lvl, tissue_set, species):
    cs = resolve_cell_set(H, tissue_set, None)
    try:
        clvl = H.level(cs)
    except Exception:                                        # noqa: BLE001
        return
    if "n_organelle" not in clvl.state_schema:
        return
    c0, c1 = clvl.state_schema["n_organelle"]
    if c1 - c0 != len(species):
        raise ValueError(f"cell block 'n_organelle' has width {c1 - c0}; the organelle set declares {len(species)} species")
    live = lvl.occ > 0
    for sp in species:
        sel = live & (lvl.node_type == sp["id"])
        n = torch.bincount(lvl.parent[sel], minlength=clvl.state.shape[0]).to(clvl.state.dtype)
        clvl.state[:, c0 + sp["id"]] = n[: clvl.state.shape[0]]


# ---------------------------------------------------------------------------------------------
@register_operator("organelle_seed", family="seed", set="particle", kind="seed")
class OrganelleSeed(Seed):
    """Place every live piece in its species' region of its cell, once, spread apart."""
    REQUIRES_PARAMS = ["tissue"]
    PARAM_ROLES = {"tissue": "tissue_set"}
    OPTIONAL_TYPE_PROPS = ["count", "radius", "region", "on_divide", "body"]
    SUPPORTED_DIMS = (3,)
    READS = ["pos"]
    WRITES = ["pos"]
    MECHANISM_TAGS = ["seed", "organelle_placement"]
    REFERENCE = "Plexus (this work); counts per cell after the cell atlas."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "organelle")
        self.tissue = str(params["tissue"])
        self.seed = int(params.get("seed", 0))
        self.tries = int(params.get("spread_tries", 8))
        self.params = dict(params)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        fans = _fans(H, self.tissue)
        if fans is None:
            raise ValueError(f"organelle_seed: {self.tissue!r} carries no mesh at seed time; put seed_mesh before it")
        if lvl.parent is None:
            raise ValueError(f"organelle_seed: set {self.at!r} declares no parent; it needs `parent: cell`")
        species = _species(lvl, self.params)
        dev = lvl.state.device
        gen = torch.Generator(device=dev).manual_seed(self.seed)
        nF = fans["nF"]
        p0, p1 = lvl.state_schema["pos"]
        live_cells = torch.zeros(lvl.parent.max().item() + 1, dtype=torch.bool, device=dev)
        live_cells[:nF] = True
        # a piece whose cell has no face (a dormant cell slot) is dormant too
        occ = (lvl.occ > 0) & live_cells[lvl.parent]
        lvl.occ[:] = 0.0
        lvl.occ[occ] = 1.0
        lvl.state[~occ, p0:p1] = PARK
        placed = (torch.zeros(0, 3, device=dev, dtype=lvl.state.dtype), torch.zeros(0, dtype=torch.long, device=dev),
                  torch.zeros(0, device=dev, dtype=lvl.state.dtype))
        for sp in sorted(species, key=lambda s: -s["radius"]):     # the big ones first, the small ones fit around them
            idx = torch.nonzero(occ & (lvl.node_type == sp["id"])).flatten()
            if idx.numel() == 0:
                continue
            faces = lvl.parent[idx]
            x = _sample(fans, faces, sp["region"], sp["radius"], gen)
            x = _spread(fans, faces, sp["region"], sp["radius"], x, placed, self.tries, gen)
            lvl.state[idx, p0:p1] = x
            _set_block(lvl, "vel", idx, 0.0)
            placed = (torch.cat([placed[0], x], 0), torch.cat([placed[1], faces], 0),
                      torch.cat([placed[2], torch.full((idx.numel(),), sp["radius"], device=dev, dtype=lvl.state.dtype)], 0))
            half = fans["csep"].norm(dim=1)
            tight = int((half[: nF] < sp["radius"]).sum().item())
            print(f"[organelle_seed] {int(idx.numel())} {sp['name']} pieces in {nF} cells "
                  f"({sp['count']} per cell, radius {sp['radius']:g}, region {sp['region']})"
                  + (f"; {tight} cells thinner than the radius, piece on the mid-surface" if tight else ""), flush=True)
        _write_count(H, lvl, self.tissue, species)
        return {}


# ---------------------------------------------------------------------------------------------
@register_operator("organelle_project", family="mechanics", set="particle", kind="structural")
class OrganelleProject(Structural):
    """Carry every piece with its cell and keep it in its region; apply `on_divide` at a division."""
    REQUIRES_PARAMS = ["tissue"]
    PARAM_ROLES = {"tissue": "tissue_set"}
    OPTIONAL_TYPE_PROPS = ["count", "radius", "region", "on_divide", "body"]
    SUPPORTED_DIMS = (3,)
    READS = ["pos"]
    WRITES = ["pos"]
    MECHANISM_TAGS = ["region_constraint", "inheritance_at_division"]
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "organelle")
        self.tissue = str(params["tissue"])
        self.seed = int(params.get("seed", 0))
        self.params = dict(params)
        self._prev_live = None
        self._prev_age = None
        self._prev_cen = None
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
        cs = resolve_cell_set(H, self.tissue, None); clvl = H.level(cs)
        nF = fans["nF"]
        live_cells = clvl.occ[:nF] > 0 if getattr(clvl, "occ", None) is not None else torch.ones(nF, dtype=torch.bool, device=dev)
        par = lvl.parent
        p0, p1 = lvl.state_schema["pos"]
        park = torch.full((3,), PARK, device=dev, dtype=lvl.state.dtype)
        # ---- carried by the cell: the centroid's displacement since last frame ---------------
        cen = fans["cen"]
        if self._prev_cen is not None and self._prev_cen.shape[0] <= nF:
            k = self._prev_cen.shape[0]
            shift = torch.zeros(nF, 3, device=dev, dtype=cen.dtype)
            shift[:k] = cen[:k] - self._prev_cen
            occ = lvl.occ > 0
            ok = occ & (par < nF)
            lvl.state[ok, p0:p1] += shift[par[ok]]
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
                    dist = (cen[cand] - cen[d]).norm(dim=1)
                    mother = int(cand[dist.argmin()].item())
                    if mother == d:
                        continue
                    self._divide(lvl, fans, species, mother, d, par, p0, p1, park)
        self._prev_live = live_cells.clone()
        self._prev_age = age.clone() if age is not None else None
        self._prev_cen = cen.clone()
        # ---- orphans retire, the rest are kept in their region ------------------------------
        occ = lvl.occ > 0
        orphan = occ & ((par >= nF) | (~live_cells[par.clamp(max=nF - 1)]))
        if bool(orphan.any()):
            lvl.kill(torch.nonzero(orphan).flatten(), park=park)
            occ = lvl.occ > 0
        for sp in species:
            idx = torch.nonzero(occ & (lvl.node_type == sp["id"])).flatten()
            if idx.numel():
                lvl.state[idx, p0:p1] = _in_region(fans, par[idx], sp["region"], sp["radius"], lvl.state[idx, p0:p1])
                _set_block(lvl, "vel", idx, 0.0)
        _write_count(H, lvl, self.tissue, species)
        return {}

    def _divide(self, lvl, fans, species, mother, d, par, p0, p1, park):
        """The mother's pieces go to the nearer daughter; then each species' rule."""
        occ = lvl.occ > 0
        cen = fans["cen"]
        for sp in species:
            mine = torch.nonzero(occ & (par == mother) & (lvl.node_type == sp["id"])).flatten()
            if mine.numel() == 0:
                continue
            x = lvl.state[mine, p0:p1]
            nearer_d = (x - cen[d]).norm(dim=1) < (x - cen[mother]).norm(dim=1)
            if sp["rule"] == "halve" and mine.numel() >= 2:
                # exactly half, the half nearest the daughter, as protein clusters are shared
                dd = (x - cen[d]).norm(dim=1)
                give = mine[torch.argsort(dd)[: mine.numel() // 2]]
                par[give] = d
            elif sp["rule"] == "duplicate":
                par[mine[nearer_d]] = d
                have_m = int((~nearer_d).sum().item()); have_d = int(nearer_d.sum().item())
                for who, have in ((mother, have_m), (d, have_d)):
                    need = sp["count"] - have
                    if need <= 0:
                        continue
                    src = mine[:1].repeat(need)
                    new, _ = lvl.spawn(src)
                    if new.numel() == 0:
                        print(f"[organelle_project] reserve exhausted: {sp['name']} not duplicated in cell {who}", flush=True)
                        continue
                    par[new] = who
                    lvl.node_type[new] = sp["id"]
                    faces = torch.full((new.numel(),), who, dtype=torch.long, device=x.device)
                    lvl.state[new, p0:p1] = _sample(fans, faces, sp["region"], sp["radius"], self._gen)
                    _set_block(lvl, "vel", new, 0.0)
                    _set_block(lvl, "age", new, 0.0)
            else:                                               # none: the daughter holding it keeps it
                par[mine[nearer_d]] = d


# ---------------------------------------------------------------------------------------------
@register_operator("organelle_express", family="population", set="particle", kind="structural")
class OrganelleExpress(Structural):
    """Per cell and species, relax the number of pieces to the declared `count`: `dN/dt = (count - N) / tau`."""
    REQUIRES_PARAMS = ["tissue"]
    PARAM_ROLES = {"tissue": "tissue_set"}
    OPTIONAL_TYPE_PROPS = ["count", "radius", "region", "on_divide", "body", "tau"]
    SUPPORTED_DIMS = (3,)
    READS = ["pos"]
    WRITES = ["pos"]
    MECHANISM_TAGS = ["biogenesis", "turnover", "organelle_number"]
    REFERENCE = ("Mitochondrial mass restored over the cell cycle after its halving at division: "
                 "Posakony et al. (1977) J. Cell Biol. 74:468; the set-point law is this work's.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "organelle")
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
        species = [sp for sp in _species(lvl, self.params) if sp["tau"] != float("inf")]
        if not species:
            return {}
        dev = lvl.state.device
        if self._gen is None:
            self._gen = torch.Generator(device=dev).manual_seed(self.seed)
        dt = float(getattr(H, "dt", 1.0) or 1.0)
        cs = resolve_cell_set(H, self.tissue, None); clvl = H.level(cs)
        nF = fans["nF"]
        n_sp = len(getattr(lvl, "type_names", []) or [])
        if self._carry is None or self._carry.shape != (clvl.state.shape[0], n_sp):
            self._carry = torch.zeros(clvl.state.shape[0], n_sp, device=dev, dtype=lvl.state.dtype)
        live_cells = clvl.occ[:nF] > 0 if getattr(clvl, "occ", None) is not None else torch.ones(nF, dtype=torch.bool, device=dev)
        par = lvl.parent
        p0, p1 = lvl.state_schema["pos"]
        park = torch.full((3,), PARK, device=dev, dtype=lvl.state.dtype)
        if "age" in lvl.state_schema:
            a, b = lvl.state_schema["age"]; lvl.state[lvl.occ > 0, a:b] += dt
        for sp in species:
            i = sp["id"]
            occ = lvl.occ > 0
            mine = occ & (lvl.node_type == i)
            N = torch.bincount(par[mine], minlength=clvl.state.shape[0])[:nF].to(lvl.state.dtype)
            want = (sp["count"] - N) * (dt / sp["tau"]) * live_cells.to(lvl.state.dtype) + self._carry[:nF, i]
            k = torch.where(want >= 0, torch.floor(want), -torch.floor(-want)).long()     # toward zero
            self._carry[:nF, i] = want - k.to(want.dtype)
            # births, at a uniform point of the species' region of the cell
            born = k.clamp_min(0)
            total = int(born.sum().item())
            if total:
                free = lvl.free_slots(total)
                n_born = int(free.numel())
                if n_born < total:
                    print(f"[organelle_express] reserve exhausted: {total - n_born} {sp['name']} not born", flush=True)
                if n_born:
                    parents = torch.repeat_interleave(torch.arange(nF, device=dev), born)[:n_born]
                    par[free] = parents
                    lvl.occ[free] = 1.0
                    lvl.node_type[free] = i
                    lvl.state[free, p0:p1] = _sample(fans, parents, sp["region"], sp["radius"], self._gen)
                    _set_block(lvl, "vel", free, 0.0)
                    _set_block(lvl, "age", free, 0.0)
            # retirements, the oldest first, where the cell holds more than its count
            over = (-k).clamp_min(0)
            if int(over.sum().item()):
                cells = torch.nonzero(over > 0).flatten()
                age = lvl.get("age")[:, 0] if "age" in lvl.state_schema else torch.zeros(par.shape[0], device=dev)
                kill = []
                for c in cells.tolist():
                    rows = torch.nonzero(mine & (par == c)).flatten()
                    if rows.numel():
                        kill.append(rows[torch.argsort(age[rows], descending=True)[: int(over[c].item())]])
                if kill:
                    lvl.kill(torch.cat(kill), park=park)
        _write_count(H, lvl, self.tissue, _species(lvl, self.params))
        return {}
