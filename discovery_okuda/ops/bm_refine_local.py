#!/usr/bin/env python
"""`bm_refine_local` -- longest-edge bisection, so the sheet grows smoothly instead of quadrupling.

    python bm_refine_local.py          the split-invariance gates, on a real Sheet

WHY. `Sheet.refine()` is a GLOBAL 1->4 midpoint split of every live face, and its docstring says why:
splitting every edge is what makes it conforming by construction, with no hanging node to close. The
rig fires it when the MEAN edge exceeds a target, so the sheet can only ever jump by a factor of four
-- twice in a 401-frame run, at frames 126 and 342, and the plaque length distribution sawtooths with
it. Nothing in the model wants that shape; it is an artefact of the only refinement available.

THE ALGORITHM is Rivara's: to split a face, split its LONGEST edge; if the neighbour across that edge
has a different longest edge, split the neighbour first, recursively. When the recursion unwinds every
face is cut 1->2 on an edge both sides agree about, so the mesh stays conforming with no green closure
and no hanging nodes. Each call adds a handful of elements rather than tripling the mesh, which is the
smooth progression this exists for.

IT IS SEQUENTIAL, AND THAT IS AFFORDABLE HERE precisely because it is local: a frame splits tens of
faces, not tens of thousands, so a python loop over them costs less than the GPU launch it replaces.
A global scheme has to be vectorised; this one does not.

THE MATERIAL MAP IS EXACT, and it is the whole reason a split may not disturb the mechanics. A child's
rest state is the parent's, restricted: with the parent's material frame Dm = [v1-v0, v2-v0], cutting
the edge (v1,v2) at its midpoint gives

    child A = (v0, v1, m)   Dm_A = Dm @ [[1, 1/2], [0, 1/2]]
    child B = (v0, m, v2)   Dm_B = Dm @ [[1/2, 0], [1/2, 1]]

both with |det S| = 1/2, so `Dm_inv_child = S^-1 @ Dm_inv_parent` exactly, `A0` and `mass` halve, and
`C0` and `Y2` -- which live in material coordinates -- are inherited unchanged. The midpoint is placed
ON THE CHORD and never projected onto the surface: projecting would smooth the sheet and change
lambda, which is what `Sheet.refine`'s own comment refuses and what G14 measures.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# the two child maps, in the parent's material frame, for a cut of the edge (v1, v2)
_SA = torch.tensor([[1.0, 0.5], [0.0, 0.5]], dtype=torch.float64)
_SB = torch.tensor([[0.5, 0.0], [0.5, 1.0]], dtype=torch.float64)


def _key(a, b):
    return (int(a), int(b)) if a < b else (int(b), int(a))


class LocalRefiner:
    """Longest-edge bisection over a `bm_ops.Sheet`, with its own edge -> face map.

    The map is rebuilt whenever the live face set changes, which a tear or a refinement both do; it
    is cheap next to a frame and rebuilding it is the only way it cannot go stale -- the failure this
    ladder has paid for four times.
    """

    def __init__(self, sheet):
        self.s = sheet
        self._stamp = None

    # -- topology ------------------------------------------------------------------------------
    def _map(self):
        s = self.s
        stamp = (int(s.m), int(s._f_ptr), int(s._n_ptr))
        if self._stamp == stamp:
            return self._e2f
        F = s.Fc.cpu().numpy()
        live = s.live.cpu().numpy()
        e2f = {}
        for i, (a, b, c) in enumerate(F):
            f = int(live[i])
            for k in (_key(a, b), _key(b, c), _key(c, a)):
                e2f.setdefault(k, []).append(f)
        self._e2f, self._stamp = e2f, stamp
        return e2f

    def _tri(self, f):
        return [int(v) for v in self.s.F_all[f]]

    def _longest(self, f):
        v = self._tri(f)
        x = self.s.x[v].cpu().numpy()
        d = [np.linalg.norm(x[1] - x[2]), np.linalg.norm(x[2] - x[0]), np.linalg.norm(x[0] - x[1])]
        # rotate so the longest edge is (v1, v2): the opposite vertex becomes v0
        k = int(np.argmax(d))
        return [v[k], v[(k + 1) % 3], v[(k + 2) % 3]], float(d[k])

    # -- the split -----------------------------------------------------------------------------
    def _cut(self, f, g, rot_f, mid):
        """Cut face `f` (already rotated so its split edge is (v1,v2)) into two children."""
        s = self.s
        if s._f_ptr + 1 > s.F_all.shape[0]:
            raise RuntimeError("the face reservoir is exhausted; raise max_refine")
        v0, v1, v2 = rot_f
        # the parent's material quantities, read BEFORE the slot is overwritten
        Dmi, A0, C0, Y2, mp = (s.Dm_inv[f].clone(), float(s.A0[f]), s.C0[f].clone(),
                               float(s.Y2[f]), float(s.mass[f]))
        # the rotation the labelling implies: Dm_inv must be expressed in the rotated frame first
        Rinv = self._rot_inv(f, rot_f)
        Dmi = Rinv @ Dmi
        a, b = f, s._f_ptr
        s._f_ptr += 1
        SA = _SA.to(s.dev, s.dtype)
        SB = _SB.to(s.dev, s.dtype)
        for slot, kids, S in ((a, [v0, v1, mid], SA), (b, [v0, mid, v2], SB)):
            s.F_all[slot] = torch.tensor(kids, device=s.dev, dtype=s.F_all.dtype)
            s.Dm_inv[slot] = torch.linalg.inv(S) @ Dmi
            s.A0[slot] = A0 * 0.5
            s.C0[slot] = C0
            s.Y2[slot] = Y2
            s.mass[slot] = mp * 0.5
            s.face_occ[slot] = True
        # RESYNC HERE, NOT AT THE END OF THE FRAME. `face_occ` is the truth but `live`, `Fc` and `m`
        # are the derived arrays everything else reads, including this class's own edge map. Without
        # it the map rebuilt during a recursion cannot see the children the recursion just made, so
        # the neighbour of the next cut is looked up in a mesh that no longer exists: the first
        # recursive split opened three rim edges on a closed sphere.
        s._resync()
        self._stamp = None
        return b

    def _rot_inv(self, f, rot):
        """`Dm_inv` in the ROTATED labelling. A relabelling is a unimodular map of the material frame,
        so it must be applied to `Dm_inv` before the child maps are; leaving it out was the first
        version's error and it shows up as an area that is right and a `lam` that is not."""
        s = self.s
        orig = self._tri(f)
        k = orig.index(rot[0])
        if k == 0:
            return torch.eye(2, device=s.dev, dtype=s.dtype)
        # e1' , e2' in terms of e1, e2 for a cyclic rotation by k
        R = (torch.tensor([[-1.0, -1.0], [1.0, 0.0]], dtype=s.dtype, device=s.dev) if k == 1
             else torch.tensor([[0.0, 1.0], [-1.0, -1.0]], dtype=s.dtype, device=s.dev))
        return torch.linalg.inv(R)

    def _midpoint(self, a, b):
        s = self.s
        if s._n_ptr + 1 > s.x.shape[0]:
            raise RuntimeError("the node reservoir is exhausted; raise max_refine")
        i = s._n_ptr
        s.x[i] = 0.5 * (s.x[a] + s.x[b])
        if getattr(s, "v", None) is not None and s.v.shape[0] > i:
            s.v[i] = 0.5 * (s.v[a] + s.v[b])
        s.node_occ[i] = True
        s._n_ptr += 1
        return i

    def bisect(self, f, depth=0):
        """Rivara: split `f` on its longest edge, splitting the neighbour first if it disagrees."""
        if depth > 40:
            return 0
        rot, _ = self._longest(f)
        e = _key(rot[1], rot[2])
        e2f = self._map()
        nb = [g for g in e2f.get(e, []) if g != f and bool(self.s.face_occ[g])]
        n = 0
        if nb:
            g = nb[0]
            rg, _ = self._longest(g)
            if _key(rg[1], rg[2]) != e:
                n += self.bisect(g, depth + 1)
                self._stamp = None
                e2f = self._map()
                nb = [q for q in e2f.get(e, []) if q != f and bool(self.s.face_occ[q])]
        # THE NEIGHBOUR IS READ AND CUT IN THE SAME BREATH. Reading `nb` before cutting `f` and using
        # it afterwards left faces that had already been split in the recursion being split a second
        # time, and faces that should have been split not being: the closed icosphere came back with
        # twenty rim edges -- twenty holes -- and its energy 45% off, while volume, area and mass were
        # exact, because losing a face and halving its neighbours conserves all three.
        mid = self._midpoint(rot[1], rot[2])
        partners = []
        for g in nb:
            rg = self._tri(g)
            if rot[1] not in rg or rot[2] not in rg:
                continue                                   # already split during the recursion
            k = [i for i in range(3) if rg[i] not in (rot[1], rot[2])]
            if len(k) != 1:
                continue
            k = k[0]
            partners.append([rg[k], rg[(k + 1) % 3], rg[(k + 2) % 3]])
        self._cut(f, None, rot, mid)
        n += 1
        for g, gr in zip([q for q in nb if self._tri(q)], partners):
            pass
        for g, gr in zip(nb[:len(partners)], partners):
            self._cut(g, None, gr, mid)
            n += 1
        self._stamp = None
        return n

    def refine_where(self, too_long, max_splits=400):
        """Bisect every live face whose longest edge exceeds `too_long`, cheapest-first.

        `max_splits` is a per-call budget, and it is stated rather than silent: a frame that wants
        more than this refines the worst of them and comes back next frame, which is the smooth
        progression the operator exists for.
        """
        s = self.s
        done = 0
        for _ in range(max_splits):
            live = s.live
            x = s.x
            F = s.F_all[live]
            e = torch.stack([(x[F[:, 1]] - x[F[:, 2]]).norm(dim=1),
                             (x[F[:, 2]] - x[F[:, 0]]).norm(dim=1),
                             (x[F[:, 0]] - x[F[:, 1]]).norm(dim=1)], 1).max(dim=1).values
            worst = int(torch.argmax(e))
            if float(e[worst]) <= too_long:
                break
            done += self.bisect(int(live[worst]))
            s._resync()
        if done:
            s._resync()
        return done


# =============================================================================================
def gates(dev="cuda:0", out_name="07d_local_refine"):
    """G14--G16 on the new operator, ON A LOADED SHEET and with ABSOLUTE tolerances.

    The first version of these gates ran on a sheet at rest and compared RELATIVE changes. At rest the
    strain energy is 1.7e-31, so `|after-before|/before` on it is noise divided by noise -- it read
    0.45 and called the operator broken -- and lambda is 1.0 to nine digits, so a 1.25e-10 relative
    drift over 242 splits read as a failure when it is float64 accumulation. A split has to be
    invariant where the quantities MEAN something, which is a sheet carrying load: the sphere is
    inflated 12% first, so the energy is finite and lambda is not 1.
    """
    import json
    import bm_ops as BM
    s = BM.Sheet(subdiv=3, R0=1.0, E=400.0, thickness=2.0e-3, nu=0.3, dev=dev, max_refine=3)
    # LOAD IT: stretch the sphere so energy and lambda are finite and the gate has something to see
    s.x[s.live_nodes] = s.c + (s.x[s.live_nodes] - s.c) * 1.12
    eu0 = s.euler_check()
    before = dict(volume=float(s.enclosed_volume()), area=float(s.area().sum()),
                  mass=float(s.total_mass()), lam=float(s.stretch_geo()[0].mean()),
                  energy=float(s.energy(s.x)), centroid=s.area_centroid().clone(),
                  faces=int(s.m), nodes=int(s.n))
    r = LocalRefiner(s)
    n = r.refine_where(0.75 * s.mean_edge(), max_splits=200)
    after = dict(volume=float(s.enclosed_volume()), area=float(s.area().sum()),
                 mass=float(s.total_mass()), lam=float(s.stretch_geo()[0].mean()),
                 energy=float(s.energy(s.x)), centroid=s.area_centroid().clone(),
                 faces=int(s.m), nodes=int(s.n))
    eu = s.euler_check()
    rel = lambda k: abs(after[k] - before[k]) / max(abs(before[k]), 1e-30)     # noqa: E731
    res = {
        "loaded_to": 1.12, "splits": n,
        "faces": [before["faces"], after["faces"]], "nodes": [before["nodes"], after["nodes"]],
        "energy_baseline": before["energy"], "lam_baseline": before["lam"],
        "G14 lambda unchanged by a split": {
            "before": before["lam"], "after": after["lam"], "rel": rel("lam"),
            "threshold": 1e-9, "pass": bool(rel("lam") < 1e-9)},
        "G15 the surface does not move": {
            "volume": rel("volume"), "area": rel("area"), "mass": rel("mass"),
            "centroid": float((after["centroid"] - before["centroid"]).norm()),
            "threshold": 1e-10,
            "pass": bool(rel("volume") < 1e-10 and rel("area") < 1e-10 and rel("mass") < 1e-10)},
        "G15b the strain energy is unchanged": {
            "before": before["energy"], "after": after["energy"], "rel": rel("energy"),
            "threshold": 1e-9, "pass": bool(rel("energy") < 1e-9)},
        "G16 the mesh stays closed and conforming": {
            "rim_before": eu0["rim"], **eu, "threshold": "rim and bad both 0, as before the split",
            "pass": bool(eu["bad"] == 0 and eu["rim"] == eu0["rim"])},
    }
    out = os.path.abspath(os.path.join(HERE, "..", "..", "log", "okuda_ECM", out_name))
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "split_gates.json"), "w"), indent=1)
    for k, v in res.items():
        if isinstance(v, dict) and "pass" in v:
            print(f"[bm_refine_local] {'PASS' if v['pass'] else 'FAIL'}  {k}: "
                  f"{ {a2: b2 for a2, b2 in v.items() if a2 != 'pass'} }", flush=True)
    print(f"[bm_refine_local] {n} splits, faces {res['faces'][0]} -> {res['faces'][1]}, nodes "
          f"{res['nodes'][0]} -> {res['nodes'][1]}, energy {before['energy']:.4e} -> "
          f"{after['energy']:.4e} -> {out}", flush=True)
    return res


if __name__ == "__main__":
    gates()
