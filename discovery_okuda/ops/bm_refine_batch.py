#!/usr/bin/env python
"""`bm_refine_local`, batched on the GPU -- the same refinement without the python recursion.

    python bm_refine_batch.py            sequential vs batched: precision and duration

WHY. The sequential version is Rivara's recursion, and it is a python loop with a device sync per
split: find the worst face (argmax -> int), read its vertices (.cpu().numpy()), walk a dict, cut,
resync. At a 150-split budget that is ~150 round trips per refinement event, and the run sat at 3% GPU
with the refinement rather than the physics as the cost. A tick reduces how OFTEN that happens; it
does not make one event cheaper.

THE BATCHED FORM IS THE STANDARD PARALLEL LONGEST-EDGE ALGORITHM, and it is conforming for the same
reason the recursion is. Per pass:

    1  every face's LONGEST edge, on the GPU
    2  an edge is AGREED when every face touching it names it as its own longest -- then both sides
       want the same cut and no hanging node can appear
    3  at most one agreed edge per face, so a face is cut once per pass
    4  cut the whole selected set at once: one midpoint per edge, two children per face

Faces whose longest edge is not agreed are simply not cut this pass; the pass that does cut their
neighbour changes which edge is longest, and they are picked up next time. That is the recursion,
unrolled into passes, with no dict and no sync.

THE MATERIAL MAP IS THE SEQUENTIAL ONE, unchanged: `Dm_inv_child = S^-1 R^-1 Dm_inv_parent`, `A0` and
`mass` halve, `C0` and `Y2` are inherited, and the midpoint sits on the chord. If the two versions
disagree on lambda, area, mass or energy, one of them is wrong -- which is what the comparison below
is for.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# child maps for a cut of the edge (v1, v2), in the parent's material frame
_SA_INV = torch.tensor([[1.0, -1.0], [0.0, 2.0]], dtype=torch.float64)      # inv([[1,.5],[0,.5]])
_SB_INV = torch.tensor([[2.0, 0.0], [-1.0, 1.0]], dtype=torch.float64)      # inv([[.5,0],[.5,1]])
# the relabelling that puts the split edge at (v1, v2): rotate the triple by k
_RINV = torch.stack([
    torch.eye(2, dtype=torch.float64),
    torch.linalg.inv(torch.tensor([[-1.0, -1.0], [1.0, 0.0]], dtype=torch.float64)),
    torch.linalg.inv(torch.tensor([[0.0, 1.0], [-1.0, -1.0]], dtype=torch.float64))])


class BatchRefiner:
    def __init__(self, sheet):
        self.s = sheet
        d, t = sheet.dev, sheet.dtype
        self.SA, self.SB, self.R = _SA_INV.to(d, t), _SB_INV.to(d, t), _RINV.to(d, t)

    def _pass(self, too_long):
        """One conforming pass. Returns how many faces were cut."""
        s = self.s
        live = s.live
        F = s.F_all[live]
        x = s.x
        # 1 -- the longest edge of every live face. Edge j is opposite vertex j, i.e. (v_{j+1},v_{j+2})
        e = torch.stack([(x[F[:, 1]] - x[F[:, 2]]).norm(dim=1),
                         (x[F[:, 2]] - x[F[:, 0]]).norm(dim=1),
                         (x[F[:, 0]] - x[F[:, 1]]).norm(dim=1)], 1)
        elen, k = e.max(dim=1)
        want = elen > too_long
        if not bool(want.any()):
            return 0
        # every face's three edges, as sorted vertex pairs, and a global id per distinct edge
        pairs = torch.stack([F[:, [1, 2]], F[:, [2, 0]], F[:, [0, 1]]], 1)          # (m,3,2)
        lo = pairs.min(dim=2).values
        hi = pairs.max(dim=2).values
        key = lo.to(torch.int64) * (int(s.x.shape[0]) + 1) + hi.to(torch.int64)     # (m,3)
        uniq, inv = torch.unique(key.reshape(-1), return_inverse=True)
        inv = inv.reshape(-1, 3)
        my = inv.gather(1, k[:, None]).squeeze(1)                                   # my longest edge
        # 2 -- AGREED: an edge every one of its faces names as longest. `n_touch` counts the faces on
        # an edge and `n_want` those naming it; equal means unanimous.
        n_touch = torch.zeros(uniq.numel(), device=s.dev, dtype=torch.int32)
        n_touch.index_add_(0, inv.reshape(-1), torch.ones(inv.numel(), device=s.dev,
                                                          dtype=torch.int32))
        n_want = torch.zeros_like(n_touch)
        n_want.index_add_(0, my, torch.ones(my.numel(), device=s.dev, dtype=torch.int32))
        agreed = (n_touch == n_want) & (n_want > 0)
        cand = want & agreed[my]
        if not bool(cand.any()):
            return 0
        # 3 -- one cut per face: an edge is kept only if it wins on BOTH its faces. Ties are broken by
        # the lowest edge id, which is deterministic and does not depend on face order.
        big = uniq.numel() + 1
        best = torch.full((F.shape[0],), big, device=s.dev, dtype=torch.int64)
        best = torch.where(cand, my, best)
        win = torch.full((uniq.numel(),), big, device=s.dev, dtype=torch.int64)
        win.scatter_reduce_(0, my[cand], best[cand], reduce="amin", include_self=True)
        keep_edge = torch.zeros(uniq.numel(), device=s.dev, dtype=torch.bool)
        keep_edge[my[cand]] = True
        # a face may host only one kept edge; drop a face whose OTHER edges are also kept
        hosted = torch.zeros(F.shape[0], device=s.dev, dtype=torch.int32)
        hosted.index_add_(0, torch.arange(F.shape[0], device=s.dev),
                          keep_edge[inv].sum(dim=1).to(torch.int32))
        ok_face = hosted <= 1
        cut = cand & ok_face
        # and both faces of an edge must be cuttable, or the cut would leave a hanging node
        ok_edge = torch.ones(uniq.numel(), device=s.dev, dtype=torch.bool)
        badf = ~ok_face
        if bool(badf.any()):
            ok_edge[inv[badf].reshape(-1)] = False
        cut = cut & ok_edge[my]
        if not bool(cut.any()):
            return 0
        return self._cut_batch(live, F, k, my, cut, uniq)

    def _cut_batch(self, live, F, k, my, cut, uniq):
        s = self.s
        idx = torch.nonzero(cut, as_tuple=False).squeeze(1)
        eid = my[idx]
        ue, inv_e = torch.unique(eid, return_inverse=True)
        ne = ue.numel()
        nf = idx.numel()
        if s._n_ptr + ne > s.x.shape[0] or s._f_ptr + nf > s.F_all.shape[0]:
            raise RuntimeError("the reservoir is exhausted; raise max_refine")
        # one midpoint per distinct edge. `ue` holds INDICES into `uniq`, not the keys themselves --
        # decoding them as keys put every midpoint between two essentially arbitrary vertices, which
        # is why the mesh stayed conforming (rim 0, bad 0, the topology is untouched by where a point
        # is) and mass stayed exact (it only halves) while lambda came out 206x, area 48x and energy
        # 1e12 wrong. The key is `uniq[ue]`; only then does the division decode a vertex pair.
        ekey = uniq[ue]
        va = (ekey // (int(s.x.shape[0]) + 1)).to(torch.long)
        vb = (ekey % (int(s.x.shape[0]) + 1)).to(torch.long)
        mid = torch.arange(ne, device=s.dev) + s._n_ptr
        s.x[mid] = 0.5 * (s.x[va] + s.x[vb])
        s.node_occ[mid] = True
        s._n_ptr += ne
        mid_of_face = mid[inv_e]
        # the rotated triple: v0 opposite the split edge
        kk = k[idx]
        gath = torch.stack([(kk + 0) % 3, (kk + 1) % 3, (kk + 2) % 3], 1)
        tri = F[idx].gather(1, gath)
        f_old = live[idx]
        f_new = torch.arange(nf, device=s.dev) + s._f_ptr
        s._f_ptr += nf
        Dmi = self.R[kk] @ s.Dm_inv[f_old]
        A0, C0, Y2, mp = s.A0[f_old], s.C0[f_old], s.Y2[f_old], s.mass[f_old]
        for slot, cols, S in ((f_old, (0, 1, None), self.SA), (f_new, (0, None, 2), self.SB)):
            kids = torch.stack([tri[:, 0],
                                tri[:, 1] if cols[1] is not None else mid_of_face,
                                mid_of_face if cols[1] is not None else tri[:, 2]], 1)
            s.F_all[slot] = kids
            s.Dm_inv[slot] = S @ Dmi
            s.A0[slot] = A0 * 0.5
            s.C0[slot] = C0
            s.Y2[slot] = Y2
            s.mass[slot] = mp * 0.5
            s.face_occ[slot] = True
        s._resync()
        return nf

    def refine_where(self, too_long, max_passes=24):
        n = 0
        for _ in range(max_passes):
            c = self._pass(too_long)
            if not c:
                break
            n += c
        return n


# =============================================================================================
def _fresh(subdiv, dev, load=1.12):
    import bm_ops as BM
    s = BM.Sheet(subdiv=subdiv, R0=1.0, E=400.0, thickness=2.0e-3, nu=0.3, dev=dev, max_refine=3)
    s.x[s.live_nodes] = s.c + (s.x[s.live_nodes] - s.c) * load
    return s


def _state(s):
    return dict(faces=int(s.m), nodes=int(s.n), volume=float(s.enclosed_volume()),
                area=float(s.area().sum()), mass=float(s.total_mass()),
                lam=float(s.stretch_geo()[0].mean()), energy=float(s.energy(s.x)),
                euler=s.euler_check())


def compare(dev="cuda:0", subdiv=4, factor=0.9, max_splits=100000,
            out_name="07e_refine_batch"):
    """Sequential vs batched, on the SAME sheet and the same trigger: precision, then duration."""
    import json
    from bm_refine_local import LocalRefiner
    res = {}

    s1 = _fresh(subdiv, dev)
    tgt = factor * s1.mean_edge()
    b1 = _state(s1)
    t0 = time.time()
    n1 = LocalRefiner(s1).refine_where(tgt, max_splits=max_splits)
    torch.cuda.synchronize()
    t_seq = time.time() - t0
    a1 = _state(s1)

    s2 = _fresh(subdiv, dev)
    b2 = _state(s2)
    t0 = time.time()
    n2 = BatchRefiner(s2).refine_where(tgt)
    torch.cuda.synchronize()
    t_bat = time.time() - t0
    a2 = _state(s2)

    # torch.compile on the pass. Shapes change every pass, so this is expected to recompile; the
    # number is reported either way rather than assumed.
    t_cmp, n3, a3, err = None, None, None, None
    try:
        s3 = _fresh(subdiv, dev)
        r3 = BatchRefiner(s3)
        r3._pass = torch.compile(r3._pass, dynamic=True)
        t0 = time.time()
        n3 = r3.refine_where(tgt)
        torch.cuda.synchronize()
        t_cmp = time.time() - t0
        a3 = _state(s3)
    except Exception as e:                                   # noqa: BLE001
        err = f"{type(e).__name__}: {e}"[:300]

    rel = lambda p, q, k: abs(q[k] - p[k]) / max(abs(p[k]), 1e-30)          # noqa: E731
    res = {
        "sheet": {"subdiv": subdiv, "faces_seeded": b1["faces"], "loaded_to": 1.12, "factor": factor,
                  "trigger": tgt},
        "sequential": {"splits": n1, "faces": a1["faces"], "nodes": a1["nodes"],
                       "seconds": t_seq, "euler": a1["euler"]},
        "batched": {"splits": n2, "faces": a2["faces"], "nodes": a2["nodes"],
                    "seconds": t_bat, "euler": a2["euler"]},
        "batched_compiled": ({"splits": n3, "faces": a3["faces"], "seconds": t_cmp,
                              "euler": a3["euler"]} if a3 else {"error": err}),
        "speedup_batched": (t_seq / t_bat) if t_bat else None,
        "invariance of the batched split (its own before/after)": {
            "lam": rel(b2, a2, "lam"), "volume": rel(b2, a2, "volume"),
            "area": rel(b2, a2, "area"), "mass": rel(b2, a2, "mass"),
            "energy": rel(b2, a2, "energy"),
            "pass": bool(rel(b2, a2, "lam") < 1e-9 and rel(b2, a2, "volume") < 1e-10
                         and rel(b2, a2, "area") < 1e-10 and rel(b2, a2, "mass") < 1e-10
                         and rel(b2, a2, "energy") < 1e-9)},
        "agreement between the two": {
            "faces": [a1["faces"], a2["faces"]],
            "lam": abs(a1["lam"] - a2["lam"]), "area": abs(a1["area"] - a2["area"]),
            "energy": abs(a1["energy"] - a2["energy"]),
            "note": "the two need not produce the SAME mesh -- they choose different cuts in a "
                    "different order -- so this compares the invariants, not the topology"},
    }
    out = os.path.abspath(os.path.join(HERE, "..", "..", "log", "okuda_ECM", out_name))
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "batch_vs_sequential.json"), "w"), indent=1)
    print(f"[batch] seeded {b1['faces']} faces, trigger {tgt:.5f}", flush=True)
    print(f"[batch] sequential : {n1:6d} splits -> {a1['faces']:6d} faces in {t_seq:7.2f}s  "
          f"rim {a1['euler']['rim']} bad {a1['euler']['bad']}", flush=True)
    print(f"[batch] batched    : {n2:6d} splits -> {a2['faces']:6d} faces in {t_bat:7.2f}s  "
          f"rim {a2['euler']['rim']} bad {a2['euler']['bad']}  "
          f"({res['speedup_batched']:.1f}x)", flush=True)
    if a3:
        print(f"[batch] +compile   : {n3:6d} splits -> {a3['faces']:6d} faces in {t_cmp:7.2f}s  "
              f"rim {a3['euler']['rim']} bad {a3['euler']['bad']}", flush=True)
    else:
        print(f"[batch] +compile   : FAILED -- {err}", flush=True)
    inv = res["invariance of the batched split (its own before/after)"]
    print(f"[batch] batched invariance {'PASS' if inv['pass'] else 'FAIL'}: lam {inv['lam']:.2e} "
          f"vol {inv['volume']:.2e} area {inv['area']:.2e} mass {inv['mass']:.2e} "
          f"energy {inv['energy']:.2e} -> {out}", flush=True)
    return res


if __name__ == "__main__":
    compare()
