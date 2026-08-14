#!/usr/bin/env python
"""07c -- THE PLAQUE BELONGS TO A CELL, and divides with it.

    python test_07c_cell_plaque.py --frames 40        smoke
    python test_07c_cell_plaque.py --frames 401       the run

THE ONE CHANGE, and it is smaller than it looks. `RealDriver` triangulates the epithelium as ONE
TRIANGLE PER HALF-EDGE at frame-0 topology: 1,188 triangles that never divide, and `ct_face` indexes
those. Here it is ONE TRIANGLE PER CELL -- the cell's centroid and two of its own vertices, rebuilt
every frame from the CURRENT half-edge table -- so `ct_face` becomes the real cell id. That is the
index `Clutch.bind` has always wanted (`cell_of_edge`, `area_cell`, `N_f` per cell), so the certified
force path is untouched: the plaque still binds barycentrically to a triple of tissue vertices and
still returns its reaction to all three.

WHAT THAT BUYS, against what 07a measured:

    07a   plaques follow the SHEET's mesh: 2,562 -> 40,962 in two jumps of up to +300%, plaques per
          cell 12.81 -> 6.42, and median N_b per plaque 0.453 -> 0.000137 -- the adhesion empties as
          the same pool is divided among sixteen times more patches.
    07c   plaques follow the CELL: N0 per cell, seeded per cell, and a division re-labels the
          mother's plaques between mother and daughter (the pairing the lineage gate certified,
          3,869 of 3,869 exact) with their bonds travelling WITH them. Top-ups are seeded empty, so
          the total may only rise through the supply term.

GATES: G70 (plaques per cell holds), G71 (the count's step never exceeds the CELL count's step -- the
restated form, since the replay itself divides up to 8.5% of its cells in one kept frame), G72 (the
length distribution is stationary), G73 (a plaque born in a division starts at rest), G74 (integrin
per plaque stationary), G75 (division splits, never creates).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_05b_plaque as B                                              # noqa: E402
from plaque_lineage import PlaqueOwner                                   # noqa: E402
from test_05l_supply import Rig05l                                       # noqa: E402

# A DETERMINISTIC SPREAD INSIDE THE TRIANGLE, so a cell's plaques are distinct points and the same
# seed gives the same run. Barycentric, summing to one, biased toward the centroid because that is
# where a cell's basal face actually is.
_WEIGHTS = torch.tensor([[0.34, 0.33, 0.33], [0.50, 0.30, 0.20], [0.50, 0.20, 0.30],
                         [0.30, 0.50, 0.20], [0.30, 0.20, 0.50], [0.20, 0.50, 0.30],
                         [0.20, 0.30, 0.50], [0.40, 0.40, 0.20], [0.40, 0.20, 0.40],
                         [0.20, 0.40, 0.40], [0.60, 0.20, 0.20], [0.25, 0.35, 0.40]],
                        dtype=torch.float64)

_FAN = 8            # wedges per cell to test; a vertex-model cell has about six
NAME = "07c_cell_plaque"
SRC = "06_spheroid_ecm"


def surface_tris(pos, srce, trgt, face, nF, nv):
    """The FULL surface: one triangle per half-edge, (cell centroid, srce, trgt), and the cell each
    belongs to.

    TWO TABLES, NOT ONE, AND THAT IS THE WHOLE FIX. The first version of this file replaced the
    half-edge triangulation with one sliver triangle per cell, because `ct_face` had to become a cell
    id. But `F_epi` is read by TWO things: the adhesion's barycentric frame, and the contact operator
    that keeps the sheet outside the tissue. Slivers cannot represent a surface -- measured, the
    degenerate ones had area 2.5e-23, contact fired on 60 nodes with a force of 1.13, and at M = 2.09
    over 70 substeps that is a displacement of 2.4 in a box half a unit across. The run diverged on
    frame 0. So the surface keeps its half-edge triangulation, and the adhesion gets its own triple.
    """
    cen = torch.zeros((nF, 3), device=pos.device, dtype=pos.dtype)
    cnt = torch.zeros((nF,), device=pos.device, dtype=pos.dtype)
    cen.index_add_(0, face, pos[srce])
    cnt.index_add_(0, face, torch.ones_like(cnt[face]))
    cen = cen / cnt.clamp(min=1.0)[:, None]
    V = torch.cat([pos, cen], dim=0)
    F = torch.stack([nv + face, srce, trgt], dim=1)
    return V, F, face


class Rig07c(Rig05l):
    """05l with the epithelium as CELLS, and an adhesion that divides with them."""

    def __init__(self, N0=12, Nf0=3.0, **P):
        super().__init__(**P)
        self.N0 = int(N0)
        # THE RECEPTOR BUDGET PER CELL, and it is a parameter because it turned out to be the thing
        # holding three gates shut. At the inherited 3.0 a cell's twelve clusters share three
        # receptors: measured on 07e, the free pool falls 2.73 -> 0.114 per cell and a plaque holds a
        # MEDIAN OF 0.0125 BONDS, so the adhesion's stiffness is 5 x 0.0125 = 0.06 against the
        # sheet's elastic rate of ~4.5 -- one and a half percent of what it is pulling on. The 3.0
        # dates from the icosphere era, when a patch was one sheet node and nobody asked what it
        # contained. A plaque IS an integrin cluster of 20-50 (Kanchanawong 2010) and a cell carries
        # hundreds, so a few hundred per cell is the biological number.
        self.Nf0 = float(Nf0)
        z = self.z
        self.nv_of = [int(z[f"m{j}_pos"].shape[0]) for j in range(self.n_mesh)]
        self.nF_of = [int(z[f"m{j}_nF"]) for j in range(self.n_mesh)]
        self.ndiv_of = [np.asarray(z[f"m{j}_ndiv"], float) for j in range(self.n_mesh)]
        self._mesh_j = 0
        self._build_epi(0)
        self._seed_by_cell()
        print(f"[07c] cells {self.nF_of[0]} -> {self.nF_of[-1]}, N0 {self.N0} per cell, "
              f"{self.ct_node.numel()} plaques at frame 0", flush=True)

    # -- the epithelium, as cells --------------------------------------------------------------
    def _mesh_at(self, t):
        f = min(int(self.mesh_frames[-1]), max(0, int(t)))
        return min(max(int(np.searchsorted(self.mesh_frames, f, side="right") - 1), 0),
                   self.n_mesh - 1)

    def _build_epi(self, j):
        z = self.z
        pos = torch.as_tensor(z[f"m{j}_pos"], device=self.dev, dtype=self.dtype)
        pos = self.c + pos * self.scale
        srce = torch.as_tensor(z[f"m{j}_E_srce"].astype(np.int64), device=self.dev)
        trgt = torch.as_tensor(z[f"m{j}_E_trgt"].astype(np.int64), device=self.dev)
        face = torch.as_tensor(z[f"m{j}_E_face"].astype(np.int64), device=self.dev)
        nF, nv = self.nF_of[j], self.nv_of[j]
        live = face < nF
        self._srce, self._trgt, self._face = srce[live], trgt[live], face[live]
        V, F, cof = surface_tris(pos, self._srce, self._trgt, self._face, nF, nv)
        self.x_epi, self.F_epi, self.cell_of_tri = V, F, cof
        self.v_epi = torch.zeros_like(V)
        # `u_epi` IS INDEXED BY `F_epi`, so it has to grow with it. RealDriver sets it once, at 596
        # rows for the frame-0 topology; here the table is rebuilt every mesh and reaches ten
        # thousand, so the parent's contact builder read past the end of it -- a device-side assert
        # far from its cause, which is the fourth time this ladder has paid for exactly that.
        du = V - self.c
        self.u_epi = du / du.norm(dim=1, keepdim=True).clamp_min(1e-30)
        self._mesh_j, self._nF = j, nF
        # THE CENTROID INDEX IS NOT STABLE, AND THAT IS WHAT KILLED THE FIRST RUNS. A plaque's triple
        # is (cell centroid, v_a, v_b); the centroid lives at `nv + cell` in the stacked array, and
        # `nv` GROWS as the replay appends vertices. So a triple stored at one mesh points, at the
        # next, into the real vertices instead -- garbage geometry, a NaN normal, and a run that was
        # healthy at frame 31 and non-finite at 32 with nothing growing beforehand. The row is
        # re-pointed here, at the only place the mesh changes; the other two columns are real
        # vertices and are stable by construction.
        if getattr(self, "ct_tri", None) is not None:
            self.ct_tri = self.ct_tri.clone()
            self.ct_tri[:, 0] = nv + self.ct_face
        # the contact map indexes F_epi ROWS, and F_epi is rebuilt with the mesh, so it is rebuilt too
        if getattr(self, "cx_node", None) is not None:
            self._recontact()

    def _epi_anchor(self, t):
        """The REPLAYED TARGET at the current topology -- not the current state.

        The epithelium here is a dynamic body pulled toward its recording by `k_drive`; returning
        `self.x_epi` makes that term identically zero, the tissue is then driven by the adhesion's
        reaction alone, and the run diverges on frame 0. That is what the first version of this
        method did. A change of mesh rebuilds the cell table FIRST, so a division is a change in the
        SET rather than a silent re-indexing of the old one.
        """
        j = self._mesh_at(t)
        if j != self._mesh_j:
            self._on_division(j)
        return self._target(t)

    def _target(self, t):
        """The recorded positions at frame `t`, triangulated at the CURRENT cell topology."""
        z, j = self.z, self._mesh_j
        f = min(int(self.mesh_frames[-1]), max(0, int(t)))
        nv, nF = self.nv_of[j], self.nF_of[j]
        a = torch.as_tensor(z[f"m{j}_pos"][:nv], device=self.dev, dtype=self.dtype)
        if j + 1 < self.n_mesh:
            b = torch.as_tensor(z[f"m{j+1}_pos"][:nv], device=self.dev, dtype=self.dtype)
            span = float(self.mesh_frames[j + 1] - self.mesh_frames[j])
            al = (f - float(self.mesh_frames[j])) / span if span > 0 else 0.0
            a = (1.0 - al) * a + al * b
        pos = self.c + a * self.scale
        return surface_tris(pos, self._srce, self._trgt, self._face, nF, nv)[0]

    def _geom(self):
        """05d's geometry, with the adhesion reading its OWN triple and the area taken per CELL.

        `ct_tri` is the plaque's barycentric frame -- three tissue vertices, stable ids -- and
        `ct_face` is the cell that owns it, which is the index `Clutch.bind` wants for `N_f` and for
        `area_cell`. The reaction still goes to the three vertices `vf` names, so momentum is
        conserved exactly as G9--G13 certified it.
        """
        vf = self.ct_tri
        tri = self.x_epi[vf]
        p = (tri * self.ct_w[:, :, None]).sum(1)
        n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
        n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-30)
        n = n * torch.sign(((p - self.c) * n).sum(1, keepdim=True)).clamp(min=-1.0, max=1.0)
        d = ((self.sheet.x[self.ct_node] - p) * n).sum(1)
        te = self.x_epi[self.F_epi]
        a_tri = 0.5 * torch.cross(te[:, 1] - te[:, 0], te[:, 2] - te[:, 0], dim=1).norm(dim=1)
        a_cell = torch.zeros(self._nF, device=self.dev, dtype=self.dtype)
        a_cell.index_add_(0, self.cell_of_tri, a_tri)      # the cell's WHOLE basal face
        return vf, p, n, d, a_cell.clamp_min(1e-30)

    def build_contact(self):
        """A SHEET REFINEMENT DOES NOT TOUCH THE ADHESION -- that is the whole thesis of 07c.

        The parent rebuilds the contact set from the sheet's nodes whenever `bm_refine` fires, which
        is how the count came to follow the membrane's mesh: 2,562 -> 40,962 in two jumps. Here the
        set is owned by the CELLS, so a finer sheet changes nothing about it. Node ids survive a
        refinement (the pool is only appended to), so the existing plaques stay valid as they are.

        Left inherited, this raised a device-side assert at the first refinement: the parent re-seeds
        `ct_face` as a row of `F_epi`, which is no longer what `ct_face` means.

        It IS still needed once, during `super().__init__()`, before this rig has replaced the set --
        so the first call is delegated.

        AFTER THAT IT RE-POINTS RATHER THAN RE-SEEDS. A refinement does change the sheet's node set,
        which is why the parent rebuilds here, and leaving it a pure no-op left every plaque holding a
        stale node id: the run died exactly at frame 121, the first refinement. So each plaque keeps
        its CELL, its weights and its bonds, and is re-pointed at the live sheet node most nearly
        above its own attachment point. Ownership and integrin content survive; only the sheet-side
        endpoint moves, which is what a finer membrane means.
        """
        # TWO SETS, BECAUSE THEY ARE TWO OPERATORS. `bm_contact` is non-penetration and applies at
        # EVERY sheet node -- the note's own table says so -- while `plaque_pull` applies to the
        # cell-owned adhesion. Sharing one set was fine while there was a plaque per node; with the
        # adhesion owned by cells, many nodes have no plaque, so contact stopped being applied there
        # and the sheet sank 3.3e-3 inside (five l0) before the first refinement NaN'd on it.
        # `super()` builds the node-owned map, which is kept as `cx_*` for contact alone.
        keep = None
        if getattr(self, "ct_tri", None) is not None:
            keep = (self.ct_node, self.ct_face, self.ct_w, self.ct_tri,
                    self.clutch.Nb, self.clutch.D)
        # AND THE CLUTCH IS HIDDEN WHILE IT RUNS. 05d's `build_contact` regrids the bond arrays onto
        # the new node set, which is right when the adhesion IS the node set and wrong here: it would
        # map 4,848 cell-owned plaques onto 10,242 nodes, and it asserts on the size mismatch. The
        # bonds belong to the plaques and survive a refinement untouched.
        cl = self.clutch
        self.clutch = None
        try:
            super().build_contact()
        finally:
            self.clutch = cl
        self.cx_node, self.cx_face, self.cx_w = self.ct_node, self.ct_face, self.ct_w
        if keep is None:
            return
        self.ct_node, self.ct_face, self.ct_w, self.ct_tri = keep[:4]
        self.clutch.Nb, self.clutch.D = keep[4], keep[5]
        att = (self.x_epi[self.ct_tri] * self.ct_w[:, :, None]).sum(1) - self.c
        att = att / att.norm(dim=1, keepdim=True).clamp_min(1e-30)
        us = self.sheet.x - self.c
        us = us / us.norm(dim=1, keepdim=True).clamp_min(1e-30)
        # `sheet.live` IS A LIST OF FACE INDICES, not a node mask -- the node one is `live_nodes`.
        # Used as a mask it selects nodes by FACE id: in range, never asserting, and wrong.
        idx = self.sheet.live_nodes
        out = torch.empty(att.shape[0], dtype=torch.long, device=self.dev)
        step = 4096                                    # chunked: P x N would be billions of entries
        for k in range(0, att.shape[0], step):
            out[k:k + step] = idx[torch.argmax(att[k:k + step] @ us[idx].T, dim=1)]
        self.ct_node = out

    def _recontact(self):
        """Point every live sheet node at the epithelial triangle most nearly under it."""
        ln = self.sheet.live_nodes
        us = self.sheet.x[ln] - self.c
        us = us / us.norm(dim=1, keepdim=True).clamp_min(1e-30)
        ct = self.x_epi[self.F_epi].mean(1) - self.c
        ct = ct / ct.norm(dim=1, keepdim=True).clamp_min(1e-30)
        out = torch.empty(ln.numel(), dtype=torch.long, device=self.dev)
        for k in range(0, ln.numel(), 4096):
            out[k:k + 4096] = torch.argmax(us[k:k + 4096] @ ct.T, dim=1)
        self.cx_node, self.cx_face = ln, out
        self.cx_w = torch.full((ln.numel(), 3), 1.0 / 3.0, device=self.dev, dtype=self.dtype)

    def contact(self):
        """`bm_contact`, reading the ADHESION's triple instead of `F_epi[ct_face]`.

        The parent takes the plaque's triangle as `F_epi[ct_face]`, which was true while `ct_face`
        indexed the half-edge table. Here it is a CELL id, so the parent tested every plaque against
        an arbitrary triangle: measured, 2,376 nodes reported penetrating with a force of 19.07
        against the baseline's 393 and 0.0145, and the run diverged on frame 0. The law itself is
        unchanged -- one-sided, pushes out and never pulls in, reaction to the three vertices.
        """
        if self.k_c <= 0 or getattr(self, "cx_node", None) is None or self.cx_node.numel() == 0:
            z = torch.zeros_like(self.sheet.x)
            return z, torch.zeros_like(self.x_epi), 0, 0.0
        vf = self.F_epi[self.cx_face]
        tri = self.x_epi[vf]
        p = (tri * self.cx_w[:, :, None]).sum(1)
        n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
        n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-30)
        n = n * torch.sign(((p - self.c) * n).sum(1, keepdim=True)).clamp(min=-1.0, max=1.0)
        d = ((self.sheet.x[self.cx_node] - p) * n).sum(1)
        pen = (d - self.l0).clamp(max=0.0)
        f = -self.k_c * pen[:, None] * n
        fb = torch.zeros_like(self.sheet.x)
        fe = torch.zeros_like(self.x_epi)
        fb.index_add_(0, self.cx_node, f)
        fe.index_add_(0, vf.reshape(-1), (-f[:, None, :] * self.cx_w[:, :, None]).reshape(-1, 3))
        return fb, fe, int((pen < 0).sum()), float(-pen.min())

    # -- the division --------------------------------------------------------------------------
    def _on_division(self, j):
        old_nF, old_j = self._nF, self._mesh_j
        att = (self.x_epi[self.ct_tri] * self.ct_w[:, :, None]).sum(1)
        self._build_epi(j)
        nF = self._nF
        if nF <= old_nF:
            return
        nd0, nd1 = self.ndiv_of[old_j], self.ndiv_of[j]
        kk = min(len(nd0), len(nd1))
        moth = np.flatnonzero(nd1[:kk] - nd0[:kk] > 0)
        daug = np.arange(old_nF, nF)
        if len(moth) != len(daug):
            raise SystemExit(f"[07c] mesh {j}: {len(moth)} mothers, {len(daug)} daughters -- the "
                             f"lineage the gate certified does not hold here")
        cen = self.x_epi[self.nv_of[j] + np.arange(nF)].float().cpu().numpy()
        own = PlaqueOwner(self.ct_face.cpu().numpy(), nF, self.N0)
        own.divide(moth, daug, att.float().cpu().numpy(), cen)
        self.ct_face = torch.as_tensor(own.cell, device=self.dev)
        need = own.deficit()
        if need.sum():
            self._seed_into(np.repeat(np.arange(nF), need))

    def _seed_by_cell(self):
        self._seed_into(np.repeat(np.arange(self._nF), self.N0), fresh=True)

    def _seed_into(self, cells, fresh=False):
        """New plaques on the named cells: DISTINCT patches, not copies.

        The first version gave every plaque of a cell the same sheet node and the same barycentric
        weights, because they all started from the cell's centroid -- so a cell's twelve adhesions
        were twelve copies of one, stacking twelve times the stiffness on a single node. It ran for
        twenty-two frames and diverged, before any division. A cell's adhesions are distinct patches:
        each takes its own sheet node (the k-th most nearly above it) and its own point inside the
        cell's triangle, from a fixed low-discrepancy set so the seeding is deterministic.
        """
        cells_np = np.asarray(cells, np.int64)
        if not cells_np.size:
            return
        cells_t = torch.as_tensor(cells_np, device=self.dev)
        # rank of each plaque WITHIN its cell: the groups are contiguous, so it is a running count
        _, idx0, cnt = np.unique(cells_np, return_index=True, return_counts=True)
        # the groups are contiguous and sorted, so repeating each group's FIRST index by its own
        # count already gives, for every element, the index its group starts at
        first = np.repeat(idx0, cnt)
        rank = torch.as_tensor(np.arange(cells_np.size) - first, device=self.dev)
        kmax = int(rank.max().item()) + 1

        # THE ATTACHMENT POINT GOES UNDER THE NODE, not at the cell's centre. Giving every plaque of
        # a cell the same triangle with spread barycentric weights anchors them all at the middle of
        # the cell while their sheet ends are the twelve nearest nodes -- so the outer ones reach
        # SIDEWAYS by about a cell radius, five microns, and the link they form is diagonal across
        # the cell rather than radial across the 40 nm a cluster spans. Measured: with a binding
        # distance of 3 l0 = 2.1 um, 2,200 of 2,400 plaques were out of reach at seeding, one per cell
        # surviving. Here each node's own direction is ray-cast against the wedges of ITS cell -- the
        # half-edge triangles the surface table already holds -- and the plaque binds the wedge that
        # contains it, with that ray's barycentric weights. The link is then radial by construction.
        # the cell's wedges, as a padded table: sort the triangles by cell, rank each within its
        # cell, and scatter. A vertex-model cell has about six half-edges, so `_FAN` = 8 covers it.
        cot = self.cell_of_tri
        order = torch.argsort(cot)
        cs = cot[order]
        counts = torch.bincount(cs, minlength=self._nF)
        starts = torch.cumsum(counts, 0) - counts
        rank_in = torch.arange(cs.numel(), device=self.dev) - starts[cs]
        room = rank_in < _FAN
        fan = torch.zeros((self._nF, _FAN), dtype=torch.long, device=self.dev)
        fan[cs[room], rank_in[room]] = order[room]
        tri_fan = self.F_epi[fan]                   # (nF, FAN, 3)
        # the sheet node each plaque takes: the k-th most nearly overhead LIVE node, k being its rank
        # within its cell, so a cell's clusters are distinct patches rather than copies
        cen = self.x_epi[self.F_epi[fan[cells_t, 0], 0]]
        u = cen - self.c
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-30)
        us = self.sheet.x - self.c
        us = us / us.norm(dim=1, keepdim=True).clamp_min(1e-30)
        ln = self.sheet.live_nodes
        top = torch.topk(u @ us[ln].T, k=min(kmax, ln.numel()), dim=1).indices
        node = ln[top.gather(1, rank.clamp(max=top.shape[1] - 1)[:, None]).squeeze(1)]
        U = self.u_epi[tri_fan]                                  # unit directions of each wedge
        Minv = torch.linalg.pinv(U[cells_t].transpose(-1, -2))    # (P, FAN, 3, 3)
        un = (self.sheet.x[node] - self.c)
        un = un / un.norm(dim=1, keepdim=True).clamp_min(1e-30)
        bc = torch.einsum("pkij,pj->pki", Minv, un)              # (P, FAN, 3)
        good = bc.min(dim=2).values                              # the wedge that contains the ray
        pick = good.argmax(dim=1)
        tri = tri_fan[cells_t, pick]
        w = bc[torch.arange(bc.shape[0], device=self.dev), pick].clamp_min(0.0)
        w = w / w.sum(1, keepdim=True).clamp_min(1e-30)

        # A PLAQUE IS ONLY CREATED WHERE THE MEMBRANE IS WITHIN REACH. `bind_max` is in rest lengths;
        # beyond it an integrin cannot span the gap, so no cluster forms and the cell simply has fewer
        # than N0 until the sheet comes close enough. Default is infinite, which is 07c's behaviour.
        bmax = getattr(self, "bind_max", float("inf")) * self.plq.l0 if hasattr(self, "plq") else \
            getattr(self, "bind_max", float("inf")) * 6.0e-4
        if np.isfinite(bmax):
            att = (self.x_epi[tri] * w[:, :, None]).sum(1)
            near = (self.sheet.x[node] - att).norm(dim=1) <= bmax
            if not bool(near.all()):
                node, cells_t, w, tri = node[near], cells_t[near], w[near], tri[near]
                if not node.numel():
                    return
        if fresh:
            self.ct_node, self.ct_face, self.ct_w, self.ct_tri = node, cells_t, w, tri
            self.clutch.provision(node.numel(), self._nF, Nf0=self.Nf0)
            return
        self.ct_node = torch.cat([self.ct_node, node])
        self.ct_face = torch.cat([self.ct_face, cells_t])
        self.ct_w = torch.cat([self.ct_w, w])
        self.ct_tri = torch.cat([self.ct_tri, tri])
        z3 = torch.zeros(node.numel(), device=self.dev, dtype=self.dtype)
        self.clutch.Nb = torch.cat([self.clutch.Nb, z3])
        self.clutch.D = torch.cat([self.clutch.D, torch.zeros((node.numel(), 3), device=self.dev,
                                                              dtype=self.dtype)])
        if self.clutch.Nf.shape[0] < self._nF:
            nf0 = float(self.clutch.Nf.mean())
            self.clutch.Nf = torch.cat([self.clutch.Nf, torch.full(
                (self._nF - self.clutch.Nf.shape[0],), nf0, device=self.dev, dtype=self.dtype)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames", type=int, default=401)
    ap.add_argument("--N0", type=int, default=12)
    ap.add_argument("--name", default=NAME)
    a = ap.parse_args()
    d = os.path.join(B.LOG, a.name)
    os.makedirs(d, exist_ok=True)
    P = dict(subdiv=4, E=400.0, thickness=2.0e-3, nu=0.3, kn=5.0, sigma_T=7.0, zeta=20.0,
             s_target=1.0, k_drive=50.0, dev=a.device, max_refine=2, edge_trigger=1.45,
             reseed=True, tau_bm=40.0, rho_crit=0.0)
    rig = Rig07c(N0=a.N0, **P)
    S = {k: [] for k in ("t", "cells", "plaques", "ppc", "nb_med", "nf_mean", "receptor_total",
                         "lam")}
    store, i = {}, 0
    t0 = time.time()
    for t in range(a.frames):
        rig.frame(t)
        if not rig.alive():
            print(f"[07c] DIVERGED at {t}", flush=True)
            break
        if t % 2 == 0:
            nb = rig.clutch.Nb
            l1, _ = rig.sheet.stretch_geo()
            S["t"].append(t); S["cells"].append(int(rig._nF))
            S["plaques"].append(int(rig.ct_node.numel()))
            S["ppc"].append(float(rig.ct_node.numel() / max(rig._nF, 1)))
            S["nb_med"].append(float(nb.median()))
            S["nf_mean"].append(float(rig.clutch.Nf.mean()))
            S["receptor_total"].append(float(nb.sum() + rig.clutch.Nf.sum()))
            S["lam"].append(float(l1.mean()))
            # the same store 06 writes, so `render_std` can draw this run with no special case
            att = (rig.x_epi[rig.ct_tri] * rig.ct_w[:, :, None]).sum(1)
            store[f"t{i}"] = np.int32(t)
            store[f"x{i}"] = rig.sheet.x.float().cpu().numpy()
            store[f"f{i}"] = rig.sheet.Fc.cpu().numpy().astype(np.int32)
            store[f"v{i}"] = l1.float().cpu().numpy()
            store[f"r{i}"] = (rig.sheet.areal_density() / rig.sheet.rho0).float().cpu().numpy()
            store[f"e{i}"] = rig.x_epi.float().cpu().numpy()
            store[f"n{i}"] = rig.ct_node.cpu().numpy().astype(np.int32)
            store[f"p{i}"] = att.float().cpu().numpy()
            store[f"nb{i}"] = nb.float().cpu().numpy()
            store[f"nf{i}"] = rig.clutch.Nf.float().cpu().numpy()
            store[f"cf{i}"] = rig.ct_face.cpu().numpy().astype(np.int32)
            i += 1
    np.savez_compressed(os.path.join(d, "bm_frames.npz"), n_kept=np.int32(i),
                        FE=rig.F_epi.cpu().numpy().astype(np.int32),
                        centre=rig.c.float().cpu().numpy(), scale=np.float64(rig.scale), **store)
    json.dump(dict(run=a.name, N0=a.N0, frames=len(S["t"]), series=S),
              open(os.path.join(d, "metrics.json"), "w"), indent=1)
    from spec_06 import write_spec
    write_spec(d, rig, name=a.name, frames=a.frames, matrix_src=SRC,
               extra=dict(kind="cell-owned adhesion", N0=a.N0,
                          cells=[S["cells"][0], S["cells"][-1]],
                          plaques=[S["plaques"][0], S["plaques"][-1]],
                          ppc=[S["ppc"][0], S["ppc"][-1]]))
    print(f"[07c] {len(S['t'])} kept frames in {time.time()-t0:.0f}s -- cells {S['cells'][0]} -> "
          f"{S['cells'][-1]}, plaques {S['plaques'][0]} -> {S['plaques'][-1]}, ppc "
          f"{S['ppc'][0]:.2f} -> {S['ppc'][-1]:.2f}, median Nb {S['nb_med'][0]:.4f} -> "
          f"{S['nb_med'][-1]:.4f}, lam_geo {S['lam'][-1]:.3f} -> {d}", flush=True)


if __name__ == "__main__":
    main()
