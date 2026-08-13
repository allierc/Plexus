"""mesh_contact_ops -- the 03 interface, generalised from a flat patch to the spheroid's own mesh.

WHAT IS NEW HERE AND WHAT IS NOT. `test_03_mesh_contact.py` established the scheme
(ICFEMP, Chen et al. 2015 CMAME 293:1): a particle is tested against a mesh FACE rather than
against a grid node, the reaction is distributed to that face's vertices by the same weights that
built the force, and the penalty is written as a fraction of the explicit stability ceiling. All
three survive verbatim. What that rig could not do is the geometry: its patch is flat and
axis-aligned, so "which face is this particle under" is integer arithmetic on a lattice. A
spheroid's surface is curved, moving, and re-meshed every frame by division and T1.

THE LOOKUP, WHICH IS THE ONLY THING A CURVED SURFACE NEEDS THAT A FLAT ONE DOES NOT. The tissue is
star-shaped about its own centroid -- `ecm_from_cell[replay]` already relies on this and P11 is the
premise that reports when a tissue stops being -- so a direction (theta, phi) names at most a few
faces, and the face a particle is under is found by binning its own direction. That is O(1) per
particle with no tree, and it is the same bin structure `apical_map` already uses for R(theta,phi).
The difference from the radius map is the whole point of this operator: a bin of the map is a
NUMBER and can only push radially and can return its reaction to nothing, while a bin here holds
the TRIANGLES themselves, so the contact has a real face normal, real barycentric weights, and
three real vertices to hand the reaction back to.

THE BIN GRID IS RE-SIZED EVERY FRAME, and it has to be. A bucket lookup that tests a particle's own
bin and its eight neighbours is exact only while a triangle is smaller than a bin, and the tissue's
faces shrink by a factor of ~6 in angular size between frame 0 (edge 0.77 tissue units at radius
4.66, i.e. 0.17 rad) and the last (0.49 at 17.6, i.e. 0.028 rad). A grid fixed at either end is
either wrong at the other or a hundred times more expensive than it needs to be, so `n_theta` is
set from the frame's own largest triangle.

AND THE BINS ARE SQUARE EVERYWHERE, WHICH A (theta, phi) LATTICE IS NOT. On a plain lattice a phi
bin near the pole is a sliver -- at theta = 0.1 rad it is a tenth of the row's own height -- so a
triangle spans ten of them and the nine-bin query misses the face the particle is standing on. It
fails silently, only near the poles, and the symptom is matrix inside the tissue, which is exactly
the artefact `cell_exclude` was written to sweep up rather than to explain. The grid here gives
each row as many phi bins as it can hold square ones, so the pole row collapses to three bins by
the same formula that gives the equator two hundred, and `selftest()` certifies the whole lookup
against brute force before any run is trusted to it.

THE CLAMP IS THIS OPERATOR'S, NOT THE SCATTER'S. `mpm_scatter` clamps the external acceleration at
`a_max` after this operator has returned, so a contact that saturates it would have its reaction
recorded at the unclamped value and the momentum test -- the one measurement that catches a missing
reaction -- would pass on a force that was never applied. The clamp is applied here instead, before
the reaction is computed, and `a_max` on the scatter is set above it.

WHAT IT STILL DOES NOT DO. The reaction is computed, conserved and recorded, and then the tissue
does not feel it, because the tissue is a replay: pass 1 finished before pass 2 began. What this
operator changes is that the reaction now EXISTS as a force on named vertices rather than as a
pressure binned by direction -- so making the coupling mutual is a schedule change (hand
`vertex_force` to a live epithelium) rather than an architecture change.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator

# ONE ROW PER FRAME, appended by the operator and read by the run script after the engine returns.
# Module-level for the reason `ecm_ops.STRESS_HISTORY` is: the operator instance is built inside the
# engine and there is no handle to it afterwards.
CONTACT_HISTORY: list = []          # per frame: the dict of scalars below
VERTEX_FORCE: list = []             # per frame: the reaction on the tissue, one row per vertex
PRESSURE_MAP: list = []             # per frame: that reaction as a pressure on a (theta, phi) grid
INSIDE_HISTORY: list = []           # per frame: how many particles ended up behind the surface
_LIVE: dict = {}                    # the live contact operator, so the counter can share its bins


def _grid(nrow, dev):
    """A REDUCED grid: rows uniform in theta, and each row given as many phi bins as it can hold
    SQUARE ones -- `n_phi(row) = 2 pi sin(theta) / d(theta)`.

    A plain (theta, phi) lattice is square only at the equator. At theta = 0.1 rad a phi bin is a
    tenth of the row's own height, so a triangle spans ten of them and a lookup that tests a
    particle's own bin and its eight neighbours misses the face it is standing on -- silently, and
    only near the poles, which is the worst way for a contact to fail. Here every bin is d(theta)
    on a side wherever it sits, so the 3x3 query is exact everywhere, and the pole row collapses to
    three bins by the same formula rather than by a special case.
    """
    dth = math.pi / nrow
    th_c = (torch.arange(nrow, device=dev, dtype=torch.float32) + 0.5) * dth
    npr = (2 * math.pi * th_c.sin() / dth).round().long().clamp_min(1)
    off = torch.zeros(nrow + 1, dtype=torch.long, device=dev)
    off[1:] = torch.cumsum(npr, 0)
    return dict(nrow=nrow, dth=dth, npr=npr, off=off, nbin=int(off[-1]))


def _bins_of(u, G):
    """Direction -> (row, bucket)."""
    th = torch.acos(u[:, 2].clamp(-1, 1))
    ph = torch.atan2(u[:, 1], u[:, 0]) % (2 * math.pi)
    it = (th / G["dth"]).long().clamp(0, G["nrow"] - 1)
    npr = G["npr"][it]
    ip = (ph / (2 * math.pi) * npr).long().clamp(torch.zeros_like(npr), npr - 1)
    return it, ph, G["off"][it] + ip


def _neighbours(it, ph, G):
    """The 3x3 neighbourhood, as bucket indices. Each of the three rows has its own phi count, so
    the phi bin is recomputed per row rather than carried across -- carrying it is what a uniform
    lattice lets you do and a reduced one does not."""
    dev = it.device
    rows = (it[:, None] + torch.tensor([-1, 0, 1], device=dev)[None, :]).clamp(0, G["nrow"] - 1)
    npr = G["npr"][rows]                                              # [n,3]
    ip = (ph[:, None] / (2 * math.pi) * npr).long()
    out = []
    for d in (-1, 0, 1):
        out.append(G["off"][rows] + (ip + d) % npr)
    return torch.stack(out, 2).reshape(it.shape[0], 9)                # [n,9]


@register_operator("mesh_contact", family="mechanics", set="particle", kind="lateral")
class MeshContact(Lateral):
    """Particle-to-surface contact against a replayed triangulated tissue.

    Penalty in the face normal, regularised Coulomb friction against the face's own velocity, and
    the reaction distributed to the face's vertices by the barycentric weights that built it.
    """

    EMIT = "mpm_acceleration"          # consumed by `mpm_scatter` as a_ext, like `ecm_from_cell`
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["tissue"]
    MECHANISM_TAGS = ["cell_matrix_contact", "particle_to_surface", "friction", "moving_boundary"]
    PARAM_ROLES = {"k_frac": "penalty_fraction_of_ceiling", "mu": "friction_coefficient",
                   "scale": "tissue_to_box_scale", "eps_v": "slip_regularisation_velocity"}
    REFERENCE = ("Chen, Z., Qiu, X., Zhang, X. & Lian, Y. (2015). Improved coupling of finite "
                 "element method with material point method based on a particle-to-surface contact "
                 "algorithm (ICFEMP). Comput. Methods Appl. Mech. Engrg. 293:1-19. "
                 "doi:10.1016/j.cma.2015.04.005 -- the scheme this operator implements. Grid-node "
                 "coupling (CFEMP, Lian, Y. P., Zhang, X. & Liu, Y. (2011) CMAME 200:3482-3494, "
                 "doi:10.1016/j.cma.2011.07.014) is the alternative it was chosen over: it needs "
                 "mesh and grid cells of comparable size, which this prototype violates. The "
                 "surface is Okuda, S. et al. (2018) Sci. Rep. 8:2386.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        self.scale = float(params.get("scale", 1.0))
        # THE PENALTY AS A FRACTION OF THE EXPLICIT CEILING. In acceleration form the contact is
        # a = k*d, integrated at the substep, so it is stable only while dt_sub*sqrt(k) < 1, i.e.
        # k < 1/dt_sub^2. Written as a fraction so the number cannot be quoted as an absolute
        # stiffness that happens to be three hundred times past its own limit, which is how the
        # first version of the flat rig and `flat_mpm` both failed.
        self.k_frac = float(params.get("k_frac", 0.15))
        self.mu = float(params.get("mu", 0.4))
        self.eps_v = float(params.get("eps_v", 1.0e-3))
        self.a_max = float(params.get("a_max", 3.0e4))
        self.dt_frame = float(params.get("dt", 3.2e-3))
        self.band_cells = float(params.get("band_cells", 3.0))   # prefilter width, in grid cells
        self.n_grid = int(params.get("n_grid", 64))
        self.cap_rows = int(params.get("cap_rows", 2))
        self.map_theta = int(params.get("map_theta", 32))        # the pressure map's own resolution
        self.map_phi = int(params.get("map_phi", 64))
        # PASS-2 FRAMES PER KEPT MESH. The tissue cache keeps 200 meshes out of 402 tissue frames,
        # so a pass-2 run of 400 frames advances one mesh every two frames; `stride` is that number,
        # and the mesh in between is interpolated rather than held. Held, the surface would stand
        # still for a frame and then jump, and a jump into a penalty contact is a shock the material
        # has to absorb -- the interpolation is also exactly what the friction law's vertex velocity
        # already assumes, so holding would have made the two disagree.
        self.stride = max(1, int(params.get("mesh_stride", 1)))
        self.verbose = bool(params.get("verbose", True))

        z = np.load(str(params["tissue"]))
        self.n_mesh = int(len(z["mesh_frames"]))
        self._z = {k: z[k] for k in z.files if k.startswith("m")}
        self._nmesh_keys = z["mesh_frames"]
        self._frame = -2
        self._j = -1
        self._built = None
        self._dom = None
        self._newframe = False
        # THE COUNTER SHARES THIS OPERATOR'S BINS rather than rebuilding them: two bin structures
        # that disagree would have the diagnostic measuring a different surface from the one the
        # contact acted on, which is the one way a non-penetration count can be wrong and look right.
        _LIVE["contact"] = self

    # ---- the mesh of one frame ------------------------------------------------------------
    def _mesh(self, j, dev, dt_):
        """Vertices, half-edges and face sizes of kept mesh `j`, in BOX units and centred."""
        p = torch.as_tensor(self._z[f"m{j}_pos"], device=dev, dtype=dt_) * self.scale
        es = torch.as_tensor(self._z[f"m{j}_E_srce"].astype(np.int64), device=dev)
        et = torch.as_tensor(self._z[f"m{j}_E_trgt"].astype(np.int64), device=dev)
        ef = torch.as_tensor(self._z[f"m{j}_E_face"].astype(np.int64), device=dev)
        return p, es, et, ef

    def _build(self, j, dev, dt_, alpha=0.0):
        """Everything about the surface that changes once per frame: triangles, their bins, and the
        vertex velocity the friction law needs.

        `alpha` in [0,1) is how far this frame sits between kept mesh `j` and `j+1`. The TOPOLOGY is
        always mesh `j`'s -- a vertex that does not exist yet cannot be interpolated toward -- and
        the positions of the vertices the two meshes share are blended. `cell_divide` appends vertices
        and never renumbers them (checked: Nv is monotone and a vertex's position is continuous
        across a kept-frame pair), so the shared set is the common prefix.
        """
        V, es, et, ef = self._mesh(j, dev, dt_)
        Vn = None
        if j + 1 < self.n_mesh:
            Vn, _, _, _ = self._mesh(j + 1, dev, dt_)
            if alpha > 0:
                m = min(V.shape[0], Vn.shape[0])
                V = V.clone()
                V[:m] = (1.0 - alpha) * V[:m] + alpha * Vn[:m]
        nv, nF = V.shape[0], int(ef.max()) + 1
        # FACE CENTROIDS. Every face contributes each of its vertices exactly once as a half-edge
        # source, so a scatter-mean over `E_srce` is the polygon's centroid with no ring ordering
        # required -- and the ordering is exactly what a re-meshed tissue does not hand over.
        cnt = torch.bincount(ef, minlength=nF).clamp_min(1)
        cen = torch.zeros(nF, 3, device=dev, dtype=dt_)
        cen.index_add_(0, ef, V[es])
        cen = cen / cnt[:, None].to(dt_)
        # ONE TRIANGLE PER HALF-EDGE: (face centroid, source vertex, target vertex). A fan, but
        # built from the half-edge list rather than from a ring, so it is correct for a polygon of
        # any size and needs no ordering. The centroid is a VIRTUAL vertex: whatever reaction it
        # receives is handed to the face's real vertices, equally, at the end.
        A, B, C = cen[ef], V[es], V[et]
        nrm = torch.cross(B - A, C - A, dim=1)
        nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp_min(1e-20)
        # OUTWARD, DECIDED BY THE STAR-SHAPE. The mesh is centroid-referenced, so a face's own
        # position is its outward direction and the winding never has to be trusted.
        flip = ((nrm * (A + B + C)).sum(1) < 0)
        nrm = torch.where(flip[:, None], -nrm, nrm)

        # THE BIN GRID, SIZED BY THIS FRAME'S LARGEST TRIANGLE. The 3x3 query below is exact only
        # while a triangle fits inside a bin, and the faces shrink sixfold in angular size between
        # frame 0 (edge 0.17 rad) and the last (0.028 rad); a fixed grid is wrong at one end or a
        # hundred times more expensive than it needs to be at the other. The 99.9th percentile
        # rather than the maximum, because one sliver of a just-divided cell should not set the
        # resolution of the whole lookup -- and the largest triangles are the ones a 3x3 query has
        # the most slack for, since they are found from any of their three corners.
        rA, rB, rC = (A.norm(dim=1), B.norm(dim=1), C.norm(dim=1))
        rmin = torch.minimum(torch.minimum(rA, rB), rC).clamp_min(1e-9)
        ang = (torch.maximum((B - A).norm(dim=1),
                             torch.maximum((C - A).norm(dim=1), (C - B).norm(dim=1))) / rmin)
        a999 = float(torch.quantile(ang.float(), 0.999))
        nrow = int(min(200, max(4, math.floor(math.pi / max(a999, 1e-3)))))
        G = _grid(nrow, dev)
        nb = G["nbin"]

        # EACH TRIANGLE INTO THE BUCKETS OF ITS THREE CORNERS. With the grid sized as above a
        # triangle spans at most one bin, so a particle whose direction lies inside the triangle has
        # one of its corners in its own bin or in a neighbouring one -- which is what the 3x3 query
        # covers. Registered three times rather than once so a triangle straddling a bin edge is
        # found from either side.
        u3 = torch.cat([A, B, C]) / torch.cat([rA, rB, rC]).clamp_min(1e-20)[:, None]
        _, _, b3 = _bins_of(u3, G)
        tri3 = torch.arange(A.shape[0], device=dev).repeat(3)
        order = torch.argsort(b3 * (A.shape[0] + 1) + tri3)
        b3, tri3 = b3[order], tri3[order]
        counts = torch.bincount(b3, minlength=nb)
        K = int(counts.max())
        first = torch.zeros(nb + 1, dtype=torch.long, device=dev)
        first[1:] = torch.cumsum(counts, 0)
        slot = torch.arange(b3.shape[0], device=dev) - first[b3]
        table = torch.full((nb, K), -1, dtype=torch.long, device=dev)
        table[b3, slot] = tri3
        # AND THE OUTERMOST RADIUS IN EACH BUCKET, which is the prefilter: a particle further out
        # than any corner of any triangle the bucket holds, plus a margin, cannot be in contact and
        # is never tested. This is what keeps the query over ~1% of the matrix instead of all of it.
        # Taken through the table rather than by a scatter-max, which has no deterministic
        # implementation and would make a re-run non-reproducible for a number that is a maximum.
        rtri = torch.maximum(torch.maximum(rA, rB), rC)
        rmax = torch.where(table >= 0, rtri[table.clamp_min(0)],
                           torch.zeros_like(rtri[0]).expand(nb, K)).max(dim=1).values

        # VERTEX VELOCITY, from the next kept mesh. `cell_divide` appends vertices and never renumbers
        # them (checked: Nv is monotone and a vertex's position is continuous across a kept-frame
        # pair), so the common prefix is the same vertex on both sides; a vertex that does not yet
        # exist in the next mesh gets zero.
        Vv = torch.zeros_like(V)
        if Vn is not None:
            # PER PASS-2 FRAME, so the divisor is the time between kept meshes and not the frame's.
            # With `stride` frames to a mesh the surface covers that displacement in `stride` frames,
            # and a velocity computed as if it took one would make the friction law read a surface
            # sliding `stride` times faster than the positions it is handed.
            m = min(V.shape[0], Vn.shape[0])
            Vv[:m] = (Vn[:m] - V[:m]) / (self.stride * self.dt_frame)
        return dict(V=V, Vv=Vv, es=es, et=et, ef=ef, cnt=cnt, A=A, B=B, C=C, nrm=nrm,
                    table=table, rmax=rmax, G=G, nv=nv, nF=nF, K=K, n_tri=int(A.shape[0]))

    # ---- the contact ----------------------------------------------------------------------
    def _query(self, M, x, u, r):
        """For each candidate particle, the outermost triangle its own direction hits.

        Returns (hit, tri, t, w) -- whether a face was found, which one, the radius of the surface
        along the particle's ray, and the three barycentric weights.
        """
        dev = x.device
        it, ph, _ = _bins_of(u, M["G"])
        bn = _neighbours(it, ph, M["G"])                                       # [n,9]
        cand = M["table"][bn].reshape(x.shape[0], -1)                          # [n, 9K]
        ok = cand >= 0
        ci = cand.clamp_min(0)
        A = M["A"][ci]; B = M["B"][ci]; C = M["C"][ci]                         # [n, 9K, 3]
        # MOLLER-TRUMBORE, with the ray from the tissue centroid along the particle's own
        # direction. `t` is then the radius of the surface in that direction -- the curved-surface
        # form of the flat rig's interpolated `z_face` -- and the weights fall out of the same
        # solve rather than being a second, separate projection.
        e1, e2 = B - A, C - A
        uu = u[:, None, :]
        p = torch.cross(uu.expand_as(e2), e2, dim=2)
        det = (e1 * p).sum(2)
        inv = 1.0 / torch.where(det.abs() < 1e-20, torch.full_like(det, 1e-20), det)
        s = -A                                              # ray origin is the centroid = 0
        w1 = (s * p).sum(2) * inv
        q = torch.cross(s, e1, dim=2)
        w2 = (uu * q).sum(2) * inv
        t = (e2 * q).sum(2) * inv
        tol = 1e-6
        good = ok & (det.abs() > 1e-20) & (w1 >= -tol) & (w2 >= -tol) & (w1 + w2 <= 1 + tol) & (t > 0)
        tt = torch.where(good, t, torch.full_like(t, -1.0))
        best = tt.argmax(dim=1)                             # the OUTERMOST face on the ray
        ar = torch.arange(x.shape[0], device=dev)
        hit = good[ar, best]
        tri = ci[ar, best]
        tb = t[ar, best]
        w = torch.stack([1.0 - w1[ar, best] - w2[ar, best], w1[ar, best], w2[ar, best]], 1)
        return hit, tri, tb, w

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        # `or -1` WOULD MAKE FRAME 0 READ AS -1, because 0 is falsy -- and frame 0 is the frame
        # that opens the history, so the aggregation below would index an empty list.
        f = getattr(H, "frame", None)
        f = -1 if f is None else int(f)
        j = min(self.n_mesh - 1, max(0, f // self.stride))
        alpha = (f - j * self.stride) / self.stride if self.stride > 1 else 0.0
        if f != self._frame:
            self._frame, self._newframe = f, True
            # REBUILT EVERY FRAME WHEN THE MESH IS INTERPOLATED, because the geometry now changes
            # between kept meshes as well as at them. At stride 1 this is the same rebuild-on-change
            # it always was; the build is a few milliseconds against a frame's eight substeps.
            if j != self._j or alpha > 0:
                self._j = j
                self._built = self._build(j, dev, dt_, alpha)
        elif self._built is None:
            self._j = j
            self._built = self._build(j, dev, dt_, alpha)
        M = self._built
        dt_sub = float(getattr(H, "sub_dt", None) or self.dt_frame)
        k = (self.k_frac / dt_sub) ** 2
        dx = 1.0 / self.n_grid

        c = torch.tensor(self.centre, device=dev, dtype=dt_)
        d = pos - c
        r = d.norm(dim=1).clamp_min(1e-12)
        u = d / r[:, None]
        # THE PREFILTER: only particles inside the outermost corner their own bucket holds, plus a
        # margin. Everything else is beyond any face this direction can offer and is never tested.
        _, _, b = _bins_of(u, M["G"])
        near = r < (M["rmax"][b] + self.band_cells * dx)
        acc = torch.zeros_like(pos)
        if not bool(near.any()):
            self._record(f, 0, 0.0, 0.0, 0.0, acc, None, None, M, dev, dt_)
            return {self.at: acc}
        idx = torch.nonzero(near).squeeze(1)
        hit, tri, t, w = self._query(M, pos[idx], u[idx], r[idx])
        if not bool(hit.any()):
            self._record(f, 0, 0.0, 0.0, 0.0, acc, None, None, M, dev, dt_)
            return {self.at: acc}
        idx, tri, t, w = idx[hit], tri[hit], t[hit], w[hit]
        n_hat = M["nrm"][tri]
        xs = c + t[:, None] * u[idx]                       # the point on the surface, on the ray
        depth = ((xs - pos[idx]) * n_hat).sum(1).clamp_min(0.0)
        inside = depth > 0
        idx, tri, w, n_hat, depth = idx[inside], tri[inside], w[inside], n_hat[inside], depth[inside]
        if idx.numel() == 0:
            self._record(f, 0, 0.0, 0.0, 0.0, acc, None, None, M, dev, dt_)
            return {self.at: acc}
        a_n = k * depth
        # THE FACE'S OWN VELOCITY at the contact point: the two real vertices by their weights, and
        # the virtual centroid by the mean of the face's vertices. Friction against the GRID's
        # velocity instead would be friction against whatever else is in the cell, which is the weld
        # this scheme exists to avoid.
        ef = M["ef"][tri]
        vface = torch.zeros(M["nF"], 3, device=dev, dtype=dt_)
        vface.index_add_(0, M["ef"], M["Vv"][M["es"]])
        vface = vface / M["cnt"][:, None].to(dt_)
        v_surf = (w[:, 0:1] * vface[ef] + w[:, 1:2] * M["Vv"][M["es"][tri]]
                  + w[:, 2:3] * M["Vv"][M["et"][tri]])
        v_p = lvl.get("vel")[idx] if "vel" in lvl.state_schema else torch.zeros_like(v_surf)
        dv = v_p - v_surf
        dv_t = dv - (dv * n_hat).sum(1)[:, None] * n_hat
        speed = dv_t.norm(dim=1)
        # REGULARISED COULOMB: saturates at mu*a_n, linear below `eps_v`, so a resting contact does
        # not chatter. `eps_v` is a slip velocity and not a fudge factor.
        a_t = torch.minimum(self.mu * a_n, self.mu * a_n * speed / self.eps_v)
        a_par = a_n[:, None] * n_hat - a_t[:, None] * dv_t / speed.clamp_min(1e-12)[:, None]
        # THE CLAMP IS HERE, not in the scatter, so what is recorded as the reaction is what acts.
        mag = a_par.norm(dim=1)
        a_par = a_par * (self.a_max / mag.clamp_min(1e-20)).clamp(max=1.0)[:, None]
        acc.index_add_(0, idx, a_par)

        # THE REACTION. Mass is uniform across the matrix, so force and acceleration differ by one
        # constant and the residual below is the same number either way; it is written in force
        # units because that is what a vertex would receive.
        m = float(getattr(lvl, "mass", torch.ones(1, device=dev)).reshape(-1)[0])
        if self.verbose and not getattr(self, "_m_printed", False):
            print(f"[mesh_contact] particle mass {m:.4g}; penalty k = {k:.4g} in acceleration "
                  f"units = ({self.k_frac}/dt_sub)^2 against the explicit ceiling "
                  f"1/dt_sub^2 = {1.0 / dt_sub ** 2:.4g}; own clamp at a = {self.a_max:.4g}",
                  flush=True)
            self._m_printed = True
        fvert = torch.zeros(M["nv"], 3, device=dev, dtype=dt_)
        fvert.index_add_(0, M["es"][tri], (-m * w[:, 1:2]) * a_par)
        fvert.index_add_(0, M["et"][tri], (-m * w[:, 2:3]) * a_par)
        # the virtual centroid's share, handed to the face's own vertices in equal parts
        fface = torch.zeros(M["nF"], 3, device=dev, dtype=dt_)
        fface.index_add_(0, ef, (-m * w[:, 0:1]) * a_par)
        fvert.index_add_(0, M["es"], fface[M["ef"]] / M["cnt"][M["ef"], None].to(dt_))

        self._record(f, int(idx.numel()), float(depth.max()), float(speed.mean()),
                     float(a_n.max()), acc * m, fvert, (idx, tri, depth, a_par, m), M, dev, dt_)
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc * lvl.occ[:, None].float()}

    # ---- what it writes down --------------------------------------------------------------
    def _record(self, f, n_con, dmax, slip, a_max_seen, fpart, fvert, detail, M, dev, dt_):
        """ONE ROW PER FRAME, AGGREGATED OVER THE FRAME'S OWN SUBSTEPS -- not the first substep's
        values wearing the frame's name.

        The first version kept whichever substep had the most contacts, which reported the frame's
        pressure map from a single instant and its residual from another; the peak quantities
        (contacts, penetration, residual) are maxima over the substeps and the continuous ones
        (slip, pressure) are means, so each is aggregated the way its own question asks.
        """
        resid = 0.0
        if fvert is not None:
            resid = float((fpart.sum(0) + fvert.sum(0)).norm()
                          / fpart.norm(dim=1).sum().clamp_min(1e-30))
        if self._newframe:
            CONTACT_HISTORY.append(dict(frame=f, n_contact=0, depth_max=0.0, slip=0.0, a_max=0.0,
                                        momentum_residual=0.0, n_tri=M["n_tri"],
                                        nrow=M["G"]["nrow"], K=M["K"], n_sub=0))
            PRESSURE_MAP.append(np.zeros((self.map_theta, self.map_phi), np.float32))
            VERTEX_FORCE.append(None)
            self._newframe = False
            if self.verbose and (f < 2 or f % 50 == 0):
                print(f"[mesh_contact] frame {f}: {M['n_tri']} triangles, {M['G']['nbin']} bins in "
                      f"{M['G']['nrow']} rows (max {M['K']} per bucket), {n_con} contacts, "
                      f"penetration {dmax / (1.0 / self.n_grid):.2f} cells, momentum residual "
                      f"{resid:.2e}", flush=True)
        row = CONTACT_HISTORY[-1]
        row["n_sub"] += 1
        row["n_contact"] = max(row["n_contact"], n_con)
        row["depth_max"] = max(row["depth_max"], dmax)
        row["a_max"] = max(row["a_max"], a_max_seen)
        row["momentum_residual"] = max(row["momentum_residual"], resid)
        row["slip"] += (slip - row["slip"]) / row["n_sub"]
        PRESSURE_MAP[-1] += (self._pressure(detail, M, dev, dt_) - PRESSURE_MAP[-1]) / row["n_sub"]
        if fvert is not None:
            VERTEX_FORCE[-1] = fvert.detach().to("cpu", torch.float32).numpy()

    def _pressure(self, detail, M, dev, dt_):
        """The reaction as a pressure by direction, on the same (theta, phi) grid `apical_map` uses.

        Binned by direction and divided by the area each bin covers -- R^2 dOmega, with dOmega
        exact per row rather than sin(theta) d(theta), which diverges from it precisely at the poles
        where the bins are slivers and the error would read as a polar pressure.
        """
        nth, nph = self.map_theta, self.map_phi
        P = torch.zeros(nth * nph, device=dev, dtype=dt_)
        Rm = torch.zeros(nth * nph, device=dev, dtype=dt_)
        cntm = torch.zeros(nth * nph, device=dev, dtype=dt_)
        if self._dom is None:
            e = torch.linspace(0, math.pi, nth + 1, device=dev, dtype=dt_)
            self._dom = (e[:-1].cos() - e[1:].cos()) * (2 * math.pi / nph)
        # the surface radius per bin, from the mesh's own vertices
        Vv = M["V"]
        rv = Vv.norm(dim=1).clamp_min(1e-12)
        uv = Vv / rv[:, None]
        thv = torch.acos(uv[:, 2].clamp(-1, 1)); phv = torch.atan2(uv[:, 1], uv[:, 0]) % (2 * math.pi)
        bv = ((thv / math.pi * nth).long().clamp(0, nth - 1) * nph
              + (phv / (2 * math.pi) * nph).long().clamp(0, nph - 1))
        Rm.index_add_(0, bv, rv); cntm.index_add_(0, bv, torch.ones_like(rv))
        Rm = Rm / cntm.clamp_min(1.0)
        if detail is not None:
            idx, tri, depth, a_par, m = detail
            xs = M["A"][tri]
            r_ = xs.norm(dim=1).clamp_min(1e-12); u_ = xs / r_[:, None]
            th = torch.acos(u_[:, 2].clamp(-1, 1)); ph = torch.atan2(u_[:, 1], u_[:, 0]) % (2 * math.pi)
            b = ((th / math.pi * nth).long().clamp(0, nth - 1) * nph
                 + (ph / (2 * math.pi) * nph).long().clamp(0, nph - 1))
            P.index_add_(0, b, m * a_par.norm(dim=1))
        area = (Rm.reshape(nth, nph) ** 2) * self._dom[:, None]
        return (P.reshape(nth, nph) / area.clamp_min(1e-12)).detach().to("cpu",
                                                                        torch.float32).numpy()


@register_operator("mesh_inside", family="hierarchy", set="particle", kind="lateral")
class MeshInsideCount(Lateral):
    """How many matrix particles are behind the surface, and how deep -- measured, not fixed.

    `cell_exclude` answers the same question by PROJECTING the offenders out, which makes the
    count unmeasurable by construction: with the backstop on, the answer is always zero and the
    contact is never tested. This operator only counts, so "the contact holds the matrix out on its
    own" is a number that can come back wrong.

    It shares `mesh_contact`'s bin structure through the same module and runs once per FRAME, at
    frame level, because a per-substep count would report the transient depth inside a substep
    rather than what the frame ended with.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["diagnostic", "non_penetration"]
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.n_grid = int(params.get("n_grid", 64))

    def forward(self, H, mask=None):
        op = _LIVE.get("contact")
        if op is None or op._built is None:
            return {}
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        c = torch.tensor(op.centre, device=dev, dtype=dt_)
        d = pos - c
        r = d.norm(dim=1).clamp_min(1e-12)
        u = d / r[:, None]
        M = op._built
        _, _, b = _bins_of(u, M["G"])
        near = r < M["rmax"][b] + 1e-9
        n_in, dmax = 0, 0.0
        if bool(near.any()):
            idx = torch.nonzero(near).squeeze(1)
            hit, tri, t, w = op._query(M, pos[idx], u[idx], r[idx])
            deep = hit & (t > r[idx])
            n_in = int(deep.sum())
            if n_in:
                dmax = float((t[deep] - r[idx][deep]).max())
        _f = getattr(H, "frame", None)
        INSIDE_HISTORY.append(dict(frame=-1 if _f is None else int(_f), n_inside=n_in,
                                   depth_max_cells=dmax * self.n_grid))
        return {}


def reset():
    """Clear every module-level record. Called by the run script before the engine starts, because
    a second run in the same process would otherwise append to the first one's history."""
    for L in (CONTACT_HISTORY, VERTEX_FORCE, PRESSURE_MAP, INSIDE_HISTORY):
        L.clear()
    _LIVE.clear()


# --------------------------------------------------------------------------- the self-test
def selftest(tissue, scale=0.00853, dev="cuda:0", n=40000, n_brute=400, meshes=(0, 100, 199)):
    """THE LOOKUP IS THE ONE THING HERE THAT FAILS SILENTLY, so it is certified against brute force
    before any run is trusted to it.

    A miss is not an error and not a NaN: the particle simply feels no contact, ends up inside the
    tissue, and the movie shows matrix in the lumen -- which is exactly the artefact
    `cell_exclude` was written to sweep up rather than to explain. Two questions, both of which
    can come back wrong:

      COVERAGE  a ray from the centroid of a CLOSED star-shaped surface must hit it. The fraction of
                random directions that find a face is therefore 1, and anything less is the bin
                structure losing faces -- which is what a plain (theta, phi) lattice does near the
                poles.
      AGREEMENT for a subsample, the face and the radius the bins return must be the ones a test
                against EVERY triangle returns. Coverage alone would pass on a lookup that
                confidently returns the wrong face.
    """
    z = np.load(tissue)
    op = MeshContact({"tissue": tissue, "scale": scale, "verbose": False}, device=dev)
    g = torch.Generator(device="cpu").manual_seed(0)
    ok_all = True
    for j in meshes:
        j = min(j, op.n_mesh - 1)
        M = op._build(j, torch.device(dev), torch.float32)
        u = torch.randn(n, 3, generator=g).to(dev)
        u = u / u.norm(dim=1, keepdim=True)
        r = torch.full((n,), 1e-6, device=dev)
        x = u * r[:, None]
        hit, tri, t, w = op._query(M, x, u, r)
        cov = float(hit.float().mean())
        # brute force: every triangle, for a subsample
        s = torch.randperm(n, generator=g)[:n_brute].to(dev)
        A, B, C = M["A"], M["B"], M["C"]
        e1, e2 = B - A, C - A
        us = u[s][:, None, :]
        p = torch.cross(us.expand(-1, A.shape[0], -1), e2[None], dim=2)
        det = (e1[None] * p).sum(2)
        inv = 1.0 / torch.where(det.abs() < 1e-20, torch.full_like(det, 1e-20), det)
        sv = -A[None]
        w1 = (sv * p).sum(2) * inv
        q = torch.cross(sv.expand(n_brute, -1, -1), e1[None].expand(n_brute, -1, -1), dim=2)
        w2 = (us * q).sum(2) * inv
        tb = (e2[None] * q).sum(2) * inv
        good = (det.abs() > 1e-20) & (w1 >= -1e-6) & (w2 >= -1e-6) & (w1 + w2 <= 1 + 1e-6) & (tb > 0)
        tb = torch.where(good, tb, torch.full_like(tb, -1.0))
        t_ref = tb.max(dim=1).values
        d = (t[s] - t_ref).abs() / t_ref.clamp_min(1e-9)
        bad = int((d > 1e-4).sum())
        print(f"[selftest] mesh {j:3d}: {M['n_tri']:6d} triangles, {M['G']['nbin']:6d} bins in "
              f"{M['G']['nrow']:3d} rows, max {M['K']:3d} per bucket | coverage {cov:.5f} | "
              f"brute-force disagreement {bad}/{n_brute} (max relative {float(d.max()):.2e})",
              flush=True)
        ok_all = ok_all and cov > 0.9999 and bad == 0
    print(f"[selftest] {'PASS' if ok_all else 'FAIL'}", flush=True)
    return ok_all


if __name__ == "__main__":
    import sys as _sys
    _t = (_sys.argv[_sys.argv.index("--tissue") + 1] if "--tissue" in _sys.argv else
          os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
              os.path.abspath(__file__)))), "log", "okuda_ECM", "_tissue",
              "cellfix_B_new_f401_x4_c4a5698982.npz"))
    _d = _sys.argv[_sys.argv.index("--device") + 1] if "--device" in _sys.argv else "cuda:0"
    raise SystemExit(0 if selftest(_t, dev=_d) else 1)
