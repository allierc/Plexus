"""Where a triangulated surface meets a continuum, and what each tells the other.

    mesh_contact      the vertex mesh pushes MPM particles out of itself, and feels the reaction
    mesh_inside       which particles are inside the closed surface -- the test the contact needs
    surface_track     the surface's own moving frame, kept across division and death
    plate_confine     a rigid half-space (a projection; `block_seed` is the material version)
    bm_sense          the epithelium reads the membrane it is resting on
    ecm_load          the load the matrix puts back on the tissue
    ecm_gate_growth   and what that load does to growth -- entry condition `'mg_scale' in m`

WHY NOT GRID-BASED CONTACT. CFEMP (Lian et al. 2011, CMAME 200:3482) resolves contact by comparing
the two bodies' velocities at shared grid nodes, and needs mesh and grid to be comparable in size.
Ours are not: a cell is 0.73 dx and the basement membrane 0.1 dx, so both bodies live inside one
grid cell and the grid hands them ONE velocity -- the weld that `test_03_mesh_contact` measured.
"""
from __future__ import annotations
import math
import os
import numpy as np
import torch
from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.models.base import Structural
from plexus.models.registry import register_entity, register_operator


# ==========================================================================================================
# FROM `discovery_okuda/ops/mesh_contact_ops.py` -- mesh_contact_ops -- the 03 interface, generalised from a flat patch to the spheroid's own mesh.
# ==========================================================================================================
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


@register_operator("mesh_contact", family="boundary", set="particle", kind="lateral")
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
        # BY THE FACE'S OWN RADIUS, NOT ITS NEAREST CORNER'S. `rmin` is the minimum corner radius, so
        # one vertex near the centroid sends a face's apparent angular size to infinity -- and a
        # budded tissue has them: measured on 08b_s2_big's final mesh, rmin reaches 0.14 against a
        # median radius of 14.7, which put the 99.9th percentile of `ang` at 6.19 RADIANS (355 deg)
        # and collapsed `nrow` to its floor of 4. That is ~32 bins for 184,974 triangles, ~5,800 to a
        # bin, which destroys this structure's own premise ("a triangle spans at most one bin") and
        # makes the query allocate [n, 9K] with K ~ 5,800. On a sphere rmin ~ rmed so it never showed:
        # on the reference tissue both forms give 0.0586 rad and the same 53 rows.
        rmed = torch.maximum(torch.minimum(torch.maximum(rA, rB), torch.maximum(rB, rC)),
                             torch.minimum(rA, rC)).clamp_min(1e-20)     # the MEDIAN corner radius
        ang = (torch.maximum((B - A).norm(dim=1),
                             torch.maximum((C - A).norm(dim=1), (C - B).norm(dim=1))) / rmed)
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


# ==========================================================================================================
# FROM `discovery_okuda/ops/bm_sense_ops.py` -- #!/usr/bin/env python
# ==========================================================================================================
SENSE_TRACE: list = []


@register_operator("bm_sense", family="signalling", set="vertex", kind="structural")
class BMSense3D(Structural):
    """Write the basement-membrane deficit under each cell into `cell.chem[:, chan]`."""

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["map"]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["mechanosensing", "integrin_signalling", "anchorage_dependence",
                      "matrix_to_cell_feedback", "morphogen_source"]
    PARAM_ROLES = {"p_ref": "membrane_reference_level", "sharp": "deficit_sharpness",
                   "chan": "chem_channel_written"}
    REFERENCE = ("Streuli, C. H. (2009) Curr. Opin. Cell Biol. 21:194 (anchorage and the cycle); "
                 "Frantz, C. et al. (2010) J. Cell Sci. 123:4195 (the ECM as a signal).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.cat = params.get("cell_set", "cell")
        z = np.load(str(params["map"]))
        self.P = torch.as_tensor(np.asarray(z["pmap"], np.float32))
        self.T = int(self.P.shape[0])
        self.p_ref = float(params.get("p_ref", 1.0))
        self.sharp = float(params.get("sharp", 1.0))
        self.chan = int(params.get("chan", 0))
        self._frame, self._t, self._said = -1, 0, False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        clvl = H.level(self.cat)
        if m is None or clvl is None or "chem" not in getattr(clvl, "state_schema", {}):
            return {}
        nF = int(m["nF"])
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        f = int(getattr(H, "frame", -1) or -1)
        if f != self._frame:
            self._frame = f
            self._t = min(self.T - 1, max(0, f))

        # PER-CELL DIRECTION, centroid-referenced -- the same construction the map was binned with,
        # and the same one `ecm_gate_growth` uses, so the two cannot drift apart.
        es, ef = m["E_srce"], m["E_face"]
        live = ef < nF
        e_s, e_f = es[live].long(), ef[live].long()
        cnt = torch.zeros(nF, device=dev, dtype=dt_).index_add_(
            0, e_f, torch.ones_like(e_f, dtype=dt_))
        cen = torch.zeros(nF, 3, device=dev, dtype=dt_).index_add_(0, e_f, pos[e_s].to(dt_))
        ok = cnt > 0
        cen[ok] /= cnt[ok, None]
        origin = cen[ok].mean(0) if ok.any() else torch.zeros(3, device=dev, dtype=dt_)
        d = cen - origin
        u = d / d.norm(dim=1).clamp_min(1e-9)[:, None]
        M = self.P[self._t].to(dev, dt_)
        nth, nph = M.shape
        th = torch.acos(u[:, 2].clamp(-1, 1))
        ph = torch.atan2(u[:, 1], u[:, 0]) % (2 * math.pi)
        lig = M[(th / math.pi * nth).long().clamp(0, nth - 1),
                (ph / (2 * math.pi) * nph).long().clamp(0, nph - 1)]
        # A CELL WITH NO LIVE HALF-EDGE HAS NO DIRECTION, so it is given no deficit rather than the
        # deficit of whatever direction the origin happens to point in.
        def_ = (1.0 - lig / max(self.p_ref, 1e-12)).clamp(0.0, 1.0) ** self.sharp
        def_ = torch.where(ok, def_, torch.zeros_like(def_))

        ci, _ = clvl.state_schema["chem"]
        clvl.state[:nF, ci + self.chan] = def_.to(clvl.state.dtype)
        SENSE_TRACE.append((int((def_ > 0.5).sum()), float(def_.max()), float(def_.mean())))
        if not self._said:
            print(f"[bm_sense] writing the membrane deficit into {self.cat}.chem[:, {self.chan}] "
                  f"({self.T} frames, p_ref {self.p_ref}, sharp {self.sharp}); frame {f}: "
                  f"{int((def_ > 0.5).sum())} of {nF} cells above 0.5, max {float(def_.max()):.3f}",
                  flush=True)
            self._said = True
        return {}


# ==========================================================================================================
# FROM `discovery_okuda/ops/plate_ops.py` -- plate_ops -- two rigid solid blocks, top and bottom, that the growing tissue cannot get past.
# ==========================================================================================================
PLATE_CONTACT: list = []


@register_operator("plate_confine", family="boundary", set="vertex", kind="structural")
class PlateConfine3D(Structural):
    """Confine a set between two rigid plates normal to `axis`, at `centre` +/- `gap_half`."""

    EMIT = None                        # moves positions in place; no integrable delta
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["gap_half"]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["rigid_confinement", "anisotropic_boundary", "solid_obstacle"]
    PARAM_ROLES = {"gap_half": "free_half_gap", "stiff": "projection_fraction",
                   "axis": "confined_axis", "centre": "gap_centre_on_axis"}
    REFERENCE = "Plexus (this work); the confinement geometry of Okuda, S. et al. (2018) Sci. Rep. 8:2386."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.axis = int(params.get("axis", 2))
        self.centre = float(params.get("centre", 0.0))
        self.gap_half = float(params["gap_half"])
        self.stiff = float(params.get("stiff", 0.6))
        self.damp_normal = bool(params.get("damp_normal", True))
        self._said = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        ax = self.axis
        # LIVE ENTITIES ONLY, when the level knows how many it has. A vertex buffer is mostly empty
        # slots; projecting them is harmless but it makes the contact COUNT meaningless, and that
        # count is the only evidence of when confinement began.
        m = getattr(lvl, "_mesh", None)
        n = int(m["Nv"]) if (m is not None and "Nv" in m) else pos.shape[0]

        z = pos[:n, ax] - self.centre
        over = z.abs() - self.gap_half
        hit = over > 0
        n_hit = int(hit.sum())
        if n_hit:
            pos[:n, ax] = torch.where(
                hit, pos[:n, ax] - self.stiff * torch.sign(z) * over.clamp_min(0.0), pos[:n, ax])
            # `lvl.state` IS THE TENSOR; `lvl.state_schema` is what knows the block names. Probing
            # the tensor calls Tensor.__contains__ with a string, which raises -- so the vertex set,
            # which has no `vel` block at all, took down the whole run at frame 1.
            if self.damp_normal and "vel" in lvl.state_schema:
                v = lvl.get("vel")
                # KILL ONLY THE INTO-PLATE COMPONENT. Zeroing the whole velocity would apply a
                # friction the plate does not have, and a matrix particle sliding ALONG a plate is
                # exactly what the free directions are for.
                vn = v[:n, ax]
                v[:n, ax] = torch.where(hit & (vn * torch.sign(z) > 0),
                                        torch.zeros_like(vn), vn)
        PLATE_CONTACT.append((n_hit, float(over[hit].max()) if n_hit else 0.0))
        if not self._said:
            print(f"[plate_confine] {self.at}: rigid plates at {self.centre:+.4g} "
                  f"+/- {self.gap_half:.4g} along axis {ax}, stiff={self.stiff}; "
                  f"{n_hit} of {n} in contact at frame 0", flush=True)
            self._said = True
        return {}


def block_fraction(gap_half, half_extent):
    """Fraction of the domain's volume the two blocks occupy, for a box of half-width `half_extent`.

    Reported rather than specified, because the two things a caller wants to control -- how squashed
    the tissue is, and how much of the box is solid -- are the SAME number seen twice, and only one of
    them can be set. See `PLATES.md` for the arithmetic that pins them together.
    """
    free = min(1.0, max(0.0, gap_half / half_extent))
    return 1.0 - free


# ==========================================================================================================
# FROM `discovery_okuda/ops/surface_ops.py` -- surface -- the epithelial surface as a LEVEL, instead of a lookup table.
# ==========================================================================================================
@register_entity("surface")            # BY SET NAME. Registering this as "surface_element"
class SurfaceElement:  # while the set is called "surface" means the provision never runs, and the
    # operator dies on a missing `u` buffer -- the same trap the membrane set fell into.
    """One patch of the epithelial surface: a direction, a radius, and the velocity of that radius.

    Registered because entities resolve BY SET NAME -- an unregistered name silently falls back to a bare
    pos/vel schema, which for this set is actually all that is needed, but relying on a fallback is how
    the membrane set died inside `mpm_strain` with a missing attribute.
    """
    @staticmethod
    def provision(lvl, parent, s, H, device):
        n = lvl.get("pos").shape[0]
        lvl.register_buffer("u", torch.zeros(n, 3, device=device))
        lvl.register_buffer("R", torch.zeros(n, device=device))


@register_operator("surface_track", family="hierarchy", set="particle", kind="structural")
class SurfaceTrack(Structural):
    """Write the epithelial surface into the `surface` Level each frame, WITHOUT binning.

    The old lookup took `R(u, t)` from the cell of a 32x64 table that `u` fell into -- a nearest-bin
    interpolation, i.e. the crudest one available, on a field that is smooth. Here each element's radius
    is a distance-weighted average over the `k` nearest directions of the recorded map, which is
    continuous in `u` and has no cell edges for a strain field to remember.

    The map is still the pass-1 recording; what changes is how it is read.
    """
    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["surface"]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["prescribed_boundary", "replay", "smooth_interpolation"]
    PARAM_ROLES = {"k": "interpolation_neighbours", "scale": "surface_rescale",
                   "seed": "lattice_seed", "jitter": "lattice_disorder"}
    REFERENCE = "Plexus (this work); the surface is Okuda, S. et al. (2018) Sci. Rep. 8:2386."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        import numpy as _np
        self.at = params.get("_at", "surface")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        self.scale = float(params.get("scale", 1.0))
        self.k = int(params.get("k", 6))
        self.jitter = float(params.get("jitter", 0.35))
        self.seed = int(params.get("seed", 0))
        z = _np.load(str(params["surface"]))
        self.smap = torch.as_tensor(z["smap"], dtype=torch.float32) * self.scale
        self.T = int(self.smap.shape[0])
        self._u = None
        self._mu = None            # map directions, built once
        self._prevR = None
        self._frame = -1

    def _lattice(self, n, dev, dt_):
        g = torch.Generator().manual_seed(self.seed)
        i = torch.arange(n, dtype=torch.float64) + 0.5
        ct = 1.0 - 2.0 * i / n
        st = torch.sqrt((1.0 - ct * ct).clamp_min(0.0))
        phi = (math.pi * (1.0 + 5.0 ** 0.5) * i) % (2 * math.pi)
        u = torch.stack([st * torch.cos(phi), st * torch.sin(phi), ct], 1).to(torch.float32)
        if self.jitter > 0:
            sp = math.sqrt(4.0 * math.pi / max(n, 1))
            e1 = torch.stack([-u[:, 1], u[:, 0], torch.zeros_like(u[:, 0])], 1)
            nr = e1.norm(dim=1, keepdim=True)
            e1 = torch.where(nr > 1e-6, e1 / nr.clamp_min(1e-12),
                             torch.tensor([[1.0, 0.0, 0.0]]).expand_as(e1))
            e2 = torch.cross(u, e1, dim=1)
            u = u + e1 * (torch.randn(n, generator=g) * (self.jitter * sp))[:, None] \
                  + e2 * (torch.randn(n, generator=g) * (self.jitter * sp))[:, None]
            u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return u.to(dev, dt_)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        n = pos.shape[0]
        if self._u is None:
            # THE MEMBRANE'S OWN DIRECTIONS, handed over by `bm_seed`, so element i and
            # particle i share a direction exactly rather than by reproducing an RNG sequence. Falls back
            # to rebuilding the lattice only when there is no membrane in the run.
            # THIS LEVEL OWNS THE LATTICE. `bm_seed` reads it back rather than
            # rebuilding it, so element i and particle i share a direction because they are the same
            # array -- not because two functions happen to draw from their generators in the same order.
            self._u = self._lattice(n, dev, dt_)
            lvl.u[:] = self._u
            nth, nph = self.smap.shape[1], self.smap.shape[2]
            th = (torch.arange(nth, dtype=torch.float32) + 0.5) / nth * math.pi
            ph = (torch.arange(nph, dtype=torch.float32) + 0.5) / nph * 2 * math.pi
            T2, P2 = torch.meshgrid(th, ph, indexing="ij")
            self._mu = torch.stack([torch.sin(T2) * torch.cos(P2),
                                    torch.sin(T2) * torch.sin(P2),
                                    torch.cos(T2)], -1).reshape(-1, 3).to(dev, dt_)
            # nearest map directions, once: the lattice is fixed, so the stencil is too
            cs = self._u @ self._mu.T
            self._nb = torch.topk(cs, self.k, dim=1).indices
            w = (1.0 - torch.gather(cs, 1, self._nb)).clamp_min(1e-6)
            self._w = (1.0 / w) / (1.0 / w).sum(1, keepdim=True)

        f = int(getattr(H, "frame", -1) or -1)
        if f != self._frame:
            self._frame = f
        t = min(self.T - 1, max(0, f))
        M = self.smap[t].to(dev, dt_).reshape(-1)
        R = (M[self._nb] * self._w).sum(1)
        c = torch.tensor(self.centre, device=dev, dtype=dt_)
        newpos = c + self._u * R[:, None]
        if "vel" in lvl.state_schema:
            lvl.get("vel")[:] = (newpos - pos)          # per frame, which is this Level's time unit
        pos[:] = newpos
        lvl.R[:] = R
        H.surface_level = self.at
        return {}


# ==========================================================================================================
# FROM `discovery_okuda/ops/load_ops.py` -- load_ops -- the other half of the coupling: the matrix pushing back on the cells.
# ==========================================================================================================
LOAD_TRACE: list = []


@register_operator("ecm_load", family="mechanics", set="vertex", kind="structural")
class ECMLoad3D(Structural):
    """Push the vertex mesh inward with a recorded matrix pressure map P(theta, phi, t)."""

    EMIT = None                        # moves positions in place; no integrable delta
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["load"]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["matrix_to_cell_feedback", "mechanical_resistance", "partitioned_coupling"]
    PARAM_ROLES = {"gain": "load_coupling_gain", "mu": "vertex_mobility",
                   "dt": "frame_timestep", "cap_frac": "max_step_as_radius_fraction"}
    REFERENCE = "Plexus (this work); the reaction to Okuda, S. et al. (2018) Sci. Rep. 8:2386 contact."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        import numpy as _np
        self.at = params.get("_at", "vertex")
        z = _np.load(str(params["load"]))
        P = _np.asarray(z["pmap"], _np.float32)
        # NORMALISED BY THE p99 OF THE NONZERO MAP -- same reason and same measurement as
        # `ecm_gate_growth` below: the peak is a single bin in a single frame, an order of magnitude
        # above anything typical, so normalising by it silently scales the coupling to nothing.
        nz = P[P > 0]
        self.pk = max(float(_np.percentile(nz, 99)) if nz.size else 1.0, 1e-12)
        self.P = torch.as_tensor(P / self.pk, dtype=torch.float32)
        self.T = int(self.P.shape[0])
        self.gain = float(params.get("gain", 1.0))
        self.mu = float(params.get("mu", 1.0))
        self.dt = float(params.get("dt", 1.0))
        self.cap_frac = float(params.get("cap_frac", 0.04))
        self._frame, self._t, self._said = -1, 0, False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        m = getattr(lvl, "_mesh", None)
        n = int(m["Nv"]) if (m is not None and "Nv" in m) else pos.shape[0]
        f = int(getattr(H, "frame", -1) or -1)
        if f != self._frame:
            self._frame = f
            self._t = min(self.T - 1, max(0, f))
        M = self.P[self._t].to(dev, dt_)
        nth, nph = M.shape

        # CENTROID-REFERENCED, because the recorded map is: `tissue.apical_map` subtracts the vertex
        # centroid before binning, and the vesicle drifts. Binning against the world origin instead
        # would rotate the load off the surface it was measured on, a little more every frame.
        p = pos[:n]
        c = p.mean(0)
        d = p - c
        r = d.norm(dim=1).clamp_min(1e-9)
        u = d / r[:, None]
        th = torch.acos(u[:, 2].clamp(-1, 1))
        ph = torch.atan2(u[:, 1], u[:, 0]) % (2 * math.pi)
        it = (th / math.pi * nth).long().clamp(0, nth - 1)
        ip = (ph / (2 * math.pi) * nph).long().clamp(0, nph - 1)
        press = M[it, ip]

        step = (self.gain * press / max(self.mu, 1e-12)) * self.dt
        step = torch.minimum(step, self.cap_frac * r)          # never more than a slice of the radius
        pos[:n] = p - step[:, None] * u                        # inward, along the surface normal
        nz = int((step > 0).sum())
        LOAD_TRACE.append((nz, float(step.max()) if n else 0.0, float(press.mean())))
        if not self._said:
            print(f"[ecm_load] {self.at}: recorded matrix load, {self.T} frames, peak pressure "
                  f"{self.pk:.4g} (normalised to 1); gain={self.gain}, cap={self.cap_frac} of r; "
                  f"{nz} of {n} vertices loaded at frame 0", flush=True)
            self._said = True
        return {}


@register_operator("ecm_gate_growth", family="population", set="vertex", kind="structural")
class ECMGrowthGate3D(Structural):
    """The matrix's stress slows the CELL CYCLE where it presses hardest.

    THE MECHANISM, AND WHY IT IS STRONGER THAN A FORCE. `ecm_load` pushes the vertices inward: a
    mechanical correction that fights the growth every frame and is bounded by how hard you dare push
    before cells invert. This operator instead gates the RATE -- a cell facing a stressed matrix grows
    its target volume more slowly, so it reaches `cell_divide`'s volume-doubling threshold later and
    DIVIDES LESS OFTEN. That difference integrates over 400 frames, so a few percent of stress
    anisotropy becomes a visible shape anisotropy, which a force of the same size cannot do.

    It is also the biology: proliferation under mechanical load is suppressed, not just deformed
    (Helmlinger 1997; Montel 2011 measured spheroids stalling under external pressure). The tissue is
    not being pushed into an ovoid -- it is GROWING into one, because the directions differ in how much
    the matrix objects.

    HOW IT INTERCEPTS WITHOUT REWRITING THE GROWTH OPERATOR. `cell_grow` keeps a per-cell
    cumulative scale `mg_scale` and multiplies it by (1 + rate.(rho + Hill(a))) each tick, then derives
    A0/P0/V0f/R0 from it. This operator runs AFTER it, remembers the scale it left behind last frame,
    and reads the factor growth just applied as `f = s_now / s_prev`. The gated scale is then
    `s_prev * (1 + gate.(f - 1))` -- exact, and it needs to know nothing about `rate`, `rho` or the Hill
    function, so it cannot drift out of step with them. Five lines of A0/P0/V0f/R0 bookkeeping are
    duplicated from that operator, which is the price of not editing a shared file; they are marked.

    THE GATE IS A CHOICE OF FUNCTIONAL FORM, and it is stated rather than buried: a Hill in the
    normalised pressure, `gate = floor + (1 - floor) / (1 + (P/p_half)^n)`. `floor` matters -- with
    floor 0 a cell in the most stressed direction stops dividing entirely and the tissue can only grow
    the other way, which produces a dramatic shape for a reason that is closer to a wall than to
    mechanosensing.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["load"]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["mechanosensitive_growth", "contact_inhibition",
                      "matrix_to_cell_feedback", "anisotropic_growth"]
    PARAM_ROLES = {"p_half": "half_suppression_pressure", "hill": "gate_sharpness",
                   "floor": "minimum_growth_fraction"}
    REFERENCE = ("Helmlinger, G. et al. (1997) Nat. Biotechnol. 15:778; "
                 "Montel, F. et al. (2011) Phys. Rev. Lett. 107:188102.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        import numpy as np
        self.at = params.get("_at", "vertex")
        z = np.load(str(params["load"]))
        P = np.asarray(z["pmap"], np.float32)

        # SMOOTHED, AND THIS WAS THE LAST THING IN THE WAY. The recorded map is a CONTACT LEDGER, not a
        # stress field: at frame 250 only 24% of its bins are nonzero, because 140,000 particles binned
        # into 2,048 directions against a moving surface touch a different quarter of it every frame. A
        # zero bin gates to 1.0 -- no suppression -- so three quarters of the polar cells were
        # unsuppressed in any given frame and a different three quarters the next, which time-averages to
        # a nearly UNIFORM weak suppression. That is why a 15x instantaneous directional contrast moved
        # the aspect ratio by 3%: the contrast was real and it was not persistent for any given cell.
        #
        # The patchiness is a sampling artefact of the estimator, not a feature of the field, so it is
        # averaged out: a running mean over `smooth_frames` and a box over `smooth_phi_deg` of longitude
        # and one row of colatitude. `smooth_phi_deg = 360` makes the map AXISYMMETRIC, which is the
        # right estimator when the matrix is (dense polar caps are), and is wrong the moment the
        # experiment puts structure in longitude -- so it is a parameter and not a default of the code.
        sf = int(params.get("smooth_frames", 25))
        sp = float(params.get("smooth_phi_deg", 360.0))
        if sf > 1 and P.shape[0] > sf:
            k = np.ones(sf, np.float32) / sf
            flat = P.reshape(P.shape[0], -1)
            P = np.stack([np.convolve(flat[:, j], k, mode="same") for j in range(flat.shape[1])],
                         axis=1).reshape(P.shape).astype(np.float32)
        if sp >= 359.0:
            P = np.repeat(P.mean(axis=2, keepdims=True), P.shape[2], axis=2)
        elif sp > 0:
            w = max(1, int(round(sp / 360.0 * P.shape[2])))
            ker = np.ones(w, np.float32) / w
            P = np.stack([[np.convolve(np.tile(P[t, i], 3), ker, mode="same")[P.shape[2]:2 * P.shape[2]]
                           for i in range(P.shape[1])] for t in range(P.shape[0])]).astype(np.float32)
        # one row of colatitude, clamped at the poles
        P = (P + np.pad(P, ((0, 0), (1, 0), (0, 0)), mode="edge")[:, :-1]
             + np.pad(P, ((0, 0), (0, 1), (0, 0)), mode="edge")[:, 1:]) / 3.0
        # NORMALISED so `p_half` means the same thing whatever the matrix's stiffness was -- otherwise
        # stiffening the matrix moves both the pressure and the gate's operating point and a stiffness
        # sweep measures two things at once.
        #
        # BY THE p99 OF THE NONZERO MAP, NOT BY THE PEAK, and that was measured the hard way. The peak
        # is ONE bin in ONE frame: on 49_aniso_i0_fibres it is 4.0e5 while the per-axis mean pressures
        # are 1448-2176, so peak-normalisation put the whole run at press ~0.005 and `p_half = 0.25` left
        # the gate at 1.000 -- an operator that ran 400 times and did nothing. Against the p99 (4.6e4)
        # the same run spans mean 0.007 -> 0.159 with the loaded directions reaching p90 0.45, which is a
        # range a Hill function can actually act on. Only bins that ever saw contact count: three
        # quarters of the map is identically zero, and including it would drag the scale toward the
        # tissue's own solid angle rather than the pressures it produced.
        nz = P[P > 0]
        self.pk = max(float(np.percentile(nz, 99)) if nz.size else 1.0, 1e-12)
        self.P = torch.as_tensor(P / self.pk, dtype=torch.float32)
        self.T = int(self.P.shape[0])
        # SELF-CALIBRATING OPERATING POINT. `p_half` in units of the p99 was a hand-picked number and it
        # put the gate on the wrong part of its own curve: a real map's typical pressures sit far below
        # its p99, which is set by a few hot contact bins. With dense polar caps giving poles 7234 and
        # equator 2631 -- a genuine 2.75x pattern -- both landed at 0.145 and 0.053 against p_half 0.10,
        # i.e. on the flat foot of the Hill, for a rate difference of only 1.7x where the clean synthetic
        # map gave 5x. `auto` puts the half-suppression point at the MEDIAN of the pressure the surface
        # actually carries late in the run, so half the loaded tissue sits either side of it and the
        # pattern falls across the steep part of the curve instead of under it.
        # `relative` IS A DIFFERENT MECHANISM AND IS LABELLED ONE. The absolute gate is the physical
        # reading -- a cell responds to the stress it carries -- and it has a timing problem that is the
        # MATRIX's, not the gate's: the measured pressure only passes the half-point around frame 350 of
        # 402, so a 15x directional rate difference arrives with 50 frames left to shape anything. In
        # `relative` mode the reference is the current frame's own mean over loaded directions, so the
        # gate reads the PATTERN and ignores the amplitude, and it acts from first contact. That is
        # adaptive mechanosensing (cells habituating to ambient stress and responding to the excess),
        # which is a real mechanism and a WEAKER claim than the absolute one -- it cannot be reported as
        # "the matrix's stress shaped the tissue" without saying which mode produced it.
        self.relative = str(params.get("p_half", "")).lower() == "relative"
        ph = params.get("p_half", "auto")
        if self.relative:
            self.p_half = float(params.get("rel_half", 1.0))
        elif isinstance(ph, str) and ph.lower() == "auto":
            late = P[int(0.75 * P.shape[0]):]
            lnz = late[late > 0]
            self.p_half = float(np.median(lnz) / self.pk) if lnz.size else 0.10
        else:
            self.p_half = float(ph)
        # SHARPER THAN 2, AND THAT IS A CLAIM ABOUT MECHANOSENSING, not a fitting knob. A linear-ish
        # gate cannot turn a 2.75x stress pattern into a shape: it needs to be switch-like, which is what
        # a Hill exponent of 4-6 is and what mechanotransduction actually looks like (a threshold, not a
        # proportionality). Stated here because the alternative reading -- "the exponent was raised until
        # the picture worked" -- is the one a reader should be able to rule out from the numbers above.
        self.hill = float(params.get("hill", 4.0))
        self.floor = float(params.get("floor", 0.15))
        self._prev = None
        self._frame, self._t, self._said = -1, 0, False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        if m is None or "mg_scale" not in m:
            return {}                      # growth has not run yet this run; nothing to gate
        nF = int(m["nF"])
        s_now = m["mg_scale"]
        dev, dt_ = s_now.device, s_now.dtype
        f = int(getattr(H, "frame", -1) or -1)
        if f != self._frame:
            self._frame = f
            self._t = min(self.T - 1, max(0, f))

        # PER-CELL DIRECTION -> PRESSURE BIN. Centroid-referenced, matching how the map was built.
        pos = lvl.get("pos")
        es, ef = m["E_srce"], m["E_face"]
        live = ef < nF
        e_s, e_f = es[live].long(), ef[live].long()
        cnt = torch.zeros(nF, device=dev, dtype=dt_).index_add_(
            0, e_f, torch.ones_like(e_f, dtype=dt_))
        cen = torch.zeros(nF, 3, device=dev, dtype=dt_).index_add_(0, e_f, pos[e_s].to(dt_))
        ok = cnt > 0
        cen[ok] /= cnt[ok, None]
        origin = cen[ok].mean(0) if ok.any() else torch.zeros(3, device=dev, dtype=dt_)
        d = cen - origin
        r = d.norm(dim=1).clamp_min(1e-9)
        u = d / r[:, None]
        M = self.P[self._t].to(dev, dt_)
        nth, nph = M.shape
        th = torch.acos(u[:, 2].clamp(-1, 1))
        ph = torch.atan2(u[:, 1], u[:, 0]) % (2 * math.pi)
        press = M[(th / math.pi * nth).long().clamp(0, nth - 1),
                  (ph / (2 * math.pi) * nph).long().clamp(0, nph - 1)]
        press = torch.where(ok, press, torch.zeros_like(press))
        ref = self.p_half
        if self.relative:
            # THE CURRENT FRAME'S OWN MEAN OVER LOADED DIRECTIONS. Zero bins are excluded: three
            # quarters of the map is untouched surface early on, and averaging that in would put the
            # reference far below anything real and saturate the gate everywhere.
            live = M[M > 0]
            if live.numel() < 8:
                return {}                      # nothing in contact yet: nothing to be relative to
            ref = float(live.mean()) * self.p_half
        gate = self.floor + (1.0 - self.floor) / (
            1.0 + (press / max(ref, 1e-12)).clamp_min(0.0) ** self.hill)

        # ---- GATE THE TARGET VOLUME, NOT `mg_scale` -------------------------------------------------
        # THE TRAP THIS AVOIDS, MEASURED. `cell_grow` reallocates `mg_scale` to ONES and
        # re-bases its A0_init/P0_init/V0f_init snapshots from the current values whenever `nF` changes
        # -- and with `cell_divide` firing every 4 frames, nF changes on most frames. A gate that reads
        # `s_now / s_prev` therefore saw a size change almost every frame, skipped its correction, and
        # ran 400 times without altering anything: 5,933 cells against 5,968 ungated and the same final
        # radius to two decimals. `V0f` is CONTINUOUS across that re-base (only its reference moves), so
        # gating its increment works where gating the scale cannot.
        V = m["V0f"]
        prev = self._prev
        if prev is None or prev.shape[0] != nF:
            p2 = V.detach().clone()
            if prev is not None:
                k = min(prev.shape[0], nF)
                p2[:k] = prev[:k]
            prev = p2
        dV = V - prev
        # A DIVISION IS NOT GROWTH. A daughter's target volume is set by `cell_divide`, not by an
        # increment, so its `dV` is a large jump in either direction; gating it would shrink or inflate a
        # cell that had just been created. Growth is ~0.3% per frame, so anything past 20% is a topology
        # event and is passed through untouched -- and re-based, so the next frame gates normally.
        topo = dV.abs() > 0.2 * prev.clamp_min(1e-12)
        Vg = torch.where(topo, V, prev + gate * dV)
        q = (Vg / V.clamp_min(1e-12)).clamp(0.2, 1.0)      # how much of the wanted growth was allowed
        m["V0f"] = V * q
        m["V0"] = float(m["V0f"].sum())
        # A0 ~ R^2 and P0 ~ R with R ~ V^(1/3), so one volume ratio sets all three consistently. Applied
        # to the CURRENT values rather than to the *_init snapshots, which is what makes this survive the
        # re-base above.
        m["A0"] = m["A0"] * q.pow(2.0 / 3.0)
        m["P0"] = m["P0"] * q.pow(1.0 / 3.0)
        m["mg_scale"] = s_now * q.pow(1.0 / 3.0)
        if "R0" in m:
            m["R0"] = float((3.0 * max(m["V0"], 1e-12) / (4.0 * math.pi)) ** (1.0 / 3.0))
        self._prev = m["V0f"].detach().clone()
        LOAD_TRACE.append((int((gate < 0.99).sum()), float(gate.min()), float(press.mean())))
        if not self._said:
            print(f"[ecm_gate_growth] recorded matrix load, {self.T} frames, p99 {self.pk:.4g}; "
                  f"p_half={self.p_half} hill={self.hill} floor={self.floor}; gate range "
                  f"[{float(gate.min()):.3f}, {float(gate.max()):.3f}] over {nF} cells at frame {f}",
                  flush=True)
            self._said = True
        return {}

