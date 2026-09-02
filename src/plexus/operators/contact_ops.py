"""Where a triangulated surface meets a continuum, and what each tells the other.

    seed_plate        an open planar half-edge patch -- sheet, disc, disc with a hole, or grid
    plate_drive       and its prescribed rigid descent, so the surface is a piston
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
    """Particle-to-surface contact against a LIVE triangulated surface.

    Penalty in the face normal, regularised Coulomb friction against the face's own velocity, and
    the reaction distributed to the face's vertices by the barycentric weights that built it.

    NO REPLAY. The surface is a vertex set in this hierarchy, read at the frame it is in --
    `surface: <set>`. The archive path (`tissue: <npz>`, a cache of 200 meshes with a `mesh_stride`
    and a linear interpolation between kept frames) IS DELETED, along with the interpolation, the
    stride arithmetic and the finite-differenced vertex velocity that went with it.

    WHY, AND WHAT IT COST TO LEARN. A replayed input is a claim about a FILE, not about a model.
    Gate 04 read a 32.7 MB npz built in August from a spec that no longer resolved -- its parent had
    drifted to `rate: 0.03` against the `0.003457` that made the cache -- so the gate was green
    against an artefact nobody could rebuild, and `tools/export_tissue.py` existed only to paper
    over that. The surface now comes from a `seed` operator like every other initial condition
    (`seed_plate`, `seed_mesh`), which means it is declared, regenerated on every run, and cannot
    silently disagree with the spec that names it.

    WHAT THIS RULES OUT, STATED RATHER THAN HIDDEN: two subsystems whose clocks differ by ~10^5 --
    an epithelium at 600 s a frame against a matrix at 3.2 ms -- cannot share a schedule, and the
    replay was how that was dodged. A surface on this path must therefore be PRESCRIBED (kinematics
    written by an operator, e.g. `plate_drive`) or solved on the matrix's own clock. That is a
    modelling constraint, not a regression.
    """

    EMIT = "mpm_acceleration"          # consumed by `mpm_scatter` as a_ext, like `ecm_from_cell`
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["surface"]      # a live vertex set; there is no archive path any more
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
        # THE STAR-SHAPE REFERENCE, AND IT MAY MOVE WITH THE SURFACE.
        #
        #   centre: [x, y, z]   a fixed point in the box, as before.
        #   centre: centroid    the LIVE surface's own centroid, re-read every frame.
        #
        # A fixed point is right for a surface that stays put and wrong for one that travels: the
        # forbidden region is the side of the surface the reference sits on, so a sphere driven
        # 0.15 of a box downward leaves its own reference behind and the ray cast starts pointing
        # at the wrong face. `centroid` also removes the drift the plate had -- its declared
        # standoff of 0.30 grew to 0.50 over the descent, which is a 90 -> 150 walk in the contact's
        # bin rows for a geometry that never changed.
        #
        # `centre_offset` is added to whichever, so a plate says `centre: centroid` with an offset
        # into the piston body and the standoff is then constant by construction.
        _c = params.get("centre", [0.5, 0.5, 0.5])
        self.centre_track = isinstance(_c, str) and str(_c).lower() in ("centroid", "surface")
        self.centre = [0.0, 0.0, 0.0] if self.centre_track else [float(v) for v in _c]
        self.centre_offset = [float(v) for v in params.get("centre_offset", [0.0, 0.0, 0.0])]
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
        self.verbose = bool(params.get("verbose", True))

        # THE SURFACE IS A SET, NOT A FILE. See the class docstring for why the archive path is gone.
        self.surface = str(params["surface"])
        if "tissue" in params:
            raise ValueError("mesh_contact: `tissue:` (the replayed mesh archive) is deleted. Seed "
                             "the surface as a vertex set -- `seed_plate` or `seed_mesh` -- and "
                             f"name it with `surface: <set>` instead of {params['tissue']!r}")
        if "mesh_stride" in params:
            raise ValueError("mesh_contact: `mesh_stride:` counted pass-2 frames per KEPT mesh in "
                             "the archive. A live surface has one mesh per frame and no cadence to "
                             "reconcile, so the parameter has no meaning; remove it")
        if self.scale != 1.0:
            # A LIVE SET IS ALREADY IN BOX COORDINATES. `scale` existed because the archive held the
            # tissue in its own units about its own centroid. Applied to a set the engine is already
            # integrating in box units it would move the surface away from the particles it is
            # meant to touch, and the run would report no contacts -- which reads as "nothing
            # happened" rather than as an error.
            raise ValueError(f"mesh_contact: `scale: {self.scale}` -- a live surface is already in "
                             f"box units, so the only meaningful scale is 1.0")
        self._frame = -2
        self._built = None
        self._dom = None
        self._newframe = False
        # THE COUNTER SHARES THIS OPERATOR'S BINS rather than rebuilding them: two bin structures
        # that disagree would have the diagnostic measuring a different surface from the one the
        # contact acted on, which is the one way a non-penetration count can be wrong and look right.
        _LIVE["contact"] = self

    # ---- the mesh of one frame ------------------------------------------------------------
    def _mesh_live(self, H, dev, dt_):
        """Vertices, half-edges and the surface's own velocity, read off the live vertex set.

        IN THE STAR FRAME, WHICH IS THE ONE CONVERSION THIS OWES. Everything below assumes the
        surface is given about the ray origin, and a live set is in box coordinates -- so `centre`
        is subtracted here and added back in `forward` when the hit point is reconstructed. Getting
        this wrong does not crash: it reports zero contacts, which reads as "the surface never
        touched anything".

        `Vv` comes off the mesh table because the surface KNOWS its velocity -- `plate_drive` writes
        it. The archive path had to recover it by differencing two cached meshes and dividing by a
        stride, which is one place a cadence could disagree with the positions it was handed.
        """
        lvl = H.level(self.surface)
        m = getattr(lvl, "_mesh", None)
        if m is None or not int(m.get("Nv", 0)):
            return None
        nv = int(m["Nv"])
        if self.centre_track:
            self.centre = (lvl.get("pos")[:nv].mean(0).detach().to("cpu").tolist())
            self.centre = [a + b for a, b in zip(self.centre, self.centre_offset)]
        c = torch.tensor(self.centre, device=dev, dtype=dt_)
        V = lvl.get("pos")[:nv].to(dt_) - c
        vv = m.get("Vv", None)
        Vv = torch.zeros_like(V) if vv is None else vv[:nv].to(device=dev, dtype=dt_)
        return V, m["E_srce"].to(dev), m["E_trgt"].to(dev), m["E_face"].to(dev), Vv

    def _build(self, dev, dt_, H):
        """Everything about the surface that changes once per frame: triangles, their bins, and the
        vertex velocity the friction law needs. Rebuilt every frame, because a live surface has
        moved by definition.
        """
        got = self._mesh_live(H, dev, dt_)
        return None if got is None else self._build_from(*got, dev=dev, dt_=dt_)

    def _build_from(self, V, es, et, ef, Vv, dev, dt_):
        """The geometry, given the four arrays -- split out so `selftest` can certify the lookup on
        a surface it builds in memory instead of one a run has to produce first."""
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

        # VERTEX VELOCITY comes off the surface itself (`_mesh_live`), because the operator that
        # moves it knows it exactly. Nothing is differenced here.
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
        # REBUILT ONCE PER FRAME AND NOT PER SUBSTEP. The surface's position is a frame-level fact --
        # `plate_drive` writes it once, before the substep block -- so rebuilding inside the block
        # would cost eight bin tables to describe one geometry. The build is a few milliseconds
        # against a frame's eight substeps.
        if f != self._frame:
            self._frame, self._newframe = f, True
            self._built = self._build(dev, dt_, H)
        elif self._built is None:
            # the seed ran between this frame's substeps; pick the surface up without opening a
            # second history row for a frame that already has one
            self._built = self._build(dev, dt_, H)
        M = self._built
        if M is None:
            # THE SEED HAS NOT RUN. `seed` operators are confined to the opening frames and the
            # contact may be scheduled inside a substep block that runs on the same frame, so an
            # empty surface is an ordering fact and not an error -- but a silent zero would make a
            # mis-ordered schedule look like a run in which nothing ever touched.
            if not getattr(self, "_said_empty", False):
                print(f"[mesh_contact] surface `{self.surface}` is empty at frame {f}; no contact "
                      f"until its seed has run", flush=True)
                self._said_empty = True
            return {self.at: torch.zeros_like(pos)}
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
def selftest(surface="sphere", dev="cuda:0", n=40000, n_brute=400, **kw):
    """THE LOOKUP IS THE ONE THING HERE THAT FAILS SILENTLY, so it is certified against brute force
    before any run is trusted to it.

    A miss is not an error and not a NaN: the particle simply feels no contact, ends up behind the
    surface, and the movie shows matrix where the solid is -- which is exactly the artefact
    `cell_exclude` was written to sweep up rather than to explain. Two questions, both of which can
    come back wrong:

      COVERAGE  a ray from inside a CLOSED star-shaped surface must hit it, so the fraction of
                random directions that find a face is 1 and anything less is the bin structure
                losing faces -- which is what a plain (theta, phi) lattice does near the poles. An
                OPEN surface (a plate) covers only its own solid angle, so the number is reported
                and not asserted; there the agreement test is the whole check.
      AGREEMENT for a subsample, the face and the radius the bins return must be the ones a test
                against EVERY triangle returns. Coverage alone would pass on a lookup that
                confidently returns the wrong face.

    THE SURFACE IS BUILT HERE, NOT LOADED. It used to take a path to a 32.7 MB mesh archive, so the
    one check that certifies the lookup could not run without an artefact somebody had generated --
    and the check would then be certifying the lookup against whatever that file happened to hold.

    RUN IT AS AN IMPORT, NOT AS `python -m`: this module registers its operators at import, and
    `-m` executes it a SECOND time under the name `__main__`, so the decorators raise
    "operator 'mesh_contact' already has variant 'default'" before the check starts.

        python -c "import plexus.operators as _; from plexus.operators.contact_ops import selftest;\
                   raise SystemExit(0 if selftest('sphere') else 1)"
        python -c "import plexus.operators as _; from plexus.operators.contact_ops import selftest;\
                   raise SystemExit(0 if selftest('plate', shape='disc_hole') else 1)"
    """
    if surface == "sphere":
        from plexus.operators.vertex_ops import build_sphere_mesh
        verts, es, et, ef, _nF = build_sphere_mesh(int(kw.get("n_cells", 400)),
                                                   float(kw.get("radius", 5.0)),
                                                   float(kw.get("jitter", 0.15)), 0)
        V = np.asarray(verts, np.float64)
        closed = True
    elif surface == "plate":
        # THE PLATE IN ITS STAR FRAME: the patch sits `standoff` below the ray origin, which is
        # where `seed_plate` + a `centre` above it put it. Open, so coverage is a fraction.
        xy, quads = _plate_lattice(int(kw.get("nq", 48)), float(kw.get("half_width", 0.25)),
                                   str(kw.get("shape", "sheet")))
        es, et, ef = _plate_half_edges(quads)
        h = float(kw.get("standoff", 0.30))
        V = np.stack([xy[:, 0], np.full(len(xy), -h), xy[:, 1]], -1)
        closed = False
    else:
        raise ValueError(f"selftest: surface must be 'sphere' or 'plate', got {surface!r}")

    dv = torch.device(dev)
    op = MeshContact({"surface": "vertex", "verbose": False}, device=dev)
    Vt = torch.as_tensor(V, device=dv, dtype=torch.float32)
    M = op._build_from(Vt, torch.as_tensor(np.asarray(es, np.int64), device=dv),
                       torch.as_tensor(np.asarray(et, np.int64), device=dv),
                       torch.as_tensor(np.asarray(ef, np.int64), device=dv),
                       torch.zeros_like(Vt), dev=dv, dt_=torch.float32)
    g = torch.Generator(device="cpu").manual_seed(0)
    u = torch.randn(n, 3, generator=g).to(dv)
    u = u / u.norm(dim=1, keepdim=True)
    r = torch.full((n,), 1e-6, device=dv)
    hit, tri, t, w = op._query(M, u * r[:, None], u, r)
    cov = float(hit.float().mean())

    # BRUTE FORCE OVER EVERY TRIANGLE, for a subsample, and only where the lookup found one: on an
    # open surface most rays legitimately hit nothing, and comparing those would measure the
    # geometry rather than the bins.
    pool = torch.nonzero(hit).squeeze(1)
    s_ = pool[torch.randperm(pool.numel(), generator=g).to(dv)[:n_brute]] if pool.numel() else pool
    nb = int(s_.numel())
    bad, dmax = 0, 0.0
    if nb:
        A, B, C = M["A"], M["B"], M["C"]
        e1, e2 = B - A, C - A
        us = u[s_][:, None, :]
        pp = torch.cross(us.expand(-1, A.shape[0], -1), e2[None], dim=2)
        det = (e1[None] * pp).sum(2)
        inv = 1.0 / torch.where(det.abs() < 1e-20, torch.full_like(det, 1e-20), det)
        sv = -A[None]
        w1 = (sv * pp).sum(2) * inv
        q = torch.cross(sv.expand(nb, -1, -1), e1[None].expand(nb, -1, -1), dim=2)
        w2 = (us * q).sum(2) * inv
        tb = (e2[None] * q).sum(2) * inv
        good = (det.abs() > 1e-20) & (w1 >= -1e-6) & (w2 >= -1e-6) & (w1 + w2 <= 1 + 1e-6) & (tb > 0)
        tb = torch.where(good, tb, torch.full_like(tb, -1.0))
        t_ref = tb.max(dim=1).values
        d = (t[s_] - t_ref).abs() / t_ref.clamp_min(1e-9)
        bad, dmax = int((d > 1e-4).sum()), float(d.max())
    ok = (bad == 0) and (cov > 0.9999 if closed else cov > 0.0)
    print(f"[selftest] {surface}: {M['n_tri']:6d} triangles, {M['G']['nbin']:6d} bins in "
          f"{M['G']['nrow']:3d} rows, max {M['K']:3d} per bucket | coverage {cov:.5f}"
          f"{' (closed: must be 1)' if closed else ' (open: reported, not asserted)'} | "
          f"brute-force disagreement {bad}/{nb} (max relative {dmax:.2e})", flush=True)
    print(f"[selftest] {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


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
    PARAM_ROLES = {"gap_half": "free_half_gap", "gap_half_end": "final_free_half_gap",
                   "close_from": "frame_closing_starts", "close_to": "frame_closing_ends", "stiff": "projection_fraction",
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
        # A MOVING PLATEN. `gap_half` alone is a gap that was always there, which answers "how does
        # a cell behave when confined" but not "what does confining it DO" -- and the second is the
        # question a squashing experiment asks. With `gap_half_end` the plates close linearly from
        # `close_from` to `close_to` (frames), so one run shows the whole sequence: sphere, ovoid,
        # then the flattened, mutually-pressed shape an epithelium has.
        #
        # WHY THE PLATEN MUST BE THIS OPERATOR AND NOT AN `obstacles:` BOX. The obstacle path
        # rasterises solid cells onto the MPM grid and zeroes their velocity, which stops a particle
        # entering but cannot EXPEL one already inside: a cell seeded across a plate keeps its shape
        # and simply freezes the overlapping shell (measured on cell_10 -- z/x aspect 1.00 against a
        # gap 13% narrower than the cell). This operator PROJECTS positions, so a plate that arrives
        # where matter already is pushes that matter out, which is what squashing means.
        self.gap_half_end = params.get("gap_half_end", None)
        self.gap_half_end = None if self.gap_half_end is None else float(self.gap_half_end)
        self.close_from = int(params.get("close_from", 0))
        self.close_to = int(params.get("close_to", 0))
        self._gap0 = self.gap_half
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

        if self.gap_half_end is not None and self.close_to > self.close_from:
            u = (int(getattr(H, "frame", 0)) - self.close_from) / (self.close_to - self.close_from)
            u = min(1.0, max(0.0, u))
            self.gap_half = self._gap0 + u * (self.gap_half_end - self._gap0)
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
# THE PISTON: a surface this repository can SEED, so the interaction has a second geometry.
#
# `mesh_contact` has only ever met one surface -- a growing sphere, replayed from a cache -- and every
# gate row about it is a row about that sphere. What the operator claims is more general than that:
# a triangulated surface, of any shape, pushing a continuum. A claim exercised on one geometry is a
# claim about that geometry.
#
# SO THE SURFACE BECOMES A SEED OPERATOR RATHER THAN A FILE. `seed_plate` builds the patch at frame 0
# the way `seed_mesh` builds the vesicle, and `plate_drive` prescribes its motion, so a spec declares
# the piston instead of pointing at an npz somebody generated. The four shapes are one lattice under
# four masks, which is what makes them comparable: same resolution, same edge length, same bin count
# per unit area -- geometry is the only thing that differs across the ladder.
# ==========================================================================================================
def _plate_lattice(nq, half_width, shape, hole_frac=0.35, bars=4, bar_frac=0.5):
    """One `nq` x `nq` quad lattice minus whatever `shape` removes: (xy of kept vertices, quads).

    THE MASK IS ON THE QUAD CENTRES, not on the vertices, so a removed quad takes no vertex its
    neighbours still need and the kept set is exactly the boundary of what survives.
    """
    g = np.linspace(-half_width, half_width, nq + 1)
    X, Y = np.meshgrid(g, g, indexing="ij")
    vid = np.arange((nq + 1) ** 2).reshape(nq + 1, nq + 1)
    i, j = np.meshgrid(np.arange(nq), np.arange(nq), indexing="ij")
    quads = np.stack([vid[i, j], vid[i + 1, j], vid[i + 1, j + 1], vid[i, j + 1]], -1).reshape(-1, 4)
    cx = 0.5 * (g[:-1] + g[1:])
    CX, CZ = (a.reshape(-1) for a in np.meshgrid(cx, cx, indexing="ij"))
    r = np.hypot(CX, CZ)
    if shape == "sheet":
        keep = np.ones(len(quads), bool)
    elif shape == "disc":
        keep = r <= half_width
    elif shape == "disc_hole":
        keep = (r <= half_width) & (r >= hole_frac * half_width)
    elif shape == "grid":
        # A LATTICE OF BARS: solid strips with square windows between them, so the material is
        # loaded in patches and can extrude through the gaps. `bar_frac` is the fraction of each
        # period the strip occupies, so 1.0 recovers `sheet` and the shape is a continuum.
        per = 2 * half_width / bars
        u = ((CX + half_width) % per) / per
        v = ((CZ + half_width) % per) / per
        w = 0.5 * bar_frac
        keep = (np.minimum(u, 1 - u) < w) | (np.minimum(v, 1 - v) < w)
    else:
        raise ValueError(f"seed_plate: shape must be sheet|disc|disc_hole|grid, got {shape!r}")
    quads = quads[keep]
    if not len(quads):
        raise ValueError(f"seed_plate: shape {shape!r} kept no quads at nq={nq}")
    used = np.unique(quads)
    remap = np.full(vid.size, -1, np.int64)
    remap[used] = np.arange(len(used))
    return np.stack([X.reshape(-1)[used], Y.reshape(-1)[used]], -1), remap[quads]


def _plate_half_edges(quads):
    """(E_srce, E_trgt, E_face) -- four half-edges per quad, in ring order, grouped by face.

    QUADS AND NOT TRIANGLES because `mesh_contact` fans every face from its own centroid, one
    sub-triangle per half-edge, and needs no ring ordering to do it. A quad therefore costs four
    sub-triangles and buys a lattice that can have holes punched in it by dropping whole faces.
    """
    nf = len(quads)
    es = np.concatenate([quads[:, k] for k in (0, 1, 2, 3)])
    et = np.concatenate([quads[:, k] for k in (1, 2, 3, 0)])
    ef = np.tile(np.arange(nf), 4)
    o = np.argsort(ef, kind="stable")
    return es[o], et[o], ef[o]


@register_operator("seed_plate", family="seed", set="vertex", kind="seed")
class SeedPlate(Structural):
    """Frame-0: an open planar half-edge patch normal to `axis`, and its half-edge table.

    WHERE THE PATCH SITS RELATIVE TO `mesh_contact`'s `centre:` IS THE WHOLE DESIGN, and it is the
    one thing a spec can get wrong silently. The contact is star-shaped: it casts a ray from
    `centre` along each particle's own direction, calls the outward normal the one pointing AWAY
    from `centre`, and pushes anything it finds BETWEEN `centre` and the surface further out. So
    `centre` names the region the material is forbidden to enter -- the tissue's interior for the
    spheroid, and for a piston the BODY OF THE PISTON, i.e. a point on the far side of the plate
    from the material. Declare `centre` in the plate's plane and the ray cast is degenerate; declare
    it on the material's side and the plate pushes the wrong way.

    `standoff` REPORTS THE CONSEQUENCE AT FRAME 0 rather than leaving it to be discovered: the
    distance from the declared `centre` to the plate, which is what sizes the contact's direction
    bins. Too small and a rim face is seen edge-on and the bin grid collapses to its floor of four
    rows; too large and the angular size of a face falls under the 200-row cap and a face spans more
    than one bin, which is the assumption the 3x3 lookup rests on.
    """

    EMIT = None                        # writes positions and the mesh table; no integrable delta
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["planar_patch", "half_edge_mesh", "rigid_indenter", "initial_condition"]
    PARAM_ROLES = {"shape": "patch_topology", "nq": "quads_across", "half_width": "patch_half_extent",
                   "height": "position_on_axis", "axis": "patch_normal_axis",
                   "hole_frac": "inner_over_outer_radius", "bars": "strips_per_side",
                   "bar_frac": "strip_width_over_period"}
    REFERENCE = ("Plexus (this work). The indenter geometry of a nanoindentation / parallel-plate "
                 "compression assay; the half-edge layout is Okuda, S. et al. (2013) "
                 "Biomech. Model. Mechanobiol. 12:627-644, as used by `seed_mesh`.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.shape = str(params.get("shape", "sheet"))
        self.nq = int(params.get("nq", 48))
        self.half_width = float(params.get("half_width", 0.25))
        self.axis = int(params.get("axis", 1))
        self.height = float(params.get("height", 0.55))
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5])]   # the two IN-PLANE axes
        # THE PROFILE IS ORTHOGONAL TO THE SHAPE, and keeping them apart is the point. `shape`
        # decides which quads of the lattice survive (sheet / disc / disc with a hole / grid) and
        # `profile` displaces the survivors along the normal -- so `disc` + `cone` is a conical
        # indenter and `sheet` + `pyramid` a Vickers-like one, from one seeder and one lattice.
        # Fold them into a single `shape:` list and every new tip costs a new mask.
        #
        #   flat     the plane, and the default: 461 specs' worth of nothing changes.
        #   pyramid  apex DOWN at the centre, four faces rising to the rim -- Chebyshev distance.
        #   cone     the same with a radial distance, i.e. the axisymmetric tip.
        #
        # `profile_depth` is the APEX-TO-RIM rise. It is declared rather than an angle because the
        # angle also depends on `half_width`, and two numbers that pin one geometry is one too many;
        # the half-angle it implies is printed instead, since that is what an indenter is specified
        # by (Berkovich 70.3 deg, Vickers 68 deg, both as equivalent cones).
        self.profile = str(params.get("profile", "flat"))
        self.profile_depth = float(params.get("profile_depth", 0.0))
        self.hole_frac = float(params.get("hole_frac", 0.35))
        self.bars = int(params.get("bars", 4))
        self.bar_frac = float(params.get("bar_frac", 0.5))
        self.standoff = params.get("standoff", None)
        self.standoff = None if self.standoff is None else float(self.standoff)

    def forward(self, H, mask=None):
        from plexus.models.mesh import MeshTable
        lvl = H.level(self.at)
        dev, dt_ = lvl.state.device, lvl.state.dtype
        xy, quads = _plate_lattice(self.nq, self.half_width, self.shape,
                                   self.hole_frac, self.bars, self.bar_frac)
        es, et, ef = _plate_half_edges(quads)
        Nv, nF = len(xy), len(quads)
        Nbuf = lvl.state.shape[0]
        if Nv > Nbuf:
            raise ValueError(f"seed_plate: the {self.shape} patch has {Nv} vertices but the "
                             f"`{self.at}` set declares n={Nbuf}")
        ip = [a for a in (0, 1, 2) if a != self.axis]              # the two in-plane axes
        pos = torch.zeros(Nbuf, 3, dtype=dt_, device=dev)
        pos[:Nv, ip[0]] = torch.as_tensor(xy[:, 0] + self.centre[0], dtype=dt_, device=dev)
        pos[:Nv, ip[1]] = torch.as_tensor(xy[:, 1] + self.centre[1], dtype=dt_, device=dev)
        pos[:Nv, self.axis] = self.height
        if self.profile != "flat":
            if self.profile == "pyramid":
                d = np.maximum(np.abs(xy[:, 0]), np.abs(xy[:, 1]))       # square, apex-down
            elif self.profile == "cone":
                d = np.hypot(xy[:, 0], xy[:, 1])                         # axisymmetric
            else:
                raise ValueError(f"seed_plate: profile must be flat|pyramid|cone, "
                                 f"got {self.profile!r}")
            # THE APEX SITS AT `height` AND THE RIM RISES ABOVE IT, so `height` keeps meaning "the
            # lowest point of the tool" across the whole ladder -- the number a descent is measured
            # against. Defining it at the rim instead would make the flat plate and the pyramid
            # start their contact at different depths for the same declared height.
            pos[:Nv, self.axis] += torch.as_tensor(
                self.profile_depth * d / max(self.half_width, 1e-12), dtype=dt_, device=dev)
        p0, p1 = lvl.state_schema["pos"]
        st = lvl.state.clone(); st[:, p0:p1] = pos; lvl.state = st
        if getattr(lvl, "occ", None) is not None:
            occ = torch.zeros(Nbuf, device=dev); occ[:Nv] = 1.0; lvl.occ = occ
        seeded = dict(E_srce=torch.as_tensor(es, device=dev),
                      E_trgt=torch.as_tensor(et, device=dev),
                      E_face=torch.as_tensor(ef, device=dev), nF=nF, Nv=Nv,
                      # THE SURFACE VELOCITY LIVES ON THE TABLE, not in a `vel` state block, because
                      # it is a property of the SURFACE and `mesh_contact` reads it as one -- the
                      # replay path derives exactly this by differencing consecutive cached meshes.
                      # A vertex set that declared `vel` would also have the engine integrate it,
                      # and a prescribed piston must not be integrated.
                      Vv=torch.zeros(Nv, 3, dtype=dt_, device=dev),
                      plate_axis=self.axis, plate_height=self.height,
                      verts0=pos[:Nv].detach().to("cpu").numpy())
        m = getattr(lvl, "_mesh", None)
        if isinstance(m, MeshTable):
            m.clear(); m.update(seeded)
        else:
            lvl._mesh = MeshTable(**seeded)
        edge = 2.0 * self.half_width / self.nq
        so = "" if self.standoff is None else (
            f"; standoff {self.standoff:.4g} from the declared contact centre -> a face subtends "
            f"~{edge / self.standoff:.4f} rad, i.e. ~{int(math.pi / max(edge / self.standoff, 1e-3))} "
            f"bin rows (the contact caps at 200 and floors at 4)")
        pr = ""
        if self.profile != "flat":
            pr = (f", {self.profile} apex-down: rim {self.profile_depth:.4g} above the apex, "
                  f"half-angle {math.degrees(math.atan2(self.half_width, max(self.profile_depth, 1e-12))):.1f} "
                  f"deg from the axis (Berkovich 70.3, Vickers 68 as equivalent cones)")
        print(f"[seed_plate] {self.shape}: {nF} quads ({4 * nF} sub-triangles), {Nv} vertices, "
              f"edge {edge:.4f}, half-width {self.half_width} at axis-{self.axis} = "
              f"{self.height}{pr}{so}", flush=True)
        return {}


@register_operator("surface_drive", "plate_drive", family="mechanics", set="vertex",
                   kind="structural")
class SurfaceDrive(Structural):
    """Move a seeded SURFACE along one axis at a prescribed rate, and publish its velocity.

    `plate_drive` IS AN ALIAS AND NOT THE NAME. The operator translates whatever surface the set
    holds -- a plate, a sphere, a cap -- so naming it for one of them made the sphere rung read as
    a mistake. Canonical name first, on the `seed_mesh`/`mesh_seed` precedent.

    KINEMATIC, NOT DYNAMIC, AND THAT IS THE MODELLING DECISION. The plate is a rigid indenter whose
    position is imposed; it does not accelerate under the reaction it collects. `mesh_contact` still
    computes that reaction and still accumulates it per vertex (`VERTEX_FORCE`), so the load the
    material puts back is MEASURED here even though it is not integrated -- which is the difference
    between a one-way coupling that reports its own residual and one that hides it.

    THE VELOCITY IS PUBLISHED, NOT INFERRED. The contact's friction law needs the surface's velocity
    at the contact point, and on the replay path it gets it by differencing consecutive cached
    meshes. A prescribed plate KNOWS its velocity exactly, so it writes it, and the friction is then
    reading the motion that is happening rather than a finite difference of it.
    """

    EMIT = None                        # moves positions in place; no integrable delta
    SUPPORTED_DIMS = [3]
    # `target`, NOT `to`: the schema reserves `to`/`from` for FIELD references (`mpm_scatter`'s
    # `to: mpm_grid`), so a scalar there is resolved as a field name and the spec dies with
    # "references unknown field 0.35".
    # `by` OR `target`, AND `by` IS THE ONE A LADDER WANTS. `target` is an absolute coordinate of
    # the surface's CENTROID, which is the tool's low point for a flat plate, its centre for a
    # sphere, and neither for a pyramid -- so three rungs meant to differ only in the tool would
    # have had to declare three different numbers to travel the same distance. `by` is that
    # distance, signed along the axis, and it means the same thing for every shape.
    REQUIRES_PARAMS = []
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["prescribed_kinematics", "rigid_indenter", "moving_boundary"]
    PARAM_ROLES = {"by": "signed_displacement_along_axis",
                   "target": "final_centroid_position_on_axis", "over": "frames_of_travel",
                   "rigid": "reset_from_seed_or_translate_live",
                   "axis": "motion_axis", "hold": "frames_held_before_moving"}
    REFERENCE = "Plexus (this work); the loading protocol of a displacement-controlled indentation."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.axis = int(params.get("axis", 1))
        if ("target" in params) == ("by" in params):
            raise ValueError("surface_drive: declare exactly one of `by:` (a signed displacement "
                             "along the axis, the same for every tool shape) or `target:` (an "
                             "absolute coordinate of the surface's centroid)")
        self.target = float(params["target"]) if "target" in params else None
        self.by = float(params["by"]) if "by" in params else None
        self.over = int(params.get("over", 0))          # 0 -> the whole run, resolved at frame 0
        self.hold = int(params.get("hold", 0))
        # RIGID BY DEFAULT: a tool's shape is its own, and resetting from the seeded positions is
        # exact at every frame instead of accumulating one increment's rounding per step.
        self.rigid = bool(params.get("rigid", True))
        self._from = None
        self._v0 = None
        self._y_prev = None
        self._said = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        if m is None or not int(m.get("Nv", 0)):
            return {}                                   # the seed has not run yet
        Nv = int(m["Nv"])
        pos = lvl.get("pos")
        if self._from is None:
            v0 = m.get("verts0", None)
            self._v0 = (torch.as_tensor(v0[:Nv, self.axis], device=pos.device, dtype=pos.dtype)
                        if v0 is not None else pos[:Nv, self.axis].clone())
            self._from = float(self._v0.mean())      # the surface's own centroid on that axis
            if self.target is None:
                self.target = self._from + self.by
            self._y_prev = self._from
            self.over = self.over or max(1, int(getattr(H, "n_frames", 0) or 1) - self.hold)
        f = int(getattr(H, "frame", 0) or 0)
        u = min(1.0, max(0.0, (f - self.hold) / float(self.over)))
        y = self._from + u * (self.target - self._from)
        # A TRANSLATION OF THE SEEDED SHAPE, NOT AN ASSIGNMENT. `pos[:Nv, axis] = y` is the same
        # thing for a plate, whose vertices all share one coordinate on that axis -- and it FLATTENS
        # a sphere into a disc on the first frame. The offset is measured from the seeded positions,
        # so the shape is exact at every frame rather than drifting with whatever the last one did.
        if self.rigid:
            pos[:Nv, self.axis] = self._v0 + (y - self._from)
        else:
            # INCREMENTAL, so whatever else moved the surface this frame SURVIVES. Resetting from
            # the seeded positions is exact for a rigid tool and destroys a solved one: a spheroid
            # under `cell_mechanics` would have its shape overwritten on the drive axis every frame,
            # so the mechanics would run and be discarded and the surface would read as rigid while
            # a schedule full of operators said otherwise.
            pos[:Nv, self.axis] = pos[:Nv, self.axis] + (y - self._y_prev)
        self._y_prev = y
        # ZERO BEFORE AND AFTER THE TRAVEL, not the mean rate: a friction law told the plate is
        # still sliding while it is parked would shear the material it is resting on for the whole
        # hold, and the hold exists precisely to show the material at rest under a static load.
        moving = (self.hold <= f < self.hold + self.over)
        v = (self.target - self._from) / (self.over * float(getattr(H, "dt", 1.0))) if moving else 0.0
        # THE VELOCITY BUFFER IS THIS OPERATOR'S, NOT THE SEED'S. `seed_plate` allocated it and
        # `seed_mesh` -- which predates all of this and serves 461 specs -- does not, so a surface
        # built by the second and driven by this died on `KeyError: 'Vv'`. The operator that
        # publishes a quantity is the one that should create the place to put it.
        vv = m.get("Vv", None)
        if vv is None or vv.shape[0] < Nv:
            vv = torch.zeros(pos.shape[0], 3, device=pos.device, dtype=pos.dtype)
            m["Vv"] = vv
        vv[:Nv] = 0.0
        vv[:Nv, self.axis] = v
        m["plate_height"] = y
        if not self._said:
            print(f"[plate_drive] {self.at}: axis {self.axis}, {self._from:.4g} -> {self.target:.4g} "
                  f"over {self.over} frames after a {self.hold}-frame hold; "
                  f"speed {abs(v):.5g} box units per unit time", flush=True)
            self._said = True
        return {}


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

