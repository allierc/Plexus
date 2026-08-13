"""bm_ops -- the basement membrane as a CODIMENSION-1 SHEET, and the certification of its strain.

    python bm_ops.py [--device cuda:0]        -> the certification, printed; nothing is written

WHY A MESH AND NOT A CLOUD. Sixty-six archived runs (`note_spheroid_bm_ecm` S8) modelled the sheet as
MPM material and every one of them ran into the same wall: a basement membrane is ~100 nm on a ~200 um
spheroid, one part in 2000 of the radius, so resolving it through a background grid needs dx ~ 3e-4 and
a grid of 3500^3. It is not a thin solid that we cannot afford to resolve; it is a SURFACE, and its
thickness belongs in the constitutive law as a number, never in the geometry as a length.

WHAT THAT BUYS, AND IT IS THE WHOLE POINT OF STEP 05a. In MPM the deformation gradient F is advected by
the GRID's velocity gradient, so a particle moved by anything other than the grid carries no record of
having moved: run 130 reported a stretch of 0.31 where the geometry had moved the sheet by 2.44, i.e.
13% of the truth, and run 121 -- which routed the same force through the grid and paid for it with the
standoff -- reported 2.25 of 2.30. On a mesh there is nothing to launder. F is MEASURED, per triangle,
from where its three nodes are now against where they were at seeding:

    Ds = [x1-x0, x2-x0]  (3x2, now)      Dm = the same in the reference triangle's own 2D frame
    F  = Ds Dm^-1        (3x2)           C = F^T F  (2x2)      lambda_1,2 = sqrt(eig C)

so a node moved by a tether, by a plaque, by a grid or by hand is seen identically. The claim that this
measure is exact is not an argument; `selftest()` below applies maps whose singular values are known in
closed form -- a rigid rotation, a uniform dilation, an anisotropic affine map -- and reports the error.

THE MECHANICS. St Venant-Kirchhoff membrane, one constant-strain triangle per face, energy

    W_f = A0_f [ mu tr(Eg^2) + (lam/2) (tr Eg)^2 ],   Eg = (C - C0)/2 in the reference metric,
    mu  = Y2 / (2(1+nu)),   lam = Y2 nu / (1-nu^2),   Y2 = E * T  (a 2D modulus: force per LENGTH)

with forces taken as -dW/dx by autograd, so the force and the energy cannot disagree. MASSLESS AND
OVERDAMPED, x <- x + dt M f with M = 1/gamma: a sheet with no mass has no wave speed, so its stability
limit is a RATE, dt * M * lambda_max(Hessian) < 2, about twenty times weaker than the CFL an inertial
sheet would carry. That bound is not asserted here either -- `spectral_rate()` measures lambda_max by
power iteration on Hessian-vector products, and the 05a sweep runs either side of it.

REMODELLING is the reference metric creeping toward the current one, C0 <- C0 + (dt/tau_r)(C - C0),
which is the 2D form of the archived line's per-edge l0_dot = (L - l0)/tau_r. It is what separates the
two stretches this rig reports and which must never be quoted as one number:

    lambda_geo  = stretch against the metric the sheet was SEEDED with   -> the tissue's growth, 3.4x
    lambda_el   = stretch against the metric it currently REMEMBERS      -> what the material feels

AREAL DENSITY rides the same measure: rho = rho0 / J_geo with J_geo = det F against the seeded metric,
so with no secretion a sheet enclosing a tripling radius thins by ~11x. That is the baseline `bm_secrete`
has to beat, and the number run 128 failed to move because the material it added was never in the sheet.

WHAT IS NOT HERE. No bending stiffness (the sheet resists stretch, not curvature -- for a 100 nm sheet
on a 200 um ball the bending term is smaller by (T/R)^2 ~ 2.5e-7, but a fold or a hole would need it),
no bond breaking, no secretion, no plaque: those are 05c and 05b. The tether used by 05a is a one-sided
spring to a frozen direction, exactly what S9 says a plaque is NOT; it is here so that the sheet has a
load, and it is replaced in 05b.
"""
from __future__ import annotations

import math
import sys

import torch


# ---------------------------------------------------------------------------------------------
#  geometry: an icosphere, because the sheet's own mesh quality is a term in every measurement
# ---------------------------------------------------------------------------------------------
def icosphere(subdiv=4, device="cpu", dtype=torch.float64):
    """Unit sphere by recursive subdivision of an icosahedron: V (n,3), F (m,3), E (k,2).

    WHY NOT A LAT-LON GRID, which would be three lines shorter: its triangles at the pole are a
    thousand times smaller than at the equator, and every per-face quantity this rig reports would then
    be an average over a distribution the mesh invented. The icosphere's edge lengths span a factor of
    1.24 at subdiv 4 (measured, printed by `selftest`), so a per-face mean means what it says.
    """
    t = (1.0 + 5.0 ** 0.5) / 2.0
    V = torch.tensor([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                      [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                      [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=dtype)
    F = torch.tensor([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                      [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                      [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                      [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], dtype=torch.long)
    V = V / V.norm(dim=1, keepdim=True)
    for _ in range(subdiv):
        cache, newF = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                nonlocal V
                m = (V[a] + V[b]) * 0.5
                V = torch.cat([V, (m / m.norm())[None]], 0)
                cache[key] = V.shape[0] - 1
            return cache[key]

        for f in F.tolist():
            a, b, c = f
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            newF += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        F = torch.tensor(newF, dtype=torch.long)
    e = torch.cat([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0)
    e = torch.stack([e.min(dim=1).values, e.max(dim=1).values], 1)
    E = torch.unique(e, dim=0)
    return V.to(device), F.to(device), E.to(device)


# ---------------------------------------------------------------------------------------------
#  the strain measure -- the thing 05a exists to certify
# ---------------------------------------------------------------------------------------------
def reference_frame(X, F):
    """Dm (m,2,2) and A0 (m,) for the reference configuration X: each triangle in its OWN 2D frame.

    The frame is built from the triangle, not from the world, which is why a rigid rotation of the
    whole sheet leaves every entry of C unchanged rather than merely leaving its eigenvalues unchanged.
    """
    x0, x1, x2 = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    u, v = x1 - x0, x2 - x0
    a = u.norm(dim=1)
    e1 = u / a.clamp_min(1e-30)[:, None]
    n = torch.cross(u, v, dim=1)
    e2 = torch.cross(n / n.norm(dim=1).clamp_min(1e-30)[:, None], e1, dim=1)
    # COLUMNS are the two reference edge vectors expressed in (e1, e2): [[a, v.e1], [0, v.e2]].
    # Transposing this -- which the first version did -- leaves det(Dm) and therefore J untouched
    # while corrupting every individual principal stretch, so it passes an area check and fails a
    # stretch check. That is exactly what `selftest` caught: J to 1e-13, lambda off by 0.95.
    Dm = torch.stack([torch.stack([a, (v * e1).sum(1)], 1),
                      torch.stack([torch.zeros_like(a), (v * e2).sum(1)], 1)], 1)
    A0 = 0.5 * n.norm(dim=1)
    return Dm, A0


def cauchy_green(x, F, Dm_inv):
    """C = F^T F per triangle (m,2,2) -- the only place deformation is ever read in this prototype."""
    x0, x1, x2 = x[F[:, 0]], x[F[:, 1]], x[F[:, 2]]
    Ds = torch.stack([x1 - x0, x2 - x0], 2)       # (m,3,2)
    Fd = Ds @ Dm_inv                              # (m,3,2)
    return Fd.transpose(1, 2) @ Fd


def principal_stretches(C):
    """(lambda_1, lambda_2) per triangle, largest first. Closed form for 2x2, so no eig on 20k faces.

    ONE NUMERICAL PROPERTY, so that `selftest`'s output is not misread. The discriminant is
    sqrt(tr^2/4 - det), and under an ISOTROPIC stretch those two terms are equal: their difference is
    computed as a cancellation, so it carries an absolute error of ~eps and its square root carries
    ~sqrt(eps) = 1e-8 in float64. That is why the rigid and dilation cases certify at 1e-8 while the
    anisotropic affine case -- where the eigenvalues are well separated and nothing cancels -- certifies
    at 1e-15. The 1e-8 is the degeneracy, not the measure; J = l1*l2 is unaffected (1e-13) because it is
    det C and never touches the discriminant.
    """
    a, b, d = C[:, 0, 0], C[:, 0, 1], C[:, 1, 1]
    tr, det = a + d, (a * d - b * b).clamp_min(0.0)
    disc = ((tr * tr) / 4 - det).clamp_min(0.0).sqrt()
    l1 = (tr / 2 + disc).clamp_min(0.0).sqrt()
    l2 = (tr / 2 - disc).clamp_min(0.0).sqrt()
    return l1, l2


def relative_stretches(C, C0):
    """Stretch of C measured against a reference metric C0 -- eigenvalues of C0^-1 C, square-rooted.

    This is what makes `lambda_el` a different number from `lambda_geo` rather than the same number
    twice: C0 is the metric the sheet currently REMEMBERS (remodelled), the seeded one is the metric
    the tissue's growth is measured against.
    """
    a, b, d = C0[:, 0, 0], C0[:, 0, 1], C0[:, 1, 1]
    det0 = (a * d - b * b).clamp_min(1e-30)
    inv = torch.stack([torch.stack([d, -b], 1), torch.stack([-b, a], 1)], 1) / det0[:, None, None]
    M = inv @ C
    tr, det = M[:, 0, 0] + M[:, 1, 1], (M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]).clamp_min(0)
    disc = ((tr * tr) / 4 - det).clamp_min(0.0).sqrt()
    return (tr / 2 + disc).clamp_min(0).sqrt(), (tr / 2 - disc).clamp_min(0).sqrt()


# ---------------------------------------------------------------------------------------------
#  the sheet
# ---------------------------------------------------------------------------------------------
# The four children of a midpoint (1->4) split, as constant maps in the PARENT's material
# coordinates. If the parent's reference edge vectors are the columns (e1, e2) of Dm, then child k's
# are Dm @ S_k, so Dm_inv_child = S_k^-1 @ Dm_inv_parent -- and F, C and lambda come out IDENTICAL for
# parent and child at the moment of the split. That identity is gate G14/G15: a child whose reference
# frame is rebuilt from its CURRENT shape instead would silently reset lambda to 1, which is the mesh
# version of the laundering that made run 130 report 13% of its true stretch.
#   A: (v0, m01, m20)    B: (v1, m12, m01)    C: (v2, m20, m12)    D: (m01, m12, m20)
# All four have det S = 1/4 > 0, so the orientation of every child matches its parent.
_SPLIT_S = [[[0.5, 0.0], [0.0, 0.5]],
            [[-0.5, -0.5], [0.5, 0.0]],
            [[0.0, 0.5], [-0.5, -0.5]],
            [[0.0, -0.5], [0.5, 0.5]]]


class Sheet:
    """A codim-1 StVK membrane, massless and overdamped, with a remodelling reference metric.

    Everything is float64 by default. The sheet is 10k nodes, not 10M particles, so double precision
    costs nothing here and removes float32 from the list of explanations for any residual -- which
    matters, because 05b's falsifier IS a residual at machine precision.

    THE SET IS A RESERVOIR. Nodes and faces are allocated to their maximum at seeding and carry `occ`,
    the framework's own dormancy flag -- `engine.py:453` allocates `grow_reserve` dormant MPM particles
    the same way for `cell_grow` to wake. Everything that changes the sheet's SIZE is then a flip of
    `face_occ`: `refine` wakes three faces per parent, `tear` puts faces to sleep, and a future
    `bm_secrete` wakes them on a mass balance. Nothing is ever reallocated, so an index held by a plaque
    or a measurement stays valid across every one of those operations.

    `max_refine = 0` allocates no reserve and every array is exactly the live set, so the runs that
    predate the reservoir are bit-identical.
    """

    def __init__(self, subdiv=4, R0=0.0875, centre=(0.5, 0.5, 0.5), E=400.0, thickness=2.0e-3,
                 nu=0.3, beta=0.0, mobility=1.0, tau_r=0.0, rho0=1.0, max_refine=0, dev="cuda:0",
                 dtype=torch.float64):
        self.dev, self.dtype, self.R0 = dev, dtype, float(R0)
        V, Fc, Ed = icosphere(subdiv, device=dev, dtype=dtype)
        self.c = torch.tensor(centre, device=dev, dtype=dtype)
        self.subdiv, self.max_refine = int(subdiv), int(max_refine)
        n0, m0, e0 = V.shape[0], Fc.shape[0], Ed.shape[0]
        # sizes after `max_refine` global 1->4 splits: faces x4 each time, one new node per edge
        n_max = n0 + (e0 * (4 ** self.max_refine - 1)) // 3
        m_max = m0 * 4 ** self.max_refine
        self.x = torch.zeros(n_max, 3, device=dev, dtype=dtype)
        self.x[:n0] = self.c + V * R0
        self.node_occ = torch.zeros(n_max, dtype=torch.bool, device=dev)
        self.node_occ[:n0] = True
        self.F_all = torch.zeros(m_max, 3, dtype=torch.long, device=dev)
        self.F_all[:m0] = Fc
        self.face_occ = torch.zeros(m_max, dtype=torch.bool, device=dev)
        self.face_occ[:m0] = True
        self._n_ptr, self._f_ptr = n0, m0            # first never-used slot of each pool
        self.u0 = V.clone()                          # the seeded direction of every ORIGINAL node
        self.Dm_inv = torch.zeros(m_max, 2, 2, device=dev, dtype=dtype)
        self.A0 = torch.zeros(m_max, device=dev, dtype=dtype)
        self.C0 = torch.zeros(m_max, 2, 2, device=dev, dtype=dtype)
        self.Y2 = torch.zeros(m_max, device=dev, dtype=dtype)
        # MASS IS THE STATE; AREAL DENSITY IS DERIVED FROM IT. rho = m_f / A_f(now), so the dilution
        # term -rho (Adot/A) of the mass balance is not something to integrate -- it is what m/A does
        # on its own, and writing it as a separate update would count it twice. Only secretion and
        # proteolysis change `mass`; stretching and refinement do not. A split divides it four ways,
        # which is what makes `sum_f mass_f` invariant across a remesh (gate G18b) and what the first
        # version, which carried rho as a state, could not have given.
        self.mass = torch.zeros(m_max, device=dev, dtype=dtype)
        self.E0, self.T, self.nu, self.beta = float(E), float(thickness), float(nu), float(beta)
        self.rho0, self.tau_r, self.M = float(rho0), float(tau_r), float(mobility)
        self._S = torch.tensor(_SPLIT_S, device=dev, dtype=dtype)
        self._Sinv = torch.linalg.inv(self._S)
        self._resync()
        self.reseed()
        self.v = torch.zeros_like(self.x)
        self.n_refinements = 0

    # -- the reservoir ---------------------------------------------------------------------------
    def _resync(self):
        """Recompute the live index sets after any topology change. Everything downstream indexes
        through `self.live`, so a dead face is never touched -- which matters because a dead face's
        node indices point at reservoir slots whose positions are meaningless and whose reference
        frame would be singular."""
        self.live = torch.nonzero(self.face_occ, as_tuple=False).flatten()
        self.live_nodes = torch.nonzero(self.node_occ, as_tuple=False).flatten()
        self.m, self.n = int(self.live.numel()), int(self.live_nodes.numel())
        e = torch.cat([self.Fc[:, [0, 1]], self.Fc[:, [1, 2]], self.Fc[:, [2, 0]]], 0)
        e = torch.stack([e.min(dim=1).values, e.max(dim=1).values], 1)
        self.Ed = torch.unique(e, dim=0)

    @property
    def Fc(self):
        return self.F_all[self.live]

    @property
    def Dm_inv_seed(self):        # kept under its old name: every measurement is against the SEED
        return self.Dm_inv[self.live]

    def reseed(self, x_live=None):
        """Declare the current configuration to be the reference one. Called once at construction and
        again by the rigs, which seed the sheet on the recorded epithelial surface rather than on a
        sphere -- if the reference metric were a sphere's, frame 0 would already report a strain the
        tissue never applied."""
        if x_live is not None:
            self.x[self.live_nodes] = x_live
        Dm, A0 = reference_frame(self.x, self.Fc)
        self.Dm_inv[self.live] = torch.linalg.inv(Dm)
        self.A0[self.live] = A0
        self.C0[self.live] = torch.eye(2, device=self.dev, dtype=self.dtype)
        self.Y2[self.live] = self.E0 * self.T
        self.mass[self.live] = self.rho0 * A0            # so rho = mass/A = rho0 at seeding
        self.x_seed = self.x.clone()
        self.A_seed_total = float(A0.sum())
        self.mean_edge_seed = float((self.x[self.Ed[:, 1]] - self.x[self.Ed[:, 0]]).norm(dim=1).mean())

    def refine(self):
        """Global 1->4 midpoint split of every live face. Conforming by construction -- every edge is
        split, so no face is left with a hanging node on one of its sides.

        THE MIDPOINTS ARE PLACED ON THE CHORD AND NOT PROJECTED ONTO THE SURFACE. Projecting would
        smooth the sheet and change lambda, which would break G14 by construction: the smoothing has to
        come from the dynamics, not from the remesher.
        """
        F = self.Fc                                                       # (m,3), live only
        m = F.shape[0]
        pairs = torch.cat([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0)
        srt = torch.stack([pairs.min(dim=1).values, pairs.max(dim=1).values], 1)
        uniq, inv = torch.unique(srt, dim=0, return_inverse=True)
        ne = uniq.shape[0]
        if self._n_ptr + ne > self.x.shape[0] or self._f_ptr + 3 * m > self.F_all.shape[0]:
            raise RuntimeError(f"the reservoir is exhausted: refining {m} faces needs {ne} nodes and "
                               f"{3*m} face slots, {self.x.shape[0]-self._n_ptr} and "
                               f"{self.F_all.shape[0]-self._f_ptr} are free. Raise `max_refine`.")
        mid = torch.arange(ne, device=self.dev) + self._n_ptr
        self.x[mid] = 0.5 * (self.x[uniq[:, 0]] + self.x[uniq[:, 1]])
        self.node_occ[mid] = True
        self._n_ptr += ne
        eid = inv.reshape(3, m).T                                          # (m,3): edges 01, 12, 20
        m01, m12, m20 = mid[eid[:, 0]], mid[eid[:, 1]], mid[eid[:, 2]]
        kids = [torch.stack([F[:, 0], m01, m20], 1), torch.stack([F[:, 1], m12, m01], 1),
                torch.stack([F[:, 2], m20, m12], 1), torch.stack([m01, m12, m20], 1)]
        slots = [self.live] + [torch.arange(m, device=self.dev) + self._f_ptr + k * m
                               for k in range(3)]                          # child A reuses the parent
        Dmi, A0p, C0p, Y2p = (self.Dm_inv[self.live], self.A0[self.live], self.C0[self.live],
                              self.Y2[self.live])
        mp = self.mass[self.live]
        for k in range(4):
            s = slots[k]
            self.F_all[s] = kids[k]
            # Dm_child = Dm_parent @ S_k  =>  Dm_inv_child = S_k^-1 @ Dm_inv_parent, EXACTLY
            self.Dm_inv[s] = self._Sinv[k] @ Dmi
            self.A0[s] = A0p * 0.25                       # |det S_k| = 1/4 for all four children
            self.C0[s] = C0p                              # the memory is in material coordinates
            self.Y2[s] = Y2p
            self.mass[s] = mp * 0.25                      # EXTENSIVE: divides, so the total is exact
            self.face_occ[s] = True
        self._f_ptr += 3 * m
        self._resync()
        self.n_refinements += 1
        return ne, 3 * m

    def tear(self, mask_live):
        """Put faces to sleep. `mask_live` is a boolean over the LIVE faces; a hole is then simply a
        region with no live faces, and its rim is free. Nothing is reallocated and no node is removed,
        so a plaque bound to a node inside the hole keeps its index and stops being loaded by a sheet
        that is no longer there."""
        idx = self.live[mask_live]
        self.face_occ[idx] = False
        self._resync()
        return int(idx.numel())

    def mean_edge(self):
        return float((self.x[self.Ed[:, 1]] - self.x[self.Ed[:, 0]]).norm(dim=1).mean())

    def euler_check(self):
        """G16: every edge of the live mesh is shared by exactly two live faces, or it is a rim edge.
        A refinement that leaves a hanging node shows up here as an edge with three."""
        F = self.Fc
        pairs = torch.cat([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0)
        srt = torch.stack([pairs.min(dim=1).values, pairs.max(dim=1).values], 1)
        _, cnt = torch.unique(srt, dim=0, return_counts=True)
        return dict(edges=int(cnt.numel()), interior=int((cnt == 2).sum()),
                    rim=int((cnt == 1).sum()), bad=int((cnt > 2).sum()))

    # -- energy and force ------------------------------------------------------------------------
    def energy(self, x):
        C = cauchy_green(x, self.Fc, self.Dm_inv_cur())
        Eg = 0.5 * (C - torch.eye(2, device=self.dev, dtype=self.dtype))
        Y2 = self.Y2[self.live]
        mu = Y2 / (2 * (1 + self.nu))
        lam = Y2 * self.nu / (1 - self.nu ** 2)
        trE = Eg[:, 0, 0] + Eg[:, 1, 1]
        trE2 = (Eg @ Eg)[:, 0, 0] + (Eg @ Eg)[:, 1, 1]
        return (self.A0[self.live] * (mu * trE2 + 0.5 * lam * trE * trE)).sum()

    def Dm_inv_cur(self):
        """Dm^-1 for the CURRENT reference metric. With no remodelling this is the seeded one; with
        remodelling C0 has moved, and the elastic energy must be measured against where it moved to --
        a sheet that forgets being stretched is a sheet whose rest state is not the one it was born in.

        With C0 = L L^T (Cholesky), F_el = F_geo L^-T has C_el = L^-1 C_geo L^-T, whose eigenvalues are
        those of C0^-1 C_geo -- the same pair `stretch_elastic` reports, so the energy and the reported
        elastic stretch cannot drift apart.
        """
        if self.tau_r <= 0:
            return self.Dm_inv_seed
        L = torch.linalg.cholesky(self.C0[self.live])
        return self.Dm_inv_seed @ torch.linalg.inv(L.transpose(1, 2))

    def elastic_force(self, x):
        xr = x.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(self.energy(xr), xr)
        return -g

    # -- the measures ----------------------------------------------------------------------------
    def stretch_geo(self, x=None):
        C = cauchy_green(self.x if x is None else x, self.Fc, self.Dm_inv_seed)
        return principal_stretches(C)

    def stretch_elastic(self, x=None):
        C = cauchy_green(self.x if x is None else x, self.Fc, self.Dm_inv_seed)
        return relative_stretches(C, self.C0[self.live])

    def areal_density(self):
        """rho = mass / current area, per live face. This is the quantity `bm_secrete` must hold flat
        and the quantity a tear criterion reads; it falls as 1/J under stretch WITHOUT anything
        integrating a dilution term, because that is what dividing by a growing area does."""
        return self.mass[self.live] / self.area().clamp_min(1e-30)

    def total_mass(self):
        """The conserved quantity: nothing but secretion and proteolysis may change it. G18b."""
        return float(self.mass[self.live].sum())

    def enclosed_volume(self):
        """Signed volume of the closed live surface, by the divergence theorem. Part of G18d: a remesh
        must not move the physical surface, and a midpoint split places new nodes on the chord, so the
        volume is the one invariant that WOULD move if the remesher smoothed anything."""
        x = self.x[self.Fc] - self.c
        return float(torch.abs((x[:, 0] * torch.cross(x[:, 1], x[:, 2], dim=1)).sum() / 6.0))

    def area_centroid(self):
        """Area-weighted centroid of the live surface -- the other half of G18d."""
        a = self.area()
        c = self.x[self.Fc].mean(1)
        return (c * a[:, None]).sum(0) / a.sum().clamp_min(1e-30)

    def area(self, x=None):
        x = self.x if x is None else x
        F = self.Fc
        return 0.5 * torch.cross(x[F[:, 1]] - x[F[:, 0]], x[F[:, 2]] - x[F[:, 0]], dim=1).norm(dim=1)

    def spectral_rate(self, iters=60, v0=None, return_vec=False):
        """lambda_max of the elastic Hessian at the CURRENT configuration, by power iteration on Hvp.

        This is the number the overdamped stability bound is written in: the step is stable while
        dt * M * lambda_max < 2. It is measured rather than assumed, and it is measured AGAIN as the
        run goes on, because it is not a constant -- a StVK membrane's tangent stiffens with stretch,
        so a bound taken at the seeded configuration is an underestimate everywhere else. In 05a it
        grew 19.1x over 401 frames (4.56 -> 87.2) and the substep count with it, 21 -> 194.

        `v0` warm-starts the iteration from the previous call's eigenvector, which is why re-measuring
        every ten frames costs a fraction of the first measurement.
        """
        x = self.x.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(self.energy(x), x, create_graph=True)
        v = torch.randn_like(x) if v0 is None or v0.shape != x.shape else v0.clone()
        v = v / v.norm().clamp_min(1e-30)
        lam = 0.0
        for _ in range(iters):
            (hv,) = torch.autograd.grad((g * v).sum(), x, retain_graph=True)
            lam = float(hv.norm())
            v = hv / max(lam, 1e-30)
        return (lam, v.detach()) if return_vec else lam

    # -- the step --------------------------------------------------------------------------------
    def step(self, dt, extra_force=None):
        """One overdamped step. `extra_force` is whatever is loading the sheet -- 05a's tether, 05b's
        plaques -- and is passed in rather than owned, so the sheet never knows what is pulling it."""
        f = self.elastic_force(self.x)
        if extra_force is not None:
            f = f + extra_force
        self.advance(dt * self.M * f, dt)
        return f

    def advance(self, dx, dt):
        """Apply a displacement and age the material by it. Separate from `step` because 05b solves
        its own tangential friction semi-implicitly and hands the result in: an explicit -xi*v is a
        stiffness of xi/dt, which GROWS as the substep shrinks, so refining the step to stabilise it
        makes it worse. Every state update that depends on where the sheet ended up lives here, so it
        happens exactly once per step whoever computed the displacement."""
        self.v = dx / dt
        self.x = self.x + dx
        if self.beta > 0:
            # E(lambda) = E0 [1 + beta (lambda - 1)]: the tangent modulus rises with the principal
            # stretch (Candiello 2007 measures 0.4 -> 3 MPa on native BM). Applied explicitly, from
            # the stretch the previous step left, so it is a state and not a hidden nonlinearity in
            # the energy -- which keeps `elastic_force` the exact gradient of `energy`.
            l1, _ = self.stretch_elastic()
            self.Y2[self.live] = self.E0 * self.T * (1.0 + self.beta * (l1 - 1.0).clamp_min(0.0))
        if self.tau_r > 0:
            C = cauchy_green(self.x, self.Fc, self.Dm_inv_seed)
            # the seeded frame's C is the geometric one; the memory creeps toward it
            self.C0[self.live] = self.C0[self.live] + (dt / self.tau_r) * (C - self.C0[self.live])


# ---------------------------------------------------------------------------------------------
#  certification
# ---------------------------------------------------------------------------------------------
def selftest(dev="cuda:0", subdiv=4, verbose=True):
    """Apply maps whose singular values are known, and report the error in the measured stretch.

    Four cases, each of which kills the measure if it fails:
      rigid      translate + rotate the whole sheet     -> lambda_1 = lambda_2 = 1
      dilation   scale by s about the centre            -> lambda_1 = lambda_2 = s, EVERY triangle
      affine     a known 3x3 map A applied to a FLAT patch -> the singular values of A's in-plane part
      energy     -dW/dx from autograd against a central finite difference of W
    The affine case is done on a flat patch and not on the sphere on purpose: an anisotropic map of a
    sphere gives a different stretch on every triangle, so agreeing with it would prove nothing about
    the measure that a plot of two curves could not also produce by accident.
    """
    out, dt_ = {}, torch.float64
    S = Sheet(subdiv=subdiv, dev=dev, dtype=dt_)
    el = (S.x[S.Ed[:, 1]] - S.x[S.Ed[:, 0]]).norm(dim=1)
    out["nodes"], out["faces"], out["edges"] = S.n, S.m, S.Ed.shape[0]
    out["edge_len_ratio"] = float(el.max() / el.min())
    # 1. rigid
    th = 0.7
    R = torch.tensor([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]],
                     device=dev, dtype=dt_)
    xr = (S.x - S.c) @ R.T + S.c + torch.tensor([0.13, -0.02, 0.4], device=dev, dtype=dt_)
    l1, l2 = S.stretch_geo(xr)
    out["rigid_max_err"] = float(max((l1 - 1).abs().max(), (l2 - 1).abs().max()))
    # 2. uniform dilation
    s = 3.397714
    l1, l2 = S.stretch_geo(S.c + (S.x - S.c) * s)
    out["dilation_s"] = s
    out["dilation_max_err"] = float(max((l1 - s).abs().max(), (l2 - s).abs().max()))
    out["dilation_J_max_err"] = float((l1 * l2 - s * s).abs().max())
    # 3. anisotropic affine on a flat patch
    nx = 33
    g = torch.linspace(-0.2, 0.2, nx, device=dev, dtype=dt_)
    GX, GY = torch.meshgrid(g, g, indexing="ij")
    Xf = torch.stack([GX.reshape(-1), GY.reshape(-1), torch.zeros(nx * nx, device=dev, dtype=dt_)], 1)
    ii = torch.arange(nx * nx, device=dev).reshape(nx, nx)
    q = torch.stack([ii[:-1, :-1].reshape(-1), ii[1:, :-1].reshape(-1),
                     ii[1:, 1:].reshape(-1), ii[:-1, 1:].reshape(-1)], 1)
    Ff = torch.cat([q[:, [0, 1, 2]], q[:, [0, 2, 3]]], 0)
    A = torch.tensor([[2.4, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 1.0]], device=dev, dtype=dt_)
    Dm, _ = reference_frame(Xf, Ff)
    C = cauchy_green(Xf @ A.T, Ff, torch.linalg.inv(Dm))
    l1, l2 = principal_stretches(C)
    sv = torch.linalg.svdvals(A[:2, :2])
    out["affine_singular_values"] = [float(v) for v in sv]
    out["affine_max_err"] = float(max((l1 - sv[0]).abs().max(), (l2 - sv[1]).abs().max()))
    # 4. force is the gradient of the energy
    S2 = Sheet(subdiv=2, dev=dev, dtype=dt_)
    S2.x = S2.x + 0.004 * torch.randn(S2.n, 3, device=dev, dtype=dt_, generator=None)
    f_an = S2.elastic_force(S2.x)
    h, err, scale = 1e-7, 0.0, 0.0
    gen = torch.Generator(device="cpu").manual_seed(0)
    for i in torch.randint(0, S2.n, (12,), generator=gen).tolist():
        for k in range(3):
            xp, xm = S2.x.clone(), S2.x.clone()
            xp[i, k] += h; xm[i, k] -= h
            fd = -(S2.energy(xp) - S2.energy(xm)) / (2 * h)
            err = max(err, abs(float(fd) - float(f_an[i, k])))
            scale = max(scale, abs(float(fd)))
    out["force_vs_fd_max_abs_err"] = err
    out["force_vs_fd_rel_err"] = err / max(scale, 1e-30)
    out["spectral_rate_at_rest"] = S.spectral_rate()
    out.update(refine_test(dev=dev, subdiv=min(subdiv, 3)))
    if verbose:
        print(f"[bm_ops selftest] subdiv {subdiv}: {out['nodes']} nodes, {out['faces']} faces, "
              f"{out['edges']} edges; edge length max/min {out['edge_len_ratio']:.4f}")
        print(f"  rigid motion         max |lambda - 1|   = {out['rigid_max_err']:.3e}")
        print(f"  dilation by {s:.4f}   max |lambda - s|   = {out['dilation_max_err']:.3e}"
              f"   (J: {out['dilation_J_max_err']:.3e})")
        print(f"  affine {out['affine_singular_values'][0]:.4f}/"
              f"{out['affine_singular_values'][1]:.4f}  max |lambda - sv| = "
              f"{out['affine_max_err']:.3e}")
        print(f"  force vs finite difference: {out['force_vs_fd_rel_err']:.3e} relative")
        print(f"  Hessian lambda_max at rest = {out['spectral_rate_at_rest']:.6g} "
              f"(the overdamped bound is dt*M*lambda_max < 2)")
        print(f"  refine, unloaded      max |d lambda|     = {out['refine_unloaded_dlambda']:.3e}"
              f"   [G14]")
        print(f"  refine, LOADED        max |d lambda|     = {out['refine_loaded_dlambda']:.3e}"
              f"   area {out['refine_loaded_darea_rel']:.3e}, energy "
              f"{out['refine_loaded_denergy_rel']:.3e}   [G15]")
        print(f"  refine, conformity    edges with 3+ faces = {out['refine_bad_edges']}, rim "
              f"{out['refine_rim_edges']}   [G16]")
        print(f"  refine, conservation  mass {out['remesh_mass_rel']:.3e}, volume "
              f"{out['remesh_volume_rel']:.3e}, area {out['remesh_area_rel']:.3e}, centroid "
              f"{out['remesh_centroid_rel']:.3e}   [G18b, G18d]")
        print(f"  tear, conformity      rim edges opened    = {out['tear_rim_edges']} around "
              f"{out['tear_faces_killed']} dead faces   [G21 prep]")
    return out


def refine_test(dev="cuda:0", subdiv=3):
    """G14--G16: refinement must be invisible to the strain measure, and must leave no hanging node.

    THIS IS THE GATE THE WHOLE RESERVOIR DESIGN TURNS ON. A child triangle whose reference frame is
    rebuilt from its CURRENT shape reports lambda = 1 the instant it is created, so a sheet that
    refines as it grows would silently forget everything it had been stretched by -- the mesh version of
    the laundering that made run 130 report 13% of its true stretch. Inheriting Dm_inv through the
    constant map S_k instead makes parent and child report the SAME lambda, and this test is the proof.
    """
    out, dt_ = {}, torch.float64
    # unloaded
    S = Sheet(subdiv=subdiv, max_refine=1, dev=dev, dtype=dt_)
    l1a, l2a = S.stretch_geo()
    S.refine()
    l1b, l2b = S.stretch_geo()
    out["refine_unloaded_dlambda"] = float(max((l1b - l1a.mean()).abs().max(),
                                               (l2b - l2a.mean()).abs().max()))
    # loaded: an anisotropic map applied BEFORE the split, so every triangle carries a real strain
    S = Sheet(subdiv=subdiv, max_refine=1, dev=dev, dtype=dt_)
    A = torch.tensor([[2.3, 0.4, 0.0], [0.0, 1.7, 0.15], [0.0, 0.0, 1.2]], device=dev, dtype=dt_)
    S.x[S.live_nodes] = (S.x[S.live_nodes] - S.c) @ A.T + S.c
    l1a, _ = S.stretch_geo()
    area_a, en_a = float(S.area().sum()), float(S.energy(S.x))
    par = S.live.clone()
    S.refine()
    l1b, _ = S.stretch_geo()
    # compare each parent against its four children: child A reuses the parent's slot and the other
    # three are contiguous, so the mapping is known without storing it
    m0 = par.numel()
    kids = torch.cat([torch.arange(m0, device=dev),
                      torch.arange(3 * m0, device=dev) + m0]).reshape(-1)
    ref = l1a.repeat(4)
    out["refine_loaded_dlambda"] = float((l1b[kids] - ref).abs().max())
    out["refine_loaded_darea_rel"] = abs(float(S.area().sum()) - area_a) / max(area_a, 1e-30)
    out["refine_loaded_denergy_rel"] = abs(float(S.energy(S.x)) - en_a) / max(abs(en_a), 1e-30)
    ec = S.euler_check()
    out["refine_bad_edges"], out["refine_rim_edges"] = ec["bad"], ec["rim"]
    out["refine_faces"], out["refine_nodes"] = S.m, S.n
    # G18b / G18d: A REMESH IS NUMERICAL ONLY. It changes the triangulation and must change nothing
    # else -- not the mass the sheet carries, not the surface it occupies. These are the two halves of
    # that claim, and they are separate from G15 (which says the MECHANICS is unchanged) because a
    # split could preserve every stretch and still lose material or move the surface.
    S3 = Sheet(subdiv=subdiv, max_refine=1, dev=dev, dtype=dt_)
    S3.x[S3.live_nodes] = (S3.x[S3.live_nodes] - S3.c) @ A.T + S3.c
    m_a, v_a, c_a = S3.total_mass(), S3.enclosed_volume(), S3.area_centroid().clone()
    ar_a = float(S3.area().sum())
    S3.refine()
    out["remesh_mass_rel"] = abs(S3.total_mass() - m_a) / max(abs(m_a), 1e-30)
    out["remesh_volume_rel"] = abs(S3.enclosed_volume() - v_a) / max(abs(v_a), 1e-30)
    out["remesh_area_rel"] = abs(float(S3.area().sum()) - ar_a) / max(ar_a, 1e-30)
    out["remesh_centroid_rel"] = float((S3.area_centroid() - c_a).norm()) / max(
        float(c_a.norm()), 1e-30)
    # A HOLE, so that the rim is a thing that exists before 05d needs it: kill a polar cap and check
    # that the mesh is still conforming with a free boundary rather than a corrupt one.
    r = S.x[S.Fc].mean(1) - S.c
    S.tear(r[:, 2] / r.norm(dim=1) > 0.9)
    ec2 = S.euler_check()
    out["tear_rim_edges"] = ec2["rim"]
    out["tear_bad_edges"] = ec2["bad"]
    out["tear_faces_killed"] = int(4 * m0 - S.m)
    return out


if __name__ == "__main__":
    _d = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda:0"
    if not torch.cuda.is_available():
        _d = "cpu"
    selftest(dev=_d, subdiv=int(sys.argv[sys.argv.index("--subdiv") + 1])
             if "--subdiv" in sys.argv else 4)
