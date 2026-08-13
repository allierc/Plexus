"""adhesion_ops -- adhesion as a BOND DENSITY: receptor on the cell, bonds on the edge.

WHY THE DISCRETE PLAQUE WAS RETIRED. A `plaque` was one integrin nanocluster -- 20-50 integrins,
35-100 nm across (Changede & Sheetz 2017) -- and clusters sit at a measured ~555 nm centre-to-centre
spacing. On this sheet that is ~1.03 million clusters against 40,962 nodes: 25 clusters and ~3000
integrins PER NODE. Above one object per element the discrete description is the approximation and the
density is exact, so 05b's 1166 discrete plaques were the wrong object by a factor of 883.

THE THREE STATES, each on the entity that owns it:

    cell.N_f      free receptor on the basal face   an integrin is the CELL's, not the sheet's
    plaque.N_b    bound bonds, ONE EDGE PER NODE    a bond has two ends, so it is a relation
    bm_face.rho_L ligand                            laminin is sheet material; this is rho

NO DIFFUSION OPERATOR, and that is a result rather than a convenience. The plasma membrane is
continuous within a cell and interrupted at cell-cell junctions, so integrins do not diffuse between
cells; transport between them is exo/endocytosis, a per-cell source and sink. WITHIN one cell,
sqrt(4 D t) = 15.5 um at D = 0.1 um^2/s over a 600 s frame against a ~10 um cell: the basal face is
WELL MIXED at our timestep. That removes the Laplacian and the dt*D/h^2 < 1/2 bound -- which would have
cost ~14 substeps and, unlike the sheet's rate bound, would have TIGHTENED at every refinement. The
assumption is stated, not silent: resolve a cell with more than one face and the diffusion term returns.

UNITS. No force scale is declared for this prototype (plexus/units.py), so bond numbers are normalised:
n_max = 1 is a saturated patch, and N_f is in the same units. Only ratios are quoted. Writing an
absolute integrin count would be fake precision on a sheet whose thickness is already 23x too large.
"""
from __future__ import annotations

import torch


class Clutch:
    """The adhesion, as a bond density on the contact map -- one edge per live sheet node.

    Replaces `Plaques`. The edge set is not seeded and never re-seeded: it IS the contact map, which
    `bm_contact` already rebuilds on refinement, so an adhesion exists everywhere the sheet touches the
    cell layer and nowhere else. What varies is how many bonds each patch is holding.
    """

    def __init__(self, kappa_b=5.0, l0=6.0e-4, k_on=0.6, k_off0=0.05, f_bell=3.0e-3, n_max=1.0,
                 dev="cuda:0", dtype=torch.float64):
        self.kappa_b, self.l0 = float(kappa_b), float(l0)
        # f_bell IS A LOAD, so it has to be set against a length the bond actually reaches. The
        # first choice, 0.02, needed a stretch of 4.7 um to engage -- larger than the 3 um node
        # spacing -- so k_off never rose above k_off0 and Bell's law was inert. A bond breaks when it
        # is stretched by something like its own length, so f_bell ~ kappa_b * l0 = 3e-3 here.
        self.k_on, self.k_off0, self.f_bell = float(k_on), float(k_off0), float(f_bell)
        self.n_max = float(n_max)
        self.dev, self.dtype = dev, dtype
        self.Nb = None            # bonds per edge (one edge per live sheet node)
        self.Nf = None            # free receptor per CELL (one per epithelial face)
        # THE MEAN BOND STRETCH, and it is the state the first version was missing. A bond resists
        # displacement in EVERY direction from where it formed, not only along the normal: a force
        # that is purely normal is a roller, and the sheet then slides freely whatever k_off is --
        # which is exactly what the first slip sweep measured (3.747 deg of slip at k_off = 0 against
        # 3.748 at k_off = 0.5, i.e. no dependence at all). Bonds form UNSTRETCHED, so the patch's
        # mean stretch grows with the relative sliding and is reset by the arrival of new bonds:
        #     d(Delta)/dt = v_rel - (on/N_b) Delta
        # `off` removes bonds at the current mean and so does not change it. This is the clutch: the
        # sheet advances because loaded bonds let go and unloaded ones take their place.
        self.D = None             # (n_edges, 3) mean tangential stretch of the bonds in a patch

    # -- allocation ------------------------------------------------------------------------------
    def provision(self, n_edges, n_cells, Nf0=1.0):
        self.Nb = torch.zeros(n_edges, device=self.dev, dtype=self.dtype)
        self.D = torch.zeros(n_edges, 3, device=self.dev, dtype=self.dtype)
        self.Nf = torch.full((n_cells,), float(Nf0), device=self.dev, dtype=self.dtype)

    def regrid(self, n_edges, old_node, new_node):
        """A refinement changes the edge set (one per node), so the bonds have to be carried over.
        A node that existed keeps its bonds; a node born in the split starts with none and binds from
        its cell's free pool like any other patch -- which is the honest statement, since new membrane
        is not born pre-adhered."""
        nb = torch.zeros(n_edges, device=self.dev, dtype=self.dtype)
        dd = torch.zeros(n_edges, 3, device=self.dev, dtype=self.dtype)
        if self.Nb is not None and old_node is not None:
            keep = torch.zeros(int(new_node.max()) + 1, dtype=torch.long, device=self.dev)
            keep[new_node] = torch.arange(new_node.numel(), device=self.dev)
            live_old = old_node[old_node <= new_node.max()]
            idx_old = torch.arange(old_node.numel(), device=self.dev)[old_node <= new_node.max()]
            nb[keep[live_old]] = self.Nb[idx_old]
            dd[keep[live_old]] = self.D[idx_old]
        self.Nb, self.D = nb, dd

    # -- the operators ---------------------------------------------------------------------------
    def force(self, x_bm, p, nhat):
        """`plaque_pull` as a density: a spring per bond, in three dimensions.

        The rest configuration of a bond is the ligand sitting l0 OUTSIDE its receptor along the
        surface normal, so the extension is the full vector

            Delta_total = (x_bm - p) - l0 * nhat   +   Delta_tangential(the patch's memory)

        and the force is N_b springs on it. Taking only the normal component -- which the first
        version did -- makes the adhesion a roller: it holds the sheet off the cells and lets it slide
        freely, so k_off has nothing to act on. The load per bond is independent of how many bonds
        share it, which is what lets Bell's law read a force that means something.
        """
        dvec = (x_bm - p) - self.l0 * nhat + self.D
        f = -(self.Nb[:, None] * self.kappa_b) * dvec
        return f, self.kappa_b * dvec.norm(dim=1)

    def slide(self, dt, v_rel, nhat, on_per_bond):
        """Advance the patch's mean bond stretch. New bonds arrive unstretched and dilute the mean;
        sliding grows it. Only the TANGENTIAL part is carried -- the normal offset is geometry and is
        already in `force`."""
        vt = v_rel - (v_rel * nhat).sum(1, keepdim=True) * nhat
        self.D = (self.D + dt * vt) / (1.0 + dt * on_per_bond)[:, None]
        self.D = self.D - (self.D * nhat).sum(1, keepdim=True) * nhat

    def bind(self, dt, cell_of_edge, area_edge, rho_L, f_per_bond, area_cell):
        """`plaque_bind` + `integrin_turnover`, one step of Eq. (clutch).

        RECEPTOR IS MOVED, NOT CREATED: every bond formed is subtracted from its cell's free pool and
        every bond broken is returned to it, so N_f + sum N_b is invariant under binding by
        construction. That is what makes G30 a test of the implementation rather than of the algebra.
        """
        n_b = self.Nb / area_edge.clamp_min(1e-30)
        n_f = (self.Nf / area_cell.clamp_min(1e-30))[cell_of_edge]
        on = self.k_on * n_f * rho_L * (1.0 - n_b / self.n_max).clamp_min(0.0) * area_edge * dt
        # a cell cannot hand out more receptor than it has: cap the demand per cell, proportionally
        demand = torch.zeros_like(self.Nf)
        demand.index_add_(0, cell_of_edge, on)
        scale = torch.ones_like(self.Nf)
        hot = demand > self.Nf
        scale[hot] = (self.Nf[hot] / demand[hot].clamp_min(1e-30))
        on = on * scale[cell_of_edge]
        k_off = self.k_off0 * torch.exp((f_per_bond / self.f_bell).clamp(max=40.0))
        off = (k_off * self.Nb * dt).clamp(max=self.Nb)
        # a bond that breaks takes its share of the stretch with it and a bond that forms adds none,
        # so the MEAN stretch is diluted by arrivals only -- handled in `slide`, which needs the
        # per-bond arrival rate this line returns
        self.on_per_bond = on / dt / self.Nb.clamp_min(1e-12)
        self.Nb = self.Nb + on - off
        net = torch.zeros_like(self.Nf)
        net.index_add_(0, cell_of_edge, off - on)
        self.Nf = (self.Nf + net).clamp_min(0.0)
        return float(on.sum()), float(off.sum()), k_off

    def turnover(self, dt, s_i, tau_i):
        """The cell makes and recycles its own receptor. This is the ONLY thing that changes the
        total; binding merely moves it between the free and the bound column."""
        if tau_i <= 0 and s_i <= 0:
            return
        self.Nf = self.Nf + dt * (s_i - (self.Nf / tau_i if tau_i > 0 else 0.0))

    def total(self, cell_of_edge):
        return float(self.Nf.sum() + self.Nb.sum())
