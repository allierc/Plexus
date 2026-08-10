"""medioapical_ops -- the SECOND myosin pool, and what having two of them fixes.

WHAT THE ONE-POOL MODEL GETS WRONG, AND IT IS NOT A DETAIL. `junction_myosin` carries a number `m_e`
per cell--cell contact and multiplies the line tension by it. That number is a DENSITY -- myosin per
unit length of the junctional belt -- because the energy term it multiplies, `Lambda * m_e * l_e`, is
a tension times a length. The feedback that sets it, however, is keyed to `l_e`, the junction's own
length, and length is EXTENSIVE: when `divide_3d` inserts a vertex into an existing edge (a,b), the
contact is cut into (a,n) and (n,b), each about half as long -- and NOTHING PHYSICAL HAS HAPPENED. The
same two cells still touch over the same interface; only the mesh's description of it changed.

So a length-keyed setpoint reads a 50% collapse in its own drive at every division, and the myosin on
a junction that was never disturbed decays toward half its value over `tau`. That is a re-meshing
artifact wearing the clothes of a mechanism. The rule it violates is simple and general: THE DRIVE OF
A FEEDBACK MUST BE INTENSIVE. Tension is (cutting a cable in two does not halve the tension in it);
strain and strain rate are; length is not.

THE FIX IS NOT A BETTER DRIVE, IT IS A CONSERVED AMOUNT. Myosin is a protein: what the model should
carry is an AMOUNT, and the density should be derived from it. Then a cut splits the amount in
proportion to length, the density is unchanged on both halves, and the setpoint the halves relax
toward is unchanged too, because the supply that sets it is per unit length. Nothing has to know that
a division happened.

    N_e (amount on junction e)  =  n_e (density) * l_e        <- what is stored is n_e
    M_f (amount on cell f)      =  rho_f (density) * A_f      <- what is stored is rho_f

STORING THE DENSITY IS WHAT MAKES THE EXISTING CARRIES CORRECT. A junction's state is inherited across
a division by COPYING the parent's value onto both halves (`junction_ops._lookup`), and a face's state
is inherited by copying the parent face's value onto both daughters (`divide_3d`'s `keep` map). For a
density both copies conserve the amount exactly, since l_an + l_nb = l_ab and A_d1 + A_d2 = A_f. For
an amount both would DOUBLE it. The same physical statement -- myosin is conserved through a division
-- is two different lines of code depending on which of the two the state is, and choosing wrongly is
invisible: the arrays are the same shape and the run does not fail.

WHY A SECOND POOL AT ALL, biologically. An epithelial cell has two distinct apical actomyosin
populations, and germband extension uses both (Munjal, Philippe, Munro & Lecuit, Nature 524:351,
2015): a MEDIOAPICAL meshwork spread over the apical face, which pulses, and a JUNCTIONAL belt at the
adherens junction, which is a cable along the apical rim. They are coupled by flow -- medioapical
pulses move outward and deliver myosin onto the junctions. The first is AREAL and the second is
LINEAR, which is the distinction the question "should myosin scale with area or with length?" is
really about: BOTH, but they are two different pools.

WHAT IS HERE AND WHAT IS NOT. Here: the two pools, the flux between them, turnover on each, and an
optional tension bias on where the flux lands. Not here: pulsatility. Munjal's pulses come from
advection-mediated positive feedback (myosin flow concentrates myosin, which raises the flow) with
dissociation as the negative arm; that is a third state variable and an advection term, and it is the
next operator, not this one. This one is transport plus turnover, which is the control that
pulsatility has to beat.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral, Structural
from plexus.models.registry import register_operator

import junction_ops as JO
from tyssue_ops3d import face_geometry_3d

# Per frame: (total medioapical amount, total junctional amount, mean areal density, mean line
# density). The first two are the conservation ledger -- a transport model that quietly creates
# myosin shows up here and nowhere else.
POOL_TRACE: list = []


def _face_carry(m, name):
    """Ask the topology operators to carry a per-face array across a rebuild.

    `divide_3d` and `apoptosis_3d` reindex every per-face array through the `keep` map (new face ->
    old face) so nothing is left pointing at a face that has moved. The set of names they carry was a
    literal tuple, so a per-face state added by a new operator was silently dropped -- the same class
    of defect as the per-half-edge myosin that `junction_myosin_sync` exists to fix, one level up.
    `face_carry` makes the list open: an operator declares its own array once and the topology
    operators still know nothing about what is in it.
    """
    m.setdefault("face_carry", set()).add(name)


@register_operator("medioapical_myosin", family="mechanics", set="cell", kind="lateral")
class MedioapicalMyosin(Lateral):
    """The apical meshwork: an AREAL myosin density on each cell, and its flux onto that cell's
    junctions.

        M_f  = rho_f * A_f                                  amount, from the stored areal density
        dM_f/dt = k_on * A_f  -  M_f / tau_med  -  sum_e J_(f->e)
        J_(f->e) = k_ex * rho_f * l_e * (1 + beta_T (T_e/<T> - 1))

    EVERY TERM IS PER WHAT. `k_on` is myosin assembled per unit APICAL AREA per frame, so a cell that
    grows assembles more, which is what makes this pool areal rather than a per-cell budget.
    `tau_med` is the meshwork's own turnover time IN FRAMES. `k_ex` is the fraction of the cell's
    areal density handed to each unit LENGTH of its junctional belt per frame -- the flux is
    proportional to `l_e` because the belt is what receives it and there is `l_e` of it.

    THE SHAPE FACTOR IS A PREDICTION, NOT A PARAMETER. At steady state
    rho* = k_on / (1/tau_med + k_ex * P_f/A_f), so a cell with more perimeter per unit area drains
    faster and holds less medioapical myosin. P_f/A_f is fixed by shape alone, so this model says
    elongated cells are medioapically poorer than round ones of the same area, with nothing added to
    say so.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = False        # writes m["myo_med"] / m["myo_influx"], not positions
    MECHANISM_TAGS = ["actomyosin_contraction", "medioapical_pool", "cortical_flow",
                      "topology_persistent"]
    PARAM_ROLES = {"k_on": "areal_assembly_rate", "tau_med": "medioapical_turnover_time",
                   "k_ex": "export_rate_onto_the_belt", "beta_T": "tension_bias_of_the_export"}
    REFERENCE = ("Munjal, A., Philippe, J.-M., Munro, E. & Lecuit, T. (2015) Nature 524:351 "
                 "(two apical pools, medioapical pulses flowing onto junctions); "
                 "Rauzi, M. et al. (2010) Nature 468:1110 (planar-polarised junctional myosin).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        # ACTS ON `cell` AND READS THE MESH, which lives on the vertex Level. The mesh is the only
        # place the face incidence exists, so the operator is declared on the set whose state it owns
        # and reaches the geometry through `mesh_at`.
        self.at = params.get("_at", "cell")
        self.mesh_at = params.get("mesh_at", "vertex")
        self.k_on = float(params.get("k_on", 0.05))
        self.tau_med = float(params.get("tau_med", 20.0))
        self.k_ex = float(params.get("k_ex", 0.05))
        self.beta_T = float(params.get("beta_T", 0.0))
        self.rho0 = float(params.get("rho0", 1.0))
        self.dt = float(params.get("dt", 1.0))
        self.lam = float(params.get("lam", 1.0))
        self.k_perim = float(params.get("k_perim", 1.0))
        self.gam = float(params.get("gam", 0.0))
        self._said = False

    def forward(self, H, mask=None):
        lvl = H.level(self.mesh_at)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        es, live, vi, vj, stride, key, length = JO._live_edges(m, pos)
        if not bool(live.any()):
            return {}
        nF = int(m["nF"])
        ef = m["E_face"][live].long()
        # THE SAME AREA `shape_energy_3d` MINIMISES AGAINST -- the Newell area-vector magnitude, from
        # the same function, not a re-derivation. `eocc` masks the dead half-edges exactly as the
        # energy's live-only path does.
        eocc = live.to(dt_)
        area, _, _, _ = face_geometry_3d(pos, m["E_srce"].long(), m["E_trgt"].long(),
                                         m["E_face"].long().clamp(max=nF - 1), nF, eocc)
        area = area.clamp_min(1e-9)

        # ---- the stored state is a DENSITY, so `divide_3d`'s copy conserves the amount ------------
        rho = m.get("myo_med")
        if rho is None or rho.shape[0] < nF:
            rho = torch.full((nF,), self.rho0, device=dev, dtype=dt_)
        rho = rho[:nF].to(dt_)
        _face_carry(m, "myo_med")

        # ---- where the export lands ---------------------------------------------------------------
        # With `beta_T` at 0 this is pure transport: every unit of belt receives at the same rate and
        # the pattern of junctional myosin is set by geometry alone. That is deliberately the control
        # -- a tension bias added on top has to beat it, and cannot be credited with what transport
        # already does.
        w = torch.ones_like(length)
        if self.beta_T != 0.0:
            myo_now = m.get("myo")
            myo_l = (myo_now[live].to(dt_) if myo_now is not None and myo_now.shape[0] == es.shape[0]
                     else torch.ones_like(length))
            T = JO.edge_tension(m, length, myo_l, ef, self.lam, self.k_perim, self.gam, dev, dt_)
            w = (1.0 + self.beta_T * (T / T.mean().clamp_min(1e-9) - 1.0)).clamp_min(0.0)

        # ---- the flux, per half-edge, and its reaction on the cell --------------------------------
        # A HALF-EDGE HAS EXACTLY ONE FACE, so `J` computed per half-edge IS the flux from that face
        # onto that piece of belt, and a junction -- two half-edges, one per cell -- collects a
        # contribution from BOTH of the cells it separates. That is the geometry doing the
        # bookkeeping: nothing has to look up "the other cell".
        J = self.k_ex * rho[ef] * length * w * self.dt
        drain = torch.zeros(nF, device=dev, dtype=dt_).index_add(0, ef, J)

        M = rho * area
        M = M + (self.k_on * area - M / max(self.tau_med, 1e-9)) * self.dt - drain
        rho = (M / area).clamp_min(0.0)
        m["myo_med"] = rho.detach()
        m["myo_med_total"] = float((rho * area).sum())          # the amount, for the ledger
        # WHAT THE JUNCTION OPERATOR CONSUMES: an amount per half-edge, full-length so it is indexed
        # the same way `myo` is, and re-derived every frame so it can never be stale. Filled with 0 on
        # dead slots: an amount, not a multiplier.
        m["myo_influx"] = JO._scatter_full(es, live, J.detach(), dev, dt_, fill=0.0)
        if not self._said:
            print(f"[medioapical_myosin] {nF} cells, k_on={self.k_on}, tau_med={self.tau_med}, "
                  f"k_ex={self.k_ex}, beta_T={self.beta_T}; stored as an AREAL DENSITY so a division "
                  f"conserves the amount", flush=True)
            self._said = True
        return {}


@register_operator("junction_myosin", family="mechanics", set="vertex", kind="structural",
                   model="two_pool")
class JunctionMyosinTwoPool(Structural):
    """The junctional belt fed by the medioapical pool: a conserved AMOUNT, a DERIVED density.

        N_e = n_e * l_e                                  amount, from the stored line density
        dN_e/dt = sum_(f in e) J_(f->e)  -  N_e / tau_jun
        m_e = a * n_e / <n_e>                             what the mechanics multiplies Lambda by

    A DIFFERENT MODEL OF THE SAME CONTRACT, not a different implementation of it. `junction_myosin`'s
    default model recruits from a mechanical drive with parameters {activity, tau, beta, keyed_on};
    this one receives a flux from another pool with parameters {tau_jun, activity} and has no drive
    at all. Per plexus2 those are disjoint parameter sets and therefore two biological hypotheses in
    one slot, which is what the `model=` axis is for -- swapping them is an experiment. They share the
    contract's kind and the state channel `m["myo"]`, so `shape_energy_3d` cannot tell them apart and
    the comparison is at the same operating point.

    WHY THE DENSITY IS NORMALISED. `m_e` multiplies `Lambda`, and `Lambda` is calibrated against a
    tissue whose mean multiplier is 1. Dividing by the tissue mean fixes the overall level at
    `activity` and leaves this model predicting the PATTERN of junctional myosin, which is the part it
    is entitled to predict; the absolute level is set by `Lambda * activity` as it always was. Without
    it, `k_on` and `k_ex` would be a second, hidden line tension.

    THE PROPERTY THIS EXISTS FOR. `n_e` is intensive and its supply is per unit length, so at a
    division the two halves of a cut junction keep the density they had AND relax toward the setpoint
    they were already at. The one-pool model keeps the first and loses the second.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = False
    MECHANISM_TAGS = ["actomyosin_contraction", "junction_state", "cortical_flow",
                      "topology_persistent", "conserved_amount"]
    PARAM_ROLES = {"tau_jun": "junctional_turnover_time", "activity": "global_myosin_activity",
                   "myo_new": "line_density_of_a_newborn_junction"}
    REFERENCE = ("Munjal, A. et al. (2015) Nature 524:351 (medioapical -> junctional flow); "
                 "Curran, S. et al. (2017) Dev. Cell 43:480 (junctional myosin fluctuations).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.activity = float(params.get("activity", 1.0))
        self.tau_jun = float(params.get("tau_jun", 20.0))
        self.myo_new = float(params.get("myo_new", 1.0))
        self.inherit = bool(params.get("inherit", True))
        self.dt = float(params.get("dt", 1.0))
        self._said = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        es, live, vi, vj, stride, key, length = JO._live_edges(m, pos)
        if not bool(live.any()):
            return {}
        influx = m.get("myo_influx")
        if influx is None or influx.shape[0] != es.shape[0]:
            raise RuntimeError(
                "junction_myosin[two_pool] has no flux to integrate: `medioapical_myosin` must be "
                "scheduled before it, on the same frame. Refusing to run the belt with an influx of "
                "zero, which would decay to an empty junction and look like a turnover result.")
        J = influx[live].to(dt_)

        # ONE JUNCTION, NOT TWO HALF-EDGES. A cell--cell contact appears twice in the half-edge arrays,
        # once from each of the cells it separates, and it is ONE belt with ONE myosin density -- so
        # the two cells' fluxes are SUMMED onto it and integrated once. Integrating the two half-edges
        # separately gives a junction two different densities, and since the keyed store is indexed by
        # the vertex pair, the two would then collide on one key and `searchsorted` would return
        # whichever happened to sort first. The bug is silent, order-dependent, and exactly the class
        # `junction_myosin_sync` was written to remove; it is avoided here by never creating it.
        uq, inv = torch.unique(key, return_inverse=True)
        z = torch.zeros(uq.numel(), device=dev, dtype=dt_)
        J_j = z.clone().index_add(0, inv, J)                        # both cells feed the same belt
        cnt = z.clone().index_add(0, inv, torch.ones_like(length))
        l_j = z.clone().index_add(0, inv, length) / cnt.clamp_min(1.0)

        # THE STORED VALUE IS THE LINE DENSITY, and `_lookup` copies it onto both halves of a cut
        # junction. Multiplying by the CURRENT length is what turns that copy into a conservative
        # split: N_an + N_nb = n*(l_an + l_nb) = n*l_ab = N_ab, to the accuracy with which the
        # inserted vertex lies on the parent edge.
        n_e, _ = JO._lookup(m, key, length, vi, vj, stride, self.myo_new, self.inherit, dev, dt_)
        n_j = (z.clone().index_add(0, inv, n_e) / cnt.clamp_min(1.0))
        N_j = n_j * l_j
        N_j = N_j + J_j - N_j * (self.dt / max(self.tau_jun, 1e-9))
        n_j = (N_j / l_j.clamp_min(1e-9)).clamp_min(0.0)
        n_e = n_j[inv]
        N = N_j[inv] / cnt.clamp_min(1.0)[inv]      # the junction's amount, split back over its halves

        # THE STORE HAS ONE ENTRY PER JUNCTION, not one per half-edge. Duplicate keys in a sorted array
        # make `searchsorted` ambiguous whenever their values differ, which is a latent hazard the
        # one-pool operator only escapes because its drive depends on length alone.
        m["myo_keys"], m["myo_vals"] = uq.detach().clone(), n_j.detach().clone()
        vs_seen = torch.zeros(stride, dtype=torch.bool, device=dev)
        vs_seen[vi] = True; vs_seen[vj] = True
        m["myo_vseen"] = vs_seen
        # THE GAIN, LEFT ON THE MESH FOR `junction_myosin_sync` TO REAPPLY. The store is a density and
        # the mechanics wants a multiplier whose tissue mean is `activity`; putting the conversion on
        # the mesh rather than baking it into `myo` is what lets one sync operator serve both models.
        gain = self.activity / n_e.mean().clamp_min(1e-9)
        m["myo_gain"] = float(gain)
        m["myo"] = JO._scatter_full(es, live, (gain * n_e).clamp(0.0, 5.0), dev, dt_)
        m["myo_amount"] = JO._scatter_full(es, live, N.detach(), dev, dt_, fill=0.0)
        POOL_TRACE.append((float(m.get("myo_med_total", 0.0)), float(N.sum()),
                           float(n_e.mean()), float((gain * n_e).mean())))
        if not self._said:
            print(f"[junction_myosin/two_pool] {int(live.sum())} junctions fed by the medioapical "
                  f"pool, tau_jun={self.tau_jun}; the stored state is a LINE DENSITY and the "
                  f"integrated one is an AMOUNT", flush=True)
            self._said = True
        return {}
