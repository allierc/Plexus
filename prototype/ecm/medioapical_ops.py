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
        # BOTH ARE DENSITIES, so `divide_3d`'s copy-onto-both-daughters conserves them. `myo_area` is
        # NOT declared here on purpose: an area is extensive, copying it would give two daughters the
        # mother's area each, and anything needing it after a division must recompute it.
        _face_carry(m, "myo_med")
        _face_carry(m, "myo_nstar_per_tau")

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
        m["myo_area"] = area.detach()
        m["myo_k_ex"] = self.k_ex
        # THE LOCAL STEADY STATE OF THE BELT, PER CELL. Setting dN/dt = 0 in the junctional equation
        # with the flux from ONE side gives n* = tau_jun * k_ex * rho_f, which is the density a mature
        # junction of this cell carries. It is the only scale in the model against which "a newborn
        # junction starts weak" or "the cytokinetic ring is myosin-rich" can be stated as a FRACTION
        # rather than as an absolute number that silently means something different at frame 400 than
        # at frame 0. `tau_jun` belongs to the junction operator, so what is left here is the part this
        # operator owns and the consumer multiplies by its own tau.
        m["myo_nstar_per_tau"] = (self.k_ex * rho).detach()
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
        self.new_rel = bool(params.get("myo_new_rel", True))
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
        # A NEWBORN JUNCTION IS A FRACTION OF WHAT A MATURE ONE HERE WOULD CARRY, not an absolute
        # number. `myo_new` used to be a bare density, and because the tissue's mean line density runs
        # 1.07 -> 1.97 -> 1.48 over the run, `myo_new = 1.0` set a new junction to 51% of its
        # neighbours at frame 100 and 93% at frame 0 -- a visible, meaningless dimming that had nothing
        # to do with the parameter's intent.
        nsp = m.get("myo_nstar_per_tau")
        ef_l = m["E_face"][live].long()
        n_star = (self.tau_jun * nsp[ef_l] if nsp is not None
                  else torch.ones_like(length))
        # A FLAG AND NOT A REPLACEMENT, because the absolute reading is the one every cache written
        # before 10 August was built with, and a silent change of meaning under an unchanged cache key
        # is how an archived run stops being reproducible without anything failing. `False` reproduces
        # those bit-for-bit; `True` is the corrected default and is what enters the cache key.
        new_val = (self.myo_new * n_star if self.new_rel
                   else torch.full_like(length, self.myo_new))
        n_e, _ = JO._lookup(m, key, length, vi, vj, stride, self.myo_new, self.inherit, dev, dt_,
                            new_val=new_val)
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


@register_operator("cytokinetic_ring", family="mechanics", set="vertex", kind="structural")
class CytokineticRing(Structural):
    """The myosin a division leaves on the junction it just built, taken from the cortex that built it.

    WHY THE MODEL NEEDED THIS, AND IT WAS VISIBLE IN A MOVIE BEFORE IT WAS MEASURED. In the two-pool
    run the brightest junctions at a division site are the two HALVES of a contact the division cut
    (m/<m> = 1.066, because a cell about to divide is large, has a low perimeter-to-area ratio, holds
    more cortical myosin and feeds its belts harder) -- while the daughter--daughter interface, the one
    place a real dividing cell puts almost all of its myosin, is the DIMMEST thing in the frame at
    0.628. The cytokinetic ring is the most myosin-II-rich structure a cell assembles and it
    constricts exactly there; in epithelia the nascent adherens junction between the daughters is
    built out of it, with the neighbouring cell contributing (Herszterg et al., Dev. Cell 24:256,
    2013; Founounou et al., Dev. Cell 24:242, 2013). A model whose division sites are bright for the
    wrong reason and dark in the right place is getting this backwards twice.

    WHAT IT DOES. A junction born with BOTH endpoints new is a daughter--daughter interface -- exactly
    the distinction `_lookup`'s inheritance already draws, since a split half has one new endpoint and
    one old one. Those junctions are written into the keyed store at

        n_e = ring * n*_f,        n*_f = tau_jun * k_ex * rho_f

    i.e. `ring` times the density a mature junction of that cell carries, and the myosin is DEBITED
    from the medioapical pool of the adjacent cell. That second half is not bookkeeping pedantry: the
    ring is assembled from cortical actomyosin, so a ring that appears from nowhere would be a source
    term in a model whose whole point is that myosin is a conserved amount, and the conservation
    ledger would stop being able to detect a leak.

    IT RELAXES ON ITS OWN. Nothing here decays the deposit: the junctional equation already does,
    with dN/dt = J - N/tau_jun pulling the interface from `ring * n*` back to `n*` over tau_jun. So
    `ring` sets how bright a new junction starts and tau_jun sets how long it stays that way, and
    neither needs a second timescale invented for it.

    WHERE IT GOES IN THE SCHEDULE. After the topology operators, before `junction_myosin_sync`:

        ... -> reconnect_t1_3d -> divide_3d -> cytokinetic_ring -> junction_myosin_sync
                                                                -> topo_snapshot_3d

    It must see the new vertices, which means after `divide_3d`, and it must read `myo_vseen` from the
    PREVIOUS frame's `junction_myosin` to know which vertices are new -- which it does, because
    nothing between the two writes it. Placing it before the sync operator is what makes the deposit
    visible in the frame it happens rather than the frame after.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = False
    MECHANISM_TAGS = ["actomyosin_contraction", "cytokinesis", "junction_state", "conserved_amount"]
    PARAM_ROLES = {"ring": "newborn_junction_density_as_a_multiple_of_the_local_steady_state",
                   "tau_jun": "the_belt_turnover_that_relaxes_it_back"}
    REFERENCE = ("Herszterg, S. et al. (2013) Dev. Cell 24:256 and Founounou, N. et al. (2013) "
                 "Dev. Cell 24:242 (the daughter--daughter junction is built from the cytokinetic "
                 "ring, with the neighbouring cell contributing).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.ring = float(params.get("ring", 3.0))
        self.tau_jun = float(params.get("tau_jun", 20.0))
        self.debit = bool(params.get("debit", True))
        self._said = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        if m is None or m.get("myo_keys") is None or m.get("myo_nstar_per_tau") is None:
            return {}
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        es, live, vi, vj, stride, key, length = JO._live_edges(m, pos)
        if not bool(live.any()):
            return {}
        vseen = m.get("myo_vseen")
        if vseen is None:
            return {}
        # BOTH ENDPOINTS NEW = the interface between the two daughters. One new and one old is a split
        # half of a pre-existing contact, which has a history and must not be overwritten with a ring.
        oi = vseen[vi.clamp(max=vseen.numel() - 1)] & (vi < vseen.numel())
        oj = vseen[vj.clamp(max=vseen.numel() - 1)] & (vj < vseen.numel())
        fresh = (~oi) & (~oj)
        if not bool(fresh.any()):
            RING_TRACE.append((0, 0.0))
            return {}
        nF = int(m["nF"])
        ef = m["E_face"][live].long()
        nsp = m["myo_nstar_per_tau"]
        if nsp.shape[0] < nF:
            # THE CARRY DID NOT HAPPEN, which means this operator is scheduled somewhere `divide_3d`
            # has grown the face count without `_carry_face_state` running. Said rather than clamped:
            # a clamp here would silently give every new face the last old face's supply.
            raise RuntimeError(
                f"cytokinetic_ring: myo_nstar_per_tau has {nsp.shape[0]} entries against {nF} faces. "
                f"`medioapical_myosin` must declare it in m['face_carry'] and the topology operators "
                f"must run `_carry_face_state`.")
        n_star = self.tau_jun * nsp[ef]
        n_ring = self.ring * n_star

        uq, inv = torch.unique(key[fresh], return_inverse=True)
        z = torch.zeros(uq.numel(), device=dev, dtype=dt_)
        cnt = z.clone().index_add(0, inv, torch.ones_like(length[fresh]))
        n_j = z.clone().index_add(0, inv, n_ring[fresh]) / cnt.clamp_min(1.0)
        l_j = z.clone().index_add(0, inv, length[fresh]) / cnt.clamp_min(1.0)

        # THE DEBIT, ON THE CELLS THAT BUILT IT. The amount deposited on a junction is n*l; each of the
        # two half-edges belongs to one of the two daughters, so each daughter pays for its own side.
        if self.debit:
            rho = m.get("myo_med")
            if rho is not None and rho.shape[0] >= nF:
                # THE AREA IS RECOMPUTED, NOT READ. `myo_area` on the mesh is the pre-division one and
                # an area is extensive, so it is not carried across the rebuild; the daughters' areas
                # are what this debit must be spread over.
                area, _, _, _ = face_geometry_3d(pos, m["E_srce"].long(), m["E_trgt"].long(),
                                                 m["E_face"].long().clamp(max=nF - 1), nF,
                                                 live.to(dt_))
                area = area.clamp_min(1e-9)
                paid = (n_ring * length)[fresh]
                per_face = torch.zeros(nF, device=dev, dtype=dt_).index_add(0, ef[fresh], paid)
                rho = (rho[:nF] - per_face / area).clamp_min(0.0)
                m["myo_med"] = rho.detach()
                m["myo_med_total"] = float((rho * area).sum())

        # APPENDED TO THE STORE, not written to `m["myo"]`: the store is what the next frame's
        # `junction_myosin` reads and what `junction_myosin_sync` renders from, so putting the deposit
        # there is what makes it survive to both. Existing keys are kept; a fresh junction cannot
        # collide with one, since its key contains a vertex index that did not exist before.
        m["myo_keys"] = torch.cat([m["myo_keys"], uq.detach()])
        m["myo_vals"] = torch.cat([m["myo_vals"], n_j.detach().to(m["myo_vals"].dtype)])
        RING_TRACE.append((int(uq.numel()), float((n_j * l_j).sum())))
        if not self._said:
            print(f"[cytokinetic_ring] {int(uq.numel())} newborn interface(s) seeded at {self.ring}x "
                  f"the local steady state, debited from the medioapical pool", flush=True)
            self._said = True
        return {}


# Per frame: (newborn daughter--daughter interfaces seeded, myosin moved onto them). The second column
# is what the medioapical pool paid, so a ring that creates myosin instead of moving it shows up as a
# ledger that no longer closes.
RING_TRACE: list = []
