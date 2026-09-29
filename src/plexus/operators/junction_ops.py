"""Myosin, and the two places an epithelial cell can put it: on its junctions, or across its apex.

Myosin is what makes a vertex model contractile beyond a constant line tension: where the vertex
model writes one Lambda for every edge, these operators give each junction its own multiplier
and a rule for how it changes. All of them keep that multiplier keyed by VERTEX PAIR rather than
by half-edge index, which is what lets it survive a T1, a division or a death without any
topology operator knowing it exists.

In the order they appear below:

    junction_myosin     structural   per-junction myosin, recruited by tension
    junction_sync       rewire       re-key the store onto half-edge arrays topology has changed
    medioapical_myosin  lateral      the apical pool: an areal density on the face, not its edges
    cytokinetic_ring    structural   the ring a dividing cell leaves on the junction it just built
    junction_pcp        lateral      planar cell polarity: two complexes on each side of each junction

then the second model of `junction_myosin`, a different hypothesis in the same slot:

    junction_myosin[two_pool]  the belt fed by the medioapical pool, as a conserved amount
    junction_myosin[rest_length]  each junction an elastic element whose rest length remodels (exp 13)

The medioapical operators live in this file because the two-pool model is not a separate
mechanism: it is the same junction bookkeeping with a second reservoir on the face, and it calls
the same helpers -- `_live_edges`, `_lookup`, `_scatter_full`, `edge_tension`. Split across two
files, the shared half of one model would be private to the other.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from plexus.models.base import Rewire, Structural
from plexus.models.registry import register_operator
from plexus.models.base import Lateral, Structural
from plexus.operators.vertex_ops import face_geometry_3d


MYOSIN_TRACE: list = []


def _edge_key(vi, vj, stride):
    """Unordered vertex pair -> one integer. `stride` must exceed the vertex buffer size."""
    lo = torch.minimum(vi, vj).long()
    hi = torch.maximum(vi, vj).long()
    return lo * stride + hi


def _lookup(m, key, length, vi, vj, stride, myo_new, inherit, dev, dt_, new_val=None):
    """The keyed store on the mesh, mapped onto the half-edge arrays as they are NOW.

    One function with two callers -- `junction_myosin` and `junction_sync` -- so the myosin a
    frame is recorded with is by construction the myosin that frame's mechanics used. They
    cannot drift apart because only one piece of code decides it.

    Returns `(myo, n_new)` and touches nothing: the store is read, never written.

    `new_val` is what a junction with no history gets, per half-edge, overriding the scalar
    `myo_new`. A scalar is right only when the stored quantity has a fixed scale, and in the
    two-pool model it does not -- the tissue's mean line density drifts by a factor of about two
    over a run, so a newborn junction pinned at an absolute 1.0 would be set to an arbitrary
    fraction of what its neighbours happen to hold. Passing new_val = myo_new * n*_f, where
    n*_f = tau_jun * k_ex * rho_f is the density the supply into that cell's belt sustains, makes
    `myo_new` mean what its name says: a newborn junction as a FRACTION of what a mature one
    there would carry.
    """
    keys = m.get("myo_keys")
    vals = m.get("myo_vals")
    if keys is None or vals is None:
        keys = torch.empty(0, dtype=torch.long, device=dev)
        vals = torch.empty(0, dtype=dt_, device=dev)
    order = torch.argsort(keys)
    ks, vs = keys[order], vals[order]
    idx = torch.searchsorted(ks, key)
    idx_c = idx.clamp(max=max(ks.numel() - 1, 0))
    found = (ks.numel() > 0) & (idx_c < ks.numel())
    hit = found & (ks[idx_c] == key) if ks.numel() else torch.zeros_like(key, dtype=torch.bool)
    base = vs[idx_c] if vs.numel() else torch.zeros_like(length)
    fresh = (new_val if new_val is not None else torch.full_like(length, myo_new))
    myo = torch.where(hit, base, fresh)
    n_new = int((~hit).sum())

    # ---- A JUNCTION WITH A PARENT INHERITS FROM IT ------------------------------------------
    # Not every edge that misses the lookup is a new junction. `cell_divide` inserts a new vertex on
    # each of two edges of the dividing cell and then joins them, so a division produces two KINDS
    # of edge, and only one of them is new:
    #
    #   one endpoint new, one old   -- a SPLIT HALF of the cut junction (a,b). The same physical
    #                                  contact, now in two pieces. It has a myosin history and
    #                                  giving it `myo_new` throws that history away.
    #   both endpoints new          -- the interface between the two daughters. This one really is
    #                                  new, and `myo_new` is the honest answer for it.
    #
    # The parent is recoverable without any help from `cell_divide`, which is the point of keying by
    # vertex pair: a new vertex `n` has exactly two OLD neighbours, and they are the endpoints of the
    # edge it was inserted into. So the parent key is (min, max) over `n`'s old neighbours, and both
    # halves look it up. Falls back to `myo_new` where there is no parent to find.
    vseen = m.get("myo_vseen")
    if inherit and n_new and vseen is not None:
        oi = vseen[vi.clamp(max=vseen.numel() - 1)] & (vi < vseen.numel())
        oj = vseen[vj.clamp(max=vseen.numel() - 1)] & (vj < vseen.numel())
        half = (~hit) & (oi ^ oj)                       # exactly one endpoint is new
        if bool(half.any()):
            newv = torch.where(oi[half], vj[half], vi[half])     # the inserted vertex
            oldv = torch.where(oi[half], vi[half], vj[half])     # its old neighbour
            uq, inv = torch.unique(newv, return_inverse=True)
            lo = torch.full((uq.numel(),), stride, dtype=torch.long, device=dev)
            hi_ = torch.zeros(uq.numel(), dtype=torch.long, device=dev)
            lo = lo.scatter_reduce(0, inv, oldv, reduce="amin", include_self=True)
            hi_ = hi_.scatter_reduce(0, inv, oldv, reduce="amax", include_self=True)
            pkey = lo * stride + hi_                             # the edge the vertex was cut into
            pidx = torch.searchsorted(ks, pkey).clamp(max=max(ks.numel() - 1, 0))
            phit = (ks.numel() > 0) & (ks[pidx] == pkey) & (lo < hi_)
            # A HALF THAT CANNOT FIND ITS PARENT falls back to the same value a junction with no
            # history gets, taken per-edge so it follows `new_val` when one is supplied.
            fb = fresh[half]
            fb_u = torch.zeros(uq.numel(), device=dev, dtype=dt_).scatter_reduce(
                0, inv, fb, reduce="amax", include_self=False)
            inh = torch.where(phit, vs[pidx] if vs.numel() else torch.zeros_like(pkey, dtype=dt_),
                              fb_u)
            myo = myo.masked_scatter(half, inh[inv])
            n_inherited = int(phit[inv].sum())
            INHERIT_TRACE.append((n_new, int(half.sum()), n_inherited))
    return myo, n_new


def edge_tension(m, length, myo, ef, lam, k_perim, gam, dev, dt_):
    """Per-edge tension, as dE/dl_e of the energy `cell_mechanics` actually minimises.

    E = K_A(A-A0)^2 + K_P(P-P0)^2 + 0.5 Gam P^2 + K_V(...)^2 + Lambda*sum(m_e l_e), and only the
    perimeter and line terms depend on an individual edge length, so

        T_e = Lambda*m_e + 2 K_P (P_f - P0_f) + Gam * P_f      (f = the edge's own face)

    The CURRENT perimeter is not stored on the mesh -- `face_geometry_3d` computes and discards it --
    so it is recomputed here by the same index_add. P0 is on the mesh; K_P, Gam and Lambda come from
    the spec and MUST match the values `cell_mechanics` was given, or this is the tension of a
    different tissue.

    Raises rather than falling back. Falling back to `length` would make a tension-keyed run a
    LENGTH experiment wearing a tension label, and length is the exact variable such a run exists
    to replace -- a silent fallback to the hypothesis under test is the worst available failure
    mode.

    A module function rather than a method, because the medioapical operators need the same
    expression: the tension a junction is under is a property of the mesh and the energy, not of
    whichever operator happens to be asking. Two copies of it would be two tensions.
    """
    nF = int(m["nF"])
    P0 = m.get("P0")
    if P0 is None:
        raise RuntimeError(
            "edge_tension: this needs the target perimeter P0 on the mesh and it is absent. Refusing "
            "to fall back to length -- that is the variable a tension-keyed run exists to replace, and "
            "the fallback would have produced a length-keyed run labelled as tension.")
    perim = torch.zeros(nF, device=dev, dtype=dt_).index_add(0, ef, length)
    f = ef.long()
    T = lam * myo + 2.0 * k_perim * (perim - P0.to(dt_))[f] + gam * perim[f]
    return T.clamp_min(0.0)


def _scatter_full(es, live, myo, dev, dt_, fill=1.0):
    """Live values -> a per-half-edge array sized to the CURRENT buffer, `fill` on the dead slots.

    `fill` is 1.0 for a MULTIPLIER (a dead half-edge then contributes its usual tension if anything
    ever reads it unmasked) and 0.0 for an AMOUNT or a flux (a dead half-edge holds no myosin, and
    filling it with 1.0 would put the reservoir into the conservation ledger).
    """
    full = torch.full((es.shape[0],), float(fill), device=dev, dtype=dt_)
    full[live] = myo
    return full


def _live_edges(m, pos):
    """The live half-edges and their identities, for whoever is asking this frame.

    `live` is `E_face < nF`, the same mask `cell_mechanics` uses, NOT `E_srce < Nv`. The two differ
    after a division and using the wrong one is how a half-edge ends up with a myosin belonging to a
    face that no longer exists.
    """
    es, et, ef = m["E_srce"], m["E_trgt"], m["E_face"]
    nF, Nv = int(m["nF"]), int(m["Nv"])
    live = ef < nF
    vi, vj = es[live].long(), et[live].long()
    stride = max(pos.shape[0], Nv) + 1
    length = (pos[vj] - pos[vi]).norm(dim=-1).to(pos.dtype)
    return es, live, vi, vj, stride, _edge_key(vi, vj, stride), length


@register_operator("junction_myosin", family="mechanics", set="vertex", kind="structural",
                   equation=r"""$$m^{\mathrm{ss}}=a\,\frac{\ell_e}{\ell_{\mathrm{ref}}},\qquad \frac{dm}{dt}=\frac{m^{\mathrm{ss}}-m}{\tau}$$""")
class JunctionMyosin(Structural):
    """Per-junction myosin, recruited by tension: a stretched junction recruits more, pulls
    harder, and so shrinks -- the positive mechanical feedback a constant line tension cannot
    express. Survives T1, division and death by construction, being keyed by vertex pair.

    vertex -> vertex: reads pos and the keyed myosin store on the mesh, writes m["myo"] in place.
    Structural rather than lateral because it writes mesh state, not a per-vertex delta.

        myo_ss = a (l_e / l_ref)                the setpoint, rising with junction length
        d myo  = (myo_ss - myo) dt / tau        first-order relaxation toward it

    l_e is the junction's current length and l_ref the reference it is measured against, both in
    world units; a is `activity`, the dimensionless global myosin level. The multiplier myo then
    scales the vertex model's line tension Lambda, so myo = 1 is the uninhibited baseline.

    `activity` is the myosin-inhibition knob, the in-silico blebbistatin, and it multiplies the
    SETPOINT rather than the tension directly -- so inhibition takes effect over tau, the way a
    drug does, rather than instantly.

    l_ref is the CURRENT mean live edge length, not a constant. A growing tissue triples in
    radius, so a fixed reference would eventually call every junction stretched and raise myosin
    everywhere for a reason that is growth rather than tension.

    Reference: Rauzi, M. et al. (2008). Nature and function of lateral tension during tissue
    elongation. Nat. Cell Biol. 10:1401-1410 (myosin on shrinking junctions);
    Fernandez-Gonzalez, R. et al. (2009). Myosin II dynamics are regulated by tension in
    intercalating cells. Dev. Cell 17:736-743 (tension-dependent recruitment).
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = False        # writes m["myo"], not positions
    MECHANISM_TAGS = ["actomyosin_contraction", "mechanosensitive_recruitment",
                      "junction_state", "topology_persistent"]
    PARAM_ROLES = {"activity": "global_myosin_activity", "tau": "recruitment_timescale",
                   "myo_new": "myosin_of_a_newborn_junction", "beta": "tension_sensitivity",
                   "activity_from": "per-type cell property giving each junction its drive"}
    REFERENCE = ("Rauzi, M. et al. (2008). Nature and function of lateral tension during tissue "
                 "elongation. Nat. Cell Biol. 10:1401-1410; Fernandez-Gonzalez, R. et al. (2009). "
                 "Myosin II dynamics are regulated by tension in intercalating cells. Dev. Cell "
                 "17:736-743.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.activity = float(params.get("activity", 1.0))
        self.tau = float(params.get("tau", 20.0))
        self.beta = float(params.get("beta", 1.0))
        self.myo_new = float(params.get("myo_new", 1.0))
        # "length" reproduces every run up to 83 bit-for-bit; "tension" and "strain_rate" are the two
        # the literature supports. `destabilising` flips the sign so high drive -> more myosin -> higher
        # drive, which is the positive feedback that produces T1s rather than suppressing them.
        self.keyed_on = str(params.get("keyed_on", "length")).lower()
        # HETEROGENEITY, THROUGH THE TYPES. `activity` is one scalar over the whole tissue, so nothing
        # in a spec could say "these cells contract harder" -- and two operator instances cannot say it
        # either, because each rewrites the WHOLE keyed store every frame (`m["myo_keys"] = key`), so
        # the second would erase the first. `mask` does not help: the selector is per NODE of `at:`,
        # i.e. per vertex, while myosin lives per junction.
        #
        # `activity_from: <prop>` names a PER-TYPE property of the cell set, which is how every other
        # heterogeneous spec in this corpus says the same thing (`types: {t0: {fraction, p}, ...}`).
        # Each half-edge takes the value of the cell that owns it, and the junction takes the MEAN of
        # its two -- a junctional belt is fed from the cortices on both sides, and averaging is the
        # only choice symmetric in them, which matters because the store keeps ONE value per
        # undirected edge and would otherwise depend on which half-edge was written last.
        self.act_prop = params.get("activity_from")
        self.cell_set = params.get("cell_set", "cell")
        self.destabilising = bool(params.get("destabilising", True))
        self._prev_len = None
        self.lam = float(params.get("lam", 1.0))          # Lambda, for the tension expression
        self.k_perim = float(params.get("k_perim", 1.0))
        self.gam = float(params.get("gam", 0.0))
        self.inherit = bool(params.get("inherit", True))
        self.dt = float(params.get("dt", 1.0))
        # THE STORE IS ON THE MESH (`myo_keys` / `myo_vals` / `myo_vseen`), not here. See the module
        # docstring: state of an edge-set has to be reachable by any operator scheduled after a
        # topology change, and an operator attribute is reachable by exactly one.
        self._said = False

    def _edge_tension(self, m, length, myo, live, ef, dev, dt_):
        return edge_tension(m, length, myo, ef, self.lam, self.k_perim, self.gam, dev, dt_)

    def _edge_strain_rate(self, key, length):
        """d ln(l_e)/dt, matched to the PREVIOUS frame by the same vertex-pair key the myosin uses.

        Cheaper than tension and closer to Gustafson et al. 2022, whose recruitment variable is strain
        rate. Keyed rather than positional because cell_divide and edge_flip permute the half-edge
        arrays every frame -- comparing arrays by index would silently difference unrelated junctions.
        """
        cur = torch.stack([key, length])
        if self._prev_len is None:
            self._prev_len = cur
            return torch.zeros_like(length)
        pk, pl = self._prev_len
        order = torch.argsort(pk)
        ks, vs = pk[order], pl[order]
        idx = torch.searchsorted(ks, key).clamp(max=max(ks.numel() - 1, 0))
        hit = (ks.numel() > 0) & (ks[idx] == key)
        prev = torch.where(hit, vs[idx], length)
        self._prev_len = cur
        return torch.log(length.clamp_min(1e-9) / prev.clamp_min(1e-9))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        pos = lvl.get("pos")
        ef = m["E_face"]
        nF = int(m["nF"])
        dev, dt_ = pos.device, pos.dtype
        es, live, vi, vj, stride, key, length = _live_edges(m, pos)
        if not bool(live.any()):
            return {}
        l_ref = length.mean().clamp_min(1e-9)

        # ---- look each live junction up by its identity ------------------------------------------
        myo, n_new = _lookup(m, key, length, vi, vj, stride, self.myo_new, self.inherit, dev, dt_)

        # ---- recruitment ------------------------------------------------------------------------
        # ---- WHAT THE FEEDBACK IS KEYED TO ------------------------------------------------------
        # `length` was the original choice and it is the wrong variable, in two separate ways.
        #
        #   WRONG VARIABLE. Edge tension in this energy is Lambda*m_e + sum_f 2 K_P (P_f - P_f^0); length
        #   does not appear in it, so a long edge is not necessarily a taut one. The docstring used to
        #   assert "setpoint rises with junction length, i.e. with tension", which is an identification,
        #   not a derivation, and it is false here.
        #
        #   WRONG SIGN. Keyed to length the feedback is longer -> more myosin -> shorter: a STABILISER
        #   that homogenises junction lengths. The measured relationship runs the other way -- Bertet
        #   et al. 2004 find myosin enriched in DISASSEMBLING junctions, Fernandez-Gonzalez et al. 2009
        #   find more myosin on linked edges "regardless of edge length" -- and the resulting feedback is
        #   POSITIVE, a mechanical instability that generates T1s. The modelling literature keys on
        #   tension or strain rate and never on length; Sknepnek et al. 2023 say so verbatim.
        #
        # And keyed to length the operator is not even independent evidence for itself: in the tau -> 0
        # limit (the biologically correct one, myosin FRAP t1/2 ~ 6 s) substituting the setpoint gives
        # Lambda*a*sum[(1-beta) l_e + beta l_e^2/<l>], i.e. a line tension plus a HARMONIC EDGE SPRING of
        # zero rest length. Lowering the coefficient of variation of edge lengths is what such a spring
        # does by definition, so "beta lowers CV at constant radius" is a tautology and cannot
        # discriminate a right law from a wrong one. The T1 rate can, and the two signs move it oppositely.
        if self.keyed_on == "tension":
            drive = self._edge_tension(m, length, myo, live, ef[live].long(), dev, dt_)
        elif self.keyed_on == "strain_rate":
            drive = self._edge_strain_rate(key, length)
        else:
            drive = length
        d_ref = drive.mean().clamp_min(1e-9) if self.keyed_on != "strain_rate" else torch.ones((), device=dev, dtype=dt_)
        # SIGN. Getting this backwards silently converts the experiment into its own control, so it is
        # written out rather than inferred:
        #   length-keyed, +   longer -> more myosin -> shorter.  Negative feedback ON LENGTH: homogenises.
        #   tension-keyed, +  tauter -> more myosin -> tauter.   POSITIVE feedback ON TENSION: this is the
        #                     germband-extension instability, and it should RAISE the T1 rate.
        #   tension-keyed, -  tension homeostasis, the stabilising variant, kept as the contrast.
        # So the destabilising choice is +1, not -1. I had it as -1 on the first pass, which would have
        # made every "tension" run a stabiliser and the whole 84-91 comparison vacuous.
        sgn = 1.0 if self.destabilising else -1.0
        act = self.activity
        if self.act_prop:
            cl = H.level(self.cell_set)
            a_cell = getattr(cl, self.act_prop, None)
            if a_cell is None:
                raise ValueError(
                    f"junction_myosin activity_from={self.act_prop!r}: set {self.cell_set!r} has no "
                    f"per-node buffer of that name -- declare it as a per-type scalar under "
                    f"`sets.{self.cell_set}.types`.")
            a_he = a_cell[ef[live].long()].to(dt_)          # per half-edge, from the cell that owns it
            uk, inv = torch.unique(key, return_inverse=True)
            tot = torch.zeros(int(uk.numel()), device=dev, dtype=dt_).index_add_(0, inv, a_he)
            cnt = torch.zeros(int(uk.numel()), device=dev, dtype=dt_).index_add_(
                0, inv, torch.ones_like(a_he))
            act = self.activity * (tot / cnt.clamp_min(1.0))[inv]
        ss = act * (1.0 + sgn * self.beta * (drive / d_ref - 1.0)).clamp_min(0.0)
        myo = myo + (ss - myo) * (self.dt / max(self.tau, 1e-9))
        myo = myo.clamp(0.0, 5.0)

        # ---- store back, keyed. Junctions that no longer exist are simply not written, so a T1 or a
        # death drops them without anyone having to notice.
        m["myo_keys"], m["myo_vals"] = key.detach().clone(), myo.detach().clone()
        # which vertices existed this frame, so next frame can tell an inserted vertex from an old one
        vs_seen = torch.zeros(stride, dtype=torch.bool, device=dev)
        vs_seen[vi] = True; vs_seen[vj] = True
        m["myo_vseen"] = vs_seen
        # the per-half-edge array the mechanics reads: full length, 1.0 on dead slots so a masked-out
        # half-edge contributes its usual tension if anything ever reads it unmasked
        m["myo"] = _scatter_full(es, live, myo, dev, dt_)
        MYOSIN_TRACE.append((int(live.sum()), float(myo.mean()), float(myo.min()),
                             float(myo.max()), n_new))
        if not self._said:
            if self.act_prop:
                _u = torch.unique(act)
                print(f"[junction_myosin] activity_from={self.act_prop!r} on set {self.cell_set!r}: "
                      f"{int(_u.numel())} distinct drives across {int(live.sum()):,} junctions, "
                      f"{[round(float(v), 3) for v in _u[:6]]}"
                      f"{' ...' if _u.numel() > 6 else ''} (a boundary junction gets the mean of its "
                      f"two cells)", flush=True)
            print(f"[junction_myosin] {int(live.sum())} live junctions, activity={self.activity}, "
                  f"tau={self.tau}, myo_new={self.myo_new}; keyed by vertex pair so T1 / division / "
                  f"death need no edits", flush=True)
            self._said = True
        return {}


# Per frame, only when inheritance fires: (new edges, split halves, halves that found a parent).
INHERIT_TRACE: list = []

# Per frame: (half-edges now, half-edges when myosin was written, live junctions re-keyed). A run
# whose second column never differs from the first is a run in which no topology operator changed
# the edge buffer at all, which in a growing tissue is itself a defect worth noticing.
SYNC_TRACE: list = []


@register_operator("junction_sync", family="mechanics", set="vertex", kind="rewire",
                   equation=r"""$$m_e=\text{gain}\cdot\mathrm{store}\big[\mathrm{key}(v_i,v_j)\big],\qquad N_e=\mathrm{store}\big[\mathrm{key}(v_i,v_j)\big]\,\ell_e$$""")
class JunctionMyosinSync(Rewire):
    """Re-key the per-junction myosin onto the half-edge arrays a topology operator has just
    changed, so what is recorded for a frame is what that frame's mechanics actually used.

    vertex -> vertex: reads pos and the keyed store, rewrites m["myo"] and m["myo_amount"].

        myo_e  = gain * store[key(v_i, v_j)]     re-read at the CURRENT half-edge indices
        N_e    = store[key(v_i, v_j)] * l_e      the amount, for the conservation ledger

    The kind is `rewire` because the relation is what changed and this is the state following it.

    It is an operator rather than a line inside `cell_divide` because the carry is done from the
    other side. Per-FACE state already survives topology: the topology operators rebuild the mesh
    and reindex every per-face array through the `keep` map, so A0, P0, V0f, age and ndiv are
    never left pointing at a face that moved. Per-HALF-EDGE state has no such carry, and adding
    one to each topology operator would mean editing them again for the next per-junction state.
    Keying by vertex pair instead lets one operator map the store onto whatever half-edge arrays
    now exist.

    In the schedule it goes after every operator that can rewire or resize the half-edge arrays,
    and before `topo_record`:

        junction_myosin -> cell_mechanics -> edge_flip -> cell_divide
                        -> junction_sync -> topo_record

    It cannot change a trajectory, by construction. `cell_mechanics` reads m["myo"] in the same
    frame `junction_myosin` writes it, before any topology operator runs, and the next frame's
    `junction_myosin` overwrites it from the store; nothing in between reads it. So the array
    this writes is read by exactly one thing, `topo_record`, and adding it to a schedule leaves
    every position, every division and every T1 bit-identical -- the fix to a recording defect
    must not be able to alter what is being recorded.

    Reference: none -- this is bookkeeping that keeps a mechanism correct, not a mechanism.
    Plexus (this work).
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = False
    MECHANISM_TAGS = ["junction_state", "topology_persistent", "bookkeeping"]
    PARAM_ROLES = {"myo_new": "myosin_of_a_newborn_junction"}
    REFERENCE = "Plexus (this work); bookkeeping that keeps a mechanism correct, not a mechanism."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.myo_new = float(params.get("myo_new", 1.0))
        self.inherit = bool(params.get("inherit", True))
        self._said = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        if m is None or m.get("myo_keys") is None:
            return {}                      # no junction myosin in this specification: nothing to carry
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        es, live, vi, vj, stride, key, length = _live_edges(m, pos)
        if not bool(live.any()):
            return {}
        was = int(m["myo"].shape[0]) if m.get("myo") is not None else 0
        val, _ = _lookup(m, key, length, vi, vj, stride, self.myo_new, self.inherit, dev, dt_)
        # The store holds what the model chose to store; `myo` is what the mechanics reads. For
        # the default model those are the same number and `myo_gain` is 1. For `two_pool` the
        # store is a line density and the multiplier is that density scaled to a tissue mean of
        # `activity`, so the gain is activity / <n>. The owning operator leaves the gain on the
        # mesh and this operator applies it, which is what keeps this one model-agnostic. The
        # gain used is the one computed BEFORE the topology operators ran -- a mean over some
        # thousands of junctions, of which a handful were just cut, so it does not move at that
        # scale.
        gain = float(m.get("myo_gain", 1.0))
        m["myo"] = _scatter_full(es, live, (gain * val).clamp(0.0, 5.0), dev, dt_)
        # And the amount, on the same footing. `myo` is a density -- normalised, for `two_pool` --
        # so no sum of it says how much myosin there is; N_e = (stored density) * l_e does, and it
        # is the quantity the conservation ledger is written in. Zero on dead slots: an amount.
        m["myo_amount"] = _scatter_full(es, live, val * length, dev, dt_, fill=0.0)
        SYNC_TRACE.append((int(es.shape[0]), was, int(live.sum())))
        if not self._said:
            print(f"[junction_sync] re-keying myosin after the topology operators "
                  f"({was} half-edges written -> {int(es.shape[0])} now)", flush=True)
            self._said = True
        return {}


POOL_TRACE: list = []


def _face_carry(m, name):
    """Ask the topology operators to carry a per-face array across a rebuild.

    `cell_divide` and `cell_die` reindex every per-face array through the `keep` map (new face ->
    old face), so nothing is left pointing at a face that has moved. The list of names they carry
    is open rather than literal: an operator declares its own array here once, and the topology
    operators still know nothing about what is in it. A closed list would silently drop the
    per-face state of any operator added later -- the same class of defect as the per-half-edge
    myosin that `junction_sync` exists to fix, one level up.
    """
    m.setdefault("face_carry", set()).add(name)


@register_operator("medioapical_myosin", family="mechanics", set="cell", kind="lateral",
                   equation=r"""$$M_f=\rho_f A_f,\qquad \frac{dM_f}{dt}=k_{\mathrm{on}}A_f-\frac{M_f}{\tau_{\mathrm{med}}}-\sum_e J_{f\to e},\qquad J_{f\to e}=k_{\mathrm{ex}}\rho_f\ell_e\!\left(1+\beta_T\!\left(\frac{T_e}{\langle T\rangle}-1\right)\right)$$""")
class MedioapicalMyosin(Lateral):
    """The apical meshwork: a second myosin pool, spread over the FACE as an areal density
    rather than along its edges, which assembles there and flows outward onto the belt.

    cell -> cell: reads the face geometry and the per-edge tension, writes m["myo_med"] (the
    areal density) and m["myo_influx"] (the per-half-edge flux) on the mesh.

        M_f      = rho_f A_f                                 the amount, from the stored density
        dM_f/dt  = k_on A_f  -  M_f / tau_med  -  sum_e J_(f->e)
        J_(f->e) = k_ex rho_f l_e (1 + beta_T (T_e/<T> - 1))

    rho_f is the areal density of myosin on face f, in amount per world unit squared, and A_f the
    face's apical area. k_on is myosin assembled per unit APICAL AREA per frame, so a cell that
    grows assembles more -- which is what makes this pool areal rather than a per-cell budget.
    tau_med is the meshwork's own turnover time, in frames. k_ex is the fraction of the cell's
    areal density handed to each unit LENGTH of its junctional belt per frame; the flux is
    proportional to l_e because the belt is what receives it and there is l_e of it. beta_T is
    the dimensionless tension bias: at beta_T = 0 the flux is blind to mechanics, and above it a
    junction under more than the mean tension T/<T> draws proportionally more.

    The shape factor is a PREDICTION, not a parameter. At steady state

        rho* = k_on / (1/tau_med + k_ex P_f/A_f)

    so a cell with more perimeter per unit area drains faster and holds less medioapical myosin.
    P_f/A_f is fixed by shape alone, so the model says elongated cells are medioapically poorer
    than round ones of the same area -- with nothing added to make it say so.

    Reference: Munjal, A., Philippe, J.-M., Munro, E. & Lecuit, T. (2015). A self-organized
    biomechanical network drives shape changes during tissue morphogenesis. Nature 524:351-355
    (two apical pools, medioapical pulses flowing onto junctions); Rauzi, M., Lenne, P.-F. &
    Lecuit, T. (2010). Planar polarized actomyosin contractile flows control epithelial junction
    remodelling. Nature 468:1110-1114 (planar-polarised junctional myosin).
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
    REFERENCE = ("Munjal, A., Philippe, J.-M., Munro, E. & Lecuit, T. (2015). A self-organized "
                 "biomechanical network drives shape changes during tissue morphogenesis. Nature "
                 "524:351-355; Rauzi, M., Lenne, P.-F. & Lecuit, T. (2010). Planar polarized "
                 "actomyosin contractile flows control epithelial junction remodelling. Nature "
                 "468:1110-1114.")

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
        es, live, vi, vj, stride, key, length = _live_edges(m, pos)
        if not bool(live.any()):
            return {}
        nF = int(m["nF"])
        ef = m["E_face"][live].long()
        # THE SAME AREA `cell_mechanics` MINIMISES AGAINST -- the Newell area-vector magnitude, from
        # the same function, not a re-derivation. `eocc` masks the dead half-edges exactly as the
        # energy's live-only path does.
        eocc = live.to(dt_)
        area, _, _, _ = face_geometry_3d(pos, m["E_srce"].long(), m["E_trgt"].long(),
                                         m["E_face"].long().clamp(max=nF - 1), nF, eocc)
        area = area.clamp_min(1e-9)

        # ---- the stored state is a DENSITY, so `cell_divide`'s copy conserves the amount ------------
        rho = m.get("myo_med")
        if rho is None or rho.shape[0] < nF:
            rho = torch.full((nF,), self.rho0, device=dev, dtype=dt_)
        rho = rho[:nF].to(dt_)
        # BOTH ARE DENSITIES, so `cell_divide`'s copy-onto-both-daughters conserves them. `myo_area` is
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
            T = edge_tension(m, length, myo_l, ef, self.lam, self.k_perim, self.gam, dev, dt_)
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
        m["myo_influx"] = _scatter_full(es, live, J.detach(), dev, dt_, fill=0.0)
        if not self._said:
            print(f"[medioapical_myosin] {nF} cells, k_on={self.k_on}, tau_med={self.tau_med}, "
                  f"k_ex={self.k_ex}, beta_T={self.beta_T}; stored as an AREAL DENSITY so a division "
                  f"conserves the amount", flush=True)
            self._said = True
        return {}


@register_operator("junction_myosin", family="mechanics", set="vertex", kind="structural",
                   model="two_pool")
class JunctionMyosinTwoPool(Structural):
    """The junctional belt as a RECEIVER rather than a recruiter: it is fed by the medioapical
    pool, holds a conserved amount, and its density follows from that amount and its length.

    vertex -> vertex: reads the medioapical influx and pos, writes m["myo"] and the store.

        N_e      = n_e l_e                                the amount, from the line density
        dN_e/dt  = sum_(f in e) J_(f->e)  -  N_e / tau_jun
        m_e      = a n_e / <n_e>                          what the mechanics multiplies Lambda by

    n_e is the line density of myosin on junction e, in amount per world unit of length, and l_e
    its length. tau_jun is the belt's turnover time in frames -- the only timescale here, since
    the supply comes from elsewhere. a is `activity`, the dimensionless global level.

    This is a different MODEL of the same contract, not a different implementation of it. The
    default model recruits from a mechanical drive, with parameters {activity, tau, beta,
    keyed_on}; this one receives a flux from another pool, with {tau_jun, activity}, and has no
    drive at all. Those parameter sets are disjoint, so there is no operating point at which the
    two agree and swapping them is an experiment -- which is what the `model=` axis is for. They
    share the contract's kind and the channel m["myo"], so `cell_mechanics` cannot tell them
    apart and the comparison is at least fair on that side.

    The density is NORMALISED by the tissue mean because m_e multiplies Lambda, and Lambda is
    calibrated against a tissue whose mean multiplier is 1. Dividing by <n_e> fixes the overall
    level at `activity` and leaves this model predicting the PATTERN of junctional myosin, which
    is the part it is entitled to predict. Without it, k_on and k_ex would be a second, hidden
    line tension.

    The property this model exists for: n_e is intensive and its supply is per unit length, so at
    a division the two halves of a cut junction keep both the density they had AND the setpoint
    they were relaxing toward. The one-pool model keeps the first and loses the second.

    Reference: Munjal, A. et al. (2015). Nature 524:351-355 (medioapical to junctional flow);
    Curran, S. et al. (2017). Myosin II controls junction fluctuations to guide epithelial tissue
    ordering. Dev. Cell 43:480-492 (junctional myosin fluctuations).
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
    REFERENCE = ("Munjal, A. et al. (2015). Nature 524:351-355 (medioapical to junctional flow); "
                 "Curran, S. et al. (2017). Myosin II controls junction fluctuations to guide "
                 "epithelial tissue ordering. Dev. Cell 43:480-492.")

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
        es, live, vi, vj, stride, key, length = _live_edges(m, pos)
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
        # `junction_sync` was written to remove; it is avoided here by never creating it.
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
        # number. As a bare density it would mean nothing stable: the tissue's mean line density
        # drifts by about a factor of two over a run, so a fixed absolute value sets a new junction
        # to an arbitrary and time-varying fraction of what its neighbours hold -- a visible dimming
        # that has nothing to do with the parameter's intent.
        # n* IS A TWO-SIDED QUANTITY, and getting that wrong is worth a factor of two. A junction is
        # fed by BOTH cells it separates, so setting dN/dt = 0 gives n* = tau_jun * k_ex * (rho_f +
        # rho_g), not tau_jun * k_ex * rho_f. The one-sided version put a newborn junction at half the
        # density of a mature one while claiming `myo_new = 1.0` meant "the same as a mature one" --
        # measured, it moved the newborn from 0.628 to 0.625, i.e. it did nothing, because the units
        # error it introduced was the same size as the one it repaired. Summing over the half-edges
        # sharing a key is exactly the sum over the two cells.
        nsp = m.get("myo_nstar_per_tau")
        ef_l = m["E_face"][live].long()
        n_star = (self.tau_jun * (z.clone().index_add(0, inv, nsp[ef_l]))[inv]
                  if nsp is not None else torch.ones_like(length))
        # A FLAG AND NOT A REPLACEMENT, because the absolute reading is what every existing cache
        # was built with, and a silent change of meaning under an unchanged cache key is how an
        # archived run stops being reproducible without anything failing. `False` reproduces those
        # bit-for-bit; `True` is the corrected reading, and it enters the cache key.
        new_val = (self.myo_new * n_star if self.new_rel
                   else torch.full_like(length, self.myo_new))
        n_e, _ = _lookup(m, key, length, vi, vj, stride, self.myo_new, self.inherit, dev, dt_,
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
        # THE GAIN, LEFT ON THE MESH FOR `junction_sync` TO REAPPLY. The store is a density and
        # the mechanics wants a multiplier whose tissue mean is `activity`; putting the conversion on
        # the mesh rather than baking it into `myo` is what lets one sync operator serve both models.
        gain = self.activity / n_e.mean().clamp_min(1e-9)
        m["myo_gain"] = float(gain)
        m["myo"] = _scatter_full(es, live, (gain * n_e).clamp(0.0, 5.0), dev, dt_)
        m["myo_amount"] = _scatter_full(es, live, N.detach(), dev, dt_, fill=0.0)
        POOL_TRACE.append((float(m.get("myo_med_total", 0.0)), float(N.sum()),
                           float(n_e.mean()), float((gain * n_e).mean())))
        if not self._said:
            print(f"[junction_myosin/two_pool] {int(live.sum())} junctions fed by the medioapical "
                  f"pool, tau_jun={self.tau_jun}; the stored state is a LINE DENSITY and the "
                  f"integrated one is an AMOUNT", flush=True)
            self._said = True
        return {}


@register_operator("cytokinetic_ring", family="mechanics", set="vertex", kind="structural",
                   equation=r"""$$n_e=\text{ring}\cdot n^{*}_f,\qquad n^{*}_f=\tau_{\mathrm{jun}}\,k_{\mathrm{ex}}\,\rho_f$$""")
class CytokineticRing(Structural):
    """The cytokinetic ring: the myosin a division leaves on the junction it just built, debited
    from the cortex that built it.

    vertex -> vertex: reads which vertices are new and the medioapical density, writes the keyed
    store and debits m["myo_med"]. Runs after `cell_divide`, before `junction_sync`.

    A junction born with BOTH endpoints new is a daughter-daughter interface -- exactly the
    distinction the inheritance rule already draws, since a junction merely split by a division
    has one new endpoint and one old one. Those junctions are written into the store at

        n_e  = ring * n*_f,        n*_f = tau_jun k_ex rho_f

    n*_f is the line density the supply into that cell's belt sustains at steady state, so `ring`
    is dimensionless: how many times the density of a MATURE junction of that same cell a newborn
    daughter-daughter interface starts at. The deposit is debited from the medioapical pool of the
    adjacent cell, which is not bookkeeping pedantry -- the ring is assembled from cortical
    actomyosin, and a ring appearing from nowhere would be a source term in a model whose whole
    point is that myosin is conserved, after which the ledger could no longer detect a leak.

    Nothing here decays the deposit, because the junctional equation already does:
    dN/dt = J - N/tau_jun pulls the interface from ring * n* back to n* over tau_jun. So `ring`
    sets how bright a new junction starts and tau_jun how long it stays that way, and neither
    needs a second timescale invented for it.

    Without this operator the two-pool model gets the division site backwards twice: the brightest
    junctions there are the two halves of a contact the division CUT -- a cell about to divide is
    large, has a low perimeter-to-area ratio, holds more cortical myosin and feeds its belts
    harder -- while the daughter-daughter interface, the one place a real dividing cell puts
    almost all of its myosin, is the dimmest thing in the frame.

    Reference: Herszterg, S., Leibfried, A., Bosveld, F., Martin, C. & Bellaiche, Y. (2013).
    Interplay between the dividing cell and its neighbors regulates adherens junction formation
    during cytokinesis in epithelial tissue. Dev. Cell 24:256-270; Founounou, N., Loyer, N. & Le
    Borgne, R. (2013). Septins regulate the contractility of the actomyosin ring to enable
    adherens junction remodeling during cytokinesis. Dev. Cell 24:242-255.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = False
    MECHANISM_TAGS = ["actomyosin_contraction", "cytokinesis", "junction_state", "conserved_amount"]
    PARAM_ROLES = {"ring": "newborn_junction_density_as_a_multiple_of_the_local_steady_state",
                   "tau_jun": "the_belt_turnover_that_relaxes_it_back"}
    REFERENCE = ("Herszterg, S. et al. (2013). Interplay between the dividing cell and its "
                 "neighbors regulates adherens junction formation during cytokinesis in epithelial "
                 "tissue. Dev. Cell 24:256-270; Founounou, N., Loyer, N. & Le Borgne, R. (2013). "
                 "Dev. Cell 24:242-255.")

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
        es, live, vi, vj, stride, key, length = _live_edges(m, pos)
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
            # THE CARRY DID NOT HAPPEN, which means this operator is scheduled somewhere `cell_divide`
            # has grown the face count without `_carry_face_state` running. Said rather than clamped:
            # a clamp here would silently give every new face the last old face's supply.
            raise RuntimeError(
                f"cytokinetic_ring: myo_nstar_per_tau has {nsp.shape[0]} entries against {nF} faces. "
                f"`medioapical_myosin` must declare it in m['face_carry'] and the topology operators "
                f"must run `_carry_face_state`.")
        # TWO-SIDED, as in `junction_myosin[two_pool]`: the belt a ring hands its myosin to is fed by
        # both daughters, so the mature density it should be measured against is the sum over the two.
        uq, inv = torch.unique(key[fresh], return_inverse=True)
        z = torch.zeros(uq.numel(), device=dev, dtype=dt_)
        cnt = z.clone().index_add(0, inv, torch.ones_like(length[fresh]))
        n_star_j = self.tau_jun * z.clone().index_add(0, inv, nsp[ef[fresh]])
        n_j = self.ring * n_star_j
        l_j = z.clone().index_add(0, inv, length[fresh]) / cnt.clamp_min(1.0)
        n_ring = torch.zeros_like(length)
        n_ring[fresh] = n_j[inv] / cnt.clamp_min(1.0)[inv]     # each side pays for its own half

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
        # `junction_myosin` reads and what `junction_sync` renders from, so putting the deposit
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


@register_operator("junction_myosin", family="mechanics", set="vertex", kind="structural",
                   model="oriented", title="Junction tension set by orientation and position",
                   equation=r"""$$m_e=a\,\big(1+\alpha\,w_e\big)\big(1+g\,\phi_e\big),\qquad w_e=(\hat{\mathbf t}_e\cdot\hat{\mathbf c}_e)^2\,\frac{\rho_e}{\rho_{\max}},\quad \phi_e=1-\Big(\frac{z_e}{z_{\max}}\Big)^2$$""")
class JunctionMyosinOriented(Structural):
    """In-surface ANISOTROPIC junction tension: circumferential junctions pull harder than
    meridional ones about a declared body axis -- the "molecular corset" -- optionally scaled by an
    A-P gradient that is strongest at the centre and vanishes at the poles.

    vertex -> vertex: reads pos, writes m["myo"] (one multiplier per half-edge) for THIS frame's
    topology. No store, no dynamics: the multiplier is recomputed from geometry every call, so it
    survives T1, division and death by construction and needs no `junction_sync`.

        m_e   = a (1 + alpha w_e)(1 + g phi_e)          what cell_mechanics multiplies Lambda by
                (x l_e / l_0 under `law: stiffness`: a resistance to stretch, not a squeeze)
        w_e   = (t_e . c_e)^2  rho_e / rho_max          circumferential alignment, faded at the poles
        phi_e = 1 - (z_e / z_max)^2                      the A-P profile: 1 at the centre, 0 at a pole

    t_e is the junction's unit direction, c_e the unit circumferential direction at its midpoint
    (the axis crossed with the midpoint's offset from the tissue centroid), rho_e the midpoint's
    distance from the axis and rho_max the largest over the tissue, z_e the midpoint's coordinate
    along the axis from the centroid and z_max the largest |z| of any vertex. a is `activity` (the
    absolute level), alpha is `aniso` (0 = isotropic: every junction pulls alike) and g is `grad`
    (0 = no A-P pattern). All three are dimensionless.

    WHY THE FADE AT THE POLES. The circumferential direction is undefined on the axis itself, where
    the circles of latitude shrink to a point; weighting by rho_e makes the anisotropy vanish there
    smoothly, which is what a corset does -- it grips the girth and nothing at the ends. It is the
    same construction `bm_bond`'s `aniso` uses on the basement membrane (membrane_ops.py), applied
    here to the epithelium's own junctions.

    WHAT THIS IS AND IS NOT. The fly follicle's anisotropic resistance sits in the basement membrane
    (Crest et al. 2017: AFM finds it in the BM, not in the epithelium, and it is a GRADIENT of
    stiffness along A-P more than a directional anisotropy). This operator puts the corset in the
    epithelium's line tension, which is the in-surface half of the roadmap's M6: a stated
    reduction, standing in for the membrane until the tissue and the membrane are coupled live.
    Line tension is contractile, not elastic: it biases where growth goes by pulling harder around
    the girth, it does not store a rest length.

    Reference: Haigo, S.L. & Bilder, D. (2011). Global tissue revolutions in a morphogenetic
    movement controlling elongation. Science 331:1071-1074 (the molecular corset); Crest, J.,
    Diz-Munoz, A., Chen, D.-Y., Fletcher, D.A. & Bilder, D. (2017). Organ sculpting by patterned
    extracellular matrix stiffness. eLife 6:e24958 (the A-P stiffness gradient).
    """
    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = False          # writes m["myo"], not positions
    MECHANISM_TAGS = ["planar_polarity", "anisotropic_tension", "molecular_corset", "ap_gradient"]
    PARAM_ROLES = {"activity": "absolute_tension_level", "aniso": "circumferential_excess",
                   "grad": "ap_gradient_amplitude", "axis": "body_axis_direction"}
    PARAM_UNITS = {"activity": "fraction", "aniso": "fraction", "grad": "fraction"}
    REFERENCE = ("Haigo, S.L. & Bilder, D. (2011). Science 331:1071-1074; Crest, J. et al. (2017). "
                 "Organ sculpting by patterned extracellular matrix stiffness. eLife 6:e24958.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.activity = float(params.get("activity", 1.0))
        self.aniso = float(params.get("aniso", 0.0))
        self.grad = float(params.get("grad", 0.0))
        ax = torch.as_tensor(params.get("axis", [0.0, 0.0, 1.0]), dtype=torch.float64)
        self.axis = ax / ax.norm().clamp(min=1e-12)
        # `law` -- A SQUEEZE OR A RESISTANCE. `tension` (the default) is a constant force per
        # junction: it deforms the tissue at once and is outgrown as the tissue grows (exp 5 round
        # 1: aspect ratio 1.88 by frame 100, before any division, relaxing to 1.24 by frame 1601).
        # `stiffness` multiplies each junction's factor by its length over the tissue's mean
        # junction length at the first call, so the energy Lambda m_e l_e becomes quadratic in the
        # length -- a spring of zero rest length, whose force grows with STRETCH. That is what the
        # follicle's basement membrane does (Crest et al. 2017: it resists expansion, and it
        # stiffens 30 -> 70 kPa as the follicle grows), and what a corset is.
        self.law = str(params.get("law", "tension")).lower()
        # `hoop` -- THE CORSET AS A SUPRACELLULAR SHEET. A junction-level law cannot feel the tissue
        # grow: divisions keep junctions at their length while the tissue widens by adding cells
        # (round 2: `stiffness` squeezed at once and relaxed like `tension`). The follicle's
        # membrane is one continuous sheet that does not divide and resists the GIRTH widening. So
        # under `hoop` the circumferential excess is proportional to the local hoop strain,
        #   m_e = a (1 + max(0, rho_e / rho_0 - 1) (alpha (t_e.c_e)^2 + g phi_e)),
        # rho_0 the tissue's largest distance from the axis at the first call: zero at the start,
        # rising only where the tissue has widened -- tension that grows with hoop strain.
        if self.law not in ("tension", "stiffness", "hoop"):
            raise ValueError(f"junction_myosin[oriented]: law must be tension|stiffness|hoop, "
                             f"got {self.law!r}")
        self._l0 = None
        self._rho0 = None
        # `grad_mode` -- WHERE THE A-P GRADIENT LIVES. `isotropic` (default): every junction near the
        # centre is stiffer, whatever its direction. `corset`: the gradient multiplies only the
        # CIRCUMFERENTIAL term -- a corset that is densest at the centre and fades toward the poles,
        # which is what the follicle's membrane is (circumferential fibrils, Col IV highest at the
        # centre; Crest et al. 2017 Fig 4-5). Round 4 found the isotropic gradient does not elongate.
        self.grad_mode = str(params.get("grad_mode", "isotropic")).lower()
        # BOUNDING THE HOOP LAW (round 5: at excess 8-16 the multiplier reached 10-25x the base
        # tension as the girth grew 2-3x, and cells squeezed into slivers inverted from frame
        # ~1400). The follicle's membrane stiffens ~2.3x over development (30 -> 70 kPa) and is
        # continuously remodelled, so two bounded forms:
        #   `m_max`   -- the multiplier saturates there (0 = unbounded, the round-4/5 law);
        #   `tau_ref` -- the reference radius rho_0 relaxes toward the current girth with this time
        #                constant, in frames (0 = never: strain from the seed, the round-4/5 law).
        self.m_max = float(params.get("m_max", 0.0))
        self.tau_ref = float(params.get("tau_ref", 0.0))
        if self.grad_mode not in ("isotropic", "corset"):
            raise ValueError(f"junction_myosin[oriented]: grad_mode must be isotropic|corset, got {self.grad_mode!r}")

    def forward(self, H, mask=None):
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        Nv = int(m["Nv"]); es, et = m["E_srce"].long(), m["E_trgt"].long()
        pos = lvl.get("pos")[:Nv].detach()
        a = self.axis.to(device=pos.device, dtype=pos.dtype)
        c = pos.mean(0)
        mid = 0.5 * (pos[es] + pos[et]) - c
        t = pos[et] - pos[es]
        t = t / t.norm(dim=1, keepdim=True).clamp(min=1e-12)
        circ = torch.cross(a.expand_as(mid), mid, dim=1)
        rho = circ.norm(dim=1)
        ch = circ / rho.clamp(min=1e-12)[:, None]
        w = (t * ch).sum(1) ** 2 * (rho / rho.max().clamp(min=1e-12))
        z = mid @ a
        zmax = ((pos - c) @ a).abs().max().clamp(min=1e-12)
        phi = (1.0 - (z / zmax) ** 2).clamp(min=0.0)
        if self.law == "hoop":
            if self._rho0 is None:
                self._rho0 = float(rho.max().clamp(min=1e-12))
            elif self.tau_ref > 0.0:
                _dt = float(getattr(H, "dt", 1.0))
                self._rho0 += (float(rho.max()) - self._rho0) * min(1.0, _dt / self.tau_ref)
            strain = (rho / self._rho0 - 1.0).clamp(min=0.0)
            # BOTH patterns resist only once stretched: the directional corset (alpha) and the A-P
            # stiffness gradient (g, Crest et al.'s stiffer centre) are stiffnesses, so each scales
            # with the local hoop strain and is zero on the unstretched seed.
            align = (t * ch).sum(1) ** 2
            if self.grad_mode == "corset":
                myo = self.activity * (1.0 + strain * align * (self.aniso + self.grad * phi))
            else:
                myo = self.activity * (1.0 + strain * (self.aniso * align + self.grad * phi))
            if self.m_max > 0.0:
                myo = myo.clamp(max=self.activity * self.m_max)
            m["myo"] = myo
            return {}
        myo = self.activity * (1.0 + self.aniso * w) * (1.0 + self.grad * phi)
        if self.law == "stiffness":
            l_e = (pos[et] - pos[es]).norm(dim=1)
            if self._l0 is None:
                self._l0 = float(l_e.mean().clamp(min=1e-12))
            myo = myo * (l_e / self._l0)
        m["myo"] = myo
        return {}


# junction_myosin[type_pair] -- the line tension per pair of cell TYPES (experiment 9, Graner & Glazier 1992's
# J table on the vertex model). Moved here from sorting_ops.py on 2026-09-27, unchanged.
def type_pair_multiplier(es, et, ef, face_type, table, medium=None):
    """Per half-edge: table[type of its face, type of the face across], or table[type, medium] on a
    free edge (no twin). `table` [K+1, K+1] when `medium` is its last index, else [K, K]."""
    from plexus.operators.junction_ops import pcp_twins
    stride = int(max(int(es.max()), int(et.max()))) + 1 if es.numel() else 1
    tw = pcp_twins(es, et, stride)
    t_in = face_type[ef]
    has = tw >= 0
    t_out = torch.where(has, face_type[ef[tw.clamp(min=0)]], torch.full_like(t_in, -1))
    if medium is None and not bool(has.all()):
        raise ValueError("junction_myosin[type_pair]: the mesh has free edges and the table no `medium` entry")
    t_out = torch.where(has, t_out, torch.full_like(t_in, medium if medium is not None else 0))
    return table[t_in, t_out]


@register_operator("junction_myosin", model="type_pair", family="mechanics", set="vertex", kind="structural",
                   title="Junction tension set by the two cells' types",
                   equation=r"""$$m_e=\gamma_{\tau(f_e)\,\tau(f'_e)},\qquad m_e^{\mathrm{free}}=\gamma_{\tau(f_e)\,M}$$""")
class JunctionMyosinTypePair(Structural):
    """The vertex model's differential adhesion: each junction's line tension set by the TYPES of the
    two cells it separates, and by the cell's type and the medium on a free edge.

    vertex -> vertex: reads the cell set's node_type, writes m["myo"] (one multiplier per half-edge)
    for this frame's topology. No store: recomputed every call, so it follows T1s, divisions and a
    retyped cell, and needs no `junction_sync`.

        m_e      = gamma[tau(f_e), tau(f'_e)]     what cell_mechanics multiplies Lambda by
        m_e_free = gamma[tau(f_e), medium]        an edge with no cell across (the aggregate's surface)

    f_e is the face (cell) owning half-edge e, f'_e the face across (its twin's owner), tau a cell's
    type and gamma the symmetric table `tensions: {A: {A: .., B: .., medium: ..}, B: {B: .., medium: ..}}`
    keyed by the cell set's type names plus `medium`. With gamma = J / J_ref this is Graner & Glazier
    1992's Potts surface energies on a vertex model's junctions: J(d,d)=2, J(d,l)=11, J(l,l)=14,
    J(cell, medium)=16 sort (their Eq. 3); a lower homotypic tension is a higher adhesion.

    WHY A MODEL OF junction_myosin. `interface_tension` is a purse-string on the activator
    interface with one tension; `cell_mechanics` already multiplies its line tension junction by
    junction by m["myo"], which `junction_myosin` writes. A table read by type is one more rule for
    that multiplier, not a new force.

    Reference: Graner, F. & Glazier, J.A. (1992). Phys. Rev. Lett. 69:2013-2016; Steinberg, M.S.
    (1963). Science 141:401-408; Farhadifar, R. et al. (2007). Curr. Biol. 17:2095-2104 (the line
    tension of the vertex model).
    """
    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = False             # writes m["myo"], not positions
    REQUIRES_PARAMS = ["tensions"]
    MECHANISM_TAGS = ["differential_adhesion", "cell_sorting", "type_pair_tension"]
    PARAM_ROLES = {"tensions": "line_tension_multiplier_per_pair_of_types_and_medium",
                   "noise": "active_cortical_fluctuation_amplitude_in_W", "tau": "fluctuation_correlation_frames",
                   "noise_end": "amplitude_at_the_end_of_the_ramp", "frames": "ramp_length_calls",
                   "seed": "rng_seed"}
    PARAM_UNITS = {"tensions": "fraction"}
    REFERENCE = ("Graner, F. & Glazier, J.A. (1992). Phys. Rev. Lett. 69:2013-2016; Steinberg, M.S. "
                 "(1963). Science 141:401-408.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self._cat = params.get("cell_set")
        self.spec = params["tensions"]
        self._tab = None
        self._medium = None
        # ACTIVE CORTICAL FLUCTUATIONS (experiment 9, Phase 2, Finding 20). A deterministic vertex
        # aggregate only descends to the nearest local minimum: two halves never engulf and a labyrinth
        # never rounds. Cells in an aggregate fluctuate their cortex; here each CELL f carries an
        # Ornstein-Uhlenbeck variable eta_f (unit variance, correlation time `tau` frames) and a junction
        # between f and g adds `noise` x (eta_f + eta_g) / sqrt(2) to its multiplier (a rim junction,
        # `noise` x eta_f). `noise` is in the table's own units (a W); 0, the default, is the table alone.
        self.noise = float(params.get("noise", 0.0))
        # `noise_end` and `frames`: the amplitude cooled linearly from `noise` to `noise_end` over
        # `frames` calls, then held (an annealing schedule, Finding 23); absent, the amplitude is constant.
        self.noise0 = self.noise
        self.noise_end = float(params.get("noise_end", self.noise))
        self.frames = max(1, int(params.get("frames", 1)))
        self._calls = 0
        self.tau = float(params.get("tau", 10.0))
        self.seed = int(params.get("seed", 0))
        self._eta = None
        self._gen = None

    def _table(self, clvl, device, dtype):
        names = list(getattr(clvl, "type_names", None) or [])
        want_m = any("medium" in (row or {}) for row in self.spec.values())
        K = len(names) + (1 if want_m else 0)
        idx = {n: i for i, n in enumerate(names)}
        if want_m:
            idx["medium"] = len(names)
        tab = torch.full((K, K), float("nan"), device=device, dtype=dtype)
        for a, row in self.spec.items():
            for b, g in (row or {}).items():
                if a not in idx or b not in idx:
                    raise ValueError(f"junction_myosin[type_pair]: {a!r}/{b!r} is not one of {list(idx)}")
                tab[idx[a], idx[b]] = tab[idx[b], idx[a]] = float(g)
        miss = [(n, m) for n in names for m in list(idx) if torch.isnan(tab[idx[n], idx[m]])]
        if miss:
            raise ValueError(f"junction_myosin[type_pair]: no tension for {miss}")
        return tab, (idx["medium"] if want_m else None)

    def forward(self, H, mask=None):
        from plexus.operators.vertex_ops import resolve_cell_set
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        clvl = H.level(resolve_cell_set(H, self.at, self._cat))
        es, et, ef = m["E_srce"].long(), m["E_trgt"].long(), m["E_face"].long()
        pos = lvl.get("pos")
        if self._tab is None or self._tab.device != pos.device:
            self._tab, self._medium = self._table(clvl, pos.device, pos.dtype)
        nF = int(m["nF"])
        ft = clvl.node_type[:nF].to(pos.device).long()
        myo = type_pair_multiplier(es, et, ef, ft, self._tab, self._medium)
        self.noise = self.noise0 + (self.noise_end - self.noise0) * min(1.0, self._calls / self.frames)
        self._calls += 1
        if self.noise0 > 0.0 or self.noise_end > 0.0:
            from plexus.operators.junction_ops import pcp_twins
            n_buf = clvl.node_type.shape[0]
            if self._eta is None:
                self._gen = torch.Generator(device="cpu").manual_seed(self.seed)
                self._eta = torch.randn(n_buf, generator=self._gen, dtype=torch.float64)
            h = float(getattr(H, "dt", 1.0))
            a = float(np.exp(-h / max(self.tau, 1e-9)))
            self._eta = a * self._eta + float(np.sqrt(1.0 - a * a)) * torch.randn(n_buf, generator=self._gen, dtype=torch.float64)
            eta = self._eta.to(device=pos.device, dtype=myo.dtype)
            stride = int(max(int(es.max()), int(et.max()))) + 1 if es.numel() else 1
            tw = pcp_twins(es, et, stride)
            has = tw >= 0
            other = torch.where(has, eta[ef[tw.clamp(min=0)]], torch.zeros_like(eta[ef]))
            myo = myo + self.noise * torch.where(has, (eta[ef] + other) / float(np.sqrt(2.0)), eta[ef])
        m["myo"] = myo
        return {}


def rest_length_step(L, L0, dt, tau, kappa, m_max):
    """One step of a junction's viscoelastic rest length, and the line-tension multiplier it gives.

        L0'  = L0 + (L - L0) dt / tau               the rest length remodels toward the length
        m    = clip(1 + kappa (L - L0') / L0', 0, m_max)

    A module function so the model and its test share one expression."""
    L0n = L0 + (L - L0) * (dt / max(tau, 1e-9))
    m = (1.0 + kappa * (L - L0n) / L0n.clamp_min(1e-12)).clamp(0.0, m_max)
    return L0n, m


@register_operator("junction_myosin", model="rest_length", family="mechanics", set="vertex", kind="structural",
                   title="Junction tension from a slowly remodelling rest length",
                   equation=r"""$$T_e=\Lambda\Big(1+\kappa\,\frac{\ell_e-\ell^0_e}{\ell^0_e}\Big),\qquad \frac{d\ell^0_e}{dt}=\frac{\ell_e-\ell^0_e}{\tau}$$""")
class JunctionRestLength(Structural):
    """Each junction an elastic element with a REST LENGTH that remodels toward its length over tau: a
    junction stretched faster than it remodels pulls harder, one compressed faster pulls less.

    vertex -> vertex: reads pos, keeps each junction's rest length in its own keyed store on the mesh
    (`rl_keys` / `rl_vals`, keyed by vertex pair as the myosin store is), writes m["myo"], the multiplier
    `cell_mechanics` puts on Lambda:

        m_e   = clip(1 + kappa (l_e - l0_e) / l0_e, 0, m_max)       so  T_e = Lambda m_e + perimeter terms
        dl0_e = (l_e - l0_e) dt / tau

    l_e is the junction's length and l0_e its rest length, world units; kappa is the junction's
    stiffness over Lambda (dimensionless); tau the remodelling time, frames. A junction with no history
    (a new interface, a split half) starts at rest, l0 = l. Under steady stretching at strain rate r
    the strain settles at r tau, so the tension records the RECENT DEFORMATION of each junction --
    orientation-dependent wherever the tissue deforms anisotropically.

    WHY, for exp 13 Phase 2 (P9): in the default energy a junction's tension is the sum of its two
    cells' (Lambda + 2 K_P (P - P0) + Gamma P), so a cut junction's recoil cannot depend on its
    orientation, and LeGoff et al. 2013's rim (tangential junctions recoil ~2.6x the radial ones)
    cannot be expressed at all. Length-keyed myosin (the default model) is a zero-rest-length spring
    that collapsed the pouch's junctions (P26, P30); this is a spring with a rest length that tracks
    the junction, which is gentle at the start (every m = 1) and stiff only to recent strain.

    `start` (frames, default 0): before it the rest length IS the length (every m = 1), so a seeded mesh
    settling to its own equilibrium -- exp 13's disc halves its junction lengths in its first 60 frames --
    is not read as compression (a first smoke without it: median m 0.16 at frame 60, cells ballooned,
    wrecked). Set it to the spec's settle frame.

    `tau_m` (frames, default 0 = off): the multiplier itself relaxes toward 1 + kappa (l - l0) / l0 over
    tau_m instead of jumping to it. `cell_mechanics` holds m fixed through a frame's relaxation, so an
    undamped m is a bang-bang loop -- a stretched junction pulls with its whole m, overshoots into
    compression, reads m = 0 the next frame, springs back -- which wrecked every batch-14 run 20 frames
    after `start`, gain 0 and kappa 1 included (a vertex jumping 4.5 edge lengths). The damped m is kept
    in the same keyed store as l0 (`rl_mvals`).

    WITH `cell_mechanics[model: junction_spring]` the law is a spring in the mechanics' ENERGY: that model
    reads the store (`rl_keys`, `rl_vals`, `rl_kappa`, `rl_stride`) and adds (kappa Lambda / 2) (l - l0)^2 /
    l0 per junction, evaluated at every relaxation iteration, and ignores m["myo"], which is then only a
    readout of the junction's total tension (for the rulers and the growth law's stress). Use tau_m 0 there.
    Without it, m multiplies Lambda and is held fixed through the relaxation -- which cascades into T1s
    (exp 13 P45).

    SCHEDULE IT BEFORE `cell_mechanics`, as the other junction_myosin models are, and -- for the record --
    a second instance with `advance: false` after the topology operators. The advancing instance
    relaxes l0 and writes m for the half-edges the mechanics is about to use. The read-only one
    recomputes m from the stored l0 for the half-edges as they are after divisions and flips, so what
    `topo_record` writes is aligned with the recorded mesh. It never moves l0 or the damped m. A single
    instance after the topology operators (the first version) handed the next frame's mechanics a
    multiplier indexed by the previous layout. At multipliers within 1 % of 1, vertices then jumped
    0.2 -> 1.8 edge lengths over frames 65-80 (batch 15, `p2_rl_k1_tm20`).

    Reference: Staddon, M.F. et al. (2019). Mechanosensitive junction remodeling promotes robust
    epithelial morphogenesis. Biophys. J. 117:1739-1750 (junction rest-length remodelling);
    Noll, N. et al. (2017). Active tension network model suggests an exotic mechanical state
    realized in epithelial tissues. Nat. Phys. 13:1221-1226.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = False
    MECHANISM_TAGS = ["junction_state", "viscoelastic", "rest_length_remodeling", "topology_persistent"]
    PARAM_ROLES = {"kappa": "junction_stiffness_over_line_tension", "tau": "rest_length_remodeling_time",
                   "m_max": "multiplier_ceiling", "start": "first_frame_of_remodelling",
                   "tau_m": "multiplier_relaxation_time", "advance": "false = recompute m only, for the record"}
    REFERENCE = ("Staddon, M.F. et al. (2019). Biophys. J. 117:1739-1750; Noll, N. et al. (2017). "
                 "Nat. Phys. 13:1221-1226.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.kappa = float(params.get("kappa", 4.0))
        self.tau = float(params.get("tau", 150.0))
        self.m_max = float(params.get("m_max", 5.0))
        self.dt = float(params.get("dt", 1.0))
        self.start = int(params.get("start", 0))
        self.tau_m = float(params.get("tau_m", 0.0))
        self.advance = bool(params.get("advance", True))
        self._k = 0
        self._said = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        es, live, vi, vj, stride, key, length = _live_edges(m, pos)
        if not bool(live.any()):
            return {}
        keys, vals = m.get("rl_keys"), m.get("rl_vals")
        if not self.advance:
            # READ-ONLY: m from the stored l0 (and the stored damped m) on the half-edges as they are now
            mult = torch.ones_like(length)
            if keys is not None and keys.numel():
                order = torch.argsort(keys)
                ks = keys[order]
                idx = torch.searchsorted(ks, key).clamp(max=ks.numel() - 1)
                hit = ks[idx] == key
                mv = m.get("rl_mvals")
                if mv is not None and mv.numel() == keys.numel():
                    mult = torch.where(hit, mv[order].to(dt_)[idx], mult)
            m["myo"] = _scatter_full(es, live, mult, dev, dt_)
            return {}
        self._k += 1
        L0 = length.clone()
        m_prev = torch.ones_like(length)
        if keys is not None and keys.numel() and self._k > self.start:
            order = torch.argsort(keys)
            ks, vs = keys[order], vals[order].to(dt_)
            idx = torch.searchsorted(ks, key).clamp(max=ks.numel() - 1)
            hit = ks[idx] == key
            L0 = torch.where(hit, vs[idx], length)
            mv = m.get("rl_mvals")
            if mv is not None and mv.numel() == keys.numel():
                m_prev = torch.where(hit, mv[order].to(dt_)[idx], m_prev)
        L0n, mult = rest_length_step(length, L0, self.dt, self.tau, self.kappa, self.m_max)
        if self.tau_m > 0:
            mult = m_prev + (mult - m_prev) * min(self.dt / self.tau_m, 1.0)
        m["rl_keys"], m["rl_vals"], m["rl_mvals"] = key.detach().clone(), L0n.detach().clone(), mult.detach().clone()
        # for cell_mechanics[junction_spring]: NO spring before `start`. The store is written from the first
        # frame (l0 = l), and a spring at a rest length reset every frame to the current length is a drag on
        # every junction's change within the relaxation; batch 18 ran with it (all four arms wrecked).
        m["rl_kappa"], m["rl_stride"] = (self.kappa if self._k > self.start else 0.0), int(stride)
        m["myo"] = _scatter_full(es, live, mult, dev, dt_)
        MYOSIN_TRACE.append((int(live.sum()), float(mult.mean()), float(mult.min()), float(mult.max()), 0))
        if not self._said:
            print(f"[junction_myosin/rest_length] {int(live.sum())} live half-edges, kappa={self.kappa}, "
                  f"tau={self.tau}, m_max={self.m_max}; rest lengths keyed by vertex pair", flush=True)
            self._said = True
        return {}


# ================================================================================================ #
#  junction_pcp -- planar cell polarity on the junction sides
# ================================================================================================ #
PCP_TRACE: list = []


def pcp_twins(es, et, stride):
    """Index of each half-edge's twin (t, s), or -1 on the free boundary."""
    key = es.long() * stride + et.long()
    tw = et.long() * stride + es.long()
    order = torch.argsort(key)
    ks = key[order]
    j = torch.searchsorted(ks, tw).clamp(max=max(ks.numel() - 1, 0))
    hit = ks[j] == tw
    return torch.where(hit, order[j], torch.full_like(j, -1))


def pcp_side_pairs(ef, ang, perim, nF, spread):
    """The within-cell kernel of `junction_pcp`: pairs (a, b) of sides of one cell and their weights.

    w_ab = exp(-d_ab / (spread * s_i)), d_ab the distance between the two sides' midpoints ALONG the
    perimeter, taken as (P_i / 2 pi) times their angular separation about the centroid (exact for a
    circle, close for a convex cell), s_i = P_i / n_i the cell's mean side length. Returns
    (a, b, w) with the self pairs (w = 1) included; `spread` = 0 returns only the self pairs.
    """
    E = ef.shape[0]
    idx = torch.arange(E, device=ef.device)
    if spread <= 0.0:
        return idx, idx, torch.ones(E, dtype=ang.dtype, device=ef.device)
    order = torch.argsort(ef, stable=True)
    fs = ef[order]
    deg = torch.bincount(ef, minlength=nF)
    start = torch.cumsum(deg, 0) - deg
    kmax = int(deg.max())
    a_l, b_l = [], []
    for k in range(kmax):
        ok = deg[fs] > k
        j = start[fs] + k
        a_l.append(order[ok]); b_l.append(order[j[ok]])
    a, b = torch.cat(a_l), torch.cat(b_l)
    dphi = torch.remainder(ang[a] - ang[b] + torch.pi, 2 * torch.pi) - torch.pi
    f = ef[a]
    d = perim[f] / (2 * torch.pi) * dphi.abs()
    side = perim[f] / deg[f].clamp_min(1).to(perim.dtype)
    return a, b, torch.exp(-d / (spread * side).clamp_min(1e-12))


def pcp_rates(F, V, twin, ef, L, nF, Ftot, Vtot, A, k_on, k_off, g, beta, Kb, Kp, kern=None, AV=None,
              hill=None):
    """dF/dt and dV/dt per half-edge for `junction_pcp` (its docstring has the equations).

    F, V, L, A: [E] line densities, lengths and the cue factor of the live half-edges; twin: [E]
    index into the same arrays or -1; ef: [E] face; Ftot, Vtot: [nF] each cell's conserved amounts;
    kern: (a, b, w) from `pcp_side_pairs`, or None for the side-local inhibition; AV: [E] a factor on
    V's unbinding (the elongation coupling), None for 1; hill: (n, K) makes the recruitment by the
    partner across the junction cooperative, x -> (1 + K^n) x^n / (K^n + x^n) (equal to x at x = 1),
    None for linear (`junction_pcp[cooperative]`); (n, K, p) with p [E] in [0, 1] blends the two per
    side, (1 - p) x + p T(x) (`junction_pcp[primed]`).
    A module function so the two-cell test calls exactly what the operator integrates.
    """
    perim = torch.zeros(nF, dtype=F.dtype, device=F.device).index_add(0, ef, L).clamp_min(1e-12)
    Fb = torch.zeros_like(perim).index_add(0, ef, F * L)
    Vb = torch.zeros_like(perim).index_add(0, ef, V * L)
    cF = (Ftot - Fb).clamp_min(0.0) / perim                     # the free pool, per unit perimeter
    cV = (Vtot - Vb).clamp_min(0.0) / perim
    has = twin >= 0
    tw = twin.clamp_min(0)
    Vn = torch.where(has, V[tw], torch.zeros_like(V))           # the neighbour's side of the junction
    Fn = torch.where(has, F[tw], torch.zeros_like(F))
    if kern is None:
        Vs = V
    else:                                                       # Vang nearby along the membrane
        a, b, w = kern
        num = torch.zeros_like(V).index_add(0, a, w * V[b] * L[b])
        den = torch.zeros_like(V).index_add(0, a, w * L[b]).clamp_min(1e-12)
        Vs = num / den
    B = 1.0 + Kb * Vs.clamp_min(0.0) ** Kp                      # Vang on and near this side blocks Fz binding
    if hill is not None:                                        # cooperative recruitment across the junction
        n_, K_ = hill[0], hill[1]
        Tv = (1 + K_ ** n_) * Vn.clamp_min(0.0) ** n_ / (K_ ** n_ + Vn.clamp_min(0.0) ** n_)
        Tf = (1 + K_ ** n_) * Fn.clamp_min(0.0) ** n_ / (K_ ** n_ + Fn.clamp_min(0.0) ** n_)
        if len(hill) > 2:                                       # a per-side weight p: (1 - p) x + p T(x)
            p_ = hill[2]
            Tv, Tf = (1 - p_) * Vn + p_ * Tv, (1 - p_) * Fn + p_ * Tf
        Vn, Fn = Tv, Tf
    dF = k_on * cF[ef] * (beta + g * Vn) - k_off * A * B * F
    dV = k_on * cV[ef] * (beta + g * Fn) - k_off * (V if AV is None else AV * V)
    return dF, dV


def pcp_celsr_q(P, vi, vj, ef, cen, I, nF):
    """Each cell's Celsr1-like nematic q_i (complex), Aigouy et al. 2010's angular integral of the border
    intensity I round the cell: q_i = sum_h I_h int_{phi_a}^{phi_b} exp(2 i phi) dphi / sum_h I_h (phi_b - phi_a),
    phi_a, phi_b the angles of side h's ends about the centroid -- the same readout as the exp08.celsr
    ruler (a uniform I reads 0 on any cell shape). Returns (Re q, Im q), each [nF]; the axis of the
    enriched borders' POSITIONS is half the angle of q.
    """
    ra, rb = P[vi] - cen[ef], P[vj] - cen[ef]
    pa, pb = torch.atan2(ra[:, 1], ra[:, 0]), torch.atan2(rb[:, 1], rb[:, 0])
    dphi = torch.remainder(pb - pa + math.pi, 2 * math.pi) - math.pi
    pb = pa + dphi
    re = torch.zeros(nF, dtype=P.dtype, device=P.device).index_add(0, ef, I * (torch.sin(2 * pb) - torch.sin(2 * pa)) / 2)
    im = torch.zeros(nF, dtype=P.dtype, device=P.device).index_add(0, ef, I * (torch.cos(2 * pa) - torch.cos(2 * pb)) / 2)
    den = torch.zeros(nF, dtype=P.dtype, device=P.device).index_add(0, ef, I * dphi)
    den = torch.where(den.abs() > 1e-30, den, torch.full_like(den, 1e-30))
    return re / den, im / den


def pcp_elongation(P, vi, vj, ef, nF):
    """Each cell's elongation (e1, e2) = sum over its sides of l^2 (cos 2 psi, sin 2 psi) / (2 A), psi
    the side's direction and A the cell's area (shoelace): Aigouy et al. 2010, Suppl. Eq. 25, written
    with one consistent factor. 0 for a regular polygon, positive e1 for a cell long along axis 0."""
    d = P[vj] - P[vi]
    l2 = (d ** 2).sum(1)
    psi = torch.atan2(d[:, 1], d[:, 0])
    A = 0.5 * torch.zeros(nF, dtype=P.dtype, device=P.device).index_add(
        0, ef, P[vi, 0] * P[vj, 1] - P[vj, 0] * P[vi, 1]).abs().clamp_min(1e-12)
    e1 = torch.zeros_like(A).index_add(0, ef, l2 * torch.cos(2 * psi)) / (2 * A)
    e2 = torch.zeros_like(A).index_add(0, ef, l2 * torch.sin(2 * psi)) / (2 * A)
    return e1, e2


@register_operator("junction_pcp", family="polarity", set="vertex", kind="lateral",
                   equation=r"""$$\frac{dF_h}{dt}=k_{\mathrm{on}}c^F_i(\beta+gV_{\bar h})-k_{\mathrm{off}}A_hB_hF_h,\quad \frac{dV_h}{dt}=k_{\mathrm{on}}c^V_i(\beta+gF_{\bar h})-k_{\mathrm{off}}V_h,\quad B_h=1+K_bV_h^{K_p}$$""")
class JunctionPCP(Lateral):
    """Planar cell polarity on the half-edges: two junctional complexes on each cell's side of each
    junction, exchanged across the junction with the neighbour's side.

    vertex -> vertex: reads pos and the half-edge table, writes m["fz"] and m["vang"] (one line
    density per half-edge, recorded as `e_fz` / `e_vang`) and, when the cell set declares them, the
    per-cell blocks `mutant` (the clone), `pcp_vec` (the polarity arrow p_i and its asymmetry) and,
    with `color_block: chem`, three colour channels for the movie: the arrow's share along +cue,
    along -cue and across it, each times min(1, 2 * asymmetry) -- the renderers colour a face from
    `chem` only, so a spec drawing them red / blue / green (`plotting.species`, additive) shows an
    aligned sheet as one red field, a reversed row as blue, a swirl as green. `color_mode: celsr`
    colours the AXIS instead of the arrow, as Aw et al. 2016 image Celsr1 (on both sides of a junction,
    so it has no direction): each cell's nematic q_i of F + V round the cell (`pcp_celsr_q`, the
    exp08.celsr ruler's angular integral) as a colour wheel of its axis theta_i, measured from the
    cue axis: channel k = max(0, cos(2 theta_i - 2 pi k / 3)) times min(1, 2 |q_i|) -- enrichment on
    the borders lying along the cue axis lights channel 0 alone, across it channels 1 and 2 equally,
    the obliques mixtures; an A-P ordered sheet reads one colour whichever way its arrows point.

    THE MODEL is Amonlirdviman et al. 2005's feedback loop (SOM, reactions S1-S10 and PDEs S21-S30)
    reduced to the two complexes the minimal model keeps: F, the Fz(-Dsh) complex, and V, the
    Vang(-Pk) complex. On half-edge h of cell i, with h-bar its twin (cell j's side of the same
    junction) and l_h its length:

        dF_h/dt = k_on c^F_i (beta + g V_hbar) - k_off A_h B_h F_h
        dV_h/dt = k_on c^V_i (beta + g F_hbar) - k_off V_h
        B_h     = 1 + K_b Vs_h^K_p,    Vs_h = sum_{k in i} w_hk V_k l_k / sum_{k in i} w_hk l_k
        w_hk    = exp(-d_hk / (spread * s_i))
        c^F_i   = (F_tot,i - sum_{h in i} F_h l_h) / P_i          (and c^V_i likewise)

    F_h, V_h are line densities (amount per unit junction length) of the complexes on the side h.
    c^F_i is cell i's FREE pool per unit of its perimeter P_i: the SOM lets Dsh and Pk diffuse freely
    in the cell interior, and a well-mixed pool is that diffusion's fast limit (Burak & Shraiman
    2009 make the same fast-diffusion assumption, their Eq. 6). F_tot,i and V_tot,i are conserved per
    cell (SOM: "the total amount of each protein in a cell is always conserved"), set to
    `fz_total`, `vang_total` times the cell's perimeter at the first call.
      g       the EXCHANGE: a complex on one side recruits its partner on the neighbour's side --
              reaction S2, Fz on one cell binding Vang on the next (and S4, S6, S9 on the larger
              complexes). g = 0 switches every cross-junction interaction off (the control).
      beta    recruitment that needs no partner across (the basal rate, relative to g's unit).
      B_h     Vang/Pk on and NEAR the side blocks Fz-Dsh binding: SOM Eq. S1's backward rate times
              B = 1 + K_b (...)^K_p, the local Pk and Vang concentration to the exponent K_p.
      Vs_h    that local concentration, spread along the membrane: d_hk is the distance between the
              midpoints of sides h and k of the same cell along its perimeter, s_i the cell's mean
              side length, `spread` the range in side lengths. WITHOUT IT THE SHEET CANNOT ORDER,
              and that was measured: side-local inhibition (spread 0) polarised every cell (median
              asymmetry 0.07 -> 0.46) and left neighbours uncorrelated (local order 0.07), because
              then any orientation of the junctions is a steady state and nothing puts a cell's Fz
              and Vang on OPPOSITE HALVES. The range is Burak & Shraiman 2009's non-local inhibitory
              field (their Eq. 3, range 1/kappa = 0.45 cell spacings at locus A, Table 1) and the
              reduced form of the SOM's membrane diffusion of Vang and the complexes (PDEs S24-S30).
      A_h     the global cue: the SOM's reaction-based bias multiplies the Dsh-Fz backward rates by
              M1 < 1 "in a region of the distal edge of each cell"; here A_h = `cue_m1` on the sides
              whose outward in-plane normal lies within arccos(`cue_cos`) of `cue_axis`, else 1.
              `cue_until` (a frame, default none) switches it off from that frame on: the SOM
              removed its cue half way through and found the loop keeps the polarity it has.
      k_on, k_off   the binding and unbinding rates, per model time unit.
      elong   Aigouy et al. 2010's coupling of polarity to cell SHAPE (their Eq. 1, -J3 eps_a . Q_a,
              J3/J1 = 0.05 and 0.5 in their Figs. 6C and S5B-D): both complexes unbind from side h at
              k_off times exp(-elong (e1_i cos 2 phi_h + e2_i sin 2 phi_h)), phi_h the side's outward
              normal angle and (e1_i, e2_i) cell i's elongation tensor (`pcp_elongation`, their Suppl.
              Eq. 25). The proteins then prefer the sides facing along the cell's long axis, so the
              polarity AXIS follows elongation. Default 0: no coupling.
      elong_ref   `seed` couples to each cell's elongation RELATIVE TO ITS SHAPE AT THE FIRST CALL,
              (e1 - e1_0, e2 - e2_0), instead of the absolute shape. Measured reason: `seed_mesh`'s
              disc is anisotropic before anything moves (mean |e| 0.18, long axis along y), and with
              the absolute form the coupling read that lattice artefact as a cue -- at elong 1.5 the
              order snapped to y during establishment, at 0.15 it turned to y once the cue was off,
              while the cells' real long axis after the flow was -54 deg (exp08 Finding 12). Relative
              to the seed, a sheet that does not deform feels nothing, exactly. Default `none`.
    A clone removes one complex: `clone: {center, radius, removes: fz|vang}` sets that complex's
    total to 0 in the cells whose centroid lies within `radius` of `center` (in the sheet plane),
    the SOM's loss-of-function clones.

    THE START is random: every side gets `init_bound` of its cell's total density times
    (1 + `init_noise` * U(-1, 1)), independently, from `seed` -- so each cell starts with a random
    arrow and nothing is aligned. `noise` adds a multiplicative Gaussian kick of that relative size
    per step (Burak & Shraiman 2009's stochastic term, Eq. 4), default 0: deterministic, as in the SOM.

    THE STATE IS KEYED BY THE ORDERED VERTEX PAIR (srce, trgt), which names one cell's side of one
    junction, so it follows the half-edge through a renumbering the way the myosin store follows the
    junction. A side with no history takes its cell's current mean. On a sheet whose topology
    changes AFTER this operator in the tick, m["fz"] / m["vang"] are the arrays this operator saw;
    a run with T1s or divisions needs them re-keyed before `topo_record`, as `junction_sync` does
    for myosin -- the fixed sheets of exp08 batch 1 do not.

    A NEW OPERATOR, NOT A VARIANT, because no registered contract holds TWO values per junction that
    talk ACROSS it. The ones it could not be a model of, and why:
      junction_myosin   one multiplier per junction (keyed by the unordered vertex pair) feeding the
                        edge tension `cell_mechanics` reads -- no cell's side, no partner across
      junction_sync     bookkeeping that re-keys that one value after topology changes, no dynamics
      protein_seed / protein_express / protein_project
                        clusters placed by REGION of a cell (apical, basal, mid-surface, interior),
                        not by junction side, and nothing exchanged between neighbours
      seed_polarity     one heading per cell, written once as a seed
      cell_chem_react / cell_chem_diffuse
                        one concentration per cell on the cell graph; a polarity needs one per side
    Its family is `polarity`, of which it is the only member; it lives here, beside the other
    per-junction state and the helpers it shares (`_live_edges`, `_scatter_full`).

    Integrated by explicit Euler, `substeps` steps of `dt` model time units per call; amounts are
    clamped at 0.

    Reference: Amonlirdviman, K., Khare, N. A., Tree, D. R. P., Chen, W.-S., Axelrod, J. D. &
    Tomlin, C. J. (2005). Mathematical modeling of planar cell polarity to understand domineering
    nonautonomy. Science 307:423-426, Supporting Online Material; Burak, Y. & Shraiman, B. I. (2009).
    Order and stochastic dynamics in Drosophila planar cell polarity. PLoS Comput Biol 5:e1000628.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    DIFFERENTIABLE = False
    # DERIVED READOUTS ON THE CELL SET: `mutant`, `pcp_vec` and the colour channels are written from
    # the half-edge state and integrated by nothing, which is what the flag is for (as for the
    # centroid aggregate). No position or velocity is touched; the dynamics live on the mesh table.
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["planar_cell_polarity", "junction_state", "contact_dependent_signalling",
                      "topology_persistent"]
    PARAM_ROLES = {"g": "exchange_gain_across_the_junction", "beta": "basal_recruitment",
                   "spread": "range_of_the_cis_inhibition_along_the_membrane_in_side_lengths",
                   "elong": "coupling_of_polarity_to_cell_elongation",
                   "elong_ref": "elongation_measured_from_the_seeded_shape_or_absolute",
                   "K_b": "cis_inhibition_strength", "K_p": "cis_inhibition_exponent",
                   "cue_m1": "distal_unbinding_factor_of_the_global_cue", "k_on": "binding_rate",
                   "k_off": "unbinding_rate"}
    REFERENCE = ("Amonlirdviman, K. et al. (2005). Mathematical modeling of planar cell polarity to "
                 "understand domineering nonautonomy. Science 307:423-426 (SOM); Burak, Y. & Shraiman, "
                 "B. I. (2009). Order and stochastic dynamics in Drosophila planar cell polarity. "
                 "PLoS Comput Biol 5:e1000628.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.cell_at = params.get("cell_at", "cell")
        self.k_on = float(params.get("k_on", 1.0))
        self.k_off = float(params.get("k_off", 1.0))
        self.g = float(params.get("g", 10.0))
        self.beta = float(params.get("beta", 0.1))
        self.Kb = float(params.get("K_b", 5.0))
        self.Kp = float(params.get("K_p", 2.0))
        self.fz_total = float(params.get("fz_total", 1.0))
        self.vang_total = float(params.get("vang_total", 1.0))
        self.cue_m1 = float(params.get("cue_m1", 1.0))
        self.cue_until = params.get("cue_until", None)
        self.cue_axis = [float(x) for x in params.get("cue_axis", [1.0, 0.0, 0.0])]
        self.cue_cos = float(params.get("cue_cos", 0.5))
        self.plane_axis = int(params.get("plane_axis", 2))
        self.init_bound = float(params.get("init_bound", 0.5))
        self.init_noise = float(params.get("init_noise", 0.5))
        self.noise = float(params.get("noise", 0.0))
        self.seed = int(params.get("seed", 0))
        self.dt = float(params.get("dt", 0.05))
        self.substeps = int(params.get("substeps", 4))
        self.spread = float(params.get("spread", 1.0))
        self.elong = float(params.get("elong", 0.0))
        self.elong_ref = str(params.get("elong_ref", "none"))
        self._e0 = None
        self.clone = params.get("clone") or None
        self.color_block = params.get("color_block") or None
        self.color_mode = str(params.get("color_mode", "arrow"))
        if self.color_mode not in ("arrow", "celsr"):
            raise ValueError(f"junction_pcp: color_mode must be 'arrow' or 'celsr', not {self.color_mode!r}")
        self._tot = None
        self._gen = None
        self._said = False

    def _hill(self):
        """The cooperativity of the cross-junction recruitment: None (linear) for this model."""
        return None

    def _cell_block(self, H, name):
        try:
            lvl = H.level(self.cell_at)
        except Exception:                                                    # noqa: BLE001
            return None
        if name not in getattr(lvl, "state_schema", {}):
            return None
        return lvl.get(name)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        es, live, vi, vj, stride, _ukey, L = _live_edges(m, pos)
        if not bool(live.any()):
            return {}
        nF = int(m["nF"])
        ef = m["E_face"][live].long()
        ax = [i for i in range(pos.shape[1]) if i != self.plane_axis]
        P = pos[:, ax]
        cen = torch.zeros(nF, 2, device=dev, dtype=dt_).index_add(0, ef, P[vi])
        cen = cen / torch.bincount(ef, minlength=nF).clamp_min(1).to(dt_)[:, None]
        d = P[vj] - P[vi]
        n = torch.stack([d[:, 1], -d[:, 0]], 1) / L.clamp_min(1e-12)[:, None]
        n = n * torch.sign(((0.5 * (P[vi] + P[vj]) - cen[ef]) * n).sum(1))[:, None]
        perim = torch.zeros(nF, device=dev, dtype=dt_).index_add(0, ef, L)
        twin = pcp_twins(vi, vj, stride)
        okey = vi * stride + vj
        mid = 0.5 * (P[vi] + P[vj]) - cen[ef]
        kern = pcp_side_pairs(ef, torch.atan2(mid[:, 1], mid[:, 0]), perim, nF, self.spread)

        # ---- first call: totals, clone, the random start --------------------------------------
        if self._tot is None:
            Ft = self.fz_total * perim
            Vt = self.vang_total * perim
            mut = torch.zeros(nF, dtype=torch.bool, device=dev)
            if self.clone:
                c0 = torch.tensor([float(x) for x in self.clone.get("center", [0.0, 0.0])][:2], device=dev, dtype=dt_)
                mut = (cen - c0).norm(dim=1) <= float(self.clone.get("radius", 0.0))
                rem = str(self.clone.get("removes", "fz"))
                if rem not in ("fz", "vang"):
                    raise ValueError(f"junction_pcp: clone.removes must be 'fz' or 'vang', not {rem!r}")
                (Ft if rem == "fz" else Vt)[mut] = 0.0
            self._tot = (Ft, Vt)
            blk = self._cell_block(H, "mutant")
            if blk is not None:
                blk[..., :nF, 0] = mut.to(blk.dtype)
            self._gen = torch.Generator(device="cpu").manual_seed(self.seed)
            u = lambda: (2 * torch.rand(L.shape[0], generator=self._gen, dtype=torch.float64) - 1).to(dev, dt_)  # noqa: E731
            F = self.init_bound * self.fz_total * (1 + self.init_noise * u())
            V = self.init_bound * self.vang_total * (1 + self.init_noise * u())
            F = torch.where(Ft[ef] > 0, F, torch.zeros_like(F))            # a clone cell has none to place
            V = torch.where(Vt[ef] > 0, V, torch.zeros_like(V))
        else:
            keys, sF, sV = m["pcp_keys"], m["pcp_F"], m["pcp_V"]
            order = torch.argsort(keys)
            ks = keys[order]
            j = torch.searchsorted(ks, okey).clamp(max=max(ks.numel() - 1, 0))
            hit = ks[j] == okey
            mF = torch.zeros(nF, device=dev, dtype=dt_).index_add(0, ef, torch.where(hit, sF[order[j]], torch.zeros_like(L)))
            nh = torch.zeros(nF, device=dev, dtype=dt_).index_add(0, ef, hit.to(dt_)).clamp_min(1)
            mV = torch.zeros(nF, device=dev, dtype=dt_).index_add(0, ef, torch.where(hit, sV[order[j]], torch.zeros_like(L)))
            F = torch.where(hit, sF[order[j]], (mF / nh)[ef])
            V = torch.where(hit, sV[order[j]], (mV / nh)[ef])
        Ft, Vt = self._tot
        if Ft.shape[0] != nF:
            raise RuntimeError(f"junction_pcp: the sheet has {nF} cells, the totals were set for "
                               f"{Ft.shape[0]} -- a division or death needs the totals carried")

        cue = torch.tensor(self.cue_axis, device=dev, dtype=dt_)[ax]
        cue = cue / cue.norm().clamp_min(1e-12)
        m1 = self.cue_m1
        if self.cue_until is not None and int(getattr(H, "frame", 0) or 0) >= int(self.cue_until):
            m1 = 1.0
        A = torch.where((n @ cue) >= self.cue_cos, torch.full_like(L, m1), torch.ones_like(L))
        AV = None
        if self.elong != 0.0:
            e1, e2 = pcp_elongation(P, vi, vj, ef, nF)
            if self.elong_ref == "seed":
                if self._e0 is None:
                    self._e0 = (e1.clone(), e2.clone())
                if self._e0[0].shape[0] != nF:
                    raise RuntimeError("junction_pcp: elong_ref seed needs a fixed set of cells")
                e1, e2 = e1 - self._e0[0], e2 - self._e0[1]
            self._eps = (e1, e2, ef)                                         # read by `junction_pcp[primed]`
            phi = torch.atan2(n[:, 1], n[:, 0])
            AV = torch.exp(-self.elong * (e1[ef] * torch.cos(2 * phi) + e2[ef] * torch.sin(2 * phi)))
            A = A * AV
        for _ in range(self.substeps):
            dF, dV = pcp_rates(F, V, twin, ef, L, nF, Ft, Vt, A, self.k_on, self.k_off, self.g,
                               self.beta, self.Kb, self.Kp, kern, AV, self._hill())
            F = (F + self.dt * dF).clamp_min(0.0)
            V = (V + self.dt * dV).clamp_min(0.0)
            if self.noise > 0.0:
                kF = torch.randn(L.shape[0], generator=self._gen, dtype=torch.float64).to(dev, dt_)
                kV = torch.randn(L.shape[0], generator=self._gen, dtype=torch.float64).to(dev, dt_)
                s = self.noise * (self.dt ** 0.5)
                F = (F * (1 + s * kF)).clamp_min(0.0)
                V = (V * (1 + s * kV)).clamp_min(0.0)

        m["pcp_keys"], m["pcp_F"], m["pcp_V"] = okey.detach(), F.detach(), V.detach()
        m["fz"] = _scatter_full(es, live, F.detach(), dev, dt_, fill=0.0)
        m["vang"] = _scatter_full(es, live, V.detach(), dev, dt_, fill=0.0)
        vec = self._cell_block(H, "pcp_vec")
        col = self._cell_block(H, self.color_block) if self.color_block else None
        if vec is not None or col is not None:
            p = torch.zeros(nF, 2, device=dev, dtype=dt_).index_add(0, ef, ((F - V) * L)[:, None] * n)
            s_ = torch.zeros(nF, device=dev, dtype=dt_).index_add(0, ef, (F + V) * L).clamp_min(1e-12)
            asym = p.norm(dim=1) / s_
            if vec is not None:
                vec[..., :nF, 0:2] = p.to(vec.dtype)
                if vec.shape[-1] > 2:
                    vec[..., :nF, 2] = asym.to(vec.dtype)
            if col is not None and col.shape[-1] >= 3 and self.color_mode == "celsr":
                qr, qi = pcp_celsr_q(P, vi, vj, ef, cen, F + V, nF)
                two = torch.atan2(qi, qr) - 2.0 * math.atan2(float(cue[1]), float(cue[0]))   # 2 x the axis, from the cue's
                w = (2.0 * torch.sqrt(qr * qr + qi * qi)).clamp(max=1.0)
                for k in range(3):                                     # a colour wheel of the axis
                    col[..., :nF, k] = (torch.cos(two - 2.0 * math.pi * k / 3).clamp_min(0.0) * w).to(col.dtype)
            elif col is not None and col.shape[-1] >= 3:
                u = p / p.norm(dim=1, keepdim=True).clamp_min(1e-12)
                along = u @ cue
                w = (2.0 * asym).clamp(max=1.0)
                col[..., :nF, 0] = (along.clamp_min(0.0) * w).to(col.dtype)
                col[..., :nF, 1] = ((-along).clamp_min(0.0) * w).to(col.dtype)
                col[..., :nF, 2] = ((1 - along ** 2).clamp_min(0.0).sqrt() * w).to(col.dtype)
        PCP_TRACE.append((nF, float((F * L).sum()), float((V * L).sum())))
        if not self._said:
            print(f"[junction_pcp] {nF} cells, {int(live.sum())} junction sides, g={self.g}, "
                  f"K_b={self.Kb}, K_p={self.Kp}, cue_m1={self.cue_m1}"
                  + (f", clone removes {self.clone.get('removes')}" if self.clone else ""), flush=True)
            self._said = True
        return {}


@register_operator("junction_pcp", model="cooperative", family="polarity", set="vertex", kind="lateral",
                   equation=r"""$$\frac{dF_h}{dt}=k_{\mathrm{on}}c^F_i\big(\beta+g\,T(V_{\bar h})\big)-k_{\mathrm{off}}A_hB_hF_h,\quad T(x)=\frac{(1+K^n)\,x^n}{K^n+x^n}$$""")
class JunctionPCPCooperative(JunctionPCP):
    """`junction_pcp` with a COOPERATIVE exchange: the recruitment of a complex by its partner across
    the junction is a Hill function of the partner's density instead of linear in it.

        dF_h/dt = k_on c^F_i (beta + g T(V_hbar)) - k_off A_h B_h F_h      (and V with T(F_hbar))
        T(x)    = (1 + K^n) x^n / (K^n + x^n)                              T(1) = 1, so g keeps its scale

    n = `hill_n` (default 2) is the cooperativity, K = `hill_k` (default 1) the partner density,
    in the same line-density units as F and V (a cell's total is 1 per unit perimeter), at which the
    recruitment is half its saturated value. Everything else is the default model's.

    K MUST SIT AT THE OPERATING DENSITY, and the two-cell test measured why: at K = 0.5, below the
    ~1-1.5 the junction sides hold, the recruitment saturates where it acts, the junction no longer
    orients and both cells put Vang on the shared side; at K = 1 the slope of T at x = 1 is 2 against
    the linear model's 1 and the junction orients more sharply (Fz 2.12 against 0.018 on the two
    sides, linear 2.32 against 0.11). n = 3 at K = 1 is too cooperative: nothing is recruited from
    low density and the pair sits in the empty state.

    WHY A SECOND MODEL, measured (exp08 Findings 9, 16, 17): with the linear exchange a clone lacking
    Vang makes its wild-type neighbour hold Vang on BOTH ends -- recruited distally by the clone's
    Fz, proximally by its other neighbour's -- so the neighbour's Fz escapes sideways and the rows
    beside the clone circulate round it instead of reversing (0 reversed rows against the paper's
    2-3). In the SOM the clone's unopposed Fz floods that one junction and drains the neighbour's
    Vang pool, which frees its proximal side and lets it flip; a cooperative recruitment makes the
    stronger junction take the pool outright. The SOM's own reactions are cooperative in this sense:
    the complexes grow by successive bindings across the junction (S2, S4, S6, S9 then S5, S8, S10),
    so a junction that already holds a complex recruits the next partner faster.

    Reference: Amonlirdviman, K. et al. (2005). Science 307:423-426, SOM reactions S1-S10.
    """

    PARAM_ROLES = dict(JunctionPCP.PARAM_ROLES, hill_n="cooperativity_of_the_exchange",
                       hill_k="partner_density_at_half_saturation")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.hill_n = float(params.get("hill_n", 2.0))
        self.hill_k = float(params.get("hill_k", 1.0))

    def _hill(self):
        return (self.hill_n, self.hill_k)


@register_operator("junction_pcp", model="primed", family="polarity", set="vertex", kind="lateral",
                   equation=r"""$$\frac{dF_h}{dt}=k_{\mathrm{on}}c^F_i\big(\beta+g\,[(1-p_i)V_{\bar h}+p_i T(V_{\bar h})]\big)-k_{\mathrm{off}}A_hB_hF_h,\quad p_i=\min\big(1,\max_{t'\le t}|\epsilon_i(t')-\epsilon_i(0)|/\epsilon_*\big)$$""")
class JunctionPCPPrimed(JunctionPCPCooperative):
    """`junction_pcp[cooperative]` whose cooperativity a cell ACQUIRES by being stretched: its exchange
    is the default (linear) model's until its shape has changed, and the cooperative model's once it
    has changed by `prime_strain`, for good.

        recruitment by the partner x:  (1 - p_i) x + p_i T(x)       T the cooperative model's Hill function
        p_i = min(1, s_i / prime_strain),   s_i = max over past frames of |eps_i(t) - eps_i(0)|

    eps_i = (e1, e2) is cell i's elongation tensor (`pcp_elongation`, Aigouy et al. 2010), measured from
    its seeded shape (so `elong_ref: seed` and a non-zero `elong` are required, the coupling that
    computes it); |.| its magnitude, dimensionless (0 for a cell with its seed shape; ~0.35 for the
    rig's 20 % stretch). p_i is a memory: it never decreases, so the relaxed tissue keeps the lock.

    WHY (exp08 Phase 2 Findings P8-P13): the linear exchange makes domains on the undeformed sheet
    (r_c 3-4 cell spacings) but forgets the stretch's axis once the cells relax; the cooperative
    exchange keeps the axis but freezes an undeformed sheet in its first rows (r_c 0.28) -- the same
    bistability doing both. Softening it everywhere (Hill n 1.5) lost the memory without making
    domains. Priming by strain keeps the linear model where nothing has been stretched and the
    cooperative one where the stretch has set an axis. It stands for a mechanically induced
    stabilisation of the junctional complexes (Aw et al. 2016 propose the deformation reorganises the
    junctions); the per-cell max-strain memory is this model's assumption, not a measured law.
    """

    PARAM_ROLES = dict(JunctionPCPCooperative.PARAM_ROLES,
                       prime_strain="cell_shape_change_at_which_the_exchange_is_fully_cooperative")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.prime_strain = float(params.get("prime_strain", 0.1))
        if self.elong == 0.0 or self.elong_ref != "seed":
            raise ValueError("junction_pcp[primed] reads each cell's strain from the elongation "
                             "coupling: set elong != 0 and elong_ref: seed")
        self._eps = None
        self._smax = None

    def _hill(self):
        e1, e2, ef = self._eps
        s = torch.sqrt(e1 * e1 + e2 * e2)
        self._smax = s if self._smax is None or self._smax.shape != s.shape else torch.maximum(self._smax, s)
        p = (self._smax / self.prime_strain).clamp(max=1.0)
        return (self.hill_n, self.hill_k, p[ef])
