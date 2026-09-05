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

then the second model of `junction_myosin`, a different hypothesis in the same slot:

    junction_myosin[two_pool]  the belt fed by the medioapical pool, as a conserved amount

The medioapical operators live in this file because the two-pool model is not a separate
mechanism: it is the same junction bookkeeping with a second reservoir on the face, and it calls
the same helpers -- `_live_edges`, `_lookup`, `_scatter_full`, `edge_tension`. Split across two
files, the shared half of one model would be private to the other.
"""
from __future__ import annotations
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


@register_operator("junction_myosin", family="mechanics", set="vertex", kind="structural")
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


@register_operator("junction_sync", family="mechanics", set="vertex", kind="rewire")
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


@register_operator("medioapical_myosin", family="mechanics", set="cell", kind="lateral")
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
        # number. `myo_new` used to be a bare density, and because the tissue's mean line density runs
        # 1.07 -> 1.97 -> 1.48 over the run, `myo_new = 1.0` set a new junction to 51% of its
        # neighbours at frame 100 and 93% at frame 0 -- a visible, meaningless dimming that had nothing
        # to do with the parameter's intent.
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
        # A FLAG AND NOT A REPLACEMENT, because the absolute reading is the one every cache written
        # before 10 August was built with, and a silent change of meaning under an unchanged cache key
        # is how an archived run stops being reproducible without anything failing. `False` reproduces
        # those bit-for-bit; `True` is the corrected default and is what enters the cache key.
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


@register_operator("cytokinetic_ring", family="mechanics", set="vertex", kind="structural")
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
