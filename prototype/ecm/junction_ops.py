"""junction_ops -- myosin as a per-JUNCTION state, and the one operator that makes it survive topology.

WHY A NEW LEVEL AT ALL. Actomyosin is already in the model, as the scalar `Lambda` of
`shape_energy_3d`'s line-tension term `Lambda * sum_e l_e` -- one number for the whole tissue, fixed in
the spec. That is enough to ask "what if there were less myosin everywhere" and not enough for anything
local: a junction cannot be weaker than its neighbours, myosin cannot be recruited where tension is
high, and "the junctions between these two cells failed" is not expressible. `tyssue_atlas.yaml` lists
`contract/increase_linear_tension -> actomyosin_contraction` at `status: refine`, i.e. identified from
tyssue and never built. This is that operator, plus the two things it needs to be usable.

THE HARD PART IS NOT THE DYNAMICS, IT IS TOPOLOGY. `reconnect_t1_3d` (kind `rewire`) rewires the
half-edge relation; `divide_3d` (kind `structural`) adds cells and edges; cell death removes them. In
Plexus's own terms those kinds "mutate the relation or the membership in place and return nothing"
(plexus2, Schedules) -- there is no delta for the engine to reconcile, so any state living on the
junctions is silently invalidated by every one of them. A T1 destroys an edge that HAD a myosin level; a
division creates edges that have none.

SO THE STATE IS KEYED BY TOPOLOGICAL IDENTITY, NOT BY ARRAY INDEX. `junction_myosin` stores myosin
against the unordered vertex pair (min(v_i,v_j), max(v_i,v_j)) that defines the junction, and looks it
up by that key every frame. A rewire can permute the half-edge arrays, a division can lengthen them and
a death can shorten them: an edge that still exists finds its own value, an edge that is new finds
nothing and takes `myo_new`, and an edge that is gone simply stops being asked. `reconnect_t1_3d`,
`divide_3d` and any future death operator need NO edits and no knowledge that myosin exists.

That is the whole design claim: adding a per-junction state must not mean editing three topology
operators, or the next state added has to edit them again. The alternative -- each topology operator
knowing about each state -- is the pattern the discovery engine was refactored to escape.

WHAT `myo_new` IS WORTH ARGUING ABOUT. A junction born from a division has no myosin history, and what
it starts with is a modelling claim with consequences: at `myo_new` below 1 new junctions are
transiently WEAK, which is a real phenomenon and also exactly the sort of thing that will look like an
explanation for anything. It is a parameter, it is reported, and it is swept.

THE KEYED STORE LIVES ON THE MESH, NOT ON THE OPERATOR INSTANCE. It used to be `self._keys`/`self._vals`,
which made `junction_myosin` the only thing in the process that could say what a junction's myosin is.
That is what `junction_myosin_sync` (below) exists to undo: myosin is state of the junction edge-set, so
it is stored where the edge-set is stored and any operator scheduled after a topology change can rebuild
the per-half-edge array from it. This is the `occ`-style discipline of plexus2 sec. "Levels" -- a
cardinality-changing operator must not be able to leave state indexed against a buffer that no longer
exists -- reached without teaching `divide_3d` or `reconnect_t1_3d` that myosin exists.

WHAT WENT WRONG WITHOUT IT, MEASURED. `junction_myosin` writes `m["myo"]` sized to the half-edge buffer
at its own schedule slot (before `shape_energy_3d`); `reconnect_t1_3d` and `divide_3d` then rewire and
lengthen those arrays later in the same frame, and `topo_snapshot_3d` records the new edges beside the
old myosin. In `cellfix_B_new_f401_x4_2cedf4bcc6.npz`, 56 of 200 recorded snapshots had `len(myo)`
short of `len(E_srce)` by 6 to 1356 entries. Every reader downstream indexes myosin positionally
against the edges, so on those frames a junction was drawn and measured with another junction's myosin:
the two half-edges of one junction agreed to 0.0011 on the 144 aligned snapshots and disagreed by 0.216
on the 56 misaligned ones, and the tracked junction of `junction_model.png` moved 0.0095 per snapshot
step between clean frames against 0.1415 across a misaligned one. The high-frequency buzz on that
figure's myosin trace was that misalignment, not the dynamics -- the leaky integrator at tau=20 cannot
move faster than 0.03 per step.
"""
from __future__ import annotations

import torch

from plexus.models.base import Rewire, Structural
from plexus.models.registry import register_operator

# Per frame: (junctions, mean myosin, min, max, how many were NEW this frame). New-junction count is the
# diagnostic that says whether the keying works: it should equal roughly twice the divisions per frame,
# and a run where it equals the whole edge count every frame is a run whose keys are not matching.
MYOSIN_TRACE: list = []


def _edge_key(vi, vj, stride):
    """Unordered vertex pair -> one integer. `stride` must exceed the vertex buffer size."""
    lo = torch.minimum(vi, vj).long()
    hi = torch.maximum(vi, vj).long()
    return lo * stride + hi


def _lookup(m, key, length, vi, vj, stride, myo_new, inherit, dev, dt_, new_val=None):
    """The keyed store on the mesh, mapped onto the half-edge arrays AS THEY ARE NOW.

    Split out of `JunctionMyosin.forward` so that `junction_myosin_sync` can perform exactly the same
    mapping after a topology operator has rewired or lengthened those arrays. One function, two callers,
    so the myosin a frame is RECORDED with is by construction the myosin that frame's mechanics used --
    the two cannot drift apart because there is only one piece of code that decides it.

    Returns `(myo, n_new)` and touches nothing: the store is read, never written.

    `new_val` IS WHAT A JUNCTION WITH NO HISTORY GETS, per half-edge, overriding the scalar `myo_new`.
    A scalar is the right answer only when the stored quantity has a fixed scale, and in the two-pool
    model it does not: the tissue's mean line density runs 1.07 -> 1.97 -> 1.48 over 401 frames, so a
    newborn junction pinned at an absolute 1.0 is set to between 51% and 93% of whatever its
    neighbours happen to hold, for no reason. Passing `new_val = myo_new * n*_f`, with
    n*_f = tau_jun * k_ex * rho_f the density the supply into that cell's belt sustains, makes
    `myo_new` mean what its name says -- a newborn junction as a FRACTION of what a mature one there
    would carry -- and makes it a claim rather than a units accident.
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
    # Not every edge that misses the lookup is a new junction. `divide_3d` inserts a new vertex on
    # each of two edges of the dividing cell and then joins them, so a division produces two KINDS
    # of edge, and only one of them is new:
    #
    #   one endpoint new, one old   -- a SPLIT HALF of the cut junction (a,b). The same physical
    #                                  contact, now in two pieces. It has a myosin history and
    #                                  giving it `myo_new` throws that history away.
    #   both endpoints new          -- the interface between the two daughters. This one really is
    #                                  new, and `myo_new` is the honest answer for it.
    #
    # The parent is recoverable without any help from `divide_3d`, which is the point of keying by
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
    """Per-edge tension, as dE/dl_e of the energy `shape_energy_3d` actually minimises.

    E = K_A(A-A0)^2 + K_P(P-P0)^2 + 0.5 Gam P^2 + K_V(...)^2 + Lambda*sum(m_e l_e), and only the
    perimeter and line terms depend on an individual edge length, so

        T_e = Lambda*m_e + 2 K_P (P_f - P0_f) + Gam * P_f      (f = the edge's own face)

    The CURRENT perimeter is not stored on the mesh -- `face_geometry_3d` computes and discards it --
    so it is recomputed here by the same index_add. P0 is on the mesh; K_P, Gam and Lambda come from
    the spec and MUST match the values `shape_energy_3d` was given, or this is the tension of a
    different tissue.

    RAISES rather than falling back. The first version printed a warning and returned `length`, and
    the smoke test duly printed it -- which means run 86 would have been a LENGTH experiment wearing
    a tension label, and the 86-88 comparison would have been vacuous while looking fine. A silent
    fallback to the exact hypothesis under test is the worst possible failure mode.

    A MODULE FUNCTION rather than a method, because `medioapical_ops` needs the same expression: the
    tension a junction is under is a property of the mesh and the energy, not of whichever operator
    happens to be asking. Two copies of it would be two tensions.
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

    `live` is `E_face < nF`, the same mask `shape_energy_3d` uses, NOT `E_srce < Nv`. The two differ
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
    """Per-junction myosin, recruited by tension, surviving T1 / division / death by construction.

    DYNAMICS, deliberately the simplest thing that is still mechanosensitive:

        myo_ss = activity * (l_e / l_ref)          setpoint rises with junction length, i.e. with tension
        d myo  = (myo_ss - myo) * dt / tau         first-order relaxation toward it

    so a stretched junction recruits myosin and pulls harder, which is the feedback the AVM's constant
    `Lambda` cannot express. `activity` is the myosin-inhibition knob -- the in-silico blebbistatin --
    and it multiplies the setpoint rather than the tension directly, so inhibition takes effect over
    `tau` the way a drug does rather than instantly.

    `l_ref` is the CURRENT MEAN live edge length, not a constant: the tissue triples in radius, so a
    fixed reference would make every junction "stretched" by the end and myosin would rise everywhere
    for a reason that is growth, not tension.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = False        # writes m["myo"], not positions
    MECHANISM_TAGS = ["actomyosin_contraction", "mechanosensitive_recruitment",
                      "junction_state", "topology_persistent"]
    PARAM_ROLES = {"activity": "global_myosin_activity", "tau": "recruitment_timescale",
                   "myo_new": "myosin_of_a_newborn_junction", "beta": "tension_sensitivity"}
    REFERENCE = ("Rauzi, M. et al. (2008) Nat. Cell Biol. 10:1401 (myosin on shrinking junctions); "
                 "Fernandez-Gonzalez, R. et al. (2009) Dev. Cell 17:736 (tension-dependent recruitment).")

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
        rate. Keyed rather than positional because divide_3d and reconnect_t1_3d permute the half-edge
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
        ss = self.activity * (1.0 + sgn * self.beta * (drive / d_ref - 1.0)).clamp_min(0.0)
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
            print(f"[junction_myosin] {int(live.sum())} live junctions, activity={self.activity}, "
                  f"tau={self.tau}, myo_new={self.myo_new}; keyed by vertex pair so T1 / division / "
                  f"death need no edits", flush=True)
            self._said = True
        return {}


# Per frame, only when inheritance fires: (new edges, split halves, halves that found a parent).
INHERIT_TRACE: list = []

# Per frame: (half-edges now, half-edges when myosin was written, live junctions re-keyed). A run whose
# second column never differs from the first is a run in which no topology operator changed the edge
# buffer -- which for `cellfix_B_new` would itself be the bug.
SYNC_TRACE: list = []


@register_operator("junction_myosin_sync", family="mechanics", set="vertex", kind="rewire")
class JunctionMyosinSync(Rewire):
    """Re-key `m["myo"]` onto the half-edge arrays a topology operator has just changed.

    WHY THIS IS AN OPERATOR AND NOT A LINE IN `divide_3d`. Per-FACE state already survives topology:
    `divide_3d` and `apoptosis_3d` both rebuild the mesh through `flat_from_rings_3d` and carry every
    per-face array across with the `keep` map, so `A0`, `P0`, `V0f`, `age` and `ndiv` are never left
    indexed against a buffer that has moved. There is no such carry for per-HALF-EDGE state, because
    until `junction_myosin` there was none. Adding one to each topology operator is the pattern the
    module docstring exists to refuse -- the next per-junction state would have to edit them again.

    So the carry is done from the other side: myosin is keyed by vertex pair, and this operator maps
    the store back onto whatever half-edge arrays now exist. It is `rewire` because it is the relation
    that changed and this is the state following it; it returns nothing and writes one buffer.

    WHERE IT GOES IN THE SCHEDULE. After every operator that can rewire or resize the half-edge arrays
    and before `topo_snapshot_3d`:

        junction_myosin -> shape_energy_3d -> reconnect_t1_3d -> divide_3d
                        -> junction_myosin_sync -> topo_snapshot_3d

    IT CANNOT CHANGE A TRAJECTORY, BY CONSTRUCTION. `shape_energy_3d` reads `m["myo"]` in the same
    frame `junction_myosin` writes it, before any topology operator runs, and next frame's
    `junction_myosin` overwrites it from the store. Nothing between reads it. So the array this
    operator writes is read by exactly one thing -- `topo_snapshot_3d` -- and adding it to a schedule
    leaves every position, every division and every T1 bit-identical. That is a property worth having:
    the fix to a recording defect must not be able to alter what is being recorded.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = False
    MECHANISM_TAGS = ["junction_state", "topology_persistent", "bookkeeping"]
    PARAM_ROLES = {"myo_new": "myosin_of_a_newborn_junction"}
    REFERENCE = "Plexus (this work)."

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
        # THE STORE HOLDS WHAT THE MODEL CHOSE TO STORE, AND `myo` IS WHAT THE MECHANICS READS. For the
        # default model those are the same number and `myo_gain` is 1. For `two_pool` the store is a
        # line density and the multiplier is that density scaled to a tissue mean of `activity`, so the
        # gain is `activity / <n>`; the owning operator leaves it on the mesh and this operator applies
        # it, which is what keeps this one model-agnostic. The gain is the one computed before the
        # topology operators ran, i.e. over ~2,000 junctions of which a handful were just cut -- a mean
        # that does not move at that scale, and stated here rather than hidden.
        gain = float(m.get("myo_gain", 1.0))
        m["myo"] = _scatter_full(es, live, (gain * val).clamp(0.0, 5.0), dev, dt_)
        # AND THE AMOUNT, on the same footing. `myo` is a density (and, for `two_pool`, a normalised
        # one), so no sum of it says how much myosin there is; N_e = (stored density) * l_e does, and
        # it is the quantity the conservation ledger is written in. Zero on dead slots: an amount.
        m["myo_amount"] = _scatter_full(es, live, val * length, dev, dt_, fill=0.0)
        SYNC_TRACE.append((int(es.shape[0]), was, int(live.sum())))
        if not self._said:
            print(f"[junction_myosin_sync] re-keying myosin after the topology operators "
                  f"({was} half-edges written -> {int(es.shape[0])} now)", flush=True)
            self._said = True
        return {}
