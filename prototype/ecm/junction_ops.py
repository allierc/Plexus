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
"""
from __future__ import annotations

import torch

from plexus.models.base import Structural
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
        self.inherit = bool(params.get("inherit", True))
        self._vseen = None
        self.dt = float(params.get("dt", 1.0))
        self._keys = None            # int64 [K] topological identities seen so far
        self._vals = None            # float  [K] their myosin
        self._said = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        pos = lvl.get("pos")
        es, et, ef = m["E_srce"], m["E_trgt"], m["E_face"]
        nF, Nv = int(m["nF"]), int(m["Nv"])
        dev, dt_ = pos.device, pos.dtype
        live = ef < nF
        if not bool(live.any()):
            return {}
        vi, vj = es[live].long(), et[live].long()
        stride = max(pos.shape[0], Nv) + 1
        key = _edge_key(vi, vj, stride)

        length = (pos[vj] - pos[vi]).norm(dim=-1).to(dt_)
        l_ref = length.mean().clamp_min(1e-9)

        # ---- look each live junction up by its identity ------------------------------------------
        if self._keys is None:
            self._keys = torch.empty(0, dtype=torch.long, device=dev)
            self._vals = torch.empty(0, dtype=dt_, device=dev)
        order = torch.argsort(self._keys)
        ks, vs = self._keys[order], self._vals[order]
        idx = torch.searchsorted(ks, key)
        idx_c = idx.clamp(max=max(ks.numel() - 1, 0))
        found = (ks.numel() > 0) & (idx_c < ks.numel())
        hit = found & (ks[idx_c] == key) if ks.numel() else torch.zeros_like(key, dtype=torch.bool)
        base = vs[idx_c] if vs.numel() else torch.zeros_like(length)
        myo = torch.where(hit, base, torch.full_like(length, self.myo_new))
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
        if self.inherit and n_new and self._vseen is not None:
            oi = self._vseen[vi.clamp(max=self._vseen.numel() - 1)] & (vi < self._vseen.numel())
            oj = self._vseen[vj.clamp(max=self._vseen.numel() - 1)] & (vj < self._vseen.numel())
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
                inh = torch.where(phit, vs[pidx] if vs.numel() else torch.zeros_like(pkey, dtype=dt_),
                                  torch.full_like(pkey, self.myo_new, dtype=dt_))
                myo = myo.masked_scatter(half, inh[inv])
                n_inherited = int(phit[inv].sum())
                INHERIT_TRACE.append((n_new, int(half.sum()), n_inherited))

        # ---- recruitment ------------------------------------------------------------------------
        ss = self.activity * (1.0 + self.beta * (length / l_ref - 1.0)).clamp_min(0.0)
        myo = myo + (ss - myo) * (self.dt / max(self.tau, 1e-9))
        myo = myo.clamp(0.0, 5.0)

        # ---- store back, keyed. Junctions that no longer exist are simply not written, so a T1 or a
        # death drops them without anyone having to notice.
        self._keys, self._vals = key.detach().clone(), myo.detach().clone()
        # which vertices existed this frame, so next frame can tell an inserted vertex from an old one
        vs_seen = torch.zeros(stride, dtype=torch.bool, device=dev)
        vs_seen[vi] = True; vs_seen[vj] = True
        self._vseen = vs_seen
        # the per-half-edge array the mechanics reads: full length, 1.0 on dead slots so a masked-out
        # half-edge contributes its usual tension if anything ever reads it unmasked
        full = torch.ones(es.shape[0], device=dev, dtype=dt_)
        full[live] = myo
        m["myo"] = full
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
