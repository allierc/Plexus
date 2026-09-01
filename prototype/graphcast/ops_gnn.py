"""A GENERAL MESSAGE-PASSING RULE as a learnable Plexus2 field operator, with `embedding:` as the
switch between the plain GNN and the GNN + Instant-NGP hashtable.

WHERE THIS SITS IN THE LADDER. Four families fit the same data, each strictly more general than the
last, so a failure can be attributed rather than merely observed:

    ops_known_ode.py   the true equation, constants learnable       upper bound
    ops_gnn.py         THIS FILE -- message + update, both MLPs     "can a graph rule find it?"
    ops_graphcast.py   the GraphCast form, edge latents, deep       "does the extra machinery pay?"

KIND IS `field`, AND THE LATTICE IS THE GRAPH. The toy has no node set: both its rules are field
rules on a grid, so a message-passing operator over the 4 (2-D) or 6 (3-D) nearest neighbours is
computed by rolls. That is not an approximation of a GNN, it IS one -- a translation-invariant
message-passing layer on a regular lattice and a convolution are the same operator (checked
elsewhere in this workspace to 1.19e-6), and writing it with rolls costs nothing and keeps the
operator dimension-generic. When the ZAPBench point cloud arrives the same message and update MLPs
take an explicit edge list instead; the neighbourhood changes, the rule does not.

WHY THIS FORM AND NOT AN ARBITRARY NETWORK. The known-ODE file shows that the Kuramoto rule, written
in the observables, IS a message-passing layer:

    r_i      =  omega_i  +  K SUM_j ( v_j w_i - w_j v_i )
    dv_i/dt  =  w_i r_i m_i,     dw_i/dt = -v_i r_i m_i

with `K` the edge weight, `omega_i` an ADDITIVE PER-NODE EMBEDDING, and (w_i, -v_i) a receiver-side
gauge. So the general form below contains the truth as a special case, and the question a fit
answers is whether gradient descent finds it -- not whether the architecture could express it.

THE MESSAGE IS MULTIPLICATIVE IN THE EMBEDDING'S SPIRIT, NOT CONCATENATIVE ONLY. `a_i` enters both
the message input and the update input, so heterogeneity acts at the point of interaction rather
than being added afterwards. That placement is the one thing the GraphCast form otherwise lacks
(GraphCast has no per-node embedding at all -- weather grid points have no hidden identity), and it
is the property this prototype exists to keep.

THE `embedding:` SWITCH IS THE PARTITION BETWEEN THE TWO FAMILIES, expressed as an option and not a
fork, per the prototype's first requirement:

    none    no a_i. The rule is homogeneous; it CANNOT represent omega_i, so this is the control
            that says how much of the fit the heterogeneity is responsible for.
    free    one unconstrained vector per cell. Maximum capacity, no prior, and at 1024^2 that is
            more parameters than there are observations per frame.
    ngp     the Instant-NGP ladder-hashtable of `ops_embedding.py`, indexed by POSITION with corner
            interpolation. Far fewer parameters, and a spatial smoothness prior for free -- measured
            cosine similarity 0.744 within 0.01 of the domain against 0.010 beyond 0.5.

WHAT `ngp` IS AND IS NOT EXPECTED TO DO. It is NOT expected to discover the toy's coarse/fine split:
levels do not specialise by frequency (measured fine/smooth energy ratio 0.58-1.14) and collisions
do not prioritise (gradient mass at a shared row splits 0.507 at equal loss weight). What it is
expected to buy is capacity spent only where samples land -- this toy's fine rule occupies 15% of
the domain -- and smoothness without a regulariser. Whether that beats `free` at a matched parameter
budget is exactly the open question, and it is why both are options of one operator rather than two
operators with a story attached.
"""

from __future__ import annotations

import torch
from torch import nn

from ops_embedding import Ladder
from plexus.models.base import FieldUpdate, Lateral
from plexus.models.registry import register_operator


def _mlp(sizes, act=nn.GELU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


@register_operator("gnn_field", family="signalling", set="field", kind="field",
                   model="message_passing")
class GNNField(FieldUpdate):
    """dstate/dt from a learned message and a learned update over the lattice neighbourhood.

        m_ij  =  lin_edge( s_i, s_j, a_i, a_j )       one MLP, shared over edges
        M_i   =  SUM_{j in N(i)} m_ij                  sum, so the aggregate is permutation-invariant
        ds_i  =  lin_phi( s_i, M_i, a_i ) * mask_i     one MLP, shared over cells

    `a_i` ENTERS BOTH, which is the placement ParticleGraph uses and the one that matters. In
    `Interaction_Particle` the embedding is concatenated into `lin_edge`'s input (`embedding_i`, and
    `embedding_j` too in the PDE_A_bis and PDE_E variants) AND into `lin_phi`'s. Putting it only in
    the update would make heterogeneity a per-cell gain applied after the interaction; putting it in
    the message makes it modify the interaction itself, which is what a cell type does. GraphCast
    has no per-node embedding at all -- weather grid points carry no hidden identity -- so this is
    the one place the two lineages genuinely differ, and it is ours to keep.

    THE AGGREGATE IS A SUM AND NOT A MEAN, because the true rule's aggregate is a sum: a mean would
    divide by a neighbour count that is constant on a lattice, folding a factor of 2D into the
    update MLP where it cannot be read off. On an irregular graph that choice stops being free and
    the sum is still the one that matches the physics.

    THE MASK MULTIPLIES THE OUTPUT, not the input. The fine rule acts only inside its regions, and
    outside them the field is identically zero; masking the increment says "no dynamics here",
    while masking the input would say "no neighbours here" and would silently change the stencil at
    the boundary of every disc.
    """

    INPUTS: list = []
    OUTPUTS: list = []
    READS: list = []
    WRITES: list = []
    MAPS: list = []
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["embedding", "hidden_dim"]
    MECHANISM_TAGS = ["message_passing", "learned_interaction", "heterogeneity"]
    PARAM_ROLES = {"lin_edge": "message_function", "lin_phi": "update_function",
                   "a": "per_cell_embedding", "n_passes": "message_passing_depth"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")
        self.embedding = str(params.get("embedding", "free"))
        if self.embedding not in ("none", "free", "ngp"):
            raise ValueError(f"gnn_field embedding must be none|free|ngp, got {self.embedding!r}")
        self.emb_dim = 0 if self.embedding == "none" else int(params.get("embedding_dim", 4))
        self.hidden = int(params.get("hidden_dim", 32))
        self.n_passes = int(params.get("n_passes", 1))
        self.dt = self.tunable(params.get("dt"), 1.0)
        self.substeps = int(params.get("substeps", 1))
        self.ngp = dict(params.get("ngp") or {})
        self.seed = int(params.get("seed", 0))
        self.device_ = device
        self.lin_edge = self.lin_phi = self.a = self.ladder = None
        self._mask = None
        self._pos = None

    def bind(self, shape, mask=None):
        """Allocate against the field's shape. Same contract as `KuramotoKnownODE.bind`.

        The MLPs cannot be built in `__init__` because their input width depends on the number of
        CHANNELS the field carries, and the embedding cannot be allocated because its extent is the
        grid's. Both are properties of the hierarchy, which does not exist until a spec is built.
        """
        n_dim = len(shape)
        self._shape = tuple(shape)
        self._mask = (torch.ones(shape, device=self.device_) if mask is None
                      else mask.to(self.device_))
        C = self._C
        torch.manual_seed(self.seed)
        # NAMED AFTER ParticleGraph's `Interaction_Particle`, the reference implementation in this
        # workspace: `lin_edge` is the message, `lin_phi` the update. connectome-gnn calls the same
        # two `g_phi` and `f_theta`. Matching a name costs nothing and means the regularisers, the
        # plots and the gates written against those repos read straight across.
        self.lin_edge = _mlp([2 * C + 2 * self.emb_dim, self.hidden, self.hidden]).to(self.device_)
        self.lin_phi = _mlp([C + self.hidden + self.emb_dim, self.hidden, C]).to(self.device_)
        if self.embedding == "free":
            # ONES, NOT ZEROS, following ParticleGraph (`self.a = nn.Parameter(torch.ones(...))`)
            # and connectome-gnn's NeuralGNN. It is the convention every fitted model in this
            # workspace uses, so a recovered `a` compares with those runs without a rescale.
            self.a = nn.Parameter(torch.ones(self.emb_dim, *shape, device=self.device_))
        elif self.embedding == "ngp":
            self.ladder = Ladder(self.emb_dim, n_dim,
                                 int(self.ngp.get("n_levels", 8)),
                                 int(self.ngp.get("n_min", 4)),
                                 int(self.ngp.get("n_max", 512)),
                                 int(self.ngp.get("table_size", 2 ** 14)),
                                 self.seed, self.device_)
            axes = torch.meshgrid(*[(torch.arange(n, device=self.device_,
                                                  dtype=torch.float32) + 0.5) / n
                                    for n in shape], indexing="ij")
            self._pos = torch.stack([x.reshape(-1) for x in axes], -1)
        return self

    def _embedding(self):
        """[E, *shape], or None. `ngp` is recomputed each call because its table is what is fitted."""
        if self.embedding == "none":
            return None
        if self.embedding == "free":
            return self.a
        return self.ladder(self._pos).T.reshape(self.emb_dim, *self._shape)

    def _increment(self, s, a):
        """One message-passing pass. `s` is [C, *shape], `a` is [E, *shape] or None."""
        D = s.dim() - 1
        si = s if a is None else torch.cat([s, a], 0)
        agg = None
        for d in range(D):
            for shift in (1, -1):
                sj = torch.roll(si, shift, d + 1)
                x = torch.cat([si, sj], 0)                       # [2(C+E), *shape]
                m = self.lin_edge(x.flatten(1).T).T                   # [H, n_cells] -> [H, *shape]
                agg = m if agg is None else agg + m
        agg = agg.reshape(self.hidden, *s.shape[1:])
        z = torch.cat([s, agg] + ([] if a is None else [a]), 0)
        return self.lin_phi(z.flatten(1).T).T.reshape(s.shape)

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        if self.lin_edge is None:
            self._C = fld.grid.shape[0]
            self.bind(tuple(fld.grid.shape[1:]))
        s = fld.grid
        a = self._embedding()
        for _ in range(self.substeps):
            ds = s
            for _ in range(self.n_passes):
                ds = self._increment(ds, a)
            s = s + self.dt * ds * self._mask
        fld.grid = s
        return {}


@register_operator("gnn_message", family="signalling", set="particle", kind="lateral",
                   model="message_passing")
class GNNMessage(Lateral):
    """THE SAME RULE ON A SET, over whatever graph a `rewire` operator built. The canonical form.

    `GNNField` above computes its neighbourhood with `torch.roll`, which bakes the 4/6-point
    lattice stencil INTO THE CODE. That is fast, dimension-generic and wrong in one specific way:
    the neighbourhood stops being a statement in the spec. It cannot be widened, it cannot be a
    kNN, and it cannot move to an irregular point cloud -- so the operator that will eventually
    read ZAPBench's 71,721 somata would have to be a different operator.

    This is the same message and update over `lvl.edge_index`, which is the composition every
    interaction model in the library uses (config/active_matter/vicsek_4t.yaml):

        operators: [{op: radius_graph, at: pixel, radius: 0.006}, {op: gnn_message, at: pixel, ...}]
        schedule:  [radius_graph, gnn_message]

    Three consequences, and the first is the one that was hiding:

      * THE NEIGHBOURHOOD IS A NUMBER IN THE FILE. Measured on a lattice, `radius: 1.5 cells`
        gives EIGHT neighbours, not four -- it catches the diagonals at sqrt(2). The roll version
        was silently choosing four, and no reader of the spec could have known.
      * `EMIT = "velocity"` -- this returns a per-node dv/dt, and `plexus.engine._integrate` reads
        EMIT to decide whether the delta is a rate (x += dt*delta) or an acceleration. A
        `FieldUpdate` has no delta and inherits `EMIT = None`; a `Lateral` must declare one, and
        omitting it is not a style lapse but a wrong integration.
      * `REQUIRES_PARAMS` is checked by `plexus.schema` AT LOAD TIME, so a spec missing `radius`
        or `hidden_dim` is rejected before anything builds, with the offending key named.

    THE SCALE CEILING IS REAL AND IS RECORDED HERE. `radius_graph` is O(N^2) blockwise -- its own
    docstring says 1e4-1e5 nodes. Measured on a lattice: 64^2 0.05 s / 0.2 GB, 256^2 1.02 s /
    3.2 GB, 512^2 16.5 s / 12.9 GB, and 1024^2 would need ~206 GB. So this operator runs the toy
    at 256^2 and the 1024^2 fine field stays with `GNNField`. Two operators for one rule is a cost;
    pretending the graph version scales to a million nodes would be a bigger one.
    """

    EMIT = "velocity"                            # returns dv/dt: a RATE, not an acceleration
    INPUTS: list = []
    OUTPUTS: list = []
    READS: list = []
    WRITES: list = []
    MAPS: list = []
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["embedding", "hidden_dim"]
    MECHANISM_TAGS = ["message_passing", "learned_interaction", "heterogeneity"]
    PARAM_ROLES = {"lin_edge": "message_function", "lin_phi": "update_function",
                   "a": "per_node_embedding", "n_passes": "message_passing_depth",
                   "block": "the state block the rule acts on"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at") or params.get("at", "particle")
        self.block = str(params.get("block", "voltage"))
        self.embedding = str(params.get("embedding", "free"))
        if self.embedding not in ("none", "free", "ngp"):
            raise ValueError(f"gnn_message embedding must be none|free|ngp, got {self.embedding!r}")
        self.emb_dim = 0 if self.embedding == "none" else int(params.get("embedding_dim", 16))
        self.hidden = int(params["hidden_dim"])
        self.n_passes = int(params.get("n_passes", 1))
        self.ngp = dict(params.get("ngp") or {})
        self.seed = int(params.get("seed", 0))
        self.device_ = device
        self.lin_edge = self.lin_phi = self.a = self.ladder = None

    def _build(self, C, N, pos):
        torch.manual_seed(self.seed)
        self.lin_edge = _mlp([2 * C + 2 * self.emb_dim, self.hidden, self.hidden]).to(self.device_)
        self.lin_phi = _mlp([C + self.hidden + self.emb_dim, self.hidden, C]).to(self.device_)
        if self.embedding == "free":
            self.a = nn.Parameter(torch.ones(N, self.emb_dim, device=self.device_))
        elif self.embedding == "ngp":
            self.ladder = Ladder(self.emb_dim, pos.shape[-1],
                                 int(self.ngp.get("n_levels", 8)), int(self.ngp.get("n_min", 4)),
                                 int(self.ngp.get("n_max", 512)),
                                 int(self.ngp.get("table_size", 2 ** 14)), self.seed, self.device_)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        x = lvl.get(self.block)
        N, C = x.shape[0], x.shape[-1]
        if self.lin_edge is None:
            self._build(C, N, pos)
        a = (None if self.embedding == "none"
             else (self.a if self.embedding == "free" else self.ladder(pos)))
        ei = lvl.edge_index                       # [2, E]: row 0 receiver i, row 1 neighbour j
        if ei.numel() == 0:
            return {self.at: torch.zeros_like(x)}
        i, j = ei[0], ei[1]
        s = x if a is None else torch.cat([x, a], -1)
        for _ in range(self.n_passes):
            m = self.lin_edge(torch.cat([s[i], s[j]], -1))          # [E, H]
            agg = torch.zeros(N, self.hidden, device=x.device, dtype=x.dtype)
            agg.index_add_(0, i, m * lvl.occ[j][:, None])           # SUM at the receiver
            z = torch.cat([x, agg] + ([] if a is None else [a]), -1)
            dx = self.lin_phi(z)
            s = dx if a is None else torch.cat([dx, a], -1)
        return {self.at: dx * lvl.occ[:, None]}
