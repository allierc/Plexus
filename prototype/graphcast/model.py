"""The model: four architectural options behind one contract.

    encoder_decoder  off | on          transfer through a background grid, or straight on the set
    message          simple | graphcast  NeuralGNN's form, or the edge-stateful one
    n_passes         >= 1               1 is NeuralGNN, 16 is GraphCast
    embedding        none | free | multires

THE `simple` PATH IS NeuralGNN, ARITHMETICALLY, and that is the point of G5. It is written here to
match `connectome_gnn.models.neural_gnn.NeuralGNN._compute_messages` term for term:

    msg_i = SUM_{e: post(e)=i}  W_e * g_phi([v_pre(e), a_pre(e)])          (neural_gnn.py:630-653)
    dv_i  = f_theta([v_i, a_i, msg_i, excitation_i])                       (neural_gnn.py:685-686)

so that with the weights copied across, the two produce the same numbers. Everything after G5 is
then a controlled variation on something already known to work, rather than a new model whose
failures cannot be attributed.

THE `graphcast` PATH is the processor of supplement Eqs 11-13, with the residuals of Eq 13:

    e_ij <- e_ij + LN(MLP_e([e_ij || v_j || v_i]))
    v_i  <- v_i  + LN(MLP_v([v_i  || SUM_j e_ij]))

Both MLPs' final layers are initialised to ZERO, so every residual block is the identity at
initialisation and `n_passes` 1 and 16 give bit-identical output before training. That is G6, and
it is a property of the construction rather than something to be checked after the fact: a residual
stack that does not start at the identity is a different model at every depth.

THE `multires` EMBEDDING is a hashed multi-level lookup indexed by NODE, not by position. Keying it
on position would be wrong here and the toy proves it: types are spatially unstructured by
construction (G16), so a positional embedding cannot carry type. Indexing by node with each level
holding FEWER ROWS THAN NODES is the version that has a mechanism -- capacity below the cluster
count forces neurons to share an entry, and the sharing is what a type is. On the real ladder in
ngp-demo the first nine levels are dense and collide not at all, which is why the sharing there
came from grid quantisation and not from collisions; here the undersizing is deliberate.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def mlp(sizes, activation=nn.SiLU, zero_final=False):
    layers = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        if zero_final and i == len(sizes) - 2:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class MultiResNodeEmbedding(nn.Module):
    """a_i as a concatenation of hashed lookups, each level holding fewer rows than there are nodes.

    Level l has T_l = max(2, ceil(N / ratio^(L-1-l))) rows, so the coarsest level is the most
    contended. A node reads one row per level and the rows are concatenated, exactly as a
    multiresolution encoding concatenates its level slices.
    """

    def __init__(self, n_nodes: int, dim: int, n_levels: int = 3, ratio: float = 6.0,
                 seed: int = 0):
        super().__init__()
        self.n_levels = n_levels
        g = torch.Generator().manual_seed(seed)
        per = max(1, dim // n_levels)
        self.slice_dims = [per] * (n_levels - 1) + [dim - per * (n_levels - 1)]
        self.tables = nn.ParameterList()
        idx = []
        for l in range(n_levels):
            rows = max(2, int(math.ceil(n_nodes / (ratio ** (n_levels - 1 - l)))))
            self.tables.append(nn.Parameter(torch.zeros(rows, self.slice_dims[l])))
            # a fixed pseudo-random assignment of nodes to rows: the collision pattern is a
            # property of the model, not of the run, so it is a buffer and not re-drawn.
            idx.append(torch.randint(0, rows, (n_nodes,), generator=g))
        for l, ix in enumerate(idx):
            self.register_buffer(f"idx{l}", ix)

    def forward(self) -> torch.Tensor:
        return torch.cat([self.tables[l][getattr(self, f"idx{l}")] for l in range(self.n_levels)],
                         dim=-1)


class GraphCastModel(nn.Module):
    """One module, four options. `forward` returns dv/dt for every node."""

    def __init__(self, ms, n_nodes: int, n_edges: int, device: str = "cpu", seed: int = 0):
        super().__init__()
        self.ms = ms
        self.n_nodes, self.n_edges = n_nodes, n_edges
        self.emb_dim = 0 if ms.embedding == "none" else ms.embedding_dim

        if ms.embedding == "free":
            # ones, matching the NeuralGNN convention (drosophila_cx_task_gnn.py:459)
            self.a = nn.Parameter(torch.ones(n_nodes, self.emb_dim))
            self.multires = None
        elif ms.embedding == "multires":
            self.a = None
            mr = ms.multires or {}
            self.multires = MultiResNodeEmbedding(
                n_nodes, self.emb_dim, int(mr.get("n_levels", 3)),
                float(mr.get("ratio", 6.0)), seed)
        else:
            self.a, self.multires = None, None

        # the edge weight, learnable, and the object G9 scores against the ground-truth kernel
        self.W = nn.Parameter(torch.zeros(n_edges))
        nn.init.normal_(self.W, 0.0, 1.0 / math.sqrt(max(n_edges / n_nodes, 1.0)))

        # the per-neuron stimulus gain: the object G13 scores, and the readout on real data
        self.b = nn.Parameter(torch.zeros(n_nodes))

        h, nh = ms.hidden_dim, ms.n_hidden_layers
        if ms.message == "simple":
            self.g_phi = mlp([2 + self.emb_dim] + [h] * nh + [1])
            self.f_theta = mlp([1 + self.emb_dim + 1 + 1] + [h] * nh + [1])
        else:
            d = ms.latent_dim
            self.encode_v = mlp([1 + self.emb_dim + 1] + [h] * nh + [d])
            self.encode_e = mlp([1] + [h] * nh + [d])        # from the scalar edge weight
            self.mlp_e = nn.ModuleList(
                [mlp([3 * d] + [h] * nh + [d], zero_final=True) for _ in range(ms.n_passes)])
            self.mlp_v = nn.ModuleList(
                [mlp([2 * d] + [h] * nh + [d], zero_final=True) for _ in range(ms.n_passes)])
            self.ln_e = nn.ModuleList([nn.LayerNorm(d) for _ in range(ms.n_passes)])
            self.ln_v = nn.ModuleList([nn.LayerNorm(d) for _ in range(ms.n_passes)])
            self.decode = mlp([d] + [h] * nh + [1])
        self.to(device)

    # ------------------------------------------------------------------ #
    def embedding(self) -> torch.Tensor | None:
        if self.a is not None:
            return self.a
        if self.multires is not None:
            return self.multires()
        return None

    def _cat(self, *parts):
        return torch.cat([p for p in parts if p is not None], dim=-1)

    def forward(self, v: torch.Tensor, edge_index: torch.Tensor,
                stim: torch.Tensor, return_msg: bool = False,
                return_edge_feat: bool = False):
        """v [N,1] state, edge_index [2,E] as (pre, post), stim [N,1] the per-node drive.

        `return_msg` also hands back the aggregated message, which is what G9 scores against the
        true field gradient: on this toy the fine rule IS a spatial derivative, so "did the model
        recover the interaction" and "did the message become a gradient operator" are the same
        question, and the second is the one that can be measured directly.
        """
        pre, post = edge_index[0], edge_index[1]
        a = self.embedding()
        drive = self.b[:, None] * stim

        if self.ms.message == "simple":
            # THE SENDER'S DRIVE IS PART OF THE MESSAGE. The fine rule is du/dx, so the message
            # has to carry the neighbour's value of u; built from v alone it could only ever
            # approximate the gradient through the states u already produced, which is a
            # different and much weaker signal. NeuralGNN passes only (v_j, a_j) because its
            # excitation is per-receiver; here the drive is per-sender and spatial.
            feat = self._cat(v[pre], stim[pre], None if a is None else a[pre])
            g = self.g_phi(feat)                                   # [E, 1]
            edge_msg = self.W[:, None] * g
            msg = torch.zeros(self.n_nodes, 1, device=v.device, dtype=v.dtype)
            msg = msg.index_add(0, post, edge_msg)                 # sum over incoming edges
            out = self.f_theta(self._cat(v, a, msg, drive))
            if return_edge_feat:
                return out, feat
            return (out, msg) if return_msg else out

        hv = self.encode_v(self._cat(v, a, drive))
        he = self.encode_e(self.W[:, None])
        for k in range(self.ms.n_passes):
            he = he + self.ln_e[k](self.mlp_e[k](torch.cat([he, hv[pre], hv[post]], dim=-1)))
            agg = torch.zeros_like(hv).index_add(0, post, he)
            hv = hv + self.ln_v[k](self.mlp_v[k](torch.cat([hv, agg], dim=-1)))
        out = self.decode(hv)
        if return_edge_feat:
            return out, None            # the graphcast path has no separable g_phi to regularise
        return (out, agg[:, :1]) if return_msg else out


def copy_weights_from_neural_gnn(dst: "GraphCastModel", src) -> None:
    """G5: make the two models share parameters so their outputs can be compared directly.

    Copies NeuralGNN's `W`, `a`, `g_phi` and `f_theta` into the `simple` path. Raises if the
    shapes disagree, because a silent partial copy would turn G5 from a check into a formality.
    """
    with torch.no_grad():
        dst.W.copy_(src.W.detach().reshape(-1)[: dst.n_edges])
        if dst.a is not None:
            dst.a.copy_(src.a.detach()[: dst.n_nodes, : dst.emb_dim])
        for d, s in ((dst.g_phi, src.g_phi), (dst.f_theta, src.f_theta)):
            dl = [m for m in d if isinstance(m, nn.Linear)]
            sl = [m for m in s.modules() if isinstance(m, nn.Linear)]
            if len(dl) != len(sl):
                raise ValueError(f"layer count differs: {len(dl)} here vs {len(sl)} in NeuralGNN")
            for a_, b_ in zip(dl, sl):
                if a_.weight.shape != b_.weight.shape:
                    raise ValueError(f"shape differs: {tuple(a_.weight.shape)} vs "
                                     f"{tuple(b_.weight.shape)}")
                a_.weight.copy_(b_.weight)
                a_.bias.copy_(b_.bias)


class PositionHashEmbedding(nn.Module):
    """a_i as an INSTANT-NGP encoding of the node's POSITION: spatial hash + corner interpolation.

    THE TWO THINGS THAT MAKE AN NGP AN NGP, and that `MultiResNodeEmbedding` above does not have.
    That class hashes a node's INDEX through a fixed random draw, so it has the collisions and the
    level ladder but nothing spatial: two adjacent nodes get unrelated rows, and two distant nodes
    are as likely to share one as two neighbours are. Here:

        1. the index is a SPATIAL hash of the integer corner coordinates of the node's cell;
        2. the 2^D corners of that cell are MULTILINEARLY INTERPOLATED.

    Together those change the prior qualitatively rather than quantitatively. Nearby nodes share
    corners, so their embeddings are correlated BY CONSTRUCTION -- spatial smoothness for free,
    where `free` and the index-hash must learn it or not have it. And collisions become a property
    of distance: at a level whose grid is finer than the table, the nodes sharing a row are far
    apart, which is the structure the paper relies on.

    WHAT THAT LAST PROPERTY MEANS HERE IS NOT WHAT IT MEANS IN A NeRF, and the difference is the
    reason this is an experiment and not an improvement. With a pointwise decoder a shared row is
    shared capacity and the concatenation across levels still separates the two points. With a
    MESSAGE-PASSING decoder, two distant nodes sharing a feature is a manufactured long-range
    coupling, and recovering which nodes couple to which is the entire deliverable. So the spatial
    hash does not remove that risk -- it makes it a known function of distance instead of an
    arbitrary draw, which is what makes it measurable.

    WHAT IT DOES NOT BUY, stated because the opposite was assumed for a while. It does NOT make the
    levels specialise by frequency: measured independently (fine/smooth energy ratio 0.58-1.14),
    levels do not sort themselves into coarse and fine content. The multiresolution is the ladder
    the spec sets -- `n_min`, `n_max`, `n_levels` -- plus sparsity, which spends capacity only where
    samples land. On this toy that sparsity is the real win, because the fine rule occupies 15% of
    the domain and the fine levels are simply unused over the other 85%.
    """

    # Müller et al. 2022, eq. 4. Coprime constants; pi_0 = 1 so the first axis is untouched.
    PRIMES = (1, 2654435761, 805459861, 3674653429)

    def __init__(self, dim: int, n_dim: int = 2, n_levels: int = 8, n_min: int = 4,
                 n_max: int = 256, table_size: int = 2 ** 14, seed: int = 0):
        super().__init__()
        self.n_levels = n_levels
        self.n_dim = n_dim          # SPATIAL dimension: decides when a level is dense, not 3
        per = max(1, dim // n_levels)
        self.slice_dims = [per] * (n_levels - 1) + [dim - per * (n_levels - 1)]
        if self.slice_dims[-1] <= 0:
            raise ValueError(f"embedding_dim {dim} cannot be split over {n_levels} levels")
        # GEOMETRIC LADDER, the paper's eq. 3: b = exp((ln n_max - ln n_min) / (L - 1)).
        b = 1.0 if n_levels == 1 else math.exp((math.log(n_max) - math.log(n_min)) / (n_levels - 1))
        self.res = [int(round(n_min * b ** l)) for l in range(n_levels)]
        self.table_size = table_size
        g = torch.Generator().manual_seed(seed)
        self.tables = nn.ParameterList()
        for l in range(n_levels):
            # A LEVEL COARSER THAN THE TABLE IS DENSE, not hashed -- (res+1)^D <= T means a 1:1
            # map exists and collisions would be gratuitous. This is the paper's own rule and it is
            # what makes the coarse levels unambiguous, which is what lets the fine ones collide.
            rows = min(table_size, (self.res[l] + 1) ** n_dim)
            self.tables.append(nn.Parameter(torch.empty(rows, self.slice_dims[l]).uniform_(
                -1e-4, 1e-4, generator=g)))

    def _hash(self, c: torch.Tensor, rows: int) -> torch.Tensor:
        """XOR-product spatial hash of integer corner coordinates `c` [..., D]."""
        h = torch.zeros(c.shape[:-1], dtype=torch.long, device=c.device)
        for d in range(c.shape[-1]):
            h = h ^ (c[..., d].long() * self.PRIMES[d])
        return h % rows

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """`pos` is [N, D] in [0, 1]. Returns [N, sum(slice_dims)]."""
        N, D = pos.shape
        # the 2^D corner offsets of a cell, as a [2^D, D] table of 0/1
        off = torch.stack(torch.meshgrid(*[torch.tensor([0, 1], device=pos.device)] * D,
                                         indexing="ij"), -1).reshape(-1, D)
        out = []
        for l, table in enumerate(self.tables):
            x = pos * self.res[l]
            c0 = torch.floor(x).long()                       # [N, D] lower corner
            f = x - c0                                       # [N, D] fractional position in cell
            corners = c0[:, None, :] + off[None, :, :]       # [N, 2^D, D]
            idx = self._hash(corners, table.shape[0])        # [N, 2^D]
            feat = table[idx]                                # [N, 2^D, F]
            # multilinear weight of each corner: prod_d (f_d if off_d else 1 - f_d)
            w = torch.where(off[None].bool(), f[:, None, :], 1.0 - f[:, None, :]).prod(-1)
            out.append((feat * w[..., None]).sum(1))
        return torch.cat(out, -1)
