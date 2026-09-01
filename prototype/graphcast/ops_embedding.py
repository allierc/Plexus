"""The Instant-NGP ladder-hashtable as a LEARNABLE PLEXUS2 OPERATOR.

WHY IT IS AN OPERATOR AND NOT A LAYER, which is the whole point of putting it here. In `model.py`
the same encoding is an `nn.Module` owned by a network: its parameters belong to the model, and
what it produces is an activation. As an operator it owns its parameters itself and WRITES A FIELD
that other operators READ, so the heterogeneity becomes a named mechanism in a schedule:

    schedule:  ngp_embedding  ->  kuramoto_known_ode

and `kuramoto_known_ode` takes its `omega` from the field the encoding wrote. The consequence is
the one plexus2.tex asks for: a residual is attributable to a mechanism. If the fit fails, "the
encoding cannot represent the heterogeneity" and "the Kuramoto rule is wrong" are different
operators with different parameters, and the schedule says which is which. Wired as a layer they
are one blob of weights and the question cannot be asked.

WHAT THIS ENCODING IS. For a position x, at each of L levels with resolutions on a geometric ladder
n_min -> n_max, take the 2^D corners of the cell containing x, hash each corner's integer
coordinates into that level's table, and multilinearly interpolate the F-vectors found there. The
L slices are concatenated. Muller et al. 2022, eqs. 3-4.

WHAT IT BUYS, MEASURED HERE:

    spatial smoothness for free.  Nearby positions share corners, so their encodings are correlated
    BY CONSTRUCTION. Measured cosine similarity against separation: 0.744 within 0.01 of the
    domain, 0.399 at 0.01-0.05, 0.148 at 0.05-0.2, 0.010 beyond 0.5. The index-hash version in
    `model.MultiResNodeEmbedding` is flat noise at every separation -- 0.005, 0.002, 0.003 -- so it
    has the collisions and the ladder without anything spatial.

    capacity spent only where samples land.  The fine levels are simply unused over the 85% of this
    toy's domain that the discs do not cover. On a masked problem that is the real economy.

WHAT IT DOES NOT BUY, and this was assumed for a while before it was measured. IT DOES NOT MAKE THE
LEVELS SPECIALISE BY FREQUENCY. An independent measurement puts the fine/smooth energy ratio across
levels at 0.58-1.14 -- levels do not sort themselves into coarse content and fine content. Nor do
collisions "prioritise": at equal loss weight the gradient mass at a shared row splits 0.507, and
only a 100x weight difference moves it to 0.970. Collisions are SURVIVABLE, not selective. So this
operator will not discover the toy's coarse/fine split on its own; the multiresolution is the
ladder the spec sets, plus sparsity.

THE RISK THAT IS SPECIFIC TO A GRAPH MODEL, kept here because it is easy to inherit silently. With
a pointwise decoder a shared table row is shared capacity and the cross-level concatenation still
separates the two points. With a MESSAGE-PASSING decoder, two distant nodes sharing a row is a
manufactured long-range coupling -- and recovering which nodes couple to which is the deliverable.
The spatial hash does not remove that; it makes collision a known function of distance rather than
an arbitrary draw, which is what makes it measurable.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from plexus.models.base import Operator
from plexus.models.registry import register_operator

# Muller et al. 2022, eq. 4. Coprime; pi_0 = 1 leaves the first axis untouched.
PRIMES = (1, 2654435761, 805459861, 3674653429)


class Ladder(nn.Module):
    """The tables and the lookup. Shared by the operator and by `model.py`'s embedding option."""

    def __init__(self, out_dim: int, n_dim: int, n_levels: int, n_min: int, n_max: int,
                 table_size: int, seed: int = 0, device: str = "cpu"):
        super().__init__()
        self.n_levels, self.n_dim = n_levels, n_dim
        per = max(1, out_dim // n_levels)
        self.slice_dims = [per] * (n_levels - 1) + [out_dim - per * (n_levels - 1)]
        if self.slice_dims[-1] <= 0:
            raise ValueError(
                f"out_dim {out_dim} cannot be split over {n_levels} levels: each level needs at "
                f"least one feature, so out_dim >= n_levels. Muller et al. use F=2 per level over "
                f"L=16, i.e. out_dim 32; here out_dim {out_dim} allows at most {out_dim} levels.")
        b = 1.0 if n_levels == 1 else math.exp((math.log(n_max) - math.log(n_min)) / (n_levels - 1))
        self.res = [int(round(n_min * b ** l)) for l in range(n_levels)]
        g = torch.Generator().manual_seed(seed)
        self.tables = nn.ParameterList()
        for l in range(n_levels):
            # A LEVEL COARSER THAN THE TABLE IS DENSE, NOT HASHED. If (res+1)^D <= T a 1:1 map
            # exists, so colliding would be gratuitous. This is the paper's own rule, and it is
            # what makes the coarse levels unambiguous -- which is what licenses the fine ones to
            # collide at all.
            rows = min(table_size, (self.res[l] + 1) ** n_dim)
            t = torch.empty(rows, self.slice_dims[l], device=device)
            self.tables.append(nn.Parameter(t.uniform_(-1e-4, 1e-4, generator=g)))

    def _hash(self, c: torch.Tensor, rows: int) -> torch.Tensor:
        h = torch.zeros(c.shape[:-1], dtype=torch.long, device=c.device)
        for d in range(c.shape[-1]):
            h = h ^ (c[..., d].long() * PRIMES[d])
        return h % rows

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """`pos` [N, D] in [0, 1] -> [N, sum(slice_dims)]."""
        D = pos.shape[-1]
        off = torch.stack(torch.meshgrid(*[torch.tensor([0, 1], device=pos.device)] * D,
                                         indexing="ij"), -1).reshape(-1, D)
        out = []
        for l, table in enumerate(self.tables):
            x = pos * self.res[l]
            c0 = torch.floor(x).long()
            f = x - c0
            idx = self._hash(c0[:, None, :] + off[None], table.shape[0])
            w = torch.where(off[None].bool(), f[:, None, :], 1.0 - f[:, None, :]).prod(-1)
            out.append((table[idx] * w[..., None]).sum(1))
        return torch.cat(out, -1)


@register_operator("ngp_embedding", family="fields", set="field", kind="field", model="hash_ladder")
class NGPEmbedding(Operator):
    """Writes the ladder encoding of every cell's POSITION into the field it is `at:`.

    KIND IS `field`, NOT `broadcast`, and the distinction is not pedantry. `broadcast` in Plexus2
    means L_{k+1} -> L_k through the containment map of the hierarchy. A hash table is not a level
    of the hierarchy -- it has no entities and nothing is contained in it -- so calling this a
    broadcast would claim a containment map that does not exist. What it actually does is compute a
    field's values from the field's own coordinates, which is what `field` means.

    THE OUTPUT IS THE FIELD'S CHANNELS. A field with C components receives an encoding of width C,
    so `out_dim` is not free: it is whatever the target field declares. That keeps the arity in the
    spec, where a reader can see it, rather than in this class.
    """

    EMIT = None
    INPUTS: list = []
    OUTPUTS: list = []
    READS: list = []
    WRITES: list = []
    MAPS: list = []
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS: list = []
    MECHANISM_TAGS = ["encoding", "multiresolution", "heterogeneity"]
    PARAM_ROLES = {"tables": "hashed_multiresolution_feature_tables",
                   "n_levels": "ladder_length", "n_min": "coarsest_resolution",
                   "n_max": "finest_resolution", "table_size": "rows_per_level"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")
        self.n_levels = int(params.get("n_levels", 8))
        self.n_min = int(params.get("n_min", 4))
        self.n_max = int(params.get("n_max", 512))
        self.table_size = int(params.get("table_size", 2 ** 14))
        self.scale = float(params.get("scale", 1.0))
        self.seed = int(params.get("seed", 0))
        self.device_ = device
        self.ladder = None
        self._pos = None

    def bind(self, shape, mask=None):
        """Allocate the ladder against the field's shape, and cache the cell-centre positions.

        THE POSITIONS ARE COMPUTED ONCE. They are a property of the grid, not of the state, so
        recomputing a [1024*1024, 2] coordinate table every tick would be the dominant cost of an
        operator whose actual work is a gather.
        """
        n_dim = len(shape)
        if self.ladder is None:
            self.ladder = Ladder(self.out_dim, n_dim, self.n_levels, self.n_min, self.n_max,
                                 self.table_size, self.seed, self.device_)
        axes = torch.meshgrid(*[(torch.arange(n, device=self.device_, dtype=torch.float32) + 0.5)
                                / n for n in shape], indexing="ij")
        self._pos = torch.stack([a.reshape(-1) for a in axes], -1)
        self._shape = tuple(shape)
        return self

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        if self.ladder is None:
            self.out_dim = fld.grid.shape[0]
            self.bind(tuple(fld.grid.shape[1:]))
        e = self.ladder(self._pos) * self.scale             # [n_cells, C]
        fld.grid = e.T.reshape(fld.grid.shape).contiguous()
        return {}
