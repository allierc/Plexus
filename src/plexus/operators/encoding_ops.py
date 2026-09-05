"""Encoding: a learnable field written from its own coordinates, and nothing else.

In the order they appear below:

    hash_encoding   field   f(x, y[, t]) from a multiresolution hash table plus an MLP head

An encoder is an operator here rather than a layer inside a model class, and that is the whole
reason it lives in the library. As an operator it owns its parameters and writes a field other
operators read, so an encoded quantity becomes a named mechanism in a schedule:

    operators: [{op: hash_encoding, at: omega, ...}, {op: kuramoto_fit, at: v, omega_from: omega}]
    schedule:  [hash_encoding, kuramoto_fit]

The consequence is the one plexus2.tex asks for: a residual is attributable to a mechanism. If
that fit fails, "the encoding cannot represent the heterogeneity" and "the rule is wrong" are
two different operators with two different parameter sets, and the schedule says which is
which. Wired as a layer inside one model they are one blob of weights and the question cannot
be asked.

THREE KNOBS, AND THE REST DERIVED. The encoder has six parameters in the reference and they
interact, so settling them by hand is how one ends up with a ladder whose finest level is
either below the pixel grid or far above it. A specification sets the three the data has an
opinion about:

    n_levels               L, how many levels in the ladder
    log2_hashmap_size      log2 T, the per-level table capacity, as a power of two
    px_per_finest_cell     how many PIXELS one cell of the finest level spans
    frames_per_finest_cell how many FRAMES one cell of the finest level spans   (3D only)

and the ladder follows:

    n_min  = (8, 8, 2)                                 fixed: a coarse level is a coarse level
    n_max  = (W/px, H/px, T/frames)                    the finest level, in the data's own units
    b      = exp((ln n_max - ln n_min) / (L - 1))      the per-axis growth factor between levels

The time axis is in frames for the same reason space is in pixels, and it is not cosmetic:
"200 cells along t" means nothing without knowing how many frames the run is, where "2 frames
per cell" is a statement about what the data can support.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from plexus.models.base import FieldUpdate
from plexus.models.hashgrid import MultiResHashGrid
from plexus.models.registry import register_operator

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "softplus": nn.Softplus, "tanh": nn.Tanh}


@register_operator("hash_encoding", family="fields", set="field", kind="field",
                   model="multires_hash")
class HashEncoding(FieldUpdate):
    """A learnable field: the value at every cell is computed from that cell's own coordinates,
    through a multiresolution hash table and a small MLP (multi-layer perceptron) head.

    field -> field: reads the coordinates of the field named by `at:`, writes its grid in place.
    It reads no other field -- everything it knows is in its tables.

        c(x) = scale * MLP( concat_{l=1..L} interp( T_l[ hash(floor(x n_l)) ] ) )
        n_l  = n_min b^(l-1),   b = (n_max / n_min)^(1 / (L - 1))

    x is the cell centre in normalised [0, 1] coordinates, with the frame index appended and
    divided by `n_frames` when `use_time` is set. L is `n_levels` and n_l the grid resolution of
    level l in cells per axis, growing geometrically from n_min to n_max; each level hashes its
    cell corners into a table T_l of 2^log2_hashmap_size entries holding
    `n_features_per_level` numbers each, and interpolates between them. The L results are
    concatenated and passed through `n_hidden_layers` layers of `n_neurons`. `scale` multiplies
    the head's output and carries the unit of the target field, so the head itself stays O(1).

    The kind is `field`, not `broadcast`. In Plexus `broadcast` means L_{k+1} -> L_k through the
    hierarchy's containment map; a hash table is not a level of the hierarchy -- it holds no
    entities and nothing is contained in it -- so calling this a broadcast would claim a
    containment map that does not exist. It computes a field's values from that field's own
    coordinates, which is what `field` means.

    `use_time` decides whether the encoded quantity is static or spatiotemporal, and getting it
    wrong is not a tuning error. A property of POSITION -- a per-cell rate, a gain, a cell type
    -- is encoded in 2D or 3D and is the same at every tick. A property of position AND time
    takes the tick as a further input. A static parameter given a time axis can memorise the
    trajectory, and then it is not a parameter.

    Interpolation defaults to smoothstep in space and linear in time. Smoothstep makes the
    encoding C^1, which anything taking a second derivative through it needs; but its weight
    derivative vanishes at every cell boundary, so on an axis whose cells line up with the
    sampled frames it would force df/dt to zero AT every sample and inflate it in between.

    Reference: Muller, T., Evans, A., Schied, C. & Keller, A. (2022). Instant neural graphics
    primitives with a multiresolution hash encoding. ACM Trans. Graph. 41(4):102.
    """

    INPUTS: list = []
    OUTPUTS: list = []
    READS: list = []
    WRITES: list = []
    MAPS: list = []
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["n_levels", "log2_hashmap_size", "px_per_finest_cell"]
    MECHANISM_TAGS = ["encoding", "multiresolution", "heterogeneity", "implicit_field"]
    PARAM_ROLES = {
        "n_levels": "number_of_resolution_levels",
        "log2_hashmap_size": "log2_entries_per_level",
        "px_per_finest_cell": "pixels_spanned_by_a_finest_cell",
        "frames_per_finest_cell": "frames_spanned_by_a_finest_cell",
        "use_time": "whether the encoded quantity varies in time",
        "scale": "multiplier on the head's output, in the unit of the target field",
    }
    REFERENCE = ("Muller, T., Evans, A., Schied, C. & Keller, A. (2022). Instant neural "
                 "graphics primitives with a multiresolution hash encoding. ACM Trans. Graph. "
                 "41(4):102.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")
        self.n_levels = int(params["n_levels"])
        self.log2_T = int(params["log2_hashmap_size"])
        self.ppc = max(1.0, float(params["px_per_finest_cell"]))
        self.fpc = max(1.0, float(params.get("frames_per_finest_cell", 1.0)))
        self.use_time = bool(params.get("use_time", False))
        self.n_frames = int(params.get("n_frames", 1))
        self.features = int(params.get("n_features_per_level", 2))
        self.n_neurons = int(params.get("n_neurons", 64))
        self.n_hidden = int(params.get("n_hidden_layers", 2))
        self.activation = str(params.get("activation", "gelu"))
        self.scale = float(params.get("scale", 1.0))
        self.seed = int(params.get("seed", 0))
        self.device_ = device
        self.grid = self.head = None
        self._coords = None
        self._shape = None

    # ---------------------------------------------------------------- build
    def bind(self, shape, mask=None, out_dim=None):
        """Allocate once the field's shape is known, and cache the cell-centre coordinates.

        The ladder cannot be built in `__init__` because `n_max` is derived from the field's
        resolution, and the field does not exist until the hierarchy is built. The coordinates are
        cached because they are a property of the grid, not of the state -- recomputing a
        [1024*1024, 3] table every tick would be the dominant cost of an operator whose real work
        is a gather.
        """
        # The output width is the target field's channel count. It is passed in rather than read
        # off the field, because a trainer may call `bind` before `forward` ever runs.
        if out_dim is not None:
            self._out_dim = int(out_dim)
        self._out_dim = int(getattr(self, "_out_dim", 1))
        D = len(shape)
        n_in = D + (1 if self.use_time else 0)
        n_min = [8.0] * D + ([2.0] if self.use_time else [])
        n_max = [max(9.0, round(s / self.ppc)) for s in shape]
        if self.use_time:
            n_max.append(max(3.0, round(self.n_frames / self.fpc)))
        b = [math.exp((math.log(mx) - math.log(mn)) / max(1, self.n_levels - 1))
             for mn, mx in zip(n_min, n_max)]
        interp = ["smoothstep"] * D + (["linear"] if self.use_time else [])

        torch.manual_seed(self.seed)
        self.grid = MultiResHashGrid(
            n_input_dims=n_in, n_levels=self.n_levels,
            n_features_per_level=self.features, log2_hashmap_size=self.log2_T,
            base_resolution=n_min, per_level_scale=b, max_resolution=n_max,
            interpolation=interp).to(self.device_)
        act = _ACTIVATIONS[self.activation]
        layers = [nn.Linear(self.grid.n_output_dims, self.n_neurons), act()]
        for _ in range(self.n_hidden - 1):
            layers += [nn.Linear(self.n_neurons, self.n_neurons), act()]
        layers += [nn.Linear(self.n_neurons, int(self._out_dim))]
        self.head = nn.Sequential(*layers).to(self.device_)

        axes = torch.meshgrid(*[(torch.arange(n, device=self.device_, dtype=torch.float32) + 0.5)
                                / n for n in shape], indexing="ij")
        self._coords = torch.stack([a.reshape(-1) for a in axes], -1)
        self._shape = tuple(shape)
        return self

    @torch.no_grad()
    def sample(self, shape=None, t: float = 0.0) -> torch.Tensor:
        """Evaluate the encoding on an ARBITRARY grid, [C, *shape]. For figures, not for the fit.

        The resolution is an argument rather than the bound field's because a montage needs each
        level drawn at its own lattice: a level with 12 cells across cannot represent anything
        finer than 12 cells, and rendering it at 1024 only interpolates that fact into a blur.
        """
        shape = tuple(shape or self._shape)
        axes = torch.meshgrid(*[(torch.arange(n, device=self.device_, dtype=torch.float32) + 0.5)
                                / n for n in shape], indexing="ij")
        x = torch.stack([a.reshape(-1) for a in axes], -1)
        if self.use_time:
            x = torch.cat([x, torch.full((x.shape[0], 1), float(t), device=x.device)], -1)
        return (self.head(self.grid(x)) * self.scale).T.reshape(self._out_dim, *shape)

    def set_level_window(self, alpha: float) -> None:
        """Coarse-to-fine: enable levels up to `alpha`. Forwarded to the grid; see hashgrid."""
        if self.grid is not None:
            self.grid.set_level_window(alpha)

    # ---------------------------------------------------------------- run
    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        if self.grid is None or self._out_dim != fld.grid.shape[0]:
            self.bind(tuple(fld.grid.shape[1:]), out_dim=fld.grid.shape[0])
        x = self._coords
        if self.use_time:
            t = float(getattr(H, "frame", 0)) / max(1, self.n_frames - 1)
            x = torch.cat([x, torch.full((x.shape[0], 1), t, device=x.device)], -1)
        out = self.head(self.grid(x)) * self.scale             # [n_cells, C]
        fld.grid = out.T.reshape(fld.grid.shape).contiguous()
        return {}

    def describe(self) -> str:
        return "" if self.grid is None else self.grid.extra_repr()
