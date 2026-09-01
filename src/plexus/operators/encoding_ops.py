"""ENCODING OPERATORS: a learnable field written from its own coordinates.

`hash_encoding` is the Instant-NGP multiresolution hash encoding (`models/hashgrid.py`) with an MLP
head, as an ordinary Plexus operator. It reads no other field: what it consumes is WHERE each cell
is, and when. Everything it knows is in its tables.

WHY THIS IS AN OPERATOR RATHER THAN A LAYER, which is the whole reason it is in the library instead
of inside somebody's model class. As an operator it owns its parameters and WRITES A FIELD that
other operators read, so an encoded quantity becomes a named mechanism in a schedule:

    operators: [{op: hash_encoding, at: omega, ...}, {op: kuramoto_fit, at: v, omega_from: omega}]
    schedule:  [hash_encoding, kuramoto_fit]

and the consequence is the one plexus2.tex asks for -- a residual is attributable to a MECHANISM. If
that fit fails, "the encoding cannot represent the heterogeneity" and "the rule is wrong" are two
different operators with two different parameter sets, and the schedule says which is which. Wired
as a layer inside one model they are one blob of weights and the question cannot be asked.

===============================================================================================
THREE KNOBS, AND EVERYTHING ELSE FOLLOWS FROM THEM
===============================================================================================

The encoder has six parameters in the paper and they interact; settling them by hand is how one
ends up with a ladder whose finest level is either below the pixel grid or far above it. So the
spec sets three numbers that the DATA has an opinion about, and the rest is derived:

    n_levels               L, how many levels
    log2_hashmap_size      log2 T, the per-level table capacity, as a power of two
    px_per_finest_cell     how many PIXELS one cell of the finest level spans
    frames_per_finest_cell how many FRAMES one cell of the finest level spans   (3-D only)

and then, exactly as `ngp-demo/scripts/gui_scalar_time.py` settles it:

    n_min  = (8, 8, 2)                                 fixed: a coarse level is a coarse level
    n_max  = (W/px, H/px, T/frames)                    the finest level, in the data's own units
    b      = exp((ln n_max - ln n_min) / (L - 1))      per axis

THE TIME AXIS IS IN FRAMES, THE SAME WAY SPACE IS IN PIXELS, and that is not cosmetic. "200 cells
along t" means nothing without knowing the run is 201 frames long; "2 frames per cell" is a
statement about what the data can support. Measured on the two-scale toy this was written against,
the fine component's lag-1 autocorrelation is 0.829 -- past half correlation after ONE frame -- so
1 frame per cell keeps its per-frame content and 2 already averages pairs.

INTERPOLATION DEFAULTS TO SMOOTHSTEP IN SPACE AND LINEAR IN TIME. Smoothstep makes the encoding C^1,
which anything taking a second derivative through it needs; but its weight derivative vanishes at
every cell boundary, so on an axis whose cells line up with the sampled frames it would force df/dt
to zero AT every sample and inflate it in between. See `models/hashgrid.py`.
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
    """Writes f(x, y[, t]) into the field it is `at:`, from a hash encoding plus an MLP head.

    KIND IS `field`, NOT `broadcast`. `broadcast` in Plexus means L_{k+1} -> L_k through the
    hierarchy's containment map; a hash table is not a level of the hierarchy -- it holds no
    entities and nothing is contained in it -- so calling this a broadcast would claim a containment
    map that does not exist. It computes a field's values from the field's own coordinates, which is
    what `field` means.

    STATIC OR SPATIOTEMPORAL, by `use_time`. A quantity that is a property of POSITION -- a per-cell
    rate, a gain, a cell type -- is encoded in 2-D or 3-D space and is the same at every tick. A
    quantity that is a property of position AND time -- an observed field being represented -- takes
    the tick as a further input, normalised by `n_frames`. Getting this wrong is not a tuning error:
    a static parameter given a time axis can memorise the trajectory, and then it is not a parameter.
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
    REFERENCE = ("Müller, T. et al. (2022). Instant neural graphics primitives with a "
                 "multiresolution hash encoding. ACM ToG 41(4):102.")

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
        # THE OUTPUT WIDTH IS THE TARGET FIELD'S CHANNEL COUNT, and `bind` may be called by a
        # trainer before `forward` ever runs -- so it is passed in, defaulted, and only overridden
        # by `forward` if it turns out to differ. An earlier version read it off the field inside
        # `forward` and crashed when bind came first.
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
