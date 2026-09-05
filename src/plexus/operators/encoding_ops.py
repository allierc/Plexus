"""Fields that are a REPRESENTATION rather than a simulated continuum: one learned from its own
coordinates, one splatted from a set.

Neither is a biological mechanism, and that is what they have in common. Every other field in
the library is solved -- a morphogen diffuses, a grid carries momentum -- and its values are an
output of the model. These two produce a field that stands FOR something else: the parameters an
encoder has fitted, or a picture of a set at a chosen resolution. Both write into a Field other
operators read, so the thing being represented becomes nameable in a schedule.

In the order they appear below:

    hash_encoding   field      f(x, y[, t]) from a multiresolution hash table plus an MLP head
    voxelize        exchange   set -> field: splat a per-element scalar onto a regular grid

    {x_i(t), r_i}_{i=1..N}          discrete entities carrying state
              |
              |   voxelize
              v
       A(r, t)  on a regular grid   a continuous field, at a chosen resolution

`voxelize` changes how the system is LOOKED AT and not what it does: the neurons remain the
model, and a voxel grid of their activity is a picture of the model, in the same sense that a
microscope image is a picture of a tissue and not the tissue. The result is a Field rather than a
state block on the set, because a projection has its own parameters -- a kernel, a width, a
normalisation, a resolution -- and none of them is a property of any neuron. In `neuron.state`,
changing the render would be an edit to the model; as a Field, the resolution or the kernel or
the rendered quantity can change and the mechanism is untouched. Its family is `harness` for the
same reason: `OPERATOR_FAMILIES` says what an operator is FOR, and the honest answer there is
bookkeeping that is not biology.

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

from plexus.models.base import Exchange, FieldUpdate
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


# ----------------------------------------------------------------------------------------------
# `voxelize` -- the other direction: a field built FROM a set, rather than from coordinates.
# ----------------------------------------------------------------------------------------------
@register_operator("voxelize", family="harness", set="neuron", kind="exchange")
class Voxelize(Exchange):
    """Splat a per-element scalar onto a regular grid: the discrete set becomes a continuous
    field that can be viewed, saved, or handed to a model that expects a volume.

    neuron -> field: reads pos and the `block:` state, writes the `to:` field in place.

        A(v) = sum_i K(|c_v - r_i|) x_i                     normalize: none
        A(v) = sum_i K(|c_v - r_i|) x_i / (eps + sum_i K)   normalize: density

    c_v is the centre of voxel v and r_i the position of element i, both in world units; x_i is
    the scalar being rendered, taken from the state block named by `block`. K is the splat
    kernel: `gaussian`, exp(-|d|^2 / 2 sigma^2), or `nearest`, which deposits into the
    containing voxel only. sigma is `radius`-independent and measured in VOXELS, not world
    units, so it follows the field's resolution rather than the domain -- doubling `res` at
    fixed sigma renders a proportionally finer kernel. The stencil is truncated at `radius`
    voxels, defaulting to 3 sigma, which captures 99.7% of a Gaussian; a tighter cut is a
    different kernel and is recorded as one rather than called an optimisation.

    The two normalisations answer different questions, so the choice belongs in the record and
    not in a default nobody reads. `none` is an activity DENSITY: bright where neurons are
    dense as well as where they are active, defined everywhere, and zero far from any neuron.
    `density` divides by the local neuron density, giving a local MEAN of the rendered quantity
    -- independent of how many cells happen to be there, but ill-posed in empty space, where
    `eps` is what decides the answer. Neither is more correct, and a downstream model trained
    on one is not trained on the other.

    The grid is rebuilt every tick, never accumulated. A splat adding into the previous tick's
    grid would integrate the activity in time, and the result would look like a plausible field
    while being a running sum -- a difference invisible in any single frame.

    Resolution belongs to the field, not to this operator: the field declares `res:` and this
    reads its shape, so 64 -> 128 is a one-line change in one place.

    Reference: none -- this is a rendering, not a mechanism. Plexus (this work).
    """

    EMIT = None                        # writes a Field in place, returns no delta
    INPUTS = ["neuron"]
    OUTPUTS = []                       # a field, not a set
    READS = ["pos", "voltage"]
    WRITES = []                        # the field's grid; not a state block on any set
    MAPS = []                          # positional coupling, not a named map
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True              # index_add of a Gaussian weight: gradients flow to x_i
    REQUIRES_PARAMS = []               # the field is the spec's `to:`; every knob has a default
    MECHANISM_TAGS = ["rasterization", "observation", "projection", "volume_render"]
    PARAM_ROLES = {"block": "source_state_block", "sigma": "kernel_width_voxels",
                   "radius": "stencil_half_width_voxels", "normalize": "density_normalisation",
                   "kernel": "splat_kernel", "eps": "empty_space_regulariser"}
    REFERENCE = "Plexus (this work); a rendering, not a mechanism."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "neuron")
        self.field = params.get("to") or params.get("field")
        self.block = params.get("block", "voltage")
        self.channel = int(params.get("channel", 0))
        self.kernel = str(params.get("kernel", "gaussian"))
        self.sigma = float(params.get("sigma", 1.5))          # in VOXELS, so it follows `res`
        self.radius = int(params.get("radius", max(1, int(round(3 * self.sigma)))))
        self.normalize = str(params.get("normalize", "none"))
        self.eps = float(params.get("eps", 1e-6))
        self._stencil = None

    def _offsets(self, D, device):
        if self._stencil is None or self._stencil.shape[1] != D:
            r = self.radius
            ax = torch.arange(-r, r + 1, device=device)
            self._stencil = torch.stack(torch.meshgrid(*([ax] * D), indexing="ij"),
                                        -1).reshape(-1, D)
        return self._stencil

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        fld = H.field(self.field)
        pos = lvl.get("pos")                                   # [N, D] in the world box
        val = lvl.get(self.block)[:, :1] * lvl.occ[:, None]    # [N, 1] dormant contribute nothing
        if mask is not None:
            val = val * mask[:, None].float()
        dev = pos.device
        D = pos.shape[-1]
        shape = torch.tensor(fld.shape[:D], device=dev)        # (nx, ny[, nz])
        # world -> voxel. Axis 0 spans the world width; the others span 1 (the ScalarField
        # convention), so the scale is per-axis and a non-cubic box does not silently stretch.
        box = torch.tensor([fld.width] + [1.0] * (D - 1), device=dev, dtype=pos.dtype)
        g = pos / box * shape.to(pos.dtype)                    # [N, D] continuous voxel coords
        base = torch.floor(g).long()
        offs = self._offsets(D, dev)                           # [K, D]
        idx = base[:, None, :] + offs[None, :, :]              # [N, K, D]
        d = g[:, None, :] - (idx.to(g.dtype) + 0.5)            # to the voxel CENTRE
        r2 = (d * d).sum(-1)                                   # [N, K]
        if self.kernel == "gaussian":
            w = torch.exp(-r2 / (2.0 * self.sigma ** 2))
        elif self.kernel == "nearest":
            w = (r2 <= 0.75).to(g.dtype)                       # the containing voxel only
        else:
            raise ValueError(f"voxelize: kernel must be 'gaussian' or 'nearest', got {self.kernel!r}")
        inside = ((idx >= 0) & (idx < shape)).all(-1)          # [N, K]
        w = w * inside.to(w.dtype)
        flat = torch.zeros_like(idx[..., 0])
        stride = 1
        for k in range(D - 1, -1, -1):                         # row-major over (nx, ny, nz)
            flat = flat + idx[..., k].clamp(0, int(shape[k]) - 1) * stride
            stride *= int(shape[k])
        num = torch.zeros(stride, device=dev, dtype=w.dtype)
        num.index_add_(0, flat.reshape(-1), (w * val).reshape(-1))
        if self.normalize == "density":
            den = torch.zeros(stride, device=dev, dtype=w.dtype)
            den.index_add_(0, flat.reshape(-1), w.reshape(-1))
            num = num / (den + self.eps)
        elif self.normalize != "none":
            raise ValueError(f"voxelize: normalize must be 'none' or 'density', "
                             f"got {self.normalize!r}")
        grid = fld.grid.clone()                                # rebuilt, never accumulated
        grid[self.channel] = num.reshape(tuple(int(s) for s in shape))
        fld.grid = grid
        return {}

    def render_metadata(self, H) -> dict:
        """What this render IS, for the record that travels with the volume.

        A rendered volume without its kernel and its normalisation is a stack of numbers that
        cannot be reproduced or compared with another stack. The downstream adapter
        (`plexus.io.walrus`) writes this verbatim into the dataset."""
        fld = H.field(self.field)
        lvl = H.level(self.at)
        man = getattr(lvl, "region_manifest", None)
        return {
            "source_set": self.at, "source_field": self.block,
            "resolution": [int(s) for s in fld.shape],
            "world_box": [float(fld.width)] + [1.0] * (len(fld.shape) - 1),
            "kernel": self.kernel, "sigma_voxels": self.sigma,
            "stencil_radius_voxels": self.radius,
            "normalize": self.normalize, "eps": self.eps,
            "region": (man or {}).get("region"),
            "voxel_size_um": (None if not man else
                              man["region"]["side_um"] / float(fld.shape[0])),
        }
