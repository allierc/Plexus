"""Observation: turning a Plexus state into a representation of it.

Nothing in this module is a biological mechanism, which is why it has its own file. An
operator here does not change what the system does; it changes how the system is looked at.
The neurons remain the model, and a voxel grid of their activity is a picture of the model --
in the same sense that a microscope image is a picture of a tissue and not the tissue.

    {x_i(t), r_i}_{i=1..N}          discrete entities carrying state
              |
              |   voxelize
              v
       A(r, t)  on a regular grid   a continuous field, at a chosen resolution

In the order they appear below:

    voxelize   exchange   set -> field: splat a per-element scalar onto a regular grid

The result is a Field rather than a state block on the set, because it is a projection and not
a second copy of the dynamical state. The projection has its own parameters -- a kernel, a
width, a normalisation, a resolution -- and none of them is a property of any neuron. In
`neuron.state`, changing the render would be an edit to the model; as a Field attached to the
containing set, the resolution or the kernel or the rendered quantity can change and the
mechanism is untouched.

The family is `harness` for the same reason: `OPERATOR_FAMILIES` says what an operator is for,
and the honest answer here is bookkeeping that is not biology. An `observation` family would
grow a closed vocabulary for one module, where `harness` already means this.
"""
from __future__ import annotations

import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator


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
