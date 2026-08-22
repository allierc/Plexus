"""observation -- turning a Plexus state into a REPRESENTATION of it.

NOTHING IN THIS MODULE IS A BIOLOGICAL MECHANISM, and that is the point of it having its own
file. An operator here does not change what the system does; it changes how the system is
LOOKED AT. The neurons remain the model; a voxel grid of their activity is a picture of the
model, in the same sense that a microscope image is a picture of a tissue and not the tissue.

    {x_i(t), r_i}_{i=1..N}          discrete entities carrying state
              |
              |   voxelize          <- this module
              v
       A(r, t)  on a regular grid   a continuous field, at a chosen resolution

WHY THE FIELD AND NOT A STATE BLOCK. The rendered cube is not a second copy of the dynamical
state: it is a projection of it, and the projection has its own parameters -- a kernel, a
width, a normalisation, a resolution -- none of which are properties of any neuron. Putting
it in `neuron.state` would make changing the render an edit to the model. As a `Field`
attached to the containing set, the resolution can change, the kernel can change, or a
different quantity can be rendered, and the mechanism is untouched.

WHY A FAMILY OF `harness`. `OPERATOR_FAMILIES` says what an operator is FOR, and the honest
answer here is "bookkeeping and scaffolding that is not biology". Inventing an `observation`
family would grow a closed vocabulary for one module; `harness` already means this.
"""
from __future__ import annotations

import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator


@register_operator("voxelize", family="harness", set="neuron", kind="exchange")
class Voxelize(Exchange):
    """set -> field: splat a per-node scalar onto a regular grid, once per tick.

        A(v) = SUM_i K(|c_v - r_i|) x_i          kernel = "gaussian", normalize = "none"
        A(v) = SUM_i K x_i / (eps + SUM_i K)     normalize = "density"

    THE TWO NORMALISATIONS ANSWER DIFFERENT QUESTIONS and the choice belongs in the record,
    not in a default nobody reads. `none` is an activity DENSITY: it is bright where neurons
    are dense as well as where they are active, it is defined everywhere, and it is zero far
    from any neuron. `density` divides that by the local neuron density, giving a local MEAN
    membrane potential -- independent of how many cells happen to be there, but ill-posed in
    empty space, where `eps` is what decides the answer. Neither is more correct; a downstream
    model trained on one is not trained on the other.

    THE GRID IS REBUILT EVERY TICK, not accumulated. A splat that added into last tick's grid
    would integrate the activity in time, and the result would look like a plausible field
    while being a running sum -- the difference is invisible in a single frame.

    RESOLUTION IS THE FIELD'S, NOT THIS OPERATOR'S. The field declares `res:`; this operator
    reads its shape. So changing 64 -> 128 is a one-line change to the field, and the record
    of what was rendered stays in one place.
    """

    EMIT = None                        # writes a Field in place; returns {} -- no integrable delta
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
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "neuron")
        self.field = params.get("to") or params.get("field")
        self.block = params.get("block", "voltage")
        self.channel = int(params.get("channel", 0))
        self.kernel = str(params.get("kernel", "gaussian"))
        self.sigma = float(params.get("sigma", 1.5))          # in VOXELS, so it follows `res`
        # the stencil is truncated at `radius` voxels. 3 sigma captures 99.7% of a Gaussian; a
        # tighter cut is a different kernel, and is recorded as such rather than called an
        # optimisation.
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
