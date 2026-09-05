"""A continuum bound to a set: what writes into it, what happens inside it, what reads it.

    deposit        set -> field   (a cell lays a trail)
    diffuse        field -> field (finite_difference | spectral)
    decay          field -> field
    sense          field -> set   (a cell reads the value under it)
    chemotax       field -> set   (and moves up the gradient)
    playback       a PRESCRIBED field: a video or a measured stack, not a solved one
    pacemaker / activation_pulse   an excitable field's source terms
    signal         set -> set along an edge-set (the synapse case)

THE FOUR-STEP SHAPE IS THE POINT OF THE GROUPING. deposit / diffuse / decay / sense is one
mechanism written as four operators so that each can be swapped, and reading them apart is how a
spec ends up depositing into a field nothing senses.
"""
from __future__ import annotations
import torch
from plexus.models.base import Field
from plexus.models.registry import register_field
from plexus.models.base import Exchange
from plexus.models.registry import register_operator
import math
import torch.fft as fft
import torch.nn.functional as Fnn
from plexus.models.base import FieldUpdate
import os
from plexus.models.base import Field, FieldUpdate
from plexus.models.registry import register_field, register_operator
from plexus.paths import graphs_data_path
import torch.nn.functional as F
from plexus.models.base import Lateral


@register_field("grid", frame="grid")
class ScalarField(Field):
    """A C-channel scalar field on a square-pixel grid over the box [0,W]x[0,1](x[0,1]).

    Pure state: a `grid` buffer `[C, *shape]` (shape = (nx, ny) in 2D, (nx, ny, nz) in
    3D) plus the geometry to map a world position to a voxel (`pix`). Pixels are
    square, dx = 1/R; axis 0 spans the world width W, the other axes span 1. Operators
    read/write `grid` directly.
    """

    def __init__(self, name, couples_to=None, components=1, res=200, width=1.0, dim=2, device="cpu"):
        super().__init__(name, couples_to)
        self.C = int(components)
        self.R = int(res)
        self.width = float(width)
        self.dim = int(dim)
        self.nx = int(round(self.width * self.R))      # square pixels, dx = 1/R
        self.ny = self.R
        if self.dim == 2:
            self.shape = (self.nx, self.ny)
        else:                                          # 3D: axes 1,2 span [0,1]
            self.nz = self.R
            self.shape = (self.nx, self.ny, self.nz)
        self.periodic = False                          # set by the engine from the spec boundary
        self.register_buffer("grid", torch.zeros((self.C,) + self.shape, device=device))

    def pix(self, *coords):
        """Nearest voxel indices of world positions (per-axis int-cast of the shader).

        Accepts D coordinate tensors (x, y[, z]) and returns a D-tuple of index
        tensors. Axis 0 spans [0, W], every other axis spans [0, 1]; each maps by the
        common pixels-per-unit R. The 2D call `pix(x, y)` returns `(gx, gy)` exactly as
        before (back-compatible). When `self.periodic`, indices WRAP modulo the grid
        (a torus) instead of clamping to the edge -- so a sensor reaching past one
        side reads the other (matching the periodic particle wrap in `_integrate`)."""
        out = []
        for k, c in enumerate(coords):
            box = self.width if k == 0 else 1.0
            if getattr(self, "periodic", False):
                # floor (not trunc-toward-0) so a coord just below 0 wraps to the far edge
                out.append(torch.remainder(torch.floor(c * self.R).long(), self.shape[k]))   # torus wrap
            else:
                out.append((c.clamp(0, box - 1e-6) * self.R).long().clamp(0, self.shape[k] - 1))
        return tuple(out)


@register_operator("deposit", family="fields", set="cell", kind="exchange")
class Deposit(Exchange):
    """object -> field. Writes `to:` field in place; returns {}."""

    EMIT = None                                # set->field: scatters onto the grid in place (stigmergy write); returns {} — no integrable delta
    # typed signature (Plexus2 sec. 2.1): set -> field (Exchange). The true output is
    # the `to:` field grid, not set state, so WRITES (set-state blocks) is empty and the
    # field coupling is the "field" map.
    INPUTS = ["cell"]
    OUTPUTS = []                               # writes the `to:` field, no set-state output
    READS = ["pos"]
    WRITES = []                                # no set-state block written (the grid is mutated in place)
    MAPS = ["field"]                           # Exchange: a scatter map onto the `to:` field
    SUPPORTED_DIMS = [2, 3]                     # N-D scatter onto the grid field
    REQUIRES_PARAMS = ["to"]
    MECHANISM_TAGS = ["deposition", "stigmergy", "field_write"]
    PARAM_ROLES = {"amount": "deposit_rate"}
    REFERENCE = "Grasse, P.-P. (1959). La reconstruction du nid et les coordinations interindividuelles (stigmergy). Insectes Sociaux 6:41-80."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("to")
        self.amount = float(params.get("amount", 0.9))
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        N = lvl.n
        fld = H.fields[self.field_name]
        pos = lvl.get("pos")                                      # [N, D] (D = 2 or 3)
        D = pos.shape[1]
        nt = lvl.node_type
        dt = float(getattr(H.config, "dt", 1.0))
        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ

        gidx = fld.pix(*[pos[:, k] for k in range(D)])           # D-tuple of voxel indices
        # channel-major, row-major flat index over the N-D grid (== the 2D
        # `nt*(nx*ny) + gx*ny + gy` exactly when D == 2).
        ravel = torch.zeros(N, dtype=torch.long, device=dev)
        stride = 1
        for k in reversed(range(D)):
            ravel = ravel + gidx[k] * stride
            stride *= fld.shape[k]
        flat = nt * stride + ravel                               # stride == prod(shape)
        amt = torch.full((N,), self.amount * dt, device=dev) * m
        fld.grid.view(-1).index_add_(0, flat, amt)
        fld.grid.clamp_(max=1.0)
        return {}


@register_operator("diffuse", family="fields", set="field", kind="field",
                   implementation="finite_difference")
class Diffuse(FieldUpdate):
    """field -> field: acts on the field named by `at:` (no set involved).

    The `finite_difference` implementation of the `diffuse` contract: a 3x3 box-blur
    lerp (an explicit Laplacian step). `spectral` below is a second implementation of
    the SAME contract -- select it with `{op: diffuse, at: chemical, implementation:
    spectral}`; both advance dc/dt = D nabla^2 c one step, differing only in numerics."""

    EMIT = None                                # field->field: writes the grid in place; returns {} — no integrable delta
    SUPPORTED_DIMS = [2, 3]                     # 3x3 (2D) / 3x3x3 (3D) box-blur step
    REQUIRES_PARAMS = []                        # no required params — target field comes from `at:` (engine-injected)
    MECHANISM_TAGS = ["diffusion", "field_smoothing", "laplacian"]
    PARAM_ROLES = {"rate": "diffusion_rate"}
    REFERENCE = "Fick, A. (1855). Ueber Diffusion. Ann. Phys. 170:59-86; Turing, A. M. (1952). Phil. Trans. R. Soc. B 237:37-72."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")   # the field at `at:`
        self.rate = float(params.get("rate", 0.35))     # diffusion weight per unit time

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        g = fld.grid                                                    # [C, *shape]
        dt = float(getattr(H.config, "dt", 1.0))
        # periodic field -> wrap the blur across the seam (`circular`); else edge-clamp.
        pmode = "circular" if getattr(fld, "periodic", False) else "replicate"
        if g.dim() == 3:                                               # 2D field [C, nx, ny]
            gp = Fnn.pad(g.unsqueeze(0), (1, 1, 1, 1), mode=pmode)
            blur = Fnn.avg_pool2d(gp, 3, stride=1).squeeze(0)             # 3x3 mean, same size
        else:                                                         # 3D field [C, nx, ny, nz]
            gp = Fnn.pad(g.unsqueeze(0), (1, 1, 1, 1, 1, 1), mode=pmode)
            blur = Fnn.avg_pool3d(gp, 3, stride=1).squeeze(0)            # 3x3x3 mean, same size
        dw = min(max(self.rate * dt, 0.0), 1.0)                        # saturate(rate*dt)
        fld.grid = g * (1.0 - dw) + blur * dw
        return {}


@register_operator("diffuse", family="fields", set="field", kind="field",
                   implementation="spectral")
class DiffuseSpectral(FieldUpdate):
    """`spectral` implementation of the `diffuse` contract: one EXACT heat-kernel step of
    dc/dt = D nabla^2 c on a periodic grid -- c_hat *= exp(-D k^2 dt) in Fourier space.
    Same contract as the finite-difference box-blur (Diffuse); differs only in numerics
    (spectral accuracy, periodic boundary). Differentiable via torch.fft, so an inverse
    loop that filters `capabilities()` for `differentiable` keeps it."""

    EMIT = None
    SUPPORTED_DIMS = [2]                        # FFT step is 2D here (N-D is a follow-up)
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["diffusion", "field_smoothing", "spectral"]
    PARAM_ROLES = {"rate": "diffusion_coefficient"}
    REFERENCE = "Fick, A. (1855). Ueber Diffusion. Ann. Phys. 170:59-86; Turing, A. M. (1952). Phil. Trans. R. Soc. B 237:37-72."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")
        self.rate = float(params.get("rate", 0.35))     # diffusion coefficient D

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        g = fld.grid                                                    # [C, nx, ny]
        if g.dim() != 3:
            raise NotImplementedError("diffuse:spectral is 2D-only (grid must be [C, nx, ny])")
        dt = float(getattr(H.config, "dt", 1.0))
        _, nx, ny = g.shape
        kx = fft.fftfreq(nx, device=g.device) * (2 * math.pi)          # radians / cell
        ky = fft.fftfreq(ny, device=g.device) * (2 * math.pi)
        k2 = kx[:, None] ** 2 + ky[None, :] ** 2                        # [nx, ny]
        decay = torch.exp(-self.rate * dt * k2)                         # exact heat kernel
        ghat = fft.fftn(g, dim=(-2, -1))
        fld.grid = fft.ifftn(ghat * decay, dim=(-2, -1)).real
        return {}


@register_operator("decay", family="fields", set="field", kind="field")
class Decay(FieldUpdate):
    """field -> field: acts on the field named by `at:` (no set involved)."""

    EMIT = None                                # field->field: writes the grid in place (evaporation); returns {} — no integrable delta
    SUPPORTED_DIMS = [2, 3]                     # elementwise evaporation, dimension-agnostic
    REQUIRES_PARAMS = []                        # no required params — field target from `at:`; `rate` optional
    MECHANISM_TAGS = ["evaporation", "field_decay", "stigmergy"]
    PARAM_ROLES = {"rate": "evaporation_rate"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")   # the field at `at:`
        self.rate = float(params.get("rate", 0.012))    # evaporation per unit time

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        dt = float(getattr(H.config, "dt", 1.0))
        fld.grid = (fld.grid - self.rate * dt).clamp(min=0.0)
        return {}


_RING = 6                                                  # 3D sensors around the heading axis


def _perp_basis(h):
    """D-1 orthonormal unit vectors spanning the plane perpendicular to each heading
    h [N, D]. In 2D the perpendicular is unique; in 3D we build two (robust to h ~
    +/-z by falling back to a different reference there)."""
    D = h.shape[1]
    if D == 2:
        return [torch.stack([-h[:, 1], h[:, 0]], dim=1)]   # the unique perpendicular
    ref = h.new_tensor([0.0, 0.0, 1.0]).expand_as(h)
    u = torch.cross(h, ref, dim=1)
    small = u.norm(dim=1) < 1e-4                            # h nearly parallel to z
    ref2 = h.new_tensor([0.0, 1.0, 0.0]).expand_as(h)
    u = torch.where(small[:, None], torch.cross(h, ref2, dim=1), u)
    u = u / u.norm(dim=1, keepdim=True).clamp(min=1e-9)
    v = torch.cross(h, u, dim=1)                            # (h, u, v) orthonormal
    return [u, v]


def _ring_dirs(h, ca, sa):
    """The tilted sensor directions around the heading: `cos(ang)*h + sin(ang)*r` for
    each unit r in the perpendicular plane. 2D -> {ahead-left, ahead-right}; 3D -> a
    ring of `_RING` directions. Returns a list of [N, D] unit vectors."""
    D = h.shape[1]
    basis = _perp_basis(h)
    if D == 2:
        rs = [basis[0], -basis[0]]                         # left / right
    else:
        u, v = basis
        rs = [math.cos(2.0 * math.pi * k / _RING) * u + math.sin(2.0 * math.pi * k / _RING) * v
              for k in range(_RING)]
    return [ca * h + sa * r for r in rs]


def _read(fld, centers, weights, ssz):
    """Windowed, species-weighted trail read at a BATCH of sensors (field -> [N, S]).

    Sums dot(weights, grid[:, *window]) over a (2k+1)^D voxel window around each of the
    S sensor centres [N, S, D]; per-agent `ssz` masks offsets outside that agent's own
    window (the 2D `sensor_size` semantics, generalised to N-D). Vectorised over BOTH
    the S sensors and the (2k+1)^D window voxels -- one gather, no Python voxel/sensor
    loop (was 27 gathers per tick in 2D: 3 sensors x 9 voxels)."""
    N, S, D = centers.shape
    dev = centers.device
    g = fld.grid                                           # [C, *shape]
    shape = fld.shape
    per = getattr(fld, "periodic", False)                  # torus field: wrap the window across the seam
    ssz = ssz if torch.is_tensor(ssz) else centers.new_full((N,), float(ssz))
    ks = int(ssz.max().item())

    flat = centers.reshape(N * S, D)
    gidx = torch.stack(fld.pix(*[flat[:, k] for k in range(D)]), dim=-1).reshape(N, S, D)   # [N, S, D]
    rng = torch.arange(-ks, ks + 1, device=dev)
    offs = torch.stack(torch.meshgrid(*([rng] * D), indexing="ij"), dim=-1).reshape(-1, D)  # [W, D] window
    W = offs.shape[0]

    # per-axis wrapped/clamped voxel index for the whole [N, S, W] window (D=2/3, not a hot loop)
    axes = []
    for k in range(D):
        col = gidx[:, :, None, k] + offs[None, None, :, k]                 # [N, S, W]
        axes.append(torch.remainder(col, shape[k]) if per else col.clamp(0, shape[k] - 1))
    vals = g[(slice(None),) + tuple(axes)].permute(1, 2, 3, 0)             # [N, S, W, C]

    inwin = (offs.abs()[None, :, :] <= ssz[:, None, None]).all(-1)         # [N, W] offset inside agent window
    contrib = (weights[:, None, None, :] * vals).sum(-1)                   # [N, S, W]
    return (contrib * inwin[:, None, :].float()).sum(-1)                   # [N, S]


@register_operator("sense", family="signalling", set="cell", kind="exchange")
class Sense(Exchange):
    EMIT = None                                 # writes `heading` in place (steering); returns {} — not an integrable delta
    SUPPORTED_DIMS = [2, 3]                      # dimension-generic (heading is a [N,D] unit vector)
    REQUIRES_PARAMS = ["from"]
    REQUIRES_TYPE_PROPS = ["turn_speed", "sensor_angle", "sensor_dist", "sensor_size"]
    MECHANISM_TAGS = ["trail_following", "stigmergy", "physarum_sensing"]
    PARAM_ROLES = {"cross": "inter_species_coupling_sign", "noise": "steer_noise"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("from")
        self.cross = float(params.get("cross", -1.0))      # sense weight on OTHER species' channels
        self.noise = float(params.get("noise", 0.0))       # steer-noise knob in [0,1]: 0 = deterministic
        self.at = params.get("_at", "cell")                # turn (theta = turn_speed); 1 = uniform[0, turn_speed]

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        N = lvl.n
        pos = lvl.get("pos")                               # [N, D]
        h = lvl.heading                                    # [N, D] unit heading
        fld = H.fields[self.field_name]
        C = fld.C
        nt = lvl.node_type

        ts = lvl.turn_speed                                # [N]
        ang = lvl.sensor_angle * (math.pi / 180.0)         # SpeciesSettings in degrees -> rad [N]
        sd = lvl.sensor_dist[:, None]                      # [N, 1]
        ssz = lvl.sensor_size                              # [N] per-agent window half-width
        ca, sa = torch.cos(ang)[:, None], torch.sin(ang)[:, None]

        # senseWeight: +1 on own channel, `cross` on the others
        w = torch.full((N, C), self.cross, device=dev)
        w[torch.arange(N, device=dev), nt] = 1.0

        dirs = _ring_dirs(h, ca, sa)                       # list of [N, D] tilted directions
        stacked = torch.stack(dirs, dim=1)                 # [N, K, D]
        # centre sensor (heading) + K ring sensors -> one batched windowed read [N, 1+K]
        dir_all = torch.cat([h[:, None, :], stacked], dim=1)           # [N, 1+K, D]
        centers = pos[:, None, :] + dir_all * sd[:, None, :]           # [N, 1+K, D] sensor centres
        reads = _read(fld, centers, w, ssz)                # [N, 1+K]
        centre, ring = reads[:, 0], reads[:, 1:]           # [N] centre, [N, K] ring

        best_val, best_idx = ring.max(1)                   # strongest fan sensor
        target = stacked[torch.arange(N, device=dev), best_idx]        # [N, D]
        straight = centre >= best_val                      # centre wins -> keep heading

        # turn magnitude toward the winning sensor. `noise` knob (default 0) blends a
        # deterministic full turn (frac=1 -> theta=turn_speed) with the stochastic
        # Physarum turn (frac ~ uniform[0,1]); noise=1 reproduces the old `rand*ts`.
        if self.noise > 0.0:
            frac = (1.0 - self.noise) + self.noise * torch.rand(N, generator=H.rng, device=dev)
        else:
            frac = torch.ones(N, device=dev)
        theta = (ts * frac)[:, None]                                                   # turn angle <= turn_speed
        t_perp = target - (target * h).sum(1, keepdim=True) * h         # toward target, perp to h
        t_perp = t_perp / t_perp.norm(dim=1, keepdim=True).clamp(min=1e-9)
        turned = torch.cos(theta) * h + torch.sin(theta) * t_perp      # rotate h by theta toward target
        new_h = torch.where(straight[:, None], h, turned)
        new_h = new_h / new_h.norm(dim=1, keepdim=True).clamp(min=1e-9)

        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ
        keep = (m > 0)[:, None]                            # only live, selected agents turn
        lvl.heading = torch.where(keep, new_h, h)
        return {}


@register_operator("chemotax", family="fields", set="particle", kind="exchange")
class Chemotax(Exchange):
    EMIT = "velocity"                           # default routing; override in the spec with `emit: mpm_acceleration`
    # typed signature (Plexus2 sec. 2.1): field -> set (Exchange). Reads the `from:`
    # field gradient at each node's position, writes a velocity/accel on the node.
    INPUTS = ["particle"]
    OUTPUTS = ["particle"]
    READS = ["pos"]
    WRITES = ["pos"]                            # gain*grad(field) as a velocity (or mpm_acceleration)
    MAPS = ["field"]                            # Exchange: a gather map from the `from:` field
    SUPPORTED_DIMS = [2]                         # Field.grad_at is 2D for now (N-D is a follow-up)
    REQUIRES_PARAMS = ["from"]
    MECHANISM_TAGS = ["gradient_following", "field_templated_aggregation", "field_templated_flow"]
    PARAM_ROLES = {"gain": "field_sensitivity", "noise": "exploration_noise"}
    REFERENCE = "Keller, E. F. & Segel, L. A. (1971). Model for chemotaxis. J. Theor. Biol. 30:225-234."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("from")
        self.gain = float(params.get("gain", 1.0))
        ch = params.get("channel", None)                    # None -> sum all channels (any trail)
        self.channel = None if ch is None else int(ch)
        self.by_material = bool(params.get("by_material", False))  # solids climb (+|gain|), liquids flee (-|gain|)
        self.noise = float(params.get("noise", 0.0))        # isotropic exploration noise (off by default)
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        fld = H.fields[self.field_name]
        grad = fld.grad_at(pos, self.channel, periodic=getattr(H, "periodic", False))   # [N, D]
        if self.by_material and getattr(lvl, "is_liquid", None) is not None:
            # same field, opposite pull per phase: solid climbs the filaments (+|gain|),
            # liquid is pushed into the voids (-|gain|).
            sign = torch.where(lvl.is_liquid, -1.0, 1.0).to(grad.dtype)[:, None]
            d = abs(self.gain) * sign * grad
        else:
            d = self.gain * grad
        d = d * lvl.occ[:, None]
        if self.noise > 0.0:                                # exploratory noise on the chemotactic delta
            d = d + self.noise * torch.randn(d.shape[0], d.shape[-1],
                                             generator=getattr(H, "rng", None),
                                             device=d.device) * lvl.occ[:, None]
        if mask is not None:
            d = d * mask[:, None].float()
        return {self.at: d}


@register_field("prescribed", frame="prescribed")
class PrescribedField(Field):
    """A 1-channel scalar field whose grid is read from a video `[T, ny, nx]` (tif).
    Pure state: the `video` buffer `[T, nx, ny]`, the current `grid` `[1, nx, ny]`,
    and the world<->pixel geometry. No dynamics -- `playback` drives it."""

    def __init__(self, name, source=None, res=None, width=1.0, device="cpu"):
        super().__init__(name)                                 # a video binds to no set (no couples_to)
        import tifffile
        path = source if os.path.isabs(source) else graphs_data_path(source)
        vid = tifffile.imread(path).astype("float32")          # [T, ny, nx] (image rows top->bottom)
        vid = vid[:, ::-1, :].copy()                           # flip vertically: image-top -> domain-top
        v = torch.tensor(vid, device=device).permute(0, 2, 1).contiguous()  # -> [T, nx, ny]
        self.C = 1
        self.T = v.shape[0]
        self.nx, self.ny = v.shape[1], v.shape[2]
        self.width = float(width)
        self.R = self.nx / self.width                          # pixels per world unit (x)
        self.register_buffer("video", v)                       # [T, nx, ny]
        self.register_buffer("grid", v[0:1].clone())           # [1, nx, ny]

    def pix(self, x, y):
        gx = (x.clamp(0, self.width - 1e-6) / self.width * self.nx).long().clamp(0, self.nx - 1)
        gy = (y.clamp(0, 1 - 1e-6) * self.ny).long().clamp(0, self.ny - 1)
        return gx, gy


@register_operator("playback", family="harness", set="field", kind="field")
class Playback(FieldUpdate):
    """field <- data: set the field grid to the current tick's video frame (looping).
    Reads the engine's current frame from `H.frame`. Mutates the field, returns {}."""

    EMIT = None                 # field->field: writes the grid in place from the video; returns {} — no integrable delta
    SUPPORTED_DIMS = [2]        # 2D grid field playback
    REQUIRES_PARAMS = []        # no required params — `_at` (the field to advance) is engine-injected
    MECHANISM_TAGS = ["prescribed_field", "video_playback", "data_driven_field"]
    PARAM_ROLES = {}            # reads no tunable params (only the structural `_at`)
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at")

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        t = int(getattr(H, "frame", 0)) % fld.T
        fld.grid = fld.video[t:t + 1].clone()
        return {}


@register_operator("pacemaker", family="fields", set="field", kind="field")
class Pacemaker(FieldUpdate):
    EMIT = None                 # writes `H.signals[name]` scalar in place; returns {} — no integrable delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []        # no required params — all knobs optional (defaults in __init__)
    MECHANISM_TAGS = ["periodic_source", "clock", "pacemaker"]
    PARAM_ROLES = {"period": "beat_interval", "duration": "active_width", "phase": "beat_offset"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.signal = str(params.get("name", "pacemaker"))   # the H.signals key it writes
        self.period = float(params.get("period", 180.0))     # ticks between beats
        self.duration = float(params.get("duration", 20.0))  # active width (ticks)
        self.phase = float(params.get("phase", 0.0))         # tick offset

    def clock(self, frame: int) -> float:
        s = (frame + self.phase) % self.period
        if s < self.duration:
            return math.sin(math.pi * s / max(self.duration, 1e-9))   # smooth 0 -> 1 -> 0 bump
        return 0.0

    def forward(self, H, mask=None):
        if getattr(H, "signals", None) is None:
            H.signals = {}
        H.signals[self.signal] = float(self.clock(int(getattr(H, "frame", 0))))
        return {}


@register_operator("activation_pulse", family="fields", set="field", kind="field")
class ActivationPulse(FieldUpdate):
    EMIT = None                       # writes a prescribed field; never engine-integrated
    SUPPORTED_DIMS = [2, 3]           # dimension-generic: N-D Gaussian/uniform profile on the [C,nx,ny(,nz)] field
    REQUIRES_PARAMS = []              # no required params — field target from `at:`; all timing knobs optional
    MECHANISM_TAGS = ["activation_field", "gaussian_source", "phase_delay", "travelling_wave", "spatial_clock"]
    PARAM_ROLES = {"radius": "stimulus_width", "center": "stimulus_site", "clock": "pacemaker_signal",
                   "period": "beat_interval", "duration": "active_width",
                   "max_delay": "phase_delay_gain", "delay_from": "delay_map"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")   # activation field at `at:`
        self.channel = int(params.get("channel", 0))
        self.delay_from = params.get("delay_from")                 # None -> shared clock; set -> per-pixel wave
        # shared-clock mode (old pulse_stimulus):
        self.clock = str(params.get("clock", "pacemaker"))         # H.signals key to read p(t)
        self.profile = str(params.get("profile", "gaussian"))      # "gaussian" (localised) | "uniform" (global)
        c = params.get("center", [0.5, 0.5])
        self.center = [float(x) for x in c]                        # N-D site; missing axes default to 0.5 at forward
        self.sigma = float(params.get("radius", 0.12))
        # per-pixel wave mode (old phase_delay_pulse):
        self.period = float(params.get("period", 150.0))           # ticks between beats
        self.duration = float(params.get("duration", 30.0))        # active width (ticks)
        self.phase = float(params.get("phase", 0.0))               # global tick offset
        self.max_delay = float(params.get("max_delay", 10.0))      # ticks of delay at map==1

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        if self.delay_from is None:
            # --- shared clock x spatial profile (old pulse_stimulus) ------------------ #
            dev = fld.grid.device
            shape, R, D = fld.shape, fld.R, len(fld.shape)         # (nx,ny) 2D or (nx,ny,nz) 3D
            # pixel-centre world coordinates per axis: axis 0 spans [0, width], the rest [0, 1]
            axes = [(torch.arange(shape[k], device=dev) + 0.5) / R for k in range(D)]
            grids = torch.meshgrid(*axes, indexing="ij")          # D tensors, each [*shape]
            if self.profile == "uniform":
                bump = torch.ones(shape, device=dev)               # global stimulus: a(x,t) = p(t)
            else:
                ctr = [self.center[k] if k < len(self.center) else 0.5 for k in range(D)]
                r2 = sum((grids[k] - ctr[k]) ** 2 for k in range(D))
                bump = torch.exp(-r2 / (2.0 * self.sigma * self.sigma))   # localised Gaussian site (N-D)
            p = float((getattr(H, "signals", None) or {}).get(self.clock, 0.0))   # this tick's clock value
            fld.grid[self.channel] = p * bump
        else:
            # --- per-pixel delayed wave (old phase_delay_pulse) ----------------------- #
            out = fld.grid[self.channel]                           # [nx, ny] activation channel to write
            delay = H.fields[self.delay_from].grid[0].to(out.device)   # [nx, ny] normalised 0..1
            if delay.shape != out.shape:                           # map at a different resolution: resample
                mode = "bilinear" if delay.dim() == 2 else "trilinear"   # 2D grid vs 3D volume
                delay = Fnn.interpolate(delay[None, None].float(), size=tuple(out.shape),
                                        mode=mode, align_corners=True)[0, 0]
            tau = self.max_delay * delay                           # per-pixel delay (ticks)
            t = float(getattr(H, "frame", 0))
            s = torch.remainder(t - tau + self.phase, self.period)   # local phase, handles t-tau < 0
            act = torch.where(s < self.duration,
                              torch.sin((math.pi / max(self.duration, 1e-9)) * s),
                              torch.zeros_like(s))                 # smooth bump while active, else 0
            fld.grid[self.channel] = act
        return {}


_ACT = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "softplus": F.softplus,
    "sigmoid": torch.sigmoid,
    "identity": lambda x: x,
}


@register_operator("signal", family="signalling", set="neuron", kind="lateral")
class Signal(Lateral):
    EMIT = "velocity"                     # first-order voltage ODE (dv/dt); engine integrates the `voltage` coordinate
    # typed signature (Plexus2 sec. 2.1): a morphism from (neuron, synapse) to neuron,
    # reading neuron voltage + synapse weight, writing the neuron voltage derivative,
    # traversing the pre/post incidence maps. The maps are PART of the signature.
    INPUTS = ["neuron", "synapse"]
    OUTPUTS = ["neuron"]
    READS = ["voltage", "w"]              # neuron membrane voltage; synapse weight block W_e
    WRITES = ["voltage"]                  # returns dv/dt on the neuron voltage
    MAPS = ["pre", "post"]                # gather phi(v) along `pre`; aggregate current along `post`
    SUPPORTED_DIMS = [2, 3]               # voltage is scalar -- the operator ignores spatial dimension
    REQUIRES_PARAMS = ["tau", "edge_set"]
    MECHANISM_TAGS = ["signal_propagation", "connectome", "recurrent"]
    PARAM_ROLES = {
        "tau": "membrane_time_constant",
        "edge_set": "connectome_synapse_set",
        "activation": "presynaptic_nonlinearity",
        "bias": "resting_drive",
        "weight": "synapse_weight_block",
    }
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.tau = float(params["tau"])
        self.edge_set = params["edge_set"]
        self.act = _ACT[params.get("activation", "relu")]
        self.bias = float(params.get("bias", 0.0))
        self.weight_block = params.get("weight", "w")     # synapse state block holding W_e
        self.block = params.get("block", "voltage")       # the neuron state block to evolve
        self.at = params.get("_at", "neuron")

    def forward(self, H, mask=None):
        neuron = H.level(self.at)
        v = neuron.get(self.block)                                 # [N, 1]  membrane voltage
        es = H.level(self.edge_set)
        v_pre = H.gather(self.edge_set, "pre", self.block)         # [E, 1]  presynaptic voltage per edge (lift along `pre`)
        w = es.get(self.weight_block)                              # [E, 1]  fixed synaptic weight W_e
        edge_msg = w * self.act(v_pre)                             # [E, 1]  W_e * phi(v_pre)
        current = H.scatter_along(self.edge_set, "post", edge_msg) # [N, 1]  synaptic current onto post neuron (Aggregate along `post`)
        dv = (-v + self.bias + current) / self.tau                 # [N, 1]  first-order voltage derivative
        dv = dv * neuron.occ[:, None]                              # dormant neurons do not move
        if mask is not None:
            dv = dv * mask[:, None].float()
        return {self.at: dv}
