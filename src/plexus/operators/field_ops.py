"""A continuum bound to a set: what writes into it, what happens inside it, and what reads it.

Deposit, diffuse, decay and sense are one mechanism -- stigmergy, a trail laid and followed --
written as four operators so each can be swapped independently. Reading them apart is also how
a specification ends up depositing into a field that nothing senses.

In the order they appear below:

    grid              field     a C-channel scalar grid; pure state, no behaviour
    deposit           exchange  set -> field: each element adds to the voxel it stands on
    diffuse           field     field -> field: one step of dc/dt = D grad^2 c
    decay             field     field -> field: remove a constant amount everywhere
    sense             exchange  field -> set: read a sensor fan, turn toward the strongest
    chemotax          exchange  field -> set: move along the field's gradient
    prescribed        field     a field read from a video, not solved
    playback          field     advance a prescribed field to this tick's frame
    pacemaker         field     a periodic scalar clock signal p(t)
    activation_pulse  field     paint a clocked activation field, shared clock or travelling wave
    signal            lateral   set -> set along an edge set: connectome signalling

then the alternative implementation, which changes only the numerics:

    diffuse[spectral] the exact heat kernel in Fourier space, against the box-blur default
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
    """A C-channel scalar field on a square-pixel grid. Pure state: it holds a continuum and
    the geometry to index it, and has no dynamics of its own -- operators supply those.

    The domain is the box [0, W] x [0, 1] in 2D, or [0, W] x [0, 1] x [0, 1] in 3D, where W is
    `width` in world units. `res` is R, the resolution in pixels per world unit, so a pixel is
    dx = 1/R world units on every axis and the grid is nx = round(W R) by R (by R). The state is
    one `grid` buffer of shape [C, nx, ny(, nz)], C being the number of channels: a channel per
    species is the usual reading, and `deposit` writes each element into the channel of its own
    type.

    Reference: none -- a regular scalar grid is a representation, not a result.
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
        """The voxel a world position falls in: nearest-voxel, not interpolated.

        Takes D coordinate tensors (x, y[, z]) and returns a D-tuple of index tensors. Axis 0
        spans [0, W], every other axis spans [0, 1], and all of them map through the same
        pixels-per-world-unit R. Under `self.periodic` the index wraps modulo the grid rather
        than clamping to the edge, so a sensor reaching past one side reads the other -- the
        same torus the periodic particle wrap uses."""
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
    """Deposition: each element adds to the field at the voxel it stands on. The write half
    of stigmergy -- an ant laying pheromone, a slime mould laying trail.

    cell -> field: reads pos and node_type, writes the `to:` field in place.

        g[t_i, pix(x_i)] <- g[t_i, pix(x_i)] + a dt,   then  g <- min(g, 1)

    a is `amount`, the deposition rate in field units per unit time, so the amount actually
    laid in one tick is a dt. t_i is the element's own type, which selects the channel: two
    species lay into two channels of the same field and `sense` can then weigh them
    differently. The write is nearest-voxel and additive, so co-located elements accumulate.

    The field saturates at 1 rather than growing without bound. That ceiling is part of the
    mechanism, not a guard: it is what makes an established trail stop getting more attractive
    and lets a second trail compete with it.

    Reference: Grasse, P.-P. (1959). La reconstruction du nid et les coordinations
    interindividuelles chez Bellicositermes natalensis et Cubitermes sp. (stigmergy).
    Insectes Sociaux 6:41-80.
    """

    EMIT = None                                # writes the grid in place, returns no delta
    # Typed signature: the output is the `to:` field grid, not set state, so OUTPUTS and
    # WRITES are empty and the field coupling appears as the "field" map.
    INPUTS = ["cell"]
    OUTPUTS = []                               # writes the `to:` field, no set-state output
    READS = ["pos"]
    WRITES = []                                # no set-state block written (the grid is mutated in place)
    MAPS = ["field"]                           # Exchange: a scatter map onto the `to:` field
    SUPPORTED_DIMS = [2, 3]                     # N-D scatter onto the grid field
    REQUIRES_PARAMS = ["to"]
    MECHANISM_TAGS = ["deposition", "stigmergy", "field_write"]
    PARAM_ROLES = {"amount": "deposit_rate"}
    REFERENCE = ("Grasse, P.-P. (1959). La reconstruction du nid et les coordinations "
                 "interindividuelles chez Bellicositermes natalensis et Cubitermes sp. "
                 "(stigmergy). Insectes Sociaux 6:41-80.")

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
    """Diffusion: the field spreads down its own gradient. One step of the heat equation.

    field -> field: acts on the field named by `at:`, writing its grid in place. No set is
    involved, which is why the contract's set is `field`.

        dc/dt = D grad^2 c

    c is the field value, per channel and independently, and D the diffusion coefficient in
    world units squared per unit time. This implementation steps it by blending toward a
    3x3 (2D) or 3x3x3 (3D) box mean B(c):

        c <- (1 - w) c + w B(c),        w = saturate(rate * dt)

    Expanding B in Taylor series gives B(c) - c = (dx^2 / 3) grad^2 c in both 2D and 3D, so the
    coefficient this implementation actually realises is D = rate * dx^2 / 3 = rate / (3 R^2)
    world units squared per time, R being the field's pixels per world unit. `rate` is
    therefore a blend weight and not D itself. Saturating w at 1 is what keeps an explicit step
    stable at large rate * dt, at the cost of silently capping the diffusion once it binds.

    A periodic field wraps the blur across the seam; otherwise the edge value is replicated,
    which is a no-flux (reflecting) boundary.

    Reference: Fick, A. (1855). Ueber Diffusion. Ann. Phys. 170:59-86; Turing, A. M. (1952).
    The chemical basis of morphogenesis. Phil. Trans. R. Soc. B 237:37-72.
    """

    EMIT = None                                # field->field: writes the grid in place; returns {} — no integrable delta
    SUPPORTED_DIMS = [2, 3]                     # 3x3 (2D) / 3x3x3 (3D) box-blur step
    REQUIRES_PARAMS = []                        # no required params — target field comes from `at:` (engine-injected)
    MECHANISM_TAGS = ["diffusion", "field_smoothing", "laplacian"]
    PARAM_ROLES = {"rate": "diffusion_rate"}
    REFERENCE = ("Fick, A. (1855). Ueber Diffusion. Ann. Phys. 170:59-86; Turing, A. M. "
                 "(1952). The chemical basis of morphogenesis. Phil. Trans. R. Soc. B "
                 "237:37-72.")

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
    """The same diffusion, stepped exactly instead of approximately: in Fourier space the
    heat equation is diagonal, so one step is a multiplication rather than a stencil.

        c_hat(k) <- c_hat(k) exp(-D k^2 dt)

    k is the wavenumber, and the step is exact for any dt -- there is no stability limit and no
    numerical broadening, which is the reason to choose it. Differentiable through torch.fft,
    so an inverse loop filtering `capabilities()` for differentiability keeps this one.

    2D only, and periodic by construction: an FFT cannot express any other boundary.

    THE OPERATING POINT IS NOT THE SAME AS THE DEFAULT IMPLEMENTATION'S. Here `rate` is D
    measured in pixels squared per unit time, because the wavenumbers are built per grid cell.
    In `Diffuse` the realised coefficient is rate / 3 in those same units. The same `rate` in a
    specification therefore diffuses three times faster through this implementation than
    through the box-blur one, so the two are not interchangeable at a fixed parameter value.

    Reference: Fick, A. (1855). Ueber Diffusion. Ann. Phys. 170:59-86; Turing, A. M. (1952).
    The chemical basis of morphogenesis. Phil. Trans. R. Soc. B 237:37-72.
    """

    EMIT = None
    SUPPORTED_DIMS = [2]                        # FFT step is 2D here (N-D is a follow-up)
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["diffusion", "field_smoothing", "spectral"]
    PARAM_ROLES = {"rate": "diffusion_coefficient"}
    REFERENCE = ("Fick, A. (1855). Ueber Diffusion. Ann. Phys. 170:59-86; Turing, A. M. "
                 "(1952). The chemical basis of morphogenesis. Phil. Trans. R. Soc. B "
                 "237:37-72.")

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
    """Evaporation: the field loses a fixed amount everywhere, floored at zero. What stops a
    deposited trail from being permanent, and so what sets how long the past is remembered.

    field -> field: acts on the field named by `at:`, writing its grid in place.

        c <- max(c - r dt, 0)

    r is `rate`, in field units per unit time. Note the form: this is a CONSTANT amount removed
    per unit time, not exponential decay -- it is not dc/dt = -r c. A voxel at 0.2 and one at
    1.0 lose the same absolute amount each tick, so a faint trail vanishes proportionally much
    sooner than a strong one, and every voxel reaches exactly zero in finite time rather than
    approaching it. Combined with `deposit`'s ceiling of 1, the field's memory is at most
    1 / (r dt) ticks.

    Reference: none -- linear removal is a modelling choice here, not a published law. Plexus
    (this work).
    """

    EMIT = None                                # writes the grid in place, returns no delta
    SUPPORTED_DIMS = [2, 3]                     # elementwise evaporation, dimension-agnostic
    REQUIRES_PARAMS = []                        # no required params — field target from `at:`; `rate` optional
    MECHANISM_TAGS = ["evaporation", "field_decay", "stigmergy"]
    PARAM_ROLES = {"rate": "evaporation_rate"}
    REFERENCE = "Plexus (this work); linear removal, not exponential decay."

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

    Sums dot(weights, grid[:, *window]) over a (2k+1)^D voxel window around each of the S
    sensor centres [N, S, D]; the per-agent `ssz` masks out offsets falling outside that
    agent's own window half-width. Vectorised over both the S sensors and the (2k+1)^D window
    voxels, so the whole fan is one gather rather than a Python loop over sensors and voxels."""
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
    """Trail following: read the field on a fan of sensors around the heading and turn toward
    the strongest. The read half of stigmergy, and the steering rule of the Physarum model.

    cell -> cell: reads pos, heading and the field named by `from:`, writes heading in place.

    Each element places one sensor straight ahead and K to the side -- in 2D the two at
    +/- sensor_angle, in 3D a ring of six around the heading axis -- each at distance
    sensor_dist, in world units, from the element:

        x_sensor = x_i + d_i (cos(alpha_i) n_i + sin(alpha_i) r)

    n_i is the unit heading, alpha_i the per-type `sensor_angle` (given in degrees and
    converted here), d_i the per-type `sensor_dist`, and r a unit vector perpendicular to n_i.
    Each sensor returns the weighted sum of the field over a window of half-width
    `sensor_size` voxels: weight +1 on the element's own channel and `cross` on every other,
    so cross = -1 means another species' trail repels as strongly as its own attracts, and
    cross = +1 means the species are indistinguishable.

    If the centre sensor reads at least as much as the best side sensor the heading is kept.
    Otherwise the heading rotates toward the winning direction by

        theta_i = turn_speed_i * ((1 - eta) + eta u),   u ~ uniform[0, 1]

    where turn_speed is the per-type maximum turn per tick in radians and eta is `noise`, in
    [0, 1]: eta = 0 always turns the full amount and is deterministic; eta = 1 turns a uniform
    random fraction of it, the stochastic Physarum rule.

    Reference: Jones, J. (2010). Characteristics of pattern formation and evolution in
    approximations of Physarum transport networks. Artificial Life 16:127-153.
    """

    EMIT = None                                 # writes heading in place, returns no delta
    SUPPORTED_DIMS = [2, 3]                      # dimension-generic (heading is a [N,D] unit vector)
    REQUIRES_PARAMS = ["from"]
    REQUIRES_TYPE_PROPS = ["turn_speed", "sensor_angle", "sensor_dist", "sensor_size"]
    MECHANISM_TAGS = ["trail_following", "stigmergy", "physarum_sensing"]
    PARAM_ROLES = {"cross": "inter_species_coupling_sign", "noise": "steer_noise"}
    REFERENCE = ("Jones, J. (2010). Characteristics of pattern formation and evolution in "
                 "approximations of Physarum transport networks. Artificial Life 16:127-153.")

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

        # Turn magnitude toward the winning sensor: `noise` blends a deterministic full turn
        # (frac = 1, theta = turn_speed) with the stochastic Physarum turn (frac uniform on
        # [0, 1]). Default 0, i.e. deterministic.
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
    """Chemotaxis: move along a chemical gradient, up it or down it. The continuum-sensing
    counterpart of `sense`, which samples the field at discrete points instead.

    particle -> particle: reads pos and the gradient of the `from:` field, emits a velocity.

        dx_i/dt = chi grad c(x_i)  +  eta xi_i

    chi is `gain`, the chemotactic sensitivity, in world units squared per unit time per field
    unit -- it converts a field gradient into a speed. Its sign is the direction of travel:
    positive climbs the gradient (attraction), negative descends it (repulsion). eta is
    `noise`, an isotropic exploration velocity, and xi_i a standard normal vector. `channel`
    picks one channel of a multi-species field; omitting it sums over all of them, so the
    element responds to any trail.

    With `by_material: true` the sign is taken from the particle's own phase instead of from
    `gain`: solids climb the gradient with +|chi| and liquids descend it with -|chi|, so one
    field drives both phases in opposite directions -- solid into the filaments, liquid into
    the voids.

    Emits a velocity by default, the overdamped reading. A specification writing
    `emit: mpm_acceleration` routes the same quantity to the MPM substep as a body force
    instead, which is a different physical claim about what the gradient does.

    Reference: Keller, E. F. & Segel, L. A. (1971). Model for chemotaxis. J. Theor. Biol.
    30:225-234.
    """

    EMIT = "velocity"                           # first-order by default; `emit: mpm_acceleration` reroutes it
    INPUTS = ["particle"]
    OUTPUTS = ["particle"]
    READS = ["pos"]
    WRITES = ["pos"]                            # gain*grad(field) as a velocity (or mpm_acceleration)
    MAPS = ["field"]                            # Exchange: a gather map from the `from:` field
    SUPPORTED_DIMS = [2]                         # Field.grad_at is 2D for now (N-D is a follow-up)
    REQUIRES_PARAMS = ["from"]
    MECHANISM_TAGS = ["gradient_following", "field_templated_aggregation", "field_templated_flow"]
    PARAM_ROLES = {"gain": "field_sensitivity", "noise": "exploration_noise"}
    REFERENCE = ("Keller, E. F. & Segel, L. A. (1971). Model for chemotaxis. J. Theor. Biol. "
                 "30:225-234.")

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
    """A field that is measured rather than solved: its values come from a video, one frame
    per tick, so the continuum is an input to the model instead of an output of it.

    Pure state: the whole `video` buffer [T, nx, ny] read from a TIFF, the current `grid`
    [1, nx, ny], and the world-to-pixel geometry. One channel only. No dynamics of its own --
    `playback` is the operator that advances it, and nothing writes back into it.

    The video is flipped vertically on load, because image rows run top to bottom while the
    domain's y axis runs bottom to top; without the flip every gradient read from it would
    point the wrong way.

    Reference: none -- a prescribed field is data, not a model.
    """

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
    """Advance a prescribed field to this tick's frame, looping when the video runs out.

    field -> field: reads the engine's frame counter, writes the field's grid in place.

        c(x, t) = video[t mod T](x)

    T is the number of frames in the video. Looping means a specification longer than the
    recording repeats it, which is a claim that the process is periodic -- if it is not, the
    seam is an artefact the model will still respond to.

    Reference: none -- playback is bookkeeping, not a mechanism. Plexus (this work).
    """

    EMIT = None                 # field->field: writes the grid in place from the video; returns {} — no integrable delta
    SUPPORTED_DIMS = [2]        # 2D grid field playback
    REQUIRES_PARAMS = []        # no required params — `_at` (the field to advance) is engine-injected
    MECHANISM_TAGS = ["prescribed_field", "video_playback", "data_driven_field"]
    PARAM_ROLES = {}            # reads no tunable params (only the structural `_at`)
    REFERENCE = "Plexus (this work); playback of a measured field, not a mechanism."

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
    """A clock: one periodic scalar p(t), shared by every operator that reads it. Not a field
    over space -- a single number per tick, published under `name` for others to consume.

    field -> signal: reads the engine's frame counter, writes H.signals[name].

        s(t) = (t + phi) mod P
        p(t) = sin(pi s / d)  if s < d,  else 0

    P is `period`, the interval between beats in ticks; d is `duration`, how many ticks each
    beat stays active; phi is `phase`, a tick offset that lets two pacemakers run out of step.
    The active part is a half sine, so p rises smoothly from 0 to 1 and back rather than
    switching -- a square pulse would inject a discontinuity into whatever integrates it. The
    duty cycle is d / P, and p = 0 for the rest of the period.

    Reference: none -- a periodic forcing term is a modelling choice, not a published law.
    Plexus (this work).
    """

    EMIT = None                 # writes a scalar into H.signals, returns no delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []        # no required params — all knobs optional (defaults in __init__)
    MECHANISM_TAGS = ["periodic_source", "clock", "pacemaker"]
    PARAM_ROLES = {"period": "beat_interval", "duration": "active_width", "phase": "beat_offset"}
    REFERENCE = "Plexus (this work); a half-sine periodic forcing term."

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
    """Paint a clocked activation field: where a stimulus is, and when it arrives there. One
    operator with two timing modes, chosen by whether a delay map is given.

    field -> field: writes one channel of the field named by `at:`, in place.

    Without `delay_from`, every point shares one clock and differs only in how strongly it is
    stimulated:

        a(x, t) = p(t) exp(-|x - x0|^2 / 2 sigma^2)        profile: gaussian
        a(x, t) = p(t)                                     profile: uniform

    p(t) is the scalar published by a `pacemaker` under the key named in `clock`, x0 is
    `center` in world coordinates and sigma is `radius`, the stimulus width in world units.
    Every point beats at the same instant, which is a claim that conduction is instantaneous.

    With `delay_from` naming a normalised [0, 1] field m(x), each point instead runs the same
    beat shifted in time:

        tau(x) = T_max m(x)
        s(x, t) = (t - tau(x) + phi) mod P
        a(x, t) = sin(pi s / d)  if s < d,  else 0

    T_max is `max_delay`, the delay in ticks where the map reads 1, and P, d, phi are the
    period, duration and phase in ticks as in `pacemaker`. The activation is then a wave
    travelling outward along whatever gradient the delay map encodes, at a speed set by
    1 / grad tau -- which is how a conduction system is expressed without simulating one. A
    delay map at a different resolution from the field is resampled by interpolation.

    Reference: none -- a prescribed stimulus is a boundary condition, not a mechanism. Plexus
    (this work).
    """

    EMIT = None                       # writes a prescribed field in place; never engine-integrated
    SUPPORTED_DIMS = [2, 3]           # dimension-generic: N-D Gaussian/uniform profile on the [C,nx,ny(,nz)] field
    REQUIRES_PARAMS = []              # no required params — field target from `at:`; all timing knobs optional
    MECHANISM_TAGS = ["activation_field", "gaussian_source", "phase_delay", "travelling_wave", "spatial_clock"]
    PARAM_ROLES = {"radius": "stimulus_width", "center": "stimulus_site", "clock": "pacemaker_signal",
                   "period": "beat_interval", "duration": "active_width",
                   "max_delay": "phase_delay_gain", "delay_from": "delay_map"}
    REFERENCE = "Plexus (this work); a prescribed stimulus, i.e. a boundary condition."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")   # activation field at `at:`
        self.channel = int(params.get("channel", 0))
        self.delay_from = params.get("delay_from")                 # None -> shared clock; set -> per-pixel wave
        # shared-clock mode: one clock, a spatial profile
        self.clock = str(params.get("clock", "pacemaker"))         # H.signals key to read p(t)
        self.profile = str(params.get("profile", "gaussian"))      # "gaussian" (localised) | "uniform" (global)
        c = params.get("center", [0.5, 0.5])
        self.center = [float(x) for x in c]                        # N-D site; missing axes default to 0.5 at forward
        self.sigma = float(params.get("radius", 0.12))
        # per-pixel wave mode: the same beat, delayed by a map
        self.period = float(params.get("period", 150.0))           # ticks between beats
        self.duration = float(params.get("duration", 30.0))        # active width (ticks)
        self.phase = float(params.get("phase", 0.0))               # global tick offset
        self.max_delay = float(params.get("max_delay", 10.0))      # ticks of delay at map==1

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        if self.delay_from is None:
            # --- shared clock x spatial profile: every point beats at once --------------- #
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
            # --- per-pixel delayed wave: the beat arrives late where the map is high ----- #
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
    """Passive connectome signalling: a neuron relaxes toward the summed input arriving along
    its incoming synapses. A firing-rate network, with no spikes and no channel dynamics.

    (neuron, synapse) -> neuron: reads the neuron voltage and the synapse weight, traverses
    the `pre` and `post` incidence maps, emits dv/dt.

        dv_i/dt = ( -v_i + b + sum_{e : post(e) = i} W_e phi(v_pre(e)) ) / tau

    v_i is the membrane voltage of neuron i and tau its membrane time constant, in the same
    time units as the specification's dt -- it is the only timescale in the operator, so
    everything is measured against it. b is `bias`, a constant resting drive in voltage units.
    W_e is the fixed weight of synapse e, taken from the synapse set's `weight` block, and its
    sign is what makes the synapse excitatory or inhibitory. phi is `activation`, the
    presynaptic nonlinearity -- relu (the default, a threshold-linear rate), tanh, softplus,
    sigmoid, or identity for a purely linear network.

    The two maps are part of the signature, not an implementation detail: `pre` lifts each
    neuron's voltage onto the synapses leaving it, and `post` aggregates the resulting currents
    back onto the neuron receiving them. That aggregation is what makes the connectome, rather
    than a dense matrix, the object the operator acts over.

    Reference: Wilson, H. R. & Cowan, J. D. (1972). Excitatory and inhibitory interactions in
    localized populations of model neurons. Biophys. J. 12:1-24; Hopfield, J. J. (1984).
    Neurons with graded response have collective computational properties like those of
    two-state neurons. PNAS 81:3088-3092.
    """

    EMIT = "velocity"                     # first-order: dv/dt, engine-integrated on the voltage block
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
    REFERENCE = ("Wilson, H. R. & Cowan, J. D. (1972). Excitatory and inhibitory interactions "
                 "in localized populations of model neurons. Biophys. J. 12:1-24; Hopfield, "
                 "J. J. (1984). PNAS 81:3088-3092.")

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
