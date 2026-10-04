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

    `per_axis: true` (exp19, 2026-10-02) makes EVERY axis follow the world box: axis k spans
    [0, world_size[k]] and holds round(world_size[k] R) pixels, still dx = 1/R on every axis. A
    recorded volume that is not square in its last two axes (a larval zebrafish imaged laterally,
    13 x 107 x 144 voxels) is then held as it is, not padded. Off by default: every existing spec,
    whatever its `world:`, keeps the nx = round(W R) by R (by R) grid above.

    Reference: none -- a regular scalar grid is a representation, not a result.
    """

    def __init__(self, name, couples_to=None, components=1, res=200, width=1.0, dim=2, device="cpu",
                 per_axis=False, world_size=None):
        super().__init__(name, couples_to)
        self.C = int(components)
        self.R = int(res)
        self.width = float(width)
        self.dim = int(dim)
        self.nx = int(round(self.width * self.R))      # square pixels, dx = 1/R
        self.ny = self.R
        if per_axis:
            if world_size is None or len(world_size) < self.dim:
                raise ValueError(f"field {name}: `per_axis: true` needs the world box on all {self.dim} axes")
            self.box = tuple(float(w) for w in world_size[:self.dim])
            self.shape = tuple(int(round(w * self.R)) for w in self.box)
            self.nx, self.ny = self.shape[0], self.shape[1]
            if self.dim == 3:
                self.nz = self.shape[2]
        elif self.dim == 2:
            self.shape = (self.nx, self.ny)
        else:                                          # 3D: axes 1,2 span [0,1]
            self.nz = self.R
            self.shape = (self.nx, self.ny, self.nz)
        if not per_axis:
            self.box = (self.width,) + (1.0,) * (self.dim - 1)
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
            box = self.box[k]
            if getattr(self, "periodic", False):
                # floor (not trunc-toward-0) so a coord just below 0 wraps to the far edge
                out.append(torch.remainder(torch.floor(c * self.R).long(), self.shape[k]))   # torus wrap
            else:
                out.append((c.clamp(0, box - 1e-6) * self.R).long().clamp(0, self.shape[k] - 1))
        return tuple(out)


@register_operator("deposit", family="fields", set="cell", kind="exchange",
                   equation=r"""$$c_{s}\big(\mathrm{pix}(\mathbf x_i)\big) \;\leftarrow\;
\min\!\Big(1,\; c_{s}\big(\mathrm{pix}(\mathbf x_i)\big) + a\,\Delta t\Big),
\qquad s=\text{type}(i)$$""")
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
                   implementation="finite_difference",
                   equation=r"""$$c \;\leftarrow\; (1-w)\,c \;+\; w\,\overline{c}_{3\times3},
\qquad w=\operatorname{sat}(\text{rate}\cdot\Delta t)$$""")
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


@register_operator("diffuse", family="fields", set="field", kind="field", model="graphcast",
                   title="Learned neighbour dynamics of a field (GraphCast message passing)",
                   equation=r"""$$h_i=\phi_v(s_i),\;\; e_{ij}\!\leftarrow e_{ij}+\psi^{l}(e_{ij},h_i,h_j),\;\;
h_i\!\leftarrow h_i+\chi^{l}\big(h_i,\textstyle\sum_{j\in N(i)}e_{ij}\big),\;\; s_i\!\leftarrow s_i+\delta(h_i)$$""")
class DiffuseGraphCast(FieldUpdate):
    """The field's own neighbour dynamics, LEARNED: a GraphCast processor on the grid's lattice.

    field -> field, like the two implementations above: one tick moves every voxel's value by an
    amount that depends on it and on its lattice neighbours. `diffuse` fixes that dependence to a
    Laplacian; this model lets a message-passing network learn it from a recording, and a zero
    network leaves the field unchanged (persistence). It is a MODEL of `diffuse`, not a new
    operator, because the contract is the same -- a field's value, updated from its neighbours,
    written in place -- and a diffusion law is one of the laws it can represent (a message
    D (s_j - s_i), summed). What differs is the biology claimed: none. The law is whatever the
    data say, and reading it is the analysis.

    ONE STEP, encode -> process -> decode (Lam et al. 2023, Science 382:1416, "GraphCast", its
    supplementary section 3), on the graph whose nodes are the voxels and whose edges join each
    voxel to its 2D lattice neighbours along each axis (4 in 2D, 6 in 3D):

        h_i    = phi_v(s_i)                                   node latent, from the voxel's value
        e_ij   = phi_e(d_ij)                                  edge latent, from the step d_ij in um
        for l in 1..L (unshared layers):
          e_ij <- e_ij + psi_l(e_ij, h_i, h_j)                edge update, RESIDUAL: e persists
          h_i  <- h_i + chi_l(h_i, sum_{j in N(i)} e_ij)      node update, from RECEIVED edges only
        s_i    <- s_i + delta(h_i)                            the increment, one recorded interval

    s_i is the field's value at voxel i (C channels), h_i its latent (width `latent`), e_ij the
    latent of the edge from j to i, and d_ij the vector from i to j in um (`spacing` per grid
    axis), so an anisotropic voxel (5 um in z against 3.3 um in x and y) is an edge feature and
    not an assumption. Every MLP is Linear -> SiLU -> Linear followed by a LayerNorm, the residual
    added after it (GraphCast's post-norm), except `delta`, which has no norm and whose last layer
    STARTS AT ZERO -- so an untrained model is exactly persistence, s(t+1) = s(t), and the first run
    must reproduce the persistence baseline to the last bit. A neighbour outside the box has no
    edge: its message is dropped, never wrapped around (the recording's 14 z-planes are not a
    torus).

    THE WEIGHTS ARE ONE FLAT TENSOR, `theta`. The trainer owns every parameter and hands it to
    each rollout through `on_ready` (`plexus.trainer.Learnables`, `{param: theta, op: diffuse}`),
    because `engine.run` builds fresh operators on every call. A flat leaf is the one shape that
    mechanism already carries; the MLPs below are read from views of it, functionally. `theta` is
    built in the constructor from `seed`, `channels` and `spacing` -- so a spec rolled out without a
    trainer is the untrained (persistence) model, and a trained one is this spec plus the trainer's
    `theta`.

    THE MULTI-MESH, `mesh_levels: L_m` (> 0; GraphCast's own architecture). The processor above runs on
    the grid itself, so a layer carries information one voxel. GraphCast instead encodes the grid
    onto a coarser MESH, processes on a multi-mesh whose long edges cross the domain in one layer,
    and decodes back (supplement sections 3.1-3.6; `weathernext1_graph/graphcast.py`). On a lattice:

        mesh nodes     one per block of `mesh_stride` voxels (e.g. [1, 2, 2]: 14 x 64 x 64 for a
                       14 x 128 x 128 grid), at the block's centre
        multi-mesh     level k = 0 .. L_m - 1 joins the nodes on the stride-2^k sub-lattice to their
                       neighbours 2^k mesh steps away along each axis; the edges of every level are
                       ONE graph, as GraphCast merges its refinements M0..M6
        grid2mesh      each voxel sends to its block's node
        mesh2grid      each voxel receives from the (up to) 2 nearest nodes along every strided axis

        encoder   g_i = phi_g(s_i),  m_a = phi_m(0),  e_ia = phi_e(d_ia)                grid, mesh, edges
                  e_ia <- e_ia + psi(e_ia, g_i, m_a);  m_a <- m_a + chi(m_a, sum_i e_ia);  g_i <- g_i + xi(g_i)
        processor e_ab = phi_e'(d_ab);  L layers of the edge and node updates above, on the multi-mesh
        decoder   e_ai = phi_e''(d_ai);  e_ai <- e_ai + psi'(e_ai, m_a, g_i);  g_i <- g_i + chi'(g_i, sum_a e_ai)
                  s_i <- s_i + delta(g_i)

    g_i is voxel i's latent, m_a mesh node a's, e the latent of an edge, d its step from sender to
    receiver in um divided by the finest mesh edge's length, with that length appended. Mesh nodes
    carry no input of their own (GraphCast gives them only their position; this law is
    translation-invariant, so they carry nothing). `mesh_levels: 0` is the grid processor above,
    unchanged.

    A PER-VOXEL EMBEDDING, `embedding: <field>` (`embedding_dim: k`). The law stays ONE law, shared by
    every voxel, but each voxel's encoder also reads k numbers of its own from another field of the
    model -- connectome-gnn's per-neuron a_i, and the place GraphCast puts its static grid features
    (land-sea mask, orography), here learned instead of given. The trainer owns that field and its
    representation (`{field: embed, with: tensor | hash}`); this operator only reads it. What the
    embedding is FOR is the analysis: voxels the law must treat differently -- one cell against its
    neighbour, a lumen against cytoplasm -- can only be told apart there, so cells, which the
    recording does not segment, may appear as clusters of voxels in the embedding.

    TRANSPORT, `transport: true` (exp16, Cedric 2026-09-29: "GraphCast is not able to learn a simple
    translation rule"). A second decoder head writes a VELOCITY per voxel, v_i = max_speed tanh(delta_v(h_i)),
    in voxels per tick, and the tick first MOVES the field by it -- semi-Lagrangian: each voxel takes the value
    at its own position minus v, trilinear from the 2^D voxels around it (0 past the box) -- then adds the
    local increment: s <- advect(s, v) + delta(h). The interpolation is written out rather than
    `grid_sample`, so at v = 0 it returns its input exactly: the velocity head starts at 0 and the untrained
    law stays persistence. `transport_embedding: true` moves the per-voxel embedding by the same v each
    tick, so a voxel's identity rides with the tissue instead of staying where it was at t0 (the embedding is
    then state of the rollout, seeded by the trainer). `max_speed` (default 1 voxel per tick) bounds v.

    CHECKPOINTED, `checkpoint: true`: under a tape each tick is recomputed during backward instead of
    stored (`torch.utils.checkpoint`), so a 68-step rollout costs one step's activations, not 68 --
    the same numbers, slower by about one extra forward.

    ONE GLOBAL FORCING, `forcing: T` (exp16, Cedric 2026-09-29). GraphCast feeds its forcings -- solar
    radiation, time of day -- to every grid node beside the state; here one learned number per recorded
    volume, I(t), `self.I` [T], is appended to EVERY voxel's encoder input at the tick that advances
    volume t. It is the one learnable allowed to vary with time, and it is allowed because it is GLOBAL:
    one number per volume for 229,376 voxels can move the organoid-wide level (the washout's source) but
    cannot draw a single spatial pattern -- a per-voxel or per-region time course would be a leak that
    explains the recording by storing it. The rollout's absolute volume index is `t_origin + tick`,
    `t_origin` set by the caller at `on_ready` (the trainer). `forcing: 0` (default) is no forcing.

    A LEARNED STIMULUS, `forcing_model: siren` (exp19, Cedric 2026-10-03). Without a recorded stimulus the forcing
    is learned; a free number per volume (`forcing_model: free`, the default above) can follow ANY global time course,
    the recording's own included. The SIREN (Sitzmann et al. 2020: sine activations, their init) makes it a smooth
    function of time instead, s(t) = SIREN(2 t / (T - 1) - 1), `forcing_channels` K global channels, every voxel
    reading all K through its encoder as exp17's neurons read their stimulus features. Its capacity is limited on
    purpose -- `siren_hidden` units, `siren_layers` layers, `siren_omega` the first layer's frequency scale -- so that
    it cannot explain the data alone: the no-network control (`messages: false`) is its test. The last layer starts
    at 0, so s(t) = 0 and the untrained law is the forcing-free one. Learnable as `{param: I_mlp, op: diffuse}`.

    THE NETWORK CONTROLS, `messages: false` (exp19, Cedric 2026-10-03, exp17's slides 26-27). Every voxel's node
    update then receives NOTHING from its neighbours (the lattice's `agg` is 0; on the multi-mesh the grid skips
    the mesh entirely): each voxel evolves from its own inputs and the forcing alone. Set in a spec, it is the
    law trained with NO NETWORK from the start; set on a trained law's instance (`op.messages = False`) before
    a rollout, it is the trained law with its network weights ZEROED and the forcing kept. Default true.

    Reference: Battaglia, P. W. et al. (2018). Relational inductive biases, deep learning, and
    graph networks. arXiv:1806.01261; Lam, R. et al. (2023). Learning skillful medium-range global
    weather forecasting. Science 382:1416-1421.
    """

    EMIT = None                                  # field->field: writes the grid in place
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["message_passing", "learned_law", "graphcast"]
    PARAM_ROLES = {"theta": "GraphCast model to learn the field's dynamics", "latent": "latent width",
                   "layers": "message-passing layers, unshared", "spacing": "voxel size per axis, um",
                   "channels": "the observed channels", "inputs": "past states read (GraphCast: 2)",
                   "seed": "the initial weights' seed", "mesh_levels": "multi-mesh levels (0: none)",
                   "mesh_stride": "voxels per mesh node, per axis",
                   "embedding": "the field holding each voxel's learned embedding",
                   "embedding_dim": "its width k",
                   "forcing": "length of the global forcing I(t), one per recorded volume (0: none)",
                   "I": "I(t) to learn a global external cue",
                   "checkpoint": "recompute each tick during backward instead of storing it",
                   "transport": "move the field by a learned velocity each tick",
                   "transport_embedding": "move the embedding with the same velocity",
                   "max_speed": "largest velocity, voxels per tick",
                   "messages": "false: no message from any neighbour (the network controls)",
                   "forcing_model": "free (one number per volume, default) or siren (a smooth function of time)",
                   "forcing_channels": "siren: K global channels of the learned stimulus",
                   "siren_hidden": "siren: units per hidden layer", "siren_layers": "siren: layers, the last linear",
                   "siren_omega": "siren: omega_0, the first layer's frequency scale (t in [-1, 1])",
                   "I_mlp": "the SIREN's weights, flat"}
    REFERENCE = ("Lam, R. et al. (2023). Learning skillful medium-range global weather forecasting. "
                 "Science 382:1416-1421; Battaglia, P. W. et al. (2018). arXiv:1806.01261.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")
        self.latent = int(params.get("latent", 16))
        self.layers = int(params.get("layers", 2))
        self.spacing = [float(v) for v in (params.get("spacing") or [])]
        self.seed = int(params.get("seed", 0))
        self.channels = int(params.get("channels", 1))
        # GRAPHCAST READS TWO STATES, x(t - 1) and x(t) (Lam et al. 2023, supplement section 3): with
        # `inputs: k` the field holds k copies of its channels, newest first, the law reads them all,
        # updates the newest and shifts it down. `inputs: 1` is the one-state law.
        self.inputs = int(params.get("inputs", 1))
        self.mesh_levels = int(params.get("mesh_levels", 0))
        self.messages = bool(params.get("messages", True))
        self.mesh_stride = [int(v) for v in (params.get("mesh_stride") or [])]
        self.embedding = params.get("embedding")
        self.embedding_dim = int(params.get("embedding_dim", 0)) if self.embedding else 0
        if self.embedding and not self.embedding_dim:
            raise ValueError("diffuse[graphcast] `embedding:` needs `embedding_dim:`, the field's width")
        self.checkpoint = bool(params.get("checkpoint", False))
        self.transport = bool(params.get("transport", False))
        self.transport_embedding = bool(params.get("transport_embedding", False))
        self.max_speed = float(params.get("max_speed", 1.0))
        if self.transport and self.inputs > 1:
            raise ValueError("diffuse[graphcast]: `transport:` moves one state; it excludes `inputs:` > 1")
        if self.transport_embedding and not (self.transport and self.embedding):
            raise ValueError("diffuse[graphcast]: `transport_embedding:` needs `transport: true` and `embedding:`")
        self.forcing = int(params.get("forcing", 0))
        self.I = torch.zeros(self.forcing, device=device) if self.forcing else None
        self.forcing_model = str(params.get("forcing_model", "free"))
        if self.forcing_model not in ("free", "siren"):
            raise ValueError("diffuse[graphcast] `forcing_model:` free (default) or siren")
        self.fK = int(params.get("forcing_channels", 1)) if self.forcing else 0
        if self.forcing_model == "free" and self.fK > 1:
            raise ValueError("diffuse[graphcast]: `forcing_channels:` > 1 needs `forcing_model: siren`")
        if self.forcing_model == "siren":
            if not self.forcing:
                raise ValueError("diffuse[graphcast] `forcing_model: siren` needs `forcing:`, the number of volumes")
            self._siren_init(int(params.get("siren_hidden", 16)), int(params.get("siren_layers", 3)),
                             float(params.get("siren_omega", 30.0)), device)
        self.t_origin = 0                          # the rollout's first volume, set by the caller
        # GRAPHCAST'S INPUT AND OUTPUT NORMALISATION (supplement 3.7; exp17's `state_diffuse[graphcast]`):
        # mu, sd, dsd = the recording's mean, standard deviation and SD of the ONE-STEP difference, set by the
        # trainer at `on_ready` when the task asks (`reference.normalise: true`). The law reads (s - mu) / sd and
        # its increment is dsd x delta. (0, 1, 1) -- the default -- is no normalisation, every earlier run.
        self.norm = (0.0, 1.0, 1.0)
        self._tick = 0
        self.device_ = device
        if not self.spacing:
            raise ValueError("diffuse[graphcast] needs `spacing:` -- the voxel size along each grid "
                             "axis in um, one entry per dimension (it is the edges' feature)")
        if self.mesh_levels and len(self.mesh_stride) != len(self.spacing):
            raise ValueError("diffuse[graphcast] with `mesh_levels:` needs `mesh_stride:`, the voxels per "
                             "mesh node along each grid axis (e.g. [1, 2, 2])")
        # BUILT HERE, not at the first step: the trainer's `on_ready` reads `theta` before any tick.
        self.theta = self.init_theta(self.channels, len(self.spacing))
        self._graph = None

    # ------------------------------------------------------------------ the forcing
    def _siren_init(self, hidden, layers, omega, device):
        """The SIREN's weights, flat in `I_mlp`: Sitzmann et al.'s init (first layer U(-1/fan, 1/fan), the others
        U(-sqrt(6/fan)/omega, +)), the last layer 0 so s(t) = 0 at the start (exp17's `modulation: siren` init)."""
        import math
        dims = [1] + [hidden] * (layers - 1) + [self.fK]
        self._siren_shapes = [s_ for i in range(layers) for s_ in ((dims[i + 1], dims[i]), (dims[i + 1],))]
        self._siren_omega = omega
        g = torch.Generator().manual_seed(self.seed + 7)
        parts = []
        for i, sh in enumerate(self._siren_shapes):
            if i >= len(self._siren_shapes) - 2 or len(sh) == 1:
                parts.append(torch.zeros(sh))
            else:
                fan = sh[1]
                bnd = 1.0 / fan if i == 0 else math.sqrt(6.0 / fan) / omega
                parts.append((torch.rand(sh, generator=g) * 2 - 1) * bnd)
        self.I_mlp = torch.cat([q.reshape(-1) for q in parts]).to(device)

    def forcing_at(self, t):
        """[fK] the global forcing at volume index t: I[t] (free) or the SIREN at t (siren)."""
        ti = min(max(int(t), 0), self.forcing - 1)
        if self.forcing_model == "free":
            return self.I[ti].reshape(1)
        x = torch.tensor([[2.0 * ti / max(self.forcing - 1, 1) - 1.0]], device=self.I_mlp.device, dtype=self.I_mlp.dtype)
        o, n_lin = 0, len(self._siren_shapes) // 2
        for i in range(n_lin):
            (wo, wi), (bo,) = self._siren_shapes[2 * i], self._siren_shapes[2 * i + 1]
            w = self.I_mlp[o:o + wo * wi].reshape(wo, wi)
            o += wo * wi
            b = self.I_mlp[o:o + bo]
            o += bo
            x = x @ w.T + b
            if i < n_lin - 1:
                x = torch.sin(self._siren_omega * x)
        return x.reshape(self.fK)

    # ------------------------------------------------------------------ the weights, flat
    def layout(self, C, D):
        """[(name, shape)] of every tensor inside `theta`, in order. Linear weights are [out, in].
        The encoder reads all `inputs` states of the C channels; the decoder writes C."""
        H = self.latent
        out = []

        def mlp(name, n_in, n_out, norm=True):
            out.extend([(f"{name}.w1", (H, n_in)), (f"{name}.b1", (H,)),
                        (f"{name}.w2", (n_out, H)), (f"{name}.b2", (n_out,))])
            if norm:
                out.extend([(f"{name}.g", (n_out,)), (f"{name}.beta", (n_out,))])
        if self.mesh_levels:
            F = D + 1                                               # the step d, and its length
            nin = C * self.inputs + self.embedding_dim + self.fK
            for name, n_in, n_out in (("enc_g", nin, H), ("enc_m", nin, H),
                                      ("enc_e_g2m", F, H), ("g2m_edge", 3 * H, H), ("g2m_node", 2 * H, H),
                                      ("g2m_grid", H, H), ("enc_e_mm", F, H)):
                mlp(name, n_in, n_out)
            for layer in range(self.layers):
                mlp(f"mm_edge{layer}", 3 * H, H)
                mlp(f"mm_node{layer}", 2 * H, H)
            mlp("enc_e_m2g", F, H)
            mlp("m2g_edge", 3 * H, H)
            mlp("m2g_node", 2 * H, H)
            mlp("dec", H, C, norm=False)
            if self.transport:
                mlp("dec_v", H, D, norm=False)
            return out
        mlp("enc_v", C * self.inputs + self.embedding_dim + self.fK, H)
        mlp("enc_e", D, H)
        for layer in range(self.layers):
            mlp(f"edge{layer}", 3 * H, H)
            mlp(f"node{layer}", 2 * H, H)
        mlp("dec", H, C, norm=False)
        if self.transport:
            mlp("dec_v", H, D, norm=False)
        return out

    def init_theta(self, C, D):
        """PyTorch's own Linear initialisation (uniform in +-1/sqrt(fan_in)), LayerNorm gain 1 and
        bias 0, and the decoder's last layer 0 -- the persistence start."""
        g = torch.Generator().manual_seed(self.seed)
        lay = self.layout(C, D)
        shapes = dict(lay)
        parts = []
        for name, shape in lay:
            if name.split(".")[0] in ("dec", "dec_v") and name.endswith(("w2", "b2")):
                t = torch.zeros(shape)
            elif name.endswith(".g"):
                t = torch.ones(shape)
            elif name.endswith(".beta"):
                t = torch.zeros(shape)
            else:
                fan_in = shapes[name.replace(".b", ".w")][1]        # a bias takes its weight's fan-in
                t = (torch.rand(shape, generator=g) * 2 - 1) / math.sqrt(fan_in)
            parts.append(t.reshape(-1))
        return torch.cat(parts).to(self.device_)

    def _views(self, C, D):
        views, k = {}, 0
        for name, shape in self.layout(C, D):
            n = math.prod(shape)
            views[name] = self.theta[k:k + n].view(shape)
            k += n
        if k != self.theta.numel():
            raise ValueError(f"diffuse[graphcast]: theta has {self.theta.numel()} values, the layout "
                             f"{k} (latent {self.latent}, layers {self.layers}, {C} channel(s))")
        return views

    @staticmethod
    def _mlp(W, name, x, norm=True):
        y = Fnn.linear(Fnn.silu(Fnn.linear(x, W[f"{name}.w1"], W[f"{name}.b1"])),
                       W[f"{name}.w2"], W[f"{name}.b2"])
        return Fnn.layer_norm(y, y.shape[-1:], W[f"{name}.g"], W[f"{name}.beta"]) if norm else y

    # ------------------------------------------------------------------ the lattice graph
    def graph(self, shape, device):
        """Per direction: the flat index of each voxel's neighbour, whether it exists, and the step."""
        if self._graph is not None and self._graph[0] == shape:
            return self._graph[1]
        D = len(shape)
        sp = self.spacing or [1.0] * D
        if len(sp) != D:
            raise ValueError(f"diffuse[graphcast]: spacing has {len(sp)} entries for a {D}-D field")
        idx = torch.arange(math.prod(shape), device=device).view(shape)
        dirs = []
        for ax in range(D):
            for sgn in (1, -1):
                nb = torch.roll(idx, -sgn, ax)            # the voxel one step along +-ax
                ok = torch.ones(shape, dtype=torch.bool, device=device)
                edge = [slice(None)] * D
                edge[ax] = -1 if sgn == 1 else 0          # no neighbour past the box
                ok[tuple(edge)] = False
                step = torch.zeros(D, device=device)
                step[ax] = sgn * sp[ax]
                dirs.append((nb.reshape(-1), ok.reshape(-1, 1).float(), step / max(sp)))
        self._graph = (shape, dirs)
        return dirs

    def mesh(self, shape, device):
        """The multi-mesh and its two bipartite graphs, as (senders, receivers, features) index
        triples: `g2m` grid -> mesh, `mm` the merged multi-mesh, `m2g` mesh -> grid; and the node count."""
        key = ("mesh", shape)
        if self._graph is not None and self._graph[0] == key:
            return self._graph[1]
        D = len(shape)
        sp = torch.tensor(self.spacing, dtype=torch.float32, device=device)
        st = torch.tensor(self.mesh_stride, device=device)
        gsz = torch.tensor(shape, device=device)
        msh = [-(-n // k) for n, k in zip(shape, self.mesh_stride)]

        def coords(shp):
            return torch.stack(torch.meshgrid(*[torch.arange(n, device=device) for n in shp],
                                              indexing="ij"), -1).reshape(-1, D)

        def flat(ix, shp):
            f = torch.zeros(ix.shape[0], dtype=torch.long, device=device)
            for a in range(D):
                f = f * shp[a] + ix[:, a]
            return f

        def centre(blk):                          # a block's centre, in voxel units (a partial block too)
            return (blk * st).float() + (torch.minimum(st, gsz - blk * st).float() - 1) / 2

        gi, mi = coords(shape), coords(msh)
        gpos, mpos = gi.float() * sp, centre(mi) * sp
        L0 = float(max(sp[a] * st[a] for a in range(D) if msh[a] > 1))   # the finest mesh edge

        def feat(ps, pr):
            d = (ps - pr) / L0
            return torch.cat([d, d.norm(dim=-1, keepdim=True)], -1)

        blk = gi // st
        g_all = torch.arange(gi.shape[0], device=device)
        m_of = flat(blk, msh)
        out = {"n_mesh": mi.shape[0], "g2m": (g_all, m_of, feat(gpos, mpos[m_of]))}
        snd, rcv = [], []
        for k in range(self.mesh_levels):
            sk = 2 ** k
            nodes = mi[(mi % sk == 0).all(-1)]
            for a in range(D):
                for sgn in (1, -1):
                    nb = nodes.clone()
                    nb[:, a] += sgn * sk
                    ok = (nb[:, a] >= 0) & (nb[:, a] < msh[a])
                    rcv.append(flat(nodes[ok], msh))
                    snd.append(flat(nb[ok], msh))
        sn, rc = torch.cat(snd), torch.cat(rcv)
        out["mm"] = (sn, rc, feat(mpos[sn], mpos[rc]))
        cands = [blk]
        c = centre(blk)
        for a in range(D):
            if self.mesh_stride[a] == 1:
                continue
            side = torch.where(gi[:, a].float() > c[:, a], 1, -1)   # the nearer neighbouring block
            more = []
            for b in cands:
                b2 = b.clone()
                b2[:, a] += side
                more.append(b2)
            cands = cands + more
        snd, rcv = [], []
        for b in cands:
            ok = ((b >= 0) & (b < torch.tensor(msh, device=device))).all(-1)
            snd.append(flat(b[ok], msh))
            rcv.append(g_all[ok])
        sn, rc = torch.cat(snd), torch.cat(rcv)
        out["m2g"] = (sn, rc, feat(mpos[sn], gpos[rc]))
        self._graph = (key, out)
        return out

    def _step_mesh(self, s, W, C, shape, a=None):
        """encode (grid -> mesh) -> process (multi-mesh) -> decode (mesh -> grid), one tick."""
        G = self.mesh(shape, s.device)
        H, Nm = self.latent, G["n_mesh"]
        mu, sd, _ = self.norm
        x = ((s - mu) / sd).reshape(s.shape[0], -1).T               # [Ng, C * inputs], normalised
        if a is not None:
            x = torch.cat([x, a.reshape(a.shape[0], -1).T], -1)     # + the voxel's own embedding
        Ng = x.shape[0]

        def agg(rc, e, n):
            return torch.zeros(n, H, device=e.device, dtype=e.dtype).index_add(0, rc, e)

        g = self._mlp(W, "enc_g", x)
        if not self.messages:                                       # no network: the grid never meets the mesh
            g = g + self._mlp(W, "g2m_grid", g)
            return self._mlp(W, "dec", g, norm=False), g
        m = self._mlp(W, "enc_m", torch.zeros(1, x.shape[1], device=x.device, dtype=x.dtype)).expand(Nm, H)
        sn, rc, f = G["g2m"]
        e = self._mlp(W, "enc_e_g2m", f)
        e = e + self._mlp(W, "g2m_edge", torch.cat([e, g[sn], m[rc]], -1))
        m = m + self._mlp(W, "g2m_node", torch.cat([m, agg(rc, e, Nm)], -1))
        g = g + self._mlp(W, "g2m_grid", g)
        sn, rc, f = G["mm"]
        e = self._mlp(W, "enc_e_mm", f)
        for layer in range(self.layers):
            e = e + self._mlp(W, f"mm_edge{layer}", torch.cat([e, m[sn], m[rc]], -1))
            m = m + self._mlp(W, f"mm_node{layer}", torch.cat([m, agg(rc, e, Nm)], -1))
        sn, rc, f = G["m2g"]
        e = self._mlp(W, "enc_e_m2g", f)
        e = e + self._mlp(W, "m2g_edge", torch.cat([e, m[sn], g[rc]], -1))
        g = g + self._mlp(W, "m2g_node", torch.cat([g, agg(rc, e, Ng)], -1))
        return self._mlp(W, "dec", g, norm=False), g                # [Ng, C], and the grid latents

    # ------------------------------------------------------------------ one tick
    @staticmethod
    def _advect(x, v):
        """x [C, *shape] moved by v [D, *shape] (voxels): out(i) = x(i - v(i)), trilinear over the 2^D voxels
        around i - v, weight 0 for a voxel past the box. Exact at v = 0 (one corner of weight 1)."""
        import itertools
        C, shape = x.shape[0], tuple(x.shape[1:])
        D = len(shape)
        ax = torch.meshgrid(*[torch.arange(n, device=x.device, dtype=x.dtype) for n in shape], indexing="ij")
        p = [ax[d] - v[d] for d in range(D)]
        f0 = [torch.floor(q) for q in p]
        fr = [q - q0 for q, q0 in zip(p, f0)]
        i0 = [q0.long() for q0 in f0]
        xf = x.reshape(C, -1)
        out = torch.zeros_like(x)
        for corner in itertools.product((0, 1), repeat=D):
            w = torch.ones_like(fr[0])
            ok = torch.ones(shape, dtype=torch.bool, device=x.device)
            idx = torch.zeros(shape, dtype=torch.long, device=x.device)
            for d in range(D):
                i = i0[d] + corner[d]
                w = w * (fr[d] if corner[d] else 1 - fr[d])
                ok = ok & (i >= 0) & (i < shape[d])
                idx = idx * shape[d] + i.clamp(0, shape[d] - 1)
            out = out + (w * ok) * xf[:, idx.reshape(-1)].reshape(C, *shape)
        return out

    def step(self, s, a=None, t=None, return_v=False):
        """s [C * inputs, *shape] -> the newest state plus delta, the older ones shifted down.
        `a` [embedding_dim, *shape], each voxel's embedding, when the law reads one; `t` the absolute
        volume index this tick advances FROM, when the law reads the global forcing I(t)."""
        C, shape = self.channels, tuple(s.shape[1:])
        D = len(shape)
        if s.shape[0] != C * self.inputs or D != len(self.spacing):
            raise ValueError(f"diffuse[graphcast] was built for {C} channel(s) x {self.inputs} input state(s) "
                             f"in {len(self.spacing)}-D (`channels:`, `inputs:`, `spacing:`); the field is "
                             f"{s.shape[0]} x {shape}")
        W = self._views(C, D)
        if (a is None) != (not self.embedding_dim) or (a is not None and tuple(a.shape) != (self.embedding_dim,) + tuple(s.shape[1:])):
            raise ValueError(f"diffuse[graphcast] reads an embedding of {self.embedding_dim} per voxel; "
                             f"it was given {None if a is None else tuple(a.shape)}")
        if self.forcing:
            if t is None:
                raise ValueError("diffuse[graphcast] with `forcing:` needs the volume index t of each tick")
            f = self.forcing_at(t).reshape(self.fK, *([1] * D)).expand(self.fK, *shape)
            a = f if a is None else torch.cat([a, f], 0)   # a global input, beside the embedding
        if self.mesh_levels:
            ds, h = self._step_mesh(s, W, C, shape, a)
        else:
            dirs = self.graph(shape, s.device)
            mu, sd, _ = self.norm
            x = ((s - mu) / sd).reshape(s.shape[0], -1).T           # [N, C * inputs], normalised
            if a is not None:
                x = torch.cat([x, a.reshape(a.shape[0], -1).T], -1)
            h = self._mlp(W, "enc_v", x)                            # [N, H]
            e = [self._mlp(W, "enc_e", d[2][None]).expand(h.shape[0], -1) for d in dirs]
            for layer in range(self.layers):
                agg = 0.0
                for k, (nb, ok, _) in enumerate(dirs):
                    e[k] = e[k] + self._mlp(W, f"edge{layer}", torch.cat([e[k], h, h[nb]], -1))
                    agg = agg + e[k] * ok                           # a missing neighbour sends nothing
                if not self.messages:                               # the network controls: nothing arrives
                    agg = torch.zeros_like(h)
                h = h + self._mlp(W, f"node{layer}", torch.cat([h, agg], -1))
            ds = self._mlp(W, "dec", h, norm=False)                 # [N, C]
        ds = self.norm[2] * ds                                      # the increment in the field's units
        if self.transport:
            v = (self.max_speed * torch.tanh(self._mlp(W, "dec_v", h, norm=False))).T.reshape((D,) + shape)
            new = self._advect(s[:C], v) + ds.T.reshape((C,) + shape)
            return (new, v) if return_v else new
        new = s[:C] + ds.T.reshape((C,) + shape)
        return torch.cat([new, s[:-C]], 0) if self.inputs > 1 else new

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        a = H.fields[self.embedding].grid if self.embedding else None
        t = self.t_origin + int(getattr(H, "frame", self._tick))
        run = self.step
        if self.checkpoint and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            run = lambda *q, **k: checkpoint(self.step, *q, use_reentrant=False, **k)   # noqa: E731
        if self.transport_embedding:
            fld.grid, v = run(fld.grid, a, t, return_v=True)
            # THE EMBEDDING RIDES WITH THE TISSUE: moved by the same velocity as the field it describes.
            H.fields[self.embedding].grid = self._advect(H.fields[self.embedding].grid, v)
        else:
            fld.grid = run(fld.grid, a, t)
        self._tick += 1
        return {}


@register_operator("diffuse", family="fields", set="field", kind="field", model="known_ode",
                   title="Per-voxel relaxation, exchange between neighbours and a global drive (known ODE)",
                   equation=r"""$$\frac{dr_i}{dt}=\frac{r^*_i-r_i}{\tau_i}+\sum_{j\in N(i)}\kappa_{ij}\,(r_j-r_i)+\beta_i\,I(t),
\qquad \kappa_{ij}=\kappa_{\rm axis}\,e^{-\lVert a_i-a_j\rVert^2}$$""")
class DiffuseKnownODE(FieldUpdate):
    """The INTERPRETABLE RIVAL of `diffuse[model: graphcast]` (exp16, Cedric 2026-10-01): exp17's known ODE (a leak
    toward a rest value, coupling, stimulus weights) translated into the metabolism of a voxel of tissue.

        dr_i/dt = (r*_i - r_i) / tau_i  +  sum_{j in N(i)} kappa_ij (r_j - r_i)  +  beta_i I(t)

    r_i is voxel i's redox ratio (NADH / FAD); r*_i its METABOLIC SET POINT, the ratio its cell returns to at rest;
    1/tau_i its RELAXATION RATE back to it (`rate`, per tick); kappa_ij the EXCHANGE between neighbouring voxels i, j
    (the 2D lattice neighbours, no wrap), beta_i its SENSITIVITY to the washout and I(t) the washout's GLOBAL time
    course, one number per recorded volume (the one time-varying learnable allowed). The exchange is

        kappa_ij = kappa_axis  exp(-|a_i - a_j|^2)

    with a_i voxel i's learned embedding and kappa_axis one rate per grid axis (z planes are 5 um apart, x and y
    3.3 um). Inside a cell the cytoplasm mixes NADH within seconds, so two voxels of one cell should share an
    embedding and exchange at kappa_axis; across a membrane only gap junctions connect hepatocytes (connexin-32), so
    their embeddings should differ and the exchange vanish: THE CELLS, IF THE DATA HOLD THEM, ARE THE SURFACES OF LOW
    kappa. The coupling conserves the total sum r over the box (kappa_ij = kappa_ji), so it moves NADH/FAD balance
    between voxels and never creates it.

    r*_i = mu + rest_i: the `rest` field is an OFFSET from the tissue's mean ratio mu (the trainer's `normalise: true`),
    so set points start at the mean. The per-voxel constants are FIELDS of the model (`rest`, `rate`, `beta`, `embedding`), learned by the trainer as
    fields (`{field: rest, with: tensor | hash}`); kappa_axis (`kappa`) and I(t) (`I`) are tensors of this operator,
    learned as `{param: kappa, op: diffuse}` and `{param: I, op: diffuse}`. All rates start at 0 and beta at 0, so the
    untrained law is EXACTLY persistence. Explicit Euler with `substeps` per tick; the trainer bounds rates >= 0.

    Not a variant of `diffuse[graphcast]`'s options: no MLP, no latent -- every learned number has a name above.
    Reference: Skala, M. C. et al. (2007) PNAS 104:19494 and Walsh, A. J. et al. (2014) Cancer Res 74:5184 (the
    NADH/FAD optical redox ratio); exp17's known ODE, `state_diffuse[model: known_ode]` (cell_ops.py).
    """

    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["relaxation", "exchange", "global_drive", "known_ode"]
    PARAM_ROLES = {"kappa": "exchange rate between neighbouring voxels, per axis", "I": "I(t) the washout's global time course",
                   "rest": "the field of metabolic set points r*", "rate": "the field of relaxation rates 1/tau",
                   "beta": "the field of sensitivities to the washout", "embedding": "the field whose distances set the exchange",
                   "forcing": "length of I(t), one per recorded volume", "substeps": "Euler steps per tick",
                   "spacing": "voxel size per axis, um", "barrier": "the field of per-voxel barriers b >= 0 to exchange",
                   "drive_offset": "the drive is (1 + beta) I(t), so I(t) learns from rest"}
    REFERENCE = ("Skala, M. C. et al. (2007). PNAS 104:19494-19499; Walsh, A. J. et al. (2014). Cancer Res "
                 "74:5184-5194.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")
        self.spacing = [float(v) for v in (params.get("spacing") or [])]
        if not self.spacing:
            raise ValueError("diffuse[known_ode] needs `spacing:`, the voxel size per grid axis in um")
        self.rest, self.rate, self.beta = params.get("rest"), params.get("rate"), params.get("beta")
        self.embedding = params.get("embedding")
        # A PER-VOXEL BARRIER b_i >= 0 (`barrier: <field>`): kappa_ij = kappa_axis exp(-(b_i + b_j)). Membranes are where
        # b is high. It replaces the embedding gate for learning from rest: exp(-|a_i - a_j|^2) has ZERO gradient when the
        # embeddings agree, as they do at the start (v34: the factor stayed 0.999), while exp(-(b_i + b_j)) at b = 0 has
        # gradient -1. The embedding gate is kept so v34-v37 reproduce.
        self.barrier = params.get("barrier")
        # THE DRIVE IS (1 + beta_i) I(t) with `drive_offset: true`: with beta_i I(t) and both starting at 0, each one's
        # gradient is the other, 0, and neither ever moved (v34-v37: beta and I stayed exactly 0).
        self.drive_offset = bool(params.get("drive_offset", False))
        self.substeps = int(params.get("substeps", 4))
        self.forcing = int(params.get("forcing", 0))
        D = len(self.spacing)
        self.kappa = torch.zeros(D, device=device)                 # per axis, learned; 0 = no exchange
        self.I = torch.zeros(self.forcing, device=device) if self.forcing else None
        self.t_origin, self._tick = 0, 0
        self.norm = (0.0, 1.0, 1.0)                                 # the trainer's mean / SD / one-step SD; mean used

    @staticmethod
    def _shift(x, ax, sgn):
        """x's neighbour one step along +-ax (x[i + sgn]), and whether it exists (no wrap)."""
        n = x.shape[ax]
        idx = torch.arange(n, device=x.device) + sgn
        ok = (idx >= 0) & (idx < n)
        y = torch.index_select(x, ax, idx.clamp(0, n - 1))
        shp = [1] * x.dim()
        shp[ax] = n
        return y, ok.view(shp).to(x.dtype)

    def rhs(self, r, rest, rate, beta, a, t, b=None):
        """dr/dt per voxel; r [C, *shape], the per-voxel fields [1, *shape] or None, a [k, *shape] or None."""
        out = torch.zeros_like(r)
        if rate is not None and rest is not None:
            # THE SET POINT IS AN OFFSET FROM THE TISSUE'S MEAN (`norm[0]`, set by the trainer under `normalise: true`):
            # the field starts at 0, so every r* starts at the mean ratio rather than at 0 (0.58 below the tissue).
            out = out + rate * (self.norm[0] + rest - r)
        D = r.dim() - 1
        for d in range(D):
            ax = d + 1
            for sgn in (1, -1):
                rj, ok = self._shift(r, ax, sgn)
                w = self.kappa[d] * ok
                if a is not None:
                    aj, _ = self._shift(a, ax, sgn)
                    w = w * torch.exp(-((a - aj) ** 2).sum(0, keepdim=True))
                if b is not None:
                    bj, _ = self._shift(b, ax, sgn)
                    w = w * torch.exp(-(b + bj))
                out = out + w * (rj - r)
        if self.forcing and (beta is not None or self.drive_offset):
            g = (1.0 + (beta if beta is not None else 0.0)) if self.drive_offset else beta
            out = out + g * self.I[min(max(int(t), 0), self.forcing - 1)]
        return out

    def step(self, r, rest=None, rate=None, beta=None, a=None, t=0, b=None):
        h = 1.0 / self.substeps
        for _ in range(self.substeps):
            r = r + h * self.rhs(r, rest, rate, beta, a, t, b)
        return r

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        get = lambda n: H.fields[n].grid if n else None              # noqa: E731
        t = self.t_origin + int(getattr(H, "frame", self._tick))
        fld.grid = self.step(fld.grid, get(self.rest), get(self.rate), get(self.beta), get(self.embedding), t,
                             get(self.barrier))
        self._tick += 1
        return {}


@register_operator("decay", family="fields", set="field", kind="field",
                   equation=r"""$$c \;\leftarrow\; \max\!\big(0,\; c - k\,\Delta t\big)$$""")
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



@register_operator("decay", family="fields", set="field", kind="field", model="surface_death",
                   title="Cell death at the tissue's surface: the alive fraction of exposed voxels decays",
                   equation=r"""$$\frac{dA_i}{dt}=-k\,e^{\gamma (z_i/Z-1/2)}\,e_i\,g_i(t)\,A_i,\qquad
e_i=1-\tfrac16\sum_{j\in N(i)}A_j,\qquad g_i=\sigma\!\Big(\frac{r_i-r_c}{w}\Big)$$""")
class DecaySurfaceDeath(FieldUpdate):
    """CELL DEATH OF A LIVE ORGANOID (exp16, Cedric 2026-10-01): the tissue mask shrinks from 100 % of frame 1 to 74 %
    by frame 69, lost in frames 10-45 with the washout, one voxel deep at the rim (median depth 3.3 um, no core), and
    lost voxels are more reduced than the rim kept (ratio 0.619 against 0.605). So a voxel dies only where it is
    EXPOSED, at a rate a stress function sets:

        dA_i/dt = -k  exp(gamma (z_i / Z - 1/2))  e_i  g_i(t)  A_i,      e_i = 1 - (1/6) sum_{j in N(i)} A_j

    A_i in [0, 1] is voxel i's ALIVE FRACTION (the field at `at:`; the trainer seeds it with the recorded tissue mask
    at the rollout's origin); e_i its EXPOSURE, the share of its 6 face neighbours that are not alive (outside the box
    counts as not alive), 0 deep inside, ~1/2 on a flat surface; k the death rate per tick (`k`, per 10 min);
    gamma a top-to-bottom gradient of it over the stack (`gamma`, 0 = none; z_i / Z from 0 at the first plane to 1 at
    the last) -- the top planes lose 61 % of their tissue, the bottom one gains 11 %. The stress g_i(t), `hazard:`

        ratio          sigma((r_i - r_c) / w)          the voxel's own redox ratio past a threshold r_c (width w)
        washout        softplus(h(t))                  a learned global time course, one number per recorded volume
        ratio_washout  sigma((r_i - r_c) / w + h(t))   both
        constant       1                               steady erosion, the null

    with r_i read from the field `ratio:`. Every learned number starts where its gradient is not 0: k > 0, r_c at the
    tissue's mean ratio plus `rc` (0), w 0.03, gamma 0, h 0 (sigma(0) = 1/2, softplus(0) = log 2). Explicit Euler with
    `substeps` per tick; A only falls (the tissue the recording GAINS, 11 % of the bottom plane, is not modelled).
    The redox law is untouched: the trainer scores the ratio on the recorded tissue and A against the recorded mask
    (`mask_iou` / `mask_bce`); the movie draws the model on its own tissue, A > 1/2.

    Reference: none for this form -- a surface hazard is a modelling choice read off the recording (exp16, finding
    on the shrinkage), not a published law. Plexus (this work).
    """

    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["cell_death", "surface_erosion", "field_decay"]
    PARAM_ROLES = {"k": "death rate at full exposure and full stress, per tick", "rc": "stress threshold on the ratio, "
                   "offset from the tissue mean", "w": "width of the stress threshold, ratio units",
                   "gamma": "top-to-bottom gradient of the death rate over the stack", "h": "h(t) the global stress time course",
                   "hazard": "what stresses a voxel: ratio | washout | ratio_washout | constant",
                   "ratio": "the field of the redox ratio the stress reads", "forcing": "length of h(t), one per recorded volume",
                   "substeps": "Euler steps per tick",
                   "rc0": "the threshold's origin in ratio units, when the trainer does not normalise (default its mean)",
                   "k0": "the death rate's starting value"}
    REFERENCE = "Plexus (this work); a surface hazard read off the exp16 recording, not a published law."
    HAZARDS = ("ratio", "washout", "ratio_washout", "constant")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")
        self.ratio = params.get("ratio", "ratio")
        self.hazard = params.get("hazard", "ratio")
        if self.hazard not in self.HAZARDS:
            raise ValueError(f"decay[surface_death] hazard {self.hazard!r}; one of {self.HAZARDS}")
        self.substeps = int(params.get("substeps", 4))
        self.forcing = int(params.get("forcing", 0))
        if "washout" in self.hazard and not self.forcing:
            raise ValueError("decay[surface_death] hazard with `washout` needs `forcing:`, the length of h(t)")
        self.k = torch.tensor([float(params.get("k0", 0.05))], device=device)
        self.rc = torch.zeros(1, device=device)
        self.w = torch.tensor([0.03], device=device)
        self.gamma = torch.zeros(1, device=device)
        self.h = torch.zeros(max(self.forcing, 1), device=device)
        self.t_origin, self._tick = 0, 0
        self.norm = (0.0, 1.0, 1.0)                                 # the trainer's mean ratio: r_c = mean + rc
        # THE THRESHOLD'S ORIGIN: the trainer's tissue mean (`normalise: true`), or `rc0` given in ratio units for a law
        # trained without normalising (GraphCast): with r_c at 0 the sigmoid sits at 1 and r_c, w get no gradient.
        self.rc0 = params.get("rc0")

    @staticmethod
    def exposure(A):
        """1 - the mean alive fraction of the 6 face neighbours (2 per axis), outside the box counting as 0."""
        tot = torch.zeros_like(A)
        for ax in range(1, A.dim()):
            for sgn in (1, -1):
                aj, ok = DiffuseKnownODE._shift(A, ax, sgn)
                tot = tot + aj * ok
        return 1.0 - tot / (2 * (A.dim() - 1))

    def stress(self, r, t):
        ht = self.h[min(max(int(t), 0), self.h.numel() - 1)]
        if self.hazard == "constant":
            return torch.ones_like(r)
        if self.hazard == "washout":
            return torch.nn.functional.softplus(ht).expand_as(r)
        x = (r - ((self.norm[0] if self.rc0 is None else float(self.rc0)) + self.rc)) / self.w.clamp(min=1e-3)
        return torch.sigmoid(x + ht if self.hazard == "ratio_washout" else x)

    def step(self, A, r, t=0):
        Z = A.shape[1]
        zf = (torch.arange(Z, device=A.device, dtype=A.dtype) / max(Z - 1, 1) - 0.5).view(1, Z, *([1] * (A.dim() - 2)))
        rate = self.k * torch.exp(self.gamma * zf) * self.stress(r, t)
        h = 1.0 / self.substeps
        for _ in range(self.substeps):
            A = A - h * rate * self.exposure(A) * A
        return A.clamp(0.0, 1.0)

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        t = self.t_origin + int(getattr(H, "frame", self._tick))
        fld.grid = self.step(fld.grid, H.fields[self.ratio].grid[:1], t)
        self._tick += 1
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


@register_operator("sense", family="signalling", set="cell", kind="exchange",
                   equation=r"""$$S_k = \!\!\sum_{\mathbf p\in W_k}\!\Big(c_{\text{own}}(\mathbf p) + \kappa\!\!\sum_{s\ne\text{own}}\!\! c_s(\mathbf p)\Big),
\quad k\in\{C\}\cup\text{fan};
\qquad
\hat{\mathbf h}_i \leftarrow \mathrm{normalize}\big(\hat{\mathbf h}_i + \omega\,\mathbf r_{k^\star}\big)$$""")
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


@register_operator("chemotax", family="fields", set="particle", kind="exchange",
                   equation=r"""$$\dot{\mathbf x}_i=\chi\,\nabla c(\mathbf x_i)+\eta\,\boldsymbol\xi_i$$""")
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


@register_operator("playback", family="harness", set="field", kind="field",
                   equation=r"""$$\phi(\cdot,\,t) \;=\; V\big[\,t \bmod N_{\text{frames}}\,\big]$$""")
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


@register_operator("pacemaker", family="fields", set="field", kind="field",
                   equation=r"""$$p(t) \;=\;
\begin{cases}
\sin\!\big(\pi\, s/\tau_d\big) & s<\tau_d\\[2pt]
0 & \text{otherwise}
\end{cases},
\qquad s=(t+\varphi)\bmod T$$""")
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


@register_operator("activation_pulse", family="fields", set="field", kind="field",
                   equation=r"""$$a(\mathbf x,t)=p(t)\,e^{-\lVert\mathbf x-\mathbf x_0\rVert^2/2\sigma^2}$$""")
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


@register_operator("signal", family="signalling", set="neuron", kind="lateral",
                   equation=r"""$$\frac{dv_i}{dt}=\frac{1}{\tau}\Big(-v_i+b+\!\!\sum_{e\,:\,\mathrm{post}(e)=i}\!\! W_e\,\phi\big(v_{\mathrm{pre}(e)}\big)\Big)$$""")
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
