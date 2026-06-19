"""Slime / Physarum agent model, expressed in the Plexus framework.

Faithful port of Sebastian Lague's GPU slime simulation
(https://github.com/SebLague/Slime-Simulation --
 Assets/Scripts/Slime/SlimeSim.compute, Assets/Scripts/Slime/SlimeSettings.cs),
which is itself an implementation of Jones 2010, "Characteristics of pattern
formation and evolution in approximations of Physarum transport networks".

The original is two GPU compute kernels over an agent buffer and an RGBA trail
texture. Both map one-to-one onto Plexus primitives -- one set, one field, two
operators -- with NO engine change:

    Update  kernel  ->  `slime`         (Exchange @ cell): each agent senses the
                        trail field at three sensors (ahead-left / ahead /
                        ahead-right), turns toward the strongest, steps forward,
                        and deposits onto *its own species' channel*.
    Diffuse kernel  ->  `trail.diffuse` (TrailField.step): per-channel 3x3
                        box-blur toward neighbours (lerp by diffuse weight) plus
                        linear decay -- exactly the shader's `Diffuse`.

Categorical separation (the paper's highest-leverage primitive) is how the
multi-species case is carried. Each Unity `SpeciesSettings` struct becomes a cell
`type` holding its own move / turn / sensor parameters (the engine broadcasts
them to per-agent buffers), and the 4-channel `TrailMap` becomes a C-channel
`TrailField` -- one channel per type. An agent is attracted (+1) to its own
channel and weighted by `cross` on every other channel; Lague's
`senseWeight = speciesMask*2 - 1` is exactly the `cross = -1` special case
(attract self, repel others). One vs two vs three vs four species is then a
change of `types` only -- no new code.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as Fnn

from plexus.models.base import Field, Exchange, Rewire
from plexus.models.registry import register_field, register_operator


# --------------------------------------------------------------------------- #
#  TrailField -- the multi-channel pheromone continuum (the RGBA TrailMap)
# --------------------------------------------------------------------------- #
@register_field("trail", frame="eulerian")
class TrailField(Field):
    """A C-channel trail map on a square grid over [0,W]x[0,1] (one channel per
    species). Implements the Field contract for the slime model:

      deposit_at  (object -> field)  : agents add to their channel at their pixel
                                       (nearest cell, like the shader's int cast),
                                       clamped to 1 (shader `min(1, ...)`).
      sense       (field -> object)  : a windowed, species-weighted read at a
                                       sensor position (the shader's `sense()`).
      step        (self-update)      : per-channel 3x3 box-blur lerp + linear
                                       decay (the shader's `Diffuse` kernel).
    """

    def __init__(self, name, couples_to, n_channels=1, res=200, deposit=5.0,
                 decay=0.2, diffuse=3.0, dt=1.0, device="cpu", width=1.0):
        super().__init__(name, couples_to)
        self.C = int(n_channels)
        self.R = int(res)
        self.width = float(width)
        self.nx = int(round(self.width * self.R))      # square pixels dx = 1/R
        self.ny = self.R
        self.deposit = float(deposit)
        self.decay = float(decay)
        self.diffuse = float(diffuse)
        self.dt = float(dt)
        self.device = device
        self.register_buffer("grid", torch.zeros(self.C, self.nx, self.ny, device=device))

    # --- pixel index of a world position (nearest, matching the int2() cast) -- #
    def _pix(self, x, y):
        gx = (x.clamp(0, self.width - 1e-6) * self.R).long().clamp(0, self.nx - 1)
        gy = (y.clamp(0, 1 - 1e-6) * self.R).long().clamp(0, self.ny - 1)
        return gx, gy

    def deposit_at(self, pos, channel, amount):
        """Add `amount[i]` to channel `channel[i]` at agent i's pixel (object->field)."""
        gx, gy = self._pix(pos[:, 0], pos[:, 1])
        flat = channel * (self.nx * self.ny) + gx * self.ny + gy
        self.grid.view(-1).index_add_(0, flat, amount)
        self.grid.clamp_(max=1.0)

    def sense(self, pos, heading, angle_off, sensor_dist, sensor_size, weights):
        """Windowed, species-weighted trail read at one sensor (field->object).

        Returns, per agent, sum over a (2k+1)^2 window around the sensor centre of
        dot(weights, trail[:, x, y]) -- the shader's `sense()` with `senseWeight`.
        `sensor_dist` and `sensor_size` are per-agent; `weights` is [N, C] (own
        channel +1, others `cross`). A single max-window loop masks out offsets
        beyond each agent's own sensor_size, so per-species sensors still work.
        """
        a = heading + angle_off
        cx, cy = self._pix(pos[:, 0] + torch.cos(a) * sensor_dist,
                           pos[:, 1] + torch.sin(a) * sensor_dist)
        N = pos.shape[0]
        out = pos.new_zeros(N)
        g = self.grid                                   # [C, nx, ny]
        ks = int(sensor_size.max().item()) if torch.is_tensor(sensor_size) else int(sensor_size)
        ssz = sensor_size if torch.is_tensor(sensor_size) else torch.full((N,), float(sensor_size), device=pos.device)
        for ox in range(-ks, ks + 1):
            px = (cx + ox).clamp(0, self.nx - 1)
            for oy in range(-ks, ks + 1):
                py = (cy + oy).clamp(0, self.ny - 1)
                inwin = ((abs(ox) <= ssz) & (abs(oy) <= ssz)).float()      # per-agent window mask
                vals = g[:, px, py].t()                                    # [N, C]
                out = out + (weights * vals).sum(1) * inwin
        return out

    def step(self):
        """Diffuse kernel: per-channel 3x3 box-blur lerp + linear decay, clamp >=0."""
        g = self.grid                                                       # [C, nx, ny]
        gp = Fnn.pad(g.unsqueeze(0), (1, 1, 1, 1), mode="replicate")        # edge-clamp like the shader
        blur = Fnn.avg_pool2d(gp, 3, stride=1).squeeze(0)                   # 3x3 mean, same size
        dw = min(max(self.diffuse * self.dt, 0.0), 1.0)                     # saturate(diffuseRate*dt)
        g = g * (1.0 - dw) + blur * dw
        self.grid = (g - self.decay * self.dt).clamp(min=0.0)               # max(0, blurred - decay*dt)


# --------------------------------------------------------------------------- #
#  slime -- the Update kernel: sense -> steer -> move -> deposit
# --------------------------------------------------------------------------- #
@register_operator("slime", level="cell", kind="exchange")
class SlimeOperator(Exchange):
    """One agent step (Lague `Update`). Acts on `cell`, reading per-agent species
    parameters the engine broadcast from `types`. Self-propelled at fixed speed
    (first-order / overdamped), so it integrates its own position and deposits
    into the field -- an Exchange that mutates `H` and returns `{}` (sanctioned by
    the base contract). Shares `cell.heading` with `motility`/`trail`.
    """

    REQUIRES_PARAMS = ["from", "to"]
    # Each maps to a Unity SpeciesSettings field; required per species (per type).
    REQUIRES_TYPE_PROPS = ["move_speed", "turn_speed", "sensor_angle",
                           "sensor_dist", "sensor_size"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.field_name = params.get("from")
        self.cross = float(params.get("cross", -1.0))   # sense weight on OTHER species' channels

    def forward(self, H, mask=None):
        cell = H.level("cell")
        dev = cell.state.device
        N = cell.n
        pos = cell.state[:, :2]
        th = cell.heading
        fld = H.fields[self.field_name]
        C = fld.C
        nt = cell.node_type

        # per-agent species parameters (broadcast from `types` by the engine)
        ms = cell.move_speed
        ts = cell.turn_speed
        ang = cell.sensor_angle * (math.pi / 180.0)      # SpeciesSettings is in degrees
        sd = cell.sensor_dist
        ssz = cell.sensor_size

        # senseWeight: +1 on own channel, `cross` on the others (mask*2-1 -> cross=-1)
        w = torch.full((N, C), self.cross, device=dev)
        w[torch.arange(N, device=dev), nt] = 1.0

        wF = fld.sense(pos, th, 0.0, sd, ssz, w)
        wL = fld.sense(pos, th, ang, sd, ssz, w)
        wR = fld.sense(pos, th, -ang, sd, ssz, w)

        # steering -- the shader's 4-branch if/elif, scaled by a random strength
        rnd = torch.rand(N, generator=H.rng, device=dev)
        straight = (wF > wL) & (wF > wR)
        randturn = (wF < wL) & (wF < wR)
        turn = torch.where(
            straight, torch.zeros_like(th),
            torch.where(randturn, (rnd - 0.5) * 2.0 * ts,
            torch.where(wR > wL, -rnd * ts,
            torch.where(wL > wR, rnd * ts, torch.zeros_like(th)))))

        new_head = th + turn

        # move forward at the species speed (first-order self-propulsion)
        nx_ = pos[:, 0] + torch.cos(new_head) * ms
        ny_ = pos[:, 1] + torch.sin(new_head) * ms
        W = float(getattr(H, "world_width", 1.0))

        if getattr(H, "periodic", False):
            nx_ = torch.remainder(nx_, W)
            ny_ = torch.remainder(ny_, 1.0)
            out = torch.zeros(N, dtype=torch.bool, device=dev)             # never leaves the torus
        else:
            out = (nx_ < 0) | (nx_ >= W) | (ny_ < 0) | (ny_ >= 1)          # hit a wall
            randang = torch.rand(N, generator=H.rng, device=dev) * 2 * math.pi
            new_head = torch.where(out, randang, new_head)                 # pick a fresh direction
            nx_ = nx_.clamp(0, W - 1e-6)
            ny_ = ny_.clamp(0, 1 - 1e-6)

        new_pos = torch.stack([nx_, ny_], dim=1)

        # apply only to selected agents (selector mask); others are untouched
        mf = (mask.float() if mask is not None else torch.ones(N, device=dev))
        keep = mf > 0
        cell.heading = torch.where(keep, new_head, th)
        cell.state[:, :2] = torch.where(keep[:, None], new_pos, pos)

        # deposit onto own channel (only in-bounds, selected agents) -- shader's else branch
        amount = torch.full((N,), fld.deposit * fld.dt, device=dev) * mf * (~out).float()
        fld.deposit_at(cell.state[:, :2], nt, amount)
        return {}


# --------------------------------------------------------------------------- #
#  trail_graph -- a STRUCTURE (Rewire) operator: build a node+edge graph that
#  approximates the slime transport network from the trail field.
# --------------------------------------------------------------------------- #
def extract_trail_graph(grid, n_nodes=160, radius=0.06, thresh=0.12,
                        edge_min=0.06, edge_k=12):
    """Approximate the slime network as a graph (nodes + edges) from a trail field.

    grid: [nx, ny] non-negative trail density (one channel; sum channels outside).
    Returns (nodes_world [M,2], edges [2,E]) -- the relation E the Rewire operator
    builds. Method: (1) NODES = farthest-point sampling over pixels above `thresh`
    (seeded at the brightest), so nodes spread evenly *along* the network; (2) EDGES
    = node pairs within `radius` whose connecting segment stays on the trail
    (min sampled density > `edge_min`), so an edge exists only where a filament
    actually links two nodes. The result hugs the slime filaments."""
    dev = grid.device
    nx, ny = grid.shape
    g = grid / (grid.max() + 1e-9)
    pix = (g > thresh).nonzero(as_tuple=False).float()        # [P,2] pixel (ix,iy)
    P = pix.shape[0]
    empty = (torch.zeros(0, 2, device=dev), torch.zeros(2, 0, dtype=torch.long, device=dev))
    if P < 8:
        return empty
    M = min(int(n_nodes), P)
    vals = g[(g > thresh)]                                     # [P] density at masked pixels
    chosen = torch.zeros(M, dtype=torch.long, device=dev)
    chosen[0] = vals.argmax()                                  # seed at the brightest pixel
    d = ((pix - pix[chosen[0]]) ** 2).sum(1)
    for i in range(1, M):                                      # farthest-point sampling
        chosen[i] = d.argmax()
        d = torch.minimum(d, ((pix - pix[chosen[i]]) ** 2).sum(1))
    npix = pix[chosen]                                         # [M,2]
    nodes = (npix + 0.5) / ny                                  # world coords (square cells dx=1/ny)

    rp = radius * ny                                           # radius in pixels
    D = torch.cdist(npix, npix)                                # [M,M]
    iu = torch.triu_indices(M, M, 1, device=dev)
    cand = D[iu[0], iu[1]] < rp
    a, b = iu[0][cand], iu[1][cand]                            # candidate pairs within radius
    if a.numel() == 0:
        return nodes, torch.zeros(2, 0, dtype=torch.long, device=dev)
    t = torch.linspace(0, 1, edge_k, device=dev)
    seg = npix[a][:, None, :] * (1 - t[None, :, None]) + npix[b][:, None, :] * t[None, :, None]
    sx = seg[..., 0].round().long().clamp(0, nx - 1)
    sy = seg[..., 1].round().long().clamp(0, ny - 1)
    support = g[sx, sy].min(1).values                         # weakest trail along the segment
    keep = support > edge_min
    edges = torch.stack([a[keep], b[keep]])                   # [2,E]
    return nodes, edges


@register_operator("trail_graph", level="cell", kind="rewire")
class TrailGraphOperator(Rewire):
    """Structure operator (Rewire): builds a relation E -- a node+edge graph
    approximating the slime network -- from the trail field, and stores it on H
    (H.graph_nodes [M,2], H.graph_edges [2,E]; per-species lists if `per_species`).
    Emits no state delta (it maintains a scaffold, per plexus.tex). With
    `per_species`, one graph per trail channel (coloured by species)."""

    REQUIRES_PARAMS = ["from"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.field_name = params.get("from")
        self.n_nodes = int(params.get("n_nodes", 160))
        self.radius = float(params.get("radius", 0.06))
        self.thresh = float(params.get("thresh", 0.12))
        self.edge_min = float(params.get("edge_min", 0.06))
        self.edge_k = int(params.get("edge_k", 12))
        self.per_species = bool(params.get("per_species", False))

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        if self.per_species and fld.C > 1:
            graphs = []
            for c in range(fld.C):
                n, e = extract_trail_graph(fld.grid[c], self.n_nodes, self.radius,
                                           self.thresh, self.edge_min)
                graphs.append((n, e, c))
            H.graph_channels = graphs
            # also a combined view for the simple renderer
            H.graph_nodes, H.graph_edges = graphs[0][0], graphs[0][1]
        else:
            n, e = extract_trail_graph(fld.grid.sum(0), self.n_nodes, self.radius,
                                       self.thresh, self.edge_min)
            H.graph_nodes, H.graph_edges = n, e
            H.graph_channels = None
        return {}
