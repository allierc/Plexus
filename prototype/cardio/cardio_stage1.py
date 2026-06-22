#!/usr/bin/env python
"""Stage 1 of the cardio model: an excitable active material (electrical only).

Set:        tissue_particle  -- a grid of material points, state (pos, u, w).
Operators:  grid_graph      (rewire, fixed-once)  -- material adjacency E_grid
            trigger_pulse   (lateral source)      -- sparse S1/S2 current into u
            excitable_nagumo(lateral)             -- FHN + graph Laplacian, updates u,w

No cells, no segmentation: the nodes are tissue material points. The electrical
state (u,w) is auxiliary and stepped in place (the `heading` precedent); Stage 1
needs no integrator. Built on the packaged Plexus contract (plexus.models.base) so
these operators can graduate to src/plexus/operators/ unchanged.

Run (PYTHONPATH=src so `plexus` imports):
  cd /workspace/Plexus
  PYTHONPATH=src .../python prototype/cardio/cardio_stage1.py specs/s1_planar.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from plexus.models.base import Hierarchy, Level, Lateral, Rewire   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FFMPEG = "/workspace/.conda_envs/neural-graph-linux/bin/ffmpeg"

# FHN rest fixed point for (a,b)=(0.7,0.8): u on the left (stable, excitable) branch.
SCHEMA = {"pos": (0, 2), "u": (2, 3), "w": (3, 4)}


# --------------------------------------------------------------------------- #
#  Set builder: a grid of material points
# --------------------------------------------------------------------------- #
def build_tissue(shape, a=0.7, b=0.8, device="cpu"):
    """A tissue_particle Level on an Hy x Wx grid; u,w initialised at FHN rest."""
    Hy, Wx = shape
    N = Hy * Wx
    ys, xs = np.meshgrid(np.linspace(0, 1, Hy), np.linspace(0, 1, Wx), indexing="ij")
    state = torch.zeros(N, 4, device=device)
    state[:, 0] = torch.tensor(xs.ravel(), dtype=torch.float32)
    state[:, 1] = torch.tensor(ys.ravel(), dtype=torch.float32)
    # rest fixed point: w = (u+a)/b on the cubic nullcline u - u^3/3 = w
    u0 = -1.1994
    w0 = (u0 + a) / b
    state[:, 2] = u0
    state[:, 3] = w0
    lvl = Level("tissue_particle", 0, state, state_schema=SCHEMA)
    lvl.shape = (Hy, Wx)
    lvl.register_buffer("I_ext", torch.zeros(N, device=device))   # per-tick injected current
    H = Hierarchy()
    H.add_level(lvl)
    H.dim = 2
    return H


# --------------------------------------------------------------------------- #
#  Operators
# --------------------------------------------------------------------------- #
class GridGraph(Rewire):
    """Fixed 4- or 8-neighbour lattice adjacency. Built once; never rewired."""
    KIND = "rewire"

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "tissue_particle")
        self.neighbours = int(params.get("neighbours", 4))
        self._built = False

    def forward(self, H, mask=None):
        if self._built:
            return {}
        lvl = H.level(self.at)
        Hy, Wx = lvl.shape
        idx = np.arange(Hy * Wx).reshape(Hy, Wx)
        offs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.neighbours == 8:
            offs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        src, dst = [], []
        for dy, dx in offs:
            ys0, ys1 = max(0, -dy), Hy - max(0, dy)
            xs0, xs1 = max(0, -dx), Wx - max(0, dx)
            a = idx[ys0:ys1, xs0:xs1]
            b = idx[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
            src.append(a.ravel()); dst.append(b.ravel())
        src = np.concatenate(src); dst = np.concatenate(dst)
        lvl.edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long,
                                      device=lvl.state.device)
        self._built = True
        return {}


class TriggerPulse(Lateral):
    """Sparse, time-gated current into u over rectangular regions (S1, S2, ...).

    Conceptually a set-local *source*; pragmatically a Lateral with PREDICTION=None
    that writes the per-tick injected current buffer `I_ext` in place (returns {}).
    Each pulse: {region: [x0,x1,y0,y1] in [0,1], t_start, dur, amp, mode}.
    mode 'set' clamps u to amp inside the region during the pulse (robust ignition);
    mode 'current' adds amp to I_ext (the FHN I_ext term).
    """
    KIND = "lateral"
    PREDICTION = None

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "tissue_particle")
        self.pulses = params["pulses"]

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        lvl.I_ext.zero_()
        t = int(H.tick)
        for p in self.pulses:
            if not (p["t_start"] <= t < p["t_start"] + p["dur"]):
                continue
            x0, x1, y0, y1 = p["region"]
            m = ((pos[:, 0] >= x0) & (pos[:, 0] < x1) &
                 (pos[:, 1] >= y0) & (pos[:, 1] < y1))
            uc = lvl.state_schema["u"][0]
            if p.get("mode", "set") == "set":
                lvl.state[m, uc] = float(p["amp"])
            else:
                lvl.I_ext[m] += float(p["amp"])
        return {}


class ExcitableNagumo(Lateral):
    """FitzHugh-Nagumo reaction + graph Laplacian of u; Euler-steps u,w in place.

      du = u - u^3/3 - w + D * sum_{j~i}(u_j - u_i) + I_ext
      dw = eps * (u + a - b w)
    Kernel identical to NeuralGraph/ODEs/Fitzhug_Nagumo.py; Plexus adds the
    spatial Laplacian over the fixed grid_graph edges. Auxiliary state -> returns {}.
    """
    KIND = "lateral"
    PREDICTION = None

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "tissue_particle")
        self.D = float(params.get("D", 0.6))
        self.a = float(params.get("a", 0.7))
        self.b = float(params.get("b", 0.8))
        self.eps = float(params.get("eps", 0.08))
        self.dt = float(params.get("dt", 0.1))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        u = lvl.get("u").squeeze(1)
        w = lvl.get("w").squeeze(1)
        i, j = lvl.edge_index
        if getattr(lvl, "_deg", None) is None:
            lvl._deg = torch.zeros_like(u).index_add_(0, i, torch.ones_like(u[i])).clamp(min=1)
        lap = torch.zeros_like(u).index_add_(0, i, u[j] - u[i]) / lvl._deg   # normalized Laplacian
        du = u - u.pow(3) / 3.0 - w + self.D * lap + lvl.I_ext
        dw = self.eps * (u + self.a - self.b * w)
        uc = lvl.state_schema["u"][0]; wc = lvl.state_schema["w"][0]
        lvl.state[:, uc] = u + self.dt * du
        lvl.state[:, wc] = w + self.dt * dw
        return {}


OPS = {"grid_graph": GridGraph, "trigger_pulse": TriggerPulse,
       "excitable_nagumo": ExcitableNagumo}


# --------------------------------------------------------------------------- #
#  Minimal engine: build the set, instantiate ops, run the schedule, record u
# --------------------------------------------------------------------------- #
def run(spec, device="cpu"):
    g = spec["general"]
    sset = spec["sets"]["tissue_particle"]
    nag = next(o for o in spec["operators"] if o["op"] == "excitable_nagumo")
    H = build_tissue(tuple(sset["shape"]), a=nag.get("a", 0.7), b=nag.get("b", 0.8),
                     device=device)
    lvl = H.level("tissue_particle")
    Hy, Wx = lvl.shape

    ops = {}
    for o in spec["operators"]:
        params = {**o, "_at": "tissue_particle", "dt": g["dt"]}
        ops[o["op"]] = OPS[o["op"]](params, device=device)
    schedule = spec["schedule"]

    ops["grid_graph"].forward(H)                       # build adjacency once
    record_every = int(g.get("record_every", 2))
    frames = []
    for t in range(int(g["n_frames"])):
        H.tick = t
        for name in schedule:
            ops[name].forward(H)
        if t % record_every == 0:
            frames.append(lvl.get("u").detach().cpu().numpy().reshape(Hy, Wx).copy())
    return np.stack(frames), (Hy, Wx)


# --------------------------------------------------------------------------- #
#  Render u(t) as an mp4
# --------------------------------------------------------------------------- #
def render(u_rec, out, fps=24, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    if os.path.exists(FFMPEG):
        plt.rcParams["animation.ffmpeg_path"] = FFMPEG
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_position([0, 0, 1, 1]); ax.axis("off")
    im = ax.imshow(u_rec[0], cmap="inferno", vmin=-2.0, vmax=2.0, interpolation="nearest")
    writer = FFMpegWriter(fps=fps, bitrate=4000)
    with writer.saving(fig, out, dpi=110):
        for f in range(u_rec.shape[0]):
            im.set_data(u_rec[f])
            writer.grab_frame()
    plt.close(fig)
    print(f"saved {out}  ({os.path.getsize(out)/1e6:.1f} MB, {u_rec.shape[0]} frames)")


def montage(u_rec, out, n=6, title=""):
    """Evolution montage: n evenly spaced u snapshots in a row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.linspace(0, u_rec.shape[0] - 1, n).round().astype(int)
    fig, axs = plt.subplots(1, n, figsize=(2.6 * n, 2.8))
    for k, (ax, f) in enumerate(zip(axs, idx)):
        ax.imshow(u_rec[f], cmap="inferno", vmin=-2.0, vmax=2.0, interpolation="nearest")
        ax.set_title(f"frame {f}", fontsize=9); ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


def archive_run(spec, u_rec, root=None):
    """Write <root>/archive/<name>/{<name>.mp4, <name>_montage.png}."""
    name = spec["general"]["name"]
    d = os.path.join(root or HERE, "archive", name)
    os.makedirs(d, exist_ok=True)
    render(u_rec, os.path.join(d, f"{name}.mp4"), fps=int(spec["general"].get("fps", 24)))
    montage(u_rec, os.path.join(d, f"{name}_montage.png"), title=name)
    with open(os.path.join(d, "spec.yaml"), "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", help="path to a stage-1 spec yaml")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    with open(args.spec) as f:
        spec = yaml.safe_load(f)
    u_rec, _ = run(spec, device=args.device)
    print(f"u range [{u_rec.min():.2f},{u_rec.max():.2f}]  frames {u_rec.shape}")
    d = archive_run(spec, u_rec)
    print(f"archived -> {d}")


if __name__ == "__main__":
    main()
