"""neural_assemblies -- a recurrent E/I connectome run as a Plexus signalling *spec*.

Demonstrates PR1 (State) + PR2 (incidence / edge-sets) + PR3 (the `signal` operator):
a block-structured random recurrent network -- K assemblies, balanced random weights, gain
in the chaotic regime -- is expressed as one Plexus hierarchy

    network
      |-- neuron   (state: voltage)                 -- a set
      +-- synapse  (edge-set: pre/post -> neuron, w) -- the connectome

and evolved by the single `signal` operator

    tau * dv_i/dt = -v_i + sum_{e: post(e)=i} W_e * phi(v_{pre(e)}) .

No neuroscience is baked into the engine -- this is a specification over the same
primitives that run particles, boids, and MPM. We plot the resulting voltage traces,
coloured by assembly, in the stacked connectome-cx style.

    python prototype/neural_assemblies/run.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
torch.use_deterministic_algorithms(True, warn_only=True)

import plexus.operators  # noqa: F401  self-registers `signal`
from plexus.schema import Spec, OpSpec, Selector
from plexus.engine import build, _resolve_emit, _integrate
from plexus.models.registry import get_operator

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
#  the connectome: K assemblies, block-structured balanced-random weights (chaotic RNN)
# --------------------------------------------------------------------------- #
def make_connectome(N=90, K=3, p_within=0.5, p_cross=0.3,
                    g=4.2, within_boost=1.6, seed=0):
    """Block-structured random recurrent connectome. Weights are balanced (mean-0)
    Gaussian -- the classic chaotic-RNN regime -- with within-assembly coupling boosted,
    so the K assemblies form correlated chaotic clusters. `g` sets the gain (past the
    chaos onset, tanh keeps activity bounded)."""
    rng = np.random.default_rng(seed)
    assembly = np.repeat(np.arange(K), N // K)                 # assembly id per neuron
    scale = g / np.sqrt(N)
    edges, weights = [], []
    for j in range(N):                                         # presynaptic
        for i in range(N):                                     # postsynaptic
            if i == j:
                continue
            same = assembly[i] == assembly[j]
            if rng.random() < (p_within if same else p_cross):
                boost = within_boost if same else 1.0          # assemblies = stronger within-block recurrence
                w = boost * scale * rng.standard_normal()
                edges.append([j, i]); weights.append(float(w))
    return assembly, edges, weights


# --------------------------------------------------------------------------- #
#  run it through the Plexus engine
# --------------------------------------------------------------------------- #
def simulate(N=90, K=3, T=900, dt=0.1, tau=1.0, seed=0):
    assembly, edges, weights = make_connectome(N=N, K=K, seed=seed)

    sets = {
        "network": {"n": 1},
        "neuron":  {"parent": "network", "per_parent": N, "state": {"voltage": 1}},
        "synapse": {"parent": "network", "edge_set": True, "pre": "neuron", "post": "neuron",
                    "edges": edges, "weights": weights,
                    "state": {"w": {"width": 1, "integration": "none", "record": False}}},
    }
    ops = [OpSpec(op="signal", on=Selector("neuron"),
                  params={"edge_set": "synapse", "tau": tau, "activation": "tanh", "bias": 0.0})]
    sim = Spec(name="neural_assemblies", seed=seed, n_frames=T, dt=dt, sets=sets, fields={},
               operators=ops, schedule=["signal"])

    H = build(sim, device="cpu")
    H.emit_order = _resolve_emit(sim)
    neuron = H.level("neuron")

    rng = np.random.default_rng(seed + 1)                      # random initial voltages break the v=0 fixed point
    neuron.state[:, 0] = torch.tensor(rng.normal(0.0, 0.5, N), dtype=torch.float32)

    op = get_operator("signal")({**ops[0].params, "_at": "neuron"}, "cpu")
    traces = np.empty((T, N), dtype=np.float32)
    for t in range(T):
        H.zero_delta()
        H.add_delta("neuron", op(H, None)["neuron"])
        _integrate(H, dt)
        traces[t] = neuron.get("voltage").squeeze(-1).numpy()
    return traces, assembly, dt


# --------------------------------------------------------------------------- #
#  connectome-cx-style stacked voltage traces
# --------------------------------------------------------------------------- #
def plot(traces, assembly, dt, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, N = traces.shape
    time = np.arange(T) * dt
    K = int(assembly.max()) + 1
    palette = ["#d1495b", "#2e86ab", "#1b9e77", "#e6a817", "#8e44ad"]    # categorical, one per assembly
    order = np.argsort(assembly)                                        # group neurons by assembly

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [2.1, 1]})

    # (a) stacked per-neuron voltage traces, coloured by assembly
    ax = axes[0]
    step = 2.2
    for row, i in enumerate(order):
        ax.plot(time, traces[:, i] + row * step, lw=0.6,
                color=palette[assembly[i] % len(palette)], alpha=0.9)
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel("time", fontsize=15, labelpad=2)
    ax.set_ylabel("neurons (grouped by assembly)", fontsize=15, labelpad=2)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.text(-0.02, 1.02, "a", transform=ax.transAxes, fontsize=20, fontweight="bold", va="bottom")

    # (b) assembly-mean activity -- the collective dynamics of each assembly
    ax = axes[1]
    for k in range(K):
        m = traces[:, assembly == k].mean(axis=1)
        ax.plot(time, m, lw=2.0, color=palette[k % len(palette)], label=f"assembly {k}")
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel("time", fontsize=15, labelpad=2)
    ax.set_ylabel(r"mean $v$", fontsize=15, labelpad=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    ax.text(-0.04, 1.02, "b", transform=ax.transAxes, fontsize=20, fontweight="bold", va="bottom")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[neural_assemblies] wrote {out_png}")


def main():
    traces, assembly, dt = simulate()
    print(f"[neural_assemblies] traces {traces.shape}  v range [{traces.min():.2f}, {traces.max():.2f}]")
    plot(traces, assembly, dt, os.path.join(HERE, "neural_assemblies.png"))


if __name__ == "__main__":
    main()
