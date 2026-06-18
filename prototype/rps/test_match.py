"""Numerically verify this prototype's reaction_diffusion delta equals ParticleGraph's
RD_RPS.forward to float precision on the same state.

RD_RPS message-passes a discrete graph Laplacian (message = L_ij * u_j, add-agg). On a
periodic grid with the 5-point stencil (weight +1 to each of 4 neighbours, -4 self),
that graph Laplacian IS our `_lap`. Build that exact grid graph, run RD_RPS.forward,
and compare to our single Euler step's vector field D*lap + react.

    PYTHONPATH=/workspace/ParticleGraph/src python test_match.py
"""
import os
import sys
import types
import importlib.util
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rps_engine

# load RD_RPS.py in isolation (stub its only ParticleGraph helper to skip the heavy __init__)
pg = types.ModuleType("ParticleGraph"); pu = types.ModuleType("ParticleGraph.utils")
pu.to_numpy = lambda x: x.detach().cpu().numpy(); pg.utils = pu
sys.modules.setdefault("ParticleGraph", pg); sys.modules.setdefault("ParticleGraph.utils", pu)
RD_PATH = "/workspace/ParticleGraph/src/ParticleGraph/generators/RD_RPS.py"
spec = importlib.util.spec_from_file_location("RD_RPS_mod", RD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules["RD_RPS_mod"] = mod                         # pyg's inspector reads sys.modules
spec.loader.exec_module(mod)
RD_RPS = mod.RD_RPS


def grid_graph(n, device):
    """edges (source->target): 4 periodic neighbours (+1) and a self-loop (-4) per node,
    so the add-aggregated message is the 5-point Laplacian."""
    idx = lambda i, j: (i % n) * n + (j % n)
    src, dst, w = [], [], []
    for i in range(n):
        for j in range(n):
            c = idx(i, j)
            for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                src.append(idx(i + di, j + dj)); dst.append(c); w.append(1.0)
            src.append(c); dst.append(c); w.append(-4.0)
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    return edge_index, torch.tensor(w, dtype=torch.float32, device=device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n, S, D, a = 24, 3, 0.37, 0.6
    torch.manual_seed(0)
    uvw = torch.rand(n * n, S, device=device)                       # node-major (i*n + j)
    ei, ea = grid_graph(n, device)

    # --- reference: ParticleGraph RD_RPS.forward ---
    class Data:  # minimal duck-typed pyg.data.Data
        pass
    d = Data(); d.x = torch.zeros(n * n, 9, device=device); d.x[:, 6:9] = uvw
    d.edge_index = ei; d.edge_attr = ea
    ref = RD_RPS(aggr_type="add", coeff=D).forward(d)               # [N,3], a=0.6 inside

    # --- ours: D*lap + react on the same field ---
    g = uvw.t().reshape(S, n, n).contiguous()                       # [S,i,j]
    p = g.sum(0, keepdim=True); nxt = torch.roll(g, -1, 0)
    ours = D * rps_engine._lap(g, "periodic") + g * (1.0 - p - a * nxt)
    ours = ours.reshape(S, n * n).t()                              # back to [N,3]

    diff = (ref - ours).abs().max().item()
    rel = diff / (ref.abs().max().item() + 1e-12)
    print(f"max|RD_RPS - prototype| = {diff:.3e}   (relative {rel:.3e})")
    print("MATCH ✓" if diff < 1e-4 else "MISMATCH ✗")


if __name__ == "__main__":
    main()
