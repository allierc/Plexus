import sys
sys.path.insert(0, "/workspace/Plexus/src")
import torch
from plexus.schema import Spec
from plexus.engine import build

sim = Spec.from_yaml("/workspace/Plexus/prototype/embryogenesis/specs/embryo_1E_split_xrep.yaml")
H = build(sim, device="cpu")
# find the agent level
lvl = None
for L in H.levels:
    if getattr(L, "name", "") == "agent":
        lvl = L
        break
nt = lvl.node_type
occ = lvl.occ > 0
live_nt = nt[occ]
print("n_live:", int(occ.sum()))
print("type counts (live):", torch.bincount(live_nt).tolist())
x = lvl.state[occ, 0]
for tid in range(int(live_nt.max()) + 1):
    m = live_nt == tid
    print(f"  type {tid}: n={int(m.sum())}  mean_x={x[m].mean():.4f}  x_range=[{x[m].min():.3f},{x[m].max():.3f}]")
