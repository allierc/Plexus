"""ucb_tree.py -- HOO: a UCB tree search over the continuous parameter cube.

HOO (Hierarchical Optimistic Optimization, Bubeck et al.) is the canonical UCB
bandit *tree* for black-box continuous optimization. Each node owns a box of the
unit cube; an optimistic bound B (empirical reward + a UCB exploration term + a
depth-decaying diameter term) guides traversal to the most promising leaf, which is
sampled, then split. It needs no gradients and no surrogate model -- the sound,
model-free baseline. Because our simulator is deterministic given a fixed seed, the
objective is noise-free, so HOO here behaves as optimistic tree search (the UCB term
just adds exploration); this is exactly the parameter half of the paper's mechanism
search tree.

`minimize(f, D, n_evals)` minimizes f: [0,1]^D -> R. Returns (best_u, best_val,
history) where history[i] is best-so-far distance after i+1 evaluations.
"""
from __future__ import annotations

import math
import numpy as np


def _node(lo, hi, depth):
    return {"lo": lo, "hi": hi, "depth": depth, "T": 0, "mu": 0.0,
            "children": None, "U": math.inf, "B": math.inf}


def minimize(f, D, n_evals, rho=0.66, nu=1.0, lo=None, hi=None, on_eval=None):
    lo = np.zeros(D) if lo is None else np.asarray(lo, float)
    hi = np.ones(D) if hi is None else np.asarray(hi, float)
    root = _node(lo, hi, 0)
    nodes = [root]
    best_val, best_u, hist = math.inf, None, []

    for t in range(n_evals):
        # 1. traverse to a leaf by max B-value
        node, path = root, []
        while True:
            path.append(node)
            if node["children"] is None:
                break
            c0, c1 = node["children"]
            node = c0 if c0["B"] >= c1["B"] else c1

        # 2. evaluate the leaf's box center (deterministic objective)
        x = 0.5 * (node["lo"] + node["hi"])
        y = f(x)
        r = -y                                   # reward = -distance (maximize)
        if y < best_val:
            best_val, best_u = y, x.copy()
        hist.append(best_val)
        if on_eval is not None:
            on_eval(t, best_val, best_u)

        # 3. update running mean/count along the path
        for nd in path:
            nd["T"] += 1
            nd["mu"] += (r - nd["mu"]) / nd["T"]

        # 4. split the leaf along its widest dimension
        d = int(np.argmax(node["hi"] - node["lo"]))
        mid = 0.5 * (node["lo"][d] + node["hi"][d])
        h1 = node["hi"].copy(); h1[d] = mid
        l2 = node["lo"].copy(); l2[d] = mid
        node["children"] = (_node(node["lo"].copy(), h1, node["depth"] + 1),
                            _node(l2, node["hi"].copy(), node["depth"] + 1))
        nodes.extend(node["children"])

        # 5. recompute optimistic bounds U, then back up B (small trees -> cheap)
        N = t + 1
        for nd in nodes:
            nd["U"] = (math.inf if nd["T"] == 0 else
                       nd["mu"] + math.sqrt(2 * math.log(N) / nd["T"]) + nu * rho ** nd["depth"])
        for nd in sorted(nodes, key=lambda z: -z["depth"]):
            if nd["children"] is None:
                nd["B"] = nd["U"]
            else:
                c0, c1 = nd["children"]
                nd["B"] = min(nd["U"], max(c0["B"], c1["B"]))

    return best_u, best_val, hist
