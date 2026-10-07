"""exp17: A MESH RUN'S LEVELS CUT ONE AT A TIME (Cedric, 2026-10-05: "mesh-3 ablation rollouts per level"). Local.

A multi-level mesh run (graph: mesh) learns one W per edge in three sets, but its levels share them: short = the finest
level (every neuron), mid = the 16- and 32-um levels together, long = 64 um and coarser. `task.rollouts` can zero a whole
set; this tool zeroes ONE LEVEL's edges inside its set -- the trained model, the W of every edge of that level set to 0
at inference (an edge that two levels share counts as both levels'), not retrained -- and runs the free rollout as the
trainer's variants (trainer._trace_free), writing results/<run>_no_M<k>_movie.npz, the test json's `rollouts` entry and
the comparison movie (exp17_ablation.render_compare): the full model left, the cut right.

    PYTHONPATH=src:tools python tools/exp17_mesh_level_ablation.py zap_e15_cur_siren_mesh3 [levels ...]
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]


def main(name, levels=None, device="cuda:0"):
    from plexus import trainer as T
    from plexus.operators.cell_ops import neuron_mesh_levels
    from exp17_ablation import render_compare
    T.engine.quiet(True)
    spec = T.load(name)
    out = T.out_dir(spec, None)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    learn = T.Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    box = T._trace_setup(spec, device)
    from exp17_ablation import neuron_graph_op
    g = neuron_graph_op(spec, "cpu")                                   # the run's own operator: its graph and options
    L, b = g.mesh_levels, g.mesh_bin
    pos = np.load(g._pos_file[0])[g._pos_file[1]].astype(np.float64)
    lv = neuron_mesh_levels(pos, L, b)
    N = len(pos)
    code = lambda e: np.minimum(e[:, 0], e[:, 1]).astype(np.int64) * N + np.maximum(e[:, 0], e[:, 1])   # noqa: E731
    tj = os.path.join(out, "results", f"{name}_test.json")
    J = json.load(open(tj))
    for k, (lab, cube, nodes, e) in enumerate(lv):
        if levels and k not in levels:
            continue
        s_ = "short" if lab == "every neuron" else ("mid" if cube <= 2 * b else "long")
        snd, rcv = (t.cpu().numpy() for t in g._E[s_])
        hit = np.isin(code(np.stack([snd, rcv], 1)), code(e))           # the set's directed edges on this level
        key = T.Learnables.key(next(x for x in spec["learnable"] if x.get("param") == f"W_{s_}"))
        nm = f"no_M{k}"
        saved = learn.p[key].detach().clone()
        with torch.no_grad():
            learn.p[key][torch.as_tensor(hit, device=learn.p[key].device)] = 0.0
            fv = T._trace_free(spec, learn, box, device, out, f"{name}_{nm}")[0]
            learn.p[key].copy_(saved)
        what = f"W_{s_} on M{k}'s {int(hit.sum()):,} edges"          # short: it is a table cell of the slide
        J.setdefault("rollouts", {})[nm] = {**fv, "spec": {"name": nm, "zero": [what]}}
        json.dump(J, open(tj, "w"))
        print(f"[levels] {name} {nm}: {what} = 0 ({lab}, of the {len(hit):,} {s_} edges); free R2 den {fv['r2_denoised']:+.3f}", flush=True)
        render_compare(name, nm)


if __name__ == "__main__":
    main(sys.argv[1], [int(a) for a in sys.argv[2:]] or None)
