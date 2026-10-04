"""exp20 LEAK CHECK (Cedric, 2026-10-03: "make sure there is no leak of neuron activity in the inference, that the
inference is only driven by stimuli"): the free rollout of a run re-made with every recorded frame after its start
replaced by zeros, and by noise; the predictions must not move beyond the run-to-run GPU rounding (the same rollout on
the true recording twice). Generalised from the exp17 session's /tmp/leak_test.py (devcontainer-14, 2026-10-03).

    PYTHONPATH=src python tools/exp20_leak_check.py <run> [--frames 300] [--device cuda:0]

Writes experiments/exp20_gutbrain_graphcast/data/leak_<run>.json: per checked frame the largest |difference| over
cells of (true vs true again), (true vs zeroed), (true vs noise), and LEAK = whether zeroed or noise departs from the
true rollout by more than 10x the true-vs-true gap (+ 1e-6 dF/F) at any frame. silence=None, as exp17's: the trainer's
runaway freeze reads the recording (its mean over all frames), a separate, known leak (md, Cedric's review item 1).
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
OUT = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast", "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    from plexus import engine
    from plexus import trainer as T
    engine.quiet(True)
    spec = T.load(a.run)
    out = T.out_dir(spec, None)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=a.device)
    learn = T.Learnables(spec["learnable"], a.device)
    learn.restore(ck["fitted"])
    box = T._trace_setup(spec, a.device)
    w = T._warmup(spec)
    o0, n = box["n_in"] - 1 + w, a.frames

    def run(X):
        b, preds = dict(box, X=X), []
        sim = T._model(spec, train=False, n_frames=w + n - 1)
        with torch.no_grad():
            T._trace_rollout_stream(sim, learn, spec, b, o0, n, a.device, lambda k, p: preds.append(p.clone()), w=w)
        return torch.stack(preds)
    torch.manual_seed(0)
    p_true = run(box["X"])
    Xz = box["X"].clone(); Xz[o0 + 1:] = 0.0
    Xr = box["X"].clone(); Xr[o0 + 1:] = torch.randn_like(Xr[o0 + 1:])
    p_zero, p_rand, p_again = run(Xz), run(Xr), run(box["X"])
    d = {k: (p_true - p).abs().amax(1).cpu() for k, p in (("self", p_again), ("zeroed", p_zero), ("noise", p_rand))}
    ks = sorted({0, 1, 2, 5, 10, 50, 100, n - 1} & set(range(n)))
    rows = [{"frame": k + 1, **{f"max_abs_{m}": float(d[m][k]) for m in d}} for k in ks]
    tol = 10 * d["self"] + 1e-6
    leak = bool(((d["zeroed"] > tol) | (d["noise"] > tol)).any())
    res = {"run": spec["name"], "seed_frames": int(box["n_in"]), "warmup": int(w), "first_origin": int(o0), "frames": n,
           "drive": spec["task"].get("drive"), "leak": leak, "rows": rows}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"leak_{spec['name']}.json"), "w"), indent=1)
    print(f"[leak] {spec['name']}: seed frames {box['n_in']}, warm-up {w}, {n} free frames -- LEAK {leak}")
    for r in rows:
        print(f"  frame {r['frame']:4d}: |true - true again| {r['max_abs_self']:.2e}   |true - zeroed| "
              f"{r['max_abs_zeroed']:.2e}   |true - noise| {r['max_abs_noise']:.2e}")


if __name__ == "__main__":
    main()
