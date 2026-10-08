"""exp17: THE LEARNED MODEL'S TRACES, EVERY FRAME (Cedric, 2026-10-08: "twins of the ARTR and the antiphase slides with the
learned 19.25 model"). Local, a landed run's models/best.pt, no training.

The test's own free rollout of the whole recording (trainer._trace_rollout_stream from the law's first origin, the
runaway elements silenced as the test does), every predicted frame kept -- the movie npz keeps 800 of the 7,879, too
coarse for a 60-s rotation cycle. The frames before the first origin are the recorded ones (the law starts from them).

    PYTHONPATH=src:tools python tools/exp17_model_traces.py zap_n19_nom
-> data/model_traces_<run>.npy, float16 [T, N] dF/F (read with np.load(..., mmap_mode="r"))
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")


def path(run):
    return os.path.join(EXP, "data", f"model_traces_{run}.npy")


def main(run):
    import torch
    from plexus import trainer as T
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    spec = T.load(run)
    ck = torch.load(os.path.join(T.out_dir(spec, None), "models", "best.pt"), weights_only=False, map_location=dev)
    learn = T.Learnables(spec["learnable"], dev)
    learn.restore(ck["fitted"])
    box = T._trace_setup(spec, dev)
    X, Tn = box["X"], box["T"]
    w = T._warmup(spec)
    o0 = box["n_in"] - 1 + w
    n = Tn - 1 - o0
    sim = T._model(spec, train=False, n_frames=w + n - 1)
    out = np.lib.format.open_memmap(path(run), mode="w+", dtype=np.float16, shape=(Tn, X.shape[1]))
    out[:o0 + 1] = X[:o0 + 1].cpu().numpy().astype(np.float16)
    dead = torch.zeros(X.shape[1], dtype=torch.bool, device=X.device)
    fill = X.mean(0)
    lo, hi = T.SILENCE_RANGE

    def silence(blk):                                      # the test's rule: a runaway element frozen at its mean
        v = blk.reshape(-1, blk.shape[-1])
        bad = ~torch.isfinite(v).all(1) | (v < lo).any(1) | (v > hi).any(1)
        dead.logical_or_(bad)
        if not bool(dead.any()):
            return None
        v = v.clone()
        v[dead] = fill[dead][:, None].to(v.dtype)
        return v.reshape(blk.shape)

    def on_pred(k, p):
        out[o0 + k + 1] = p.reshape(-1).float().cpu().numpy().astype(np.float16)
    with torch.no_grad():
        T._trace_rollout_stream(sim, learn, spec, box, o0, n, dev, on_pred, silence=silence, w=w)
    out.flush()
    m = np.asarray(out[o0 + 1:], np.float32).mean(1)
    x = X[o0 + 1:].mean(1).cpu().numpy()
    print(f"[model] {path(run)}: {Tn} x {X.shape[1]}, brain-mean r {np.corrcoef(m, x)[0, 1]:.3f} (the test's: as "
          f"results/{run}_test.json), silenced {int(dead.sum())}")


if __name__ == "__main__":
    for n_ in sys.argv[1:]:
        main(n_)
