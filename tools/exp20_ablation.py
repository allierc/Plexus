"""exp20: exp17's network ablation on an exp20 run -- the free rollout of the whole session with every edge weight W = 0
and with the left half's stimulus weights B = 0, beside the full law (Cedric, 2026-10-03: "show network dynamics, not
only a function of the stimulus"; exp17 slides 26-27).

    PYTHONPATH=src:tools python tools/exp20_ablation.py <run> [<run> ...] [--device cuda:0] [--no-movie]

`tools/exp17_ablation.py` does the work unchanged (it reads the run's own spec, recording and edge sets); this wrapper
only points its outputs at exp20's folder (data/ablation_<run>.json, data/figs/ablation_<run>.png) and its time axis at
the recording's own frame period (1.117 s for glucose fish 1, not ZAPBench's 0.914 s). Another experiment's tool is
called, never edited (INSTRUCTION.md).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")

import exp17_ablation as A  # noqa: E402

A.EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")


def _render(spec, out, stem, path, law, rec):
    """exp17's `render` with the recording named for what it is (exp17's labels it ZAPBench / destriped)."""
    import time
    from plexus.tasks import trace_recording as TR
    t1 = time.time()
    mv = np.load(os.path.join(out, "results", f"{stem}_movie.npz"))
    frames = mv["frames"]
    TR.render_movie(rec["dff"][frames], mv["pred"].astype(np.float32), frames, rec.get("pos_view", rec["pos_um"]),
                    mv["r2_raw"], mv["r2_denoised"], A.FRAME_S, rec["condition"][frames], rec["names"], path,
                    emb=None, labels=None, r2_t=mv["r2_t"], r2_raw_all=mv["r2_raw_all"],
                    r2_den_all=mv["r2_denoised_all"], silenced_all=mv["silenced_all"],
                    mean_obs_all=mv["mean_obs_all"], mean_pred_all=mv["mean_pred_all"], cond_all=mv["cond_all"],
                    split_all=mv["split_all"], split_name=spec["task"]["reference"].get("split", "all"), law=law,
                    rec_name=str(spec["task"]["reference"].get("trace_recording")).replace("gutbrain_", "gut-brain "))
    print(f"[ablation] {path} ({len(frames)} frames, {time.time() - t1:.0f} s)", flush=True)


A.render = _render


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    names = [a for a in argv if not a.startswith("--")]
    if names:
        from plexus import trainer as T
        from plexus.tasks import trace_recording as TR
        rec = TR.load(T.load(names[0])["task"]["reference"]["trace_recording"])
        A.FRAME_S = float(np.median(np.diff(rec["t_s"])))
    return A.main(argv)


if __name__ == "__main__":
    main()
