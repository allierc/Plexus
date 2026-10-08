"""exp17: A RUN'S MOVIE IN THE ATLAS FRAME, FROM ABOVE AND FROM THE SIDE (Cedric, 2026-10-08: "the zebrafish are not
elongated as in the other slides; add the side view to the two main panels"). Local, no training.

The trainer's own run movie (trace_recording.render_movie: recorded left, learned right, the brain mean, the free
rollout's R2, the neuron-constant insets), re-rendered from results/<run>_movie.npz with every neuron at its ATLAS
position (head left, x' = atlas y, y' = 620 x 0.798 um - atlas x, z dorsal: the deck's other fish) and, under each
panel, the side view. The insets' clusters are the trainer's (KMeans(8) on each neuron's z-scored constants).

    PYTHONPATH=src:tools python tools/exp17_run_movie.py zap_n19_nom
-> presentation/Movies/<run>_atlas.mp4 (+ .png)
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")


def main(run, workers=24):
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    from exp17_flow_pruned import atlas_frame
    spec = T.load(run)
    out = T.out_dir(spec, None)
    rec = TR.load(spec["task"]["reference"]["trace_recording"])
    fit = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location="cpu")["fitted"]
    have = [b for b in ("tau", "rest", "gain", "input") if f"neuron.{b}" in fit]
    blocks = [torch.nn.functional.softplus(fit[f"neuron.{b}"].float()) if b == "tau" else fit[f"neuron.{b}"].float()
              for b in have]
    emb = torch.cat([((b - b.mean(0)) / b.std(0).clamp(min=1e-9)) / b.shape[1] ** 0.5 for b in blocks], 1).numpy()
    from sklearn.cluster import KMeans
    labels = KMeans(8, n_init=4, random_state=0).fit_predict(emb)
    src = "(" + ", ".join({"tau": "1/tau", "rest": "V", "gain": "g", "input": "B"}[b] for b in have) + ")"
    mv = np.load(os.path.join(out, "results", f"{run}_movie.npz"))
    fr = mv["frames"]
    P3 = atlas_frame(np.load(os.path.join(EXP, "data", "atlas_destripe.npz"))["atlas_um"].astype(np.float64))
    path = os.path.join(EXP, "presentation", "Movies", f"{run}_atlas.mp4")
    TR.render_movie(rec["dff"][fr], mv["pred"].astype(np.float32), fr, P3, mv["r2_raw"], mv["r2_denoised"],
                    T._frame_s(rec), rec["condition"][fr], rec["names"], path, emb=emb, labels=labels,
                    r2_t=mv["r2_t"], r2_raw_all=mv["r2_raw_all"], r2_den_all=mv["r2_denoised_all"],
                    silenced_all=mv["silenced_all"], mean_obs_all=mv["mean_obs_all"], mean_pred_all=mv["mean_pred_all"],
                    cond_all=mv["cond_all"], split_all=mv["split_all"],
                    split_name=spec["task"]["reference"].get("split", "all"), emb_name="neuron constants " + src,
                    inputs=(fit["neuron.input"].float().norm(dim=1).numpy() if "neuron.input" in fit else None),
                    law=T._law_name(spec), metric=spec["task"].get("movie_metric", "cells"), rec_name="destriped",
                    workers=workers, side=True, pos_is_view=True)
    print(f"[run movie] {path}")


if __name__ == "__main__":
    for n_ in sys.argv[1:]:
        main(n_)
