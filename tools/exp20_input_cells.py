"""exp20: THE GUT- AND VISUAL-INPUT CELLS OF THE BATCH-6 ARMS, counted (Cedric, 2026-10-05: "number of gut input cells
before / after the control threshold, number of visual input cells with the 10 % rule and the 3-sigma rule, the
intersection of gut and visual input cells"). Glucose fish 1, training frames only, as the masks were made.

    PYTHONPATH=src:tools python tools/exp20_input_cells.py [--device cuda:0]

Chen 2026's abstract: "many gut-responsive neurons also responded to swimming and visual stimuli, with brainstem areas
primarily integrating gut and motor signals and midbrain regions integrating jointly gut, visual and motor signals" --
the intersection asks how many of each rule's gut-input cells the grating also drives.

  gut, per UV rule (tools/exp20_input_mask.py):
    batches 1-5   the top 10 % by |t| after every training pulse, the control included: no control threshold
    10 %          before: the top 10 % by t after the training GUT pulses; after: minus the cells the CONTROL pulses
                  also excite (t_ctrl >= 2)
    paper K SD    before: the gut + all-UV regression's r above mode + K sd and above the all-UV one's; after: and the
                  cell's mean change after the gut pulses above that after the control pulses (the paper's selection)
    atlas         the registered regions' cells; no control threshold
  visual:
    10 %          the top 10 % by coherence with the grating speed (every arm's grating cells)
    paper 3 SD    the paper's rule on the grating: each cell's smoothness-regularised regression on the grating speed
                  (the paper's 5-s visual kernel, lambda_R 2, lambda_F 20, tools/gutbrain_baselines.py), its r on the
                  training frames above mode + 3 sd of the left half-Gaussian (fitLeftGetThresh)
Writes data/input_cells_f1.json and data/visual_paper3sd_gutbrain_glucose_f1.npz (mask, r, threshold).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
DATA = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast", "data")
REC = "gutbrain_glucose_f1"
RULES = [("batches 1-5", "", "gb_sx_f1_mask_siren", "gb_b6_glucose_f1_mesh4"),
         ("batch 6, 10 %", "_bio", "gb_b6_glucose_f1_bio", "gb_b6_glucose_f1_mesh4_bio"),
         ("batch 6, 2 SD", "_paper2sd", "gb_b6_glucose_f1_paper2", "gb_b6_glucose_f1_mesh4_paper2"),
         ("batch 6, 3 SD", "_paper3sd", "gb_b6_glucose_f1_paper3", "gb_b6_glucose_f1_mesh4_paper3"),
         ("atlas: area postrema + vagal ganglia", "_anat_apvg", "gb_b6_glucose_f1_anat_apvg", "gb_b6_glucose_f1_mesh4_anat_apvg"),
         ("atlas: area postrema + vagal ganglia + dorsal vagal complex", "_anat_dvc", "gb_b6_glucose_f1_anat_dvc", "gb_b6_glucose_f1_mesh4_anat_dvc")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    import gutbrain_baselines as GB
    import exp20_input_mask as IM
    from plexus.paths import graphs_data_path
    z = np.load(graphs_data_path("zebrafish", f"{REC}_recording.npz"))
    X = torch.as_tensor(z["dff"], device=a.device)
    S, tr, split = z["stimulus"], z["trials"], z["split"]
    T, N = X.shape
    dt = float(np.median(np.diff(z["t_s"])))
    pre, ev = int(round(IM.PRE_S / dt)), int(round(IM.EVOKED_S / dt))
    ld = lambda t: np.load(graphs_data_path("zebrafish", f"input_mask_{REC}{t}.npz"))["mask_by_input"]   # noqa: E731

    # the 10 % rule's two sets, as exp20_input_mask.py --bio makes them
    on = [int(f) for f, h in zip(tr[:, 0], tr[:, 6]) if not h and f - pre >= 0 and f + ev <= T
          and (split[int(f) - pre:int(f) + ev] == 0).all()]
    sites = {int(f): int(s_) for f, s_ in zip(tr[:, 0], tr[:, 2])}

    def t_of(sel):
        dd = torch.stack([X[f:f + ev].mean(0) - X[f - pre:f].mean(0) for f in sel])
        return (dd.mean(0) / (dd.std(0) / np.sqrt(len(sel))).clamp(min=1e-9)).cpu().numpy()
    t_gut, t_ctl = t_of([f for f in on if sites[f] != 1]), t_of([f for f in on if sites[f] == 1])
    top_gut = t_gut > np.quantile(t_gut, 0.9)

    # the paper's rule, before and after its control comparison
    bz = np.load(os.path.join(DATA, f"baselines_{REC}_cells.npz"))
    bj = json.load(open(os.path.join(DATA, f"baselines_{REC}.json")))["threshold"]
    paper = {k: (bz["r_full"] > bj["mode"] + k * bj["left_sd"]) & (bz["r_full"] > bz["r_part"]) for k in (2, 3)}
    gut_ok = bz["evoked_gut"] > bz["evoked_ctrl"]

    # the visual cells: the 10 % rule (every arm's grating mask) and the paper's rule on the grating
    vis10 = ld("_bio")[:, 3] > 0
    fit = split == 0
    K = int(round(GB.KERNEL_S["visual"] / dt))
    pred, _ = GB.fused_ridge(X, [S[:, 3].astype(float)], [K], fit, a.device)
    rows = torch.as_tensor(np.where(fit)[0], device=a.device)
    r_vis = GB.corr_cols(pred, X, rows).cpu().numpy()
    del pred
    thr, mode, sd = GB.left_gauss_thresh(r_vis, 3.0)
    vis3 = r_vis > thr
    np.savez(os.path.join(DATA, f"visual_paper3sd_{REC}.npz"), mask=vis3, r=r_vis, threshold=thr, mode=mode, left_sd=sd)

    before_after = {"": (ld("")[:, 0] > 0, None), "_bio": (top_gut, ld("_bio")[:, 0] > 0),
                    "_paper2sd": (paper[2], ld("_paper2sd")[:, 0] > 0), "_paper3sd": (paper[3], ld("_paper3sd")[:, 0] > 0),
                    "_anat_apvg": (ld("_anat_apvg")[:, 0] > 0, None), "_anat_dvc": (ld("_anat_dvc")[:, 0] > 0, None)}
    assert (before_after["_paper3sd"][1] == (paper[3] & gut_ok)).all() and (before_after["_bio"][1] == (top_gut & (t_ctl < 2))).all()
    out = {"cells": int(N), "visual": {"10 %": int(vis10.sum()), "paper 3 SD": int(vis3.sum()),
                                       "paper 3 SD threshold": {"r": float(thr), "mode": float(mode), "left_sd": float(sd)},
                                       "both": int((vis10 & vis3).sum())}, "rules": []}
    for (lab, tag, ng, m4), (b4, af) in zip(RULES, before_after.values()):
        g = af if af is not None else b4                          # the set the arms used
        out["rules"].append({"rule": lab, "tag": tag, "neuron_graph": ng, "mesh4": m4, "gut_before": int(b4.sum()),
                             "gut_after": int(af.sum()) if af is not None else None, "gut_used": int(g.sum()),
                             "and_visual_10": int((g & vis10).sum()), "and_visual_3sd": int((g & vis3).sum())})
    json.dump(out, open(os.path.join(DATA, "input_cells_f1.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
