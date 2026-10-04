#!/usr/bin/env python
"""A movie of the message passing between touching cells in a coupled cardiac-sheet fit.

    PYTHONPATH=src python tools/cardio_message_movie.py healthy_coupled --device cuda:0
    -> prototype/cardio_mpm/strain/out/movies/<name>_message.mp4

The fit is `active_strain[model: coupled]`: each frame, cell j's activation takes the message
summed over the cells whose outlines touch it,

    a_j(t)   = clip( gamma_j(t) + msg_j(t), 0, 1.5 )
    msg_j(t) = sum_{i in N(j)} W2 tanh( W1 [a_i(t-1), a_j(t-1)] )

The trained model is rolled out again, beat by beat, exactly as the trainer scores it (edge band
prescribed from the recording). Each frame records every cell's activation a_j, and msg_j is
recomputed from the previous frame's activations with the fit's own W1, W2 over the same contact
graph (`cell_neighbours[model: label_image]`). Each value is painted onto that cell's pixels of
the segmentation:

    left    a_j(t), the activation the contraction follows (0 = rest)
    right   msg_j(t), what its neighbours add (red) or take away (blue); the colour scale is
            symmetric about zero and its limit is printed, because a message can be small
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="a training spec whose model uses active_strain[model: coupled]")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4, help="pixels of the label image per movie pixel")
    ap.add_argument("--out", default=os.path.join(ROOT, "prototype", "cardio_mpm", "strain", "out", "movies"))
    a = ap.parse_args()

    import imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tifffile
    from plexus import engine
    from plexus import trainer as T
    from plexus.paths import graphs_data_path

    engine.quiet(True)
    spec = T.load(a.name)
    ck = torch.load(os.path.join(T.out_dir(spec), "models", "best.pt"), weights_only=False,
                    map_location=a.device)
    learn = T.Learnables(spec["learnable"], a.device)
    learn.restore(ck["fitted"])
    box = T._recording_setup(spec, a.device)
    sim0 = T._model(spec, train=False, n_frames=1)
    ast = next(o for o in sim0.operators if o.op == "active_strain")
    lbl_src = next(f for n, f in sim0.fields.items() if f.get("frame") == "label_image")["source"]
    lab = tifffile.imread(graphs_data_path(lbl_src)).astype(np.int64)[::a.stride, ::a.stride]

    # THE FIT'S OWN WEIGHTS: the trained values when W1/W2 were learnable, else the operator's start.
    W1 = ck["fitted"].get("active_strain.W1")
    W2 = ck["fitted"].get("active_strain.W2")
    if W1 is None or W2 is None:
        from plexus.models.registry import get_operator
        inst = get_operator("active_strain", model="coupled")({**ast.params, "_at": ast.on.set}, "cpu")
        W1 = W1 if W1 is not None else inst.W1
        W2 = W2 if W2 is not None else inst.W2
    W1, W2 = W1.detach().cpu().double(), W2.detach().cpu().double()

    acts, msgs, tags = [], [], []
    for bi, w in enumerate(box["wins"]):
        sim = T._model(spec, train=False, n_frames=len(w["frames"]) - 1)
        rec, ei = [], {}

        def watch(H, tick):
            cell = H.level("cell")
            if "e" not in ei and getattr(cell, "edge_index", None) is not None:
                ei["e"] = cell.edge_index.detach().cpu()
            rec.append(cell.get("gam_prev")[:, 0].detach().cpu().double().numpy().copy())

        with torch.no_grad():
            T._recording_rollout(sim, learn, spec, box, bi, a.device, False, watch=watch)
        A = np.stack(rec)                                            # [T, C] activation a_j(t)
        src, dst = ei["e"][0].numpy(), ei["e"][1].numpy()
        M = np.zeros_like(A)
        for t in range(1, A.shape[0]):                               # msg(t) from a(t - 1)
            x = torch.tensor(np.stack([A[t - 1, src], A[t - 1, dst]], -1))
            m = (torch.tanh(x @ W1.T) @ W2.T)[:, 0].numpy()
            np.add.at(M[t], dst, m)
        acts.append(A); msgs.append(M); tags += [(w["k"], t) for t in range(A.shape[0])]
        print(f"beat {w['k']}: {A.shape[0]} frames, activation max {A.max():.3f}, "
              f"|message| max {np.abs(M).max():.2e}", flush=True)
    A, M = np.concatenate(acts), np.concatenate(msgs)
    a_hi = max(float(A.max()), 1e-6)
    m_hi = max(float(np.abs(M).max()), 1e-12)

    def paint(v, cmap, lo, hi):
        rgb = np.zeros(lab.shape + (3,))
        on = lab > 0
        rgb[on] = matplotlib.colormaps[cmap](np.clip((v[lab[on] - 1] - lo) / (hi - lo), 0, 1))[:, :3]
        return rgb

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.name}_message.mp4")
    wr = imageio.get_writer(path, fps=a.fps, codec="libx264", quality=8, macro_block_size=1)
    fig, axs = plt.subplots(1, 2, figsize=(12.8, 6.6), facecolor="black")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.08, wspace=0.03)
    for i, (k, t) in enumerate(tags):
        for ax in axs:
            ax.clear(); ax.axis("off")
        axs[0].imshow(paint(A[i], "magma", 0.0, a_hi), interpolation="nearest")
        axs[1].imshow(paint(M[i], "RdBu_r", -m_hi, m_hi), interpolation="nearest")
        axs[0].text(0.01, 0.99, f"{a.name}   beat {k}   frame {t}\nactivation a_j (0 - {a_hi:.2f})",
                    transform=axs[0].transAxes, va="top", ha="left", color="white", fontsize=11)
        axs[1].text(0.01, 0.99, f"message from touching cells\nred adds, blue takes away   |msg| <= {m_hi:.1e}",
                    transform=axs[1].transAxes, va="top", ha="left", color="white", fontsize=11)
        fig.canvas.draw()
        wr.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
    wr.close()
    plt.close(fig)
    print(f"-> {path}  ({len(tags)} frames at {a.fps} fps; activation max {a_hi:.3f}, |message| max {m_hi:.2e})")


if __name__ == "__main__":
    main()
