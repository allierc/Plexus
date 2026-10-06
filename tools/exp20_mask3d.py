"""exp20: THE INPUT MASKS IN 3-D, turning (Cedric, 2026-10-04: "a slide for the new mask, 3D view"): glucose fish 1's
cells a faint cloud, the cells each input enters coloured -- left the batch 1-5 mask (uv: |t| over every training
pulse, control included; grating and swim: coherence; tools/exp20_input_mask.py), right the batch-6 mask closer to
biology (uv: the cells EXCITED by the training gut pulses and not by the control pulses; grating as before; no swim,
the motor output), then the paper's own gut-responsive rule at 2 and 3 SD (exp20_input_mask.py --paper) -- all
from the same oblique camera turning once about the vertical, head left at the start.

    PYTHONPATH=src:tools python tools/exp20_mask3d.py
Writes presentation/Movies/mask3d_f1.mp4 (+ .png) and a copy in png/.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")
ZF = os.path.join(os.environ["GNN_OUTPUT_ROOT"], "graphs_data", "zebrafish")
COL = {"uv": "#ff4040", "visual": "#4fc3f7", "swim": "#81c784"}


def main(rec="gutbrain_glucose_f1", n_frames=200, fps=25):
    import pyvista as pv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.tasks import trace_recording as TR
    pv.OFF_SCREEN = True
    P = np.asarray(TR.load(rec)["pos_view"], np.float64)                  # head left (x mirrored, drawing only)
    Q = (P - (P.max(0) + P.min(0)) / 2).astype(np.float32)
    ld = lambda t: np.load(os.path.join(ZF, f"input_mask_{rec}{t}.npz"))["mask_by_input"]     # noqa: E731
    sets = [("batches 1-5", ld(""), (("uv", 0), ("visual", 3), ("swim", 4))),           # the four UV rules (batch 6)
            ("batch 6, 10 %", ld("_bio"), (("uv", 0), ("visual", 3))),
            ("batch 6, the paper's rule, 2 SD", ld("_paper2sd"), (("uv", 0), ("visual", 3))),
            ("batch 6, the paper's rule, 3 SD", ld("_paper3sd"), (("uv", 0), ("visual", 3))),
            ("batch 6, atlas: area postrema + vagal ganglia", ld("_anat_apvg"), (("uv", 0), ("visual", 3))),
            ("batch 6, atlas: area postrema + vagal ganglia + dorsal vagal complex", ld("_anat_dvc"), (("uv", 0), ("visual", 3)))]
    rr = 1.75 * float(np.ptp(P, 0).max())
    az0, el = np.deg2rad(-60.0), np.deg2rad(32.0)
    tmp = tempfile.mkdtemp(prefix="mask3d_")
    pls = []
    for title, by, ins in sets:
        pl = pv.Plotter(off_screen=True, window_size=(900, 600))
        pl.set_background("black")
        pl.add_mesh(pv.PolyData(Q[::4]), color="#6a6a6a", point_size=1.2, opacity=0.18)
        for k, j in ins:
            m = by[:, j] > 0
            pl.add_mesh(pv.PolyData(Q[m]), color=COL[k], point_size=2.0, opacity=0.75)
        pl.camera.focal_point = (0.0, 0.0, 0.0)
        pl.camera.up = (0.0, 0.0, 1.0)
        pls.append(pl)
    n = {t: {k: int((by[:, j] > 0).sum()) for k, j in ins} for t, by, ins in sets}
    union = {t: int((by.max(1) > 0).sum()) for t, by, _ in sets}
    frames = []
    for i in range(n_frames):
        az = az0 + 2 * np.pi * i / n_frames
        shots = []
        for pl in pls:
            pl.camera.position = (rr * np.cos(el) * np.cos(az), rr * np.cos(el) * np.sin(az), rr * np.sin(el))
            pl.render()
            shots.append(pl.screenshot(return_img=True))
        fig = plt.figure(figsize=(18, 9.2), facecolor="black")
        for j, ((title, by, ins), im) in enumerate(zip(sets, shots)):
            x0, y0 = (j % 3) / 3, 0.5 * (1 - j // 3)
            ax = fig.add_axes([x0, y0 + 0.04, 1 / 3, 0.40])
            ax.imshow(im)
            ax.axis("off")
            fig.text(x0 + 0.01, y0 + 0.48, f"{'abcdef'[j]}   {title}", color="white", fontsize=11, va="top")
            fig.text(x0 + 0.01, y0 + 0.03, "   ".join(f"{k} {n[title][k]:,}" for k, _ in ins), color="0.8", fontsize=9)
            for q, (k, _) in enumerate(ins):
                fig.text(x0 + 0.20 + 0.04 * q, y0 + 0.03, k, color=COL[k], fontsize=10, weight="bold")
        fig.text(0.5, 0.005, "red: the gut-input cells (the law's UV-pulse and beam inputs); blue: the grating's; green: the swim's "
                 "(batches 1-5 only)", color="0.7", fontsize=9, ha="center")
        f_ = os.path.join(tmp, f"g{i:05d}.png")
        fig.savefig(f_, dpi=100, facecolor="black")
        plt.close(fig)
        frames.append(f_)
    for pl in pls:
        pl.close()
    out = os.path.join(EXP, "presentation", "Movies", "mask3d_f1.mp4")
    subprocess.run([TR._ffmpeg(), "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "g%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", out], check=True)
    shutil.copy(frames[0], out.replace(".mp4", ".png"))
    for f_ in (out, out.replace(".mp4", ".png")):
        shutil.copy(f_, os.path.join(EXP, "png", os.path.basename(f_)))
    shutil.rmtree(tmp)
    print("[mask3d]", out, n, union)
    return n, union


if __name__ == "__main__":
    main()
