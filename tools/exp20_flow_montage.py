"""exp20: THE FLOW MOVIES SIDE BY SIDE, exp17's slides 28-29 (tools/exp17_flow_montage.py) on batch 4's fish.

  fish     the wind movies (tools/exp20_wind.py) of 8 fish in exp17's graph-slide layout (4 x 2), each cell its
           excitatory map above its inhibitory map, the fish's name above; every movie has the same number of frames
           over its own session (exp17_wind samples the run's movie frames), so the cells run side by side in frame,
           not in minutes
  gut      exp17's slide 29 has a consensus over graphs on ONE brain; the fish's brains differ, so here instead, per
           fish, the excitatory flow speed during the gut windows (0-20 s after a gut pulse) minus the rest
           (data/gutflow_<run>.npz), the paper's stations in yellow (tools/exp20_landmarks.py)

    PYTHONPATH=src:tools python tools/exp20_flow_montage.py
Writes presentation/Movies/flow_fish.mp4 (+ .png still), presentation/figs/flow_gut_fish.png, copies in png/.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")
G = os.path.join(os.environ["GNN_OUTPUT_ROOT"], "log", "training", "gutbrain")
FISH = [("D-glucose fish 1", "gb_sx_f1_mask_siren"), ("D-glucose fish 2", "gb_sx_f2_mask_siren"),
        ("D-glucose fish 3", "gb_sx_f3_mask_siren"), ("D-glucose fish 4", "gb_sx_f4_mask_siren"),
        ("glutamate fish 1", "gb_b4_glutamate_f1"), ("L-glucose fish 1 (control)", "gb_b4_Lglucose_f1"),
        ("fish water fish 2 (control)", "gb_b4_fish_water_f2"), ("blood glucose fish 2 (vessel)", "gb_b4_blood_glucose_f2")]
EX, IN = "784:384:8:64", "784:384:808:64"            # the two maps inside a 1600 x 640 wind frame (exp20_wind's layout)


def _run(cmd, out):
    from plexus.tasks import trace_recording as TR
    subprocess.run([TR._ffmpeg(), "-y", "-loglevel", "error"] + cmd + ["-pix_fmt", "yuv420p", "-c:v", "libx264", out],
                   check=True)
    still = out.replace(".mp4", ".png")
    subprocess.run([TR._ffmpeg(), "-y", "-loglevel", "error", "-ss", "16", "-i", out, "-frames:v", "1", still], check=True)
    for f in (out, still):
        shutil.copy(f, os.path.join(EXP, "png", os.path.basename(f)))
    print("[montage]", out)


def fish():
    from exp17_flow_montage import _labels
    ok = [(t, n) for t, n in FISH if os.path.exists(os.path.join(G, n, "results", "movie_wind.mp4"))]
    if len(ok) < len(FISH):
        print("[montage] missing wind movies:", [n for t, n in FISH if (t, n) not in ok])
    ins, f = [], []
    for i, (title, n) in enumerate(ok):
        ins += ["-i", os.path.join(G, n, "results", "movie_wind.mp4")]
        f.append(f"[{i}:v]split[a{i}][b{i}];[a{i}]crop={EX},scale=392:192[e{i}];[b{i}]crop={IN},scale=392:192[n{i}];"
                 f"[e{i}][n{i}]vstack,pad=392:420:0:36:black[c{i}]")
    k = len(ok)
    lay = "|".join(f"{(i % 4) * 392}_{(i // 4) * 420}" for i in range(k))
    rows = (k + 3) // 4
    lab = _labels((1568, 420 * rows), [(((i % 4) * 392 + 8, (i // 4) * 420 + 8), t) for i, (t, _) in enumerate(ok)], 18)
    ins += ["-loop", "1", "-i", lab]
    fc = (";".join(f) + ";" + "".join(f"[c{i}]" for i in range(k))
          + f"xstack=inputs={k}:layout={lay}:fill=black[g];[g][{k}:v]overlay=0:0:shortest=1[v]")
    _run(ins + ["-filter_complex", fc, "-map", "[v]", "-r", "25"], os.path.join(EXP, "presentation", "Movies", "flow_fish.mp4"))


def gut():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus import trainer as T
    from exp20_landmarks import landmarks, annotate
    fig, ax = plt.subplots(2, 4, figsize=(16, 5.6), facecolor="black")
    rows = []
    for a_, (title, n) in zip(ax.ravel(), FISH):
        a_.set_facecolor("black"); a_.axis("off")
        p, j = os.path.join(EXP, "data", f"gutflow_{n}.npz"), os.path.join(EXP, "data", f"gutflow_{n}.json")
        if not (os.path.exists(p) and os.path.exists(j)):
            a_.set_title(f"{title}: not rendered", color="0.6", fontsize=9)
            continue
        z, J = np.load(p), json.load(open(j))
        d = z["ex"]
        lim = np.nanpercentile(np.abs(d), 99)
        a_.imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
        gx0, gy0, gh = z["grid"]
        rn = T.load(n)["task"]["reference"]["trace_recording"]
        try:
            annotate(a_, landmarks(rn), lambda x, y: ((x - gx0) / gh, (y - gy0) / gh), fontsize=7)
        except FileNotFoundError:
            pass
        a_.set_title(f"{title}: x{J['ex']['ratio']:.2f} in the gut windows", color="white", fontsize=9)
        rows.append({"fish": title, "run": n, "ratio_ex": J["ex"]["ratio"], "ratio_in": J["in"]["ratio"],
                     "toward_tail_gut": J["ex"]["toward_tail_gut"], "toward_tail_rest": J["ex"]["toward_tail_rest"],
                     "frames_gut": J["frames_gut"]})
    fig.text(0.5, 0.01, "excitatory flow speed, the gut windows (0-20 s after a gut pulse) minus the rest, each fish on its "
             "own scale (red faster in the gut windows); yellow: the paper's stations, inferred from the fish's "
             "gut-responsive cells (no atlas)", color="0.85", fontsize=9, ha="center")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(EXP, "presentation", "figs", "flow_gut_fish.png")
    fig.savefig(out, dpi=150, facecolor="black")
    plt.close(fig)
    shutil.copy(out, os.path.join(EXP, "png", "flow_gut_fish.png"))
    json.dump(rows, open(os.path.join(EXP, "data", "flow_gut_fish.json"), "w"), indent=1)
    print("[montage]", out)
    return rows


if __name__ == "__main__":
    what = sys.argv[1:] or ["fish", "gut"]
    if "fish" in what:
        fish()
    if "gut" in what:
        gut()
