"""exp17: the flow movies side by side (Cedric, 2026-10-03). Local; ffmpeg only, from the movies already made.

  graphs   the wind movies (tools/exp17_wind.py) of batch 17's 8 graphs in the graph slide's layout (4 x 2), each
           cell its excitatory map above its inhibitory map, the graph's name above
  summary  the consensus of the 5 graphs Cedric kept (tools/exp17_wind_consensus.py --tag g17sel): the MEAN flow and
           the MEDIAN flow (each excitatory | inhibitory), and below them the RECORDED dF/F (the base run's movie, left
           panel) -- the same 800 frames over the 2 h in every panel

    PYTHONPATH=src:tools python tools/exp17_flow_montage.py
Writes presentation/Movies/flow_graphs.mp4 / flow_summary.mp4 (+ .png stills) and copies in png/.
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
G = os.path.join(os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData"), "log", "training", "zapbench")
GRAPHS = [("base: axes, 32 / 128 um", "zap_e15_cur"), ("axes turned 45 deg", "zap_g17_rot45"),
          ("random directions", "zap_g17_randdir"), ("18 nearest only", "zap_g17_knn18"), ("no highways", "zap_g17_nolong"),
          ("reaches 16 / 64 um", "zap_g17_r16_64"), ("reaches 64 / 256 um", "zap_g17_r64_256"),
          ("random graph (the null)", "zap_g17_random")]
EX, IN = "784:384:8:64", "784:384:808:64"            # the two maps inside a 1600 x 640 wind frame (exp17_wind layout)
FONT = os.path.join(os.path.dirname(__import__("matplotlib").__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")


def _labels(size, items, fs):
    """A transparent PNG of the canvas `size` with white text at (x, y): this ffmpeg has no drawtext."""
    import tempfile
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype(FONT, fs)
    for (x, y), t in items:
        dr.text((x, y), t, fill=(255, 255, 255, 255), font=font)
    f = tempfile.mktemp(suffix=".png")
    im.save(f)
    return f


def run(cmd, out):
    from plexus.tasks import trace_recording as TR
    subprocess.run([TR._ffmpeg(), "-y", "-loglevel", "error"] + cmd + ["-pix_fmt", "yuv420p", "-c:v", "libx264", out],
                   check=True)
    still = out.replace(".mp4", ".png")
    subprocess.run([TR._ffmpeg(), "-y", "-loglevel", "error", "-ss", "16", "-i", out, "-frames:v", "1", still], check=True)
    for f in (out, still):
        shutil.copy(f, os.path.join(EXP, "png", os.path.basename(f)))
    print("[montage]", out)


def graphs():
    ins, f = [], []
    for i, (title, n) in enumerate(GRAPHS):
        ins += ["-i", os.path.join(G, n, "results", "movie_wind.mp4")]
        f.append(f"[{i}:v]split[a{i}][b{i}];[a{i}]crop={EX},scale=392:192[e{i}];[b{i}]crop={IN},scale=392:192[n{i}];"
                 f"[e{i}][n{i}]vstack,pad=392:420:0:36:black[c{i}]")
    lay = "|".join(f"{(i % 4) * 392}_{(i // 4) * 420}" for i in range(8))
    lab = _labels((1568, 840), [(((i % 4) * 392 + 8, (i // 4) * 420 + 8), t) for i, (t, _) in enumerate(GRAPHS)], 18)
    ins += ["-loop", "1", "-i", lab]
    fc = (";".join(f) + ";" + "".join(f"[c{i}]" for i in range(8)) + f"xstack=inputs=8:layout={lay}[g];"
          "[g][8:v]overlay=0:0:shortest=1[v]")
    run(ins + ["-filter_complex", fc, "-map", "[v]", "-r", "25"], os.path.join(EXP, "presentation", "Movies", "flow_graphs.mp4"))


def summary(tag="g17sel"):
    d = os.path.join(EXP, "png")
    ins = ["-i", os.path.join(d, f"wind_movie_consensus_mean_{tag}.mp4"), "-i", os.path.join(d, f"wind_movie_consensus_median_{tag}.mp4"),
           "-i", os.path.join(d, f"wind_movie_consensus_median_rec_{tag}.mp4")]     # the median over the RECORDED dF/F
    fc = (f"[0:v]split[m0][m1];[m0]crop={EX},scale=600:294[me];[m1]crop={IN},scale=600:294[mi];"
          f"[me][mi]hstack,pad=1200:330:0:36:black[r0];"
          f"[1:v]split[d0][d1];[d0]crop={EX},scale=600:294[de];[d1]crop={IN},scale=600:294[di];"
          f"[de][di]hstack,pad=1200:330:0:36:black[r1];"
          f"[2:v]split[q0][q1];[q0]crop={EX},scale=600:294[qe];[q1]crop={IN},scale=600:294[qi];"
          f"[qe][qi]hstack,pad=1200:330:0:36:black[r2];"
          f"[r0][r1][r2]vstack=inputs=3[g];[g][3:v]overlay=0:0:shortest=1[v]")
    ins += ["-loop", "1", "-i", _labels((1200, 990), [((8, 8), "MEAN flow of the 5 graphs   (excitatory | inhibitory)"),
                                                       ((8, 338), "MEDIAN flow of the 5 graphs   (excitatory | inhibitory)"),
                                                       ((8, 668), "MEDIAN flow over the RECORDED dF/F   (excitatory | inhibitory)")], 22)]
    run(ins + ["-filter_complex", fc, "-map", "[v]", "-r", "25"], os.path.join(EXP, "presentation", "Movies", "flow_summary.mp4"))


if __name__ == "__main__":
    graphs()
    summary()
