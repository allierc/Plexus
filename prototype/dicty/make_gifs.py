"""make_gifs.py -- render a gif for every dicty spec (+ the real movie) for review."""
import os, sys, glob, importlib, traceback
sys.path.insert(0, "."); sys.path.insert(0, ".."); sys.path.insert(0, "/workspace/Plexus/src")
import numpy as np, cv2, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from scenario_schema import load
from make_target import CROP, WIN
import dicty_engine, dicty_engine_mpm   # noqa: F401 — registers ALL operators (point-cell + MPM) before load()
GREEN = LinearSegmentedColormap.from_list("c", ["#000", "#0a3a0a", "#1d7a1d", "#7CFC00"])
HERE = os.path.dirname(os.path.abspath(__file__)); DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
VIDEO = "/groups/saalfeld/home/allierc/GraphData/graphs_data/dicty/260228InterfaceTransferMirrorB1-DB-floor-fl2.mp4"


def engine_for(path):
    blob = open(path).read()
    name = "dicty_engine_mpm" if ("inflow_mpm" in blob or "op: mpm" in blob or "per_parent" in blob) else "dicty_engine"
    return importlib.import_module(name)


def render_spec(name):
    sc = load(os.path.join(HERE, "specs", name + ".yaml"))
    sc.n_frames = min(getattr(sc, "n_frames", 240), 240); sc.record_every = max(1, sc.n_frames // 40)
    if sc.sets["cell"].get("n", 0) > 1500: sc.sets["cell"]["n"] = 1500   # cap for fast render
    eng = engine_for(os.path.join(HERE, "specs", name + ".yaml"))
    _, hist = eng.run(sc, device=DEV)
    T = len(hist); fmax = max(float(h["field"].max()) for h in hist) or 1.0
    fig, ax = plt.subplots(figsize=(4.2, 4.2)); fig.patch.set_facecolor("black")
    im = ax.imshow(hist[0]["field"].T, origin="lower", extent=[0, 1, 0, 1], cmap=GREEN, vmin=0, vmax=fmax * 0.5)
    pts = ax.scatter(hist[0]["pos"][:, 0], hist[0]["pos"][:, 1], s=5, c="#FFA500", edgecolors="none")
    ax.set_xticks([]); ax.set_yticks([]); tt = ax.set_title("", color="white", fontsize=8)
    def upd(f):
        h = hist[f]; im.set_data(h["field"].T); pts.set_offsets(h["pos"])
        tt.set_text("%s  %d/%d  N=%d" % (name, f, T - 1, h["count"])); return [im, pts, tt]
    out = os.path.join(HERE, "gif_" + name + ".gif")
    FuncAnimation(fig, upd, frames=T, blit=False).save(out, writer=PillowWriter(fps=12)); plt.close(fig)
    return out, hist[-1]


def render_real():
    cap = cv2.VideoCapture(VIDEO); n = int(cap.get(7)); W = int(cap.get(3)); H = int(cap.get(4))
    x0, x1 = int(CROP["x0"] * W), int(CROP["x1"] * W); y0, y1 = int(CROP["y0"] * H), int(CROP["y1"] * H)
    frames = []
    for tf in np.linspace(WIN[0], WIN[1], 40):
        cap.set(1, int(tf * (n - 1))); _, fr = cap.read()
        frames.append(cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)[::-1])
    cap.release()
    fig, ax = plt.subplots(figsize=(4.2, 4.2)); fig.patch.set_facecolor("black")
    im = ax.imshow(frames[0], origin="lower"); ax.set_xticks([]); ax.set_yticks([])
    tt = ax.set_title("", color="white", fontsize=9)
    def upd(f): im.set_data(frames[f]); tt.set_text("REAL movie %d/%d" % (f, len(frames) - 1)); return [im, tt]
    out = os.path.join(HERE, "gif_REAL_data.gif")
    FuncAnimation(fig, upd, frames=len(frames), blit=False).save(out, writer=PillowWriter(fps=12)); plt.close(fig)
    return out


if __name__ == "__main__":
    print("REAL ->", render_real(), flush=True)
    specs = [os.path.basename(p)[:-5] for p in sorted(glob.glob(os.path.join(HERE, "specs", "*.yaml")))]
    for nm in specs:
        try:
            out, last = render_spec(nm)
            print(f"{nm} -> {os.path.basename(out)}  (final N={last['count']})", flush=True)
        except Exception as e:
            print(f"{nm} FAILED: {e}", flush=True); traceback.print_exc()
    print("DONE", flush=True)
