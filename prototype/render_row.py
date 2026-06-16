"""Run several ball scenarios and tile them into ONE row gif + montage, colored by
material layer, with an objective "break" metric per ball.

    PYTHONPATH=../src python render_row.py layer_ball_1 layer_ball_2 layer_ball_3 layer_ball_4 [out]

Each panel colors particles by their concentric layer (0=core red, 1=middle green,
2=membrane blue). break% = fraction of a ball's particles whose distance from the
ball centroid exceeds 1.8*radius at the final frame -> 0% means it stayed intact,
high% means it shed material / broke apart on the bounce.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenario_schema import load
import engine2

LAYER_COL = np.array([[0.85, 0.10, 0.10],     # 0 core   -> red
                      [0.15, 0.70, 0.20],     # 1 middle -> green
                      [0.25, 0.50, 1.00],     # 2 membrane -> blue
                      [0.6, 0.3, 0.8]])        # 3+ -> purple


def membrane_youngs(sc):
    for t in sc.sets["cell"]["types"].values():
        if t.get("layers"):
            return t["layers"][-1]["youngs"]
    return None


def run_one(name):
    sc = load("scenarios/%s.yaml" % name)
    rad = float(sc.sets["particle"]["radius"])
    _, a = engine2.run(sc, None, device="cuda")
    pp = a["particle_pos"]; lid = a["layer_id"]
    if lid is None:
        lid = np.zeros(pp.shape[1], int)
    com = a["cell_pos"][:, 0, :]                       # [T,2]  (single ball)
    # break%: MAX over time of the fraction of particles flung far (>1.8*rad) from the
    # ball centroid -> catches transient shedding during the bounce, not just the end.
    far = (np.linalg.norm(pp - com[:, None, :], axis=2) > 1.8 * rad)   # [T,Np]
    broke = float(far.mean(axis=1).max()) * 100.0
    return dict(name=name, pp=pp, lid=lid, mem=membrane_youngs(sc), broke=broke, com=com)


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("-")]
    out = "layer_row"
    if len(names) >= 2 and names[-1].startswith("layer_row") or names[-1] == "out":
        pass
    runs = [run_one(n) for n in names]
    T = min(r["pp"].shape[0] for r in runs)
    nP = len(runs)

    fig, axes = plt.subplots(1, nP, figsize=(3.2 * nP, 3.4)); fig.patch.set_facecolor("white")
    if nP == 1:
        axes = [axes]
    scs = []
    for ax, r in zip(axes, runs):
        ax.axhspan(0, 2 / 128, color="#cccccc")
        col = LAYER_COL[np.clip(r["lid"], 0, 3)]
        s = ax.scatter(r["pp"][0][:, 0], r["pp"][0][:, 1], s=2.0, c=col)
        scs.append(s)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("membrane E=%g\nbreak=%.0f%%" % (r["mem"], r["broke"]), fontsize=9)

    def upd(fr):
        for s, r in zip(scs, runs):
            s.set_offsets(r["pp"][fr])
        fig.suptitle("frame %d/%d  (red=core, green=middle, blue=membrane)" % (fr, T - 1), fontsize=9)
        return scs

    FuncAnimation(fig, upd, frames=T, blit=False).save(out + ".gif", writer=PillowWriter(fps=25))
    plt.close(fig)

    g = Image.open(out + ".gif"); frames = []
    try:
        while True:
            frames.append(g.copy().convert("RGB")); g.seek(g.tell() + 1)
    except EOFError:
        pass
    n = len(frames); idx = [0, int(n * 0.16), int(n * 0.24), int(n * 0.34), int(n * 0.6), n - 1]
    sel = [frames[i] for i in idx]; w, h = sel[0].size
    m = Image.new("RGB", (w, h * len(sel)), "white")
    for k, im in enumerate(sel):
        m.paste(im, (0, k * h))
    m.save("/tmp/%s_montage.png" % out)
    print("[done] %s.gif  " % out + "  ".join("%s(E=%g):break=%.0f%%" % (r["name"], r["mem"], r["broke"]) for r in runs),
          flush=True)


if __name__ == "__main__":
    main()
