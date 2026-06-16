"""Render a falling/bouncing-ball scenario as a gif (+ mp4 via gen_mp4-style ffmpeg).

    python render_bounce.py <scenario_name> [--cpu]

Runs engine2 (generic engine, unchanged) on the named scenario in scenarios/, then
draws the MPM particles each frame with the floor marked, and writes <name>.gif and
<name>.mp4 in prototype/. Particle color encodes height-at-rest band so deformation
is visible; the COM trajectory is drawn as a faint line so the bounce is legible.
"""
import os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenario_schema import load
import engine2

P = os.path.dirname(os.path.abspath(__file__)) + "/"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name = args[0] if args else "bounce_ball"
    device = "cpu" if "--cpu" in sys.argv else ("cuda" if torch.cuda.is_available() else "cpu")
    sc = load("scenarios/%s.yaml" % name)
    W = float(getattr(sc, "world", 1.0))
    Nc = int(sc.sets["cell"]["n"]); ppc = int(sc.sets["particle"]["per_parent"])
    print(f"[spec ok] {sc.name}: {Nc} ball(s) x {ppc} particles, {sc.n_frames} frames, dev={device}",
          flush=True)
    _, a = engine2.run(sc, None, device=device)
    pp = a["particle_pos"]; par = a["parent"]; T = pp.shape[0]
    com = a["cell_pos"]                                   # [T, Nc, 2]
    ctype = a["cell_type"]; tnames = a["type_names"]

    # color particles by their parent ball's type (red=softer, blue=stiffer)
    palette = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    pcol = np.array(palette)[ctype[par] % len(palette)]

    H_in = 5.0 / max(W, 1.0)
    fig, ax = plt.subplots(figsize=(5, max(H_in, 2.0))); fig.patch.set_facecolor("white")
    floor = 2.0 / int(next(o.params.get("n_grid", 128) for o in sc.operators if o.op == "mpm"))
    ax.axhspan(0, floor, color="#cccccc")                 # the floor band (reflective wall)
    sc_pts = ax.scatter(pp[0][:, 0], pp[0][:, 1], s=2.0, c=pcol)
    trails = [ax.plot([], [], "-", color=palette[ctype[i] % len(palette)],
                      lw=0.8, alpha=0.4)[0] for i in range(Nc)]
    ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    legend = "  ".join("%s=%s" % (("red","blue","green","purple","orange")[i % 5], t)
                       for i, t in enumerate(tnames))
    tt = ax.set_title("", fontsize=8)

    def upd(fr):
        sc_pts.set_offsets(pp[fr])
        for i, tr in enumerate(trails):
            tr.set_data(com[:fr + 1, i, 0], com[:fr + 1, i, 1])
        tt.set_text("%s  frame %d/%d   %s" % (sc.name, fr, T - 1, legend))
        return [sc_pts, *trails, tt]

    FuncAnimation(fig, upd, frames=T, blit=False).save(P + name + ".gif",
                                                       writer=PillowWriter(fps=25))
    plt.close(fig)
    msg = "  ".join("ball%d(%s) y:%.3f->min%.3f->%.3f" %
                    (i, tnames[ctype[i]], com[0, i, 1], com[:, i, 1].min(), com[-1, i, 1])
                    for i in range(Nc))
    print("[done] wrote %s.gif  %s" % (name, msg), flush=True)


if __name__ == "__main__":
    main()
