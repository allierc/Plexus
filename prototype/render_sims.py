"""Render the 10 sim gifs (periodic), resumable: skips ones already done this round.
    python render_sims.py        # does all not-yet-done; re-run until 'ALL DONE'
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scenario_schema import load
import engine2
NAMES = {1:"aggregate",2:"disperse",3:"swarm",4:"wander",5:"squish",
         6:"sort",7:"chase",8:"mixed",9:"crystal",10:"collapse"}
for i, nm in NAMES.items():
    mark = "/tmp/psim_%d.done" % i
    if os.path.exists(mark):
        print("skip sim_%d (done)" % i, flush=True); continue
    sc = load("scenarios/%s.yaml" % nm); sc.n_frames = 900; sc.record_every = 1
    _, a = engine2.run(sc, None, device="cuda")
    pp, par = a["particle_pos"], a["parent"]; pt = a["cell_type"][par]; f = a["field"]; T = f.shape[0]
    soft, stiff = pt == 0, pt == 1
    fig, ax = plt.subplots(figsize=(5, 5)); fig.patch.set_facecolor("black")
    im = ax.imshow(f[0].T, origin="lower", extent=[0,1,0,1], cmap="inferno", vmin=0, vmax=max(f.max()*0.35,1e-6))
    sb = ax.scatter(pp[0][stiff,0], pp[0][stiff,1], s=0.4, c="#3a6ea5", alpha=0.5)
    sr = ax.scatter(pp[0][soft,0], pp[0][soft,1], s=0.5, c="cyan")
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xticks([]); ax.set_yticks([]); tt = ax.set_title("", color="white", fontsize=7)
    def upd(fr, im=im, sr=sr, sb=sb, tt=tt, pp=pp, f=f, soft=soft, stiff=stiff, nm=nm, T=T, i=i):
        im.set_data(f[fr].T); sr.set_offsets(pp[fr][soft]); sb.set_offsets(pp[fr][stiff])
        tt.set_text("sim_%d %s (periodic)  %d/%d" % (i, nm, fr, T-1)); return [im, sr, sb, tt]
    FuncAnimation(fig, upd, frames=T, blit=False).save("sim_%d.gif" % i, writer=PillowWriter(fps=20)); plt.close(fig)
    open(mark, "w").close(); print("sim_%d.gif = %s" % (i, nm), flush=True)
print("ALL DONE", flush=True)
