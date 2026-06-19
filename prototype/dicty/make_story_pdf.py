"""make_story_pdf.py -- narrative PDF of the agentic loop: 10 "agent did X -> found Y" beats,
each with one final-frame simulation panel; the REAL data shown once at the top."""
import os, sys, traceback
sys.path.insert(0, "."); sys.path.insert(0, ".."); sys.path.insert(0, "/workspace/Plexus/src")
import numpy as np, cv2, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scenario_schema import load
import dicty_engine, dicty_engine_mpm   # register all ops
import opt_dicty as O
from make_target import CROP, WIN
GREEN = LinearSegmentedColormap.from_list("c", ["#000", "#0a3a0a", "#1d7a1d", "#7CFC00"])
VIDEO = "/groups/saalfeld/home/allierc/GraphData/graphs_data/dicty/260228InterfaceTransferMirrorB1-DB-floor-fl2.mp4"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))

TPL = """name: beat
seed: 0
boundary: periodic
n_frames: {nf}
record_every: {re}
dt: 0.5
vmax: 0.061
sets: {{cell: {{n: {n}, buffer: {buf}, init_npz: target_density.npz, types: {{dicty: {{fraction: 1.0, youngs: 60.0}}}}}}{pset}}}
fields: {{camp: {{res: 140, diffusion: {D}, decay: {decay}, couples_to: cell}}}}
operators:
{ops}
schedule: [{sched}]
"""

def spec(name, ops, sched, n=700, nf=320, D=0.02, decay=0.10, buf=2200, pset=""):
    body = TPL.format(nf=nf, re=max(1, nf // 8), n=n, buf=buf, D=D, decay=decay, ops="\n".join("  - " + o for o in ops),
                      sched=", ".join(sched), pset=pset)
    p = os.path.join(HERE, "specs", "_beat_" + name + ".yaml"); open(p, "w").write(body); return "_beat_" + name

SPRING = "{op: spring, at: cell, k_rep: 60.0, r0: 0.018, kadh: %s, r_on: %s, delta: 0.001, mu_f: 0.05}"
SEC = "{op: secrete, at: cell, to: camp, rate: %s}"
SENSE = "{op: sense, at: cell, from: camp, gain: %s}"
SAT = "{op: sense_sat, at: cell, from: camp, gain: %s, c_sat: %s, sat_n: 2.1}"
INFLOW = "{op: inflow, at: cell, rate: 4.0}"
RW = "{op: random_walk, at: cell, strength: 0.003}"
RELAY = "{op: relay, at: cell, gain: 140.0, thr: 0.10, field: camp}"
DREP = "{op: density_repel, at: cell, strength: 1.5, thr: 1.5, field: camp}"

# (name, DID, FOUND, build) -- build returns (specname, engine)
def B(ops, sched, **kw): return lambda: (spec(kw.pop("nm"), ops, sched, **kw), "pc")
BEATS = []
def add(nm, did, found, ops, sched, mpm=False, **kw):
    if mpm:
        BEATS.append((nm, did, found, lambda: ("dicty_mpm_base", "mpm")))
    else:
        BEATS.append((nm, did, found, lambda nm=nm, ops=ops, sched=sched, kw=kw: (spec(nm, ops, sched, **kw), "pc")))

base_sched = ["spring", "secrete", "camp.diffuse", "sense", "inflow", "random_walk", "integrate"]
add("1_baseline", "Influx + linear chemotaxis (secrete->sense), repulsion only",
    "Cells drift; weak diffuse clump, no structure",
    [SPRING % ("0.0", "0.02"), SEC % "4", SENSE % "8", INFLOW, RW], base_sched, D=0.02, decay=0.10)
add("2_collapse", "Crank chemotaxis (gain high, low cAMP diffusion)",
    "Keller-Segel collapse -> ONE central blob",
    [SPRING % ("0.0", "0.02"), SEC % "8", SENSE % "40", INFLOW, RW], base_sched, D=0.004, decay=0.10)
add("3_adhesion", "Add spring adhesion (kadh high, long r_on)",
    "Filament / strand network, not compact mounds",
    [SPRING % ("150.0", "0.25"), SEC % "8", SENSE % "30", INFLOW, RW], base_sched, D=0.006, decay=0.20)
add("4_relay", "Add excitable cAMP relay waves (relay op)",
    "Still 1-few mounds; relay NOT sufficient",
    [SPRING % ("10.0", "0.20"), SEC % "8", RELAY, SENSE % "30", INFLOW, RW],
    ["spring", "secrete", "camp.diffuse", "relay", "sense", "inflow", "random_walk", "integrate"], D=0.006, decay=0.10)
add("5_sense_sat", "Replace sense with Hill-saturated chemotaxis (sense_sat)",
    "BREAKTHROUGH - first MULTI-MOUND (5-6 compact mounds)",
    [SPRING % ("10.0", "0.20"), SEC % "11", SAT % ("500", "0.50"), INFLOW, RW],
    ["spring", "secrete", "camp.diffuse", "sense_sat", "inflow", "random_walk", "integrate"], D=0.008, decay=0.10, n=900)
add("6_tuned", "Tune the sense_sat parent (D=0.0042, secrete=11)",
    "REAL-like 4-8 dense mounds across seeds (best match)",
    [SPRING % ("10.0", "0.20"), SEC % "11", SAT % ("500", "0.50"), INFLOW, RW],
    ["spring", "secrete", "camp.diffuse", "sense_sat", "inflow", "random_walk", "integrate"], D=0.0042, decay=0.07, n=1200, nf=400)
add("7_metric", "Fix the metric (SSIM-on-image + mound-count; inner_mass was gameable)",
    "Revealed real target ~8 mounds; old metric hid morphology",
    [SPRING % ("20.0", "0.19"), SEC % "11", SAT % ("750", "0.80"), INFLOW, RW],
    ["spring", "secrete", "camp.diffuse", "sense_sat", "inflow", "random_walk", "integrate"], D=0.0042, decay=0.07, n=1200, nf=400)
add("8_density_repel", "Add density_repel (finite cell volume)",
    "Rescues high-density runaway but no stable mounds",
    [SPRING % ("10.0", "0.20"), SEC % "11", SAT % ("500", "0.50"), DREP, INFLOW, RW],
    ["spring", "secrete", "camp.diffuse", "sense_sat", "density_repel", "inflow", "random_walk", "integrate"], D=0.0042, decay=0.07, n=1200, nf=400)
add("9_collapse_long", "Stress test: run far longer (n_frames 400 -> 2000)",
    "HEADLINE: multi-mound is a TRANSIENT -> collapses to a single blob",
    [SPRING % ("10.0", "0.20"), SEC % "11", SAT % ("500", "0.50"), INFLOW, RW],
    ["spring", "secrete", "camp.diffuse", "sense_sat", "inflow", "random_walk", "integrate"], D=0.0042, decay=0.07, n=1200, nf=2000, buf=2600)
add("10_mpm", "Fork to soft-MPM bodies (8 particles/cell)",
    "Diffuse, inner~0.2 - deformable body NOT the missing ingredient", [], [], mpm=True)


def final_panel(ax, name, eng):
    sc = load(os.path.join(HERE, "specs", name + ".yaml"))
    E = dicty_engine_mpm if eng == "mpm" else dicty_engine
    _, hist = E.run(sc, device=DEV)
    h = hist[-1]; fmax = max(float(hh["field"].max()) for hh in hist) or 1.0
    ax.imshow(h["field"].T, origin="lower", extent=[0, 1, 0, 1], cmap=GREEN, vmin=0, vmax=fmax * 0.5)
    pos = h.get("ppos", h["pos"])
    ax.scatter(pos[:, 0], pos[:, 1], s=(0.6 if eng == "mpm" else 2.5), c="#FFA500", edgecolors="none")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    nm = O.n_mounds(O.fov_image(h["pos"])); inner = float(O.radial_profile(O.sim_density_seq([h["pos"]])[0])[:3].sum())
    return nm, inner, h["count"]


# REAL final panel
cap = cv2.VideoCapture(VIDEO); n = int(cap.get(7)); W = int(cap.get(3)); Hh = int(cap.get(4))
x0, x1 = int(CROP["x0"] * W), int(CROP["x1"] * W); y0, y1 = int(CROP["y0"] * Hh), int(CROP["y1"] * Hh)
cap.set(1, int(WIN[1] * (n - 1))); _, fr = cap.read(); real_img = cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)[::-1]; cap.release()

fig = plt.figure(figsize=(13, 20)); fig.patch.set_facecolor("black")
gs = fig.add_gridspec(6, 4, height_ratios=[1.4, 1, 1, 1, 1, 1], hspace=0.45, wspace=0.12)
axR = fig.add_subplot(gs[0, 1:3]); axR.imshow(real_img, origin="lower"); axR.set_xticks([]); axR.set_yticks([])
axR.set_title("REAL Dictyostelium data (target)\ndense, multi-mound aggregate (~8 mounds)", color="cyan", fontsize=12)
for i, (nm, did, found, build) in enumerate(BEATS):
    r, c = 1 + i // 2, (i % 2) * 2
    ax = fig.add_subplot(gs[r, c:c + 2]); ax.set_facecolor("black"); ax.set_xticks([]); ax.set_yticks([])
    try:
        sname, eng = build(); mc, inner, N = final_panel(ax, sname, eng)
        tag = f"mounds={mc}, inner={inner:.2f}, N={N}"
    except Exception as e:
        ax.text(0.5, 0.5, "render failed", color="red", ha="center"); tag = str(e)[:40]; traceback.print_exc()
    ax.set_title(f"{did}", color="orange", fontsize=9)
    ax.set_xlabel(f"-> {found}\n[{tag}]", color="white", fontsize=8)
fig.suptitle("How the agentic loop discovered (and falsified) a Dictyostelium model\n"
             "10 steps: agent DID -> FOUND -> final simulation.  Bottom line: the model makes multi-mound only\n"
             "transiently and collapses; no 2D mechanism (12 tested) reproduces the real dense multi-mound aggregate.",
             color="w", fontsize=12, y=0.995)
plt.savefig(os.path.join(HERE, "agentic_story.pdf"), facecolor="black", bbox_inches="tight")
plt.savefig(os.path.join(HERE, "agentic_story.png"), dpi=80, facecolor="black", bbox_inches="tight")
print("wrote agentic_story.pdf/png")
