"""Build the exp20 deck (a learned law on the gut-brain recordings of Chen, James, Ruetten et al. 2026): figures and
slide bodies, no number typed by hand.

    PYTHONPATH=src:tools /workspace/.conda_envs/neural-graph-linux/bin/python tools/exp20_slides.py
    python presentation/fit_deck.py --dir experiments/exp20_gutbrain_graphcast/presentation --deck exp20.tex

The template is exp17's (`tools/exp17_slides.py`, `presentation/`): the picture LEFT, the specifics -- numbers,
equations, the data's provenance -- RIGHT in a `\\fitcol`. Every number on a slide is read from the file it describes
(the deposit's listing, the exported recording and its provenance, the paper's figure values in PAPER, each with its
figure and panel). Runs in the devcontainer only (INSTRUCTION.md, the cluster rule).

    01  the paper: all-optical gut-brain interrogation                  (Fig. 1 page)
    02  the deposit: conditions, fish, what each file holds              (figure)
    03  one fish: anatomy, cells, the known inputs over the session      (figure, from the exported recording)
    04  the target of the integration gate: Fig. 3d                       (figure crop + GV/GM)
    05  the law: the neuron graph on this fish (exp17's figure, drawn from the operator)
    06  the references before training: baselines, input mask
    then per batch (exp17's structure, Cedric 2026-10-03): its levers slide, the shown run's movie and curves, Cedric's
    two controls (W = 0 on the trained law; a law trained with no network) on the gut response in the free rollout;
    90  overview of every landed run
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")
PRES = os.path.join(EXP, "presentation")
DATA = os.path.join(EXP, "data")
PAPERS = os.path.join(EXP, "papers")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
DECK_TITLE = "multi-level GNN on brain-gut fish"
HIDDEN = {"04_fig3d", "90_overview"}
HIDDEN_BATCHES = {"1", "2", "3"}  # Cedric, 2026-10-04: "comment batch 1 2 and 3 slides" (kept in slides/, out of the deck)
HIDDEN_PAT = ("gb_ex_f4",)
# batch 4's per-fish W = 0 slide and no-network-twin movie (Cedric, 2026-10-04: "put in comments slide W=0 and no_W"):
# the 24-fish table and the site slides carry both controls' numbers
HIDDEN_RE = re.compile(r"^b4_.*_(controls|nonet_movie)$")       # batch 2's fish 4 slides: another law than fish 1's shown one (W prior), not comparable (Cedric)
NET_DECK = "batch {b} $\\cdot$ network dynamics"   # the control slides' title
# PER BATCH, ONE GROUP PER FISH (Cedric, 2026-10-03: "batch 2 on the template of batch 1, results for the two fish"):
# (fish, the run whose movie and curves are shown, the run the network test is read on, its no-network twin or None)
SHOWN = {"3": [(f"glucose fish {k}", f"gb_sx_f{k}_mask_nol1", f"gb_sx_f{k}_mask_nol1",
                {1: "gb_sx_now", 4: "gb_sx_f4_now"}.get(k)) for k in range(1, 7)],
         "1": [("glucose fish 1", "gb_ng_mask", "gb_ng_nol1", "gb_ng_now")],
         "2": [("glucose fish 1", "gb_ex_mask", "gb_ex_nol1", "gb_ex_now"),
               ("glucose fish 4", "gb_ex_f4", "gb_ex_f4", None)]}
# ONE TITLE PER BATCH, on every slide of it (Cedric, 2026-10-03)
BATCH_DECK = {"1": "batch 1 $\\cdot$ gut-brain glucose fish 1 $\\cdot$ sweep of multi-level GNN models",
              "2": "batch 2 $\\cdot$ gut-brain glucose fish 1 and 4 $\\cdot$ sweep of multi-level GNN models, stable integrator",
              "3": "batch 3 $\\cdot$ gut-brain glucose fish 1-6 $\\cdot$ the nominal law with its input mask"}            # slides written but commented out of all.tex (Cedric, 2026-10-03: slide 6, Fig. 3d)

# THE PAPER'S NUMBERS, each with its figure and panel (read off papers/figs/*_crop.png, +-1 on the bars)
PAPER = {
    "fig3d": {"regions": ["OT", "PBN", "nodose", "LH", "medial idMO", "DVC"],
              "GMV": [55, 35, 14, 62, 33, 30], "GM": [28, 31, 12, 48, 27, 26], "GV": [33, 9, 4, 26, 10, 7],
              "midbrain": ["OT"], "hindbrain": ["PBN", "medial idMO", "DVC"]},
    "fig2g": {"foregut": 1440, "midgut": 350, "N": 4},
    "fig2i": {"AP": (30, 165), "PBN": (25, 110), "N": 3},
    "fig1e": {"AP": 6, "PBN": 6, "DVC": 6, "medial idMO": 6, "OT": 5, "LH": 4, "nodose": 4, "N": 6},
}
COND_ORDER = ["glucose", "glutamate", "Lglucose", "fish_water", "blood_glucose"]
COND_LABEL = {"glucose": "D-glucose, gut", "glutamate": "glutamate, gut", "Lglucose": "L-glucose, gut (control)",
              "fish_water": "fish water, gut (control)", "blood_glucose": "D-glucose, blood (portal)"}
SITE_COL = {1: "#9e9e9e", 2: "#ff4040", 3: "#ffa040"}


def no_stimuli_only(t):
    """Cedric, 2026-10-04: "remove stimuli only in all slides" -- every law since batch 3 is driven by the stimuli only,
    so no slide says it; applied to every slide's text as it is written, the md's "what changed" cells included."""
    t = re.sub(r"\s*[,+;]\s*stimuli only(?![\w-])", "", t, flags=re.I)
    t = re.sub(r"\(\s*stimuli only\s*\)\s*", "", t, flags=re.I)
    return re.sub(r"stimuli only[,;:]?\s*", "", t, flags=re.I)


def frame(title, left, right, src, deck_title=None, left_gap=False):
    """`left_gap`: the picture moved down by 2 lines, to sit centred beside the text (Cedric, 2026-10-03)."""
    gap = (f"\\vspace*{{{left_gap if not isinstance(left_gap, bool) else 2}\\baselineskip}}\n" if left_gap else "")
    return (f"% generated by tools/exp20_slides.py from {src} ({title})\n"
            f"\\begin{{frame}}[t]{{{deck_title or DECK_TITLE}}}\n\\vspace*{{\\bandgap}}\n"
            "\\begin{columns}[T,onlytextwidth]\n\\begin{column}{0.58\\textwidth}\n"
            f"{gap}{left}\n\\end{{column}}\n\\begin{{column}}{{0.4\\textwidth}}\n\\vspace*{{2\\baselineskip}}\n"
            f"\\fitcol{{%\n{right}}}\n\\end{{column}}\n\\end{{columns}}\n\\end{{frame}}\n")


def head(s):
    return f"{{\\normalsize\\textbf{{{s}}}}}\\\\[4pt]\n"


def rows(pairs):
    body = "".join(f"{k} & {v} \\\\\n" for k, v in pairs)
    return ("{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{7pt}}l@{}}\n" + body + "\\end{tabular}\\par}\n"
            "\\vspace{8pt}")


def _black(ax):
    ax.set_facecolor("black")
    for s in ax.spines.values():
        s.set_color("0.6")
    ax.tick_params(colors="0.8", labelsize=8)
    ax.xaxis.label.set_color("0.9")
    ax.yaxis.label.set_color("0.9")
    ax.title.set_color("white")


# ============================================================================== 01 the paper
def slide_paper():
    import fitz
    pdf = os.path.join(PAPERS, "Chen_2026_NatCommun_s41467-026-76242-8_gut_vascular_interoception.pdf")
    d = fitz.open(pdf)
    pg = d[2]
    W, H = pg.rect.width, pg.rect.height
    png = os.path.join(PRES, "figs", "01_paper_fig1.png")
    pg.get_pixmap(dpi=220, clip=fitz.Rect(W * 0.06, H * 0.055, W * 0.95, H * 0.845)).save(png)    # the whole of Fig. 1, no body text
    f1 = PAPER["fig1e"]
    right = (head("Chen, James, Ruetten et al. 2026") + rows([
        ("", "Nat Commun 17:9881, Ahrens \\& Fitzgerald labs"),
        ("fish", "Tg(elavl3:H2B-jGCaMP7f), 7-8 dpf, light sheet"),
        ("rate", "1-3 whole-brain volumes per second"),
        ("inputs", "caged nutrient in the gut, released by a UV beam;"),
        ("", "a drifting grating (optomotor); fictive swimming"),
        ("controls", "UV off the fish; caged L-glucose; fish water")])
        + head("what they found (Fig. 1e, 2, 3)") + rows([
            ("D-glucose", ", ".join(f"{k} {v}/{f1['N']}" for k, v in f1.items() if k != "N") + " fish"),
            ("", "L-glucose and fish water: 0 fish (Fig. 2c,d)"),
            ("site", f"foregut {PAPER['fig2g']['foregut']:,} vs midgut {PAPER['fig2g']['midgut']:,} cells (Fig. 2g)"),
            ("integration", "hindbrain: gut + motor; midbrain: gut + visual + motor")])
        + head("their model") + "{\\scriptsize one neuron at a time: lagged linear kernels on gut UV, control UV,\\\\ "
        "visual and motor regressors, smoothness-regularised (Fig. 1d)\\par}\n")
    return ("01_paper", frame("Whole-brain imaging while a nutrient is released in the gut by light",
                              "\\panel{figs/01_paper_fig1.png}", right, os.path.basename(pdf),
                              deck_title="{\\fontsize{7}{8.4}\\selectfont Whole-brain, all-optical interrogation of neuronal "
                                         "dynamics underlying gut and vascular interoception in zebrafish}"))


# ============================================================================== 02 the deposit
def deposit():
    j = json.load(open(os.path.join(DATA, "figshare_article.json")))
    by = defaultdict(lambda: defaultdict(float))
    fish = defaultdict(set)
    for f in j["files"]:
        m = re.match(r"(Lglucose|blood_glucose|fish_water|glucose|glutamate)_fish(\d+)_", f["name"])
        if not m:
            continue
        c, k = m.group(1), int(m.group(2))
        kind = "cells" if "cells0" in f["name"] else ("volume" if "volume0" in f["name"] else "ephys")
        by[c][kind] += f["size"] / 1e9
        fish[c].add(k)
    return j, by, fish


SITE_NAME = {1: "off the fish (control)", 2: "gut / vessel", 3: "gut", 4: "gut / vessel", 5: "gut / vessel",
             6: "vessel", 7: "vessel"}
SITE_COLS = {1: "#9e9e9e", 2: "#ff4040", 3: "#ffa040", 4: "#ffd54f", 5: "#ba68c8", 6: "#4dd0e1", 7: "#81c784"}


def slide_deposit():
    """THE EXPERIMENTAL DESIGN (Cedric, 2026-10-03: "the title should be about the experimental design; print the total
    of fish"): one row per fish of the deposit, its session on a minutes axis, each UV pulse a tick coloured by its
    galvo site label (`ch_gpos`), from data/design.json (tools/exp20_design.py)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = json.load(open(os.path.join(DATA, "design.json")))
    j, by, fish = deposit()
    fig, ax = plt.subplots(figsize=(7.2, 6.0), facecolor="black")
    _black(ax)
    y, yt, yl, prev = 0, [], [], None
    for d in D:
        if prev is not None and d["condition"] != prev:
            y += 0.8
        prev = d["condition"]
        ax.plot([0, d["minutes"]], [y, y], color="0.45", lw=3, solid_capstyle="butt")
        for p_ in d["pulses"]:
            ax.plot([p_["volume"] * d["volume_s"] / 60] * 2, [y - 0.35, y + 0.35], color=SITE_COLS.get(p_["site"], "w"), lw=1.3)
        yt.append(y)
        yl.append(f"{COND_LABEL[d['condition']].split(',')[0]} {d['fish']}")
        y += 1
    ax.set_yticks(yt, yl, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("minutes of the session")
    used = sorted({p_["site"] for d in D for p_ in d["pulses"] if p_["site"] in SITE_COLS})
    for k in used:
        ax.plot([], [], color=SITE_COLS[k], lw=4, label=f"site {k}: {SITE_NAME[k]}")
    ax.legend(frameon=False, fontsize=9, labelcolor="white", loc="lower right", handlelength=1.5)
    fig.tight_layout()
    fig.savefig(os.path.join(PRES, "figs", "02_design.png"), dpi=200, facecolor="black")
    plt.close(fig)
    n_f = len(D)
    per = {c: [d for d in D if d["condition"] == c] for c in COND_ORDER}
    rate = lambda ds: ", ".join(sorted({f"{d['volume_s']:.2f} s" for d in ds}))
    right = (head(f"{n_f} fish, one session each, all on disk") + rows(
        [(COND_LABEL[c], f"{len(per[c])} fish, {np.mean([d['minutes'] for d in per[c]]):.0f} min, "
                         f"{np.mean([len(d['pulses']) for d in per[c]]):.0f} pulses, volume {rate(per[c])}") for c in COND_ORDER])
        + head("a session") + rows([
            ("UV pulses", "200 ms, blocked by site: grey off the fish (control),"),
            ("", "coloured on the gut, or on the portal vessel in the"),
            ("", "blood-glucose fish (sites are the galvo's label only)"),
            ("grating", "forward or still, ~60-s blocks, open loop"),
            ("swim", "fictive, two motor-nerve electrodes")])
        + head("per fish, three files") + rows([
            ("cells", f"the traces of every segmented cell ({np.mean([d['cells'] for d in D]) / 1e3:.0f}k on average)"),
            ("ephys", "26 channels at 6 kHz: swim, triggers, UV, galvo, grating"),
            ("volume", "the mean anatomy (for pictures only)")]))
    return ("02_deposit", frame(f"The experimental design: {n_f} fish, five conditions, UV pulses on and off the gut",
                                "\\panel{figs/02_design.png}", right, "data/design.json", left_gap=0.6,
                                deck_title=f"The experimental design: {n_f} fish, five conditions, UV pulses on and off the gut"))


def slide_anatomy():
    """WHERE THE UV WAS AIMED (Cedric, 2026-10-03: "pinpoint the sites on a fish anatomy"): the WHOLISTIC whole-body
    model (tools/exp20_body_export.py, tools/exp20_body_render.py) with the paper's target regions; the deposit's site
    labels per condition with their galvo x range, from data/design.json. The label -> region mapping is NOT recorded:
    the slide says so."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    A = os.path.join(DATA, "anatomy")
    if not os.path.exists(os.path.join(A, "body_lateral.png")):
        return ("02b_anatomy", "")
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.2), facecolor="black")
    for a_, v in zip(ax, ("lateral", "dorsal")):
        im = plt.imread(os.path.join(A, f"body_{v}.png"))
        r0, r1 = (0.12, 0.78) if v == "lateral" else (0.28, 0.72)      # full width in both: head and tail align
        a_.imshow(im[int(r0 * im.shape[0]):int(r1 * im.shape[0])])
        # SCALE BAR (Cedric, 2026-10-03): the model has no units; its skin is 19.7 units long, taken as a 3.8-mm 7-dpf
        # larva (1 unit ~ 0.19 mm; the retina then ~300 um across). The render's parallel scale (tools/exp20_body_render.py)
        # gives 23 units across the 2400-px width, so 500 um = (0.5 / 0.193) units = that many px
        px = 0.5 / (3.8 / 19.7) * 2400 / 23.0
        h_ = (r1 - r0) * im.shape[0]
        a_.plot([60, 60 + px], [h_ - 40, h_ - 40], color="white", lw=2.5)
        a_.text(60 + px / 2, h_ - 55, "\u2248 500 \u00b5m", color="white", fontsize=8, ha="center", va="bottom")
        a_.axis("off")
        a_.set_title(f"{v} view, head left", color="0.8", fontsize=9)
    marks = [("off the fish (control)", "#bdbdbd"), ("foregut", "#ff4040"), ("midgut", "#ffd54f"),
             ("hepatic portal system", "#66bb6a")]
    org = [("brain + spinal cord", "#4fc3f7"), ("gut", "#ff8a65"), ("liver", "#8d4a3a"), ("heart", "#e53935"),
           ("kidney", "#7e57c2"), ("swim bladder", "#e0e0e0")]
    h = [Line2D([], [], marker="o", ls="", color=c, markersize=9, label=l) for l, c in marks] + \
        [Line2D([], [], marker="s", ls="", color=c, markersize=7, label=l) for l, c in org]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=8, labelcolor="white")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(PRES, "figs", "02b_anatomy.png"), dpi=200, facecolor="black")
    plt.close(fig)
    D = json.load(open(os.path.join(DATA, "design.json")))
    from collections import defaultdict
    gx = defaultdict(list)
    for d in D:
        for p_ in d["pulses"]:
            if "galvo_x" in p_:
                gx[(d["condition"], p_["site"])].append(p_["galvo_x"])
    rw = []
    for c in COND_ORDER:
        ss = sorted({k[1] for k in gx if k[0] == c and k[1] > 0})
        rw.append((COND_LABEL[c], "sites " + ", ".join(str(k) for k in ss)))
    right = (head("the paper's targets (markers)") + rows([
        ("control", "UV outside the fish, on the tail or the swim bladder"),
        ("gut", "foregut, midgut (Fig. 1b, 2e): foregut ~4x more cells"),
        ("blood", "hepatic portal system near the liver (Fig. 5a)")])
        + head("the deposit's site labels") + rows(rw)
        + head("what we can and cannot say") + rows([
            ("", "site 1 = off the fish in every session (control block)"),
            ("", "the other labels are not calibrated to the body: one"),
            ("", "site moves 0.6 V fish to fish; blood sites 2, 4, 5, 6 share"),
            ("", "one voltage. Marker positions are the paper's regions"),
            ("", "on the WHOLISTIC body model, not measured beam spots")]))
    return ("02b_anatomy", frame("Where the UV was aimed: the paper's target regions on the larval body",
                                 "\\panel{figs/02b_anatomy.png}", right, "data/anatomy, data/design.json", left_gap=1,
                                 deck_title="Where the UV was aimed: the paper's target regions on a larval zebrafish"))


# ============================================================================== 03 one fish
def slide_fish(cond="glucose", k=1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.paths import graphs_data_path
    name = f"gutbrain_{cond}_f{k}"
    f = graphs_data_path("zebrafish", f"{name}_recording.npz")
    if not os.path.exists(f):
        return (f"03_{name}", "")
    z = np.load(f)
    prov = json.load(open(f.replace(".npz", ".json")))
    from plexus.tasks import trace_recording as TR
    X, P, S, tr, t = z["dff"], TR.load(name)["pos_view"], z["stimulus"], z["trials"], z["t_s"] / 60   # head left (exp17)
    fig = plt.figure(figsize=(7.2, 6.6), facecolor="black")
    gs = fig.add_gridspec(5, 1, height_ratios=[4.6, 0.5, 0.5, 0.5, 1.4], hspace=0.35)
    ax = fig.add_subplot(gs[0])
    _black(ax)
    # WHERE THE GUT SIGNAL REACHES (Cedric, 2026-10-03): the gut-responsive cells (tools/gutbrain_baselines.py, the
    # paper's selection on the training frames) coloured by their mean evoked change after the training gut pulses,
    # over every cell in grey; top view of the brain, no frame, a 100-um scale bar
    cz = np.load(os.path.join(DATA, f"baselines_{name}_cells.npz"))
    resp, evg = cz["responsive"], cz["evoked_gut"]
    sub = np.random.default_rng(0).choice(len(P), min(60000, len(P)), replace=False)
    ax.scatter(P[sub, 0], P[sub, 1], s=0.6, c="0.35", lw=0)
    o = np.argsort(evg[resp])
    ax.scatter(P[resp, 0][o], P[resp, 1][o], s=3.0, c=evg[resp][o], cmap="autumn", vmin=0,
               vmax=np.percentile(evg[resp], 98), lw=0)
    ax.set_aspect("equal")
    ax.axis("off")
    x0, y0 = P[:, 0].min(), P[:, 1].min() - 0.06 * np.ptp(P[:, 1])
    ax.plot([x0, x0 + 100], [y0, y0], color="white", lw=2.5)
    ax.text(x0 + 50, y0 - 0.02 * np.ptp(P[:, 1]), "100 \u00b5m", color="white", fontsize=8, ha="center", va="top")
    ax.set_title(f"the {int(resp.sum()):,} gut-responsive cells, coloured by their evoked dF/F after a gut pulse\n"
                 f"(yellow = strongest), on all {len(P):,} cells (grey); brain, top view, head left", fontsize=8)
    names = ["UV pulses", "grating speed", "swim power (L)"]
    for i, (nm, col) in enumerate(zip(names, (None, 3, 4))):
        a = fig.add_subplot(gs[1 + i])
        _black(a)
        if col is None:
            for r in tr:
                a.axvline(t[int(r[0])], color=SITE_COL.get(int(r[2]), "w"), lw=1.5 if r[6] else 0.8,
                          ls="-" if not r[6] else "--")
            a.set_ylim(0, 1)
        else:
            a.plot(t, S[:, col], color="#4fc3f7" if col == 3 else "#81c784", lw=0.6)
        a.set_yticks([]); a.set_xticks([])
        a.set_ylabel(nm, rotation=0, ha="right", va="center", fontsize=8)
        a.set_xlim(t[0], t[-1])
    a = fig.add_subplot(gs[4])
    _black(a)
    a.plot(t, np.median(X, 1), color="white", lw=0.6)
    for r in tr:
        a.axvline(t[int(r[0])], color=SITE_COL.get(int(r[2]), "w"), lw=0.5, alpha=0.6)
    a.set_xlim(t[0], t[-1])
    a.set_xlabel("min")
    a.set_ylabel("median\ndF/F", rotation=0, ha="right", va="center", fontsize=8)
    png = os.path.join(PRES, "figs", f"03_{name}.png")
    fig.savefig(png, dpi=200, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    sites = {int(s): int((tr[:, 2] == s).sum()) for s in np.unique(tr[:, 2])}
    ext = prov["extent_um"]
    right = (head(f"{cond} fish {k}") + rows([
        ("volumes", f"{prov['frames']:,}, {prov['volume_s']:.3f} s apart, {prov['frames'] * prov['volume_s'] / 60:.1f} min"),
        ("cells", f"{prov['cells_kept']:,} of {prov['cells_in_file']:,} (baseline $\\geq$ {prov['min_signal_counts']:.0f} counts above background)"),
        ("brain", f"{ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} \\textmu m (voxel inferred)"),
        ("dF/F", f"clipped to [{prov['clip'][0]}, {prov['clip'][1]}]: {prov['frac_clipped'] * 100:.2f} \\% of values")])
        + head("the known inputs (forcings)") + rows([
            ("UV", f"{len(tr)} pulses of {np.median(tr[:, 1]):.0f} ms; site 1 (grey, off the fish) {sites.get(1, 0)},"),
            ("", f"site 2 (red) {sites.get(2, 0)}, site 3 (orange) {sites.get(3, 0)}: gut"),
            ("", "the law reads the pulse and the beam's galvo x, y only"),
            ("grating", f"forward, open loop, on {np.mean(S[:, 3] > 0) * 100:.0f} \\% of volumes"),
            ("swim", f"left: bouts on {np.mean(S[:, 4] > 0.1) * 100:.1f} \\% of volumes; right: "
                     f"{'live' if prov['swim_live'][1] else 'no bouts (dead channel)'}")])
        + head("held out") + rows([
            ("", f"the last full pulse of every site (dashed): {int((z['split'] == 2).sum())} volumes"),
            ("", f"window {prov['window_volumes'][0]} before + {prov['window_volumes'][1]} after a pulse (55 s)")]))
    return (f"03_{name}", frame(f"One session: {COND_LABEL[cond]}, fish {k}, and its known inputs",
                                f"\\panel{{figs/03_{name}.png}}", right, f"{name}_recording.npz"))


# ============================================================================== 04 Fig. 3d
def slide_fig3d():
    import shutil
    import fitz                                            # Fig. 3d whole: its bars AND the region names under them
    d = fitz.open(os.path.join(PAPERS, "Chen_2026_NatCommun_s41467-026-76242-8_gut_vascular_interoception.pdf"))
    pg = d[5]
    W, H = pg.rect.width, pg.rect.height
    pg.get_pixmap(dpi=450, clip=fitz.Rect(W * 0.665, H * 0.262, W * 0.955, H * 0.418)).save(
        os.path.join(PRES, "figs", "04_fig3d.png"))
    F = PAPER["fig3d"]
    gvgm = {r: gv / gm for r, gv, gm in zip(F["regions"], F["GV"], F["GM"])}
    mid = np.mean([gvgm[r] for r in F["midbrain"]])
    hind = np.mean([gvgm[r] for r in F["hindbrain"]])
    right = (head("what Fig. 3d measures") + "{\\scriptsize the gain in a neuron's model correlation from adding\\\\ "
             "motor (GM), visual (GV) or both (GMV) regressors\\\\ to the gut-only model, \\% (N = 6 fish)\\par}\\vspace{6pt}\n"
             + head("GV / GM per region") + rows([(r, f"{F['GV'][i]} / {F['GM'][i]} = {gvgm[r]:.2f}")
                                                  for i, r in enumerate(F["regions"])])
             + head("the gate (G-region)") + rows([
                 ("midbrain", f"OT: {mid:.2f}"), ("hindbrain", f"PBN, medial idMO, DVC: {hind:.2f}"),
                 ("ratio", f"{mid / hind:.1f}: the law's input ablations, read the same way,"),
                 ("", "against the same ruler on the recording")]))
    return ("04_fig3d", frame("The integration map the law should learn: Fig. 3d",
                              "\\panel{figs/04_fig3d.png}", right, "the paper's Fig. 3d", left_gap=1,
                              deck_title="The target of G-region (Fig. 3d): gut + visual in the midbrain, gut + motor "
                                         "in the hindbrain"))


# ============================================================================== 05 the law, 06 the baselines
GD = os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
RUNS = os.path.join(GD, "log", "training", "gutbrain")
REC1 = "gutbrain_glucose_f1"


def _tex(t):
    t = str(t).replace("_now", "_no_W")                 # Cedric, 2026-10-03: the no-network runs read "no_W"
    # BATCH 3's DEFAULT IS NO W PRIOR (Cedric, 2026-10-03: "no l1 is the nominal default now, do not specify"): its run
    # names drop "_nol1", and the one arm that keeps the prior (gb_sx_base, gb_sx_f4_base) reads "Wprior"
    t = re.sub(r"\bgb_sx_nol1\b", "gb_sx_f1", t)          # fish 1's nominal, named as the other fish's
    t = re.sub(r"(gb_sx_\w*?)_nol1", r"\1", t)
    t = re.sub(r"gb_sx_(f4_)?base", r"gb_sx_\1Wprior", t)
    return t.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def figure_graph(op, P, path, n_show=80, seed=0, box_um=300.0):
    """The cell graph the law trains on (the operator's own edges), TOP VIEW, HEAD LEFT (pos_view): a, the whole brain,
    every cell a faint dot and the middle and long edges INTO `n_show` random cells; b, a box around one cell, the
    short edges between the cells inside it and that cell's own 18 edges drawn thick."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    C = {"short": "#9ecae1", "mid": "#fd8d3c", "long": "#ff4040"}
    E = {k: (op._E[k][0].cpu().numpy(), op._E[k][1].cpu().numpy()) for k in ("short", "mid", "long") if k in op._E}
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), facecolor="black", gridspec_kw={"width_ratios": [2.1, 1]})
    a, b = ax
    for x_ in ax:
        x_.set_facecolor("black"); x_.set_aspect("equal"); x_.axis("off")
    sub = rng.choice(len(P), min(50000, len(P)), replace=False)
    a.scatter(P[sub, 0], P[sub, 1], s=0.2, c="0.35", lw=0)
    tgt = rng.choice(len(P), n_show, replace=False)
    for k in ("mid", "long"):
        s_, r_ = E[k]
        m = np.isin(r_, tgt)
        a.add_collection(LineCollection(np.stack([P[s_[m], :2], P[r_[m], :2]], 1), colors=C[k], lw=0.5, alpha=0.8))
    a.scatter(P[tgt, 0], P[tgt, 1], s=6, c="white", lw=0, zorder=3)
    x0, y0 = P[:, 0].min(), P[:, 1].min() - 25
    a.plot([x0, x0 + 100], [y0, y0], color="white", lw=2.5)
    a.text(x0 + 50, y0 - 8, "100 \u00b5m", color="white", fontsize=8, ha="center", va="top")
    fig.text(0.02, 0.95, f"a  the whole brain, head left: middle and long edges into {n_show} cells", color="0.9",
             fontsize=12, va="top")
    c0 = int(np.argmin(np.linalg.norm(P - np.median(P, 0), axis=1)))     # a cell at the brain's centre: a full box
    inb = np.where(np.all(np.abs(P[:, :2] - P[c0, :2]) < box_um / 2, 1) & (np.abs(P[:, 2] - P[c0, 2]) < 8.0))[0]
    sset = set(inb.tolist())
    s_, r_ = E["short"]
    m = np.isin(r_, inb) & np.isin(s_, inb)
    b.scatter(P[inb, 0], P[inb, 1], s=8, c="0.6", lw=0)
    b.add_collection(LineCollection(np.stack([P[s_[m], :2], P[r_[m], :2]], 1), colors=C["short"], lw=0.4, alpha=0.5))
    for k in ("short", "mid", "long"):
        s_, r_ = E[k]
        m = r_ == c0
        b.add_collection(LineCollection(np.stack([P[s_[m], :2], P[r_[m], :2]], 1), colors=C[k], lw=1.8))
    b.scatter([P[c0, 0]], [P[c0, 1]], s=40, c="white", zorder=4)
    b.set_xlim(P[c0, 0] - box_um / 2, P[c0, 0] + box_um / 2)
    b.set_ylim(P[c0, 1] - box_um / 2, P[c0, 1] + box_um / 2)
    xb, yb = P[c0, 0] - box_um / 2 + 8, P[c0, 1] - box_um / 2 + 8
    b.plot([xb, xb + 20], [yb, yb], color="white", lw=2.5, zorder=5)
    b.text(xb + 10, yb + 3, "20 \u00b5m", color="white", fontsize=10, ha="center", va="bottom", zorder=5)
    fig.text(0.685, 0.95, "b  one cell (white) and its 18 senders,\n    in a 16-\u00b5m-deep slab around it", color="0.9",
             fontsize=12, va="top")
    for i_, (k, c) in enumerate(C.items()):
        fig.text(0.70 + 0.09 * i_, 0.06, k, color=c, fontsize=10, ha="center", weight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.84))               # both panels under one title line, tops aligned
    pa, pb = a.get_position(), b.get_position()
    b.set_position([pb.x0, pa.y0, pb.width, pa.height])
    fig.savefig(path, dpi=200, facecolor="black")
    plt.close(fig)


MID_TURN_DEG = 45.0       # the middle edges DRAWN turned about the vertical (Cedric, 2026-10-04): in the law they lie along
                          # x, y, z like the long ones, so drawn true they hide inside them


def figure_graph3d(op, P, path, mp4=None, n_frames=200, fps=25, n_show=80, seed=0):
    """THE CELL GRAPH IN 3-D, exp17's slide-6 turntable (tools/exp17_slides.py figure_neuron_graph, adapted): to scale in
    um, VTK off-screen, black, head LEFT (pos_view). a: the whole brain, the cells a faint cloud and the middle (orange)
    and long (red) edges INTO `n_show` random cells; b: a 60-um box around one cell, the short edges between the cells
    inside and that cell's own 18 edges thick. The middle edges are drawn turned MID_TURN_DEG about the vertical through
    their receiver, so they do not hide behind the long ones (the caption says so). `mp4`: a full turn of panel a."""
    import shutil
    import subprocess
    import tempfile
    import pyvista as pv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.tasks import trace_recording as TR
    pv.OFF_SCREEN = True
    COL = {"short": "#9ecae1", "mid": "#fd8d3c", "long": "#ff4040"}
    E = {k: (op._E[k][0].cpu().numpy(), op._E[k][1].cpu().numpy()) for k in ("short", "mid", "long")}
    Q = (P - (P.max(0) + P.min(0)) / 2).astype(np.float32)
    th = np.deg2rad(MID_TURN_DEG)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]], np.float32)
    rng = np.random.default_rng(seed)
    show = set(rng.choice(len(P), n_show, replace=False).tolist())

    def ends(k, s, r):
        a_, b_ = Q[s], Q[r]
        if k == "mid":                                     # drawn turned about the receiver's vertical
            a_ = b_ + (a_ - b_) @ R.T
        return a_, b_

    def lines(pl, a_, b_, col, w, al):
        if not len(a_):
            return
        n = len(a_)
        pl.add_mesh(pv.PolyData(np.concatenate([a_, b_]), lines=np.column_stack([np.full(n, 2), np.arange(n),
                                                                                 np.arange(n) + n]).ravel()),
                    color=col, line_width=w, opacity=al)

    def cam(pl, az, r_, f=(0.0, 0.0, 0.0)):
        el = np.deg2rad(32.0)
        pl.camera.position = (f[0] + r_ * np.cos(el) * np.cos(az), f[1] + r_ * np.cos(el) * np.sin(az), f[2] + r_ * np.sin(el))
        pl.render()

    tmp = tempfile.mkdtemp(prefix="ngraph20_")
    pa = pv.Plotter(off_screen=True, window_size=(1400, 1000))
    pa.set_background("black")
    pa.add_mesh(pv.PolyData(Q[::3]), color="#8a8a8a", point_size=1.5, opacity=0.3)
    for k, w, al in (("mid", 2.5, 0.95), ("long", 2.0, 0.85)):
        s, r = E[k]
        sel = np.fromiter((i in show for i in r), bool, len(r))
        lines(pa, *ends(k, s[sel], r[sel]), COL[k], w, al)
    pa.add_mesh(pv.PolyData(Q[np.array(sorted(show))]), color="white", point_size=7, render_points_as_spheres=True)
    pa.camera.focal_point = (0.0, 0.0, 0.0)
    pa.camera.up = (0.0, 0.0, 1.0)
    rr = 1.75 * float(np.ptp(P, 0).max())               # the whole brain in view at every turn
    az0 = np.deg2rad(-60.0)                             # from +x, -y: x (head to tail) runs left to right on screen
    cam(pa, az0, rr)
    fa = os.path.join(tmp, "a.png")
    pa.screenshot(fa)
    frames = []
    if mp4:
        for i in range(n_frames):
            cam(pa, az0 + 2 * np.pi * i / n_frames, rr)
            pa.screenshot(os.path.join(tmp, f"f{i:05d}.png"))
            frames.append(os.path.join(tmp, f"f{i:05d}.png"))
    pa.close()
    c = int(np.argmin(np.linalg.norm(Q - np.median(Q, 0), axis=1)))
    inbox = np.all(np.abs(Q - Q[c]) < 30.0, axis=1)
    pb = pv.Plotter(off_screen=True, window_size=(1000, 1000))
    pb.set_background("black")
    pb.add_mesh(pv.PolyData(Q[inbox]), color="#aaaaaa", point_size=5, render_points_as_spheres=True)
    s, r = E["short"]
    sel = inbox[s] & inbox[r]
    lines(pb, Q[s[sel]], Q[r[sel]], COL["short"], 1.0, 0.35)
    for k in ("short", "mid", "long"):
        s, r = E[k]
        sel = r == c
        a_, b_ = ends(k, s[sel], r[sel])
        lines(pb, a_, b_, COL[k], 4.0, 1.0)
        pb.add_mesh(pv.PolyData(a_), color=COL[k], point_size=12, render_points_as_spheres=True)
    pb.add_mesh(pv.PolyData(Q[c:c + 1]), color="white", point_size=18, render_points_as_spheres=True)
    pb.camera.focal_point = tuple(Q[c].tolist())
    pb.camera.up = (0.0, 0.0, 1.0)
    cam(pb, az0, 420.0, tuple(Q[c].tolist()))
    fb = os.path.join(tmp, "b.png")
    pb.screenshot(fb)
    pb.close()
    st = op.graph_stats
    note = (f"middle edges drawn turned {MID_TURN_DEG:.0f}° about the vertical, to show them apart from the long "
            "ones; in the law both lie along x, y, z")

    def compose(fa_, out):
        fig = plt.figure(figsize=(12, 5.4), facecolor="black")
        for j, (f, lab) in enumerate(((fa_, f"a   the whole brain, head left: middle and long edges into {n_show} cells"),
                                      (fb, "b   one cell (white), its 18 senders, the short edges around it"))):
            ax = fig.add_axes([0.0 if j == 0 else 0.58, 0.10, 0.58 if j == 0 else 0.42, 0.82])
            ax.imshow(plt.imread(f))
            ax.axis("off")
            fig.text(0.01 if j == 0 else 0.59, 0.955, lab, color="white", fontsize=10, va="top")
        fig.text(0.01, 0.045, "   ".join(f"{k}: {st[k]['edges']:,} edges, {st[k]['per_element']:.1f} per cell, mean "
                                         f"{st[k]['mean_um']:.1f} µm" for k in ("short", "mid", "long")),
                 color="0.75", fontsize=8)
        fig.text(0.01, 0.012, note, color="#fd8d3c", fontsize=8)
        for k, x in zip(("short", "mid", "long"), (0.60, 0.72, 0.84)):
            fig.text(x, 0.10, k, color=COL[k], fontsize=11, weight="bold")
        fig.savefig(out, dpi=150, facecolor="black")
        plt.close(fig)

    compose(fa, path)
    if mp4:
        for i, f in enumerate(frames):
            compose(f, os.path.join(tmp, f"g{i:05d}.png"))
        subprocess.run([TR._ffmpeg(), "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "g%05d.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", mp4], check=True)
        shutil.copy(path, mp4.replace(".mp4", ".png"))
    shutil.rmtree(tmp)
    return st


def slide_law():
    """The neuron graph on this fish, from the operator itself (the picture IS the graph trained), head left."""
    import exp17_slides as X17
    from plexus.tasks import trace_recording as TR
    png = os.path.join(PRES, "figs", "05_neuron_graph.png")
    mp4 = os.path.join(PRES, "Movies", "05_neuron_graph.mp4")
    pf = f"zebrafish/{REC1}_recording.npz"
    op = X17.neuron_graph_op(positions_file=pf)
    rec = TR.load(REC1)
    if not os.path.exists(mp4):          # the turntable (Cedric, 2026-10-04: "rotate 3D the view as in exp17"), made once
        figure_graph3d(op, np.asarray(rec["pos_view"], np.float64), png, mp4=mp4)
    E = {s: int(op._E[s][0].numel()) for s in ("short", "mid", "long") if s in op._E}
    right = (head("the law: a known ODE on a cell graph") + rows([
        ("state", "one per cell: dF/F"),
        ("cell", "its own time constant $\\tau_i$, rest $c_i$, 6 input weights $B_i$"),
        ("edges", ", ".join(f"{k} {v / 1e6:.2f}M" for k, v in E.items()) + ", one $W$ each"),
        ("", "6 nearest, 6 at 32 \\textmu m, 6 at 128 \\textmu m"),
        ("step", "$\\tau_i\\, dz_i/dt = -z_i + c_i + \\Omega_i(t) \\sum_j W_{ij}\\tanh z_j + B_i\\cdot u$"),
        ("", "$\\Omega_i(t)$: a SIREN of position and time scaling the messages (batch 4)"),
        ("inputs", "$u$: UV pulse, beam x, y, grating; the swim zeroed (batch 3 on)"),
        ("", "into the input-mask cells only; nothing recorded is read after the start")])
        + head("training") + rows([("", "exp17's rig: warm-up 10, horizons 1-5 then 10..50, 49,000 updates"),
                                    ("", "held-out trial windows never trained on")]))
    return ("05_law", frame("The law: a leaky ODE per cell, coupled by a learned graph, driven by the known inputs",
                            "\\playmovie{Movies/05_neuron_graph}", right, "state_diffuse[neuron_graph] on glucose fish 1",
                            left_gap=3, deck_title="The law: a known ODE per cell, coupled by a learned graph, driven "
                                                   "by the known inputs"))


def slide_baselines():
    b = json.load(open(os.path.join(DATA, f"baselines_{REC1}.json")))
    m = np.load(os.path.join(GD, "graphs_data", "zebrafish", f"input_mask_{REC1}.npz"))
    png = os.path.join(EXP, "png", f"input_mask_{REC1}.png")
    import shutil
    shutil.copyfile(png, os.path.join(PRES, "figs", "06_input_mask.png"))
    right = (head("the references, before any training") + rows([
        ("gut-responsive", f"{b['gut_responsive']:,} cells (the paper's selection, training frames)"),
        ("noise", f"$\\sigma^2$ {b['noise_sigma2']:.3f} dF/F$^2$ over all cells (ZAPBench 0.0005)"),
        ("control / gut", f"{b['evoked_ratio_ctrl_over_gut']:.2f}: the brain barely answers the off-fish UV"),
        ("replicable", f"{b['top1_cells']:,} cells, held-out gut evoked {b['evoked_top1_heldout_gut']:+.3f}")])
        + head("the input mask (exp17's, per input)") + rows([
            ("", f"top 10 \\% per input, training frames: union {int(m['mask'].sum()):,} cells"),
            ("uv", "site-blind trial-locked $|t|$: holds 71 \\% of the gut-responsive cells"),
            ("visual, swim", "exp17's coherence with the input")]))
    return ("06_baselines", frame("Before training: the gut-responsive cells, the references, the input mask",
                                  "\\panel{figs/06_input_mask.png}", right, f"data/baselines_{REC1}.json",
                                  deck_title="Before training: the references, and the input mask (the cells each "
                                             "input enters)"))


def slides_kymo(rec_name=REC1, n_rows=100, bio=""):
    """ONE SLIDE PER INPUT (Cedric, 2026-10-03: "a kymograph as in exp17's slide 32, one slide per stimulus, the fish
    vertical"): left, the input over the session and the n_rows cells most locked to it (the input mask's own score:
    uv |t|, visual / swim coherence), dF/F rows z-scored, sorted by the lag of their response; the UV pulses (site
    colours) and the held-out windows (red) drawn through; right, the brain HEAD UP, that input's mask cells coloured on
    every cell in grey."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.tasks import trace_recording as TR
    rec = TR.load(rec_name)
    X, S, tr, sp = rec["dff"], rec["stimulus"], rec["trials"], rec["split"]
    t = rec["t_s"] / 60
    dt = float(np.median(np.diff(rec["t_s"])))
    bio = "_bio" if bio is True else (bio or "")       # a mask tag: "_bio", "_paper3sd" (batch 6)
    M = np.load(os.path.join(GD, "graphs_data", "zebrafish", f"input_mask_{rec_name}{bio}.npz"))
    Pv = rec["pos_view"]
    up = np.stack([Pv[:, 1], -Pv[:, 0]], 1)                 # head left -> head UP
    INP = [("uv", "UV pulse", M["score_uv"], S[:, 0], "#ff4040", "trial-locked |t| over the training pulses"),
           ("visual", "grating speed", M["score_visual"], S[:, 3], "#4fc3f7", "coherence with the grating"),
           ("swim", "swim power (left)", M["score_swim"], S[:, 4], "#81c784", "coherence with the swim power")]
    if bio:                         # batch 6 (Cedric, 2026-10-04: "reuse slide 8 for the discussion"): the UV cells of the
        INP = [("uv", "UV pulse", np.where(M["mask_by_input"][:, 0] > 0, M["score_uv"], -np.inf), S[:, 0], "#ff4040",
                "the paper's gut-responsive rule, mode + 3 SD (score: the regression's r)" if "paper" in bio else
                "the atlas's area postrema and vagal ganglia (BigWarp registration; score: |t| after every pulse)" if "anat" in bio else
                "excited after the training gut pulses (t), not after the control")]    # mask closer to biology
    out = []
    for key, label, score, u, col, how in INP:
        top = np.argsort(score)[::-1][:n_rows]
        Z = X[:, top]
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
        if key == "uv":                                     # lag = the time to peak of the trial-locked mean response
            on = [int(f) for f, h in zip(tr[:, 0], tr[:, 6]) if not h and f + 50 < len(X)]
            resp = np.mean([Z[f:f + 50] for f in on], 0)
            lag = resp.argmax(0)
        else:                                               # lag of the largest cross-correlation within 20 s
            uu = (u - u.mean()) / (u.std() + 1e-9)
            L = int(20 / dt)
            cc = np.stack([(uu[:len(uu) - k][:, None] * Z[k:]).mean(0) for k in range(L)], 0)
            lag = np.abs(cc).argmax(0)
        Z = Z[:, np.argsort(lag)]
        fig = plt.figure(figsize=(12, 6.2), facecolor="black")
        gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1], height_ratios=[0.6, 4], hspace=0.12, wspace=0.05)
        a0, a1, b_ = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
        _black(a0); _black(a1)
        a0.plot(t, u, color=col, lw=0.7)
        a0.set_xlim(t[0], t[-1]); a0.set_xticks([]); a0.set_yticks([])
        a0.set_title(f"{label} over the session", fontsize=10, loc="left", color="white")
        im = a1.imshow(Z.T, aspect="auto", cmap="inferno", vmin=-1, vmax=3, extent=[t[0], t[-1], n_rows, 0],
                       interpolation="nearest")
        for a_ in (a0, a1):
            for f, site, held in zip(tr[:, 0], tr[:, 2], tr[:, 6]):
                a_.axvline(t[int(f)], color=SITE_COLS.get(int(site), "w"), lw=0.6, alpha=0.8, ls="--" if held else "-")
            for f0 in np.where(np.diff(np.r_[0, (sp == 2).astype(int)]) == 1)[0]:
                f1 = f0 + int(np.argmax(sp[f0:] != 2)) if (sp[f0:] != 2).any() else len(sp)
                a_.axvspan(t[f0], t[min(f1, len(t) - 1)], color="#d62728", alpha=0.18, lw=0)
        a1.set_xlabel("time since the session's start, min", color="white")
        a1.set_ylabel(f"the {n_rows} cells most locked to the {label}\n(z-scored dF/F, sorted by lag)", fontsize=9)
        a1.set_title(f"rows: {how}; red bands: held-out trial windows", fontsize=9, loc="left", color="white")
        b_.set_facecolor("black"); b_.axis("off"); b_.set_aspect("equal")
        m = M["mask_by_input"][:, {"uv": 0, "visual": 3, "swim": 4}[key]] > 0
        b_.scatter(up[::8, 0], up[::8, 1], s=0.3, c="0.3", lw=0)
        b_.scatter(up[m, 0], up[m, 1], s=0.5, c=col, lw=0)
        b_.scatter(up[top, 0], up[top, 1], s=9, c="white", lw=0)
        b_.set_title(f"its input cells: {'batch 6' if bio else 'top 10 %'} ({int(m.sum()):,}),\nthe {n_rows} rows in white; "
                     "head up", fontsize=9, color="white")
        y0 = up[:, 1].min() - 20
        b_.plot([up[:, 0].min(), up[:, 0].min() + 100], [y0, y0], color="white", lw=2)
        b_.text(up[:, 0].min() + 50, y0 - 8, "100 \u00b5m", color="white", fontsize=8, ha="center", va="top")
        png = f"b6{bio}_kymo_{key}.png" if bio else f"06_kymo_{key}.png"
        fig.savefig(os.path.join(PRES, "figs", png), dpi=180, facecolor="black", bbox_inches="tight")
        plt.close(fig)
        right_rows = [("input", label), ("score", how), ("cells", f"top 10 %: {int(m.sum()):,} of {len(m):,}"),
                      ("rows", f"the {n_rows} highest scores, sorted by lag")]
        dtitle = (f"batch 6 $\\cdot$ discussion $\\cdot$ the gut-input cells of the paper's rule (mode + 3 SD)" if "paper" in bio
                  else "batch 6 $\\cdot$ discussion $\\cdot$ the gut-input cells of the atlas: area postrema + vagal ganglia"
                  if "anat" in bio
                  else f"batch 6 $\\cdot$ discussion $\\cdot$ the input cells of the {_tex(label)}, closer to biology" if bio
                  else f"{DECK_TITLE} $\\cdot$ the input cells of the {_tex(label)}")
        body = (f"% generated by tools/exp20_slides.py (input kymograph {key})\n\\begin{{frame}}[t]{{{dtitle}}}\n"
                f"\\vspace*{{\\bandgap}}\n"
                f"\\begin{{center}}\\includegraphics[width=0.98\\textwidth,height=0.80\\textheight,keepaspectratio]"
                f"{{figs/{png}}}\\end{{center}}\n\\end{{frame}}\n")
        out.append((f"b6{bio}_kymo_{key}" if bio else f"06_kymo_{key}", body))
    return out


# ============================================================================== batches, runs, controls
BATCHES = {
    "1": ("which levers carry the gut response", ["gb_ng_base", "gb_ng_s1", "gb_ng_mask", "gb_ng_now", "gb_ng_nol1",
                                                "gb_ng_wide", "gb_ng_h50", "gb_gc_base", "gb_ng_base_lglu"], "gb_ng_nol1", "gb_ng_now"),
    "2": ("the same levers, without runaway cells", ["gb_ex_base", "gb_ex_s1", "gb_ex_nol1", "gb_ex_now", "gb_ex_mask", "gb_ex_h50",
                                     "gb_ex_base_lglu", "gb_ex_f4"], "gb_ex_nol1", "gb_ex_now"),
    "3": ("the nominal law, six fish", ["gb_sx_base", "gb_sx_lin", "gb_sx_nol1", "gb_sx_now", "gb_sx_lk_snd", "gb_sx_siren",
                                     "gb_sx_base_lglu"] + [f"gb_sx_f{k}_nol1" for k in (2, 3, 4, 5, 6)]
                                    + [f"gb_sx_f{k}_mask_nol1" for k in range(1, 7)]
                                    + ["gb_sx_f4_base", "gb_sx_f4_lin", "gb_sx_f4_now", "gb_sx_f4_lk_snd", "gb_sx_f4_siren"],
          "gb_sx_f1_mask_nol1", "gb_sx_now"),
}


def md_rows():
    """{run: (v, what changed)} from the md's results table (the one place a run's change is written)."""
    md = open(os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast.md")).read()
    out = {}
    for line in md.splitlines():
        m = re.match(r"\| (\S+) \| `training/gutbrain/(\w+)` \|(.*)\|\s*$", line)
        if m:
            cells = [c.strip() for c in m.group(3).split("|")]
            out[m.group(2)] = (m.group(1), cells[-2] if len(cells) >= 2 else "")
    return out


def landed(name):
    d = os.path.join(RUNS, name, "results")
    t = os.path.join(d, f"{name}_test.json")
    if not os.path.exists(t):
        return None
    r = {"name": name, "dir": os.path.join(RUNS, name), "test": json.load(open(t))}
    for k in ("trials", "freetrial"):
        p = os.path.join(d, f"{name}_{k}.json")
        r[k] = json.load(open(p)) if os.path.exists(p) else None
    rep = os.path.join(d, "report.json")
    r["report"] = json.load(open(rep)) if os.path.exists(rep) else {}
    return r


def brain_r2(d, stem):
    """THE BRAIN-MEAN R2 (exp17's headline, Cedric 2026-10-03): 1 - sum_t (pbar_t - obar_t)^2 / sum_t (obar_t - mean)^2
    over the free rollout's frames, obar / pbar the recorded / learned mean over the alive cells (trainer
    `_brain_mean_metrics`, read from the rollout's own movie npz)."""
    from plexus import trainer as T
    p = os.path.join(d, "results", f"{stem}_movie.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return T._brain_mean_metrics(z["mean_obs_all"], z["mean_pred_all"])["brain_mean_r2"]


def cell_r2(d, stem):
    """The per-cell R2: R2 per frame over the alive cells against the denoised recording, the mean over the free
    rollout's frames (results/<stem>_free.npz, the trainer's `_trace_free` or a rollout variant)."""
    p = os.path.join(d, "results", f"{stem}_free.npz")
    return float(np.nanmean(np.load(p)["r2_denoised"])) if os.path.exists(p) else None


def bm(d, stem):
    """Brain-mean R2 and RMSE of a free rollout (trainer `_brain_mean_metrics` on its movie npz); (None, None) if absent."""
    from plexus import trainer as T
    p = os.path.join(d, "results", f"{stem}_movie.npz")
    if not os.path.exists(p):
        return None, None
    z = np.load(p)
    q = T._brain_mean_metrics(z["mean_obs_all"], z["mean_pred_all"])
    return q["brain_mean_r2"], q["brain_mean_rmse"]


def control_table(name, now):
    """THE NETWORK TEST, exp17's table (Cedric, 2026-10-03), first on every results slide: the full model, W = 0 at
    inference, the batch's model trained with no network -- brain-mean R2 and RMSE (dF/F), per-cell R2."""
    r, rn = landed(name), landed(now) if now else None
    f = lambda v, fmt="{:+.3f}": fmt.format(v) if v is not None and np.isfinite(v) else "--"
    rws = [("full model", r["dir"], name)]
    if os.path.exists(os.path.join(r["dir"], "results", f"{name}_W0_movie.npz")):
        rws.append(("W = 0 at inference", r["dir"], f"{name}_W0"))
    if rn and now != name:
        rws.append(("trained with no network", rn["dir"], now))
    body = "".join(f"{a} & {f(bm(d, st)[0])} & {f(bm(d, st)[1], '{:.4f}')} \\\\\n"
                   for a, d, st in rws)
    return (head("the network test, free rollout") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{5pt}}r@{\\hspace{5pt}}r"
            "@{}}\n& \\multicolumn{2}{c}{brain-mean dF/F} \\\\\n& R$^2$ & RMSE \\\\\n\\hline\n"
            + body + "\\end{tabular}\\par}\\vspace{6pt}\n")


def numbers(r):
    """The four numbers a batch slide shows for a run, from its own result files (None where not measured)."""
    from exp_measures import exp20 as M
    out = {"short": r["test"]["skill_short"], "free_r2": r["test"]["free"]["r2_denoised"], "trial": None, "free_gut": None,
           "brain_r2": brain_r2(r["dir"], r["name"])}
    if r["trials"]:
        g = [t for t in r["trials"]["trials"] if t["kind"] == "gut"]
        out["trial"] = M.window_skill([t["mse_resp"] for t in g], [t["mse_sta_resp"] for t in g])
    if r["freetrial"]:
        out["free_gut"] = r["freetrial"]["arms"]["full"]["evoked_gut_free_over_rec"]
        out["pattern"] = r["freetrial"]["arms"]["full"]["pattern_r_gut"]
    return out


def figure_batch(b, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = {n: landed(n) for n in names}
    N = {n: numbers(r) for n, r in L.items() if r}
    keys = [("short", "skill over the mean, h 1-3"), ("trial", "held-out trial skill over the STA"),
            ("free_gut", "free rollout: gut response / recorded"), ("brain_r2", "free rollout: brain-mean R$^2$")]
    fig, ax = plt.subplots(len(keys), 1, figsize=(7, 7.5), facecolor="black", sharex=True)
    x = np.arange(len(names))
    for a, (k, lab) in zip(ax, keys):
        _black(a)
        v = [N[n][k] if n in N and N[n].get(k) is not None else np.nan for n in names]
        a.bar(x, np.clip(v, -1.0, 1.5), 0.6, color=["#4fc3f7" if np.isfinite(t) and t >= 0 else "#ff7043" for t in v])
        for i, t in enumerate(v):
            if np.isfinite(t):
                a.text(i, (min(max(t, -1.0), 1.5)) + (0.04 if t >= 0 else -0.12), f"{t:+.2f}", ha="center",
                       color="white", fontsize=7)
        a.axhline(0, color="0.6", lw=0.6)
        a.set_ylabel(lab, fontsize=7, rotation=0, ha="right", va="center")
    ax[-1].set_xticks(x, [f"{i + 1}" for i in range(len(names))], fontsize=8)
    ax[-1].set_xlabel("arm (table, right)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="black")
    plt.close(fig)
    return N


def slide_batch(b):
    title, names, shown, now = BATCHES[b]
    N = figure_batch(b, names, os.path.join(PRES, "figs", f"batch_{b}_levers.png"))
    rw = md_rows()

    def f(n, k):
        v = N.get(n, {}).get(k)
        return f"{v:+.2f}" if v is not None and np.isfinite(v) else "--"
    tab = "".join(f"{i + 1} & {_tex(n)} & {f(n, 'short')} & {f(n, 'trial')} & {f(n, 'free_gut')} & {f(n, 'brain_r2')} \\\\\n"
                  if n in N else f"{i + 1} & {_tex(n)} & \\multicolumn{{4}}{{l}}{{running}} \\\\\n" for i, n in enumerate(names))
    right = (head(f"batch {b}: {title}") + "{\\scriptsize\\begin{tabular}{@{}r@{\\hspace{4pt}}l@{\\hspace{5pt}}r@{\\hspace{5pt}}r"
             "@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{}}\n & arm & short & trial & free gut & brain R$^2$ \\\\\n\\hline\n" + tab
             + "\\end{tabular}\\par}\\vspace{6pt}\n"
             + "{\\scriptsize short: skill over the best recent mean, h 1-3, held-out windows. trial: skill over the "
               "stimulus-triggered mean, held-out gut trials, gut-responsive cells, from the pulse. free gut: the evoked response "
               "to the held-out gut pulses INSIDE the free rollout of the whole session (stimuli and the first frame only), over "
               "the recorded one. brain R$^2$: the whole session's brain-mean dF/F, learned against recorded (exp17's headline).\\par}\n")
    return (f"batch_{b}_levers", frame(f"batch {b}: every arm", f"\\panel{{figs/batch_{b}_levers.png}}", right,
                                       "the runs' _test, _trials, _freetrial json", deck_title=BATCH_DECK[b]))


def run_deck(b, fish=""):
    """A batch's title, naming the fish when the batch runs more than one (Cedric, 2026-10-03)."""
    if b == "4" and fish:
        return f"batch 4 $\\cdot$ {fish} $\\cdot$ the SIREN law"
    return BATCH_DECK[b] if not fish else BATCH_DECK[b].replace("glucose fish 1 and 4", fish).replace("glucose fish 1-6", fish)


def slides_run(name, b, now=None, fish=""):
    """Two slides per shown run, exp17's (tools/exp17_slides.py slides_run): the movie with the network test, the MSE
    table, exp20's gut trials, the free rollout and the training; the curves with the per-trial MSE and the scale."""
    import shutil
    r = landed(name)
    if not r:
        return []
    out, rw = [], md_rows()
    v, what = rw.get(name, ("", ""))
    names = BATCHES[b][1]
    arm = names.index(name) + 1 if name in names else ""
    tag = f"batch {b}, arm {arm}" if arm and b != "4" else f"batch {b}"      # batch 4: the fish names the run
    dt = run_deck(b, fish)
    t, rep = r["test"], r["report"]
    N = numbers(r)
    S_, L_ = slice(0, 3), slice(15, 32)
    mse = {k: (np.mean(np.asarray(t[k])[S_]) * 1e3, np.mean(np.asarray(t[k])[L_]) * 1e3)
           for k in ("mse_model", "mse_mean", "mse_lookup") if k in t}
    fr = t["free"]
    stages = rep.get("stages") or []
    ft = r["freetrial"]["arms"]["full"] if r["freetrial"] else None
    pm = lambda a, sd: f"{a:+.3f} $\\pm$ {sd:.3f}" if sd is not None else f"{a:+.3f}"
    num = (head(f"{tag}: {_tex(name)}") + "{\\scriptsize " + _tex(what) + "\\par}\\vspace{6pt}\n"
           + control_table(name, now)
           + head("training") + rows([
               ("updates", f"{rep.get('iters', 0):,} (horizons {stages[0][0]}..{stages[-1][0]})" if stages else "--"),
               ("time", f"{rep.get('seconds', 0) / 3600:.1f} h"), ("weights", f"{rep.get('n_params', 0):,}")]))
    mv = os.path.join(r["dir"], "results", "movie.mp4")
    if os.path.exists(mv):
        shutil.copyfile(mv, os.path.join(PRES, "Movies", f"{name}.mp4"))
        shutil.copyfile(mv.replace(".mp4", ".png"), os.path.join(PRES, "Movies", f"{name}.png"))
        out.append((f"{name}_movie", frame(f"{tag}: the free rollout of the whole session, recorded left, learned right",
                                           f"\\playmovie{{Movies/{name}}}", num, f"{name}/results/movie.mp4", deck_title=dt,
                                           left_gap=True)))
    cp = os.path.join(r["dir"], "results", f"{name}_test.png")
    if os.path.exists(cp):
        shutil.copyfile(cp, os.path.join(PRES, "figs", f"{name}_test.png"))
        tj = r["trials"]["trials"] if r["trials"] else []
        per = "".join(f"{x['kind']} {x['onset']} & {np.mean(x['mse_resp']) * 1e3:.1f} & {np.mean(x['mse_sta_resp']) * 1e3:.1f} \\\\\n"
                      for x in tj)
        right = (head(f"{tag}: {_tex(name)}") + head("held-out trials: window MSE, $10^{-3}$")
                 + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{7pt}}r@{\\hspace{7pt}}r@{}}\ntrial & law & STA \\\\\n\\hline\n"
                 + per + "\\end{tabular}\\par}\\vspace{6pt}\n"
                 + head("the scale") + rows([
                     ("persistence, h 1", f"{t['mse_persistence'][0]:.4f}"), ("mean, h 1", f"{t['mse_mean'][0]:.4f}"),
                     ("model, h 1", f"{t['mse_model'][0]:.4f}"), ("mean, h 32", f"{t['mse_mean'][-1]:.4f}"),
                     ("model, h 32", f"{t['mse_model'][-1]:.4f}")])
                 + rows([("brain-mean R$^2$", f"{bm(r['dir'], name)[0]:+.3f}" if bm(r['dir'], name)[0] is not None else "--"),
                         ("brain-mean RMSE", f"{bm(r['dir'], name)[1]:.4f} dF/F" if bm(r['dir'], name)[1] is not None else "--")]))
        out.append((f"{name}_curves", frame(f"{tag}: the prediction against the mean baseline, step by step",
                                            f"\\panel{{figs/{name}_test.png}}", right, f"{name}_test.png", deck_title=dt, left_gap=4)))
    return out


def net_deck(b, fish=""):
    return NET_DECK.format(b=b) if not fish else f"batch {b} $\\cdot$ {fish} $\\cdot$ network dynamics"


def slides_controls(name, now, b, fish=""):
    """CEDRIC'S TWO CONTROLS (exp17 slides 26-27): the trained law with W = 0, and a law trained with no network."""
    import shutil
    r, rn = landed(name), (landed(now) if now else None)
    out = []
    dt = net_deck(b, fish)
    if r and r["freetrial"] and "W0" in r["freetrial"]["arms"]:
        full, w0 = r["freetrial"]["arms"]["full"], r["freetrial"]["arms"]["W0"]
        nw = rn["freetrial"]["arms"]["full"] if rn and rn["freetrial"] else None
        ab = os.path.join(EXP, "data", f"ablation_{name}.json")
        A = json.load(open(ab)) if os.path.exists(ab) else None
        t = lambda a: ", ".join(f"{x['kind'][0]}{x['onset']} {x['evoked_free']:+.3f}" for x in a["trials"])
        rows_ = [("recorded", ", ".join(f"{x['kind'][0]}{x['onset']} {x['evoked_rec']:+.3f}" for x in full["trials"])),
                 ("full law", t(full)), ("W = 0", t(w0))] + ([("no network", t(nw))] if nw else [])
        right = (head("1. the trained law with W = 0") + "{\\scriptsize every edge weight set to 0 after training; the "
                 "inputs kept\\par}\\vspace{4pt}\n" + head("2. trained with no network from the start")
                 + "{\\scriptsize " + _tex(now) + ": the fair control (B and the cells' constants fitted without W)\\par}\\vspace{6pt}\n"
                 + head("evoked change at the held-out pulses, free rollout") + rows(rows_)
                 + rows([("gut / recorded", f"full {full['evoked_gut_free_over_rec']:.2f}, W = 0 {w0['evoked_gut_free_over_rec']:.2f}"
                          + (f", no network {nw['evoked_gut_free_over_rec']:.2f}" if nw else "")),
                         ("pattern r", f"full {full['pattern_r_gut']:+.2f}, W = 0 {w0['pattern_r_gut']:+.2f}"
                          + (f", no network {nw['pattern_r_gut']:+.2f}" if nw else ""))]
                        + ([("free R$^2$", f"full {A['full']['r2_denoised']:+.3f}, W = 0 {A['W0']['r2_denoised']:+.3f}")] if A else [])))
        mv = os.path.join(r["dir"], "results", "movie_W0.mp4")
        left = "\\panel{figs/ablation_" + name + ".png}"
        if os.path.exists(mv):
            shutil.copyfile(mv, os.path.join(PRES, "Movies", f"{name}_W0.mp4"))
            shutil.copyfile(mv.replace(".mp4", ".png"), os.path.join(PRES, "Movies", f"{name}_W0.png"))
            left = f"\\playmovie{{Movies/{name}_W0}}"
        elif A:
            shutil.copyfile(os.path.join(EXP, "data", "figs", f"ablation_{name}.png"), os.path.join(PRES, "figs", f"ablation_{name}.png"))
        else:
            left = ""
        out.append((f"{name}_controls", frame(f"{_tex(name)} with W = 0, and a law with no network: the gut response is the network's",
                                              left, right, f"{name}_freetrial.json, ablation_{name}.json", deck_title=dt)))
    if rn:
        out += [(k.replace("_movie", "_nonet_movie"), v.replace(run_deck(b, fish), net_deck(b, fish)))
                for k, v in slides_run(now, b, now, fish) if k.endswith("_movie")]
    return out


def figure_network(name, now, path):
    """CEDRIC'S CLAIM IN ONE FIGURE: top, the gut-responsive cells' mean dF/F around each held-out pulse inside the free
    rollout of the whole session (recorded, the full law, the same law with W = 0, the law trained with no network);
    bottom, the whole brain's mean dF/F over the session (exp17's white trace), the same four."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r, rn = landed(name), (landed(now) if now else None)
    ft = r["freetrial"]
    nw = rn["freetrial"]["arms"]["full"] if rn and rn["freetrial"] else None
    pre, post = ft["window"]
    dt = float(np.median(np.diff(np.load(os.path.join(GD, "graphs_data", "zebrafish",
                                                      f"{ft['recording']}_recording.npz"))["t_s"])))
    COL = {"rec": "#4caf50", "full": "white", "W0": "#ff5252", "now": "#42a5f5"}
    LAB = {"rec": "recorded", "full": "network law", "W0": "same law, W = 0", "now": "trained with no network"}
    trials = ft["arms"]["full"]["trials"]
    fig = plt.figure(figsize=(8.6, 6.4), facecolor="black")
    gs = fig.add_gridspec(2, len(trials), height_ratios=[1.25, 1], hspace=0.45, wspace=0.28)
    tt = (np.arange(-pre, post + 1)) * dt
    for j, t0 in enumerate(trials):
        a = fig.add_subplot(gs[0, j])
        _black(a)
        cur = {"rec": t0["trace_rec"], "full": t0["trace_free"],
               "W0": ft["arms"]["W0"]["trials"][j]["trace_free"]}
        if nw:
            cur["now"] = nw["trials"][j]["trace_free"]
        for k, y in cur.items():
            a.plot(tt, y, color=COL[k], lw=2.0 if k in ("rec", "full") else 1.4, label=LAB[k])
        a.axvline(0, color="#ffd54f", lw=0.8, ls=":")
        a.set_ylim(0, 0.7)                                  # one axis for the three pulses (Cedric, 2026-10-03)
        a.set_title(f"{'control (off the fish)' if t0['kind'] == 'control' else 'gut'} pulse, held out\n"
                    f"volume {t0['onset']}", fontsize=8)
        a.set_xlabel("s from the UV pulse", fontsize=7)
        if j == 0:
            a.set_ylabel(f"mean dF/F, {ft['responsive_cells']:,}\ngut-responsive cells", fontsize=7)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=8, labelcolor="white")
    b = fig.add_subplot(gs[1, :])
    _black(b)
    mf = np.load(os.path.join(r["dir"], "results", f"{name}_movie.npz"))
    m0 = np.load(os.path.join(r["dir"], "results", f"{name}_W0_movie.npz"))
    T = mf["r2_t"] * dt / 60
    b.plot(T, mf["mean_obs_all"], color=COL["rec"], lw=1.0, label=LAB["rec"])
    b.plot(T, mf["mean_pred_all"], color=COL["full"], lw=1.0)
    b.plot(T, m0["mean_pred_all"], color=COL["W0"], lw=1.0)
    if rn:
        mn = np.load(os.path.join(rn["dir"], "results", f"{now}_movie.npz"))
        b.plot(T, mn["mean_pred_all"], color=COL["now"], lw=1.0)
    tr = np.load(os.path.join(GD, "graphs_data", "zebrafish", f"{ft['recording']}_recording.npz"))["trials"]
    for f, site, held in zip(tr[:, 0], tr[:, 2], tr[:, 6]):
        b.axvline(f * dt / 60, color="#9e9e9e" if site == 1 else "#ffd54f", lw=1.2 if held else 0.5,
                  ls="--" if held else "-", alpha=0.8)
    b.set_xlabel("min (free rollout from the first volume; UV pulses: grey off the fish, yellow gut; dashed held out)", fontsize=7)
    b.set_ylabel("whole-brain\nmean dF/F", fontsize=7)
    fig.savefig(path, dpi=200, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def slide_network(name, now, b, fish=""):
    r, rn = landed(name), (landed(now) if now else None)
    if not (r and r["freetrial"] and "W0" in r["freetrial"]["arms"]):
        return ("", "")
    png = f"batch_{b}_network_{name}.png"
    figure_network(name, now, os.path.join(PRES, "figs", png))
    full, w0 = r["freetrial"]["arms"]["full"], r["freetrial"]["arms"]["W0"]
    nw = rn["freetrial"]["arms"]["full"] if rn and rn["freetrial"] else None
    ab = os.path.join(EXP, "data", f"ablation_{name}.json")
    A = json.load(open(ab)) if os.path.exists(ab) else None
    lk = os.path.join(EXP, "data", f"leak_{name}.json")
    LK = json.load(open(lk)) if os.path.exists(lk) else None
    fr = lambda a: f"{a['evoked_gut_free_over_rec']:.2f}"
    f3 = lambda v: f"{v:+.2f}" if v is not None and np.isfinite(v) else "--"
    keep_net, keep_w0 = full["evoked_gut_free_over_rec"], w0["evoked_gut_free_over_rec"]
    claim = (f"the network law keeps {keep_net:.2f} of the recorded gut response; with W = 0 it keeps {keep_w0:.2f}"
             + (f", trained with no network {nw['evoked_gut_free_over_rec']:.2f}" if nw else "")
             + (": the response is the network's" if keep_net > 3 * max(keep_w0, 0.01) else
                ": the network carries part of it"))
    prov = json.load(open(os.path.join(GD, "graphs_data", "zebrafish", f"{r['freetrial']['recording']}_recording.json")))
    swim = any(prov.get("swim_live", [False, False])) and "nosw" not in r["freetrial"]["recording"]
    right = (head("the claim") + "{\\scriptsize " + claim + "\\par}\\vspace{6pt}\n"
             + head("gut response / recorded (free rollout)") + rows(
                 [("network law", f"{fr(full)}, pattern r {full['pattern_r_gut']:+.2f} over the cells"),
                  ("W = 0", f"{fr(w0)}, pattern r {w0['pattern_r_gut']:+.2f}")]
                 + ([("no network", f"{fr(nw)}, pattern r {nw['pattern_r_gut']:+.2f}")] if nw else []))
             + head("whole session: brain-mean R$^2$ (per-cell R$^2$)") + rows(
                 [("network law", f"{f3(bm(r['dir'], name)[0])} ({f3(cell_r2(r['dir'], name))})"),
                  ("W = 0", f"{f3(bm(r['dir'], name + '_W0')[0])} ({f3(cell_r2(r['dir'], name + '_W0'))})")]
                 + ([("no network", f"{f3(bm(rn['dir'], now)[0])} ({f3(cell_r2(rn['dir'], now))})")] if rn else []))
             + head("what drives the rollout") + rows(
                 [("start", "one recorded volume, then the inputs only"),
                  ("leak check", (f"recording after the start zeroed or noise: max {max(x['max_abs_zeroed'] for x in LK['rows']):.1e} dF/F, "
                                  f"= run-to-run {max(x['max_abs_self'] for x in LK['rows']):.1e}") if LK else "not run on this law"),
                  ("swim", "still an input here (its swim channel has bouts)" if swim else "no swim input (none recorded)")]))
    return (f"batch_{b}_network_{name}", frame("Network dynamics",
                                        f"\\panel{{figs/{png}}}", right, f"{name}_freetrial.json, {now}_freetrial.json, ablation_{name}.json",
                                        deck_title=net_deck(b, fish)))


# WHICH GUT SPOT A SITE LABEL IS (inferred: the labels are not calibrated to the body; tools/exp20_body_render.py)
# (galvo positions per site, design.json, every glucose fish: site 1 off the fish; sites 2 and 3 one spot; site 5 another)
SITE_REGION = {(f"gutbrain_glucose_f{k}", st): {1: "off", 2: "gutA", 3: "gutA", 5: "gutB"}[st]
               for k in range(1, 7) for st in (1, 2, 3, 5)}
REGION_TXT = {"off": "UV off the fish (control)",
              "gutA": "UV on the gut: foregut? (the galvo spot of fish 1's sites 2 and 3; fish 4's site 2 has its voltage)",
              "gutB": "UV on the gut, another spot: midgut? (galvo x -1.2 V against -2.4 V; over all pulses it evokes +0.078 "
                      "dF/F against +0.184 at site 2, less than half, as the paper's midgut against foregut, Fig. 2g)"}


def figure_site(name, now, site, path):
    """One pulse site in the free rollout: top, the gut-responsive cells' mean dF/F around EVERY pulse of the site,
    mean +- SD over its pulses (recorded, the network law, the same law W = 0, the no-network law); bottom left, the body
    diagram with the site; bottom right, the whole session's brain mean with this site's pulses marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r, rn = landed(name), (landed(now) if now else None)
    ft = r["freetrial"]
    pre, post = ft["window"]
    z = np.load(os.path.join(GD, "graphs_data", "zebrafish", f"{ft['recording'].replace('_nosw', '')}_recording.npz"))
    dt = float(np.median(np.diff(z["t_s"])))
    tt = np.arange(-pre, post + 1) * dt
    COL = {"rec": "#4caf50", "full": "white", "W0": "#ff5252", "now": "#42a5f5"}
    LAB = {"rec": "recorded", "full": "network law", "W0": "same law, W = 0", "now": "trained with no network (no_W)"}
    sel = lambda arm_pulses, key: np.array([p_[key] for p_ in arm_pulses if p_["site"] == site])
    series = {"rec": sel(ft["arms"]["full"]["pulses"], "trace_rec"), "full": sel(ft["arms"]["full"]["pulses"], "trace_free"),
              "W0": sel(ft["arms"]["W0"]["pulses"], "trace_free")}
    if rn and rn["freetrial"] and "pulses" in rn["freetrial"]["arms"]["full"]:
        series["now"] = sel(rn["freetrial"]["arms"]["full"]["pulses"], "trace_free")
    n = len(series["rec"])
    fig = plt.figure(figsize=(9, 6.6), facecolor="black")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], width_ratios=[1.5, 2], hspace=0.30, wspace=0.10)
    a = fig.add_subplot(gs[0, :])
    _black(a)
    for k, Y in series.items():
        if len(Y) == 0:
            continue
        m_, s_ = Y.mean(0), Y.std(0)
        a.fill_between(tt, m_ - s_, m_ + s_, color=COL[k], alpha=0.18, lw=0)
        a.plot(tt, m_, color=COL[k], lw=2.0 if k in ("rec", "full") else 1.4, label=LAB[k])
    a.axvline(0, color="#ffd54f", lw=0.8, ls=":")
    a.set_ylim(0, 0.7)
    a.set_xlabel("s from the UV pulse", fontsize=8)
    a.set_ylabel(f"mean dF/F, {ft['responsive_cells']:,}\ngut-responsive cells", fontsize=8)
    a.set_title(f"site {site}: mean $\\pm$ SD over its {n} pulses, inside the free rollout", fontsize=10)
    a.legend(frameon=False, fontsize=8, labelcolor="white", loc="upper right")
    d = fig.add_subplot(gs[1, 0])
    d.axis("off")
    reg = SITE_REGION.get((ft["recording"].replace("_nosw", ""), site), "gutA")
    im = plt.imread(os.path.join(DATA, "anatomy", f"site_{reg}.png"))
    d.imshow(im[int(0.22 * im.shape[0]):int(0.82 * im.shape[0]), :int(0.92 * im.shape[1])])
    d.set_title({"off": "site: off the fish", "gutA": "site: gut (foregut?)", "gutB": "site: gut (midgut?)"}[reg],
                color="#ffeb3b", fontsize=9)
    b = fig.add_subplot(gs[1, 1])
    _black(b)
    mf = np.load(os.path.join(r["dir"], "results", f"{name}_movie.npz"))
    T = mf["r2_t"] * dt / 60
    b.plot(T, mf["mean_obs_all"], color=COL["rec"], lw=0.8)
    b.plot(T, mf["mean_pred_all"], color=COL["full"], lw=0.8)
    w0p = os.path.join(r["dir"], "results", f"{name}_W0_movie.npz")
    if os.path.exists(w0p):
        b.plot(T, np.load(w0p)["mean_pred_all"], color=COL["W0"], lw=0.8)
    if rn:
        b.plot(T, np.load(os.path.join(rn["dir"], "results", f"{now}_movie.npz"))["mean_pred_all"], color=COL["now"], lw=0.8)
    for f, st_ in zip(z["trials"][:, 0], z["trials"][:, 2]):
        if int(st_) == site:
            b.axvline(f * dt / 60, color="#ffd54f", lw=1.0, alpha=0.8)
    b.set_xlabel("min (the whole session; this site's pulses in yellow)", fontsize=7)
    b.set_ylabel("whole-brain\nmean dF/F", fontsize=7)
    fig.savefig(path, dpi=200, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    ev = {k: float(np.mean(Y[:, pre:pre + int(round(20 / dt))].mean(1) - Y[:, :pre].mean(1))) for k, Y in series.items() if len(Y)}
    return n, ev


def slides_sites(name, now, b, fish=""):
    """Cedric, 2026-10-03: the network slide split by pulse site (fish 1: sites 1, 2, 3; fish 4: 1, 5, 2)."""
    r = landed(name)
    if not (r and r["freetrial"] and "pulses" in r["freetrial"]["arms"]["full"]):
        return []
    ft = r["freetrial"]
    sites = list(dict.fromkeys(p_["site"] for p_ in ft["arms"]["full"]["pulses"]))
    out = []
    for site in sites:
        png = f"batch_{b}_site{site}_{name}.png"
        n, ev = figure_site(name, now, site, os.path.join(PRES, "figs", png))
        reg = SITE_REGION.get((ft["recording"].replace("_nosw", ""), site), "gutA")
        rec = ev.get("rec", 0.0)
        frac = lambda k: f"{ev[k]:+.3f} ({ev[k] / rec:.2f} of recorded)" if k in ev and abs(rec) > 1e-3 else (f"{ev[k]:+.3f}" if k in ev else "--")
        right = (head(f"site {site}: {REGION_TXT[reg].split(':')[0]}") + "{\\scriptsize " + REGION_TXT[reg] + "\\par}\\vspace{6pt}\n"
                 + head(f"evoked change, 0-20 s, mean of {n} pulses") + rows(
                     [("recorded", f"{rec:+.3f} dF/F"), ("network law", frac("full")), ("W = 0", frac("W0"))]
                     + ([("no\\_W", frac("now"))] if "now" in ev else [])))
        out.append((f"batch_{b}_site{site}_{name}", frame(f"site {site}", f"\\panel{{figs/{png}}}", right,
                                                          f"{name}_freetrial.json", deck_title=net_deck(b, fish) + f" $\\cdot$ site {site}")))
    return out


def slide_params(name, b, fish=""):
    """THE LEARNED CONSTANTS ON THE BRAIN (exp17's slide 35; tools/exp20_param_maps.py), the figure centred over the
    whole slide (Cedric, 2026-10-03)."""
    png = os.path.join(PRES, "figs", f"param_maps_{name}.png")
    if not os.path.exists(png) and landed(name):
        import exp20_param_maps
        exp20_param_maps.render(name)
    if not os.path.exists(png):
        return ("", "")
    return (f"{name}_params", f"% generated by tools/exp20_slides.py (learned constants of {name})\n"
            f"\\begin{{frame}}[t]{{batch {b} $\\cdot$ {fish or SHOWN[b][0][0]} $\\cdot$ the learned constants of {_tex(name)}}}\n"
            "\\vspace*{\\bandgap}\\vfill\n\\begin{center}\\includegraphics[width=0.96\\textwidth,height=0.80\\textheight,"
            f"keepaspectratio]{{figs/param_maps_{name}.png}}\\end{{center}}\n\\vfill\n\\end{{frame}}\n")


def slides_fish_compare(b="3"):
    """FISH TO FISH (Cedric, 2026-10-03): the same law on the six glucose fish. Slide 1, a table: session, cells,
    gut-responsive cells, per-site recorded evoked change and the law's share, W = 0's share, brain-mean R2 (masked law,
    unmasked law, W = 0). Slide 2, the gut-responsive cells' mean dF/F around the gut pulses (sites 2, 3, 5 pooled),
    mean +- SD, recorded against the law and W = 0, one panel per fish."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = {d["fish"]: d for d in json.load(open(os.path.join(DATA, "design.json"))) if d["condition"] == "glucose"}
    rows_, panels = [], []
    for k in range(1, 7):
        m_, u_ = f"gb_sx_f{k}_mask_nol1", ("gb_sx_nol1" if k == 1 else f"gb_sx_f{k}_nol1")
        r = landed(m_)
        if not (r and r["freetrial"] and "pulses" in r["freetrial"]["arms"]["full"]):
            continue
        ft = r["freetrial"]
        pre = ft["window"][0]
        dt = D[k]["volume_s"]
        ev = lambda Y: float(np.mean(Y[:, pre:pre + int(round(20 / dt))].mean(1) - Y[:, :pre].mean(1)))
        gut = [x for x in ft["arms"]["full"]["pulses"] if x["site"] != 1]
        gw0 = [x for x in ft["arms"]["W0"]["pulses"] if x["site"] != 1]
        Yr, Yl, Y0 = (np.array([x[key] for x in L]) for key, L in (("trace_rec", gut), ("trace_free", gut), ("trace_free", gw0)))
        er, el, e0 = ev(Yr), ev(Yl), ev(Y0)
        bz = json.load(open(os.path.join(DATA, f"baselines_gutbrain_glucose_f{k}.json")))
        rows_.append(f"{k} & {D[k]['minutes']:.0f} & {D[k]['cells'] / 1e3:.0f}k & {bz['gut_responsive']:,} & {len(gut)} & "
                     f"{er:+.3f} & {el / er:.2f} & {e0 / er:.2f} & {bm(r['dir'], m_)[0]:+.2f} & "
                     f"{bm(landed(u_)['dir'], u_)[0] if landed(u_) else float('nan'):+.2f} & {bm(r['dir'], m_ + '_W0')[0]:+.2f} \\\\")
        panels.append((k, Yr, Yl, Y0, dt, pre))
    if not rows_:
        return []
    # ONE SLIDE, one row per constant (Cedric, 2026-10-04: "merge 75 to 78, one row per heatmap type"; it was one slide
    # per constant, the six fish 3 x 2): tools/exp20_param_compare.py grid24 b3
    if os.path.exists(os.path.join(PRES, "figs", "param_24_b3.png")):
        PARAM_FISH[:] = [("97_param_fish_all", f"% generated by tools/exp20_slides.py (six-fish constants)\n\\begin{{frame}}[t]"
                          f"{{batch {b} $\\cdot$ six glucose fish, one law: the learned constants}}\n\\vspace*{{\\bandgap}}\\vfill\n"
                          "\\begin{center}\\includegraphics[width=\\textwidth,height=0.82\\textheight,keepaspectratio]"
                          "{figs/param_24_b3.png}\\end{center}\n\\vfill\n\\end{frame}\n")]
    rows_ = [x for x in rows_ if x is not None]
    panels = [x for x in panels if x is not None]
    body = ("{\\scriptsize\\begin{tabular}{@{}r@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{\\hspace{8pt}}r"
            "@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{\\hspace{8pt}}r@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{}}\n"
            "& & & gut- & gut & \\multicolumn{3}{c}{evoked, 0-20 s (free rollout)} & \\multicolumn{3}{c}{brain-mean R$^2$} \\\\\n"
            "fish & min & cells & responsive & pulses & recorded & law / rec. & W = 0 / rec. & masked & unmasked & W = 0 \\\\\n\\hline\n"
            + "\n".join(rows_) + "\n\\end{tabular}\\par}\\vspace{10pt}\n"
            "{\\scriptsize the nominal law (known ODE on the cell graph, no W prior) with each fish's own input mask; "
            "evoked = the gut-responsive cells' mean dF/F 0-20 s after a gut pulse (sites 2, 3, 5) minus the 10 s before, mean "
            "over the fish's gut pulses, inside the free rollout of its whole session; brain-mean R$^2$ of that rollout\\par}")
    out = [("95_fish_table", f"% generated by tools/exp20_slides.py (fish to fish)\n\\begin{{frame}}[t]{{batch {b} $\\cdot$ "
            f"six glucose fish, one law: the gut response and the network test, fish by fish}}\n\\vspace*{{\\bandgap}}"
            f"\\vspace*{{1.5\\baselineskip}}\n{body}\n\\end{{frame}}\n")]
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.2), facecolor="black", sharey=True)
    for a_, (k, Yr, Yl, Y0, dt, pre) in zip(ax.ravel(), panels):
        _black(a_)
        tt = (np.arange(Yr.shape[1]) - pre) * dt
        for Y, c, lab in ((Yr, "#4caf50", "recorded"), (Yl, "white", "law"), (Y0, "#ff5252", "same law, W = 0")):
            m, sd = Y.mean(0), Y.std(0)
            a_.fill_between(tt, m - sd, m + sd, color=c, alpha=0.15, lw=0)
            a_.plot(tt, m, color=c, lw=1.6, label=lab)
        a_.axvline(0, color="#ffd54f", lw=0.8, ls=":")
        a_.set_ylim(0, 0.7)
        a_.set_title(f"glucose fish {k}: {len(Yr)} gut pulses", fontsize=10)
        a_.set_xlabel("s from the pulse", fontsize=8)
    ax[0, 0].set_ylabel("mean dF/F, gut-responsive cells", fontsize=8)
    ax[1, 0].set_ylabel("mean dF/F, gut-responsive cells", fontsize=8)
    ax[0, 0].legend(frameon=False, fontsize=8, labelcolor="white")
    fig.tight_layout()
    fig.savefig(os.path.join(PRES, "figs", "96_fish_traces.png"), dpi=170, facecolor="black")
    plt.close(fig)
    out.append(("96_fish_traces", f"% generated by tools/exp20_slides.py (fish to fish traces)\n\\begin{{frame}}[t]{{batch {b} "
                f"$\\cdot$ six glucose fish, one law: the response to a gut pulse, recorded and learned}}\n\\vspace*{{\\bandgap}}\\vfill\n"
                "\\begin{center}\\includegraphics[width=0.96\\textwidth,height=0.80\\textheight,keepaspectratio]{figs/96_fish_traces.png}"
                "\\end{center}\n\\vfill\n\\end{frame}\n"))
    return out + PARAM_FISH


PARAM_FISH = []


def slide_overview():
    rw = md_rows()
    lines = []
    for b, (title, names, _, _) in BATCHES.items():
        for n in names:
            r = landed(n)
            if not r:
                continue
            N = numbers(r)
            fmt = lambda v: f"{v:+.2f}" if v is not None and np.isfinite(v) else "--"
            lines.append(f"{b}.{names.index(n) + 1} & {_tex(n)} & {fmt(N['short'])} & {fmt(N['trial'])} & {fmt(N.get('free_gut'))} & {fmt(N['brain_r2'])} \\\\")
    body = ("{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{8pt}}l@{\\hspace{8pt}}r@{\\hspace{8pt}}r@{\\hspace{8pt}}r@{\\hspace{8pt}}r@{}}\n"
            "arm & run & short & trial & free gut & brain R$^2$ \\\\\n\\hline\n" + "\n".join(lines) + "\n\\end{tabular}\\par}")
    return ("90_overview", (f"% generated by tools/exp20_slides.py (overview)\n\\begin{{frame}}[t]{{{DECK_TITLE} $\\cdot$ every landed run}}\n"
                            f"\\vspace*{{\\bandgap}}\n{body}\n\\end{{frame}}\n"))


# ============================================================================== batch 4: the 24 fish (Cedric, 2026-10-04)
# "compile the 24 fish results in the md and the pdf, reuse the per fish results slides template and add the siren
# slides (exp17's slide 17), comparison slides (exp17's slide 31, in two slides), a comparison flux movie (slides 28
# and 29), a comprehensive summary against the paper"
B4 = [("glucose", 6), ("glutamate", 5), ("Lglucose", 4), ("fish_water", 4), ("blood_glucose", 5)]
B4_NAME = {"glucose": "D-glucose", "glutamate": "glutamate", "Lglucose": "L-glucose", "fish_water": "fish water",
           "blood_glucose": "blood glucose"}
B4_FISH = [(c, k) for c, n in B4 for k in range(1, n + 1)]


def b4_runs(cond, k):
    """(the SIREN law, its no-network twin) of a batch-4 fish; the glucose network arms are batch 3's gb_sx_f<k>_mask_siren."""
    return ((f"gb_sx_f{k}_mask_siren", f"gb_b4_glucose_f{k}_now") if cond == "glucose"
            else (f"gb_b4_{cond}_f{k}", f"gb_b4_{cond}_f{k}_now"))


BATCHES["4"] = ("24 fish: the SIREN law and its no-network twin", [r for c, k in B4_FISH for r in b4_runs(c, k)],
                "gb_sx_f1_mask_siren", "gb_b4_glucose_f1_now")
BATCH_DECK["4"] = "batch 4 $\\cdot$ 24 fish $\\cdot$ the SIREN law (input mask) and its no-network twin"
REGION_SHORT = {"off": "off the fish", "gutA": "gut (foregut?)", "gutB": "gut (midgut?)", "vessel": "vessel (portal)"}


def site_region(rec, site):
    """WHICH SPOT A PULSE SITE IS, every condition (inferred from the galvo voltages of data/design.json; the labels are
    not calibrated to the body): site 1 off the fish in every condition -- in blood glucose by the block design and
    its flat response (its galvo read-back is one value for every site); the gut's spot of sites 2-4; a second gut spot
    (midgut?) where the beam sits 1 V further along x (glucose site 5, L-glucose site 4: galvo x -0.6 to -1.2 V against
    -1.9 to -2.4 V); the portal vessel in blood glucose (the paper's Fig. 5a)."""
    cond = re.match(r"gutbrain_(\w+?)_f\d+", rec).group(1)
    if site == 1:
        return "off"
    if cond == "blood_glucose":
        return "vessel"
    return "gutB" if (cond, site) in (("glucose", 5), ("Lglucose", 4)) else "gutA"


def frame_full(title, body, src, deck_title=None):
    return (f"% generated by tools/exp20_slides.py from {src} ({title})\n\\begin{{frame}}[t]{{{deck_title or DECK_TITLE}}}\n"
            f"\\vspace*{{\\bandgap}}\n{body}\n\\end{{frame}}\n")


def _small(t):
    """The text under a wide figure, one size down everywhere (Cedric, 2026-10-04: "fontsize too big" on these slides):
    heads scriptsize, everything else tiny."""
    t = t.replace("{\\normalsize\\textbf{", "{\\scriptsize\\textbf{").replace("{\\scriptsize ", "{\\tiny ")
    t = t.replace("{\\scriptsize\\begin", "{\\tiny\\begin")
    return "{\\tiny " + t + "}"


def frame_top(title, png, w, left, right, src, deck_title=None):
    """A wide figure across the slide, two text columns under it (the batch-4 comparison slides)."""
    left, right = _small(left), _small(right)
    body = (f"\\vspace*{{0.2\\baselineskip}}{{\\centering\\includegraphics[width={w}\\textwidth]{{{png}}}\\par}}\\vspace{{4pt}}\n"
            "\\begin{columns}[T,onlytextwidth]\n\\begin{column}{0.49\\textwidth}\n" + left + "\n\\end{column}\n"
            "\\begin{column}{0.49\\textwidth}\n" + right + "\n\\end{column}\n\\end{columns}")
    return frame_full(title, body, src, deck_title)


def _pic(png, h=0.80, w=0.96):
    return (f"\\vfill\n\\begin{{center}}\\includegraphics[width={w}\\textwidth,height={h}\\textheight,keepaspectratio]"
            f"{{{png}}}\\end{{center}}\n\\vfill")


def figure_sites_grid(name, now, path):
    """BATCH 4's SITE SLIDE, one per fish (the fish have 2 to 6 sites): top, one panel per pulse site, the
    gut-responsive cells' mean dF/F around every pulse of the site, mean +- SD over its pulses, inside the free rollout
    (recorded, the network law, the same law W = 0, the no-network twin); bottom left the body diagram of the fish's
    stimulated spot, bottom right the whole session's brain mean with the pulses, coloured by site."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r, rn = landed(name), (landed(now) if now else None)
    ft = r["freetrial"]
    pre, post = ft["window"]
    rec = ft["recording"].replace("_nosw", "")
    z = np.load(os.path.join(GD, "graphs_data", "zebrafish", f"{rec}_recording.npz"))
    dt = float(np.median(np.diff(z["t_s"])))
    tt = np.arange(-pre, post + 1) * dt
    n_ev = int(round(20 / dt))
    COL = {"rec": "#4caf50", "full": "white", "W0": "#ff5252", "now": "#42a5f5"}
    LAB = {"rec": "recorded", "full": "network law", "W0": "same law, W = 0", "now": "trained with no network (no_W)"}
    sites = sorted({p_["site"] for p_ in ft["arms"]["full"]["pulses"]})
    n = len(sites)
    fig = plt.figure(figsize=(max(9.0, 2.7 * n), 6.4), facecolor="black")
    gs = fig.add_gridspec(2, n, height_ratios=[1.35, 1], hspace=0.42, wspace=0.10)
    ev, regs = {}, {}
    for j, site in enumerate(sites):
        a = fig.add_subplot(gs[0, j])
        _black(a)
        sel = lambda arm, key: np.array([p_[key] for p_ in arm["pulses"] if p_["site"] == site])
        S = {"rec": sel(ft["arms"]["full"], "trace_rec"), "full": sel(ft["arms"]["full"], "trace_free")}
        if "W0" in ft["arms"]:
            S["W0"] = sel(ft["arms"]["W0"], "trace_free")
        if rn and rn["freetrial"] and "pulses" in rn["freetrial"]["arms"]["full"]:
            S["now"] = sel(rn["freetrial"]["arms"]["full"], "trace_free")
        for k, Y in S.items():
            if len(Y) == 0:
                continue
            m_, s_ = Y.mean(0), Y.std(0)
            a.fill_between(tt, m_ - s_, m_ + s_, color=COL[k], alpha=0.15, lw=0)
            a.plot(tt, m_, color=COL[k], lw=1.8 if k in ("rec", "full") else 1.2, label=LAB[k])
        a.axvline(0, color="#ffd54f", lw=0.8, ls=":")
        a.set_ylim(0, 0.7)
        reg = regs[site] = site_region(rec, site)
        a.set_title(f"site {site}: {REGION_SHORT[reg]}\n{len(S['rec'])} pulses", fontsize=9,
                    color="0.75" if reg == "off" else "#ffeb3b")
        a.set_xlabel("s from the UV pulse", fontsize=7)
        if j == 0:
            a.set_ylabel(f"mean dF/F, {ft['responsive_cells']:,}\ngut-responsive cells", fontsize=7)
            a.legend(frameon=False, fontsize=6.5, labelcolor="white", loc="upper left")
        else:
            a.set_yticklabels([])
        ev[site] = {k: float(np.mean(Y[:, pre:pre + n_ev].mean(1) - Y[:, :pre].mean(1))) for k, Y in S.items() if len(Y)}
    main_reg = max((r_ for r_ in regs.values() if r_ != "off"), key=list(regs.values()).count, default="off")
    nd = max(1, n // 3)
    d = fig.add_subplot(gs[1, :nd])
    d.axis("off")
    im = plt.imread(os.path.join(DATA, "anatomy", f"site_{main_reg}.png"))
    d.imshow(im[int(0.22 * im.shape[0]):int(0.82 * im.shape[0]), :int(0.92 * im.shape[1])])
    d.set_title(f"stimulated: {REGION_SHORT[main_reg]}", color="#ffeb3b", fontsize=9)
    b = fig.add_subplot(gs[1, nd:])
    _black(b)
    mf = np.load(os.path.join(r["dir"], "results", f"{name}_movie.npz"))
    T_ = mf["r2_t"] * dt / 60
    b.plot(T_, mf["mean_obs_all"], color=COL["rec"], lw=0.8)
    b.plot(T_, mf["mean_pred_all"], color=COL["full"], lw=0.8)
    w0p = os.path.join(r["dir"], "results", f"{name}_W0_movie.npz")
    if os.path.exists(w0p):
        b.plot(T_, np.load(w0p)["mean_pred_all"], color=COL["W0"], lw=0.7)
    if rn:
        b.plot(T_, np.load(os.path.join(rn["dir"], "results", f"{now}_movie.npz"))["mean_pred_all"], color=COL["now"], lw=0.7)
    for f, st_ in zip(z["trials"][:, 0], z["trials"][:, 2]):
        b.axvline(f * dt / 60, color="0.55" if site_region(rec, int(st_)) == "off" else "#ffd54f", lw=0.8, alpha=0.8)
    b.set_xlabel("min (the whole session; UV pulses: yellow stimulated, grey off the fish)", fontsize=7)
    b.set_ylabel("whole-brain\nmean dF/F", fontsize=7)
    fig.savefig(path, dpi=180, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    return ev, regs


def slide_sites_b4(name, now, fish):
    r = landed(name)
    if not (r and r["freetrial"] and "pulses" in r["freetrial"]["arms"]["full"]):
        return ("", "")
    png = f"batch_4_sites_{name}.png"
    ev, regs = figure_sites_grid(name, now, os.path.join(PRES, "figs", png))
    f = lambda e, k: f"{e[k]:+.3f}" if k in e else "--"
    fr = lambda e, k: f"{e[k] / e['rec']:.2f}" if k in e and abs(e["rec"]) > 0.02 else "--"
    body = "".join(f"{s} & {_tex(REGION_SHORT[regs[s]])} & {f(e, 'rec')} & {fr(e, 'full')} & {fr(e, 'W0')} & {fr(e, 'now')} \\\\\n"
                   for s, e in ev.items())
    right = (head("evoked change at every pulse, 0-20 s") + "{\\scriptsize\\begin{tabular}{@{}r@{\\hspace{4pt}}l@{\\hspace{5pt}}r"
             "@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{}}\n& & recorded & \\multicolumn{3}{c}{over the recorded} \\\\\n"
             "site & spot & dF/F & law & W = 0 & no\\_W \\\\\n\\hline\n" + body + "\\end{tabular}\\par}")
    return (f"b4_{name}_sites", frame("the pulse sites", f"\\panel{{figs/{png}}}", right, f"{name}_freetrial.json",
                                      deck_title=f"batch 4 $\\cdot$ {fish} $\\cdot$ network dynamics, every site"))


def slide_omega(name, fish):
    """exp17's slide 17 on a batch-4 fish: the learned modulation Omega_i(t) beside the recorded activity
    (tools/exp20_modulation.py)."""
    import shutil
    res = os.path.join(RUNS, name, "results")
    js = os.path.join(DATA, f"omega_{name}.json")
    if not (os.path.exists(os.path.join(res, "movie_omega.mp4")) and os.path.exists(js)):
        return ("", "")
    for ext in ("mp4", "png"):
        shutil.copyfile(os.path.join(res, f"movie_omega.{ext}"), os.path.join(PRES, "Movies", f"{name}_omega.{ext}"))
    O = json.load(open(js))
    right = (head("the learned modulation $\\Omega_i(t)$")
             + "{\\scriptsize each cell's message sum is multiplied by $\\Omega_i(t) = 1 + f(x_i, y_i, z_i, t)$, $f$ a SIREN "
               "of the position and the absolute time; 1 = no modulation, 0 = no input from the network at that frame\\par}\\vspace{6pt}\n"
             + rows([("mean over all", f"{O['mean']:.2f}"),
                     ("5th-95th pct", f"{O['p5']:.2f} .. {O['p95']:.2f}"),
                     ("brain mean over time", f"{O['brain_mean_min']:.2f} .. {O['brain_mean_max']:.2f}"),
                     ("below 1", f"{100 * O['frac_below_1']:.1f} \\% of cell-frames")])
             + f"{{\\tiny\\color{{gray}} the movie's {O['frames']} frames over the session; left the recorded dF/F, right "
               "$\\Omega$ per cell, one colour scale centred on 1\\par}\n")
    return (f"b4_{name}_omega", frame("the learned modulation of the messages", f"\\playmovie{{Movies/{name}_omega}}", right,
                                      "tools/exp20_modulation.py", left_gap=True,
                                      deck_title=f"batch 4 $\\cdot$ {fish} $\\cdot$ modulation"))


def slides_b4_fish(cond, k):
    """THE PER-FISH TEMPLATE (batches 2-3) on a batch-4 fish: the movie with the network test, the SIREN modulation,
    the learned constants, the pulse sites (one slide, every site), W = 0 and the no-network twin's movie. The curves
    slide (MSE by step) is left out: 24 fish x 8 slides, and its numbers are in the 24-fish table."""
    net, now = b4_runs(cond, k)
    fish = f"{B4_NAME[cond]} fish {k}"
    if not landed(net):
        return []
    out = [s for s in slides_run(net, "4", now, fish) if s[0].endswith("_movie")]
    out.append(slide_omega(net, fish))
    out.append(slide_params(net, "4", fish))
    out.append(slide_sites_b4(net, now, fish))
    out += slides_controls(net, now, "4", fish)
    return [(s if s.startswith("b4_") else f"b4_{s}", b) for s, b in out if b]


def _f(v, fmt="{:+.2f}"):
    return fmt.format(v) if isinstance(v, (int, float)) and v is not None and np.isfinite(v) else "--"


def slides_b4_compare():
    """THE 24 FISH SIDE BY SIDE (data/compare24.json, tools/exp20_compare24.py), exp17's slides 28, 29 and 31."""
    import shutil
    out = []
    cj = os.path.join(DATA, "compare24.json")
    if not os.path.exists(cj):
        return out
    C = json.load(open(cj))
    dt_ = BATCH_DECK["4"]
    halves = {"a": ("glucose", "glutamate"), "b": ("Lglucose", "fish_water", "blood_glucose")}
    for h, conds in halves.items():
        lines = []
        for r in C["rows"]:
            if r["condition"] not in conds:
                continue
            g = r.get("gut") or {}
            ratio = lambda k: (_f(g[k] / g["rec"], "{:.2f}") if g.get(k) is not None and g.get("rec") and abs(g["rec"]) > 0.02 else "--")
            lines.append(f"{_tex(B4_NAME[r['condition']])} {r['fish']} & {_f(r.get('minutes'), '{:.0f}')} & "
                         f"{(r.get('cells') or 0) / 1e3:.0f}k & {_f(r.get('gut_responsive'), '{:,}')} & {g.get('n', '--')} & "
                         f"{_f(r.get('top1_heldout'), '{:+.3f}')} & {_f(g.get('rec'), '{:+.3f}')} & {ratio('law')} & {ratio('W0')} & {ratio('now')} & "
                         f"{_f(r.get('brain_r2'))} & {_f(r.get('brain_r2_W0'))} & {_f(r.get('brain_r2_now'))} & "
                         f"{_f(r.get('omega_mean'), '{:.2f}')} \\\\")
        body = ("{\\tiny\\begin{tabular}{@{}l@{\\hspace{3pt}}r@{\\hspace{3pt}}r@{\\hspace{3pt}}r@{\\hspace{3pt}}r@{\\hspace{6pt}}r"
                "@{\\hspace{6pt}}r@{\\hspace{3pt}}r@{\\hspace{3pt}}r@{\\hspace{3pt}}r@{\\hspace{6pt}}r@{\\hspace{3pt}}r@{\\hspace{3pt}}r@{\\hspace{6pt}}r@{}}\n"
                "& & & gut- & & replicable & \\multicolumn{4}{c}{evoked, 0-20 s, free rollout} & \\multicolumn{3}{c}{brain-mean R$^2$} & \\\\\n"
                "fish & min & cells & responsive & pulses & held out & recorded & law/rec & W = 0 & no\\_W & law & W = 0 & no\\_W & $\\bar\\Omega$ \\\\\n"
                "\\hline\n" + "\n".join(lines) + "\n\\end{tabular}\\par}\\vspace{8pt}\n"
                "{\\tiny replicable: the top-1 \\% cells (chosen on the training pulses) on the HELD-OUT gut pulses, recorded (the "
                "paper's test of a response); evoked: the gut-responsive cells' mean dF/F 0-20 s after a stimulated pulse minus "
                "the 10 s before, mean over the fish's stimulated pulses (blood glucose: the vessel), inside the free rollout of "
                "the whole session; law / rec: the law's over the recorded; no\\_W: the twin trained with no network; "
                "$\\bar\\Omega$: the SIREN modulation's mean (1 = none)\\par}")
        title = ("the nutrients: D-glucose and glutamate" if h == "a" else
                 "the controls (L-glucose, fish water) and the vessel (blood glucose)")
        out.append((f"b4_table_{h}", frame_full(title, "\\vspace*{1.0\\baselineskip}" + body, "data/compare24.json",
                                                deck_title=dt_ + f" $\\cdot$ the table ({'1' if h == 'a' else '2'}/2)")))
    for h, lab in (("a", "D-glucose and glutamate"), ("b", "L-glucose, fish water, blood glucose")):
        if os.path.exists(os.path.join(PRES, "figs", f"compare24_traces_{h}.png")):
            out.append((f"b4_traces_{h}", frame_full(lab, _pic(f"figs/compare24_traces_{h}.png"), "tools/exp20_compare24.py",
                                                     deck_title=f"batch 4 $\\cdot$ the response to a stimulated pulse, every fish: {lab}")))
    T = C.get("tests", {})
    if os.path.exists(os.path.join(PRES, "figs", "compare24_conditions.png")):
        fm = T.get("foregut_midgut", [])
        wd = T.get("fwhm", {})
        from scipy.stats import mannwhitneyu
        md2 = lambda v: f"{np.median(v):.1f} s ({min(v):.0f}-{max(v):.0f}, n {len(v)})" if v else "--"
        g_, v_ = wd.get("gut", []), wd.get("vessel", [])
        pr = lambda a_, b_: f"{mannwhitneyu(a_, b_, alternative='greater').pvalue:.3f}" if a_ and b_ else "--"
        left = (head("Fig. 2g: the second gut spot against the first")
                + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{8pt}}r@{\\hspace{8pt}}r@{}}\nfish & recorded & the law \\\\\n\\hline\n"
                + "".join(f"{_tex(x['fish'])} & {x['site5_over_site2_rec']:.2f} & {_f(x['site5_over_site2_law'], '{:.2f}')} \\\\\n" for x in fm)
                + "\\end{tabular}\\par}\\vspace{3pt}{\\tiny the evoked change at the second spot (glucose site 5, L-glucose site 4, "
                  "galvo x 1 V further along) over the first's (site 2); the paper: the midgut evokes about a quarter of the "
                  "foregut (350 / 1440 cells); L-glucose, which evokes little, shows no difference\\par}")
        right = (head("Fig. 5d: the response width (FWHM)")
                 + rows([("gut (glucose, glutamate)", md2([x["fwhm_rec"] for x in g_]) + " recorded"),
                         ("", md2([x["fwhm_law"] for x in g_ if x.get("fwhm_law") is not None]) + " the laws"),
                         ("vessel (blood glucose)", md2([x["fwhm_rec"] for x in v_]) + " recorded"),
                         ("", md2([x["fwhm_law"] for x in v_ if x.get("fwhm_law") is not None]) + " the laws"),
                         ("gut longer than vessel", f"p {pr([x['fwhm_rec'] for x in g_], [x['fwhm_rec'] for x in v_])} recorded, "
                          f"{pr([x['fwhm_law'] for x in g_ if x.get('fwhm_law')], [x['fwhm_law'] for x in v_ if x.get('fwhm_law')])} "
                          "the laws (Mann-Whitney)")])
                 + "{\\tiny FWHM of the gut-responsive cells' mean response to the fish's stimulated pulses; the paper: "
                   "shorter for the portal system than for the gut lumen, p < 0.001\\par}")
        out.append(("b4_conditions", frame_top("the five conditions", "figs/compare24_conditions.png", 0.98, left, right,
                                               "tools/exp20_compare24.py", deck_title="batch 4 $\\cdot$ the five conditions: "
                                               "the paper's Fig. 2 and Fig. 5 on the recordings and the laws")))
    for h, lab in (("a", "D-glucose and glutamate"), ("b", "L-glucose, fish water, blood glucose")):
        if os.path.exists(os.path.join(PRES, "figs", f"param_24_{h}.png")):
            out.append((f"b4_params_{h}", frame_full(lab, _pic(f"figs/param_24_{h}.png", h=0.82, w=1.0), "tools/exp20_param_compare.py grid24",
                                                     deck_title=f"batch 4 $\\cdot$ the learned constants on every fish: {lab}")))
    mv = os.path.join(PRES, "Movies", "flow_fish.mp4")
    fj = os.path.join(DATA, "flow_gut_fish.json")
    if os.path.exists(os.path.join(PRES, "Movies", "omega_fish.mp4")):            # Cedric, 2026-10-04: before the flow
        out.append(("b4_omega_fish", frame_full("the SIREN modulation on each fish",
                    "\\vspace*{1.0\\baselineskip}\\centering\\playmovie[0.80\\textwidth]{Movies/omega_fish}\\par"
                    "{\\tiny\\color{gray} each cell: the learned modulation $\\Omega_i(t)$ of the messages into each cell (1 = none; "
                    "red above 1, blue below), the fish's whole session in the same 800 frames, its mean over cells and frames "
                    "beside its name; each fish on its own colour scale centred on 1 (tools/exp20\\_modulation.py, "
                    "exp17\\_modulation.py), head left\\par}",
                    "tools/exp20_flow_montage.py omega",
                    deck_title="batch 4 $\\cdot$ the learned modulation $\\Omega$ (SIREN) on each fish")))
    if os.path.exists(mv):
        out.append(("b4_flow_fish", frame_full("the flow on each fish",
                    "\\vspace*{1.2\\baselineskip}\\centering\\playmovie[0.74\\textwidth]{Movies/flow_fish}\\par"
                    "{\\tiny\\color{gray} each cell: excitatory flow above, inhibitory below, smoothed over 25 \\textmu m "
                    "(tools/exp20\\_wind.py); each fish its own session, the same 800 frames; the paper's stations in yellow\\par}",
                    "tools/exp20_flow_montage.py", deck_title="batch 4 $\\cdot$ the flow (the learned messages as wind) on each fish")))
    if os.path.exists(fj) and os.path.exists(os.path.join(PRES, "figs", "flow_gut_fish.png")):
        F = json.load(open(fj))
        short = lambda t: (t.replace("D-glucose fish ", "D-glc ").replace("glutamate fish ", "glut ")
                           .replace("L-glucose fish ", "L-glc ").replace("fish water fish ", "water ")
                           .replace("blood glucose fish ", "blood "))
        body = "".join(f"{_tex(short(x['fish']))} & {x['ratio_ex']:.2f} & {x['ratio_in']:.2f} & {x['toward_tail_gut']:+.2f} & "
                       f"{x['toward_tail_rest']:+.2f} \\\\\n" for x in F)
        tab = ("{\\tiny\\begin{tabular}{@{}l@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{\\hspace{8pt}}r@{\\hspace{6pt}}r@{}}\n"
               "& \\multicolumn{2}{c}{speed, gut / rest} & \\multicolumn{2}{c}{toward the tail (ex.)} \\\\\n"
               "fish & excitatory & inhibitory & gut windows & rest \\\\\n\\hline\n" + body + "\\end{tabular}\\par}")
        out.append(("b4_flow_gut", frame_top("the flow during the gut response", "figs/flow_gut_fish.png", 0.80, tab,
                    "{\\tiny the fish's brains differ, so no mean over fish (exp17's slide 29 averages graphs on one "
                    "brain); per fish, the frames 0-20 s after a stimulated pulse against the rest. speed: the mean flow speed "
                    "over the brain, gut windows over the rest. toward the tail: the net direction of the excitatory flow "
                    "along the body, +1 all toward the tail, -1 all toward the head (the head drawn left)\\par}",
                    "tools/exp20_wind.py gut_flow, exp20_flow_montage.py gut",
                    deck_title="batch 4 $\\cdot$ the flow during the gut windows against the rest")))
    ij = os.path.join(DATA, "integration.json")
    if os.path.exists(ij) and os.path.exists(os.path.join(PRES, "figs", "integration.png")):
        I = json.load(open(ij))
        rws = []
        for o in I["fish"]:
            pb = o.get("per_bin") or []
            gv = [b_.get("GV_over_GM") for b_ in pb]
            if any(v is not None for v in gv):
                rws.append((_tex(o["recording"].replace("gutbrain_", "")),
                            ", ".join(_f(v, "{:.2f}") for v in gv) + f" ({o['cells_used']:,} cells)"))
        left = ("{\\scriptsize the gut-responsive cells' gain in model correlation when the swim (GM) or the grating (GV) "
                "joins the gut regressors (the paper's distributed-lag regression), by fifth of the brain from head to tail "
                "(no atlas: the axis stands in for the regions); white, the median over the fish. The paper: OT (midbrain) "
                "GV/GM 1.18; PBN, medial idMO, DVC (hindbrain) 0.26-0.35\\par}")
        right = ("{\\tiny\\begin{tabular}{@{}l@{\\hspace{6pt}}l@{}}\nfish & GV / GM, head to tail fifth (cells) \\\\\n\\hline\n"
                 + "".join(f"{a_} & {b_} \\\\\n" for a_, b_ in rws) + "\\end{tabular}\\par}")
        out.append(("b4_integration", frame_top("Fig. 3d on the recordings", "figs/integration.png", 0.80, left, right,
                                                "tools/exp20_integration.py",
                                                deck_title="batch 4 $\\cdot$ visuo-motor integration of the gut-responsive cells "
                                                           "(the recordings, swim kept)")))
    out += slides_timescales()
    return out


def slides_timescales():
    """THE PAPER'S FIG. 4d-g (tools/exp20_timescales.py): glucose fish 1's five clusters by decay, recorded against the
    law's forecast from each pulse; then every fish side by side."""
    out = []
    ex = os.path.join(DATA, "timescales_gb_sx_f1_mask_siren.json")
    if os.path.exists(ex) and os.path.exists(os.path.join(PRES, "figs", "timescales_gb_sx_f1_mask_siren.png")):
        E = json.load(open(ex))
        left = (head("glucose fish 1, the law forecast from each pulse")
                + rows([("cells", f"{E['cells_used']:,} gut-responsive, {E['pulses']} stimulated pulses"),
                        ("time to peak", f"{E['rec']['t_peak_median']:.1f} s recorded, {E['law']['t_peak_median']:.1f} s the law (median)"),
                        ("decay", f"{E['rec']['decay_median']:.1f} s recorded, {E['law']['decay_median']:.1f} s the law "
                                  f"(10-90 \\%: {E['rec']['decay_p10']:.0f}-{E['rec']['decay_p90']:.0f} s recorded)"),
                        ("per cell, r", f"peak {E['r_cells']['peak']:+.2f}, decay {E['r_cells']['decay']:+.2f}, "
                                        f"time to peak {E['r_cells']['t_peak']:+.2f}")]))
        right = ("{\\scriptsize the paper (Fig. 4d-g): most cells start within 5 s; decays from about 10 s to over a minute; "
                 "five k-means clusters split by decay; every region holds every timescale. Here the law starts from the "
                 "recorded volume at the pulse and runs on the inputs only for the 56-s window; the clusters are the "
                 "recorded responses scaled by their own peak, the law's mean over the same cells dashed\\par}")
        out.append(("b4_timescales_f1", frame_top("the response time courses", "figs/timescales_gb_sx_f1_mask_siren.png", 0.92,
                                                  left, right, "tools/exp20_timescales.py",
                                                  deck_title="batch 4 $\\cdot$ the response time courses (Fig. 4d-g), glucose fish 1")))
    aj = os.path.join(DATA, "timescales_all.json")
    if os.path.exists(aj) and os.path.exists(os.path.join(PRES, "figs", "timescales_all.png")):
        A = json.load(open(aj))
        nut = [r for r in A if r["condition"] in ("glucose", "glutamate")]
        ves = [r for r in A if r["condition"] == "blood_glucose"]
        ctl = [r for r in A if r["condition"] in ("Lglucose", "fish_water")]
        rg = lambda v, f="{:.1f}": f"{f.format(min(v))}-{f.format(max(v))}" if v else "--"
        left = (head("every fish: medians over the cells")
                + rows([("nutrients, gut", f"decay {rg([r['decay_rec'] for r in nut])} s recorded, {rg([r['decay_law'] for r in nut])} s the laws"),
                        ("", f"time to peak {rg([r['t_peak_rec'] for r in nut])} s recorded, {rg([r['t_peak_law'] for r in nut])} s the laws"),
                        ("vessel", f"decay {rg([r['decay_rec'] for r in ves])} s recorded, {rg([r['decay_law'] for r in ves])} s the laws"),
                        ("controls", f"decay {rg([r['decay_rec'] for r in ctl])} s recorded, {rg([r['decay_law'] for r in ctl])} s the laws"),
                        ("over the window", f"{rg([100 * r['censored_rec'] for r in A], '{:.0f}')} \\% of the cells last the 56 s (recorded)")]))
        right = (head("per cell, recorded against the law (r)")
                 + rows([("peak", f"{min(r['r_peak'] for r in A):+.2f} to {max(r['r_peak'] for r in A):+.2f}"),
                         ("decay", f"{min(r['r_decay'] for r in A):+.2f} to {max(r['r_decay'] for r in A):+.2f}"),
                         ("time to peak", f"{min(r['r_t_peak'] for r in A):+.2f} to {max(r['r_t_peak'] for r in A):+.2f}")])
                 + "{\\scriptsize the laws keep which cells answer and by how much, and the order of the conditions' "
                   "timescales (the vessel and the controls short, the nutrients long); their decays run 2-4 s longer in "
                   "the nutrient fish; a cell's own timing is matched only loosely\\par}")
        out.append(("b4_timescales_all", frame_top("the time courses, every fish", "figs/timescales_all.png", 0.92, left, right,
                                                   "tools/exp20_timescales.py summary",
                                                   deck_title="batch 4 $\\cdot$ the response time courses, every fish: recorded against the laws")))
    return out


def _b4_stats():
    """The summary's numbers, from data/compare24.json, flow_gut_fish.json and integration.json (so a fish that lands
    later updates them)."""
    from scipy.stats import mannwhitneyu
    C = json.load(open(os.path.join(DATA, "compare24.json")))
    S = {"n": 0, "law": [], "W0": [], "now": [], "bm": [], "bm_W0": [], "bm_now": [], "frac": {}, "top1": {}, "fw": {},
         "fw_law": {}, "om": []}
    for r in C["rows"]:
        g, c = r.get("gut") or {}, r["condition"]
        if r.get("gut_responsive") and r.get("cells"):
            S["frac"].setdefault(c, []).append(100.0 * r["gut_responsive"] / r["cells"])
        if r.get("top1_heldout") is not None:
            S["top1"].setdefault(c, []).append(r["top1_heldout"])
        if g.get("rec") and abs(g["rec"]) > 0.02:
            S["n"] += 1
            S["law"].append(g["law"] / g["rec"])
            if g.get("W0") is not None:
                S["W0"].append(g["W0"] / g["rec"])
            if g.get("now") is not None:
                S["now"].append(g["now"] / g["rec"])
            if g.get("fwhm_rec") is not None and g["rec"] > 0.03:
                S["fw"].setdefault(c, []).append(g["fwhm_rec"])
                if g.get("fwhm_law") is not None:
                    S["fw_law"].setdefault(c, []).append(g["fwhm_law"])
        for k, kk in (("brain_r2", "bm"), ("brain_r2_W0", "bm_W0"), ("brain_r2_now", "bm_now"), ("omega_mean", "om")):
            if r.get(k) is not None and (kk != "bm_now" or r.get("brain_r2") is not None):
                S[kk].append(r[k])
    gut = S["fw"].get("glucose", []) + S["fw"].get("glutamate", [])
    ves = S["fw"].get("blood_glucose", [])
    gl = S["fw_law"].get("glucose", []) + S["fw_law"].get("glutamate", [])
    vl = S["fw_law"].get("blood_glucose", [])
    S["fwhm"] = {"gut": gut, "vessel": ves, "gut_law": gl, "vessel_law": vl,
                 "p": float(mannwhitneyu(gut, ves, alternative="greater").pvalue) if gut and ves else None,
                 "p_law": float(mannwhitneyu(gl, vl, alternative="greater").pvalue) if gl and vl else None}
    S["fm"] = [x for x in C["tests"]["foregut_midgut"] if x["fish"].startswith("glucose")]
    fj = os.path.join(DATA, "flow_gut_fish.json")
    S["flow"] = json.load(open(fj)) if os.path.exists(fj) else []
    ij = os.path.join(DATA, "integration.json")
    S["integ"] = json.load(open(ij))["fish"] if os.path.exists(ij) else []
    return S


def slides_summary():
    """THE SESSION AGAINST THE PAPER (Cedric, 2026-10-04: "make a comprehensive summary, compare the full experiment
    session insight with the original paper: what is in line, what differs, what new jobs are needed")."""
    if not os.path.exists(os.path.join(DATA, "compare24.json")):
        return []
    S = _b4_stats()
    md_ = lambda v, f="{:.2f}": f.format(np.median(v)) if v else "--"
    rg = lambda v, f="{:.2f}": f"{f.format(min(v))}-{f.format(max(v))}" if v else "--"
    fr = S["frac"]
    nut = [x for x in S["flow"] if x["fish"].startswith(("D-glucose", "glutamate"))]
    ctl = [x for x in S["flow"] if "control" in x["fish"]]
    fast = [x for x in S["flow"] if x["fish"].startswith("D-glucose") and x["ratio_ex"] > 1.05]
    glc = [x for x in S["flow"] if x["fish"].startswith("D-glucose")]
    tail = [o for o in S["integ"] if "glucose" in o["recording"] and (o.get("per_bin") or [{}])[-1].get("GV_over_GM") is not None]
    tail_ok = [o for o in tail if 0 < o["per_bin"][-1]["GV_over_GM"] < 0.5]
    head_ok = [o for o in tail if (o["per_bin"][0].get("GV_over_GM") or 0) > 1.0]
    tail_v = [o["per_bin"][-1]["GV_over_GM"] for o in tail_ok]
    p_ = lambda v: f"{v:.3f}" if v is not None and v >= 0.001 else ("< 0.001" if v is not None else "--")
    found = (
        "\\begin{itemize}\\setlength{\\itemsep}{2pt}\n"
        f"\\item \\textbf{{The gut response is the network's.}} In the {S['n']} fish where the law and its twin landed, the "
        f"SIREN law's free rollout -- one recorded volume, then the beam and the grating only -- keeps {md_(S['law'])} of the "
        f"gut-responsive cells' recorded response (median; {rg(S['law'])}); the same law with W = 0 keeps {md_(S['W0'])} "
        f"(at most {max(S['W0']):.2f}), the twin trained with no network {md_(S['now'])} ({rg(S['now'])}). Whole brain: "
        f"brain-mean R$^2$ {md_(S['bm'], '{:+.2f}')} for the law against {md_(S['bm_W0'], '{:+.2f}')} with W = 0 and "
        f"{md_(S['bm_now'], '{:+.2f}')} with no network.\n"
        "\\item \\textbf{Fish to fish.} The law fits glucose 1-3, glutamate 1-5 and blood glucose 1-4 (12-50 min sessions: "
        "brain-mean R$^2$ 0.61-0.93), not the long glucose 4-6 (83-107 min: 0.01-0.14) nor blood glucose 5 (0.12) nor fish "
        "water (0.15-0.51), even on exp17's rig: batch 3's guess, too few long-horizon updates, is not the whole story.\n"
        f"\\item \\textbf{{The law keeps the paper's spatial and route effects from the beam position alone:}} the second gut "
        f"spot (midgut?) evokes {rg([x['site5_over_site2_rec'] for x in S['fm']])} of the foregut spot's change recorded, "
        f"{rg([x['site5_over_site2_law'] for x in S['fm']])} in the law (glucose 4-6); the vessel's response is half as long "
        f"as the gut's: FWHM {md_(S['fwhm']['vessel'], '{:.0f}')} s against {md_(S['fwhm']['gut'], '{:.0f}')} s recorded "
        f"(p {p_(S['fwhm']['p'])}), {md_(S['fwhm']['vessel_law'], '{:.0f}')} s against {md_(S['fwhm']['gut_law'], '{:.0f}')} s "
        f"in the laws (p {p_(S['fwhm']['p_law'])}).\n"
        + (f"\\item \\textbf{{The flow runs head-ward.}} In the nutrient fish the learned excitatory messages run toward the "
           f"head ({rg([x['toward_tail_gut'] for x in nut], '{:+.2f}')} net toward the tail, -1 all head-ward), the direction "
           f"of the paper's DVC/AP to PBN to forebrain pathway; the gut windows speed it up in {len(fast)} of {len(glc)} glucose fish (x"
           f"{rg([x['ratio_ex'] for x in fast])}) but leave its direction as at rest; in the controls it is not faster "
           f"(x{rg([x['ratio_ex'] for x in ctl])}).\n" if nut else "")
        + "\\item \\textbf{The controls.} L-glucose and fish water have 7-13x fewer gut-responsive cells "
        f"({md_(fr.get('Lglucose', []), '{:.2f}')} \\% and {md_(fr.get('fish_water', []), '{:.2f}')} \\% of the brain, against "
        f"{md_(fr.get('glucose', []), '{:.1f}')} \\% in glucose); their laws reproduce what the UV evokes there, a smaller, "
        f"shorter change (FWHM {md_(S['fw'].get('Lglucose', []) + S['fw'].get('fish_water', []), '{:.0f}')} s).\n"
        f"\\item \\textbf{{The SIREN modulation}} $\\Omega$ averages {rg(S['om'])} (1 = none): it scales the messages up, "
        "most in the controls; its scale trades against W's, only its changes in time and space carry information.\n"
        "\\end{itemize}")
    out = [("99a_summary", frame_full("what the session found",
                                      "\\vspace*{0.4\\baselineskip}{\\fontsize{7}{8.6}\\selectfont " + found + "}",
                                      "data/compare24.json, flow_gut_fish.json, integration.json",
                                      deck_title="summary $\\cdot$ what 4 batches and 24 fish say"))]
    T_ = [("brain-wide gut response: DVC, PBN, medial idMO, OT, LH, nodose (Fig. 1e)",
           f"{md_(fr.get('glucose', []), '{:.1f}')} \\% of the cells respond to gut glucose ({rg(fr.get('glucose', []), '{:.1f}')} \\%); "
           "clusters at the inferred DVC/AP and both PBN", "in line (no atlas)"),
          ("glutamate evokes too (Fig. 2b)", f"the largest replicable held-out response: {md_(S['top1'].get('glutamate', []), '{:.2f}')} "
           f"dF/F against {md_(S['top1'].get('glucose', []), '{:.2f}')} in glucose", "in line"),
          ("L-glucose and fish water evoke little (Fig. 2c, d)",
           f"{md_(fr.get('Lglucose', []), '{:.2f}')} and {md_(fr.get('fish_water', []), '{:.2f}')} \\% of the cells respond; "
           f"replicable {md_(S['top1'].get('Lglucose', []), '{:.2f}')} / {md_(S['top1'].get('fish_water', []), '{:.2f}')} dF/F", "in line"),
          ("the midgut evokes less than the foregut (Fig. 2g: 350 / 1440 cells)",
           f"{rg([x['site5_over_site2_rec'] for x in S['fm']])} of the foregut's change; the law {rg([x['site5_over_site2_law'] for x in S['fm']])}",
           "in line, the law too"),
          ("gut cells integrate vision and motion: OT GV/GM 1.18, hindbrain 0.26-0.35 (Fig. 3d)",
           f"tail fifth {rg(tail_v)} in {len(tail_ok)} of {len(tail)} glucose fish; head fifth above 1 in {len(head_ok)}",
           "partly; laws not tested (swim off)"),
          ("most cells start within 5 s; decays 10 s to over a minute; every region holds every timescale (Fig. 4d-g)",
           (lambda A: (f"time to peak {min(r['t_peak_rec'] for r in A if r['condition'] in ('glucose', 'glutamate')):.0f}-"
                       f"{max(r['t_peak_rec'] for r in A if r['condition'] in ('glucose', 'glutamate')):.0f} s, decay "
                       f"{min(r['decay_rec'] for r in A if r['condition'] in ('glucose', 'glutamate')):.0f}-"
                       f"{max(r['decay_rec'] for r in A if r['condition'] in ('glucose', 'glutamate')):.0f} s (median per "
                       "nutrient fish), under 1 \\% of the cells past 56 s; the laws: amplitude per cell r 0.68-0.93, "
                       "decays 2-4 s longer")
            )(json.load(open(os.path.join(DATA, "timescales_all.json"))))
           if os.path.exists(os.path.join(DATA, "timescales_all.json")) else "--",
           "in line at the short end; laws partly"),
          ("a tightly coupled network across areas (Fig. 4, inter-region correlation)",
           f"W = 0 keeps {md_(S['W0'])} of the response, no network {md_(S['now'])}: cells talking make it", "in line, stronger"),
          ("portal (vessel) responses are shorter, FWHM (Fig. 5d)",
           f"{md_(S['fwhm']['vessel'], '{:.0f}')} against {md_(S['fwhm']['gut'], '{:.0f}')} s (p {p_(S['fwhm']['p'])}); the laws "
           f"{md_(S['fwhm']['vessel_law'], '{:.0f}')} against {md_(S['fwhm']['gut_law'], '{:.0f}')} s", "in line, the law too"),
          ("a ventro-medial idMO answers the vessel only (Fig. 5b)", "needs the regions: no atlas registration in the deposit",
           "open"),
          ("each cell modelled alone: per-cell regression kernels (Fig. 3)",
           "a per-cell law (the no-network twin) loses the response and the brain mean", "differs")]
    tab = ("{\\fontsize{6.5}{7.8}\\selectfont\\begin{tabular}{@{}p{0.33\\textwidth}@{\\hspace{8pt}}p{0.45\\textwidth}@{\\hspace{8pt}}p{0.17\\textwidth}@{}}\n"
           "\\textbf{the paper} & \\textbf{this session (recordings; laws)} & \\textbf{verdict} \\\\\n\\hline\n"
           + "".join(f"{a} & {b} & {c} \\\\[2pt]\n" for a, b, c in T_) + "\\end{tabular}\\par}")
    out.append(("99b_paper", frame_full("the paper against the session", "\\vspace*{0.6\\baselineskip}" + tab,
                                        "Chen 2026; data/compare24.json, integration.json",
                                        deck_title="summary $\\cdot$ Chen 2026 against the session")))
    jobs = ("\\begin{columns}[T,onlytextwidth]\n\\begin{column}{0.48\\textwidth}\n"
            + "{\\small\\textbf{what differs, or is not shown yet}}\\\\[4pt]\n" + "{\\fontsize{7}{8.6}\\selectfont\\begin{itemize}\\setlength{\\itemsep}{2pt}\n"
            "\\item the response needs the network: the paper's per-cell kernels fit each cell, our per-cell law cannot "
            "carry the response through a free rollout\n"
            "\\item the long sessions (83-107 min) and fish water are fitted badly (brain-mean R$^2$ under 0.4)\n"
            "\\item no region test in the laws: the swim is no input since batch 3 (the leak review), so Fig. 3d's "
            "integration is measured on the recordings only (G-region unset)\n"
            "\\item the stations are inferred from clusters: no atlas, no ventro-medial idMO, the nodose a guess\n"
            "\\end{itemize}}\n\\end{column}\n\\begin{column}{0.48\\textwidth}\n"
            + "{\\small\\textbf{new jobs}}\\\\[4pt]\n" + "{\\fontsize{7}{8.6}\\selectfont\\begin{itemize}\\setlength{\\itemsep}{2pt}\n"
            "\\item the long sessions: glucose 4 with twice the updates per stage, and on its first 41 min only "
            "(data length against drift)\n"
            "\\item the swim back as an input on glucose 1 and 3 (the paper's GM model): Fig. 3d in the laws, G-region\n"
            "\\item a second seed on glucose 1, glutamate 1, blood glucose 3: the seed spread of the gut response\n"
            "\\item data: the atlas registration of the 24 fish (Z-Brain or mapzebrain), if the authors have it: the "
            "regions, the vessel-only idMO, the nodose\n"
            "\\end{itemize}}\n\\end{column}\n\\end{columns}")
    out.append(("99c_next", frame_full("what differs and the next jobs", "\\vspace*{0.8\\baselineskip}" + jobs,
                                       "the session", deck_title="summary $\\cdot$ what differs, and the next jobs")))
    return out


def slides_b6():
    """BATCH 6 (Cedric, 2026-10-04, via exp17): exp17's multi-level mesh (graph: mesh, cell_ops.neuron_mesh_levels) on
    glucose fish 1 -- exp17's mesh slides twinned (tools/exp20_mesh_figures.py: the level panels, GraphCast Fig. 1e, g,
    and a turntable per level count) -- and the input mask closer to biology (tools/exp20_mask3d.py), at the deck's end."""
    out = []
    GM_ = {L_: json.load(open(os.path.join(DATA, f"gcmesh_{L_}.json"))) for L_ in (3, 4, 5)
           if os.path.exists(os.path.join(DATA, f"gcmesh_{L_}.json"))}
    um = lambda v: f"{v:.0f}" if v >= 10 else f"{v:.1f}"                        # noqa: E731
    dt_ = "batch 6 $\\cdot$ glucose fish 1 $\\cdot$ the multi-level mesh, as GraphCast's"
    if 5 in GM_ and os.path.exists(os.path.join(PRES, "figs", "gcmesh_5_levels.png")):
        g5 = GM_[5]["per_level"]
        cubes = " / ".join(f"{l_['cube_um']:g}" for l_ in g5[:-1])
        body = ("\\vspace*{1.0\\baselineskip}\\centering\\includegraphics[width=\\textwidth,height=0.56\\textheight,"
                "keepaspectratio]{figs/gcmesh_5_levels.png}\\par\\vspace{6pt}\n"
                "\\begin{columns}[T,onlytextwidth]\n\\begin{column}{0.49\\textwidth}\n{\\tiny "
                "\\textbf{GraphCast} (Lam et al.\\ 2023, Fig.~1g): $M^0$ the icosahedron, each level splitting every "
                "triangle in 4; the nodes NESTED, a coarse vertex a vertex of every finer level; the multi-mesh = the "
                "finest level's nodes and EVERY level's edges, the messages along all of them at once (Fig.~1e).\\par}\n"
                "\\end{column}\n\\begin{column}{0.49\\textwidth}\n{\\tiny "
                f"\\textbf{{On the cells}} (exp17, cell\\_ops.neuron\\_mesh\\_levels): $M^0$..$M^{{{len(g5) - 2}}}$ cubes of "
                f"{cubes} \\textmu m on one origin; a cube's node is the coarser level's node in it, else the cell nearest "
                f"its centroid; $M^{{{len(g5) - 1}}}$ every cell ({GM_[5]['neurons']:,}) -- the cells ARE the finest nodes, "
                "so no encoder / decoder; each level the Delaunay tetrahedralisation of its nodes, the edges longer than "
                "2 of its cubes dropped, every node keeping its shortest. White arrows: the edges INTO one node, a node "
                "of every level; each level a slab one of its cubes thick, from above, head up as exp17's.\\par}\n"
                "\\end{column}\n\\end{columns}\n")
        out.append(("b6_mesh_levels", frame_full("the multi-level mesh, level by level", body,
                                                 "data/gcmesh_5.json, figs/gcmesh_5_levels.png", deck_title=dt_)))
    for L_, g_ in GM_.items():
        if not os.path.exists(os.path.join(PRES, "Movies", f"gcmesh_{L_}.mp4")):
            continue
        pl_, sd_, mg_ = g_["per_level"], g_["sets_directed"], g_["merged"]
        tab_ = ("{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{5pt}}l@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{}}\n"
                "level & nodes of & nodes & edges & median \\\\\n\\hline\n"
                + "".join(f"$M^{i_}$ & {l_['label'].replace(' um', ' \\textmu m').replace('every neuron', 'every cell')} & "
                          f"{l_['nodes']:,} & {l_['edges']:,} & {um(l_['median_um'])} \\textmu m \\\\\n" for i_, l_ in enumerate(pl_))
                + "\\end{tabular}\\par}\\vspace{6pt}\n")
        long_lv = [f"$M^{i_}$" for i_, l_ in enumerate(pl_) if l_["cube_um"] >= 4 * g_["bin_um"]]
        mid_lv = [f"$M^{i_}$" for i_, l_ in enumerate(pl_) if g_["bin_um"] <= l_["cube_um"] <= 2 * g_["bin_um"]]
        sets_ = [("short", f"$M^{len(pl_) - 1}$"), ("mid", ", ".join(mid_lv)), ("long", ", ".join(long_lv) or "none")]
        right = (head(f"the multi-level mesh, {L_} levels") + tab_
                 + "{\\scriptsize the law's three W sets, every edge both ways:\\par}\\vspace{2pt}\n"
                 + rows([(k_, f"{lv_}: {sd_[k_]['edges']:,} edges, {sd_[k_]['per_neuron']:.2f} per cell, "
                              f"mean {um(sd_[k_]['mean_um'])} \\textmu m" if sd_[k_]["edges"] else
                          "none: no level of 64 \\textmu m or coarser") for k_, lv_ in sets_])
                 + "{\\scriptsize the merged multi-mesh is ONE graph over "
                 + (f"all {g_['neurons']:,} cells" if mg_["components"] == 1 else
                    f"{mg_['largest']:,} of the {g_['neurons']:,} cells")
                 + f"; fish 1's cells lie 4.0 \\textmu m apart (median nearest neighbour; ZAPBench's 7.5), so exp17's "
                   "16-\\textmu m finest cube is kept; batch 6 trains the SIREN law of batch 4 on it, the graph alone "
                   f"changed (gb\\_b6\\_glucose\\_f1\\_mesh{L_})\\par}}\n")
        out.append((f"b6_mesh{L_}", frame(f"the multi-level mesh, {L_} levels", f"\\playmovie{{Movies/gcmesh_{L_}}}", right,
                                          f"data/gcmesh_{L_}.json (tools/exp20_mesh_figures.py)", left_gap=True,
                                          deck_title=f"batch 6 $\\cdot$ glucose fish 1 $\\cdot$ the multi-level mesh, {L_} levels")))
    # THE UV INPUT CELLS, FOUR RULES (Cedric, 2026-10-04: "batches 1-5 did not exclude the cells the off-fish UV drives;
    # batch 6 excludes them but fixes 10 % and dissociates grating and gut cells; add the +2SD and +3SD rules"):
    # tools/exp20_mask_rules.py, exp20_input_mask.py --bio / --paper K
    mr = os.path.join(DATA, "mask_rules_f1.json")
    if os.path.exists(mr) and os.path.exists(os.path.join(PRES, "figs", "mask_rules_f1.png")):
        J = json.load(open(mr))
        r_ = np.load(os.path.join(DATA, "baselines_gutbrain_glucose_f1_cells.npz"))["responsive"]
        g = lambda k: J[k]
        body = ("\\vspace*{0.2\\baselineskip}\\vfill{\\centering\\includegraphics[width=\\textwidth,height=0.84\\textheight,"
                "keepaspectratio]{figs/mask_rules_f1.png}\\par}\\vfill")      # the maps large (Cedric, 2026-10-04)
        out.append(("b6_mask_rules", frame_full("the gut-input cells, six rules", body, "tools/exp20_mask_rules.py",
                                                deck_title="batch 6 $\\cdot$ glucose fish 1 $\\cdot$ the gut-input cells (the "
                                                           "law's UV-pulse and beam inputs enter them): six rules")))
    if os.path.exists(mr) and os.path.exists(os.path.join(PRES, "figs", "mask_rules_traces_f1.png")):
        J = json.load(open(mr))
        f3 = lambda v: f"{v:+.3f}" if v is not None else "--"
        left = ("{\\scriptsize\\textbf{what the gut-input cells of each rule do}}\\\\[2pt]{\\tiny the mean change of the rule's cells "
                "after the training gut pulses (red) and the off-fish control pulses (grey), 0-20 s, dF/F:\\par}\\vspace{2pt}"
                "{\\tiny\\begin{tabular}{@{}l@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{}}\nrule & gut & control \\\\\n\\hline\n"
                + "".join(f"{lab} & {f3(J[k]['evoked_gut'])} & {f3(J[k]['evoked_control'])} \\\\\n" for lab, k in
                          (("batches 1-5", "_batch1_5"), ("batch 6, 10 \\%", "_bio"), ("batch 6, 2 SD", "_paper2sd"),
                           ("batch 6, 3 SD", "_paper3sd"), ("atlas: AP + vagal ganglia", "_anat_apvg"),
                           ("atlas: + DVC", "_anat_dvc")) if k in J)
                + "\\end{tabular}\\par}\\vspace{2pt}{\\tiny batches 1-5's gut-input cells answer the light as much as the gut: "
                  "the gut response had to come from elsewhere. The paper's rule (2 / 3 SD): the gut + all-UV regression "
                  "tracks the cell above mode + 2 / 3 SD of all cells, better than all-UV alone, and the cell changes more "
                  "after gut than control pulses; no quota\\par}")
        arms = [("batches 1-5", "(batch 4: gb\\_sx\\_f1\\_mask\\_siren)", "mesh4"),
                ("batch 6, 10 \\%", "bio", "mesh4\\_bio"), ("batch 6, 2 SD", "paper2", "mesh4\\_paper2"),
                ("batch 6, 3 SD", "paper3", "mesh4\\_paper3"), ("atlas: AP + vagal ganglia", "anat\\_apvg", "mesh4\\_anat\\_apvg"),
                ("atlas: + DVC", "anat\\_dvc", "mesh4\\_anat\\_dvc")]
        right = ("{\\scriptsize\\textbf{the batch-6 arms on fish 1}}\\\\[2pt]{\\tiny\\begin{tabular}{@{}l@{\\hspace{6pt}}l@{\\hspace{6pt}}l@{}}\n"
                 "UV rule & neuron graph & 4-level mesh \\\\\n\\hline\n"
                 + "".join(f"{a} & {b} & {c} \\\\\n" for a, b, c in arms)
                 + "\\end{tabular}\\par}\\vspace{2pt}{\\tiny gb\\_b6\\_glucose\\_f1\\_<arm>; the mesh also at 3 and 5 levels "
                   "with the batch 1-5 mask; batch 4's SIREN law, the mask or the graph alone changed\\par}")
        out.append(("b6_mask_traces", frame_top("what each rule's gut-input cells do", "figs/mask_rules_traces_f1.png", 0.64, left,
                                                right, "tools/exp20_mask_rules.py", deck_title="batch 6 $\\cdot$ glucose "
                                                "fish 1 $\\cdot$ the gut-input cells of each rule, around gut and control pulses")))
    if os.path.exists(os.path.join(PRES, "figs", "06_kymo_uv.png")):      # slide 8, again, for the discussion
        out.append(("b6_kymo_uv_old", "% generated by tools/exp20_slides.py (slide 8 again)\n\\begin{frame}[t]"
                    "{batch 6 $\\cdot$ discussion $\\cdot$ the gut-input cells of batches 1-5 (slide 8)}\n"
                    "\\vspace*{\\bandgap}\n\\begin{center}\\includegraphics[width=0.98\\textwidth,height=0.80\\textheight,"
                    "keepaspectratio]{figs/06_kymo_uv.png}\\end{center}\n\\end{frame}\n"))
    out += slides_kymo(bio="_paper3sd")
    if os.path.exists(os.path.join(GD, "graphs_data", "zebrafish", "input_mask_gutbrain_glucose_f1_anat_apvg.npz")):
        out += slides_kymo(bio="_anat_apvg")
    if os.path.exists(os.path.join(PRES, "figs", "atlas_f1.png")):          # fish 1 in Z-Brain (BigWarp, Cedric's landmarks)
        out.append(("b6_atlas", f"% generated by tools/exp20_slides.py (fish 1 in the atlas)\n\\begin{{frame}}[t]"
                    "{batch 6 $\\cdot$ glucose fish 1 in the Z-Brain atlas (BigWarp, 30 landmarks)}\n\\vspace*{\\bandgap}\\vfill\n"
                    "\\begin{center}\\includegraphics[width=\\textwidth,height=0.84\\textheight,keepaspectratio]"
                    "{figs/atlas_f1.png}\\end{center}\n\\vfill\n\\end{frame}\n"))
    if os.path.exists(os.path.join(PRES, "Movies", "mask3d_f1.mp4")):
        body = ("\\vspace*{1.0\\baselineskip}{\\centering\\playmovie[0.92\\textwidth]{Movies/mask3d_f1}\\par}\\vspace{4pt}\n"
                "{\\tiny\\color{gray} the gut-input cells of the six rules in 3-D, turning once about the vertical: red the "
                "gut-input cells (the law's UV-pulse and beam inputs), blue the grating's, green the swim's (batches 1-5 only)\\par}")
        out.append(("b6_mask3d", frame_full("the input masks in 3-D", body, "tools/exp20_mask3d.py",
                                            deck_title="batch 6 $\\cdot$ glucose fish 1 $\\cdot$ the input masks in 3-D")))
    return out


def main():
    for d in ("slides", "Movies", "figs"):
        os.makedirs(os.path.join(PRES, d), exist_ok=True)
    deck = []
    for fn in (slide_paper, slide_deposit, slide_anatomy, slide_fish, slide_fig3d, slide_law, slide_baselines):
        try:
            deck.append(fn())
        except Exception as e:                      # one slide's failure must not lose the deck
            print(f"[slides] {fn.__name__} FAILED: {e}")
    try:
        deck += slides_kymo()
    except Exception as e:
        print(f"[slides] slides_kymo FAILED: {e}")
    hidden_b = []
    for b, (_, names, shown, now) in BATCHES.items():
        if b == "4" or not any(landed(n) for n in names):      # batch 4: its own slides below
            continue
        n0 = len(deck)
        deck.append(slide_batch(b))
        groups = SHOWN.get(b, [("", shown, shown, now)])
        for fish, run, ctrl, nw in groups:
            fish_tag = fish if len(groups) > 1 else ""
            deck += slides_run(run, b, nw, fish_tag)
            deck.append(slide_params(run, b, fish_tag))
            deck += slides_sites(ctrl, nw, b, fish_tag)            # replaces the one network slide (Cedric, 2026-10-03)
            deck += slides_controls(ctrl, nw, b, fish_tag)
        if b in HIDDEN_BATCHES:
            hidden_b += [st for st, _ in deck[n0:]]
    n0 = len(deck)
    deck += slides_fish_compare("3")
    if "3" in HIDDEN_BATCHES:
        hidden_b += [st for st, _ in deck[n0:]]
    for fn in (slides_b4_compare, slides_summary):
        try:
            deck += fn()
        except Exception as e:
            print(f"[slides] {fn.__name__} FAILED: {e!r}")
    for c, k in B4_FISH:                                   # batch 4 fish by fish, after the summary (an appendix)
        try:
            deck += slides_b4_fish(c, k)
        except Exception as e:
            print(f"[slides] batch 4 {c} {k} FAILED: {e!r}")
    try:
        deck += slides_b6()                                # batch 6 at the deck's end (Cedric, 2026-10-04)
    except Exception as e:
        print(f"[slides] slides_b6 FAILED: {e!r}")
    deck.append(slide_overview())
    out = []
    for stem, body in deck:
        if not body:
            print(f"[slides] {stem}: no data yet, skipped")
            continue
        open(os.path.join(PRES, "slides", f"{stem}.tex"), "w").write(no_stimuli_only(body))
        out.append(stem)
        print(f"[slides] {stem}")
    open(os.path.join(PRES, "slides", "all.tex"), "w").write(
        "% generated by tools/exp20_slides.py\n" + "".join(f"{'% ' if s in HIDDEN or s in hidden_b or any(h in s for h in HIDDEN_PAT) or HIDDEN_RE.match(s) else ''}\\input{{slides/{s}.tex}}\n"
                                                           for s in out))


if __name__ == "__main__":
    main()
