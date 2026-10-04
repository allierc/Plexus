"""Build the exp18 deck (one broadcast angle selects the law): figures, movies and slide bodies, no number typed.

    PYTHONPATH=src /workspace/.conda_envs/neural-graph-linux/bin/python tools/exp18_slides.py
    python presentation/fit_deck.py --dir experiments/exp18_phase_modulation/presentation --deck exp18.tex

The deck template is exp17's (`tools/exp17_slides.py`, `presentation/plexus_oct_26.tex`): picture or movie on the
LEFT, the specifics on the RIGHT in a `\\fitcol`. Every number is computed here from the file it describes -- the
connectome and afferent npz, the model and training specs, a run's `results/*.json` and the rulers' cache
`experiments/specs/exp18/measures.jsonl` (INSTRUCTION.md, "The one rule"). Devcontainer only.

    01_circuit     the 285-cell oculomotor connectome as a cell-type-pair matrix              (figure)
    02_operator    the broadcast angle: cos(varphi_ab - alpha_k), the three per-edge regimes  (figure)
    03_batch1      the four arms of batch 1, their context path and what they learn           (figure)
    <run>_movie    one slide per LANDED run: results/movie.mp4, its per-law errors and rulers  (movie)
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys

import numpy as np
import yaml

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EXP = os.path.join(ROOT, "experiments", "exp18_phase_modulation")
PRES = os.path.join(EXP, "presentation")
GD = os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
NEURAL = os.path.join(GD, "graphs_data", "neural")
DECK_TITLE = "one angle, two laws"
BATCH1 = ["exp18_m3_onehot_zf285", "exp18_m3_alpha_zf285", "exp18_m3_nocontext_zf285",
          "exp18_m3_alpha_randphi_zf285"]
ROLE = {"exp18_m3_onehot_zf285": "reference: one-hot context as an input line",
        "exp18_m3_alpha_zf285": "the claim: context only as the angle $\\alpha_k$",
        "exp18_m3_nocontext_zf285": "control: context reaches nothing",
        "exp18_m3_alpha_randphi_zf285": "exploration: varphi fixed and random",
        "exp18_m5_onehot_zf285": "six laws, reference: one-hot input",
        "exp18_m5_alpha_zf285": "six laws, six angles on one circle",
        "exp18_m5_alpha_randphi_zf285": "six laws, six angles, varphi fixed and random",
        "exp18_m3_alpha_fixedW_zf285": "two laws, the connectome weights FROZEN",
        "exp18_m5_alpha_phirandlearn_zf285": "six laws, varphi learned from a random start",
        "exp18_m5_alpha_randphi_fixedW_zf285": "six laws, random varphi, connectome FROZEN",
        "exp18_m5_alpha_randphi_s1_zf285": "six laws, random varphi, training seed 1",
        "exp18_m5_alpha_t2_randphi_zf285": "six laws on a torus: two angles per law",
        "exp18_m5_alpha_phirandlearn_fixedW_zf285": "six laws, varphi learned from random, connectome FROZEN",
        "exp18_m5_alpha_phirandlearn_s1_zf285": "six laws, varphi learned from random, seed 1",
        "exp18_m5_alpha_t2_phirandlearn_zf285": "six laws on a torus, varphi learned from random",
        "exp18_m3_qlbit_phirandlearn": "two laws on a Scholes QL bit (two 20-regular subgraphs)",
        "exp18_m3_qlbit_onehot": "QL bit, reference: one-hot input",
        "exp18_m3_qlbit_nocontext": "QL bit, control: context reaches nothing",
        "exp18_m3_qlbit_phirandlearn_fixedW": "QL bit, weights FROZEN: only angles, phases, in/out learned",
        "exp18_m3_qlbit_phirandlearn_s1": "QL bit, seed 1",
        "exp18_m3_alpha_s1_zf285": "two laws, alpha, seed 1 (the card's spread)",
        "exp18_m5_phirandlearn_jit10_zf285": "six laws, trained with angle noise 0.1 rad",
        "exp18_m5_phirandlearn_spread30_zf285": "six laws, trained with receiver spread 0.3 rad",
        "exp18_m5_phirandlearn_halfturn_zf285": "six laws, angles bounded to half a turn",
        "exp18_m3_alpha_jit10_zf285": "two laws, trained with angle noise 0.1 rad",
        "exp18_m3_alpha_gain_zf285": "two laws, gain-only control (no sign reversal)",
        "exp18_m5_phirandlearn_gain_zf285": "six laws, gain-only control (no sign reversal)",
        "exp18_m3_onehot_s1_zf285": "two laws, one-hot reference, seed 1",
        "exp18_m5_onehot_s1_zf285": "six laws, one-hot reference, seed 1",
        "exp18_m3_alpha_jit10_s1_zf285": "two laws, trained with angle noise, seed 1",
        "exp18_m5_phirandlearn_jit10_s1_zf285": "six laws, trained with angle noise, seed 1",
        "exp18_m5_phirandlearn_jit05_zf285": "six laws, trained with angle noise 0.05 rad",
        "exp18_m5_phirandlearn_jit20_zf285": "six laws, trained with angle noise 0.2 rad"}


def table_runs() -> list[str]:
    """Every run of the md's results table, in table order (`tools/exp_record.table`)."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from exp import load
    from exp_record import table
    _, body = load(os.path.join(ROOT, "experiments", "exp18_phase_modulation.md"))
    return [os.path.basename(r["_run"]) for r in table(body) if r["_group"] == "training"]
GREEN, INK, MUTED, RED, BLUE = "#2e8b4f", "white", "#9a9a9a", "#d0453a", "#4a90d9"


# ============================================================================== reading the files
def model_spec(path: str) -> dict:
    return yaml.safe_load(open(os.path.join(ROOT, path)))


def training_spec(name: str) -> dict:
    return yaml.safe_load(open(os.path.join(ROOT, "config", "training", "neural", f"{name}.yaml")))


def edges(rel: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(os.path.join(GD, "graphs_data", rel))
    return z["edge_index"], z["weights"]


def types(model: dict) -> list[tuple[str, int, str]]:
    return [(k, int(v["count"]), str(v["sign"])) for k, v in model["sets"]["neuron"]["types"].items()]


def signal_op(model: dict) -> dict:
    return next(o for o in model["operators"] if o["op"] == "neuron_signal")


def run_dir(name: str) -> str:
    return os.path.join(GD, "log", "training", "neural", name)


def measures(name: str) -> dict:
    """The rulers' cached values for a run (`experiments/specs/exp18/measures.jsonl`), flattened `measure.key`."""
    out = {}
    for f in (os.path.join(ROOT, "experiments", "specs", "exp18", "measures.jsonl"),
              os.path.join(ROOT, "experiments", "specs", "exp18", "extra_measures.jsonl")):   # runs gates.yaml does not card
        if not os.path.exists(f):
            continue
        for line in open(f):
            r = json.loads(line)
            if os.path.basename(str(r.get("run", r.get("spec", "")))) == name:
                for k, v in (r.get("value") or r.get("values") or {}).items():
                    out[f"{r.get('measure')}.{k}"] = v
    return out


# ============================================================================== slide helpers (exp17's)
def frame(title, left, right, src, concl="", box=0.62):
    """Picture LEFT, specifics RIGHT; then one blank line and the slide's one-line conclusion across the full width
    (Cedric 2026-10-04). The picture and the column are shortened to leave room for it."""
    if concl:
        left = f"\\setlength{{\\panelbox}}{{{box}\\textheight}}" + left
    col = "\\setlength{\\colheight}{0.66\\textheight}" if concl else ""
    tail = (f"\\par\\vspace{{\\baselineskip}}\n{{\\centering\\footnotesize {concl}\\par}}\n" if concl else "")
    return (f"% generated by tools/exp18_slides.py from {src} ({title})\n"
            f"\\begin{{frame}}[t]{{{DECK_TITLE} - {title}}}\n\\vspace*{{\\bandgap}}\n"
            "\\begin{columns}[T,onlytextwidth]\n\\begin{column}{0.58\\textwidth}\n\\vspace*{2\\baselineskip}\n"
            f"{left}\n\\end{{column}}\n\\begin{{column}}{{0.4\\textwidth}}\n\\vspace*{{2\\baselineskip}}\n{col}"
            f"\\fitcol{{%\n{right}}}\n\\end{{column}}\n\\end{{columns}}\n{tail}\\end{{frame}}\n")


def frame_wide(title, png, src, concl):
    """A wide figure across the whole slide, one blank line, the one-line conclusion."""
    return (f"% generated by tools/exp18_slides.py from {src} ({title})\n"
            f"\\begin{{frame}}[t]{{{DECK_TITLE} - {title}}}\n\\vspace*{{\\bandgap}}\n\\vspace*{{\\baselineskip}}\n"
            f"{{\\centering\\includegraphics[width=\\textwidth,height=0.72\\textheight,keepaspectratio]{{{png}}}\\par}}\n"
            f"\\vspace{{\\baselineskip}}\n{{\\centering\\footnotesize {concl}\\par}}\n\\end{{frame}}\n")


def head(s):
    return f"{{\\normalsize\\textbf{{{s}}}}}\\\\[4pt]\n"


def rows(pairs):
    body = "".join(f"{k} & {v} \\\\\n" for k, v in pairs)
    return ("{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{7pt}}l@{}}\n" + body + "\\end{tabular}\\par}\n"
            "\\vspace{8pt}")


def note(s):
    return f"{{\\scriptsize {s}\\par}}\\vspace{{6pt}}\n"


def tex(s: str) -> str:
    return s.replace("_", "\\_")


def tex_text(s: str) -> str:
    """Plain text (an md cell) made safe for LaTeX."""
    for a, b in (("\\", "/"), ("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_"), ("$", "\\$"),
                 ("~", "--"), ("^", "")):
        s = s.replace(a, b)
    return s.replace("`", "")


def one_line(s: str, n: int = 78) -> str:
    """The first clause of a sentence, at most n characters, cut at a clause boundary."""
    s = s.strip()
    for sep in (";", ": ", " -- "):
        if sep in s and s.index(sep) > 25:
            s = s[:s.index(sep)]
            break
    if len(s) > n:
        cut = max(s.rfind(",", 0, n), s.rfind(" (", 0, n))
        s = s[:cut] if cut > 40 else s[:n].rsplit(" ", 1)[0]
    return s.rstrip(" ,.") + "."


def _black(ax):
    ax.set_facecolor("black")
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


# ============================================================================== 01 the circuit
def slide_circuit() -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model = model_spec("config/neural/zf_circuit_285.yaml")
    T = types(model)
    nt = np.concatenate([[i] * c for i, (_, c, _) in enumerate(T)])          # type_layout: ordered
    sign = np.array([1.0 if s == "E" else -1.0 for _, _, s in T])
    ei, w = edges(model["sets"]["synapse"]["edges_file"])
    k = len(T)
    S, C = np.zeros((k, k)), np.zeros((k, k), int)
    for (j, i), x in zip(ei.T, w):                                            # j pre, i post
        S[nt[i], nt[j]] += abs(x) * sign[nt[j]]
        C[nt[i], nt[j]] += 1
    fig, ax = plt.subplots(figsize=(5.6, 3.9), facecolor="black")
    _black(ax)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("rkb", [RED, "black", BLUE])     # red I, black none, blue E
    Q = np.sign(S) * np.sqrt(np.abs(S))                                       # signed sqrt: the small blocks show
    m = np.abs(Q).max()
    ax.imshow(Q, cmap=cmap, vmin=-m, vmax=m)
    names = [f"{n.replace('_', ' ')} ({c} {sg})" for n, c, sg in T]
    ax.set_xticks(range(k), names, rotation=50, ha="right", color=INK, fontsize=8)
    ax.set_yticks(range(k), names, color=INK, fontsize=8)
    ax.set_xlabel("sender type (pre)", color=INK)
    ax.set_ylabel("receiver type (post)", color=INK)
    for a in range(k):
        for b in range(k):
            if C[a, b]:
                ax.text(b, a, str(C[a, b]), ha="center", va="center", fontsize=7,
                        color=INK)
    fig.savefig(os.path.join(PRES, "figs", "01_circuit.png"), dpi=160, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    ai, _ = edges(model["sets"]["afferent"]["edges_file"])
    mo, _ = edges(model["sets"]["motor"]["edges_file"])
    right = (head("the zebrafish oculomotor connectome") +
             rows([("cells", f"{len(nt)}"), ("synapses", f"{ei.shape[1]:,}"),
                   ("cell types", f"{k} ({sum(s == 'E' for _, _, s in T)} excitatory, "
                                  f"{sum(s == 'I' for _, _, s in T)} inhibitory)"),
                   ("input edges", f"{ai.shape[1]} (sensor lines {ai[0].min()}-{ai[0].max()} onto "
                                   f"{np.unique(ai[1]).size} AF5 cells)"),
                   ("read-out edges", f"{mo.shape[1]} onto 1 output")]) +
             note("matrix: summed Dale-signed weight per (receiver, sender) type pair, signed square root; blue "
                  "excitatory, red inhibitory; the number is the synapse count; (cells, E/I) beside each type"))
    return frame("the circuit", "\\panel{figs/01_circuit.png}", right, "config/neural/zf_circuit_285.yaml", box=0.54,
                 concl=f"{len(nt)} cells, {ei.shape[1]:,} Dale-signed synapses: fixed anatomy, only the response turns.")


# ============================================================================== 02 the operator
def slide_operator() -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    op = signal_op(model_spec("config/neural/zf_circuit_285_phase_k2.yaml"))
    a0 = [float(a) for a in op["alpha"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.2), facecolor="black",
                                   gridspec_kw={"width_ratios": [1, 1.4]})
    for ax in (ax1, ax2):
        _black(ax)
    t = np.linspace(0, 2 * np.pi, 400)
    ax1.plot(np.cos(t), np.sin(t), color=MUTED, lw=1)
    ph = 0.6
    ax1.plot([0, np.cos(ph)], [0, np.sin(ph)], color=GREEN, lw=2.2)
    ax1.text(1.08 * np.cos(ph), 1.08 * np.sin(ph), r"$\varphi_{ab}$", color=GREEN, fontsize=12)
    for i, a in enumerate(a0):
        ax1.plot([0, np.cos(a)], [0, np.sin(a)], color=[BLUE, RED][i % 2], lw=2.2)
        ax1.text(1.12 * np.cos(a), 1.12 * np.sin(a) - 0.05, rf"$\alpha_{i + 1}$", color=[BLUE, RED][i % 2], fontsize=12)
    ax1.set_aspect("equal")
    ax1.set_xlim(-1.4, 1.4)
    ax1.set_ylim(-1.4, 1.4)
    ax1.set_xticks([])
    ax1.set_yticks([])
    al = np.linspace(-np.pi, np.pi, 400)
    ax2.plot(al, np.cos(ph - al), color=INK, lw=1.8)
    ax2.axhline(0, color=MUTED, lw=0.6)
    for x, lab in ((ph, "expressed"), (ph - np.pi / 2, "silent"), (ph - np.pi, "reversed")):
        ax2.axvline(x, color=MUTED, lw=0.6, ls="--")
        ax2.text(x, 1.08, lab, color=INK, ha="center", fontsize=9)
    ax2.set_xlabel(r"broadcast angle $\alpha$ (rad)", color=INK)
    ax2.set_ylabel(r"$\cos(\varphi_{ab}-\alpha)$", color=INK)
    ax2.set_ylim(-1.2, 1.25)
    fig.savefig(os.path.join(PRES, "figs", "02_operator.png"), dpi=160, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    right = (head("one angle rotates every synapse") +
             "{\\scriptsize $\\dot v_i \\mathrel{+}= g_i \\sum_j W_{ij}\\tanh(v_j)\\,\\cos(\\varphi_{ab}-\\alpha_k)$\\par}"
             "\\vspace{6pt}\n" +
             note("$a$ the sender's type, $b$ the receiver's; $\\varphi_{ab}$ one phase per ordered type pair "
                  f"({op['n_types']}$\\times${op['n_types']} = {op['n_types'] ** 2} angles); $\\alpha_k$ one angle "
                  "per law $k$, broadcast to every synapse; $W_{ij}$ the connectome, sign fixed by the sender "
                  "(Dale)") +
             head("what it can do") +
             rows([("$\\alpha_k=\\varphi_{ab}$", "pair fully expressed"),
                   ("$\\alpha_k=\\varphi_{ab}\\pm\\pi/2$", "pair silent"),
                   ("$\\alpha_k=\\varphi_{ab}\\pm\\pi$", "pair sign-reversed (opponent receptors)")]) +
             head("starting point") +
             rows([("$\\alpha$", ", ".join(f"{a:.3f}" for a in a0) + " rad"),
                   ("$\\varphi$", str(op["phi_init"]))]) +
             note("GNN\\_Transformer.tex Part II (Eq. polar); operator \\texttt{neuron\\_signal[phase\\_rotated]}"))
    return frame("the broadcast angle", "\\panel{figs/02_operator.png}", right,
                 "config/neural/zf_circuit_285_phase_k2.yaml",
                 concl="One scalar per law: the cosine expresses, silences or reverses each type pair.")


# ============================================================================== 03 batch 1
def n_learned(spec: dict, model: dict) -> int:
    n = 0
    sizes = {"synapse": edges(model["sets"]["synapse"]["edges_file"])[0].shape[1],
             "afferent": edges(model["sets"]["afferent"]["edges_file"])[0].shape[1],
             "motor": edges(model["sets"]["motor"]["edges_file"])[0].shape[1],
             "neuron": int(model["sets"]["neuron"]["per_parent"]), "output": 1}
    op = signal_op(model)
    for e in spec["learnable"]:
        if "param" in e:
            n += len(op["alpha"]) if e["param"] == "alpha" else int(op["n_types"]) ** 2
        else:
            n += sizes[e["of"]]
    return n


def slide_batch1() -> str:
    lines = []
    for name in BATCH1:
        s = training_spec(name)
        m = model_spec(s["model"])
        op = signal_op(m)
        ai, _ = edges(m["sets"]["afferent"]["edges_file"])
        ctx = ("angle $\\alpha_k$" if op.get("model") == "phase_rotated"
               else ("input line" if ai[0].max() > 0 else "none"))
        lines.append((tex(name.replace("exp18_m3_", "").replace("_zf285", "")),
                      f"{ROLE[name]}; context: {ctx}; {ai.shape[1]} input edges; "
                      f"{n_learned(s, m):,} values learned"))
    task = yaml.safe_load(open(os.path.join(ROOT, "config", "task", "m3_int_delay.yaml")))
    tr = training_spec(BATCH1[0])["training"]
    right = (head("the task: two laws, one circuit") +
             rows([(t["name"], t["law"] + (f", {t['seconds']} s" if "seconds" in t else ""))
                   for t in task["teachers"]] +
                  [("stimulus", f"band-limited noise, {task['stimulus']['f_max_hz']} Hz, "
                                f"{task['general']['duration_s']} s trials"),
                   ("training", f"{tr['epochs']} epochs, lr {tr['lr']}, batch {tr['batch']}")]) +
             head("batch 1 (gpu\\_l4)") +
             "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{7pt}}p{5.2cm}@{}}\n" +
             "".join(f"{k} & {v} \\\\[2pt]\n" for k, v in lines) + "\\end{tabular}\\par}\n")
    figure_arms()
    return frame("batch 1", "\\panel{figs/03_batch1.png}", right, "config/training/neural/exp18_m3_*.yaml", box=0.58,
                 concl="The context reaches the circuit as an input line, as the angle alone, or not at all.")


def figure_arms() -> None:
    """One row per arm: where the stimulus and the context enter, read from each arm's model spec."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor="black")
    ax.set_facecolor("black")
    ax.axis("off")
    box = dict(boxstyle="round,pad=0.35", fc="black", lw=1.2)
    for r, name in enumerate(BATCH1):
        y = len(BATCH1) - 1 - r
        m = model_spec(training_spec(name)["model"])
        op = signal_op(m)
        ai, _ = edges(m["sets"]["afferent"]["edges_file"])
        phase = op.get("model") == "phase_rotated"
        ax.text(0.0, y, tex(name.replace("exp18_m3_", "").replace("_zf285", "")).replace("\\_", " "),
                color=INK, fontsize=10, va="center", ha="left", weight="bold")
        ax.text(2.2, y + 0.18, "stimulus", color=INK, fontsize=8, va="center", ha="center", bbox={**box, "ec": MUTED})
        ax.text(2.2, y - 0.18, "context", color=INK, fontsize=8, va="center", ha="center", bbox={**box, "ec": MUTED})
        ax.text(4.6, y, f"circuit\n{len(np.unique(ai[1]))} AF5 inputs", color=INK, fontsize=8, va="center",
                ha="center", bbox={**box, "ec": GREEN})
        ax.annotate("", (3.85, y + 0.08), (2.75, y + 0.18), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
        if ai[0].max() > 0:
            ax.annotate("", (3.85, y - 0.08), (2.75, y - 0.18), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.6))
            ax.text(3.3, y - 0.36, "input line", color=BLUE, fontsize=7, ha="center")
        elif phase:
            ax.annotate("", (5.9, y - 0.05), (2.75, y - 0.18),
                        arrowprops=dict(arrowstyle="->", color=RED, lw=1.6, connectionstyle="arc3,rad=0.25"))
            ax.text(5.95, y - 0.05, r"$\alpha_k$ rotates" + "\nevery synapse", color=RED, fontsize=7, va="center")
            if op.get("phi_init") == "uniform":
                ax.text(5.95, y + 0.3, r"$\varphi$ fixed, random", color=MUTED, fontsize=7, va="center")
        else:
            ax.text(3.3, y - 0.36, "reaches nothing", color=MUTED, fontsize=7, ha="center")
    ax.set_xlim(-0.1, 7.2)
    ax.set_ylim(-0.7, len(BATCH1) - 0.4)
    fig.savefig(os.path.join(PRES, "figs", "03_batch1.png"), dpi=160, facecolor="black", bbox_inches="tight")
    plt.close(fig)


# ============================================================================== one slide per landed run
def poster(mp4: str, png: str, at: float = 0.85) -> None:
    import imageio.v3 as iio
    meta = iio.immeta(mp4)
    nf = meta.get("nframes")
    n = int(nf) if nf and np.isfinite(nf) else int(round(meta.get("duration", 10) * meta.get("fps", 48)))
    iio.imwrite(png, iio.imread(mp4, index=max(0, int(n * at) - 1)))


def dark_movie(name: str) -> str | None:
    """The run's movie in the DECK's colours (black): one held-out trial per law, re-rolled from the checkpoint
    through the rulers' own restore and drawn by plot_trainer.rollout_movie(theme='dark'). Cached by name."""
    out = os.path.join(PRES, "Movies", f"{name}_dark.mp4")
    if os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(os.path.join(run_dir(name), "models", "best.pt")):
        return out
    import torch
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from exp_measures.common import TrainingRun
    from exp_measures.exp18 import _restore
    from plexus.tasks.plot_trainer import rollout_movie
    T = TrainingRun(run_dir(name))
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    TR, spec, sim, learn, ck, U, Y, cond = _restore(T, dev)
    pick = [int(np.flatnonzero(cond == k)[0]) for k in sorted(set(cond.tolist()))][:6]
    torch.manual_seed(0)
    with torch.no_grad():
        _, Yp = TR.rollout(sim, learn, U[pick], spec["task"], dev, grad=False)
    n = min(Yp.shape[-2], Y.shape[1])
    ch = int(spec["task"]["observe"].get("channel", 0))
    rollout_movie(U[pick][..., :1].cpu().numpy(), Y[pick][:, :n, :1].cpu().numpy(), Yp[..., :n, ch:ch + 1].cpu().numpy(),
                  out=out, dt=float(sim.dt), fps=48, theme="dark", n_show=len(pick), quiet=True,
                  title=f"{name}  (one held-out trial per law)")
    return out


def md_said() -> dict:
    """run name -> the md results row's 'what the result said' cell."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from exp import load
    from exp_record import table
    _, body = load(os.path.join(ROOT, "experiments", "exp18_phase_modulation.md"))
    return {os.path.basename(r["_run"]): r.get("what the result said", "") for r in table(body)}


def slide_run(name: str, said: dict) -> str | None:
    d = run_dir(name)
    tj = os.path.join(d, "results", f"{name}_test.json")
    if not (os.path.exists(tj) and os.path.exists(os.path.join(d, "models", "best.pt"))):
        return None
    mp4 = dark_movie(name)
    shutil.copyfile(mp4, os.path.join(PRES, "Movies", f"{name}.mp4"))
    poster(mp4, os.path.join(PRES, "Movies", f"{name}.png"))
    t = json.load(open(tj))
    rep = json.load(open(os.path.join(d, "results", "report.json")))
    names = t.get("cell_names") or []
    per = [(names[int(c)] if int(c) < len(names) else c, v) for c, v in t["normalised_per_cell"].items()]
    M = measures(name)
    right = (head(ROLE.get(name, tex(name))) +
             head("held-out error, of each law's own variance") +
             rows([(n, f"{v:.4f}") for n, v in per] + [("worst law", f"{max(v for _, v in per):.4f}")]))
    extra = [(k.split(".")[-1].replace("_", " "), f"{v:.3g}") for k, v in M.items()
             if k.split(".")[-1] in ("swap_ratio_min", "mode_switch", "gap_min", "alpha_sep") and v is not None]
    if extra:
        right += head("rulers (exp18.swap, exp18.spectrum)") + rows(extra)
    right += head("training") + rows([("epochs", str(rep["epochs"])), ("values learned", f"{rep['n_params']:,}"),
                                      ("time", f"{rep['seconds'] / 60:.0f} min"),
                                      ("best val", f"{rep['best_val_mse']:.5f}")])
    right += note("movie: one held-out trial per law; green ground truth, white the circuit")
    concl = tex_text(one_line(said.get(name, ""))) if said.get(name, "--") not in ("", "--") else ""
    body = frame(tex(name), f"\\playmovie{{Movies/{name}}}", right, f"log/training/neural/{name}", concl=concl)
    open(os.path.join(PRES, "slides", f"{name}_movie.tex"), "w").write(body)
    return f"{name}_movie"


# ============================================================================== the result figures (paper Fig. 1-3, dark)
def figure_slides() -> list[str]:
    """The paper's result figures, drawn black for the deck by tools/exp18_paper_figs.py / exp18_spice_*.py; every
    conclusion's number read from the summaries those tools wrote."""
    F = os.path.join(PRES, "figs")
    S = json.load(open(os.path.join(F, "fig1_summary.json")))
    best, crowd = S["exp18_m5_alpha_phirandlearn_zf285"], S["exp18_m5_alpha_zf285"]
    net = json.load(open(os.path.join(ROOT, "experiments", "exp18_phase_modulation", "spice",
                                      "network_exp18_m5_alpha_phirandlearn_zf285.json")))["rows"]
    six = json.load(open(os.path.join(F, "spice_six_circuits.json")))
    e_other = max(r["err"] for r in six if r["law"] != "delay")
    e_delay = next(r["err"] for r in six if r["law"] == "delay")
    sp = [r["spice_vs_target"] for r in net]
    lead = best["lead_re"]
    slides = [
        ("fig1_angles", "the learned angles", "fig1_angles.png",
         head("one circle, many laws") + note("each line: one law's broadcast angle; legend: angle and held-out error"),
         f"Laws {best['closest_pair_rad']:.2f} rad apart still differ; from $\\varphi = 0$ three laws crowd and fail."),
        ("fig1_eig", "eigenvalues per law", "fig1_eig.png",
         head("one wiring, different circuits") + note("eigenvalues of $J(\\alpha_k)$ per law; large: the three leading"),
         f"Each angle makes another circuit: leading eigenvalues separate (Re {min(lead):.1f}--{max(lead):.1f} 1/s)."),
        ("fig1_freq", "frequency responses", "fig1_freq.png",
         head("what each angle computes") + note("gain of output over stimulus; solid the circuit, dashed the law"),
         "One connectome, six filters: each angle turns the same wiring into its law."),
        ("spice_six", "six laws, six circuits", "spice_six_circuits.png",
         head("the lookup alternative") + note("one textbook circuit per law, simulated in ngspice against the law"),
         f"Six textbook circuits match their laws to {e_other:.0e}: a lookup, not one wiring."),
        ("spice_cell", "one cell, as netlisted", "spice_cell.png",
         head("the analog cell") + note("every symbol is an element of the generated SPICE file"),
         "Two synapse networks, two multipliers, RC and tanh: the knob is two DC voltages."),
        ("spice_traces", "one knob, six laws", "spice_traces.png",
         head("ngspice at six knob positions") + note("green the law, white the circuit"),
         f"One analog circuit performs all six laws as one knob turns (error {min(sp):.3f}--{max(sp):.3f})."),
    ]
    order = []
    for stem, title, png, right, concl in slides:
        if not os.path.exists(os.path.join(F, png)):
            continue
        open(os.path.join(PRES, "slides", f"{stem}.tex"), "w").write(
            frame_wide(title, f"figs/{png}", f"figs/{png}", concl))
        order.append(stem)
    return order


def main():
    os.makedirs(os.path.join(PRES, "slides"), exist_ok=True)
    order = []
    for stem, fn in (("01_circuit", slide_circuit), ("02_operator", slide_operator), ("03_batch1", slide_batch1)):
        open(os.path.join(PRES, "slides", f"{stem}.tex"), "w").write(fn())
        order.append(stem)
    order += figure_slides()
    said = md_said()
    for name in table_runs():
        s = slide_run(name, said)
        if s:
            order.append(s)
    open(os.path.join(PRES, "slides", "all.tex"), "w").write("".join(f"\\input{{slides/{s}}}\n" for s in order))
    print("[slides] " + ", ".join(order))


if __name__ == "__main__":
    main()
