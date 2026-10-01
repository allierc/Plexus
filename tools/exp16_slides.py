"""Build the exp16 deck (GraphCast on the redox organoid): figures, movies and slide bodies, no number typed by hand.

    PYTHONPATH=src:tools /workspace/.conda_envs/neural-graph-linux/bin/python tools/exp16_slides.py [--movie]
    python presentation/fit_deck.py --dir experiments/exp16_redox_graphcast/presentation --deck exp16.tex

The template is exp17's (`experiments/exp17_zapbench_graphcast/presentation/`, Cedric 2026-10-01): its preamble,
its frame (picture or movie LEFT, the specifics RIGHT in a `\\fitcol`) and its helpers, imported from
`tools/exp17_slides.py` rather than copied. Slides: the data; the references every number is read against; ONE
SLIDE PER MODEL (the GraphCast law, the multi-mesh, the per-voxel embedding, the global forcing I(t), the known
ODE); one slide per key run's movie with its scores; the ladder of all landed runs. Every number is computed here
from the frozen recording and the runs' own result files (INSTRUCTION.md, "The one rule"). Devcontainer only.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.pop("DISPLAY", None)
from exp17_slides import _tex, frame, head            # noqa: E402  exp17's template, reused


def wide_frame(title, left, right, src, deck_title, left_gap=False):
    """exp17's frame with the columns 0.50 / 0.48 instead of 0.58 / 0.40 (Cedric, 2026-10-01: "use larger column for
    the right one"): for the near-square figures (multi-mesh, known-ODE maps), which are height-bound and leave the left
    column's extra width blank. The movies (1200 x 560, width-bound) keep exp17's split."""
    return (frame(title, left, right, src, left_gap=left_gap, deck_title=deck_title)
            .replace("\\begin{column}{0.58\\textwidth}", "\\begin{column}{0.50\\textwidth}")
            .replace("\\begin{column}{0.4\\textwidth}", "\\begin{column}{0.48\\textwidth}"))


def rows(pairs):
    """exp17's two-column table, one size up (footnote, not script): Cedric asked for larger text on the right."""
    body = "".join(f"{k} & {v} \\\\\n" for k, v in pairs)
    return ("{\\footnotesize\\begin{tabular}{@{}l@{\\hspace{7pt}}l@{}}\n" + body + "\\end{tabular}\\par}\n"
            "\\vspace{8pt}")


def para(t):
    """A paragraph WRAPPED to the column (a fixed 6.2 cm box): set on one line, a long sentence made the column wide and
    shrank every slide's type, since the deck takes the densest slide's scale."""
    return "{\\footnotesize\\parbox{6.2cm}{" + t + "}\\par}\\vspace{6pt}\n"


def panel(png):
    """A left picture bounded at 0.78 of the slide height (exp17's \\panelboxfull), clear of the bottom band."""
    return "\\panelboxfull\\panel{" + png + "}"


def movie(stem):
    return "\\panelboxfull\\playmovie{" + stem + "}"
from plexus.tasks import field_recording as FR        # noqa: E402

EXP = os.path.join(ROOT, "experiments", "exp16_redox_graphcast")
PRES = os.path.join(EXP, "presentation")
GD = os.environ["GNN_OUTPUT_ROOT"]
RUNS = os.path.join(GD, "log", "training", "redox")
DECK = "GraphCast on a live organoid"
GC, KO = "GraphCast", "known ODE"   # Cedric, 2026-10-01: model and movie titles say WHICH law, not the deck's name


def family(run):
    return KO if "_ko" in run else GC
# THE MOVIE SLIDES, in deck order: (run, what it shows). The landed ones are drawn, the others skipped.
# (run, what it shows, the short difference that goes into the slide's title after the deck title)
MOVIES = [("hlo_f1_base", "frame 1 only, the GraphCast law on the voxel lattice", "the law, frame 1"),
          ("hlo_f1_mesh", "frame 1 only, + the 3-level multi-mesh", "+ multi-mesh"),
          ("hlo_f1_mesh_hash", "frame 1 only, + the per-voxel embedding (best frame-1 law)", "+ embedding"),
          ("hlo_f1_mesh_hash_I", "frame 1 only, + the global forcing I(t)", "+ global forcing I(t)"),
          ("hlo_fit_mesh_hash_I", "4 frames of history, mesh + embedding + I(t) (best 4-frame law)", "4 frames of history"),
          ("hlo_f1_mh_cur40", "frame 1 only, curriculum 1..40", "curriculum 1..40"),
          ("hlo_f1_mh_norm", "frame 1 only, GraphCast's normalisation", "normalisation"),
          ("hlo_ko_full", "the known ODE: relaxation + exchange + global forcing", "relaxation + exchange"),
          ("hlo_ko_nocoup", "the known ODE without exchange", "no exchange"),
          ("hlo_ko_uniform", "the known ODE with uniform exchange (no embedding)", "uniform exchange"),
          ("hlo_ko2_full", "the known ODE, fixed: (1 + beta) I(t) and a per-voxel barrier", "barrier + drive"),
          ("hlo_ko2_sparse", "the known ODE, fixed, thin membranes (L1 on the barrier)", "thin membranes")]
# THE LADDER: every phase-2 run, grouped, in the order a reader climbs it.
LADDER = [("4 frames of history, from frame 4", ["hlo_fit_base", "hlo_fit_mesh", "hlo_fit_mesh_hash", "hlo_fit_mesh_hash_I"]),
          ("frame 1 only", ["hlo_f1_base", "hlo_f1_mesh", "hlo_f1_mesh_hash", "hlo_f1_mesh_hash_I"]),
          ("frame 1, batch 8", ["hlo_f1_mh_s1", "hlo_f1_mh_s2", "hlo_f1_mh_cur40", "hlo_f1_mh_norm"]),
          ("the known ODE", ["hlo_ko_full", "hlo_ko_nocoup", "hlo_ko_uniform", "hlo_ko_noI"]),
          ("the known ODE, fixed", ["hlo_ko2_full", "hlo_ko2_drive", "hlo_ko2_barrier", "hlo_ko2_sparse"])]


def _black(ax):
    ax.set_facecolor("black")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("0.7")
    ax.tick_params(colors="0.8", labelsize=8)
    ax.xaxis.label.set_color("0.85")
    ax.yaxis.label.set_color("0.85")


def landed(name):
    d = os.path.join(RUNS, name, "results")
    p = os.path.join(d, f"{name}_test.json")
    if not os.path.exists(p):
        return None
    t = json.load(open(p))
    if t.get("mode") != "full":
        return None
    rep = os.path.join(d, "report.json")
    return {"name": name, "dir": os.path.join(RUNS, name), "test": t,
            "report": json.load(open(rep)) if os.path.exists(rep) else {}}


def poster(mp4, png, at=-1):
    import imageio.v2 as iio
    r = iio.get_reader(mp4)
    n = r.count_frames()
    iio.imwrite(png, r.get_data(n - 1 if at < 0 else min(at, n - 1)))


# ============================================================================== the data and the references
def references(rec, recd):
    """The frame-1 protocol's references, frames 2-69 from frame 1 (the numbers gates.yaml quotes)."""
    from plexus.trainer import _r2_parts
    obs, m = rec["ratio"][1:], rec["mask"][1:] & rec["mask"][0]
    obsd = recd["ratio"][1:]
    held = lambda v: np.broadcast_to(v, obs.shape)                                 # noqa: E731
    clim = np.where(rec["mask"].sum(0) > 0, (rec["ratio"] * rec["mask"]).sum(0) / np.maximum(rec["mask"].sum(0), 1), 0)
    out = {}
    for lab, v in (("frame 1 held", rec["ratio"][0]), ("frame 1 blurred (sigma 4) held", FR.smoothed(rec, 4.0)(0, 1)),
                   ("time-mean map (uses the future)", clim)):
        out[lab] = (_r2_parts(held(v), obs, m)[0], _r2_parts(held(v), obsd, m)[0])
    sig = FR.noise_floor(FR.structure_function(rec, np.arange(rec["ratio"].shape[0]), hs=(1, 2, 3)))["sigma"]
    sigd = FR.noise_floor(FR.structure_function(recd, np.arange(rec["ratio"].shape[0]), hs=(3, 4, 5, 6)),
                          fit_hs=(3, 4, 5, 6))["sigma"]
    out["noise ceiling"] = (1 - sig ** 2 / obs[m].var(), 1 - sigd ** 2 / obsd[m].var())
    return out, sig, sigd


def slides_data(rec, recd, prov, refs, sig, sigd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = []
    stem = os.path.join(PRES, "Movies", "00_redox_data")
    if "--movie" in sys.argv or not os.path.exists(stem + ".mp4"):
        FR.render_pair_3d(rec["ratio"], recd["ratio"], rec["mask"], rec["dx_um"], rec["dz_um"], stem + ".mp4",
                          labels=("recorded", "denoised: 3 voxels x 3 frames"),
                          times_min=rec["t_s"] / 60, show_r2=False)
        poster(stem + ".mp4", stem + ".png", at=0)
    T, Z, Y, X = rec["ratio"].shape
    right = (head("a live human liver organoid, 12 h washout")
             + rows([("imaging", "two-photon, 770 nm"), ("ratio", "NADH / FAD per voxel"),
                     ("volumes", f"{T}, every {prov['interval_s']['median'] / 60:.0f} min ({prov['span_h']:.1f} h)"),
                     ("grid", f"{Z} x {Y} x {X} voxels"),
                     ("voxel", f"{rec['dx_um']:.1f} x {rec['dx_um']:.1f} x {rec['dz_um']:.0f} um"),
                     ("tissue", f"{100 * rec['mask'].mean():.0f} \\% of voxels")])
             + head("the noise")
             + rows([("raw", f"sigma {sig:.3f}: {100 * (1 - refs['noise ceiling'][0]):.0f} \\% of the variance"),
                     ("denoised", f"sigma {sigd:.3f} (3 voxels x 3 frames)")]))
    out.append(frame("the data", movie("Movies/00_redox_data"), right,
                     "graphs_data/redox/hlo_washout_recording.npz", left_gap=True, deck_title=DECK))
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor="black")
    _black(ax)
    labs = list(refs)
    x = np.arange(len(labs))
    ax.bar(x - 0.18, [refs[k][0] for k in labs], 0.36, color="0.55", label="against the raw recording")
    ax.bar(x + 0.18, [refs[k][1] for k in labs], 0.36, color="tab:orange", label="against the denoised recording")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(x, [k.replace(" (", "\n(") for k in labs], fontsize=7, color="0.85")
    ax.set_ylabel("R2, frames 2-69 from frame 1")
    ax.legend(frameon=False, fontsize=8, labelcolor="0.85")
    fig.tight_layout()
    fig.savefig(os.path.join(PRES, "figs", "01_references.png"), dpi=150, facecolor="black")
    plt.close(fig)
    right = (head("given frame 1, predict frames 2-69")
             + rows([(_tex(k), f"raw {v[0]:+.3f}, denoised {v[1]:+.3f}") for k, v in refs.items()])
             + head("the metric")
             + "{\\footnotesize\\parbox{6.2cm}{R$^2$ of the 68-step free rollout, per frame (mean $\\pm$ SD), on the voxels that are tissue "
               "in frame 1 and in the target frame; against the raw recording and against its denoised version "
               "(fixed: 3 voxels $\\times$ 3 frames mean). Trained on all frames: in sample.}\\par}\n")
    # Cedric, 2026-10-01: the references slide is deleted from the deck (figs/01_references.png is still drawn)
    return out


# ============================================================================== one slide per model
def slide_graphcast(n_params):
    right = (head("the GraphCast law, one tick = 10 min")
             + "{\\footnotesize\\begin{tabular}{@{}l@{\\hspace{7pt}}l@{}}\n"
               "encode & $h_i=\\phi_v(r_i),\\; e_{ij}=\\phi_e(d_{ij})$ \\\\\n"
               "edges & $e_{ij}\\mathrel{+}=\\psi_l(e_{ij},h_i,h_j)$ \\\\\n"
               "nodes & $h_i\\mathrel{+}=\\chi_l\\big(h_i,\\sum_j e_{ij}\\big)$ \\\\\n"
               "decode & $r_i\\mathrel{+}=\\delta(h_i)$ \\\\\n\\end{tabular}\\par}\\vspace{8pt}"
             + head("in words")
             + "{\\footnotesize\\parbox{6.2cm}{each voxel's ratio changes by an amount a message-passing network computes from it and its "
               "6 lattice neighbours; the same law everywhere; $d_{ij}$ the step in um (5 in z, 3.3 in x, y); the "
               "decoder starts at 0, so the untrained law is persistence.}\\par}\\vspace{6pt}"
             + rows([("state", "one number per voxel (the ratio)"), ("weights", f"{n_params:,} (latent 16, 2 layers)"),
                     ("no", "hidden state, time-varying learnable")]))
    left = movie("Movies/hlo_f1_base") if os.path.exists(os.path.join(PRES, "Movies", "hlo_f1_base.mp4")) else ""
    return frame("model: GraphCast law", left, right, "src/plexus/operators/field_ops.py diffuse[graphcast]",
                 left_gap=True, deck_title=f"{GC} -- the law")


def slide_mesh(rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.models.registry import get_contract
    import plexus.operators  # noqa: F401
    C = get_contract("diffuse").implementations["graphcast"]
    o = C({"_at": "ratio", "latent": 8, "layers": 1, "spacing": [rec["dz_um"], rec["dx_um"], rec["dx_um"]],
           "mesh_levels": 3, "mesh_stride": [1, 2, 2]})
    G = o.mesh(tuple(rec["ratio"].shape[1:]), "cpu")
    m = rec["mask"][0]
    z = m.shape[0] // 2
    fig, ax = plt.subplots(figsize=(6.0, 5.0), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(np.where(m[z], rec["ratio"][0, z], np.nan), cmap="gray", alpha=0.5)
    cols = ("tab:cyan", "tab:orange", "tab:red")
    for k in range(3):
        s = 2 ** k
        yy, xx = np.meshgrid(np.arange(0, 64, s), np.arange(0, 64, s), indexing="ij")
        on = m[z][(2 * yy + 1).clip(0, 127), (2 * xx + 1).clip(0, 127)]
        ax.scatter(2 * xx[on] + 0.5, 2 * yy[on] + 0.5, s=4 * (k + 1) ** 2, c=cols[k],
                   label=f"level {k}: {2 * s * rec['dx_um']:.0f} um")
    ax.legend(frameon=False, fontsize=8, labelcolor="0.85", loc="lower left")
    yy, xx = np.nonzero(m[z])                       # crop to the organoid (+4 voxels) so the picture sits at the top
    ax.set_xlim(xx.min() - 4, xx.max() + 4)
    ax.set_ylim(yy.max() + 4, yy.min() - 4)
    ax.set_axis_off()
    fig.savefig(os.path.join(PRES, "figs", "02_multimesh.png"), dpi=150, facecolor="black", bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    right = (head("GraphCast's multi-mesh on the voxels")
             + rows([("grid", f"{m.size:,} voxels"), ("mesh", f"{G['n_mesh']:,} nodes, 1 per 1 x 2 x 2 voxels"),
                     ("levels", "3: edges of 1, 2, 4 mesh steps"),
                     ("reach", f"{2 * rec['dx_um']:.1f}, {4 * rec['dx_um']:.1f}, {8 * rec['dx_um']:.1f} um per layer"),
                     ("edges in", f"{G['g2m'][0].numel():,} grid to mesh"),
                     ("", f"{G['mm'][0].numel():,} in the mesh"), ("", f"{G['m2g'][0].numel():,} mesh to grid")])
             + "{\\footnotesize\\parbox{6.2cm}{encode the voxels onto the mesh, pass messages over all levels at once, decode back: a "
               "change can cross the organoid in a few layers instead of one voxel per layer.}\\par}\n")
    return wide_frame("model: multi-mesh", panel("figs/02_multimesh.png"), right, "DiffuseGraphCast.mesh",
                      deck_title=f"{GC} -- multi-mesh")


def slide_embedding():
    r = landed("hlo_f1_mesh_hash")
    if r is None:
        return ""
    import imageio.v2 as iio
    from PIL import Image
    a = np.asarray(Image.open(os.path.join(r["dir"], "results", "hlo_f1_mesh_hash_embedding3d.png")).convert("RGB"))
    b = np.asarray(Image.open(os.path.join(r["dir"], "results", "hlo_f1_mesh_hash_embedding_scatter.png")).convert("RGB")) \
        if os.path.exists(os.path.join(r["dir"], "results", "hlo_f1_mesh_hash_embedding_scatter.png")) else None
    if b is not None:
        h = min(a.shape[0], b.shape[0])
        a = np.concatenate([np.asarray(Image.fromarray(a).resize((int(a.shape[1] * h / a.shape[0]), h))),
                            np.asarray(Image.fromarray(b).resize((int(b.shape[1] * h / b.shape[0]), h)))], 1)
    iio.imwrite(os.path.join(PRES, "figs", "03_embedding.png"), a)
    cl = json.load(open(os.path.join(r["dir"], "results", "hlo_f1_mesh_hash_clusters.json")))
    right = (head("a per-voxel embedding: heterogeneity")
             + "{\\footnotesize\\parbox{6.2cm}{each voxel's encoder also reads $a_i$, 6 numbers of its own: one law, per-voxel variation "
               "(GraphCast's static features, learned). $a_i$ is an Instant-NGP hash over the voxel's position, 3 levels, "
               "finest 26 um.}\\par}\\vspace{6pt}"
             + head("does it find cells?")
             + rows([("clusters", f"{cl['k']} (silhouette {cl['silhouette']:.2f})"),
                     ("contiguity", f"{cl['coherence']:.2f} (shuffled {cl['coherence_null']:+.2f})"),
                     ("8 clusters", f"median domain {cl['domain_um3_median_k8']:.0f} um$^3$ (a cell: 2,000-5,000)")]))
    return frame("model: per-voxel embedding", panel("figs/03_embedding.png"), right,
                 "log/training/redox/hlo_f1_mesh_hash", left_gap=True, deck_title=f"{GC} -- per-voxel embedding")


def slide_forcing(rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    r = landed("hlo_f1_mesh_hash_I")
    if r is None:
        return ""
    ck = torch.load(os.path.join(r["dir"], "models", "best.pt"), weights_only=False, map_location="cpu")
    I = ck["fitted"]["diffuse.I"].numpy()
    th = rec["t_s"] / 3600
    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor="black")
    _black(ax)
    ax.plot(th[:len(I)], I, color="tab:orange", lw=1.5, label="learned I(t)")
    ax2 = ax.twinx()
    ax2.plot(th, rec["trend_ratio"], color="0.75", lw=1, label="lab's trend (mean NADH / mean FAD)")
    ax2.tick_params(colors="0.8", labelsize=8)
    ax.set_xlabel("hours")
    ax.set_ylabel("I(t)")
    ax2.set_ylabel("organoid ratio", color="0.85")
    fig.legend(frameon=False, fontsize=8, labelcolor="0.85", loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(PRES, "figs", "04_forcing.png"), dpi=150, facecolor="black")
    plt.close(fig)
    right = (head("a global forcing I(t)")
             + "{\\footnotesize\\parbox{6.2cm}{one number per volume, the same for every voxel: it can move the organoid's level "
               "(the washout), never draw a pattern.}\\par}\\vspace{6pt}"
             + rows([("values", f"{len(I)}, one per 10-min volume"), ("range", f"{I.min():+.3f} to {I.max():+.3f}")]))
    return frame("model: global forcing I(t)", panel("figs/04_forcing.png"), right,
                 "log/training/redox/hlo_f1_mesh_hash_I", left_gap=True, deck_title=f"{GC} -- global forcing I(t)")


def slide_known_ode(rec):
    eq = ("{\\footnotesize $\\displaystyle \\frac{dr_i}{dt}=\\frac{r^*_i-r_i}{\\tau_i}+\\sum_{j}\\kappa_{ij}(r_j-r_i)+(1+\\beta_i) I(t)$,"
          "\\quad $\\kappa_{ij}=\\kappa_{\\rm axis}\\,e^{-(b_i+b_j)}$\\par}\\vspace{6pt}")
    words = rows([("$r^*_i$", "set point"), ("$1/\\tau_i$", "relaxation rate"),
                  ("$\\kappa_{ij}$", "exchange, low across membranes"), ("$\\beta_i$", "sensitivity to the global forcing")])
    run = "hlo_ko2_full" if landed("hlo_ko2_full") else "hlo_ko_full"
    r = landed(run)
    left = ""
    extra = para("training")
    if r is not None:
        png, _, kap = learned_maps(run, rec, "05_known_ode.png")
        left = panel(png)
        t = r["test"]
        extra = rows([("kappa", ", ".join(f"{k:.3f}" for k in kap) + " per 10 min (z, y, x)"),
                      ("R$^2$", f"raw {t['r2']:+.3f}, denoised {t.get('r2_denoised', float('nan')):+.3f}"),
                      ("numbers", f"{r['report'].get('n_params', 0):,}")])
    right = (head("the interpretable rival: a known ODE") + (ko_equation(run) + ko_words(run) if r is not None else eq + words)
             + extra)
    return wide_frame("model: known ODE", left, right, f"log/training/redox/{run}", deck_title=f"{KO} -- the law")


def ko_equation(run):
    """The known ODE AS TRAINED for this run, written from its own spec (config/training/redox/<run>.yaml) and the model
    file it names: a term is written only when the run learns it (kappa not learned = 0 = no exchange), the gate is the
    one the operator reads (an embedding -> exp(-|a_i - a_j|^2), a barrier -> exp(-(b_i + b_j))), and the drive is
    (1 + beta_i) I(t) under `drive_offset: true`, else beta_i I(t) -- DiffuseKnownODE.rhs, field_ops.py."""
    import yaml
    spec = yaml.safe_load(open(os.path.join(ROOT, "config", "training", "redox", f"{run}.yaml")))
    op = next(o for o in yaml.safe_load(open(os.path.join(ROOT, spec["model"])))["operators"]
              if o.get("model") == "known_ode")
    lf = {e["field"]: e for e in spec["learnable"] if "field" in e}
    lp = {e["param"] for e in spec["learnable"] if "param" in e}
    terms = ["\\frac{r^*_i-r_i}{\\tau_i}"]
    gate = ""
    if "kappa" in lp:
        if op.get("barrier") in lf:
            kij, gate = "\\kappa_{ij}", "$\\kappa_{ij}=\\kappa_{\\rm axis}\\,e^{-(b_i+b_j)}$"
        elif op.get("embedding") in lf:
            kij, gate = "\\kappa_{ij}", "$\\kappa_{ij}=\\kappa_{\\rm axis}\\,e^{-\\|a_i-a_j\\|^2}$, $a_i$ the learned embedding"
        else:
            kij = "\\kappa_{\\rm axis}"
        terms.append("\\sum_{j}" + kij + "(r_j-r_i)")
    if "I" in lp:
        terms.append("(1+\\beta_i)\\,I(t)" if op.get("drive_offset") else "\\beta_i\\,I(t)")
    eq = "{\\footnotesize $\\displaystyle \\frac{dr_i}{dt}=" + "+".join(terms) + "$\\par}\\vspace{3pt}\n"
    notes = [gate] if gate else []
    if "kappa" not in lp:
        notes.append("no exchange: $\\kappa$ not learned, held at 0")
    l1 = (lf.get(op.get("barrier"), {}).get("prior") or {}).get("l1")
    if l1:
        notes.append(f"loss $+\\,{l1:g}\\sum_i |b_i|$ (sparse barrier)")
    return eq + "".join("{\\footnotesize\\parbox{6.2cm}{" + n + "}\\par}\\vspace{2pt}\n" for n in notes) + "\\vspace{4pt}"


def ko_words(run):
    """The variables of THIS run's equation, as slide 15 lists them: one row per symbol the equation writes."""
    import yaml
    spec = yaml.safe_load(open(os.path.join(ROOT, "config", "training", "redox", f"{run}.yaml")))
    op = next(o for o in yaml.safe_load(open(os.path.join(ROOT, spec["model"])))["operators"]
              if o.get("model") == "known_ode")
    lf = {e["field"] for e in spec["learnable"] if "field" in e}
    lp = {e["param"] for e in spec["learnable"] if "param" in e}
    w = [("$r_i$", "redox ratio of voxel $i$"), ("$r^*_i$", "set point"), ("$1/\\tau_i$", "relaxation rate")]
    if "kappa" in lp:
        if op.get("barrier") in lf:
            w += [("$\\kappa_{ij}$", "exchange between neighbours $i$, $j$"), ("$b_i$", "barrier to exchange (membrane)")]
        elif op.get("embedding") in lf:
            w += [("$\\kappa_{ij}$", "exchange between neighbours $i$, $j$"), ("$a_i$", "embedding, sets the exchange")]
        else:
            w += [("$\\kappa_{\\rm axis}$", "exchange, one rate per axis")]
    if "I" in lp:
        w += [("$\\beta_i$", "sensitivity to the global forcing"), ("$I(t)$", "global forcing")]
    return rows(w)


# ============================================================================== what each run learned, as maps
def learned_maps(run, rec, name=None):
    """The run's learned per-voxel fields on the middle z-plane, one panel each, plus I(t) when the law has it. Each map is
    coloured over the tissue's 2nd-98th percentile with its own colourbar: on the full range the barrier panel read as one
    flat white (exp(-b) ~ 1 on most voxels, a few near 0). Returns (figure path, rows of medians, kappa or None);
    None when the run learned no field and no I(t) (the bare GraphCast law, the mesh)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from plexus import trainer as T
    from plexus import engine
    r = landed(run)
    ck = torch.load(os.path.join(r["dir"], "models", "best.pt"), weights_only=False, map_location="cpu")
    fitted = ck["fitted"]
    if not any(k.startswith("field.") or k == "diffuse.I" for k in fitted):
        return None
    spec = T.load(os.path.join(ROOT, "config", "training", "redox", f"{run}.yaml"))
    learn = T.Learnables(spec["learnable"], "cpu")
    learn.restore(fitted)
    H, _ = engine.run(T._model(spec, train=False, n_frames=0), device="cpu", on_seeded=learn.inject)
    z = rec["ratio"].shape[1] // 2
    m3 = rec["mask"][0]
    m = m3[z]
    g = {k: H.fields[k].grid.detach().numpy() for k in ("rest", "rate", "beta", "barrier", "embedding") if k in H.fields}
    ko = "rest" in g
    maps = []                                            # (label, 2-D map on plane z, mask, colormap, 3-D values for the row)
    if ko:
        mu = float(rec["ratio"][rec["mask"]].mean())     # the trainer's `normalise` mean (split all: every frame)
        maps += [("set point r* (ratio)", mu + g["rest"][0, z], m, "viridis", mu + g["rest"][0][m3]),
                 ("rate 1/tau, per 10 min", g["rate"][0, z], m, "magma", g["rate"][0][m3]),
                 ("sensitivity beta to I(t)", g["beta"][0, z], m, "RdBu_r", g["beta"][0][m3])]
        if "barrier" in g:
            # b itself, not exp(-b): the learned b is ~0 on most voxels (ko2_full median 0.005, 98th pct 0.31), so exp(-b)
            # drew a flat white plane on any range; b on its own range shows the few voxels that do block exchange
            maps.append(("barrier b (0 = open)", g["barrier"][0, z], m, "gray_r", g["barrier"][0][m3]))
        elif "embedding" in g and "diffuse.kappa" in fitted:      # v34: the embedding's exchange factor along x
            a = g["embedding"][:, z]
            maps.append(("embedding exchange factor along x", np.exp(-((a[:, :, 1:] - a[:, :, :-1]) ** 2).sum(0)),
                         m[:, 1:], "gray", None))
    elif "embedding" in g:
        E = g["embedding"]                               # hash: levels x features channels -> its first 3 principal components
        X = E[:, m3].T
        X = X - X.mean(0)
        _, sv, Vt = np.linalg.svd(X, full_matrices=False)
        ev = sv ** 2 / (sv ** 2).sum()
        for c in range(min(3, len(sv))):
            pc = np.tensordot(Vt[c], E - E[:, m3].mean(1)[:, None, None, None], axes=1)
            maps.append((f"embedding PC{c + 1} ({100 * ev[c]:.0f}% of its variance)", pc[z], m, "PuOr", pc[m3]))
    I = fitted["diffuse.I"].numpy() if "diffuse.I" in fitted else None
    n = len(maps) + (I is not None)
    if n <= 4:                                           # 2 x 2; with 4 maps AND I(t), I(t) is a strip below the maps
        fig, ax = plt.subplots(2, 2, figsize=(6.8, 5.4), facecolor="black")
        ax = list(ax.ravel())
        Iax = ax[len(maps)] if I is not None else None
    else:
        fig = plt.figure(figsize=(7.2, 6.6), facecolor="black")
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.42])
        ax = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
        Iax = fig.add_subplot(gs[2, :])
    for axx, (lab, img, mm, cm, _) in zip(ax, maps):
        lo, hi = np.percentile(img[mm], [2, 98])
        if hi <= lo:                                     # a constant map (sparse barrier: all 0) -- keep a visible range
            hi = lo + 1e-3
        im = axx.imshow(np.where(mm, img, np.nan), cmap=cm, vmin=lo, vmax=hi)
        axx.set_axis_off()
        axx.set_title(lab, color="0.85", fontsize=11, loc="left")
        cb = fig.colorbar(im, ax=axx, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=9, colors="0.85")
        cb.outline.set_edgecolor("0.5")
    if I is not None:
        axx = Iax
        _black(axx)
        axx.plot(np.arange(1, len(I) + 1), I, color="tab:cyan", lw=1.2)
        axx.axhline(0, color="0.5", lw=0.6)
        axx.set_xlabel("frame (10 min each)", fontsize=10)
        axx.tick_params(labelsize=9)
        axx.set_title("global forcing I(t), ratio per 10 min", color="0.85", fontsize=11, loc="left")
    for axx in ax:
        if axx is not Iax and not axx.images:
            axx.set_axis_off()
    fig.tight_layout()
    out = f"figs/{name or 'maps_' + run + '.png'}"
    fig.savefig(os.path.join(PRES, out), dpi=150, facecolor="black")
    plt.close(fig)
    short = {"set point": "r*", "rate 1/tau": "1/tau", "sensitivity beta": "beta", "barrier b": "b"}
    rows_ = [(next((v_ for k_, v_ in short.items() if lab.startswith(k_)), lab.split(" (")[0]),
              f"{np.median(v):.3g} [{np.percentile(v, 5):.3g}, {np.percentile(v, 95):.3g}]")
             for lab, _, _, _, v in maps if v is not None]
    if "barrier" in g:
        rows_.append(("b > 0.1", f"{100 * (g['barrier'][0][m3] > 0.1).mean():.1f}% of tissue voxels"))
    if I is not None:
        rows_.append(("I(t)", f"{I.min():+.4f} to {I.max():+.4f}"))
    kap = fitted["diffuse.kappa"].numpy() if "diffuse.kappa" in fitted else None
    return out, rows_, kap


def slide_maps(r, got, short=""):
    """Cedric, 2026-10-01: after each movie, a slide like the known ODE's -- what that run learned, as maps."""
    n = r["name"]
    if got is None:
        return ""
    # Cedric, 2026-10-01: the equation as trained and its variables, as slide 15 -- no value table
    right = head(_tex(n) + ": what it learned") + ko_equation(n) + ko_words(n)
    return wide_frame(_tex(n) + " maps", panel(got[0]), right, f"log/training/redox/{n}/models/best.pt",
                      deck_title=f"{family(n)} -- {_tex(short)}, learned maps" if short else f"{family(n)} -- learned maps")


# ============================================================================== movies and metrics
def slide_movie(r, what, rec, recd, short=""):
    n = r["name"]
    mv = os.path.join(r["dir"], "results", "movie.mp4")
    if not os.path.exists(mv):
        return ""
    import shutil
    shutil.copy(mv, os.path.join(PRES, "Movies", f"{n}.mp4"))
    poster(mv, os.path.join(PRES, "Movies", f"{n}.png"))
    t = r["test"]
    z = np.load(os.path.join(r["dir"], "results", f"{n}_free.npz"))
    P, o = z["pred"].astype(np.float32), int(z["origin"])
    k = P.shape[0]
    m = rec["mask"][o + 1:o + 1 + k] & rec["mask"][o]
    raw = FR._r2_upto(rec["ratio"][o + 1:o + 1 + k], P, m)[-1]
    den = FR._r2_upto(recd["ratio"][o + 1:o + 1 + k], P, m)[-1]
    right = (head(_tex(n))
             + para(_tex(what) + ".")         # Cedric, 2026-10-01: the rollout length is said once, on the data slides
             + head("R$^2$ per frame, mean $\\pm$ SD")
             + rows([("raw", f"{raw[0]:+.3f} $\\pm$ {raw[1]:.3f}"), ("denoised", f"{den[0]:+.3f} $\\pm$ {den[1]:.3f}"),
                     ("pooled, denoised", f"{t.get('r2_denoised', float('nan')):+.3f}"),
                     ("shape IoU", f"{t['shape_iou_mean']:.3f} (frame {o + 1} held {t['shape_iou_hold_mean']:.3f})"
                      if "shape_iou_mean" in t else "--")])
             + head("training")
             + rows([("weights", f"{r['report'].get('n_params', 0):,}"),
                     ("time", f"{r['report'].get('seconds', 0) / 3600:.1f} h")]))
    return frame(_tex(n), movie(f"Movies/{n}"), right, f"log/training/redox/{n}", left_gap=True,
                 deck_title=f"{family(n)} -- {_tex(short)}" if short else family(n))


def slide_death(rec):
    """PROPOSED (Cedric, 2026-10-01): a cell-death operator for the organoid's shrinkage. The figure is the recording's own
    evidence for its form -- WHEN the tissue is lost (with the washout), WHERE (one voxel deep at the rim, no core), and
    the top-to-bottom imbalance -- and the right column the operator and its loss."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage as nd
    M, R = rec["mask"], rec["ratio"]
    T = M.shape[0]
    dx, dz = rec["dx_um"], rec["dz_um"]
    vol = M.reshape(T, -1).sum(1) / M[0].sum()
    mean = np.array([R[t][M[t]].mean() for t in range(T)])
    lost_d, all_d = [], []
    for t in range(2, T - 3):                            # PERSISTENT losses: tissue in t-2..t, gone in t+1..t+3 (no flicker)
        was = M[t - 2] & M[t - 1] & M[t]
        gone = ~M[t + 1] & ~M[t + 2] & ~M[t + 3]
        d = nd.distance_transform_edt(M[t], sampling=(dz, dx, dx))
        lost_d.append(d[was & gone])
        all_d.append(d[M[t]])
    lost_d, all_d = np.concatenate(lost_d), np.concatenate(all_d)
    fig, ax = plt.subplots(2, 2, figsize=(6.8, 5.4), facecolor="black")
    a = ax[0, 0]
    _black(a)
    fr = np.arange(1, T + 1)
    a.plot(fr, 100 * vol, color="tab:orange", lw=1.4)
    a.set_ylabel("tissue volume, % of frame 1", fontsize=9, color="tab:orange")
    a.set_xlabel("frame (10 min each)", fontsize=9)
    b = a.twinx()
    b.plot(fr, mean, color="tab:cyan", lw=1.2)
    b.tick_params(labelsize=8, colors="0.85")
    b.set_ylabel("tissue-mean ratio", fontsize=9, color="tab:cyan")
    a.tick_params(labelsize=8)
    a.set_title("when: with the washout", color="0.85", fontsize=11, loc="left")
    a = ax[0, 1]
    _black(a)
    bins = np.arange(0, 40, 1.66)
    a.hist(all_d, bins=bins, density=True, color="0.6", alpha=0.6, label="all tissue")
    a.hist(lost_d, bins=bins, density=True, color="tab:red", alpha=0.8, label="lost")
    a.set_xlabel("depth below the surface, um", fontsize=9)
    a.tick_params(labelsize=8)
    a.legend(frameon=False, fontsize=8, labelcolor="0.85")
    a.set_title("where: the rim, 1 voxel deep", color="0.85", fontsize=11, loc="left")
    a = ax[1, 0]
    _black(a)
    zz = np.arange(M.shape[1]) * dz
    a.barh(zz, M[0].sum((1, 2)), height=dz * 0.8, color="0.6", label="frame 1")
    a.barh(zz, M[-1].sum((1, 2)), height=dz * 0.4, color="tab:red", label=f"frame {T}")
    a.invert_yaxis()
    a.set_ylabel("z, um (top of the stack first)", fontsize=9)
    a.set_xlabel("tissue voxels in the plane", fontsize=9)
    a.tick_params(labelsize=8)
    a.legend(frameon=False, fontsize=8, labelcolor="0.85", loc="upper right")
    a.set_title("the top planes lose most", color="0.85", fontsize=11, loc="left")
    a = ax[1, 1]
    z = M.shape[1] // 2
    img = np.zeros(M.shape[2:] + (3,))
    img[M[0][z]] = (0.45, 0.45, 0.45)
    img[M[0][z] & ~M[-1][z]] = (0.84, 0.15, 0.16)
    img[~M[0][z] & M[-1][z]] = (0.12, 0.47, 0.71)
    a.imshow(img)
    a.set_axis_off()
    a.set_title("mid-plane: lost red, gained blue", color="0.85", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(PRES, "figs", "06_death.png"), dpi=150, facecolor="black")
    plt.close(fig)
    right = (head("proposed: a cell-death operator")
             + "{\\footnotesize $\\displaystyle \\frac{dA_i}{dt}=-k\\,e_i\\,\\sigma\\!\\left(\\frac{r_i-r_c}{w}\\right)A_i$\\par}\\vspace{3pt}"
             + rows([("$A_i$", "alive fraction of voxel $i$, frame 1's mask at start"),
                     ("$e_i$", "exposure: share of the 6 neighbours outside the tissue"),
                     ("$r_i$", "redox ratio of voxel $i$"),
                     ("$r_c$, $w$", "learned stress threshold and its width"),
                     ("$k$", "learned death rate, per 10 min")])
             + head("trained and drawn on the mask")
             + "{\\footnotesize $L=L_{\\rm ratio}+\\lambda\\Big(1-\\frac{\\sum_i A_iM_i}{\\sum_i (A_i+M_i-A_iM_i)}\\Big)$\\par}\\vspace{3pt}"
             + rows([("$M_i$", "observed tissue mask of the frame"), ("movie", "right panel on the model's own tissue, $A_i>0.5$"),
                     ("score", "shape IoU per frame")])
             + head("what the recording says")
             + rows([("volume", f"{100 * vol[-1]:.0f}\\% of frame 1 by frame {T}, lost in frames 10-45"),
                     ("depth", f"lost voxels {np.median(lost_d):.1f} um deep (median), no core"),
                     ("ratio", "lost voxels more reduced than the rim kept (0.619 vs 0.605)")]))
    return wide_frame("model: cell death", panel("figs/06_death.png"), right, "the frozen recording's mask",
                      deck_title=f"{KO} -- cell death (proposed)", left_gap=True)   # top aligned with the text


def slide_ladder():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = [(g, [r for r in (landed(n) for n in names) if r is not None]) for g, names in LADDER]
    groups = [(g, rs) for g, rs in groups if rs]
    fig, ax = plt.subplots(figsize=(6.6, 4.4), facecolor="black")
    _black(ax)
    x, labs, cols = 0, [], ("tab:blue", "tab:orange", "tab:green", "tab:red")
    for gi, (g, rs) in enumerate(groups):
        for r in rs:
            ax.bar(x, r["test"].get("r2_denoised", np.nan), color=cols[gi % 4], label=g if r is rs[0] else None)
            labs.append(r["name"].replace("hlo_", ""))
            x += 1
        x += 0.6
    ax.set_ylabel("R2 against the denoised recording")
    ax.legend(frameon=False, fontsize=7, labelcolor="0.85", loc="lower right")
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(PRES, "figs", "09_ladder.png"), dpi=150, facecolor="black")
    plt.close(fig)
    body = "".join(f"{_tex(r['name'].replace('hlo_', ''))} & {r['test']['r2']:+.3f} & {r['test'].get('r2_denoised', float('nan')):+.3f} \\\\\n"
                   for _, rs in groups for r in rs)
    right = (head("every landed run, R$^2$ of the free rollout")
             + "{\\footnotesize\\begin{tabular}{@{}l@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{}}\nrun & raw & denoised \\\\\n\\hline\n"
             + body + "\\end{tabular}\\par}\\vspace{4pt}"
             + "{\\footnotesize\\parbox{6.2cm}{bars left to right as in the table; colours by group.}\\par}\n")
    return frame("metrics: every run", panel("figs/09_ladder.png"), right, "the runs' _test.json", left_gap=True,
                 deck_title=f"{GC} vs {KO} -- every run")


def main():
    for d in ("slides", "figs", "Movies"):
        os.makedirs(os.path.join(PRES, d), exist_ok=True)
    rec = FR.coarsen(FR.load("hlo_washout"), 4)
    recd = FR.denoise(rec)
    prov = json.load(open(os.path.join(GD, "graphs_data", "redox", "hlo_washout_recording.provenance.json")))
    refs, sig, sigd = references(rec, recd)
    parts = [slide_mesh(rec)] + slides_data(rec, recd, prov, refs, sig, sigd)    # Cedric: the multi-mesh first
    base = landed("hlo_f1_base")
    if base is not None and os.path.exists(os.path.join(base["dir"], "results", "movie.mp4")):
        import shutil
        shutil.copy(os.path.join(base["dir"], "results", "movie.mp4"), os.path.join(PRES, "Movies", "hlo_f1_base.mp4"))
        poster(os.path.join(PRES, "Movies", "hlo_f1_base.mp4"), os.path.join(PRES, "Movies", "hlo_f1_base.png"))
    parts += [slide_graphcast(base["report"].get("n_params", 0) if base else 0), slide_embedding(),
              slide_forcing(rec)]
    ko_law = slide_known_ode(rec)                        # Cedric, 2026-10-01: the known ODE's law opens its own movies
    for n, what, short in MOVIES:
        r = landed(n)
        if r is not None:
            if family(n) == KO and ko_law:
                parts.append(ko_law)
                ko_law = ""
            parts.append(slide_movie(r, what, rec, recd, short))
            if family(n) == KO:                          # Cedric, 2026-10-01: maps slides for the known ODE only
                parts.append(slide_maps(r, learned_maps(n, rec), short))
    parts += [ko_law, slide_death(rec), slide_ladder()]
    names = []
    for i, body in enumerate(p for p in parts if p):
        nm = f"{i:02d}"
        open(os.path.join(PRES, "slides", f"{nm}.tex"), "w").write(body)
        names.append(nm)
    open(os.path.join(PRES, "slides", "all.tex"), "w").write("".join(f"\\input{{slides/{n}}}\n" for n in names))
    print(f"[slides] {len(names)} slides -> {PRES}/slides/all.tex")


if __name__ == "__main__":
    main()
