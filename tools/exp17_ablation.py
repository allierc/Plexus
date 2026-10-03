"""exp17 ABLATION of the neuron-graph law: does the network between the neurons do anything? (Cedric, 2026-10-02:
"test full rollout with W=0 to see if the network is doing a thing, test full rollout with left neurons W=0 to see
if the lateral network does a thing").

    python tools/exp17_ablation.py <run> [<run> ...] [--device cuda:0] [--no-movie]

For each landed neuron-graph run (models/best.pt), THREE FREE ROLLOUTS of the whole recording, each the trainer's own
`_trace_free` (same start, same runaway silencing, same R2 per frame), differing only in the learned edge weights W:

    full     W as learned
    W0       every edge weight of the three edge sets (short, mid, long) set to 0: each neuron is driven only by its
             own leak toward its rest value and its stimulus input B_i . u -- no message from any other neuron
    Sleft0   the left half's stimulus weights B (its rows of the `input` block) set to 0, every W kept: whatever stimulus-locked
             activity the left half keeps reached it through the network from the right (Cedric, 2026-10-02)

LEFT AND RIGHT: the brain's left-right axis is the SHORTER of the two horizontal axes of the neuron positions
(ZAPBench's zapbench_recording.npz, extent ~504 x 818 x 253 um: x; the destriped zap-inr anatomy frame, ~984 x 550 x
350 um: y), split at the MEDIAN position along it (the midline). "Left" = the half BELOW the median along that axis;
which anatomical side that is depends on the axis's sign, a convention not checked against the anatomy.

R2 per frame is over neurons, against the raw recording and against the denoised one (each neuron's 3-frame mean),
over all neurons and separately over the left and the right half (alive neurons only: a silenced runaway neuron is
frozen at its recorded mean and left out, as in `_trace_free`).

Writes, beside the run's own results (log/.../<run>/results/):
    <run>_W0_free.npz, <run>_W0_movie.npz, <run>_Sleft0_free.npz, <run>_Sleft0_movie.npz   (as `_trace_free`)
    <run>_abl_full_free.npz                       (the full rollout re-run here, to check it reproduces the run's own)
    <run>_ablation_sides.npz                      (R2 per frame over left / right neurons, all three rollouts)
    movie_W0.mp4 / .png, movie_Sleft0.mp4 / .png  (`trace_recording.render_movie`, the learned panel relabelled)
and in the experiment folder:
    experiments/exp17_zapbench_graphcast/data/ablation_<run>.json
    experiments/exp17_zapbench_graphcast/data/figs/ablation_<run>.png
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import sys
import time

import numpy as np
import torch

EXP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments",
                   "exp17_zapbench_graphcast")
FRAME_S = 0.914                      # seconds per recorded frame (ZAPBench)
SMOOTH = 31                          # frames in the figure's running mean of R2 per frame (~28 s)
EDGE_SETS = ("short", "mid", "long")


def neuron_graph_op(spec, device):
    """The run's neuron-graph operator instance, built as the trainer builds its model; None if the law is another."""
    from plexus import engine
    from plexus import trainer as T
    from plexus.operators.cell_ops import StateDiffuseNeuronGraph
    sim = T._model(spec, train=False, n_frames=0)
    with torch.no_grad():
        H, _ = engine.run(sim, device=device, progress=False)
    ops = [o for o in H.operators if isinstance(o, StateDiffuseNeuronGraph)]
    return ops[0] if len(ops) == 1 else None


def left_right(op):
    """(left mask [N] bool, a description) from the operator's own positions file: split at the median along the
    shorter horizontal axis."""
    pos = np.asarray(np.load(op._pos_file[0])[op._pos_file[1]], dtype=np.float64)
    ext = np.ptp(pos, 0)
    ax = int(np.argmin(ext[:2]))
    mid = float(np.median(pos[:, ax]))
    left = pos[:, ax] < mid
    desc = {"positions_file": op._pos_file[0], "extent_um_xyz": [round(float(e), 1) for e in ext],
            "left_right_axis": "xy"[ax], "midline_um": mid,
            "left_is": f"{'xy'[ax]} below the median {mid:.1f} um (a sign convention, not checked against anatomy)",
            "n_left": int(left.sum()), "n_right": int((~left).sum())}
    return left, desc


@contextlib.contextmanager
def per_side_r2(left_t, store):
    """Inside it, every `_trace_rollout_stream` that `_trace_free` makes also scores R2 per frame over the left and the
    right neurons separately, reading `_trace_free`'s own recording, denoised recording and silenced-neuron mask from
    its `on_pred` (so the alive neurons are exactly the ones its own R2 uses). Results land in `store`."""
    from plexus import trainer as T
    orig = T._trace_rollout_stream

    def wrapped(sim, learn, spec, box, o, n, device, on_pred, silence=None, w=0):
        nl = inspect.getclosurevars(on_pred).nonlocals
        missing = [k for k in ("X", "Xd", "dead", "o0") if k not in nl]
        if missing:
            raise RuntimeError(f"_trace_free's on_pred no longer holds {missing}: update exp17_ablation.per_side_r2")
        X, Xd, dead, o0 = nl["X"], nl["Xd"], nl["dead"], nl["o0"]
        sides = {"left": left_t, "right": ~left_t}
        for s in sides:
            for kind in ("raw", "denoised"):
                store[f"r2_{kind}_{s}"] = np.zeros(n)
            store[f"silenced_{s}"] = np.zeros(n, np.int64)

        def r2(p, x, m):
            num = ((p[m] - x[m]) ** 2).sum()
            den = ((x[m] - x[m].mean()) ** 2).sum()
            return float(1 - num / den)

        def on_pred2(k, p):
            on_pred(k, p)
            tt = o0 + k + 1
            for s, m in sides.items():
                a = m & ~dead
                store[f"r2_raw_{s}"][k] = r2(p, X[tt], a)
                store[f"r2_denoised_{s}"][k] = r2(p, Xd[tt], a)
                store[f"silenced_{s}"][k] = int((m & dead).sum())
        return orig(sim, learn, spec, box, o, n, device, on_pred2, silence=silence, w=w)

    T._trace_rollout_stream = wrapped
    try:
        yield store
    finally:
        T._trace_rollout_stream = orig


@contextlib.contextmanager
def weights_zeroed(learn, keys, masks):
    """The learned edge weights with the masked edges set to 0, restored on exit. `masks[s]` a bool [E_s] (True =
    zero this edge) or None for every edge of set s."""
    saved = {}
    with torch.no_grad():
        for s, k in keys.items():
            saved[k] = learn.p[k].detach().clone()
            m = masks[s]
            if m is None:
                learn.p[k].zero_()
            else:
                learn.p[k][m] = 0.0
    try:
        yield
    finally:
        with torch.no_grad():
            for k, v in saved.items():
                learn.p[k].copy_(v)


@contextlib.contextmanager
def input_zeroed(learn, key, rows):
    """The learned stimulus weights B (the `input` block, [N, 22]) with the rows of `rows` (bool [N]) set to 0,
    restored on exit: those neurons get no stimulus, the network is untouched."""
    with torch.no_grad():
        saved = learn.p[key].detach().clone()
        learn.p[key][rows.to(learn.p[key].device)] = 0.0
    try:
        yield
    finally:
        with torch.no_grad():
            learn.p[key].copy_(saved)


def summarise(free, sides, lab):
    """The numbers of one rollout: means over all frames, and over ZAPBench's test / held-out frames when split."""
    out = {"r2_raw": free["r2_raw"], "r2_denoised": free["r2_denoised"], "finite": free["finite"],
           "silenced": free["silenced"], "silenced_first_t_s": free["silenced_first_t_s"]}
    for s in ("left", "right"):
        out[f"r2_raw_{s}"] = float(np.nanmean(sides[f"r2_raw_{s}"]))
        out[f"r2_denoised_{s}"] = float(np.nanmean(sides[f"r2_denoised_{s}"]))
        out[f"silenced_{s}"] = int(sides[f"silenced_{s}"][-1])
    out["r2_denoised_median"] = float(np.nanmedian(sides["_all"]))
    for s in ("left", "right"):
        out[f"r2_denoised_{s}_median"] = float(np.nanmedian(sides[f"r2_denoised_{s}"]))
    out["left_minus_right_r2_denoised"] = out["r2_denoised_left"] - out["r2_denoised_right"]
    if lab is not None:
        for name, v in (("test", 2), ("holdout", 3)):
            m = lab == v
            if m.any():
                out[f"r2_denoised_{name}_frames"] = float(np.nanmean(sides["_all"][m]))
                for s in ("left", "right"):
                    out[f"r2_denoised_{s}_{name}_frames"] = float(np.nanmean(sides[f"r2_denoised_{s}"][m]))
    return out


def run_one(name, device="cuda:0", movie=True):
    from plexus import engine
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    engine.quiet(True)
    spec = T.load(name)
    name = spec["name"]
    out = T.out_dir(spec, None)
    op = neuron_graph_op(spec, device)
    if op is None:
        raise SystemExit(f"{name}: not a neuron-graph run (no state_diffuse[neuron_graph] in {spec['model']}); "
                         "the W ablation needs its edge weights")
    keys = {}
    for e in spec["learnable"]:
        if e.get("op") == "state_diffuse" and str(e.get("param", "")).startswith("W_"):
            keys[e["param"][2:]] = T.Learnables.key(e)
    sets = [s for s in EDGE_SETS if op._E[s][0].numel()]
    if sorted(keys) != sorted(sets):
        raise SystemExit(f"{name}: learned edge sets {sorted(keys)}, the graph has {sets}: cannot ablate cleanly")
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    learn = T.Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    for s, k in keys.items():
        if learn.p[k].numel() != op._E[s][1].numel():
            raise SystemExit(f"{name}: {k} has {learn.p[k].numel()} weights, the {s} edge set {op._E[s][1].numel()}")
    left, lr_desc = left_right(op)
    left_t = torch.as_tensor(left, device=device)
    box = T._trace_setup(spec, device)
    if box["X"].shape[1] != len(left):
        raise SystemExit(f"{name}: recording has {box['X'].shape[1]} neurons, the positions file {len(left)}")
    lab = box.get("lab")

    # the edges into the left half (counts for the json: how much of the network the left half receives)
    left_masks = {s: left_t[op._E[s][1].to(device)] for s in keys}
    edges = {s: {"edges": int(op._E[s][1].numel()), "into_left": int(left_masks[s].sum()),
                 "left_to_right": int((left_t[op._E[s][0].to(device)] & ~left_masks[s]).sum()),
                 "mean_abs_w": float(learn.p[keys[s]].detach().abs().mean())} for s in keys}
    # Cedric, 2026-10-02: "the no stimulus left is not the right ablation" -- the lateral test is the left half WITHOUT ITS
    # STIMULUS (its rows of B zeroed) and every W kept: whatever stimulus-locked activity the left half still has
    # reached it through the network from the right
    in_key = next((T.Learnables.key(e) for e in spec["learnable"] if e.get("block") == "input"), None)
    if in_key is None:
        raise SystemExit(f"{name}: no learned `input` block (B): cannot remove the left half's stimulus")
    arms = {"full": None, "W0": {s: None for s in keys}, "Sleft0": ("input", left_t)}
    res, per_frame, t_axis = {}, {}, None
    for arm, masks in arms.items():
        stem = f"{name}_abl_full" if arm == "full" else f"{name}_{arm}"
        t1 = time.time()
        store = {}
        cm = (contextlib.nullcontext() if masks is None else
              input_zeroed(learn, in_key, masks[1]) if isinstance(masks, tuple) else weights_zeroed(learn, keys, masks))
        with cm, per_side_r2(left_t, store):
            free, r2r, r2d, o0, n, finite, n_dead = T._trace_free(spec, learn, box, device, out, stem)
        if arm == "full":                    # its movie frames duplicate the run's own <run>_movie.npz
            p = os.path.join(out, "results", f"{stem}_movie.npz")
            if os.path.exists(p):
                os.remove(p)
        store["_all"] = r2d
        res[arm] = summarise(free, store, None if lab is None else np.asarray(lab)[o0 + 1 + np.arange(n)])
        per_frame[arm] = {"all": r2d, "raw": r2r, **{k: v for k, v in store.items() if k != "_all"}}
        t_axis = o0 + 1 + np.arange(n)
        print(f"[ablation] {name} {arm}: {time.time() - t1:.0f} s, R2 denoised {res[arm]['r2_denoised']:.3f} "
              f"(left {res[arm]['r2_denoised_left']:.3f}, right {res[arm]['r2_denoised_right']:.3f}), "
              f"{free['silenced']} silenced", flush=True)
        if arm == "full":
            tj = os.path.join(out, "results", f"{name}_test.json")
            own = json.load(open(tj)).get("free", {}) if os.path.exists(tj) else {}
            if "r2_denoised" in own:
                d = res["full"]["r2_denoised"] - own["r2_denoised"]
                res["full"]["run_own_r2_denoised"] = own["r2_denoised"]
                res["full"]["reproduces_run_own"] = bool(abs(d) < 1e-3)
                print(f"[ablation] {name} full vs the run's own free rollout: R2 denoised {res['full']['r2_denoised']:.4f}"
                      f" vs {own['r2_denoised']:.4f} (difference {d:+.1e})", flush=True)
    del box
    torch.cuda.empty_cache()
    np.savez_compressed(os.path.join(out, "results", f"{name}_ablation_sides.npz"), t=t_axis, left=left,
                        **{f"{arm}_{k}": v for arm, d in per_frame.items() for k, v in d.items()})

    head = {
        "network_adds_r2_denoised": res["full"]["r2_denoised"] - res["W0"]["r2_denoised"],
        "network_adds_r2_raw": res["full"]["r2_raw"] - res["W0"]["r2_raw"],
        "left_minus_right_full": res["full"]["left_minus_right_r2_denoised"],
        "left_minus_right_Sleft0": res["Sleft0"]["left_minus_right_r2_denoised"],
        "left_r2_lost_by_Sleft0": res["full"]["r2_denoised_left"] - res["Sleft0"]["r2_denoised_left"],
        "right_r2_lost_by_Sleft0": res["full"]["r2_denoised_right"] - res["Sleft0"]["r2_denoised_right"],
    }
    rec = TR.load(spec["task"]["reference"]["trace_recording"])
    doc = {"run": name, "model": spec["model"], "recording": spec["task"]["reference"]["trace_recording"],
           "split": spec["task"]["reference"].get("split", "all"),
           "what": {"full": "free rollout of the whole recording with the learned edge weights W",
                    "W0": "the same with every edge weight (short, mid, long) set to 0: each neuron only its own leak, "
                          "rest and stimulus input",
                    "Sleft0": "the same with the left half's stimulus weights B set to 0 (no stimulus into the left half), every W kept"},
           "r2": "R2 per frame over neurons, then the mean over all free-rollout frames; denoised = against each "
                 "neuron's 3-frame mean; left/right = over that half's alive neurons only",
           "left_right": lr_desc, "edges": edges, "frames": int(len(t_axis)), "headline": head, **res}
    os.makedirs(os.path.join(EXP, "data", "figs"), exist_ok=True)
    jp = os.path.join(EXP, "data", f"ablation_{name}.json")
    json.dump(doc, open(jp, "w"), indent=1)
    fp = os.path.join(EXP, "data", "figs", f"ablation_{name}.png")
    figure(doc, per_frame, t_axis, np.asarray(rec["condition"]), list(rec["names"]), fp)
    print(f"[ablation] {name}: network adds R2 denoised {head['network_adds_r2_denoised']:+.3f} "
          f"(full {res['full']['r2_denoised']:.3f}, W0 {res['W0']['r2_denoised']:.3f}); left - right R2 denoised "
          f"{head['left_minus_right_full']:+.3f} full, {head['left_minus_right_Sleft0']:+.3f} with no stimulus into the left half\n"
          f"           {jp}\n           {fp}", flush=True)
    if movie:
        render_arms(spec, out, rec)
    return doc


# the learned panel's label, "learned (<this>)": short enough to fit above the panel
MOVIE_LABEL = {"W0": "neuron graph, W = 0: no network", "Sleft0": "no stimulus into the left half",
               "no_stimulus": "no stimulus (u = 0)", "lead_left_quarter": "left quarter given (recorded), rest free"}


def render_arms(spec, out, rec=None):
    from plexus.tasks import trace_recording as TR
    rec = rec if rec is not None else TR.load(spec["task"]["reference"]["trace_recording"])
    for arm, lbl in MOVIE_LABEL.items():
        render(spec, out, f"{spec['name']}_{arm}", os.path.join(out, "results", f"movie_{arm}.mp4"), lbl, rec)


def render(spec, out, stem, path, law, rec):
    """The ablated rollout's movie, as `_analyse_trace` renders a run's own (no inset strip)."""
    from plexus.tasks import trace_recording as TR
    t1 = time.time()
    mv = np.load(os.path.join(out, "results", f"{stem}_movie.npz"))
    frames = mv["frames"]
    TR.render_movie(rec["dff"][frames], mv["pred"].astype(np.float32), frames, rec["pos_um"],
                    mv["r2_raw"], mv["r2_denoised"], FRAME_S, rec["condition"][frames], rec["names"], path,
                    emb=None, labels=None, r2_t=mv["r2_t"], r2_raw_all=mv["r2_raw_all"],
                    r2_den_all=mv["r2_denoised_all"], silenced_all=mv["silenced_all"],
                    mean_obs_all=mv["mean_obs_all"], mean_pred_all=mv["mean_pred_all"], cond_all=mv["cond_all"],
                    split_all=mv["split_all"], split_name=spec["task"]["reference"].get("split", "all"), law=law,
                    rec_name="destriped" if "destripe" in str(spec["task"]["reference"].get("trace_recording"))
                    else "ZAPBench")
    print(f"[ablation] {path} ({len(frames)} frames, {time.time() - t1:.0f} s)", flush=True)


def _smooth(y, k=SMOOTH):
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    c = np.convolve(np.nan_to_num(y, nan=0.0), np.ones(k) / k, mode="same")
    return c


def figure(doc, pf, t, cond, names, path):
    """Two panels, black background, labels above: (a) R2 denoised per frame over all neurons, full vs W = 0;
    (b) over the left and the right half, under no stimulus left (solid) and the full model (dashed). Condition blocks
    shaded, y 0..1, a running mean of SMOOTH frames."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11})
    tm = t * FRAME_S / 60.0
    fig, axs = plt.subplots(2, 1, figsize=(15, 7.2), facecolor="black", sharex=True)
    fig.subplots_adjust(left=0.05, right=0.80, top=0.88, bottom=0.08, hspace=0.42)
    c = cond[t]
    edges = np.flatnonzero(np.diff(c)) + 1
    starts, stops = np.r_[0, edges], np.r_[edges, len(c)]
    r, h = doc, doc["headline"]
    lr = doc["left_right"]
    titles = [
        (f"a   R2 per frame over all {lr['n_left'] + lr['n_right']:,} neurons, against the denoised recording "
         f"(running mean of {SMOOTH} frames)\n     the network adds {h['network_adds_r2_denoised']:+.3f} R2 "
         f"(full {r['full']['r2_denoised']:.3f} - W = 0 {r['W0']['r2_denoised']:.3f}, means over the whole rollout)"),
        (f"b   R2 per frame over the left ({lr['n_left']:,}) and right ({lr['n_right']:,}) halves; left = "
         f"{lr['left_right_axis']} below the midline, a sign convention\n     left - right gap "
         f"{h['left_minus_right_Sleft0']:+.3f} with no stimulus left, {h['left_minus_right_full']:+.3f} in the full model"),
    ]
    for ax, title in zip(axs, titles):
        ax.set_facecolor("black")
        for i, (a, b) in enumerate(zip(starts, stops)):
            if i % 2 == 0:
                ax.axvspan(tm[a], tm[b - 1], color="0.16", lw=0)
            if b - a > 120 and ax is axs[0]:
                ax.text(0.5 * (tm[a] + tm[b - 1]), 1.01, str(names[int(c[a])]), color="0.6", fontsize=7,
                        ha="center", va="bottom", transform=ax.get_xaxis_transform(), clip_on=False)
        ax.set_ylim(0, 1)
        ax.set_xlim(tm[0], tm[-1])
        ax.set_ylabel("R2, denoised", color="0.85")
        ax.tick_params(colors="0.75")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(axis="y", color="0.3", lw=0.5)
        ax.text(0.0, 1.10 if ax is axs[0] else 1.04, title, color="white", fontsize=10.5, transform=ax.transAxes,
                ha="left", va="bottom")
    axs[0].plot(tm, _smooth(pf["full"]["all"]), color="white", lw=1.6, label=f"full model ({r['full']['r2_denoised']:.3f})")
    axs[0].plot(tm, _smooth(pf["W0"]["all"]), color="#ff9f1c", lw=1.6, label=f"W = 0, no network ({r['W0']['r2_denoised']:.3f})")
    for side, col in (("left", "#ff5a5f"), ("right", "#4ea8ff")):
        axs[1].plot(tm, _smooth(pf["Sleft0"][f"r2_denoised_{side}"]), color=col, lw=1.6,
                    label=f"{side} half, no stimulus left ({r['Sleft0'][f'r2_denoised_{side}']:.3f})")
        axs[1].plot(tm, _smooth(pf["full"][f"r2_denoised_{side}"]), color=col, lw=1.0, ls=(0, (3, 2)), alpha=0.8,
                    label=f"{side} half, full model ({r['full'][f'r2_denoised_{side}']:.3f})")
    for ax in axs:
        lg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5, frameon=True,
                       title="mean R2 over the rollout in ( )", title_fontsize=8)
        lg.get_title().set_color("0.7")
        lg.get_frame().set_facecolor("black")
        lg.get_frame().set_edgecolor("0.4")
        for tx in lg.get_texts():
            tx.set_color("0.9")
    axs[1].set_xlabel("time, min (free rollout of the whole recording)", color="0.85")
    fig.suptitle(f"{doc['run']}: what the edge weights W between neurons contribute to the free rollout "
                 f"({doc['recording']}, split {doc['split']})", color="white", fontsize=12, x=0.05, ha="left")
    fig.savefig(path, dpi=130, facecolor="black")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-movie", action="store_true", help="numbers and figure only")
    ap.add_argument("--render-only", action="store_true", help="re-render the movies from the saved rollouts")
    a = ap.parse_args(argv)
    for r in a.runs:
        if a.render_only:
            from plexus import trainer as T
            spec = T.load(r)
            render_arms(spec, T.out_dir(spec, None))
        else:
            run_one(r, a.device, movie=not a.no_movie)


if __name__ == "__main__" and "--compare" not in sys.argv:
    sys.exit(main())


# ------------------------------------------------------------------------------------------------------------------
# THE COMPARISON MOVIES (Cedric, 2026-10-02): the full learned model LEFT and the ablated one RIGHT, not the recording
# against the ablation. W0: under each brain its brain-mean dF/F (recorded green, the model white) and its free-rollout
# R2 (0..1) -- a 2 x 2 of curves. Sleft0: the brain-mean dF/F of the LEFT half and of the RIGHT half, each recorded
# (green), full model (white) and ablated (orange); no R2.
_CMP: dict = {}


def _brain_view(pos):
    """Horizontal, head left: the run movies' mapping (trace_recording.render_movie)."""
    if np.ptp(pos[:, 0]) > np.ptp(pos[:, 1]):
        pos = np.stack([pos[:, 1], -pos[:, 0], pos[:, 2]], 1)
    return np.stack([-pos[:, 1], pos[:, 0], pos[:, 2]], 1)


def _cmp_frames(ks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = _CMP
    P, order, vmax, tm, fr = d["P"], d["order"], d["vmax"], d["tm"], d["frames"]
    fig = plt.figure(figsize=(12, 7.6), facecolor="black")
    sc, txt = [], []
    for j, lab in enumerate(("learned: the full model", f"learned: {d['label']}")):
        ax = fig.add_axes([0.5 * j, 0.40, 0.5, 0.50])
        ax.set_facecolor("black")
        ax.axis("off")
        sc.append(ax.scatter(P[:, 0], P[:, 1], c=np.zeros(len(P)), s=0.5, cmap="inferno", vmin=0, vmax=vmax,
                             linewidths=0))
        ax.set_aspect("equal")
        fig.text(0.5 * j + 0.03, 0.985, lab, color="white", fontsize=12, va="top")
        txt.append(fig.text(0.5 * j + 0.47, 0.955, "", color="white", fontsize=9, va="top", ha="right"))
    t_txt = fig.text(0.03, 0.945, "", color="0.7", fontsize=9, va="top")
    cur = []

    def strip(x0, y0, h=0.085):
        m = fig.add_axes([x0, y0, 0.42, h])
        m.set_facecolor("black")
        for k_, sp in m.spines.items():
            sp.set_visible(k_ in ("left", "bottom"))
            sp.set_color("0.5")
        m.set_xlim(tm[0], tm[-1])
        m.tick_params(colors="0.6", labelsize=7, length=2)
        for i_, (a_, b_, c_) in enumerate(d["blocks"]):
            m.axvspan(a_, b_, color=("0.30" if i_ % 2 else "0.18"), alpha=0.6, lw=0, zorder=0)
        cur.append(m.axvline(tm[0], color="#ff7f0e", lw=0.9, zorder=5))
        return m

    def dff(m, curves):
        for y, col in curves:
            m.plot(tm, y, color=col, lw=0.7, zorder=2)
        lo_, hi_ = np.nanpercentile(np.concatenate([c for c, _ in curves]), [0.5, 99.5])
        m.set_ylim(lo_ - 0.1 * (hi_ - lo_), hi_ + 0.1 * (hi_ - lo_))
        m.set_yticks([])
    if d["mode"] != "Sleft0":                # W0 and the trainer's rollout variants (task.rollouts): a 2 x 2
        for j, key in enumerate(("full", "abl")):
            x0 = 0.05 + 0.5 * j
            m = strip(x0, 0.235)
            dff(m, [(d["mean_rec"], "#2ca02c"), (d[f"mean_{key}"], "white")])
            for a_, b_, c_ in d["blocks"]:
                m.text((a_ + b_) / 2, 1.02, d["names"][int(c_)], color="0.75", fontsize=6, ha="center", va="bottom",
                       transform=m.get_xaxis_transform())
            fig.text(x0, 0.345, ("brain-mean dF/F of the FREE neurons (the given ones left out): recorded (green), "
                                 "this model (white)" if d["given"] else "brain-mean dF/F: recorded (green), this model "
                                 "(white)"), color="0.7", fontsize=8)
            r = strip(x0, 0.06)
            r.plot(d["t_all"], d[f"r2_{key}"], color="white", lw=0.6, zorder=2)
            r.set_ylim(0, 1)
            r.set_yticks([0, 1])
            fig.text(x0, 0.155, "free rollout R$^2$ (denoised) per frame; time, min", color="0.7", fontsize=8)
    else:
        for j, side in enumerate(("left", "right")):
            x0 = 0.05 + 0.5 * j
            m = strip(x0, 0.10, h=0.20)
            dff(m, [(d[f"{side}_rec"], "#2ca02c"), (d[f"{side}_full"], "white"), (d[f"{side}_abl"], "#ff9f1c")])
            for a_, b_, c_ in d["blocks"]:
                m.text((a_ + b_) / 2, 1.02, d["names"][int(c_)], color="0.75", fontsize=6, ha="center", va="bottom",
                       transform=m.get_xaxis_transform())
            fig.text(x0, 0.335, f"the {side.upper()} half's mean dF/F: recorded (green), full model (white), "
                     "ablated (orange)", color="0.75", fontsize=8)
            fig.text(x0, 0.03, "time, min", color="0.6", fontsize=8)
    for k in ks:
        f = fr[k]
        sc[0].set_array(np.asarray(d["full"][k], np.float32)[order])
        sc[1].set_array(np.asarray(d["abl"][k], np.float32)[order])
        t_txt.set_text(f"{d['names'][d['cond'][k]]}   t = {f * FRAME_S / 60:5.1f} min")
        upto = d["t_all"] <= f * FRAME_S / 60
        for j, key in enumerate(("full", "abl")):
            rr = d[f"r2_{key}"][upto]
            txt[j].set_text(f"R2 denoised {np.mean(rr):+.3f} so far" if len(rr) else "")
        for c_ in cur:
            c_.set_xdata([f * FRAME_S / 60] * 2)
        fig.savefig(os.path.join(d["tmp"], f"{k:05d}.png"), dpi=90, facecolor="black")
    plt.close(fig)


def render_compare(name, arm, workers=16):
    """results/movie_<arm>_cmp.mp4 (+ .png): the full model and the `arm` ablation side by side (see above)."""
    import multiprocessing as mp
    import shutil as sh
    import subprocess
    import tempfile
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    spec = T.load(name)
    out = T.out_dir(spec, None)
    rec = TR.load(spec["task"]["reference"]["trace_recording"])
    full = np.load(os.path.join(out, "results", f"{name}_movie.npz"))
    abl = np.load(os.path.join(out, "results", f"{name}_{arm}_movie.npz"))
    fr = full["frames"]
    if not np.array_equal(fr, abl["frames"]):
        raise SystemExit(f"{name}: the full and the {arm} movies sample different frames")
    sides = os.path.join(out, "results", f"{name}_ablation_sides.npz")
    left = (np.load(sides)["left"].astype(bool) if os.path.exists(sides)
            else np.zeros(rec["dff"].shape[1], bool))       # a trainer rollout variant: no halves needed
    # a variant with GIVEN elements (task.rollouts clamp): the brain means over the free elements only, in both panels
    free_ = ~abl["clamped"].astype(bool) if "clamped" in abl else np.ones(rec["dff"].shape[1], bool)
    pos = _brain_view(np.asarray(rec["pos_um"], np.float64))
    order = np.argsort(pos[:, 2])
    X = rec["dff"][fr].astype(np.float32)
    pf, pa = full["pred"].astype(np.float32), abl["pred"].astype(np.float32)
    tm = fr * FRAME_S / 60
    cond = rec["condition"][fr]
    cut = np.flatnonzero(np.diff(cond)) + 1
    st, en = np.r_[0, cut], np.r_[cut, len(cond)] - 1
    blocks = [(tm[a], tm[b], cond[a]) for a, b in zip(st, en)]
    _CMP.clear()
    _CMP.update(P=pos[order], order=order, vmax=float(np.percentile(X, 97)), tm=tm, frames=fr, full=pf, abl=pa,
                cond=cond, names=[str(s) for s in rec["names"]], blocks=blocks, mode=arm,
                label=MOVIE_LABEL.get(arm, arm), t_all=full["r2_t"] * FRAME_S / 60,
                r2_full=np.asarray(full["r2_denoised_all"], float), r2_abl=np.asarray(abl["r2_denoised_all"], float),
                mean_rec=np.nanmean(X[:, free_], 1), mean_full=np.nanmean(pf[:, free_], 1),
                mean_abl=np.nanmean(pa[:, free_], 1), given=bool((~free_).any()),
                left_rec=np.nanmean(X[:, left], 1), left_full=np.nanmean(pf[:, left], 1),
                left_abl=np.nanmean(pa[:, left], 1), right_rec=np.nanmean(X[:, ~left], 1),
                right_full=np.nanmean(pf[:, ~left], 1), right_abl=np.nanmean(pa[:, ~left], 1),
                tmp=tempfile.mkdtemp(prefix="abl_cmp_"))
    ks = np.arange(len(fr))
    workers = min(workers, len(os.sched_getaffinity(0)))
    chunks = [c for c in np.array_split(ks, workers) if len(c)]
    with mp.get_context("fork").Pool(len(chunks)) as pool:
        pool.map(_cmp_frames, chunks)
    path = os.path.join(out, "results", f"movie_{arm}_cmp.mp4")
    subprocess.run([TR._ffmpeg(), "-y", "-loglevel", "error", "-framerate", "25", "-i",
                    os.path.join(_CMP["tmp"], "%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt",
                    "yuv420p", "-c:v", "libx264", path], check=True)
    sh.copy(os.path.join(_CMP["tmp"], f"{len(fr) // 2:05d}.png"), path.replace(".mp4", ".png"))
    sh.rmtree(_CMP["tmp"])
    _CMP.clear()
    print(f"[ablation] {path}", flush=True)
    return path


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "--compare":
    for n_ in sys.argv[2:]:
        for a_ in ("W0", "Sleft0"):
            render_compare(n_, a_)
