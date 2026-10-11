"""THE GRAPH PHASE (`Plexus_Main.py -o graph <run>`, `-o test_plot_graph <run>`; exp17, Cedric 2026-10-10: "whether or
not the graph is doing something, what and how if it does anything ... in a reusable way"). A landed trace run -- a
neuron-graph law fitted to a recording -- analysed through the law's own graph protocol (cell_ops: `graph_roles`,
`silencers`, `prune_sets`, `linearise`, `omega_at`) and the trainer's own rollouts (`_trace_free`, its `task.rollouts`
kinds), so every number is the run's and the same for every law. Scores in results/<stem>_graph.json, figures and
movies under results/graph/, the variants' rollouts cached there (reused while newer than the checkpoint).

THE TESTS (`task.graph.tests`, default all that apply), in order:
    constants  the learned constants: tau (s), V (dF/F), |B_i|, the edge weights per set, each sender's sign mass
    silence    the graph silenced at inference: every silencer at 0, and one edge set at a time; the stimulus off --
               brain-mean r / R2 / RMSE and the per-neuron r (brain mean removed), whole rollout and steady part
               (the first of `bootstrap.blocks` blocks left out), input and other neurons apart; the paired block
               bootstrap of the graph against each
    blocks     full against silenced, scored within each stimulus block from every-frame traces (the per-block
               baseline constant): the gap where the stimulus is weak
    controls   the trained controls named in `task.graph.controls` / `--graph-controls` (no_w, mean_field, seed,
               random), scored on the same frames and neurons, the bootstrap against the graph; the seed pair's spread
    clamp      the relay test: the input neurons given their recorded traces (clamp_in), with and without the graph,
               scored on the free neurons
    prune_w    the pruning ladder per edge set: the two-Gaussian threshold on log10 |w|, rollouts along a ladder of
               thresholds, "removable" when both metrics stay within the seed spread; the joint cut
    prune_b    the same ladder on the input weights |B_i| (Cedric: "a threshold on B_i to identify the input neurons")
    spectrum   the law linearised at the recording's mean state: the eigenvalues of largest real and imaginary part
               of J = diag(r)(M - I) per second, the leak-only abscissa, the leading modes' size and spread, the
               spectral radius of M, the net recurrent gain per neuron
    impulse    propagation: the linear impulse response from J and a nonlinear pulse rollout (the virtual
               perturbation: a region held at mu + level_z sd with the stimulus off), region by region, with and
               without the graph; a movie of the response
    leadlag    who leads the brain mean, in the recording and in the every-frame rollout (the cross-spectrum phase
               over periods 20-250 s), per region; the rank correlation of the two maps
    memorise   the functions of absolute time (Omega, alpha): their statistics, and the rollout with the clock shifted
               or frozen

    PYTHONPATH=src python Plexus_Main.py -o graph zap_n22_markall --device cuda:0 \\
        --graph-controls no_w=zap_n22_markall_now,mean_field=zap_n22_markall_mf,seed=zap_n22_nom_s1
    python -m plexus.tasks.graph_analysis zap_n22_markall [--tests silence,blocks] [--device cuda:0] [--controls ...]
-> results/<stem>_graph.json, results/graph/<stem>_card.png, results/graph/<stem>_pulse_<region>.mp4, the variants'
   <stem>_<variant>_{free,movie}.npz and _traces.npy
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import traceback

import numpy as np
import torch

TESTS = ("constants", "silence", "blocks", "controls", "clamp", "prune_w", "prune_b", "spectrum", "impulse", "leadlag",
         "memorise")
DEFAULTS = {"steady_skip_min": 5.0, "bootstrap": {"blocks": 24, "resamples": 10000, "seed": 0},
            "prune": {"ladder": 7, "span": [1.0 / 9.0, 3.0], "quantiles": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]},
            "prune_b": {"quantiles": [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]},
            "spectrum": {"k": 50, "k_imag": 20, "state": "recording_mean", "solver": "krylov_schur"},
            "pulse": {"level_z": 2.0, "duration_s": 10.0, "settle_s": 120.0, "window_s": 120.0, "regions": None},
            "movies": True, "every_frame": True}
SEED_SPREAD = {"brain_mean_r": 0.009, "per_neuron_r": 0.005}     # exp17 19.25 vs 19.26, the fallback when no seed twin
SIG_S = 1.0                                                      # a lead within +-1 s is neither leading nor trailing
MIN_REGION = 200


# ============================================================================== the context
def settings(spec) -> dict:
    """The phase's settings: DEFAULTS under `task.graph`, the CLI's controls on top."""
    g = dict(spec["task"].get("graph") or {})
    st = {}
    for k, v in DEFAULTS.items():
        st[k] = {**v, **(g.get(k) or {})} if isinstance(v, dict) else g.get(k, v)
    st["controls"] = {k: v for k, v in {**(g.get("controls") or {}), **(spec.get("_graph_controls") or {})}.items() if v}
    st["tests"] = list(g.get("tests") or TESTS)
    st["inputs"] = g.get("inputs")
    st["regions"] = spec.get("_graph_regions") or g.get("regions")
    return st


def context(spec, device="cpu", root=None) -> dict:
    """Everything the tests read: the restored run (`trainer.restore_trace`), the recording, the input neurons, the
    regions, the folders."""
    from plexus import trainer as T
    R = T.restore_trace(spec, device, root)
    out, stem = R["out"], R["stem"]
    gd = os.path.join(out, "results", "graph")
    os.makedirs(gd, exist_ok=True)
    rec, box, op = R["box"]["rec"], R["box"], R["op"]
    st = settings(spec)
    X = np.asarray(rec["dff"], np.float32)
    N = X.shape[1]
    mask_in, mask_file, stim_cols = None, None, None
    if st["inputs"]:
        mask_file = st["inputs"]
        mask_in = np.asarray(T._clamp_mask(st["inputs"], rec["pos_um"]))
    elif getattr(op, "video_mask", None) is not None:
        # a video law: the stimulus enters through the encoder at the video mask's neurons; its `input` block carries
        # the release features only where its own mask lets them in (exp17 26.x: the block markers alone)
        mask_in = op.video_mask.detach().cpu().numpy().reshape(-1) != 0
    elif getattr(op, "input_mask", None) is not None:
        # THE INPUT ELEMENTS: those whose mask lets in a stimulus column that VARIES -- a column changing at most twice
        # over the recording is a block marker (on once, off once; exp17's 9 markers reach every neuron in the markall
        # laws and name no input), a column changing more is a stimulus feature
        m = op.input_mask.detach().cpu().numpy()
        S_ = np.asarray(rec["stimulus"]) if "stimulus" in rec else None
        if m.ndim == 2 and m.shape[1] > 1 and S_ is not None and S_.shape[1] == m.shape[1]:
            varying = (np.diff(S_, axis=0) != 0).sum(0) > 2
            mask_in = (m[:, varying] != 0).any(1) if varying.any() else (m != 0).any(1)
            stim_cols = [int(j) for j in np.flatnonzero(varying)] if varying.any() and not varying.all() else None
        else:
            mask_in = (m != 0).any(1) if m.ndim == 2 else (m.reshape(-1) != 0)
    elif "input" in R["blocks"]:
        mask_in = np.linalg.norm(R["blocks"]["input"].cpu().numpy().reshape(N, -1), axis=1) > 0
    if mask_in is not None and mask_file is None:
        mask_file = os.path.join(gd, "input_neurons.npz")
        np.savez(mask_file, mask=mask_in.astype(np.float32))
        mask_file = {"mask": mask_file, "array": "mask"}
    regions = None
    if st["regions"]:
        from plexus.paths import graphs_data_path
        f = st["regions"] if os.path.isabs(st["regions"]) else graphs_data_path(*st["regions"].split("/"))
        za = np.load(f, allow_pickle=True)
        names = [str(x) for x in za["names"]]
        reg = np.asarray(za["regions"], bool)
        if reg.shape[0] != N:
            raise ValueError(f"task.graph.regions {st['regions']}: {reg.shape[0]} rows for {N} elements")
        # each element's most specific region among the table regions (the fewest elements), as exp17_atlas
        table = []
        try:
            from exp17_atlas import REGIONS
            table = [(f_, s_) for f_, s_ in REGIONS if f_ in names]
        except Exception:
            table = []
        if not table:                                      # not Z-Brain's table: the file's own regions, as named
            table = [(n_, n_) for n_ in names]
        lab = np.full(N, -1)
        best = np.full(N, np.inf)
        for k, (full, _) in enumerate(table):
            m = reg[:, names.index(full)]
            nk = int(m.sum())
            upd = m & (nk < best)
            lab[upd], best[upd] = k, nk
        inside = np.asarray(za["inside"], bool) if "inside" in za.files else np.ones(N, bool)
        lab[~inside] = -1
        regions = {"label": lab, "names": [s_ for _, s_ in table], "pos": np.asarray(za["atlas_um"], np.float64)
                   if "atlas_um" in za.files else np.asarray(rec["pos_um"], np.float64)}
    o0 = box["n_in"] - 1 + T._warmup(spec)
    ck_file = os.path.join(out, "models", f"{R['ckpt']}.pt")
    return dict(spec=spec, R=R, op=op, learn=R["learn"], box=box, device=device, out=out, stem=stem, gd=gd, st=st,
                rec=rec, X=X, N=N, T=int(X.shape[0]), o0=o0, n_free=int(X.shape[0]) - 1 - o0,
                frame_s=T._frame_s(rec), offsets=[int(v) for v in rec["offsets"]], names=list(rec["names"]),
                cond=np.asarray(rec["condition"]).astype(int), mask_in=mask_in, mask_file=mask_file, stim_cols=stim_cols,
                regions=regions,
                pos=np.asarray(rec["pos_um"], np.float64), norm=box["norm"], ck_mtime=os.path.getmtime(ck_file),
                force=bool(spec.get("_force", False)))


# ============================================================================== rollouts and scores
def rollout(ctx, name, variant, every_frame=False, n_frames=None):
    """A free rollout under `variant` (a `task.rollouts` entry without its name), cached under results/graph/ as
    <stem>_<name>_{free,movie}.npz (+ _traces.npy with every frame): reused while newer than the checkpoint. ->
    {movie, free, traces, summary}."""
    from plexus import trainer as T
    gd, stem = ctx["gd"], ctx["stem"]
    mv = os.path.join(gd, f"{stem}_{name}_movie.npz")
    tr = os.path.join(gd, f"{stem}_{name}_traces.npy") if every_frame else None
    sj = os.path.join(gd, f"{stem}_{name}_summary.json")
    v = {"name": name, **variant}
    v_rec = json.loads(json.dumps(v, default=_json_default))     # the cache is reused only for the SAME variant
    fresh = (os.path.exists(mv) and os.path.exists(sj) and os.path.getmtime(mv) > ctx["ck_mtime"]
             and (tr is None or os.path.exists(tr)))
    if fresh and not ctx["force"]:
        summary = json.load(open(sj))
        if summary.get("variant_spec", v_rec) == v_rec:           # a summary without `variant_spec` predates the check
            return {"movie": mv, "free": os.path.join(gd, f"{stem}_{name}_free.npz"), "traces": tr, "summary": summary,
                    "cached": True}
    t1 = time.time()
    free = T._trace_free(ctx["spec"], ctx["learn"], ctx["box"], ctx["device"], ctx["out"], f"{stem}_{name}",
                         variant=v, res_dir=gd, traces_path=tr, n_frames=n_frames)[0]
    free = {k: (v_ if not isinstance(v_, np.generic) else v_.item()) for k, v_ in free.items()}
    free["seconds"] = time.time() - t1
    free["variant_spec"] = v_rec                                   # the trainer's `variant` is the name alone
    json.dump(free, open(sj, "w"), indent=1, default=_json_default)
    cuts = "".join(f", {p} {c['cut']}/{c['of']} cut" for p, c in free.get("prune", {}).items()) \
        if isinstance(free.get("prune"), dict) else ""
    zi = free.get("zero_input")
    cuts += f", inputs cut {zi['n_cut']}/{zi['n_input']}" if isinstance(zi, dict) else ""
    print(f"[graph] rollout {name} ({time.time() - t1:.0f} s): brain-mean r {free['brain_mean_r']:+.3f}, R2 denoised "
          f"{free['r2_denoised']:+.3f}, {free['silenced']} silenced{cuts}", flush=True)
    return {"movie": mv, "free": os.path.join(gd, f"{stem}_{name}_free.npz"), "traces": tr, "summary": free, "cached": False}


def nominal(ctx, every_frame=False):
    """The nominal rollout: the test phase's own results/<stem>_movie.npz when it exists (the deck's numbers come from
    it), else a rollout here; with `every_frame` always a rollout here (its traces), checked against the test's."""
    tj = os.path.join(ctx["out"], "results", f"{ctx['stem']}_test.json")
    test_mv = os.path.join(ctx["out"], "results", f"{ctx['stem']}_movie.npz")
    if not every_frame and os.path.exists(test_mv):
        return {"movie": test_mv, "traces": None, "summary": (json.load(open(tj)).get("free") if os.path.exists(tj) else None),
                "cached": True, "from_test": True}
    r = rollout(ctx, "nominal", {}, every_frame=every_frame)
    if os.path.exists(tj):
        ref = json.load(open(tj)).get("free", {})
        if ref.get("r2_denoised") is not None:
            r["reproduces_test"] = {"r2_denoised_here": r["summary"]["r2_denoised"], "r2_denoised_test": ref["r2_denoised"],
                                    "difference": float(r["summary"]["r2_denoised"] - ref["r2_denoised"])}
    return r


def steady_mask(ctx, frames):
    """Which of the recording frames `frames` fall in the steady part: after the first of `bootstrap.blocks` equal
    blocks of the free frames (the opening transient from the recorded start)."""
    from plexus.tasks import trace_recording as TR
    nb, skip = int(ctx["st"]["bootstrap"]["blocks"]), 1
    blk, keep = TR.steady_blocks(ctx["n_free"], nb, skip)
    k = np.asarray(frames) - (ctx["o0"] + 1)
    k = np.clip(k, 0, ctx["n_free"] - 1)
    return keep[k], blk[k]


def movie_scores(ctx, npz, exclude=None, frames_sel=None):
    """A free rollout's scores from its movie npz: the brain-mean r / R2 / RMSE over the free frames (all, steady) and
    the per-neuron r, brain mean removed (`trace_recording.per_neuron_r`, the recording's neuron set, a flat learned
    trace 0) over the movie's frames (all, steady), the input and the other neurons apart. `exclude` (bool [N]): the
    elements left out (the clamped ones)."""
    from plexus.tasks import trace_recording as TR
    z = np.load(npz)
    fr = np.asarray(z["frames"])
    P = np.asarray(z["pred"], np.float32)
    X = ctx["X"][fr]
    keep = np.ones(ctx["N"], bool) if exclude is None else ~np.asarray(exclude, bool)
    st_m, _ = steady_mask(ctx, fr)
    o, p = np.asarray(z["mean_obs_all"], np.float64), np.asarray(z["mean_pred_all"], np.float64)
    n_all = len(o)
    st_f, _ = steady_mask(ctx, ctx["o0"] + 1 + np.arange(n_all))
    out = {"frames": {"free": int(n_all), "free_steady": int(st_f.sum()), "movie": int(len(fr)), "movie_steady": int(st_m.sum())}}
    for win, fm, ff in (("whole", np.ones(len(fr), bool), np.ones(n_all, bool)), ("steady", st_m, st_f)):
        bm = TR.brain_mean_metrics(o[ff], p[ff])
        pn = TR.per_neuron_r(P[fm][:, keep], X[fm][:, keep])
        r = np.full(ctx["N"], np.nan)
        r[keep] = pn["r"]
        row = {"brain_mean_r": bm["r"], "brain_mean_r2": bm["r2"], "brain_mean_rmse": bm["rmse"],
               "per_neuron_r": pn["mean"], "per_neuron_sd": pn["sd"], "n": pn["n"], "n_flat": pn["n_flat"]}
        if ctx["mask_in"] is not None:
            for lab, m in (("input", ctx["mask_in"] & keep), ("other", ~ctx["mask_in"] & keep)):
                v = r[m]
                v = v[np.isfinite(v)]
                row[lab] = {"per_neuron_r": float(v.mean()) if len(v) else None, "n": int(len(v)),
                            "n_flat": int((v == 0).sum())}
        out[win] = row
    out["silenced"] = int(np.asarray(z["silenced_all"])[-1]) if "silenced_all" in z.files else None
    return out


def bootstrap(ctx, npzs: dict, pairs, exclude=None):
    """The paired block bootstrap (`trace_recording.paired_block_bootstrap`) over the steady part, one neuron set for
    every law (recorded residual moving, finite in every law, not excluded): the brain-mean r over the free frames'
    blocks, the per-neuron r over the movie frames' blocks."""
    from plexus.tasks import trace_recording as TR
    bs = ctx["st"]["bootstrap"]
    Z = {k: np.load(f) for k, f in npzs.items()}
    k0 = list(Z)[0]
    fr = np.asarray(Z[k0]["frames"])
    for k in Z:
        if not np.array_equal(np.asarray(Z[k]["frames"]), fr):
            raise ValueError(f"bootstrap: {k} has other movie frames than {k0}")
    st_m, blk_m = steady_mask(ctx, fr)
    n_all = len(Z[k0]["mean_obs_all"])
    st_f, blk_f = steady_mask(ctx, ctx["o0"] + 1 + np.arange(n_all))
    X = ctx["X"][fr[st_m]].astype(np.float64)
    keep = TR.remove_brain_mean(X).std(0) > 1e-9
    if exclude is not None:
        keep &= ~np.asarray(exclude, bool)
    for k in Z:
        keep &= np.isfinite(np.asarray(Z[k]["pred"])).all(0)
    bm, loc = {}, {}
    for k in Z:
        o, p = np.asarray(Z[k]["mean_obs_all"], np.float64)[st_f], np.asarray(Z[k]["mean_pred_all"], np.float64)[st_f]
        bm[k] = TR.brain_mean_block_sums(o, p, blk_f[st_f] - 1)
        P = np.asarray(Z[k]["pred"], np.float64)[st_m]
        per, sc = TR.block_sums(P, X, blk_m[st_m] - 1, keep, ctx["device"])
        loc[k] = (per, sc, TR.local_r_from_sums(per.sum(0), sc.sum(0))[1])
    doc = TR.paired_block_bootstrap(bm, loc, pairs, int(bs["resamples"]), int(bs["seed"]), ctx["device"])
    doc["neurons"] = int(keep.sum())
    doc["window"] = "steady"
    return doc


def seed_spread(ctx, doc):
    """The seed pair's spread, from the controls test when it ran, else exp17's fallback."""
    c = (doc.get("controls") or {}).get("seed")
    if c and "scores" in c:
        n = doc["silence"]["rollouts"]["nominal"]["steady"] if "silence" in doc else None
        if n:
            return {"brain_mean_r": abs(n["brain_mean_r"] - c["scores"]["steady"]["brain_mean_r"]),
                    "per_neuron_r": abs(n["per_neuron_r"] - c["scores"]["steady"]["per_neuron_r"]), "from": c["run"]}
    return {**SEED_SPREAD, "from": "exp17 19.25 vs 19.26 (fallback)"}


# ============================================================================== the tests
def test_constants(ctx, doc):
    op, B, N = ctx["op"], ctx["R"]["blocks"], ctx["N"]
    mu, sd, _ = ctx["norm"]
    out = {"units": {"tau": "s", "V": "dF/F", "B": "L2 norm of the element's stimulus weights", "W": "normalised units"}}
    if "tau" in B:
        tau = ctx["frame_s"] / op._rate(B["tau"]).reshape(-1).double().cpu().numpy()
        out["tau_s"] = _q(tau)
        out["tau_s"]["at_floor"] = float((tau < 1.05 * tau.min()).mean())
        out["tau_s"]["at_ceiling"] = float((tau > 0.95 * tau.max()).mean())
    if "rest" in B:
        V = B["rest"].reshape(-1).double().cpu().numpy() * sd + mu
        out["V_dff"] = _q(V)
    if "input" in B:
        bn = np.linalg.norm(B["input"].double().cpu().numpy().reshape(N, -1), axis=1)
        out["B_norm"] = _q(bn[bn > 0])
        out["B_norm"]["n_nonzero"] = int((bn > 0).sum())
    roles = op.graph_roles()
    out["roles"] = roles
    out["silencers"] = op.silencers()
    out["edge_sets"] = {}
    win, wout = np.zeros(N), np.zeros(N)
    P_, N_ = np.zeros(N), np.zeros(N)
    for s, (snd, rcv, w) in op.edge_table().items():
        w_ = w.detach().double().cpu().numpy()
        snd_, rcv_ = snd.cpu().numpy(), rcv.cpu().numpy()
        out["edge_sets"][s] = {"edges": int(len(w_)), "share_positive": float((w_ > 0).mean()) if len(w_) else None,
                               "abs_w": _q(np.abs(w_)) if len(w_) else None, "per_element": float(len(w_) / N)}
        np.add.at(win, rcv_, w_)
        np.add.at(wout, snd_, w_)
        np.add.at(P_, snd_, np.clip(w_, 0, None))
        np.add.at(N_, snd_, np.clip(-w_, 0, None))
    for s, (param, strength, _) in op.prune_sets().items():
        out["edge_sets"].setdefault(s, {})["strength"] = _q(strength.double().cpu().numpy())
        out["edge_sets"][s]["learnable"] = param
    if out["edge_sets"] and (P_ + N_).max() > 0:
        tot = P_ + N_
        sender = tot > 0
        minority = np.minimum(P_, N_)[sender] / tot[sender]
        out["senders"] = {"n": int(sender.sum()), "excitatory_share": float((P_[sender] > N_[sender]).mean()),
                          "minority_share_median": float(np.median(minority)),
                          "one_sign_share": float((minority < 0.05).mean()), "W_in": _q(win), "W_out": _q(wout)}
    if ctx["regions"] is not None and "tau" in B:
        lab, names = ctx["regions"]["label"], ctx["regions"]["names"]
        out["tau_s_by_region"] = {names[k]: float(np.median(tau[lab == k])) for k in range(len(names)) if (lab == k).sum() >= MIN_REGION}
        if "senders" in out:
            out["excitatory_share_by_region"] = {names[k]: float((P_ > N_)[(lab == k) & (tot > 0)].mean())
                                                 for k in range(len(names)) if ((lab == k) & (tot > 0)).sum() >= MIN_REGION}
    if ctx["mask_in"] is not None and "tau" in B:
        out["tau_s"]["median_input"] = float(np.median(tau[ctx["mask_in"]]))
        out["tau_s"]["median_other"] = float(np.median(tau[~ctx["mask_in"]]))
    return out


def _q(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    p = np.percentile(x, [2, 25, 50, 75, 98])
    return {"n": int(len(x)), "p2": float(p[0]), "p25": float(p[1]), "median": float(p[2]), "p75": float(p[3]),
            "p98": float(p[4]), "mean": float(x.mean())}


def silence_variants(ctx) -> dict:
    """The silence test's variants: every silencer at 0 (`W0_all`), one edge set at a time when there are several,
    the stimulus off when the task has a drive."""
    op, spec = ctx["op"], ctx["spec"]
    sil = op.silencers()
    learn_names = {e.get("param") or e.get("block") for e in spec["learnable"]}
    sil = [s for s in sil if s in learn_names]
    V = {}
    if sil:
        V["W0_all"] = {"zero": sil}
        if len(sil) > 1:
            for s in sil:
                V[f"W0_{s}"] = {"zero": [s]}
    if "drive" in spec["task"]:
        V["no_stimulus"] = {"drive": "off"}
    return V


def test_silence(ctx, doc):
    every = bool(ctx["st"]["every_frame"]) and any(t in ctx["st"]["tests"] for t in ("blocks", "leadlag"))
    nom = nominal(ctx, every_frame=every)
    out = {"rollouts": {"nominal": {**movie_scores(ctx, nom["movie"]), "file": nom["movie"],
                                    "from_test": bool(nom.get("from_test"))}}, "variants": {}}
    if nom.get("reproduces_test"):
        out["reproduces_test"] = nom["reproduces_test"]
    npzs = {"nominal": nom["movie"]}
    for name, v in silence_variants(ctx).items():
        r = rollout(ctx, name, v, every_frame=(every and name == "W0_all"))
        out["rollouts"][name] = {**movie_scores(ctx, r["movie"]), "file": r["movie"], "silenced": r["summary"].get("silenced")}
        out["variants"][name] = v
        npzs[name] = r["movie"]
    names = [k for k in npzs if k != "nominal"]
    if names:
        out["tests"] = bootstrap(ctx, npzs, [("nominal", k) for k in names])
    return out


def test_blocks(ctx, doc):
    """Full against silenced within each stimulus block (tools/exp17_w0_blocks.py): every-frame traces, the first
    `steady_skip_min` minutes after the first free frame left out, the gap on each half of the block too."""
    from plexus.tasks import trace_recording as TR
    nomr = nominal(ctx, every_frame=True)
    sil = silence_variants(ctx)
    if "W0_all" not in sil:
        raise NotImplementedError("blocks: the law has nothing that silences its graph")
    w0 = rollout(ctx, "W0_all", sil["W0_all"], every_frame=True)
    F = np.load(nomr["traces"], mmap_mode="r")
    Z = np.load(w0["traces"], mmap_mode="r")
    X, off, names = ctx["X"], ctx["offsets"], ctx["names"]
    first = ctx["o0"] + 1
    t0 = first + int(round(float(ctx["st"]["steady_skip_min"]) * 60 / ctx["frame_s"]))
    fill = X.mean(0)
    per = []

    def arr(M, a, b):
        A = np.array(M[a:b], np.float32)
        bad = ~np.isfinite(A)
        if bad.any():
            A[bad] = np.broadcast_to(fill, A.shape)[bad]       # a silenced element frozen at its mean, as the traces tool
        return A
    for k, b in enumerate(names):
        a_, e_ = max(off[k], t0), off[k + 1]
        if e_ - a_ < 40:
            per.append({"block": b, "skipped": "fewer than 40 frames after the skip"})
            continue
        Xb = X[a_:e_]
        row = {"block": b, "frames": [int(a_), int(e_)]}
        row["full"] = TR.block_scores(Xb, arr(F, a_, e_), ctx["device"])
        row["w0"] = TR.block_scores(Xb, arr(Z, a_, e_), ctx["device"])
        h = (a_ + e_) // 2
        halves = []
        for s_, t_ in ((a_, h), (h, e_)):
            f_, z_ = (TR.block_scores(X[s_:t_], arr(M, s_, t_), ctx["device"]) for M in (F, Z))
            halves.append({m: f_[m] - z_[m] for m in ("brain_mean_r", "per_neuron_r")})
        row["gap"] = {m: row["full"][m] - row["w0"][m] for m in ("brain_mean_r", "per_neuron_r")}
        row["gap_halves"] = halves
        per.append(row)
    return {"skip_min": ctx["st"]["steady_skip_min"], "first_free_frame": int(first), "scored_from_frame": int(t0),
            "silenced": sil["W0_all"], "per_block": per}


def test_controls(ctx, doc):
    """The trained controls, scored as the graph is and tested against it; a control not landed is `pending`."""
    from plexus import trainer as T
    ctr = ctx["st"]["controls"]
    if not ctr:
        raise NotImplementedError("controls: none named (task.graph.controls or --graph-controls)")
    nom = nominal(ctx)
    out = {}
    npzs = {"nominal": nom["movie"]}
    for role, run in ctr.items():
        try:
            sc = T.load(run)
        except Exception as e:                           # a spec that does not exist: said, not raised
            out[role] = {"run": run, "error": str(e)}
            continue
        f = os.path.join(T.out_dir(sc, None), "results", f"{sc['name']}_movie.npz")
        if not os.path.exists(f):
            out[role] = {"run": run, "pending": True}
            continue
        try:
            out[role] = {"run": run, "scores": movie_scores(ctx, f), "file": f}
            npzs[role] = f
        except Exception as e:
            out[role] = {"run": run, "error": f"{type(e).__name__}: {e}"}
    landed = [k for k in npzs if k != "nominal"]
    if landed:
        out["tests"] = bootstrap(ctx, npzs, [("nominal", k) for k in landed])
    if "seed" in out and "scores" in out["seed"]:
        n = movie_scores(ctx, nom["movie"])["steady"]
        s = out["seed"]["scores"]["steady"]
        out["seed_spread"] = {m: abs(n[m] - s[m]) for m in ("brain_mean_r", "per_neuron_r")}
    return out


def test_clamp(ctx, doc):
    """THE RELAY TEST: the input neurons given their recorded traces every tick, with and without the graph, scored
    on the free neurons (tools/exp17_clamp.py)."""
    if ctx["mask_file"] is None or ctx["mask_in"] is None or not ctx["mask_in"].any() or ctx["mask_in"].all():
        raise NotImplementedError("clamp: no input neurons (task.graph.inputs or the law's input mask)")
    sil = silence_variants(ctx)
    cl = {"clamp": dict(ctx["mask_file"])}
    rs = {"clamp_in": rollout(ctx, "clamp_in", cl)}
    if "W0_all" in sil:
        rs["clamp_in_W0"] = rollout(ctx, "clamp_in_W0", {**cl, **sil["W0_all"]})
        rs["W0_all"] = rollout(ctx, "W0_all", sil["W0_all"])
    nom = nominal(ctx)
    ex = ctx["mask_in"]
    out = {"n_clamped": int(ex.sum()), "n_free": int((~ex).sum()), "rollouts": {}}
    npzs = {"nominal": nom["movie"]}
    out["rollouts"]["nominal"] = movie_scores(ctx, nom["movie"], exclude=ex)
    for k, r in rs.items():
        out["rollouts"][k] = {**movie_scores(ctx, r["movie"], exclude=ex), "file": r["movie"]}
        npzs[k] = r["movie"]
    pairs = [("clamp_in", "clamp_in_W0"), ("nominal", "W0_all"), ("clamp_in", "nominal")] if "W0_all" in rs else [("clamp_in", "nominal")]
    out["tests"] = bootstrap(ctx, npzs, pairs, exclude=ex)
    if "W0_all" in rs:                                   # with W = 0 the free neurons cannot see the clamped ones
        a = np.asarray(np.load(rs["W0_all"]["movie"])["pred"], np.float32)[:, ~ex]
        b = np.asarray(np.load(rs["clamp_in_W0"]["movie"])["pred"], np.float32)[:, ~ex]
        ok = np.isfinite(a) & np.isfinite(b)
        out["identity_max_abs"] = float(np.abs(a[ok] - b[ok]).max()) if ok.any() else None
        g = out["rollouts"]
        out["relay_gap"] = {m: g["clamp_in"]["steady"][m] - g["clamp_in_W0"]["steady"][m] for m in ("brain_mean_r", "per_neuron_r")}
        out["graph_gap"] = {m: g["nominal"]["steady"][m] - g["W0_all"]["steady"][m] for m in ("brain_mean_r", "per_neuron_r")}
    return out


def gmm2(x, it=200):
    """A two-Gaussian mixture on 1-D x by EM (tools/exp17_prune.py): (weights, means, sds), the lower mean first."""
    m = np.percentile(x, [25, 75]).astype(np.float64)
    s = np.array([x.std(), x.std()]) / 2
    w = np.array([0.5, 0.5])
    for _ in range(it):
        p = w / (s * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x[:, None] - m) / s) ** 2)
        g = p / np.maximum(p.sum(1, keepdims=True), 1e-300)
        n = np.maximum(g.sum(0), 1e-12)
        w, m = n / len(x), (g * x[:, None]).sum(0) / n
        s = np.maximum(np.sqrt((g * (x[:, None] - m) ** 2).sum(0) / n), 1e-6)
    o = np.argsort(m)
    return w[o], m[o], s[o]


def crossing(w, m, s):
    """Where w0 N(m0, s0) = w1 N(m1, s1) between the two means."""
    xs = np.linspace(m[0], m[1], 20001)
    d = [w[k] / s[k] * np.exp(-0.5 * ((xs - m[k]) / s[k]) ** 2) for k in (0, 1)]
    return float(xs[np.argmin(np.abs(d[0] - d[1]))])


def _ladder_judge(ctx, doc, nominal_steady, rows, spread):
    """Which rungs are removable: neither steady metric more than the seed spread BELOW the nominal's (a rung that
    scores better than the nominal is removable too: the lattice grid's weak weights, 23.3)."""
    for row in rows:
        s = row["scores"]["steady"]
        row["delta"] = {m: s[m] - nominal_steady[m] for m in ("brain_mean_r", "per_neuron_r")}
        row["removable"] = bool(all(row["delta"][m] >= -spread[m] for m in ("brain_mean_r", "per_neuron_r")))


def test_prune_w(ctx, doc):
    """THE PRUNING LADDER per edge set (tools/exp17_prune.py): the two-Gaussian threshold on log10 strength, rollouts
    along a geometric ladder around it, removable while neither steady metric falls more than the seed spread; then
    the joint cut of every set at its threshold. A set without a dead mode (the mixture's lower component under 10 %
    of the weights, or wider than two decades, or a crossing outside the 1st-99th percentile: the lattice grid) gets
    the deck's grid rule instead: the ladder is the nonzero weights' own quantiles (`prune.quantiles`)."""
    op, spec = ctx["op"], ctx["spec"]
    learn_names = {e.get("param") or e.get("block") for e in spec["learnable"]}
    sets = {s: v for s, v in op.prune_sets().items() if v[0] in learn_names}
    if not sets:
        raise NotImplementedError("prune_w: the law has no prunable edge set among the learnables")
    pr = ctx["st"]["prune"]
    nom = movie_scores(ctx, nominal(ctx)["movie"])["steady"]
    spread = seed_spread(ctx, doc)
    out = {"ruler": {"window": "steady", "metrics": ["brain_mean_r", "per_neuron_r"], "seed_spread": spread}, "per_set": {}}
    joint = {}
    for s, (param, strength, to_param) in sets.items():
        st_ = strength.double().cpu().numpy()
        pos = st_[st_ > 0]
        if len(pos) < 10:
            out["per_set"][s] = {"learnable": param, "n": int(len(st_)), "skipped": "fewer than 10 nonzero weights"}
            continue
        lg = np.log10(pos)
        w_, m_, sd_ = gmm2(lg)
        t_star = 10 ** crossing(w_, m_, sd_)
        p01, p99 = np.percentile(pos, [1, 99])
        if w_[0] < 0.1 or sd_[0] > 2.0 or not (p01 <= t_star <= p99):
            rule, qs = "quantile", [float(q) for q in pr["quantiles"]]
            lad, tag = np.quantile(pos, qs), "q"
        else:
            rule, qs = "gmm", None
            lad, tag = t_star * np.geomspace(float(pr["span"][0]), float(pr["span"][1]), int(pr["ladder"])), ""
        rows = []
        for j, t in enumerate(lad):
            tp = to_param(t)
            r = rollout(ctx, f"prune_{s}_{tag}{j}", {"prune": {"thresholds": {param: float(tp)}}})
            cut = r["summary"].get("prune", {}).get(param, {})
            rows.append({"threshold": float(t), "threshold_param": float(tp), "cut": cut.get("cut"), "of": cut.get("of"),
                         "kept_share": (1 - cut["cut"] / cut["of"]) if cut.get("of") else None,
                         "scores": movie_scores(ctx, r["movie"]), "file": r["movie"]})
        _ladder_judge(ctx, doc, nom, rows, spread)
        rem = [row for row in rows if row["removable"]]
        chosen = max(rem, key=lambda row: row["threshold"]) if rem else None
        out["per_set"][s] = {"learnable": param, "n": int(len(st_)), "n_zero": int((st_ == 0).sum()),
                             "gmm": {"weights": w_.tolist(), "means_log10": m_.tolist(), "sds_log10": sd_.tolist()},
                             "crossing": float(t_star), "ladder_rule": rule, "quantiles": qs, "ladder": rows,
                             "threshold": chosen["threshold"] if chosen else None,
                             "kept_share": chosen["kept_share"] if chosen else None}
        if chosen:
            joint[param] = chosen["threshold_param"]
    if len(joint) > 1:
        r = rollout(ctx, "prune_joint", {"prune": {"thresholds": joint}})
        row = {"thresholds_param": joint, "scores": movie_scores(ctx, r["movie"]), "file": r["movie"],
               "cut": r["summary"].get("prune")}
        _ladder_judge(ctx, doc, nom, [row], spread)
        out["joint"] = row
    return out


def test_prune_b(ctx, doc):
    """THE LADDER ON THE INPUT WEIGHTS (Cedric, 2026-10-10): the input neurons below a quantile of |B_i| silenced, the
    rollout as the ruler; where the survivors sit when regions are given. |B_i| and the cut run over the STIMULUS
    columns of the input block (those changing more than twice over the recording) when the law also carries block
    markers to every neuron (the markall laws): the markers stay, so the ladder removes stimulus weights, not context."""
    spec, B = ctx["spec"], ctx["R"]["blocks"]
    learn_names = {e.get("param") or e.get("block") for e in spec["learnable"]}
    if "input" not in learn_names or "input" not in B or "drive" not in spec["task"]:
        raise NotImplementedError("prune_b: the law has no learned input weights to cut")
    from plexus import trainer as T
    param = "video_out" if "video_out" in learn_names else "input"          # a video law: the encoder's output rows b_i
    cols = ctx.get("stim_cols") if param == "input" else None
    src = B["input"] if param == "input" else ctx["learn"].p[T._learnable_key(spec, param)].detach()
    Bm = src.double().cpu().numpy().reshape(ctx["N"], -1)
    bn = np.linalg.norm(Bm[:, cols] if cols else Bm, axis=1)
    inp = bn > 0
    if not inp.any():
        raise NotImplementedError(f"prune_b: every row of `{param}` is 0, nothing to cut")
    nom = movie_scores(ctx, nominal(ctx)["movie"])["steady"]
    spread = seed_spread(ctx, doc)
    rows = []
    for j, q in enumerate(ctx["st"]["prune_b"]["quantiles"]):
        r = rollout(ctx, f"zero_in_{j}", {"zero_input": {"quantile": float(q), **({"columns": cols} if cols else {}),
                                                         **({"param": param} if param != "input" else {})}})
        zi = r["summary"].get("zero_input", {})
        row = {"quantile": float(q), "threshold": zi.get("threshold"), "n_cut": zi.get("n_cut"), "n_input": zi.get("n_input"),
               "scores": movie_scores(ctx, r["movie"]), "file": r["movie"]}
        if ctx["regions"] is not None and zi.get("threshold") is not None:
            lab, names = ctx["regions"]["label"], ctx["regions"]["names"]
            surv = inp & (bn >= zi["threshold"])
            row["survivors_by_region"] = {names[k]: int((surv & (lab == k)).sum()) for k in range(len(names)) if (surv & (lab == k)).sum()}
        rows.append(row)
    _ladder_judge(ctx, doc, nom, rows, spread)
    rem = [row for row in rows if row["removable"]]
    chosen = max(rem, key=lambda row: row["quantile"]) if rem else None
    return {"ruler": {"window": "steady", "seed_spread": spread}, "n_input": int(inp.sum()), "B_norm": _q(bn[inp]),
            "param": param, "columns": len(cols) if cols else "all",
            "columns_rule": "the stimulus columns (changing more than twice over the recording); the block markers kept"
            if cols else f"every column of `{param}`",
            "ladder": rows, "quantile": chosen["quantile"] if chosen else None,
            "n_needed": (chosen["n_input"] - chosen["n_cut"]) if chosen else None}


def _state_star(ctx, frames=None):
    """z* in the law's normalised units: the recording's mean over `frames` (default all), or the rest block."""
    mu, sd, _ = ctx["norm"]
    if ctx["st"]["spectrum"]["state"] == "rest" and "rest" in ctx["R"]["blocks"]:
        return ctx["R"]["blocks"]["rest"].reshape(-1).double().cpu().numpy()
    X = ctx["X"] if frames is None else ctx["X"][frames]
    return (X.mean(0).astype(np.float64) - mu) / sd


def _omega_mean(ctx, frames):
    op = ctx["op"]
    if op.graph_roles().get("omega_mlp") != "time" and "omega_table" not in op.graph_roles():
        return None
    fs = np.unique(np.linspace(frames[0], frames[-1], min(64, len(frames))).round().astype(int))
    O = op.omega_at(fs)
    return O.mean(0).double().cpu().numpy() if O is not None else None


def _jacobian(ctx, z_star, omega, frame=None):
    """J per second as a LinearOperator (and M, r): J = diag(r)(M - I) / frame_s."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import LinearOperator, aslinearoperator
    op = ctx["op"]
    if frame is not None:
        op.frame = int(frame)
    M = op.linearise(z_star, omega)
    r = op._rate(ctx["R"]["blocks"]["tau"]).reshape(-1).double().cpu().numpy() / ctx["frame_s"]
    N = ctx["N"]
    Ml = aslinearoperator(M) if not isinstance(M, LinearOperator) else M

    def mv(v):
        v = np.asarray(v, np.float64).reshape(-1)
        return r * (Ml.matvec(v) - v)

    def rmv(v):
        v = np.asarray(v, np.float64).reshape(-1)
        return Ml.rmatvec(r * v) - r * v
    J = LinearOperator((N, N), matvec=mv, rmatvec=rmv, dtype=np.float64)
    J.M, J.r = M, r
    return J, M, r


def _mode_where(v, pos, lab=None, names=None):
    p = np.abs(np.asarray(v)) ** 2
    p = p / max(p.sum(), 1e-300)
    top = np.sort(p)[::-1]
    c = (p[:, None] * pos).sum(0)
    out = {"n_eff": float(1.0 / max((p ** 2).sum(), 1e-300)), "top20_share": float(top[:20].sum()),
           "spread_um": float(np.sqrt((p * ((pos - c) ** 2).sum(1)).sum()))}
    if lab is not None:
        share = {names[k]: float(p[lab == k].sum()) for k in range(len(names))}
        out["region_mass"] = dict(sorted(share.items(), key=lambda kv: -kv[1])[:3])
    return out


def _csr_torch(M, device, dtype=torch.float64):
    """A scipy sparse matrix as a torch CSR tensor on `device`."""
    import warnings
    import scipy.sparse as sps
    M = sps.csr_matrix(M)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                            # "sparse CSR tensor support is in beta"
        return torch.sparse_csr_tensor(torch.as_tensor(M.indptr, dtype=torch.int64), torch.as_tensor(M.indices, dtype=torch.int64),
                                       torch.as_tensor(M.data, dtype=dtype), size=M.shape).to(device)


def _factors(M):
    """The coupling M as sparse factors applied right to left ([M] for a sparse matrix, the grid's (Dec, Hop, Enc), the
    mean field's rank-one pair); None for a LinearOperator without `factors`."""
    import scipy.sparse as sps
    if sps.issparse(M):
        return [M]
    return list(M.factors) if getattr(M, "factors", None) is not None else None


class _Krylov:
    """A = diag(r) (F_1 ... F_n - leak I) as a complex matvec on `device` (tools/exp17_jacobian.py's Jac): the Jacobian
    per second with r = 1/tau and leak 1; the coupling M alone with r = 1, leak 0."""

    def __init__(self, factors, r, device, leak=1.0):
        self.t = [_csr_torch(f, device) for f in factors]
        self.r = torch.as_tensor(np.asarray(r, np.float64), device=device)[:, None]
        self.leak, self.N, self.dev = float(leak), len(r), device

    def mv_c(self, x):                                             # complex128 [N] -> complex128 [N]
        X = torch.stack([x.real, x.imag], 1)
        y = X
        for f in reversed(self.t):
            y = f @ y
        y = self.r * (y - self.leak * X)
        return torch.complex(y[:, 0], y[:, 1])

    def dense(self):
        N = self.N
        D = torch.eye(N, dtype=torch.float64, device=self.dev)
        for f in reversed(self.t):
            D = f @ D
        return (self.r * (D - self.leak * torch.eye(N, dtype=torch.float64, device=self.dev))).cpu().numpy()


def _krylov_schur(A, k, which, m=None, tol=1e-5, maxit=4000, seed=0):
    """THE k EIGENPAIRS OF A WITH THE LARGEST Re (LR), |Im| (LI) OR |lambda| (LM), Krylov-Schur on A's device (Stewart
    2002, SIAM J. Matrix Anal. Appl. 23:601; tools/exp17_jacobian.py): an m-step Arnoldi factorisation (complex, classical
    Gram-Schmidt twice), the Schur form of H_m reordered so the wanted Ritz values lead, truncated to them, extended
    again; converged when every wanted pair's residual |A x - lambda x| / |x| < tol max(|lambda|, 0.01). Each restart
    keeps the k wanted and a buffer of k / 2 more (the k-th sits in a dense cluster of slow modes). ARPACK on the CPU
    gives the same pairs in 10-40 min per call at 100,000 neurons; this takes seconds to minutes on the GPU."""
    import scipy.linalg as sl
    dev, N = A.dev, A.N
    m = min(m or max(4 * k, k + 80), N - 1)
    g = np.random.default_rng(seed).standard_normal(N)
    V = torch.zeros(N, m + 1, dtype=torch.complex128, device=dev)
    H = np.zeros((m + 1, m), np.complex128)
    V[:, 0] = torch.as_tensor(g / np.linalg.norm(g), device=dev)
    key = {"LR": lambda x: x.real, "LI": lambda x: np.abs(x.imag), "LM": lambda x: np.abs(x)}[which]
    p, nmv = 0, 0
    for it in range(maxit):
        for j in range(p, m):
            w = A.mv_c(V[:, j])
            nmv += 1
            Vj = V[:, :j + 1]
            h = Vj.conj().T @ w
            w = w - Vj @ h
            h2 = Vj.conj().T @ w                                   # the second pass (DGKS)
            w = w - Vj @ h2
            h = (h + h2).cpu().numpy()
            b = float(torch.linalg.vector_norm(w))
            H[:j + 1, j] = h
            H[j + 1, j] = b
            V[:, j + 1] = w / max(b, 1e-300)
        T, Q = sl.schur(H[:m, :m], output="complex")
        kb = min(k + max(10, k // 2), m - 20) if m > 20 else k
        th_ = np.sort(key(np.diag(T)))[::-1][kb - 1]
        T, Q, sdim = sl.schur(H[:m, :m], output="complex",
                              sort=lambda x: key(np.array([x]))[0] >= th_ - 1e-14 * max(1, abs(th_)))
        p = min(max(sdim, kb), m - 10) if m > 10 else max(sdim, kb)
        ev, Y = np.linalg.eig(T[:p, :p])
        bq = H[m, m - 1] * Q[m - 1, :p]                            # the residual row after the truncation
        res = np.abs(bq @ Y) / np.linalg.norm(Y, axis=0)
        o = np.argsort(-key(ev))[:k]
        ok = res[o] < tol * np.maximum(np.abs(ev[o]), 1e-2)
        Qt = torch.as_tensor(Q[:, :p], device=dev)
        Vn = V[:, :m] @ Qt
        if ok.all() or it == maxit - 1:
            X = (Vn @ torch.as_tensor(Y[:, o], device=dev)).cpu().numpy()
            return ev[o], X / np.linalg.norm(X, axis=0), {"solver": "krylov_schur", "restarts": it + 1, "matvecs": nmv,
                                                           "max_residual": float(res[o].max()), "converged": bool(ok.all()), "m": m}
        V[:, :p] = Vn
        V[:, p] = V[:, m]
        H[:] = 0
        H[:p, :p] = T[:p, :p]
        H[p, :p] = bq


def _eigs(A, k, which, N):
    """k eigenpairs of A (largest real part LR, |imaginary| LI, modulus LM) -> (vals, vecs, info | error string): a
    `_Krylov` operator goes to the Krylov-Schur solver on its device (dense numpy below 2,000 elements, exact), a scipy
    matrix or LinearOperator to ARPACK on the CPU."""
    k = int(min(k, max(1, N - 2)))
    try:
        if isinstance(A, _Krylov):
            if N <= 2000:
                vals, vecs = np.linalg.eig(A.dense())
                key = {"LR": vals.real, "LI": np.abs(vals.imag), "LM": np.abs(vals)}[which]
                o = np.argsort(-key)[:k]
                return vals[o], vecs[:, o], {"solver": "dense"}
            return _krylov_schur(A, k, which)
        from scipy.sparse.linalg import eigs
        vals, vecs = eigs(A, k=k, which=which, tol=1e-6, maxiter=20000, ncv=min(N, max(2 * k + 1, 20)))
        return vals, vecs, {"solver": "arpack"}
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def test_spectrum(ctx, doc):
    """THE LAW LINEARISED at the recording's mean state (per stimulus block for a law with an angle of time): the
    Jacobian's eigenvalues of largest real part and of largest imaginary part per second, against the leak alone; the
    leading modes' size (n_eff), concentration and spatial spread; the spectral radius of M and the net recurrent gain
    (row sums of M) per neuron."""
    sp_ = ctx["st"]["spectrum"]
    op = ctx["op"]
    roles = op.graph_roles()
    per_block = any(v == "time" and k.startswith("alpha") for k, v in roles.items())
    pos = ctx["pos"] if ctx["regions"] is None else ctx["regions"]["pos"]
    lab = ctx["regions"]["label"] if ctx["regions"] else None
    names = ctx["regions"]["names"] if ctx["regions"] else None
    N = ctx["N"]

    def one(frames, tag):
        fr0 = np.arange(ctx["T"]) if frames is None else frames
        z = _state_star(ctx, frames)
        om = _omega_mean(ctx, fr0)
        J, M, r = _jacobian(ctx, z, om, frame=None if frames is None else int(np.median(frames)))
        res = {"state": sp_["state"], "frames": [int(fr0[0]), int(fr0[-1])], "leak_abscissa_per_s": float(-r.min()),
               "leak_slowest_s": float(1.0 / r.min()), "leak_fastest_s": float(1.0 / r.max())}
        import scipy.sparse as sps
        if sps.issparse(M):
            rs = np.asarray(M.sum(1)).ravel()
            res["row_sum_gain"] = _q(rs)
            res["nnz"] = int(M.nnz)
        F = _factors(M)
        on_device = F is not None and sp_.get("solver", "krylov_schur") == "krylov_schur"
        AM = _Krylov(F, np.ones(N), ctx["device"], leak=0.0) if on_device else M       # M alone
        AJ = _Krylov(F, r, ctx["device"], leak=1.0) if on_device else J                # diag(r) (M - I)
        if sps.issparse(M) and M.nnz == 0:
            res["M_spectral_radius"] = 0.0                     # no coupling at all
        else:
            vals, vecs, info = _eigs(AM, 6, "LM", N)
            res["M_spectral_radius"] = float(np.abs(vals).max()) if vals is not None else None
            if isinstance(info, str):
                res["M_spectral_radius_error"] = info
        vals, vecs, info = _eigs(AJ, int(sp_["k"]), "LR", N)
        if isinstance(info, str):
            res["eigs_error"] = info
            return res
        res["solver"] = info
        o = np.argsort(-vals.real)
        vals, vecs = vals[o], vecs[:, o]
        res["abscissa_per_s"] = float(vals[0].real)
        res["n_unstable"] = int((vals.real > 0).sum())
        res["n_slower_than_leak"] = int((vals.real > -r.min() + 1e-9).sum())   # above the slowest leak
        res["eig_re_per_s"] = vals.real.tolist()
        res["eig_im_per_s"] = vals.imag.tolist()
        res["leading_modes"] = [{"re_per_s": float(vals[i].real), "im_per_s": float(vals[i].imag),
                                 "time_constant_s": float(-1 / vals[i].real) if vals[i].real < 0 else None,
                                 "period_s": float(2 * np.pi / abs(vals[i].imag)) if abs(vals[i].imag) > 1e-12 else None,
                                 **_mode_where(vecs[:, i], pos, lab, names)} for i in range(min(8, len(vals)))]
        res["n_brainwide_slow"] = int(sum(1 for m in res["leading_modes"] if m["n_eff"] > 0.01 * N))
        vi, vvi, info_i = _eigs(AJ, int(sp_["k_imag"]), "LI", N)
        if isinstance(info_i, str):
            res["oscillatory_error"] = info_i
        if vi is not None:
            oi = np.argsort(-np.abs(vi.imag))
            res["oscillatory"] = [{"re_per_s": float(vi[i].real), "im_per_s": float(vi[i].imag),
                                   "period_s": float(2 * np.pi / abs(vi[i].imag)) if abs(vi[i].imag) > 1e-12 else None,
                                   **_mode_where(vvi[:, i], pos, lab, names)} for i in oi[:5]]
        return res
    if per_block:
        out = {"per_block": {}}
        for k, b in enumerate(ctx["names"]):
            a_, e_ = ctx["offsets"][k], ctx["offsets"][k + 1]
            out["per_block"][b] = one(np.arange(a_, e_), b)
        out["session"] = one(None, "session")
    else:
        out = one(None, "session")
    return out


def _source_regions(ctx):
    reg = ctx["regions"]
    if reg is None:
        raise NotImplementedError("impulse: needs task.graph.regions (an atlas npz)")
    lab, names = reg["label"], reg["names"]
    want = ctx["st"]["pulse"].get("regions")
    if want:
        srcs = [(n_, names.index(n_)) for n_ in want if n_ in names]
    else:
        cnt = [(int((lab == k).sum()), k) for k in range(len(names))]
        srcs = [(names[k], k) for n_, k in sorted(cnt, reverse=True)[:6] if n_ >= MIN_REGION]
    if not srcs:
        raise NotImplementedError("impulse: no region with enough elements")
    return srcs


def test_impulse(ctx, doc):
    """PROPAGATION: the linear impulse response from the Jacobian (RK4 over window_s) and the nonlinear pulse rollout
    (drive off; a region held at mu + level_z sd for duration_s after settle_s), region by region, with and without
    the graph; the response movie of the first source."""
    srcs = _source_regions(ctx)
    reg, lab, names = ctx["regions"], ctx["regions"]["label"], ctx["regions"]["names"]
    ps = ctx["st"]["pulse"]
    fs = ctx["frame_s"]
    z = _state_star(ctx)
    om = _omega_mean(ctx, np.arange(ctx["T"]))
    J, M, r = _jacobian(ctx, z, om)
    targets = [k for k in range(len(names)) if (lab == k).sum() >= MIN_REGION]
    tnames = [names[k] for k in targets]
    out = {"sources": [n_ for n_, _ in srcs], "targets": tnames, "level_z": ps["level_z"], "duration_s": ps["duration_s"],
           "window_s": ps["window_s"], "linear": {}, "pulse": {}}
    # LINEAR: dz/dt = J z from z0 = level_z on the source, RK4, dt = frame / 4
    dt, n_steps = fs / 4.0, int(round(float(ps["window_s"]) / (fs / 4.0)))
    for sname, sk in srcs:
        z0 = np.zeros(ctx["N"])
        z0[lab == sk] = float(ps["level_z"])
        zt, mass, peak_t, peak_v = z0.copy(), np.zeros(len(targets)), np.zeros(len(targets)), np.zeros(len(targets))
        for i in range(n_steps):
            k1 = J.matvec(zt)
            k2 = J.matvec(zt + 0.5 * dt * k1)
            k3 = J.matvec(zt + 0.5 * dt * k2)
            k4 = J.matvec(zt + dt * k3)
            zt = zt + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            for j, tk in enumerate(targets):
                v = float(np.abs(zt[lab == tk]).mean())
                mass[j] += v * dt
                if v > peak_v[j]:
                    peak_v[j], peak_t[j] = v, (i + 1) * dt
        own = mass[targets.index(sk)] if sk in targets else float("nan")
        out["linear"][sname] = {"integrated_abs_z_s": dict(zip(tnames, mass.tolist())), "peak_s": dict(zip(tnames, peak_t.tolist())),
                                "peak_z": dict(zip(tnames, peak_v.tolist())),
                                "outside_share": float((mass.sum() - own) / mass.sum()) if mass.sum() > 0 else None}
    # NONLINEAR: pulse rollouts against the resting baseline (drive off), the response per target region
    settle, dur, win = float(ps["settle_s"]), float(ps["duration_s"]), float(ps["window_s"])
    n_fr = int(round((settle + dur + win) / fs)) + 2
    has_drive = "drive" in ctx["spec"]["task"]
    base_v = {"drive": "off"} if has_drive else {}
    base = rollout(ctx, "pulse_base", base_v, n_frames=n_fr)
    sil = silence_variants(ctx).get("W0_all")
    base0 = rollout(ctx, "pulse_base_W0", {**base_v, **sil}, n_frames=n_fr) if sil else None
    mu, sd, _ = ctx["norm"]
    mdir = os.path.join(ctx["gd"], "masks")
    os.makedirs(mdir, exist_ok=True)
    Pb = np.asarray(np.load(base["movie"])["pred"], np.float32)
    fr_b = np.asarray(np.load(base["movie"])["frames"])
    t_rel = (fr_b - fr_b[0]) * fs
    on = t_rel >= settle
    first_movie = None
    for sname, sk in srcs:
        mf = os.path.join(mdir, re.sub(r"[^A-Za-z0-9_]+", "_", sname) + ".npz")
        np.savez(mf, mask=(lab == sk).astype(np.float32))
        pv = {**base_v, "pulse": {"mask": mf, "array": "mask", "level_z": float(ps["level_z"]), "start_s": settle, "duration_s": dur}}
        rp = rollout(ctx, f"pulse_{re.sub(r'[^A-Za-z0-9_]+', '_', sname)}", pv, n_frames=n_fr)
        Pp = np.asarray(np.load(rp["movie"])["pred"], np.float32)
        dz = (Pp - Pb) / sd
        res = {"integrated_abs_z_s": {}, "peak_s": {}, "peak_z": {}}
        for tk, tn in zip(targets, tnames):
            v = np.nanmean(np.abs(dz[:, lab == tk]), 1)
            v = np.nan_to_num(v)
            res["integrated_abs_z_s"][tn] = float((v[on] * fs).sum())
            i = int(np.argmax(v[on]))
            res["peak_s"][tn] = float(t_rel[on][i] - settle)
            res["peak_z"][tn] = float(v[on][i])
        m_all = sum(res["integrated_abs_z_s"].values())
        own = res["integrated_abs_z_s"].get(sname, 0.0)
        res["outside_share"] = float((m_all - own) / m_all) if m_all > 0 else None
        if base0 is not None:
            rp0 = rollout(ctx, f"pulse_{re.sub(r'[^A-Za-z0-9_]+', '_', sname)}_W0", {**pv, **sil}, n_frames=n_fr)
            dz0 = (np.asarray(np.load(rp0["movie"])["pred"], np.float32) - np.asarray(np.load(base0["movie"])["pred"], np.float32)) / sd
            res["W0"] = {"integrated_abs_z_s": {}}
            for tk, tn in zip(targets, tnames):
                v = np.nan_to_num(np.nanmean(np.abs(dz0[:, lab == tk]), 1))
                res["W0"]["integrated_abs_z_s"][tn] = float((v[on] * fs).sum())
            m0 = sum(res["W0"]["integrated_abs_z_s"].values())
            res["W0"]["outside_share"] = float((m0 - res["W0"]["integrated_abs_z_s"].get(sname, 0.0)) / m0) if m0 > 0 else None
        out["pulse"][sname] = res
        if first_movie is None and ctx["st"]["movies"]:
            first_movie = os.path.join(ctx["gd"], f"{ctx['stem']}_pulse_{re.sub(r'[^A-Za-z0-9_]+', '_', sname)}.mp4")
            try:
                response_movie(ctx, dz, t_rel, settle, dur, sname, first_movie)
                out["movie"] = first_movie
            except Exception as e:
                print(f"[graph] pulse movie failed: {type(e).__name__}: {e}")
    return out


def response_movie(ctx, dz, t_rel, settle, dur, sname, path, fps=12, dpi=90):
    """The pulse's response on the brain: every neuron coloured by its response dz (blue below the resting rollout,
    red above, clipped at +-1 z), from above (head left) and from the side, the pulse's window marked."""
    import shutil
    import subprocess
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from plexus.tasks.trace_recording import _ffmpeg
    pos = ctx["pos"] if ctx["regions"] is None else ctx["regions"]["pos"]
    cm = LinearSegmentedColormap.from_list("resp", ["#4a7bff", "black", "#ff4a4a"])
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(12, 6.2), facecolor="black")
    long = int(np.argmax(np.ptp(pos, 0)))
    other = [a for a in range(3) if a != long]
    scs = []
    order = np.argsort(pos[:, other[1]])
    for rect, (i0, i1), ttl in (([0.02, 0.36, 0.47, 0.6], (long, other[0]), "from above"),
                                ([0.51, 0.36, 0.47, 0.6], (long, other[1]), "from the side")):
        a = fig.add_axes(rect)
        scs.append(a.scatter(pos[order, i0], pos[order, i1], s=0.4, c="0.2", lw=0, rasterized=True))
        a.set_aspect("equal")
        a.axis("off")
        a.set_title(ttl, fontsize=10, loc="left")
    at = fig.add_axes([0.08, 0.07, 0.88, 0.22])
    m = np.nan_to_num(np.nanmean(np.abs(dz), 1))
    at.plot(t_rel, m, color="w", lw=1.0)
    at.axvspan(settle, settle + dur, color="#ff4a4a", alpha=0.25, lw=0)
    at.set_xlabel("time from the rollout's start, s", fontsize=9)
    at.set_ylabel("mean |response|, z", fontsize=9)
    bar = at.axvline(t_rel[0], color="w", lw=1.2)
    txt = fig.text(0.98, 0.98, "", ha="right", va="top", fontsize=10)
    fig.text(0.02, 0.98, f"pulse into {sname}: the response, learned minus resting rollout", fontsize=11, va="top")
    tmp = tempfile.mkdtemp(prefix="pulse_")
    for i in range(len(t_rel)):
        c = cm(np.clip((np.nan_to_num(dz[i][order]) + 1.0) / 2.0, 0, 1))
        for sc in scs:
            sc.set_facecolors(c)
        bar.set_xdata([t_rel[i]] * 2)
        txt.set_text(f"t = {t_rel[i]:.0f} s" + ("  pulse on" if settle <= t_rel[i] < settle + dur else ""))
        fig.savefig(os.path.join(tmp, f"{i:05d}.png"), dpi=dpi, facecolor="black")
    plt.close(fig)
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", path], check=True)
    shutil.copy(os.path.join(tmp, f"{len(t_rel) // 2:05d}.png"), path.replace(".mp4", ".png"))
    shutil.rmtree(tmp)


def welch_vs(X, b, nper=256, chunk=8000):
    """Per column of X [T, N]: its power spectrum and its cross-spectrum with b [T], Welch (Hann, half overlap, each
    window's mean removed) -> f [F] (cycles per frame), Pxx [F, N], Pbb [F], Pxb [F, N] complex."""
    T = len(b)
    nper = int(min(nper, max(8, (T // 4) // 2 * 2)))              # a short recording: shorter windows, still >= 4 of them
    st = np.arange(0, T - nper + 1, nper // 2)
    w = np.hanning(nper).astype(np.float32)

    def seg(A):
        S = np.stack([A[s:s + nper] for s in st]).astype(np.float32)
        S = S - S.mean(1, keepdims=True)
        return S * (w[:, None] if S.ndim == 3 else w)
    Bf = np.fft.rfft(seg(b), axis=1)
    Pbb = (np.abs(Bf) ** 2).mean(0)
    F = Bf.shape[1]
    Pxx = np.empty((F, X.shape[1]), np.float32)
    Pxb = np.empty((F, X.shape[1]), np.complex64)
    for a in range(0, X.shape[1], chunk):
        Xf = np.fft.rfft(seg(X[:, a:a + chunk]), axis=1)
        Pxx[:, a:a + chunk] = (np.abs(Xf) ** 2).mean(0)
        Pxb[:, a:a + chunk] = (Xf * np.conj(Bf)[:, :, None]).mean(0)
    return np.fft.rfftfreq(nper, 1.0), Pxx, Pbb, Pxb


def lead_map(X, frame_s, lo_s=20.0, hi_s=250.0, r_min=0.3):
    """Per neuron its LEAD over the brain mean b, s (> 0: moves first), from the phase of the cross-spectrum over
    periods lo_s-hi_s, coherence-weighted (tools/exp17_brain_mean_lag.phase_lag); NaN for the neurons whose r with b
    is below r_min. -> (lead [N], r [N])."""
    X = np.asarray(X, np.float32)
    b = X.mean(1)
    bc = (b - b.mean()) / max(b.std(), 1e-12)
    r = ((X - X.mean(0)) / np.maximum(X.std(0), 1e-12) * bc[:, None]).mean(0)
    f, Pxx, Pbb, Pxb = welch_vs(X, b)
    fhz = f / frame_s
    sel = (fhz >= 1.0 / hi_s) & (fhz <= 1.0 / lo_s)
    om = 2 * np.pi * fhz[sel][:, None]
    phi = np.angle(Pxb[sel])
    wgt = np.abs(Pxb[sel]) ** 2 / (Pxx[sel] * Pbb[sel][:, None] + 1e-30)
    tau = -(wgt * phi * om).sum(0) / np.maximum((wgt * om ** 2).sum(0), 1e-30)
    lead = -tau
    lead[r < r_min] = np.nan
    return lead, r


def test_leadlag(ctx, doc):
    """WHO LEADS THE BRAIN MEAN, in the recording and in the every-frame rollout: the lead per neuron from the
    cross-spectrum phase (periods 20-250 s), the shares leading / trailing by more than 1 s, the medians per region,
    and the rank correlation of the two maps over the neurons with a lead in both."""
    from scipy.stats import spearmanr
    nomr = nominal(ctx, every_frame=True)
    F = np.load(nomr["traces"], mmap_mode="r")
    first = ctx["o0"] + 1
    Xr = ctx["X"][first:]
    Pm = np.array(F[first:], np.float32)
    bad = ~np.isfinite(Pm)
    if bad.any():
        Pm[bad] = np.broadcast_to(Xr.mean(0), Pm.shape)[bad]
    out = {"band_s": [20.0, 250.0], "r_min": 0.3, "sig_s": SIG_S, "frames": [int(first), int(ctx["T"] - 1)]}
    leads = {}
    for tag, A in (("recording", Xr), ("model", Pm)):
        lead, r = lead_map(A, ctx["frame_s"])
        ok = np.isfinite(lead)
        leads[tag] = lead
        res = {"with_lead": int(ok.sum()), "median_s": float(np.median(lead[ok])) if ok.any() else None,
               "lead_gt_1s_share": float((lead[ok] > SIG_S).mean()) if ok.any() else None,
               "trail_gt_1s_share": float((lead[ok] < -SIG_S).mean()) if ok.any() else None}
        if ctx["mask_in"] is not None:
            res["median_input_s"] = float(np.nanmedian(lead[ctx["mask_in"]])) if np.isfinite(lead[ctx["mask_in"]]).any() else None
            res["median_other_s"] = float(np.nanmedian(lead[~ctx["mask_in"]])) if np.isfinite(lead[~ctx["mask_in"]]).any() else None
        if ctx["regions"] is not None:
            lab, names = ctx["regions"]["label"], ctx["regions"]["names"]
            res["by_region"] = {names[k]: float(np.nanmedian(lead[lab == k])) for k in range(len(names))
                                if (lab == k).sum() >= MIN_REGION and np.isfinite(lead[lab == k]).any()}
        out[tag] = res
    both = np.isfinite(leads["recording"]) & np.isfinite(leads["model"])
    out["n_both"] = int(both.sum())
    if both.sum() > 10:
        out["spearman_model_vs_recording"] = float(spearmanr(leads["recording"][both], leads["model"][both])[0])
        if ctx["regions"] is not None and "by_region" in out["recording"] and "by_region" in out["model"]:
            common = [k for k in out["recording"]["by_region"] if k in out["model"]["by_region"]]
            if len(common) > 4:
                out["spearman_regions"] = float(spearmanr([out["recording"]["by_region"][k] for k in common],
                                                          [out["model"]["by_region"][k] for k in common])[0])
    np.savez_compressed(os.path.join(ctx["gd"], f"{ctx['stem']}_leadlag.npz"), lead_recording=leads["recording"],
                        lead_model=leads["model"])
    out["file"] = os.path.join(ctx["gd"], f"{ctx['stem']}_leadlag.npz")
    return out


def test_memorise(ctx, doc):
    """THE FUNCTIONS OF ABSOLUTE TIME (Omega(x, t), alpha(t)): their statistics over the rollout, and the rollout with
    the frame clock shifted by a quarter of the recording either way or frozen at its middle."""
    op = ctx["op"]
    roles = op.graph_roles()
    if not any(v == "time" for v in roles.values()):
        raise NotImplementedError("memorise: the law has no function of absolute time")
    out = {"time_learnables": [k for k, v in roles.items() if v == "time"]}
    fr = np.unique(np.linspace(0, ctx["T"] - 1, 400).round().astype(int))
    O = op.omega_at(fr)
    if O is not None:
        O = O.double().cpu().numpy()
        bm = O.mean(1)
        out["omega"] = {"mean": float(O.mean()), "p5": float(np.percentile(O, 5)), "p95": float(np.percentile(O, 95)),
                        "brain_mean_min": float(bm.min()), "brain_mean_max": float(bm.max()), "share_below_1": float((O < 1).mean()),
                        "sd_over_time_median": float(np.median(O.std(0)))}
    if hasattr(op, "angle"):
        f0 = getattr(op, "frame", 0)
        al = []
        with torch.no_grad():
            for f in fr:
                op.frame = int(f)
                al.append(float(op.angle()))
        op.frame = f0
        al = np.degrees(np.array(al))
        out["alpha_deg"] = {"min": float(al.min()), "max": float(al.max()), "sd": float(al.std())}
    nom = movie_scores(ctx, nominal(ctx)["movie"])["steady"]
    q = ctx["T"] // 4
    out["rollouts"] = {}
    for name, v in (("clock_plus", {"clock": {"shift_frames": q}}), ("clock_minus", {"clock": {"shift_frames": -q}}),
                    ("clock_frozen", {"clock": {"freeze_at": ctx["T"] // 2}})):
        r = rollout(ctx, name, v)
        sc = movie_scores(ctx, r["movie"])
        out["rollouts"][name] = {**sc, "variant": v, "file": r["movie"],
                                 "drop_steady": {m: nom[m] - sc["steady"][m] for m in ("brain_mean_r", "per_neuron_r")}}
    return out


TEST_FNS = {"constants": test_constants, "silence": test_silence, "blocks": test_blocks, "controls": test_controls,
            "clamp": test_clamp, "prune_w": test_prune_w, "prune_b": test_prune_b, "spectrum": test_spectrum,
            "impulse": test_impulse, "leadlag": test_leadlag, "memorise": test_memorise}


# ============================================================================== the card
def figure_card(ctx, doc, path):
    """THE GRAPH CARD, one figure: (a) brain-mean r and per-neuron r on the steady part, the graph against its
    silenced rollouts, the stimulus off, the trained controls and the clamp; (b) the per-block gap full - silenced;
    (c) the pruning ladders; (d) the Jacobian's leading eigenvalues against the leak; (e) the pulse responses, region
    by region; (f) the lead over the brain mean, model against recording."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    fig, axs = plt.subplots(2, 3, figsize=(17, 9), facecolor="black")
    fig.suptitle(f"{ctx['stem']}: the graph card", fontsize=12, x=0.01, ha="left")
    # (a)
    a = axs[0, 0]
    rows = []
    sil = doc.get("silence", {})
    for k, v in (sil.get("rollouts") or {}).items():
        rows.append((k, v["steady"]["brain_mean_r"], v["steady"]["per_neuron_r"]))
    for k, v in (doc.get("controls") or {}).items():
        if isinstance(v, dict) and "scores" in v:
            rows.append((f"ctrl {k}", v["scores"]["steady"]["brain_mean_r"], v["scores"]["steady"]["per_neuron_r"]))
    for k in ("clamp_in", "clamp_in_W0"):
        v = (doc.get("clamp", {}).get("rollouts") or {}).get(k)
        if v:
            rows.append((k, v["steady"]["brain_mean_r"], v["steady"]["per_neuron_r"]))
    if rows:
        y = np.arange(len(rows))
        a.barh(y - 0.2, [r[1] for r in rows], 0.4, color="#2ca02c", label="brain-mean r")
        a.barh(y + 0.2, [r[2] for r in rows], 0.4, color="#ff9f1c", label="per-neuron r")
        a.set_yticks(y)
        a.set_yticklabels([r[0] for r in rows], fontsize=8)
        a.invert_yaxis()
        a.legend(fontsize=8, frameon=False, loc="lower right")
    a.set_title("a  steady part: the graph against its controls", fontsize=10, loc="left")
    # (b)
    a = axs[0, 1]
    pb = [r for r in (doc.get("blocks", {}).get("per_block") or []) if "gap" in r]
    if pb:
        x = np.arange(len(pb))
        a.bar(x - 0.2, [r["gap"]["brain_mean_r"] for r in pb], 0.4, color="#2ca02c")
        a.bar(x + 0.2, [r["gap"]["per_neuron_r"] for r in pb], 0.4, color="#ff9f1c")
        a.set_xticks(x)
        a.set_xticklabels([r["block"] for r in pb], rotation=45, ha="right", fontsize=8)
        a.axhline(0, color="0.5", lw=0.6)
    a.set_title("b  full minus silenced, within each block", fontsize=10, loc="left")
    # (c)
    a = axs[0, 2]
    for s, v in (doc.get("prune_w", {}).get("per_set") or {}).items():
        lad = v.get("ladder") or []
        if lad:
            a.plot([r["threshold"] for r in lad], [r["scores"]["steady"]["per_neuron_r"] for r in lad], "o-", ms=3, label=f"{s}: per-neuron r")
            if v.get("threshold"):
                a.axvline(v["threshold"], color="0.5", lw=0.6, ls="--")
    for r_ in (doc.get("prune_b", {}).get("ladder") or []):
        a.plot(r_["quantile"], r_["scores"]["steady"]["per_neuron_r"], "s", color="#ff4a4a", ms=4)
    a.set_xscale("log")
    a.set_xlabel("threshold (edges: strength; inputs: quantile)", fontsize=8)
    a.legend(fontsize=7, frameon=False)
    a.set_title("c  the pruning ladders, steady per-neuron r", fontsize=10, loc="left")
    # (d)
    a = axs[1, 0]
    spc = doc.get("spectrum", {})
    spc = spc.get("session", spc)
    if spc.get("eig_re_per_s"):
        a.scatter(spc["eig_re_per_s"], spc["eig_im_per_s"], s=10, color="#ff9f1c", label="J, largest real part")
        for m in spc.get("oscillatory") or []:
            a.scatter([m["re_per_s"]], [m["im_per_s"]], s=12, color="#4a7bff")
        a.axvline(spc.get("leak_abscissa_per_s", 0.0), color="0.5", lw=0.6, ls="--", label="the slowest leak alone")
        a.axvline(0, color="0.3", lw=0.6)
        a.set_xlabel("Re, per s", fontsize=8)
        a.set_ylabel("Im, per s", fontsize=8)
        a.legend(fontsize=7, frameon=False)
    a.set_title("d  the Jacobian's leading eigenvalues", fontsize=10, loc="left")
    # (e)
    a = axs[1, 1]
    imp = doc.get("impulse", {})
    if imp.get("pulse"):
        srcs, tg = list(imp["pulse"]), imp["targets"]
        Mx = np.array([[imp["pulse"][s_]["integrated_abs_z_s"].get(t_, 0.0) for t_ in tg] for s_ in srcs])
        im = a.imshow(np.log10(Mx + 1e-6), aspect="auto", cmap="magma")
        a.set_xticks(range(len(tg)))
        a.set_xticklabels(tg, rotation=60, ha="right", fontsize=6)
        a.set_yticks(range(len(srcs)))
        a.set_yticklabels(srcs, fontsize=7)
        cb = fig.colorbar(im, ax=a, fraction=0.03, pad=0.01)
        cb.set_label("log10 integrated |response|, z s", fontsize=8, labelpad=2)
    a.set_title("e  a pulse into a region (rows): the response of each (columns)", fontsize=10, loc="left")
    # (f)
    a = axs[1, 2]
    ll = doc.get("leadlag", {})
    if ll.get("file") and os.path.exists(ll["file"]):
        z = np.load(ll["file"])
        lr, lm = z["lead_recording"], z["lead_model"]
        ok = np.isfinite(lr) & np.isfinite(lm)
        if ok.any():
            a.hexbin(np.clip(lr[ok], -5, 5), np.clip(lm[ok], -5, 5), gridsize=40, cmap="magma", mincnt=1)
            a.plot([-5, 5], [-5, 5], color="0.5", lw=0.6)
            a.set_xlabel("lead over the brain mean, recording, s", fontsize=8)
            a.set_ylabel("lead, model, s", fontsize=8)
            if ll.get("spearman_model_vs_recording") is not None:
                a.text(0.02, 0.95, f"Spearman {ll['spearman_model_vs_recording']:+.2f}", transform=a.transAxes, fontsize=8, va="top")
    a.set_title("f  who leads: model against recording", fontsize=10, loc="left")
    for ax in axs.ravel():
        ax.tick_params(labelsize=8)
    fig.savefig(path, dpi=120, facecolor="black", bbox_inches="tight")
    plt.close(fig)


# ============================================================================== the phase
def run(spec, device="cpu", root=None, tests=None) -> dict:
    """The graph phase on a landed trace run: every test of `task.graph.tests` (or `tests`) in TESTS' order, the json
    written after each one (results/<stem>_graph.json), the card at the end. A test the law cannot take is recorded
    under `refused`, a failing one under `errors` (printed), neither stops the phase."""
    t0 = time.time()
    ctx = context(spec, device, root)
    if tests:
        ctx["st"]["tests"] = list(tests)
    want = [t for t in TESTS if t in ctx["st"]["tests"]]
    op = ctx["op"]
    path = os.path.join(ctx["out"], "results", f"{ctx['stem']}_graph.json")
    doc = {"name": spec["name"], "checkpoint": ctx["R"]["ckpt"], "it": ctx["R"]["it"], "model": spec["model"],
           "law": {"class": type(op).__name__, "roles": op.graph_roles(), "silencers": op.silencers(),
                   "edge_sets": {s: int(v[0].numel()) for s, v in op.edge_table().items()}},
           "recording": {"name": spec["task"]["reference"].get("trace_recording"), "frames": ctx["T"], "neurons": ctx["N"],
                         "frame_s": ctx["frame_s"], "offsets": ctx["offsets"], "names": ctx["names"],
                         "first_free_frame": ctx["o0"] + 1, "n_input": int(ctx["mask_in"].sum()) if ctx["mask_in"] is not None else None,
                         "regions": ctx["regions"]["names"] if ctx["regions"] else None},
           "settings": {k: v for k, v in ctx["st"].items()}, "windows": {
               "whole": "every free frame (brain mean) / every movie frame (per neuron)",
               "steady": f"after the first of {ctx['st']['bootstrap']['blocks']} equal blocks of the free frames"},
           "tests_run": want, "refused": {}, "errors": {}, "seconds": {}}
    if os.path.exists(path) and not ctx["force"]:          # earlier tests of this run kept, re-run ones replaced
        try:
            old = json.load(open(path))
            for t in TESTS:
                if t in old and t not in want:
                    doc[t] = old[t]
        except Exception:
            pass
    for t in want:
        t1 = time.time()
        try:
            doc[t] = TEST_FNS[t](ctx, doc)
        except NotImplementedError as e:
            doc["refused"][t] = str(e)
            print(f"[graph] {t}: skipped -- {e}")
        except Exception as e:
            doc["errors"][t] = f"{type(e).__name__}: {e}"
            print(f"[graph] {t} FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        doc["seconds"][t] = time.time() - t1
        json.dump(doc, open(path, "w"), indent=1, default=_json_default)
        print(f"[graph] {t}: {doc['seconds'][t]:.0f} s", flush=True)
    card = os.path.join(ctx["gd"], f"{ctx['stem']}_card.png")
    try:
        figure_card(ctx, doc, card)
        doc["card"] = card
    except Exception as e:
        doc["errors"]["card"] = f"{type(e).__name__}: {e}"
        print(f"[graph] card FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()}")
    doc["seconds"]["total"] = time.time() - t0
    json.dump(doc, open(path, "w"), indent=1, default=_json_default)
    print(f"[graph] {spec['name']}: {len(want)} tests in {doc['seconds']['total'] / 60:.1f} min -> {path}")
    return doc


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not serialisable: {type(o).__name__}")


def main(argv=None):
    import argparse
    from plexus import trainer as T
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="a training spec name or path (a landed trace run)")
    ap.add_argument("--tests", default=None, help=f"comma-separated, among {','.join(TESTS)}")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--root", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--controls", default=None, help="no_w=<run>,mean_field=<run>,seed=<run>,random=<run>")
    ap.add_argument("--regions", default=None, help="an atlas npz (regions, names, inside, atlas_um) for the region tests")
    ap.add_argument("--force", action="store_true", help="rerun every rollout (the cache under results/graph/ ignored)")
    a = ap.parse_args(argv)
    spec = T.load(a.spec)
    if a.checkpoint:
        spec["_checkpoint"] = a.checkpoint
    if a.controls:
        spec["_graph_controls"] = dict(kv.split("=", 1) for kv in a.controls.split(",") if kv)
    if a.regions:
        spec["_graph_regions"] = a.regions
    if a.force:
        spec["_force"] = True
    run(spec, device=a.device, root=a.root, tests=a.tests.split(",") if a.tests else None)


if __name__ == "__main__":
    main()
