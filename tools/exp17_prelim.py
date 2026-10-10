"""exp17 batches 25 and 26: PRELIMINARY RESULTS (Cedric, 2026-10-10: "launch an agent to add a few results slides to 25
and 26, with preliminary results"). Local, and read-only on the run directories.

Per run, from log/training/zapbench/<run>/results/history.jsonl:
  the live evaluation, logged every `save_every` updates of the run's spec (trainer._train_trace): per step ahead,
    skill = 1 - MSE / the best mean baseline's MSE, over the trainer's fixed evaluation origins; `eval_skill_short` the
    mean over steps 1-3, `eval_skill_long` over steps 16-32 (trainer.py, sk[0:3] and sk[15:32]);
  the training loss of every update (`loss`), averaged here over the updates since the previous evaluation;
  the horizon of every update, for the stage edges;
and the same from the comparator's history.jsonl (22.3 zap_n22_markall for 22.3's law, 24.10 zap_n24_ph_edge_blk for
24.10's), read at the same updates.

Once a run has been tested (results/report.json, results/<run>_test.json at the run's last update and
results/<run>_movie.npz): the 2-h free rollout's brain-mean r and per-neuron r as the deck's run slides compute them,
exp17_slides.bm_metrics (Pearson r of the learned against the recorded brain-mean dF/F over the free frames) and
exp17_slides.local_r (per neuron, r of the learned against the recorded trace, each first regressed on its own brain
mean; the mean over the neurons), and the comparator's the same way. local_r caches its result as <npz>.local_r.json
next to the npz it is given: it is given a symlink in data/prelim_cache/, so the cache is written there and the run
directory is never written. The video runs (26.x) read zapbench_destripe_video, whose dF/F is zapbench_destripe's
(tools/exp17_video_input.py export copies it): local_r reads zapbench_destripe for every run, as the deck does.

    PYTHONPATH=src:tools python tools/exp17_prelim.py            # both batches
    PYTHONPATH=src:tools python tools/exp17_prelim.py 26         # one
-> data/prelim_b25.json, data/prelim_b26.json; presentation/figs/prelim_b25.png, prelim_b26.png
"""
from __future__ import annotations

import datetime
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
FIGS = os.path.join(EXP, "presentation", "figs")
CACHE = os.path.join(EXP, "data", "prelim_cache")
GD = os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
RUNS = os.path.join(GD, "log", "training", "zapbench")

REFS = {"22.3": "zap_n22_markall", "24.10": "zap_n24_ph_edge_blk"}
# (arm, run, law reference); the arms of one law take the colour slots in order, so the same mask has the same colour on
# both laws
BATCHES = {
    25: [("25.1", "zap_n25_bio", "22.3"), ("25.2", "zap_n25_bio_eff", "22.3"), ("25.3", "zap_n25_bio_eff_perm", "22.3"),
         ("25.4", "zap_n25_ph_bio", "24.10"), ("25.5", "zap_n25_ph_bio_eff", "24.10"),
         ("25.6", "zap_n25_ph_bio_eff_perm", "24.10")],
    26: [("26.1", "zap_n26_vid", "22.3"), ("26.2", "zap_n26_vid_bio", "22.3"),
         ("26.3", "zap_n26_ph_vid", "24.10"), ("26.4", "zap_n26_ph_vid_bio", "24.10")],
}
# FAIR COMPARISONS ONLY (Cedric, 2026-10-10: "we need to report always fair comparison ... put the results in blank if
# not available"). The tables compare each arm with the run it differs from in one thing: its law's reference, except a
# permuted mask, compared with the same mask unpermuted (25.3 with 25.2, 25.6 with 25.5: the permutation the one
# difference). Each arm's no-W twin (its inputs, W not learned, no Omega; with W = 0 the two laws coincide, so one
# twin serves an arm of each law) and the seed spread of the references and of 26.1 (seed 0 against seed 1 of one
# spec) are read once the run has landed, blank until then.
CMP = {"25.3": "25.2", "25.6": "25.5"}
NOW_TWIN = {"25.1": ("25.7", "zap_n25_bio_now"), "25.4": ("25.7", "zap_n25_bio_now"),
            "25.2": ("25.8", "zap_n25_bio_eff_now"), "25.5": ("25.8", "zap_n25_bio_eff_now"),
            "25.3": ("25.9", "zap_n25_bio_eff_perm_now"), "25.6": ("25.9", "zap_n25_bio_eff_perm_now"),
            "26.1": ("26.5", "zap_n26_vid_now"), "26.3": ("26.5", "zap_n26_vid_now"),
            "26.2": ("26.6", "zap_n26_vid_bio_now"), "26.4": ("26.6", "zap_n26_vid_bio_now")}
SEED_PAIRS = {25: [("22.3", "zap_n22_markall", "22.23", "zap_n22_markall_s1"),
                   ("24.10", "zap_n24_ph_edge_blk", "24.12", "zap_n24_ph_edge_blk_s1")],
              26: [("26.1", "zap_n26_vid", "26.7", "zap_n26_vid_s1"),
                   ("22.3", "zap_n22_markall", "22.23", "zap_n22_markall_s1"),
                   ("24.10", "zap_n24_ph_edge_blk", "24.12", "zap_n24_ph_edge_blk_s1")]}
# categorical slots 1-3 of the dataviz palette's dark steps (blue, orange, aqua), validated all-pairs on black;
# the comparator dashed in a neutral light grey
COLOURS = ["#3987e5", "#d95926", "#199e70"]
REF_COLOUR = "#d0d0d0"
SHORT_STEPS, LONG_STEPS = (1, 3), (16, 32)        # trainer.py: sk[0:3], sk[15:32] (steps ahead, 1-based)
REC = "zapbench_destripe"


def read_history(run: str) -> dict:
    """The run's history.jsonl -> its eval rows, its per-window training loss, its horizon stages, its last update and
    the file's mtime. A last line still being written is skipped."""
    p = os.path.join(RUNS, run, "results", "history.jsonl")
    mtime = os.path.getmtime(p)
    ev, loss_it, loss_v, hz = [], [], [], []
    with open(p) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            it = int(d["it"])
            loss_it.append(it)
            loss_v.append(float(d["loss"]))
            hz.append(int(d.get("horizon", 0)))
            if "eval_skill_short" in d:
                ev.append((it, float(d["eval_skill_short"]), float(d["eval_skill_long"])))
    loss_it, loss_v, hz = np.asarray(loss_it), np.asarray(loss_v), np.asarray(hz)
    win = []                                        # the mean loss over (previous eval, this eval]
    prev = 0
    for it, _, _ in ev:
        sel = (loss_it > prev) & (loss_it <= it) & np.isfinite(loss_v)
        win.append(float(loss_v[sel].mean()) if sel.any() else float("nan"))
        prev = it
    edges = [(int(loss_it[0]), int(hz[0]))] + [(int(loss_it[i]), int(hz[i])) for i in range(1, len(hz)) if hz[i] != hz[i - 1]]
    return {"eval_it": [e[0] for e in ev], "skill_short": [e[1] for e in ev], "skill_long": [e[2] for e in ev],
            "loss_window": win, "stages": edges, "it_reached": int(loss_it[-1]), "mtime": mtime}


def total_updates(run: str) -> tuple[int, int]:
    """(the updates the run's spec asks for, its save_every), from the spec copied into the run directory."""
    import yaml
    t = yaml.safe_load(open(os.path.join(RUNS, run, "config.yaml")))["training"]
    return sum(int(s["iters"]) for s in t["stages"]), int(t.get("save_every", 0))


def test_metrics(run: str, total: int) -> dict | None:
    """The tested run's 2-h free rollout: brain-mean r and per-neuron r as the deck computes them; None until the test
    has run (report.json, a test json at the last update, the movie npz)."""
    import exp17_slides as S
    res = os.path.join(RUNS, run, "results")
    tj, mv = os.path.join(res, f"{run}_test.json"), os.path.join(res, f"{run}_movie.npz")
    if not (os.path.exists(os.path.join(res, "report.json")) and os.path.exists(tj) and os.path.exists(mv)):
        return None
    try:
        T_ = json.load(open(tj))
        it_ = int(T_.get("it", -1))
    except (json.JSONDecodeError, OSError):
        return None                                  # being written
    if it_ != total:
        return None
    os.makedirs(CACHE, exist_ok=True)
    ln = os.path.join(CACHE, f"{run}_movie.npz")
    if os.path.islink(ln) and os.readlink(ln) != mv:
        os.remove(ln)
    if not os.path.islink(ln):
        os.symlink(mv, ln)
    # the test writes its json after the movie npz, but a re-test of the same run (another rollout variant) rewrites
    # them while this reads: a broken npz (BadZipFile, a short read, a missing key) is "not tested yet", never a crash
    try:
        bm = S.bm_metrics(ln)
        lr = S.local_r(ln, REC, flat0=True)          # one neuron set for every run, a flat learned trace scored 0
        z_ = np.load(ln)
        n_bm, n_mv = int(np.asarray(z_["mean_obs_all"]).size), int(np.asarray(z_["frames"]).size)
    except Exception as e_:                          # noqa: BLE001
        print(f"  [prelim] {run}: movie npz not readable ({type(e_).__name__}), counted as not tested")
        return None
    if bm is None or lr is None:
        return None
    return {"brain_mean_r": float(bm["r"]), "brain_mean_rmse": float(bm["rmse"]), "per_neuron_r": float(lr["mean"]),
            "per_neuron_r_sd": float(lr["sd"]), "per_neuron_n": int(lr["n"]), "per_neuron_flat0": int(lr["n_flat_scored_0"]),
            "it": it_,
            "brain_mean_frames": n_bm, "movie_frames": n_mv, "free_end_s": float(T_["free_t_s"][-1]),
            "movie_npz": mv, "movie_npz_mtime": os.path.getmtime(mv)}


def landed_test(run: str) -> dict | None:
    """A control's or seed twin's test as test_metrics reads it; None until the run has landed (or its directory exists)."""
    if not os.path.exists(os.path.join(RUNS, run, "config.yaml")):
        return None
    return test_metrics(run, total_updates(run)[0])


def inputs_of(run: str) -> dict:
    """What the run's state_diffuse reads, from its model spec: the neurons reading a block marker, the neurons reading a
    stimulus feature (or the video), and whether it has a learned offset per block (rest_per_block)."""
    import yaml
    from plexus.paths import graphs_data_path
    from exp17_vrest_blocks import MARKERS
    op = next(o for o in yaml.safe_load(open(os.path.join(ROOT, "config", "zapbench", f"{run}.yaml")))["operators"]
              if o.get("op") == "state_diffuse")
    M = np.asarray(np.load(graphs_data_path(*str(op["input_mask"]).split("/")))[str(op.get("input_mask_array", "mask"))]) != 0
    F = int(op.get("forcing_dim", 22))
    M = M.reshape(M.shape[0], -1) if M.ndim == 2 else np.repeat(M.reshape(-1, 1), F, 1)
    feat = [k for k in range(M.shape[1]) if k not in MARKERS]
    return {"markers": int(M[:, list(MARKERS)].any(1).sum()), "features": int(M[:, feat].any(1).sum()),
            "rest_per_block": bool(op.get("rest_per_block", False))}


def at_update(H: dict, it: int) -> tuple[float, float] | None:
    """The comparator's live skill (short, long) at update `it`, None if it logged none there."""
    if it in H["eval_it"]:
        i = H["eval_it"].index(it)
        return H["skill_short"][i], H["skill_long"][i]
    return None


def stamp(t: float) -> str:
    return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")


def collect(batch: int) -> tuple[dict, dict]:
    arms = BATCHES[batch]
    refs = sorted({r for _, _, r in arms}, key=lambda k: list(REFS).index(k))
    HR = {k: read_history(REFS[k]) for k in refs}
    out = {"batch": batch, "read": stamp(datetime.datetime.now().timestamp()),
           "short_steps": list(SHORT_STEPS), "long_steps": list(LONG_STEPS), "recording": REC,
           "comparators": {}, "runs": []}
    for k in refs:
        tot_, se_ = total_updates(REFS[k])
        out["comparators"][k] = {"run": REFS[k], "it_reached": HR[k]["it_reached"], "updates_total": tot_,
                                 "history_mtime": stamp(HR[k]["mtime"]), "save_every": se_, "stages": HR[k]["stages"],
                                 "test": test_metrics(REFS[k], tot_)}
    HA = {}
    slot = {k: 0 for k in refs}
    run_of = {a: r for a, r, _ in arms} | {k: REFS[k] for k in refs}
    for arm, run, ref in arms:
        H = read_history(run)
        HA[arm] = H
        tot_, se_ = total_updates(run)
        e_it = H["eval_it"][-1] if H["eval_it"] else None
        cmp_ = CMP.get(arm, ref)                     # the run this arm differs from in one thing
        row = {"arm": arm, "run": run, "ref": ref, "ref_run": REFS[ref], "cmp": cmp_, "cmp_run": run_of[cmp_],
               "colour": COLOURS[slot[ref]],
               "updates_total": tot_, "save_every": se_, "it_reached": H["it_reached"],
               "history_mtime": stamp(H["mtime"]), "eval_it": e_it, "test": test_metrics(run, tot_),
               "inputs": inputs_of(run), "cmp_inputs": inputs_of(run_of[cmp_])}
        slot[ref] += 1
        if arm in NOW_TWIN:                          # its no-W twin: blank until it lands
            nw_, nr_ = NOW_TWIN[arm]
            row["now"] = {"arm": nw_, "run": nr_, "test": landed_test(nr_)}
        if e_it is not None:
            row["skill_short"], row["skill_long"] = H["skill_short"][-1], H["skill_long"][-1]
            c_ = at_update(HA[cmp_] if cmp_ in HA else HR[cmp_], e_it)
            if c_:
                row["ref_skill_short"], row["ref_skill_long"] = c_
                row["d_skill_short"] = row["skill_short"] - c_[0]
                row["d_skill_long"] = row["skill_long"] - c_[1]
        rt_ = (out["comparators"][cmp_]["test"] if cmp_ in out["comparators"]
               else next(r_["test"] for r_ in out["runs"] if r_["arm"] == cmp_))
        row["cmp_test"] = rt_
        if row["test"] and rt_:
            row["d_brain_mean_r"] = row["test"]["brain_mean_r"] - rt_["brain_mean_r"]
            row["d_per_neuron_r"] = row["test"]["per_neuron_r"] - rt_["per_neuron_r"]
        out["runs"].append(row)
    out["comparator_inputs"] = {k: inputs_of(REFS[k]) for k in refs}
    out["seed_pairs"] = []                           # seed 0 against seed 1 of one spec: blank until seed 1 lands
    for a_, ra_, b_, rb_ in SEED_PAIRS.get(batch, []):
        ta_, tb_ = landed_test(ra_), landed_test(rb_)
        sp_ = {"a": a_, "a_run": ra_, "b": b_, "b_run": rb_, "test_a": ta_, "test_b": tb_}
        if ta_ and tb_:
            sp_["d_brain_mean_r"] = abs(ta_["brain_mean_r"] - tb_["brain_mean_r"])
            sp_["d_per_neuron_r"] = abs(ta_["per_neuron_r"] - tb_["per_neuron_r"])
        out["seed_pairs"].append(sp_)
    return out, {"arms": HA, "refs": HR}


def figure(J: dict, H: dict, path: str) -> None:
    """3 rows (skill short, skill long, training loss) x one column per law; each run one colour, its comparator dashed
    over the same updates; dotted verticals at the horizon stage edges."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    refs = list(J["comparators"])
    fig, axs = plt.subplots(3, len(refs), figsize=(12, 8.5), facecolor="black", squeeze=False, sharex=True)
    s0, s1 = J["short_steps"], J["long_steps"]
    rows = [("skill_short", f"live skill, steps {s0[0]}-{s0[1]} ahead"),
            ("skill_long", f"live skill, steps {s1[0]}-{s1[1]} ahead"),
            ("loss_window", "training loss")]
    for c, ref in enumerate(refs):
        arms = [r for r in J["runs"] if r["ref"] == ref]
        top = max(H["arms"][r["arm"]]["eval_it"][-1] for r in arms if H["arms"][r["arm"]]["eval_it"])
        HR = H["refs"][ref]
        ri = [i for i, it in enumerate(HR["eval_it"]) if it <= top]
        for k, (key, ttl) in enumerate(rows):
            ax = axs[k, c]
            ax.set_facecolor("black")
            for e_it, _ in HR["stages"][1:]:
                if e_it <= top:
                    ax.axvline(e_it, color="#555555", lw=0.8, ls=":", zorder=0)
            ax.plot([HR["eval_it"][i] for i in ri], [HR[key][i] for i in ri], color=REF_COLOUR, lw=1.6, ls=(0, (5, 3)),
                    label=f"{ref} ({REFS[ref]})")
            for r in arms:
                Ha = H["arms"][r["arm"]]
                ax.plot(Ha["eval_it"], Ha[key], color=r["colour"], lw=2.0, label=f"{r['arm']} ({r['run']})")
            ax.set_title(f"{ttl}, {ref}'s law", loc="left", color="white", fontsize=16)
            ax.tick_params(colors="#bbbbbb", labelsize=12)
            for s in ax.spines.values():
                s.set_color("#666666")
            ax.grid(color="#333333", lw=0.6)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
            if k == 0:
                ax.legend(loc="lower right", fontsize=12, frameon=False, labelcolor="white")
            if k == 2:
                ax.set_xlabel("update", color="white", fontsize=14)
    fig.tight_layout(h_pad=1.2, w_pad=1.6)
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)


def main() -> None:
    which = [int(a) for a in sys.argv[1:]] or sorted(BATCHES)
    os.makedirs(FIGS, exist_ok=True)
    for b in which:
        J, H = collect(b)
        jp = os.path.join(EXP, "data", f"prelim_b{b}.json")
        json.dump(J, open(jp, "w"), indent=1)
        fp = os.path.join(FIGS, f"prelim_b{b}.png")
        figure(J, H, fp)
        print(f"[prelim] batch {b} -> {jp}, {fp}")
        for r in J["runs"]:
            t_ = r["test"]
            print(f"  {r['arm']:5s} {r['run']:24s} it {r['it_reached']:6,d}/{r['updates_total']:,} eval@{r['eval_it']} "
                  f"short {r.get('skill_short', float('nan')):+.4f} (ref {r.get('ref_skill_short', float('nan')):+.4f}) "
                  f"long {r.get('skill_long', float('nan')):+.4f} (ref {r.get('ref_skill_long', float('nan')):+.4f})"
                  + (f"  test bm r {t_['brain_mean_r']:.3f} pn r {t_['per_neuron_r']:.3f}" if t_ else "  not tested"))
        for k, c in J["comparators"].items():
            t_ = c["test"]
            print(f"  ref {k} {c['run']}: " + (f"bm r {t_['brain_mean_r']:.3f} pn r {t_['per_neuron_r']:.3f}" if t_ else "no test"))


if __name__ == "__main__":
    main()
