#!/usr/bin/env python
"""The experiment log for the STAGED runs -- the ones the loop did not propose.

WHY THESE NEED A LOG AND THE CAMPAIGN'S RUNS DO NOT. Every run the loop makes is written to
`campaign/records.jsonl` with its parent, its edit, its act, the claim it bears on and the
prediction it committed to. A staged run has none of that: it is submitted by hand, it lands in
`log/okuda/` beside the campaign's runs, and the only record of WHY it exists is whatever was said
in the conversation that produced it. Ten of them accumulated in one afternoon, and the difference
between `stage_buds_web_da016` and `stage_buds_web_seed2` is not recoverable from their names.

WHAT IT READS, so it cannot go stale: the spec on disk for the structure -- which operators carry a
frame window, and what changes across it -- and `metrics.json` for the outcome. The only thing kept
by hand is the QUESTION, below, because a hypothesis is not derivable from a spec.

    python staged_log.py            rewrite campaign/staged/EXPERIMENTS.md
    python staged_log.py --print    and print it
"""
from __future__ import annotations

import argparse
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.abspath(os.path.join(HERE, "..", "log", "okuda"))
OUT = os.path.join(HERE, "campaign", "staged")

# THE QUESTION EACH RUN WAS SUBMITTED TO ANSWER. Hand-written, because nothing in a spec says what
# somebody wanted to know. A run absent from here still appears in the table with its structure --
# an unexplained experiment is better logged than dropped.
QUESTION = {
    "stage_star_then_decoupled":
        "Will the substrate accept a spec that changes partway? Two runs joined at frame 900: the "
        "star's recipe, then r020_10's. Built by hand from the two specs.",
    "stage_beta_off_at_900":
        "Is the Turing pattern MAINTAINED by the shape coupling, or only formed by it? Only "
        "`cell_chem_from_shape.beta` changes at 900 (0.5 -> 0.0); the reaction is untouched.",
    "stage_buds_wider_pattern":
        "Does a wider pattern on a built star give BUDS on it? Stage-2 `d_a` 0.08 -> 0.32, the "
        "wavelength lever, on the reasoning that ~10-cell spots make tubes and 100-200-cell spots "
        "make lobes.",
    "stage_buds_reseed":
        "The same widening plus a SECOND INDUCTION: a fresh scatter seeding at frames 900-906, on "
        "the star's geometry, the way branching morphogenesis makes secondary buds.",
    "stage_buds_web_seed2":
        "Is the web reproducible? `stage_buds_wider_pattern` at a different seed and nothing else.",
    "stage_buds_web_da016":
        "Is the web a wavelength threshold? Half the step -- stage-2 `d_a` 0.16 rather than 0.32.",
    "stage_buds_web_switch1350":
        "Does the web need an immature star? Same widening, switched at 1350 instead of 900.",
    "stage_tips_nobaseline":
        "Is `rho` the dilution? It is the activator-INDEPENDENT baseline, so a tip at activator 1.6 "
        "and a flank at 0.0 inflate alike. Stage 2 sets it to 0 and changes nothing else.",
    "stage_tips_fast":
        "Growth ENFORCED at the tips: no baseline, the Hill switch raised to 0.9 so only the top "
        "~8% of cells by activator grow, and the rate 3.5x. Does the star build structure at its "
        "tips instead of rounding off?",
    "stage_tips_fast_divide":
        "The same, plus division allowed to cycle faster (`min_cycle` 16 -> 6) once cells have "
        "grown. Does dividing at the tips make the structure bigger or just finer?",
}

# The metrics worth a column here. `n_tubes` and `protr` say what shape it is; `act_max` says
# whether the chemistry survived, which is the thing every staged run so far has turned on.
# READ OFF A REAL summary, not from memory: a staged run's `metrics.json` summary carries 23 keys
# and `act_max_final`, `n_cells_final` and `grip_final` are not among them -- the first table this
# script printed had three columns of em-dashes because I named the campaign's metric bank instead
# of what the file holds. `red_frac_final` is the chemistry column that does exist, and on these
# runs it is the one that has decided every result.
COLS = ("n_tubes_final", "protr_final", "red_frac_final", "tip_act_final", "tube_len_final")


def stages(spec):
    """(switch frames, [(op, what changed across the switch)]) read off the windows in the spec."""
    ops = spec.get("operators") or []
    cuts, split = set(), []
    by = {}
    for o in ops:
        by.setdefault(o["op"], []).append(o)
        for k in ("after_frame", "before_frame"):
            v = o.get(k)
            # 1 and 3 are the seeding window every spec carries; they are not a stage boundary.
            if isinstance(v, int) and v > 10:
                cuts.add(v)
    for op, insts in by.items():
        if len(insts) < 2:
            continue
        keys = {k for i in insts for k in i} - {"op", "at", "after_frame", "before_frame",
                                                "cell_set", "id", "impl", "model", "field"}
        diff = {k: [i.get(k) for i in insts] for k in sorted(keys)
                if len({str(i.get(k)) for i in insts}) > 1}
        if diff:
            split.append((op, diff))
    return sorted(cuts), split


def read(name):
    d = os.path.join(LOG, name)
    row = {"run": name, "question": QUESTION.get(name, "")}
    p = os.path.join(d, "spec_run.yaml")
    if os.path.exists(p):
        try:
            spec = yaml.safe_load(open(p))
            row["frames"] = (spec.get("general") or {}).get("n_frames")
            row["switch"], row["split"] = stages(spec)
        except Exception as e:
            row["spec_error"] = f"{type(e).__name__}: {e}"
    m = os.path.join(d, "metrics.json")
    if os.path.exists(m):
        try:
            s = (json.load(open(m)) or {}).get("summary") or {}
            row["metrics"] = {k: s.get(k) for k in COLS}
        except Exception:
            pass
    pr = os.path.join(d, "progress.json")
    if os.path.exists(pr):
        try:
            row["progress"] = json.load(open(pr))
        except Exception:
            pass
    return row


def render(rows):
    L = ["# Staged experiments — runs the loop did not propose\n",
         "**Generated by `staged_log.py` from the specs and metrics on disk. Do not edit above the "
         "questions.** A staged run changes its spec partway through: two instances of one operator "
         "with disjoint frame windows, which the engine gates on "
         "`after_frame <= tick < before_frame`. None of these appears in `campaign/records.jsonl` — "
         "they are hand-submitted, so this file is their only record.\n",
         f"{len(rows)} run(s).\n",
         "| run | frames | switch | what changes at the switch | n_tubes | protr | red_frac | tip_act |",
         "|---|---:|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        m = r.get("metrics") or {}
        ch = "; ".join(f"`{op}` " + ", ".join(f"{k} {v[0]}→{v[1]}" for k, v in d.items())
                       for op, d in (r.get("split") or []))[:110] or "—"

        def f(k, fmt="{:.3g}"):
            v = m.get(k)
            return fmt.format(v) if isinstance(v, (int, float)) else "—"
        state = "" if m else f" *({(r.get('progress') or {}).get('phase', 'not started')})*"
        L.append(f"| `{r['run']}`{state} | {r.get('frames') or '—'} | "
                 f"{', '.join(map(str, r.get('switch') or [])) or '—'} | {ch} | "
                 f"{f('n_tubes_final', '{:.0f}')} | {f('protr_final')} | {f('red_frac_final')} | "
                 f"{f('tip_act_final')} |")
    L.append("\n## What each one was asking\n")
    for r in rows:
        L.append(f"**`{r['run']}`** — {r.get('question') or '_no question recorded_'}\n")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", dest="show", action="store_true")
    a = ap.parse_args()
    names = sorted(d for d in os.listdir(LOG) if d.startswith("stage_"))
    rows = [read(n) for n in names]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "staged.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    txt = render(rows)
    p = os.path.join(OUT, "EXPERIMENTS.md")
    open(p, "w").write(txt)
    print(f"  {len(rows)} staged run(s) -> {os.path.relpath(p, HERE)}")
    if a.show:
        print("\n" + txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
