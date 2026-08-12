#!/usr/bin/env python
"""Audit what the campaign LEARNED, against the epistemic framework -- not what it ran.

Cedric, 12 August: *"first we need diagnosis, can you audit current loop with epistemic/ontology"*,
and then *"make a script we can reuse with a md definition file that we can modify later."*

The framework is Allier & Saalfeld 2026 (`/workspace/NeuralGraph/instructions_epistemic_analysis.md`)
and every definition, marker and constant lives in `epistemic_spec.md` beside this file. The script
holds the arithmetic and nothing else; to change what counts as a mode, edit the spec.

WHAT MAKES THIS DIFFERENT FROM THE NEURALGRAPH ANALYSIS, which parsed 348 iterations of prose by
hand. Okuda's reasoning is already STRUCTURED: a prediction is a field, its outcome is a field,
descent is a field, and the surprise a slot chases is a field. So six of the twelve modes are
COMPUTED from the record rather than tagged, and their rates are measurements. The other six live
only as prose, and for those this counts marker hits and labels them candidates -- a marker count is
a lower bound on a mode, never a measurement of it, and the output says so on every line.

THE ONE THING THIS AUDIT ADDS TO THE FRAMEWORK is the seed floor. The substrate is stochastic and
the campaign runs `intent: replicate` slots, so its own reproducibility is measurable: the spread
between a replicate and its parent. That turns "was the prediction right" into the sharper question
"was the prediction ASKABLE" -- a threshold finer than the noise cannot be answered by one run, and
counting it as refuted credits the loop with a falsification it never performed.

    python epistemic_audit.py                  write the three files
    python epistemic_audit.py --print          also print the summary
    python epistemic_audit.py --spec other.md
"""
import argparse
import collections
import json
import math
import os
import re

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "campaign", "epistemic")


# --------------------------------------------------------------------------- the spec
def load_spec(path):
    """Every ```yaml block in the markdown, merged. The prose around them is for the reader."""
    txt = open(path).read()
    spec = {}
    for block in re.findall(r"```yaml\n(.*?)```", txt, re.S):
        d = yaml.safe_load(block) or {}
        for k, v in d.items():
            if isinstance(v, dict) and isinstance(spec.get(k), dict):
                spec[k].update(v)
            elif isinstance(v, list) and isinstance(spec.get(k), list):
                spec[k].extend(v)
            else:
                spec[k] = v
    return spec


def fam(metric):
    return re.sub(r"_(peak|final|floor|span|trend|measured_frac)$", "", str(metric))


# --------------------------------------------------------------------------- detectors
def has_prediction(r):
    return bool((r.get("scored") or {}).get("predict"))


def outcome_refuted(r):
    return (r.get("scored") or {}).get("outcome") == "refuted"


def is_sweep(r):
    return r.get("intent") == "sweep" or bool(r.get("sweep"))


def is_replicate(r):
    return r.get("intent") == "replicate" or bool(r.get("replicate"))


def chases_set(r):
    return bool(r.get("chases"))


DETECT = {"has_prediction": has_prediction, "outcome_refuted": outcome_refuted,
          "is_sweep": is_sweep, "is_replicate": is_replicate, "chases_set": chases_set}


def lineage_of(name, by, depth=99):
    """The chain of ancestors. One `parent` per record, so descent is a line, not a graph."""
    out, n, seen = [], name, set()
    while n and n not in seen and len(out) < depth:
        seen.add(n); out.append(n)
        n = (by.get(n) or {}).get("parent")
    return out


def edit_key(r):
    """What an edit CHANGES, ignoring the value -- so the same lever on two lineages matches."""
    e = r.get("edit")
    if not e:
        return None
    e = e if isinstance(e[0], list) else [e]
    ks = []
    for x in e:
        if not isinstance(x, list):
            continue
        if x[0] == "set_param":
            ks.append(f"set_param:{x[1]}")
        elif x[0] in ("add_op", "remove_op", "set_impl"):
            ks.append(f"{x[0]}:{x[1]}" + (f":{x[2]}" if x[0] == "set_impl" and len(x) > 2 else ""))
    return "+".join(sorted(ks)) or None


# --------------------------------------------------------------------------- the audit
def audit(spec):
    C = spec["corpus"]
    rs = [json.loads(l) for l in open(os.path.join(HERE, C["records"]))]
    by = {r["name"]: r for r in rs}
    # EVIDENCE, NOT ATTEMPTS. A run killed by the round cap produced no metrics and reasoned about
    # nothing; counting it would flatter every rate below. It is reported separately.
    ev = [r for r in rs if (r.get("metrics") or {})] if C.get("require_metrics") else rs
    lost = len(rs) - len(ev)

    R = {"n_runs": len(rs), "n_evidence": len(ev), "n_lost": lost,
         "rounds": (min(r["round"] for r in rs), max(r["round"] for r in rs))}

    # ---- modes
    floors = spec["seed_floor"]
    rows = []
    for m in spec["modes"]:
        # `cross_lineage_edit` is computed in its own pass below: it is a property of a PAIR of
        # runs, not of one, so it cannot be a per-record detector like the rest.
        if m["kind"] != "computed" or m["detect"] not in DETECT:
            continue
        f = DETECT[m["detect"]]
        hits = [r for r in ev if f(r)]
        first = min((r["round"] for r in hits), default=None)
        c = sum(1 for r in hits if (r.get("scored") or {}).get("outcome") == "confirmed")
        x = sum(1 for r in hits if (r.get("scored") or {}).get("outcome") == "refuted")
        rows.append(dict(name=m["name"], n=len(hits), first=first,
                         val=(c / (c + x) if (c + x and m.get("validated")) else None),
                         conf=c, ref=x, note=m.get("note", "")))
    R["computed"] = rows

    # ---- transfer: the same lever, on a parent from a different lineage
    seen_on = collections.defaultdict(set)
    transfer = []
    for r in sorted(ev, key=lambda z: z["name"]):
        k = edit_key(r)
        if not k:
            continue
        root = lineage_of(r.get("parent") or r["name"], by)[-1]
        if seen_on[k] and root not in seen_on[k]:
            transfer.append((r["name"], k, root, sorted(seen_on[k])))
        seen_on[k].add(root)
    R["transfer"] = transfer
    R["levers"] = len(seen_on)

    # ---- the seed floor, remeasured from this corpus so the spec can be checked against it
    K = [k for k in floors if not k.startswith("_")]
    meas = collections.defaultdict(list)
    for r in ev:
        if not is_replicate(r):
            continue
        p = by.get(r.get("parent"))
        if not p:
            continue
        a, b = (p.get("metrics") or {}), (r.get("metrics") or {})
        for k in a:
            if fam(k) in K and isinstance(a.get(k), (int, float)) and a[k] \
                    and isinstance(b.get(k), (int, float)):
                meas[fam(k)].append(abs(b[k] - a[k]) / abs(a[k]))
    R["floor_measured"] = {k: float(np.median(v)) for k, v in sorted(meas.items()) if len(v) >= 3}

    # ---- askability: is the prediction finer than the noise?
    ask = []
    for r in ev:
        sc = r.get("scored") or {}
        m = re.match(r"\s*([a-z_0-9]+)\s*([<>]=?)\s*([0-9.eE+-]+)", str(sc.get("predict", "")))
        p = by.get(r.get("parent"))
        if not m or not p:
            continue
        met, thr = m.group(1), float(m.group(3))
        base = (p.get("metrics") or {}).get(met)
        if not isinstance(base, (int, float)) or not base:
            continue
        fl = floors.get(fam(met), floors.get("_default", 0.30))
        ask.append(dict(run=r["name"], metric=met, base=base, thr=thr,
                        rel=abs(thr - base) / abs(base), floor=fl,
                        default_floor=fam(met) not in floors,
                        outcome=sc.get("outcome")))
    R["ask"] = ask

    # ---- does a refutation change anything? (the next round's slots on the same lever)
    ref = [r for r in ev if outcome_refuted(r)]
    followed = 0
    for r in ref:
        nxt = f"r{int(r['round'][1:]) + 1:03d}"
        k = edit_key(r)
        if k and any(q["round"] == nxt and edit_key(q) == k for q in ev):
            followed += 1
    R["refutation"] = dict(n=len(ref), followed=followed)

    # ---- prose modes: candidates only
    txt = {}
    for key in ("knowledge", "analysis"):
        p = os.path.join(HERE, C[key])
        txt[key] = open(p).read() if os.path.exists(p) else ""
    R["marker"] = []
    for m in spec["modes"]:
        if m["kind"] != "marker":
            continue
        body = "\n".join(txt[f] for f in m.get("files", []))
        R["marker"].append(dict(name=m["name"], n=len(re.findall(m["pattern"], body)),
                                files=",".join(m.get("files", []))))
    return R


# --------------------------------------------------------------------------- report
def write(R, spec):
    os.makedirs(OUT, exist_ok=True)
    a = R["ask"]
    under = [x for x in a if x["rel"] < x["floor"]]
    over = [x for x in a if x["rel"] >= x["floor"]]

    def rate(g):
        c = sum(1 for x in g if x["outcome"] == "confirmed")
        f = sum(1 for x in g if x["outcome"] == "refuted")
        return (c, f, 100 * c / max(c + f, 1))

    L = []
    w = L.append
    w(f"# Epistemic audit: Okuda discovery loop\n")
    w(f"**Rounds** {R['rounds'][0]}–{R['rounds'][1]} | **runs** {R['n_runs']} | "
      f"**runs that produced evidence** {R['n_evidence']} "
      f"({100 * R['n_evidence'] / R['n_runs']:.0f}%) | "
      f"**produced nothing** {R['n_lost']}\n")
    w("Generated by `epistemic_audit.py` from `epistemic_spec.md`. Framework: Allier & Saalfeld "
      "2026. Rates below are over runs that produced evidence; the rest were killed by the round "
      "cap and reasoned about nothing.\n")

    w("\n## Priors excluded\n")
    for p in spec.get("priors", []):
        w(f"- {p}")

    w("\n## Reasoning modes — computed from the record\n")
    w("| Mode | Count | Validation | First | What it is |")
    w("|---|---:|---:|---|---|")
    for m in R["computed"]:
        v = "—" if m["val"] is None else f"**{100 * m['val']:.0f}%** ({m['conf']}/{m['conf'] + m['ref']})"
        w(f"| {m['name']} | {m['n']} | {v} | {m['first'] or '—'} | {m['note'].strip()[:90]} |")

    w("\n## Reasoning modes — prose only, candidates not measurements\n")
    w("A marker count is a lower bound. These modes have no field in the record, so the loop "
      "cannot score them, chain them, or notice their absence.\n")
    w("| Mode | Marker hits | Source |")
    w("|---|---:|---|")
    for m in R["marker"]:
        w(f"| {m['name']} | {m['n']} | {m['files']} |")

    w("\n## The seed floor\n")
    w("Median |Δ| against the parent, over replicate runs of the same composition at a fresh seed. "
      "This is the substrate's own reproducibility and therefore the finest question a single run "
      "can answer.\n")
    w("| Metric | Measured floor | In the spec |")
    w("|---|---:|---:|")
    for k, v in sorted(R["floor_measured"].items(), key=lambda kv: -kv[1]):
        w(f"| {k} | {100 * v:.0f}% | {100 * spec['seed_floor'].get(k, float('nan')):.0f}% |")

    w("\n## Askability — was the prediction finer than the noise?\n")
    if a:
        w(f"- predictions with a comparable parent value: **{len(a)}**")
        w(f"- asking for less than the metric's own floor: **{len(under)} "
          f"({100 * len(under) / len(a):.0f}%)**")
        w(f"- median change asked for: **{100 * np.median([x['rel'] for x in a]):.0f}%**, "
          f"median floor: **{100 * np.median([x['floor'] for x in a]):.0f}%**")
        c1, f1, p1 = rate(under); c2, f2, p2 = rate(over)
        w(f"- validation **below** the floor: {c1}/{c1 + f1} = **{p1:.0f}%**")
        w(f"- validation **above** the floor: {c2}/{c2 + f2} = **{p2:.0f}%**")
        w("\nA prediction below the floor is not a prediction. Scoring it `refuted` credits the "
          "loop with a falsification it did not perform, and scoring it `confirmed` credits it "
          "with a discovery.\n")

    w("\n## Cross-lineage transfer\n")
    w("COUNTED STRUCTURALLY, WHICH OVERSTATES IT. This counts a lever appearing on a parent whose "
      "basis root differs from where it was first used -- it does not check that the Proposer "
      "*intended* a transfer, or cited the earlier result. The framework's Analogy/Transfer is "
      "deliberate reuse; this is an upper bound on it.\n")
    w(f"- distinct levers used: **{R['levers']}**")
    w(f"- levers ever applied on a second lineage: **{len(R['transfer'])}**\n")
    w("The framework weights cross-context generalisation highest of all evidence "
      "(`per_block` 0.15, linear). A law tested on one lineage cannot exceed medium confidence "
      "however many times it is confirmed there.\n")
    for n, k, root, was in R["transfer"][:12]:
        w(f"- `{n}` applied `{k}` on **{root}**, first seen on {', '.join(was)}")

    w("\n## Falsification — does a refutation change anything?\n")
    f = R["refutation"]
    w(f"- refuted predictions: **{f['n']}**")
    w(f"- followed next round by another slot on the same lever: **{f['followed']}** "
      f"({100 * f['followed'] / max(f['n'], 1):.0f}%)\n")
    w("Popper's asymmetry makes refutation the strongest evidence available — but only if a "
      "hypothesis is revised because of it. A refutation nothing follows up is a discarded run.\n")

    p = os.path.join(OUT, "epistemic_analysis.md")
    open(p, "w").write("\n".join(L) + "\n")

    # detailed: every askability row, the framework's "exhaustive list"
    D = ["# Epistemic audit — detailed\n",
         "Every scored prediction, with the change it asked for against its metric's measured "
         "seed floor. `default` marks a metric with no measured floor, using the spec's fallback.\n",
         "| Run | Metric | Parent | Threshold | Asked | Floor | | Outcome |",
         "|---|---|---:|---:|---:|---:|---|---|"]
    for x in sorted(a, key=lambda z: z["rel"]):
        mark = "**below**" if x["rel"] < x["floor"] else "above"
        D.append(f"| {x['run']} | {x['metric']}{'*' if x['default_floor'] else ''} | "
                 f"{x['base']:.4g} | {x['thr']:.4g} | {100 * x['rel']:.1f}% | "
                 f"{100 * x['floor']:.0f}% | {mark} | {x['outcome']} |")
    open(os.path.join(OUT, "epistemic_detailed.md"), "w").write("\n".join(D) + "\n")

    # edges: descent and surprise-chasing, which the record already carries
    E = ["# Epistemic audit — edges\n",
         "Causal links the record carries directly: `parent` (descent) and `chases` "
         "(an unpredicted result becoming an experiment). Hand-tracing, as the NeuralGraph "
         "analysis needed, is unnecessary here — the loop writes its own edges.\n",
         "| From | Mode | To | Mode | Type | Edit |", "|---|---|---|---|---|---|"]
    rs = [json.loads(l) for l in open(os.path.join(HERE, spec["corpus"]["records"]))]
    for r in rs:
        if r.get("chases"):
            E.append(f"| {r['chases']} | Surprise | {r['name']} | Deduction | chases | "
                     f"{edit_key(r) or '—'} |")
    for r in rs:
        if r.get("parent") and r.get("edit"):
            E.append(f"| {r['parent']} | — | {r['name']} | {r.get('intent') or '—'} | descent | "
                     f"{edit_key(r) or '—'} |")
    open(os.path.join(OUT, "epistemic_edges.md"), "w").write("\n".join(E) + "\n")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.path.join(HERE, "epistemic_spec.md"))
    ap.add_argument("--print", dest="show", action="store_true")
    a = ap.parse_args()
    spec = load_spec(a.spec)
    R = audit(spec)
    p = write(R, spec)
    print(f"  {R['n_evidence']}/{R['n_runs']} runs produced evidence "
          f"({R['n_lost']} produced nothing)")
    for m in R["computed"]:
        v = "" if m["val"] is None else f"   validation {100 * m['val']:.0f}%"
        print(f"  {m['name']:20s} {m['n']:4d}{v}")
    print(f"  {'cross-lineage transfer':20s} {len(R['transfer']):4d}  of {R['levers']} levers")
    print(f"\n  -> {os.path.relpath(p, os.path.dirname(HERE))} (+ _detailed, _edges)")
    if a.show:
        print("\n" + open(p).read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
