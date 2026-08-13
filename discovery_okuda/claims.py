#!/usr/bin/env python
"""The claim ledger: knowledge as an object experiments can act on, not prose they are summarised in.

Phase 1 of `refactor_plan.md`. Every rule, threshold and act lives in `crew/claims.md`; this file
holds the arithmetic and the file format and nothing else.

WHY THIS EXISTS, in one measurement. The epistemic audit of r001-r022 found 126 predictions, 95
refutations and eleven STANDING LAWS -- and the laws were paragraphs. Nothing scored them, nothing
bred from them, and nothing noticed that two of them contradict each other: L5 says
`cell_chem_from_shape.beta < 0` extinguishes the activator *morphotype-independent*, L9 says it is
*base-dependent*. They sat in the same file for six rounds and no experiment was ever posed to
separate them, because there was no object to pose one against.

APPEND-ONLY, AND THAT IS THE POINT. A claim's history is the finding. The old `knowledge.md` was
rewritten each round, so a law contested in r006 and quietly restated in r013 was indistinguishable
from one confirmed twice. Here every state is a line, `knowledge.md` becomes a rendered VIEW, and
the render can show when a status changed and on what evidence.

EVIDENCE IS WEIGHTED, NOT COUNTED, which is where the audit's two findings meet. 65% of the old
campaign's predictions asked for less than their metric's own seed-to-seed spread; ten such
confirmations are ten coin tosses. So each entry carries `weight = min(1, effect / floor)` and a
claim resting on sub-floor experiments accumulates almost nothing.

    python claims.py --validate          well-formedness over the whole ledger
    python claims.py --render            rewrite campaign/knowledge.md from it
    python claims.py --list              one line per claim
    python claims.py --seed seeds.json   append claims (used once, to transcribe the old laws)
"""
import argparse
import json
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "crew", "claims.md")
LEDGER = os.path.join(HERE, "campaign", "claims.jsonl")
RECORDS = os.path.join(HERE, "campaign", "records.jsonl")
FLOORS = os.path.join(HERE, "epistemic_spec.md")


def load_spec(path=SPEC):
    """Every ```yaml block in the markdown, merged. The prose around them is for the reader."""
    spec = {}
    for block in re.findall(r"```yaml\n(.*?)```", open(path).read(), re.S):
        d = yaml.safe_load(block) or {}
        for k, v in d.items():
            if isinstance(v, dict) and isinstance(spec.get(k), dict):
                spec[k].update(v)
            else:
                spec[k] = v
    return spec


def floors(path=FLOORS):
    for block in re.findall(r"```yaml\n(.*?)```", open(path).read(), re.S):
        d = yaml.safe_load(block) or {}
        if "seed_floor" in d:
            return d["seed_floor"]
    return {"_default": 0.20}


def fam(metric):
    return re.sub(r"_(peak|final|floor|span|trend|measured_frac)$", "", str(metric))


# --------------------------------------------------------------------------- the ledger
def load(path=LEDGER):
    """Every line, then the LAST state of each claim. Both are returned: the history is evidence."""
    hist = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        hist.append(json.loads(line))
                    except Exception:
                        pass
    cur = {}
    for c in hist:
        cur[c["id"]] = c
    return cur, hist


def append(claim, path=LEDGER):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(claim, sort_keys=True) + "\n")


def next_id(cur, prefix="C"):
    n = 0
    for k in cur:
        m = re.match(rf"{prefix}(\d+)$", k)
        if m:
            n = max(n, int(m.group(1)))
    return f"{prefix}{n + 1:03d}"


# --------------------------------------------------------------------------- weighting
def resolvability(metric, parent_value, threshold, fl=None):
    """How much bigger than the noise the asked-for effect is, capped at 1.

    A prediction that asks for less than the floor cannot be answered by one run, so it contributes
    proportionally less. This is the ONLY place the audit's design finding enters the knowledge
    layer, and it is why `evidence_for` is a weight and not a count.
    """
    fl = fl or floors()
    f = fl.get(fam(metric), fl.get("_default", 0.20))
    if not parent_value:
        return 0.0, f
    rel = abs(float(threshold) - float(parent_value)) / abs(float(parent_value))
    return min(1.0, rel / f) if f else 1.0, f


def weigh(claim, spec):
    """(for, against) total weight."""
    ev = spec.get("evidence", {})
    dw = ev.get("default_weight", {})
    tot = {}
    for side in ("evidence_for", "evidence_against"):
        s = 0.0
        for e in (claim.get(side) or []):
            w = e.get("weight")
            if w is None:
                w = dw.get(e.get("act", "predict"), 1.0)
            s += float(w)
        tot[side] = s
    return tot["evidence_for"], tot["evidence_against"]


def status_for(claim, spec):
    """What the evidence says the status should be -- computed, never asserted by an agent."""
    ev = spec.get("evidence", {})
    f, a = weigh(claim, spec)
    if claim.get("status") == "superseded":
        return "superseded"
    if f >= ev.get("contested_min", 0.75) and a >= ev.get("contested_min", 0.75):
        return "contested"
    if a - f >= ev.get("refute_threshold", 1.5):
        return "refuted"
    if f - a >= ev.get("support_threshold", 2.0):
        return "supported"
    return "proposed" if claim.get("status") in (None, "proposed", "stale") else claim["status"]


# --------------------------------------------------------------------------- validation
def validate(cur, hist, spec, runs=None):
    """Well-formedness, per `crew/claims.md`. Returns a list of problems, empty if clean."""
    sch, kinds = spec["schema"], set(spec["kind"])
    stat = set(spec["status"])
    trans = spec.get("transitions", {})
    out = []
    for c in hist:
        cid = c.get("id", "?")
        for k in sch["required"]:
            if not c.get(k):
                out.append(f"{cid}: missing required field `{k}`")
        if c.get("kind") not in kinds:
            out.append(f"{cid}: kind {c.get('kind')!r} is not one of {sorted(kinds)}")
        if c.get("status") not in stat:
            out.append(f"{cid}: status {c.get('status')!r} is not one of {sorted(stat)}")
        sc = c.get("scope") or {}
        if not (sc.get("lineages") or sc.get("regimes")):
            out.append(f"{cid}: scope names no lineage and no regime -- it cannot be transferred")
        st = (c.get("statement") or "").strip()
        if len(st) < 15 or not re.search(r"\s", st):
            out.append(f"{cid}: statement is not an assertion ({st[:40]!r})")
        for p in (c.get("parents") or []):
            if p not in cur:
                out.append(f"{cid}: parent {p} does not exist")
        if runs is not None:
            for side in ("evidence_for", "evidence_against"):
                for e in (c.get(side) or []):
                    if e.get("run") and e["run"] not in runs:
                        out.append(f"{cid}: {side} names run {e['run']}, not in records.jsonl")
    # transitions, over the history of each id
    seq = {}
    for c in hist:
        seq.setdefault(c["id"], []).append(c.get("status"))
    for cid, ss in seq.items():
        for a, b in zip(ss, ss[1:]):
            if a != b and b not in trans.get(a, []):
                out.append(f"{cid}: illegal transition {a} -> {b}")
    return out


# --------------------------------------------------------------------------- render
def render(cur, spec, path=None):
    """`knowledge.md` as a VIEW. The Analyst writes claims; this file is output, never input."""
    r = spec.get("render", {})
    order = r.get("order", ["contested", "supported", "proposed", "stale", "refuted", "superseded"])
    path = path or os.path.join(HERE, r.get("target", "campaign/knowledge.md"))
    L = ["# Knowledge — the claim ledger, rendered\n",
         "**Generated by `claims.py --render` from `campaign/claims.jsonl`. Do not edit.** The "
         "Analyst writes claims and evidence; this file is a view of them. It is regenerated every "
         "round, and because the ledger is append-only it can show when a status changed and on "
         "what — which the hand-written version structurally could not.\n"]
    by = {}
    for c in cur.values():
        by.setdefault(c.get("status", "proposed"), []).append(c)
    for st in order:
        cs = sorted(by.get(st, []), key=lambda c: c["id"])
        if not cs:
            continue
        L.append(f"\n## {st.upper()}  ({len(cs)})\n")
        for c in cs:
            f, a = weigh(c, spec)
            L.append(f"### {c['id']} — {c['statement']}")
            sc = c.get("scope") or {}
            L.append(f"*{c['kind']}* | scope: lineages {sc.get('lineages') or '—'}, "
                     f"regimes {sc.get('regimes') or '—'} | "
                     f"weight **for {f:.1f} / against {a:.1f}**"
                     + (f" | parents {', '.join(c['parents'])}" if c.get("parents") else "")
                     + ("  | *seeded*" if c.get("seeded") else ""))
            if c.get("mechanism"):
                L.append(f"\n{c['mechanism']}")
            for side, lab in (("evidence_for", "for"), ("evidence_against", "against")):
                for e in (c.get(side) or [])[-int(r.get("show_evidence", 4)):]:
                    w = f" (w {e['weight']:.2f})" if r.get("show_weights") and "weight" in e else ""
                    L.append(f"- **{lab}** `{e.get('run', '?')}` {e.get('act', 'predict')}"
                             f"{w} {e.get('note', '')}")
            L.append("")
    open(path, "w").write("\n".join(L) + "\n")
    return path


# --------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--list", dest="lst", action="store_true")
    ap.add_argument("--seed", default=None, help="a JSON list of claims to append")
    ap.add_argument("--ledger", default=LEDGER)
    a = ap.parse_args()

    spec = load_spec()
    if a.seed:
        cur, _ = load(a.ledger)
        added = 0
        for c in json.load(open(a.seed)):
            c.setdefault("id", next_id(cur))
            c.setdefault("status", "proposed")
            c.setdefault("seeded", True)
            cur[c["id"]] = c
            append(c, a.ledger); added += 1
        print(f"  appended {added} claim(s) -> {os.path.relpath(a.ledger, HERE)}")

    cur, hist = load(a.ledger)
    if not cur:
        print("  ledger is empty"); return 0
    runs = None
    if os.path.exists(RECORDS):
        runs = {json.loads(l)["name"] for l in open(RECORDS) if l.strip()}

    if a.validate or a.seed:
        # SAY WHEN THE RUN CHECK COULD NOT RUN. `validate` skips the "does this run exist" test when
        # `runs` is None, and it was None because records.jsonl had just been cleared for a fresh
        # campaign -- so a ledger citing eight runs that had been DELETED validated clean. A check
        # that silently does not happen is worse than one that fails.
        if runs is None:
            print("  ! records.jsonl absent -- evidence run ids were NOT checked against it")
        probs = validate(cur, hist, spec, runs)
        print(f"  {len(cur)} claims, {len(hist)} ledger lines, "
              f"{len(probs)} problem(s)")
        for p in probs:
            print(f"    {p}")
        if probs:
            return 1
    if a.lst:
        for c in sorted(cur.values(), key=lambda c: c["id"]):
            f, x = weigh(c, spec)
            print(f"  {c['id']}  {c['status']:10s} {c['kind']:15s} "
                  f"for {f:4.1f} / against {x:4.1f}  {c['statement'][:64]}")
    if a.render:
        p = render(cur, spec)
        print(f"  rendered {len(cur)} claims -> {os.path.relpath(p, HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
