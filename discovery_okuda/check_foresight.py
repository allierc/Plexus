#!/usr/bin/env python
"""Run the Forecaster and the Eye against each other on the BASIS, and score the pair.

CEDRIC, 13 AUGUST: *"can you test the forecaster-VLLM difference on the baseline, even if current
round is only 02?"*

WHY THE BASIS IS THE RIGHT PLACE TO TEST THIS AND NOT A COMPROMISE. Forty-six of its members have a
spec, a strip and a landed run on disk, which is more finished specimens than the live campaign will
produce in a week -- and the whole point of the pair is that neither role sees the other, so it does
not matter that the runs already happened. The Forecaster is handed the spec and `knowledge.md` and
nothing else; the Eye is handed the frames and nothing else. Neither is shown the other's answer,
neither is shown the metrics, and there is no path between them.

WHAT IT CANNOT TELL YOU, said before the numbers so the numbers are not over-read. In the live loop
the Forecaster predicts a spec that has NEVER RUN, and here it predicts one whose result is already
in `knowledge.md` -- these are the runs the campaign's conclusions were drawn FROM. So this is an
upper bound on foresight, not an estimate of it: it measures whether the knowledge can reproduce
what it was built on, which is the easiest version of the question. A campaign that scores badly
HERE is in trouble; one that scores well here has only shown it is not incoherent.

The pairing is still worth the tokens, because it tests four things the live loop cannot test until
a round has finished: that both roles fill the form, that `foresight.py` parses what they actually
write rather than what the schema hoped for, that the slots discriminate between different bodies,
and that the two roles disagree at all -- if they agree on everything, either the schema is too
coarse or one of them is not looking.

    python check_foresight.py --runs b_star b_null_plain b_gs_plain_soft_lo
    python check_foresight.py --n 6                  six spread across the basis families
    python check_foresight.py --n 6 --eye-only       re-score without paying for forecasts again
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, os.path.join(HERE, "agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LOG = os.environ.get("OKUDA_LOG", os.path.join(ROOT, "log", "okuda"))
CONFIG = os.path.join(ROOT, "config", "okuda")
OUT = os.path.join(HERE, "campaign", "foresight_basis.json")

import foresight as F  # noqa: E402


def basis_runs():
    """Basis members with BOTH a spec and a strip -- the pair needs one input for each role."""
    out = []
    for name in sorted(os.listdir(LOG)):
        if not name.startswith("b_"):
            continue
        if os.path.exists(os.path.join(LOG, name, "strip.png")) and \
           os.path.exists(os.path.join(CONFIG, f"{name}.yaml")):
            out.append(name)
    return out


def spread(names, n):
    """`n` members from DIFFERENT families, not the first `n` alphabetically.

    The basis is sorted by prefix, so the first six are six Brusselator variants -- six spheres, in
    all likelihood, and a score over six spheres would say the roles agree about spheres. Walking
    one member per family and then looping gives the schema something to discriminate BETWEEN, which
    is the property being tested.
    """
    fams = {}
    for x in names:
        fams.setdefault(x.split("_")[1] if "_" in x else x, []).append(x)
    order, keys = [], sorted(fams)
    while len(order) < len(names):
        for k in keys:
            if fams[k]:
                order.append(fams[k].pop(0))
    return order[:n]


def forecast_one(name, knowledge, ledger):
    from crew import forecaster
    with open(os.path.join(CONFIG, f"{name}.yaml")) as f:
        spec = yaml.safe_load(f)
    spec["name"] = name
    return forecaster.run({"item": name, "specs": [spec], "history": knowledge,
                           "claim_ledger": ledger, "log_root": LOG})


def observe_one(name):
    from crew import eye
    # THE EYE IS GIVEN `metrics` AND SHOWS THE MODEL NONE OF IT -- only `camera_lbox` reaches the
    # prompt, as the scale-bar sentence. Handing over the whole diag summary here mirrors exactly
    # what the round does, so this test exercises the real blindness rather than a stricter one.
    d = os.path.join(LOG, name, "diag.json")
    m = (json.load(open(d)).get("summary") or {}) if os.path.exists(d) else {}
    return eye.run({"item": name, "metrics": {name: m}, "log_root": LOG})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--eye-only", action="store_true",
                    help="reuse the forecasts already in campaign/foresight_basis.json")
    a = ap.parse_args()

    avail = basis_runs()
    runs = a.runs or spread(avail, a.n)
    missing = [r for r in runs if r not in avail]
    if missing:
        print(f"no spec or no strip: {', '.join(missing)}")
        runs = [r for r in runs if r in avail]
    if not runs:
        print("nothing to test")
        return 1

    prev = json.load(open(OUT)) if (a.eye_only and os.path.exists(OUT)) else {}

    import claims as K
    spec = K.load_spec()
    cur, _h = K.load()
    ledger = [{"id": c["id"], "statement": c["statement"], "status": c.get("status"),
               "kind": c["kind"]} for c in cur.values()]
    kn = os.path.join(HERE, "campaign", "knowledge.md")
    knowledge = open(kn).read() if os.path.exists(kn) else ""
    print(f"{len(runs)} basis runs, {len(ledger)} claims, knowledge.md {len(knowledge)} chars\n")

    pairs = {}
    for i, name in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {name}", flush=True)
        fc = (prev.get(name) or {}).get("forecast_text") or forecast_one(name, knowledge, ledger)
        ob = observe_one(name)
        pairs[name] = {"forecast_text": fc, "observed_text": ob}
        for tag, t in (("forecast", fc), ("observed", ob)):
            print(f"    {tag:9s} " + (" | ".join(f"{k}={v}" for k, v in F.parse(t).items())
                                      or "NOTHING PARSED"))

    V = F.vocab()
    scored = {n: F.score(p["forecast_text"], p["observed_text"], V) for n, p in pairs.items()}
    means = [r["foresight"] for r in scored.values() if r["foresight"] is not None]
    rs = {"runs": scored, "scored_runs": len(means),
          "foresight": round(sum(means) / len(means), 3) if means else None,
          "forecast_only": [], "observed_only": []}
    print("\n" + F.render(rs))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({n: {**pairs[n], **{k: v for k, v in scored[n].items() if k != "per_slot"},
                       "per_slot": scored[n]["per_slot"]} for n in pairs}, f, indent=1, default=str)
    print(f"\n-> {os.path.relpath(OUT, ROOT)}")
    print("\nUPPER BOUND, NOT AN ESTIMATE: these runs are already in knowledge.md, so the "
          "Forecaster is reproducing what it was built from. The live loop forecasts specs that "
          "have never run and must score lower.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
