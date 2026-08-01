"""saturation -- the measurement the atlas exists to make.

plexus2.tex App. E.1: *"If the decomposition of a sufficiently broad collection of biological
simulation frameworks converges toward a compact and reusable operator vocabulary, then the
proposed operator algebra constitutes a meaningful intermediate representation for biological
modelling. Conversely, repeated discovery of genuinely new operator families indicates that the
language remains incomplete."*

That is a falsifiable claim, and this module is the instrument that tests it. Per mechanism, in
the order it was inspected, it plots the cumulative number of genuinely new contracts. A curve
that flattens is the language saturating. A curve that keeps climbing is the language telling us
it is not finished.

FOUR OUTCOMES, AGAINST THE PROMOTED LANGUAGE ONLY:

    alias           a contract we already have, registered and validated.
    refinement      an existing contract whose typed signature has to widen to admit this.
    new             vocabulary the promoted language does not have.
    implementation  a SECOND mechanism normalizing to a contract this run has already counted
                    as new. It is an interchangeable numerical realisation, not new vocabulary.

The fourth class was not in the plan; the first real ledger produced it. Four of jax-morph's
control steps -- three gene-network variants and their shared base -- all normalize to one
contract, `regulate`. Counting them as four new contracts would have inflated the headline number
by 36% on the very measurement the atlas exists to make, and it is exactly the outcome plexus2.tex
App. E.1 calls the GOOD one: "the first outcome strengthens numerical diversity". The curve counts
DISTINCT contracts.

The comparison is against `plexus.operators` and nothing else. Unreviewed code in `prototype/` or
in the `candidates/` anti-chamber is not part of the language and does not enter this measurement:
a saturation curve measured against code nobody has checked would be measuring the wrong thing.

    python saturation.py                    # the ledger + the curve, as text
    python saturation.py --plot             # _state/saturation.png
    python saturation.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "_state")
OUT_JSON = os.path.join(STATE, "saturation.json")
OUT_PNG = os.path.join(STATE, "saturation.png")

CLASSES = ["alias", "refinement", "new", "implementation", "out_of_scope",
           "unclassified"]


def classify(mech: dict, baseline: dict) -> str:
    """One mechanism -> one of CLASSES, using the FROZEN baseline, never the live registry.

    The record states a verdict; this function is where that verdict meets the evidence. A
    `new` whose contract name is already registered is reclassified as `alias` whatever the
    record says -- the ledger is not obliged to believe the record.
    """
    verdict = mech.get("verdict")
    if verdict in (None, ""):
        return "unclassified"
    if verdict == "out_of_scope":
        return "out_of_scope"

    name = (mech.get("contract") or {}).get("name")
    registered = set(baseline["registered"])

    if verdict == "new":
        if name in registered:
            return "alias"                    # the record is wrong; say so in the ledger
        return "new"
    return verdict                            # alias / refinement, as recorded


def ledger(doc: dict, baseline: dict) -> dict:
    """The per-mechanism table plus the cumulative curve, in inspection order.

    Inspection order is `order:` if the record gives one, else the order the mechanisms appear
    in the file. It matters: the curve is a statement about *the order things were found*, and
    re-sorting it (by name, by family) would turn a claim about discovery into an artefact of
    the sort key.
    """
    mechs = list(doc.get("mechanisms") or [])
    mechs.sort(key=lambda m: m.get("order", 10**6))

    rows, cum_new, curve = [], 0, []
    counts = {c: 0 for c in CLASSES}
    disputed = []
    seen_new = {}                     # contract name -> the mechanism that first introduced it
    for i, m in enumerate(mechs, 1):
        cls = classify(m, baseline)
        cname = (m.get("contract") or {}).get("name")
        if cls == "new" and cname:
            if cname in seen_new:
                cls = "implementation"          # same contract, another way of computing it
            else:
                seen_new[cname] = m.get("id")
        counts[cls] += 1
        if cls != m.get("verdict") and m.get("verdict") not in (None, ""):
            disputed.append({"id": m.get("id"), "recorded": m.get("verdict"), "ledger": cls,
                             "name": (m.get("contract") or {}).get("name")})
        if cls == "new":
            cum_new += 1
        rows.append({"n": i, "id": m.get("id"), "raw_name": m.get("raw_name"),
                     "class": cls, "contract": cname,
                     "of": m.get("of") or (seen_new.get(cname) if cls == "implementation"
                                           else None),
                     "status": m.get("status", "candidate")})
        curve.append({"n": i, "cum_new": cum_new})

    scored = sum(counts[c] for c in ("alias", "refinement", "new", "implementation"))
    return {
        "repository": doc.get("repository"),
        "commit": doc.get("commit"),
        "baseline_contracts": baseline["counts"]["contracts"],
        "counts": counts,
        "scored": scored,
        "yield_new_per_mechanism": (cum_new / scored) if scored else None,
        "new_contracts": sorted(seen_new),
        "curve": curve,
        "rows": rows,
        "disputed": disputed,
    }


def render(led: dict) -> str:
    c = led["counts"]
    w = max((len(str(r["raw_name"] or r["id"])) for r in led["rows"]), default=10)
    lines = [f"{led['repository']}  @ {led['commit']}",
             f"baseline: {led['baseline_contracts']} registered contracts", ""]
    for r in led["rows"]:
        tgt = f"  -> {r['of']}" if r["of"] else ""
        lines.append(f"  {r['n']:>3}  {str(r['raw_name'] or r['id']):<{w}}  "
                     f"{r['class']:<13} {r['contract'] or '':<24}{tgt}   [{r['status']}]")
    lines += ["",
              f"  alias {c['alias']}   refinement {c['refinement']}   NEW {c['new']}   "
              f"implementation {c['implementation']}   out_of_scope {c['out_of_scope']}   "
              f"unclassified {c['unclassified']}",
              f"  new contracts: {', '.join(led['new_contracts']) or '-'}"]
    if led["scored"]:
        lines.append(f"  yield: {led['yield_new_per_mechanism']:.2f} new contracts per "
                     f"scored mechanism ({c['new']}/{led['scored']})")
    if led["disputed"]:
        lines += ["", "  THE LEDGER DISAGREES WITH THE RECORD:"]
        for d in led["disputed"]:
            lines.append(f"    {d['id']}: recorded {d['recorded']!r}, ledger says {d['ledger']!r}"
                         f"  ({d['name']})")
    return "\n".join(lines)


def plot(led: dict, path=OUT_PNG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [p["n"] for p in led["curve"]]
    ys = [p["cum_new"] for p in led["curve"]]
    fig, ax = plt.subplots(figsize=(5.6, 3.6), facecolor="black")
    ax.set_facecolor("black")
    ax.step(xs, ys, where="post", color="#4FA3FF", lw=1.8)
    ax.plot(xs, xs, ls=":", lw=1.0, color="#777777")          # every mechanism new = no saturation
    for s in ax.spines.values():
        s.set_color("#888888")
    ax.tick_params(colors="#BBBBBB", labelsize=8)
    ax.set_xlabel("mechanisms inspected", color="#DDDDDD", fontsize=9)
    ax.set_ylabel("cumulative NEW contracts", color="#DDDDDD", fontsize=9)
    ax.text(0.02, 0.96, f"{led['repository']}", transform=ax.transAxes, color="white",
            fontsize=11, va="top", ha="left")
    ax.text(0.02, 0.87, "dotted = no saturation at all", transform=ax.transAxes,
            color="#999999", fontsize=8, va="top", ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="black")
    return path


# ------------------------------------------------------------------------------------------- #
def selftest():
    baseline = {"registered": {"diffuse": {}, "cell_divide": {}}, "counts": {"contracts": 2}}
    doc = {"repository": "r", "commit": "c", "mechanisms": [
        {"id": "a", "order": 1, "verdict": "alias", "of": "diffuse",
         "contract": {"name": "diffuse"}},
        {"id": "b", "order": 2, "verdict": "new", "contract": {"name": "trace_replay"}},
        {"id": "c", "order": 3, "verdict": "new", "contract": {"name": "secrete"}},
        {"id": "d", "order": 4, "verdict": "new", "contract": {"name": "diffuse"}},      # alias
        {"id": "e", "order": 5, "verdict": "refinement", "of": "cell_divide",
         "contract": {"name": "cell_divide"}},
        {"id": "f", "order": 6, "verdict": "out_of_scope", "contract": {"name": "z"}},
        {"id": "g", "order": 7},                                                          # unclassified
    ]}
    led = ledger(doc, baseline)
    c = led["counts"]
    ok = (c["alias"] == 2 and c["new"] == 2 and c["refinement"] == 1
          and c["out_of_scope"] == 1 and c["unclassified"] == 1
          and [p["cum_new"] for p in led["curve"]] == [0, 1, 2, 2, 2, 2, 2]
          and len(led["disputed"]) == 1)
    print(render(led))
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    print("  a `new` whose name is already registered was demoted to alias,")
    print("  and the demotion is reported as a dispute rather than silently applied.")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default=os.path.join(HERE, "atlas_record.yaml"))
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())

    import record
    import registry_view
    led = ledger(record.load(a.record), registry_view.load())
    print(render(led))
    os.makedirs(STATE, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(led, f, indent=2)
    if a.plot:
        print(f"\nplot -> {plot(led)}")


if __name__ == "__main__":
    main()
