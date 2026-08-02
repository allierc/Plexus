"""catalog -- the cross-repository measurement, which is the only place the claim can be tested.

One atlas answers "can Plexus say what THIS framework says". That is a shape, not an argument. The
claim `plexus2.tex` App. "Building the Plexus operator atlas" actually makes is about what happens ACROSS frameworks:

    if decomposing a sufficiently broad collection of biological simulation frameworks converges
    toward a compact and reusable operator vocabulary, the algebra is a meaningful intermediate
    representation. Conversely, repeated discovery of genuinely new operator families indicates
    the language remains incomplete.

That is a statement about a CURVE, and a curve needs more than one point. This module merges every
`atlas_*/atlas_record.yaml` into one ledger, counts each contract ONCE across repositories in the
order the repositories were done, and plots the cumulative-new curve.

WHY THIS IS SEPARATE FROM `saturation.py`. That module scores ONE repository against the frozen
promoted baseline, and its `--prior` flag was a patch for the two-repository case. It does not
scale: with N repositories you would be passing N-1 priors and the answer would depend on the
order you happened to type them. The catalog owns the cross-repository question outright.

WHAT IT DELIBERATELY DOES NOT DO. It does not promote, implement, or touch
`src/plexus/operators/`. Nothing here writes into the language. Extraction is cheap and
reversible; committing the anti-chamber to one framework's shape before the catalog exists is
neither. Implement and promote AFTER the curve, on the contracts that survive it.

    python catalog.py                    # the cross-repository ledger, as text
    python catalog.py --plot             # + the cumulative-new curve
    python catalog.py --order atlas_jax atlas_cc3d
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "log", "catalog.json")
OUT_PNG = os.path.join(HERE, "log", "catalog.png")

# Only these count as vocabulary. `out_of_scope` models no biology; `unclassified` is unfinished
# and is reported rather than silently dropped.
SCORED = ("alias", "refinement", "new", "implementation")


def load_atlases(order=None):
    """Every atlas_*/atlas_record.yaml, in the order the repositories were done.

    Order matters and is not cosmetic: the curve is a claim about the order things were FOUND, so
    re-sorting it (by name, by size) would turn a discovery history into an artefact of the sort
    key. Default is directory order; `--order` states it explicitly.
    """
    import yaml
    names = order or sorted(d for d in os.listdir(HERE)
                            if d.startswith("atlas_")
                            and os.path.isfile(os.path.join(HERE, d, "atlas_record.yaml")))
    out = []
    for n in names:
        path = os.path.join(HERE, n, "atlas_record.yaml")
        if not os.path.isfile(path):
            raise SystemExit(f"no record at {path}")
        out.append((n, yaml.safe_load(open(path))))
    return out


def baseline():
    """The frozen PROMOTED language -- the same 52 contracts every atlas is scored against.

    Read from whichever atlas has it; they are byte-identical by construction, and that is the
    point: two campaigns measured against different rulers cannot be compared.
    """
    for d in sorted(os.listdir(HERE)):
        p = os.path.join(HERE, d, "_state", "registry_baseline.json")
        if d.startswith("atlas_") and os.path.isfile(p):
            return set(json.load(open(p))["registered"])
    raise SystemExit("no registry_baseline.json under any atlas_* -- run registry_view.py --json")


def build(atlases, reg):
    seen = {}          # contract -> (repo, mechanism) that first introduced it
    rows, curve, cum = [], [], 0
    per_repo = {}
    for repo, doc in atlases:
        stats = {k: 0 for k in ("alias", "refinement", "new", "implementation",
                                "out_of_scope", "unclassified")}
        mechs = sorted(doc.get("mechanisms") or [], key=lambda m: m.get("order", 10 ** 6))
        for m in mechs:
            verdict = m.get("verdict")
            name = (m.get("contract") or {}).get("name")
            if verdict in (None, ""):
                cls = "unclassified"
            elif verdict == "out_of_scope":
                cls = "out_of_scope"
            elif verdict == "new":
                if name and name in reg:
                    cls = "alias"                     # the record is wrong; the baseline decides
                elif name and name in seen:
                    cls = "implementation"            # a second sighting, possibly in another repo
                else:
                    cls = "new"
                    if name:
                        seen[name] = (repo, m.get("id"))
            else:
                cls = verdict
            stats[cls] += 1
            if cls == "new":
                cum += 1
            first = seen.get(name)
            rows.append({"repo": repo, "id": m.get("id"), "raw_name": m.get("raw_name"),
                         "contract": name, "class": cls,
                         "first_seen_in": (first[0] if first else None),
                         "cross_repo_repeat": bool(first and first[0] != repo and
                                                   cls == "implementation")})
            curve.append({"n": len(rows), "repo": repo, "cum_new": cum})
        scored = sum(stats[k] for k in SCORED)
        per_repo[repo] = {**stats, "scored": scored, "mechanisms": len(mechs),
                          "new_here": sum(1 for r in rows
                                          if r["repo"] == repo and r["class"] == "new"),
                          "yield": (sum(1 for r in rows
                                        if r["repo"] == repo and r["class"] == "new") / scored)
                          if scored else None}
    return {"baseline_contracts": len(reg), "repos": [r for r, _ in atlases],
            "per_repo": per_repo, "new_contracts": sorted(seen),
            "first_seen": {k: {"repo": v[0], "mechanism": v[1]} for k, v in seen.items()},
            "cross_repo_repeats": [r for r in rows if r["cross_repo_repeat"]],
            "rows": rows, "curve": curve, "total_new": cum}


def render(cat):
    L = [f"CROSS-REPOSITORY CATALOG   ·   baseline {cat['baseline_contracts']} promoted contracts",
         ""]
    L.append(f"{'repository':<16}{'mech':>6}{'scored':>8}{'new':>6}{'impl':>6}{'alias':>7}"
             f"{'refine':>8}{'o-o-s':>7}{'yield':>8}")
    L.append("-" * 72)
    for repo in cat["repos"]:
        s = cat["per_repo"][repo]
        y = f"{s['yield']:.2f}" if s["yield"] is not None else "--"
        L.append(f"{repo:<16}{s['mechanisms']:>6}{s['scored']:>8}{s['new_here']:>6}"
                 f"{s['implementation']:>6}{s['alias']:>7}{s['refinement']:>8}"
                 f"{s['out_of_scope']:>7}{y:>8}")
    L.append("-" * 72)
    tot = sum(cat["per_repo"][r]["scored"] for r in cat["repos"])
    L.append(f"{'TOTAL':<16}{sum(cat['per_repo'][r]['mechanisms'] for r in cat['repos']):>6}"
             f"{tot:>8}{cat['total_new']:>6}")
    L.append("")
    L.append(f"distinct new contracts across all repositories: {cat['total_new']}")
    L.append(f"  {', '.join(cat['new_contracts'])}")
    L.append("")
    reps = cat["cross_repo_repeats"]
    L.append(f"CONTRACTS SIGHTED IN MORE THAN ONE REPOSITORY: {len(reps)}")
    if reps:
        for r in reps:
            L.append(f"  {r['repo']}/{r['id']:<28} -> {r['contract']} "
                     f"(first in {r['first_seen_in']})")
        L.append("")
        L.append("  Those are the measurement. A contract that only ever appears in the "
                 "repository that\n  introduced it is a description of that repository; one that "
                 "reappears independently is\n  evidence the vocabulary is real.")
    else:
        L.append("  none yet -- with one repository this is expected and says nothing")
    unc = sum(cat["per_repo"][r]["unclassified"] for r in cat["repos"])
    if unc:
        L.append("")
        L.append(f"  ⚠ {unc} mechanism(s) unclassified -- the curve is incomplete until they land")
    return "\n".join(L)


def plot(cat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BG = "black"
    curve = cat["curve"]
    xs = [c["n"] for c in curve]
    ys = [c["cum_new"] for c in curve]
    fig, ax = plt.subplots(figsize=(8.4, 5.2), facecolor=BG)
    ax.set_facecolor(BG)
    # the world in which every mechanism is new vocabulary -- the curve to fall away from
    ax.plot([0, xs[-1]], [0, xs[-1]], ls=":", lw=1.2, color="#777777",
            label="every mechanism new (no saturation)")
    colours = ["#4FA3FF", "#FF6B6B", "#FFD166", "#9C6ADE", "#5AD2A0"]
    start = 0
    for i, repo in enumerate(cat["repos"]):
        idx = [j for j, c in enumerate(curve) if c["repo"] == repo]
        ax.plot([xs[j] for j in idx], [ys[j] for j in idx], lw=2.6,
                color=colours[i % len(colours)], label=repo)
        if start:
            ax.axvline(start + 0.5, color="#444444", lw=1.0, ls="--")
        start = xs[idx[-1]]
    ax.set_xlabel("mechanisms inspected, in the order they were done", color="white", fontsize=10)
    ax.set_ylabel("cumulative NEW contracts", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=9)
    for s in ax.spines.values():
        s.set_color("#444444")
    leg = ax.legend(loc="upper left", fontsize=9, facecolor=BG, edgecolor="#444444")
    for t in leg.get_texts():
        t.set_color("white")
    ax.text(0.98, 0.04, "flattening = the language saturating", transform=ax.transAxes,
            color="#AAAAAA", fontsize=9, ha="right", va="bottom")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, facecolor=BG)
    plt.close(fig)
    return OUT_PNG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", nargs="*", default=None,
                    help="repositories in the order they were done (default: directory order)")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    cat = build(load_atlases(a.order), baseline())
    print(render(cat))
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(cat, f, indent=1)
    print(f"\n-> {os.path.relpath(OUT_JSON, HERE)}")
    if a.plot:
        print(f"-> {os.path.relpath(plot(cat), HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
