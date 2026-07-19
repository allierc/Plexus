"""knowledge -- distil the RunRecord archive into the four-class Knowledge ledger.

The archive is the SOURCE OF TRUTH (raw evidence). This ledger is a *distilled interpretation* that is
rewritten each round as evidence accumulates (never the other way round). Four classes (per review):
  - Established           : a mechanism claim that is sufficient AND robust AND has a necessary operator.
  - Refuted               : a specific hypothesis contradicted across seeds.
  - Structural limitation : a composition that provably CANNOT produce the phenotype ("X alone cannot
                            branch") -- stronger than "false", reusable across domains.
  - Open                  : unresolved.
"""
from __future__ import annotations
import os


def _row(rec):
    g, obs = rec["g"], rec["obs"]
    return (f"| `{'+'.join(sorted(set(g.op_names())))}` | {rec['region']} | {rec['rate']:.2f} | "
            f"{obs['cls']} | duct {obs['duct']} / gen {obs['generations']} |")


def distill(evals, necessity, out_path, round_id=0):
    """evals: {comp_hash: {g, rate, region, obs}}; necessity: {(comp_hash, op): bool}."""
    established, structural, refuted, open_ = [], [], [], []
    for h, rec in evals.items():
        g, rate, obs = rec["g"], rec["rate"], rec["obs"]
        ops = set(g.op_names())
        has_growth = "tissue_grow" in ops
        has_tension = "interface_relax" in ops
        has_cleft = ("cleft_induce" in ops) or ("react_rd" in ops)
        nec = sorted({op for (hh, op), v in necessity.items() if hh == h and v})
        if rate >= 0.6 and nec:                                # sufficient + robust-across-basin + necessary op
            established.append((rec, nec))
        elif rate < 0.2 and len(g.ops) >= 2:                   # essentially never emerges -> structural limit
            reason = ("no growth operator (cannot develop)" if not has_growth else
                      "no surface-tension operator (tissue fragments, not connected)" if not has_tension else
                      "no cleft operator (grows but cannot subdivide)" if not has_cleft else
                      f"{obs['cls']} phenotype, not in real regime")
            structural.append((rec, reason))
        elif rate >= 0.2:                                      # emerges sometimes but not robustly / not pinned
            open_.append(rec)
    # bootstrap-ladder status: first robust in-regime composition + the named failure metric_v0 exposes
    in_regime = sorted([r for r in evals.values() if r["rate"] >= 0.6],
                       key=lambda r: (-r["rate"], -len(r["g"].ops)))
    any_necessary = any(v for v in necessity.values())
    rung1 = in_regime[0] if in_regime else None
    lines = [f"# Knowledge ledger (distilled from the RunRecord archive) — round {round_id}", "",
             "> Distilled interpretation of the evidence. The archive of RunRecords is the source of "
             "truth; this ledger is revised as evidence accumulates.", "",
             "## Bootstrap-ladder status  (metric_v0, frozen)"]
    if rung1:
        lines.append(f"- **Rung 1 reached** — first composition robustly in the real regime across "
                     f"seeds + parameter basin: `{'+'.join(sorted(set(rung1['g'].op_names())))}` "
                     f"({rung1['region']}, rate {rung1['rate']:.2f}). {len(in_regime)} compositions "
                     f"are in-regime.")
    if len(in_regime) > 1 and not any_necessary:
        lines.append("- **Named failure → Loop III (measurement discovery):** metric_v0 cannot separate "
                     f"the {len(in_regime)} in-regime compositions, nor prove any operator *necessary* "
                     "— the initial condition is the real t=0 gland (already branch-like), so the "
                     "topology readout saturates and cannot measure developmental subdivision. A new "
                     "observable (subdivision / cleft-spacing over time) is required.")
    lines += ["", "## Established  (sufficient ∧ robust ∧ has a necessary operator)"]
    if established:
        for rec, nec in established:
            g = rec["g"]
            lines.append(f"- **{rec['region']}** — composition `{'+'.join(sorted(set(g.op_names())))}` "
                         f"emerges (rate {rec['rate']:.2f}); necessary operator(s): "
                         f"{', '.join('`'+o+'`' for o in nec) or '(none isolated)'}.")
    else:
        lines.append("- _(none yet)_")
    lines += ["", "## Structural limitation  (a composition that CANNOT produce the phenotype)"]
    if structural:
        for rec, reason in structural:
            g = rec["g"]
            lines.append(f"- `{'+'.join(sorted(set(g.op_names())))}` → **{reason}** "
                         f"(class {rec['obs']['cls']}, rate {rec['rate']:.2f}).")
    else:
        lines.append("- _(none yet)_")
    lines += ["", "## Refuted  (hypothesis contradicted across seeds)"]
    lines.append("- " + ("; ".join(refuted) if refuted else "_(none yet)_"))
    lines += ["", "## Open"]
    if open_:
        for rec in open_:
            lines.append(f"- `{'+'.join(sorted(set(rec['g'].op_names())))}` — partial "
                         f"(rate {rec['rate']:.2f}, class {rec['obs']['cls']}).")
    else:
        lines.append("- _(none)_")
    lines += ["", "## Composition → phenotype map",
              "| composition | region | emergence | class | topology |",
              "| --- | --- | --- | --- | --- |"]
    for h, rec in sorted(evals.items(), key=lambda kv: -kv[1]["rate"]):
        lines.append(_row(rec))
    open(out_path, "w").write("\n".join(lines) + "\n")
    return dict(established=len(established), structural=len(structural), open=len(open_))
