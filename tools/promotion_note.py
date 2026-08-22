#!/usr/bin/env python
"""Generate the promotion note's tables from the run record, then build the PDF.

WHY GENERATED. `paper/promotion_note.tex` carries two tables that a human must not type: the
twin-run gate (one row per spec, with two sha1s) and the operator inventory (105 names with their
family, kind, set and implementations). Both are facts on disk -- `log/promotion/promotion_identical.json`
and the live registry -- and a note whose numbers are transcribed is a note that disagrees with the
repository the first time either changes. So the tables are written here, into
`paper/promotion_tables.tex`, and the prose \\input{}s them.

    python tools/promotion_note.py            write the tables and build the PDF
    python tools/promotion_note.py --tables   tables only, no latex
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "promotion_tables.tex")


def _esc(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def twin_table():
    """The twin-run gate, one row per spec, from the record it actually wrote."""
    p = os.path.join(ROOT, "log", "promotion", "promotion_identical.json")
    if not os.path.exists(p):
        return "% no log/promotion/promotion_identical.json yet\n"
    rows = json.load(open(p))
    # THE FRAME COUNT IS READ BACK FROM THE SPEC THE RUN USED, not from the PAIRS table, because a
    # row can be re-run at a different length and the note must describe the run that happened.
    import yaml
    order = {"0": 0, "0.5": 1, "B": 2, "B-core": 3, "C": 4, "D": 5, "R": 9}
    rows.sort(key=lambda r: (order.get(r["phase"], 8), r["spec"]))
    L = [r"\begin{center}\small", r"\begin{tabular}{@{}llrllll@{}}\toprule",
         r"phase & spec & frames & A & B & digest & outcome\\\midrule"]
    for r in rows:
        n = ""
        for tag in ("A", "B"):
            f = os.path.join(ROOT, "config", "okuda",
                             f"promo_{r['phase'].replace('.', 'p').replace('-', '_')}_{r['spec']}_{tag}.yaml")
            if os.path.exists(f):
                try:
                    n = str(yaml.safe_load(open(f))["general"]["n_frames"]); break
                except Exception:
                    pass
        ok = r["ok"]
        mark = (r"\textcolor{good}{\textbf{identical}}" if ok
                else r"\textcolor{bad}{\textbf{" + _esc(r["why"][:38]) + r"}}")
        same = r["digest_a"] == r["digest_b"]
        dig = r"\code{" + r["digest_a"] + r"}" if same else r"\code{" + r["digest_a"] + r"} vs \code{" + r["digest_b"] + "}"
        L.append(f"{_esc(r['phase'])} & \\code{{{_esc(r['spec'])}}} & {n} & "
                 f"{_esc(r['a'])} & {_esc(r['b'])} & {dig} & {mark}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    n_ok = sum(1 for r in rows if r["ok"])
    L.append(f"\n\\noindent\\textbf{{{n_ok} of {len(rows)} rows identical.}}\n")
    return "\n".join(L) + "\n"


def operator_table():
    """Every registered operator, where it lives now, and what it declares."""
    sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "discovery_okuda", "ops"),
                    os.path.join(ROOT, "discovery_okuda")]
    import plexus.operators                                            # noqa: F401
    for m in ("mesh_ops", "chem_ops", "t1_ops", "monolayer_ops", "shape_chem_ops",
              "shape_probe_ops", "ecm_ops", "membrane_ops", "integrin_ops", "load_ops",
              "mesh_contact_ops", "bm_sense_ops", "plate_ops", "surface_ops", "block_ops",
              "junction_ops", "medioapical_ops"):
        __import__(m)
    from plexus.models.registry import _OPERATOR_REGISTRY as REG, _OP_CONTRACTS as CON
    rows = []
    for n in sorted(REG):
        c = REG[n]
        if getattr(c, "REGISTERED_NAMES", [n])[0] != n:
            continue                                                   # canonical row only
        f = inspect.getsourcefile(c) or "?"
        where = ("core" if "/src/plexus/operators/" in f else
                 "core (models)" if "/src/plexus/" in f else "okuda")
        impls = sorted(CON[n].implementations)
        rows.append((where, os.path.basename(f), n, getattr(c, "FAMILY", None),
                     getattr(c, "KIND", None), getattr(c, "SET", None), impls,
                     getattr(c, "REGISTERED_NAMES", [n])[1:]))
    L = [r"\begin{center}\scriptsize", r"\begin{longtable}{@{}lllllp{3.1cm}@{}}\toprule",
         r"module & operator & family & kind & set & implementations / alias\\\midrule\endhead"]
    for where, fn, n, fam, kind, st, impls, alias in sorted(rows, key=lambda r: (r[1], r[2])):
        extra = ", ".join(impls) if impls != ["default"] else ""
        if alias:
            extra += (("; " if extra else "") + "alias " + ", ".join(alias))
        tag = r"\textcolor{bad}{\textbf{okuda}}~" if where == "okuda" else ""
        L.append(f"{tag}\\code{{{_esc(fn)}}} & \\code{{{_esc(n)}}} & {_esc(fam)} & {_esc(kind)} & "
                 f"{_esc(st)} & {_esc(extra)}\\\\")
    L += [r"\bottomrule", r"\end{longtable}", r"\end{center}"]
    n_core = sum(1 for r in rows if r[0].startswith("core"))
    L.append(f"\n\\noindent\\textbf{{{n_core} of {len(rows)} canonical operator names resolve into "
             f"\\code{{src/plexus/}}.}} The remainder are the two \\code{{AUDIT.md}} rejects, "
             f"registered in \\code{{discovery\\_okuda}} only.\n")
    return "\n".join(L) + "\n"


def gate_table():
    """The lifted gates, from `log/gates/gate_table.json` -- the roll-up `run_gates.py` writes.

    THE TABLE CARRIES `basis` AS WELL AS `tier`, because the paper's third tier asks whether the
    model agrees with something OBSERVED IN CELLS and a threshold copied off a previous run of the
    same code is not that. Reporting the two together is the only way the fraction at the bottom
    means anything.
    """
    p = os.path.join(ROOT, "log", "gates", "gate_table.json")
    if not os.path.exists(p):
        return ("\\noindent\\textit{No gate has been run from the promoted registry yet; this "
                "table is written by \\code{tools/run\\_gates.py}.}\n")
    doc = json.load(open(p))
    MARK = {"PASS": r"\textcolor{good}{pass}", "FAIL": r"\textcolor{bad}{\textbf{FAIL}}",
            "BLOCKED": r"\textcolor{grey}{blocked}", "KNOWN_RED": r"\textcolor{warn}{known-red}",
            "TURNED_GREEN": r"\textcolor{warn}{\textbf{turned green}}",
            "INFRA_FAIL": r"\textcolor{bad}{\textbf{infra}}"}
    L = [r"\begin{center}\scriptsize", r"\begin{longtable}{@{}llllrrl@{}}\toprule",
         r"gate & row & tier & basis & declared & measured & \\\midrule\endhead"]
    for t in doc.get("tables", []):
        for m in t["measures"]:
            k, a = next(iter(m["assertion"].items()))
            v = m.get("value")
            vs = ("--" if v is None else
                  (json.dumps(v, default=str)[:30] if isinstance(v, dict)
                   else (f"{v:.6g}" if isinstance(v, float) else str(v))))
            L.append(f"\\code{{{_esc(t['gate'])}}} & \\code{{{_esc(m['name'])}}} & "
                     f"{_esc(m['tier'].replace('_', ' '))} & {_esc(m['basis'])} & "
                     f"{_esc(k)} {_esc(a)} & {_esc(vs)} & {MARK.get(m['outcome'], m['outcome'])}\\\\")
    L += [r"\bottomrule", r"\end{longtable}", r"\end{center}"]
    s = doc.get("split", {})
    if s:
        L.append(
            f"\n\\noindent\\textbf{{{s['verification']} of {s['rows']} rows are verification}} "
            f"(bookkeeping or closed form, {100 * s['verification_frac']:.0f}\\%); "
            f"{s['regression_pins']} pin a regression against a previous run of this same model; "
            f"and \\textbf{{{s['observed_in_cells']} compare the model with something observed in "
            f"cells}} ({100 * s['observed_frac']:.0f}\\%). The paper reports roughly two thirds "
            f"bookkeeping-or-closed-form for the \\code{{ecm}} study; this table looks better than "
            f"that only because \\code{{basis}} separates the regression pins out of the "
            f"measurement tier, where they would otherwise sit and read as biology.\n")
        L.append(f"\n\\noindent Outcomes: {_esc(doc.get('outcomes', {}))}.\n")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", action="store_true", help="write the tables, skip latex")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("% GENERATED by tools/promotion_note.py -- do not edit; edit the tool.\n")
        f.write("\\newcommand{\\twintable}{%\n" + twin_table() + "}\n")
        f.write("\\newcommand{\\optable}{%\n" + operator_table() + "}\n")
        f.write("\\newcommand{\\gatetable}{%\n" + gate_table() + "}\n")
    print(f"  wrote {os.path.relpath(OUT, ROOT)}")
    if a.tables:
        return 0
    tex = os.path.join(ROOT, "paper", "promotion_note.tex")
    if not os.path.exists(tex):
        print("  paper/promotion_note.tex does not exist yet -- tables only")
        return 0
    for _ in range(2):                       # twice: longtable needs the second pass
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                            "promotion_note.tex"], cwd=os.path.join(ROOT, "paper"),
                           capture_output=True, text=True, timeout=300)
    if r.returncode:
        print(r.stdout[-2500:])
        return 1
    print(f"  wrote paper/promotion_note.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
