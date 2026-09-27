#!/usr/bin/env python
"""The motor's entities containment as a TABULAR HIERARCHY, written to LaTeX from the spec.

    python tools/bfm_containment_tex.py config/bacterium/bfm_e01t_cj_membrane_avg.yaml \
           [config/bacterium/bfm_e02b_cj_scaffold_grid.yaml] -o builder/exp_02_bacterium/notes/containment.tex

One row per set, indented by its depth in the containment map, the tree drawn with VERTICAL and
HORIZONTAL rules on the grid of the text rows (the human, 2026-09-24: "a tabular hierarchy with
vertical horizontal lines neat on a grid aligned with text"), and beside the tree the columns a
reader asks for: how many, what state each element carries (block, width, integration, unit), and
which operators act on the set. Nothing is typed by hand: the tree, the counts and the blocks are
read from the spec, so the figure cannot drift from the run it describes. Plain TikZ, no tree
package -- every coordinate is on the grid, and Kile compiles it with pdflatex alone.
"""
from __future__ import annotations

import os
import sys

import yaml

LH = 0.37           # cm per text line
GAP = 0.10          # cm between rows
IND = 0.55          # cm per containment level
X_TREE, X_COUNT, X_STATE, X_OPS, X_END = 0.0, 6.35, 6.6, 13.4, 18.5   # column origins, cm (portrait A4)
W_STATE, W_OPS = 6.5, 5.1                                            # text widths of the two prose columns
CPL_STATE, CPL_OPS = 38, 32                                          # characters per line, footnotesize


def esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("#", r"\#").replace("%", r"\%").replace("&", r"\&")


def load(path):
    return yaml.safe_load(open(path))


def tree(spec):
    """(name, depth, count, blocks, parent, has_state, has_children) rows in tree order."""
    sets = spec["sets"]
    kids = {}
    for nm, st in sets.items():
        kids.setdefault(st.get("parent"), []).append(nm)
    rows = []

    def count(nm):
        st = sets[nm]
        if st.get("per_parent"):
            par = st.get("parent")
            return int(st["per_parent"]) * (count(par) if par else 1)
        return int(st.get("n") or len(st.get("start") or []) or 0)

    def walk(nm, depth):
        st = sets[nm]
        blocks = [(b, int(v.get("width", 1)), str(v.get("integration", "")), v.get("unit"))
                  for b, v in (st.get("state") or {}).items() if b not in ("pos", "vel", "copy")]
        _st = st.get("state") or {}
        has_state = bool(_st) and ("pos" in _st or "vel" in _st or "copy" in _st)   # a material set
        rows.append((nm, depth, count(nm), blocks, st.get("parent"), has_state, bool(kids.get(nm))))
        for k in kids.get(nm, []):
            walk(k, depth + 1)

    for root in kids.get(None, []):
        walk(root, 0)
    return rows


def ops_by_set(spec):
    """set -> [(op, role)] : the operators that read or write the set, with the role the spec gives it."""
    out = {}
    for o in spec.get("operators", []):
        op = o["op"]
        out.setdefault(o.get("at"), []).append((op, "at"))
        for key in ("stator", "anchor", "to"):
            if o.get(key):
                out.setdefault(o[key], []).append((op, key))
    return out


def _lines(text_plain: str, cpl: int) -> int:
    return max(1, -(-len(text_plain) // cpl))


def state_text(blocks, has_state, has_children, count):
    """(tex, plain) for the state column."""
    if blocks:
        tex = ", ".join(f"\\texttt{{{esc(b)}}}~{w}~{esc(integ) if integ else '-'}" + (f"~{esc(u)}" if u else "")
                        for b, w, integ, u in blocks)
        plain = ", ".join(f"{b} {w} {integ or '-'}" + (f" {u}" if u else "") for b, w, integ, u in blocks)
        return tex, plain
    if has_state:
        t = "position, velocity, \\texttt{copy} (symmetry-copy index); moved by MPM"
        return t, "position, velocity, copy (symmetry-copy index); moved by MPM"
    if has_children:
        return r"\textit{no state: a container}", "no state: a container"
    return r"\textit{no state: a static cloud, drawn only}", "no state: a static cloud, drawn only"


def ops_text(acting):
    seen, parts, plain = set(), [], []
    for op, role in acting:
        if (op, role) in seen:
            continue
        seen.add((op, role))
        parts.append(f"\\texttt{{{esc(op)}}}" + ("" if role == "at" else f"~({esc(role)})"))
        plain.append(op + ("" if role == "at" else f" ({role})"))
    return ", ".join(parts), ", ".join(plain)


def render(spec, title):
    rows = tree(spec)
    ops = ops_by_set(spec)
    fields = spec.get("fields") or {}
    out = []
    out.append(r"\begin{tikzpicture}[x=1cm,y=1cm,font=\footnotesize,line cap=round]")
    out.append(r"\tikzset{hd/.style={font=\footnotesize\bfseries,anchor=west,inner sep=0pt},")
    out.append(r"        tx/.style={anchor=west,inner sep=0pt}, num/.style={anchor=east,inner sep=0pt},")
    out.append(r"        mono/.style={anchor=west,inner sep=0pt,font=\footnotesize\ttfamily},")
    out.append(r"        cellt/.style={anchor=north west,inner sep=0pt,font=\footnotesize,text=black!65,align=left}}")
    # header
    y = 0.0
    out.append(rf"\node[hd] at ({X_TREE},{y}) {{set}};")
    out.append(rf"\node[hd,anchor=east] at ({X_COUNT},{y}) {{elements}};")
    out.append(rf"\node[hd] at ({X_STATE},{y}) {{state per element}};")
    out.append(rf"\node[hd] at ({X_OPS},{y}) {{operators acting}};")
    y -= 0.6 * LH
    out.append(rf"\draw[black!70,thin] ({X_TREE},{y}) -- ({X_END},{y});")
    y -= GAP
    ytop = {}
    for i, (nm, depth, cnt, blocks, parent, has_state, has_kids) in enumerate(rows):
        st_tex, st_plain = state_text(blocks, has_state, has_kids, cnt)
        op_tex, op_plain = ops_text(ops.get(nm, []))
        nl = max(_lines(st_plain, CPL_STATE), _lines(op_plain, CPL_OPS) if op_plain else 1)
        y_first = y - 0.5 * LH                      # baseline-centre of the first line
        ytop[nm] = y_first
        x = X_TREE + depth * IND
        out.append(rf"\node[mono] at ({x + 0.18},{y_first}) {{{esc(nm)}}};")
        out.append(rf"\node[num] at ({X_COUNT},{y_first}) {{{cnt:,}}};")
        out.append(rf"\node[cellt,text width={W_STATE}cm] at ({X_STATE},{y}) {{{st_tex}}};")
        if op_tex:
            out.append(rf"\node[cellt,text width={W_OPS}cm] at ({X_OPS},{y}) {{{op_tex}}};")
        if parent is not None:
            px = X_TREE + (depth - 1) * IND + 0.08
            out.append(rf"\draw[black!75,semithick] ({px},{ytop[parent] - 0.55 * LH}) -- ({px},{y_first}) -- ({x + 0.10},{y_first});")
        y -= nl * LH + GAP
    if fields:
        y -= 0.5 * LH
        out.append(rf"\draw[black!70,thin] ({X_TREE},{y}) -- ({X_END},{y});")
        y -= GAP + 0.5 * LH
        out.append(rf"\node[hd] at ({X_TREE},{y}) {{field}};")
        out.append(rf"\node[hd,anchor=east] at ({X_COUNT},{y}) {{nodes}};")
        out.append(rf"\node[hd] at ({X_STATE},{y}) {{what it holds}};")
        out.append(rf"\node[hd] at ({X_OPS},{y}) {{bound by}};")
        y -= 0.5 * LH + GAP
        for fn, fd in fields.items():
            ng = int(fd.get("n_grid", 0))
            bound = sorted({o.get("at") for o in spec.get("operators", []) if o.get("to") == fn})
            hold_tex = "mass and momentum of the bodies scattered to it; one grid = the bodies that can touch each other"
            bt = "\\texttt{mpm\\_scatter} in, \\texttt{mpm\\_gather} out, from " + ", ".join("\\texttt{" + esc(b) + "}" for b in bound)
            bp = "mpm_scatter in, mpm_gather out, from " + ", ".join(bound)
            nl = max(_lines(hold_tex, CPL_STATE), _lines(bp, CPL_OPS))
            y_first = y - 0.5 * LH
            out.append(rf"\node[mono] at ({X_TREE + 0.18},{y_first}) {{{esc(fn)}}};")
            out.append(rf"\node[num] at ({X_COUNT},{y_first}) {{${ng}^3$}};")
            out.append(rf"\node[cellt,text width={W_STATE}cm] at ({X_STATE},{y}) {{{hold_tex}}};")
            out.append(rf"\node[cellt,text width={W_OPS}cm] at ({X_OPS},{y}) {{{bt}}};")
            y -= nl * LH + GAP
    out.append(r"\end{tikzpicture}")
    return "\n".join(out), rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    out_path = next((sys.argv[i + 1] for i, t in enumerate(sys.argv) if t == "-o"), None)
    args = [a for a in args if a != out_path]
    if not args:
        raise SystemExit("give one or two spec paths")
    specs = [(p, load(p)) for p in args]
    figs = []
    for p, spec in specs:
        body, rows = render(spec, spec["general"]["name"] if "general" in spec else os.path.basename(p))
        figs.append((p, spec, body, rows))
    name = lambda sp: (sp.get("general") or {}).get("name") or "?"
    doc = [r"""\documentclass[a4paper,10pt]{article}
\usepackage[margin=12mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{tikz}
\usepackage{caption}
\hyphenpenalty=10000 \exhyphenpenalty=10000
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\begin{document}
\section*{The flagellar motor in Plexus: what contains what}
""" + f"""Every set below is one line of the run's specification. A set holds elements; each element carries the
state blocks listed beside it; a set inside another is drawn under it, joined by the rules of the tree,
and the containment map is that tree. An operator acts on the set named after \\texttt{{at}}; the roles in
brackets say when it reads a second set (the stators for the contact, the anchor cloud, the grid a body is
scattered to). Counts are per run: the seven rotor parts and the six scaffold proteins are one element each
holding their material points. \\emph{{Integration}} says how a block moves: \\texttt{{first\\_order}} is integrated each frame from
the rate its operator returns, \\texttt{{none}} is written outright by an operator. Generated by
\\texttt{{tools/bfm\\_containment\\_tex.py}} from
{', '.join('\\texttt{' + esc(os.path.relpath(p, os.getcwd())) + '}' for p, _ in specs)}.
"""]
    for p, spec, body, rows in figs:
        nsets = len(rows)
        doc.append(f"\\subsection*{{\\texttt{{{esc(name(spec))}}}}}")
        doc.append(body)
        doc.append(f"\n{nsets} sets, {sum(1 for r in rows if r[4] is None)} of them roots; the cell is the only root whose children are not its material points.\n")
        doc.append(r"\clearpage")
    doc.append(r"\end{document}")
    text = "\n".join(doc)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        open(out_path, "w").write(text); print(f"wrote {out_path}  ({sum(len(f[3]) for f in figs)} set rows)")
    else:
        print(text)


if __name__ == "__main__":
    main()
