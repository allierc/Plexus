#!/usr/bin/env python
"""Generate `paper/mpm_warp_code.tex`: the `default` and `warp` source of each MPM operator, side by
side, EXTRACTED FROM THE FILES rather than transcribed.

WHY GENERATED. A code appendix typed by hand is wrong the first time anyone edits the operator, and
wrong silently -- the reader has no way to tell. This pulls the exact line spans out of
`src/plexus/operators/mpm_ops.py` and `mpm_warp.py` at build time and stamps the line numbers it
took, so a claim in the note can always be checked against the file.

TWO OF THE FOUR OPERATORS HAVE NO `warp` VERSION. `mpm_strain` and `mpm_grid_update` still run the
`default` bodies, and that is stated on the page rather than left for the reader to notice from an
empty column -- an appendix that quietly shows one side is how "not yet written" turns into
"apparently equivalent".

    python tools/mpm_code_appendix.py && (cd paper && pdflatex mpm_warp_code.tex)
"""
from __future__ import annotations

import argparse
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OPS = os.path.join(ROOT, "src", "plexus", "operators", "mpm_ops.py")
WARP = os.path.join(ROOT, "src", "plexus", "operators", "mpm_warp.py")


def extract(path, qualname):
    """Source of `Class.method`, `func` or `class Name`, with its 1-based line span.

    The end of a block is the next line at the SAME OR LOWER indent that starts a def/class/
    decorator -- which is how Python itself delimits it, so this does not need to parse.
    """
    src = open(path).read().split("\n")
    parts = qualname.split(".")
    lo, hi, ind = 0, len(src), -1
    for depth, nm in enumerate(parts):
        pat = re.compile(r"^(\s*)(?:class|def)\s+" + re.escape(nm) + r"\b")
        for i in range(lo, hi):
            m = pat.match(src[i])
            if not m or (ind >= 0 and len(m.group(1)) <= ind):
                continue
            ind = len(m.group(1))
            j = i + 1
            while j < hi:
                s = src[j]
                if s.strip() and (len(s) - len(s.lstrip())) <= ind and \
                        re.match(r"^\s*(class|def|@)", s):
                    break
                j += 1
            # a decorator line immediately above belongs to the block
            k = i
            while k > 0 and src[k - 1].strip().startswith("@"):
                k -= 1
            lo, hi = (k, j) if depth == len(parts) - 1 else (i + 1, j)
            break
        else:
            raise SystemExit(f"  {qualname}: '{nm}' not found in {os.path.basename(path)}")
    # TRIM THE NEXT OPERATOR'S HEADER. A `# ===...` banner introducing the following class sits at
    # indent 0 and so is not a def/class line, which means the scan above walks straight past it and
    # the listing ends with three lines belonging to something else. Blank and comment-only trailing
    # lines are dropped together.
    while hi > lo and (not src[hi - 1].strip() or src[hi - 1].lstrip().startswith("#")):
        hi -= 1
    body = src[lo:hi]
    trim = min((len(l) - len(l.lstrip()) for l in body if l.strip()), default=0)
    return "\n".join(l[trim:] if l.strip() else "" for l in body), lo + 1, hi


def block(title, path, qualname, note=None):
    code, a, b = extract(path, qualname)
    hdr = f"{os.path.basename(path)}:{a}--{b}"
    L = [r"\subsection*{" + title.replace("_", r"\_")
         + r"\;{\footnotesize\color{grey}\normalfont\texttt{"
         + hdr.replace("_", r"\_") + r"}}}"]
    if note:
        L.append(r"\\[1pt]{\footnotesize " + note + r"}")
    L += [r"\vspace{2pt}", r"\begin{lstlisting}", code, r"\end{lstlisting}"]
    return "\n".join(L), b - a + 1


HEAD = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=1.5cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{titlesec}
\usepackage{booktabs}
\usepackage[hidelinks]{hyperref}
\definecolor{grey}{HTML}{666666}
\definecolor{good}{HTML}{1B7F3B}
\definecolor{bad}{HTML}{B3261E}
\definecolor{lst}{HTML}{F6F6F6}
\newcommand{\code}[1]{\texttt{\small #1}}
\newcommand{\gd}[1]{\textcolor{good}{\textbf{#1}}}
\newcommand{\bd}[1]{\textcolor{bad}{\textbf{#1}}}
\titleformat{\section}{\large\bfseries}{\thesection.}{0.6em}{}
\titleformat{\subsection}{\normalsize\bfseries}{}{0em}{}
\lstset{basicstyle=\ttfamily\scriptsize, backgroundcolor=\color{lst}, frame=none,
        columns=fullflexible, breaklines=true, showstringspaces=false, language=Python,
        keywordstyle=\color{black}, commentstyle=\color{grey}\itshape,
        aboveskip=2pt, belowskip=2pt, xleftmargin=2pt}
\setlength{\parindent}{0pt}
\title{\vspace{-1.4cm}\textbf{The four MPM operators, \code{default} against \code{warp}}\\[2pt]
\large every line of both, extracted from the source at build time}
\author{}\date{}
\begin{document}\maketitle\vspace{-1.0cm}
"""

INTRO = r"""
\section{What has a \code{warp} twin and what does not}

\begin{center}\small
\begin{tabular}{@{}llll@{}}
\toprule
operator & \code{default} & \code{warp} & per particle per substep, \code{default} allocates \\
\midrule
\code{mpm\_strain}      & \code{mpm\_ops.py} & \bd{none} --- still \code{default} & 313\,B \\
\code{mpm\_scatter}     & \code{mpm\_ops.py} & \gd{\code{mpm\_warp.py}} \code{p2g\_atomic} & 7594\,B $\to$ \gd{0} \\
\code{mpm\_grid\_update}& \code{mpm\_ops.py} & \bd{none} --- still \code{default} & 151\,B \\
\code{mpm\_gather}      & \code{mpm\_ops.py} & \gd{\code{mpm\_warp.py}} \code{g2p} & 6413\,B $\to$ \gd{0} \\
\bottomrule
\end{tabular}
\end{center}

\vspace{2pt}
Two of the four are \bd{not yet ported}, and they are now the whole remaining frame: with the scatter
and the gather fused, \code{mpm\_strain} and \code{mpm\_grid\_update} are the only operators still
writing intermediates to global memory. The empty column below is a to-do, not a claim that the
\code{default} body is already optimal. Byte figures are from the \code{TorchDispatchMode} trace in
\code{paper/mpm\_warp.pdf} \S3.1, at $N=945\,000$.

\vspace{4pt}
The \code{warp} side of each pair is two pieces: the \code{@wp.kernel} that runs on the device, and
the operator's \code{forward}, which does nothing but marshal torch tensors into it. They are shown
in that order. \code{wp.from\_torch} wraps the SAME device memory rather than copying, so the grid
the kernel writes is the field the rest of the substep reads.
"""


def stacked(left, right):
    """The two versions one after the other, full width, free to break across pages.

    NOT side-by-side minipages, which was the first attempt: a minipage is an unbreakable box, and
    `mpm_grid_update`'s default body is 143 lines -- it overflowed the page by 492pt, which LaTeX
    reports as a warning and then CLIPS. An appendix whose whole purpose is to show every line
    cannot use a container that silently drops some.
    """
    return left + "\n\\vspace{8pt}\n\n" + right + "\n"


NONE = (r"\subsection*{no \code{warp} implementation}"
        r"This operator still runs the \code{default} body under \code{warp}. "
        r"%s\vspace{4pt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "mpm_warp_code.tex"))
    a = ap.parse_args()

    secs = [HEAD, INTRO]
    counts = []

    def sec(title, blurb):
        secs.append("\n\\clearpage\n\\section{%s}\n\n%s\n\\vspace{4pt}\n" % (title, blurb))

    # --- 1. strain -------------------------------------------------------------------------
    sec(r"\code{mpm\_strain} --- the deformation gradient update",
        r"$\mathbf{F}_p \leftarrow (\mathbf{I}+\Delta t\,\mathbf{C}_p)\mathbf{F}_p$, plus the "
        r"plastic and liquid branches. A $[N,3,3]$ matrix product: it is the operator that fuses "
        r"most easily and it has not been done.")
    l, n1 = block("default", OPS, "MPMStrain.forward")
    secs.append(stacked(l, NONE % (r"It allocates 313\,B per particle per substep --- "
                                   r"\code{mul} and \code{where} on $[N,3,3]$, 68\,MB each at "
                                   r"945\,000 particles.")))
    counts.append(("mpm_strain", n1, 0))

    # --- 2. scatter ------------------------------------------------------------------------
    sec(r"\code{mpm\_scatter} --- particle $\to$ grid (P2G)",
        r"Mass and momentum scattered to the 27 surrounding nodes, with the fixed-corotated "
        r"stress folded into the affine term. \bd{This is the one with the atomics}: many "
        r"particles land on the same node, so the writes must be serialised. 108 "
        r"\code{wp.atomic\_add} per particle (27 nodes $\times$ (1 mass + 3 momentum)).")
    l, n2 = block("default", OPS, "MPMScatter.forward")
    r1, n3 = block("warp --- the device kernel", WARP, "p2g_atomic")
    r2, n4 = block("warp --- the launch", WARP, "MPMScatterWarp.forward")
    secs.append(stacked(l, r1 + "\n\\vspace{6pt}\n" + r2))
    counts.append(("mpm_scatter", n2, n3 + n4))

    # --- 3. grid_update --------------------------------------------------------------------
    sec(r"\code{mpm\_grid\_update} --- the grid solve",
        r"$\mathbf{v}_i = (m\mathbf{v})_i/\max(m_i,\epsilon)$, then every boundary condition the "
        r"model owns: gravity, walls, moving plates, CSF surface tension, buoyancy. The longest "
        r"of the four and the least like the others --- it is $O(\text{cells})$, not "
        r"$O(\text{particles})$, so a port is a different kernel shape.")
    l, n5 = block("default", OPS, "MPMGridUpdate.forward")
    secs.append(stacked(l, NONE % (r"It allocates 151\,B per particle per substep. Note that "
                                   r"CUDA-graph capture already removes most of what this costs, "
                                   r"by holding its intermediates in a resident private pool "
                                   r"instead of re-allocating them 7 times a frame "
                                   r"(\code{mpm\_warp.pdf} \S5.3).")))
    counts.append(("mpm_grid_update", n5, 0))

    # --- 4. gather -------------------------------------------------------------------------
    sec(r"\code{mpm\_gather} --- grid $\to$ particle (G2P)",
        r"Velocity and the affine matrix $\mathbf{C}$ read back from the same 27 nodes, then the "
        r"advection. \gd{A pure read}: no two threads write the same place, so there is nothing "
        r"to coordinate --- no atomics, no sort, no ownership question. That is why it was ported "
        r"first.")
    l, n6 = block("default", OPS, "MPMGather.forward")
    r1, n7 = block("warp --- the device kernel", WARP, "g2p")
    r2, n8 = block("warp --- the launch", WARP, "MPMGatherWarp.forward")
    secs.append(stacked(l, r1 + "\n\\vspace{6pt}\n" + r2))
    counts.append(("mpm_gather", n6, n7 + n8))

    secs.append("\n\\end{document}\n")
    with open(a.out, "w") as f:
        f.write("% GENERATED by tools/mpm_code_appendix.py -- do not edit\n" + "\n".join(secs))
    print(f"  -> {a.out}")
    for nm, d, w in counts:
        print(f"    {nm:<18} default {d:>4} lines   warp {w if w else '--':>4}")


if __name__ == "__main__":
    main()
