"""paper -- make the PDF readable, because the first excavator could not read it.

The first live agent call came back with a good entry and one honest complaint: *"the paper PDF
could not be rendered in this environment (no poppler/pdftotext), so the source-vs-paper check is
still open"*. It anchored `paper_section` to the library's own guides instead and said so.

That is exactly the right behaviour and exactly the wrong situation. The atlas's most valuable
output is a disagreement between what a paper claims and what its code does, and an agent that
cannot open the paper can never find one. PyMuPDF is installed in the Plexus environment, so the
fix is four lines: extract once, with page markers, and let every agent Read plain text.

Page markers matter. `paper_section: "p. 4"` is checkable; `paper_section: "the methods"` is not,
and the validator has no way to tell the difference.

    python paper.py                       # extract the target's paper
    python paper.py --pdf <path>          # any other paper in papers/
    python paper.py --grep division       # where does the paper talk about X?
"""
from __future__ import annotations

import argparse
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_PDF = os.path.join(PLEXUS, "papers", "Deshpande_2025_jax_morph.pdf")
OUT_DIR = os.path.join(HERE, "_state", "paper")


def extract(pdf=DEFAULT_PDF, out_dir=OUT_DIR):
    import fitz

    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(pdf))[0]
    out = os.path.join(out_dir, name + ".txt")
    doc = fitz.open(pdf)
    chunks = []
    for i, page in enumerate(doc, 1):
        chunks.append(f"\n\n===== PAGE {i} =====\n\n" + page.get_text())
    text = "".join(chunks)
    with open(out, "w") as f:
        f.write(text)
    words = len(text.split())
    print(f"{os.path.basename(pdf)}: {len(doc)} pages, {words} words -> {out}")
    if words < 200 * len(doc) / 10:
        print("  WARNING: very little text for this page count -- the PDF may be a scan, and "
              "an agent reading this file would be reading almost nothing. Check before "
              "citing it.")
    return out


def grep(term, out_dir=OUT_DIR, ctx=200):
    hits = 0
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".txt"):
            continue
        text = open(os.path.join(out_dir, fn), errors="replace").read()
        pages = [(m.start(), int(m.group(1))) for m in re.finditer(r"===== PAGE (\d+) =====", text)]
        for m in re.finditer(term, text, re.I):
            page = max((p for pos, p in pages if pos < m.start()), default=0)
            frag = " ".join(text[max(0, m.start() - ctx):m.start() + ctx].split())
            print(f"\n[{fn} p.{page}]  ...{frag}...")
            hits += 1
    print(f"\n{hits} hits for {term!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=DEFAULT_PDF)
    ap.add_argument("--grep")
    a = ap.parse_args()
    if a.grep:
        return grep(a.grep)
    extract(a.pdf)


if __name__ == "__main__":
    main()
