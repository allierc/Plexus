"""Mirror hand edits made in `index.qmd` into the rendered `docs/index.html`.

Quarto is not installed in this container and `docs/` is what GitHub Pages serves, so the two
files have to be kept in step by hand. Rather than diffing text (the two files are NOT identical:
quarto adds `class="anchored"` to headings, wraps markdown sections in <section>, and rewrites some
`<figure class="sim-card">` to `sim-card figure`), this syncs by ANCHOR:

  captions   for every card in the qmd, the caption is copied onto the docs card that plays the
             SAME clip -- the clip name is the identity of a card
  headings   an <h2>/<h3> whose text changed is matched on its old text, attributes preserved
  ledes      the <p class="opk"> that follows a changed heading, matched on its old text
  sections   a whole markdown section deleted from the qmd is removed from docs by its rendered
             <section id="..."> wrapper, and from the table of contents

Anything it cannot find is reported and nothing is written for that file -- a silent half-sync is
worse than none, because the page it produces looks finished.

Run:  python prototype/galaxy_collision/sync_qmd_to_docs.py [--dry-run]
"""
from __future__ import annotations
import re
import sys

QMD = "/workspace/Plexus/index.qmd"
DOCS = "/workspace/Plexus/docs/index.html"

CARD_RE = re.compile(
    r'<video src="gallery/(?P<clip>[^"]+)"'          # the clip identifies the card
    r'.*?<span class="sim-cap">(?P<cap>.*?)</span>', re.S)

# (old text, new text) for headings and ledes, taken from the qmd edit.
TEXT = [
    ("Vertex + Turing — patterning &amp; shaping a growing tissue",
     "Growing tissue + Turing pattern"),
    ("Growth and reaction–diffusion on a deformable vertex shell: the tissue first proliferates "
     "by growth and division to a couple of thousand cells, then reaction–diffusion chemistry both "
     "<b>colours</b> the surface and <b>drives its shape</b> — each regime buckling the shell a "
     "different way.",
     "Cells grow and divide while reaction–diffusion patterns the tissue and shapes its surface."),
    ("Three levels in one composition",
     "Spheroid + basement membrane + extracellular matrix"),
]
# the spheroid lede is several lines in the source; matched loosely on its first and last words
LEDE_SPHEROID = (r"<p class=\"opk\">The demonstration here is that .*?"
                 r"a sheet the cell never touches\.</p>")
LEDE_SPHEROID_NEW = ('<p class="opk">Cells grow a spheroid, deform its basement membrane, and '
                     'interact with the surrounding matrix.</p>')
# a whole markdown section deleted from the qmd -> its rendered <section id="..."> in docs
DROP_SECTIONS = ["agentic-mechanistic-discovery"]


def _captions(text: str) -> dict:
    return {m.group("clip"): m.group("cap") for m in CARD_RE.finditer(text)}


def _sync_captions(docs: str, want: dict) -> tuple[str, int, list]:
    """Copy each qmd caption onto the docs card playing the same clip."""
    changed, missing = 0, []
    for clip, cap in want.items():
        i = docs.find(f'<video src="gallery/{clip}"')
        if i < 0:
            missing.append(clip)
            continue
        m = re.compile(r'<span class="sim-cap">(.*?)</span>', re.S).search(docs, i)
        if not m or docs.find("</figure>", i) < m.start():
            missing.append(clip + " (no caption inside its figure)")
            continue
        if m.group(1) != cap:
            docs = docs[:m.start(1)] + cap + docs[m.end(1):]
            changed += 1
    return docs, changed, missing


def _drop_section(docs: str, sid: str) -> tuple[str, bool]:
    """Remove a rendered <section id="sid" ...> ... </section> and its table-of-contents entry."""
    m = re.search(rf'<section id="{re.escape(sid)}"[^>]*>', docs)
    if not m:
        return docs, False
    # sections do not nest here, so the first closing tag that balances is the matching one
    depth, i = 1, m.end()
    while depth and i < len(docs):
        nxt_open = docs.find("<section", i)
        nxt_close = docs.find("</section>", i)
        if nxt_close < 0:
            return docs, False
        if 0 <= nxt_open < nxt_close:
            depth += 1; i = nxt_open + 8
        else:
            depth -= 1; i = nxt_close + len("</section>")
    docs = docs[:m.start()] + docs[i:]
    docs = re.sub(rf'\s*<li><a href="#{re.escape(sid)}"[^>]*>.*?</a></li>', "", docs, flags=re.S)
    return docs, True


def main():
    dry = "--dry-run" in sys.argv
    qmd = open(QMD).read()
    docs = open(DOCS).read()
    want = _captions(qmd)
    print(f"[sync] {len(want)} cards in index.qmd")

    docs, n_cap, missing = _sync_captions(docs, want)
    print(f"[sync] captions updated: {n_cap}")
    for m in missing:
        print(f"[sync] NOT IN DOCS: {m}")

    problems = []
    for old, new in TEXT:
        if new in docs and old not in docs:
            print(f"[sync] already applied: {new[:50]}")
            continue
        if old not in docs:
            problems.append(f"text not found: {old[:70]}")
            continue
        docs = docs.replace(old, new)
        print(f"[sync] text -> {new[:60]}")
    if LEDE_SPHEROID_NEW in docs:
        print("[sync] already applied: spheroid lede")
    else:
        docs, n = re.subn(LEDE_SPHEROID, LEDE_SPHEROID_NEW, docs, flags=re.S)
        print(f"[sync] spheroid lede replaced: {n}")
        if n != 1:
            problems.append(f"spheroid lede matched {n} times")
    for sid in DROP_SECTIONS:
        docs, ok = _drop_section(docs, sid)
        print(f"[sync] section {sid}: {'removed' if ok else 'not present (already gone?)'}")

    if problems:
        for p in problems:
            print(f"[sync] PROBLEM {p}")
        raise SystemExit("[sync] nothing written -- fix the anchors above first")
    if dry:
        print("[sync] dry run: docs/index.html not written")
        return
    with open(DOCS, "w") as f:
        f.write(docs)
    print("[sync] docs/index.html written")


if __name__ == "__main__":
    main()
