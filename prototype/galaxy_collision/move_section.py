"""Move one <h3> gallery section of the minisite to a new place, in both files at once.

A section runs from its own <h3> to the next <h3> (or to the next markdown `##` heading, which is
where the galleries end). Quarto writes `<h3 class="anchored">` in `docs/index.html` and plain
`<h3>` in `index.qmd`, so the heading is matched on its TEXT and the tag is taken as it is found.

Refuses to move a section that contains a generated block marker: those blocks are re-inserted by
`minisite_section.py` relative to their own anchors, so moving one by hand would put the page's
two copies of the truth in different orders.

Run:  python prototype/galaxy_collision/move_section.py "<section text>" "<before this section>"
"""
from __future__ import annotations
import re
import sys

FILES = ["/workspace/Plexus/index.qmd", "/workspace/Plexus/docs/index.html"]


def _bounds(s: str, text: str) -> tuple[int, int]:
    """(start, end) of the section whose heading contains `text`."""
    m = re.search(r"<h3[^>]*>" + re.escape(text) + r"</h3>", s)
    if not m:
        raise SystemExit(f"heading not found: {text}")
    start = s.rfind("\n", 0, m.start()) + 1
    nxt_h3 = s.find("<h3", m.end())
    nxt_h2 = re.search(r"^## ", s[m.end():], re.M)              # markdown heading (qmd)
    nxt_sec = s.find("</section>", m.end())                     # rendered section end (docs)
    ends = [e for e in (nxt_h3,
                        m.end() + nxt_h2.start() if nxt_h2 else -1,
                        nxt_sec) if e > 0]
    if not ends:
        raise SystemExit(f"no section end after {text}")
    end = min(ends)
    end = s.rfind("\n", 0, end) + 1
    return start, end


def main():
    what, before = sys.argv[1], sys.argv[2]
    for path in FILES:
        s = open(path).read()
        a, b = _bounds(s, what)
        block = s[a:b]
        if "<!-- BEGIN " in block:
            raise SystemExit(f"{path}: {what} holds a generated block -- move it in the generator")
        rest = s[:a] + s[b:]
        m = re.search(r"<h3[^>]*>" + re.escape(before) + r"</h3>", rest)
        if not m:
            raise SystemExit(f"{path}: target heading not found: {before}")
        at = rest.rfind("\n", 0, m.start()) + 1
        out = rest[:at] + block + rest[at:]
        if len(out) != len(s):
            raise SystemExit(f"{path}: length changed by {len(out) - len(s)} -- refusing to write")
        open(path, "w").write(out)
        print(f"[move] {path}: '{what}' now sits directly before '{before}'")


if __name__ == "__main__":
    main()
