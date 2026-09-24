"""Give every spec a `general.title:` -- the name a human reads, beside the name a path needs.

    python tools/name_specs.py --report      # what each spec would be called, nothing written
    python tools/name_specs.py --write       # insert `title:` into every spec that lacks one

`general.name` identifies a run: it names the directory, the trajectory and the movie, so it is
terse and machine-shaped (`turing2d_gs_eta`). A caption, a slide or a figure needs the other
name -- what the run IS -- and that cannot be derived from a filename without either an
unreadable caption or an unusable filename. So both are declared.

Two sources, and the report says which one a title came from:

    gallery   the site already names 29 of these runs in prose -- the `<h3>` the clip sits
              under and its own figcaption. That wording is Cedric's, so it is used verbatim.
    derived   everything else: the identifier with its abbreviations expanded and its words
              separated. Honest but mechanical, and marked as such by `--report` so the ones
              worth rewriting by hand can be found (`--report --derived-only`).

A title is inserted TEXTUALLY, as the first key under `general:`, so the rest of a
hand-maintained spec -- its key order, its blank lines -- is returned byte-for-byte.
"""
from __future__ import annotations

import argparse
import glob
import html as htmllib
import os
import re

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Abbreviations that appear in spec identifiers. Only entries whose expansion is certain from
# the codebase are listed; an unlisted token is passed through with its own capitalisation.
EXPAND = {
    "gs": "Gray-Scott", "rps": "rock-paper-scissors", "fn": "FitzHugh-Nagumo",
    "mpm": "MPM", "mls": "MLS", "si": "solid-into-fluid", "cx": "central complex",
    "rd": "reaction-diffusion", "ab": "apico-basal", "avm": "vertex model",
    "ecm": "extracellular matrix", "hcm": "hypertrophic cardiomyopathy",
    "2d": "2D", "3d": "3D", "1d": "1D", "gnn": "GNN", "cv": "coefficient of variation",
    "nbody": "N-body", "t1": "T1", "ctrl": "control", "cfl": "CFL", "vs": "versus",
    "sh": "Swift-Hohenberg", "bm": "membrane", "pf": "phase field", "wt": "wild type",
}


def prettify(name: str) -> str:
    """`turing2d_gs_eta` -> `Turing 2D, Gray-Scott, eta`. Mechanical, and marked as such."""
    parts = [p for p in re.split(r"[_\-]+", name) if p]
    out = []
    for p in parts:
        low = p.lower()
        if low in EXPAND:
            out.append(EXPAND[low])
            continue
        m = re.fullmatch(r"([a-z]+)([23])d", low)            # turing2d -> Turing 2D
        if m and m.group(1) not in EXPAND:
            out.append(f"{m.group(1).capitalize()} {m.group(2)}D")
            continue
        out.append(p if p.isupper() else p.capitalize() if out == [] else p)
    text = " ".join(out[:1] + [o for o in out[1:]])
    return text[:1].upper() + text[1:]


def gallery_titles() -> dict:
    """spec name -> the site's own wording for it: the section heading and the clip's caption."""
    text = open(os.path.join(ROOT, "index.qmd")).read()
    out = {}
    for m in re.finditer(r'<video src="gallery/([^"]+)\.mp4".*?<pre>(.*?)</pre>', text, re.S):
        nm = re.search(r"^\s*name:\s*(\S+)", m.group(2), re.M)
        if not nm:
            continue
        head = None
        for h in re.finditer(r"<h3>(.*?)</h3>", text[: m.start()], re.S):
            head = h.group(1)
        cap = re.search(r'<span class="sim-name"[^>]*>(.*?)<span', text[m.start(): m.end()], re.S)
        head, cap = _plain(head or ""), _plain(cap.group(1) if cap else "")
        if head and cap:
            out[nm.group(1)] = f"{head} --- {cap}"
        elif head:
            out[nm.group(1)] = head
    return out


def _plain(s: str) -> str:
    return re.sub(r"\s+", " ", htmllib.unescape(re.sub(r"<[^>]+>", "", s))).strip()


def spec_name(path: str) -> str | None:
    """The declared `general.name`, read textually so a malformed spec still reports."""
    m = re.search(r"^\s{0,2}name:\s*(\S+)", open(path).read(), re.M)
    return m.group(1).strip("'\"") if m else None


def has_title(text: str) -> bool:
    return bool(re.search(r"^\s{0,4}title:\s*\S", text, re.M))


def insert_title(text: str, title: str) -> str | None:
    """Put `title:` as the first key under `general:`; None if there is no `general:` block."""
    m = re.search(r"^general:\s*$", text, re.M)
    if not m:
        return None
    indent = "  "
    nxt = text[m.end():]
    ind = re.search(r"^(\s+)\S", nxt, re.M)
    if ind:
        indent = ind.group(1)
    safe = title.replace('"', "'")
    return text[: m.end()] + f'\n{indent}title: "{safe}"' + nxt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help="write the titles into the specs")
    ap.add_argument("--report", action="store_true", help="print what each spec would be called")
    ap.add_argument("--derived-only", action="store_true", help="report only the mechanical ones")
    ap.add_argument("--glob", default="config/*/*.yaml", help="which specs to touch")
    a = ap.parse_args()

    gallery = gallery_titles()
    counts = dict(gallery=0, derived=0, already=0, no_general=0, no_name=0)
    for path in sorted(glob.glob(os.path.join(ROOT, a.glob))):
        text = open(path).read()
        name = spec_name(path)
        rel = os.path.relpath(path, ROOT)
        if not name:
            counts["no_name"] += 1
            continue
        if has_title(text):
            counts["already"] += 1
            continue
        title, source = (gallery[name], "gallery") if name in gallery else (prettify(name), "derived")
        counts[source] += 1
        if a.report and (not a.derived_only or source == "derived"):
            print(f"{source:8s} {rel:52s} {title}")
        if a.write:
            new = insert_title(text, title)
            if new is None:
                counts["no_general"] += 1
                continue
            with open(path, "w") as f:
                f.write(new)
    print("  ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
