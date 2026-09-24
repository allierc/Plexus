"""Build the slide bodies of a Plexus deck: one frame per (movie, spec) pair, no prose by hand.

    python presentation/make_slides.py                 # rebuild slides/*.tex and Movies/*
    python presentation/make_slides.py --list          # show the pool of pairs it can draw from

Every slide is a movie on the left and `tools/spec_summary.py`'s rendering of the spec that
produced it on the right, so the deck says exactly what was run and cannot drift from the spec.

The pool is the site gallery: `index.qmd` embeds, beside each clip, the spec it came from, so
the clip and the spec are paired by the site itself rather than by a filename convention that
would silently mismatch. SELECTION is seeded (`--seed`), one pair per mechanism family first so
ten slides span the library rather than showing one family ten times.
"""
from __future__ import annotations

import argparse
import glob
import html as htmllib
import os
import random
import re
import shutil
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
HERE = os.path.join(ROOT, "presentation")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import spec_summary as S                                        # noqa: E402

FFMPEG = "/workspace/.conda_envs/MPM-pytorch/bin/ffmpeg"        # no ffmpeg on PATH in this container


def pool() -> list[dict]:
    """Every gallery clip whose embedded spec resolves to a config file, with its heading.

    The heading is the `<h3>` the clip sits under on the site -- the mechanism being shown, in
    the site's own words -- and the caption is the clip's own figcaption.
    """
    text = open(os.path.join(ROOT, "index.qmd")).read()
    specs = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "config", "*", "*.yaml"))):
        specs.setdefault(os.path.splitext(os.path.basename(f))[0], f)
    out = []
    for m in re.finditer(r'<video src="gallery/([^"]+)\.mp4".*?<pre>(.*?)</pre>', text, re.S):
        clip, body = m.group(1), m.group(2)
        name = re.search(r"^\s*name:\s*(\S+)", body, re.M)
        if not name or name.group(1) not in specs:
            continue
        head = None
        for h in re.finditer(r"<h3>(.*?)</h3>", text[:m.start()], re.S):
            head = h.group(1)
        cap = re.search(r'<span class="sim-name"[^>]*>(.*?)<span', text[m.start():m.end()], re.S)
        movie = next((p for p in (os.path.join(ROOT, "gallery", clip + ".mp4"),
                                  os.path.join(ROOT, "docs", "gallery", clip + ".mp4"))
                      if os.path.exists(p)), None)
        if not movie:
            continue
        out.append(dict(clip=clip, spec=specs[name.group(1)], movie=movie,
                        family=os.path.basename(os.path.dirname(specs[name.group(1)])),
                        heading=_plain(head or ""), caption=_plain(cap.group(1) if cap else "")))
    return out


def _plain(s: str) -> str:
    """HTML fragment to plain text: tags out, entities decoded, whitespace collapsed."""
    return re.sub(r"\s+", " ", htmllib.unescape(re.sub(r"<[^>]+>", "", s))).strip()


def choose(items: list[dict], n: int, seed: int) -> list[dict]:
    """`n` pairs, one per family first, then filling at random -- a deck that spans the library."""
    rng = random.Random(seed)
    by = {}
    for it in items:
        by.setdefault(it["family"], []).append(it)
    picked = [rng.choice(v) for _, v in sorted(by.items())]
    rng.shuffle(picked)
    rest = [it for it in items if it not in picked]
    rng.shuffle(rest)
    return (picked + rest)[:n]


def icons() -> None:
    """The nine kind glyphs, on black, beside the deck -- drawn by scripts/make_op_icons.py."""
    src = os.path.join(ROOT, "figures", "icons_black")
    dst = os.path.join(HERE, "icons")
    if not os.path.isdir(src):
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_op_icons.py"),
                        "--black"], check=True)
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(os.path.join(src, "*.png")):
        if not os.path.basename(f).startswith("_"):
            shutil.copyfile(f, os.path.join(dst, os.path.basename(f)))


def poster(movie: str, dest_png: str) -> None:
    """A still from 85% of the way in: the poster the deck shows until the video is played.

    Late, not early. These runs BUILD what they are about -- an attractor is one dot at 40% of
    the way in and the whole strange attractor by the end -- so a still from near the end is the
    one that shows what the clip is of.
    """
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", movie, "-vf",
                    "select='eq(n\\,trunc(n_frames*0.85))',scale=900:-2", "-frames:v", "1",
                    dest_png], check=False)
    if not os.path.exists(dest_png) or os.path.getsize(dest_png) < 1000:
        subprocess.run([FFMPEG, "-y", "-v", "error", "-i", movie, "-vf", "scale=900:-2",
                        "-frames:v", "1", dest_png], check=True)


def build(n: int, seed: int, equations: int, max_ops: int) -> list[str]:
    os.makedirs(os.path.join(HERE, "Movies"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "slides"), exist_ok=True)
    icons()
    chosen, written = choose(pool(), n, seed), []
    for i, it in enumerate(chosen, 1):
        base = f"{i:02d}_{it['clip']}"
        shutil.copyfile(it["movie"], os.path.join(HERE, "Movies", base + ".mp4"))
        poster(it["movie"], os.path.join(HERE, "Movies", base + ".png"))
        summary = S.summarise(it["spec"])
        # The frame title is the spec's OWN `general.title` -- the name the run is called. The
        # site heading is only a fallback, and it is a section heading rather than a name.
        title = summary["header"]["title"] or it["heading"] or summary["header"]["name"]
        frame = S.render_frame(summary, title=S.tex_escape(title),
                               movie=f"Movies/{base}", figure=None,
                               caption=S.tex_escape(it["caption"] or it["clip"]),
                               max_ops=max_ops, eq_for=equations)
        dest = os.path.join(HERE, "slides", base + ".tex")
        with open(dest, "w") as f:
            f.write(f"% generated by presentation/make_slides.py from {os.path.relpath(it['spec'], ROOT)}\n")
            f.write(frame)
        written.append(base)
        print(f"  {base:44s} {os.path.relpath(it['spec'], ROOT)}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-n", type=int, default=10, help="how many slides")
    ap.add_argument("--seed", type=int, default=26, help="which pairs are drawn")
    ap.add_argument("--equations", type=int, default=2, help="equations typeset per slide")
    ap.add_argument("--max-ops", type=int, default=0, help="operators listed per slide; 0 = all")
    ap.add_argument("--list", action="store_true", help="print the pool and stop")
    a = ap.parse_args()
    if a.list:
        for it in pool():
            print(f"  {it['family']:22s} {it['clip']:32s} {os.path.relpath(it['spec'], ROOT)}")
        return
    names = build(a.n, a.seed, a.equations, a.max_ops or None)
    body = "\n".join(rf"\input{{slides/{b}.tex}}" for b in names)
    with open(os.path.join(HERE, "slides", "all.tex"), "w") as f:
        f.write("% generated by presentation/make_slides.py -- the deck inputs this one file\n")
        f.write(body + "\n")
    print(f"{len(names)} slides -> presentation/slides/all.tex")


if __name__ == "__main__":
    main()
