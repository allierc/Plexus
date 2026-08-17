"""Put the two galaxy-encounter cards on the minisite, in place of the two Coulomb ones.

The Inverse-square section of `index.qmd` is hand-authored (it is not one of the five generated
blocks), so this script edits `index.qmd` AND the rendered `docs/index.html` in lockstep -- quarto
is not installed in this container, and the committed html is what GitHub Pages serves.

What it changes, and nothing else:
  * the section heading, which no longer covers electrostatics
  * its lede and its reference line
  * the two cards that played `coulomb_3.mp4` and `coulomb_6_3d.mp4`

Each card keeps the site's convention: the title opens the run's OWN spec (the config file, which
carries the auto video description appended by the captioner), and every number in the caption is
measured from the run by `measure.py` -- none is typed here by hand.

Run:  python prototype/galaxy_collision/patch_site.py
"""
from __future__ import annotations
import html
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import measure                                        # noqa: E402

ROOT = "/workspace/Plexus"
QMD = os.path.join(ROOT, "index.qmd")
DOCS = os.path.join(ROOT, "docs/index.html")
CFG = os.path.join(ROOT, "config/inverse_square")

# quarto rewrites the heading with an anchor class, so the two files carry different markup for
# the same line: match the TEXT and keep whatever attributes the file already had.
HEAD_RE = re.compile(r"(<h3[^>]*>)Inverse-square law — self-gravity &amp; electrostatics</h3>")
HEAD_NEW = r"\1Inverse-square law — self-gravity</h3>"

LEDE_OLD = ('<p class="opk">One <code>squared_law</code> operator, two regimes: '
            '<code>law: gravity</code> (mass, always attract, all-pairs) collapses a disc into a '
            'galaxy; <code>law: coulomb</code> (signed charge, like-repel) is a plasma.</p>')
LEDE_NEW = ('<p class="opk">One <code>squared_law</code> operator with <code>law: gravity</code> — '
            'mass, always attracting, summed over every pair. It collapses a disc into a galaxy, '
            'and because the sum runs over the whole set, two galaxies in one set pull on each '
            'other through that same operator: a collision needs no second law.</p>')

REF_OLD = ('<p class="opk-ref">Reference — softened self-gravity after Philip Mocz’s '
           '<em>Create Your Own N-body Simulation</em> (2020); Coulomb after '
           '<a href="https://github.com/allierc/ParticleGraph">ParticleGraph</a>.</p>')
REF_NEW = ('<p class="opk-ref">Reference — softened self-gravity after Philip Mocz’s '
           '<em>Create Your Own N-body Simulation</em> (2020); the encounter geometry after '
           'Toomre &amp; Toomre, <em>Galactic Bridges and Tails</em> (1972).</p>')

# (video the card plays now, new clip, title, caption builder) -- in page order.
CARDS = [
    ("coulomb_3.mp4", "galaxy_collision_3d", "two galaxies, grazing",
     lambda m: (f"the same law between two discs: their centres pass {m['passage_sep']:.2f} apart "
                f"at t&nbsp;=&nbsp;{m['passage_t']:.1f}, {m['passage_sep'] / 1.2:.1f} disc radii, "
                f"which raises tidal tails and takes {100 * m['stripped']:.0f}% of the "
                f"{m['n']:,} stars out of both galaxies")),
    ("coulomb_6_3d.mp4", "galaxy_merger_3d", "two galaxies, head-on",
     lambda m: (f"one knob apart — the encounter's angular momentum set to zero, so the discs fall "
                f"straight through each other at t&nbsp;=&nbsp;{m['passage_t']:.1f}: no tails, one "
                f"remnant, red and blue mixed evenly through its core, and "
                f"{100 * m['stripped']:.0f}% of the stars thrown out into a halo")),
]


def _card(clip: str, title: str, cap: str, spec: str) -> str:
    """One gallery card, in the exact shape the neighbouring hand-written cards use."""
    return (f'  <figure class="sim-card">\n'
            f'    <video src="gallery/{clip}.mp4" autoplay loop muted playsinline '
            f'preload="metadata"></video>\n'
            f'    <figcaption>\n'
            f'      <span class="sim-name" tabindex="0">{title}<span class="sim-spec"><pre>'
            f'{html.escape(spec)}</pre></span></span>\n'
            f'      <span class="sim-cap">{cap}</span>\n'
            f'    </figcaption>\n'
            f'  </figure>\n')


def _replace_card(text: str, old_video: str, new_card: str, where: str) -> str:
    """Swap the whole <figure> that plays `old_video` for `new_card`.

    The opening tag is found by searching back for `<figure` and NOT for the full
    `<figure class="sim-card">`: quarto rewrites some of them to `class="sim-card figure"`, so
    matching the whole tag skipped the card being replaced, walked back into its NEIGHBOUR, and
    deleted that one instead -- which is how the N-body card disappeared on the first attempt.
    """
    i = text.find(f'<video src="gallery/{old_video}"')
    if i < 0:
        raise SystemExit(f"{where}: no card plays {old_video}")
    start = text.rfind("<figure", 0, i)
    if start < 0 or text.count("<figure", start, i) != 1:
        raise SystemExit(f"{where}: {old_video} is not inside a single <figure>")
    start = text.rfind("\n", 0, start) + 1                        # keep the card's indentation
    end = text.find("</figure>", i) + len("</figure>")
    if text[end:end + 1] == "\n":
        end += 1
    return text[:start] + new_card + text[end:]


def main():
    specs, caps = {}, {}
    for _old, run, _title, capfn in CARDS:
        with open(os.path.join(CFG, run + ".yaml")) as f:
            specs[run] = f.read().rstrip("\n")
        caps[run] = capfn(measure(run))
        if "descriptions:" not in specs[run]:
            print(f"[site] NOTE {run}.yaml carries no auto video description yet")
    for path, label in ((QMD, "index.qmd"), (DOCS, "docs/index.html")):
        with open(path) as f:
            t = f.read()
        t, nh = HEAD_RE.subn(HEAD_NEW, t)
        if nh != 1:
            raise SystemExit(f"{label}: matched the heading {nh} times, expected 1")
        for old, new in ((LEDE_OLD, LEDE_NEW), (REF_OLD, REF_NEW)):
            if old not in t:
                raise SystemExit(f"{label}: could not find\n  {old[:90]}")
            t = t.replace(old, new)
        for old_video, run, title, _capfn in CARDS:
            t = _replace_card(t, old_video, _card(run, title, caps[run], specs[run]), label)
        with open(path, "w") as f:
            f.write(t)
        print(f"[site] {label}: heading, lede, reference and 2 cards replaced")
    for run in specs:
        print(f"[site] {run}: {caps[run]}")
    # the clips the page now plays must exist in BOTH gallery dirs
    for _old, run, _t, _c in CARDS:
        for d in ("gallery", "docs/gallery"):
            p = os.path.join(ROOT, d, run + ".mp4")
            print(f"[site] {'OK  ' if os.path.exists(p) else 'MISSING'} {d}/{run}.mp4"
                  f"{'' if not os.path.exists(p) else f'  ({os.path.getsize(p) / 1e6:.1f} MB)'}")


if __name__ == "__main__":
    main()
