"""Build per-scenario progression montages + results.md from the reproduced winner
gifs (run after reproduce.py). Cheap: reads the gifs, no re-simulation.

    python build_gallery.py

For each scenario it stacks the FINAL frame of winner_1..N top-to-bottom (the
optimizer's improvement, end-state per design) into <scenario>_progression.png, and
writes results.md from reproduce.log (logged vs reproduced escaped counts).
"""

import glob
import os
import re

from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = ["race_pillars", "race_maze_hard"]


def last_frame(gif):
    g = Image.open(gif)
    fr = None
    try:
        while True:
            fr = g.copy().convert("RGB")
            g.seek(g.tell() + 1)
    except EOFError:
        pass
    return fr


def progression(scenario):
    gifs = sorted(glob.glob(os.path.join(_HERE, f"{scenario}_winner_*.gif")),
                  key=lambda p: int(re.search(r"_winner_(\d+)", p).group(1)))
    if not gifs:
        return 0
    frames = [last_frame(g) for g in gifs]
    w, h = frames[0].size
    m = Image.new("RGB", (w, h * len(frames)), "black")
    for k, im in enumerate(frames):
        m.paste(im.resize((w, h)), (0, k * h))
    m.save(os.path.join(_HERE, f"{scenario}_progression.png"))
    return len(gifs)


def results():
    log = os.path.join(_HERE, "reproduce.log")
    rows = []
    if os.path.exists(log):
        for line in open(log):
            parts = dict(p.split("=") for p in line.split("\t") if "=" in p)
            name = line.split("\t")[:2]
            if len(name) == 2:
                rows.append((name[0], name[1], parts.get("logged", "?"),
                             parts.get("reproduced", "?"), parts.get("n", "?")))
    with open(os.path.join(_HERE, "results.md"), "w") as f:
        f.write("# Race winners — reproduced from WINNERS_*.md\n\n")
        f.write("scenario | winner | escaped (logged) | escaped (reproduced) | fleet n\n")
        f.write("---|---|---|---|---\n")
        for s, w, lg, rp, n in rows:
            f.write(f"{s} | {w} | {lg} | {rp} | {n}\n")
    return len(rows)


if __name__ == "__main__":
    for s in SCEN:
        k = progression(s)
        print(f"{s}: {k} winners -> {s}_progression.png")
    print(f"results.md: {results()} rows")
