"""A beamer deck as a static website (Cedric, 2026-10-09: "make a local website of exp17.pdf to share with my colleagues").

Every page of the compiled PDF becomes an image; every movie the deck plays (`\\playmovie`: a link `run:Movies/<x>.mp4`
over its poster frame) becomes an HTML5 video laid exactly over that frame, playing while its slide is shown. One
`index.html`, no server-side code, no external library: the folder opens from any web server or from a zip.

    python tools/deck_site.py experiments/exp17_zapbench_graphcast/presentation/exp17.pdf [--out <dir>] [--dpi 220]
    cd <dir> && python -m http.server 8860          # then http://127.0.0.1:8860/

-> <dir> (default: <pdf dir>/site): index.html, slides/NNN.png, Movies/<the movies the deck plays>.mp4
Keys: left / right (or space), Home / End; the slide is in the URL (#12), so a link points at one slide.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import urllib.parse

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  html, body { margin: 0; height: 100%; background: #111; color: #ddd; font: 14px/1.35 Helvetica, Arial, sans-serif; }
  #side { position: fixed; left: 0; top: 0; bottom: 0; width: 260px; overflow-y: auto; background: #1a1a1a;
          border-right: 1px solid #333; }
  #side h1 { font-size: 15px; margin: 12px 12px 6px; color: #fff; }
  #side .hint { margin: 0 12px 10px; color: #888; font-size: 12px; }
  #side a { display: block; padding: 4px 12px; color: #bbb; text-decoration: none; font-size: 12.5px; }
  #side a.on { background: #2e7d32; color: #fff; }
  #side a:hover { background: #333; }
  #main { position: fixed; left: 261px; right: 0; top: 0; bottom: 0; display: flex; align-items: center;
          justify-content: center; }
  #stage { position: relative; }
  #stage img { display: block; width: 100%; height: 100%; }
  #stage video { position: absolute; background: #000; }
  #nav { position: fixed; right: 14px; bottom: 10px; color: #888; font-size: 12px; }
  #nav button { background: #333; color: #ddd; border: 0; padding: 4px 10px; margin-left: 4px; cursor: pointer; }
  @media (max-width: 800px) { #side { display: none; } #main { left: 0; } }
</style></head>
<body>
<div id="side"><h1>__TITLE__</h1><div class="hint">__HINT__</div><div id="toc"></div></div>
<div id="main"><div id="stage"></div></div>
<div id="nav"><span id="count"></span><button id="prev">&larr;</button><button id="next">&rarr;</button></div>
<script>
const S = __SLIDES__;
const AR = __AR__;
let cur = 0;
const stage = document.getElementById("stage"), toc = document.getElementById("toc");
S.forEach((s, i) => {
  const a = document.createElement("a"); a.href = "#" + (i + 1); a.textContent = (i + 1) + "  " + s.title;
  a.onclick = (e) => { e.preventDefault(); show(i); }; toc.appendChild(a);
});
function fit() {
  const m = document.getElementById("main"), W = m.clientWidth - 20, H = m.clientHeight - 40;
  const w = Math.min(W, H * AR); stage.style.width = w + "px"; stage.style.height = (w / AR) + "px";
}
function show(i) {
  cur = Math.max(0, Math.min(S.length - 1, i));
  const s = S[cur];
  stage.innerHTML = "";
  const img = document.createElement("img"); img.src = s.img; img.alt = s.title; stage.appendChild(img);
  s.movies.forEach((m) => {
    const v = document.createElement("video");
    v.src = m.src; v.muted = true; v.loop = true; v.autoplay = true; v.playsInline = true; v.controls = true;
    v.style.left = m.x + "%"; v.style.top = m.y + "%"; v.style.width = m.w + "%"; v.style.height = m.h + "%";
    stage.appendChild(v);
  });
  [...toc.children].forEach((a, k) => a.classList.toggle("on", k === cur));
  toc.children[cur].scrollIntoView({block: "nearest"});
  document.getElementById("count").textContent = (cur + 1) + " / " + S.length;
  history.replaceState(null, "", "#" + (cur + 1));
  [cur + 1, cur - 1].forEach((k) => { if (S[k]) { const p = new Image(); p.src = S[k].img; } });
}
document.addEventListener("keydown", (e) => {
  if (["ArrowRight", "PageDown", " "].includes(e.key)) { e.preventDefault(); show(cur + 1); }
  if (["ArrowLeft", "PageUp"].includes(e.key)) { e.preventDefault(); show(cur - 1); }
  if (e.key === "Home") show(0);
  if (e.key === "End") show(S.length - 1);
});
document.getElementById("prev").onclick = () => show(cur - 1);
document.getElementById("next").onclick = () => show(cur + 1);
window.addEventListener("resize", fit);
fit(); show((parseInt(location.hash.slice(1)) || 1) - 1);
</script></body></html>
"""


def build(pdf, out=None, dpi=220, title=None):
    import fitz
    base = os.path.dirname(os.path.abspath(pdf))
    out = out or os.path.join(base, "site")
    os.makedirs(os.path.join(out, "slides"), exist_ok=True)
    os.makedirs(os.path.join(out, "Movies"), exist_ok=True)
    d = fitz.open(pdf)
    W, H = d[0].rect.width, d[0].rect.height
    slides, movies = [], set()
    stamp = int(os.path.getmtime(pdf))
    for p in range(d.page_count):
        pg = d[p]
        img = f"slides/{p + 1:03d}.png"
        pg.get_pixmap(dpi=dpi).save(os.path.join(out, img))
        lines = [t.strip() for t in pg.get_text().split("\n") if t.strip()]
        mv = []
        for ln in pg.get_links():
            f = urllib.parse.unquote(ln.get("file") or ln.get("uri") or "").replace("run:", "").split("?")[0]
            if not f.endswith(".mp4"):
                continue
            src = os.path.join(base, f)
            if not os.path.exists(src):
                continue
            movies.add(f)
            r = ln["from"]
            mv.append({"src": f, "x": 100 * r.x0 / W, "y": 100 * r.y0 / H, "w": 100 * r.width / W, "h": 100 * r.height / H})
        # ?v=<build time>: a browser that cached an older build's slide images loads the new ones (Cedric, 2026-10-10:
        # "I do not see the modifications" -- the page showed a cached 61-page build after the 65-page one was served)
        slides.append({"img": f"{img}?v={stamp}", "title": lines[0] if lines else f"slide {p + 1}", "movies": mv})
    for f in sorted(movies):
        dst = os.path.join(out, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(os.path.join(base, f)):
            shutil.copy2(os.path.join(base, f), dst)
    name = title or os.path.splitext(os.path.basename(pdf))[0]
    page = (PAGE.replace("__TITLE__", html.escape(name)).replace("__SLIDES__", json.dumps(slides))
            .replace("__AR__", f"{W / H:.6f}")
            .replace("__HINT__", f"{len(slides)} slides, {len(movies)} movies &middot; &larr; &rarr; to move"))
    open(os.path.join(out, "index.html"), "w").write(page)
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs)
    print(f"[deck_site] {len(slides)} slides, {len(movies)} movies, {size / 1e6:.0f} MB -> {out}/index.html")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    build(a.pdf, a.out, a.dpi, a.title)
