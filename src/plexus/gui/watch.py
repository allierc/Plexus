"""A read-only window onto what this session is doing to the page -- served BY the page, at
`/watch`, so it needs no second port.

WHY IT LIVES HERE AND NOT ON A PORT OF ITS OWN. It did have its own port, and the port was the
problem: a server bound to 0.0.0.0 inside this container is still invisible until the editor
forwards it, and an unforwarded port reads as ERR_CONNECTION_REFUSED, which looks exactly like a
crashed server. `/watch` is on the port that already works.

IT STILL CANNOT DISTURB THE RUN. The hazard was never the port, it was the MAIN page: it polls
`/api/scene/state`, sees a version this session's build bumped, and re-seeds to match -- and a
re-seed stops an in-flight run. This page polls nothing but the four routes below, all of which
only read files off disk. There is no route here that changes anything.

THE PROBLEM IT SOLVES. The page and the API drive ONE scene on ONE VTK thread, so a person
watching at `/` and a session building through `/api/...` fight: the page polls, sees the version
the session's build just bumped, re-seeds itself to match, and a re-seed STOPS an in-flight run.
Measured three times in a row -- `[view] a run was in flight and was stopped so the scene could be
re-seeded` -- and it reads exactly like a run stuck at frame 0.

So the session works on its own server and this shows what it sees: the newest picture it took and
the journal of what it did. Read-only by construction -- there is no route here that changes
anything -- so watching cannot disturb the work.

It serves what `gui_drive.py` writes, ONE FOLDER PER KIND OF OUTPUT with the step number as the
filename, so a step reads across the folders and a kind reads down one:
  builder/png/NNNN.png     the picture
  builder/spec/NNNN.yaml   the spec that made it
  builder/why/NNNN.txt     the time and the reason
  builder/mp4/NNNN.mp4     the movie, when the step ran one
  log/gui_runs/journal.txt         one line per action

A STEP IS FOUR FILES, not one. The picture says what the scene looked like, the yaml says what
was built, the why says what it was for and when, and the mp4 -- when the step ran one -- says
what it did. Walking prev/next walks all four together, because any one of them alone is a
record of nothing.
"""
from __future__ import annotations

import glob
import html
import json
import os
import time
# up from src/plexus/gui/ to the repo
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HISTORY = os.path.join(REPO, "builder")
SHOTS = os.path.join(HISTORY, "png")
JOURNAL = os.path.join(REPO, "log", "gui_runs", "journal.txt")


def shots():
    """Every picture taken, oldest first -- the order they were made, which is the order to walk.

    Sorted by NAME here, not by time: these names are a step counter this session assigns in the
    order it builds, and that order is the thing to walk. Modification time would reorder the
    history if a file were ever touched or copied.
    """
    return sorted(glob.glob(os.path.join(SHOTS, "[0-9][0-9][0-9][0-9].png")))


def sidecar(png: str, kind: str) -> str:
    """The step's other output of kind `kind`, by number. The picture is the spine of the walk and
    everything else is found from it."""
    ext = {"spec": ".yaml", "why": ".txt", "mp4": ".mp4"}[kind]
    return os.path.join(HISTORY, kind, os.path.basename(png)[:4] + ext)


def _read(path: str) -> str:
    """The text beside a picture, or nothing. A step written before this record existed simply has
    no why and no spec, and that is shown as such rather than as an error."""
    if not os.path.exists(path):
        return ""
    with open(path, errors="replace") as f:
        return f.read()


def newest_shot():
    fs = shots()
    return fs[-1] if fs else None


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Plexus — watching</title>
<style>
 body{background:#0b0d10;color:#ccd3da;font:13px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:14px 18px}
 h1{font-size:15px;font-weight:600;margin:0 0 2px;color:#e6edf3}
 .sub{color:#7d8894;font-size:12px;margin-bottom:10px}
 .wrap{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap}
 .why{background:#12171d;border-left:3px solid #7fb069;border-radius:0 5px 5px 0;padding:8px 12px;
      margin-bottom:10px;white-space:pre-wrap;font-size:12px;color:#c8d2dc;max-width:min(62vw,820px)}
 .col{display:flex;flex-direction:column}
 video{border:1px solid #2a3139;background:#000;max-width:min(62vw,820px);margin-top:10px}
 .tabs button{font-size:12px;padding:3px 10px}
 img{border:1px solid #2a3139;background:#000;max-width:min(62vw,820px)}
 pre{background:#0f1319;border:1px solid #222a33;border-radius:5px;padding:10px 12px;
     max-height:74vh;overflow:auto;font-size:11.5px;color:#b9c3cd;flex:1;min-width:300px;margin:0}
 .t{color:#6f7a86}
 b{color:#7fb069;font-weight:600}
 button{background:#1d2530;color:#ccd3da;border:1px solid #2f3945;border-radius:4px;
        padding:4px 12px;margin-right:6px;cursor:pointer;font:inherit}
 button:hover{background:#26303c}
 button.on{background:#2d4a2d;border-color:#3f6b3f}
</style></head><body>
<h1>Plexus &mdash; watching the session</h1>
<div class=sub id=meta>&hellip;</div>
<div class=row style="margin-bottom:8px">
 <button onclick="step(-1)">&larr; prev</button>
 <button onclick="step(1)">next &rarr;</button>
 <button onclick="live()" id=livebtn>live</button>
 <span id=pos class=t></span>
</div>
<div class=wrap>
 <div class=col>
  <div class=why id=why></div>
  <img id=shot>
  <video id=mov controls loop muted style="display:none"></video>
 </div>
 <div class=col style="flex:1;min-width:300px">
  <div class="row tabs" style="margin-bottom:6px">
   <button onclick="pane('journal')" id=b_journal class=on>journal</button>
   <button onclick="pane('spec')" id=b_spec>spec</button>
  </div>
  <pre id=journal></pre>
  <pre id=spec style="display:none"></pre>
 </div>
</div>
<script>
// `idx` is which picture is shown; -1 means FOLLOW THE NEWEST, which is what a watcher wants by
// default. Pressing prev/next pins it to one shot so the view stops jumping while it is read.
let idx = -1, total = 0, shown = 'journal';
function pane(w){
 shown = w;
 for(const k of ['journal','spec']){
  document.getElementById(k).style.display = (k === w) ? '' : 'none';
  document.getElementById('b_' + k).className = (k === w) ? 'on' : '';
 }
 tick();
}
function live(){ idx = -1; tick(); }
function step(d){
 if(total === 0) return;
 if(idx < 0) idx = total - 1;           // stepping back from live starts at the newest
 idx = Math.max(0, Math.min(total - 1, idx + d));
 tick();
}
async function tick(){
 const j = await (await fetch('/api/watch/state?i=' + idx)).json();
 total = j.n_shots;
 document.getElementById('livebtn').className = (idx < 0) ? 'on' : '';
 document.getElementById('pos').textContent =
   total ? (idx < 0 ? `live — ${total} of ${total}` : `${j.index + 1} of ${total}`) : '';
 document.getElementById('meta').innerHTML =
   j.shot ? `${j.shot_name} &nbsp;&middot;&nbsp; ${j.shot_age}s ago` : 'no picture yet';
 if(j.shot) document.getElementById('shot').src = '/api/watch/shot?i=' + j.index + '&t=' + j.shot_mtime;
 document.getElementById('journal').innerHTML = j.journal;
 if(idx < 0) document.getElementById('journal').scrollTop = 1e9;
 // THE OTHER THREE FILES OF THE STEP. The why is always in view because it is the one thing a
 // picture cannot show; the spec sits behind a tab because it is 250 lines.
 document.getElementById('why').textContent = j.why || '(no why recorded for this step)';
 document.getElementById('spec').textContent = j.spec || '(no spec recorded for this step)';
 const mv = document.getElementById('mov');
 if(j.mp4){ if(!mv.src.includes('i=' + j.index)){ mv.src = '/api/watch/mp4?i=' + j.index; } mv.style.display = ''; }
 else { mv.removeAttribute('src'); mv.style.display = 'none'; }
}
document.addEventListener('keydown', e => {
 if(e.key === 'ArrowLeft') step(-1);
 if(e.key === 'ArrowRight') step(1);
});
tick();
// only follow along while LIVE; a pinned picture must stay put
setInterval(() => { if(idx < 0) tick(); }, 2000);
</script></body></html>"""




# ---------------------------------------------------------------- the four read-only routes

def g_watch(h, q):
    """The page itself. Returned as raw HTML by the server's picture path, not as JSON."""
    return PAGE


def g_watch_state(h, q):
    """Which step is shown, the three texts beside it, and the journal."""
    fs = shots()
    try:
        i = int(q.get("i", ["-1"])[0])
    except (ValueError, TypeError):
        i = -1
    i = (len(fs) - 1) if (i < 0 or i >= len(fs)) else i
    f = fs[i] if fs else None
    jr = ""
    if os.path.exists(JOURNAL):
        with open(JOURNAL, errors="replace") as fh:
            lines = fh.read().splitlines()[-200:]
        if lines:
            jr = "\n".join(html.escape(x) for x in lines[:-1])
            # the newest line stands out; a watcher wants to know what is happening NOW
            jr = (jr + "\n" if jr else "") + "<b>" + html.escape(lines[-1]) + "</b>"
    st = {"journal": jr, "n_shots": len(fs), "index": i if fs else 0}
    if f:
        st.update(shot=True, shot_name=os.path.basename(f),
                  shot_mtime=int(os.path.getmtime(f)),
                  shot_age=int(time.time() - os.path.getmtime(f)),
                  why=_read(sidecar(f, "why")), spec=_read(sidecar(f, "spec")),
                  mp4=os.path.exists(sidecar(f, "mp4")))
    return st


def _nth(q, kind):
    fs = shots()
    try:
        i = int(q.get("i", ["-1"])[0])
    except (ValueError, TypeError):
        i = -1
    f = (fs[i] if 0 <= i < len(fs) else (fs[-1] if fs else None))
    if not f:
        return None
    return f if kind == "png" else sidecar(f, kind)


def g_watch_shot(h, q):
    return _nth(q, "png")


def g_watch_mp4(h, q):
    return _nth(q, "mp4")
