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
  <record>/journal.txt            one line per action, per experiment

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
# WHICH EXPERIMENT'S RECORD THIS IS. The builder used to be one flat pile at `builder/`, which
# works until a second line of work starts and its steps interleave with the first -- the record
# then reads as one story told by two people. `PLEXUS_BUILDER` names the folder, so a new
# experiment is a new directory rather than a new numbering scheme, and the watcher and the writer
# read the SAME variable so they cannot drift apart.
HISTORY = os.environ.get("PLEXUS_BUILDER") or os.path.join(REPO, "builder", "exp_01_memiopsis")
SHOTS = os.path.join(HISTORY, "png")
# ONE WATCHER, EVERY RECORD. `PLEXUS_BUILDER` is only the record a page opens on: the page names the record it
# wants with `rec=<root>/<folder>` on every call (the human, 2026-09-26: "one watcher can toggle between
# experiments in experiments/ and builder/"), so two tabs on one server can watch two experiments and neither
# moves the other -- a server-wide switch would have. A record is any folder under these roots with the record
# layout (a `png/` of steps); `rec` is looked up in that list, never joined as a path, so a query string reaches
# nothing else on disk.
RECORD_ROOTS = ("experiments", "builder")


def records():
    """Every record the watcher can show: [{id, root, name, n}], newest activity first within each root."""
    out = []
    for root in RECORD_ROOTS:
        base = os.path.join(REPO, root)
        if not os.path.isdir(base):
            continue
        rows = []
        for name in os.listdir(base):
            d = os.path.join(base, name)
            png = os.path.join(d, "png")
            # AN EXPERIMENT IS LISTED FROM ITS FIRST DAY, before it has a step (the human, 2026-09-26: "add them to
            # the watcher selector even if still empty" -- exp06, 09, 14, 15 had just been launched): an experiment
            # folder by name (`experiments/expNN_*`, `builder/exp_*`), or any folder that already holds steps
            # ONLY PICTURES THAT EXIST: a record being rebuilt by another session holds, for a moment, a link whose
            # picture is gone (exp14's png/0013.png, 2026-09-26) -- its mtime raised and took the whole list down
            steps = ([f for f in os.listdir(png) if len(f) == 8 and f.endswith(".png") and f[:4].isdigit()
                      and os.path.exists(os.path.join(png, f))] if os.path.isdir(png) else [])
            is_exp = os.path.isdir(d) and (name[:3] == "exp" and name[3:5].isdigit() if root == "experiments"
                                           else name.startswith("exp_"))
            if not steps and not is_exp:
                continue
            def _mt(f):
                try:
                    return os.path.getmtime(os.path.join(png, f))
                except OSError:
                    return 0.0
            rows.append({"id": f"{root}/{name}", "root": root, "name": name, "n": len(steps),
                         "mtime": max((_mt(f) for f in steps), default=0.0)})
        out += sorted(rows, key=lambda r: r["name"])
    return out


def _rec_id(folder: str) -> str:
    rel = os.path.relpath(os.path.realpath(folder), os.path.realpath(REPO))
    return rel if not rel.startswith("..") else folder


def record_dir(q=None) -> str:
    """The record a request asks for (`rec=<root>/<folder>`, one of `records()`), else the server's own."""
    rec = ((q or {}).get("rec", [""])[0] or "").strip()
    if rec and any(r["id"] == rec for r in records()):
        return os.path.join(REPO, rec)
    return HISTORY


def journal_path(folder: str) -> str:
    return os.path.join(folder, "journal.txt")
# THE JOURNAL LIVES INSIDE THE RECORD, so it is per-experiment by construction. It was a fixed
# path under log/gui_runs/, which two sessions running at once both appended to -- the watcher on
# 8826 then showed the cilium campaign's lines interleaved with the bacterium's, and neither
# session could tell which line was its own. One folder per experiment already isolates the four
# step files; the journal is the fifth thing that has to be there.
JOURNAL = os.path.join(HISTORY, "journal.txt")


def shots(folder: str | None = None):
    """Every picture taken, oldest first -- the order they were made, which is the order to walk.

    Sorted by NAME here, not by time: these names are a step counter this session assigns in the
    order it builds, and that order is the thing to walk. Modification time would reorder the
    history if a file were ever touched or copied.
    """
    # a link whose picture is gone (a record mid-rebuild) is not a step: every reader of a step stats its picture
    return sorted(f for f in glob.glob(os.path.join(folder or HISTORY, "png", "[0-9][0-9][0-9][0-9].png"))
                  if os.path.exists(f))


def sidecar(png: str, kind: str) -> str:
    """The step's other output of kind `kind`, by number. The picture is the spine of the walk and
    everything else is found from it."""
    ext = {"spec": ".yaml", "why": ".txt", "mp4": ".mp4", "caption": ".txt"}[kind]
    # the record the picture is IN (<record>/png/NNNN.png), not the server's default one
    return os.path.join(os.path.dirname(os.path.dirname(png)), kind, os.path.basename(png)[:4] + ext)


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


# --------------------------------------------------------------------------- the `plexus` pane
# THE SPEC AS THE PAPER'S THREE QUESTIONS, not as 250 lines of YAML. The `spec` tab already shows
# the file verbatim, which is the right thing to keep -- it is the artefact, and it is what a
# re-run would read -- but it is not what a person wants to look at while watching a run: the box
# and the timestep are on line 4, the sets are two hundred lines further down, and the activities
# are named but not explained. `tools/spec_summary.py` answers the three questions plexus2.tex
# asks in its own order -- HEADER what is being run, ENTITIES what exists, OPERATORS what happens,
# each with its equation and the parameter values THIS spec gives it -- and everything in it is
# read from the spec and from the live operator registry, so it cannot drift from either.
#
# ON DEMAND AND CACHED, for two reasons. `summarise` parses the spec and walks the registry, and
# the page polls every 1.5 s: computing it on every tick would pay that cost forever for a pane
# nobody may have open. The cache is keyed on the file's mtime, so a step whose spec is rewritten
# is re-summarised and a step that is merely looked at again is free.
_SUMMARY_CACHE: dict = {}


def spec_summary(path: str, q_index: int = 0, rec: str = "") -> str:
    """`tools/spec_summary.py`'s reading of one spec file as HTML, or a message saying why not.

    Imported BY PATH because `tools/` is not a package -- the same way `spec_summary` itself
    imports `scripts/build_library.py` for the equation table. A failure here must never reach the
    page as a traceback: this pane is a convenience on a read-only window, and the `spec` tab
    beside it still holds the file itself.
    """
    if not path or not os.path.exists(path):
        return '<div class="ps-g">no spec recorded for this step</div>' 
    key = (path, os.path.getmtime(path), int(q_index or 0), rec)
    hit = _SUMMARY_CACHE.get(path)
    if hit is not None and hit[0] == key:
        return hit[1]
    # A MEASUREMENT STEP HAS NO SPEC, AND THAT IS NOT AN ERROR. `tools/builder_figure.py` writes a
    # two-line comment stub into `spec/` for every graph it records, so that `_next_index` cannot
    # hand the number to a later run and interleave two things under one index. Summarising it
    # raised `AttributeError: 'NoneType' object has no attribute 'get'` -- yaml parses a file of
    # pure comments to None -- and a watcher reading that would think the tool was broken rather
    # than that this step is a graph. Answered before the parse instead.
    with open(path, errors="replace") as fh:
        body = [ln for ln in fh.read().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not body:
        txt = ('<div class="ps-g">This step is a MEASUREMENT, not a run: it has a picture and a '
               'why, and no spec to summarise. What it measures is in the why beside it.</div>')
        _SUMMARY_CACHE[path] = (key, txt)
        return txt
    try:
        m = _summary_module()
        i = int(q_index or 0)
        s = _summarise(path)
        # THE EQUATION'S ADDRESS CARRIES A HASH OF ITS TEXT. `/api/watch/eq?i=&k=` names a POSITION,
        # and the route answers with `Cache-Control: max-age=86400`, so when an operator's equation
        # was corrected the browser kept showing the old image for that position for a day: slide 11
        # read "autocatalytic reactions in cyclic competition" -- the new title -- above Gray-Scott's
        # da/dt and du/dt, while the server was serving May-Leonard's three rates the whole time.
        # With the hash in the URL, new text is a new address and cannot be answered from cache.
        import hashlib as _hl
        _hashes = [_hl.sha1(f"{EQ_DPI}:{e}".encode()).hexdigest()[:12] for e in m.equations_shown(s)]
        text = m.render_html(s,
                             icon_url=lambda k: f"/api/watch/icon?kind={k}",
                             eq_url=lambda k: f"/api/watch/eq?i={i}&k={k}&h={_hashes[k] if k < len(_hashes) else 0}"
                                              + (f"&rec={rec}" if rec else ""))
    except Exception as e:                                       # noqa: BLE001
        text = (f'<div class="ps-g">spec_summary could not read this step: '
                f'{html.escape(type(e).__name__)}: {html.escape(str(e))}<br>'
                f'The spec itself is in the tab beside this one.</div>')
    _SUMMARY_CACHE[path] = (key, text)
    return text


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Plexus — watching</title>
<style>
 body{background:#0b0d10;color:#ccd3da;font:13px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:14px 18px}
 h1{font-size:15px;font-weight:600;margin:0 0 2px;color:#e6edf3}
 .sub{color:#7d8894;font-size:12px;margin-bottom:10px}
 .wrap{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap}
 .why{background:#12171d;border-left:3px solid #7fb069;border-radius:0 5px 5px 0;padding:8px 12px;
      margin-bottom:10px;white-space:pre-wrap;font-size:12px;color:#c8d2dc;max-width:min(62vw,820px)}
 .col{display:flex;flex-direction:column}
 video{border:1px solid #2a3139;background:#000;max-width:min(62vw,820px)}
 .tabs button{font-size:12px;padding:3px 10px}
 img{border:1px solid #2a3139;background:#000;max-width:min(62vw,820px)}
 #live3d{border-color:#4a6b8a;min-height:320px;min-width:320px;user-select:none;-webkit-user-drag:none}
 pre{background:#0f1319;border:1px solid #222a33;border-radius:5px;padding:10px 12px;
     max-height:74vh;overflow:auto;font-size:11.5px;color:#b9c3cd;flex:1;min-width:300px;margin:0;
     /* WRAPPED, because a `why` is a paragraph and not a log line. Unwrapped, reading one meant
        dragging a horizontal scrollbar left to right for every sentence. `pre-wrap` keeps the
        newlines that make the time/spec/camera header readable while letting long prose fold. */
     white-space:pre-wrap;overflow-wrap:anywhere}
 /* THE `plexus` PANE, laid out as the deck lays out a spec (presentation/make_slides.py, and
    `spec_summary.render_tex` beside the renderer this mirrors): two words in the paper's own
    vocabulary -- ENTITIES, then ACTIVITIES -- a right-aligned count, every annotation in grey,
    and each kind of activity introduced by the deck's own icon. It is a PANE and not a `pre`,
    because a summary is a small document and a `pre` can only be a block of text. */
 .ps-pane{background:#0f1319;border:1px solid #222a33;border-radius:5px;padding:12px 14px;
          max-height:74vh;overflow:auto;flex:1;min-width:300px;font-size:12.5px;color:#ccd3da}
 .ps-h{font-size:15px;font-weight:700;color:#f2c94c;margin:14px 0 6px}
 /* THE NAMES ARE WHAT A READER SCANS FOR, so they carry the headings' size, bold and white; the
    headings above them are yellow so the two levels never read as one. */
 .ps-en{color:#ffffff;font-weight:700;font-size:13.5px}
 .ps-h:first-child{margin-top:2px}
 /* NOT `width:100%`. Stretched to the pane, the count was flung to the far right and `cell` and
    `4,000` were read as two separate columns with a hand's width of nothing between them --
    the row says ONE thing and has to be read as one. `auto` puts the number where the name ends. */
 .ps-e,.ps-a{border-collapse:collapse;width:auto}
 .ps-e td{padding:1px 0;vertical-align:top}
 /* ENTITY NAMES LINE UP WITH ACTIVITY NAMES: the activities table spends its first column on the
    32-px icon (plus 2 px), so the entities table starts that far in. */
 .ps-e{margin-left:34px}
 .ps-n{text-align:right;padding:0 10px 0 22px !important;white-space:nowrap;color:#ccd3da}
 .ps-g{color:#6f7a86}
 /* A RELATION IS NOT A POPULATION, and the row says so in its own colour. `half_edge` is 520,032
    of nothing a reader should count as entities -- it is the map between the two sets above it,
    and reading it as a third body is the one misreading this table can produce. */
 .ps-rel{color:#d08770;font-weight:700;font-size:13.5px}
 /* WHAT EACH ELEMENT CARRIES, under its row: the state an equation below is written in. */
 .ps-st{color:#6f7a86;font-size:11.5px;padding:0 0 3px 0}
 .ps-a td{padding:1px 0;vertical-align:top}
 .ps-i{width:32px;padding-top:4px !important}
 /* NO `invert`. These files are coloured line art on an OPAQUE BLACK square -- invisible on a
    black slide, a dark tile on this pane, and a WHITE tile once inverted. The route strips the
    background instead (alpha = brightness), so what arrives here is already transparent. */
 .ps-i img{height:24px;border:0;background:none;max-width:none}
 .ps-k{color:#6f7a86;padding-top:6px !important;padding-bottom:2px !important}
 .ps-t{color:#ffffff;font-weight:700;font-size:13.5px;padding-left:2px !important}
 /* An equation is a PNG from the same LaTeX the deck uses, white on transparent, so it needs
    neither a border nor a background of its own. Sized in em so it tracks the text beside it. */
 .ps-q{max-width:100%;overflow-x:auto}
 .ps-q img{border:0;background:none;max-width:none;height:auto;margin:1px 0 5px;opacity:.92}
 .t{color:#6f7a86}
 b{color:#7fb069;font-weight:600}
 button{background:#1d2530;color:#ccd3da;border:1px solid #2f3945;border-radius:4px;
        padding:4px 12px;margin-right:6px;cursor:pointer;font:inherit}
 button:hover{background:#26303c}
 button.on{background:#2d4a2d;border-color:#3f6b3f}
 button.ic{padding:2px 4px;margin-right:3px;vertical-align:middle;line-height:0}
 button.ic svg{width:18px;height:18px;fill:none;stroke:currentColor;stroke-width:1.5;stroke-linejoin:round;stroke-linecap:round}
</style></head><body>
<h1>Plexus &mdash; watching the session</h1>
<div class=sub id=meta>&hellip;</div>
<div class=row style="margin-bottom:8px">
 <select id=recsel onchange="pickRec(this.value)" title="which experiment's record to watch"
         style="background:#1d2530;color:#ccd3da;border:1px solid #2f3945;border-radius:4px;padding:3px 6px;margin-right:10px;font:inherit"></select>
 <button onclick="step(-10)" title="back ten steps">&laquo;</button>
 <button onclick="step(-1)">&lt;</button>
 <button onclick="step(1)">&gt;</button>
 <button onclick="step(10)" title="forward ten steps">&raquo;</button>
 <button onclick="live()" id=livebtn>live</button>
 <button onclick="open3d()" id=d3btn>3D</button>
 <span id=d3cams style="display:none">
  <button class=ic onclick="view3d('top')" title="top view: looking down the up axis"><svg viewBox="0 0 20 20"><rect x="3" y="14" width="14" height="3"/><path d="M10 2v9M6.5 7.5 10 11l3.5-3.5"/></svg></button>
  <button class=ic onclick="view3d('bottom')" title="bottom view: looking up the up axis"><svg viewBox="0 0 20 20"><rect x="3" y="3" width="14" height="3"/><path d="M10 18V9M6.5 12.5 10 9l3.5 3.5"/></svg></button>
  <button class=ic onclick="view3d('nominal')" title="oblique: the movie's own camera"><svg viewBox="0 0 20 20"><path d="M4 14l7-3.5 6 2.5-7 3.5z"/><path d="M2 2l6 6M4.5 8.5H8V5"/></svg></button>
  <button class=ic onclick="view3d('side')" title="side view: level with the membrane"><svg viewBox="0 0 20 20"><rect x="9" y="8.5" width="9" height="3"/><path d="M1 10h6M4.5 6.5 8 10l-3.5 3.5"/></svg></button>
  <button class=ic id=d3cut onclick="cut3d()" title="slice: cut away the near half, through the centre, facing you"><svg viewBox="0 0 20 20"><rect x="3" y="3" width="14" height="14"/><path d="M10 1v18M11.5 7.5 15 4M11.5 12.5 17 7M11.5 17 17 11.5"/></svg></button>
 </span>
 <span id=pos class=t></span>
 <span id=d3note class=t></span>
</div>
<div class=row id=d3bar style="margin-bottom:8px;display:none">
 <span class=t>frame</span>
 <input type=range id=d3frame min=0 max=0 value=0 step=1 style="width:520px" oninput="scrub3d(this.value)">
 <span id=d3ft class=t></span>
 <span class=t>&nbsp;(&larr; &rarr; step one frame)</span>
</div>
<div class=row style="margin-bottom:8px">
 <button onclick="stepMovie(-10)" title="back ten movies">&laquo;</button>
 <button onclick="stepMovie(-1)" title="previous movie">&lt;</button>
 <button onclick="stepMovie(1)" title="next movie">&gt;</button>
 <button onclick="stepMovie(10)" title="forward ten movies">&raquo;</button>
 <span class=t>movies only &mdash; skips graphs and camera shots</span>
 <span id=mpos class=t></span>
</div>
<div class=wrap>
 <div class=col>
  <img id=shot>
  <img id=live3d style="display:none;cursor:grab">
  <video id=mov controls loop muted autoplay playsinline style="display:none"></video>
 </div>
 <div class=col style="flex:1;min-width:300px">
  <div class="row tabs" style="margin-bottom:6px">
   <button onclick="pane('plexus')" id=b_plexus class=on>plexus</button>
   <button onclick="pane('why')" id=b_why>why</button>
   <button onclick="pane('journal')" id=b_journal>journal</button>
   <button onclick="pane('spec')" id=b_spec>spec</button>
   <button onclick="pane('caption')" id=b_caption>caption</button>
  </div>
  <div id=plexus class=ps-pane></div>
  <pre id=why style="display:none"></pre>
  <pre id=journal style="display:none"></pre>
  <pre id=spec style="display:none"></pre>
  <pre id=caption style="display:none"></pre>
 </div>
</div>
<script>
// `idx` is which picture is shown; -1 means FOLLOW THE NEWEST, which is what a watcher wants by
// default. Pressing prev/next pins it to one shot so the view stops jumping while it is read.
// `shown` OPENS ON `plexus`, not on `why`. The why says what a step was FOR and the plexus pane
// says what it IS -- the box, the sets and the activities with their equations and this spec's own
// parameter values -- and that is the thing to have in view while a picture is being looked at.
let idx = -1, total = 0, shown = 'plexus';
// WHICH RECORD THIS TAB WATCHES: `rec=<root>/<folder>` (experiments/... or builder/...), kept in the page's own URL
// so a reload stays on it, and sent with every call -- the server holds no "current record", so two tabs on one
// watcher can watch two experiments. Empty = the record the server was started on (PLEXUS_BUILDER).
let rec = new URLSearchParams(location.search).get('rec') || '';
function R(){ return rec ? '&rec=' + encodeURIComponent(rec) : ''; }
async function loadRecords(){
 try{
  const j = await (await fetch('/api/watch/records')).json();
  const sel = document.getElementById('recsel');
  const cur = rec || j.default;
  sel.innerHTML = '';
  for(const root of ['experiments', 'builder']){
   const g = document.createElement('optgroup'); g.label = root;
   for(const r of j.records.filter(x => x.root === root)){
    const o = document.createElement('option'); o.value = r.id; o.textContent = r.name + '  (' + r.n + ')';
    if(r.id === cur) o.selected = true;
    g.appendChild(o);
   }
   if(g.children.length) sel.appendChild(g);
  }
 }catch(e){ /* the list is a convenience; the page still shows its record */ }
}
function pickRec(v){
 rec = v;
 history.replaceState(null, '', location.pathname + '?rec=' + encodeURIComponent(v));
 if(d3) close3d();
 idx = -1; total = 0; movies = []; pxAt = null; pxWant = null;
 const mv = document.getElementById('mov'); mv.removeAttribute('src'); mv.dataset.k = '';
 document.getElementById('shot').removeAttribute('src');
 tick();
}
loadRecords();
// THE 3-D VIEW. A step's picture is one camera; this re-opens that step's own SPEC on this
// server, seeds it, and serves the live render so it can be turned and zoomed. Server-side VTK
// streamed as a PNG, which is what the page itself does -- there is no WebGL here and a point
// cloud of 100,000 particles is not something to ship to a browser frame by frame.
//
// IT IS THE ONLY THING IN THIS PAGE THAT CHANGES ANYTHING, and it says so: seeding a scene takes
// over this server's one VTK thread. That is why it is a button and not the default.
let d3 = null;   // {i, azim, elev, zoom, drag}
function open3d(){
 if(total === 0) return;
 const i = (idx < 0) ? total - 1 : idx;
 if(d3 && d3.i === i){ close3d(); return; }
 // THE FRAME THE MOVIE IS AT, as a fraction of the run: the 3-D view opens there, not at the seed. A still (the
 // run's own 3d.png) is its last frame.
 const mv = document.getElementById('mov');
 let f = 1.0;
 if(mv && mv.style.display !== 'none' && mv.duration > 0) f = Math.max(0, Math.min(1, mv.currentTime / mv.duration));
 d3 = {i: i, azim: 20, elev: 8, zoom: 1.3, roll: 180, drag: null, name: null, frame: null, nk: 0, tns: null, f: f, cut: false};
 document.getElementById('d3cut').className = 'ic';
 // A COUNTER WHILE VTK BUILDS, because a button that does nothing visible for six seconds is
 // indistinguishable from a broken one. The server prints the same thing to the journal panel.
 const t0 = Date.now();
 const note = document.getElementById('d3note');
 const tim = setInterval(() => {
  if(d3) note.textContent = ' — VTK is building the scene… ' + ((Date.now()-t0)/1000).toFixed(1) + ' s';
 }, 200);
 note.textContent = ' — VTK is building the scene… 0.0 s';
 fetch('/api/watch/open3d?i=' + i + R()).then(r => r.json()).then(j => {
  clearInterval(tim);
  if(j.error){
   note.textContent = ' — ' + j.error; d3 = null;
   // THE SERVER RESTARTS ITSELF when the step needs operators added after it started: wait until it answers
   // again, then ask for the same step's 3-D view once more
   //
   // EACH CHECK TIMES OUT. Through VS Code's port forwarding a request to a server that is down can HANG rather than
   // fail, and one hung check stalled the whole retry: exp11's step 18 restarted the watcher and never reopened
   // (2026-09-27). Three seconds a check, for up to two minutes, the wait shown as it runs.
   if(j.restarting){
    const t_r = Date.now();
    const again = () => {
     const s_ = Math.round((Date.now() - t_r) / 1000);
     if(s_ > 120){ note.textContent = ' — the watcher did not come back within 2 min; reload the page'; return; }
     note.textContent = ' — the watcher is restarting to load new operators… ' + s_ + ' s (the 3D view reopens by itself)';
     const ac = new AbortController(); const to = setTimeout(() => ac.abort(), 3000);
     fetch('/api/watch/records', {signal: ac.signal, cache: 'no-store'})
       .then(r => { clearTimeout(to); if(r.ok) setTimeout(open3d, 500); else setTimeout(again, 1500); })
       .catch(() => { clearTimeout(to); setTimeout(again, 1500); });
    };
    setTimeout(again, 3000);
   }
   return;
  }
  // THE NAME IS CARRIED ON EVERY RENDER, because opening a spec is not seeding it: the render
  // route builds the scene itself when it is told which one, and without that it answers "no
  // scene is open; seed one first".
  d3.name = j.name;
  d3.nk = j.n_kept || 0; d3.tns = j.total_ns || null;
  // THE MOVIE'S OWN CAMERA when the spec states one: the view opens where the movie looks, and the oblique button
  // returns there. The fixed default above (roll 180) was chosen for the Platynereis larva and drew exp04 upside down.
  if(j.camera){ d3.cam = j.camera; d3.azim = j.camera.azim; d3.elev = j.camera.elev; d3.zoom = j.camera.zoom; d3.roll = j.camera.roll; }
  document.getElementById('d3cams').style.display = '';
  if(d3.nk > 1){
   d3.frame = Math.round(d3.f * (d3.nk - 1));
   const sl = document.getElementById('d3frame');
   sl.max = d3.nk - 1; sl.value = d3.frame;
   document.getElementById('d3bar').style.display = '';
   label3d();
  }
  note.textContent = ' — ' + j.name + ' ready in ' + (j.seconds || '?')
    + ' s: drag to turn, wheel to zoom, the slider moves through the run, 3D again to close';
  document.getElementById('shot').style.display = 'none';
  document.getElementById('mov').style.display = 'none';
  document.getElementById('live3d').style.display = '';
  render3d();
 });
}
// THE RUN'S TIME AT THE SLIDER: kept frame k of nk spans the whole run, so t = k / (nk - 1) x its length
// IN THE RUN'S OWN SCALE: a channel runs for nanoseconds and a tissue for days -- exp13's growth read
// "of 960,600,000,000,000 ns" when the label assumed the first
function tunit(ns){
 const u = [[86400e9, 'days'], [3600e9, 'h'], [60e9, 'min'], [1e9, 's'], [1e6, 'ms'], [1e3, 'µs'], [1, 'ns']];
 for(const [f, n] of u) if(ns >= f) return [f, n];
 return [1, 'ns'];
}
function label3d(){
 if(!d3 || d3.frame === null) return;
 let t = '';
 if(d3.tns){
  const [f, n] = tunit(d3.tns), T = d3.tns / f;
  t = ' — t = ' + (T * d3.frame / Math.max(d3.nk - 1, 1)).toFixed(T < 10 ? 2 : 1) + ' of ' + T.toFixed(1) + ' ' + n;
 }
 document.getElementById('d3ft').textContent = (d3.frame + 1) + ' of ' + d3.nk + t;
}
function scrub3d(v){
 if(!d3) return;
 d3.frame = Math.max(0, Math.min(d3.nk - 1, parseInt(v)));
 document.getElementById('d3frame').value = d3.frame;
 label3d();
 render3d();
}
window.addEventListener('keydown', e => {
 if(!d3 || d3.frame === null) return;
 if(e.target && e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
 if(e.key === 'ArrowRight'){ scrub3d(d3.frame + 1); e.preventDefault(); }
 if(e.key === 'ArrowLeft'){ scrub3d(d3.frame - 1); e.preventDefault(); }
});
// FOUR FIXED CAMERAS. Elevation is measured from the membrane plane toward the spec's up axis (+89 = straight down
// from above; +-89 and not 90 because the orbit's up vector is degenerate along the axis). Top, bottom and side keep
// the current azimuth, zoom and frame; the oblique one is the movie's own camera, zoom included.
function view3d(k){
 if(!d3) return;
 const c = d3.cam || {elev: 40, azim: 30, zoom: 1.0, roll: d3.roll};
 if(k === 'nominal'){ d3.azim = c.azim; d3.elev = c.elev; d3.zoom = c.zoom; }
 else d3.elev = {top: 89, bottom: -89, side: 0}[k];
 d3.roll = c.roll;
 render3d();
}
// THE SLICE: a toggle, kept while the view turns and the frames change; the server cuts along the view each render
function cut3d(){
 if(!d3) return;
 d3.cut = !d3.cut;
 document.getElementById('d3cut').className = 'ic' + (d3.cut ? ' on' : '');
 render3d();
}
function close3d(){
 d3 = null;
 document.getElementById('d3cams').style.display = 'none';
 document.getElementById('d3bar').style.display = 'none';
 document.getElementById('live3d').style.display = 'none';
 document.getElementById('d3note').textContent = '';
 tick();
}
// ONE RENDER IN FLIGHT AT A TIME, AND THE LAST GOOD FRAME STAYS UP.
//
// A render of a few hundred thousand particles takes seconds on the server's one VTK thread. A
// drag fires a mousemove every few milliseconds, so setting `img.src` on each one starts dozens
// of overlapping requests that each cancel the last image load -- and an <img> whose load was
// cancelled shows NOTHING. That is the black panel: not a failed render, a hundred successful
// ones none of which was ever allowed to finish.
//
// So: at most one request in flight, the newest camera queued behind it, and the picture decoded
// into an offscreen Image that is swapped in only once it is complete. The view then lags the
// mouse by one render instead of blanking.
let busy = false, pending = false;
function render3d(){
 if(!d3) return;
 if(busy){ pending = true; return; }
 busy = true;
 const url = '/api/watch/render?name=' + encodeURIComponent(d3.name || '')
   + '&azim=' + d3.azim.toFixed(1) + '&elev=' + d3.elev.toFixed(1)
   + '&zoom=' + d3.zoom.toFixed(3) + '&roll=' + d3.roll
   + (d3.frame !== null ? '&frame=' + d3.frame : '') + '&cut=' + (d3.cut ? 1 : 0) + '&t=' + Date.now();
 const pre = new Image();
 pre.onload = () => {
  if(d3) document.getElementById('live3d').src = pre.src;
  busy = false;
  if(pending){ pending = false; render3d(); }
 };
 pre.onerror = () => {
  busy = false; pending = false;
  document.getElementById('d3note').textContent = ' — the render failed; see the journal';
 };
 pre.src = url;
}
(function(){
 const im = document.getElementById('live3d');
 im.addEventListener('dragstart', e => e.preventDefault());   // or the browser drags the IMAGE
 im.addEventListener('mousedown', e => { if(d3){ d3.drag = {x: e.clientX, y: e.clientY}; im.style.cursor='grabbing'; e.preventDefault(); }});
 window.addEventListener('mouseup', () => { if(d3){ d3.drag = null; document.getElementById('live3d').style.cursor='grab'; }});
 window.addEventListener('mousemove', e => {
  if(!d3 || !d3.drag) return;
  // THE PICTURE MOVES WITH THE HAND: a rightward drag turns the camera left, a downward drag lifts it. A picture
  // rolled 180 deg is turned over on screen, so both signs flip with it.
  const s = Math.cos(d3.roll * Math.PI / 180) >= 0 ? 1 : -1;
  d3.azim -= s * (e.clientX - d3.drag.x) * 0.5;
  d3.elev = Math.max(-89, Math.min(89, d3.elev + s * (e.clientY - d3.drag.y) * 0.5));
  d3.drag = {x: e.clientX, y: e.clientY};
  render3d();
 });
 im.addEventListener('wheel', e => {
  if(!d3) return;
  e.preventDefault();
  d3.zoom = Math.max(0.2, Math.min(12, d3.zoom * (e.deltaY < 0 ? 1.12 : 1/1.12)));
  render3d();
 }, {passive: false});
})();
function pane(w){
 shown = w;
 for(const k of ['plexus','why','journal','spec','caption']){
  document.getElementById(k).style.display = (k === w) ? '' : 'none';
  document.getElementById('b_' + k).className = (k === w) ? 'on' : '';
 }
 if(w === 'plexus') loadPlexus();
 tick();
}
// THE `plexus` PANE IS FETCHED, NOT POLLED. The other three panes are files read off disk and ride
// along on `state` every 1.5 s; this one parses the spec and walks the operator registry, so it
// gets its own route and is asked for only when the step it belongs to changes. `pxAt` remembers
// which step the text on screen is FOR, which is what stops a re-render on every tick and, more
// importantly, stops step N's summary being left on screen while the picture has moved to N+1.
let pxAt = null, pxWant = null;
async function loadPlexus(){
 const i = (idx < 0) ? total - 1 : idx;
 if(total === 0 || pxAt === i || pxWant === i) return;
 pxWant = i;
 const el = document.getElementById('plexus');
 if(pxAt === null) el.innerHTML = '<div class="ps-g">reading the spec…</div>';
 try{
  const j = await (await fetch('/api/watch/plexus?i=' + i + R())).json();
  // A REPLY FOR A STEP THAT IS NO LONGER SHOWN IS DROPPED. Walking the record fast fires several
  // of these and they can land out of order; without this the pane settles on whichever was
  // slowest rather than on the step being looked at.
  if(pxWant !== i) return;
  el.innerHTML = j.text || '<div class="ps-g">no spec recorded for this step</div>';
  pxAt = i;
 }catch(e){ /* the interval retries; keep whatever is on screen */ }
 finally{ if(pxWant === i) pxWant = null; }
}
function live(){ idx = -1; tick(); }
// `d` IS ANY STRIDE, which is why the ten-step buttons needed no new function: << is step(-10)
// and >> is step(10), through the same wrap. The record runs to a hundred and fifty steps and
// walking it one at a time to reach a rung from a few hours ago is the only thing it was awkward
// at.
// THE SECOND ROW WALKS ONLY THE STEPS THAT HAVE A MOVIE. `movies` is a list of positions, filled
// by every `tick`, so `d` counts MOVIES rather than steps: >> is ten movies on, not ten steps on.
// Landing on a step with no movie (a measurement graph, a camera shot) and then pressing > must
// go to the next movie AFTER it rather than skipping one, which is what the `indexOf < 0` branch
// is for -- the neighbour in the direction of travel IS the first step, so `d` is spent by one.
let movies = [];
function stepMovie(d){
 if(!movies.length) return;
 const cur = (idx < 0) ? (total - 1) : idx;
 let pos = movies.indexOf(cur);
 if(pos < 0){
  const after = movies.findIndex(m => m > cur);
  if(d > 0){ pos = (after < 0) ? 0 : after; d -= 1; }
  else { pos = (after < 0) ? movies.length - 1 : after - 1; if(pos < 0) pos = movies.length - 1; d += 1; }
 }
 pos = ((pos + d) % movies.length + movies.length) % movies.length;
 idx = movies[pos];
 tick();
}
function step(d){
 if(total === 0) return;
 if(idx < 0) idx = total - 1;           // stepping back from live starts at the newest
 // WRAPS, rather than stopping dead at either end. The record is a loop to walk, not a list with
 // walls: next past the last step returns to the first and prev before the first goes to the
 // last, so flipping through it never needs a decision about which button has stopped working.
 idx = (idx + d + total) % total;
 tick();
}
// THE PAGE SURVIVES THE SERVER RESTARTING UNDER IT, which it does often: Python caches its
// imports, so every new operator or renderer change needs the process replaced, and until now that
// left this tab dead -- a blank page, and a fresh link had to be asked for each time. Now a failed
// poll is a STATE rather than an exception: the banner says so, the picture already on screen
// stays put, and the next successful poll clears it. The URL never changes, so the tab can simply
// be left open.
let down = false;
async function tick(){
 let j;
 try{
  const want = rec;
  j = await (await fetch('/api/watch/state?i=' + idx + R())).json();
  if(want !== rec) return;                 // a reply for the record this tab has just left
 }catch(e){
  if(!down){ down = true;
   document.getElementById('meta').innerHTML =
     '<span style="color:#e08214">server restarting &mdash; this page will reconnect by itself, '
     + 'nothing to reopen</span>'; }
  return;                                  // keep whatever is on screen; the interval retries
 }
 if(down){ down = false; }
 total = j.n_shots;
 movies = j.movies || [];
 const mAt = movies.indexOf(idx < 0 ? total - 1 : idx);
 document.getElementById('mpos').textContent =
   movies.length ? (mAt >= 0 ? `movie ${mAt + 1} of ${movies.length}`
                             : `${movies.length} movies`) : 'no movies yet';
 document.getElementById('livebtn').className = (idx < 0) ? 'on' : '';
 document.getElementById('pos').textContent =
   total ? (idx < 0 ? `live — ${total} of ${total}` : `${j.index + 1} of ${total}`) : '';
 document.getElementById('meta').innerHTML =
   j.shot ? `${j.shot_name} &nbsp;&middot;&nbsp; ${j.shot_age}s ago`
          + (j.mp4 ? ' &nbsp;&middot;&nbsp; showing the movie' : '') : 'no picture yet';
 const im = document.getElementById('shot');
 document.getElementById('journal').innerHTML = j.journal;
 if(idx < 0) document.getElementById('journal').scrollTop = 1e9;
 // THE OTHER THREE FILES OF THE STEP. The why is always in view because it is the one thing a
 // picture cannot show; the spec sits behind a tab because it is 250 lines.
 document.getElementById('why').textContent = j.why || '(no why recorded for this step)';
 document.getElementById('spec').textContent = j.spec || '(no spec recorded for this step)';
 document.getElementById('caption').textContent = j.caption || '(no caption for this step -- a measurement, or archived before captions were kept per step)';
 // ... and the fourth pane, which is fetched rather than carried on `state`. Asked for on every
 // tick so that FOLLOWING THE RECORD LIVE keeps it current: `loadPlexus` returns immediately
 // unless the step on screen has actually changed.
 if(shown === 'plexus') loadPlexus();
 // THE MOVIE REPLACES THE PICTURE WHEN THERE IS ONE, rather than sitting under it. A step that
 // ran has a still of its LAST frame and a film of the whole run, and showing both makes the
 // reader compare a frame against the thing it was taken from; the film is strictly the better
 // record. A step that only built something has no film, and then the still is all there is.
 const mv = document.getElementById('mov');
 if(d3) return;                       // the 3-D view owns the panel while it is open
 if(j.mp4){
  // KEYED BY MTIME AS WELL AS POSITION, exactly as the still below is. The movie is served
  // with a day of cache so that stepping back and forth does not re-download 75 MB each
  // way, and without a buster in the URL the browser can answer a later request for the
  // same position from that cache -- which shows the wrong run under the right caption.
  const _k = rec + ':' + j.index + '.' + (j.mp4_mtime || j.shot_mtime || 0);
  if(mv.dataset.k !== _k){ mv.src = '/api/watch/mp4?i=' + j.index + '&t=' + (j.mp4_mtime || j.shot_mtime || 0) + R();
                           mv.dataset.k = _k; mv.load(); }
  mv.style.display = ''; im.style.display = 'none';
 } else {
  mv.removeAttribute('src'); mv.removeAttribute('data-i'); mv.style.display = 'none';
  im.style.display = '';
  if(j.shot) im.src = '/api/watch/shot?i=' + j.index + '&t=' + j.shot_mtime + R();
  else im.removeAttribute('src');          // an experiment with no step yet: nothing, not the last record's picture
 }
}
document.addEventListener('keydown', e => {
 if(e.key === 'ArrowLeft') step(-1);
 if(e.key === 'ArrowRight') step(1);
});
tick();
// only follow along while LIVE; a pinned picture must stay put
// `|| down` IS WHAT MAKES A PINNED VIEW RECONNECT. Following along is only wanted while LIVE, so
// a pinned picture must stay put -- but that also meant a pinned tab never polled again, and so
// never noticed the server coming back. It polls while it is down whatever it is showing.
setInterval(() => { if(idx < 0 || down) tick(); }, 2000);
</script></body></html>"""




# ---------------------------------------------------------------- the four read-only routes

def g_watch(h, q):
    """The page itself. Returned as raw HTML by the server's picture path, not as JSON."""
    return PAGE


def g_watch_state(h, q):
    """Which step is shown, the three texts beside it, and the journal."""
    folder = record_dir(q)
    fs = shots(folder)
    try:
        i = int(q.get("i", ["-1"])[0])
    except (ValueError, TypeError):
        i = -1
    i = (len(fs) - 1) if (i < 0 or i >= len(fs)) else i
    f = fs[i] if fs else None
    jr = ""
    if os.path.exists(journal_path(folder)):
        with open(journal_path(folder), errors="replace") as fh:
            lines = fh.read().splitlines()[-200:]
        if lines:
            jr = "\n".join(html.escape(x) for x in lines[:-1])
            # the newest line stands out; a watcher wants to know what is happening NOW
            jr = (jr + "\n" if jr else "") + "<b>" + html.escape(lines[-1]) + "</b>"
    # WHICH STEPS HAVE A MOVIE, so the second button row can walk those and skip the rest. A
    # measurement graph and a pre-run camera shot are steps too, and in a record of 320 they
    # outnumber the runs -- stepping one at a time to get from one movie to the next is the thing
    # the first row is bad at. Published as positions in `fs`, which is exactly what `idx` indexes.
    st = {"journal": jr, "n_shots": len(fs), "index": i if fs else 0, "rec": _rec_id(folder),
          "movies": [k for k, ff in enumerate(fs) if os.path.exists(sidecar(ff, "mp4"))]}
    if f:
        st.update(shot=True, shot_name=os.path.basename(f),
                  shot_mtime=int(os.path.getmtime(f)),
                  shot_age=int(time.time() - os.path.getmtime(f)),
                  why=_read(sidecar(f, "why")), spec=_read(sidecar(f, "spec")),
                  # THE CAPTION, READ, NOT COMPUTED. The VLM runs once, when the step is archived,
                  # and writes `caption/NNNN.txt`; the watcher only ever reads it. Computing it here
                  # would cost a minute of GPU per page load and give a different text each time.
                  caption=_read(sidecar(f, "caption")),
                  mp4=os.path.exists(sidecar(f, "mp4")),
                  # THE MOVIE'S OWN TIME, so a movie re-rendered under an unchanged still is fetched
                  # afresh: the page keyed the <video> on the still's mtime, and a re-render (exp_02
                  # step 0052, the shells closed) played from the browser's cache as the old file.
                  mp4_mtime=(int(os.path.getmtime(sidecar(f, "mp4"))) if os.path.exists(sidecar(f, "mp4")) else 0))
    return st


def _nth(q, kind):
    fs = shots(record_dir(q))
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


ICON_DIR = os.path.join(REPO, "presentation", "icons")
EQ_CACHE = os.path.join(REPO, "log", "watch_eq")
# THE EQUATION'S PRINT SIZE, as dvipng's resolution. 105 set them larger than the text around them
# (user, 2026-09-25: "diminish the size"); 80 sits them at the pane's reading size. It is part of
# the cache key and of the image URL, so a change here is never served from an old picture.
EQ_DPI = 80
# THE KINDS THE DECK HAS AN ICON FOR. A whitelist and not a path join, because `kind` arrives in a
# query string: `?kind=../../etc/passwd` must reach nothing. It is also the honest list -- a kind
# with no icon draws nothing rather than a broken image.
ICON_KINDS = ("aggregate", "broadcast", "die", "divide", "exchange", "field", "field_dyn",
              "lateral", "rewire", "seed", "set", "structural")


def eq_png(tex: str) -> str | None:
    """A LaTeX equation as a transparent PNG, white on nothing, cached on disk. Path, or None.

    WHY LATEX AND NOT THE BROWSER. There is no network in this container (a CDN fetch returns
    nothing at all) and no KaTeX or MathJax vendored in the tree, so a page cannot typeset maths
    on its own. There IS a full TeX here, because the deck is built with it -- `latex` and
    `dvipng` are both on PATH -- so the pane sets its equations with the very same typesetter the
    slides use, and the two cannot look different.

    WHY THE TEX NEVER COMES FROM THE QUERY STRING. This shells out to `latex`. The route addresses
    an equation by WHICH ONE IT IS -- step i, equation k -- and looks the body up server-side from
    the spec, so nothing a browser sends is ever compiled.

    Cached under `log/watch_eq/<sha1>.png`: the same equation appears on many steps, and a spec
    re-opened is the commonest thing a watcher does.
    """
    import hashlib
    import shutil
    import subprocess
    import tempfile
    if not tex.strip():
        return None
    os.makedirs(EQ_CACHE, exist_ok=True)
    out = os.path.join(EQ_CACHE, hashlib.sha1(f"{EQ_DPI}:{tex}".encode()).hexdigest()[:20] + ".png")
    if os.path.exists(out):
        return out
    # ONE UNBREAKABLE BOX ON A WIDE PAGE. Set as bare inline maths in an article of default text
    # width, a long equation was broken at a relation like a sentence -- May-Leonard's came out as
    # "(1 - p -" on one line and "a u)" on the next, which reads as two wrong formulas. `\mbox`
    # forbids the break and a 100 cm page gives it room; `-T tight` then crops to the ink, so the
    # page size never shows.
    doc = (r"\documentclass[12pt]{article}\usepackage{amsmath,amssymb,bm}"
           r"\usepackage[paperwidth=100cm,paperheight=30cm,margin=1cm]{geometry}"
           r"\pagestyle{empty}\begin{document}"
           r"\mbox{$\displaystyle " + tex + r"$}\end{document}")
    tmp = tempfile.mkdtemp(prefix="plexus_eq_")
    try:
        with open(os.path.join(tmp, "e.tex"), "w") as f:
            f.write(doc)
        r = subprocess.run(["latex", "-interaction=nonstopmode", "-halt-on-error", "e.tex"],
                           cwd=tmp, capture_output=True, text=True, timeout=40)
        if not os.path.exists(os.path.join(tmp, "e.dvi")):
            print(f"[watch] latex failed on an equation: {(r.stdout or '')[-300:]}", flush=True)
            return None
        # -bg Transparent + -fg white: the watcher's page is black, and a PNG with a white box
        # around every equation is what a default render gives.
        # ONE DPI FOR EVERY EQUATION, AND THE PAGE NEVER RESCALES ONE. At 220 the images ran from
        # 308 to 1194 px wide, and `max-width:100%` then shrank the wide ones to the pane while
        # leaving the narrow ones untouched -- so a rewire's short formula was set twice the size of
        # the lateral's long one, on the same slide. A uniform font is a uniform SCALE, which means
        # no CSS size constraint at all: rendered here at a dpi whose widest output fits a pane, and
        # displayed at its natural pixel size. A formula wider than the pane scrolls rather than
        # shrinking, because shrinking it is what made the two disagree.
        subprocess.run(["dvipng", "-D", str(EQ_DPI), "-T", "tight", "-bg", "Transparent",
                        "-fg", "rgb 1.0 1.0 1.0", "-o", "e.png", "e.dvi"],
                       cwd=tmp, capture_output=True, text=True, timeout=40)
        src = os.path.join(tmp, "e.png")
        if not os.path.exists(src):
            return None
        shutil.move(src, out)
        return out
    except Exception as e:                                       # noqa: BLE001
        print(f"[watch] equation render unavailable ({type(e).__name__}: {e})", flush=True)
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_SUMMARISED: dict = {}


def _summarise(path: str) -> dict:
    """`spec_summary.summarise(path)`, and when THIS process's operator registry is older than the spec,
    the same call in a fresh interpreter.

    THE REGISTRY IS FROZEN AT THE WATCHER'S START. `plexus.operators` registers every operator once,
    when it is first imported, and a registration cannot be repeated (`register_operator` refuses a
    second variant of the same name). A watcher left running while a session adds operators -- exp04
    added `shape_match`, `slab_barrier`, `residue_slab` and `friction_zone` to `channel_ops.py` after
    its watcher started -- then failed every new spec with "operator 'shape_match' not in registry",
    while the same summary in a fresh process read all 75 operators. So a registry miss is answered
    by a subprocess that imports the library as it is NOW; its JSON is rendered here as before, and
    cached on the spec's mtime like the rest, so it is paid once per step.
    """
    key = (path, os.path.getmtime(path))
    hit = _SUMMARISED.get(path)
    if hit is not None and hit[0] == key:
        return hit[1]
    try:
        s = _summary_module().summarise(path)
    except Exception as e:                                       # noqa: BLE001
        if "not in registry" not in str(e) and "no variant" not in str(e):
            raise
        import subprocess
        import sys as _sys
        code = ("import importlib.util, json, sys; "
                "p = sys.argv[1]; t = sys.argv[2]; "
                "spec = importlib.util.spec_from_file_location('plexus_spec_summary', t); "
                "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
                "print('@@JSON@@' + json.dumps(m.summarise(p), default=str))")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(REPO, "src") + os.pathsep + env.get("PYTHONPATH", "")
        r = subprocess.run([_sys.executable, "-c", code, path, os.path.join(REPO, "tools", "spec_summary.py")],
                           capture_output=True, text=True, timeout=300, cwd=REPO, env=env)
        line = next((ln for ln in r.stdout.splitlines() if ln.startswith("@@JSON@@")), None)
        if line is None:
            raise RuntimeError(f"{e} -- and a fresh interpreter could not summarise it either: "
                               f"{(r.stderr or '').strip().splitlines()[-1:] or ['no output']}") from e
        s = json.loads(line[len("@@JSON@@"):])
    _SUMMARISED[path] = (key, s)
    return s


def _summary_module():
    """`tools/spec_summary.py`, imported by path and kept, so the registry walk is paid once."""
    m = globals().get("_SPEC_SUMMARY_MOD")
    if m is None:
        import importlib.util
        p = os.path.join(REPO, "tools", "spec_summary.py")
        spec = importlib.util.spec_from_file_location("plexus_spec_summary", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        globals()["_SPEC_SUMMARY_MOD"] = m
    return m


def g_watch_plexus(h, q):
    """`/api/watch/plexus?i=N` -- step N's spec, read as the paper's three questions.

    A ROUTE OF ITS OWN RATHER THAN A FIELD ON `state`, because `state` is polled every 1.5 s and
    this is the only thing the watcher serves that costs more than reading a file. Read-only like
    every other route here: it parses a spec and queries the registry, and writes nothing.
    """
    fs = shots(record_dir(q))
    try:
        i = int(q.get("i", ["-1"])[0])
    except (ValueError, TypeError):
        i = -1
    i = (len(fs) - 1) if (i < 0 or i >= len(fs)) else i
    rec = ((q or {}).get("rec", [""])[0] or "").strip()
    return {"text": spec_summary(_nth(q, "spec"), q_index=i, rec=rec)}


def g_watch_records(h, q):
    """`/api/watch/records` -- every record this watcher can show, and the one it opens on."""
    return {"records": [{k: r[k] for k in ("id", "root", "name", "n")} for r in records()],
            "default": _rec_id(HISTORY)}


def watch_icon(q) -> str | None:
    """The deck's glyph for an operator kind, with its background made transparent. Path, or None.

    THE FILES ARE OPAQUE BLACK BEHIND THE ART. They are drawn for a black slide, so on a slide the
    square is invisible and nobody has had to care. On this page the pane is #0f1319 rather than
    #000, so the square showed as a dark tile around every icon; inverting it in CSS -- the first
    attempt -- turned that black background WHITE and made the tile worse.

    So the alpha is DERIVED FROM THE ART instead: a pixel's opacity is its own brightness, which
    takes the background to nothing, keeps the coloured strokes exactly as drawn, and keeps the
    anti-aliased edges soft rather than stair-stepped. Done here, once, and cached, rather than in
    the browser, because no CSS filter can turn a colour into transparency.

    The deck's own files are untouched: a slide still gets the black square it wants.
    """
    kind = (q.get("kind", [""])[0] or "").strip()
    if kind not in ICON_KINDS:
        return None
    src = os.path.join(ICON_DIR, kind + ".png")
    if not os.path.exists(src):
        return None
    out = os.path.join(EQ_CACHE, f"icon_{kind}.png")
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src):
        return out
    try:
        import numpy as np
        from PIL import Image
        a = np.array(Image.open(src).convert("RGBA"))
        a[..., 3] = a[..., :3].max(axis=2)          # opacity = brightness: black -> nothing
        os.makedirs(EQ_CACHE, exist_ok=True)
        Image.fromarray(a).save(out)
        return out
    except Exception as e:                                       # noqa: BLE001
        print(f"[watch] icon {kind}: {type(e).__name__}: {e}", flush=True)
        return src                                   # the original is better than no icon at all


def watch_eq(q) -> str | None:
    """`/api/watch/eq?i=N&k=K` -- the K-th equation of step N's spec, typeset, as a PNG path.

    ADDRESSED BY POSITION, NOT BY CONTENT. The body is looked up here, from the spec the record
    already holds, so no LaTeX ever arrives from the browser -- which matters because rendering it
    means running `latex`. `spec_summary.equations_shown` walks the same dedup the HTML renderer
    walks, so the K-th image is the K-th equation that renderer emitted.
    """
    p = _nth(q, "spec")
    if not p or not os.path.exists(p):
        return None
    try:
        k = int(q.get("k", ["0"])[0])
        eqs = _summary_module().equations_shown(_summarise(p))
        return eq_png(eqs[k]) if 0 <= k < len(eqs) else None
    except Exception as e:                                       # noqa: BLE001
        print(f"[watch] equation {q.get('k')} of {p}: {type(e).__name__}: {e}", flush=True)
        return None
