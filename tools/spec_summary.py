"""Summarise a Plexus spec in three parts -- header, entities, operators -- as text or LaTeX.

    python tools/spec_summary.py config/tissue/sheet_morphogen_die.yaml
    python tools/spec_summary.py <spec> --format tex
    python tools/spec_summary.py <spec> --format tex --frame --movie Movies/sheet_die \
        --title "A sheet that dies where the morphogen is high" --out slides/

The three parts are the three questions a spec answers, in the order the paper asks them:

    HEADER      what is being run -- name, dimension, box, timestep, how long, what scale
    ENTITIES    what exists -- the sets, how many of each, what state each carries, and the
                containment and relations between them
    OPERATORS   what happens -- the activities in SCHEDULE order, each with its equation and
                the parameter values THIS spec gives it

Everything factual is read from the spec and from the live operator registry, so a summary
cannot drift from either. The equations come from one of three places, in this order, and the
source is always reported so a reader knows which they are looking at:

    1. `equation=` on the operator's own `@register_operator(...)`, stamped as `cls.EQUATION`
       exactly as `title=` is -- the operator carries its own maths;
    2. the hand-authored LaTeX in `scripts/build_library.py:ENRICH`, the equation the operator
       library page shows, so the deck and the site cannot disagree;
    3. the indented display maths in the operator's own docstring, transliterated from ASCII;
    4. no equation -- the first sentence of the docstring stands in, and the slide says so.

`--frame` emits a whole beamer frame in the deck convention: the movie on the left, this
summary on the right, so a slide is one command and no prose is written by hand.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import yaml                                     # noqa: E402

import plexus                                   # noqa: E402,F401
import plexus.operators                         # noqa: E402,F401  self-registers the library
import plexus.models.entities                   # noqa: E402,F401
from plexus import schema                       # noqa: E402
from plexus.models import registry as R         # noqa: E402


# --------------------------------------------------------------------------- #
#  The library's own equation table, imported rather than copied
# --------------------------------------------------------------------------- #
def _enrich() -> dict:
    """`ENRICH` from scripts/build_library.py -- the site's hand-authored equations.

    Imported by path, not by package, because scripts/ is not importable; the table is the
    single source of the LaTeX so the library page and a slide always show the same equation.
    """
    path = os.path.join(ROOT, "scripts", "build_library.py")
    spec = importlib.util.spec_from_file_location("_plexus_build_library", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "ENRICH", {})


ENRICH = _enrich()

# The elementary families of the algebra of activities (paper Figure 1), with the symbol each
# is written with there, so a slide names an operator's family the way the paper does.
KIND_SYMBOL = {
    "lateral":    r"\mathcal{O}_E",
    "aggregate":  r"\textstyle\sum_\pi",
    "broadcast":  r"\pi^{*}",
    "exchange":   "push/pull",
    "field":      r"\partial_t\phi",
    "rewire":     r"\mathcal{R}\!:\!E",
    "structural": r"|S|",
    "seed":       r"x_0",
}
# THE NINE KINDS: eight activities and the seed (paper Figure 1). `divide` and `die` are
# separate kinds in the registry and separate glyphs in the figure, which draws the bracket
# "structural" around the pair; an operator registered under that umbrella kind is listed
# beside them rather than silently folded into one or the other.
KIND_LABEL = {
    "lateral": "lateral", "aggregate": "aggregate", "broadcast": "broadcast",
    "exchange": "exchange", "field": "field", "rewire": "rewire", "divide": "divide",
    "die": "die", "structural": "structural", "seed": "seed",
}


def group_by_kind(ops: list[dict]) -> list[tuple[str, list[dict]]]:
    """Operators gathered under their kind, in the order the SCHEDULE reaches each kind.

    Not the order of Figure 1. A reader follows a run: the seed establishes the entities, a
    rewire then decides who neighbours whom, and only then does a lateral activity have a
    relation to act over. Sorting by the figure's order would print that backwards, which is why
    the grouping follows first appearance in the operator list -- seeds, then schedule order.
    """
    by = {}
    for o in ops:
        by.setdefault("seed" if o["phase"] == "seed" else (o["kind"] or "lateral"), []).append(o)
    return list(by.items())                      # dicts keep insertion order: first use wins


KIND_GLOSS = {
    "lateral":    "within-set interaction over a relation",
    "aggregate":  "children to parent, up the containment",
    "broadcast":  "parent to children, down the containment",
    "exchange":   "set and field exchange",
    "field":      "a field's own dynamics",
    "rewire":     "rebuild the relation each tick",
    "structural": "change the entities themselves",
    "seed":       "establish the initial condition",
}


# --------------------------------------------------------------------------- #
#  Docstring maths: find it, and turn the ASCII into LaTeX
# --------------------------------------------------------------------------- #
# An indented docstring line is DISPLAY MATHS when it states a relation (it has an `=`, a `<-`
# or a `~`) and carries at least one mathematical token. YAML lines (`key: value`) and prose
# are rejected first. This is deliberately looser than build_library._is_ascii_math, which only
# needs to DROP such lines from prose; here they are the thing being kept.
_YAMLISH = re.compile(r"^\s*[\w.]+:\s")
_MATH_TOKEN = re.compile(
    r"[Σ∑²³·×⊗≈→√∇∂]|\bsum_|\bexp\(|\bsqrt|\bgrad\b|\bd[a-z]+/d[a-z]+|"
    r"[*/^]|\b[A-Za-z]_[A-Za-z0-9{]|\|\S+\||\\[a-z]+")
# Prose gives itself away by its function words. A docstring line can hold an `=` and still be a
# sentence ("force = -grad E, overdamped, no active term -- and"), so a line carrying three or
# more of these is rejected however mathematical the rest of it looks.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "this", "that", "which", "and", "or", "not", "but",
    "with", "from", "its", "it", "of", "to", "in", "on", "by", "as", "be", "so", "no", "for",
    "at", "than", "then", "when", "what", "does", "do", "has", "have", "each", "every", "same",
    "other", "there", "they", "their", "one", "two", "into", "over", "under", "because",
}


def _is_prose(s: str) -> bool:
    """True when a fragment reads as a sentence: three or more function words of two letters up.

    Single letters are never counted, because `a` in `a = activator` is a variable, not an article.
    """
    return sum(w in _STOPWORDS for w in re.findall(r"[A-Za-z]{2,12}", s.lower())) >= 3


def _strip_annotation(s: str) -> str:
    """Drop the trailing annotation -- what follows a 3-space gutter once the equation is complete.

    A gutter only separates an annotation from the equation once the equation has actually been
    stated, so nothing is cut until a relation symbol has been seen: `A_f   = (1/2)|...|` is one
    equation whose gutter is alignment, while `da/dt = ...   a = activator, and its own
    autocatalyst` is an equation followed by a gloss. A kept tail must carry no function word at
    all, which is what separates `u = substrate` from a sentence about it.
    """
    parts = re.split(r"\s{3,}", s)
    kept, stated = [parts[0]], bool(re.search(r"=|<-|->", parts[0]))
    for p in parts[1:]:
        if not stated:                                   # still inside the equation
            kept.append(p)
            stated = bool(re.search(r"=|<-|->", p))
            continue
        words = re.findall(r"[A-Za-z]{2,12}", p.lower())
        if re.search(r"=|<-|->", p) and not any(w in _STOPWORDS for w in words) \
                and len(p.split()) <= 6:
            kept.append(p)
        else:
            break
    return "   ".join(kept).rstrip(" ,")


def docstring_math(doc: str, max_lines: int = 3) -> list[str]:
    """The indented display-maths lines of a docstring, in order, at most `max_lines` of them."""
    out = []
    for ln in (doc or "").split("\n"):
        indent = len(ln) - len(ln.lstrip())
        s = ln.strip()
        if indent < 4 or not s or _YAMLISH.match(ln):
            continue
        if ("=" not in s and "<-" not in s and "->" not in s) or not _MATH_TOKEN.search(s):
            continue
        s = _strip_annotation(s)                         # the prose tail goes before the test
        if _is_prose(s):                                 # a sentence, not an equation
            continue
        if s.endswith(".") and len(s.split()) > 6:
            continue
        out.append(s)
        if len(out) >= max_lines:
            break
    return out


# ASCII -> LaTeX, longest key first so `sum_` is not eaten by `s`. Only tokens that are
# unambiguous in this codebase's docstrings are listed; anything unlisted is passed through,
# which is why the result is always checked by eye in the compiled deck.
_TRANSLIT = [
    (r"\bsum_\{([^}]*)\}", r"\\sum_{\1}"), (r"\bsum_(\w+)", r"\\sum_{\1}"), (r"\bsum\b", r"\\sum"),
    (r"\bsqrt\(([^)]*)\)", r"\\sqrt{\1}"), (r"\bsqrt\b", r"\\sqrt"),
    (r"\bexp\(([^)]*)\)", r"e^{\1}"),
    (r"\bd([A-Za-z]\w*)/d([A-Za-z]\w*)", r"\\tfrac{d\1}{d\2}"),
    (r"\bgrad\b", r"\\nabla"), (r"\bLambda\b", r"\\Lambda"), (r"\blambda\b", r"\\lambda"),
    (r"\bDelta\b", r"\\Delta"), (r"\bdelta\b", r"\\delta"), (r"\bsigma\b", r"\\sigma"),
    (r"\bzeta\b", r"\\zeta"), (r"\btau\b", r"\\tau"), (r"\bomega\b", r"\\omega"),
    (r"\brho\b", r"\\rho"), (r"\beta\b", r"\\eta"), (r"\bmu\b", r"\\mu"), (r"\bnu\b", r"\\nu"),
    (r"\bchi\b", r"\\chi"), (r"\bphi\b", r"\\phi"), (r"\btheta\b", r"\\theta"),
    (r"\bpi\b", r"\\pi"), (r"\bepsilon\b", r"\\epsilon"), (r"\bkappa\b", r"\\kappa"),
    (r"\bdeg\(", r"\\deg("), (r"\brelu\b", r"\\mathrm{relu}"), (r"\bmin\b", r"\\min"),
    (r"\bmax\b", r"\\max"), (r"\bmean\b", r"\\mathrm{mean}"), (r"\bclamp\b", r"\\mathrm{clamp}"),
    (r"<->", r"\\leftrightarrow"), (r"->", r"\\to"), (r"<-", r"\\leftarrow"),
    (r"\s~\s", r" \\sim "), (r"\bhat\b", r"\\hat"),
]
_SUB = re.compile(r"(?<=[A-Za-z0-9])_([A-Za-z0-9]{2,})\b")      # v_eq -> v_{eq}


_WORD = re.compile(r"(?<![\\{])\b([A-Za-z]{3,})\b(?![}])")


def translit(s: str) -> str:
    """ASCII docstring maths to inline LaTeX. Conservative: unknown tokens pass through.

    A docstring writes its maths for a terminal, so it says `sum_f`, ` x ` for a cross product
    and spells whole words inside the formula. Left alone in maths mode those words set as a
    product of italic letters, so every alphabetic run of three letters or more that is not
    already a macro is wrapped upright in `\\mathrm`.
    """
    s = s.replace("\\", " ")
    prefix = re.match(r"^(.{0,80}?[a-z]{3,}[^:]*):\s+(?=\S)", s)   # a prose lead-in, then the maths
    if prefix and _is_prose(prefix.group(1)):
        s = s[prefix.end():]
    for pat, rep in _TRANSLIT:
        s = re.sub(pat, rep, s)
    s = re.sub(r"(?<= )in(?= )", r"\\in", s)
    s = re.sub(r"(?<= )x(?= )", r"\\times", s)
    s = _SUB.sub(r"_{\1}", s)
    s = re.sub(r"\^(\w+)", r"^{\1}", s)
    s = _WORD.sub(lambda m: r"\mathrm{%s}" % m.group(1), s)
    s = s.replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
    s = re.sub(r"\s{2,}", r" \\quad ", s.strip()).rstrip(" .")
    return s


def equation_for(name: str, cls) -> tuple[str, str, list[str]]:
    """(LaTeX, provenance, the ASCII lines) for one operator.

    The ASCII lines are returned alongside so the text renderer prints exactly what the LaTeX was
    made from -- re-deriving them from the registry would read a different class whenever the
    spec selected an implementation variant, and the two outputs would disagree.
    """
    declared = getattr(cls, "EQUATION", "") if cls else ""
    enriched = (ENRICH.get(name) or {}).get("equation")
    lines = docstring_math(cls.__doc__ or "")
    if declared:                                    # the operator's own, and it outranks the rest
        return declared.strip(), "registered", lines
    if enriched:
        return enriched.strip(), "library", lines
    if lines:
        return "", "docstring", lines
    return "", "none", []


# --------------------------------------------------------------------------- #
#  Reading the spec
# --------------------------------------------------------------------------- #
def prettify(name: str) -> str:
    """`cell_chem_diffuse` -> `Cell chem diffuse`: the fallback when an operator has no title."""
    t = name.replace("_", " ").strip()
    return t[:1].upper() + t[1:]


def lower_first(text: str) -> str:
    """Lowercase the opening letter, unless the first word is a name or an acronym.

    Everything in the right-hand column is a LABEL, not a sentence, and a column of labels that
    each start with a capital reads as a list of headings. So `Build the cell mesh` is set as
    `build the cell mesh` -- while `T1 neighbour exchange`, `MPM` and `Gray-Scott` keep their
    capitals, which are part of the name rather than the start of a sentence.
    """
    if not text:
        return text
    head = text.split()[0]
    if head[1:] != head[1:].lower() or head.isupper():     # T1, MPM, Gray-Scott
        return text
    return text[0].lower() + text[1:]


def op_title(name: str, cls) -> str:
    """The name an operator is CALLED, declared at registration as `title=`.

    `register_operator(**tags)` stamps every tag as an upper-case class attribute, so a title
    needs no registry change -- only the word. An operator that has not been given one falls
    back to its identifier with the underscores taken out, which reads as a machine name and is
    meant to: `--audit` lists exactly those, so the gap is visible instead of silent.
    """
    return (getattr(cls, "TITLE", "") or prettify(name)) if cls else prettify(name)


def first_sentence(doc: str) -> str:
    """The opening sentence of a docstring, however many lines it wraps over."""
    para = []
    for ln in (doc or "").strip().split("\n"):
        if not ln.strip():
            break
        para.append(ln.strip())
    text = re.sub(r"^[\w<>./()-]+\s*(--|—|:)\s*", "", " ".join(para))
    m = re.search(r"(?<![A-Z])(?<!\b[A-Za-z])\.(?=\s+[A-Z(`]|$)", text)
    if m:
        text = text[:m.end()]
    return (text[:1].upper() + text[1:]) if text else ""


def set_count(name: str, sets: dict, seen: tuple = ()) -> int | None:
    """How many entities a set holds, following `parent`/`per_parent` up the containment.

    A child set states its size per parent, so its own size is that number times the parent's --
    the containment is what makes the count, and reading `n:` alone under-reports a child set by
    the parent's multiplicity. Returns None when the count is not declared (a seed operator
    builds it), rather than guessing.
    """
    if name in seen:                                    # a cycle: refuse rather than recurse
        return None
    s = sets.get(name) or {}
    if "per_parent" in s and s.get("parent"):
        p = set_count(s["parent"], sets, seen + (name,))
        return None if p is None else p * int(s["per_parent"])
    if "n" in s:
        return int(s["n"])
    return None


# A growing tissue declares `n:` as a CAPACITY -- the buffer division and death are allowed to
# fill -- while the seed says how many entities actually exist at frame 0. Reporting the buffer
# alone puts "65,004 cells" beside a movie of a 2,000-cell vesicle, so both are read and the
# slide prints the seeded count first. Only the `cell` set is looked up this way, from the one
# parameter name that means it across the library; nothing else is guessed.
SEEDED_FROM = {"cell": ("n_cells",)}


def seeded_counts(raw: dict) -> dict:
    """set name -> how many entities its seed operator establishes, where a seed says so."""
    out = {}
    for o in (raw.get("seed") or []) + (raw.get("operators") or []):
        if not isinstance(o, dict):
            continue
        for set_name, keys in SEEDED_FROM.items():
            for k in keys:
                if k in o and set_name in (raw.get("sets") or {}):
                    out[set_name] = int(o[k])
    return out


def read_sets(raw: dict) -> list[dict]:
    """One row per set: its size, what it is, its containment, its relation, its state."""
    sets = raw.get("sets") or {}
    seeded = seeded_counts(raw)
    rows = []
    for name, s in sets.items():
        maps = s.get("maps") or {}
        state = s.get("state") or {}
        types = s.get("types") or {}
        rows.append(dict(
            name=name,
            n=set_count(name, sets),
            seeded=seeded.get(name),
            entity=s.get("entity") or s.get("mesh") or "",
            parent=s.get("parent") or "",
            per_parent=s.get("per_parent"),
            relation=(maps.get("srce"), maps.get("trgt")) if maps else None,
            state=[dict(name=k,
                        role=(v or {}).get("role", ""),
                        width=(v or {}).get("width"),
                        integration=(v or {}).get("integration", ""))
                   for k, v in state.items()],
            types=list(types),
        ))
    return rows


def read_fields(raw: dict) -> list[dict]:
    out = []
    for name, f in (raw.get("fields") or {}).items():
        f = f or {}
        out.append(dict(name=name, frame=f.get("frame", ""),
                        n_grid=f.get("n_grid") or f.get("n") or "",
                        params={k: v for k, v in f.items() if k not in ("frame", "n_grid")}))
    return out


def schedule_order(raw: dict) -> list[tuple[str, float | None]]:
    """The schedule flattened to (operator name, substep dt) in the order it runs."""
    out = []
    for step in (raw.get("schedule") or []):
        if isinstance(step, dict) and "steps" in step:
            dt = step.get("substep_dt")
            out.extend((s, dt) for s in step["steps"])
        elif isinstance(step, str):
            out.append((step, None))
    return out


def read_operators(raw: dict) -> list[dict]:
    """One row per operator, in schedule order, seeds first, with its equation and parameters."""
    declared = {}
    for o in (raw.get("operators") or []):
        declared.setdefault(o.get("op"), []).append(o)
    rows = []

    def row(o: dict, phase: str, substep_dt=None) -> dict:
        name = o.get("op")
        try:
            cls = R.get_operator(name, o.get("implementation") or o.get("model"))
        except Exception:
            cls = R._OPERATOR_REGISTRY.get(name)
        kind = getattr(cls, "KIND", "") if cls else ""
        eq, prov, math = equation_for(name, cls) if cls else ("", "none", [])
        roles = dict(getattr(cls, "PARAM_ROLES", {}) or {}) if cls else {}
        required = list(getattr(cls, "REQUIRES_PARAMS", []) or []) if cls else []
        params = [dict(name=k, value=v, role=roles.get(k, ""), required=k in required)
                  for k, v in o.items()
                  if k not in ("op", "at", "to", "from", "emit", "implementation", "model")]
        return dict(name=name, title=op_title(name, cls), kind=kind, phase=phase,
                    probe=bool(getattr(cls, "PROBE", False)) if cls else False,
                    at=o.get("at", ""),
                    to=o.get("to"), frm=o.get("from"), emit=o.get("emit"),
                    impl=o.get("implementation") or o.get("model"),
                    substep_dt=substep_dt, equation=eq, provenance=prov, math=math,
                    gloss=first_sentence(cls.__doc__ or "") if cls else "",
                    params=params)

    for o in (raw.get("seed") or []):
        rows.append(row(o, "seed"))
    for name, dt in schedule_order(raw):
        for o in declared.pop(name, [{"op": name}]):
            rows.append(row(o, "schedule", dt))
    for name, os_ in declared.items():                  # declared but never scheduled
        for o in os_:
            rows.append(row(o, "unscheduled"))
    return rows


def human_time(seconds: float) -> str:
    """A duration a reader can picture, with the seconds kept beside it."""
    for scale, unit in ((86400.0, "day"), (3600.0, "hour"), (60.0, "minute")):
        if seconds >= 2 * scale:
            return f"{seconds / scale:.3g} {unit}s ({fmt_num(seconds)} s)"
    if seconds < 1e-3:
        return f"{seconds * 1e3:.3g} ms"
    return f"{fmt_num(seconds)} s"


def summarise(path: str) -> dict:
    """The whole summary: header, entities, fields, operators."""
    raw = yaml.safe_load(open(path)) or {}
    with contextlib.redirect_stdout(sys.stderr):        # the loader's `[units]` note is not output
        sp = schema.load(path)                          # validates; raises on a malformed spec
    g = raw.get("general") or {}
    units = g.get("units") or {}
    n_rec = min(sp.n_frames, sp.record_cap)
    sets = read_sets(raw)
    total = sum(r["n"] for r in sets if r["n"] and not r["relation"])
    um = float(units["length_um"]) if units.get("length_um") else None
    return dict(
        path=os.path.relpath(path, ROOT),
        header=dict(
            name=sp.name, title=sp.title, dim=sp.dim, world=sp.world_size, dt=sp.dt,
            n_frames=sp.n_frames,
            recorded=n_rec, boundary=sp.boundary, seed=sp.seed, engine=sp.engine,
            units=units,
            seconds=(sp.n_frames * sp.dt * float(units.get("time_s", 1.0))) if units else None,
            um_per_unit=um,
            box_um=[w * um for w in sp.world_size] if um else None,
            n_entities=total,
        ),
        sets=sets, fields=read_fields(raw), operators=read_operators(raw),
        plotting=raw.get("plotting") or {},
    )


# --------------------------------------------------------------------------- #
#  Rendering: text
# --------------------------------------------------------------------------- #
def fmt_num(v) -> str:
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.3g}"
        return f"{v:g}"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fmt_num(x) for x in v) + "]"
    return str(v)


def render_text(s: dict, max_ops: int | None = None) -> str:
    h, out = s["header"], []
    out.append(h["name"])
    out.append("=" * len(h["name"]))
    box = " x ".join(fmt_num(w) for w in h["world"])
    line = (f"{h['dim']}D, box {box}, dt {fmt_num(h['dt'])}, {h['n_frames']:,} frames "
            f"({h['recorded']:,} recorded), {h['boundary']} boundary, seed {h['seed']}")
    out.append(line)
    if h["seconds"] is not None:
        scale = human_time(h["seconds"]) + " of physical time"
        if h["box_um"]:
            scale += ", box " + " x ".join(fmt_num(w) for w in h["box_um"]) + " um"
        out.append(scale)
    else:
        out.append("no units declared -- the run is dimensionless")
    out.append(f"{h['n_entities']:,} entities in {len([r for r in s['sets'] if not r['relation']])} sets, "
               f"{len(s['operators'])} operators")
    out.append(f"spec: {s['path']}")

    out.append("\nENTITIES AND CONTAINMENT")
    for r in s["sets"]:
        if r["relation"]:
            out.append(f"  {r['name']:<18} relation  {r['relation'][0]} -> {r['relation'][1]}"
                       + (f", {fmt_num(r['n'])} of them" if r["n"] else ""))
            continue
        n = fmt_num(r["n"]) if r["n"] is not None else "seeded"
        head = f"  {r['name']:<18} {n:>10}"
        if r["entity"]:
            head += f"  {r['entity']}"
        if r["parent"]:
            head += f"  [{fmt_num(r['per_parent'])} per {r['parent']}]"
        out.append(head)
        if r["state"]:
            shown = r["state"][:6]
            txt = ", ".join(f"{v['name']}" + (f" ({v['role']})" if v["role"] else "")
                            + (f" x{v['width']}" if (v["width"] or 1) > 1 else "")
                            for v in shown)
            if len(r["state"]) > len(shown):
                txt += f", +{len(r['state']) - len(shown)} more"
            out.append("      state: " + txt)
        if r["types"]:
            out.append("      types: " + ", ".join(r["types"]))
    for f in s["fields"]:
        out.append(f"  {f['name']:<18} {'field':>10}  frame {f['frame']}, grid {f['n_grid']}")

    out.append("\nOPERATORS AND EQUATIONS")
    ops = s["operators"][:max_ops] if max_ops else s["operators"]
    for o in ops:
        tag = {"seed": "seed", "unscheduled": "declared, not scheduled"}.get(o["phase"], o["kind"])
        head = f"  {o['name']:<22} {tag:<12} at {o['at']}"
        if o["to"]:
            head += f" -> {o['to']}"
        if o["frm"]:
            head += f" <- {o['frm']}"
        if o["emit"]:
            head += f", emits {o['emit']}"
        if o["impl"]:
            head += f" [{o['impl']}]"
        out.append(head)
        if o["math"]:
            for ln in o["math"]:
                out.append(f"      {ln}")
        elif o["gloss"]:
            out.append(f"      {o['gloss']}")
        if o["params"]:
            out.append("      " + ", ".join(
                f"{p['name']}={fmt_num(p['value'])}" + (f" ({p['role']})" if p["role"] else "")
                for p in o["params"]))
    if max_ops and len(s["operators"]) > max_ops:
        out.append(f"  ... and {len(s['operators']) - max_ops} more")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
#  Rendering: LaTeX
# --------------------------------------------------------------------------- #
def tex_escape(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_")
            .replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
            .replace("$", r"\$").replace("^", r"\^{}"))


def containment(sets: list[dict], fields: list[dict]) -> list[dict]:
    """The sets as a TREE, because containment is the thing a flat table loses.

    A child set declares `parent:` and `per_parent:`, so it is printed one level in under its
    parent, carrying the multiplicity that makes its own count: 5,000,000 material points *per*
    the one cell is a different claim from two sets that happen to have those sizes. A relation
    set is printed at the level of the set it maps, as an edge between entities rather than as a
    population of its own. Fields close the list: they are not entities and are not counted as any.
    """
    by_name = {r["name"]: r for r in sets}
    kids = {}
    for r in sets:
        if r["parent"] and r["parent"] in by_name:
            kids.setdefault(r["parent"], []).append(r)
    rows = []

    def walk(r: dict, depth: int) -> None:
        rows.append(dict(kind="set", depth=depth, row=r))
        for rel in sets:                                  # a relation sits under what it maps
            if rel["relation"] and rel["relation"][0] == r["name"] and rel is not r:
                rows.append(dict(kind="relation", depth=depth + 1, row=rel))
        for k in kids.get(r["name"], []):
            walk(k, depth + 1)

    for r in sets:
        if r["relation"] or (r["parent"] and r["parent"] in by_name):
            continue
        walk(r, 0)
    for f in fields:
        rows.append(dict(kind="field", depth=0, row=f))
    return rows


def render_tex(s: dict, max_ops: int | None = None, show_params: bool = False,
               eq_for: int = 0, probes: bool = False) -> str:
    """The summary as a beamer column body: the entities, then the activities.

    The two words are the paper's (plexus2.tex Sec. 2, and the site's front page): a model
    names the ENTITIES -- with their states, fields, relations and the hierarchy organizing
    them -- and the ACTIVITIES that transform them. The slide uses the same two nouns so a
    listener who has read either one is not learning a second vocabulary here.

    No header: the frame title already says which run this is, and repeating its box, timestep
    and frame count above every slide spends the column's height on what nobody reads. What is
    left is the hypothesis itself -- the entities with their containment, and the activities by
    the name each was registered under.
    """
    L = []
    L.append(r"\vspace{6pt}")          # a blank line under the title band, before the first word
    L.append(r"{\normalsize\textbf{entities}}\\[4pt]")
    L.append(r"{\scriptsize\begin{tabular}{@{}lr@{\hspace{7pt}}l@{}}")
    for e in containment(s["sets"], s["fields"]):
        r, pad = e["row"], r"\hspace*{%dpt}" % (7 * e["depth"])
        if e["kind"] == "relation":
            L.append(rf"{pad}$\hookrightarrow$ {tex_escape(r['name'])} & {fmt_num(r['n']) if r['n'] else ''} & "
                     rf"\textcolor{{gray}}{{{tex_escape(r['relation'][0])} $\to$ "
                     rf"{tex_escape(r['relation'][1])}}} \\")
        elif e["kind"] == "field":
            L.append(rf"{tex_escape(r['name'])} & & \textcolor{{gray}}{{field, "
                     rf"grid {tex_escape(r['n_grid'])}}} \\")
        else:
            n = fmt_num(r["n"]) if r["n"] is not None else "seeded"
            note = []
            if r.get("seeded") and r["n"] and r["seeded"] != r["n"]:
                n = fmt_num(r["seeded"])
                note.append(f"of {fmt_num(r['n'])} declared")
            if r["parent"]:
                note.append(f"{fmt_num(r['per_parent'])} per {tex_escape(r['parent'])}")
            if r["types"]:
                note.append(", ".join(tex_escape(t) for t in r["types"][:3]))
            lead = (rf"{pad}$\hookrightarrow$ " if e["depth"] else "")
            L.append(rf"{lead}{tex_escape(r['name'])} & {n} & "
                     rf"\textcolor{{gray}}{{{'; '.join(note)}}} \\")
    L.append(r"\end{tabular}\par}")

    L.append(r"\vspace{10pt}{\normalsize\textbf{activities}}\\[4pt]")
    # A PROBE IS NOT AN ACTIVITY. `record the mesh each frame` and `how much matrix lies inside
    # the surface` change no state: they are measurements written as operators so they can be
    # scheduled, and listing them beside the mechanism invites a reader to count them as part of
    # it. They are declared `probe=True` at registration and dropped here unless asked for.
    ops = [o for o in s["operators"] if probes or not o["probe"]]
    ops = ops[:max_ops] if max_ops else ops
    L.append(r"{\scriptsize\begin{tabular}{@{}c@{\hspace{5pt}}l@{}}")
    for kind, group in group_by_kind(ops):
        L.append(rf"\raisebox{{-4pt}}{{\includegraphics[height=15pt]{{icons/{kind}.png}}}} & "
                 rf"\textcolor{{gray}}{{{KIND_LABEL.get(kind, kind)}}} \\")
        for o in group:
            L.append(rf" & {tex_escape(lower_first(o['title']))} \\[1pt]")
    if max_ops and len(ops) == max_ops:
        L.append(r" & \textcolor{gray}{\ldots and more} \\")
    L.append(r"\end{tabular}\par}")
    return "\n".join(L)


FRAME = r"""\begin{frame}[t]{%(title)s}
\vspace*{\bandgap}
\begin{columns}[T,onlytextwidth]
\begin{column}{%(left)s\textwidth}
%(media)s
\end{column}
\begin{column}{%(right)s\textwidth}
%(body)s
\end{column}
\end{columns}
\end{frame}
"""


def render_frame(s: dict, title: str, movie: str | None, figure: str | None,
                 caption: str = "", left: float = 0.50, **kw) -> str:
    """A whole slide: media on the left, the spec summary on the right."""
    if movie:
        media = "\\playmovie{%s}" % movie
    elif figure:
        media = "\\panel{%s}" % figure
    else:
        media = "~"
    return FRAME % dict(title=title or tex_escape(s["header"]["name"]), left=f"{left:g}",
                        right=f"{1 - left - 0.02:g}", media=media,
                        body="\\fitcol{%%\n%s}" % render_tex(s, **kw))


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec", help="path to a Plexus spec YAML")
    ap.add_argument("--format", choices=["text", "tex", "both"], default="text")
    ap.add_argument("--out", help="write to this directory instead of stdout")
    ap.add_argument("--max-ops", type=int, default=None, help="list at most this many operators")
    ap.add_argument("--equations", type=int, default=2, help="typeset this many equations (tex)")
    ap.add_argument("--no-params", action="store_true", help="omit the parameter values")
    ap.add_argument("--probes", action="store_true",
                    help="list the measurement operators too (dropped by default)")
    ap.add_argument("--frame", action="store_true", help="wrap the tex in a whole beamer frame")
    ap.add_argument("--movie", help="movie basename for the left column (with --frame)")
    ap.add_argument("--figure", help="figure path for the left column (with --frame)")
    ap.add_argument("--caption", default="", help="caption under the media")
    ap.add_argument("--title", default="", help="frame title")
    ap.add_argument("--left", type=float, default=0.50, help="width of the media column")
    a = ap.parse_args()

    s = summarise(a.spec)
    pieces = {}
    if a.format in ("text", "both"):
        pieces["txt"] = render_text(s, max_ops=a.max_ops)
    if a.format in ("tex", "both"):
        kw = dict(max_ops=a.max_ops, show_params=not a.no_params, eq_for=a.equations,
                  probes=a.probes)
        pieces["tex"] = (render_frame(s, a.title, a.movie, a.figure, a.caption, a.left, **kw)
                         if a.frame else render_tex(s, **kw))
    if not a.out:
        print("\n\n".join(pieces[k] for k in ("txt", "tex") if k in pieces))
        return
    os.makedirs(a.out, exist_ok=True)
    for ext, body in pieces.items():
        dest = os.path.join(a.out, f"{s['header']['name']}.{ext}")
        with open(dest, "w") as f:
            f.write(body + "\n")
        print(dest)


if __name__ == "__main__":
    main()
