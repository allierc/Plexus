"""Print a model's hierarchy: what exists, how it is organized, and what acts on it.

The two axes of plexus2 sec. "The language", side by side and read off the SPEC rather than
narrated -- the sets and the maps that organize them (`what exists`) against the operators typed
by the sets they touch (`what happens`).

The maps are shown as they actually are, and they are of two kinds, which is the point the tree
form of a hierarchy hides:

    parent        CONTAINMENT, a function child -> parent. It partitions, so Aggregate and
                  Broadcast are well defined along it.
    maps.<role>   a NAMED FUNCTION out of a set, one row to one row. A half-edge has `srce`,
                  `trgt` and `face`, so a many-to-many relation between vertices and cells is
                  carried as a SET with two legs rather than as a new primitive.
    pre / post    the same idea for a static edge-set (a synapse between two neurons).

    python tools/print_hierarchy.py tissue/cvd_baseline
    python tools/print_hierarchy.py cell/cell_atlas_25 --live      # build it and count for real
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from plexus.paths import resolve_config  # noqa: E402

KIND_MARK = {"lateral": "<->", "aggregate": " ^ ", "broadcast": " v ", "exchange": "<=>",
             "field": " ~ ", "rewire": " # ", "structural": " * ", "seed": " 0 ",
             "divide": " * ", "die": " * "}


def set_size(sets, name, seen=()):
    """How many elements a set holds: `n`, else per_parent x its parent's size."""
    s = sets.get(name) or {}
    if name in seen:
        return None
    if "n" in s:
        return int(s["n"])
    pp = s.get("per_parent")
    if pp is None:
        return None
    par = set_size(sets, s.get("parent"), seen + (name,))
    if par is None:
        return None
    if isinstance(pp, dict):
        # A PER-TYPE `per_parent` IS KEYED ON THE PARENT'S TYPES, and those types' `count`s are
        # PER GRANDPARENT -- so the product is (children per grandparent) x (how many
        # grandparents), NOT x (how many parents). Multiplying by the parent count instead
        # reported 11,116,176,000 material points for a model that holds 10,069,000: the number
        # was over by exactly the piece count, 1,104, and looked plausible enough to print.
        types = ((sets.get(s.get("parent")) or {}).get("types") or {})
        n_pieces = sum(int(v.get("count", 0)) for v in types.values())
        per_grandparent = sum(int(pp.get(t, 0)) * int(v.get("count", 0))
                              for t, v in types.items())
        if not (n_pieces and per_grandparent):
            return None
        return per_grandparent * (par // n_pieces)
    return int(pp) * par


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    ap.add_argument("--live", action="store_true",
                    help="build the model and report the counts the ENGINE produces, not the "
                         "ones the spec implies -- they differ wherever a seed or a structural "
                         "operator has a say")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--png", default=None, help="also draw the hierarchy to this PNG")
    args = ap.parse_args()

    yaml_file, _pre, name = resolve_config(args.spec)
    raw = yaml.safe_load(open(yaml_file))
    sets = raw.get("sets") or {}
    fields = raw.get("fields") or {}
    g = raw.get("general") or {}

    live = {}
    if args.live:
        import plexus.operators  # noqa: F401
        from plexus.schema import load
        from plexus.engine import build, seed as run_seed
        sim = load(yaml_file)
        H = build(sim, device=args.device)
        run_seed(H, sim, device=args.device)
        live = {n: int(l.n) for n, l in H.levels.items()}

    print(f"\n{'=' * 86}\n{name}   ({os.path.relpath(yaml_file, ROOT)})\n{'=' * 86}")
    u = g.get("units") or {}
    scale = (f"1 world unit = {u['length_um']:g} um" if "length_um" in u else "dimensionless")
    print(f"dim {g.get('dim', 2)}   world {g.get('world', 1.0)}   {scale}   "
          f"{g.get('n_frames', '?')} frames at dt {g.get('dt', '?')}")

    # ---- WHAT EXISTS ------------------------------------------------------------------- #
    # ordered so a set appears after the set it is contained in
    order, pending = [], list(sets)
    while pending:
        progressed = False
        for n in list(pending):
            p = (sets[n] or {}).get("parent")
            if p is None or p in order or p not in sets:
                order.append(n); pending.remove(n); progressed = True
        if not progressed:                     # a cycle: emit the rest as-is rather than hang
            order += pending; break

    depth = {}
    for n in order:
        p = (sets[n] or {}).get("parent")
        depth[n] = depth.get(p, -1) + 1 if p in depth else 0

    print(f"\nWHAT EXISTS -- {len(sets)} set(s)"
          + (f", {len(fields)} field(s)" if fields else "") + "\n")
    for n in order:
        s = sets[n] or {}
        d = depth[n]
        nn = live.get(n, set_size(sets, n))
        size = f"{nn:>12,}" if isinstance(nn, int) else " " * 12
        via = ""
        if s.get("parent"):
            pp = s.get("per_parent")
            per = ("per parent by type" if isinstance(pp, dict) else f"{int(pp):,} per parent")
            via = f"  --parent-->  {s['parent']}   ({per})"
        elif s.get("maps"):
            via = "  " + "   ".join(f"--{r}-->  {t}" for r, t in s["maps"].items())
        elif s.get("pre"):
            via = f"  --pre-->  {s['pre']}   --post-->  {s['post']}"
        print(f"  {'    ' * d}{n:<22}{size}{via}")
        blocks = s.get("state")
        if blocks:
            bl = ", ".join(f"{b}[{(v.get('width', 1) if isinstance(v, dict) else v)}]"
                           for b, v in blocks.items())
            print(f"  {'    ' * d}{'':<22}{'':>12}  state: {bl}")
        types = s.get("types") or {}
        if types:
            shown = list(types)[:6]
            tl = ", ".join(f"{t} x{types[t]['count']:,}" if "count" in types[t] else t
                           for t in shown)
            more = f", +{len(types) - len(shown)} more" if len(types) > len(shown) else ""
            print(f"  {'    ' * d}{'':<22}{'':>12}  types: {tl}{more}")
    for fn, f in fields.items():
        res = f.get("n_grid")
        print(f"  {'(field)':<24}{'':>12}  {fn}"
              + (f"   {res}^{g.get('dim', 2)} grid" if res else ""))

    # ---- WHAT HAPPENS ------------------------------------------------------------------ #
    import plexus.operators  # noqa: F401
    from plexus.models import registry
    rows = []
    for section, ops in (("seed", raw.get("seed") or []), ("", raw.get("operators") or [])):
        for o in ops:
            try:
                cls = registry.get_operator(o["op"], o.get("implementation"), o.get("model"))
                kind = getattr(cls, "KIND", "?")
            except Exception:                  # noqa: BLE001
                kind = "?"
            rows.append((o["op"], kind, str(o.get("at", "")), section,
                         o.get("implementation") or o.get("model") or ""))
    print(f"\nWHAT HAPPENS -- {len(rows)} operator instance(s)\n")
    for op, kind, at, section, variant in rows:
        tag = "seed:" if section == "seed" else "     "
        v = f"  [{variant}]" if variant else ""
        print(f"  {tag} {KIND_MARK.get(kind, ' ? '):^5} {kind:<11} {op:<22} at {at}{v}")

    # the schedule, as the order it actually runs in
    print("\nSCHEDULE\n")
    for step in raw.get("schedule") or []:
        if isinstance(step, dict) and "substep_dt" in step:
            n_sub = round(float(g.get("dt", 0)) / float(step["substep_dt"])) if g.get("dt") else "?"
            print(f"  x{n_sub} at dt {step['substep_dt']:g}:  " + " -> ".join(step["steps"]))
        else:
            print(f"       {step if not isinstance(step, list) else ' -> '.join(step)}")
    print()
    if args.png:
        draw(args.png, name, g, sets, fields, order, depth, rows, live,
             plotting=raw.get("plotting"))


# --------------------------------------------------------------------------------------------
#  THE FIGURE
#
#  LAID OUT IN INCHES, and that is the whole reason it is readable. The first version placed
#  everything in AXIS FRACTIONS while the font sizes stayed in POINTS -- two units that do not
#  scale together, since the figure's height is computed from the content. A 0.017-fraction gap
#  was ~0.09 in on a short figure and a 10 pt line is 0.14 in, so every operator's second line sat
#  on top of the next operator's first, and the subtitle sat on the title. In inches a gap and a
#  line height are the same kind of number and can simply be added.
#
#  Every vertical distance below is therefore either a line height (points / 72) or one of the
#  four constants, and text is never placed closer to a rule than PAD.
# --------------------------------------------------------------------------------------------
FIG_W = 13.6                      # inches
BOX_W, BOX_H = 5.0, 0.86          # a level's card
GAP_V = 0.62                      # clearance between two cards -- the containment arrow's length
PAD = 0.16                        # EVERY line-to-text and rule-to-text gap, without exception
INDENT = 0.42                     # how far a contained level is inset from its container
OPS_X = 8.1                       # the operator column's left edge
COL_GAP = 0.30                    # between a legend swatch and its text
SWATCH = 0.22                     # a colour swatch's width

KIND_RGB = {"lateral": "#e0564e", "aggregate": "#4ea8e0", "broadcast": "#4ee0a8",
            "exchange": "#e0b44e", "field": "#8f7ae0", "rewire": "#e07ac8",
            "structural": "#7ae08f", "divide": "#7ae08f", "die": "#7ae08f",
            "seed": "#9a9a9a"}


def _lh(fs):
    """One line of `fs`-point type, in inches. The only conversion the layout needs."""
    return fs / 72.0


def type_table(sets, order):
    """The set that declares `types`, and its table. THE TYPES ARE THE MODEL, not a footnote.

    A card reading "15 types" tells you the compartment set is heterogeneous and nothing else --
    and in the cell atlas those fifteen ARE the atlas: 421 plasma-membrane patches, 77
    mitochondria, five protein species. The whole point of a level being a heterogeneous
    collection rather than one biological category is lost the moment it is summarised as a
    count, so the table is drawn in full.
    """
    for n in order:
        t = (sets[n] or {}).get("types") or {}
        if len(t) > 1:
            return n, t
    return None, {}


def draw(out, name, g, sets, fields, order, depth, rows, live, plotting=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    FS_TITLE, FS_SUB, FS_SET, FS_DET, FS_OP, FS_OPSUB, FS_MAP = 16, 10.5, 13, 9.5, 11, 9, 10

    # --- height: whichever column is taller, measured the same way it will be drawn --------
    head = PAD + _lh(FS_TITLE) + PAD + _lh(FS_SUB) + GAP_V
    op_pitch = _lh(FS_OP) + _lh(FS_OPSUB) + PAD * 1.6
    right = head + _lh(FS_SUB) + PAD * 2 + len(rows) * op_pitch
    # THE TYPES BELONG IN THE LEVEL THEY ARE THE TYPES OF. As a separate column they read as a
    # legend -- a table beside the picture -- when they are in fact the CONTENT of that level: a
    # compartment set is not "27,600 things, see table", it IS 421 membrane patches and 77
    # mitochondria and thirteen other kinds. Drawn inside the card, the heterogeneity of a level
    # is visible in the same place as the level, which is the claim the hierarchy makes.
    tset, ttypes = type_table(sets, order)
    ty_pitch = _lh(FS_DET) + PAD * 0.62
    grow = {tset: len(ttypes) * ty_pitch + PAD * 1.4} if ttypes else {}
    left = head + sum(BOX_H + grow.get(n, 0.0) + GAP_V for n in order) \
        + len(fields) * (BOX_H * 0.62 + GAP_V * 0.6)
    FIG_H = max(left, right) + PAD * 2

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("black")
    ax.set_xlim(0, FIG_W); ax.set_ylim(FIG_H, 0); ax.axis("off")   # y grows DOWNWARD

    y = PAD
    ax.text(PAD, y, name, color="white", fontsize=FS_TITLE, fontweight="bold",
            ha="left", va="top", family="monospace")
    y += _lh(FS_TITLE) + PAD
    u = (g.get("units") or {})
    sub = (f"1 world unit = {u['length_um']:g} um" if "length_um" in u else "dimensionless")
    ax.text(PAD, y, f"dim {g.get('dim', 2)}    {sub}    {g.get('n_frames', '?')} frames "
                    f"at dt {g.get('dt', '?')}",
            color="#8a8a8a", fontsize=FS_SUB, ha="left", va="top", family="monospace")
    y += _lh(FS_SUB) + GAP_V

    centres = {}
    for n in order:
        s = sets[n] or {}
        x0 = PAD + depth[n] * INDENT
        bh = BOX_H + grow.get(n, 0.0)
        ax.add_patch(FancyBboxPatch((x0, y), BOX_W, bh,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    linewidth=1.3, edgecolor="#3f6fa8",
                                    facecolor="#0c1520", zorder=2))
        nn = live.get(n, set_size(sets, n))
        cnt = f"{nn:,}" if isinstance(nn, int) else "?"
        ax.text(x0 + PAD * 1.6, y + PAD * 1.4, n, color="white", fontsize=FS_SET,
                fontweight="bold", ha="left", va="top", family="monospace", zorder=3)
        ax.text(x0 + BOX_W - PAD * 1.6, y + PAD * 1.4, cnt, color="#7fc4ff", fontsize=FS_SET,
                ha="right", va="top", family="monospace", zorder=3)
        det = []
        if s.get("state"):
            det.append("state " + " ".join(
                f"{b}[{(v.get('width', 1) if isinstance(v, dict) else v)}]"
                for b, v in list(s["state"].items())[:4]))
        if s.get("types") and n != tset:
            det.append(f"{len(s['types'])} types")
        if s.get("mesh"):
            det.append(f"mesh {s['mesh']}")
        if det:
            ax.text(x0 + PAD * 1.6, y + PAD * 1.4 + _lh(FS_SET) + PAD * 0.7, "   ".join(det),
                    color="#7f8f9f", fontsize=FS_DET, ha="left", va="top",
                    family="monospace", zorder=3)
        # ---- the types, INSIDE the card: swatch, name, how many, and how finely sampled ----
        if n == tset and ttypes:
            pal = ((plotting or {}).get("colors") or {})
            nxt = order[order.index(n) + 1] if order.index(n) + 1 < len(order) else None
            pp = (sets.get(nxt) or {}).get("per_parent") if nxt else None
            pp = pp if isinstance(pp, dict) else None
            ty = y + PAD * 1.4 + _lh(FS_SET) + PAD * 1.1
            for tn, tv in ttypes.items():
                c = pal.get(tn)
                col = tuple(float(v) for v in c) if isinstance(c, (list, tuple)) else "#6f7f8f"
                ax.add_patch(plt.Rectangle((x0 + PAD * 1.6, ty + _lh(FS_DET) * 0.18), SWATCH,
                                           _lh(FS_DET) * 0.62, facecolor=col, edgecolor="none",
                                           zorder=3))
                tx = x0 + PAD * 1.6 + SWATCH + PAD * 1.4
                ax.text(tx, ty, tn, color="#d8e2ec", fontsize=FS_DET, ha="left", va="top",
                        family="monospace", zorder=3)
                ax.text(x0 + BOX_W - PAD * 1.6 - 1.30, ty, f"x{int(tv.get('count', 0)):,}",
                        color="#7fc4ff", fontsize=FS_DET, ha="right", va="top",
                        family="monospace", zorder=3)
                if pp:
                    ax.text(x0 + BOX_W - PAD * 1.6, ty, f"{int(pp.get(tn, 0)):,} nodes",
                            color="#8a9aaa", fontsize=FS_DET, ha="right", va="top",
                            family="monospace", zorder=3)
                ty += ty_pitch
        centres[n] = (x0, y, y + bh)

        p = s.get("parent")
        if p in centres:
            px0, _pt, pbot = centres[p]
            ax_ = px0 + INDENT * 0.55
            ax.add_patch(FancyArrowPatch((ax_, pbot), (x0 + INDENT * 0.55, y),
                                         arrowstyle="-|>", mutation_scale=14, linewidth=1.4,
                                         color="#4ea8e0", zorder=1))
            pp = s.get("per_parent")
            per = "per parent, by type" if isinstance(pp, dict) else f"{int(pp):,} per parent"
            ax.text(x0 + INDENT * 0.55 + PAD * 2.0, (pbot + y) / 2.0, f"pi    {per}",
                    color="#4ea8e0", fontsize=FS_MAP, ha="left", va="center", family="monospace")
        elif s.get("maps"):
            for k, (role, tgt) in enumerate(s["maps"].items()):
                ax.text(x0 + BOX_W + PAD * 2.0,
                        y + PAD * 1.4 + k * (_lh(FS_MAP) + PAD * 0.5),
                        f"--{role}-->  {tgt}", color="#4ee0a8", fontsize=FS_MAP,
                        ha="left", va="top", family="monospace")
        y += bh + GAP_V

    for fn, f in fields.items():
        fh = BOX_H * 0.62
        ax.add_patch(FancyBboxPatch((PAD, y), BOX_W * 0.60, fh,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    linewidth=1.3, edgecolor="#6f5aa8",
                                    facecolor="#120c1e", zorder=2))
        res = f.get("n_grid")
        ax.text(PAD + PAD * 1.6, y + fh / 2.0,
                f"{fn}" + (f"    {res}^{g.get('dim', 2)} grid" if res else ""),
                color="#b8a6ff", fontsize=FS_OP, ha="left", va="center",
                family="monospace", zorder=3)
        y += fh + GAP_V * 0.6

    # ---- the operator column ------------------------------------------------------------- #
    oy = head - GAP_V + _lh(FS_SUB) * 0
    ax.text(OPS_X, oy, "operators", color="#8a8a8a", fontsize=FS_SUB, ha="left", va="top",
            family="monospace")
    oy += _lh(FS_SUB) + PAD * 2
    for op, kind, at, section, variant in rows:
        c = KIND_RGB.get(kind, "#9a9a9a")
        # the swatch sits on the FIRST line's optical centre, and the text starts COL_GAP away
        ax.plot([OPS_X, OPS_X + 0.22], [oy + _lh(FS_OP) * 0.5] * 2,
                color=c, linewidth=3.0, solid_capstyle="butt")
        tx = OPS_X + 0.22 + COL_GAP
        v = f"  [{variant}]" if variant else ""
        pre = "seed:  " if section == "seed" else ""
        ax.text(tx, oy, f"{pre}{op}{v}", color="white", fontsize=FS_OP,
                ha="left", va="top", family="monospace")
        ax.text(tx, oy + _lh(FS_OP) + PAD * 0.45, f"{kind}  at {at}", color=c,
                fontsize=FS_OPSUB, ha="left", va="top", family="monospace")
        oy += op_pitch

    fig.savefig(out, dpi=170, facecolor="black")
    plt.close(fig)
    print(f"[hierarchy] -> {out}", flush=True)


if __name__ == "__main__":
    main()
