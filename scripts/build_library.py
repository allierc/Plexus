"""Generate the Operator Library: a catalog page + one detail page per registered
operator / set / field, introspected from the live registry.

    python scripts/build_library.py            # writes library.qmd + library/*.qmd

Everything factual (kind, level, prediction, required params, mechanism tags,
parameter roles, source) is read straight from the registry and the operator
source, so the library can never drift from the code. The hand-authored prose
(equation, identifiability, failure modes, typical schedules, related) lives in
the ENRICH table below; an operator with no ENRICH entry still gets a complete,
accurate page, with those few sections shown as "to be written".

Design language (logos + figures) matches Figure 1 / paper/fig_ops.tex via the
per-kind glyphs in figures/icons (see scripts/make_op_icons.py).
"""
from __future__ import annotations

import inspect
import os
import re
import textwrap

import plexus                       # noqa: F401
import plexus.operators             # noqa: F401  self-registers the validated library
import plexus.models.entities       # noqa: F401  self-registers particle / cell / ...
from plexus.models import registry as R

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
LIBDIR = os.path.join(ROOT, "library")
GH = "https://github.com/allierc/Plexus/blob/main"

# fig_ops order; each kind has a one-line gloss and the math symbol from Figure 1.
KIND_ORDER = ["lateral", "aggregate", "broadcast", "exchange", "field", "rewire", "structural"]
KIND_INFO = {
    "lateral":    ("Lateral", r"$\mathcal{O}_E$",  "within-set interaction over a relation $E$"),
    "aggregate":  ("Aggregate", r"$\textstyle\sum_\pi$", "children &rarr; parent, up the containment $\\pi$"),
    "broadcast":  ("Broadcast", r"$\pi^*$",        "parent &rarr; children, down the containment $\\pi$"),
    "exchange":   ("Exchange", r"$\mathcal{S},\mathcal{G}$", "set &harr; field scatter / gather"),
    "field":      ("Field", r"$\partial_t\phi$",   "a field's own self-dynamics"),
    "rewire":     ("Rewire", r"$\mathcal{R}\!:E$", "rebuild the relation $E$ each tick"),
    "structural": ("Structural", r"$|S|$",         "change the node set (divide / die / spawn)"),
}
PREDICTION_GLOSS = {
    "first_derivative":  "returns a **velocity** &mdash; the engine integrates $x \\mathrel{+}= \\Delta t\\,\\delta$",
    "second_derivative": "returns an **acceleration** &mdash; $v \\mathrel{+}= \\Delta t\\,\\delta;\\ x \\mathrel{+}= \\Delta t\\,v$",
    None: "emits **no integrated force** &mdash; it mutates a field / relation / membership, or feeds a substep",
}


# --------------------------------------------------------------------------- #
#  Hand-authored enrichment (the exemplars). Add an entry to deepen any page.
# --------------------------------------------------------------------------- #
ENRICH = {
    "attraction_repulsion": dict(
        lead="A smooth, per-type pairwise force law on a neighbour graph &mdash; the analytic "
             "core of [ParticleGraph](https://github.com/allierc/ParticleGraph). A long-range "
             "Gaussian pulls, a short-range Gaussian pushes; the balance between them, set "
             "per particle type, selects the emergent phase: single clusters, multi-cluster "
             "foams, lattices, or filaments.",
        equation=r"""$$
f(r) \;=\; p_1\,e^{-r^{2p_2}/2\sigma^2} \;-\; p_3\,e^{-r^{2p_4}/2\sigma^2},
\qquad
\dot{\mathbf{x}}_i \;=\; \sum_{j\in\mathcal{N}(i)} f(r_{ij})\,(\mathbf{x}_j-\mathbf{x}_i)
$$
with $\mathbf p=[p_1,p_2,p_3,p_4]$ the receiver type's parameters and $\sigma$ a global width.
The first term is the long-range pull, the second the short-range push.""",
        spec="""sets:
  particle:
    n: 30000
    types:                       # per-type force law p = [pull, pull_range, push, push_range]
      t0: {fraction: 0.34, p: [1.62, 1.04, 1.60, 1.56]}
      t1: {fraction: 0.33, p: [1.77, 1.83, 1.09, 1.91]}
      t2: {fraction: 0.33, p: [1.72, 1.79, 1.06, 1.86]}
operators:
  - {op: radius_graph,         at: particle, radius: 0.075}   # the rewire that defines N(i)
  - {op: attraction_repulsion, at: particle, sigma: 0.005}
schedule: [radius_graph, attraction_repulsion]""",
        schedule="Always paired with a `rewire` that supplies the neighbour graph it reads. "
                 "The canonical tick is `radius_graph` &rarr; `attraction_repulsion`; the graph is "
                 "rebuilt every step so the force tracks the moving particles.",
        gallery=("arbitrary_3.mp4", "Three types, 30&nbsp;000 particles, &sigma;=0.005, periodic &mdash; "
                 "the field coarsens into a honeycomb of interconnected domains."),
        identifiability=(
            "Trajectories of a settled pattern pin down **ratios** far better than absolute scales. "
            "$\\sigma$ and the $p$ magnitudes trade off (rescaling $\\sigma$ while rescaling the "
            "exponents leaves the equilibrium spacing nearly unchanged), so recovering them needs "
            "the **transient** &mdash; the approach to equilibrium &mdash; not just the final frame. "
            "Inter-type parameters are only identifiable for type pairs that actually come into "
            "contact; types that never mix leave their cross terms unconstrained."),
        failure_modes=[
            ("`sigma` too large vs. graph `radius`", "the cutoff clips the Gaussian before it decays &rarr; forces look truncated, clusters fray."),
            ("net attraction with no repulsion (`p3`&approx;0)", "runaway collapse to a single point; add short-range push or a `radius_graph` floor."),
            ("`aggr: sum` at high density", "force scales with neighbour count &rarr; denser regions blow up; `mean` (default) is density-independent."),
            ("graph not rebuilt each tick", "particles outrun their edges; the force goes stale and the pattern freezes."),
        ],
        related=[("radius_graph", "the rewire that builds the neighbour graph this operator reads"),
                 ("Coulomb", "a fixed $1/r^2$ law &mdash; the non-parametric cousin"),
                 ("alignment", "another lateral law, velocity-aligning rather than position-based"),
                 ("separation", "short-range repulsion alone, for collision handling")],
    ),
    "diffuse": dict(
        lead="One explicit step of isotropic diffusion on a scalar field &mdash; a per-channel "
             "3&times;3 box-blur lerp that is exactly one forward-Euler step of "
             "$\\partial_t c = D\\,\\nabla^2 c$ with edge-clamped boundaries. It mutates the "
             "field in place and returns nothing; it is the spreading half of every "
             "reaction&ndash;diffusion and stigmergy schedule.",
        equation=r"""$$
c \;\leftarrow\; (1-w)\,c \;+\; w\,\overline{c}_{3\times3},
\qquad w=\operatorname{sat}(\text{rate}\cdot\Delta t)
$$
$\overline{c}_{3\times3}-c$ is a discrete Laplacian, so this is one explicit step of
$\partial_t c = D\,\nabla^2 c$ (3&times;3 in 2D, 3&times;3&times;3 in 3D).""",
        spec="""fields:
  chemical: {frame: grid, res: 96}
operators:
  - {op: diffuse, at: chemical, rate: 0.35}
schedule: [secrete, diffuse, decay, sense]""",
        schedule="Sits between deposition and sensing: a source operator (`deposit` / `secrete`) "
                 "writes the field, `diffuse` spreads it, `decay` removes it, and a `sense` / "
                 "`chemotaxis` operator reads the gradient. Multiple `diffuse` steps per tick give "
                 "a larger effective diffusion length.",
        gallery=("rd_worms.mp4", "A Gray&ndash;Scott reaction&ndash;diffusion pattern &mdash; "
                 "`diffuse` is the transport term that lets the worms spread and branch."),
        identifiability=(
            "Only the **product** rate&middot;$\\Delta t$ and the number of steps per tick are "
            "observable &mdash; doubling `rate` and halving $\\Delta t$ is invisible. The diffusion "
            "*length* $\\sqrt{D\\,t}$ is what patterns actually constrain, so from a single field "
            "snapshot `rate` and total time are confounded; you need either a known $\\Delta t$ or "
            "the time-course of a spreading front to separate them."),
        failure_modes=[
            ("`rate`&middot;`dt` &gt; 1", "the lerp saturates (clamped to 1) &rarr; the field becomes its own neighbourhood mean in one step; over-smoothing, lost structure."),
            ("too few steps for the domain", "diffusion length too short &rarr; sources stay pinned, no long-range coupling."),
            ("wrong boundary expectation", "edges are clamped (zero-flux), not periodic &mdash; mass piles up at walls unless the field is periodic by design."),
        ],
        related=[("decay", "the companion sink term; diffuse spreads, decay removes"),
                 ("deposit", "writes a scalar source into the field that diffuse then spreads"),
                 ("sense", "reads the diffused gradient back onto a set (stigmergy)"),
                 ("chemotaxis", "particle-level gradient following on a diffused field")],
    ),
    "pulse_to_contraction": dict(
        lead="The force half of an active-matter actuator: it reads an activation field "
             "$a(\\mathbf x,t)$ and converts it into a per-particle MPM body force, returned "
             "as a particle delta the MLS-MPM substep consumes. It owns only the **mechanical "
             "mapping** &mdash; not *when* the tissue fires (`pacemaker`) nor *where* "
             "(`pulse_stimulus`). This is how a beating heart wall is driven.",
        equation=r"""$$
\mathbf F_i \;=\;
\begin{cases}
\pm\,\text{amplitude}\,\nabla a(\mathbf x_i) & \text{gradient mode (inward / outward)}\\[4pt]
\text{amplitude}\,a(\mathbf x_i)\,\mathbf d(\mathbf x_i) & \text{directional mode}
\end{cases}
$$
In gradient mode the force points along $\pm\nabla a$ (contract / expand); in directional
mode a unit vector-field $\mathbf d$ sets the orientation and $a$ sets the magnitude.""",
        spec="""fields:
  activation: {frame: grid, res: 128}
  direction:  {frame: vector_grid, res: 128}
operators:
  - {op: pulse_to_contraction, at: mpm_particle, from: activation,
     mode: directional, direction_from: direction, amplitude: 25.0}""",
        schedule="A field-to-particle force in the MPM mechanics chain. The activation is timed "
                 "and placed upstream, then converted to force, then run through the solver:\n\n"
                 "```\npacemaker → pulse_stimulus → pulse_to_contraction → mpm_drag → "
                 "[mpm_strain, p2g, mpm_grid_update, g2p]×substeps\n```",
        gallery=("mat_elastic.mp4", "Active deformable matter under MLS-MPM &mdash; the same "
                 "mechanics chain this operator feeds (here an elastic body, the cardiac "
                 "scenario swaps in a `directional` activation map)."),
        identifiability=(
            "`amplitude` is identifiable only **up to the material stiffness** it works against "
            "&mdash; a strong force on stiff tissue and a weak force on soft tissue produce the "
            "same strain, so observed deformation alone confounds `amplitude` with Young's "
            "modulus (set via `apply_material_map`). Separating them needs either a known stiffness "
            "map or observations across several activation levels. `mode` and `direction_from` are "
            "structurally identifiable: inward/outward/directional give qualitatively different, "
            "distinguishable strain fields."),
        failure_modes=[
            ("`amplitude` too high vs. `mpm_drag`", "the sheet overshoots and rings / inverts elements; raise `mpm_drag` k or lower amplitude."),
            ("`mode: directional` without `direction_from`", "hard error &mdash; directional mode needs a vector_grid field for the orientation."),
            ("activation not reset each tick", "force accumulates; the engine's zero_delta must clear the particle delta each outer tick."),
            ("force scale vs. MPM `a_max`", "p2g clamps the body force at a_max; an amplitude above it is silently capped."),
        ],
        related=[("pacemaker", "sets *when* the activation field fires (the clock)"),
                 ("pulse_stimulus", "sets *where* on the field the activation appears"),
                 ("p2g", "scatters this force onto the MPM grid as the body force a_ext"),
                 ("mpm_drag", "the companion damping force summed into the same particle delta")],
    ),
}


# --------------------------------------------------------------------------- #
#  Introspection helpers
# --------------------------------------------------------------------------- #
def first_sentence(doc: str) -> str:
    """One-line purpose from a docstring: strip a leading `name --` / `name:` tag."""
    line = (doc or "").strip().split("\n", 1)[0].strip()
    line = re.sub(r"^[\w<>./()-]+\s*(--|—|:)\s*", "", line)
    return line[:1].upper() + line[1:] if line else ""


def parse_params(cls) -> list[dict]:
    """Param rows: required keys, PARAM_ROLES, and defaults parsed from __init__."""
    required = list(getattr(cls, "REQUIRES_PARAMS", []) or [])
    roles = dict(getattr(cls, "PARAM_ROLES", {}) or {})
    defaults: dict[str, str] = {}
    try:
        src = inspect.getsource(cls.__init__)
        for key, val in re.findall(r'params\.get\(\s*["\'](\w+)["\']\s*,\s*([^)\n]+?)\)', src):
            defaults[key] = val.strip()
    except (OSError, TypeError):
        pass
    names = []
    for k in required + list(roles) + list(defaults):
        if k not in names and not k.startswith("_"):
            names.append(k)
    rows = []
    for k in names:
        rows.append(dict(name=k, role=roles.get(k, ""),
                        default=("**required**" if k in required else defaults.get(k, "&ndash;")),
                        required=k in required))
    return rows


def source_block(cls) -> str:
    """The operator source: whole file if it defines one registered symbol, else the class."""
    try:
        file = inspect.getsourcefile(cls)
        n_reg = sum(1 for c in set(list(R._OPERATOR_REGISTRY.values()) + list(R._FIELD_REGISTRY.values()))
                    if inspect.getsourcefile(c) == file)
        return inspect.getsource(inspect.getmodule(cls) if n_reg == 1 else cls)
    except (OSError, TypeError):
        return inspect.getsource(cls)


def rel_github(cls) -> str:
    try:
        f = os.path.relpath(inspect.getsourcefile(cls), ROOT)
        return f"{GH}/{f}"
    except (OSError, TypeError):
        return GH


# --------------------------------------------------------------------------- #
#  Page rendering
# --------------------------------------------------------------------------- #
def badge(text: str) -> str:
    return f"[`{text}`]{{.op-badge}}"


def render_operator_page(name: str, cls) -> str:
    kind = getattr(cls, "KIND", None)
    level = getattr(cls, "LEVEL", None)
    pred = getattr(cls, "PREDICTION", None)
    doc = inspect.getdoc(inspect.getmodule(cls)) or inspect.getdoc(cls) or ""
    e = ENRICH.get(name, {})
    klabel, ksym, kgloss = KIND_INFO.get(kind, (kind, "", ""))

    purpose = first_sentence(doc)
    lead = e.get("lead") or purpose

    # ---- mechanism prose: the module docstring minus its first tag line ---- #
    body = "\n".join(doc.split("\n")[1:]).strip()
    body = re.sub(r"\n{3,}", "\n\n", body)

    tags = list(getattr(cls, "MECHANISM_TAGS", []) or [])
    morph = list(getattr(cls, "MORPHOLOGY_PRIOR", []) or [])
    dims = getattr(cls, "SUPPORTED_DIMS", []) or []
    rtp = list(getattr(cls, "REQUIRES_TYPE_PROPS", []) or [])
    rbuf = list(getattr(cls, "REQUIRES_BUFFERS", []) or [])
    req = list(getattr(cls, "REQUIRES_PARAMS", []) or [])

    out = []
    out.append("---")
    out.append(f'title: "{name}"')
    if purpose:
        out.append(f'subtitle: "{purpose}"')
    out.append("---")
    out.append("")
    # header: kind glyph + lead
    out.append('::: {.op-head}')
    out.append(f'![](../figures/icons/{kind}.png){{.op-logo}}')
    out.append("")
    out.append(f"{badge(klabel.lower())} {badge('acts on: ' + str(level))}"
               + (f" {badge('prediction: ' + (pred or 'none'))}" if True else ""))
    out.append("")
    out.append(lead)
    out.append(":::")
    out.append("")

    # Role in Plexus (the contract)
    out.append("## Role in Plexus")
    out.append("")
    out.append(f"- **Kind** &mdash; {ksym} **{klabel}**: {kgloss}.")
    out.append(f"- **Acts on** &mdash; `{level}` (the level the operator runs at).")
    reads = ", ".join(f"`{x}`" for x in (req + rtp + rbuf)) or "&ndash;"
    out.append(f"- **Reads** &mdash; {reads}"
               + ("" if not rtp else f"  (per-type props: {', '.join('`'+x+'`' for x in rtp)})"))
    out.append(f"- **Writes / returns** &mdash; {PREDICTION_GLOSS.get(pred, PREDICTION_GLOSS[None])}.")
    out.append(f"- **Prediction** &mdash; `{pred or 'none'}`.")
    out.append(f"- **Dimensions** &mdash; {', '.join(f'{d}D' for d in dims) or '&ndash;'}.")
    if getattr(cls, "TRANSITIONAL", False):
        out.append("- **Transitional** &mdash; wraps a mature multi-mechanism subsystem (fenced; scheduled for decomposition).")
    out.append("")

    # Mechanism
    out.append("## Mechanism")
    out.append("")
    out.append(body if body else "_See source below._")
    out.append("")

    # Equation
    if e.get("equation"):
        out.append("## Equation")
        out.append("")
        out.append(e["equation"])
        out.append("")

    # Parameters
    rows = parse_params(cls)
    if rows:
        out.append("## Parameters")
        out.append("")
        out.append("| parameter | role | default |")
        out.append("|---|---|---|")
        for r in rows:
            out.append(f"| `{r['name']}` | {r['role'] or '&ndash;'} | {r['default']} |")
        out.append("")

    # Minimal spec
    out.append("## Minimal spec")
    out.append("")
    if e.get("spec"):
        out.append("```yaml")
        out.append(e["spec"])
        out.append("```")
    else:
        req_line = "".join(f", {k}: ..." for k in req)
        out.append("```yaml")
        out.append(f"operators:\n  - {{op: {name}, at: {level}{req_line}}}")
        out.append("```")
    out.append("")

    # Typical schedules
    out.append("## Typical schedules")
    out.append("")
    out.append(e.get("schedule") or "_Where this operator sits in a pipeline &mdash; to be written._")
    out.append("")

    # Example output
    if e.get("gallery"):
        vid, cap = e["gallery"]
        out.append("## Example output")
        out.append("")
        out.append("```{=html}")
        out.append('<figure class="op-vid"><video src="../gallery/%s" autoplay loop muted '
                   'playsinline preload="metadata"></video>' % vid)
        out.append(f'<figcaption>{cap}</figcaption></figure>')
        out.append("```")
        out.append("")

    # Identifiability
    out.append("## Identifiability")
    out.append("")
    out.append(e.get("identifiability")
               or "_What observations can (and cannot) recover this operator's parameters &mdash; to be written._")
    out.append("")

    # Failure modes
    out.append("## Failure modes")
    out.append("")
    if e.get("failure_modes"):
        out.append("| when | what goes wrong |")
        out.append("|---|---|")
        for cause, effect in e["failure_modes"]:
            out.append(f"| {cause} | {effect} |")
    else:
        out.append("_What breaks under bad parameters &mdash; to be written._")
    out.append("")

    # Mechanism-search tags
    if tags or morph:
        out.append("## Mechanism-search tags")
        out.append("")
        if tags:
            out.append("**Mechanism** &mdash; " + " ".join(badge(t) for t in tags) + "  ")
        if morph:
            out.append("**Morphology prior** &mdash; " + " ".join(badge(t) for t in morph))
        out.append("")

    # Related operators
    out.append("## Related operators")
    out.append("")
    if e.get("related"):
        for rn, why in e["related"]:
            out.append(f"- [`{rn}`]({rn}.qmd) &mdash; {why}")
    else:
        sib = [n for n, c in R._OPERATOR_REGISTRY.items()
               if getattr(c, "KIND", None) == kind and n != name][:6]
        if sib:
            out.append("Other **" + str(kind) + "** operators: "
                       + ", ".join(f"[`{n}`]({n}.qmd)" for n in sib) + ".")
        else:
            out.append("_&ndash;_")
    out.append("")

    # Source
    out.append("## Source")
    out.append("")
    out.append(f"[`{os.path.relpath(inspect.getsourcefile(cls), ROOT)}`]({rel_github(cls)}) "
               "&mdash; the registered operator.")
    out.append("")
    out.append("```python")
    out.append(source_block(cls).rstrip())
    out.append("```")
    out.append("")
    return "\n".join(out)


def render_field_page(name: str, cls) -> str:
    frame = getattr(cls, "FRAME", None)
    couples = getattr(cls, "COUPLES_TO", None)
    doc = inspect.getdoc(inspect.getmodule(cls)) or inspect.getdoc(cls) or ""
    purpose = first_sentence(doc)
    body = "\n".join(doc.split("\n")[1:]).strip()
    out = ["---", f'title: "{name}"']
    if purpose:
        out.append(f'subtitle: "{purpose}"')
    out += ["---", "", "::: {.op-head}", "![](../figures/icons/field.png){.op-logo}", "",
            f"{badge('field')} {badge('frame: ' + str(frame))}"
            + (f" {badge('couples to: ' + str(couples))}" if couples else ""), "",
            purpose or "A continuum field discretization.", ":::", "",
            "## Role in Plexus", "",
            f"- **Frame** &mdash; `{frame}` (how the continuum is discretized).",
            f"- **Couples to** &mdash; {('`'+str(couples)+'`') if couples else 'any set, via an `exchange` operator'}.",
            "- **Used by** &mdash; `exchange` operators (scatter / gather) and `field` self-dynamics.", "",
            "## Description", "", body or "_See source below._", "",
            "## Source", "",
            f"[`{os.path.relpath(inspect.getsourcefile(cls), ROOT)}`]({rel_github(cls)})", "",
            "```python", source_block(cls).rstrip(), "```", ""]
    return "\n".join(out)


def render_set_page(name: str, cls) -> str:
    level = getattr(cls, "LEVEL", None)
    doc = inspect.getdoc(cls) or ""
    purpose = first_sentence(doc)
    schema = getattr(cls, "state_schema", None) or getattr(cls, "STATE_SCHEMA", None)
    out = ["---", f'title: "{name}"']
    if purpose:
        out.append(f'subtitle: "{purpose}"')
    out += ["---", "", "::: {.op-head}", "![](../figures/icons/set.png){.op-logo}", "",
            f"{badge('set')} {badge('level: ' + str(level))}", "",
            doc or "A node kind.", ":::", "",
            "## Role in Plexus", "",
            f"- **Level** &mdash; `{level}` (0 = leaf, higher = container).",
            "- **Is** &mdash; a set of nodes with the state schema below; operators act *within* it "
            "(`lateral`), *across* its containment (`aggregate` / `broadcast`), or *onto fields* (`exchange`).", ""]
    if isinstance(schema, dict) and schema:
        out += ["## State schema", "", "| block | columns |", "|---|---|"]
        for blk, span in schema.items():
            out.append(f"| `{blk}` | {span} |")
        out.append("")
    out += ["## Source", "",
            f"[`{os.path.relpath(inspect.getsourcefile(cls), ROOT)}`]({rel_github(cls)})", "",
            "```python", source_block(cls).rstrip(), "```", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
#  The catalog index
# --------------------------------------------------------------------------- #
def card(name: str, cls, icon: str, sub: str) -> str:
    return (f'<a class="op-card" href="library/{name}.html">'
            f'<img src="figures/icons/{icon}.png" alt="{icon}">'
            f'<span class="op-card-body"><span class="op-card-name">{name}</span>'
            f'<span class="op-card-sub">{sub}</span></span></a>')


def render_index() -> str:
    ops = R._OPERATOR_REGISTRY
    fields = R._FIELD_REGISTRY
    sets = R._ENTITY_REGISTRY
    n = len(ops) + len(fields) + len(sets)

    out = ["---", 'title: "Operator library"',
           'subtitle: "Every set, field, and operator in Plexus &mdash; the validated registry to date"',
           "resources:", "  - gallery/", "---", "",
           f"The whole calculus is **{len(sets)} sets**, **{len(fields)} fields**, and "
           f"**{len(ops)} operators** ({n} primitives in all). Every simulation on this site is a "
           "schedule composed from this list &mdash; nothing else. Each entry below is introspected "
           "live from the registry, so the catalog never drifts from the code. The logo marks the "
           "operator **kind** in the visual language of [Figure 1](index.qmd) "
           "(`paper/fig_ops.tex`).", "",
           STYLE, ""]

    # Operators grouped by kind
    out.append("## Operators")
    out.append("")
    for kind in KIND_ORDER:
        members = sorted(n for n, c in ops.items() if getattr(c, "KIND", None) == kind)
        if not members:
            continue
        klabel, ksym, kgloss = KIND_INFO[kind]
        out.append(f'### <img class="kind-h" src="figures/icons/{kind}.png"> {klabel} '
                   f'<span class="kind-sym">{ksym}</span>')
        out.append("")
        out.append(f"*{kgloss[0].upper() + kgloss[1:]}.*")
        out.append("")
        out.append('::: {.op-grid}')
        for nm in members:
            c = ops[nm]
            sub = first_sentence(inspect.getdoc(inspect.getmodule(c)) or inspect.getdoc(c) or "") \
                or f"{kind} @ {getattr(c, 'LEVEL', '')}"
            out.append(card(nm, c, kind, sub))
        out.append(":::")
        out.append("")

    # Sets
    out.append("## Sets")
    out.append("")
    out.append("*Node kinds &mdash; the levels a hierarchy is built from.*")
    out.append("")
    out.append('::: {.op-grid}')
    for nm in sorted(sets, key=lambda x: getattr(sets[x], "LEVEL", 0)):
        c = sets[nm]
        out.append(card(nm, c, "set", first_sentence(inspect.getdoc(c) or "")))
    out.append(":::")
    out.append("")

    # Fields
    out.append("## Fields")
    out.append("")
    out.append("*Continuum discretizations &mdash; coupled to sets through `exchange` operators.*")
    out.append("")
    out.append('::: {.op-grid}')
    for nm in sorted(fields):
        c = fields[nm]
        out.append(card(nm, c, "field", first_sentence(inspect.getdoc(inspect.getmodule(c)) or inspect.getdoc(c) or "")))
    out.append(":::")
    out.append("")
    return "\n".join(out)


STYLE = """```{=html}
<style>
.op-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:.7rem;margin:1rem 0 1.6rem}
.op-card{display:flex;align-items:center;gap:.7rem;padding:.6rem .75rem;border:1px solid var(--bs-border-color,#dee2e6);
  border-radius:10px;text-decoration:none;color:inherit;background:var(--bs-body-bg,#fff);transition:.12s}
.op-card:hover{border-color:#1f77b4;box-shadow:0 2px 8px rgba(31,119,180,.13);transform:translateY(-1px)}
.op-card img{width:42px;height:42px;flex:0 0 42px;object-fit:contain}
.op-card-body{display:flex;flex-direction:column;min-width:0}
.op-card-name{font-weight:600;font-family:var(--bs-font-monospace,monospace);color:#1f77b4}
.op-card-sub{font-size:.8em;color:#6c757d;line-height:1.25;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
.kind-h{height:1.5em;vertical-align:-.35em;margin-right:.25rem}
.kind-sym{color:#adb5bd;font-weight:400;margin-left:.3rem}
.op-head{display:block;border-left:3px solid #1f77b4;padding:.2rem 0 .2rem 1rem;margin:.5rem 0 1.5rem}
.op-logo{width:74px;height:74px;float:right;margin:-.2rem 0 .4rem 1rem;object-fit:contain}
.op-badge{font-size:.78em;background:rgba(31,119,180,.1);color:#1f77b4;border-radius:5px;padding:.05rem .4rem;margin-right:.2rem;white-space:nowrap}
.op-vid{margin:.4rem 0}.op-vid video{width:100%;max-width:520px;border-radius:8px;background:#000;display:block}
.op-vid figcaption{font-size:.85em;color:#6c757d;margin-top:.3rem;max-width:520px}
</style>
```
"""


def main():
    os.makedirs(LIBDIR, exist_ok=True)
    with open(os.path.join(LIBDIR, "_library_style.qmd"), "w") as f:
        f.write(STYLE)
    # per-item style include (detail pages also need the css)
    for name, cls in R._OPERATOR_REGISTRY.items():
        page = render_operator_page(name, cls)
        page = page.replace(":::\n\n## Role", ":::\n\n" + STYLE + "\n## Role", 1)
        with open(os.path.join(LIBDIR, f"{name}.qmd"), "w") as f:
            f.write(page)
    for name, cls in R._FIELD_REGISTRY.items():
        with open(os.path.join(LIBDIR, f"{name}.qmd"), "w") as f:
            f.write(render_field_page(name, cls).replace(":::\n\n## Role", ":::\n\n" + STYLE + "\n## Role", 1))
    for name, cls in R._ENTITY_REGISTRY.items():
        with open(os.path.join(LIBDIR, f"{name}.qmd"), "w") as f:
            f.write(render_set_page(name, cls).replace(":::\n\n## Role", ":::\n\n" + STYLE + "\n## Role", 1))
    with open(os.path.join(ROOT, "library.qmd"), "w") as f:
        f.write(render_index())
    total = len(R._OPERATOR_REGISTRY) + len(R._FIELD_REGISTRY) + len(R._ENTITY_REGISTRY)
    print(f"wrote library.qmd + {total} detail pages "
          f"({len(R._OPERATOR_REGISTRY)} ops, {len(R._FIELD_REGISTRY)} fields, {len(R._ENTITY_REGISTRY)} sets)")


if __name__ == "__main__":
    main()
