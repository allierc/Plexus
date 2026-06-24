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
    "exchange":   ("Exchange", "push / pull", "set &harr; field"),
    "field":      ("Field", r"$\partial_t\phi$",   "a field's own self-dynamics"),
    "rewire":     ("Rewire", r"$\mathcal{R}\!:E$", "rebuild the relation $E$ each tick"),
    "structural": ("Structural", r"$|S|$",         "change the node set (divide / die / spawn)"),
}
def kind_icon(kind):
    """Icon filename for an operator KIND. The field-dynamics operator (dphi/dt)
    gets its own glyph so it is not confused with the field data-structure icon."""
    return "field_dyn" if kind == "field" else kind


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

    # --- equation-only enrichments (math straight from each operator's source) --- #
    "Coulomb": dict(equation=r"""$$
\ddot{\mathbf x}_i \;=\; \sum_{j\in\mathcal N(i)} -\,q_i q_j\,
\frac{\mathbf x_j-\mathbf x_i}{\lVert\mathbf x_j-\mathbf x_i\rVert^{3}}
$$
The inverse-square law with superposition (`aggr: add`). Like charges $q_iq_j>0$
repel, opposite charges attract; $\mathcal N(i)$ are the edges a `rewire` leaves."""),

    "advance": dict(equation=r"""$$
\dot{\mathbf x}_i \;=\; s_i\,\hat{\mathbf h}_i,\qquad \lVert\hat{\mathbf h}_i\rVert=1
$$
Self-propulsion at speed $s_i$ (`move_speed`) along the unit heading vector
$\hat{\mathbf h}_i$ (the same $[N,D]$ orientation in 2D and 3D); the engine
integrates the position ($\mathbf x \mathrel{+}= \Delta t\,\dot{\mathbf x}$)."""),

    "aggregate": dict(equation=r"""$$
\mathbf x_P \;=\; \frac{\sum_{i\in P} o_i\,\mathbf x_i}{\sum_{i\in P} o_i}
$$
The occupancy-weighted centroid of a parent $P$'s children ($o_i$ = occupancy). A
derived readout written straight onto the parent, not an integrated force."""),

    "alignment": dict(equation=r"""$$
\mathbf a_i \;=\; a_2\,w^a_i\,\big\langle \mathbf v_j-\mathbf v_i\big\rangle_{j\in\mathcal N(i)}
$$
The boids velocity-matching rule: steer toward the mean neighbour velocity.
$w^a_i$ is the receiver type's `alignment` weight; $\langle\cdot\rangle$ is the mean over neighbours."""),

    "cohesion": dict(equation=r"""$$
\mathbf a_i \;=\; a_1\,w^c_i\,\big\langle \mathbf x_j-\mathbf x_i\big\rangle_{j\in\mathcal N(i)}
$$
The boids centring rule: steer toward the local centre of mass (mean offset to
neighbours). $w^c_i$ is the receiver type's `cohesion` weight."""),

    "separation": dict(equation=r"""$$
\mathbf a_i \;=\; -\,a_3\,w^s_i\,
\Big\langle \frac{\mathbf x_j-\mathbf x_i}{\lVert\mathbf x_j-\mathbf x_i\rVert^{2}}\Big\rangle_{j\in\mathcal N(i)}
$$
The boids anti-crowding rule: push away from close neighbours, weighted by
$1/\lVert d\rVert^{2}$. The graph's `min_radius` keeps coincident neighbours finite."""),

    "drag": dict(equation=r"""$$
\mathbf a_i \;=\; -\,k\,\mathbf v_i
$$
Viscous friction as an acceleration ($k$ = drag coefficient). A force in the
schedule, not an integrator knob, so it composes with the other second-derivative laws."""),

    "gravity": dict(equation=r"""$$
\mathbf a_{\text{ext}} \;=\; \mathbf g \qquad (\text{default } \mathbf g=(0,-g_y))
$$
A uniform body acceleration returned as a cell delta; in an MPM scene the substep
reads it as the external body force rather than the engine integrating it."""),

    "broadcast": dict(equation=r"""$$
\dot{\mathbf x}_i \;=\; k\,\big(\mathbf x_{P(i)}-\mathbf x_i\big)
$$
The containment lift: each child is pulled toward its parent $P(i)$'s position with
stiffness $k$ — the down-the-hierarchy dual of `aggregate`."""),

    "chemotaxis": dict(equation=r"""$$
\dot{\mathbf x}_i \;=\; g\,\nabla\phi(\mathbf x_i)
$$
The Keller–Segel coupling: climb (or, for $g<0$, flee) the gradient of field $\phi$
sampled at the particle. A first-derivative velocity that simply sums with any other."""),

    "decay": dict(equation=r"""$$
c \;\leftarrow\; \max\!\big(0,\; c - k\,\Delta t\big)
$$
Linear evaporation of a scalar field at rate $k$ — the sink that lets stigmergic
trails fade so the network coarsens. The companion of `deposit` / `diffuse`."""),

    "deposit": dict(equation=r"""$$
c_{s}\big(\mathrm{pix}(\mathbf x_i)\big) \;\leftarrow\;
\min\!\Big(1,\; c_{s}\big(\mathrm{pix}(\mathbf x_i)\big) + a\,\Delta t\Big),
\qquad s=\text{type}(i)
$$
Each agent adds $a\,\Delta t$ (`amount`) to its own species channel $s$ at the pixel
under its position, clamped to 1 — the only place agents imprint the trail."""),

    "radius_graph": dict(equation=r"""$$
E \;=\; \big\{\,(i,j)\;:\; r_{\min} \le \lVert\mathbf x_i-\mathbf x_j\rVert < r\,\big\}
$$
Rebuilds the neighbour relation each tick: an edge for every pair within `radius` $r$
(and beyond an optional `min_radius` floor). Emits no delta — lateral ops read these edges."""),

    "pacemaker": dict(equation=r"""$$
p(t) \;=\;
\begin{cases}
\sin\!\big(\pi\, s/\tau_d\big) & s<\tau_d\\[2pt]
0 & \text{otherwise}
\end{cases},
\qquad s=(t+\varphi)\bmod T
$$
A smooth raised bump of width $\tau_d$ (`duration`) once per period $T$, offset by
phase $\varphi$, in outer ticks $t$. Written to `H.signals[name]` for downstream ops."""),

    "pulse_stimulus": dict(equation=r"""$$
a(\mathbf x,t) \;=\; p(t)\,\exp\!\Big(-\tfrac{\lVert\mathbf x-\mathbf x_0\rVert^{2}}{2\sigma^{2}}\Big),
\qquad \sigma=\text{radius}
$$
Paints the clock $p(t)$ (from a `pacemaker`) as a Gaussian bump centred at $\mathbf x_0$.
Owns *where* the stimulus is — not *when* (the pacemaker) nor its mechanical effect."""),

    "phase_delay_pulse": dict(
        lead="The spatial generalization of the `pacemaker`&rarr;`pulse_stimulus` pair: "
             "every pixel runs the **same** raised-bump waveform, but offset by a per-pixel "
             "delay $\\tau(\\mathbf x)$ read from an image map. A delay **gradient** makes "
             "neighbouring regions fire in sequence &mdash; a travelling activation wave &mdash; and "
             "under an elastic tissue (with a downstream active-stress / contraction operator) that "
             "timing gradient becomes curved, rotating, cardiac-like motion. It owns both **when** "
             "(per pixel) and **where** (via the map), so it schedules in place of the "
             "pacemaker+stimulus pair it subsumes.",
        equation=r"""$$
a(\mathbf x,t) \;=\; \mathrm{pulse}\big(t-\tau(\mathbf x)\big),
\qquad \tau(\mathbf x)=\tau_{\max}\,m(\mathbf x),\quad m(\mathbf x)\in[0,1]
$$
$$
\mathrm{pulse}(t') \;=\;
\begin{cases}
\sin\!\big(\pi\, s/\tau_d\big) & s<\tau_d\\[2pt]
0 & \text{otherwise}
\end{cases},
\qquad s=(t'+\varphi)\bmod T
$$
The same smooth bump as `pacemaker` (width $\tau_d$ = `duration`, period $T$, phase $\varphi$),
evaluated per pixel at its own local time $t-\tau$. The delay map $m$ is an `image` field;
$\tau_{\max}$ (`max_delay`) is the delay where the map saturates. So `pacemaker` is the special
case $\tau\equiv 0$, and a linear $m$ (e.g. $m=x$) gives a plane wave, a radial $m$ a target wave.""",
        spec="""fields:
  delay:      {frame: image, source: material/delay_radial.tif}   # tau(x,y) normalised to [0,1]
  activation: {frame: grid, res: 128}
operators:
  - {op: phase_delay_pulse, at: activation, delay_from: delay,
     period: 150, duration: 30, max_delay: 40}                    # subsumes pacemaker + pulse_stimulus
  - {op: pulse_to_active_stress, at: mpm_particle, from: activation, amplitude: 25.0}
schedule: [phase_delay_pulse, pulse_to_active_stress, mpm_drag,
           mpm_strain, p2g, mpm_grid_update, g2p]""",
        schedule="Drops in **where `pacemaker` + `pulse_stimulus` would go** &mdash; it is a field "
                 "source, scheduled before the operator that reads the activation, then the MPM "
                 "mechanics chain:\n\n"
                 "```\nphase_delay_pulse → pulse_to_active_stress → mpm_drag → "
                 "[mpm_strain, p2g, mpm_grid_update, g2p]×substeps\n```\n\n"
                 "Swap `pulse_to_active_stress` for `pulse_to_contraction` to drive the tissue by "
                 "body force instead of active stress &mdash; the activation field is identical.",
        identifiability=(
            "Only the delay **modulo the period** is observable: the activation is periodic in $t$, "
            "so $\\tau_{\\max}$ and $T$ are confounded unless `max_delay` &lt; `period` (otherwise the "
            "late regions alias back onto the early phase and the wave appears to re-enter). The delay "
            "**map shape** $m(\\mathbf x)$ is what sets the wavefront geometry and is well constrained "
            "by the *direction* of the observed travelling wave; its overall *scale* $\\tau_{\\max}$ "
            "trades off against both the period and the tissue stiffness it works against, so recovering "
            "it needs the transient (the wave sweeping across) rather than a single phase. The map is "
            "analytic now (a TIFF) and learnable later, exactly like a stiffness map."),
        failure_modes=[
            ("`max_delay` &ge; `period`", "the latest pixels wrap past a full beat &rarr; spurious re-entry / the wave appears to fold back; keep the spread of $\\tau$ below one period."),
            ("delay map not normalised to [0,1]", "$\\tau=\\tau_{\\max}m$ is mis-scaled &mdash; the operator expects a normalised map (like a stiffness map); an unnormalised TIFF gives the wrong wave speed."),
            ("missing `delay_from`", "hard error &mdash; the operator requires an image field for $\\tau(\\mathbf x)$ (there is no sensible default map)."),
            ("`duration` too short vs. substep response", "the bump passes a pixel before the tissue mechanically responds &rarr; weak / no motion; widen `duration` or raise `amplitude` downstream."),
            ("map resolution &ne; field resolution", "silently bilinear-resampled to the activation grid &mdash; harmless, but a coarse map blurs the wavefront."),
        ],
        related=[("pacemaker", "the global clock this subsumes &mdash; the $\\tau\\equiv 0$ special case (owns *when*)"),
                 ("pulse_stimulus", "the fixed spatial paint this subsumes (owns *where*)"),
                 ("pulse_to_active_stress", "reads the delayed activation and turns it into contractile stress"),
                 ("pulse_to_contraction", "the body-force alternative downstream; same activation field"),
                 ("apply_material_map", "the same image-field idiom &mdash; a TIFF sampled per particle/pixel")],
    ),

    "pulse_to_active_stress": dict(equation=r"""$$
\boldsymbol\sigma_{\text{active}}(\mathbf x) \;=\;
-\,A\,a(\mathbf x)\,\mathbf n(\mathbf x)\,\mathbf n(\mathbf x)^{\!\top},
\qquad \lVert\mathbf n\rVert=1
$$
Activation $a$ becomes a rank-1 contractile stress along the unit axis $\mathbf n$
($A$ = `amplitude`). `p2g` adds it to the elastic stress, so tissue feels its
**divergence**: a patch under uniform $-A\,\mathbf n\mathbf n^{\!\top}$ shortens along $\mathbf n$."""),

    "playback": dict(equation=r"""$$
\phi(\cdot,\,t) \;=\; V\big[\,t \bmod N_{\text{frames}}\,\big]
$$
Prescribes the field from a recorded movie: the grid is set to the current tick's
frame. A coupling op (e.g. `chemotaxis`) then drives particles from it."""),

    "apply_material_map": dict(equation=r"""$$
E_i \;=\; E_{\min} + m(\mathbf x_i)\,(E_{\max}-E_{\min}),
\qquad (\mu_i,\lambda_i) = \mathrm{Lam\acute e}(E_i)
$$
Bilinearly samples the image field $m\in[0,1]$ at each particle and maps it to a
per-particle Young's modulus $E_i$, then to the Lamé buffers the MPM stress law reads.
Not a force — it sets the *material*, breaking symmetry without changing MPM physics."""),

    "mpm_strain": dict(equation=r"""$$
\mathbf F \;\leftarrow\; (\mathbf I + \Delta t\,\mathbf C)\,\mathbf F
\qquad\text{(liquid: } \mathbf F\leftarrow J^{1/d}\mathbf I,\ J=\det\mathbf F\text{)}
$$
Advances the deformation gradient from the affine velocity $\mathbf C$, then applies
the material correction: liquids drop shape memory, snow clamps the singular values of
$\mathbf F$ and hardens via the plastic ratio $J_p$. Step 1 of the MLS-MPM decomposition."""),

    "p2g": dict(equation=r"""$$
\boldsymbol\sigma = 2\mu(\mathbf F-\mathbf R)\mathbf F^{\!\top} + \lambda(J-1)J\,\mathbf I,
\qquad
\mathbf Q = -\frac{4\,\Delta t}{\Delta x^{2}}\,V\,\boldsymbol\sigma + m\,\mathbf C
$$
$$
m_g \mathrel{+}= \sum_i w_{ig}\,m_i,
\qquad
\mathbf p_g \mathrel{+}= \sum_i w_{ig}\big(m_i\mathbf v_i + \mathbf Q_i(\mathbf x_g-\mathbf x_i)\big)
$$
The fixed-corotated stress ($\mathbf R$ = polar rotation of $\mathbf F$) forms the
affine momentum matrix $\mathbf Q$, scattered to the grid by the quadratic B-spline
weights $w_{ig}$. Step 2 of the MLS-MPM decomposition."""),

    "mpm_grid_update": dict(equation=r"""$$
\mathbf v_g = \frac{\mathbf p_g}{m_g},
\qquad
\mathbf v_g \mathrel{+}= \Delta t\,\mathbf f^{\,\text{CSF}}_g,
\qquad
\mathbf v_g \leftarrow \mathrm{BC}(\mathbf v_g)
$$
Grid momentum $\to$ velocity, plus the continuum surface force (CSF) from the liquid
colour field, then boundary conditions: reflective walls, obstacle friction, interior
mask. A pure field $\to$ field solve. Step 3 of the MLS-MPM decomposition."""),

    "g2p": dict(equation=r"""$$
\mathbf v_i = \sum_g w_{ig}\,\mathbf v_g,
\qquad
\mathbf C_i = \frac{4}{\Delta x^{2}}\sum_g w_{ig}\,\mathbf v_g\,(\mathbf x_g-\mathbf x_i)^{\!\top},
\qquad
\mathbf x_i \mathrel{+}= \Delta t\,\mathbf v_i
$$
Gathers grid velocity back to particles (new velocity + affine matrix $\mathbf C$),
applies wall restitution and the CFL cap, then advects. Step 4 of the MLS-MPM decomposition."""),

    "mpm_drag": dict(equation=r"""$$
\mathbf a_i = -\,k\,\mathbf v_i
\qquad\longrightarrow\qquad
\mathbf a_{\text{ext}} \mathrel{+}= \mathbf a_i
$$
Viscous body drag as a particle delta that `p2g` consumes as a body force (not an
integrator knob). Large $k$ gives the overdamped, tissue-like regime that creeps to
$\mathbf a/k$ without ringing. Replaces `p2g`'s built-in `drag`."""),

    "sense": dict(equation=r"""$$
S_k = \!\!\sum_{\mathbf p\in W_k}\!\Big(c_{\text{own}}(\mathbf p) + \kappa\!\!\sum_{s\ne\text{own}}\!\! c_s(\mathbf p)\Big),
\quad k\in\{C\}\cup\text{fan};
\qquad
\hat{\mathbf h}_i \leftarrow \mathrm{normalize}\big(\hat{\mathbf h}_i + \omega\,\mathbf r_{k^\star}\big)
$$
Dimension-generic Physarum sensing: a centre sensor along $\hat{\mathbf h}$ plus a
fan tilted by `sensor_angle` -- two sensors (ahead-left / ahead-right) in 2D, a ring
of $K$ in 3D. Each window $W_k$ is a species-weighted sum (own channel $+1$, others
$\kappa$ = `cross`). If a fan direction $k^\star$ beats the centre, rotate
$\hat{\mathbf h}$ toward it (up to `turn_speed` $\omega$) and renormalise."""),
}


# --------------------------------------------------------------------------- #
#  Introspection helpers
# --------------------------------------------------------------------------- #
def first_sentence(doc: str) -> str:
    """One-line purpose from a docstring: strip a leading `name --` / `name:` tag."""
    line = (doc or "").strip().split("\n", 1)[0].strip()
    line = re.sub(r"^[\w<>./()-]+\s*(--|—|:)\s*", "", line)
    return line[:1].upper() + line[1:] if line else ""


# --- prose cleanup: strip unreadable ASCII display-math from docstrings -------- #
# The canonical maths live in the LaTeX `equation` (ENRICH); the Mechanism prose
# should refer to it, not repeat an ASCII version. We drop *indented* display
# equations (but keep indented YAML snippets / pipeline lists) and inline
# integrator notes like "(v += dt*a; x += dt*v)".
_MATH_SIG = re.compile(r"[Σ∑²·×⊗≈→√∇]|\bexp\(|\bsin\(|\bcos\(|\bgrad\(|\bmod\b|\bsqrt\b|\^")
_NONMATH = re.compile(r"[{}\[\]]|->|\b\w+:\s|::")          # yaml / pipeline / tag markers


def _is_ascii_math(ln: str) -> bool:
    if len(ln) - len(ln.lstrip()) < 4:                    # only indented "display" lines
        return False
    s = ln.strip()
    if not s or _NONMATH.search(s):                       # blank, or yaml/pipeline -> keep
        return False
    if _MATH_SIG.search(s):
        return True
    return bool(re.search(r"\b\w[\w]*\s*=\s*\S", s) and re.search(r"[*/^+]", s))


def clean_prose(text: str) -> str:
    """Drop ASCII display-equation lines and inline `(... += ...)` integrator notes."""
    kept = [ln for ln in (text or "").split("\n") if not _is_ascii_math(ln)]
    text = "\n".join(kept)
    text = re.sub(r"\s*\([^()]*\+=[^()]*\)", "", text)    # inline integrator parentheticals
    return re.sub(r"\n{3,}", "\n\n", text).strip()


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
    out.append(f'![](../figures/icons/{kind_icon(kind)}.png){{.op-logo}}')
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

    # Mechanism (prose + the canonical LaTeX equation, one section)
    out.append("## Mechanism")
    out.append("")
    out.append(clean_prose(body) if body else "_See source below._")
    out.append("")
    if e.get("equation"):
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

    # Typical schedules (only when authored — no placeholder)
    if e.get("schedule"):
        out.append("## Typical schedules")
        out.append("")
        out.append(e["schedule"])
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

    # Identifiability (only when authored — no placeholder)
    if e.get("identifiability"):
        out.append("## Identifiability")
        out.append("")
        out.append(e["identifiability"])
        out.append("")

    # Failure modes (only when authored — no placeholder)
    if e.get("failure_modes"):
        out.append("## Failure modes")
        out.append("")
        out.append("| when | what goes wrong |")
        out.append("|---|---|")
        for cause, effect in e["failure_modes"]:
            out.append(f"| {cause} | {effect} |")
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
    body = clean_prose("\n".join(doc.split("\n")[1:]).strip())
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
            "- **Used by** &mdash; `exchange` operators (push / pull) and `field` self-dynamics.", "",
            "## Description", "", body or "_See source below._", "",
            "## Source", "",
            f"[`{os.path.relpath(inspect.getsourcefile(cls), ROOT)}`]({rel_github(cls)})", "",
            "```python", source_block(cls).rstrip(), "```", ""]
    return "\n".join(out)


def render_set_page(name: str, cls) -> str:
    level = getattr(cls, "LEVEL", None)
    doc = clean_prose(inspect.getdoc(cls) or "")
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


def _page_header(title, subtitle):
    return ["---", f'title: "{title}"', f'subtitle: "{subtitle}"',
            "resources:", "  - gallery/", "---", "", STYLE, ""]


def _nav_card(href, icon, name, sub):
    return (f'<a class="op-card" href="{href}">'
            f'<img src="figures/icons/{icon}.png" alt="{icon}">'
            f'<span class="op-card-body"><span class="op-card-name">{name}</span>'
            f'<span class="op-card-sub">{sub}</span></span></a>')


def render_landing() -> str:
    """library.qmd &mdash; a short overview that links to the three sub-pages."""
    ops, fields, sets = R._OPERATOR_REGISTRY, R._FIELD_REGISTRY, R._ENTITY_REGISTRY
    n = len(ops) + len(fields) + len(sets)
    out = _page_header("Operator library",
                       "Every set, field, and operator in Plexus &mdash; the validated registry to date")
    out.append(f"The whole calculus is **{len(sets)} sets**, **{len(fields)} fields**, and "
               f"**{len(ops)} operators** ({n} primitives in all). Every simulation on this site is a "
               "schedule composed from this list &mdash; nothing else. The catalog is introspected "
               "live from the registry, so it never drifts from the code. It is split into three "
               "pages &mdash; the two objects (**sets** and **fields**) and the **operators** that "
               "move state between them:")
    out.append("")
    out.append('::: {.op-grid}')
    out.append(_nav_card("library_sets.html", "set", "Sets",
                         f"{len(sets)} node kinds &mdash; the levels a hierarchy is built from."))
    out.append(_nav_card("library_fields.html", "field", "Fields",
                         f"{len(fields)} continuum discretizations &mdash; coupled to sets through exchange."))
    out.append(_nav_card("library_operators.html", "lateral", "Operators",
                         f"{len(ops)} operators, grouped by kind &mdash; how state moves."))
    out.append(":::")
    out.append("")
    return "\n".join(out)


def render_operators() -> str:
    """library_operators.qmd &mdash; operators grouped by kind."""
    ops = R._OPERATOR_REGISTRY
    out = _page_header("Operators",
                       "How state moves &mdash; grouped by kind, in the visual language of Figure 1")
    out.append("The logo marks the operator **kind** in the visual language of "
               "[Figure 1](index.qmd) (`paper/fig_ops.tex`).")
    out.append("")
    for kind in KIND_ORDER:
        members = sorted(nm for nm, c in ops.items() if getattr(c, "KIND", None) == kind)
        if not members:
            continue
        klabel, ksym, kgloss = KIND_INFO[kind]
        out.append(f'## <img class="kind-h" src="figures/icons/{kind_icon(kind)}.png"> {klabel} '
                   f'<span class="kind-sym">{ksym}</span>')
        out.append("")
        out.append(f"*{kgloss[0].upper() + kgloss[1:]}.*")
        out.append("")
        out.append('::: {.op-grid}')
        for nm in members:
            c = ops[nm]
            sub = first_sentence(inspect.getdoc(inspect.getmodule(c)) or inspect.getdoc(c) or "") \
                or f"{kind} @ {getattr(c, 'LEVEL', '')}"
            out.append(card(nm, c, kind_icon(kind), sub))
        out.append(":::")
        out.append("")
    return "\n".join(out)


def render_sets() -> str:
    """library_sets.qmd &mdash; the node kinds."""
    sets = R._ENTITY_REGISTRY
    out = _page_header("Sets", "Node kinds &mdash; the levels a hierarchy is built from")
    out.append('::: {.op-grid}')
    for nm in sorted(sets, key=lambda x: getattr(sets[x], "LEVEL", 0)):
        c = sets[nm]
        out.append(card(nm, c, "set", first_sentence(inspect.getdoc(c) or "")))
    out.append(":::")
    out.append("")
    return "\n".join(out)


def render_fields() -> str:
    """library_fields.qmd &mdash; the continuum discretizations."""
    fields = R._FIELD_REGISTRY
    out = _page_header("Fields",
                       "Continuum discretizations &mdash; coupled to sets through `exchange` operators")
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
    for fname, renderer in [("library.qmd", render_landing),
                            ("library_operators.qmd", render_operators),
                            ("library_fields.qmd", render_fields),
                            ("library_sets.qmd", render_sets)]:
        with open(os.path.join(ROOT, fname), "w") as f:
            f.write(renderer())
    total = len(R._OPERATOR_REGISTRY) + len(R._FIELD_REGISTRY) + len(R._ENTITY_REGISTRY)
    print(f"wrote library.qmd + 3 sub-pages + {total} detail pages "
          f"({len(R._OPERATOR_REGISTRY)} ops, {len(R._FIELD_REGISTRY)} fields, {len(R._ENTITY_REGISTRY)} sets)")


if __name__ == "__main__":
    main()
