"""phase1_table -- the Phase 1 box: every mechanism, its equation, and what it cost the language.

The note has said "eight new contracts, six implementations, two aliases, two refinements" since
the first ledger, and a reader has had to take that on trust. This emits the evidence: all 24
mechanisms, the equation each one actually implements, and the class the LEDGER gives it -- not
the class the record claims, because those disagree on six entries and the disagreement is the
interesting part.

WHAT IS DERIVED AND WHAT IS WRITTEN BY HAND, because the distinction is the whole point:

  derived   name, contract, class, and what a class is "of", from `atlas_record.yaml` and the
            ledger. If a verdict changes, this table changes with it and nobody has to remember.
  by hand   the one-line equation. The record's `equations` field is 20-40 lines of code-faithful
            derivation per mechanism -- correct, and unreadable in a table. Compressing it is a
            judgement, so it is made once, here, in the open.

The hand-written half is checked against the derived half: every mechanism in the record must have
an equation here, and every equation here must match a mechanism in the record. Either way round
it raises. A table that silently dropped a mechanism would understate the language's debt, which
is the one direction this campaign must never fail in.

    python phase1_table.py            # -> _state/phase1_operators.tex
"""
from __future__ import annotations

import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "_state")
RECORD = os.path.join(HERE, "atlas_record.yaml")
OUT_TEX = os.path.join(STATE, "phase1_operators.tex")

# --------------------------------------------------------------------------------------------- #
#  The hand-compressed equation, one per mechanism. Sourced from the record's `equations` field
#  (which cites file:line); reduced to the line that decides the contract. Where the code and the
#  paper disagree the CODE is shown -- that is the loop's standing rule -- and the disagreement is
#  named in the note beside the table.
# --------------------------------------------------------------------------------------------- #
EQ = {
 "gene_network_connectionist":
   (r"$\dot g=\sigma(W_g g+W_{\mathrm{in}}u+b)-\gamma g$, \ "
    r"$\sigma(x)=\tfrac12+\tfrac{x}{2\sqrt{1+x^{2}}}$",
    "A per-cell gene circuit: dense gene$\\to$gene coupling through a saturating nonlinearity, "
    "plus a learned drive from sensed fields, minus linear decay. The \\emph{algebraic} sigmoid, "
    "not the logistic one. Paper adds the sensed input \\emph{outside} the sigmoid; the code mixes "
    "it inside through a trainable matrix --- source wins."),
 "neural_ode":
   (r"$\dot y=\mathrm{MLP}\!\left(\left[u,y\right]\right)$",
    "The same contract with the vector field replaced by a network: one shared MLP maps sensed "
    "inputs and evolving state to a derivative. Nothing about the state it reads or writes "
    "differs --- only the arithmetic in between."),
 "gene_network_mwc":
   (r"$F_i=F^0_i+\sum_j H^{g}_{ij}\ln\!\left(1+\tfrac{g_j}{K^{g}_{ij}}\right)"
    r"+\sum_k H^{\mathrm{in}}_{ik}\ln\!\left(1+\tfrac{u_k}{K^{\mathrm{in}}_{ik}}\right)$,\ \ "
    r"$\dot g_i=\rho_i\,\mathrm{sig}(F_i)-g_i/\tau_i$",
    "The same contract again, with a thermodynamic drive: log-occupancy binding terms instead of "
    "a linear matrix. Production saturates on the clamped concentration, decay uses the raw one."),
 "odecontroller":
   (r"$y(\Delta t)=\mathrm{ODESolve}\!\left(\dot y=f(y,u),\,y_0,\,[0,\Delta t]\right)$, "
    r"emit $y(\Delta t)-y_0$",
    "The base class the three above share: hold the sensed drivers fixed, integrate whatever "
    "vector field the subclass supplies over one macro-step with adaptive Dopri5, and emit the "
    "\\emph{increment} rather than the new state."),
 "stochastic_step":
   (r"$\Phi(s)=\mathrm{replay}\!\left(s,\tau\right)$, \ $\tau\sim\mathrm{sample}(s;\xi)$",
    "The mixin that fixes the sample / replay / score composition. It writes no physical state of "
    "its own; it constrains how every stochastic step must be built."),
 "death":
   (r"$p_i=1-e^{-\lambda_i\Delta t}$, \ $\mathrm{died}_i\sim\mathrm{Bern}(p_i)\wedge"
    r"\mathrm{alive}_i$",
    "Apoptosis as a scorable jump: an independent per-cell hazard over the macro-step flips the "
    "liveness bit. The slot is marked dead, not freed, within the same step."),
 "free_screened_diffusion":
   (r"$D\nabla^{2}c-Kc+S=0$ \ solved as \ "
    r"$c_i=\sum_j \mathrm{alive}_j\,G(r_{ij},a_j,\kappa)\,S_j$, \ $\kappa=\sqrt{K/D}$",
    "Steady screened diffusion, overwritten every macro-step. Not a lattice solve: an analytic "
    "free-space Green's-function superposition over secreting cells, with the receiver distance "
    "clamped to the source's surface so a cell reads its own secretion."),
 "division":
   (r"$p_i=1-e^{-\lambda_i\Delta t}$; \ on commit \ $r\rightarrow r\,2^{-1/d}$ (both), \ "
    r"$\Delta x=r\,2^{-1/d}\hat n$, \ $\hat n\propto s\hat a+\xi/\sqrt d$",
    "Stochastic division with volume conservation: mother and daughter each take the radius that "
    "halves the mother's $d$-volume, and the daughter is placed one \\emph{new} radius away along "
    "a partly-biased, partly-random axis."),
 "saturating_cell_growth":
   (r"$\dot r=k\left(1-r/R\right)$ \ integrated exactly: \ "
    r"$\Delta r=(R-r)\left(1-e^{-k\Delta t/R}\right)$",
    "Von Bertalanffy growth in closed form, so the radius approaches the target and can never "
    "overshoot it --- no clamp is needed. The paper states a constant increment plus a "
    "\\code{min} clamp; the code does not implement that. Source wins."),
 "active_brownian_dynamics2_d":
   (r"$\Delta x=\Delta t\!\left(\tfrac{F}{\gamma}+v_0 e(\theta)\right)"
    r"+\sqrt{\tfrac{2k_BT\Delta t}{\gamma}}\,\xi$, \ "
    r"$\Delta\theta=\sqrt{2D_r\Delta t}\,\xi_r$",
    "Self-propulsion along a heading that itself diffuses. The propulsion term is \\emph{not} "
    "divided by the drag, and the heading has no drift --- both are load-bearing and both are "
    "easy to get wrong."),
 "brownian_dynamics":
   (r"$\Delta x=-\tfrac{\nabla U}{\gamma}\Delta t+\sqrt{\tfrac{2k_BT\Delta t}{\gamma}}\,\xi$",
    "One Euler--Maruyama step of overdamped Langevin motion. Drift scales with $\\Delta t$, noise "
    "with $\\sqrt{\\Delta t}$, and the realised displacement is recorded so the step can be "
    "scored later."),
 "no_force":
   (r"$U\equiv 0$, \ $F_i=0$",
    "The null potential the position-moving steps fall back to when no interaction is given. Real "
    "code, no biology."),
 "pairwise_potential":
   (r"$E=\tfrac12\!\!\sum_{i\neq j}\!U(r_{ij})$, \ $F_i=-\partial E/\partial x_i$, \ "
    r"$p_i=-\tfrac{1}{2dV_i}\sum_j r_{ij}U'(r_{ij})$",
    "The base class every pair law inherits: half-summed energy over live non-self pairs, forces "
    "by autodiff rather than a hand-coded law, and the per-cell virial pressure."),
 "morse":
   (r"$U=\epsilon\!\left[\left(1-e^{-\alpha(r-\sigma)}\right)^{2}-1\right]S(r)$, \ "
    r"$\sigma=r_i+r_j$",
    "A repulsive core with an adhesive tail whose minimum sits exactly at contact. The code "
    "multiplies in a smooth cutoff that the paper's equation does not have."),
 "soft_sphere":
   (r"$U=\tfrac{\epsilon}{2}\left(1-r/\sigma\right)^{2}$ \ for $r<\sigma$, \ else $0$",
    "Purely repulsive excluded volume with compact support: strictly zero beyond contact, so no "
    "cutoff is needed and no adhesion exists. That compactness is what makes it a different "
    "word from the Morse family, not a parameter of it."),
 "hertzian":
   (r"$U=\tfrac{2}{5}\epsilon\left(1-r/\sigma\right)^{5/2}$ \ for $r<\sigma$",
    "Hertzian elastic contact: the same compact repulsive contract, at the exponent classical "
    "contact mechanics gives for two elastic spheres."),
 "harmonic":
   (r"$U=\tfrac{k}{2}\left[(r-\sigma)^{2}-(r_c-\sigma)^{2}\right]$ \ for $r<r_c$",
    "A spring with its parabola shifted to vanish at the cutoff. The energy is continuous there "
    "and the force is not --- a genuine discontinuity, recorded rather than smoothed over."),
 "lennard_jones":
   (r"$U=\epsilon\!\left[\left(\tfrac{\sigma}{r}\right)^{12}"
    r"-2\left(\tfrac{\sigma}{r}\right)^{6}\right]S(r)$",
    "Lennard-Jones in its $r_{\\min}$ form, so the well sits at contact like every other law "
    "here, with the same smooth switch truncating the tail."),
 "mechanical_relaxation":
   (r"$x^{\star}:\ \nabla_x U(x^{\star})=0$ \ (FIRE); \ "
    r"$\partial x^{\star}/\partial\theta$ by the implicit function theorem",
    "Relax to a force balance and overwrite position with it. The gradient is taken of the "
    "\\emph{equilibrium}, not back through the minimiser's iterations --- which is why the "
    "iteration count does not enter the tape."),
 "virial_stress":
   (r"$p_i=-\tfrac{1}{2dV_i}\sum_{j\neq i} r_{ij}\,U'(r_{ij})$",
    "Irving--Kirkwood virial pressure written to a transient per-cell field. It moves nothing: "
    "the observable \\emph{is} the number it writes, which is why its evidence is a still frame "
    "and not a movie."),
 "lie__trotter_macro_step_split":
   (r"$s_{n+1}=\left(\Phi_{\mathrm{disc}}\circ\Phi_{\mathrm{dyn}}"
    r"\circ\Phi_{\mathrm{qs}}\right)(s_n)$",
    "The macro-step as an explicit first-order operator split with a stated error term. Plexus "
    "has a schedule instead. Infrastructure, not biology --- and a candidate to improve the "
    "language."),
 "step_type:_quasistatic___dynamic___discrete":
   (r"$\mathrm{step\_type}\in\{\mathrm{qs},\mathrm{dyn},\mathrm{disc}\}$",
    "Each step declares its time-scale, which fixes both the phase it runs in and how its return "
    "value is interpreted. Plexus classifies an operator by \\emph{which state it touches} --- "
    "orthogonal, and we do not have theirs."),
 "declared_field_dataflow_validation":
   (r"$|Q(f)|\leq 1$, \ $Q$-owned XOR $D$-owned, \ "
    r"$\mathrm{total}_f=\sum_d \delta_d[f]$",
    "Steps couple only through named fields, and the model refuses at build time to assemble two "
    "quasistatic writers of one field. A composition contract, checked before anything runs."),
 "stochastic_trace___replay___score":
   (r"$\nabla_\theta\,\mathbb{E}[L]=\mathbb{E}\!\left[L\,"
    r"\nabla_\theta\log p(\tau\mid s)\right]$",
    "Record what was sampled, replay it, and score it --- how the reference gets a gradient "
    "through a \\emph{discrete} division at all. The single largest thing our language is "
    "missing, and the thing Figure~5 proper needs."),
}

BADGE = {"new": r"\cNew", "implementation": r"\cImpl", "alias": r"\cAlias",
         "refinement": r"\cRef", "out_of_scope": r"\cOut", "unclassified": r"\cUnk"}


def esc(s):
    return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&")
             .replace("%", r"\%").replace("#", r"\#"))


def breakable(name):
    r"""Let a CamelCase class name wrap inside a narrow column.

    `ActiveBrownianDynamics2D` is one 24-character word with no hyphenation point, so LaTeX sets
    it as a single box that runs straight through the next column. A discretionary break at each
    internal capital is where a reader would break it anyway.
    """
    out = []
    for i, ch in enumerate(name):
        if i and ch.isupper() and not name[i - 1].isupper():
            out.append(r"\-")
        out.append(ch)
    return esc("".join(out)).replace(r"\textbackslash{}-", r"\-")


def main():
    doc = yaml.safe_load(open(RECORD))
    led = json.load(open(os.path.join(STATE, "saturation.json")))
    rows = {r["id"]: r for r in led["rows"]}

    ids = [m["id"] for m in doc["mechanisms"]]
    missing = [i for i in ids if i not in EQ]
    extra = [i for i in EQ if i not in ids]
    if missing or extra:                       # never silently short the table
        raise SystemExit(f"phase1_table out of sync with the record.\n"
                         f"  in the record, no equation here: {missing}\n"
                         f"  written here, not in the record: {extra}")

    L = [r"% generated by phase1_table.py -- do not edit by hand",
         r"\begingroup\footnotesize",
         r"\setlength{\LTpre}{4pt}\setlength{\LTpost}{4pt}",
         r"\begin{longtable}{@{}p{0.32cm} >{\raggedright\arraybackslash}p{2.75cm}"
         r" p{9.5cm} >{\raggedright\arraybackslash}p{2.5cm}@{}}",
         r"\toprule",
         r"& \textbf{mechanism} & \textbf{what it computes} & \textbf{cost to the language}\\",
         r"\midrule\endfirsthead",
         r"\toprule",
         r"& \textbf{mechanism} & \textbf{what it computes} & \textbf{cost to the language}\\",
         r"\midrule\endhead"]

    for m in sorted(doc["mechanisms"], key=lambda x: x.get("order", 10**6)):
        mid, r = m["id"], rows.get(m["id"], {})
        cls = r.get("class", "unclassified")
        eq, desc = EQ[mid]
        contract = r.get("contract") or "--"
        of = r.get("of")
        # what the class is "of": the mechanism this repeats, or the contract it widens
        # `\\` inside a p-column cell ENDS THE ROW -- it does not break the line. Using it here
        # threw every "of <mechanism>" out of its cell and onto a row of its own.
        tail = ""
        if cls in ("implementation", "alias", "refinement") and of:
            tail = r"\newline{\scriptsize of \code{%s}}" % breakable(of)
        L.append(
            r"%d & \textbf{%s}\newline{\scriptsize\code{%s}} & %s\newline{\scriptsize %s} & %s%s\\"
            % (m.get("order", 0), breakable(m["raw_name"]), esc(contract), eq, desc,
               BADGE.get(cls, r"\cUnk"), tail))
        L.append(r"\addlinespace[2pt]")

    L += [r"\bottomrule", r"\end{longtable}", r"\endgroup"]
    os.makedirs(STATE, exist_ok=True)
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(L) + "\n")

    c = led["counts"]
    print(f"[phase1] {len(ids)} mechanisms -> {os.path.relpath(OUT_TEX, HERE)}")
    print(f"[phase1] new {c['new']} · implementation {c['implementation']} · alias {c['alias']} "
          f"· refinement {c['refinement']} · out_of_scope {c['out_of_scope']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
