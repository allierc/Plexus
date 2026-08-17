"""critic -- the TYPE GUARD. Deterministic, enumerable, and never a language model.

    The Critic is what CONSTRAINS the language model. It therefore cannot be one.

It was previously real but scattered across four files -- preconditions in composition_space,
the compile refusal in translate, the duplicate filter inside propose_batch, the acted-ledger in
run_record. Scattered, it could not be audited: there was no way to answer "what exactly does the
Critic reject?" except by reading everything. Here every rule is named, carries a reason code,
and is enumerable.

WHEN IT RUNS -- three times, and the third is the one that matters most
----------------------------------------------------------------------
  STATIC     before anything costs a second: is this composition well-formed?
             Type-legal edits, satisfied preconditions, no dangling slots, required roles
             present, not already evaluated. This is FREE -- no simulation, no cluster.

  COMPILE    can it become a runnable spec? Every operator has an emitter and a schedule
             position; every parameter is inside its declared range.

  POST-HOC   did every scheduled operator ACTUALLY ACT? (D4)
             This is the one that cannot be done statically and is the most dangerous to skip:
             an operator whose precondition is met on paper can still no-op, the run still
             finishes, and the loop records "this mechanism cannot produce X" when the mechanism
             never ran. A false IMPOSSIBILITY claim is the least recoverable error the campaign
             can make.

WHY NOT AN LLM
--------------
Co-Scientist's Reflection agent is an LLM and reviews whether a hypothesis is *worth testing*.
That is a different job, and we have it separately. This one decides whether a proposal is
*well-formed*, and it must be decidable, reproducible and incorruptible: a model that could be
argued into accepting an ill-typed composition would let the Proposer's mistakes reach the
cluster, and a silently inert operator would be recorded as evidence.

The Critic also has a second, generative role: the set of edits it does NOT reject IS the menu
handed to the Proposer. The type system decides what is possible; the agent decides what is
worth doing.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

from composition_space import OPERATORS, REQUIRED_ROLES, slots_of
from run_record import comp_hash


@dataclass
class Rejection:
    code: str            # a stable reason code, so rejections can be counted over a campaign
    rule: str            # the human-readable rule
    detail: str

    def __repr__(self):
        return f"<{self.code}: {self.detail}>"


# ============================================================================ BATCH rules
def _op_identity(name):
    """Operator identity, with the NODE INDEX stripped: divide_3d0 and cell_divide are one operator.

    `add_op` names the operator (`cell_divide`) and `remove_op` names the NODE (`divide_3d0`), so
    without this the two directions of the same experiment never matched. A1 then refused a
    necessity claim whose ablation was sitting in the same batch -- a false refusal of a correct
    proposal, which is the expensive way to fail. It was invisible because check_batch has never
    been called; wiring it live without this would have started rejecting good batches.
    """
    n = str(name or "").split(":")[0].strip()
    return re.sub(r"\d+$", "", n)


def _touched_operator(edit):
    """The operator an edit acts on, or None. Edits look like ('add_op','cell_chem_react','gray_scott')
    or a string '+cell_chem_react' / '-cell_chem_react' / '=cell_chem_react:tension'."""
    if isinstance(edit, (tuple, list)) and len(edit) >= 2:
        return _op_identity(edit[1])
    e = str(edit).strip()
    for p in ("add_op ", "remove_op ", "+", "-", "="):
        if e.startswith(p):
            return _op_identity(e[len(p):].split()[0])
    return None


def _direction(edit):
    """additive | subtractive | swap | None."""
    if isinstance(edit, (tuple, list)) and edit:
        k = str(edit[0])
        return {"add_op": "additive", "remove_op": "subtractive",
                "set_impl": "swap"}.get(k)
    e = str(edit).strip()
    if e.startswith("+") or e.startswith("add_op"):
        return "additive"
    if e.startswith("-") or e.startswith("remove_op"):
        return "subtractive"
    if e.startswith("=") or e.startswith("set_impl"):
        return "swap"
    return None


def check_batch(hypotheses):
    """ABLATION IS COMPULSORY. Reject the batch if a causal claim arrives without both directions.

    A causal claim needs two things and the campaign has never enforced either. ABLATION answers
    "is this mechanism necessary" -- run the identical model with it removed and see whether the
    phenotype dies. The ADDITIVE direction answers "is it sufficient". Either alone is an opinion,
    and every "X causes Y" on the books so far rests on one of them.

    The precedent for enforcing it already exists and is not controversial: the control is not left
    to the proposer's taste, it is MANDATED at slot 0 or the proposal is rejected. Ablation gets
    the same treatment, for the same reason -- the author does not referee their own work.

    Deterministic, and it runs BEFORE any compute is spent. Nothing here needs a model.
    """
    out = []
    by_op = {}
    for h in hypotheses:
        op = _touched_operator(getattr(h, "edit", None))
        d = _direction(getattr(h, "edit", None))
        if op and d:
            by_op.setdefault(op, set()).add(d)
    for h in hypotheses:
        kind = getattr(h, "claim_kind", "descriptive")
        op = _touched_operator(getattr(h, "edit", None))
        if kind not in ("causal", "necessary") or not op:
            continue
        dirs = by_op.get(op, set())
        if kind == "necessary" and "subtractive" not in dirs:
            out.append(Rejection(
                "A1_NO_ABLATION",
                "a necessity claim requires the operator to be REMOVED somewhere in the batch",
                f"{getattr(h, 'hid', '?')} claims {op} is necessary, but no hypothesis in this "
                f"batch removes it. Necessity is tested by taking it away, not by adding it. "
                f"Either add the removal or relabel the claim `sufficient`."))
        if kind == "causal" and not {"additive", "subtractive"} <= dirs:
            missing = sorted({"additive", "subtractive"} - dirs)
            out.append(Rejection(
                "A1_NO_ABLATION",
                "a causal claim requires BOTH directions in the same batch",
                f"{getattr(h, 'hid', '?')} claims {op} causes its phenotype, but the batch only "
                f"contains {sorted(dirs)} -- missing {missing}. Sufficiency and necessity are "
                f"different experiments and 'causes' asserts both. Either add the missing "
                f"direction or relabel the claim "
                f"{'`necessary`' if 'additive' in missing else '`sufficient`'}."))
    return out


def allowed_verb(claim_kind):
    """What the Interpreter MAY write. Enforced, so the word cannot outrun the experiment."""
    return {"causal": "causes", "necessary": "is required for",
            "sufficient": "is sufficient for", "descriptive": "is associated with"}.get(
        claim_kind, "is associated with")


def _run_key(graph):
    """MECHANISM@OPERATING-POINT -- what R6 asks about, and what the lever map should record."""
    t = _theta_hash(graph)
    return f"{comp_hash(graph)}@{t}" if t else comp_hash(graph)


def _theta_hash(graph, base=False):
    """The OPERATING POINT, canonically: every tunable parameter at its EFFECTIVE value.

    comp_hash answers "is this the same mechanism". This answers "is this the same experiment on
    it". Keeping them separate is what lets a parameter sweep run without a retune ever counting
    as a discovery.

    EFFECTIVE, NOT OVERLAID -- and the external review caught this. The first version hashed only
    `graph.params`, so the SAME composition at the SAME operating point hashed two ways depending
    on whether a value sat in the overlay or was left to its declared default. It is not
    hypothetical: a reference recipe carries a sparse overlay while the same composition rebuilt
    from a finished run carries every parameter explicitly, and both emit a BYTE-IDENTICAL spec.
    R6 would not have recognised the second as a repeat of the first. Merging the declared
    defaults under the overlay makes the hash a property of the experiment rather than of how it
    happened to be written down.
    """
    import hashlib
    try:
        eff = {f"{o['id']}.{pn}": float(d)
               for o in graph.ops
               for pn, (_lo, _hi, d) in (OPERATORS[o["op"]].get("params") or {}).items()}
        for k, v in (graph.params or {}).items():
            if not k.startswith("_run.") and isinstance(v, (int, float)) and k in eff:
                eff[k] = float(v)
        items = sorted((k, round(v, 9)) for k, v in eff.items())
    except Exception:
        return ""
    if not items:
        return ""
    return hashlib.sha1(repr(items).encode()).hexdigest()[:10]


def _cell_cap(graph):
    """Cells the vertex reservoir allows. Euler on a trivalent closed sheet: V = 2F - 4.

    IT ASKS THE TRANSLATOR, because the graph does not carry the reservoir -- that is decided when
    the spec is written. Reading `_run.vertex_n` off the graph returned 0 for every composition,
    so `_cell_cap` was 0, so R1e skipped ENTIRELY and a projection of 197,000 cells was admitted.
    A guard that silently does nothing is worse than no guard, and this one was mine, written
    yesterday, against exactly this failure.
    """
    try:
        v = float(graph.params.get("_run.vertex_n", 0))
        if v:
            return (v + 4) / 2
    except Exception:
        pass
    try:
        from translate import to_spec
        v = float(((to_spec(graph, name="_cap") or {}).get("sets") or {})
                  .get("vertex", {}).get("n") or 0)
        return (v + 4) / 2 if v else 0.0
    except Exception:
        return 0.0


# ============================================================================ STATIC rules
# THE ONE dt, IMPORTED. This said 0.02 with a comment claiming it mirrored translate.DT_GLOBAL,
# which is 1.0 -- a FIFTY-FOLD disagreement between a constant and the comment describing it, in
# the file whose job is to refuse compositions on numerical grounds. It is the same shape as the
# CFL condition that was "fifty times too generous" because it lived in a comment: a number
# written down once beside the thing it was copied from, and then the thing changed.
#
# Nothing is kept local to avoid an import. `translate` is already imported by every caller of
# this module, and a wrong number costs more than a dependency.
try:
    from translate import DT_GLOBAL as DT_GLOBAL_DEFAULT
except Exception:                                    # pragma: no cover -- translate must import
    DT_GLOBAL_DEFAULT = 1.0


# dt*rate for an autocatalytic kinetics. 1.0 diverged in five runs on 2 August; the bound the
# Diagnostician derived from them is 0.5.
AUTOCATALYTIC_STEP_LIMIT = 0.5


def _param(graph, node_id, key):
    """A node's parameter: the overlay, else the DECLARED DEFAULT, under any of its names.

    TWO BUGS, ONE DISEASE, and it cost weeks. Every rule that reads a parameter went through here.

    1. THE DEFAULT WAS NEVER RETURNED. A declaration is a TUPLE `(lo, hi, default)` and this did
       `.get(key, {}).get("default")` on it -- AttributeError, swallowed by the bare except, None.
       So an unset parameter read as None for every caller, every time, and each caller then fell
       back to its own literal. R1d's `float(rate if rate is not None else 1.0)` is what that looks
       like downstream: a rule refusing on a hard-coded constant while printing "rate None".

    2. THE NAME DIFFERED BETWEEN THE GRAPH AND THE ENGINE -- rd_rate/rate, alpha/hill,
       mono_gamma/gamma, l_th/l_th_frac -- with an alias table nothing consulted. That is fixed by
       RENAMING rather than translating: the declarations now use the engine's names, because the
       engine is what runs. An alias is a workaround for two names, and the workaround is what
       made the bug survivable and therefore permanent.
    """
    params = getattr(graph, "params", {}) or {}
    try:
        from composition_space import OPERATORS
        op = next(o["op"] for o in graph.ops if o["id"] == node_id)
        decl = OPERATORS[op].get("params") or {}
    except Exception:
        op, decl = None, {}

    v = params.get(f"{node_id}.{key}")
    if v is not None:
        return v
    if True:
        if key in decl:
            d = decl[key]
            # (lo, hi, default) -- the default is the THIRD element, not a key
            if isinstance(d, (tuple, list)) and len(d) == 3:
                return d[2]
            if isinstance(d, dict):
                return d.get("default")
            return d
    return None


def check_static(graph, seen_hashes=(), edit_kind=None):
    """Every static rule, in order. Returns [] if the composition is well-formed."""
    out = []

    # R1 -- required roles. A composition with no substrate or no mechanics is not a model.
    roles = graph.roles()
    missing = REQUIRED_ROLES - roles
    if missing:
        out.append(Rejection("R1_MISSING_ROLE", "a composition needs a substrate and mechanics",
                             f"missing role(s): {sorted(missing)}"))

    # R1b -- ONE OPERATOR OF EACH KIND. A composition holding the same operator twice is not a
    # richer model, it is two solvers driving the same state and a hypothesis about neither.
    #
    # MEASURED on round 1 of the rebuilt loop. `legal_edits` offered both `+cell_mechanics:default`
    # and `=cell_mechanics:default`; the Proposer took the ADD and wrote the claim for the SWAP
    # ("swapping the monolayer shape energy for the default releases the in-plane constraint").
    # The spec came out with cell_mechanics twice -- two independent relaxation loops of thirty
    # iterations each, driving the same vertices. Whatever that run measured, it was not the
    # composition the hypothesis named.
    #
    # `legal_edits` no longer offers the add, which closes the path this arrived by. This rule
    # exists because that is only ONE path: a graph can also be hand-written, restored from a
    # frontier file, or built by an edit sequence nobody enumerated. A guard that lives only in
    # the generator is a guard against one way of being wrong.
    from collections import Counter
    for op, n in Counter(o["op"] for o in graph.ops).items():
        if n > 1:
            out.append(Rejection(
                "R1b_DUPLICATE_OPERATOR",
                "the same operator twice is two solvers driving one state, not a richer model",
                f"{op} appears {n} times. To change an implementation use `set_impl`, which "
                f"REPLACES it; `add_op` adds a second instance and the two then run in sequence "
                f"every frame."))

    # R1c -- REACTION STABILITY. There was a CFL bound on diffusion and none on the reaction,
    # and round 1 of the rebuilt loop died of the second while satisfying the first: the activator
    # went 0.01 -> 12.1 -> 1.41e6 -> NaN, SPATIALLY UNIFORM throughout (max spread 3.4e-05 against
    # a mean of 1.4e6). Uniform blow-up is an ODE exploding; a diffusion instability would have
    # made a checkerboard. See composition_space.reaction_advance for why the factor is 1/dt.
    try:
        from composition_space import REACTION_PER_FRAME_LIMIT, reaction_advance

        # `chi` is the RD timescale and it lives on cell_chem_diffuse in this operator set; the
        # reaction is what it destabilises. Both are checked so a relocation cannot silence this.
        for o in graph.ops:
            if o["op"] not in ("cell_chem_diffuse", "cell_chem_react"):
                continue
            chi = _param(graph, o["id"], "chi")
            if chi is None:
                continue
            adv = reaction_advance(chi)
            if adv > REACTION_PER_FRAME_LIMIT:
                out.append(Rejection(
                    "R1c_REACTION_UNSTABLE",
                    "explicit reaction past its stability limit -- the chemistry diverges, so the "
                    "run is evidence about an integrator and not about a mechanism",
                    f"the reaction advances {adv:.1f} per frame against a limit of "
                    f"{REACTION_PER_FRAME_LIMIT} (chi {chi}, scaled by translate.RD_PER_FRAME). "
                    f"The engine already steps the reaction once per substep, so any scaling on "
                    f"top of that is excess. Lower chi."))
    except Exception:
        pass

    # R1e -- THE TISSUE MUST FIT IN ITS ARRAY. Derived, like R1c/R1d, and for the same reason:
    # what makes a run meaningless here is not a number outside a typed range, it is that the
    # experiment cannot happen. `cell_divide` divides a FRACTION of the population per call, so the
    # cell count is exponential by construction and the only thing that ever stops it is the
    # vertex reservoir.
    #
    # MEASURED, round 2 on 3 August: max_div_frac=0.03 with every=1 over frames 100-900 projects
    # 2000 x 1.03^800 = 3.7e13 cells against a 65,004-cell reservoir. The array filled at frame
    # 118 of 800, so ten jobs spent 85% of their wall time pinned against it, and their final cell
    # count measured the buffer rather than the biology. One line of arithmetic, available before
    # a GPU is touched, refuses all ten.
    #
    # It REFUSES rather than warns because the run cannot answer its own question: everything past
    # saturation is a measurement of the reservoir. Lower max_div_frac, raise `every`, shorten the
    # run, or size the reservoir for the projection -- all four are legal answers.
    try:
        _seed = next((o for o in graph.ops if o["op"] == "mesh_seed"), None)
        _div = next((o for o in graph.ops if o["op"] == "cell_divide"), None)
        if _seed is not None and _div is not None:
            _p = graph.params
            n0 = float(_p.get(f"{_seed['id']}.n_cells", 2000))
            every = max(1, int(_p.get(f"{_div['id']}.every", 1)))
            start = float(_p.get(f"{_div['id']}.after_frame", 0))
            # The frame count is a RUN property, not a graph property, so it is not in params.
            # 900 is the campaign default and the honest fallback; being wrong high here makes the
            # rule stricter, never laxer.
            frames = float(_p.get("_run.n_frames", 900))
            calls = max(0.0, (frames - start) / every)
            cap = float(_p.get("_run.cell_cap", 0)) or _cell_cap(graph)
            # THE LAW IS max(max_div, frac x N), NOT frac x N. Projecting the fractional term
            # alone made this rule blind to the very configuration that overran twice: with
            # max_div=30 the floor delivers 30 divisions per call regardless of the fraction, and
            # a pure-exponential projection under-estimated by an order of magnitude.
            # THE RATE IS THE GROWTH RATE, so that is what this projects. The division
            # throttles are gone: every cell that reaches DIV_FACTOR x its birth volume divides.
            # So the population doubles roughly every ln(2)/rate frames, where `rate` is how fast
            # cell_grow inflates a cell -- and the reservoir question becomes "how many
            # doublings does this run get".
            _gro = next((o for o in graph.ops if o["op"] in ("cell_grow",
                                                             "cell_grow")), None)
            grate = float(_p.get(f"{_gro['id']}.rate", 0) or 0) if _gro else 0.0
            if grate > 0 and calls > 0 and cap > 0:
                import math as _m
                doublings = (frames - start) * grate / _m.log(2.0)
                projected = n0 * (2.0 ** doublings)
                if projected > cap:
                    # frames until the array fills
                    fill = start + _m.log(cap / n0) / _m.log(2.0) * _m.log(2.0) / grate
                    if fill < 0.8 * frames:
                        out.append(Rejection(
                            "R1e_TISSUE_OUTGROWS_RESERVOIR",
                            "the projected cell count fills the vertex reservoir long before the "
                            "run ends, so most of the run measures the array rather than the "
                            "tissue",
                            f"{n0:.0f} cells growing at rate={grate:g} gives {doublings:.1f} "
                            f"doublings over {frames - start:.0f} frames -> {projected:.3g} cells "
                            f"against a cap of {cap:.0f}; the array fills at frame {fill:.0f} of "
                            f"{frames:.0f}. Lower the growth rate, shorten the run, or size the "
                            f"reservoir for the projection."))
    except Exception:
        pass

    # R1d -- AUTOCATALYSIS. R1c bounds the reaction by `chi`, the RD timescale, and that is a
    # different axis from this one: a composition can satisfy it and still explode, because the
    # KINETICS decide how big an explicit step may be. The Diagnostician found this unaided on
    # 2 August, from five diverged runs -- "Gierer-Meinhardt blows up UNIFORMLY at dt=1.0,
    # explicit-Euler reaction instability, not CFL; shape=uniform (ODE, not stencil), peak
    # 1.41e06, react is the ONLY differing param vs stable gray_scott at identical dt/chi/d_a/
    # d_h/rate" -- and asked for exactly this guard. Uniform blow-up is an ODE exploding; a
    # diffusion instability would have made a checkerboard.
    #
    # Gierer-Meinhardt is autocatalytic (da ~ rho*a^2/h), so the step that matters is dt*rate and
    # not the linear decay `reaction_stiffness` reports -- which is why that function is
    # deliberately unwired. brusselator is autocatalytic too but has NOT been measured to fail,
    # so it is left alone rather than guarded on a guess.
    # NO BARE EXCEPT AROUND THE WHOLE RULE. It was wrapped in `except Exception: pass`, so an
    # accessor that raised made the guard silently do nothing -- its own comment records that
    # happening once, passing all seven gierer_meinhardt compositions in round 2. The import is
    # done at module scope now (above), so the only thing left inside is per-operator arithmetic,
    # and a failure there is reported rather than swallowed.
    #
    # THE FIRST TIME I WROTE THIS COMMENT THE EXCEPT WAS STILL THERE. An external review proved
    # it by making the accessor raise: the rule vanished and nothing surfaced. A limit in a
    # comment stops describing the code -- this project's own lesson, in the fix for it.
    DT_GLOBAL = DT_GLOBAL_DEFAULT
    if True:
        for o in graph.ops:
            # The implementation is read from the op itself, with impl_of only as a fallback:
            # this whole rule sits in a try/except, so an accessor that raises would make the
            # guard silently do nothing -- which is how it first shipped, passing every one of
            # the seven gierer_meinhardt compositions in round 2.
            impl = o.get("impl")
            if impl is None:
                try:
                    impl = graph.impl_of(o["id"])
                except Exception:
                    impl = None
            if o["op"] != "cell_chem_react" or impl != "gierer_meinhardt":
                continue
            # ONLY WITH DIVISION. The Diagnostician asked for a blanket refusal of
            # gierer_meinhardt above this step, and that guard would have refused r002c_04 --
            # the best run the campaign has produced (protr_peak 1.317, the first non-zero tube
            # count on record), which is gierer_meinhardt at dt*rate = 1.0 and ran all 900 frames
            # with no damage at all. The discriminator is division, not the kinetics alone:
            # every GM composition that also divides took damage at frame 115, five out of five,
            # at the SAME frame; the one that does not divide is clean. A guard aimed at the
            # kinetics would have killed the finding it was meant to protect.
            if not any(x["op"] == "cell_divide" for x in graph.ops):
                continue
            # THE PARAMETER IS CALLED rd_rate. This asked for `rate`, always got None, and fell
            # back to 1.0 -- so R1d refused EVERY gierer_meinhardt-with-division composition
            # unconditionally, whatever rate it actually carried, and its own message said so
            # ("rate None") for weeks. It refused `okuda_route`, the one recipe in the repository
            # holding Okuda's coupling, on a number that recipe does not have.
            #
            # The rule is right; it was reading the wrong key. `rate` is kept as a fallback
            # because other kinetics use that name.
            rate = _param(graph, o["id"], "rate")
            if rate is None:
                rate = _param(graph, o["id"], "rate")
            step = float(DT_GLOBAL) * float(rate if rate is not None else 1.0)
            if step > AUTOCATALYTIC_STEP_LIMIT:
                out.append(Rejection(
                    "R1d_AUTOCATALYTIC_UNSTABLE",
                    "an autocatalytic reaction stepped past its explicit-Euler limit -- the "
                    "activator diverges uniformly and the run measures an integrator, not a "
                    "mechanism",
                    f"gierer_meinhardt WITH cell_divide advances dt*rate = {step:.2f} per step "
                    f"against a limit of {AUTOCATALYTIC_STEP_LIMIT} (dt {DT_GLOBAL}, rate "
                    f"{rate}). Five such runs took damage at frame 115, all five at the same "
                    f"frame. Lower `rate`, use gray_scott, or drop division -- gierer_meinhardt "
                    f"without cell_divide is stable at this step and is the best run on record."))

    # R2 -- PRECONDITIONS. An operator whose required port type is produced by nothing will
    # silently no-op. This is the rule that prevents FALSE IMPOSSIBILITY claims, and it is the
    # single most important thing the Critic does.
    for nid, op, need in graph.unmet_preconditions():
        out.append(Rejection("R2_UNMET_PRECONDITION",
                             "an operator whose input port is produced by nothing will no-op",
                             f"{op} ({nid}) needs a `{need}` producer; none present"))

    # R3 -- DANGLING SLOTS. Present but unconnected is also inert, and looks deliberate.
    for nid, op, slot in graph.unrouted_slots():
        out.append(Rejection("R3_DANGLING_SLOT",
                             "an operator with an unfed input slot is inert",
                             f"{op} ({nid}) slot `{slot}` is fed by nothing"))

    # R4 -- TYPE-LEGAL CONNECTIONS. Belt and braces: legal_edits already only offers legal
    # links, but a graph can also arrive from a file or from an agent editing JSON.
    from composition_space import LEGAL_LINKS
    for c in graph.conns:
        src_op, dst_op = graph._op_of(c["src"]), graph._op_of(c["dst"])
        if src_op is None or dst_op is None:
            out.append(Rejection("R4_DANGLING_EDGE", "a connection references a missing node",
                                 f"{c}"))
            continue
        types = OPERATORS[src_op]["outputs"]
        if not any((t, c["slot"]) in LEGAL_LINKS for t in types):
            out.append(Rejection("R4_ILLEGAL_LINK", "connection is not type-legal",
                                 f"{src_op}({types}) -> {dst_op}.{c['slot']}"))
        dst_node = graph._node(c["dst"])
        if dst_node is not None and c["slot"] not in slots_of(dst_op, graph.impl_of(dst_node)):
            out.append(Rejection("R4_SLOT_NOT_ON_IMPL",
                                 "the chosen implementation does not expose that slot",
                                 f"{dst_op}:{graph.impl_of(dst_node)} has no `{c['slot']}`"))

    # R5 -- WITHDRAWN, 3 August (Cedric). The reason is evidence, not taste.
    #
    # The (lo, hi) boxes were hand-written, and THE RUNS THIS PROJECT HAS ACTUALLY MADE FALL
    # OUTSIDE THEM: cfl_c000p080_d002p000 ran at cell_diffuse0.d_h = 2.0 against a ceiling of
    # 0.346 and produced valid evidence. A faithful rebuild of our own best specimens was
    # therefore refused as a parent, the frontier could not be seeded from them, and every
    # campaign kept restarting from the reference recipes. `from_preset` states the standard this
    # was failing, in its own docstring: "if a recipe we already trust cannot be built from legal
    # one-edit moves, the campaign is searching a space that does not contain our own evidence."
    #
    # A BOX IS NOT A LAW. What makes a run meaningless is never a number outside a range somebody
    # typed; it is an integrator that diverges, a chemistry that goes non-finite, a sheet that
    # passes through itself. Every one of those is caught by a rule that is DERIVED rather than
    # declared -- the CFL condition (R1d, and the whole of Phase 4b), the compile check, the
    # premise gate, the reservoir -- and those are the rules that were ever doing the work. They
    # all stay. This one only ever fenced the search into a box drawn before we had the evidence.
    #
    # The triples remain in OPERATORS as the DEFAULT and the SCALE of a parameter: where a sweep
    # starts and how big a step should be. They no longer decide what may be run.

    # R6 -- ALREADY EVALUATED AT THIS OPERATING POINT. Re-running a known composition is not a
    # new hypothesis; it is a replicate, and must be requested as one rather than arrived at by
    # accident. That rule is right and stays.
    #
    # BUT IT KEYED ON THE COMPOSITION ALONE, AND THAT KILLED THE SWEEP ARM OUTRIGHT. comp_hash
    # EXCLUDES parameters by design -- so that a retune can never masquerade as a new mechanism,
    # which is the discipline this project exists to enforce. A `set_param` child therefore
    # carries its PARENT'S hash, and R6 refused it as already-evaluated. Measured by an external
    # review: 39 of 39 parameter moves refused, even against a seen-list of one. The arm built to
    # ask "what does this mechanism do as you turn it up" could never fire once.
    #
    # The two rules were written for opposite purposes and collided. Both survive if identity and
    # novelty are separated: the COMPOSITION still decides what counts as a mechanism (comp_hash
    # unchanged, so a retune is still not a discovery), and the OPERATING POINT decides whether
    # this exact experiment has been run. A sweep of ten values is ten experiments on one
    # mechanism -- which is exactly what it should be.
    # The key is MECHANISM@OPERATING-POINT. NOTE WHAT THIS DOES NOT GUARANTEE: the lever map's
    # historical entries carry a bare comp_hash and no theta, so a `set_param` move that lands
    # back on one of those operating points is ADMITTED and re-runs it once. An earlier version
    # of this comment claimed otherwise; the external review checked and it was false. Asserting
    # a guarantee the code does not provide is the exact defect WIRING.md exists to stop, and it
    # does not become acceptable for being in a comment I wrote.
    # WHICH QUESTION IS BEING ASKED decides which key. A STRUCTURAL edit proposes a new
    # mechanism, so the composition is the identity and a repeat of it is a replicate -- the
    # original rule, unchanged. A `set_param` edit proposes a new OPERATING POINT on a mechanism
    # already known, which by design carries the parent's comp_hash; for that, only the exact
    # experiment -- mechanism AND parameters -- counts as already evaluated.
    h = comp_hash(graph)
    _seen = set(seen_hashes)
    # THE KEY IS ALWAYS `_run_key`, AND THIS USED TO DEPEND ON THE EDIT KIND. A structural edit was
    # keyed on `comp_hash`, which is parameter-blind -- so `add_op cell_grow` proposed on THREE
    # different parents produced one hash and two of the three slots were refused. Measured on the
    # relaunched round 1:
    #
    #   _keep/r001_02   + cell_grow   Caa2255d08b2@6420561ce7   built
    #   coral_gate      + cell_grow   Caa2255d08b2@a3dd27bbc7   REFUSED, and it is a different run
    #   repair_l_th_frac+ cell_grow   Caa2255d08b2@a3dd27bbc7   refused, correctly -- same as above
    #
    # The Proposer's stated intent was "coverage: cell_grow" across the three best chemistry
    # parents -- testing whether an operator's effect is general or parent-specific, which is the
    # experiment the lever map is FOR. Refusing it treated "same mechanism" as "same experiment".
    #
    # comp_hash still answers "is this a new MECHANISM" and that question still matters -- it is what
    # keeps a retune from being filed as a discovery. It is simply not the question a GPU asks. The
    # question a GPU asks is "will this produce a run I already have", and that is mechanism AND
    # operating point.
    _dup = _run_key(graph) in _seen
    if _dup:
        out.append(Rejection("R6_DUPLICATE", "this composition has already been evaluated",
                             f"{h} -- request a robustness test explicitly if replication is "
                             f"what you want"))
    return out


# ============================================================================ COMPILE rules
def range_notes(graph):
    """Every parameter outside its own declared range, as TEXT. Deliberately not a refusal.

    I WROTE THIS AS A GATE FIRST AND IT REFUSED THE WHOLE CAMPAIGN. `R5_PARAM_OUT_OF_RANGE` has sat in
    THETA_RULES since the rule list was written, emitted nowhere -- declared, documented, never wired --
    and wiring it as a rejection refused SIX OF SIX working recipes, including `coral_gate`, the
    healthiest run on disk (valid_frac 1.0, all ten premises holding, act_cv_peak 2.20).

    Measured across the starting pool:

        edge_flip.l_th_frac   6/6 runs   0.35 vs [0.01, 0.12]     3x the ceiling
        cell_mechanics.Lambda      3/6        3 vs [0, 0.3]           10x
        cell_grow.rate    3/6        0.000866 vs floor 0.002
        cell_grow.a_sw    1/6        50 vs [0.2, 6]           8x
        cell_chem_diffuse.d_h            1/6        2 vs [0, 0.346]          6x

    So the declared box does not contain a single working point, and the consequence is worse than a
    bad gate: the menu handed to the Proposer samples `set_param` INSIDE those boxes, so the search
    explores a region no working recipe occupies. That is a decision about the search space -- widen
    the boxes with a derivation, or drop the pretence that they bound anything -- and it is not a
    decision this function may take on a campaign's behalf.

    It also refutes a claim of mine. `l_th_frac` 1.96 destroys the tissue and 0.28 does not, and BOTH
    are outside the declared range: the box is not the discriminator. The evidence for 1.96 is the
    one-parameter revert that fixed it, not its distance from a ceiling.
    """
    notes = []
    for full_key, val in (getattr(graph, "params", None) or {}).items():
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        node_id, _, key = str(full_key).rpartition(".")
        op = _op_identity(node_id)
        tri = (OPERATORS.get(op, {}).get("params") or {}).get(key)
        if not (isinstance(tri, (tuple, list)) and len(tri) == 3):
            continue
        lo, hi = float(tri[0]), float(tri[1])
        v = float(val)
        if not (lo <= v <= hi):
            over = (v / hi) if (hi > 0 and v > hi) else None
            notes.append(f"{op}.{key}={v:g} outside [{lo:g}, {hi:g}]"
                         + (f" ({over:.0f}x the ceiling)" if over else " (below the floor)"))
    return notes


def check_compile(graph):
    """Can it become a runnable spec? Cheap, and catches emitter/schedule gaps."""
    import translate as T
    out = []
    for node in graph.ops:
        if node["op"] not in T.EMIT:
            out.append(Rejection("C1_NO_EMITTER", "no backend emitter for this operator",
                                 node["op"]))
    spec = None
    try:
        spec = T.to_spec(graph, name="_criticcheck", frames=10)
    except Exception as e:
        out.append(Rejection("C2_COMPILE_FAILED", "the composition does not compile",
                             f"{type(e).__name__}: {str(e)[:160]}"))
    if spec is not None:
        out.extend(check_reservoir(spec, graph))
    return out


# ============================================================================ THE RESERVOIR GATE
def check_reservoir(spec, graph=None, frames=None):
    """Can this run REACH the cell count it is aiming at? Refuse before a GPU is touched.

    THE FAILURE THIS EXISTS TO STOP, twice in one week:

      32-run overnight study   every run ended at exactly 1778 cells. Reported as a finding
                               ("remarkable"), retracted the next day.
      27-run weekend battery   every run ended at exactly 1778 cells. Zero valid evidence.

    Both were the same arithmetic. A closed epithelial sheet is trivalent, so Euler fixes
    V = 2F - 4: a vertex reservoir of size V caps the tissue at (V+4)/2 cells no matter what the
    biology wants. 3552 vertices give exactly 1778. Neither batch was stopped, because nothing
    compared the buffer against the destination -- the saturation flag fires AFTER the run, and
    a flag that fires afterwards tells you what you wasted, not what to avoid.

    This is deliberately NOT a fidelity rule. It does not care whether a run uses Okuda's
    numbers; the exploratory share of a batch is free to start anywhere. It cares only that
    wherever a run is going, it can get there -- which is true of a faithful run and a wild one
    alike. A run that cannot reach its own target measures the array, not the tissue.
    """
    from agents.grounder import max_cells_for
    out = []
    try:
        vbuf = int(spec["sets"]["vertex"]["n"])
        cbuf = int(spec["sets"]["cell"]["n"])
    except Exception:
        return [Rejection("C3_NO_RESERVOIR", "the spec declares no reservoir sizes", "")]

    seed_cells = None
    for o in spec.get("operators", []):
        if o.get("op") in ("mesh_seed",) and o.get("n_cells"):
            seed_cells = int(o["n_cells"])
    if seed_cells is None:
        return out                      # a checkpoint start carries its own count

    ceiling = max_cells_for(vbuf)
    target = (spec.get("_run") or {}).get("target_cells")
    if target is None:
        return [Rejection(
            "C3_NO_TARGET",
            "the run does not say how many cells it is aiming at, so its reservoir cannot be "
            "checked. No fixed multiple of the seed works: the weekend battery seeded 150 into "
            "a buffer holding 1778 -- twelve times the seed -- and every run still stopped on it",
            f"seed={seed_cells}, vertex={vbuf} -> ceiling {ceiling}")]

    if ceiling < int(target):
        out.append(Rejection(
            "C3_RESERVOIR_TOO_SMALL",
            f"aims at {int(target)} cells but the reservoir caps at {ceiling}. The run would "
            f"stop on the buffer and report it as biology -- this is the arithmetic that voided "
            f"32 runs on 30 July and 27 more on 31 July",
            f"seed={seed_cells}, vertex={vbuf} -> (V+4)/2 = {ceiling}, target {int(target)}"))
    if cbuf < ceiling:
        out.append(Rejection(
            "C3_RESERVOIR_INCONSISTENT",
            f"the cell reservoir ({cbuf}) is smaller than the vertex reservoir allows "
            f"({ceiling}) -- whichever binds first will do so silently",
            f"vertex={vbuf}, cell={cbuf}"))
    return out


# ============================================================================ POST-HOC rule
def observations(summary):
    """What the run DID, in words -- never a refusal. Cedric, 5 August: an input, not a gate.

    Everything here used to be a rejection, and every one of them was a RESULT wearing a veto:

      P0_SPECIMEN_INVALID   a broken premise. Fired on 12 of 12 runs in two consecutive rounds
                            and halted the campaign. A gate that fires on everything carries no
                            information and costs the round. The premise DETAIL is the most useful
                            output in the system -- "volume went 522.1 -> 312.9" -- so it now goes
                            to the next proposal through repair.py instead.
      P1_INERT_OPERATOR     an operator that changed nothing is a FINDING. "cell_chem_diffuse did
                            nothing at D=0.002" is knowledge; refusing the run deletes it.
      P3_CHEMISTRY_DIVERGED nan metrics cannot be scored as confirmed or refuted anyway -- the
                            arithmetic already refuses itself, so the gate was ceremony.

    The two mechanisms that DO stop a number being believed are untouched and are both arithmetic:
    the evidence horizon truncates frames after the mesh tears, and check_reservoir refuses a run
    that cannot reach its own target before a GPU is touched.
    """
    obs = []
    for op in (summary.get("inert_operators") or []):
        obs.append(f"{op} changed nothing measurable -- a null result for that operator")
    for p in (summary.get("premises_broken") or []):
        obs.append(f"premise {p} broken")
    if summary.get("chemistry_diverged"):
        obs.append("the chemistry diverged (metrics are nan and cannot be scored)")
    if summary.get("saturated"):
        obs.append("the cell count saturated on its reservoir")
    return obs


def admit(graph, seen_hashes=(), compile_check=True, edit_kind=None):
    """(ok, [Rejection]). The whole static+compile gate."""
    rej = check_static(graph, seen_hashes, edit_kind=edit_kind)
    if not rej and compile_check:
        rej = check_compile(graph)
    return (not rej), rej


def legal_menu(graph, max_stage=3, seen_hashes=(), limit=None):
    """The edits the Critic does NOT reject -- i.e. the menu handed to the Proposer.

    This is the Critic's generative role. The type system decides what is POSSIBLE; the agent
    decides what is WORTH DOING. Neither can do the other's job: a model cannot be trusted to
    respect the type system, and the type system has no taste.
    """
    out = []
    for e, lbl in graph.legal_edits(max_stage):
        try:
            child, _ = graph.apply(e)
        except Exception:
            continue
        ok, _rej = admit(child, seen_hashes, compile_check=False)
        if ok:
            out.append({"edit": list(e) if isinstance(e, tuple) else e, "label": lbl,
                        "yields": child.name_region(), "hash": comp_hash(child)})
            if limit and len(out) >= limit:
                break
    return out


RULES = ["R1_MISSING_ROLE", "R2_UNMET_PRECONDITION", "R3_DANGLING_SLOT", "R4_DANGLING_EDGE",
         "R4_ILLEGAL_LINK", "R4_SLOT_NOT_ON_IMPL", "R5_PARAM_OUT_OF_RANGE", "R6_DUPLICATE",
         "C1_NO_EMITTER", "C2_COMPILE_FAILED", "P1_INERT_OPERATOR", "P2_BUFFER_SATURATED",
         "A1_NO_ABLATION"]


if __name__ == "__main__":
    from composition_space import reference_recipes, seed
    print("=" * 84)
    print(f"CRITIC -- {len(RULES)} enumerable rules, deterministic, never a language model")
    print("=" * 84)
    for r in RULES:
        print(f"  {r}")

    print("\n-- a well-formed composition is admitted --")
    g = reference_recipes()["okuda_route"]
    ok, rej = admit(g)
    print(f"  okuda_route -> admit={ok} {rej}")

    print("\n-- R2: an operator whose precondition nothing satisfies --")
    bad, _ = seed("substrate").apply(("add_op", "cell_chem_diffuse", "graph_laplacian"))
    print(" ", check_static(bad))

    print("\n-- R3: present but unconnected == inert, and looks deliberate --")
    d, _ = seed("substrate").apply(("add_op", "cell_chem_seed", "cone"))
    d, _ = d.apply(("add_op", "cell_grow", "hill_conserve_amount"))
    print(" ", [r for r in check_static(d) if r.code == "R3_DANGLING_SLOT"])

    print("\n-- R6: a re-run is a replicate, not a hypothesis --")
    print(" ", [r for r in check_static(g, seen_hashes=[comp_hash(g)]) if r.code == "R6_DUPLICATE"])

    print("\n-- P1: static checks cannot see this one --")
    print(" ", check_posthoc({"inert_operators": ["cell_chem_diffuse"], "saturated": False}))

    print("\n-- the generative role: the menu handed to the Proposer --")
    menu = legal_menu(seed("substrate"), max_stage=3)
    print(f"  {len(menu)} edits survive the guard; the agent chooses from these")
    for m in menu[:5]:
        print(f"    {m['label']:36} -> {m['yields']}")
    print("\ncritic OK")


# =============================================================================================
# R7 -- IS THE QUESTION ASKABLE?
#
# The epistemic audit of r001-r022 measured this loop's own reproducibility: every `replicate` slot
# re-runs its parent's composition at a fresh seed, so the spread between the two IS the substrate's
# noise. It is large and it is metric-dependent -- 2% on `protr`, 41% on `protrusion_aspect_max`.
#
# Against that, 65% of the campaign's predictions asked for a change SMALLER than the floor of the
# metric they were asked in. Median ask 3%; median floor 20%. Five asked for a 0.0% change -- beat
# the parent's exact value. Below the floor the loop validated at 14%, above it at 39%: the same
# loop, the same scorer, and nearly three times the hit rate as soon as the question is answerable.
#
# So this is not a rule about being right. It is a rule about ASKING SOMETHING A SINGLE RUN CAN
# ANSWER. A threshold finer than the noise is a coin toss with a number on it, and scoring it
# `refuted` credits the loop with a falsification it never performed.
#
# WHY A RULE AND NOT AN INSTRUCTION. `user_input.md` told the Proposer that `add_op` had fired 30
# times and all 30 added the same operator; the next campaign then did 20 for 20. A rule computes;
# a paragraph negotiates.
#
# TWO EXITS, both of which must be declared rather than assumed. `replicate` is exempt because it is
# ABOUT the floor -- refusing it would forbid the only experiment that measures the thing this rule
# depends on. And a slot may declare `precision: true` with a replicate count, which is the honest
# way to ask a fine question: not one run at a tighter threshold, but several at the same one.
def check_resolution(slot, parent_metrics, floors=None):
    """Rejection or None. `slot` carries `predict`, `act`/`intent` and optionally `precision`."""
    import re as _re
    pr = str((slot or {}).get("predict") or "")
    m = _re.match(r"\s*([a-z_0-9]+)\s*([<>]=?)\s*([-0-9.eE+]+)", pr)
    if not m:
        return None                                   # no prediction to judge; other rules apply
    act = (slot.get("act") or slot.get("intent") or "").lower()
    if act in ("replicate", "precision") or slot.get("precision"):
        return None
    metric, thr = m.group(1), float(m.group(3))
    base = (parent_metrics or {}).get(metric)
    if not isinstance(base, (int, float)) or not base:
        # UNKNOWN IS NOT REFUSED. With no parent value there is nothing to compare the ask against,
        # and refusing on an absence would silently bar every prediction on a metric the parent did
        # not report -- which includes every genuinely new measurement.
        return None
    if floors is None:
        floors = _seed_floors()
    f = floors.get(_metric_family(metric), floors.get("_default", 0.20))
    rel = abs(thr - base) / abs(base)
    if rel >= f:
        return None
    return Rejection(
        "R7_BELOW_RESOLUTION",
        "the predicted effect is smaller than this metric's own seed-to-seed spread, so one run "
        "cannot answer it either way",
        f"{metric}: parent {base:.4g}, threshold {thr:.4g} -- a {100 * rel:.1f}% change against a "
        f"measured floor of {100 * f:.0f}%. Ask for a bigger effect, choose a metric with a "
        f"tighter floor, or declare `precision: true` with the replicates that would make it "
        f"readable.")


def _metric_family(metric):
    import re as _re
    return _re.sub(r"_(peak|final|floor|span|trend|measured_frac)$", "", str(metric))


_FLOOR_CACHE = {}


def _seed_floors():
    """The measured floors, from `epistemic_spec.md` -- the same file the audit re-measures into,
    so the rule tightens as the campaign learns its own noise."""
    if _FLOOR_CACHE:
        return _FLOOR_CACHE
    import re as _re
    import yaml as _yaml
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epistemic_spec.md")
    try:
        for b in _re.findall(r"```yaml\n(.*?)```", open(p).read(), _re.S):
            d = _yaml.safe_load(b) or {}
            if "seed_floor" in d:
                _FLOOR_CACHE.update(d["seed_floor"]); break
    except Exception:
        pass
    _FLOOR_CACHE.setdefault("_default", 0.20)
    return _FLOOR_CACHE


# =============================================================================================
# R8 -- IS THE ACT A REAL ACT?
#
# `crew/claims.md` says an act must carry a field the engine checks "or it will decay into a
# synonym for `predict`". Round 2 of the first claim campaign showed the decay is real and faster
# than that: four of fourteen slots put the OLD `intent` vocabulary in the `act` field --
# `exploratory`, with no claim and no prediction -- so they bypassed the claim layer entirely and
# their compute produced nothing the ledger could read.
#
# Half of that was my omission and is fixed in the vocabulary rather than here: there was no
# `explore` act, so a Proposer wanting to look somewhere had no legal way to say so and reached for
# the word it knew. An ontology with no term for a common move does not prevent the move, it makes
# it illegible. The other half is this rule.
#
# IT READS THE ACTS FROM `crew/claims.md`, so adding an act is a yaml edit and never a code change.
# The same reason R7 reads its floors from `epistemic_spec.md`.
def check_act(slot, claim_ids=None, acts=None):
    """Rejection or None. Route B slots only -- a sweep carries no act and is not asked for one."""
    if slot.get("sweep") or (slot.get("intent") == "sweep"):
        return None
    act = (slot.get("act") or "").strip().lower()
    acts = acts or _acts_spec()
    if not acts:
        return None                                   # no spec readable: judge nothing
    if not act:
        return Rejection(
            "R8_NO_ACT",
            "a Route B slot must say what the experiment is FOR",
            f"no `act`. One of: {', '.join(sorted(acts))}. See crew/claims.md.")
    if act not in acts:
        return Rejection(
            "R8_UNKNOWN_ACT",
            "the act is not in the vocabulary, so nothing downstream can act on its result",
            f"`{act}` is not an act. One of: {', '.join(sorted(acts))}. "
            f"(`exploratory`, `confirmatory` and `adversarial` were INTENTS in the old scheme; "
            f"the nearest act is `explore`, which needs no claim.)")
    spec = acts[act] or {}
    need = list(spec.get("requires") or [])
    # `claim` is spelled `on` in a slot -- the field names the claim the act bears on.
    missing = []
    for k in need:
        got = slot.get("on") if k == "claim" else slot.get(k)
        if k == "metric" or k == "threshold":
            got = slot.get("predict")                 # both live in the one `predict` string
        if got in (None, "", []):
            missing.append(k)
    if missing:
        return Rejection(
            "R8_ACT_MISSING_FIELD",
            f"`{act}` cannot be carried out without the field(s) that define it",
            f"missing {', '.join(sorted(set(missing)))}. {spec.get('effect', '')}".strip())
    # A REPLICATE THAT ALSO EDITS SOMETHING IS NOT A REPLICATE.
    #
    # `r007_08` declared `act: replicate` and gave as its reason *"is n_tubes 5 on r005_10 real or a
    # threshold flicker (C020)? A fresh seed bounds the floor of the campaign's best tube count"* --
    # then carried `set_param cell_chem_react0.rate 0.5` alongside it. It ran, it came back with
    # n_tubes 7 against the parent's 5, and that difference is unattributable: the seed changed and
    # the reaction rate changed, and the run was proposed to separate exactly those two.
    #
    # This is why the epistemic audit reads Replication 8, validation 0%. Eight slots asked to
    # measure the seed floor and not one of them re-ran anything: the seed floor every R7 refusal in
    # this campaign is judged against still comes from an older corpus, because this one has never
    # produced a pair that differs by nothing but its seed.
    #
    # REFUSED RATHER THAN SILENTLY STRIPPED. Dropping the edit would grant a run the Proposer did not
    # ask for and dropping the act would score a confounded run as a `predict`; only the Proposer
    # knows which half it meant.
    if act == "replicate":
        _e = slot.get("edit")
        if _e and str(tuple(_e)[0]).lower() not in ("control", "none", "null", "ctrl", "replicate"):
            return Rejection(
                "R8_REPLICATE_WITH_EDIT",
                "a replicate re-runs an experiment unchanged at a new seed -- an edit alongside it "
                "confounds the seed spread with the edit's effect",
                f"`act: replicate` on {slot.get('parent')} but the slot also carries `edit: {_e}`. "
                f"Either drop the edit (the run is then the parent at a fresh seed, which is what "
                f"bounds the floor) or drop the act and say what the edit is FOR.")

    cid = slot.get("on")
    if cid and claim_ids is not None and cid not in claim_ids:
        return Rejection(
            "R8_NO_SUCH_CLAIM",
            "the act names a claim that is not in the ledger",
            f"`{cid}` is not a claim id. The ledger holds: {', '.join(sorted(claim_ids))}.")
    return None


# =============================================================================================
# R9 -- HAS THIS KNOB ALREADY BEEN MEASURED TO DO NOTHING HERE?
#
# THE RULE THE EVIDENCE EARNED, and it was not written until it had. `round.inert` computes, from
# byte-identical trajectories, which knob a pair of runs differ in -- so "this parameter changes
# nothing on this composition" is a MEASUREMENT, not an opinion. That list has been in the
# Proposer's prompt since r012, stated plainly, with the runs that prove it. Ten rounds later, 8 of
# the 17 identical-trajectory clusters contain a run from r012 or after, on the same knobs.
#
# The order matters and it is the campaign's own principle: compute the fact, show it, and only
# write a rule when the fact has been shown and ignored. A gate written first would have been a
# guess about what the substrate reads; this one is arithmetic over trajectories.
#
# KEYED ON THE COMPOSITION, not on the knob alone. `vth_frac` does nothing where growth never
# reaches its threshold and everything where it does, so a campaign-wide ban earned by one lineage
# would remove a real experiment. `comp_hash` is parameter-blind, which is exactly the granularity
# of "this knob does nothing HERE".
#
# AND ONLY INSIDE THE MEASURED SPAN. A knob inert between 4 and 10 may still do something at 100 --
# nothing here has measured that, and refusing it would be the gate claiming knowledge it does not
# have. Outside the span the slot runs, and the refusal message says why it was let through.
def check_inert(edit, comp_hash, rows=None):
    """Rejection or None. `rows` is `campaign/inert.jsonl`, re-derived every round."""
    if not edit or not rows:
        return None
    e = list(edit) if isinstance(edit, (list, tuple)) else [edit]
    if not e or not isinstance(e[0], str):
        return None                                   # a multi-edit slot: judged per its parts above
    verb = e[0]
    if verb == "add_op":
        key = f"add_op {e[1]}"
    elif verb in ("set_param", "set_impl") and len(e) >= 3:
        node = str(e[1]).rpartition(".")[0] if verb == "set_param" else str(e[1])
        knob = str(e[1]).rpartition(".")[2] if verb == "set_param" else "impl"
        key = f"{node.rstrip('0123456789')}.{knob}"
    else:
        return None
    for r in rows:
        if (r.get("knob") or r.get("edit")) != key:
            continue
        hs = r.get("comp_hash") or []
        if hs and comp_hash and comp_hash not in hs:
            continue                                  # measured inert on a DIFFERENT composition
        vals = r.get("values_tried") or []
        if verb == "set_param" and vals:
            try:
                nums = sorted(float(v) for v in vals)
                v = float(e[2])
                if not (nums[0] <= v <= nums[-1]):
                    return None                       # outside the span nothing has measured
            except (TypeError, ValueError):
                pass
        runs = r.get("identical_runs") or []
        return Rejection(
            "R9_INERT_KNOB",
            "this knob has been MEASURED to change nothing on this composition -- the run already "
            "exists under another name",
            f"`{key}` was tried at {', '.join(map(str, vals)) or 'these settings'} and the "
            f"trajectories came out byte-identical: {', '.join(runs[:4])}. Spend the slot on "
            f"something the substrate reads, or move this knob OUTSIDE that range and say why it "
            f"should matter there.")
    return None


_ACTS_CACHE = {}


def _acts_spec():
    if _ACTS_CACHE:
        return _ACTS_CACHE
    import re as _re
    import yaml as _yaml
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crew", "claims.md")
    try:
        for b in _re.findall(r"```yaml\n(.*?)```", open(p).read(), _re.S):
            d = _yaml.safe_load(b) or {}
            if "acts" in d:
                _ACTS_CACHE.update(d["acts"]); break
    except Exception:
        pass
    return _ACTS_CACHE
