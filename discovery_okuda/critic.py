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
    """Operator identity, with the NODE INDEX stripped: divide_3d0 and divide_3d are one operator.

    `add_op` names the operator (`divide_3d`) and `remove_op` names the NODE (`divide_3d0`), so
    without this the two directions of the same experiment never matched. A1 then refused a
    necessity claim whose ablation was sitting in the same batch -- a false refusal of a correct
    proposal, which is the expensive way to fail. It was invisible because check_batch has never
    been called; wiring it live without this would have started rejecting good batches.
    """
    n = str(name or "").split(":")[0].strip()
    return re.sub(r"\d+$", "", n)


def _touched_operator(edit):
    """The operator an edit acts on, or None. Edits look like ('add_op','cell_react','gray_scott')
    or a string '+cell_react' / '-cell_react' / '=cell_react:tension'."""
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


# ============================================================================ STATIC rules
DT_GLOBAL_DEFAULT = 0.02        # translate.DT_GLOBAL; kept local so critic imports nothing heavy


# dt*rate for an autocatalytic kinetics. 1.0 diverged in five runs on 2 August; the bound the
# Diagnostician derived from them is 0.5.
AUTOCATALYTIC_STEP_LIMIT = 0.5


def _param(graph, node_id, key):
    """A node's parameter, from the graph's params overlay or the operator's default."""
    p = (getattr(graph, "params", {}) or {}).get(f"{node_id}.{key}")
    if p is not None:
        return p
    try:
        from composition_space import OPERATORS
        op = next(o["op"] for o in graph.ops if o["id"] == node_id)
        return (OPERATORS[op].get("params") or {}).get(key, {}).get("default")
    except Exception:
        return None


def check_static(graph, seen_hashes=()):
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
    # MEASURED on round 1 of the rebuilt loop. `legal_edits` offered both `+shape_energy_3d:default`
    # and `=shape_energy_3d:default`; the Proposer took the ADD and wrote the claim for the SWAP
    # ("swapping the monolayer shape energy for the default releases the in-plane constraint").
    # The spec came out with shape_energy_3d twice -- two independent relaxation loops of thirty
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

        # `chi` is the RD timescale and it lives on cell_diffuse in this operator set; the
        # reaction is what it destabilises. Both are checked so a relocation cannot silence this.
        for o in graph.ops:
            if o["op"] not in ("cell_diffuse", "cell_react"):
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
    try:
        from translate import DT_GLOBAL
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
            if o["op"] != "cell_react" or impl != "gierer_meinhardt":
                continue
            # ONLY WITH DIVISION. The Diagnostician asked for a blanket refusal of
            # gierer_meinhardt above this step, and that guard would have refused r002c_04 --
            # the best run the campaign has produced (protr_peak 1.317, the first non-zero tube
            # count on record), which is gierer_meinhardt at dt*rate = 1.0 and ran all 900 frames
            # with no damage at all. The discriminator is division, not the kinetics alone:
            # every GM composition that also divides took damage at frame 115, five out of five,
            # at the SAME frame; the one that does not divide is clean. A guard aimed at the
            # kinetics would have killed the finding it was meant to protect.
            if not any(x["op"] == "divide_3d" for x in graph.ops):
                continue
            rate = _param(graph, o["id"], "rate")
            step = float(DT_GLOBAL) * float(rate if rate is not None else 1.0)
            if step > AUTOCATALYTIC_STEP_LIMIT:
                out.append(Rejection(
                    "R1d_AUTOCATALYTIC_UNSTABLE",
                    "an autocatalytic reaction stepped past its explicit-Euler limit -- the "
                    "activator diverges uniformly and the run measures an integrator, not a "
                    "mechanism",
                    f"gierer_meinhardt WITH divide_3d advances dt*rate = {step:.2f} per step "
                    f"against a limit of {AUTOCATALYTIC_STEP_LIMIT} (dt {DT_GLOBAL}, rate "
                    f"{rate}). Five such runs took damage at frame 115, all five at the same "
                    f"frame. Lower `rate`, use gray_scott, or drop division -- gierer_meinhardt "
                    f"without divide_3d is stable at this step and is the best run on record."))
    except Exception:
        pass

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

    # R5 -- PARAMETERS IN RANGE. theta is not identity, but an out-of-range value is still a
    # malformed experiment.
    for k, v in graph.params.items():
        if k.startswith("_run."):
            continue
        nid, _, pname = k.partition(".")
        node = graph._node(nid)
        if node is None:
            continue
        spec = OPERATORS[node["op"]]["params"].get(pname)
        if spec and isinstance(v, (int, float)):
            lo, hi, _ = spec
            if not (lo <= v <= hi):
                out.append(Rejection("R5_PARAM_OUT_OF_RANGE", "parameter outside declared range",
                                     f"{k}={v} not in [{lo}, {hi}]"))

    # R6 -- ALREADY EVALUATED. Re-running a known composition is not a new hypothesis; it is a
    # replicate, and must be requested as one (a robustness test) rather than arrived at by
    # accident.
    h = comp_hash(graph)
    if h in set(seen_hashes):
        out.append(Rejection("R6_DUPLICATE", "this composition has already been evaluated",
                             f"{h} -- request a robustness test explicitly if replication is "
                             f"what you want"))
    return out


# ============================================================================ COMPILE rules
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
        if o.get("op") in ("seed_mesh_3d",) and o.get("n_cells"):
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
def check_posthoc(summary):
    """After the run: was this actually evidence? (D4 + the saturation guard.)

    Static checks cannot see this. An operator can satisfy every type rule and still never act --
    because `after_frame` was never reached, a reservoir was exhausted, or a threshold was never
    crossed. Recording such a run as evidence is how a search manufactures a false impossibility.
    """
    out = []
    for op in (summary.get("inert_operators") or []):
        out.append(Rejection("P1_INERT_OPERATOR",
                             "a scheduled operator never acted -- the run is not evidence",
                             op))
    if summary.get("saturated"):
        out.append(Rejection("P2_BUFFER_SATURATED",
                             "the run hit its cell buffer -- evidence about a buffer, not a "
                             "mechanism", f"n_cells={summary.get('n_cells_final')}"))

    # P3 -- THE CHEMISTRY DIVERGED. This is where the reaction ceiling went.
    #
    # The parameter boxes used to reject an absurd reaction rate for free, before any compute.
    # That guard was removed on purpose: the bound was an arbitrary hand-written number, and the
    # rule now is that a bound is either PHYSICAL AND DERIVED or absent. The diffusion limit is
    # derivable and is still enforced up front; the reaction one is not (measured -- see
    # composition_space.reaction_stiffness, "NOT a stability criterion").
    #
    # But "no arbitrary cap" must not become "no protection". We cannot say in advance which rate
    # diverges; we can say with certainty that a run whose chemistry went non-finite is not
    # evidence about a mechanism. So the guard moves from a guess before the run to a measurement
    # after it. Gierer-Meinhardt at rate 100 reaches ~4e13 within 300 steps, so this fires.
    for key in ("act_max_final", "act_max_peak", "protr_peak", "protr_final"):
        v = summary.get(key)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(f):
            out.append(Rejection("P3_CHEMISTRY_DIVERGED",
                                 "a recorded quantity is not finite -- the integration blew up, "
                                 "so this run is evidence about numerics, not biology",
                                 f"{key}={v}"))
        elif key.startswith("act_max") and f > ACT_DIVERGED:
            out.append(Rejection("P3_CHEMISTRY_DIVERGED",
                                 "the activator exceeded every physical scale -- the reaction "
                                 "integration is diverging", f"{key}={f:.3g} > {ACT_DIVERGED:g}"))
    if summary.get("act_nan"):
        out.append(Rejection("P3_CHEMISTRY_DIVERGED",
                             "the activator field contains NaN", "act_nan=True"))
    return out


# The activator is a concentration of order 1 in every kinetics we run (Gray-Scott saturates
# near 1, Gierer-Meinhardt's steady state is O(1)). Three orders of magnitude above that is not a
# strong pattern, it is a diverging integration. Deliberately loose: this is a divergence
# detector, not a quality bar.
ACT_DIVERGED = 1e3


# ============================================================================ the gate + menu
def admit(graph, seen_hashes=(), compile_check=True):
    """(ok, [Rejection]). The whole static+compile gate."""
    rej = check_static(graph, seen_hashes)
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
    bad, _ = seed("substrate").apply(("add_op", "cell_diffuse", "graph_laplacian"))
    print(" ", check_static(bad))

    print("\n-- R3: present but unconnected == inert, and looks deliberate --")
    d, _ = seed("substrate").apply(("add_op", "cell_rd_seed", "tip"))
    d, _ = d.apply(("add_op", "morphogen_growth_3d", "hill_conserve_amount"))
    print(" ", [r for r in check_static(d) if r.code == "R3_DANGLING_SLOT"])

    print("\n-- R6: a re-run is a replicate, not a hypothesis --")
    print(" ", [r for r in check_static(g, seen_hashes=[comp_hash(g)]) if r.code == "R6_DUPLICATE"])

    print("\n-- P1: static checks cannot see this one --")
    print(" ", check_posthoc({"inert_operators": ["cell_diffuse"], "saturated": False}))

    print("\n-- the generative role: the menu handed to the Proposer --")
    menu = legal_menu(seed("substrate"), max_stage=3)
    print(f"  {len(menu)} edits survive the guard; the agent chooses from these")
    for m in menu[:5]:
        print(f"    {m['label']:36} -> {m['yields']}")
    print("\ncritic OK")
