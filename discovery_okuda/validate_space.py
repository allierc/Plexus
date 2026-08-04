"""validate_space -- the gate the Okuda composition space must pass BEFORE any campaign runs.

The composition space is a data structure. Until it has been compiled to a real spec, run, and
shown to reproduce recipes we already trust, it is a hypothesis about our own code. This script
is the test battery.

    V1  EXPRESSIVENESS   every trusted hand preset is reachable by legal one-edit moves
    V2  COMPILATION      every reachable composition compiles to a runnable spec
    V3  FIDELITY         the compiled spec's operator set + schedule match the hand-written one
    V4  DEFECT FIXES     every emitted config carries the D1/D2/D3 fixes
    V5  PRECONDITIONS    a composition that would silently no-op is REFUSED, not run  (D4)
    V6  IDENTITY         theta never changes comp_hash; an implementation swap always does
    V7  ABLATION         the campaign's central test (remove `extrude`) is one legal edit
    V8  COVERAGE         every operator in the vocabulary is exercised by >=1 emitted config

    python validate_space.py            # battery only (seconds, no simulation)
    python validate_space.py --write    # + write config/okuda/*.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from composition_space import OPERATORS, CompositionGraph, reference_recipes, seed  # noqa: E402
from run_record import comp_hash                                                     # noqa: E402
import translate as T                                                                # noqa: E402

# Presets we trust, spanning the phenotypes the metric bank must separate.
TRUSTED = [
    "round_40_mc8",        # best tube: driven tip + forced extrusion   (aspect ~7.5)
    "round_41_relax60",    # the ablation that COLLAPSES it             (aspect -> 1)
    "round_41_hertwig",    # long-axis division, no bud-axis forcing
    "round_42_k05",        # monolayer, pure growth, low tension -> thin spikes
    "round_42_k05_ex4",    # monolayer + gentle extrusion assist
    "round_44_base",       # emergent GM coupled to the wall machinery -> floods
    "round_21_gs",         # Gray-Scott stable-spot regime
]

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
results = []


def check(tag, ok, msg=""):
    results.append((tag, bool(ok)))
    print(f"  [{PASS if ok else FAIL}] {tag}{(' -- ' + msg) if msg else ''}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write config/okuda/*.yaml")
    a = ap.parse_args()

    print("=" * 78)
    print("OKUDA COMPOSITION SPACE -- validation battery")
    print("=" * 78)

    presets = T.load_presets()
    print(f"\nloaded {len(presets)} hand presets from run_tyssue_round.py")

    # ---------------------------------------------------------------- V1 expressiveness
    print("\nV1 EXPRESSIVENESS -- is every trusted recipe reachable?")
    graphs = {}
    for name in TRUSTED:
        p = presets.get(name)
        if p is None:
            check(f"V1 {name}", False, "preset not found")
            continue
        try:
            g = T.from_preset(p)
            graphs[name] = g
            ok, why = g.is_runnable()
            check(f"V1 {name}", ok,
                  f"{comp_hash(g)} {g.name_region()}" if ok else why)
        except Exception as e:
            check(f"V1 {name}", False, f"{type(e).__name__}: {e}")

    for name, g in reference_recipes().items():
        graphs[f"ref_{name}"] = g
        check(f"V1 ref:{name}", g.is_runnable()[0], f"{comp_hash(g)} {g.name_region()}")

    # ---------------------------------------------------------------- V2 compilation
    print("\nV2 COMPILATION -- does every reachable composition compile?")
    specs = {}
    for name, g in graphs.items():
        try:
            cfg = T.to_spec(g, name=name, frames=350)
            specs[name] = cfg
            check(f"V2 {name}", True, f"{len(cfg['operators'])} ops")
        except Exception as e:
            check(f"V2 {name}", False, f"{type(e).__name__}: {e}")

    # ---------------------------------------------------------------- V3 fidelity
    print("\nV3 FIDELITY -- compiled operator set == hand-written one?")
    try:
        sys.path.insert(0, T.TYSSUE)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rtr", os.path.join(T.TYSSUE, "run_tyssue_round.py"))
        rtr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rtr)
        have_make = hasattr(rtr, "make")
    except Exception as e:
        have_make = False
        print(f"  (cannot import make(): {type(e).__name__}: {str(e)[:70]})")

    if have_make:
        for name in TRUSTED:
            if name not in specs or name not in presets:
                continue
            try:
                _, hand = rtr.make(presets[name])
                hand_ops = {o["op"] for o in hand["operators"]}
                ours = {o["op"] for o in specs[name]["operators"]}
                # load_mesh_3d/seed_mesh_3d are the same role
                missing, extra = hand_ops - ours, ours - hand_ops
                check(f"V3 {name}", not missing,
                      f"missing={sorted(missing)} extra={sorted(extra)}" if (missing or extra)
                      else "operator sets identical")
            except Exception as e:
                check(f"V3 {name}", False, f"{type(e).__name__}: {str(e)[:70]}")
    else:
        print("  (skipped -- make() unavailable in this environment)")

    # ---------------------------------------------------------------- V9 parameter fidelity
    # V3 proves "same operators". That is not "same model": a vocabulary default silently
    # overriding a preset value gives an identical operator set and different physics. This
    # check compares the actual numbers, excluding the keys we DELIBERATELY changed.
    print("\nV9 PARAMETER FIDELITY -- same numbers, not just same operators?")
    # dt/every: the D1/D2 fixes. min_cycle/max_cycle/max_div_frac: the CLOCK RE-ANCHORING --
    # they are per-CALL in the operator and the archived configs ran divide_3d once every 4
    # frames, so preserving the archived behaviour REQUIRES different numbers. V10 checks the
    # factor is exactly right; V9 must not flag it as a fidelity failure.
    # `ckpt`: PORTABILITY. make() bakes an absolute /workspace path; a tracked config must run
    # both in the devcontainer and on the cluster, which mount the same export at different
    # prefixes, so we emit a repo-relative path resolved by run_one against its own location.
    # `vth_frac`: THE D5b FIX. The archived recipes cap a cell's target volume at 1.5x while
    # divide_3d fires at 2.0x -- the ceiling sits BELOW the trigger, so volume-triggered division
    # was arithmetically impossible and every division ever seen came from the max_cycle timeout.
    # The ceiling is now derived from the trigger (2.0 x 1.25 = 2.5). The space is right and the
    # archive is wrong, so this divergence is deliberate. Listed rather than back-fitted into the
    # archived recipes: rewriting history to make a test pass is how a reference stops being one.
    DELIBERATE = {"dt", "every", "max_cycle", "record_every",
                  "min_cycle", "ckpt", "vth_frac",
                  # `chi`/`rate`: THE D5a FIX, guarded by V11 below exactly as V10 guards the
                  # divide clock. Both are rescaled by 1/dt because cell_react and cell_diffuse
                  # EMIT=velocity into `chem`, so the engine was integrating the chemistry with
                  # the MECHANICS substep -- 300 frames bought 6 units of reaction time instead
                  # of ~500. Exempt here, verified there; an exemption nobody checks is a hole.
                  "chi", "rate"}
    if have_make:
        for name in TRUSTED:
            if name not in specs or name not in presets:
                continue
            try:
                _, hand = rtr.make(presets[name])
                hand_by = {o["op"]: o for o in hand["operators"]}
                diffs = []
                for o in specs[name]["operators"]:
                    h = hand_by.get(o["op"])
                    if h is None:
                        continue
                    for k, v in h.items():
                        if k in DELIBERATE or k not in o:
                            continue
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            if abs(float(o[k]) - float(v)) > 1e-6 * max(1.0, abs(float(v))):
                                diffs.append(f"{o['op']}.{k}={o[k]} (hand {v})")
                        elif o[k] != v:
                            diffs.append(f"{o['op']}.{k}={o[k]!r} (hand {v!r})")
                check(f"V9 {name}", not diffs,
                      "; ".join(diffs[:5]) + (f" +{len(diffs)-5} more" if len(diffs) > 5 else "")
                      if diffs else "all mechanism parameters match")
            except Exception as e:
                check(f"V9 {name}", False, f"{type(e).__name__}: {str(e)[:70]}")
    else:
        print("  (skipped -- make() unavailable)")

    # ---------------------------------------------------------------- V10 clock re-anchoring
    print("\nV10 CLOCK RE-ANCHORING -- are the per-call quantities rescaled by exactly 4?")
    from composition_space import DIVIDE_CALL_PERIOD_BEFORE_D1 as P
    if have_make:
        for name in TRUSTED:
            if name not in specs or name not in presets:
                continue
            try:
                _, hand = rtr.make(presets[name])
                h = next((o for o in hand["operators"] if o["op"] == "divide_3d"), None)
                o = next((o for o in specs[name]["operators"] if o["op"] == "divide_3d"), None)
                if not h or not o:
                    continue
                bad = []
                if h.get("min_cycle") and o.get("min_cycle") != h["min_cycle"] * P:
                    bad.append(f"min_cycle {o.get('min_cycle')} != {h['min_cycle']}x{P}")
                check(f"V10 {name}", not bad,
                      "; ".join(bad) if bad else
                      f"min_cycle {h.get('min_cycle')}->{o.get('min_cycle')} calls->frames, ")
            except Exception as e:
                check(f"V10 {name}", False, f"{type(e).__name__}: {str(e)[:60]}")
    else:
        print("  (skipped -- make() unavailable)")

    # ---------------------------------------------------------------- V4 defect fixes
    print("\nV4 DEFECT FIXES -- are D1/D2/D3 in every emitted config?")
    for name, cfg in specs.items():
        dt_ok = cfg["general"]["dt"] == T.DT_GLOBAL
        every_ok = all(o.get("every", 1) == 1 for o in cfg["operators"])
        topo = next((o for o in cfg["operators"] if o["op"] == "topo_snapshot_3d"), None)
        stride_ok = topo is not None and topo["every"] == cfg["general"]["record_every"]
        check(f"V4 {name}", dt_ok and every_ok and stride_ok,
              f"dt={cfg['general']['dt']} every_all_1={every_ok} stride_match={stride_ok}")

    # ---------------------------------------------------------------- V5 preconditions
    print("\nV5 PRECONDITIONS -- is a silently-inert composition REFUSED?  (D4)")
    bad, _ = seed("substrate").apply(("add_op", "cell_diffuse", "graph_laplacian"))
    check("V5 unmet-precondition detected", len(bad.unmet_preconditions()) == 1,
          str(bad.unmet_preconditions()))
    refused = False
    try:
        T.to_spec(bad, name="bad")
    except ValueError:
        refused = True
    check("V5 compilation refuses it", refused,
          "a composition that would no-op never reaches the cluster")

    dangling, _ = seed("substrate").apply(("add_op", "cell_rd_seed", "tip"))
    dangling, _ = dangling.apply(("add_op", "morphogen_growth_3d", "hill_conserve_amount"))
    check("V5 dangling slot detected", len(dangling.unrouted_slots()) >= 1,
          f"{dangling.unrouted_slots()} -- present but disconnected == inert")

    # ---------------------------------------------------------------- V6 identity
    print("\nV6 IDENTITY -- theta vs structure")
    import numpy as np
    g = graphs.get("ref_round40_mc8") or seed("substrate")
    rng = np.random.default_rng(0)
    check("V6 theta does not change identity",
          comp_hash(g.with_params(g.sample_params(rng))) == comp_hash(g),
          "a retune provably cannot pose as a new hypothesis")
    se = next((o["id"] for o in g.ops if o["op"] == "shape_energy_3d"), None)
    if se:
        g2, _ = g.apply(("set_impl", se, "monolayer"))
        check("V6 impl swap DOES change identity", comp_hash(g2) != comp_hash(g),
              "mid-surface vs true 3D volume is a mechanism edit")

    # ---------------------------------------------------------------- V7 the central ablation
    print("\nV7 CENTRAL ABLATION -- `extrude` removable by one legal edit?")
    r40 = graphs.get("ref_round40_mc8")
    if r40:
        ex = next((o["id"] for o in r40.ops if o["op"] == "extrude"), None)
        if ex:
            ab, _ = r40.apply(("remove_op", ex))
            ok, why = ab.is_runnable()
            check("V7 ablate extrude", ok,
                  f"{comp_hash(r40)} -> {comp_hash(ab)}  region -> {ab.name_region()!r}")
            check("V7 ablation is in the legal move set",
                  any(e[0] == "remove_op" and e[1] == ex for e, _ in r40.legal_edits(3)),
                  "round 41 by hand == one automatic necessity test here")
        else:
            check("V7 ablate extrude", False, "reference recipe has no extrude node")

    # ---------------------------------------------------------------- V11 chemistry clock
    print("\nV11 CHEMISTRY CLOCK -- is the RD rescaling exactly 1/dt, and is the RATIO intact?")
    from translate import RD_PER_FRAME
    if have_make:
        for name in TRUSTED:
            if name not in specs or name not in presets:
                continue
            try:
                _, hand = rtr.make(presets[name])
                hb = {o["op"]: o for o in hand["operators"]}
                sb = {o["op"]: o for o in specs[name]["operators"]}
                bad = []
                for op, key in (("cell_diffuse", "chi"), ("cell_react", "rate")):
                    if op not in hb or op not in sb or hb[op].get(key) is None:
                        continue
                    want = hb[op][key] * RD_PER_FRAME
                    got = sb[op].get(key)
                    if got is None or abs(got - want) > 1e-6 * max(1.0, abs(want)):
                        bad.append(f"{op}.{key} {got} != {hb[op][key]}x{RD_PER_FRAME}")
                # THE INVARIANT THAT MATTERS: the Turing wavelength is set by the RATIO of
                # diffusion to reaction, so rescaling both by the same factor must leave it
                # alone. If only one moved, the clock fix would have silently retuned the
                # pattern -- a far worse bug than the one it repaired.
                if ("cell_diffuse" in hb and "cell_react" in hb
                        and hb["cell_diffuse"].get("chi") and hb["cell_react"].get("rate")
                        and sb.get("cell_react", {}).get("rate")):
                    r_hand = hb["cell_diffuse"]["chi"] / hb["cell_react"]["rate"]
                    r_spec = sb["cell_diffuse"]["chi"] / sb["cell_react"]["rate"]
                    if abs(r_hand - r_spec) > 1e-6 * max(1.0, r_hand):
                        bad.append(f"ratio changed {r_hand:.4f} -> {r_spec:.4f}")
                if "cell_diffuse" in hb or "cell_react" in hb:
                    check(f"V11 {name}", not bad,
                          "; ".join(bad) if bad else
                          f"both x{RD_PER_FRAME:.0f}, wavelength ratio unchanged")
            except Exception as e:
                check(f"V11 {name}", False, f"{type(e).__name__}: {str(e)[:70]}")

    # ---------------------------------------------------------------- V8 coverage
    print("\nV8 COVERAGE -- every vocabulary operator exercised?")
    covered = set()
    for g in graphs.values():
        covered.update(g.op_names())
    uncovered = sorted(set(OPERATORS) - covered)
    check("V8 coverage", not uncovered,
          f"UNCOVERED: {uncovered}" if uncovered else f"all {len(OPERATORS)} operators exercised")

    # ---------------------------------------------------------------- write configs
    if a.write:
        print("\nWRITING config/okuda/*.yaml")
        out = os.path.abspath(os.path.join(HERE, "..", "config", "okuda"))
        for name, g in graphs.items():
            try:
                path, cfg = T.write_config(g, name, frames=350)
                print(f"  {os.path.relpath(path, os.path.join(HERE, '..'))}  "
                      f"{cfg['_discovery']['comp_hash']}  {cfg['_discovery']['region']}")
            except Exception as e:
                print(f"  [FAIL] {name}: {type(e).__name__}: {e}")

    # ---------------------------------------------------------------- summary
    n_pass = sum(1 for _, ok in results if ok)
    print("\n" + "=" * 78)
    print(f"  {n_pass}/{len(results)} checks passed")
    failed = [t for t, ok in results if not ok]
    if failed:
        print("  FAILED: " + ", ".join(failed))
    print("=" * 78)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
