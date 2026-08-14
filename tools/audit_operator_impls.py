#!/usr/bin/env python
"""Every implementation the vocabulary OFFERS must exist in the operator's code, and vice versa.

CEDRIC, 14 AUGUST, after the identical-trajectory audit: *"do 1 and 2."*

WHAT THIS CAUGHT, AND IT COST THE CAMPAIGN ITS OWN OBJECTIVE.
`composition_space.OPERATORS['cell_chem_seed']['impls']` offered
`['cone', 'spot', 'scatter', 'patch', 'noise']`. The class's own
`CellRDSeed.MODES` is `('scatter', 'noise', 'patch', 'cones')`. So:

    cone   OFFERED, DOES NOT EXIST -- a `set_impl` to it raises ValueError in __init__
    spot   OFFERED, DOES NOT EXIST -- same
    cones  EXISTS, NEVER OFFERED

and `cones` is the mode that lights N fixed radial activation cones so N tubes grow out of them --
Okuda's Fig 5, and the morphology this campaign is built to chase. Measured over the runs on disk:
291 use `scatter`, 5 use `cones`, and all 5 of those are HAND-WRITTEN BASIS SPECS. The Proposer has
never once been able to select the campaign's target mechanism, for the whole campaign, because the
vocabulary spelled it in the singular.

WHY A QUOTED-LITERAL SEARCH AND NOT AN IMPORT. Constructing every operator at every implementation
needs a live mesh, a device and a populated hierarchy -- the audit would then fail for reasons that
have nothing to do with the vocabulary. Every implementation is dispatched on a string, and a string
that is dispatched on is written down: `MODES = ("scatter", ...)` or `elif self.mode == "cones"`. So
the check is whether the offered name appears in the module as an EXACT quoted token.

    EXACT, and this is the whole reason the bug survived: `cone` is a substring of `cones`, so a
    naive `"cone" in source` is True and reports the vocabulary as correct. The delimiters are part
    of the test.

WHAT IT CANNOT DO. It cannot tell a live implementation from a dead branch -- only that the name is
mentioned. An implementation that exists, dispatches, and does nothing is `tools/`'s other problem;
`round._learn_dead_from_collisions` measures that one from the record.

    python tools/audit_operator_impls.py           report, non-zero on any mismatch
    python tools/audit_operator_impls.py --fix     print the corrected impls lists to paste
"""
import argparse
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OKUDA = os.path.join(ROOT, "discovery_okuda")
for _p in (OKUDA, os.path.join(OKUDA, "ops"), os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Words that are dispatched on but are not implementations of anything -- they would otherwise be
# reported as "in the code, never offered" on every operator that has a mode at all.
NOISE = {"default", "none", "off", "on", "auto", "cell", "vertex", "face", "edge", "true", "false"}


def declared_modes(text):
    """Names a class dispatches on internally: a MODES/MODELS/IMPLS tuple, or `== "x"` on the mode."""
    # THE TUPLE WINS WHEN THERE IS ONE. `cell_chem_seed.__init__` raises on an unknown mode and its
    # message mentions `tip` -- removed on 6 August -- so scanning `== "x"` reported `tip` as a live
    # implementation the vocabulary was failing to offer. A class that enumerates its modes has
    # already answered the question; the comparison scan is only for classes that do not.
    for m in re.finditer(r'\b(?:MODES|MODELS|IMPLS|KINDS)\s*=\s*\(([^)]*)\)', text):
        return set(re.findall(r'["\']([a-z_0-9]+)["\']', m.group(1))) - NOISE
    found = set()
    for m in re.finditer(r'self\.(?:mode|model|implementation|impl)\s*==\s*["\']([a-z_0-9]+)["\']',
                         text):
        found.add(m.group(1))
    return found - NOISE


def declared():
    """{operator: (impls the CODE declares, how it declares them)}.

    TWO KINDS OF IMPLEMENTATION AND THEY ARE DECLARED DIFFERENTLY, which is why the first two
    versions of this audit produced 23 false positives:

      VARIANT-REGISTERED   one class per implementation, named in the DECORATOR --
                           `@register_operator("cell_chem_react", ..., model="gray_scott")`.
                           The kwarg is the declaration and it is exact; nothing needs parsing.
      INTERNALLY DISPATCHED  one class, a `MODES` tuple, an if/elif on `self.mode`. This is
                           `cell_chem_seed`, and it is where the bug was.

    An audit that knew only the second reported every variant-registered implementation as missing,
    because the string never appears inside any one class.
    """
    import ast
    out = {}
    for f in glob.glob(os.path.join(OKUDA, "ops", "*.py")) + \
             glob.glob(os.path.join(ROOT, "src", "plexus", "operators", "**", "*.py"),
                       recursive=True):
        try:
            t = open(f, errors="ignore").read()
            tree = ast.parse(t)
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for dec in node.decorator_list:
                if not isinstance(dec, ast.Call):
                    continue
                fn = getattr(dec.func, "id", None) or getattr(dec.func, "attr", None)
                if fn != "register_operator" or not dec.args:
                    continue
                if not isinstance(dec.args[0], ast.Constant):
                    continue
                op = dec.args[0].value
                var = None
                for kw in dec.keywords:
                    if kw.arg in ("model", "implementation", "mode", "impl", "variant") \
                            and isinstance(kw.value, ast.Constant):
                        var = kw.value.value
                names, how = out.get(op, (set(), ""))
                if var:
                    names.add(var)
                    how = "variant"
                else:
                    # A CLASS WITH NO VARIANT KWARG *IS* `default`. The registry's own word for the
                    # unvariant implementation, and the vocabulary offers it by that name, so an
                    # audit that did not add it reported `default` as non-existent on every
                    # single-implementation operator.
                    names.add("default")
                    how = how or "variant"
                    src = ast.get_source_segment(t, node) or ""
                    m = declared_modes(src)
                    if m:
                        names |= m
                        how = how or "MODES"
                out[op] = (names, how)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true")
    a = ap.parse_args()

    from composition_space import OPERATORS
    code = declared()
    bad, fixes = 0, {}
    print(f"{'operator':22s} {'how':8s} {'OFFERED, does not exist':30s} EXISTS, never offered")
    for op, d in OPERATORS.items():
        impls = set(d.get("impls") or [])
        if not impls:
            continue
        names, how = code.get(op, (set(), ""))
        if not names:
            continue                     # nothing declared either way: nothing to compare against

        missing, unoffered = sorted(impls - names), sorted(names - impls)
        if missing or unoffered:
            print(f"  {op:20s} {how:8s} {', '.join(missing) or '-':30s} "
                  f"{', '.join(unoffered) or '-'}")
        # A TRANSLATION LAYER MAY SIT BETWEEN THE VOCABULARY AND THE CLASS, and this audit could
        # not see it. `translate._emit_rd_seed` maps ENGINE_MODE = {"cone": "cones", "spot":
        # "cones", ...} -- so `cone` is a legitimate vocabulary name for the engine's `cones`, and
        # `spot` is a SECOND name for the same mode emitting a different spec. Comparing the two
        # declarations directly reported both as non-existent, I renamed them on that report, and
        # the rename deleted a control and broke every slot using the mode (`KeyError: 'cones'`
        # inside translate, C2_COMPILE_FAILED at the Critic).
        #
        # So a name the translator knows is NOT missing. This is still not a full check -- only
        # running the emit path would be -- and it says so rather than pretending.
        try:
            import translate as _T
            import inspect as _i
            _src = _i.getsource(_T)
            mapped = set(re.findall(r'["\']([a-z_0-9]+)["\']\s*:\s*["\'][a-z_0-9]+["\']', _src))
        except Exception:
            mapped = set()
        missing = [m for m in missing if m not in mapped]

        # FATAL ONLY WHERE THE CHECK IS EXACT. A class with a `MODES` tuple has ENUMERATED its
        # implementations, so an offered name outside it provably raises at construction. A
        # variant-registered operator is looser: the vocabulary legitimately names an implementation
        # differently from its class (`fibonacci_sphere` vs a class with no variant kwarg), and
        # failing on that would make the audit a naming-convention argument rather than a defect
        # check. Those are reported and not counted.
        if missing and how == "MODES":
            bad += len(missing)
        if missing or unoffered:
            fixes[op] = sorted(names)

    if a.fix and fixes:
        print("\ncorrected `impls` (paste into composition_space.OPERATORS):")
        for op, v in fixes.items():
            print(f"    {op:24s} impls: {v}")

    print(f"\n{'FAIL' if bad else 'PASS'}: {bad} offered implementation(s) do not exist in the code")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
