"""inventory -- the mechanism list, extracted from the clone rather than remembered.

The first thing an agent wants to do with a paper is *list what is in it*. That list is also the
first place a reproduction goes wrong: a mechanism nobody wrote down is a mechanism nobody
ablates, and the reproduction then differs from the reference for a reason that is invisible in
the record. (The Okuda track lost two days to exactly this shape of omission -- an emitter that
never passed its implementation through, so an ablation reported "no effect" without ever running
the new code.)

So the inventory is mechanical: walk the clone's AST, take every class that subclasses a step or
potential base, record its module, its line, its docstring's first line, its constructor
parameters, and the state fields it declares. No LLM, no judgement -- an enumeration a human can
check against `git grep class` in ten seconds.

What it produces is a record at status `candidate`: named, located, and NOT yet believed.
Inspection, normalization and the verdict are the agent's work in later phases.

    python inventory.py                # print
    python inventory.py --write        # seed atlas_record.yaml (refuses to clobber)
"""
from __future__ import annotations

import argparse
import ast
import os
import subprocess

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
CLONE = os.path.join(PLEXUS, "papers", "jax-morph")
PKG = os.path.join(CLONE, "jax_morph")
RECORD = os.path.join(HERE, "atlas_record.yaml")

# Base classes that mark "this is a mechanism", in the reference's own vocabulary.
STEP_BASES = {"SimulationStep", "StochasticStep", "ODEController"}
POTENTIAL_BASES = {"Potential", "PairwisePotential"}


def _first_doc_line(node):
    d = ast.get_docstring(node) or ""
    return d.strip().splitlines()[0] if d else None


def _init_params(node):
    """Constructor parameters, which are the tunables the atlas has to give roles to."""
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            args = [a.arg for a in item.args.args if a.arg != "self"]
            args += [a.arg for a in item.args.kwonlyargs]
            return args
    return []


def _bases(node):
    out = []
    for b in node.bases:
        if isinstance(b, ast.Name):
            out.append(b.id)
        elif isinstance(b, ast.Attribute):
            out.append(b.attr)
    return out


def scan() -> list:
    found = []
    for dirpath, dirnames, filenames in os.walk(PKG):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", "guides")]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, PLEXUS)
            with open(path, errors="replace") as f:
                src = f.read()
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                bases = _bases(node)
                if STEP_BASES & set(bases):
                    role = "step"
                elif POTENTIAL_BASES & set(bases):
                    role = "potential"
                else:
                    continue
                found.append({
                    "raw_name": node.name,
                    "raw_kind": f"{role} ({', '.join(bases)})",
                    "code_path": f"{rel}:L{node.lineno}",
                    "summary": _first_doc_line(node),
                    "params": {p: None for p in _init_params(node)},
                })
    found.sort(key=lambda m: (m["code_path"]))
    return found


def _slug(name):
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i and not name[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


# Mechanisms that are NOT classes: architectural contracts stated in the guides. They are the
# most interesting entries in this whole record -- the reference's operator-composition semantics
# -- and an AST walk cannot see them, so they are listed explicitly with their source.
ARCHITECTURAL = [
    {"raw_name": "Lie-Trotter macro-step split",
     "raw_kind": "integration contract",
     "code_path": "papers/jax-morph/jax_morph/core/step.py:L1",
     "paper_section": "guides/concepts.md, 'A simulation integrates a hybrid dynamical system'",
     "summary": "One macro-step = discrete o dynamic o quasistatic, each advancing the shared "
                "state over dt; first-order accurate operator splitting."},
    {"raw_name": "step type: quasistatic / dynamic / discrete",
     "raw_kind": "step taxonomy",
     "code_path": "papers/jax-morph/jax_morph/core/step.py:L1",
     "paper_section": "guides/concepts.md, 'Models are ordered step pipelines'",
     "summary": "A step is classified by its TIME-SCALE SEMANTICS (fast constraint / finite rate "
                "/ instantaneous event), orthogonal to what part of the state it touches."},
    {"raw_name": "declared field dataflow validation",
     "raw_kind": "composition contract",
     "code_path": "papers/jax-morph/jax_morph/core/state.py:L1",
     "paper_section": "guides/concepts.md, 'Physics and control compose through fields'",
     "summary": "Steps couple ONLY through named state fields; reads/writes are declared and "
                "cross-validated when the model is built, so recomposition is add/remove/reorder."},
    {"raw_name": "stochastic trace / replay / score",
     "raw_kind": "stochasticity contract",
     "code_path": "papers/jax-morph/jax_morph/core/logp.py:L1",
     "paper_section": "guides/concepts.md, 'Differentiability -- score-based'",
     "summary": "A stochastic step samples parameter-free noise, records an ephemeral trace, "
                "replays the trace to produce the effect, and scores the same trace in logp -- "
                "which is what makes a discrete event (division, death) differentiable."},
]


def build() -> dict:
    sha = subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip() or None
    mechs = []
    for i, m in enumerate(scan() + ARCHITECTURAL, 1):
        mechs.append({
            "id": _slug(m["raw_name"].replace(" ", "_").replace("/", "_").replace("-", "_")),
            "order": i,
            "raw_name": m["raw_name"],
            "raw_kind": m["raw_kind"],
            "code_path": m["code_path"],
            "paper_section": m.get("paper_section"),
            "summary": m["summary"],
            "params": m.get("params", {}),
            "verdict": None,
            "of": None,
            "why": None,
            "contract": None,
            "status": "candidate",
            "module": None,
            "test": None,
            "evidence": {"oracle_run": None, "diff_metric": None,
                         "threshold": None, "passed": None},
        })
    return {
        "repository": "fmottes/jax-morph",
        "paper": "10.1038/s43588-025-00851-4  (Deshpande, Mottes et al., "
                 "Engineering morphogenesis of cell clusters with differentiable programming, "
                 "Nat Comput Sci 2025);  local: papers/Deshpande_2025_jax_morph.pdf",
        "model_family": "off_lattice_particle_multicellular_differentiable",
        "commit": sha,
        "license": "Apache-2.0",
        "clone": "papers/jax-morph",
        "scale": ["cell", "cluster"],
        "sets": ["cell"],
        "fields": ["chemical (per-cell concentration, screened free-space kernel)"],
        "maps": ["cell_cell_pairwise"],
        "note": "Seeded mechanically by inventory.py. Every entry is at status `candidate`: "
                "named and located, nothing inspected, nothing believed.",
        "mechanisms": mechs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    doc = build()
    mechs = doc["mechanisms"]
    print(f"{doc['repository']} @ {doc['commit'][:8] if doc['commit'] else '?'}   "
          f"{len(mechs)} candidate mechanisms\n")
    w = max(len(m["raw_name"]) for m in mechs)
    for m in mechs:
        print(f"  {m['order']:>3}  {m['raw_name']:<{w}}  {m['raw_kind']:<34} {m['code_path']}")
    if a.write:
        if os.path.exists(RECORD):
            raise SystemExit(f"\n{RECORD} exists -- refusing to clobber a record that may carry "
                             f"inspected entries. Delete it deliberately if you mean to reseed.")
        with open(RECORD, "w") as f:
            yaml.safe_dump(doc, f, sort_keys=False, width=100, allow_unicode=True)
        print(f"\nwrote {RECORD}")


if __name__ == "__main__":
    main()
