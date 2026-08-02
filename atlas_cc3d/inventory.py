"""inventory -- seed the record from CompuCell3D's own machine-readable spec registry.

The jax-morph atlas scanned its clone's syntax tree for classes. CompuCell3D offers something
better and more honest: `cc3d.core.PyCoreSpecs` IS the framework's declaration of what a mechanism
is. Every plugin, solver and initialiser a model can switch on appears there with a
`registered_name` and a `type` (Plugin or Steppable). Enumerating it is not an interpretation of
the source, it is reading the source's own index -- and a human can check it against
`dir(cc3d.core.PyCoreSpecs)` in ten seconds.

AND THE PART A SCAN CANNOT SEE, which the first atlas learned the hard way: 4 of jax-morph's 24
mechanisms were not classes, and they were among the most interesting entries in the record. The
same is true here, more so. A Cellular Potts model's defining commitments -- what a cell IS, how
time advances, how energies compose, how a move is accepted -- are not plugins. They are the
framework. They are added here BY HAND, and the record says so in each one's `surprises`.

    python inventory.py                 # print what would be seeded
    python inventory.py --write         # seed atlas_record.yaml at `candidate`
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD = os.path.join(HERE, "atlas_record.yaml")
ENV = os.environ.get("CC3D_ENV", "/workspace/.conda_envs/cc3d-oracle")
PY = os.path.join(ENV, "bin", "python")

# Configuration objects rather than mechanisms: `Metadata` is bookkeeping and `PottsCore` is the
# lattice/dynamics declaration itself, which enters the record as the architectural entries below
# rather than as a plugin. Excluded by NAME so the exclusion is auditable.
NOT_MECHANISMS = {"Metadata", "PottsCore"}

SCAN_SRC = '''
import warnings, inspect, json; warnings.filterwarnings("ignore")
import cc3d, cc3d.core.PyCoreSpecs as S
from cc3d.core.PyCoreSpecs import _PyCoreSpecsBase
out = []
for name in sorted(dir(S)):
    o = getattr(S, name)
    if not (inspect.isclass(o) and issubclass(o, _PyCoreSpecsBase)) or name.startswith("_"):
        continue
    if getattr(o, "type", None) not in ("Plugin", "Steppable"):
        continue
    reg = getattr(o, "registered_name", None)
    if not isinstance(reg, str):
        reg = name          # a couple expose it as a property object; fall back to the class name
    doc = (o.__doc__ or "").strip().split("\\n")[0][:200]
    try:
        src_file, lineno = inspect.getsourcefile(o), inspect.getsourcelines(o)[1]
    except (OSError, TypeError):
        src_file, lineno = inspect.getsourcefile(S), 1
    out.append({"class": name, "registered": reg, "kind": o.type,
                "module": inspect.getmodule(o).__name__, "doc": doc,
                "file": src_file, "line": lineno})
print(json.dumps({"cc3d_version": cc3d.__version__, "specs": out}))
'''

# --------------------------------------------------------------------------------------------- #
#  The mechanisms that are NOT classes -- the framework itself. For a Cellular Potts model this is
#  exactly where the interesting vocabulary question lives: jax-morph's cells are points carrying
#  state, CompuCell3D's are regions of a lattice, and none of that difference is a plugin.
# --------------------------------------------------------------------------------------------- #
ARCHITECTURAL = [
    dict(id="cell_as_lattice_domain",
         raw_name="cell as a set of lattice sites",
         raw_kind="representation (not a class)",
         summary="A cell is not a point with a radius but a connected set of lattice sites sharing "
                 "one id. Volume is a site count, surface a boundary count, position a derived "
                 "centre of mass. Every other mechanism is defined in terms of this.",
         code_path="SITE/cc3d/core/PySteppables.py",
         why_hand="not a plugin -- it is the framework's data model, and the single largest "
                  "difference from the first atlas's target"),
    dict(id="metropolis_acceptance",
         raw_name="Metropolis acceptance of pixel copies",
         raw_kind="core dynamics (not a class)",
         summary="Time advances by attempting to copy one lattice site's id into a neighbour, "
                 "accepted with probability 1 if the total energy falls and exp(-dE/T) otherwise, "
                 "where T is the fluctuation amplitude. No plugin implements this; every plugin "
                 "only contributes a term to dE.",
         code_path="SITE/cc3d/core/PyCoreSpecs.py",
         why_hand="the dynamics themselves -- discrete, stochastic, accept/reject, with no "
                  "pathwise derivative"),
    dict(id="energy_sum_composition",
         raw_name="the Hamiltonian as a sum of plugin terms",
         raw_kind="composition contract (not a class)",
         summary="Mechanisms compose by ADDING energy terms to one Hamiltonian and interact only "
                 "through the accept/reject decision. Nothing is applied in sequence and nothing "
                 "returns a delta.",
         code_path="SITE/cc3d/core/PyCoreSpecs.py",
         why_hand="the direct counterpart of jax-morph's Lie-Trotter operator split, and the "
                  "contrast is itself a finding: operator splitting versus energy summation are "
                  "two different answers to how mechanisms compose"),
    dict(id="mcs_time_unit",
         raw_name="Monte Carlo Step as the time unit",
         raw_kind="time-scale contract (not a class)",
         summary="One MCS is one attempted copy per lattice site, not a duration. There is no dt "
                 "and no integrator; a rate must be expressed as a per-MCS probability.",
         code_path="SITE/cc3d/core/PyCoreSpecs.py",
         why_hand="a time-scale taxonomy with no dt at all -- the sharpest possible contrast with "
                  "an ODE-integrating framework"),
    dict(id="pixel_neighbourhood",
         raw_name="neighbour order / pixel-copy neighbourhood",
         raw_kind="relation contract (not a class)",
         summary="Which lattice sites count as adjacent, for both the copy attempt and every "
                 "contact energy. It is the relation E of the model, chosen by an integer.",
         code_path="SITE/cc3d/core/PyCoreSpecs.py",
         why_hand="the model's relation is a lattice adjacency, not an edge set built from "
                  "positions"),
]

BLANK_EVIDENCE = {"oracle_run": None, "diff_metric": None, "threshold": None, "passed": None}


def scan():
    r = subprocess.run([PY, "-c", SCAN_SRC], capture_output=True, text=True, timeout=600)
    line = [x for x in r.stdout.strip().split("\n") if x.startswith("{")]
    if not line:
        raise RuntimeError(f"spec scan failed:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    return json.loads(line[-1])


def build(data):
    mechs, order = [], 0
    for s in data["specs"]:
        if s["class"] in NOT_MECHANISMS:
            continue
        order += 1
        mechs.append({
            "id": s["registered"].lower(), "order": order, "raw_name": s["registered"],
            "raw_kind": f"{s['kind']} ({s['class']})",
            # a real file:line, so R3_code_path can CHECK it rather than take a dotted name on trust
            "code_path": f"{s['file']}:L{s['line']}",
            "paper_section": "Swat et al. (2012) Methods Cell Biol 110:325-366, "
                             "'Multi-Scale Modeling of Tissues Using CompuCell3D'",
            "summary": s["doc"] or None,
            "equations": None, "params": {}, "state_io": None, "surprises": None,
            "verdict": None, "of": None, "implementation_of": None, "why": None,
            "contract": None, "status": "candidate", "module": None, "test": None,
            "evidence": dict(BLANK_EVIDENCE),
        })
    for a in ARCHITECTURAL:
        order += 1
        mechs.append({
            "id": a["id"], "order": order, "raw_name": a["raw_name"], "raw_kind": a["raw_kind"],
            "code_path": a["code_path"].replace(
                "SITE", os.path.join(ENV, "lib", "python3.12", "site-packages")),
            "paper_section": "Swat et al. (2012) Methods Cell Biol 110:325-366",
            "summary": a["summary"], "equations": None, "params": {}, "state_io": None,
            "surprises": [f"ADDED BY HAND: {a['why_hand']}"],
            "verdict": None, "of": None, "implementation_of": None, "why": None,
            "contract": None, "status": "candidate", "module": None, "test": None,
            "evidence": dict(BLANK_EVIDENCE),
        })
    return {
        "repository": "CompuCell3D/CompuCell3D",
        "paper": "10.1016/B978-0-12-388403-9.00013-8",
        "model_family": "cellular_potts",
        "commit": f"conda compucell3d {data['cc3d_version']} py312",
        "license": "MIT (CompuCell3D core)",
        "clone": ENV,
        "scale": "cell (lattice domain)",
        "sets": ["cell"], "fields": ["chemical"], "maps": [],
        "note": "Seeded by inventory.py from cc3d.core.PyCoreSpecs -- the framework's own registry "
                "-- plus 5 architectural mechanisms added BY HAND that no scan can see. Every "
                "entry is at `candidate`: nothing has been read at source yet.",
        "mechanisms": mechs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    doc = build(scan())
    hand = sum(1 for m in doc["mechanisms"] if m["surprises"])
    print(f"[inventory] CompuCell3D {doc['commit']}")
    print(f"[inventory] {len(doc['mechanisms'])} mechanisms: "
          f"{len(doc['mechanisms']) - hand} scanned from PyCoreSpecs, {hand} added by hand")
    for m in doc["mechanisms"]:
        print(f"  {m['order']:>2} [{'hand' if m['surprises'] else 'scan'}] "
              f"{m['raw_name']:<34} {m['raw_kind']}")
    if a.write:
        import yaml
        with open(RECORD, "w") as f:
            yaml.safe_dump(doc, f, sort_keys=False, width=100)
        print(f"\n-> {os.path.relpath(RECORD, HERE)}")
    else:
        print("\n(dry run; pass --write to seed the record)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
