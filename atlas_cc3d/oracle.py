"""oracle -- CompuCell3D in its own interpreter, with provenance on every artefact.

The rule the first atlas earned and this one inherits: **`import cc3d` must never succeed in a
Plexus process.** A process that can reach the reference implementation can borrow its answer, and
a differential test that can be contaminated is not a test. CompuCell3D lives in
`/workspace/.conda_envs/cc3d-oracle` (4.10.0, py312); Plexus lives in `neural-graph-linux`.

HOW CC3D IS DRIVEN, and why not the obvious way. CompuCell3D 4.x advertises a pure-Python route
(`PyCoreSpecs` + `service_cc3d`) that would be the natural fit for a scriptable oracle. In 4.10.0
it does not work: passing a list of specs where a simulation *file* is expected reaches
`persistent_globals.get_custom_settings_path()` -> `Path(<list>)` (TypeError), and past that
`CC3DSimService._run` asserts `os.path.isfile(<list>)`. Both are upstream bugs, not configuration.

So the oracle uses `PyCoreSpecs` for what it is good at -- GENERATING correct CC3DML, including
`<RandomSeed>` -- writes a real `.cc3d` project, and runs it through `cc3d/run_script.py`, the
officially supported headless entry point, which never touches that code path. The model is still
defined in Python; only the transport is a file.

    python oracle.py verify           # isolation + headless run + determinism
    python oracle.py smoke            # the cell-sorting reference -> _oracle/runs/smoke/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.join(HERE, "_oracle")
RUNS = os.path.join(ORACLE, "runs")
ENV = os.environ.get("CC3D_ENV", "/workspace/.conda_envs/cc3d-oracle")
PY = os.path.join(ENV, "bin", "python")
RUN_SCRIPT = os.path.join(ENV, "lib", "python3.12", "site-packages", "cc3d", "run_script.py")
PLEXUS_PY = "/workspace/.conda_envs/neural-graph-linux/bin/python"

# The steppable that makes a run READABLE. CC3D writes VTK for its player; an oracle needs the
# per-cell state as data, so a steppable dumps it at `finish()`.
STEPPABLE = '''
import os, json
from cc3d.core.PySteppables import *

class DumpSteppable(SteppableBasePy):
    """Write every live cell's state at the end of the run."""
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def finish(self):
        rows = sorted((int(c.id), int(c.type), float(c.volume), float(c.surface),
                       round(float(c.xCOM), 6), round(float(c.yCOM), 6))
                      for c in self.cell_list)
        with open(os.environ["CC3D_DUMP"], "w") as f:
            json.dump({"n": len(rows),
                       "columns": ["id", "type", "volume", "surface", "xCOM", "yCOM"],
                       "cells": rows}, f, indent=1)
'''

# CC3D execs the PythonScript with its own globals, so the steppable CLASS must live in a separate
# importable module and only the registration goes here. A single-file version raises
# `NameError: SteppableBasePy is not defined` at registration -- found the hard way.
MAIN = '''
from cc3d import CompuCellSetup
from sortSteppables import DumpSteppable
CompuCellSetup.register_steppable(steppable=DumpSteppable(frequency=1))
CompuCellSetup.run()
'''

PROJECT = '''<Simulation version="4.10.0">
   <XMLScript Type="XMLScript">Simulation/model.xml</XMLScript>
   <PythonScript Type="PythonScript">Simulation/main.py</PythonScript>
</Simulation>
'''

# The reference model: cell sorting, CompuCell3D's canonical demonstration. Two adhesive cell types
# whose contact energies drive one to engulf the other -- an outcome that is a TOPOLOGY rather than
# a trajectory, which is the kind of observable a Potts differential test can actually use.
SORT_SPECS = '''
import warnings, sys; warnings.filterwarnings("ignore")
from cc3d.core.PyCoreSpecs import (PottsCore, CellTypePlugin, VolumePlugin, ContactPlugin,
                                   BlobInitializer, CenterOfMassPlugin)
seed, steps, dim = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
potts = PottsCore(dim_x=dim, dim_y=dim, dim_z=1, steps=steps, fluctuation_amplitude=10.0,
                  neighbor_order=2, random_seed=seed)
ct = CellTypePlugin("Condensing", "NonCondensing")
vol = VolumePlugin()
vol.param_new("Condensing", target_volume=25, lambda_volume=2.0)
vol.param_new("NonCondensing", target_volume=25, lambda_volume=2.0)
con = ContactPlugin(neighbor_order=2)
con.param_new("Medium", "Condensing", 16); con.param_new("Medium", "NonCondensing", 16)
con.param_new("Condensing", "Condensing", 2); con.param_new("NonCondensing", "NonCondensing", 11)
con.param_new("Condensing", "NonCondensing", 11)
com = CenterOfMassPlugin()
blob = BlobInitializer()
blob.region_new(width=5, radius=dim // 3, center=(dim // 2, dim // 2, 0),
                cell_types=("Condensing", "NonCondensing"))
body = "\\n".join(s.xml.getCC3DXMLElementString() for s in (potts, ct, vol, con, com, blob))
sys.stdout.write('<CompuCell3D Revision="0" Version="4.10.0">\\n' + body + '\\n</CompuCell3D>\\n')
'''


def _sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=1800, **kw)


def build_project(dest, seed, steps, dim):
    """Write a complete .cc3d project. The CC3DML is GENERATED from PyCoreSpecs, never hand-typed."""
    os.makedirs(os.path.join(dest, "Simulation"), exist_ok=True)
    open(os.path.join(dest, "model.cc3d"), "w").write(PROJECT)
    open(os.path.join(dest, "Simulation", "sortSteppables.py"), "w").write(STEPPABLE)
    open(os.path.join(dest, "Simulation", "main.py"), "w").write(MAIN)
    gen = os.path.join(dest, "_gen_specs.py")
    open(gen, "w").write(SORT_SPECS)
    r = _sh([PY, "-u", gen, str(seed), str(steps), str(dim)])
    if r.returncode != 0 or "<CompuCell3D" not in r.stdout:
        raise RuntimeError(f"CC3DML generation failed:\n{r.stderr[-1200:]}")
    open(os.path.join(dest, "Simulation", "model.xml"), "w").write(r.stdout)
    return dest


def run_project(dest, dump_path):
    if os.path.exists(dump_path):
        os.remove(dump_path)          # never read a stale dump as if it were this run's
    # CC3D refuses an output directory inside the project folder, so it goes in a sibling.
    out_dir = os.path.join(ORACLE, "_out", os.path.basename(dest.rstrip("/")))
    env = dict(os.environ, CC3D_DUMP=dump_path)
    r = _sh([PY, "-u", RUN_SCRIPT, "-i", os.path.join(dest, "model.cc3d"),
             "-o", out_dir, f"--current-dir={dest}"], env=env)
    if not os.path.exists(dump_path):
        raise RuntimeError(f"run produced no dump.\nstdout:\n{r.stdout[-1500:]}\n"
                           f"stderr:\n{r.stderr[-1500:]}")
    return json.load(open(dump_path))


def provenance():
    r = _sh([PY, "-c", "import warnings;warnings.filterwarnings('ignore');"
                       "import cc3d,sys;print(cc3d.__version__);print(sys.version.split()[0])"])
    lines = [x for x in r.stdout.strip().split("\n") if x][-2:]
    ver, pyver = (lines + ["?", "?"])[:2]
    return {"cc3d_version": ver, "python": pyver, "env": ENV, "run_script": RUN_SCRIPT}


def isolation_ok():
    """The Plexus interpreter must NOT be able to import cc3d -- the contamination guarantee,
    checked rather than assumed."""
    return _sh([PLEXUS_PY, "-c", "import cc3d"]).returncode != 0


def cmd_verify(a):
    prov = provenance()
    print(f"[oracle] CompuCell3D {prov['cc3d_version']} · python {prov['python']}")
    iso = isolation_ok()
    print(f"[oracle] isolation (Plexus cannot import cc3d): {'OK' if iso else 'FAILED'}")

    work = os.path.join(ORACLE, "_verify")
    build_project(work, seed=42, steps=100, dim=30)
    a1 = run_project(work, os.path.join(work, "d1.json"))
    a2 = run_project(work, os.path.join(work, "d2.json"))
    build_project(work, seed=7, steps=100, dim=30)
    b1 = run_project(work, os.path.join(work, "d3.json"))

    det, seedy = (a1 == a2), (a1 != b1)
    print(f"[oracle] headless run: OK ({a1['n']} cells)")
    print(f"[oracle] deterministic at a fixed seed: {'OK' if det else 'FAILED'}")
    print(f"[oracle] a different seed differs:      {'OK' if seedy else 'FAILED'}")
    ok = iso and det and seedy
    print(f"[oracle] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_smoke(a):
    dest = os.path.join(RUNS, "smoke")
    build_project(dest, seed=a.seed, steps=a.steps, dim=a.dim)
    data = run_project(dest, os.path.join(dest, "reference.json"))
    prov = provenance()
    prov.update({"seed": a.seed, "steps": a.steps, "dim": a.dim, "model": "cell_sorting",
                 "xml_sha1": hashlib.sha1(
                     open(os.path.join(dest, "Simulation", "model.xml"), "rb").read()).hexdigest()})
    json.dump(prov, open(os.path.join(dest, "_provenance.json"), "w"), indent=1)
    vols = [c[2] for c in data["cells"]]
    types = {}
    for c in data["cells"]:
        types[c[1]] = types.get(c[1], 0) + 1
    summary = {"n_cells": data["n"], "by_type": types,
               "volume_mean": sum(vols) / len(vols),
               "volume_min": min(vols), "volume_max": max(vols)}
    json.dump(summary, open(os.path.join(dest, "summary.json"), "w"), indent=1)
    print(f"[oracle] smoke: {data['n']} cells {types}, volume "
          f"{summary['volume_min']:.0f}-{summary['volume_max']:.0f} "
          f"(mean {summary['volume_mean']:.1f}) -> {os.path.relpath(dest, HERE)}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    s = sub.add_parser("smoke")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--steps", type=int, default=1000)
    s.add_argument("--dim", type=int, default=60)
    a = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    return {"verify": cmd_verify, "smoke": cmd_smoke}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
