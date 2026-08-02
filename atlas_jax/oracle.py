"""oracle -- the reference implementation, isolated, pinned, and RUNNABLE.

This is the one thing the Okuda track never had. Okuda et al. published a paper with no code,
so every disagreement between our simulation and theirs was unfalsifiable: we could not tell a
wrong operator from a wrong parameter from a wrong reading of a figure. Here the authors'
own code exists, so we can put our reconstruction and their implementation on the SAME initial
condition and diff the trajectories. Faithfulness stops being a judgement call.

Three rules, all learned from the discovery loop.

  1. THE ORACLE LIVES IN ITS OWN INTERPRETER.  jax-morph wants JAX; Plexus wants torch. They
     share a filesystem and nothing else. Every oracle call is a subprocess into `_oracle/venv`,
     never an import. A Plexus process that can `import jax_morph` is a Plexus process that can
     silently borrow the reference implementation's answer -- which is exactly the contamination
     a differential test exists to detect.

  2. NOTHING IS AN ORACLE UNTIL ITS PROVENANCE IS WRITTEN DOWN.  Every artefact carries the
     clone's git SHA, the resolved package versions, the interpreter, and the script that made
     it. A reference trajectory with no provenance is a number of unknown origin, and the
     campaign has already been burned once by trusting one of those.

  3. FAIL LOUDLY.  No try/except around an artefact. If the reference will not run, the Atlas
     stops -- it does not proceed on a remembered figure.

Usage
-----
    python oracle.py setup            # build the venv, install the pinned reference, record it
    python oracle.py verify           # provenance + import + a 3-line physics check
    python oracle.py smoke            # a real short simulation -> _oracle/runs/smoke/
    python oracle.py run  <script>    # any script, inside the venv, artefacts under _oracle/runs/
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
CLONE = os.path.join(PLEXUS, "papers", "jax-morph")

ORACLE = os.path.join(HERE, "_oracle")
VENV = os.path.join(ORACLE, "venv")
PY = os.path.join(VENV, "bin", "python")
RUNS = os.path.join(ORACLE, "runs")
PROVENANCE = os.path.join(ORACLE, "provenance.json")

# The reference's own declared dependencies (pyproject) plus what a reference RUN needs.
# Deliberately not pinned to exact versions on first build: we record what pip resolved, which
# is the honest thing, and pin from that record afterwards.
REQUIREMENTS = ["jax>=0.4.35", "equinox>=0.11.7", "diffrax>=0.6.0", "numpy", "matplotlib"]


# ------------------------------------------------------------------------------------------- #
#  setup
# ------------------------------------------------------------------------------------------- #
def _run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def _git_sha(path):
    out = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def setup(force=False):
    """Build `_oracle/venv` and install the reference from the LOCAL clone.

    Installed from `papers/jax-morph`, not from PyPI, so the atlas record's `code_path`
    evidence points at bytes we can read. Non-editable: the clone stays read-only.
    """
    if os.path.isdir(VENV) and not force:
        print(f"venv exists: {VENV}   (use --force to rebuild)")
    else:
        if force and os.path.isdir(VENV):
            import shutil
            shutil.rmtree(VENV)
        os.makedirs(ORACLE, exist_ok=True)
        _run([sys.executable if sys.version_info >= (3, 11) else "python3", "-m", "venv", VENV])

    _run([PY, "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    _run([PY, "-m", "pip", "install", "--quiet", *REQUIREMENTS])
    _run([PY, "-m", "pip", "install", "--quiet", "--no-deps", CLONE])

    freeze = subprocess.run([PY, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    ver = subprocess.run(
        [PY, "-c", "import jax, jax_morph as j; print(jax.__version__); print(j.__version__); "
                   "print([d.platform for d in jax.devices()])"],
        capture_output=True, text=True)
    if ver.returncode != 0:
        raise SystemExit(f"reference will not import:\n{ver.stdout}\n{ver.stderr}")
    jax_v, jm_v, devices = ver.stdout.strip().splitlines()

    prov = {
        "clone": CLONE,
        "clone_git_sha": _git_sha(CLONE),
        "interpreter": subprocess.run([PY, "-V"], capture_output=True, text=True).stdout.strip(),
        "jax": jax_v,
        "jax_morph": jm_v,
        "devices": devices,
        "requirements_asked": REQUIREMENTS,
        "pip_freeze": freeze.splitlines(),
    }
    with open(PROVENANCE, "w") as f:
        json.dump(prov, f, indent=2)
    print(f"\noracle ready:  jax {jax_v} · jax-morph {jm_v} · devices {devices}")
    print(f"provenance  -> {PROVENANCE}")
    return prov


def provenance():
    if not os.path.exists(PROVENANCE):
        raise SystemExit("no provenance -- run `python oracle.py setup` first")
    with open(PROVENANCE) as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------- #
#  run
# ------------------------------------------------------------------------------------------- #
def run_script(src, name, env=None):
    """Execute `src` (python source text) inside the oracle venv.

    The script gets `OUT` in its environment: a per-run directory under `_oracle/runs/<name>/`
    where every artefact must be written. A copy of the script and the provenance are dropped
    beside the artefacts, so a trajectory can always be traced to the code and the versions
    that produced it.
    """
    out = os.path.join(RUNS, name)
    os.makedirs(out, exist_ok=True)
    script = os.path.join(out, "_script.py")
    with open(script, "w") as f:
        f.write(src)
    with open(os.path.join(out, "_provenance.json"), "w") as f:
        json.dump(provenance(), f, indent=2)

    e = dict(os.environ, OUT=out, JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"))
    e.update(env or {})
    print(f"+ {PY} {script}   (OUT={out})", flush=True)
    p = subprocess.run([PY, script], env=e)
    if p.returncode != 0:
        raise SystemExit(f"oracle run '{name}' FAILED (exit {p.returncode}) -- see above. "
                         f"Nothing is recorded as reference.")
    print(f"artefacts -> {out}")
    return out


# ------------------------------------------------------------------------------------------- #
#  verify  --  the reference is imported, its physics answers a question with a known sign
# ------------------------------------------------------------------------------------------- #
VERIFY_SRC = textwrap.dedent("""
    import json, os
    import jax, jax.numpy as jnp
    import jax_morph as jxm

    out = {}
    out["jax"] = jax.__version__
    out["jax_morph"] = jxm.__version__
    out["public_api"] = sorted(n for n in dir(jxm) if not n.startswith("_"))
    print(json.dumps(out, indent=2))
    with open(os.path.join(os.environ["OUT"], "verify.json"), "w") as f:
        json.dump(out, f, indent=2)
""")


def verify():
    prov = provenance()
    print(json.dumps({k: prov[k] for k in
                      ("clone_git_sha", "interpreter", "jax", "jax_morph", "devices")}, indent=2))
    return run_script(VERIFY_SRC, "verify")


# ------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["setup", "verify", "smoke", "run"])
    ap.add_argument("script", nargs="?", help="path to a python script (cmd=run)")
    ap.add_argument("--name", default=None, help="run name (cmd=run)")
    ap.add_argument("--force", action="store_true", help="rebuild the venv (cmd=setup)")
    a = ap.parse_args()

    if a.cmd == "setup":
        setup(force=a.force)
    elif a.cmd == "verify":
        verify()
    elif a.cmd == "smoke":
        from smoke import SMOKE_SRC          # kept beside this file, edited often
        run_script(SMOKE_SRC, "smoke")
    elif a.cmd == "run":
        if not a.script:
            raise SystemExit("cmd=run needs a script path")
        with open(a.script) as f:
            src = f.read()
        run_script(src, a.name or os.path.splitext(os.path.basename(a.script))[0])


if __name__ == "__main__":
    main()
