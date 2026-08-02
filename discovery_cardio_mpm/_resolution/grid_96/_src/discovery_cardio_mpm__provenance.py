#!/usr/bin/env python
"""provenance -- a number must be recoverable from its own folder.

WHAT WENT WRONG BEFORE
================================================================================================
Two batches of the previous campaign were destroyed by ordinary branch traffic. A checkout moved
HEAD, uncommitted working-tree edits went with it, and the code that produced a headline result
ceased to exist -- the result survives only as a line in an archived text file that nobody can
re-derive. Separately, five batches of distillation were reverted the same way.

The obvious cure -- "commit before you run" -- does not work here, because /workspace/Plexus is a
shared tree with several agentic loops running in it at once. Right now `git status` shows over a
hundred modified paths belonging to other campaigns. A recorded `git diff` of that tree is noise,
and a rule requiring a clean tree would simply never be satisfiable.

SO THE ARTEFACT CARRIES ITS OWN SOURCE
------------------------------------------------------------------------------------------------
Every run copies the ACTUAL BYTES of every project module it imported into `<run>/_src/`, with a
sha256 of each. That is clobber-proof, campaign-scoped, and needs nothing of the rest of the tree:
a branch switch, a concurrent loop, or a deleted file cannot reach backwards into a finished run.
The commit sha is recorded too, but as a label, not as the mechanism.

WHAT A MANIFEST HOLDS
------------------------------------------------------------------------------------------------
  when, where, which GPU        so a hardware-dependent number is attributable
  the exact argv                so "what was this run?" is never reconstructed from a folder name
  the commit and dirty flag     as a label
  every project source file     bytes + sha256, under _src/
  every input array             path, sha256, and the CONTENT id of the specimen
  the seed and the arithmetic   what determinism.enforce actually set

The rule this enforces: **a run that cannot say what produced it is not evidence.**
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))


def _sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _git(*args):
    try:
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:                                  # pragma: no cover - git absent
        return ""


def imported_project_sources(roots=(HERE,)):
    """Every loaded module whose file sits under one of `roots`. This is the set that actually
    ran -- not a hand-maintained list, which would drift the first time a file was added."""
    out = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        f = os.path.abspath(f)
        # `os.path.exists` is not paranoia: some libraries register modules whose `__file__`
        # points at a path that was never written (torch does this for `_classes`). Archiving is
        # best-effort over what is really on disk; anything phantom is not our source anyway.
        if (any(f.startswith(os.path.abspath(r) + os.sep) for r in roots)
                and f.endswith(".py") and os.path.exists(f)):
            out.append(f)
    return sorted(set(out))


def write_manifest(outdir, argv=None, inputs=(), extra=None, roots=(HERE,)):
    """Write `<outdir>/run_manifest.json` and copy the run's own source into `<outdir>/_src/`.

    `inputs` is a sequence of (label, path) or (label, path, content_id).
    Returns the manifest dict.
    """
    os.makedirs(outdir, exist_ok=True)
    src_dir = os.path.join(outdir, "_src")
    os.makedirs(src_dir, exist_ok=True)

    sources = {}
    for f in imported_project_sources(roots):
        rel = os.path.relpath(f, REPO).replace(os.sep, "__")
        shutil.copy2(f, os.path.join(src_dir, rel))
        sources[os.path.relpath(f, REPO)] = _sha256_file(f)

    ins = []
    for item in inputs:
        label, path = item[0], item[1]
        cid = item[2] if len(item) > 2 else None
        rec = {"label": label, "path": os.path.abspath(path) if path else None}
        if path and os.path.exists(path):
            rec["sha256"] = _sha256_file(path)
            rec["bytes"] = os.path.getsize(path)
        else:
            rec["sha256"] = None
            rec["missing"] = True
        if cid:
            rec["content_id"] = cid
        ins.append(rec)

    dirty = _git("status", "--porcelain", "--", "discovery_cardio_mpm")
    man = {
        "argv": list(argv if argv is not None else sys.argv),
        "cwd": os.getcwd(),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "campaign_dirty": bool(dirty),
        "campaign_dirty_paths": [l[3:] for l in dirty.splitlines()] if dirty else [],
        "sources": sources,
        "inputs": ins,
    }
    if extra:
        man.update(extra)
    try:
        import torch
        man["torch"] = torch.__version__
        man["gpu"] = ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                      if torch.cuda.is_available() else [])
    except Exception:                                  # pragma: no cover
        pass

    with open(os.path.join(outdir, "run_manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    return man


def check_manifest(outdir):
    """Is this run self-describing? Returns (ok, reasons). Used by the gate and by any later
    reader that wants to know whether a number may be cited."""
    p = os.path.join(outdir, "run_manifest.json")
    if not os.path.exists(p):
        return False, ["no run_manifest.json"]
    man = json.load(open(p))
    bad = []
    if not man.get("sources"):
        bad.append("no source files recorded")
    for rel, sha in (man.get("sources") or {}).items():
        arch = os.path.join(outdir, "_src", rel.replace(os.sep, "__"))
        if not os.path.exists(arch):
            bad.append(f"source not archived: {rel}")
        elif _sha256_file(arch) != sha:
            bad.append(f"archived source does not match its recorded hash: {rel}")
    if not man.get("inputs"):
        bad.append("no inputs recorded")
    for rec in man.get("inputs") or []:
        if rec.get("missing") or not rec.get("sha256"):
            bad.append(f"input not hashed: {rec.get('label')}")
    if "seed" not in json.dumps(man):
        bad.append("no seed recorded")
    return (not bad), bad


if __name__ == "__main__":
    import tempfile
    d = tempfile.mkdtemp(prefix="prov_")
    import data                                        # noqa: F401  - so it appears in sources
    m = write_manifest(d, inputs=[("recording", data.DEFAULT_NPZ, "demo")], extra={"seed": 0})
    ok, why = check_manifest(d)
    print(f"  [provenance] {len(m['sources'])} sources archived, dirty={m['campaign_dirty']}")
    print(f"  [provenance] check {'PASS' if ok else 'FAIL ' + str(why)}  ->  {d}")
    sys.exit(0 if ok else 1)
