#!/usr/bin/env python
"""op_probe -- does changing this parameter change the RUN?

THE QUESTION NOTHING IN THIS PROJECT WAS ASKING. Three layers of defence each missed that
`shape_to_chem.beta` is inert, and they missed it the same way -- none of them measured the
operator's effect on the trajectory:

  * the operator's own self-test certifies `_feature()`: curvature reads 1/R on a sphere of known
    radius, a bump is positive, a dimple negative, the standardisation is unit-invariant. Every
    check passes and every check is about arithmetic INSIDE the operator. Its one check about
    `forward` is `chk(True, "beta = 0 returns zeros by construction")` -- an assertion hardcoded
    to pass;
  * `instrument.install()` asks "did it act?" as `bool(out) or fingerprint moved`, and
    `shape_to_chem` returns a full-size tensor every call, so `bool(out)` is True even when that
    tensor is all zeros. It scores 100% acted on every run it has ever been in;
  * the campaign's own scoring records `refuted` for a prediction whose knob was never connected,
    which is how 8 runs across 13 rounds became evidence about a mechanism that never ran.

So the primitive here is deliberately dumber than all three: run the composition twice, change one
number, and compare what came out. It cannot be fooled by an emission that is discarded
downstream, because it never looks at the emission.

    LIVE     the trajectories differ            -> the parameter is connected
    DEAD     bit-identical trajectories         -> the parameter cannot matter here
    UNREAD   the class never reads the key      -> dead by construction, no simulation needed

UNREAD is free and runs first: `params` is handed over as a dict that records every key read
during __init__ and forward. `seed_cell_rd.amp` has sat in every spec this campaign has ever
written and is not read by any branch of the operator.

RUN:  python op_probe.py <spec.yaml> --op shape_to_chem --param beta --values -2 -4
      python op_probe.py <spec.yaml> --all            # every param of every scheduled operator
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (os.path.join(ROOT, "src"), HERE, os.path.join(ROOT, "discovery_okuda")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------- the UNREAD probe
class RecordingParams(dict):
    """A params dict that remembers which keys were actually read.

    Cheaper and stricter than any simulation: a parameter the class never looks up cannot
    influence anything, whatever the sweep does to it.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.read = set()

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self.read.add(key)
        return super().__contains__(key)


def unread_params(op_name, params, implementation=None):
    """Which of `params` does the operator never read? Instantiation only -- no mesh needed."""
    from plexus.models import registry as R
    cls = R.get_operator(op_name, implementation)
    rec = RecordingParams(params)
    try:
        cls(rec, device="cpu")
    except Exception as e:                       # a param may only be reachable inside forward()
        return None, f"could not instantiate: {type(e).__name__}: {e}"
    # `implementation` is consumed by the REGISTRY to choose the class, never by the class, so it
    # reads UNREAD on every operator that has more than one implementation. Excluding it is not a
    # convenience: leaving it in puts a false positive beside every true one, which is how a
    # detector stops being read.
    declared = {k for k in params
                if not k.startswith("_") and k not in ("op", "id", "at", "implementation")}
    return sorted(declared - rec.read), None


# --------------------------------------------------------------------------- the LIVE/DEAD probe
def _fingerprint(H):
    """A trajectory fingerprint: every level's state, plus the mesh's own targets.

    Deliberately NOT the metric bank. A metric can be blind to a change (n_spots reads 1 on a
    six-domain pattern) and then a live parameter reads DEAD. State cannot.
    """
    import torch
    out = {}
    for name in sorted(H.levels):
        lvl = H.level(name)
        try:
            s = lvl.state.detach().cpu().numpy()
            out[f"{name}.state"] = np.asarray(s, dtype=np.float64).copy()
        except Exception:
            pass
        m = getattr(lvl, "_mesh", None)
        if isinstance(m, dict):
            for k in sorted(m):
                v = m[k]
                if k in ("hist",) or not hasattr(v, "shape"):
                    continue
                try:
                    out[f"{name}.mesh.{k}"] = np.asarray(
                        v.detach().cpu().numpy() if hasattr(v, "detach") else v,
                        dtype=np.float64).copy()
                except Exception:
                    pass
    return out


def _distance(a, b):
    """Relative L2 between two fingerprints; a shape change is an infinite distance."""
    keys = set(a) | set(b)
    num = den = 0.0
    shape_changed = []
    for k in sorted(keys):
        x, y = a.get(k), b.get(k)
        if x is None or y is None or x.shape != y.shape:
            shape_changed.append(k)
            continue
        d = np.nan_to_num(x - y)
        num += float((d ** 2).sum())
        den += float((np.nan_to_num(y) ** 2).sum())
    rel = (num ** 0.5) / max(den ** 0.5, 1e-30)
    return rel, shape_changed


def _set_param(spec, op_name, key, value, occurrence=0):
    """Set `key` on the `occurrence`-th instance of `op_name`. Returns the value replaced."""
    seen = 0
    for o in spec.get("operators", []):
        if o.get("op") != op_name:
            continue
        if seen == occurrence:
            old = o.get(key)
            o[key] = value
            return old
        seen += 1
    raise KeyError(f"{op_name} (occurrence {occurrence}) not in the spec")


def run_spec(spec_dict, frames, device="cpu", tag=""):
    """Run a spec dict for `frames` frames and return its trajectory fingerprint."""
    import tempfile
    import yaml
    import plexus.operators                                             # noqa: F401
    import tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d, tyssue_monolayer  # noqa: F401
    import tyssue_shape_to_chem                                         # noqa: F401
    import ckpt                                                         # noqa: F401  load_mesh_3d
    import plexus.schema as S
    from plexus.engine import run as engine_run

    d = copy.deepcopy(spec_dict)
    d.setdefault("general", {})["n_frames"] = int(frames)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(d, fh)
        path = fh.name
    try:
        H, _ = engine_run(S.load(path), device=device)
        return _fingerprint(H)
    finally:
        os.unlink(path)


# --------------------------------------------------------------------------- the WARM START
# Cedric, 6 August: "we do not need to generate hundreds of frames to test a knob, just start
# middle of coral_gate_div and switch on or off one knob across 50 frames? there autograd could be
# tested too?"
#
# Both halves of that are right, and the second is the stronger reason. A cold probe spends 100
# frames before `morphogen_growth_3d` even switches on, and it starts from a SPHERE -- where the
# curvature feature is uniform by construction, so `_standardise` returns exactly zero and any
# shape-sensing operator is untestable on the geometry it is handed. Starting mid-run gives the
# probe a specimen that already has curvature, division history and a live pattern.
#
# And the tape. A 900-frame rollout with `grad=True` will not fit in memory; 50 frames will. The
# warm start is not merely cheaper, it is what makes the GRADIENT probe possible at all.
def make_checkpoint(spec_dict, frame, path, device="cpu"):
    """Run `spec_dict` to `frame` and save the state. One run, reusable by every probe after it."""
    import tempfile
    import yaml
    import plexus.operators                                             # noqa: F401
    import tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d, tyssue_monolayer  # noqa: F401
    import tyssue_shape_to_chem                                         # noqa: F401
    import ckpt                                                         # noqa: F401  load_mesh_3d
    import ckpt as _ckpt
    import plexus.schema as S
    from plexus.engine import run as engine_run

    d = copy.deepcopy(spec_dict)
    d.setdefault("general", {})["n_frames"] = int(frame)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(d, fh)
        p = fh.name
    try:
        H, _ = engine_run(S.load(p), device=device)
        _ckpt.save_state(H, path)
    finally:
        os.unlink(p)
    return path


STRUCTURAL = ("divide_3d", "reconnect_t1_3d", "cell_extrude_3d")


def warm_spec(spec_dict, ckpt_path, frames=50, freeze_topology=False):
    """Turn a spec into one that RESTARTS from `ckpt_path` instead of seeding a fresh sphere.

    THE TRAP, and it is silent: the checkpoint already holds the chemistry, and a restart begins
    at frame 0 again. An initial-condition operator gated `before_frame: 3` would therefore fire
    on frames 0-2 of the warm run and OVERWRITE the pattern we just restored -- the probe would
    measure a re-seeded sphere and call every chemistry parameter live for the wrong reason.
    So a GATED `seed_cell_rd` is dropped (its job is done, it is in the checkpoint), while an
    UNGATED one is kept, because there it is not an initial condition at all: `mode: tip`
    re-seeds every frame BY DESIGN, and removing it would change the mechanism under test.
    """
    # MEASURED, 6 August, and it invalidated the first version of this probe's magnitude column:
    #
    #     d_a x1.001  rel 2.62e-01      d_a x1.05  rel 2.69e-01      d_a x2.0  rel 2.80e-01
    #
    # A THOUSAND-FOLD difference in the perturbation moved the distance by 7%. `divide_3d` runs
    # every 4 frames, so any perturbation changes WHICH cells divide, the cell count diverges, and
    # the trajectory distance saturates at a generic "different run" level within a few frames.
    # Fifteen mesh arrays changed SHAPE in every one of those probes, and `_distance` skips
    # mismatched shapes -- so the number was computed on whatever happened to still line up.
    #
    # Sensitivity needs a fixed-dimensional state space. It also needs a tape, and `divide_3d` /
    # `reconnect_t1_3d` are both DIFFERENTIABLE = False, so with them in the chain NO parameter
    # anywhere downstream can carry a gradient. Freezing topology fixes both at once.
    #
    # So the battery has two modes and they answer different questions:
    #   freeze_topology=False   IS IT CONNECTED?  exactly-zero vs not. Chaos cannot fake a zero,
    #                           so LIVE/DEAD stays valid on the full composition.
    #   freeze_topology=True    HOW MUCH, AND IS IT LEARNABLE?  ranking and d/dtheta, on a mesh
    #                           whose cell count cannot change under the perturbation.
    d = copy.deepcopy(spec_dict)
    d.setdefault("general", {})["n_frames"] = int(frames)
    ops = []
    for o in d.get("operators", []):
        if freeze_topology and o.get("op") in STRUCTURAL:
            continue
        if o.get("op") == "seed_mesh_3d":
            ops.append({"op": "load_mesh_3d", "at": o.get("at", "vertex"),
                        "cell_set": o.get("cell_set", "cell"), "ckpt": ckpt_path,
                        "before_frame": 1})
            continue
        if o.get("op") == "seed_cell_rd" and any(k in o for k in ("before_frame", "every")):
            continue                       # an initial condition, already restored
        o = dict(o)
        for k in ("after_frame",):          # the warm run starts past the spin-up gate
            o.pop(k, None)
        ops.append(o)
    d["operators"] = ops
    # THE SPEC NAMES ITS OPERATORS TWICE. `operators` declares them and `schedule` orders them, and
    # rewriting only the first raises "schedule step 'seed_mesh_3d' is not a declared operator" --
    # which is the schema catching me, and is exactly the kind of two-place edit the metric
    # registry was built to remove elsewhere. Derive the schedule from what survived instead of
    # editing it in parallel, so the two cannot drift.
    if "schedule" in d:
        kept = [o["op"] for o in ops]
        d["schedule"] = [s for s in
                         ["load_mesh_3d" if s == "seed_mesh_3d" else s for s in d["schedule"]]
                         if s in kept]
    return d


# --------------------------------------------------------------------------- the GRADIENT probe
def gradient(spec_dict, op_name, key, frames, device="cpu"):
    """d(final state)/d(param) through the rollout, for operators that carry a tape.

    Returns (grad, note). A gradient of exactly zero where the finite difference is NOT zero is
    the informative case: the parameter reaches the state through a path that carries no
    derivative -- a clamp at its bound, an argmax, a `where` on a hard threshold -- so the
    operator is LIVE but not LEARNABLE, and the probe names the barrier instead of guessing.
    """
    import tempfile
    import torch
    import yaml
    import plexus.operators                                             # noqa: F401
    import tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d, tyssue_monolayer  # noqa: F401
    import tyssue_shape_to_chem                                         # noqa: F401
    import ckpt                                                         # noqa: F401  load_mesh_3d
    import plexus.schema as S
    from plexus.engine import run as engine_run
    from plexus.models import registry as R

    cls = R.get_operator(op_name, _impl_of(spec_dict, op_name))
    if not getattr(cls, "DIFFERENTIABLE", True):
        note = f"{op_name} declares DIFFERENTIABLE = False"
    else:
        note = ""
    d = copy.deepcopy(spec_dict)
    d.setdefault("general", {})["n_frames"] = int(frames)
    theta = None
    for o in d.get("operators", []):
        if o.get("op") == op_name:
            theta = torch.tensor(float(o[key]), requires_grad=True)
            o[key] = theta
            break
    if theta is None:
        return None, f"{op_name}.{key} not in the spec"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump({k: v for k, v in d.items() if k != "operators"}, fh)
        p = fh.name
    try:
        sim = S.load(p)
        sim.operators = d["operators"]                  # keep the tensor param, not its yaml copy
        H, _ = engine_run(sim, device=device, grad=True)
        loss = sum(float(0) + lvl.state.pow(2).sum() for lvl in
                   (H.level(n) for n in sorted(H.levels)))
        g, = torch.autograd.grad(loss, theta, allow_unused=True, retain_graph=False)
        return (None if g is None else float(g)), note or "ok"
    finally:
        os.unlink(p)


def _impl_of(spec_dict, op_name):
    for o in spec_dict.get("operators", []):
        if o.get("op") == op_name:
            return o.get("model") or o.get("implementation")
    return None


def probe(spec_dict, op_name, key, values, frames, device="cpu", baseline=None):
    """Is `op_name.key` connected? Returns one row per tested value."""
    if baseline is None:
        baseline = run_spec(spec_dict, frames, device, tag="baseline")
    rows = []
    for v in values:
        d = copy.deepcopy(spec_dict)
        old = _set_param(d, op_name, key, v)
        fp = run_spec(d, frames, device, tag=f"{op_name}.{key}={v}")
        rel, changed = _distance(fp, baseline)
        rows.append({"op": op_name, "param": key, "from": old, "to": v,
                     "rel_change": rel, "shape_changed": changed,
                     "verdict": "DEAD" if rel == 0.0 and not changed else "LIVE"})
    return rows, baseline


# --------------------------------------------------------------------------- the module self-test
def selftest(fixture_spec, ckpt, ops, frames=50, device="cpu"):
    """The three checks every operator module should run from its own `__main__`.

    Cedric, 6 August: *"a main should be able to test forward and differentiation stand alone in
    each script?"* -- yes, and the ORDER is the point. The existing `shape_to_chem` self-test does
    a thorough job of check 0 and skips 1 and 2 entirely, which is how an operator certified
    against spheres of known radius spent 13 rounds contributing nothing:

        0  ARITHMETIC     does the internal feature compute the right number?
        1  EFFECT         does the operator change the STATE?  (what `instrument` cannot see,
                          because it scores `bool(out)` and a tensor of zeros is still an object)
        2  CONNECTION     does each PARAMETER change the state?  LIVE / DEAD / UNREAD
        3  DERIVATIVE     does a gradient reach the parameter, and does it match the difference?

    Check 3's informative outcome is not "a gradient exists". It is a NONZERO finite difference
    beside a ZERO gradient: the parameter reaches the state through a path that carries no
    derivative, so the operator is live but not learnable, and the barrier is named rather than
    guessed at.

    RUN IT ON A FIXTURE WITH REAL GEOMETRY. A cold run starts from a sphere, where the curvature
    feature is uniform and `_standardise` returns exactly zero BY DESIGN -- so a shape-sensing
    operator is untestable on the shape a cold self-test hands it.
    """
    warm = warm_spec(fixture_spec, ckpt, frames=frames)
    base = run_spec(warm, frames, device)
    rows = []
    for op_name, params in ops.items():
        declared = {k: v for k, v in _params_of(warm, op_name).items()}
        missing, err = unread_params(op_name, declared, _impl_of(warm, op_name))
        for key, values in params.items():
            if err is None and key in (missing or []):
                rows.append({"op": op_name, "param": key, "verdict": "UNREAD",
                             "rel": 0.0, "grad": None})
                continue
            r, _ = probe(warm, op_name, key, values, frames, device, baseline=base)
            rel = max(x["rel_change"] for x in r)
            g, note = (None, "not attempted")
            if rel > 0:
                try:
                    g, note = gradient(warm, op_name, key, frames, device)
                except Exception as e:
                    g, note = None, f"{type(e).__name__}: {e}"
            verdict = ("DEAD" if rel == 0 else
                       "LIVE, no derivative" if not g else "LIVE, differentiable")
            rows.append({"op": op_name, "param": key, "verdict": verdict,
                         "rel": rel, "grad": g, "note": note})
    return rows


def _params_of(spec_dict, op_name):
    for o in spec_dict.get("operators", []):
        if o.get("op") == op_name:
            return {k: v for k, v in o.items()
                    if k not in ("op", "id", "at", "every", "when",
                                 "before_frame", "after_frame")}
    return {}


def report(rows):
    print(f"\n{'operator':24}{'parameter':20}{'rel change':>12}{'d/dtheta':>14}   verdict")
    for r in sorted(rows, key=lambda x: (x["verdict"], x["op"], x["param"])):
        g = "--" if r["grad"] is None else f"{r['grad']:.3e}"
        print(f"  {r['op']:22}{r['param']:20}{r['rel']:12.3e}{g:>14}   {r['verdict']}")
    dead = [r for r in rows if r["verdict"] in ("DEAD", "UNREAD")]
    if dead:
        print(f"\n  {len(dead)} of {len(rows)} parameters cannot influence this composition:")
        for r in dead:
            print(f"     {r['op']}.{r['param']}  {r['verdict']}")
    return dead


# --------------------------------------------------------------------------- the whole battery
def perturbations(value):
    """A grid around the spec's OWN value, because that is what a sweep actually varies.

    Not a point from the declared range: all six pool parents sit outside their declared range on
    at least one parameter, so "inside the range" is not a safety property here and a probe that
    jumped to a range midpoint would be testing a different composition, not this one's knob.
    """
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, int):
        return [value + 1, max(0, value - 1)] if value != 1 else [2]
    if isinstance(value, float):
        if value == 0.0:
            return [0.1, -0.1]                 # a zero has no factor; step off it in both signs
        return [value * 0.5, value * 2.0]
    return []                                   # strings/lists: a mode, not a knob


def battery(fixture_spec, ckpt, frames=50, device="cpu", only=None, verbose=True,
            freeze_topology=False):
    """Every operator, every parameter, on one warm fixture. One baseline, N probes."""
    warm = warm_spec(fixture_spec, ckpt, frames=frames, freeze_topology=freeze_topology)
    if verbose:
        print(f"baseline: {frames} frames from {os.path.basename(ckpt)}", flush=True)
    base = run_spec(warm, frames, device)

    rows = []
    for o in warm.get("operators", []):
        name = o["op"]
        if only and name not in only:
            continue
        declared = _params_of(warm, name)
        missing, err = unread_params(name, declared, o.get("model") or o.get("implementation"))
        for key, val in sorted(declared.items()):
            if err is None and key in (missing or []):
                rows.append(dict(op=name, param=key, verdict="UNREAD", rel=0.0,
                                 grad=None, note="never looked up"))
                if verbose:
                    print(f"  {name}.{key:18} UNREAD", flush=True)
                continue
            grid = perturbations(val)
            if not grid:
                continue
            try:
                r, _ = probe(warm, name, key, grid, frames, device, baseline=base)
            except Exception as e:
                rows.append(dict(op=name, param=key, verdict="ERROR", rel=0.0, grad=None,
                                 note=f"{type(e).__name__}: {e}"))
                if verbose:
                    print(f"  {name}.{key:18} ERROR {type(e).__name__}", flush=True)
                continue
            rel = max(x["rel_change"] for x in r)
            g, note = None, ""
            if rel > 0 and isinstance(val, float):
                try:
                    g, note = gradient(warm, name, key, frames, device)
                except Exception as e:
                    g, note = None, f"{type(e).__name__}"
            verdict = ("DEAD" if rel == 0 else
                       "LIVE+grad" if g else "LIVE, no derivative")
            rows.append(dict(op=name, param=key, verdict=verdict, rel=rel, grad=g, note=note))
            if verbose:
                print(f"  {name}.{key:18} {verdict:20} rel {rel:.3e}"
                      f"{'' if g is None else f'  d/dth {g:.3e}'}", flush=True)
    return rows


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--op", default=None)
    ap.add_argument("--param", default=None)
    ap.add_argument("--values", nargs="*", default=None)
    ap.add_argument("--frames", type=int, default=140)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--unread-only", action="store_true")
    ap.add_argument("--all", action="store_true", help="the whole battery, on a warm fixture")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "fixtures", "coral_gate_div_f400.npz"))
    ap.add_argument("--build-fixture", type=int, default=0, metavar="FRAME")
    ap.add_argument("--out", default=None, help="write the table as json")
    ap.add_argument("--freeze-topology", action="store_true",
                    help="drop divide/T1: fixed state space, required for a meaningful "
                         "magnitude or any gradient")
    a = ap.parse_args()

    import yaml
    import plexus.operators                                             # noqa: F401
    import tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d, tyssue_monolayer  # noqa: F401
    import tyssue_shape_to_chem                                         # noqa: F401
    import ckpt                                                         # noqa: F401  load_mesh_3d
    spec = yaml.safe_load(open(a.spec))

    print(f"UNREAD probe -- parameters the operator never looks up\n")
    for o in spec.get("operators", []):
        name = o.get("op")
        params = {k: v for k, v in o.items() if k not in ("op", "id", "at", "every",
                                                          "when", "before_frame", "after_frame")}
        missing, err = unread_params(name, params, o.get("model") or o.get("implementation"))
        if err:
            print(f"  {name:24} -- {err}")
        elif missing:
            print(f"  {name:24} UNREAD: {missing}")
    if a.unread_only:
        return

    if a.build_fixture:
        os.makedirs(os.path.dirname(a.ckpt), exist_ok=True)
        make_checkpoint(spec, a.build_fixture, a.ckpt, a.device)
        return

    if a.all:
        print(f"\nBATTERY -- {a.frames} frames per probe, warm from "
              f"{os.path.basename(a.ckpt)}\n")
        rows = battery(spec, a.ckpt, a.frames, a.device,
                       only=[a.op] if a.op else None,
                       freeze_topology=a.freeze_topology)
        report(rows)
        if a.out:
            import json
            # THE FIXTURE TRAVELS WITH THE VERDICT. A bare list of rows says `max_flips` is DEAD
            # and does not say that it is dead only because THIS mesh never reached 20 flips per
            # call. The round's menu withholds a DEAD parameter, so an unattributed verdict would
            # withhold a working limiter on every future parent.
            json.dump({"fixture": os.path.basename(a.ckpt),
                       "spec": os.path.basename(a.spec),
                       "frames": a.frames,
                       "topology": "frozen" if a.freeze_topology else "full composition",
                       "rows": rows}, open(a.out, "w"), indent=1)
            print(f"\n  wrote {a.out}")
        return

    if a.op and a.param:
        vals = [float(v) if _isnum(v) else v for v in (a.values or [])]
        rows, _ = probe(spec, a.op, a.param, vals, a.frames, a.device)
        print(f"\nTRAJECTORY probe -- {a.frames} frames\n")
        for r in rows:
            print(f"  {r['op']}.{r['param']}: {r['from']} -> {r['to']}   "
                  f"rel change {r['rel_change']:.3e}   {r['verdict']}")


def _isnum(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
