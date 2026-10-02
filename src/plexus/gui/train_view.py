"""A TRAINING RUN IN THE WATCHER'S 3-D VIEW (exp17, Cedric 2026-10-01: "I would like to viz in 3D the learned zebrafish
activity"; the 3D button answered "simulation missing required key: 'sets'" on every training step).

The 3-D view re-opens a step's saved SIMULATION spec and replays its recorded trajectory. A training step saves a
TRAINING spec, and what it learned is not a trajectory but forecasts: `<run>_movie.npz`, the law's free rollout over a
window of the recording (`pred` [frames, elements]). This module turns one into the other, once per run:

    config/train_view/<run>_view.yaml              a simulation spec the view can open: the observed set's elements at
                                                   their positions (`pos`, um, the box the brain's own extent) and the
                                                   observed block (`dff`), coloured by it (`plotting.color_field`), no
                                                   operator -- nothing is simulated, the frames are replayed
    graphs_data/train_view/<run>_view/trajectory.npz   `<set>__pos` and `<set>__<block>` per frame: the LEARNED frames

Nothing of the training run is changed; the view files are rebuilt when the run's movie is newer than them.
"""
from __future__ import annotations

import os

import numpy as np
import yaml

CONFIG = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "config"))


def view_spec_for(train_spec_path: str) -> str:
    """The view spec of a landed training run (built if missing or older than the run's movie)."""
    from plexus.paths import graphs_data_path
    from plexus.trainer import load, out_dir
    # BY NAME: the watcher's record links a step's spec to the run's own copy (log/training/<model>/<run>/config.yaml),
    # from which the run's folder would resolve one level wrong; the name finds config/training/<model>/<run>.yaml
    name = str(yaml.safe_load(open(train_spec_path))["name"])
    spec = load(name)
    out = out_dir(spec, None)
    mv_p = os.path.join(out, "results", f"{name}_movie.npz")
    if not os.path.exists(mv_p):
        raise ValueError(f"{name}: no learned frames yet ({mv_p}) -- the run has not been tested")
    model = yaml.safe_load(open(spec["model"]))
    obs = spec["task"]["observe"]
    sname, block = obs["set"], obs["block"]
    sset = model["sets"][sname]
    seed = next(s for s in model.get("seed", []) if s.get("op") == "seed_state_from_file" and s.get("at") == sname)
    pos_array = (seed.get("blocks") or {}).get(spec_pos(model, sname), "pos_um")   # {xyz: pos_um} in exp17
    z = np.load(graphs_data_path(seed["file"]))
    P = np.asarray(z[pos_array], np.float32)
    P = P - P.min(0)
    ext = P.max(0)
    view = f"{name}_view"
    vdir = graphs_data_path("train_view", view)
    vspec = os.path.join(CONFIG, "train_view", f"{view}.yaml")
    if os.path.exists(vspec) and os.path.exists(os.path.join(vdir, "trajectory.npz")) and \
            os.path.getmtime(vspec) >= os.path.getmtime(mv_p):
        return vspec
    mv = np.load(mv_p)
    pred = np.asarray(mv["pred"], np.float32)                       # [frames, elements], the learned rollout
    T, N = pred.shape
    lo, hi = np.percentile(pred[:: max(1, T // 20)], [2, 98])
    os.makedirs(vdir, exist_ok=True)
    np.savez(os.path.join(vdir, "trajectory.npz"),
             **{f"{sname}__pos": np.broadcast_to(P, (T, N, 3)).astype(np.float32),
                f"{sname}__{block}": pred[:, :, None]})
    np.save(os.path.join(vdir, "positions.npy"), P)
    vs = {"general": {"name": view, "seed": 0, "n_frames": int(T), "dt": float(model["general"].get("dt", 1.0)),
                      "boundary": "wall", "dim": 3, "world": [float(e) for e in ext], "save_data": False},
          "sets": {"group": {"n": 1},
                   sname: {"parent": "group", "per_parent": int(N), "grow_reserve": 0,
                           "title": f"{sset.get('title', sname)} -- the LEARNED frames of {name}",
                           "state": {"pos": {"width": 3, "role": "coordinate", "integration": "none",
                                             "boundary": "free", "title": "position, um"},
                                     block: {"width": 1, "integration": "none", "boundary": "free",
                                             "title": "the learned value (the law's free rollout)"}}}},
          "fields": {},
          "seed": [{"op": "seed_state_from_file", "at": sname, "file": os.path.relpath(
              os.path.join(vdir, "positions.npz"), graphs_data_path()), "blocks": {"pos": "pos"}}],
          "operators": [], "schedule": [],
          "plotting": {"renderer": "vtk_points", "background": "black", "color_field": block,
                       "color_range": [float(lo), float(hi)], "colormap": "inferno"}}
    np.savez(os.path.join(vdir, "positions.npz"), pos=P)
    os.makedirs(os.path.dirname(vspec), exist_ok=True)
    yaml.safe_dump(vs, open(vspec, "w"), sort_keys=False)
    return vspec


def spec_pos(model: dict, sname: str) -> str:
    """The name of the set's position block (the operator's `positions:` key, `xyz` in exp17)."""
    for op in model.get("operators", []):
        if op.get("at") == sname and op.get("positions"):
            return str(op["positions"])
    return "pos"


def is_training_spec(spec: dict) -> bool:
    return isinstance(spec, dict) and "learnable" in spec and "model" in spec and "task" in spec
