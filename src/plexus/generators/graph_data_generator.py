"""Ground-truth data generation: run a spec's forward simulation, save it.

This is the Plexus analogue of connectome-gnn's `data_generate`: given a validated
`Spec`, it builds the Hierarchy, rolls the schedule forward through the
engine, and writes the trajectory dataset (+ a preview) under

    {data_root}/graphs_data/<pre_folder>/<name>/

The saved trajectory is the dataset a future inverse GNN will train on (recover
the operators/parameters from the dynamics). For now generation == the forward
run; the engine stays the interpreter, the generator owns the data layout + viz.
"""
from __future__ import annotations

import os
import shutil

import numpy as np

from plexus.schema import Spec
from plexus.engine import run
from plexus.paths import graphs_data_path


def data_generate(
    sim: Spec,
    pre_folder: str,
    device: str = "cpu",
    erase: bool = False,
    save: bool = True,
    live_every_frac: float | None = 0.05,
    live_movie: dict | None = None,
) -> tuple[str, dict]:
    """forward-simulate `sim` and write its trajectory under
    graphs_data/<pre_folder>/<sim.name>/. Returns (data_dir, out).

    generation writes DATA ONLY -- the trajectory + metadata. Visualization is a
    separate, external concern (plexus.plot, run as `Plexus_Main -o plot`); the
    generator never imports matplotlib, so adding simulations never grows a plot
    switch in here (the ParticleGraph anti-pattern).

    `live_movie` is the ONE EXCEPTION, and it earns it by being the case the separate
    concern cannot serve: at 100 M particles a single recorded frame is 1.2 GB, so
    `plot_dataset` has nothing to read back from. It is an `on_frame` hook -- the same
    extension point `live_every_frac` already uses -- and its pyvista import lives inside
    `plexus.live_movie`, below the branch that decides whether to build one, so this module
    still imports no rendering stack. Pass `None` (what `--no-viz` does) and nothing is
    built, which is how a throughput measurement is taken."""
    folder = pre_folder.rstrip("/")
    data_dir = graphs_data_path(folder, sim.name)
    if erase and os.path.isdir(data_dir):
        # EMPTY THE DIRECTORY, DO NOT REMOVE IT -- because the data root is NFS. When a run is
        # killed while writing its mp4, NFS keeps the still-open file alive under a silly-rename
        # `.nfsXXXX` handle and refuses to unlink either it or its parent until the holder exits.
        # `shutil.rmtree(data_dir)` then dies with `OSError: [Errno 16] Device or resource busy`
        # and `--force` -- whose entire job is to clear the way -- becomes the thing that blocks
        # the run. Deleting the CONTENTS and skipping the `.nfs*` handles leaves nothing stale
        # behind (they vanish by themselves the moment the writer dies) and cannot fail on a
        # directory that is merely in use.
        for entry in os.scandir(data_dir):
            if entry.name.startswith(".nfs"):
                continue
            try:
                shutil.rmtree(entry.path) if entry.is_dir() else os.unlink(entry.path)
            except OSError as e:
                print(f"[generate] --force could not remove {entry.name}: {e}", flush=True)
    os.makedirs(data_dir, exist_ok=True)

    out_path = os.path.join(data_dir, "simulation.zarr") if save else None
    print(f"[generate] {folder}/{sim.name}: {sim.n_frames} frames, "
          f"sets={ {k: int(v.get('n', 0)) for k, v in sim.sets.items() if 'n' in v} } -> {data_dir}",
          flush=True)
    # THE LIVE SNAPSHOT, ON BY DEFAULT. A 1,800-frame run writes nothing anyone can look at for
    # half an hour; this rewrites `2d.png`/`3d.png` in the data directory every 5% of the frames --
    # TWENTY pictures over a run rather than ten. At 10% a 6,000-frame Turing run showed nothing for
    # the first ten minutes and then jumped 600 frames at a time, which is too coarse to catch a
    # pattern forming or a run going non-finite while there is still time to kill it. The cost is a
    # matplotlib figure per snapshot, a fraction of a second against the minutes between them.
    # frame number on it, so the run can be watched rather than only waited for. Off with
    # `live_every_frac=None`. The matplotlib import is inside `plexus.live.snapshot`, so this module
    # still imports no plotting stack -- see its own docstring on why that matters.
    hooks = []
    # THE VTK MOVIE SUPERSEDES THE MATPLOTLIB PNG SNAPSHOT -- do not run both. They answer the same
    # question ("what is this run doing right now") and the movie answers it more often and far more
    # cheaply: 0.1 s a frame against seconds-to-minutes for a matplotlib scatter of the full set.
    # Running both is how a 100 M render came to spend its time in `snapshot` while the movie it was
    # asked for sat idle. `--no-viz` still turns off both; asking for the movie now turns off the
    # stills, and a run that wants the stills can pass `live_movie=None`.
    if live_movie is not None:
        live_every_frac = None
    if live_every_frac:
        from plexus.live import every_n, snapshot
        _stride = every_n(sim.n_frames, live_every_frac)

        def _snap(H, tick, _d=data_dir, _n=sim.n_frames, _name=sim.name, _s=_stride,
                  _st=(sim.plotting or {})):
            if tick % _s == 0 or tick == _n:
                # THE SPEC'S OWN COLOUR TABLE. Which chem column is drawn in which colour is a
                # property of the model (a Gray-Scott substrate is usually not drawn; May-Leonard's
                # three species are a partition and want RGB), so it travels with the spec rather
                # than being guessed by the renderer.
                snapshot(H, tick, _n, _d, name=_name, style=_st)
        hooks.append(_snap)

    movs = []
    # A MESH SPEC GETS NO POINT MOVIE. `live_movie` draws the largest set that carries positions,
    # and for a vertex-model run that is the VERTEX set -- a few hundred points scattered on a
    # surface, in the corner of a world box sized for the grown tissue. The result is a mostly empty
    # frame that is nonetheless called `movie.mp4`, so the run's headline artefact was the one
    # picture of it that shows nothing. When the spec has declared `renderer: vtk_mesh` it has said
    # which renderer it wants; `render_vtk` then writes `movie.mp4` itself.
    _want = str(((sim.plotting or {}).get("renderer", "") or "")).strip().lower()
    if live_movie is not None and _want == "vtk_mesh":
        print("[live-movie] skipped: plotting.renderer is vtk_mesh, so the mesh renderer writes "
              "movie.mp4 -- a point cloud of a mesh set's vertices is not this run's picture",
              flush=True)
        live_movie = None
    if live_movie is not None:
        from plexus.live_movie import LiveMovie
        # SEVERAL MOVIES FROM ONE SIMULATION. At 200 M particles the trajectory is not stored --
        # 1000 frames would be 2.4 TB -- so a movie cannot be re-rendered at a different draw count
        # afterwards; the only way to see the same run at 10 M, 50 M and 100 M drawn is to write all
        # three DURING it. Each gets its own subsample and its own file; they share the simulation.
        _cfg = dict(live_movie)
        _ns = _cfg.pop("render_n")
        _ns = [int(x) for x in (_ns if isinstance(_ns, (list, tuple)) else [_ns])]
        for _n in _ns:
            _tag = "" if len(_ns) == 1 else f"_{_n // 1000000}M" if _n >= 1000000 else f"_{_n // 1000}k"
            movs.append(LiveMovie(out=os.path.join(data_dir, f"movie{_tag}.mp4"),
                                  world=list(sim.world_size), n_frames=sim.n_frames,
                                  up=int((sim.plotting or {}).get("up_axis", 2)),
                                  name=sim.name, sim=sim, style=(sim.plotting or {}),
                                  # A FREE BOUNDARY PUTS NOTHING AT [0, world]: this spec's box is a
                                  # camera hint, and the content is about the origin.
                                  # `free` SAYS NOTHING ABOUT WHERE THE CONTENT IS. A vesicle is
                                  # built about the ORIGIN and an MPM block is seeded inside
                                  # [0, world]; a spec with both has one of each, and framing on
                                  # +-world/2 then drew the gel outside the box it was in. When
                                  # there are material points the box is [0, world] and is right.
                                  centred=(str(getattr(sim, "boundary", "") or "").lower() == "free"
                                           and not any(k in (sim.sets or {}) for k in
                                                       ("mpm_particle", "cytosol", "nucleus"))),
                                  can_curve=False,      # the live path has only the current frame
                                  render_n=_n, stills=(_cfg.get("stills", 10) if _n == _ns[0] else 0),
                                  **{k: v for k, v in _cfg.items() if k != "stills"}))
        hooks.extend(movs)

    # COMPOSED, not replaced. The live PNG snapshot and the live movie are independent answers to
    # "what is this run doing right now" and a run may want both; `on_frame` is a single slot, so
    # the composition happens here rather than by one hook knowing about the other.
    on_frame = None
    if hooks:
        def on_frame(H, tick, _hs=tuple(hooks)):
            for h in _hs:
                h(H, tick)

    # WHICH ENGINE. `general.engine` selects the driver; the default is plexus.engine.run and every
    # existing spec omits the key, so nothing changes for them. A named engine is a MODE OF RUNNING,
    # not a different physics -- `continuous` still delegates to the same loop, and exists to refuse
    # a spec that its operators alone cannot check.
    _engine = str(getattr(sim, "engine", "default") or "default").lower()
    _run = run
    if _engine not in ("", "default"):
        try:
            import importlib
            _run = importlib.import_module(f"plexus.{_engine}_engine").run
            print(f"[engine] using the {_engine!r} engine (plexus.{_engine}_engine.run)", flush=True)
        except Exception as e:                                       # noqa: BLE001
            raise ValueError(f"general.engine: {_engine!r} -- no such engine "
                             f"(expected plexus/{_engine}_engine.py with a run(); {e})") from e
    try:
        H, out = _run(sim, out_path=out_path, device=device, progress=True, on_frame=on_frame)
    finally:
        for _m in movs:
            _m.close()

    # also save a light, framework-agnostic .npz (positions/occupancy per set) so
    # downstream code need not depend on zarr to read a generated dataset back.
    if save:
        flat = {}
        for sname, d in out["sets"].items():
            # A SET NEED NOT HAVE POSITIONS. The vertex model's `cell` set carries `chem`, `cen` and
            # `area` and no `pos` block at all, so `d["pos"]` is None -- and writing None into an
            # npz makes a 0-d OBJECT array, which `np.load` then refuses without `allow_pickle`.
            # Every read of that file died on `cell__pos`, including `plot.py`'s own.
            if d["pos"] is not None:
                flat[f"{sname}__pos"] = d["pos"]
            # AND ITS RECORDED BLOCKS WERE NOT WRITTEN AT ALL. `_assemble` fills `state` with every
            # block the schema marks recorded -- the morphogen concentrations, the per-cell area,
            # the centroid -- and the npz writer dropped the lot, so a reaction-diffusion run wrote
            # a trajectory with no chemistry in it. The zarr path had them; the npz is what every
            # offline reader actually opens.
            for bname, arr in (d.get("state") or {}).items():
                flat[f"{sname}__{bname}"] = arr
            flat[f"{sname}__occ"] = d["occ"]
            # THE RECORDED TOPOLOGY, FLATTENED -- and flattened rather than pickled on purpose. A
            # list of per-row dicts goes into an npz only as a 0-d object array, and `np.load`
            # refuses those without `allow_pickle=True`, which every reader would then have to pass
            # and which makes a trajectory file executable. So: the three half-edge columns
            # concatenated end to end, plus the row offsets, plus nF/Nv per row. Row t's table is
            # `E_srce[off[t]:off[t+1]]`.
            ms = d.get("mesh")
            if ms:
                off = np.cumsum([0] + [len(m["E_srce"]) for m in ms]).astype(np.int64)
                flat[f"{sname}__mesh_offsets"] = off
                flat[f"{sname}__mesh_nF"] = np.asarray([m["nF"] for m in ms], np.int64)
                flat[f"{sname}__mesh_Nv"] = np.asarray([m["Nv"] for m in ms], np.int64)
                # A SECOND SET OF OFFSETS, because the per-face arrays are indexed by FACE and the
                # half-edge columns by HALF-EDGE, and the two ragged lengths are different numbers
                # (roughly 6 half-edges per face). One offsets array for both would silently slice
                # the wrong rows out of the myosin.
                foff = np.cumsum([0] + [int(m["nF"]) for m in ms]).astype(np.int64)
                flat[f"{sname}__mesh_face_offsets"] = foff
                for col in ("E_srce", "E_trgt", "E_face"):
                    flat[f"{sname}__mesh_{col}"] = (np.concatenate([m[col] for m in ms])
                                                    .astype(np.int64))
                from plexus.models.mesh import mesh_row_columns
                # the per-face state the renderer colours by. THE UNION of the names any row
                # carries, zero-filled where a row lacks one -- see `mesh_row_columns`, which
                # records why the intersection this replaced deleted `apop`, `age` and `ndiv` from
                # entire runs and made an apoptosis scene render as a plain ball.
                scal, edge, face, fill = mesh_row_columns(ms)
                # the operators' own SCALAR counters: one value per row, not a ragged column
                for col in scal:
                    flat[f"{sname}__mesh_{col}"] = np.asarray([fill(m, col) for m in ms], np.float64)
                # per-HALF-EDGE columns ride `mesh_offsets`; per-FACE columns ride
                # `mesh_face_offsets`. A column whose per-row length does not match the store it
                # would ride is written with its OWN offsets rather than silently mis-sliced.
                for col in edge:
                    vals = [fill(m, col) for m in ms]
                    flat[f"{sname}__mesh_{col}"] = np.concatenate(vals).astype(np.float32)
                    flat[f"{sname}__mesh_{col}_offsets"] = (
                        np.cumsum([0] + [len(v) for v in vals]).astype(np.int64))
                for col in face:
                    flat[f"{sname}__mesh_{col}"] = (np.concatenate([fill(m, col) for m in ms])
                                                    .astype(np.float32))
            if d.get("node_type") is not None:
                flat[f"{sname}__node_type"] = d["node_type"]
            if d.get("parent") is not None:                  # containment: child -> parent index
                flat[f"{sname}__parent"] = d["parent"]
                flat[f"{sname}__parent_name"] = np.asarray(d["parent_name"])
        for fname, fd in out.get("fields", {}).items():     # continuum fields (heatmap movies)
            flat[f"{fname}__grid"] = fd["grid"]
            flat[f"{fname}__colors"] = fd["colors"]
        np.savez(os.path.join(data_dir, "trajectory.npz"), world=out["world"],
                 world_size=out["world_size"], **flat)

    # count the rows off a set that HAS them -- `occ` is recorded for every set, spatial or not
    nrec = next(iter(out["sets"].values()))["occ"].shape[0]
    print(f"[generate] done: {nrec} recorded frames -> {data_dir}", flush=True)

    # AND THE MESH RENDERER RUNS HERE, so `-o generate` on a mesh spec still leaves a movie behind.
    # The point movie was skipped above precisely because it is the wrong picture for this spec; a
    # generation that therefore produced NO picture would have traded a misleading artefact for a
    # missing one. It needs `trajectory.npz`, which is why it is after the save and not beside the
    # skip. `-o plot` reaches the same two files by the same path.
    # A CURVE PANEL NEEDS THE WHOLE CLIP, so a live generate cannot draw one -- its axes are fixed
    # over the run and the run is not over. The movie it wrote is therefore the right picture minus
    # the panels, and whichever of `-o generate` / `-o plot` ran LAST decided what `movie.mp4`
    # contains. Re-rendering here off the trajectory that was just written costs one replay and
    # makes the two entry points produce the same file again.
    # ...AND ONLY WHEN THE COLOUR SURVIVES THE TRIP. `deformation` and friends are computed from
    # the per-particle deformation gradient, which the trajectory does not store, so re-rendering
    # such a spec off disk would REPLACE a correctly coloured movie with one the replay can draw.
    # The panels are worth less than the field.
    _cf = str(((sim.plotting or {}).get("color_field", "") or "")).lower()
    if _cf in ("deformation", "strain", "volume", "pressure", "vorticity"):
        if (sim.plotting or {}).get("curve"):
            print(f"[render] plotting.curve declared but `color_field: {_cf}` cannot be recomputed "
                  f"from a trajectory -- keeping the movie this run just made, panels and all "
                  f"omitted. Use `color_field: speed` if the panels matter more.", flush=True)
    elif save and (sim.plotting or {}).get("curve") and live_movie is not None:
        try:
            from plexus import live_movie as _lm, render_vtk as _rv
            if _rv.available():
                print("[render] plotting.curve declared -- re-rendering the movie off the "
                      "trajectory, which is the first point the panel's axes can be fixed",
                      flush=True)
                # `stills=10` so `3d.png` is rewritten from the SAME render as the movie.
                # Left at the default 0 the re-render replaced `movie.mp4` and kept the
                # live path's still -- so the folder held a movie with the curve panels and
                # a 3d.png without them, which is the file a browser previews.
                _lm.replay(data_dir, sim, name=sim.name, stills=10)
        except Exception as e:                              # noqa: BLE001
            print(f"[render] curve re-render unavailable ({type(e).__name__}: {e})", flush=True)
    if save and _want == "vtk_mesh" and live_movie is None:
        try:
            from plexus import render_vtk
            if render_vtk.available():
                render_vtk.still(data_dir, style="flat",
                                 out=os.path.join(data_dir, "3d.png"), name=sim.name)
                render_vtk.render_all(data_dir, seq=int((sim.plotting or {}).get("vtk_seq", 2)),
                                      name=sim.name)
        except Exception as e:                              # noqa: BLE001 -- never lose a finished run
            print(f"[render] mesh movie unavailable ({type(e).__name__}: {e})", flush=True)
    return data_dir, out
