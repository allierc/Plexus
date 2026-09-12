"""The `-o generate` pipeline as ONE function, so every entry point runs the same run.

`Plexus_Main.py` parses a command line and calls `generate`; the web page's RUN button calls
`generate` in a thread with an `on_frame` hook. Before this module existed the page had its own
loop around `engine.run` -- its own substep, its own wall model, its own frame keeping, no movie,
nothing in `graphs_data/` -- and the same three bodies ran at 202 ms/frame on the page against 68
from the command line. That was not two engines; it was two callers. This is the one caller.

What `generate` does, in order, is exactly what `Plexus_Main.main` did for `-o generate`:

  1. resolve the spec (a config name or a yaml path) and its pre-folder;
  2. the MPM guards that REWRITE the spec file -- the grid CFL (substep against the elastic and
     capillary wave speeds) and particles-per-cell;
  3. `schema.load`, the gatekeeper;
  4. a copy of the spec into `log/<type>/<name>/spec.yaml`;
  5. the VRAM projection warning for captured runs;
  6. `data_generate` with the live movie shaped by the spec's `plotting:` block (`max_frames`,
     `stills`, `keep_stills`) and the CLI defaults where the spec says nothing;
  7. a copy of the spec next to the data, the completion markers, and the caption pass.

`StopRun` is what a stop button raises inside `on_frame`; `generate` catches it, closes the movie
and returns what was written so far with `stopped=True`.
"""
from __future__ import annotations

import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class StopRun(Exception):
    """Raised from an `on_frame` hook to end a run early; not an error."""


def generate(config_name: str, *, device: str = "cuda:0", force: bool = False, viz: bool = True,
             render_n="500000000", render_max_frames: int = 300, real_time: bool = True,
             keep_stills: bool = False, render_stills: int = 10, render_dot=None,
             describe: bool = True, describe_out: str | None = None, grid: bool = False,
             movie: bool = False, plot: bool = False, on_frame=None, argv_line: str | None = None) -> dict:
    """Run one spec through the generate pipeline. Returns {name, pre_folder, data_dir, run_log_dir,
    spec_path, stopped, frame_ms} -- `frame_ms` is the engine's own per-tick wall clock."""
    import plexus.operators  # noqa: F401  self-register the operator library
    from plexus.schema import load
    from plexus.paths import resolve_config, validate_pre_folder, log_path
    from plexus.generators.graph_data_generator import data_generate

    if device.startswith("cuda"):                    # fall back to CPU if no GPU is present
        import torch
        if not torch.cuda.is_available():
            print(f"[device] {device} unavailable -> falling back to cpu", flush=True)
            device = "cpu"

    yaml_file, pre_folder, name = resolve_config(config_name)
    validate_pre_folder(pre_folder)
    if not os.path.isfile(yaml_file):
        raise FileNotFoundError(f"config not found: {yaml_file}")
    print(f"task=generate  type={pre_folder.rstrip('/')}  config={name}  ({yaml_file})", flush=True)
    # MPM grid-dt CFL: auto-correct the SPEC (not the engine) so dt_sub respects the
    # Courant condition before we generate; idempotent for non-MPM / already-stable specs.
    from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition, particles_per_cell
    Courant_Friedrichs_Lewy_condition(yaml_file)
    # The grid's OTHER discretisation constraint. CFL bounds the time step; this bounds the
    # space step against the particle count, and it had no check at all until a spec was
    # raised to n_grid 192 at a fixed particle count and its snow quietly collapsed.
    particles_per_cell(yaml_file)
    sim = load(yaml_file)

    # self-describing run dir: snapshot the spec into log/<type>/<name>/
    run_log_dir = log_path(pre_folder.rstrip("/"), name)
    os.makedirs(run_log_dir, exist_ok=True)
    shutil.copy2(yaml_file, os.path.join(run_log_dir, "spec.yaml"))

    describe = describe and viz
    # THE mp4 IS WRITTEN BY THE RUN ITSELF, not by a second script and not by a second pass over
    # the trajectory. `plot_dataset` below still runs and still renders from the recorded data;
    # this hook exists for the runs where that is impossible, and at 100 M particles one
    # recorded frame is 1.2 GB so it is impossible often. `viz=False` turns off every renderer.
    _dot = (None if render_dot is None else render_dot if render_dot == "auto" else float(render_dot))
    if viz and device.startswith("cuda"):
        _vram_warning(sim, device)
    _rn = [int(x) for x in str(render_n).split(",") if x.strip()]
    # THE SPEC MAY SET THE MOVIE'S OWN SHAPE. `max_frames` (how many frames the movie keeps, the run
    # strided to fit), `stills` (how many PNGs are dropped through it) and `keep_stills` are render
    # decisions a spec is entitled to make about itself -- a benchmark that wants 400 movie frames
    # and ten pictures should say so once, in the file, rather than on every command line and in
    # every cluster job script that runs it. The CLI values remain the defaults for a spec that
    # says nothing.
    _pl = getattr(sim, "plotting", None) or {}
    _units = getattr(sim, "units", None)
    _declared = bool(getattr(_units, "declared", False))
    lm = None if not viz else {"render_n": (_rn if len(_rn) > 1 else _rn[0]),
                               "max_frames": int(_pl.get("max_frames", render_max_frames)),
                               "dot": _dot,
                               "stills": int(_pl.get("stills", render_stills)),
                               "keep_stills": bool(_pl.get("keep_stills", keep_stills)),
                               # the movie can only be timed if the run has a clock
                               "dt": getattr(sim, "dt", None),
                               "time_s": (_units.time_s if _declared else None),
                               "real_time": real_time,
                               "length_um": (_units.length_um if _declared else None)}
    stopped = False
    frame_ms = None
    try:
        data_dir, out = data_generate(sim, pre_folder, device=device, erase=force, save=True,
                                      live_every_frac=(None if not viz else 0.05),
                                      live_movie=lm, on_frame=on_frame)
        frame_ms = out.get("frame_ms") if isinstance(out, dict) else None
    except StopRun:
        # THE RUN WAS ENDED ON PURPOSE. The movie was closed by `data_generate`'s own finally; the
        # trajectory and the markers are not written, because there is no complete run to mark.
        from plexus.paths import graphs_data_path
        stopped = True
        data_dir = graphs_data_path(pre_folder.rstrip("/"), sim.name)
        print(f"[generate] stopped early -> {data_dir}", flush=True)
    if not stopped:
        shutil.copy2(yaml_file, os.path.join(data_dir, "spec.yaml"))   # co-locate the spec with its data
        _mark(run_log_dir, "_completed_generate", data_dir)

    # render movies if plotting was asked OR describing (the captioner needs the mp4s)
    if viz and not stopped and (plot or describe):
        from plexus.plot import plot_dataset
        data_dir = plot_dataset(sim, pre_folder, movie=(movie or describe))
        if plot:
            _mark(run_log_dir, "_completed_plot", data_dir)

    # optional MLS-MPM grid-diagnostic movie (re-runs the sim to capture F/C/Jp/stress/grid)
    if grid and viz and not stopped:
        from plexus.generators.mpm_grid_diag import generate_grid_movie
        generate_grid_movie(sim, data_dir, device=device)

    # caption the freshly rendered movies (default on for -o generate; --no-describe to skip)
    if describe and not stopped and data_dir:
        _describe(data_dir, describe_out, device=device)
        _mark(run_log_dir, "_completed_describe", describe_out or "graphs_data/video_descriptions.txt")
    if not stopped:
        _mark(run_log_dir, "_complete", argv_line or f"plexus.pipeline.generate({config_name!r})")
    return {"name": name, "pre_folder": pre_folder.rstrip("/"), "data_dir": data_dir,
            "run_log_dir": run_log_dir, "spec_path": yaml_file, "stopped": stopped,
            "frame_ms": frame_ms}


def _vram_warning(sim, device: str) -> None:
    """A CAPTURED GRAPH AND A RENDERER COMPETE FOR THE SAME CARD, and the failure is silent: the
    allocator retries rather than raising, so the run sits at 100% CPU with no output and no
    error. That is what a 100 M render did -- the capture pool plus the renderer plus the
    recording buffers on a 47.4 GiB card -- and the tell was that `[engine] substep captured as a
    CUDA graph` never printed.

    THE TEST IS THE FOOTPRINT AGAINST *THIS* CARD, NOT A PARTICLE COUNT AND NOT A CARD NAME. 20 M is
    a stall on a 48 GiB A6000 and unremarkable on an 80 GiB H100, so a fixed threshold would nag on
    the big card and stay silent on a 24 GiB one. The coefficients are measured, on `warp`, over
    500 k -> 100 M (paper/mpm_warp.pdf 5.2-5.3): 0.309 GiB per million particles eager, 0.42 with
    capture -- capture's private pool stays resident, which is where the ~39% comes from."""
    # `per_parent` MAY BE A MAPPING from the parent's type name to a count (a cell atlas gives a
    # membrane patch 50 points and a nuclear-envelope patch 50,000), so the total is the sum over
    # types of count x per-type budget rather than one product. Getting this wrong is not
    # cosmetic: the number feeds the VRAM projection, which is what warns before a capture stalls
    # the card.
    def _child_total(v):
        pp = v.get("per_parent", 0)
        par = sim.sets.get(v.get("parent"), {}) or {}
        ptypes = par.get("types") or {}
        if isinstance(pp, dict):
            npar = int(par.get("n", par.get("per_parent", 1)) or 1)
            return sum(int(pp.get(tn, 0)) * int(t.get("count", 0) or
                                                round(float(t.get("fraction", 0.0)) * npar))
                       for tn, t in ptypes.items())
        return int(pp) * int(par.get("n", 1))
    _npart = sum(_child_total(v) for v in sim.sets.values() if isinstance(v, dict) and "per_parent" in v)
    _cap_on = any(isinstance(x, dict) and x.get("capture") for x in sim.schedule)
    if _npart and _cap_on:
        import torch
        _tot = torch.cuda.get_device_properties(device).total_memory / 2 ** 30
        _proj = 0.42 * _npart / 1e6            # GiB, captured
        if _proj > 0.80 * _tot:                # the renderer and the recorder want the rest
            print(f"[capture] WARNING: {_npart:,} particles with capture ON projects "
                  f"~{_proj:.0f} GiB on a {_tot:.0f} GiB "
                  f"{torch.cuda.get_device_properties(device).name}, and a live "
                  f"renderer needs room too. Without capture it is ~{0.309 * _npart / 1e6:.0f} "
                  f"GiB. If the run stalls at 100% CPU with no output and never prints "
                  f"'substep captured as a CUDA graph', that is why -- set "
                  f"`capture: false` on the spec's substep block.", flush=True)


def _describe(data_dir: str, out_file: str | None, device: str = "cuda:0") -> None:
    """Caption this run's movies with the local VLM, appending to the aggregate file.
    Runs describe_video.py as a subprocess so a missing/broken VLM never breaks a run."""
    import glob
    import subprocess
    from plexus.paths import graphs_data_path
    gemma = os.environ.get("GEMMA_DIR", os.path.join(REPO, "VLLM", "gemma-4-12B-it"))
    if not os.path.isdir(gemma):
        print(f"[describe] skip: no VLM weights at {gemma} (pass --no-describe to silence)", flush=True)
        return
    # EVERY mp4 THE RUN WROTE, not just the ones named `movie_*`. The two-panel composition view
    # lands as `movie.mp4` and the VTK products as `vtk_*.mp4`, so a run whose ONLY output is the
    # panels reported "no movies found to describe" one line after printing the path of the movie
    # it had just written. The captioner's business is "what did this run produce", and that is a
    # question about the directory, not about a prefix.
    movies = sorted(f for f in glob.glob(os.path.join(data_dir, "*.mp4"))
                    if not os.path.basename(f).startswith("grid_"))   # the MPM grid diagnostic
    if not movies:
        print(f"[describe] no .mp4 in {data_dir} to describe", flush=True)
        return
    gd = graphs_data_path()
    out_file = out_file or os.path.join(gd, "video_descriptions.txt")
    script = os.path.join(REPO, "VLLM", "describe_video.py")
    print(f"[describe] captioning {len(movies)} movie(s) -> {out_file}", flush=True)
    # a caption that did not happen must SAY SO. check=False keeps a broken VLM from killing a run,
    # which is right, but on its own it also let an out-of-memory captioner pass for a successful one.
    before = os.path.getsize(out_file) if os.path.exists(out_file) else 0
    # NO `Loading weights: 100%|####...| 677/677`. It is a full terminal width of blocks, redrawn,
    # for a load the line above already announced, and it reports nothing anyone can act on -- the
    # load either finishes or the caption says UNAVAILABLE. This path runs the captioner as a
    # SUBPROCESS, so the switch has to travel in its environment. THE PATH THIS CHECKED IS THE PATH
    # THE CHILD LOADS: `gemma` is resolved against this checkout and tested with `isdir`; passing it
    # down closes the gap where the parent verifies one directory and the subprocess then goes
    # looking in another.
    _env = {**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TRANSFORMERS_VERBOSITY": "error",
            "GEMMA_DIR": gemma}
    r = subprocess.run([sys.executable, script, *movies, "--root", gd,
                        "--out", out_file, "--append", "--device", device], check=False, env=_env)
    after = os.path.getsize(out_file) if os.path.exists(out_file) else 0
    if r.returncode != 0 or after <= before:
        print(f"[describe] *** NO CAPTIONS WERE WRITTEN *** (exit {r.returncode}, "
              f"{out_file} unchanged at {after} bytes). The movies exist and are undescribed.",
              flush=True)
    else:
        print(f"[describe] wrote {after - before} bytes of captions", flush=True)


def _mark(run_log_dir: str, marker: str, info: str) -> None:
    with open(os.path.join(run_log_dir, marker), "w") as f:
        f.write(f"{info}\n")
