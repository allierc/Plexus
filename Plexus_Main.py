"""Plexus entry point.

    python Plexus_Main.py -o <task> <config_name> [--output_root ROOT] [--force]

`<task>` is one or more of generate / train / test / plot, optionally chained
(e.g. `generate_plot`). `<config_name>` selects a spec; its simulation *type*
(the pre-folder: interaction / boids / mpm / ...) is inferred from the name, so

    python Plexus_Main.py -o generate attraction_repulsion

loads  config/interaction/attraction_repulsion.yaml  and writes the trajectory to
{data_root}/graphs_data/interaction/attraction_repulsion/ . A name with a slash
(interaction/attraction_repulsion) or an absolute .yaml path names its folder
explicitly. The data root defaults to the shared GraphData area; override with
--output_root or $PLEXUS_OUTPUT_ROOT / $GNN_OUTPUT_ROOT.

Only `generate` is implemented today (the forward simulator); train/test/plot are
stubbed for the inverse-problem stages and fail until built.
"""
from __future__ import annotations

import os
import sys
import shutil
import argparse

# ensure src/ is importable when run from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import plexus.operators  # noqa: F401  self-register the operator library
from plexus.schema import load
from plexus.paths import resolve_config, validate_pre_folder, set_data_root, log_path
from plexus.generators.graph_data_generator import data_generate


def main():
    # --- manual / debug entry --------------------------------------------------- #
    # Run Plexus_Main.py with NO CLI args (e.g. the IDE "Debug" button) and it falls
    # back to these hardcoded values. Edit the list to debug a different task/config.
    # Ignored as soon as any real CLI argument is passed.
    if len(sys.argv) == 1:
        # `--no-viz` DELIBERATELY, because this path exists for stepping through operators: without
        # it the VTK plotter is built and fires on every frame while you are stopped on a breakpoint.
        # For line-by-line work also set CUDA_LAUNCH_BLOCKING=1 in the debug environment -- CUDA is
        # asynchronous, so without it a tensor may not be computed when you inspect it and an error
        # surfaces at the wrong line. (`--device cpu` is synchronous and this spec is only 18,000
        # particles.) The MPM `forward`s to break in are all in src/plexus/operators/mpm_ops.py.
        sys.argv += ["-o", "generate",
                     "/workspace/Plexus/config/material/material_3d_multimaterial.yaml",
                     "--device", "cuda:0", "--no-viz"]

    parser = argparse.ArgumentParser(description="plexus")
    parser.add_argument("-o", "--option", nargs="+", required=True,
                        help="<task> <config_name>, e.g. -o generate attraction_repulsion")
    parser.add_argument("--output_root", default=None,
                        help="root for graphs_data/ and log/ (default: $PLEXUS_OUTPUT_ROOT / $GNN_OUTPUT_ROOT / shared GraphData)")
    parser.add_argument("--device", default="cuda:0", help="cuda:N (default) or cpu")
    parser.add_argument("--force", action="store_true",
                        help="erase + regenerate data even if it already exists")
    parser.add_argument("--movie", action="store_true",
                        help="on -o plot, also render a gif movie per set")
    parser.add_argument("--grid", action="store_true",
                        help="render the MLS-MPM 6-panel grid-diagnostic movie (objects/C/F/Jp/stress/grid)")
    parser.add_argument("--no-viz", action="store_true",
                        help="NO RENDERING AT ALL: no live mp4, no live png snapshots, no plot pass, "
                             "no captioning. This is how a throughput measurement is taken -- the "
                             "ms/frame a render is folded into is not the simulation's")
    # DRAW EVERYTHING BY DEFAULT, capped at 500 M. 400,000 was a 2018-era guess and it silently
    # turned every large run into a picture of 0.4% of itself: at 5 M drawn/simulated a body that is
    # 2% of the scene got 2% of 8% and read as absent, which cost a real debugging session. The cap
    # is a cap, not a target -- a spec asking for fewer still gets fewer, and the render is a small
    # share of frame time next to the substeps at every size measured here.
    parser.add_argument("--render-n", default="500000000",
                        help="particles DRAWN in the live mp4 (default: all, capped at 500 M); "
                             "the run still simulates all of them. "
                             "COMMA-SEPARATED writes one movie per value (movie_10M.mp4, "
                             "movie_50M.mp4, ...) from the SAME simulation -- the only way to see a "
                             "run at several draw counts when its trajectory is too big to store")
    parser.add_argument("--render-max-frames", type=int, default=300,
                        help="cap on rendered frames; longer runs are strided down to this")
    parser.add_argument("--no-real-time", action="store_true",
                        help="draw every frame instead of choosing the stride that makes 60 fps "
                             "playback equal real time. Only has an effect when the spec declares "
                             "`general.units`, since without them a second means nothing")
    parser.add_argument("--keep-stills", action="store_true",
                        help="keep the numbered still_NN_*.png after the run. They exist to let a "
                             "run be WATCHED while it runs; once the mp4 is written they are "
                             "redundant copies of frames it already holds, so they are deleted by "
                             "default and only 3d.png (the final frame) is kept")
    parser.add_argument("--render-stills", type=int, default=10,
                        help="how many PNG stills to drop through the run, copied from the movie's "
                             "own rendered frames (no extra render). 0 disables. The newest is "
                             "always also written as 3d.png so a long run can be watched")
    parser.add_argument("--render-dot", default=None,
                        help="dot size in px, or 'auto' to size it to the drawn particles' median "
                             "nearest-neighbour spacing. DEFAULT: the spec's `plotting.dot_size`, "
                             "then auto -- so the size lives in the config, not in the command")
    parser.add_argument("--no-describe", action="store_true",
                        help="skip the automatic VLM video description that -o generate runs by default")
    parser.add_argument("--describe-out", default=None,
                        help="aggregate description file (default: graphs_data/video_descriptions.txt)")
    args = parser.parse_args()

    if args.device.startswith("cuda"):               # fall back to CPU if no GPU is present
        import torch
        if not torch.cuda.is_available():
            print(f"[device] {args.device} unavailable -> falling back to cpu", flush=True)
            args.device = "cpu"

    if args.output_root:
        assert os.path.isdir(args.output_root), f"--output_root does not exist: {args.output_root}"
        assert os.access(args.output_root, os.W_OK), f"--output_root not writable: {args.output_root}"
        set_data_root(args.output_root)

    task = args.option[0]
    config_name = args.option[1] if len(args.option) > 1 else None
    if config_name is None:
        parser.error("a config name is required: -o <task> <config_name>")

    # resolve spec + simulation type, then validate (the gatekeeper)
    yaml_file, pre_folder, name = resolve_config(config_name)
    validate_pre_folder(pre_folder)
    if not os.path.isfile(yaml_file):
        parser.error(f"config not found: {yaml_file}")
    print(f"task={task}  type={pre_folder.rstrip('/')}  config={name}  ({yaml_file})")
    # MPM grid-dt CFL: auto-correct the SPEC (not the engine) so dt_sub respects the
    # Courant condition before we generate; idempotent for non-MPM / already-stable specs.
    if "generate" in task:
        from plexus.generators.mpm_cfl import (Courant_Friedrichs_Lewy_condition,
                                               particles_per_cell)
        Courant_Friedrichs_Lewy_condition(yaml_file)
        # The grid's OTHER discretisation constraint. CFL bounds the time step; this bounds the
        # space step against the particle count, and it had no check at all until a spec was
        # raised to n_grid 192 at a fixed particle count and its snow quietly collapsed.
        particles_per_cell(yaml_file)
    sim = load(yaml_file)

    # self-describing run dir: snapshot the spec into log/<type>/<name>/
    run_log_dir = log_path(pre_folder.rstrip("/"), name)
    os.makedirs(run_log_dir, exist_ok=True)
    shutil.copy2(yaml_file, os.path.join(run_log_dir, "spec.yaml"))   # same name as the data-dir copy (line ~104)

    describe = not args.no_describe and not args.no_viz
    data_dir = None

    if "generate" in task:
        # THE mp4 IS WRITTEN BY THE RUN ITSELF, not by a second script and not by a second pass over
        # the trajectory. `plot_dataset` below still runs and still renders from the recorded data;
        # this hook exists for the runs where that is impossible, and at 100 M particles one
        # recorded frame is 1.2 GB so it is impossible often. `--no-viz` turns off every renderer.
        _dot = (None if args.render_dot is None
                else args.render_dot if args.render_dot == "auto" else float(args.render_dot))
        # A CAPTURED GRAPH AND A RENDERER COMPETE FOR THE SAME CARD, and the failure is silent:
        # the allocator retries rather than raising, so the run sits at 100% CPU with no output and
        # no error. That is what a 100 M render did -- the capture pool plus the renderer plus the
        # recording buffers on a 47.4 GiB card -- and the tell was that `[engine] substep captured
        # as a CUDA graph` never printed.
        #
        # THE TEST IS THE FOOTPRINT AGAINST *THIS* CARD, NOT A PARTICLE COUNT AND NOT A CARD NAME.
        # 20 M is a stall on a 48 GiB A6000 and unremarkable on an 80 GiB H100, so a fixed
        # threshold would nag on the big card and stay silent on a 24 GiB one. The coefficients are
        # measured, on `warp`, over 500 k -> 100 M (paper/mpm_warp.pdf 5.2-5.3): 0.309 GiB per
        # million particles eager, 0.42 with capture -- capture's private pool stays resident, which
        # is where the ~39% comes from.
        if not args.no_viz and args.device.startswith("cuda"):
            _npart = sum(int(v.get("per_parent", 0)) * int(sim.sets.get(v.get("parent"), {}).get("n", 1))
                         for v in sim.sets.values() if isinstance(v, dict) and "per_parent" in v)
            _cap_on = any(isinstance(x, dict) and x.get("capture") for x in sim.schedule)
            if _npart and _cap_on:
                import torch
                _tot = torch.cuda.get_device_properties(args.device).total_memory / 2 ** 30
                _proj = 0.42 * _npart / 1e6            # GiB, captured
                if _proj > 0.80 * _tot:                # the renderer and the recorder want the rest
                    print(f"[capture] WARNING: {_npart:,} particles with capture ON projects "
                          f"~{_proj:.0f} GiB on a {_tot:.0f} GiB "
                          f"{torch.cuda.get_device_properties(args.device).name}, and a live "
                          f"renderer needs room too. Without capture it is ~{0.309 * _npart / 1e6:.0f} "
                          f"GiB. If the run stalls at 100% CPU with no output and never prints "
                          f"'substep captured as a CUDA graph', that is why -- set "
                          f"`capture: false` on the spec's substep block.", flush=True)
        _rn = [int(x) for x in str(args.render_n).split(",") if x.strip()]
        lm = None if args.no_viz else {"render_n": (_rn if len(_rn) > 1 else _rn[0]),
                                       "max_frames": args.render_max_frames, "dot": _dot,
                                       "stills": args.render_stills,
                                       "keep_stills": args.keep_stills,
                                       # the movie can only be timed if the run has a clock
                                       "dt": getattr(sim, "dt", None),
                                       "time_s": (sim.units.time_s if getattr(
                                           getattr(sim, "units", None), "declared", False) else None),
                                       "real_time": not args.no_real_time,
                                       "length_um": (sim.units.length_um if getattr(
                                           getattr(sim, "units", None), "declared", False) else None)}
        data_dir, _ = data_generate(sim, pre_folder, device=args.device,
                                    erase=args.force, save=True,
                                    live_every_frac=(None if args.no_viz else 0.05),
                                    live_movie=lm)
        shutil.copy2(yaml_file, os.path.join(data_dir, "spec.yaml"))   # co-locate the spec with its data
        _mark(run_log_dir, "_completed_generate", data_dir)

    # render movies if plotting was asked OR describing (the captioner needs the mp4s)
    if not args.no_viz and ("plot" in task or (describe and "generate" in task)):
        from plexus.plot import plot_dataset
        data_dir = plot_dataset(sim, pre_folder, movie=(args.movie or describe))
        if "plot" in task:
            _mark(run_log_dir, "_completed_plot", data_dir)

    # optional MLS-MPM grid-diagnostic movie (re-runs the sim to capture F/C/Jp/stress/grid)
    if args.grid and data_dir is None and ("generate" in task or "plot" in task):
        from plexus.paths import graphs_data_path
        data_dir = os.path.join(graphs_data_path(), pre_folder.rstrip("/"), name)
    if args.grid and data_dir and not args.no_viz:
        from plexus.generators.mpm_grid_diag import generate_grid_movie
        generate_grid_movie(sim, data_dir, device=args.device)

    # caption the freshly rendered movies (default on for -o generate; --no-describe to skip)
    if describe and "generate" in task and data_dir:
        _describe(data_dir, args.describe_out, device=args.device)
        _mark(run_log_dir, "_completed_describe", args.describe_out or "graphs_data/video_descriptions.txt")

    for stage in ("train", "test"):
        if stage in task:
            raise NotImplementedError(
                f"task stage {stage!r} is not built yet (inverse-problem stage).")

    _mark(run_log_dir, "_complete", " ".join(sys.argv))


def _describe(data_dir: str, out_file: str | None, device: str = "cuda:0") -> None:
    """Caption this run's movies with the local VLM, appending to the aggregate file.
    Runs describe_video.py as a subprocess so a missing/broken VLM never breaks a run."""
    import glob
    import subprocess
    from plexus.paths import graphs_data_path
    repo = os.path.dirname(os.path.abspath(__file__))
    gemma = os.environ.get("GEMMA_DIR", os.path.join(repo, "VLLM", "gemma-4-12B-it"))
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
    script = os.path.join(repo, "VLLM", "describe_video.py")
    print(f"[describe] captioning {len(movies)} movie(s) -> {out_file}", flush=True)
    # a caption that did not happen must SAY SO. check=False keeps a broken VLM from killing a run,
    # which is right, but on its own it also let an out-of-memory captioner pass for a successful one.
    before = os.path.getsize(out_file) if os.path.exists(out_file) else 0
    # NO `Loading weights: 100%|####...| 677/677`. It is a full terminal width of blocks, redrawn,
    # for a load the line above already announced, and it reports nothing anyone can act on -- the
    # load either finishes or the caption says UNAVAILABLE. `discovery_okuda/caption_wave.py`
    # disables it in-process for the same reason; this path runs the captioner as a SUBPROCESS, so
    # the switch has to travel in its environment instead.
    _env = {**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TRANSFORMERS_VERBOSITY": "error"}
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


if __name__ == "__main__":
    main()


# python Plexus_Main.py -o generate attraction_repulsion
# python Plexus_Main.py -o generate interaction/attraction_repulsion --force
# PLEXUS_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData python Plexus_Main.py -o generate attraction_repulsion
# cd /workspace/Plexus && PYTHONPATH=src /workspace/.conda_envs/neural-graph-linux/bin/python -u \
#   tools/mpm_live_movie.py --spec config/material/material_3d_water_bench_100m.yaml \
#   --frames 90 --render-n 400000 --device cuda:1 \
#   --out graphs_data/cell/mpm_100m/movie.mp4
# bsub -n 8 -gpu "num=1" -q gpu_a100 -W 96:00 \
#   "cd /groups/saalfeld/home/allierc/Graph/Plexus && PYTHONPATH=src python -u Plexus_Main.py \
#      -o generate material_3d_water_bench_100mL --device cuda:0 \
#      --render-n 100000008 --render-max-frames 500 --no-describe"


