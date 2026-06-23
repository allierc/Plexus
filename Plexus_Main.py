"""Plexus entry point (mirrors connectome-gnn's GNN_Main).

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
stubbed for the inverse-problem stages and fail loudly until built.
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
    parser = argparse.ArgumentParser(description="plexus")
    parser.add_argument("-o", "--option", nargs="+", required=True,
                        help="<task> <config_name>, e.g. -o generate attraction_repulsion")
    parser.add_argument("--output_root", default=None,
                        help="root for graphs_data/ and log/ (default: $PLEXUS_OUTPUT_ROOT / $GNN_OUTPUT_ROOT / shared GraphData)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--force", action="store_true",
                        help="erase + regenerate data even if it already exists")
    parser.add_argument("--movie", action="store_true",
                        help="on -o plot, also render a gif movie per set")
    parser.add_argument("--grid", action="store_true",
                        help="render the MLS-MPM 6-panel grid-diagnostic movie (objects/C/F/Jp/stress/grid)")
    parser.add_argument("--no-describe", action="store_true",
                        help="skip the automatic VLM video description that -o generate runs by default")
    parser.add_argument("--describe-out", default=None,
                        help="aggregate description file (default: graphs_data/video_descriptions.txt)")
    args = parser.parse_args()

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
        from plexus.generators.mpm_cfl import enforce_grid_cfl
        enforce_grid_cfl(yaml_file)
    sim = load(yaml_file)

    # self-describing run dir: snapshot the spec into log/<type>/<name>/
    run_log_dir = log_path(pre_folder.rstrip("/"), name)
    os.makedirs(run_log_dir, exist_ok=True)
    shutil.copy2(yaml_file, os.path.join(run_log_dir, "config.yaml"))

    describe = not args.no_describe
    data_dir = None

    if "generate" in task:
        data_dir, _ = data_generate(sim, pre_folder, device=args.device,
                                    erase=args.force, save=True)
        shutil.copy2(yaml_file, os.path.join(data_dir, "spec.yaml"))   # co-locate the spec with its data
        _mark(run_log_dir, "_completed_generate", data_dir)

    # render movies if plotting was asked OR describing (the captioner needs the mp4s)
    if "plot" in task or (describe and "generate" in task):
        from plexus.plot import plot_dataset
        data_dir = plot_dataset(sim, pre_folder, movie=(args.movie or describe))
        if "plot" in task:
            _mark(run_log_dir, "_completed_plot", data_dir)

    # optional MLS-MPM grid-diagnostic movie (re-runs the sim to capture F/C/Jp/stress/grid)
    if args.grid and data_dir is None and ("generate" in task or "plot" in task):
        from plexus.paths import graphs_data_path
        data_dir = os.path.join(graphs_data_path(), pre_folder.rstrip("/"), name)
    if args.grid and data_dir:
        from plexus.generators.mpm_grid_diag import generate_grid_movie
        generate_grid_movie(sim, data_dir, device=args.device)

    # caption the freshly rendered movies (default on for -o generate; --no-describe to skip)
    if describe and "generate" in task and data_dir:
        _describe(data_dir, args.describe_out)
        _mark(run_log_dir, "_completed_describe", args.describe_out or "graphs_data/video_descriptions.txt")

    for stage in ("train", "test"):
        if stage in task:
            raise NotImplementedError(
                f"task stage {stage!r} is not built yet (inverse-problem stage).")

    _mark(run_log_dir, "_complete", " ".join(sys.argv))


def _describe(data_dir: str, out_file: str | None) -> None:
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
    movies = sorted(glob.glob(os.path.join(data_dir, "movie_*.mp4")))
    if not movies:
        print("[describe] no movies found to describe", flush=True)
        return
    gd = graphs_data_path()
    out_file = out_file or os.path.join(gd, "video_descriptions.txt")
    script = os.path.join(repo, "VLLM", "describe_video.py")
    print(f"[describe] captioning {len(movies)} movie(s) -> {out_file}", flush=True)
    subprocess.run([sys.executable, script, *movies, "--root", gd,
                    "--out", out_file, "--append"], check=False)


def _mark(run_log_dir: str, marker: str, info: str) -> None:
    with open(os.path.join(run_log_dir, marker), "w") as f:
        f.write(f"{info}\n")


if __name__ == "__main__":
    main()


# python Plexus_Main.py -o generate attraction_repulsion
# python Plexus_Main.py -o generate interaction/attraction_repulsion --force
# PLEXUS_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData python Plexus_Main.py -o generate attraction_repulsion
