"""Data-root, config, and pre-folder resolution (mirrors connectome-gnn's utils).

Plexus keeps three roots:
  * the **repo** holds `config/<pre_folder>/<name>.yaml` (the specs, version-controlled);
  * the **data root** holds `graphs_data/<pre_folder>/<name>/` (generated trajectories)
    and `log/<pre_folder>/<name>/` (run logs);  default = the shared GraphData area,
    overridable with `--output_root` / `$PLEXUS_OUTPUT_ROOT` / `$GNN_OUTPUT_ROOT`.

The **pre-folder** is the simulation *type* (interaction / boids / mpm / divide ...).
It is inferred from the config name -- the way the prototype scenarios were named
("attraction_repulsion", "tissue_two_types_mpm", "divide_cell") -- so a bare
`-o generate attraction_repulsion` routes to `config/interaction/...` and writes to
`graphs_data/interaction/...`. A name containing a `/` names its folder explicitly.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
#  data root (graphs_data/ + log/)
# --------------------------------------------------------------------------- #
_DEFAULT_DATA_ROOT = "/groups/saalfeld/home/allierc/GraphData"
_data_root = os.environ.get("PLEXUS_OUTPUT_ROOT") or os.environ.get("GNN_OUTPUT_ROOT") or _DEFAULT_DATA_ROOT


def get_data_root() -> str:
    return _data_root


def set_data_root(path: str) -> None:
    """Set the root under which graphs_data/ and log/ live (call early in main)."""
    global _data_root
    _data_root = path


def graphs_data_path(*parts: str) -> str:
    """`graphs_data_path('interaction', 'ar') -> {data_root}/graphs_data/interaction/ar`"""
    return os.path.join(get_data_root(), "graphs_data", *parts)


def log_path(*parts: str) -> str:
    """`log_path('interaction', 'ar') -> {data_root}/log/interaction/ar`"""
    return os.path.join(get_data_root(), "log", *parts)


def get_repo_root() -> str:
    # this file is src/plexus/paths.py -> repo root is three levels up
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def config_path(*parts: str) -> str:
    """`config_path('interaction', 'ar.yaml') -> {repo}/config/interaction/ar.yaml`.
    Configs always load from the repo (version-controlled), never the data root;
    use an absolute path to load a spec from elsewhere."""
    return os.path.join(get_repo_root(), "config", *parts)


# --------------------------------------------------------------------------- #
#  pre-folder (simulation type) inference
# --------------------------------------------------------------------------- #
# Ordered (folder, trigger-substrings). First folder whose any trigger appears
# in the config name wins. Grow this as new simulation types are added.
_PRE_FOLDER_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("attraction_repulsion", ("attraction_repulsion", "attract", "repuls", "arbitrary")),
    ("boids",       ("boids", "flock", "swarm")),
    ("slime",       ("slime", "physarum", "trail")),
    ("material",    ("material",)),
    ("mpm",         ("mpm", "tissue", "elastic", "soft")),
    ("divide",      ("divide", "grow", "mitosis", "morula")),
    ("forage",      ("forage", "maze", "race", "graze")),
    ("gravity",     ("gravity",)),
]
_VALID_PRE_FOLDERS = {f for f, _ in _PRE_FOLDER_RULES}


def validate_pre_folder(pre_folder: str) -> None:
    if not pre_folder:
        return
    name = pre_folder.rstrip("/")
    assert name in _VALID_PRE_FOLDERS, (
        f"pre_folder {pre_folder!r} is not a known simulation type "
        f"(expected one of: {', '.join(sorted(_VALID_PRE_FOLDERS))})")


def add_pre_folder(config_name: str) -> tuple[str, str]:
    """Map a bare config name to (`<pre_folder>/<name>`, `<pre_folder>/`).

    Idempotent: a name that already contains `/` is taken to name its folder
    explicitly (the parent), so re-prefixing is avoided."""
    if "/" in config_name:
        head = config_name.split("/", 1)[0]
        return config_name, head + "/"
    low = config_name.lower()
    # leftmost trigger wins: the type keyword leads the name, so `slime_two_attract`
    # routes to slime (`slime` at pos 0), not attraction_repulsion (`attract` at pos 10).
    best = None                                    # (position, folder)
    for folder, triggers in _PRE_FOLDER_RULES:
        hits = [low.find(t) for t in triggers if t in low]
        if hits and (best is None or min(hits) < best[0]):
            best = (min(hits), folder)
    if best is not None:
        return os.path.join(best[1], config_name), best[1] + "/"
    raise ValueError(
        f"cannot infer a simulation type (pre-folder) from config name {config_name!r}; "
        f"known types: {', '.join(sorted(_VALID_PRE_FOLDERS))}. "
        f"Either rename the config to contain a type keyword, pass it as "
        f"'<type>/{config_name}', or add a rule in plexus.paths._PRE_FOLDER_RULES.")


def resolve_config(config_name: str) -> tuple[str, str, str]:
    """Resolve a CLI config argument to (yaml_file, pre_folder, name).

    Accepts an absolute/relative filesystem path (folder = parent dir) or a
    repo-relative bare/`type/`-prefixed name (folder inferred / explicit)."""
    if os.path.isfile(config_name) or os.path.isabs(config_name) or config_name.endswith(".yaml"):
        yaml_file = config_name if config_name.endswith(".yaml") else config_name + ".yaml"
        pre_folder = os.path.basename(os.path.dirname(os.path.abspath(yaml_file))) + "/"
        name = os.path.splitext(os.path.basename(yaml_file))[0]
        return yaml_file, pre_folder, name
    rel, pre_folder = add_pre_folder(config_name)
    yaml_file = config_path(f"{rel}.yaml")
    name = os.path.basename(rel)
    return yaml_file, pre_folder, name
