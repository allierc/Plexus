"""The tabs of the one page: each is a FORM that writes a spec, and nothing more.

A tab module declares

    NAME, TITLE          the tab's name (also its URL, `/?tab=<NAME>`) and the heading
    PICK_DIR             where OPEN starts browsing
    CORPUS_MODE          which priming corpus a Claude session gets (`gui/corpus.py`)
    BRIEF                Claude's system prompt for driving this tab
    DEFAULT_FORM         the scene the tab opens on
    FORM_HTML, FORM_JS   the form's rows and the JS that reads/fills them (`tabForm`, `tabFill`, `tabInit`)
    build_spec(form)     -> spec dict, the way a person would write it
    form_from_spec(spec) -> form dict, so an opened spec refills the form
    normalise(spec)      optional: derived consequences of the form (widths, fractions) -- bio
    write_spec(spec, path) -> raw text: dump + whatever guard the pipeline would run (CFL)

Everything after the form -- seed, view, run, play, YAML, Claude -- is the shared panel in
`gui/app.py` over the shared routes, and the run is `plexus.pipeline.generate`, the body of
`Plexus_Main.py -o generate`.
"""
from __future__ import annotations

from importlib import import_module

ORDER = ("bio", "material", "neurons", "metabolism")


def get(name: str):
    if name not in ORDER:
        raise KeyError(f"no tab {name!r}; tabs are {', '.join(ORDER)}")
    return import_module(f"plexus.gui.tabs.{name}")


def write_spec(tab, spec: dict, path: str) -> str:
    """Write through the tab's own writer when it has one, else dump and run the CFL guard."""
    w = getattr(tab, "write_spec", None)
    if w is not None:
        return w(spec, path)
    import os
    from plexus.gui.server import _dump_yaml
    from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(_dump_yaml(spec))
    Courant_Friedrichs_Lewy_condition(path)
    return open(path).read()
