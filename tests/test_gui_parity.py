"""The web page is the UI of Plexus_Main.py, held to it by two gates.

WHY THESE TWO. The material page ran the same three bodies at 202 ms/frame where the command line
ran 68, and the cause was not the engine: the page built its own spec from a template (the
template's substep, 127 a frame where 19 would do; the template's wall model) and ran its own loop
around `engine.run` (no movie, no `graphs_data/`). See `notes/campaigns/GUI_UNIFY.md`.

  1. SAME YAML: the page's default form, written out, IS `config/si_material/si_three_balls.yaml`.
     The benchmark everyone quotes and the scene the page opens on are one file.
  2. SAME ENGINE: `Plexus_Main.py -o generate` and the page's RUN both call
     `plexus.pipeline.generate`; the CLI has no second body, and the page's run is a `studio.Job`,
     which runs `Plexus_Main.main()` itself. Held here by inspection of the entry points, because a
     third implementation is exactly what this test exists to refuse.
"""
from __future__ import annotations

import ast
import os

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _strip_name(spec: dict) -> dict:
    spec = dict(spec)
    spec["general"] = {k: v for k, v in spec["general"].items() if k != "name"}
    return spec


def test_form_writes_the_reference_spec(tmp_path):
    from plexus.gui import material
    spec = material.build_spec(material.DEFAULT_FORM)
    out = tmp_path / "from_form.yaml"
    material.write_spec(spec, str(out))                    # dump + the pipeline's CFL guard
    got = yaml.safe_load(open(out))
    ref = yaml.safe_load(open(material.REFERENCE))
    assert _strip_name(got) == _strip_name(ref), "the form no longer writes si_three_balls.yaml"


def test_form_round_trips_through_the_spec():
    """A spec opened from disk refills the form with what wrote it (else OPEN + BUILD drifts)."""
    from plexus.gui import material
    spec = material.build_spec(material.DEFAULT_FORM)
    f = material.form_from_spec(spec)
    again = material.build_spec(f)
    assert again == spec


def test_bodies_keep_their_order():
    """`type_layout: ordered`: the i-th body is at the i-th centre, whatever the seed."""
    from plexus.gui import material
    spec = material.build_spec(material.DEFAULT_FORM)
    assert spec["sets"]["cell"]["type_layout"] == "ordered"
    assert list(spec["sets"]["cell"]["types"]) == [b["name"] for b in material.DEFAULT_FORM["bodies"]]


def test_cli_and_page_share_one_pipeline():
    """`Plexus_Main.main` calls `plexus.pipeline.generate` and nothing else runs a generate."""
    src = open(os.path.join(REPO, "Plexus_Main.py")).read()
    tree = ast.parse(src)
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "generate" in calls and "data_generate" not in calls
    assert "from plexus.pipeline import generate" in src
    # the page's RUN is the same function, on the VTK thread, with the hook that feeds the view
    view = open(os.path.join(REPO, "src", "plexus", "gui", "bio_view.py")).read()
    assert "pipeline.generate(self.spec_path" in view and "engine.run(" not in view
    # and the studio's worker path, when used, is the CLI's own main()
    job = open(os.path.join(REPO, "src", "plexus", "gui", "studio.py")).read()
    assert '["-o", "generate", f"studio/{name}"' in job
    worker = open(os.path.join(REPO, "src", "plexus", "gui", "worker.py")).read()
    assert "Plexus_Main.main()" in worker


@pytest.mark.parametrize("bad", [{"bodies": []}, {"bodies": [{"name": "a"}, {"name": "a"}]}])
def test_the_form_refuses_what_the_engine_would_misread(bad):
    from plexus.gui import material
    form = dict(material.DEFAULT_FORM, **bad)
    with pytest.raises(ValueError):
        material.build_spec(form)
