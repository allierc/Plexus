"""The web page is the UI of Plexus_Main.py, held to it by three gates.

WHY THESE. The material page ran the same three bodies at 202 ms/frame where the command line
ran 68, and the cause was not the engine: the page built its own spec from a template (the
template's substep, 127 a frame where 19 would do; the template's wall model) and ran its own loop
around `engine.run` (no movie, no `graphs_data/`). See `notes/campaigns/GUI_UNIFY.md`.

  1. SAME YAML: every tab's default form, written out, IS its reference spec under `config/`
     (`si_material/si_three_balls.yaml`, `neural/ctrnn_gui.yaml`, `metabolism/massaction_toy.yaml`).
     The benchmark everyone quotes and the scene the page opens on are one file.
  2. SAME ENGINE: `Plexus_Main.py -o generate` and the page's RUN both call
     `plexus.pipeline.generate`; the CLI has no second body and the page has no `engine.run`.
     Held by inspection of the entry points, because a third implementation is exactly what this
     test exists to refuse.
  3. ONE PAGE: every tab renders through the one shell with the form's three JS hooks, and the
     route table carries the shared panel once.
"""
from __future__ import annotations

import ast
import os

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABS_WITH_REFERENCE = ("material", "neurons", "metabolism")


def _strip_name(spec: dict) -> dict:
    spec = dict(spec)
    spec["general"] = {k: v for k, v in spec["general"].items() if k != "name"}
    return spec


@pytest.mark.parametrize("name", TABS_WITH_REFERENCE)
def test_form_writes_the_reference_spec(name, tmp_path):
    from plexus.gui import tabs
    T = tabs.get(name)
    spec = T.build_spec(T.DEFAULT_FORM)
    out = tmp_path / f"{name}.yaml"
    tabs.write_spec(T, spec, str(out))                    # dump + the pipeline's guard (CFL on MPM)
    got = yaml.safe_load(open(out))
    ref = yaml.safe_load(open(T.REFERENCE))
    assert _strip_name(got) == _strip_name(ref), f"the {name} form no longer writes {T.REFERENCE}"


def test_material_form_round_trips_through_the_spec():
    """A spec opened from disk refills the form with what wrote it (else OPEN + BUILD drifts)."""
    from plexus.gui.tabs import material
    spec = material.build_spec(material.DEFAULT_FORM)
    assert material.build_spec(material.form_from_spec(spec)) == spec


def test_bodies_keep_their_order():
    """`type_layout: ordered`: the i-th body is at the i-th centre, whatever the seed."""
    from plexus.gui.tabs import material
    spec = material.build_spec(material.DEFAULT_FORM)
    assert spec["sets"]["cell"]["type_layout"] == "ordered"
    assert list(spec["sets"]["cell"]["types"]) == [b["name"] for b in material.DEFAULT_FORM["bodies"]]


def test_cli_and_page_share_one_pipeline():
    """`Plexus_Main.main` calls `plexus.pipeline.generate`; the page's RUN calls the same function
    and nothing in the GUI runs `engine.run` on its own."""
    src = open(os.path.join(REPO, "Plexus_Main.py")).read()
    tree = ast.parse(src)
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "generate" in calls and "data_generate" not in calls
    assert "from plexus.pipeline import generate" in src
    gui = os.path.join(REPO, "src", "plexus", "gui")
    view = open(os.path.join(gui, "bio_view.py")).read()
    assert "pipeline.generate(self.spec_path" in view
    for fn in os.listdir(gui):
        if fn.endswith(".py"):
            assert "engine.run(" not in open(os.path.join(gui, fn)).read(), f"{fn} runs the engine on its own"
    assert not os.path.exists(os.path.join(gui, "worker.py")), "the worker process is retired"


@pytest.mark.parametrize("name", ("bio", "material", "neurons", "metabolism"))
def test_every_tab_renders_through_the_one_shell(name):
    from plexus.gui import app, tabs
    html = app.page(name)
    for hook in ("window.tabForm", "window.tabFill", "window.tabInit"):
        assert hook in html, f"{name}: the form must define {hook}"
    assert f"const TAB={name!r}".replace("'", '"') in html
    for other in tabs.ORDER:                                     # the tab bar names every tab
        assert f"switchTab('{other}')" in html


def test_the_route_table_carries_the_shared_panel_once():
    from plexus.gui import server
    for x in ("state", "spec", "seed", "run", "frames", "artefacts", "ls", "open", "view", "claude"):
        assert f"/api/scene/{x}" in server.GET_ROUTES, x
    for x in ("reset", "run", "save", "refine", "visible", "claude"):
        assert f"/api/scene/{x}" in server.POST_ROUTES, x
    assert server.GET_ROUTES["/api/bio/state"] is server.GET_ROUTES["/api/scene/state"]


@pytest.mark.parametrize("bad", [{"bodies": []}, {"bodies": [{"name": "a"}, {"name": "a"}]}])
def test_the_material_form_refuses_what_the_engine_would_misread(bad):
    from plexus.gui.tabs import material
    form = dict(material.DEFAULT_FORM, **bad)
    with pytest.raises(ValueError):
        material.build_spec(form)


def test_metabolism_operators_conserve_what_the_stoichiometry_says():
    """One reaction A + B -> 2B on the CPU: v = k a b, da/dt = -v, db/dt = +v; the flux is the
    stoichiometric sum and the rate the mass-action product, to the tolerance of float32."""
    import torch
    from plexus import engine, schema
    from plexus.gui.tabs import metabolism as M
    form = dict(M.DEFAULT_FORM, n_metabolites=2, n_reactions=1, max_per_reaction=1, cycle_fraction=0.0,
                homeostasis=0.0, k_min=0.2, k_max=0.2, seed=1, name="two_species")
    spec = M.build_spec(form)
    spec["sets"]["stoich"]["edges"] = [[0, 0], [1, 0]]
    spec["sets"]["stoich"]["weights"] = [-1.0, 1.0]
    spec["operators"][0].pop("flux_limit_dt", None)
    path = os.path.join(os.path.dirname(__file__), "_two_species.yaml")
    try:
        with open(path, "w") as f:
            yaml.safe_dump(spec, f, sort_keys=False)
        sim = schema.load(path)
        H = engine.build(sim, "cpu"); engine.seed(H, sim, "cpu")
        import plexus.operators.metabolism as ops
        c = H.level("metabolite").get("conc").clone()
        rate = ops.ReactionRate({"_at": "reaction", "edge_set": "stoich"})
        flux = ops.MetaboliteFlux({"_at": "metabolite", "edge_set": "stoich"})
        rate.forward(H)
        v = H.level("reaction").get("v")
        assert torch.allclose(v[0, 0], 0.2 * c[0, 0], rtol=1e-4)      # only A is a substrate
        d = flux.forward(H)["metabolite"]
        assert torch.allclose(d[0, 0], -v[0, 0]) and torch.allclose(d[1, 0], v[0, 0])
    finally:
        if os.path.exists(path):
            os.remove(path)


# ----------------------------------------------------------------- the circuit panel
def test_dale_weights_take_the_presynaptic_sign():
    """Under `dale: 1` every synapse's sign is its presynaptic neuron's, E first then I inside
    every assembly block (`type_layout: ordered`), and the I weights carry the n_E/n_I balance."""
    from plexus.gui.tabs import neurons as N
    spec = N.build_spec(N.DEFAULT_FORM)
    types = spec["sets"]["neuron"]["types"]
    per = spec["sets"]["neuron"]["per_parent"]
    n_e = sum(t["count"] for nm, t in types.items() if t["sign"] == "E")
    for (pre, post), w in zip(spec["sets"]["synapse"]["edges"], spec["sets"]["synapse"]["weights"]):
        assert (w > 0) == ((pre % per) < n_e), (pre, w)


def test_the_panel_draws_a_tiny_circuit(tmp_path):
    """A 2-assembly Dale circuit on the CPU: the panel captures ticks and renders an RGB frame;
    the movie and a still are written by the pipeline's own hook contract."""
    import numpy as np
    from plexus import engine, schema
    from plexus.gui.tabs import neurons as N, write_spec
    from plexus.neural_panel import NeuralPanel
    form = dict(N.DEFAULT_FORM, n_assemblies=2, per_assembly=6, n_frames=12, movie_frames=6, stills=1, name="tiny")
    path = str(tmp_path / "tiny.yaml")
    write_spec(N, N.build_spec(form), path)
    sim = schema.load(path)
    H = engine.build(sim, "cpu"); engine.seed(H, sim, "cpu")
    panel = NeuralPanel(out=str(tmp_path / "movie.mp4"), n_frames=12, sim=sim, style=sim.plotting,
                        max_frames=6, stills=1, keep_stills=True, name="tiny")
    engine.run(sim, out_path=None, device="cpu", on_frame=panel)
    panel.close()
    assert panel.failed is None and panel.rendered == 6
    img = panel.frame_at(len(panel.hist) - 1)
    assert isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3
    assert (tmp_path / "movie.mp4").exists() and (tmp_path / "3d.png").exists()
    assert panel.N == 12 and panel.dale and [b[0] for b in panel.blocks] == ["E_aff", "E", "I"]
