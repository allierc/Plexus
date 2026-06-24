#!/usr/bin/env python
"""Generate the Simulations gallery page (experiment.qmd) for the Quarto minisite.

For each curated simulation it (1) transcodes the prototype render (a large GIF, or
an mp4 from graphs_data) into a small, web-friendly looping mp4 under gallery/, and
(2) reads the simulation's spec.yaml. It then emits experiment.qmd as a grid of
cards: each card autoplays its clip, and hovering (or focusing) the title reveals
the full spec.yaml in a popover.

Run:  python scripts/make_gallery.py     (from the repo root)

The gallery/ mp4s live outside prototype/ (which is gitignored for media), so they
ship with the site; Quarto copies them into docs/gallery/ on render.
"""
from __future__ import annotations
import html
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
FFMPEG = "/workspace/.conda_envs/neural-graph-linux/bin/ffmpeg"
GD = "/groups/saalfeld/home/allierc/GraphData/graphs_data"
OUT = "gallery"

# family -> [(name, source video (gif/mp4), spec.yaml, caption)]
CURATED: dict[str, list[tuple[str, str, str, str]]] = {
    "Attraction–repulsion & boids — validated against ParticleGraph": [
        ("arbitrary_3", f"{GD}/attraction_repulsion/arbitrary_3/movie_particle.mp4",
         "config/attraction_repulsion/arbitrary_3.yaml",
         "3 types, 30k particles; per-type difference-of-Gaussians law (reproduces PDE_A)"),
        ("arbitrary_3d", f"{GD}/attraction_repulsion/arbitrary_3d/movie_particle.mp4",
         "config/attraction_repulsion/arbitrary_3d.yaml",
         "the same 3-type force law in 3D — the dimension contract (dim: 3, periodic box)"),
        ("boids_16", f"{GD}/boids/boids_16/movie_particle.mp4",
         "config/boids/boids_16.yaml",
         "16 types of boid; cohesion/alignment/separation (reproduces PDE_B)"),
    ],
    "Slime mould — Physarum networks": [
        ("slime_default", "prototype/slime/slime_default.gif", "prototype/slime/specs/slime_default.yaml", "self-reinforcing transport network"),
        ("slime_curly", "prototype/slime/slime_curly.gif", "prototype/slime/specs/slime_curly.yaml", "high turning rate, curly filaments"),
        ("slime_filaments", "prototype/slime/slime_filaments.gif", "prototype/slime/specs/slime_filaments.yaml", "fine filamentary structure"),
        ("slime_labyrinth", "prototype/slime/slime_labyrinth.gif", "prototype/slime/specs/slime_labyrinth.yaml", "uniform seed + high decay → a persistent space-filling labyrinth"),
        ("slime_fine_graph", "prototype/slime/slime_fine_graph.gif", "prototype/slime/specs/slime_fine.yaml", "fine network, extracted as a graph"),
        ("slime_two_repel", "prototype/slime/slime_two_repel.gif", "prototype/slime/specs/slime_two_repel.yaml", "two species repel: separate territories, sharp interface"),
        ("slime_six", "prototype/slime/slime_six.gif", "prototype/slime/specs/slime_six.yaml", "six sources"),
        ("slime_eight", "prototype/slime/slime_eight.gif", "prototype/slime/specs/slime_eight.yaml", "eight sources"),
        ("slime_torus", "prototype/slime/slime_torus.gif", "prototype/slime/specs/slime_torus.yaml", "periodic (toroidal) domain"),
    ],
    "MPM fluids & materials": [
        ("mat_liquid_1", "prototype/water/mat_liquid_1.gif", "prototype/scenarios/mat_liquid_1.yaml", "a big water ball drops, splashes, and settles into a pool"),
        ("mat_elastic", "prototype/water/mat_elastic.gif", "prototype/scenarios/mat_elastic.yaml", "elastic material (shape memory)"),
        ("mat_snow", "prototype/water/mat_snow.gif", "prototype/scenarios/mat_snow.yaml", "snow / granular material"),
        ("ph_crown_splash", "prototype/water/ph_crown_splash.gif", "prototype/scenarios/ph_crown_splash.yaml", "a drop hits a pool: crown splash"),
        ("ob_zigzag", "prototype/water/ob_zigzag.gif", "prototype/scenarios/ob_zigzag.yaml", "water cascades down a zig-zag channel"),
        ("ph_slosh", "prototype/water/ph_slosh.gif", "prototype/scenarios/ph_slosh.yaml", "sloshing in a vessel"),
        ("ob_dam_break", "prototype/water/ob_dam_break.gif", "prototype/scenarios/ob_dam_break.yaml", "dam break against an obstacle"),
        ("ob_funnel", "prototype/water/ob_funnel.gif", "prototype/scenarios/ob_funnel.yaml", "draining through a funnel"),
        ("ob_wedge", "prototype/water/ob_wedge.gif", "prototype/scenarios/ob_wedge.yaml", "a falling stream splits around a wedge"),
    ],
    "The Well — fields & active matter": [
        ("am_flock", "prototype/well/am_flock.gif", "prototype/well/scenarios/am_flock.yaml", "active matter: a coherent polar flock (Vicsek)"),
        ("rd_worms", "prototype/well/rd_worms.gif", "prototype/well/scenarios/rd_worms.yaml", "field: Gray–Scott reaction–diffusion, worms"),
        ("wave_double_slit", "prototype/well/wave_double_slit.gif", "prototype/well/scenarios/wave_double_slit.yaml", "field: acoustic double-slit diffraction"),
        ("wave_maze", "prototype/well/wave_maze.gif", "prototype/well/scenarios/wave_maze.yaml", "field: acoustic propagation through a maze of sound-hard corridors"),
        ("wave_lens", "prototype/well/wave_lens.gif", "prototype/well/scenarios/wave_lens.yaml", "field: acoustic waves focused by a lens"),
        ("ns_rayleigh_benard", "prototype/well/ns_rayleigh_benard.gif", "prototype/well/scenarios/ns_rayleigh_benard.yaml", "field: Rayleigh–Bénard convection rolls (Navier–Stokes)"),
        ("ns_vortices", "prototype/well/ns_vortices.gif", "prototype/well/scenarios/ns_vortices.yaml", "field: decaying 2-D turbulence, vortices merge (Navier–Stokes)"),
        ("ns_taylor_green", "prototype/well/ns_taylor_green.gif", "prototype/well/scenarios/ns_taylor_green.yaml", "field: Taylor–Green vortex decay (Navier–Stokes)"),
        ("ns_rayleigh_taylor", "prototype/well/ns_rayleigh_taylor.gif", "prototype/well/scenarios/ns_rayleigh_taylor.yaml", "field: Rayleigh–Taylor fingering instability (Navier–Stokes)"),
    ],
    "Reaction–diffusion — cyclic competition (rock–paper–scissors)": [
        ("rps_random", "prototype/rps/rps_random.gif", "prototype/rps/scenarios/rps_random.yaml", "3 species quenched from disorder → spiral turbulence"),
        ("rps_species_4", "prototype/rps/rps_species_4.gif", "prototype/rps/scenarios/rps_species_4.yaml", "4-species cyclic competition"),
        ("rps_species_6", "prototype/rps/rps_species_6.gif", "prototype/rps/scenarios/rps_species_6.yaml", "6-species cyclic competition (validated vs ParticleGraph RD_RPS)"),
    ],
    "Mixtures — several mechanisms in one world": [
        ("mix_avoid", "prototype/well/mix_avoid.gif", "prototype/well/scenarios/mix_avoid.yaml", "particles flee a reaction–diffusion ridge field"),
        ("mix_slime4", "prototype/well/mix_slime4.gif", "prototype/well/scenarios/mix_slime4.yaml", "four slime species run two coupled networks"),
        ("mix_taxis_maze_A", "prototype/well/mix_taxis_maze_A.gif", "prototype/well/scenarios/mix_taxis_maze.yaml", "particles trace a Gray–Scott maze morphogen"),
        ("mix_swarm_flow", "prototype/well/mix_swarm_flow.gif", "prototype/well/scenarios/mix_swarm_flow.yaml", "a Vicsek swarm swept by a turbulent flow"),
        ("mix_rd_flow", "prototype/well/mix_rd_flow.gif", "prototype/well/scenarios/mix_rd_flow.yaml", "a reaction–diffusion pattern stirred by turbulence"),
        ("mix_acoustic_push", "prototype/well/mix_acoustic_push.gif", "prototype/well/scenarios/mix_acoustic_push.yaml", "expanding acoustic wavefronts herd a swarm"),
    ],
    "Building a microswimmers simulation": [
        ("motile", "prototype/microswimmer/motile.gif", "prototype/microswimmer/motile.yaml", "active self-propelled swimmers"),
        ("slipwave_motile", "prototype/microswimmer/slipwave_motile.gif", "prototype/microswimmer/motile.yaml", "motile swimmer — slip-wave field overlay"),
        ("vorticity_motile", "prototype/microswimmer/vorticity_motile.gif", "prototype/microswimmer/motile.yaml", "motile swimmer — vorticity field overlay"),
        ("vortzoom_fast", "prototype/microswimmer/vortzoom_fast.gif", "prototype/microswimmer/motile.yaml", "motile swimmer — vorticity close-up"),
        ("feeding", "prototype/microswimmer/feeding.gif", "prototype/microswimmer/feeding.yaml", "feeding on a resource field"),
        ("sessile", "prototype/microswimmer/sessile.gif", "prototype/microswimmer/sessile.yaml", "attached, beating cilia"),
    ],
}


# clips whose motion reads too fast -> stretch playback time by this factor
SLOW: dict[str, float] = {
    "ns_rayleigh_benard": 2.0,
    "ns_rayleigh_taylor": 2.0,
}


def transcode(src: str, dst: str, slow: float = 1.0) -> bool:
    """GIF/mp4 -> small looping mp4 (≤460px wide, 20 fps, H.264). `slow`>1 stretches
    playback time (2.0 = half speed) via setpts before the fps resample."""
    if not os.path.exists(src):
        print(f"  skip (missing): {src}")
        return False
    vf = (f"setpts={slow}*PTS," if slow != 1.0 else "") + "fps=20,scale='min(460,iw)':-2:flags=lanczos"
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", src, "-an",
           "-movflags", "+faststart", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "30",
           "-vf", vf, dst]
    subprocess.run(cmd, check=True)
    return True


def card(name: str, mp4: str, spec_path: str, caption: str) -> str:
    if os.path.exists(spec_path):
        raw = open(spec_path).read()
        for marker in ("\n# --- auto: video descriptions", "\ndescriptions:"):  # drop VLM blurb
            i = raw.find(marker)
            if i != -1:
                raw = raw[:i]
        spec = html.escape(raw.rstrip())
    else:
        spec = "(spec not found)"
    return f"""  <figure class="sim-card">
    <video src="{mp4}" autoplay loop muted playsinline preload="metadata"></video>
    <figcaption>
      <span class="sim-name" tabindex="0">{name}<span class="sim-spec"><pre>{spec}</pre></span></span>
      <span class="sim-cap">{html.escape(caption)}</span>
    </figcaption>
  </figure>"""


STYLE = """<style>
.sim-gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:1.1rem;margin:1rem 0 2rem}
.sim-card{margin:0}
.sim-card video{width:100%;border-radius:7px;background:#000;display:block;aspect-ratio:1/1;object-fit:cover}
.sim-card figcaption{margin-top:.35rem;line-height:1.25}
.sim-name{font-weight:600;cursor:help;position:relative;border-bottom:1px dotted #999}
.sim-cap{display:block;font-size:.84em;color:#777;margin-top:.1rem}
.sim-spec{display:none;position:absolute;left:0;top:1.5em;z-index:40;width:max(300px,108%);max-height:360px;overflow:auto;
  background:#1e1e1e;border-radius:7px;box-shadow:0 8px 28px rgba(0,0,0,.4);padding:.5rem .7rem;text-align:left;cursor:auto}
.sim-name:hover .sim-spec,.sim-name:focus .sim-spec{display:block}
.sim-spec pre{margin:0;font-size:.72em;line-height:1.3;white-space:pre;color:#e9e9e9;background:none;border:none;padding:0}
</style>"""

HEADER = """---
title: "A spread of simulations"
subtitle: "One engine, many living systems — hover a title to read its spec"
resources:
  - gallery/
---

::: {.callout-note appearance="simple"}
**Operators adapted from prior work.** Plexus re-implements published simulation code
as registered operators: attraction–repulsion, boids, and rock–paper–scissors
reaction–diffusion from [ParticleGraph](https://github.com/allierc/ParticleGraph);
the slime-mould (Physarum)
agent from [Sebastian Lague's Slime-Simulation](https://github.com/SebLague/Slime-Simulation);
the field PDEs — reaction–diffusion, acoustics, Navier–Stokes, and active matter — from
[*The Well*](https://github.com/PolymathicAI/the_well) (Ohana et al., NeurIPS 2024); and
the ciliate microswimmer from
[Liu, Costello & Kanso](https://github.com/jingyiliu1900/Flow-Physics-drives-functional-design-of-microswimmers)
(*Nat. Commun.* 2025).
:::

Each clip below is one `spec.yaml` over the identical operators and schedule grammar;
**hover (or tab to) a simulation's title to read the exact spec that generated it.**
The first two are validated to floating-point precision against
[ParticleGraph](https://github.com/allierc/ParticleGraph).

::: {.callout-note collapse="true"}
## How a clip is made
`Plexus_Main.py -o generate_plot <config>` runs the forward simulation, writes the
trajectory, and renders the movie — no per-simulation code, only the spec differs.
:::
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):                       # clean slate: drop clips of de-curated sims
        if f.endswith(".mp4"):
            os.remove(os.path.join(OUT, f))

    blocks = []
    for family, sims in CURATED.items():
        cards = []
        for name, src, spec, cap in sims:
            dst = f"{OUT}/{name}.mp4"
            if transcode(src, dst, SLOW.get(name, 1.0)):
                cards.append(card(name, dst, spec, cap))
                print(f"  ok: {name}  ({os.path.getsize(dst)//1024} KB)")
        if cards:
            blocks.append(f"<h2>{html.escape(family)}</h2>\n<div class=\"sim-gallery\">\n" +
                          "\n".join(cards) + "\n</div>")

    body = HEADER + "\n```{=html}\n" + STYLE + "\n" + "\n\n".join(blocks) + "\n```\n"
    open("experiment.qmd", "w").write(body)
    print(f"wrote experiment.qmd and {len(os.listdir(OUT))} clips in {OUT}/")


if __name__ == "__main__":
    main()
