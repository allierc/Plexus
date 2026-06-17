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
        ("boids_16", f"{GD}/boids/boids_16/movie_particle.mp4",
         "config/boids/boids_16.yaml",
         "16 types of boid; cohesion/alignment/separation (reproduces PDE_B)"),
    ],
    "Slime mould — Physarum networks": [
        ("slime_default", "prototype/slime/slime_default.gif", "prototype/slime/specs/slime_default.yaml", "self-reinforcing transport network"),
        ("slime_curly", "prototype/slime/slime_curly.gif", "prototype/slime/specs/slime_curly.yaml", "high turning rate, curly filaments"),
        ("slime_filaments", "prototype/slime/slime_filaments.gif", "prototype/slime/specs/slime_filaments.yaml", "fine filamentary structure"),
        ("slime_eight", "prototype/slime/slime_eight.gif", "prototype/slime/specs/slime_eight.yaml", "eight sources"),
    ],
    "MPM fluids & materials": [
        ("mat_liquid", "prototype/water/mat_liquid.gif", "prototype/scenarios/mat_liquid.yaml", "liquid material (no shear memory)"),
        ("mat_elastic", "prototype/water/mat_elastic.gif", "prototype/scenarios/mat_elastic.yaml", "elastic material (shape memory)"),
        ("mat_snow", "prototype/water/mat_snow.gif", "prototype/scenarios/mat_snow.yaml", "snow / granular material"),
        ("ph_crown_splash", "prototype/water/ph_crown_splash.gif", "prototype/scenarios/ph_crown_splash.yaml", "a drop hits a pool: crown splash"),
        ("ph_coalesce", "prototype/water/ph_coalesce.gif", "prototype/scenarios/ph_coalesce.yaml", "two drops coalesce under surface tension"),
        ("ph_slosh", "prototype/water/ph_slosh.gif", "prototype/scenarios/ph_slosh.yaml", "sloshing in a vessel"),
        ("ob_dam_break", "prototype/water/ob_dam_break.gif", "prototype/scenarios/ob_dam_break.yaml", "dam break against an obstacle"),
        ("ob_funnel", "prototype/water/ob_funnel.gif", "prototype/scenarios/ob_funnel.yaml", "draining through a funnel"),
        ("water_bowl_1", "prototype/water/water_bowl_1.gif", "prototype/scenarios/water_bowl_1.yaml", "pouring water into a bowl"),
    ],
    "The Well — single mechanisms vs. mixtures": [
        ("rd_worms", "prototype/well/rd_worms.gif", "prototype/well/scenarios/rd_worms.yaml", "single field: Gray–Scott reaction–diffusion"),
        ("wave_lens", "prototype/well/wave_lens.gif", "prototype/well/scenarios/wave_lens.yaml", "single field: acoustic waves through a lens"),
        ("am_swirl", "prototype/well/am_swirl.gif", "prototype/well/scenarios/am_swirl.yaml", "single set: Vicsek active matter, a swirl"),
        ("mix_world", "prototype/well/mix_world.gif", "prototype/well/scenarios/mix_world.yaml", "a mixture: field + active set in one world"),
        ("mix_taxis_spots", "prototype/well/mix_taxis_spots.gif", "prototype/well/scenarios/mix_taxis_spots.yaml", "a mixture: chemotaxis toward emergent spots"),
    ],
    "Microswimmers": [
        ("motile", "prototype/microswimmer/motile.gif", "prototype/microswimmer/motile.yaml", "active self-propelled swimmers"),
        ("slipwave_motile", "prototype/microswimmer/slipwave_motile.gif", "prototype/microswimmer/motile.yaml", "motile swimmer — slip-wave field overlay"),
        ("vorticity_motile", "prototype/microswimmer/vorticity_motile.gif", "prototype/microswimmer/motile.yaml", "motile swimmer — vorticity field overlay"),
        ("feeding", "prototype/microswimmer/feeding.gif", "prototype/microswimmer/feeding.yaml", "feeding on a resource field"),
        ("sessile", "prototype/microswimmer/sessile.gif", "prototype/microswimmer/sessile.yaml", "attached, beating cilia"),
    ],
}

# Four representatives for the "What it looks like" hero strip: maximally different
# collective behaviours, each a different operator family. (name, hero caption)
HERO: list[tuple[str, str]] = [
    ("boids_16", "flocking — collective motion from local rules"),
    ("slime_filaments", "a self-organizing Physarum transport network"),
    ("ph_crown_splash", "an MPM fluid: a drop crowns on impact"),
    ("rd_worms", "a Gray–Scott reaction–diffusion field"),
]


def transcode(src: str, dst: str) -> bool:
    """GIF/mp4 -> small looping mp4 (≤460px wide, 20 fps, H.264)."""
    if not os.path.exists(src):
        print(f"  skip (missing): {src}")
        return False
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", src, "-an",
           "-movflags", "+faststart", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "30",
           "-vf", "fps=20,scale='min(460,iw)':-2:flags=lanczos", dst]
    subprocess.run(cmd, check=True)
    return True


def card(name: str, mp4: str, spec_path: str, caption: str) -> str:
    spec = html.escape(open(spec_path).read().rstrip()) if os.path.exists(spec_path) else "(spec not found)"
    return f"""  <figure class="sim-card">
    <video src="{mp4}" autoplay loop muted playsinline preload="metadata"></video>
    <figcaption>
      <span class="sim-name" tabindex="0">{name}<span class="sim-spec"><pre>{spec}</pre></span></span>
      <span class="sim-cap">{html.escape(caption)}</span>
    </figcaption>
  </figure>"""


STYLE = """<style>
.sim-gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:1.1rem;margin:1rem 0 2rem}
.sim-hero{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:1.3rem;margin:1.2rem 0 2.5rem}
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

## What it looks like

Qualitatively different collective behaviours, all produced from the same code by
changing only a ~20-line specification — no new software per phenomenon.
"""

INTRO = """
## Browse the families

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

    # one lookup of spec paths so the hero can reuse already-transcoded clips
    spec_of = {name: spec for sims in CURATED.values() for name, _, spec, _ in sims}

    blocks = [STYLE]
    for family, sims in CURATED.items():
        cards = []
        for name, src, spec, cap in sims:
            dst = f"{OUT}/{name}.mp4"
            if transcode(src, dst):
                cards.append(card(name, dst, spec, cap))
                print(f"  ok: {name}  ({os.path.getsize(dst)//1024} KB)")
        if cards:
            blocks.append(f"<h2>{html.escape(family)}</h2>\n<div class=\"sim-gallery\">\n" +
                          "\n".join(cards) + "\n</div>")

    # hero strip of four representatives (clips already transcoded above)
    hero_cards = [card(name, f"{OUT}/{name}.mp4", spec_of[name], cap) for name, cap in HERO]
    hero = "<div class=\"sim-hero\">\n" + "\n".join(hero_cards) + "\n</div>"

    body = (HEADER + "\n```{=html}\n" + STYLE + "\n" + hero + "\n```\n" +
            INTRO + "\n```{=html}\n" + "\n\n".join(blocks[1:]) + "\n```\n")
    open("experiment.qmd", "w").write(body)
    print(f"wrote experiment.qmd and {len(os.listdir(OUT))} clips in {OUT}/")


if __name__ == "__main__":
    main()
