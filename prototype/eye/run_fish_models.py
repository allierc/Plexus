"""run_fish_models -- build and probe the four MEASURED plants, eye_F .. eye_I.

Models A-E were all the same guessed (mammalian) anatomy with the materials, the
strap length, a pulley and the drive amplitude varied in turn. F-I vary the thing
those five never touched: the anatomy itself, now measured off Fig. 12.1A of
Tulenko & Currie (2020) rather than assumed. One change per model, so each one
answers a question:

    F  larva      the plant as drawn at 96 hpf: obliques from the rostral orbit onto
                  the dorsal and ventral faces, SR/IR/MR from one caudal plate, LR
                  from outside the orbit onto the caudal sclera; globe flattened to
                  0.676 of its equator; per-muscle widths from the tracing.
                  -> what do the six actions become when the anatomy is a fish's?
    G  adult      F, but with Kasprick's adult insertions: all six migrated onto the
                  sclera-corneal junction, SO sharing SR's dorsal station and IO
                  sharing IR's ventral one.
                  -> what does that late insertion overlap buy the animal?
    H  pulley     F, plus a connective-tissue sleeve on LR, which is the one muscle
                  Kasprick describes as turning nearly 90 degrees before it reaches
                  the globe.
                  -> does the turn need a pulley to work?
    I  round      F, but with the OLD 0.82 flattening and nothing else changed.
                  -> how much of F is the new anatomy and how much is the new shape?

Each model is one baseline spec plus six open-loop step responses, one muscle at a
time (`muscle_probe`), which is the same protocol A-E were measured with, so the
numbers are comparable.

    python run_fish_models.py                    # all four, all six muscles
    python run_fish_models.py --models F --frames 120 --no-movie    # a smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np
import yaml

import plexus.operators            # noqa: F401
import eye_ops                     # noqa: F401
import muscle_ops                  # noqa: F401
import probe_plant
import eye_spec as ES
import fish_anatomy as FA

ARCHIVE = os.path.join(HERE, "archive")

MODELS = {
    "F": dict(short="measured larva",
              description="the 96 hpf plant as traced off Fig. 12.1A: obliques from the "
                          "rostral orbit onto the dorsal/ventral faces, SR+IR+MR from one "
                          "caudal plate, LR from outside the orbit onto the caudal sclera; "
                          "globe 0.676; per-muscle widths",
              kw=dict(plant="fish_larva", k_sleeve=0.0)),
    "G": dict(short="adult insertions",
              description="F with Kasprick's adult insertions -- all six on the "
                          "sclera-corneal junction, SO sharing SR's station and IO sharing "
                          "IR's (the overlap that appears late)",
              kw=dict(plant="fish_adult", k_sleeve=0.0)),
    "H": dict(short="LR pulley",
              description="F plus a muscle_sleeve, the connective-tissue pulley that would "
                          "hold LR against the globe through the near-90-degree turn "
                          "Kasprick describes",
              kw=dict(plant="fish_larva", k_sleeve=2500.0, c_sleeve=30.0,
                      sleeve_free=(0.70, 0.88))),
    "I": dict(short="round-globe control",
              description="F with the OLD 0.82 flattening and nothing else changed -- the "
                          "control that separates the new anatomy from the new globe shape",
              kw=dict(plant="fish_larva", k_sleeve=0.0, axial_ratio=0.82)),
}

# the materials and drive of model B/E, so that F-I differ from A-E in ANATOMY ONLY
BASE = dict(preset="probe", n_particles=45000, n_muscle_particles=2200, n_grid=112,
            dt=0.003, sclera_youngs=420.0, vitreous_youngs=45.0, choroid_youngs=130.0,
            muscle_youngs=240.0, contract=67.0, k_bone=9000.0, c_bone=60.0,
            k_socket=5000.0, k_fat=4000.0, c_fat=90.0, drag=5.0, muscle_drag=6.0,
            tonic=0.14, mus_arc=30.0, mus_gap=0.042, mus_embed=-0.013)


def build_baseline(label, outdir):
    m = MODELS[label]
    kw = dict(BASE)
    kw.update(m["kw"])
    spec = ES.build_spec(name=f"eye_{label}_{m['kw']['plant']}", **kw)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "baseline_spec.yaml")
    with open(path, "w") as fh:
        fh.write(f"# model {label} -- {m['short']}\n# {m['description']}\n")
        yaml.safe_dump(spec, fh, sort_keys=False, width=100)
    return spec, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--muscles", type=int, nargs="*", default=list(range(FA.N_MUSCLE)))
    ap.add_argument("--frames", type=int, default=320)
    ap.add_argument("--t_on", type=int, default=60)
    ap.add_argument("--t_off", type=int, default=240)
    ap.add_argument("--a_hi", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--no-movie", action="store_true")
    a = ap.parse_args()

    for label in a.models:
        m = MODELS[label]
        outdir = os.path.join(ARCHIVE, f"eye_{label}")
        spec, path = build_baseline(label, outdir)
        print(f"\n=== model {label}: {m['short']} -> {outdir}", flush=True)
        runs = []
        for mi in a.muscles:
            key = FA.MUSCLE_KEYS[mi]
            t0 = time.time()
            probe_plant.run_probe(spec, mi, a.device, a.a_hi, float(BASE["tonic"]),
                                  a.t_on, a.t_off, a.frames, outdir,
                                  stride=a.stride, movie=not a.no_movie)
            # The probe writes into its own scratch dir, `archive/tNN_probe_<KEY>/`.
            # Copy the results in beside the model as <MODEL>_<MUSCLE>_*: the tNN tag is
            # the scratch directory's name, not part of this model's identity, so it goes
            # in meta.json as provenance instead of into every filename.
            tdir = sorted(d for d in os.listdir(ARCHIVE)
                          if d.startswith("t") and d.endswith(f"probe_{key}"))[-1]
            src = os.path.join(ARCHIVE, tdir)
            for f in os.listdir(src):
                stem, ext = os.path.splitext(f)
                shutil.copy2(os.path.join(src, f),
                             os.path.join(outdir, f"{label}_{key}"
                                                  + ("" if stem == "movie" else f"_{stem}") + ext))
            os.replace(os.path.join(outdir, f"probe_{key}.npz"),
                       os.path.join(outdir, f"{label}_{key}_probe.npz"))
            runs.append(dict(muscle=key, scratch_dir=tdir,
                             seconds=round(time.time() - t0, 1)))
            print(f"[{label}] {key} done in {runs[-1]['seconds']}s", flush=True)
        with open(os.path.join(outdir, "meta.json"), "w") as fh:
            json.dump(dict(label=label, variant=f"eye_{m['kw']['plant']}", short=m["short"],
                           description=m["description"], plant=m["kw"]["plant"],
                           anatomy="measured off Fig. 12.1A (Tulenko & Currie 2020, "
                                   "after Easter & Nicola 1996) + Kasprick et al. 2011",
                           params=m["kw"], runs=runs), fh, indent=2)
        print(f"[{label}] archived -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
