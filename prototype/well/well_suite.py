"""Exhaustive test suite for the Well-in-Plexus prototype.

Runs every scenario, computes a per-family intent metric ("it ran" is not "it
worked"), renders a gif + montage for each, and assembles a gallery + results.md +
run.log -- the same discipline the water suite (overnight_suite.py) used.

    PYTHONPATH=/workspace/Plexus/src python well_suite.py [--quick]

Families and their intent checks:
  rd_*    Gray-Scott   -> pattern contrast (spatial std of A) must exceed the
                          near-uniform initial transient -> a Turing pattern formed.
  wave_*  acoustic     -> finite, bounded pressure (no blow-up); report energy decay
                          (open BC radiates) or nnear-conservation (closed maze).
  am_*    active matter-> polar order phi: ordered regimes rise toward 1, the
                          disordered regime stays low -> the Vicsek transition.
"""
import os, sys, glob, time, traceback
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import well_schema, well_engine
import render_well, render_am

OUT = os.path.dirname(os.path.abspath(__file__))


def _field_metrics(name, out):
    fn = next(iter(out["fields"]))
    F = out["fields"][fn]
    if name.startswith("rd"):
        A = F[:, 0]
        contrast = float(A[-1].std())                       # pattern contrast (0 = uniform)
        bfrac = float((F[-1, 1] > 0.2).mean())              # area covered by species B
        return dict(metric="contrast", value=round(contrast, 4),
                    extra=f"B_area={bfrac:.3f}", ok=contrast > 0.05)
    else:  # wave
        P = F[:, 0]
        amp = float(np.abs(P).max())
        e0 = float((P[0] ** 2).mean()); e1 = float((P[-1] ** 2).mean())
        finite = np.isfinite(P).all() and amp < 50.0
        ratio = e1 / max(e0, 1e-9)
        return dict(metric="max_amp", value=round(amp, 3),
                    extra=f"E_end/E_0={ratio:.3f}", ok=bool(finite))


def _set_metrics(name, out):
    S = out["set"]; theta = S[:, :, 2]
    phi = np.abs(np.exp(1j * theta).mean(axis=1))
    ordered = "disorder" not in name
    ok = (phi[-1] > 0.6) if ordered else (phi[-1] < 0.7)
    return dict(metric="order_phi", value=round(float(phi[-1]), 3),
                extra=f"phi_0={phi[0]:.2f}", ok=bool(ok))


def run_one(name, log):
    t = time.time()
    sc = well_schema.load(f"scenarios/{name}.yaml")
    is_set = len(sc.sets) > 0
    if is_set:
        out, _ = render_am.render(name, out_dir=OUT)
        m = _set_metrics(name, out)
        n_part = out["set"].shape[1]; info = f"{n_part} particles"
    else:
        out = render_well.render(name, out_dir=OUT)
        m = _field_metrics(name, out)
        fn = next(iter(out["fields"])); info = f"{out['fields'][fn].shape[2]}x{out['fields'][fn].shape[3]} grid"
    T = (out["set"].shape[0] if is_set else next(iter(out["fields"].values())).shape[0])
    rec = dict(name=name, frames=T, size=info, dt=round(time.time() - t, 1), **m)
    log(f"    done: {rec}")
    return rec


def gallery(names, path):
    """Stack each scenario's montage into one tall gallery image."""
    rows = []
    for nm in names:
        p = os.path.join(OUT, nm + "_montage.png")
        if os.path.exists(p):
            rows.append(Image.open(p).convert("RGB"))
    if not rows:
        return
    w = max(r.width for r in rows); h = sum(int(r.height * w / r.width) for r in rows)
    canvas = Image.new("RGB", (w, h), "white"); y = 0
    for r in rows:
        rh = int(r.height * w / r.width); r = r.resize((w, rh))
        canvas.paste(r, (0, y)); y += rh
    canvas.save(path)
    print("[gallery]", path, canvas.size)


def main():
    families = {
        "Gray-Scott reaction-diffusion (FIELD self-update)":
            ["rd_spots", "rd_gliders", "rd_bubbles", "rd_maze", "rd_worms"],
        "Acoustic scattering (FIELD wave, heterogeneous medium)":
            ["wave_inclusions", "wave_split", "wave_lens", "wave_gradient", "wave_double_slit", "wave_maze"],
        "Active matter (SET: Vicsek particles + radius graph)":
            ["am_flock", "am_disorder", "am_bands", "am_swirl"],
    }
    all_names = [n for v in families.values() for n in v]
    logf = open(os.path.join(OUT, "run.log"), "w")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    results = {}
    i = 0
    for fam, names in families.items():
        for nm in names:
            i += 1
            log(f"[{i}/{len(all_names)}] {nm} ...")
            try:
                results[nm] = run_one(nm, log)
            except Exception:
                log(f"    FAILED: {nm}\n{traceback.format_exc()}")
                results[nm] = dict(name=nm, ok=False, metric="-", value="-", extra="error", frames=0, size="-", dt=0)

    # results.md
    md = ["# The Well in Plexus -- exhaustive test suite\n",
          "Each row is one scenario: a spec.yaml driving the generic engine, no per-",
          "scenario code. `ok` is the per-family intent check (behaviour occurred, not",
          "just 'it ran'). Cite: The Well, Ohana et al., NeurIPS 2024 "
          "(github.com/PolymathicAI/the_well).\n"]
    for fam, names in families.items():
        md.append(f"\n## {fam}\n")
        md.append("name | frames | grid/N | metric | value | extra | ok | sec")
        md.append("---|---|---|---|---|---|---|---")
        for nm in names:
            r = results[nm]
            md.append(f"{r['name']} | {r.get('frames','-')} | {r.get('size','-')} | "
                      f"{r.get('metric','-')} | {r.get('value','-')} | {r.get('extra','-')} | "
                      f"{'OK' if r.get('ok') else 'FAIL'} | {r.get('dt','-')}")
    n_ok = sum(1 for r in results.values() if r.get("ok"))
    md.append(f"\n**{n_ok}/{len(all_names)} scenarios passed their intent check.**\n")
    open(os.path.join(OUT, "results.md"), "w").write("\n".join(md))
    log(f"[suite done] {n_ok}/{len(all_names)} ok -> results.md")

    gallery(all_names, os.path.join(OUT, "gallery.png"))
    logf.close()


if __name__ == "__main__":
    main()
