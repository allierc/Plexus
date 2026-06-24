"""render_field_2x2.py -- 2x2 black-background "field used" map for the cardio_mpm loop runs.

Recreates the spirit of prototype/cardio/archive/3_p1_aniso/p1_aniso_properties.png (stiffness,
active contraction gain, fibre angle mod pi, fibre directions over gain) but as a 2x2 grid on a
black background, for each of the 2x2_{A,B,C,D} loop-test runs. Reads each run's spec.yaml to find
the material TIFFs (iso vs structured) and the apply_material_map min/max, loads the fields from
$GraphData/graphs_data/material/, and writes <run>/<run_basename>_properties.png next to the movie.

Run:  python render_field_2x2.py
"""
import os
import numpy as np
import yaml
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARCHIVE = "/groups/saalfeld/home/allierc/Graph/Plexus/prototype/cardio_mpm/archive"
GRAPHDATA = "/groups/saalfeld/home/allierc/GraphData/graphs_data"
RUNS = ["2x2_A_free_iso", "2x2_B_free_struct", "2x2_C_anch_iso", "2x2_D_anch_struct"]


def load_scalar(path):
    a = np.asarray(tifffile.imread(path), dtype=np.float32)
    if a.ndim == 3:                       # collapse any channel dim to a scalar field
        a = a[..., 0]
    return a


def load_angle(path):
    """vector_grid TIFF: single channel stores angle/(2*pi); recover fibre angle in radians."""
    a = np.asarray(tifffile.imread(path), dtype=np.float32)
    if a.ndim == 3:                       # 2-channel (cos,sin) -> atan2
        return np.arctan2(a[..., 1], a[..., 0])
    return a * 2.0 * np.pi


def material_map_range(spec, target):
    """min/max from the apply_material_map operator writing `target` (e.g. youngs, gain)."""
    for op in spec.get("operators", []):
        if op.get("op") == "apply_material_map" and op.get("target") == target:
            return float(op.get("min", 0.0)), float(op.get("max", 1.0))
    return 0.0, 1.0


def render(run):
    d = os.path.join(ARCHIVE, run)
    spec = yaml.safe_load(open(os.path.join(d, "spec.yaml")))
    fields = spec["fields"]
    stiff_src = os.path.join(GRAPHDATA, fields["stiffness_map"]["source"])
    gain_src = os.path.join(GRAPHDATA, fields["gain_map"]["source"])
    dir_src = os.path.join(GRAPHDATA, fields["direction"]["source"])

    smin, smax = material_map_range(spec, "youngs")
    gmin, gmax = material_map_range(spec, "gain")
    stiff = smin + load_scalar(stiff_src) * (smax - smin)        # -> Young's modulus
    gain = gmin + load_scalar(gain_src) * (gmax - gmin)          # -> active contraction gain
    fib = load_angle(dir_src)                                    # -> fibre angle (rad)
    Hy, Wx = fib.shape

    name = spec["general"].get("name", run)
    cond = run.replace("2x2_", "").replace("_", " ")
    fig, axs = plt.subplots(2, 2, figsize=(11, 11), facecolor="black")
    fig.suptitle(f"field used  --  {cond}", color="#ddd", fontsize=15, y=0.965)

    def panel(ax, m, cmap, title, **kw):
        ax.set_facecolor("black")
        im = ax.imshow(m, cmap=cmap, **kw)
        ax.set_title(title, color="#ccc", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors="white", labelsize=8)
        plt.setp(cb.ax.get_yticklabels(), color="white")
        cb.outline.set_edgecolor("#555")

    panel(axs[0, 0], stiff, "viridis", "stiffness  (Young's modulus)")
    panel(axs[0, 1], gain, "magma", "active contraction gain")
    panel(axs[1, 0], fib % np.pi, "twilight", "fibre angle (mod π)")

    ax = axs[1, 1]; ax.set_facecolor("black")
    s = max(1, Hy // 26)
    yy, xx = np.meshgrid(np.arange(0, Hy, s), np.arange(0, Wx, s), indexing="ij")
    ax.imshow(gain, cmap="gray", alpha=0.4)
    ax.quiver(xx, yy, np.cos(fib[::s, ::s]), np.sin(fib[::s, ::s]),
              color="red", pivot="mid", scale=26)
    ax.set_title("fibre directions over gain", color="#ccc", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

    out = os.path.join(d, f"{run}_properties.png")
    fig.savefig(out, dpi=120, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}  (stiff {stiff.min():.0f}-{stiff.max():.0f}, "
          f"gain {gain.min():.2f}-{gain.max():.2f}, fibre {fib.min():.2f}-{fib.max():.2f} rad)")


if __name__ == "__main__":
    for run in RUNS:
        render(run)
